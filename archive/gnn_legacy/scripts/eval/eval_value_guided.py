#!/usr/bin/env python3
"""Evaluate value-guided best-first search vs blind search on gate3_v2.

Compares two configurations:
  BLIND:  Pure lemma-scoring, no value network (value_weight=0.0)
  VALUE:  Value-guided, lemma_score * 0.7 + value_estimate * 0.3

Runs both on all 64 gate3_v2 theorems and reports the comparison.

Output: data/value_net_result.json

Usage:
  python scripts/eval/eval_value_guided.py \
      --gnn-checkpoint checkpoints/gnn/10m_hybrid.pt \
      --value-checkpoint checkpoints/value_network.pt \
      --graph data/graph/dependency_graph_full \
      --theorems data/raw/gate3_v2.jsonl \
      --output data/value_net_result.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import threading
from collections import Counter
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.explorer.dependency_graph import DependencyGraph
from src.explorer.gnn_config import GNNConfig
from src.explorer.gnn_encoder import (
    GNNEncoder,
    extract_initial_features,
    prepare_graph_tensors,
)
from src.explorer.gnn_best_first_search import GNNBestFirstSearch, GNNBestFirstConfig
from src.explorer.value_network import ValueNetwork
from src.explorer.proof_state import ProofState, Tactic
from src.proof_checker.batch_checker import BatchChecker
from src.proof_checker.formats import wrap_theorem_with_proof
from scripts.eval.eval_gnn_prover import (
    build_lemma_index,
    build_lemma_norm_index,
    normalize_expression,
    extract_conclusion,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def save_json(data: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def classify_proof_pattern(proof_steps: list[str]) -> str:
    if not proof_steps:
        return "empty"
    tactic_types = set()
    for step in proof_steps:
        s = step.strip().lower()
        if s.startswith("rw"): tactic_types.add("rw")
        elif s.startswith("exact"): tactic_types.add("exact")
        elif s.startswith("apply"): tactic_types.add("apply")
        elif s.startswith("intro"): tactic_types.add("intro")
        elif s.startswith("have"): tactic_types.add("have")
        elif s in ("ring", "simp", "linarith", "field_simp", "positivity",
                    "norm_num", "nlinarith"):
            tactic_types.add(s)
        elif s.startswith("calc"): tactic_types.add("calc")
        else: tactic_types.add("other")
    if len(tactic_types) >= 2:
        return "multi"
    steps_text = " ".join(proof_steps).lower()
    patterns = ["rfl", "add_comm", "mul_comm", "ring", "field_simp",
                "linarith", "simp", "intro", "apply", "nlinarith"]
    for p in patterns:
        if p in steps_text:
            return p
    return "other"


class TimeoutError(Exception):
    pass


def _search_with_timeout(bf_search, stmt: str, domain: str, timeout_s: float):
    """Run bf_search.search(stmt, domain=domain) with a timeout."""
    result = [None]
    exception = [None]
    done = threading.Event()

    def _run():
        try:
            result[0] = bf_search.search(stmt, domain=domain, verbose=False)
        except Exception as e:
            exception[0] = e
        finally:
            done.set()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    if not done.wait(timeout=timeout_s):
        raise TimeoutError(f"Search timed out after {timeout_s}s")

    if exception[0]:
        raise exception[0]
    return result[0]


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup_search(
    gnn: GNNEncoder,
    graph: DependencyGraph,
    lemma_to_idx: dict[str, int],
    idx_to_norm: dict[int, str],
    config: GNNBestFirstConfig,
    checker: BatchChecker | None = None,
    value_network=None,
) -> GNNBestFirstSearch:
    """Set up the GNN best-first search with shared resources."""

    print("Computing GNN node embeddings...")
    features = extract_initial_features(graph, gnn.config)
    sources, targets, edge_types, num_nodes = prepare_graph_tensors(graph)

    with torch.no_grad():
        node_embeddings = gnn(features, sources, targets, edge_types, num_nodes)

    return GNNBestFirstSearch(
        gnn=gnn,
        graph=graph,
        node_embeddings=node_embeddings,
        lemma_index=lemma_to_idx,
        idx_to_norm=idx_to_norm,
        config=config,
        proof_checker=checker if config.use_proof_checker else None,
        value_network=value_network,
    )


# ---------------------------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------------------------

def run_evaluation(
    bf_search: GNNBestFirstSearch,
    theorems: list[dict],
    checker: BatchChecker,
    timeout_per_theorem: float = 120.0,
    label: str = "blind",
) -> dict:
    """Run evaluation on gate3_v2 theorems."""
    print(f"\n{'=' * 70}")
    print(f"RUNNING: {label}")
    print(f"{'=' * 70}")

    results = []
    t_start = time.time()
    passed = []
    timeouts = 0
    failed_reasons: dict[str, int] = {}

    for i, t in enumerate(theorems):
        stmt = t["statement"]
        name = t["name"]
        domain = t.get("domain", "unknown")
        era = t.get("era", "unknown")
        ground_truth = t.get("proof", "?")

        t0 = time.time()

        try:
            proof_steps, final_state = _search_with_timeout(
                bf_search, stmt, domain, timeout_per_theorem
            )
        except TimeoutError:
            search_time = time.time() - t0
            proof_steps = []
            ok = False
            err = f"timeout ({timeout_per_theorem}s)"
            failed_reasons["timeout"] = failed_reasons.get("timeout", 0) + 1
            timeouts += 1
        else:
            search_time = time.time() - t0
            proof_text = ProofState._render_proof(proof_steps)

            if not proof_steps:
                ok = False
                err = "no proof found"
                failed_reasons["no_proof"] = failed_reasons.get("no_proof", 0) + 1
            else:
                full_code = wrap_theorem_with_proof(stmt, proof_text)
                check_results = checker.check_batch([full_code])
                ok = check_results[0].success
                err = check_results[0].errors[0][:200] if check_results[0].errors else ""
                if not ok:
                    reason_key = f"lean_reject:{err[:60]}"
                    failed_reasons[reason_key] = failed_reasons.get(reason_key, 0) + 1

        steps_str = [s.to_lean() for s in proof_steps[:10]]
        pattern = classify_proof_pattern(steps_str) if ok else "failed"

        result = {
            "name": name,
            "era": era,
            "domain": domain,
            "success": ok,
            "error": err,
            "steps": steps_str,
            "num_steps": len(proof_steps),
            "search_time_s": round(search_time, 1),
            "pattern": pattern,
        }
        results.append(result)

        if ok:
            passed.append(result)

        status = "✓" if ok else "✗"
        elapsed_total = time.time() - t_start
        eta = elapsed_total / (i + 1) * (len(theorems) - i - 1) if i > 0 else 0
        print(f"  [{i+1:2d}/{len(theorems)}] {status} {name:45s} "
              f"[{pattern:12s}] {search_time:.1f}s  "
              f"ETA: {eta/60:.0f}m  ({len(passed)} passed)")

        if ok and len(proof_steps) > 0:
            print(f"         Proof: {steps_str}")
            if len(proof_steps) >= 2:
                print(f"         ★ MULTI-STEP ({len(proof_steps)} steps)")

    elapsed = time.time() - t_start
    n_total = len(theorems)
    n_passed = len(passed)
    rate = n_passed / max(1, n_total)

    multi = [r for r in passed if r["num_steps"] >= 2]

    # Domain breakdown
    domains = Counter(r["domain"] for r in results)
    domain_stats = {}
    for dom in sorted(domains.keys()):
        dom_total = domains[dom]
        dom_passed = sum(1 for r in passed if r["domain"] == dom)
        dom_ms = sum(1 for r in multi if r["domain"] == dom)
        domain_stats[dom] = {
            "total": dom_total,
            "passed": dom_passed,
            "rate": dom_passed / max(1, dom_total),
            "multi_step": dom_ms,
        }

    print(f"\n--- {label} Results ---")
    print(f"  Passed: {n_passed}/{n_total} ({rate:.0%})")
    print(f"  Multi-step: {len(multi)}")
    print(f"  Timeouts: {timeouts}")
    print(f"  Elapsed: {elapsed:.0f}s")

    for dom, stats in domain_stats.items():
        print(f"    {dom:<20} {stats['passed']}/{stats['total']} "
              f"({stats['rate']:.0%}) MS: {stats['multi_step']}")

    return {
        "label": label,
        "total": n_total,
        "passed": n_passed,
        "rate": rate,
        "multi_step": len(multi),
        "timeouts": timeouts,
        "elapsed_s": elapsed,
        "failed_reasons": dict(failed_reasons),
        "domain_stats": domain_stats,
        "passed_theorems": [
            {
                "name": r["name"],
                "domain": r["domain"],
                "proof": " ".join(r["steps"]),
                "pattern": r["pattern"],
                "num_steps": r["num_steps"],
            }
            for r in passed
        ],
        "results": results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate value-guided vs blind search on gate3_v2"
    )
    parser.add_argument(
        "--gnn-checkpoint",
        default="checkpoints/gnn/10m_hybrid.pt",
        help="GNN checkpoint path",
    )
    parser.add_argument(
        "--value-checkpoint",
        default="checkpoints/value_network.pt",
        help="Value network checkpoint path",
    )
    parser.add_argument(
        "--graph",
        default="data/graph/dependency_graph_full",
        help="Dependency graph path",
    )
    parser.add_argument(
        "--theorems",
        default="data/raw/gate3_v2.jsonl",
        help="Theorem JSONL",
    )
    parser.add_argument(
        "--output",
        default="data/value_net_result.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--max-expansions", type=int, default=1000,
        help="Max expansions for best-first search",
    )
    parser.add_argument(
        "--timeout-per-theorem", type=float, default=120.0,
        help="Per-theorem timeout in seconds",
    )
    parser.add_argument(
        "--value-weight", type=float, default=0.3,
        help="Value weight (0=blind, 1=pure value)",
    )
    parser.add_argument(
        "--num-threads", type=int, default=6,
        help="Number of CPU threads",
    )
    parser.add_argument(
        "--skip-blind", action="store_true",
        help="Skip blind search run (only run value-guided)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("VALUE NETWORK EVALUATION: Blind vs Value-Guided Search")
    print("=" * 70)
    print(f"GNN checkpoint: {args.gnn_checkpoint}")
    print(f"Value checkpoint: {args.value_checkpoint}")
    print(f"Value weight: {args.value_weight}")
    print(f"Max expansions: {args.max_expansions}")
    print(f"Per-theorem timeout: {args.timeout_per_theorem}s")
    print(f"Threads: {args.num_threads}")
    print()

    # Hardware constraint
    torch.set_num_threads(min(args.num_threads, 6))

    # --- Load GNN ---
    ckpt_path = _PROJECT_ROOT / args.gnn_checkpoint
    if not ckpt_path.exists():
        print(f"ERROR: GNN checkpoint not found: {ckpt_path}")
        return 1

    print(f"Loading GNN: {ckpt_path}")
    gnn = GNNEncoder.load(str(ckpt_path))
    gnn.eval()
    n_params = sum(p.numel() for p in gnn.parameters())
    print(f"  GNN: {n_params:,} params, hidden={gnn.config.hidden_dim}")

    # --- Load value network ---
    value_net = None
    vn_path = _PROJECT_ROOT / args.value_checkpoint
    if vn_path.exists():
        print(f"Loading value network: {vn_path}")
        value_net = ValueNetwork.load(str(vn_path), gnn, freeze_encoder=True)
        value_net.eval()
        vn_params = sum(p.numel() for p in value_net.value_head.parameters())
        print(f"  Value head: {vn_params:,} params")
    else:
        print(f"WARNING: Value checkpoint not found: {vn_path}")
        print("  Running with value_weight=0.0 (blind search only)")
        args.value_weight = 0.0

    # --- Load graph ---
    graph_path = _PROJECT_ROOT / args.graph
    if not graph_path.with_suffix(".nx.pkl").exists():
        graph_path = _PROJECT_ROOT / "data/graph/dependency_graph"
        if not graph_path.with_suffix(".nx.pkl").exists():
            print(f"ERROR: Graph not found")
            return 1

    print(f"Loading graph: {graph_path}")
    graph = DependencyGraph.load(graph_path)
    print(f"  Graph: {graph.summary()}")

    # --- Load theorems ---
    theorems_path = _PROJECT_ROOT / args.theorems
    if not theorems_path.exists():
        print(f"ERROR: Theorems not found: {theorems_path}")
        return 1

    theorems = load_jsonl(theorems_path)
    print(f"Theorems: {len(theorems)}")

    # --- Build indices ---
    print("Building indices...")
    lemma_to_idx = build_lemma_index(graph)
    idx_to_norm = build_lemma_norm_index(graph, lemma_to_idx)
    print(f"  Lemma index: {len(lemma_to_idx)} entries")

    # --- Setup proof checker ---
    print("Setting up proof checker...")
    checker = BatchChecker(max_workers=4, timeout=args.timeout_per_theorem)

    # --- Run blind search ---
    blind_result = None
    if not args.skip_blind:
        blind_config = GNNBestFirstConfig(
            max_expansions=args.max_expansions,
            value_weight=0.0,  # blind
            value_prune_threshold=None,
            num_threads=args.num_threads,
            top_k_lemmas=30,
        )
        blind_search = setup_search(
            gnn, graph, lemma_to_idx, idx_to_norm,
            blind_config, checker, value_network=None,
        )
        blind_result = run_evaluation(
            blind_search, theorems, checker,
            timeout_per_theorem=args.timeout_per_theorem,
            label="BLIND (value_weight=0.0)",
        )

    # --- Run value-guided search ---
    value_config = GNNBestFirstConfig(
        max_expansions=args.max_expansions,
        value_weight=args.value_weight,
        value_prune_threshold=0.1,
        num_threads=args.num_threads,
        top_k_lemmas=30,
    )
    value_search = setup_search(
        gnn, graph, lemma_to_idx, idx_to_norm,
        value_config, checker, value_network=value_net,
    )
    value_result = run_evaluation(
        value_search, theorems, checker,
        timeout_per_theorem=args.timeout_per_theorem,
        label=f"VALUE-GUIDED (value_weight={args.value_weight})",
    )

    # --- Comparison ---
    print(f"\n{'=' * 70}")
    print("COMPARISON")
    print(f"{'=' * 70}")

    if blind_result:
        print(f"  Blind:      {blind_result['passed']}/{blind_result['total']} "
              f"({blind_result['rate']:.0%}), "
              f"{blind_result['multi_step']} multi-step")
    print(f"  Value:      {value_result['passed']}/{value_result['total']} "
          f"({value_result['rate']:.0%}), "
          f"{value_result['multi_step']} multi-step")

    if blind_result:
        delta = value_result['passed'] - blind_result['passed']
        delta_ms = value_result['multi_step'] - blind_result['multi_step']
        print(f"  Δ:          {delta:+d} proofs, {delta_ms:+d} multi-step")
        print(f"  Improvement: {delta/blind_result['total']:+.1%}")

    # --- Find differences ---
    new_proofs: set[str] = set()
    lost_proofs: set[str] = set()
    if blind_result:
        blind_passed = {r["name"] for r in blind_result["passed_theorems"]}
        value_passed = {r["name"] for r in value_result["passed_theorems"]}
        new_proofs = value_passed - blind_passed
        lost_proofs = blind_passed - value_passed

        if new_proofs:
            print(f"\n  New proofs found (value-guided only):")
            for name in sorted(new_proofs):
                for r in value_result["passed_theorems"]:
                    if r["name"] == name:
                        print(f"    + {name}: {r['proof']} [{r['pattern']}]")
        if lost_proofs:
            print(f"\n  Proofs lost (blind only):")
            for name in sorted(lost_proofs):
                for r in blind_result["passed_theorems"]:
                    if r["name"] == name:
                        print(f"    - {name}: {r['proof']} [{r['pattern']}]")

    # --- Domain-level comparison ---
    if blind_result:
        print(f"\n  Domain breakdown:")
        all_domains = set(blind_result["domain_stats"]) | set(value_result["domain_stats"])
        for dom in sorted(all_domains):
            bs = blind_result["domain_stats"].get(dom, {"passed": 0, "total": 0})
            vs = value_result["domain_stats"].get(dom, {"passed": 0, "total": 0})
            delta_d = vs["passed"] - bs["passed"]
            print(f"    {dom:<20} blind={bs['passed']}/{bs['total']}  "
                  f"value={vs['passed']}/{vs['total']}  "
                  f"Δ={delta_d:+d}")

    # --- Build output ---
    out = {
        "task": "Value network evaluation: blind vs value-guided best-first search",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "architecture": "Frozen GNN GoalEncoder → ValueHead MLP (768→256→1) → best-first priority blending",
        "config": {
            "gnn_checkpoint": str(ckpt_path),
            "value_checkpoint": str(vn_path) if vn_path.exists() else "none",
            "value_weight": args.value_weight,
            "max_expansions": args.max_expansions,
            "timeout_per_theorem_s": args.timeout_per_theorem,
            "num_threads": args.num_threads,
            "gnn_params": n_params,
        },
        "blind": blind_result,
        "value_guided": value_result,
        "comparison": {
            "blind_passed": blind_result["passed"] if blind_result else None,
            "value_passed": value_result["passed"],
            "delta": (value_result["passed"] - blind_result["passed"])
                if blind_result else None,
            "new_proofs": list(new_proofs) if blind_result else [],
            "lost_proofs": list(lost_proofs) if blind_result else [],
        },
    }

    output_path = _PROJECT_ROOT / args.output
    save_json(out, output_path)
    print(f"\nResults saved to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
