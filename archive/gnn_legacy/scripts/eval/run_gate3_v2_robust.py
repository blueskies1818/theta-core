#!/usr/bin/env python3
"""Focused Gate 3 run: Hybrid architecture on FULL gate3_v2 benchmark (64 theorems, 5 domains).

Architecture: GNN cosine similarity (MRR 0.786) + best-first search + dense rewards.
12 CPU threads. WITH per-theorem timeout to avoid getting stuck.
Output: data/gate3_v2_full_result.json

Key fix over run_full_gate3_v2.py: per-theorem timeout (120s default) prevents
the search from hanging indefinitely on difficult theorems like alg_cross_multiply.

Usage:
    python scripts/eval/run_gate3_v2_robust.py [--timeout-per-theorem N] [--max-expansions N]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import sys
import time
import threading
from collections import Counter
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import torch

from src.explorer.dependency_graph import DependencyGraph
from src.explorer.gnn_encoder import (
    GNNEncoder,
    extract_initial_features,
    prepare_graph_tensors,
)
from src.explorer.gnn_best_first_search import GNNBestFirstSearch, GNNBestFirstConfig
from src.explorer.proof_state import ProofState, Tactic
from src.proof_checker.batch_checker import BatchChecker
from src.proof_checker.formats import wrap_theorem_with_proof
from scripts.eval.eval_gnn_prover import (
    build_lemma_index,
    extract_conclusion,
    normalize_expression,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _project_path(rel: str) -> Path:
    return _PROJECT_ROOT / rel


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
        elif s.startswith("constructor"): tactic_types.add("constructor")
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


def is_lemma_novelty(proof_steps: list[str]) -> bool:
    structural = {"simp", "ring", "linarith", "field_simp", "rfl", "norm_num",
                   "nlinarith", "positivity", "omega", "native_decide"}
    has_lemma = False
    for step in proof_steps:
        s = step.strip().lower()
        tactic = s.split()[0] if s else ""
        if tactic not in structural and not s.startswith("exact"):
            has_lemma = True
            break
    lemma_refs = re.findall(r'rw\s*\[([^\]]+)\]', " ".join(proof_steps))
    for ref in lemma_refs:
        parts = ref.split(",")
        for p in parts:
            p = p.strip()
            if p not in structural and p not in ("h", "h1", "h2", "h3", "h'"):
                has_lemma = True
                break
    return has_lemma


def build_norm_index(graph: DependencyGraph, lemma_to_idx: dict[str, int]) -> dict[int, str]:
    idx_to_norm: dict[int, str] = {}
    for node_id in graph.node_ids:
        idx = lemma_to_idx.get(node_id)
        if idx is None:
            continue
        node = graph.get_node(node_id)
        if node:
            statement = node.get("statement", "")
            if statement:
                conclusion = extract_conclusion(statement)
                if conclusion:
                    idx_to_norm[idx] = normalize_expression(conclusion)
    return idx_to_norm


# ---------------------------------------------------------------------------
# Timeout wrapper for per-theorem search
# ---------------------------------------------------------------------------

class TimeoutError(Exception):
    pass


def _search_with_timeout(bf_search, stmt: str, timeout_s: float):
    """Run bf_search.search(stmt) with a timeout, returning (steps, state) or raising TimeoutError."""
    result = [None]
    exception = [None]
    done = threading.Event()

    def _run():
        try:
            result[0] = bf_search.search(stmt, verbose=False)
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
# Main Gate 3 run
# ---------------------------------------------------------------------------

def run_gate3_full_robust(
    gnn: GNNEncoder,
    graph: DependencyGraph,
    theorems: list[dict],
    config: GNNBestFirstConfig,
    lemma_to_idx: dict[str, int],
    idx_to_norm: dict[int, str],
    checker: BatchChecker,
    output_path: Path,
    timeout_per_theorem: float = 120.0,
) -> dict:
    print("\n" + "=" * 70)
    print("GATE 3 FULL: Hybrid (GNN + Best-First) on FULL gate3_v2 (64 theorems)")
    print("=" * 70)
    print(f"Per-theorem timeout: {timeout_per_theorem}s")
    print(f"Max expansions: {config.max_expansions}")

    # --- Compute node embeddings ---
    print("\nComputing GNN node embeddings on full graph...")
    features = extract_initial_features(graph, gnn.config)
    sources, targets, edge_types, num_nodes = prepare_graph_tensors(graph)
    print(f"  Graph: {num_nodes} nodes, {sources.size(0)} edges")

    with torch.no_grad():
        node_embeddings = gnn(features, sources, targets, edge_types, num_nodes)
    print(f"  Embeddings: {node_embeddings.shape}")

    # --- Setup search ---
    bf_search = GNNBestFirstSearch(
        gnn=gnn,
        graph=graph,
        node_embeddings=node_embeddings,
        lemma_index=lemma_to_idx,
        idx_to_norm=idx_to_norm,
        config=config,
        proof_checker=checker if config.use_proof_checker else None,
    )

    print(f"\n--- Running GNN best-first search on {len(theorems)} theorems ---")
    print(f"    Max expansions: {config.max_expansions}, "
          f"Top-K lemmas: {config.top_k_lemmas}")
    print(f"    Search threads: {config.num_threads}")
    print(f"    Per-theorem timeout: {timeout_per_theorem}s")
    print()

    results = []
    t_start = time.time()
    passed = []
    failed_reasons: dict[str, int] = {}
    timeouts = 0

    for i, t in enumerate(theorems):
        stmt = t["statement"]
        name = t["name"]
        domain = t.get("domain", "unknown")
        era = t.get("era", "unknown")
        ground_truth = t.get("proof", "?")

        t0 = time.time()

        try:
            proof_steps, final_state = _search_with_timeout(
                bf_search, stmt, timeout_per_theorem
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
        lemma_novel = is_lemma_novelty(steps_str) if ok else False

        result = {
            "name": name,
            "era": era,
            "domain": domain,
            "success": ok,
            "error": err,
            "hybrid_steps": steps_str,
            "num_steps": len(proof_steps),
            "ground_truth": ground_truth,
            "search_time_s": round(search_time, 1),
            "pattern": pattern,
            "lemma_novelty": lemma_novel,
        }
        results.append(result)

        if ok:
            passed.append(result)

        status = "\u2713" if ok else "\u2717"
        elapsed_total = time.time() - t_start
        eta = elapsed_total / (i + 1) * (len(theorems) - i - 1) if i > 0 else 0
        print(f"  [{i+1:2d}/{len(theorems)}] {status} {name:45s} "
              f"[{pattern:12s}] {search_time:.1f}s  "
              f"ETA: {eta/60:.0f}m  ({len(passed)} passed){' TIMEOUT' if err.startswith('timeout') else ''}")

        if ok and len(proof_steps) > 0:
            print(f"         Proof: {steps_str}")
            if len(proof_steps) >= 2:
                print(f"         \u2605 MULTI-STEP ({len(proof_steps)} steps)")

    elapsed = time.time() - t_start
    n_total = len(theorems)
    n_passed = len(passed)
    rate = n_passed / max(1, n_total)

    multi = [r for r in passed if r["num_steps"] >= 2]
    lemma_novel = [r for r in passed if r["lemma_novelty"]]
    structural = [r for r in passed if not r["lemma_novelty"]]

    # --- Stats ---
    print(f"\n{'=' * 70}")
    print("GATE 3 FULL RESULTS")
    print(f"{'=' * 70}")
    print(f"  Total:    {n_passed}/{n_total} ({rate:.0%})")
    print(f"  Multi-step: {len(multi)}")
    print(f"  Lemma-novelty: {len(lemma_novel)}")
    print(f"  Structural-only: {len(structural)}")
    print(f"  Timeouts:  {timeouts}")
    print(f"  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}m)")

    # Domain breakdown
    domains = Counter(r["domain"] for r in results)
    print(f"\n  By domain:")
    for dom in sorted(domains.keys()):
        dom_total = domains[dom]
        dom_passed = sum(1 for r in passed if r["domain"] == dom)
        dom_ln = sum(1 for r in lemma_novel if r["domain"] == dom)
        dom_ms = sum(1 for r in multi if r["domain"] == dom)
        print(f"    {dom:<20} {dom_passed}/{dom_total} "
              f"({dom_passed/max(1,dom_total)*100:.0f}%) "
              f"LN: {dom_ln}  MS: {dom_ms}")

    print(f"\n  Failure reasons:")
    for reason, count in sorted(failed_reasons.items(), key=lambda x: -x[1])[:10]:
        print(f"    {reason:<60} {count}")

    # --- Build output ---
    out = {
        "task": "FULL gate3_v2 benchmark: Hybrid (GNN + Best-First + Dense Rewards) [robust]",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "architecture": "GNN cosine similarity (MRR 0.786) + Best-first search with per-theorem timeout",
        "config": {
            "max_depth": config.max_depth,
            "max_expansions": config.max_expansions,
            "top_k_lemmas": config.top_k_lemmas,
            "depth_penalty": config.depth_penalty,
            "use_proof_checker": config.use_proof_checker,
            "num_threads": config.num_threads,
            "max_graph_candidates": config.max_graph_candidates,
            "timeout_per_theorem_s": timeout_per_theorem,
            "gnn_params": sum(p.numel() for p in gnn.parameters()),
        },
        "graph": {
            "num_nodes": graph.num_nodes,
            "num_edges": graph.num_edges,
        },
        "gate3": {
            "status": "PASS" if n_passed > 0 else "FAIL",
            "total": n_total,
            "passed": n_passed,
            "rate": rate,
            "multi_step": len(multi),
            "lemma_novelty": len(lemma_novel),
            "structural_only": len(structural),
            "timeouts": timeouts,
            "elapsed_s": elapsed,
            "failed_reasons": dict(failed_reasons),
            "domains": {dom: {
                "total": domains[dom],
                "passed": sum(1 for r in passed if r["domain"] == dom),
                "lemma_novelty": sum(1 for r in lemma_novel if r["domain"] == dom),
                "multi_step": sum(1 for r in multi if r["domain"] == dom),
            } for dom in domains},
            "passed_theorems": [
                {
                    "name": r["name"],
                    "domain": r["domain"],
                    "proof": " ".join(r["hybrid_steps"]),
                    "pattern": r["pattern"],
                    "num_steps": r["num_steps"],
                    "lemma_novelty": r["lemma_novelty"],
                }
                for r in passed
            ],
            "multi_step_theorems": [
                {
                    "name": r["name"],
                    "domain": r["domain"],
                    "proof": " ".join(r["hybrid_steps"]),
                    "num_steps": r["num_steps"],
                    "lemma_novelty": r["lemma_novelty"],
                }
                for r in multi
            ],
        },
        "all_results": results,
    }

    save_json(out, output_path)
    print(f"\n  Results saved to: {output_path}")

    print(f"\n  Gate 3: {out['gate3']['status']} ({n_passed}/{n_total} proofs, "
          f"{len(multi)} multi-step, {len(lemma_novel)} lemma-novelty)")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run hybrid architecture on FULL gate3_v2 benchmark with per-theorem timeout"
    )
    parser.add_argument(
        "--gnn-checkpoint",
        default="checkpoints/gnn/gate2_fullgraph_finetuned.pt",
        help="GNN checkpoint path",
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
        "--max-expansions", type=int, default=1000,
        help="Max expansions for best-first search",
    )
    parser.add_argument(
        "--timeout-per-theorem", type=float, default=120.0,
        help="Per-theorem timeout in seconds (default: 120s)",
    )
    parser.add_argument(
        "--top-k", type=int, default=30,
        help="Top-K lemmas per state",
    )
    parser.add_argument(
        "--depth-penalty", type=float, default=0.05,
        help="Depth penalty factor",
    )
    parser.add_argument(
        "--num-threads", type=int, default=12,
        help="Number of CPU threads",
    )
    parser.add_argument(
        "--output", default="data/gate3_v2_full_result.json",
        help="Output JSON file",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("FULL gate3_v2 BENCHMARK: Hybrid Architecture Run (ROBUST)")
    print("=" * 70)
    print(f"Target: 64 theorems, 5 domains")
    print(f"Architecture: GNN cosine similarity + Best-first search")
    print(f"Threads: {args.num_threads}")
    print(f"Max expansions: {args.max_expansions}")
    print(f"Top-K lemmas: {args.top_k}")
    print(f"Per-theorem timeout: {args.timeout_per_theorem}s")
    print()

    # Hardware constraint: max 12 threads
    if args.num_threads > 12:
        print(f"WARNING: Reducing threads from {args.num_threads} to 12 (max per constraints)")
        args.num_threads = 12

    torch.set_num_threads(args.num_threads)
    print(f"PyTorch threads: {torch.get_num_threads()}")

    # --- Load GNN ---
    ckpt_path = _project_path(args.gnn_checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        return 1

    print(f"Loading GNN checkpoint: {ckpt_path}")
    gnn = GNNEncoder.load(str(ckpt_path))
    gnn.eval()
    n_params = sum(p.numel() for p in gnn.parameters())
    print(f"GNN: {n_params:,} params, hidden={gnn.config.hidden_dim}")

    # --- Load graph ---
    graph_path = _project_path(args.graph)
    if not graph_path.with_suffix(".nx.pkl").exists():
        print(f"ERROR: Graph not found: {graph_path}.nx.pkl")
        return 1

    print(f"Loading graph: {graph_path}")
    graph = DependencyGraph.load(graph_path)
    print(f"Graph: {graph.summary()}")

    # --- Load theorems ---
    theorems_path = _project_path(args.theorems)
    if not theorems_path.exists():
        print(f"ERROR: Theorems not found: {theorems_path}")
        return 1

    theorems = load_jsonl(theorems_path)
    print(f"Theorems loaded: {len(theorems)}")

    # --- Indexes ---
    print("Building lemma index...")
    lemma_to_idx = build_lemma_index(graph)
    idx_to_norm = build_norm_index(graph, lemma_to_idx)
    print(f"Lemma index: {len(lemma_to_idx)} entries")

    # --- Config ---
    config = GNNBestFirstConfig(
        max_depth=20,
        max_expansions=args.max_expansions,
        top_k_lemmas=args.top_k,
        depth_penalty=args.depth_penalty,
        use_proof_checker=True,  # Root verification for quality
        verify_timeout=5.0,
        num_threads=args.num_threads,
        max_graph_candidates=200,
    )

    checker = BatchChecker(timeout=15, max_workers=8, cache_size=128)

    output_path = _project_path(args.output)

    # --- Run ---
    result = run_gate3_full_robust(
        gnn=gnn,
        graph=graph,
        theorems=theorems,
        config=config,
        lemma_to_idx=lemma_to_idx,
        idx_to_norm=idx_to_norm,
        checker=checker,
        output_path=output_path,
        timeout_per_theorem=args.timeout_per_theorem,
    )

    n_passed = result["gate3"]["passed"]
    n_multi = result["gate3"]["multi_step"]
    n_ln = result["gate3"]["lemma_novelty"]
    n_timeouts = result["gate3"].get("timeouts", 0)

    print(f"\n{'=' * 70}")
    print("FINAL")
    print(f"{'=' * 70}")
    print(f"  Proofs found: {n_passed}/64 ({result['gate3']['rate']:.0%})")
    print(f"  Multi-step:   {n_multi}")
    print(f"  Lemma-novelty: {n_ln}")
    print(f"  Timeouts:      {n_timeouts}")
    print(f"  Output:       {output_path}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
