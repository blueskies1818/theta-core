#!/usr/bin/env python3
"""PATH 3: Iterative proof search with Lean feedback loop.

Run search → get best candidate proof → verify with Lean → if rejected,
use rejection to re-weight GNN lemma rankings → search again.

Multiple lightweight passes instead of one heavy one. The rejection IS the
training signal — same binary proof-checker output, just used iteratively.

Key metric: proofs found after iteration that weren't found in first pass.

Usage:
    python scripts/eval/iterative_search.py [--max-iterations N] [--max-expansions N]
"""

from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
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
from scripts.eval.run_full_gate3_v2 import (
    classify_proof_pattern,
    is_lemma_novelty,
    build_norm_index,
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


def extract_lemmas_from_proof(proof_steps: list[Tactic]) -> set[str]:
    """Extract lemma names used in a proof.

    Collects lemma names from APPLY and REWRITE tactics.
    Skips hypothesis references (bare names like 'h', 'h1').
    """
    lemmas: set[str] = set()
    hypothesis_names = {"h", "h1", "h2", "h3", "h4", "h5", "h6", "h7", "h8", "h9",
                        "h'", "h0", "h_ind", "h_intro", "this"}
    for step in proof_steps:
        if step.lemma and step.lemma not in hypothesis_names:
            lemmas.add(step.lemma)
    return lemmas


def parse_lean_error_for_hint(error: str) -> str:
    """Extract a hint from a Lean error message for smarter re-weighting.

    Returns a hint category: 'type_mismatch', 'unknown_identifier',
    'unsolved_goals', 'invalid_tactic', or 'unknown'.
    """
    error_lower = error.lower()
    if "type mismatch" in error_lower or "could not unify" in error_lower:
        return "type_mismatch"
    if "unknown identifier" in error_lower or "unknown constant" in error_lower:
        return "unknown_identifier"
    if "unsolved goals" in error_lower:
        return "unsolved_goals"
    if "invalid tactic" in error_lower or "unknown tactic" in error_lower:
        return "invalid_tactic"
    return "unknown"


# ---------------------------------------------------------------------------
# Iterative search on a single theorem
# ---------------------------------------------------------------------------

def iterative_search_theorem(
    bf_search: GNNBestFirstSearch,
    checker: BatchChecker,
    stmt: str,
    name: str,
    domain: str | None,
    max_iterations: int,
    max_expansions_per_pass: int,
    verbose: bool = False,
) -> dict:
    """Run iterative search on one theorem.

    Returns dict with keys:
        name, domain, success, error, pass_found, num_passes, proof_steps,
        proof_text, search_time_s, rejected_lemma_count
    """
    rejected_lemmas: set[str] = set()
    all_proofs_tried: list[dict] = []
    t_start = time.time()

    for iteration in range(max_iterations):
        # Update config with current rejection set
        bf_search.config.rejected_lemmas = rejected_lemmas
        # Reset verification cache for fresh search
        bf_search._verification_cache.clear()
        bf_search._goal_embed_cache.clear()
        bf_search._tiebreaker = 0

        # Configure reduced expansions for later passes (lighter passes)
        expansions = max_expansions_per_pass
        if iteration > 0:
            # Later passes: more aggressive penalty, fewer expansions
            bf_search.config.rejection_penalty = min(
                0.9, 0.3 + iteration * 0.15
            )
            # Halve expansions after pass 1 for lightweight passes
            if iteration >= 2:
                expansions = max(200, max_expansions_per_pass // (iteration + 1))

        bf_search.config.max_expansions = expansions

        pass_t0 = time.time()
        proof_steps, _final_state = bf_search.search(
            stmt, domain=domain, verbose=False
        )
        pass_time = time.time() - pass_t0

        if not proof_steps:
            # No proof found in this pass
            all_proofs_tried.append({
                "iteration": iteration,
                "found": False,
                "num_steps": 0,
                "pass_time_s": round(pass_time, 1),
            })
            if verbose:
                print(f"    iteration {iteration}: no proof found "
                      f"({expansions} expansions, {pass_time:.1f}s)")
            continue

        proof_text = ProofState._render_proof(proof_steps)
        steps_str = [s.to_lean() for s in proof_steps[:10]]

        # Verify with Lean
        full_code = wrap_theorem_with_proof(stmt, proof_text)
        check_results = checker.check_batch([full_code])
        ok = check_results[0].success
        err = check_results[0].errors[0][:200] if check_results[0].errors else ""

        if ok:
            search_time = time.time() - t_start
            if verbose:
                print(f"    iteration {iteration}: ✓ VERIFIED "
                      f"({len(proof_steps)} steps, {pass_time:.1f}s)")
            # Check if this was a novel proof not in pass 0
            pass_0_pass = any(
                p["found"] and p["iteration"] == 0
                for p in all_proofs_tried
            )
            return {
                "name": name,
                "domain": domain or "unknown",
                "success": True,
                "error": "",
                "pass_found": iteration,
                "is_newly_found": iteration > 0,
                "num_passes": iteration + 1,
                "proof_steps": steps_str,
                "proof_text": proof_text,
                "search_time_s": round(search_time, 1),
                "rejected_lemma_count": len(rejected_lemmas),
                "num_steps": len(proof_steps),
            }

        # Failed — extract lemmas and add to rejection set
        used_lemmas = extract_lemmas_from_proof(proof_steps)
        new_rejections = used_lemmas - rejected_lemmas
        rejected_lemmas.update(used_lemmas)

        error_hint = parse_lean_error_for_hint(err)

        all_proofs_tried.append({
            "iteration": iteration,
            "found": False,
            "num_steps": len(proof_steps),
            "pass_time_s": round(pass_time, 1),
            "lean_error": err,
            "error_hint": error_hint,
            "rejected_lemmas_added": len(new_rejections),
            "proof_steps": steps_str,
        })

        if verbose:
            print(f"    iteration {iteration}: ✗ Lean rejected "
                  f"[{error_hint}] {err[:80]}... "
                  f"(+{len(new_rejections)} new rejected, "
                  f"total {len(rejected_lemmas)})")

    # Max iterations exhausted
    search_time = time.time() - t_start
    return {
        "name": name,
        "domain": domain or "unknown",
        "success": False,
        "error": f"max iterations ({max_iterations}) exhausted, "
                 f"{len(rejected_lemmas)} lemmas rejected",
        "pass_found": -1,
        "is_newly_found": False,
        "num_passes": max_iterations,
        "proof_steps": [],
        "proof_text": "",
        "search_time_s": round(search_time, 1),
        "rejected_lemma_count": len(rejected_lemmas),
        "num_steps": 0,
        "all_attempts": all_proofs_tried,
    }


# ---------------------------------------------------------------------------
# Main iterative benchmark
# ---------------------------------------------------------------------------

def run_iterative_gate3(
    gnn: GNNEncoder,
    graph: DependencyGraph,
    theorems: list[dict],
    base_config: GNNBestFirstConfig,
    lemma_to_idx: dict[str, int],
    idx_to_norm: dict[int, str],
    checker: BatchChecker,
    output_path: Path,
    max_iterations: int = 5,
    use_domain_filter: bool = True,
) -> dict:
    print("\n" + "=" * 70)
    print("PATH 3: Iterative Proof Search with Lean Rejection Feedback")
    print("=" * 70)
    print(f"  Max iterations: {max_iterations}")
    print(f"  Max expansions per pass: {base_config.max_expansions}")
    print(f"  Top-K lemmas: {base_config.top_k_lemmas}")
    print(f"  Rejection penalty: {base_config.rejection_penalty} (initial)")
    print(f"  Domain filter: {'ON' if use_domain_filter else 'OFF'}")
    print()

    # --- Compute node embeddings ---
    print("Computing GNN node embeddings on full graph...")
    features = extract_initial_features(graph, gnn.config)
    sources, targets, edge_types, num_nodes = prepare_graph_tensors(graph)
    print(f"  Graph: {num_nodes} nodes, {sources.size(0)} edges")

    with torch.no_grad():
        node_embeddings = gnn(features, sources, targets, edge_types, num_nodes)
    print(f"  Embeddings: {node_embeddings.shape}")

    # --- Run iterative search on each theorem ---
    print(f"\n{'─' * 70}")
    print(f"Running iterative search on {len(theorems)} theorems...")
    print(f"{'─' * 70}\n")

    results = []
    t_start = time.time()
    pass_0_passed: list[dict] = []
    newly_found: list[dict] = []
    never_found: list[dict] = []

    for i, t in enumerate(theorems):
        stmt = t["statement"]
        name = t["name"]
        domain = t.get("domain", "unknown") if use_domain_filter else None
        era = t.get("era", "unknown")
        ground_truth = t.get("proof", "?")

        # Fresh search instance per theorem
        search_config = GNNBestFirstConfig(
            max_depth=base_config.max_depth,
            max_expansions=base_config.max_expansions,
            top_k_lemmas=base_config.top_k_lemmas,
            depth_penalty=base_config.depth_penalty,
            use_proof_checker=False,  # handled externally
            verify_timeout=base_config.verify_timeout,
            num_threads=base_config.num_threads,
            max_graph_candidates=base_config.max_graph_candidates,
            rejection_penalty=base_config.rejection_penalty,
        )

        bf_search = GNNBestFirstSearch(
            gnn=gnn,
            graph=graph,
            node_embeddings=node_embeddings,
            lemma_index=lemma_to_idx,
            idx_to_norm=idx_to_norm,
            config=search_config,
            proof_checker=checker,
        )

        result = iterative_search_theorem(
            bf_search=bf_search,
            checker=checker,
            stmt=stmt,
            name=name,
            domain=domain,
            max_iterations=max_iterations,
            max_expansions_per_pass=base_config.max_expansions,
            verbose=True,
        )

        result["era"] = era
        result["ground_truth"] = ground_truth
        result["pattern"] = classify_proof_pattern(result["proof_steps"]) if result["success"] else "failed"
        result["lemma_novelty"] = is_lemma_novelty(result["proof_steps"]) if result["success"] else False

        results.append(result)

        if result["success"]:
            if result["is_newly_found"]:
                newly_found.append(result)
            else:
                pass_0_passed.append(result)
        else:
            never_found.append(result)

        status = "✓" if result["success"] else "✗"
        novelty = " NEW" if result["is_newly_found"] else ""
        eta = (time.time() - t_start) / (i + 1) * (len(theorems) - i - 1)
        print(f"  [{i+1:2d}/{len(theorems)}] {status}{novelty} {name:45s} "
              f"pass={result['pass_found']}/{result['num_passes']} "
              f"{result['search_time_s']:.1f}s  "
              f"ETA: {eta/60:.0f}m  "
              f"(P0:{len(pass_0_passed)} NF:{len(newly_found)})")

    # --- Summary ---
    elapsed = time.time() - t_start
    n_total = len(theorems)
    n_pass0 = len(pass_0_passed)
    n_new = len(newly_found)
    n_total_found = n_pass0 + n_new
    n_missed = n_total - n_total_found
    rate_total = n_total_found / max(1, n_total)

    print(f"\n{'=' * 70}")
    print("PATH 3 RESULTS: Iterative Search with Lean Feedback")
    print(f"{'=' * 70}")
    print(f"  Pass 0 (no feedback):   {n_pass0}/{n_total} ({n_pass0/max(1,n_total):.0%})")
    print(f"  Newly found (iterative): {n_new}/{n_total} ({n_new/max(1,n_total):.0%})")
    print(f"  ─────────────────────────────")
    print(f"  Total found:             {n_total_found}/{n_total} ({rate_total:.0%})")
    print(f"  Not found:               {n_missed}")
    print(f"  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}m)")

    # Iteration distribution
    pass_dist = Counter(r["pass_found"] for r in results if r["success"])
    print(f"\n  Found by pass:")
    for p in sorted(pass_dist.keys()):
        print(f"    Pass {p}: {pass_dist[p]} proofs")

    # Domain breakdown
    domains = Counter(r["domain"] for r in results)
    print(f"\n  By domain:")
    for dom in sorted(domains.keys()):
        dom_total = domains[dom]
        dom_p0 = sum(1 for r in pass_0_passed if r["domain"] == dom)
        dom_nf = sum(1 for r in newly_found if r["domain"] == dom)
        print(f"    {dom:<20} P0:{dom_p0} NF:{dom_nf} "
              f"({dom_p0+dom_nf}/{dom_total} = {(dom_p0+dom_nf)/dom_total*100:.0f}%)")

    # Multi-step and lemma-novelty stats
    multi = [r for r in results if r["success"] and r["num_steps"] >= 2]
    ln = [r for r in results if r["success"] and r["lemma_novelty"]]
    print(f"\n  Multi-step: {len(multi)} ({sum(1 for r in multi if r['is_newly_found'])} newly found)")
    print(f"  Lemma-novelty: {len(ln)} ({sum(1 for r in ln if r['is_newly_found'])} newly found)")

    # --- Build output ---
    out = {
        "task": "PATH 3: Iterative proof search with Lean rejection feedback",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "architecture": "GNN cosine similarity + Iterative Best-first search with Lean feedback",
        "config": {
            "max_iterations": max_iterations,
            "max_depth": base_config.max_depth,
            "max_expansions_per_pass": base_config.max_expansions,
            "top_k_lemmas": base_config.top_k_lemmas,
            "depth_penalty": base_config.depth_penalty,
            "rejection_penalty_initial": base_config.rejection_penalty,
            "num_threads": base_config.num_threads,
            "max_graph_candidates": base_config.max_graph_candidates,
            "gnn_params": sum(p.numel() for p in gnn.parameters()),
            "domain_filtering": use_domain_filter,
        },
        "graph": {
            "num_nodes": graph.num_nodes,
            "num_edges": graph.num_edges,
        },
        "results": {
            "total": n_total,
            "pass_0": n_pass0,
            "newly_found": n_new,
            "total_found": n_total_found,
            "not_found": n_missed,
            "rate_total": rate_total,
            "elapsed_s": elapsed,
            "pass_distribution": dict(pass_dist),
            "domains": {dom: {
                "total": domains[dom],
                "pass_0": sum(1 for r in pass_0_passed if r["domain"] == dom),
                "newly_found": sum(1 for r in newly_found if r["domain"] == dom),
            } for dom in domains},
            "multi_step": {
                "total": len(multi),
                "newly_found": sum(1 for r in multi if r["is_newly_found"]),
            },
            "lemma_novelty": {
                "total": len(ln),
                "newly_found": sum(1 for r in ln if r["is_newly_found"]),
            },
        },
        "theorems": results,
        "newly_found_theorems": [
            {"name": r["name"], "domain": r["domain"], "pass_found": r["pass_found"],
             "proof": " ".join(r["proof_steps"]), "num_steps": r["num_steps"]}
            for r in newly_found
        ],
    }

    save_json(out, output_path)
    print(f"\n  Results saved to: {output_path}")

    # Key metric
    print(f"\n{'─' * 70}")
    print(f"KEY METRIC: Newly found proofs via iteration: {n_new}")
    if n_new > 0:
        print(f"  Proofs found after pass 0 that weren't in first pass:")
        for r in newly_found:
            print(f"    {r['name']} (pass {r['pass_found']}, "
                  f"{r['num_steps']} steps, {r['domain']})")

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="PATH 3: Iterative proof search with Lean rejection feedback"
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
        "--max-iterations", type=int, default=5,
        help="Max iterations of search→verify→re-weight (default: 5)",
    )
    parser.add_argument(
        "--max-expansions", type=int, default=1000,
        help="Max expansions per pass for best-first search (default: 1000)",
    )
    parser.add_argument(
        "--top-k", type=int, default=30,
        help="Top-K lemmas per state (default: 30)",
    )
    parser.add_argument(
        "--depth-penalty", type=float, default=0.05,
        help="Depth penalty factor",
    )
    parser.add_argument(
        "--rejection-penalty", type=float, default=0.3,
        help="Initial rejection penalty (increases with iterations)",
    )
    parser.add_argument(
        "--num-threads", type=int, default=4,
        help="Number of CPU threads (max 4 for eval)",
    )
    parser.add_argument(
        "--output", default="data/iterative_search_result.json",
        help="Output JSON file",
    )
    parser.add_argument(
        "--no-domain-filter", action="store_true",
        help="Disable domain filtering",
    )
    args = parser.parse_args()

    # Hardware constraint
    if args.num_threads > 4:
        print(f"WARNING: Reducing threads from {args.num_threads} to 4 (eval constraint)")
        args.num_threads = 4

    print("=" * 70)
    print("PATH 3: Iterative Proof Search with Lean Rejection Feedback")
    print("=" * 70)
    print(f"Target: {args.max_iterations} iterations on 64 theorems (5 domains)")
    print(f"Threads: {args.num_threads}")
    print(f"Max expansions/pass: {args.max_expansions}")
    print(f"Top-K lemmas: {args.top_k}")
    print(f"Initial rejection penalty: {args.rejection_penalty}")
    print(f"Domain filter: {'OFF' if args.no_domain_filter else 'ON'}")
    print()

    torch.set_num_threads(args.num_threads)
    print(f"PyTorch threads: {torch.get_num_threads()}")

    # --- Load GNN ---
    ckpt_path = _project_path(args.gnn_checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        return 1

    gnn = GNNEncoder.load(str(ckpt_path))
    gnn.eval()
    n_params = sum(p.numel() for p in gnn.parameters())
    print(f"GNN: {n_params:,} params, hidden={gnn.config.hidden_dim}")

    # --- Load graph ---
    graph_path = _project_path(args.graph)
    if not graph_path.with_suffix(".nx.pkl").exists():
        print(f"ERROR: Graph not found: {graph_path}.nx.pkl")
        return 1

    graph = DependencyGraph.load(graph_path)
    print(f"Graph: {graph.summary()}")

    # --- Load theorems ---
    theorems_path = _project_path(args.theorems)
    if not theorems_path.exists():
        print(f"ERROR: Theorems not found: {theorems_path}")
        return 1

    theorems = load_jsonl(theorems_path)
    print(f"Theorems loaded: {len(theorems)}")

    # Only test first N theorems if running a quick test
    # (uncomment for dev: theorems = theorems[:10])
    # theorems = theorems[:10]  # DEV: quick test

    # --- Indexes ---
    lemma_to_idx = build_lemma_index(graph)
    idx_to_norm = build_norm_index(graph, lemma_to_idx)
    print(f"Lemma index: {len(lemma_to_idx)} entries")

    # --- Config ---
    base_config = GNNBestFirstConfig(
        max_depth=20,
        max_expansions=args.max_expansions,
        top_k_lemmas=args.top_k,
        depth_penalty=args.depth_penalty,
        use_proof_checker=True,
        verify_timeout=5.0,
        num_threads=args.num_threads,
        max_graph_candidates=200,
        rejection_penalty=args.rejection_penalty,
    )

    checker = BatchChecker(timeout=15, max_workers=8, cache_size=128)
    output_path = _project_path(args.output)

    # --- Run ---
    result = run_iterative_gate3(
        gnn=gnn,
        graph=graph,
        theorems=theorems,
        base_config=base_config,
        lemma_to_idx=lemma_to_idx,
        idx_to_norm=idx_to_norm,
        checker=checker,
        output_path=output_path,
        max_iterations=args.max_iterations,
        use_domain_filter=not args.no_domain_filter,
    )

    n_p0 = result["results"]["pass_0"]
    n_nf = result["results"]["newly_found"]
    n_tot = result["results"]["total_found"]
    rate = result["results"]["rate_total"]

    print(f"\n{'=' * 70}")
    print("FINAL")
    print(f"{'=' * 70}")
    print(f"  Pass 0 proofs:     {n_p0}")
    print(f"  Newly found (iter): {n_nf}")
    print(f"  Total found:       {n_tot}/64 ({rate:.0%})")
    print(f"  Output:            {output_path}")

    if n_nf > 0:
        print(f"\n  ★ PATH 3 SUCCESS: {n_nf} proofs newly found via iteration")
    else:
        print(f"\n  No new proofs found via iteration (all found in pass 0 or not at all)")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
