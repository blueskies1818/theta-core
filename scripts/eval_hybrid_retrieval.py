#!/usr/bin/env python3
"""Evaluate hybrid GNN-powered best-first search on gate3_v2 theorems.

Replaces CharCNN contrastive embeddings with GNN cosine similarity (MRR 0.786)
while keeping the best-first priority-queue architecture.

Tests:
  - Proof success rate on gate3_v2 (64 multi-step theorems)
  - Multi-step proof count
  - Per-domain breakdown
  - Comparison to CharCNN baseline (0%)

Usage:
    python scripts/eval_hybrid_retrieval.py

Output:
    data/hybrid_retrieval_result.json
"""

import sys, json, time, re
from pathlib import Path
from collections import Counter

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

import torch
import torch.nn.functional as F

from src.explorer.dependency_graph import DependencyGraph
from src.explorer.gnn_config import GNNConfig
from src.explorer.gnn_encoder import (
    GNNEncoder,
    extract_initial_features,
    prepare_graph_tensors,
)
from src.explorer.gnn_best_first_search import GNNBestFirstSearch, GNNBestFirstConfig
from src.explorer.proof_state import ProofState, Tactic
from src.proof_checker.batch_checker import BatchChecker
from src.proof_checker.formats import wrap_theorem_with_proof
from scripts.eval_gnn_prover import (
    build_lemma_index,
    build_lemma_norm_index,
    extract_conclusion,
    normalize_expression,
)


# ---------------------------------------------------------------------------
# Proof pattern classification
# ---------------------------------------------------------------------------

def classify_proof_pattern(proof_steps: list[str]) -> str:
    """Classify a proof into its primary pattern category."""
    if not proof_steps:
        return "empty"

    steps_text = " ".join(proof_steps).lower()
    tactic_types = set()

    for step in proof_steps:
        s = step.strip().lower()
        if s.startswith("rw"):
            tactic_types.add("rw")
        elif s.startswith("exact"):
            tactic_types.add("exact")
        elif s.startswith("apply"):
            tactic_types.add("apply")
        elif s.startswith("intro"):
            tactic_types.add("intro")
        elif s.startswith("have"):
            tactic_types.add("have")
        elif s in ("ring", "simp", "linarith", "field_simp", "positivity",
                    "norm_num", "nlinarith"):
            tactic_types.add(s)
        elif s.startswith("calc"):
            tactic_types.add("calc")
        elif s.startswith("constructor"):
            tactic_types.add("constructor")
        elif s.startswith("refine"):
            tactic_types.add("refine")
        else:
            tactic_types.add("other")

    if len(tactic_types) >= 2:
        return "multi"

    if any(tok in steps_text for tok in ("rfl", "eq.refl")):
        return "rfl"
    if "add_comm" in steps_text:
        return "add_comm"
    if "mul_comm" in steps_text:
        return "mul_comm"
    if "ring" in steps_text:
        return "ring"
    if "field_simp" in steps_text:
        return "field_simp"
    if "linarith" in steps_text:
        return "linarith"
    if "simp" in steps_text:
        return "simp"
    if "intro" in steps_text:
        return "intro"
    if "apply" in steps_text:
        return "apply"
    if "nlinarith" in steps_text:
        return "nlinarith"
    return "other"


# ---------------------------------------------------------------------------
# Lemma index building helpers
# ---------------------------------------------------------------------------

def build_norm_index(graph: DependencyGraph, lemma_to_idx: dict[str, int]) -> dict[int, str]:
    """Build normalized lemma conclusion index: idx → normalized conclusion."""
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


def build_lemma_index_from_graph(graph: DependencyGraph) -> dict[str, int]:
    """Build lemma index: lemma_name → node integer index."""
    return build_lemma_index(graph)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def load_theorems(path: Path, max_theorems: int | None = None) -> list[dict]:
    """Load theorems from JSONL file."""
    with open(path) as f:
        theorems = [json.loads(line) for line in f]
    if max_theorems:
        theorems = theorems[:max_theorems]
    return theorems


def run_hybrid_search(
    bf_search: GNNBestFirstSearch,
    theorems: list[dict],
    checker: BatchChecker,
    verbose: bool = False,
) -> list[dict]:
    """Run GNN-powered best-first search on all theorems."""
    results = []
    t_start = time.time()

    for i, t in enumerate(theorems):
        stmt = t["statement"]
        name = t["name"]
        era = t.get("era", "unknown")
        zone = t.get("frontier_zone", "unknown")
        ground_truth = t.get("proof", "?")
        domain = t.get("domain", "unknown")

        t0 = time.time()
        proof_steps, final_state = bf_search.search(stmt, verbose=False)
        search_time = time.time() - t0

        proof_text = ProofState._render_proof(proof_steps)

        if not proof_steps:
            ok = False
            err = "no proof found"
        else:
            full_code = wrap_theorem_with_proof(stmt, proof_text)
            check_results = checker.check_batch([full_code])
            ok = check_results[0].success
            err = check_results[0].errors[0][:200] if check_results[0].errors else ""

        steps_str = [s.to_lean() for s in proof_steps[:10]]
        pattern = classify_proof_pattern(steps_str) if ok else "failed"

        result = {
            "name": name,
            "era": era,
            "zone": zone,
            "domain": domain,
            "success": ok,
            "error": err,
            "hybrid_steps": steps_str,
            "num_steps": len(proof_steps),
            "ground_truth": ground_truth,
            "search_time_s": search_time,
            "pattern": pattern,
        }
        results.append(result)

        status = "\u2713" if ok else "\u2717"
        print(f"  [{i+1:2d}/{len(theorems)}] {status} {name:45s} "
              f"[{pattern:12s}] {search_time:.1f}s  "
              f"{len(proof_steps)} steps")
        if ok:
            print(f"         Proof: {steps_str}")
        elif verbose and err:
            print(f"         Error: {err[:120]}")

    elapsed = time.time() - t_start
    passed = sum(1 for r in results if r["success"])
    print(f"\nHybrid result: {passed}/{len(results)} "
          f"({passed/max(1,len(results))*100:.0f}%) in {elapsed:.0f}s")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate hybrid GNN-powered best-first search on gate3_v2"
    )
    parser.add_argument(
        "--gnn-checkpoint",
        default="checkpoints/gnn/gate2_fullgraph_finetuned.pt",
        help="Path to GNN checkpoint",
    )
    parser.add_argument(
        "--graph",
        default="data/graph/dependency_graph_full",
        help="Path to dependency graph (without extension)",
    )
    parser.add_argument(
        "--theorems",
        default="data/raw/gate3_v2.jsonl",
        help="Path to theorem JSONL file",
    )
    parser.add_argument(
        "--max-theorems",
        type=int,
        default=None,
        help="Max theorems to test (None = all)",
    )
    parser.add_argument(
        "--max-expansions",
        type=int,
        default=5000,
        help="Max node expansions for best-first search",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=30,
        help="Top-K lemmas to consider per state",
    )
    parser.add_argument(
        "--depth-penalty",
        type=float,
        default=0.05,
        help="Depth penalty factor for priority",
    )
    parser.add_argument(
        "--use-proof-checker",
        action="store_true",
        help="Verify candidates with Lean during expansion",
    )
    parser.add_argument(
        "--max-graph-candidates",
        type=int,
        default=200,
        help="Max graph candidates per theorem",
    )
    parser.add_argument(
        "--output",
        default="data/hybrid_retrieval_result.json",
        help="Output JSON file",
    )
    parser.add_argument(
        "--algebra-only",
        action="store_true",
        help="Limit search to Algebra domain subgraph",
    )
    args = parser.parse_args()

    ckpt_path = _project_root / args.gnn_checkpoint
    graph_path = _project_root / args.graph
    theorems_path = _project_root / args.theorems
    output_path = _project_root / args.output

    print("=" * 70)
    print("HYBRID: GNN Cosine Retrieval + Best-First Search")
    print("=" * 70)
    print(f"GNN checkpoint: {ckpt_path}")
    print(f"Graph:          {graph_path}")
    print(f"Theorems:       {theorems_path}")

    # ---- Load GNN ----
    if not ckpt_path.exists():
        print(f"ERROR: GNN checkpoint not found: {ckpt_path}")
        return 1

    print("Loading GNN...")
    torch.set_num_threads(4)
    gnn = GNNEncoder.load(str(ckpt_path))
    gnn.eval()
    n_params = sum(p.numel() for p in gnn.parameters())
    print(f"  Loaded: {n_params:,} params, hidden={gnn.config.hidden_dim}")

    # ---- Load graph ----
    if not graph_path.with_suffix(".nx.pkl").exists():
        print(f"ERROR: Graph not found: {graph_path}.nx.pkl")
        return 1

    print(f"Loading graph from {graph_path}...")
    graph = DependencyGraph.load(str(graph_path))
    print(f"  {graph.summary()}")

    # ---- Optionally filter to Algebra domain ----
    if args.algebra_only:
        print("Filtering to Algebra domain...")
        graph = graph.domain_subgraph("Algebra")
        print(f"  {graph.summary()}")

    # ---- Compute GNN node embeddings ----
    print("Computing GNN node embeddings...")
    features = extract_initial_features(graph, gnn.config)
    sources, targets, edge_types, num_nodes = prepare_graph_tensors(graph)
    print(f"  Graph: {num_nodes} nodes, {sources.size(0)} edges")

    with torch.no_grad():
        node_embeddings = gnn(features, sources, targets, edge_types, num_nodes)
    print(f"  Embeddings: {node_embeddings.shape}")

    # ---- Build lemma index and normalized conclusions ----
    print("Building lemma index...")
    lemma_to_idx = build_lemma_index_from_graph(graph)
    idx_to_norm = build_norm_index(graph, lemma_to_idx)
    print(f"  Lemma index: {len(lemma_to_idx)} entries")
    print(f"  Norm index:  {len(idx_to_norm)} entries")

    # ---- Load theorems ----
    if not theorems_path.exists():
        print(f"ERROR: Theorem file not found: {theorems_path}")
        return 1

    theorems = load_theorems(theorems_path, args.max_theorems)
    print(f"Theorems: {len(theorems)} loaded")

    # ---- Setup search ----
    config = GNNBestFirstConfig(
        max_depth=20,
        max_expansions=args.max_expansions,
        top_k_lemmas=args.top_k,
        depth_penalty=args.depth_penalty,
        use_proof_checker=args.use_proof_checker,
        verify_timeout=5.0,
        num_threads=4,
        max_graph_candidates=args.max_graph_candidates,
    )

    checker = BatchChecker(timeout=30, max_workers=1, cache_size=128)

    bf_search = GNNBestFirstSearch(
        gnn=gnn,
        graph=graph,
        node_embeddings=node_embeddings,
        lemma_index=lemma_to_idx,
        idx_to_norm=idx_to_norm,
        config=config,
        proof_checker=checker if args.use_proof_checker else None,
    )

    # ---- Run search ----
    print()
    print("-" * 70)
    print("Running GNN-powered best-first search on gate3_v2...")
    print("-" * 70)
    t0 = time.time()
    results = run_hybrid_search(bf_search, theorems, checker, verbose=False)
    total_elapsed = time.time() - t0

    # ---- Aggregate results ----
    passed = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    n_total = len(results)

    multi = [r for r in passed if r["pattern"] == "multi"]
    single = [r for r in passed if r["pattern"] != "multi"]

    lemma_novelty = [
        r for r in passed
        if any(kw in r["name"].lower() for kw in ("poly_", "ana_", "alg_"))
        and "simp" not in " ".join(r["hybrid_steps"]).lower()
    ]

    print()
    print("=" * 70)
    print("RESULTS: GNN Best-First (Hybrid) on gate3_v2")
    print("=" * 70)
    print(f"  Theorems tested:     {n_total}")
    print(f"  Total passed:        {len(passed)} ({len(passed)/max(1,n_total)*100:.1f}%)")
    print(f"  Multi-step proofs:   {len(multi)}")
    print(f"  Single-step proofs:  {len(single)}")
    print(f"  Lemma-novelty proofs:{len(lemma_novelty)}")
    print(f"  Failed:              {len(failed)}")
    print(f"  Total time:          {total_elapsed:.0f}s")

    patterns = Counter(r["pattern"] for r in passed)
    print(f"\n  Proof patterns (successful):")
    for pat, count in patterns.most_common():
        print(f"    {pat:<15} {count:>3}")

    domains = Counter(r["domain"] for r in results)
    print(f"\n  By domain:")
    for dom in sorted(domains.keys()):
        dom_total = domains[dom]
        dom_passed = sum(1 for r in passed if r["domain"] == dom)
        print(f"    {dom:<20} {dom_passed}/{dom_total} "
              f"({dom_passed/max(1,dom_total)*100:.0f}%)")

    if multi:
        print(f"\n  Multi-step proofs ({len(multi)}):")
        for r in multi:
            print(f"    ✓ {r['name']:<45s} [{r['pattern']}] "
                  f"→ {r['hybrid_steps']}")
            print(f"      ground_truth: {r['ground_truth']}")

    # ---- Comparison with baseline ----
    print()
    print("=" * 70)
    print("COMPARISON: Hybrid (GNN+Best-First) vs Baselines")
    print("=" * 70)

    # CharCNN baseline
    charcnn_passed = 0  # Known to be 0% on gate3_v2
    charcnn_rate = 0.0

    # MCTS GNN baseline (from gate3_fullgraph_result.json)
    mcts_path = _project_root / "data/gate3_fullgraph_result.json"
    mcts_gnn_rate = 0.0
    mcts_gnn_passed = 0
    if mcts_path.exists():
        with open(mcts_path) as f:
            mcts_data = json.load(f)
        mcts_algebra = mcts_data.get("results_algebra_subgraph", {})
        mcts_gnn_passed = mcts_algebra.get("gnn_h0_proved", 0)
        mcts_gnn_rate = mcts_algebra.get("gnn_h0_rate", 0.0)

    hybrid_passed = len(passed)
    hybrid_rate = hybrid_passed / max(1, n_total)

    print(f"\n  {'':30} {'Proved':>10} {'Rate':>10} {'Multi-step':>12}")
    print(f"  {'-'*65}")
    print(f"  {'Hybrid (GNN+Best-First)':30} "
          f"{hybrid_passed:>10} {hybrid_rate:>9.0%} "
          f"{len(multi):>12}")
    print(f"  {'CharCNN (Best-First)':30} "
          f"{charcnn_passed:>10} {charcnn_rate:>9.0%} "
          f"{'---':>12}")
    print(f"  {'MCTS GNN (gate3 original)':30} "
          f"{mcts_gnn_passed:>10} {mcts_gnn_rate:>9.0%} "
          f"{'---':>12}")

    # ---- Win condition check ----
    print()
    print("=" * 70)
    print("WIN CONDITION CHECK")
    print("=" * 70)
    win_lemma_novelty = len(lemma_novelty) > 0
    win_proof_success = hybrid_rate > charcnn_rate

    print(f"  Lemma-novelty proofs > 0:  {'PASS' if win_lemma_novelty else 'FAIL'} "
          f"({len(lemma_novelty)} proofs)")
    print(f"  Proof success > CharCNN (0%): {'PASS' if win_proof_success else 'FAIL'} "
          f"({hybrid_rate*100:.0f}% vs {charcnn_rate*100:.0f}%)")
    print(f"  Overall: {'PASS' if win_lemma_novelty and win_proof_success else 'FAIL'}")

    # ---- Save results ----
    output_data = {
        "task": "HYBRID: Wire GNN cosine retrieval into best-first search",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": {
            "checkpoint": str(ckpt_path),
            "architecture": "GNN (GAT) with GoalEncoder — cosine similarity retrieval",
            "params": n_params,
            "hidden_dim": gnn.config.hidden_dim,
        },
        "graph": {
            "path": str(graph_path),
            "num_nodes": graph.num_nodes,
            "num_edges": graph.num_edges,
            "algebra_only": args.algebra_only,
        },
        "config": {
            "max_expansions": args.max_expansions,
            "top_k_lemmas": args.top_k,
            "depth_penalty": args.depth_penalty,
            "use_proof_checker": args.use_proof_checker,
            "max_graph_candidates": args.max_graph_candidates,
        },
        "test_config": {
            "theorems": str(theorems_path),
            "num_theorems": n_total,
        },
        "results": {
            "total": n_total,
            "passed": hybrid_passed,
            "rate": hybrid_rate,
            "multi_step_passed": len(multi),
            "multi_step_rate": len(multi) / max(1, n_total),
            "lemma_novelty_passed": len(lemma_novelty),
            "single_step_passed": len(single),
            "failed": len(failed),
            "patterns": dict(patterns),
            "passed_theorems": [
                {
                    "name": r["name"],
                    "proof": " ".join(r["hybrid_steps"]),
                    "pattern": r["pattern"],
                    "ground_truth": r["ground_truth"],
                    "num_steps": r["num_steps"],
                    "search_time_s": r["search_time_s"],
                    "domain": r["domain"],
                    "era": r["era"],
                }
                for r in passed
            ],
            "all_results": results,
        },
        "comparison": {
            "hybrid_rate": hybrid_rate,
            "hybrid_passed": hybrid_passed,
            "hybrid_multi_step": len(multi),
            "charcnn_baseline_rate": charcnn_rate,
            "charcnn_baseline_passed": charcnn_passed,
            "mcts_gnn_baseline_rate": mcts_gnn_rate,
            "mcts_gnn_baseline_passed": mcts_gnn_passed,
            "note": (
                "Hybrid replaces CharCNN contrastive embeddings with GNN cosine similarity "
                "(MRR 0.786) while keeping best-first search architecture. "
                "CharCNN baseline: 0% on gate3_v2. "
                "MCTS GNN baseline: 28.6% on gate3_lemma_novelty (14 theorems, Algebra subgraph). "
                "gate3_v2 has 64 theorems, most requiring ≥2 tactics — harder than gate3_lemma_novelty."
            ),
        },
        "win_conditions": {
            "lemma_novelty_proofs_gt_0": win_lemma_novelty,
            "proof_success_gt_charcnn_baseline": win_proof_success,
            "overall_pass": win_lemma_novelty and win_proof_success,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Return code: 0 if pass, 1 if fail
    return 0 if (win_lemma_novelty and win_proof_success) else 1


if __name__ == "__main__":
    sys.exit(main())
