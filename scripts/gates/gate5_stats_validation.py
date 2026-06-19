#!/usr/bin/env python3
"""
Gate 5 Statistical Validation: Multi-replicate evaluation of hybrid architecture
on gate3_v2 (64 theorems). Runs ≥3 independent replicates with 12 CPU threads each.

Reports mean ± std of proof success rate. Std target < 3pp.
If all replicates show consistent Gate 3 pass → tag v1.0.

Usage:
    python scripts/gates/gate5_stats_validation.py [--replicates 3] [--max-expansions 5000]
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import torch

from src.explorer.dependency_graph import DependencyGraph
from src.explorer.gnn_config import GNNConfig
from src.explorer.gnn_encoder import (
    GNNEncoder,
    extract_initial_features,
    prepare_graph_tensors,
)
from src.explorer.gnn_best_first_search import GNNBestFirstSearch, GNNBestFirstConfig
from src.explorer.proof_state import ProofState
from src.proof_checker.batch_checker import BatchChecker
from src.proof_checker.formats import wrap_theorem_with_proof
from scripts.gates.hybrid_gates import (
    build_lemma_index,
    build_norm_index,
    classify_proof_pattern,
    is_lemma_novelty,
    load_jsonl,
)


def run_single_replicate(
    gnn: GNNEncoder,
    graph,
    theorems: list[dict],
    config: GNNBestFirstConfig,
    lemma_to_idx: dict[str, int],
    idx_to_norm: dict[int, str],
    replicate_id: int,
    verbose: bool = True,
) -> dict:
    """Run one full evaluation replicate on all theorems."""
    label = f"[Replicate {replicate_id}]"
    print(f"\n{'='*60}")
    print(f"  {label} Starting evaluation on {len(theorems)} theorems")
    print(f"{'='*60}")
    print(f"  Max expansions: {config.max_expansions}, Top-K: {config.top_k_lemmas}")
    print(f"  CPU threads: {torch.get_num_threads()}")

    # Compute node embeddings fresh for this replicate (though deterministic)
    features = extract_initial_features(graph, gnn.config)
    sources, targets, edge_types, num_nodes = prepare_graph_tensors(graph)

    with torch.no_grad():
        node_embeddings = gnn(features, sources, targets, edge_types, num_nodes)

    # Fresh checker and search instance per replicate
    checker = BatchChecker(timeout=30, max_workers=1, cache_size=128)
    bf_search = GNNBestFirstSearch(
        gnn=gnn,
        graph=graph,
        node_embeddings=node_embeddings,
        lemma_index=lemma_to_idx,
        idx_to_norm=idx_to_norm,
        config=config,
        proof_checker=checker if config.use_proof_checker else None,
    )

    results = []
    t_start = time.time()
    passed = []
    failed_reasons: dict[str, int] = {}

    for i, t in enumerate(theorems):
        stmt = t["statement"]
        name = t["name"]
        domain = t.get("domain", "unknown")
        era = t.get("era", "unknown")
        ground_truth = t.get("proof", "?")

        t0 = time.time()
        proof_steps, final_state = bf_search.search(stmt, verbose=False)
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
                reason_key = f"lean_reject:{err[:50]}"
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
        eta = (time.time() - t_start) / (i + 1) * (len(theorems) - i - 1)
        pct = (i + 1) / len(theorems) * 100
        print(
            f"  [{i+1:2d}/{len(theorems)}] {status} {name:40s} "
            f"[{pattern:12s}] {search_time:.1f}s  "
            f"{pct:.0f}%  ({len(passed)} passed)  ETA: {eta/60:.0f}m"
        )

        if ok and len(proof_steps) > 0:
            print(f"         Proof: {steps_str}")
            if len(proof_steps) >= 2:
                print(f"         ** MULTI-STEP ({len(proof_steps)} steps)")

    elapsed = time.time() - t_start
    n_total = len(theorems)
    n_passed = len(passed)
    rate = n_passed / max(1, n_total)

    multi = [r for r in passed if r["num_steps"] >= 2]
    lemma_novel = [r for r in passed if r["lemma_novelty"]]
    structural = [r for r in passed if not r["lemma_novelty"]]

    # Domain breakdown
    domains = Counter(r["domain"] for r in results)
    domain_stats = {}
    for dom in sorted(domains.keys()):
        dom_total = domains[dom]
        dom_passed = sum(1 for r in passed if r["domain"] == dom)
        dom_ln = sum(1 for r in lemma_novel if r["domain"] == dom)
        dom_multi = sum(1 for r in multi if r["domain"] == dom)
        domain_stats[dom] = {
            "total": dom_total,
            "passed": dom_passed,
            "rate": dom_passed / max(1, dom_total),
            "lemma_novelty": dom_ln,
            "multi_step": dom_multi,
        }

    print(f"\n  --- {label} Results ---")
    print(f"  Total:    {n_passed}/{n_total} ({rate:.0%})")
    print(f"  Multi-step: {len(multi)}")
    print(f"  Lemma-novelty: {len(lemma_novel)}")
    print(f"  Structural-only: {len(structural)}")
    print(f"  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}m)")

    try:
        checker.shutdown()
    except Exception:
        pass

    return {
        "replicate_id": replicate_id,
        "total": n_total,
        "passed": n_passed,
        "rate": rate,
        "multi_step": len(multi),
        "lemma_novelty": len(lemma_novel),
        "structural_only": len(structural),
        "elapsed_s": elapsed,
        "failed_reasons": dict(failed_reasons),
        "domains": domain_stats,
        "passed_theorems": [
            {
                "name": r["name"],
                "proof": " ".join(r["hybrid_steps"]),
                "pattern": r["pattern"],
                "num_steps": r["num_steps"],
                "domain": r["domain"],
                "lemma_novelty": r["lemma_novelty"],
            }
            for r in passed
        ],
        "all_results": results,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Gate 5 Statistical Validation: Multi-replicate hybrid evaluation"
    )
    parser.add_argument(
        "--replicates", type=int, default=3,
        help="Number of independent evaluation replicates (default: 3)",
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
        help="Theorem JSONL for Gate 3 evaluation",
    )
    parser.add_argument(
        "--max-expansions", type=int, default=5000,
        help="Max expansions for best-first search",
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
        help="Number of CPU threads for PyTorch (default: 12)",
    )
    parser.add_argument(
        "--algebra-only", action="store_true",
        help="Limit to Algebra domain subgraph (faster, 26 theorems)",
    )
    parser.add_argument(
        "--max-theorems", type=int, default=None,
        help="Max theorems to evaluate per replicate",
    )
    parser.add_argument(
        "--output", default="data/gate5_stats_validation.json",
        help="Output JSON file",
    )
    args = parser.parse_args()

    # ── Hardware config ──────────────────────────────────────────────────
    torch.set_num_threads(args.num_threads)
    print(f"PyTorch threads: {torch.get_num_threads()}")

    # ── Load GNN (once) ──────────────────────────────────────────────────
    ckpt_path = _PROJECT_ROOT / args.gnn_checkpoint
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        return 1

    print(f"\nLoading GNN: {ckpt_path}")
    gnn = GNNEncoder.load(str(ckpt_path))
    gnn.eval()
    n_params = sum(p.numel() for p in gnn.parameters())
    print(f"  {n_params:,} params, hidden={gnn.config.hidden_dim}, "
          f"layers={gnn.config.num_layers}")

    # ── Load graph ───────────────────────────────────────────────────────
    graph_path = _PROJECT_ROOT / args.graph
    if not graph_path.with_suffix(".nx.pkl").exists():
        print(f"ERROR: Graph not found: {graph_path}.nx.pkl")
        return 1

    print(f"\nLoading graph: {graph_path}")
    graph = DependencyGraph.load(graph_path)
    print(f"  {graph.summary()}")

    # ── Load theorems ────────────────────────────────────────────────────
    theorems_path = _PROJECT_ROOT / args.theorems
    if not theorems_path.exists():
        print(f"ERROR: Theorems not found: {theorems_path}")
        return 1

    theorems = load_jsonl(theorems_path)
    print(f"\nTheorems: {len(theorems)} from {args.theorems}")

    # -- Algebra-only: filter graph and theorems to Algebra domain --
    if args.algebra_only:
        print("  Algebra-only mode: filtering graph and theorems to Algebra domain")
        graph = graph.domain_subgraph("Algebra")
        print(f"  Graph filtered: {graph.summary()}")

        # Also filter theorems to algebra-related domains
        algebra_domains = {"algebra", "Algebra"}
        theorems = [t for t in theorems if t.get("domain", "") in algebra_domains]
        print(f"  Theorems filtered: {len(theorems)} algebra theorems")

        # Rebuild lemma index on filtered graph
        lemma_to_idx = build_lemma_index(graph)
        idx_to_norm = build_norm_index(graph, lemma_to_idx)
        print(f"  Lemma index rebuilt: {len(lemma_to_idx)} entries")
    else:
        pass  # lemma_to_idx and idx_to_norm already built above

    # -- Max theorems cap --
    if args.max_theorems:
        theorems = theorems[:args.max_theorems]
        print(f"  Capped to {len(theorems)} theorems")

    # Domain/era overview
    domains = Counter(t.get("domain", "?") for t in theorems)
    eras = Counter(t.get("era", "?") for t in theorems)
    print(f"  Domains: {dict(domains)}")
    print(f"  Eras: {dict(eras)}")

    # ── Build lemma index ─────────────────────────────────────────────────
    lemma_to_idx = build_lemma_index(graph)
    idx_to_norm = build_norm_index(graph, lemma_to_idx)
    print(f"\nLemma index: {len(lemma_to_idx)} entries")

    # ── Search config ────────────────────────────────────────────────────
    config = GNNBestFirstConfig(
        max_depth=20,
        max_expansions=args.max_expansions,
        top_k_lemmas=args.top_k,
        depth_penalty=args.depth_penalty,
        use_proof_checker=True,
        verify_timeout=5.0,
        num_threads=args.num_threads,
        max_graph_candidates=200,
    )

    # ── Run replicates ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"GATE 5 STATISTICAL VALIDATION: {args.replicates} replicates")
    print("=" * 70)

    all_replicates = []
    t_total_start = time.time()

    for rep_id in range(1, args.replicates + 1):
        rep_result = run_single_replicate(
            gnn=gnn,
            graph=graph,
            theorems=theorems,
            config=config,
            lemma_to_idx=lemma_to_idx,
            idx_to_norm=idx_to_norm,
            replicate_id=rep_id,
            verbose=True,
        )
        all_replicates.append(rep_result)

    t_total = time.time() - t_total_start

    # ── Statistical summary ──────────────────────────────────────────────
    rates = [r["rate"] for r in all_replicates]
    passed_counts = [r["passed"] for r in all_replicates]
    multi_counts = [r["multi_step"] for r in all_replicates]
    ln_counts = [r["lemma_novelty"] for r in all_replicates]

    mean_rate = statistics.mean(rates)
    std_rate = statistics.stdev(rates) if len(rates) > 1 else 0.0
    mean_pct = mean_rate * 100
    std_pp = std_rate * 100

    mean_passed = statistics.mean(passed_counts)
    std_passed = statistics.stdev(passed_counts) if len(passed_counts) > 1 else 0.0

    print("\n" + "=" * 70)
    print("STATISTICAL SUMMARY")
    print("=" * 70)
    print(f"  Replicates: {len(all_replicates)}")
    print(f"  Theorems per replicate: {len(theorems)}")
    print()
    print(f"  Proof success:  {mean_passed:.1f} ± {std_passed:.1f} / {len(theorems)} "
          f"({mean_pct:.1f}% ± {std_pp:.1f}pp)")
    print(f"  Multi-step:     {statistics.mean(multi_counts):.1f} ± "
          f"{statistics.stdev(multi_counts) if len(multi_counts) > 1 else 0:.1f}")
    print(f"  Lemma-novelty:  {statistics.mean(ln_counts):.1f} ± "
          f"{statistics.stdev(ln_counts) if len(ln_counts) > 1 else 0:.1f}")
    print()

    # Gate 3 pass: at least 1 proof in all replicates
    all_pass_gate3 = all(r["passed"] > 0 for r in all_replicates)
    std_target_met = std_pp < 3.0

    print(f"  Gate 3 pass (all reps > 0): {'PASS' if all_pass_gate3 else 'FAIL'}")
    print(f"  Std < 3pp:                  {'PASS' if std_target_met else 'FAIL'} "
          f"({std_pp:.2f}pp)")

    gate5_overall = all_pass_gate3 and std_target_met
    print(f"\n  Gate 5 overall: {'PASS' if gate5_overall else 'FAIL'}")
    print(f"  Tag v1.0:       {'YES' if gate5_overall else 'NO'}")
    print(f"\n  Total elapsed: {t_total:.0f}s ({t_total/60:.1f}m)")

    # ── Save results ────────────────────────────────────────────────────
    output_path = _PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    final_results = {
        "task": "Gate 5 Statistical Validation — Hybrid Architecture",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "gnn_checkpoint": args.gnn_checkpoint,
            "graph": args.graph,
            "theorems": args.theorems,
            "max_expansions": args.max_expansions,
            "top_k_lemmas": args.top_k,
            "depth_penalty": args.depth_penalty,
            "num_threads": args.num_threads,
            "n_replicates": args.replicates,
        },
        "statistics": {
            "n_replicates": len(all_replicates),
            "n_theorems": len(theorems),
            "mean_rate": mean_rate,
            "std_rate": std_rate,
            "mean_pct": mean_pct,
            "std_pp": std_pp,
            "mean_passed": mean_passed,
            "std_passed": std_passed,
            "mean_multi_step": statistics.mean(multi_counts),
            "mean_lemma_novelty": statistics.mean(ln_counts),
            "std_target_met": std_target_met,
            "gate3_all_pass": all_pass_gate3,
            "gate5_overall": gate5_overall,
            "tag_v1_0": gate5_overall,
        },
        "per_replicate": all_replicates,
        "elapsed_total_s": t_total,
    }

    with open(output_path, "w") as f:
        json.dump(final_results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    return 0 if gate5_overall else 1


if __name__ == "__main__":
    sys.exit(main())
