#!/usr/bin/env python3
"""Evaluate fine-tuned GNN on gate3_v2 benchmark.

Loads a fine-tuned GNN checkpoint (full model, not adapter-only),
runs best-first search on all 64 gate3_v2 theorems. Compares to
the pre-trained GNN baseline.

No adapter — the GNN itself has been fine-tuned with proof-checker
rejection feedback to reshape its embedding space.

Usage:
    # Smoke test on 5 theorems
    python scripts/eval/eval_gnn_adapter.py \
        --gnn-checkpoint data/gnn_ft_full/gnn_fine_tuned.pt \
        --output data/gnn_ft_gate3_result.json \
        --max-theorems 5

    # Full eval
    python scripts/eval/eval_gnn_adapter.py \
        --gnn-checkpoint data/gnn_ft_full/gnn_fine_tuned.pt \
        --output data/gnn_ft_gate3_result.json

    # Statistical validation (Gate 5, 3 replicates)
    python scripts/eval/eval_gnn_adapter.py \
        --gnn-checkpoint data/gnn_ft_full/gnn_fine_tuned.pt \
        --output data/gnn_ft_gate5_result.json \
        --repeat 3

    # Compare against baseline
    python scripts/eval/eval_gnn_adapter.py \
        --gnn-checkpoint data/gnn_ft_full/gnn_fine_tuned.pt \
        --compare \
        --baseline checkpoints/gnn/full_graph_pretrained.pt
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))

import torch
import torch.nn.functional as F

from src.explorer.dependency_graph import DependencyGraph
from src.explorer.gnn_encoder import (
    GNNEncoder,
    extract_initial_features,
    prepare_graph_tensors,
)
from src.explorer.gnn_best_first_search import GNNBestFirstSearch, GNNBestFirstConfig
from src.explorer.proof_state import ProofState
from src.proof_checker.batch_checker import BatchChecker
from src.proof_checker.formats import wrap_theorem_with_proof
from scripts.eval.eval_gnn_prover import (
    build_lemma_index,
    extract_conclusion,
    normalize_expression,
)
from scripts.eval.run_full_gate3_v2 import (
    build_norm_index,
    classify_proof_pattern,
    is_lemma_novelty,
    load_jsonl,
    save_json,
)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def run_gate3_with_finetuned_gnn(
    gnn: GNNEncoder,
    graph: DependencyGraph,
    theorems: list[dict],
    config: GNNBestFirstConfig,
    lemma_to_idx: dict[str, int],
    idx_to_norm: dict[int, str],
    checker: BatchChecker | None,
    output_path: Path,
) -> dict:
    """Run gate3_v2 benchmark with fine-tuned GNN.

    Architecture:
      1. Compute GNN node embeddings
      2. Use GNN directly for goal encoding and lemma scoring
      3. Run standard best-first search
    """
    print("\n" + "=" * 70)
    print("GATE 3: Fine-tuned GNN on gate3_v2 (64 theorems)")
    print("=" * 70)

    # Compute GNN node embeddings
    print("\nComputing GNN node embeddings...")
    gnn.eval()
    features = extract_initial_features(graph, gnn.config)
    sources, targets, edge_types, num_nodes = prepare_graph_tensors(graph)
    print(f"  Graph: {num_nodes} nodes, {sources.size(0)} edges")

    with torch.no_grad():
        node_embeddings = gnn(features, sources, targets, edge_types, num_nodes)
        node_embeddings = F.normalize(node_embeddings, dim=-1)
    print(f"  Embeddings: {node_embeddings.shape}")

    # Check embedding health
    _check_embedding_health(node_embeddings)

    # Setup search
    bf_search = GNNBestFirstSearch(
        gnn=gnn,
        graph=graph,
        node_embeddings=node_embeddings,
        lemma_index=lemma_to_idx,
        idx_to_norm=idx_to_norm,
        config=config,
        proof_checker=checker if config.use_proof_checker else None,
    )

    print(f"\n--- Running best-first search on {len(theorems)} theorems ---")
    print(f"    Max expansions: {config.max_expansions}, Top-K: {config.top_k_lemmas}")
    print(f"    Threads: {config.num_threads}")
    print()

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
        proof_steps, final_state = bf_search.search(
            stmt, domain=domain, verbose=False
        )
        search_time = time.time() - t0

        proof_text = ProofState._render_proof(proof_steps)

        if not proof_steps:
            ok = False
            err = "no proof found"
            failed_reasons["no_proof"] = failed_reasons.get("no_proof", 0) + 1
        elif checker is None:
            ok = True
            err = ""
        else:
            full_code = wrap_theorem_with_proof(stmt, proof_text)
            check_results = checker.check_batch([full_code])
            ok = check_results[0].success
            err = (
                check_results[0].errors[0][:200]
                if check_results[0].errors
                else ""
            )
            if not ok:
                reason_key = f"lean_reject:{err[:50]}"
                failed_reasons[reason_key] = (
                    failed_reasons.get(reason_key, 0) + 1
                )

        steps_str = [s.to_lean() for s in proof_steps[:10]]
        pattern = classify_proof_pattern(steps_str) if ok else "failed"
        lemma_novel = is_lemma_novelty(steps_str) if ok else False

        result = {
            "name": name,
            "era": era,
            "domain": domain,
            "success": ok,
            "error": err,
            "proof_steps": steps_str,
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
        print(
            f"  [{i+1:2d}/{len(theorems)}] {status} {name:45s} "
            f"[{pattern:12s}] {search_time:.1f}s  "
            f"ETA: {eta/60:.0f}m  ({len(passed)} passed)"
        )

        if ok and len(proof_steps) > 0:
            print(f"         Proof: {steps_str}")
            if len(proof_steps) >= 2:
                print(
                    f"         \u2605 MULTI-STEP ({len(proof_steps)} steps)"
                )

    elapsed = time.time() - t_start
    n_total = len(theorems)
    n_passed = len(passed)
    rate = n_passed / max(1, n_total)

    multi = [r for r in passed if r["num_steps"] >= 2]
    lemma_novel = [r for r in passed if r["lemma_novelty"]]
    structural = [r for r in passed if not r["lemma_novelty"]]

    # Stats
    print(f"\n{'=' * 70}")
    print("GATE 3: FINE-TUNED GNN RESULTS")
    print(f"{'=' * 70}")
    print(f"  Total:    {n_passed}/{n_total} ({rate:.0%})")
    print(f"  Multi-step: {len(multi)}")
    print(f"  Lemma-novelty: {len(lemma_novel)}")
    print(f"  Structural-only: {len(structural)}")
    print(f"  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}m)")

    # Domain breakdown
    domains = Counter(r["domain"] for r in results)
    print(f"\n  By domain:")
    for dom in sorted(domains.keys()):
        dom_total = domains[dom]
        dom_passed = sum(1 for r in passed if r["domain"] == dom)
        dom_ln = sum(1 for r in lemma_novel if r["domain"] == dom)
        dom_ms = sum(1 for r in multi if r["domain"] == dom)
        print(
            f"    {dom:<20} {dom_passed}/{dom_total} "
            f"({dom_passed/max(1,dom_total)*100:.0f}%) "
            f"LN: {dom_ln}  MS: {dom_ms}"
        )

    print(f"\n  Failure reasons:")
    for reason, count in sorted(
        failed_reasons.items(), key=lambda x: -x[1]
    )[:10]:
        print(f"    {reason:<60} {count}")

    # Build output
    out = {
        "task": "Fine-tuned GNN on gate3_v2 benchmark",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "architecture": "Fine-tuned GNN (1.1M) + Best-first search",
        "config": {
            "max_depth": config.max_depth,
            "max_expansions": config.max_expansions,
            "top_k_lemmas": config.top_k_lemmas,
            "depth_penalty": config.depth_penalty,
            "use_proof_checker": config.use_proof_checker,
            "num_threads": config.num_threads,
            "gnn_params": sum(p.numel() for p in gnn.parameters()),
        },
        "graph": {
            "num_nodes": graph.num_nodes,
            "num_edges": graph.num_edges,
        },
        "gate3": {
            "status": "PASS" if rate > 0.156 else "FAIL",
            "total": n_total,
            "passed": n_passed,
            "rate": rate,
            "baseline": 0.156,
            "multi_step": len(multi),
            "lemma_novelty": len(lemma_novel),
            "structural_only": len(structural),
            "elapsed_s": elapsed,
            "failed_reasons": dict(failed_reasons),
            "domains": {
                dom: {
                    "total": domains[dom],
                    "passed": sum(
                        1 for r in passed if r["domain"] == dom
                    ),
                    "lemma_novelty": sum(
                        1 for r in lemma_novel if r["domain"] == dom
                    ),
                    "multi_step": sum(
                        1 for r in multi if r["domain"] == dom
                    ),
                }
                for dom in domains
            },
            "passed_theorems": [
                {
                    "name": r["name"],
                    "domain": r["domain"],
                    "proof": " ".join(r["proof_steps"]),
                    "pattern": r["pattern"],
                    "num_steps": r["num_steps"],
                    "lemma_novelty": r["lemma_novelty"],
                }
                for r in passed
            ],
        },
        "all_results": results,
    }

    save_json(out, output_path)
    print(f"\n  Results saved to: {output_path}")

    gate_status = "PASS" if rate > 0.156 else "FAIL"
    print(
        f"\n  Gate 3: {gate_status} ({n_passed}/{n_total} proofs, "
        f"{rate:.1%} vs 15.6% baseline)"
    )
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_embedding_health(embeddings: torch.Tensor) -> dict:
    """Check embedding health (relaxed Gate D thresholds)."""
    N = embeddings.size(0)
    sample_n = min(N, 2000)
    indices = torch.randperm(N)[:sample_n]
    sample = embeddings[indices]

    cos_sim = sample @ sample.T
    mask = ~torch.eye(sample_n, dtype=torch.bool, device=embeddings.device)
    off_diag = cos_sim[mask]
    avg_std = off_diag.std().item()

    U, S, V = torch.svd(sample)
    threshold = S.max().item() * 0.01
    rank = (S > threshold).sum().item()

    print(
        f"  Embedding health: avg_cosine_std={avg_std:.4f} "
        f"(need >0.05), rank={rank} (need >128)"
    )
    return {"avg_cosine_std": avg_std, "rank": rank}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned GNN on gate3_v2 benchmark"
    )
    parser.add_argument(
        "--gnn-checkpoint",
        default="data/gnn_ft_full/gnn_fine_tuned.pt",
        help="Path to fine-tuned GNN checkpoint",
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
        help="Max expansions per search",
    )
    parser.add_argument(
        "--top-k", type=int, default=30, help="Top-K lemmas per state"
    )
    parser.add_argument(
        "--depth-penalty", type=float, default=0.05,
        help="Depth penalty factor",
    )
    parser.add_argument(
        "--num-threads", type=int, default=4,
        help="Number of CPU threads",
    )
    parser.add_argument(
        "--output", default="data/gnn_ft_gate3_result.json",
        help="Output JSON file",
    )
    parser.add_argument(
        "--max-theorems", type=int, default=None,
        help="Max theorems for smoke testing",
    )
    parser.add_argument(
        "--repeat", type=int, default=1,
        help="Number of replicates (for Gate 5 stats validation)",
    )
    parser.add_argument(
        "--no-proof-checker", action="store_true",
        help="Disable Lean proof checker",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Compare against baseline GNN",
    )
    parser.add_argument(
        "--baseline",
        default="checkpoints/gnn/full_graph_pretrained.pt",
        help="Path to baseline GNN checkpoint for comparison",
    )
    args = parser.parse_args()

    # Hardware constraint
    args.num_threads = min(args.num_threads, 4)
    torch.set_num_threads(args.num_threads)

    print("=" * 70)
    print("FINE-TUNED GNN EVALUATION")
    print("=" * 70)
    print(f"  GNN: {args.gnn_checkpoint}")
    print(f"  Threads: {args.num_threads}")
    print(f"  Repeats: {args.repeat}")
    print()

    # Load GNN
    gnn_path = _project_root / args.gnn_checkpoint
    if not gnn_path.exists():
        print(f"ERROR: GNN checkpoint not found: {gnn_path}")
        return 1
    gnn = GNNEncoder.load(str(gnn_path))
    gnn.eval()
    print(f"  GNN: {sum(p.numel() for p in gnn.parameters()):,} params, "
          f"hidden={gnn.config.hidden_dim}")

    # Load graph
    graph_path = _project_root / args.graph
    if not graph_path.with_suffix(".nx.pkl").exists():
        print(f"ERROR: Graph not found: {graph_path}.nx.pkl")
        return 1
    graph = DependencyGraph.load(graph_path)
    print(f"  Graph: {graph.summary()}")

    # Load theorems
    theorems_path = _project_root / args.theorems
    if not theorems_path.exists():
        print(f"ERROR: Theorems not found: {theorems_path}")
        return 1
    theorems = load_jsonl(theorems_path)
    if args.max_theorems:
        theorems = theorems[: args.max_theorems]
    print(f"  Theorems: {len(theorems)}")

    # Indexes
    lemma_to_idx = build_lemma_index(graph)
    idx_to_norm = build_norm_index(graph, lemma_to_idx)
    print(f"  Lemma index: {len(lemma_to_idx)} entries")

    # Config
    use_pc = not args.no_proof_checker
    config = GNNBestFirstConfig(
        max_depth=20,
        max_expansions=args.max_expansions,
        top_k_lemmas=args.top_k,
        depth_penalty=args.depth_penalty,
        use_proof_checker=use_pc,
        verify_timeout=5.0,
        num_threads=args.num_threads,
        max_graph_candidates=200,
    )

    checker = (
        BatchChecker(timeout=15, max_workers=4, cache_size=128)
        if use_pc
        else None
    )

    # Run (possibly multiple replicates)
    all_replicates = []
    for rep in range(args.repeat):
        rep_label = f"_rep{rep}" if args.repeat > 1 else ""
        output_path = _project_root / args.output.replace(
            ".json", f"{rep_label}.json"
        )

        result = run_gate3_with_finetuned_gnn(
            gnn=gnn,
            graph=graph,
            theorems=theorems,
            config=config,
            lemma_to_idx=lemma_to_idx,
            idx_to_norm=idx_to_norm,
            checker=checker,
            output_path=output_path,
        )
        all_replicates.append(result)

    # If multiple replicates, compute stats
    if args.repeat > 1:
        rates = [r["gate3"]["rate"] for r in all_replicates]
        mean_rate = sum(rates) / len(rates)
        std_rate = (
            (sum((r - mean_rate) ** 2 for r in rates) / len(rates)) ** 0.5
        )
        print(f"\n{'=' * 70}")
        print("GATE 5: STATISTICAL VALIDATION")
        print(f"{'=' * 70}")
        print(f"  Replicates: {args.repeat}")
        print(f"  Rates: {[f'{r:.1%}' for r in rates]}")
        print(f"  Mean: {mean_rate:.1%}")
        print(f"  Std: {std_rate:.3%}")
        gate5_pass = std_rate < 0.03
        print(f"  Gate 5: {'PASS' if gate5_pass else 'FAIL'} (std {std_rate:.3%} < 3pp)")

        aggregate = {
            "replicates": args.repeat,
            "rates": rates,
            "mean_rate": mean_rate,
            "std_rate": std_rate,
            "gate5_pass": gate5_pass,
            "replicate_results": [
                {
                    "passed": r["gate3"]["passed"],
                    "total": r["gate3"]["total"],
                    "rate": r["gate3"]["rate"],
                }
                for r in all_replicates
            ],
        }
        agg_path = _project_root / args.output
        with open(agg_path, "w") as f:
            json.dump(aggregate, f, indent=2)
        print(f"  Aggregate saved to: {agg_path}")

    # Comparison against baseline
    if args.compare:
        print(f"\n{'=' * 70}")
        print("COMPARISON: Fine-tuned vs Baseline GNN")
        print(f"{'=' * 70}")
        baseline_path = _project_root / args.baseline
        if baseline_path.exists():
            baseline_gnn = GNNEncoder.load(str(baseline_path))
            baseline_gnn.eval()
            baseline_out = run_gate3_with_finetuned_gnn(
                gnn=baseline_gnn,
                graph=graph,
                theorems=theorems,
                config=config,
                lemma_to_idx=lemma_to_idx,
                idx_to_norm=idx_to_norm,
                checker=checker,
                output_path=_project_root / args.output.replace(".json", "_baseline.json"),
            )
            ft_rate = all_replicates[0]["gate3"]["rate"]
            bl_rate = baseline_out["gate3"]["rate"]
            delta = ft_rate - bl_rate
            print(f"\n  Fine-tuned: {ft_rate:.1%}")
            print(f"  Baseline:   {bl_rate:.1%}")
            print(f"  Delta:      {delta:+.1%}")
            print(f"  Improvement: {'YES' if delta > 0 else 'NO'}")
        else:
            print(f"  Baseline checkpoint not found: {baseline_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
