#!/usr/bin/env python3
"""
Gate 4 v2: Era-Separated Negative Control with 60-theorem test set.

Evaluates two independently-trained GNNs (GNN-A continuous, GNN-B quantized)
on the expanded 60-theorem mixed-era test set. Uses MCTS with GNN embeddings
for lemma retrieval (hybrid architecture).

If checkpoints don't exist, trains GNNs first on the Algebra graph.
Output: data/gate4_v2_result.json

Usage:
    python scripts/gate4_v2.py
    python scripts/gate4_v2.py --mcts-sims 200 --epochs 100
    python scripts/gate4_v2.py --skip-eval  # train only
"""

from __future__ import annotations

import json
import random
import sys
import time
import argparse
from pathlib import Path
from collections import defaultdict

import torch

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from src.explorer.dependency_graph import DependencyGraph
from src.explorer.gnn_config import GNNConfig
from src.explorer.gnn_encoder import GNNEncoder, prepare_graph_tensors, extract_initial_features
from src.explorer.mcts import MCTS, MCTSConfig
from src.explorer.proof_state import ProofState
from src.proof_checker.batch_checker import BatchChecker
from src.proof_checker.formats import wrap_theorem_with_proof

# Import pretrain_gnn
import importlib.util as _importlib_util
_pg_spec = _importlib_util.spec_from_file_location(
    "pretrain_gnn", _project_root / "scripts" / "pretrain_gnn.py"
)
_pg_module = _importlib_util.module_from_spec(_pg_spec)
_pg_spec.loader.exec_module(_pg_module)
pretrain_gnn = _pg_module.pretrain_gnn


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def classify_era_binary(era: str) -> str:
    continuous_eras = {"classical", "classical_crisis", "pre_relativity", "pre_gr"}
    quantized_eras = {"old_quantum", "pre_qed", "pre_sm", "sm_construction",
                       "sm_confirmed", "precision_era", "modern"}
    if era in continuous_eras:
        return "continuous"
    elif era in quantized_eras:
        return "quantized"
    else:
        if "quantum" in era or "modern" in era or "sm_" in era or "precision" in era:
            return "quantized"
        return "continuous"


def run_inference(
    checkpoint: str,
    graph: DependencyGraph,
    test_theorems: list[dict],
    mcts_sims: int = 200,
    heuristic_scale: float = 0.0,
    verbose: bool = False,
) -> list[dict]:
    """Run MCTS inference using GNN checkpoint for lemma retrieval (hybrid architecture)."""
    device = torch.device("cpu")

    ckpt_path = Path(checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = _project_root / ckpt_path
    gnn = GNNEncoder.load(str(ckpt_path))
    gnn.eval()
    gnn = gnn.to(device)

    features = extract_initial_features(graph, gnn.config)
    sources, targets, edge_types, num_nodes = prepare_graph_tensors(graph)
    with torch.no_grad():
        embeddings = gnn(features.to(device), sources.to(device),
                         targets.to(device), edge_types.to(device), num_nodes)

    checker = BatchChecker(timeout=30, max_workers=4, cache_size=128)

    mcts_config = MCTSConfig(
        num_simulations=mcts_sims,
        max_depth=10,
        top_k_lemmas=30,
        c_puct=1.4,
        heuristic_scale=heuristic_scale,
        use_proof_checker=True,
        verify_timeout=5.0,
    )
    mcts = MCTS(gnn_encoder=gnn, dependency_graph=graph, config=mcts_config,
                proof_checker=checker)
    mcts.set_embeddings(embeddings, sorted(graph.node_ids))

    results = []
    for i, t in enumerate(test_theorems):
        stmt = t['statement']
        name = t['name']
        era = t.get('era', 'unknown')
        zone = t.get('frontier_zone', 'unknown')
        ground_truth = t.get('proof', '?')

        t0 = time.time()
        best_steps, root = mcts.search(stmt, verbose=False)
        search_time = time.time() - t0

        proof_text = ProofState._render_proof(best_steps)
        full_code = wrap_theorem_with_proof(stmt, proof_text or 'sorry')

        # Truncation heuristic: for multi-step proofs, try single-step version
        if len(best_steps) > 1:
            first_action = best_steps[0]
            if (first_action.tactic_type.value in ("rewrite", "apply")
                    and first_action.lemma in ("add_comm", "mul_comm", "rfl", "Eq.refl")):
                single_text = ProofState._render_proof(best_steps[:1])
                full_code = wrap_theorem_with_proof(stmt, single_text or 'sorry')

        check_results = checker.check_batch([full_code])
        ok = check_results[0].success
        err = check_results[0].errors[0][:120] if check_results[0].errors else ""
        mcts_steps = [s.to_lean() for s in best_steps[:5]]

        result = {
            "name": name,
            "era": era,
            "zone": zone,
            "success": ok,
            "error": err,
            "mcts_steps": mcts_steps,
            "num_steps": len(best_steps),
            "ground_truth": ground_truth,
            "search_time_s": search_time,
            "heuristic_scale": heuristic_scale,
        }
        results.append(result)

        if verbose:
            status = "OK" if ok else "FAIL"
            print(f"  [{i+1:2d}/{len(test_theorems)}] {status:4s} {name:45s} [{era}]")
        elif ok:
            print(f"  OK   {name} [{era}]")

    return results


def compute_statistical_test(
    results_gnn_a: list[dict],
    results_gnn_b: list[dict],
) -> dict:
    """Compute interaction effect with Fisher's exact test."""
    for r in results_gnn_a:
        r["era_binary"] = classify_era_binary(r["era"])
    for r in results_gnn_b:
        r["era_binary"] = classify_era_binary(r["era"])

    a = sum(1 for r in results_gnn_a if r["era_binary"] == "continuous" and r["success"])
    c = sum(1 for r in results_gnn_a if r["era_binary"] == "quantized" and r["success"])
    b = sum(1 for r in results_gnn_b if r["era_binary"] == "continuous" and r["success"])
    d = sum(1 for r in results_gnn_b if r["era_binary"] == "quantized" and r["success"])

    total_continuous = sum(1 for r in results_gnn_a if r["era_binary"] == "continuous")
    total_quantized = sum(1 for r in results_gnn_a if r["era_binary"] == "quantized")

    gnn_a_continuous_pct = (a / total_continuous * 100) if total_continuous > 0 else 0
    gnn_a_quantized_pct = (c / total_quantized * 100) if total_quantized > 0 else 0
    gnn_b_continuous_pct = (b / total_continuous * 100) if total_continuous > 0 else 0
    gnn_b_quantized_pct = (d / total_quantized * 100) if total_quantized > 0 else 0

    try:
        from scipy.stats import fisher_exact
        table = [[a, b], [c, d]]
        odds_ratio, p_value = fisher_exact(table, alternative="two-sided")
        test_name = "Fisher's exact test"
    except ImportError:
        import math
        n = a + b + c + d
        row1, row2 = a + b, c + d
        col1, col2 = a + c, b + d
        e_a = row1 * col1 / n if n > 0 else 0
        e_b = row1 * col2 / n if n > 0 else 0
        e_c = row2 * col1 / n if n > 0 else 0
        e_d = row2 * col2 / n if n > 0 else 0
        chi2 = 0
        for obs, exp in [(a, e_a), (b, e_b), (c, e_c), (d, e_d)]:
            if exp > 0:
                chi2 += (abs(obs - exp) - 0.5) ** 2 / exp
        p_value = math.exp(-chi2 / 2) if chi2 > 0 else 1.0
        odds_ratio = (a * d) / (b * c) if (b * c) > 0 else float('inf')
        test_name = "Chi-squared (Yates)"

    interaction_detected = (gnn_a_continuous_pct > gnn_a_quantized_pct and
                            gnn_b_quantized_pct > gnn_b_continuous_pct)
    gnn_a_diff = gnn_a_continuous_pct - gnn_a_quantized_pct
    gnn_b_diff = gnn_b_quantized_pct - gnn_b_continuous_pct
    interaction_magnitude = gnn_a_diff + gnn_b_diff

    return {
        "contingency_table": {
            "gnn_a": {"continuous_correct": a, "continuous_total": total_continuous,
                       "quantized_correct": c, "quantized_total": total_quantized},
            "gnn_b": {"continuous_correct": b, "continuous_total": total_continuous,
                       "quantized_correct": d, "quantized_total": total_quantized},
        },
        "success_rates": {
            "gnn_a": {"continuous_pct": round(gnn_a_continuous_pct, 1),
                       "quantized_pct": round(gnn_a_quantized_pct, 1),
                       "overall_pct": round((a + c) / (total_continuous + total_quantized) * 100, 1)},
            "gnn_b": {"continuous_pct": round(gnn_b_continuous_pct, 1),
                       "quantized_pct": round(gnn_b_quantized_pct, 1),
                       "overall_pct": round((b + d) / (total_continuous + total_quantized) * 100, 1)},
        },
        "interaction": {
            "detected": interaction_detected,
            "magnitude": round(interaction_magnitude, 1),
            "gnn_a_era_diff": round(gnn_a_diff, 1),
            "gnn_b_era_diff": round(gnn_b_diff, 1),
            "description": (
                f"GNN-A {gnn_a_continuous_pct:.1f}%→{gnn_a_quantized_pct:.1f}% "
                f"(continuous→quantized), "
                f"GNN-B {gnn_b_continuous_pct:.1f}%→{gnn_b_quantized_pct:.1f}%"
            ),
        },
        "statistical_test": {
            "test": test_name,
            "odds_ratio": round(odds_ratio, 3) if isinstance(odds_ratio, float) else "inf",
            "p_value": round(p_value, 4),
            "significant": p_value <= 0.05,
            "note": "p ≤ 0.05 indicates significant interaction between GNN training seed and era"
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Gate 4 v2: 60-theorem Negative Control")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Link-prediction epochs (default: 100)")
    parser.add_argument("--mcts-sims", type=int, default=200,
                        help="MCTS simulations per proof (default: 200)")
    parser.add_argument("--threads", type=int, default=12,
                        help="PyTorch CPU threads (default: 12)")
    parser.add_argument("--domain", default="Algebra",
                        help="Graph domain filter")
    parser.add_argument("--graph", default="data/graph/dependency_graph",
                        help="Graph path prefix")
    parser.add_argument("--output", default="data/gate4_v2_result.json",
                        help="Output JSON path")
    parser.add_argument("--seed-a", type=int, default=42)
    parser.add_argument("--seed-b", type=int, default=12345)
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training, use existing checkpoints")
    parser.add_argument("--force-train", action="store_true",
                        help="Force retraining even if checkpoints exist")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    print(f"PyTorch threads: {torch.get_num_threads()}")

    # Paths
    graph_path = _project_root / args.graph
    output_path = _project_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    test_path = _project_root / "data/raw/gate4_test_mixed_v2.jsonl"

    ckpt_dir = _project_root / "checkpoints/gate4_fullcpu"
    ckpt_a_path = ckpt_dir / "gnn_a" / "gnn_final.pt"
    ckpt_b_path = ckpt_dir / "gnn_b" / "gnn_final.pt"
    ckpt_a_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_b_path.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    print("=" * 70)
    print("GATE 4 v2: Era-Separated Negative Control (60-theorem test set)")
    print("=" * 70)

    test_theorems = load_jsonl(test_path)
    era_counts = defaultdict(int)
    for t in test_theorems:
        era_counts[classify_era_binary(t.get("era", "unknown"))] += 1
    print(f"\nTest set: {len(test_theorems)} theorems "
          f"({era_counts.get('continuous', 0)} continuous, "
          f"{era_counts.get('quantized', 0)} quantized)")

    # Load graph
    print("\nLoading graph...")
    full_graph = DependencyGraph.load(graph_path)
    train_graph = full_graph.domain_subgraph(args.domain)
    eval_graph = train_graph
    print(f"  Domain '{args.domain}': {train_graph.summary()}")

    # GNN config
    gnn_config = GNNConfig(
        hidden_dim=256,
        num_layers=3,
        num_heads=8,
        input_dim=768,
        learning_rate=1e-3,
        weight_decay=1e-5,
        num_epochs=args.epochs,
        batch_size=512,
        num_neighbors=[25, 15, 10],
    )

    # Training (if needed)
    need_training = args.force_train or (not ckpt_a_path.exists() or not ckpt_b_path.exists())
    if args.skip_training:
        need_training = False

    if need_training:
        # GNN-A: pre-1905 continuous
        print(f"\n{'=' * 70}")
        print(f"TRAINING GNN-A (seed={args.seed_a}, continuous era)")
        print(f"{'=' * 70}")
        random.seed(args.seed_a)
        torch.manual_seed(args.seed_a)
        save_dir_a = ckpt_a_path.parent
        train_graph.save(save_dir_a / "dependency_graph")
        pretrain_gnn(
            graph=train_graph, config=gnn_config,
            num_epochs=args.epochs, batch_size=512,
            device=torch.device("cpu"), output_dir=str(save_dir_a),
        )
        print(f"  GNN-A saved to {ckpt_a_path}")

        # GNN-B: post-1925 quantized
        print(f"\n{'=' * 70}")
        print(f"TRAINING GNN-B (seed={args.seed_b}, quantized era)")
        print(f"{'=' * 70}")
        random.seed(args.seed_b)
        torch.manual_seed(args.seed_b)
        save_dir_b = ckpt_b_path.parent
        train_graph.save(save_dir_b / "dependency_graph")
        pretrain_gnn(
            graph=train_graph, config=gnn_config,
            num_epochs=args.epochs, batch_size=512,
            device=torch.device("cpu"), output_dir=str(save_dir_b),
        )
        print(f"  GNN-B saved to {ckpt_b_path}")
    else:
        print("\nUsing existing checkpoints (--skip-training implied)")

    # Evaluation
    print(f"\n{'=' * 70}")
    print(f"EVALUATION: Hybrid MCTS ({args.mcts_sims} sims) on 60-theorem test set")
    print(f"{'=' * 70}")

    print(f"\n  GNN-A (seed={args.seed_a}) on mixed test set...")
    if ckpt_a_path.exists():
        results_a = run_inference(
            checkpoint=str(ckpt_a_path), graph=eval_graph,
            test_theorems=test_theorems, mcts_sims=args.mcts_sims,
            verbose=args.verbose,
        )
    else:
        print("  ERROR: GNN-A checkpoint not found!")
        return 1

    print(f"\n  GNN-B (seed={args.seed_b}) on mixed test set...")
    if ckpt_b_path.exists():
        results_b = run_inference(
            checkpoint=str(ckpt_b_path), graph=eval_graph,
            test_theorems=test_theorems, mcts_sims=args.mcts_sims,
            verbose=args.verbose,
        )
    else:
        print("  ERROR: GNN-B checkpoint not found!")
        return 1

    if not results_a or not results_b:
        print("\nERROR: Missing results")
        return 1

    # Statistics
    print(f"\n{'=' * 70}")
    print("RESULTS")
    print(f"{'=' * 70}")

    sa = sum(1 for r in results_a if r["success"])
    sb = sum(1 for r in results_b if r["success"])
    print(f"\nOverall: GNN-A={sa}/{len(results_a)} ({sa/len(results_a)*100:.1f}%), "
          f"GNN-B={sb}/{len(results_b)} ({sb/len(results_b)*100:.1f}%)")

    stats = compute_statistical_test(results_a, results_b)
    s = stats

    print(f"\nEra breakdown:")
    print(f"  GNN-A (seed={args.seed_a}):")
    print(f"    Continuous: {s['success_rates']['gnn_a']['continuous_pct']}%")
    print(f"    Quantized:  {s['success_rates']['gnn_a']['quantized_pct']}%")
    print(f"    Overall:    {s['success_rates']['gnn_a']['overall_pct']}%")
    print(f"  GNN-B (seed={args.seed_b}):")
    print(f"    Continuous: {s['success_rates']['gnn_b']['continuous_pct']}%")
    print(f"    Quantized:  {s['success_rates']['gnn_b']['quantized_pct']}%")
    print(f"    Overall:    {s['success_rates']['gnn_b']['overall_pct']}%")

    print(f"\nInteraction: {s['interaction']['description']}")
    direction_ok = s['interaction']['detected']
    print(f"  Direction: {'CORRECT' if direction_ok else 'WRONG'}")
    print(f"  Magnitude: {s['interaction']['magnitude']}pp")

    print(f"\nStatistical test: {s['statistical_test']['test']}")
    print(f"  p-value: {s['statistical_test']['p_value']:.4f}")
    print(f"  Significant at α=0.05: {'YES' if s['statistical_test']['significant'] else 'NO'}")
    print(f"  Odds ratio: {s['statistical_test']['odds_ratio']}")

    passes = s['statistical_test']['significant'] and direction_ok

    print(f"\n{'=' * 70}")
    print(f"GATE 4 v2 VERDICT: {'PASS' if passes else 'FAIL'}")
    print(f"  Required: p ≤ 0.05, correct direction")
    print(f"  Current:  p = {s['statistical_test']['p_value']:.4f}, "
          f"direction {'correct' if direction_ok else 'wrong'}")
    print(f"{'=' * 70}")

    # Write output
    result_json = {
        "gate": "gate4_v2",
        "name": "Negative Control Experiment (60-Theorem Test Set)",
        "description": (
            "Two GNNs independently trained via link prediction on the Algebra "
            "dependency graph. GNN-A labeled 'continuous' (seed=42), GNN-B labeled "
            "'quantized' (seed=12345). Both evaluated on expanded 60-theorem mixed-era "
            "test set (30 continuous, 30 quantized) with MCTS proof search using "
            "GNN embeddings for lemma retrieval (hybrid architecture)."
        ),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "verdict": "PASS" if passes else "FAIL",
        "thresholds": {
            "p_value_max": 0.05,
            "interaction_direction_required": "GNN-A better on continuous, GNN-B better on quantized",
            "epochs": args.epochs,
            "mcts_sims": args.mcts_sims,
            "threads": args.threads,
            "test_set_size": len(test_theorems),
            "continuous_theorems": era_counts.get("continuous", 0),
            "quantized_theorems": era_counts.get("quantized", 0),
        },
        "data": {
            "train_gnn_a": {
                "seed": args.seed_a,
                "era": "continuous (classical/pre-relativity)",
                "description": "GNN trained with seed=42 on Algebra dependency graph",
            },
            "train_gnn_b": {
                "seed": args.seed_b,
                "era": "quantized (quantum/modern)",
                "description": "GNN trained with seed=12345 on identical Algebra graph",
            },
            "test_mixed": {
                "file": "data/raw/gate4_test_mixed_v2.jsonl",
                "num_continuous": era_counts.get("continuous", 0),
                "num_quantized": era_counts.get("quantized", 0),
                "total": len(test_theorems),
                "sources": {
                    "existing": "10 cont + 10 quant from gate4_test_mixed.jsonl",
                    "richer_theorems": "20 continuous-era from richer_theorems.jsonl",
                    "generated": "20 quantized-era from gate4_quantized_generated.jsonl",
                }
            },
            "graph": {
                "domain": args.domain,
                "nodes": train_graph.num_nodes,
                "edges": train_graph.num_edges,
                "note": "Both GNNs trained on identical graph with different random seeds",
            },
        },
        "training_config": {
            "epochs": args.epochs if need_training else "skipped (used existing checkpoints)",
            "hidden_dim": gnn_config.hidden_dim,
            "num_layers": gnn_config.num_layers,
            "num_heads": gnn_config.num_heads,
            "learning_rate": gnn_config.learning_rate,
            "objective": "link_prediction",
        },
        "results": {
            "gnn_a_checkpoint": str(ckpt_a_path.relative_to(_project_root)),
            "gnn_b_checkpoint": str(ckpt_b_path.relative_to(_project_root)),
            "mcts_sims": args.mcts_sims,
            "overall": {
                "gnn_a": {"correct": sa, "total": len(results_a),
                           "pct": round(sa/len(results_a)*100, 1)},
                "gnn_b": {"correct": sb, "total": len(results_b),
                           "pct": round(sb/len(results_b)*100, 1)},
            },
            "success_rates": s["success_rates"],
            "interaction": s["interaction"],
            "statistical_test": s["statistical_test"],
            "per_theorem": {
                "gnn_a": results_a,
                "gnn_b": results_b,
            },
        },
    }

    with open(output_path, "w") as f:
        json.dump(result_json, f, indent=2)
    print(f"\nResults written to {output_path}")

    return 0 if passes else 1


if __name__ == "__main__":
    sys.exit(main())
