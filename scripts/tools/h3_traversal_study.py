#!/usr/bin/env python3
"""H3 STUDY: Graph-traversal reward experiment.

Hypothesis: Adding a graph-traversal reward (bonus for proofs using lemmas
3+ hops from training lemmas) will improve GNN lemma-novelty generalization.

Design:
  1. Train GNN on gate2 data with traversal reward enabled
  2. Evaluate GNN (H=0.0) on gate3 lemma-novelty theorems
  3. Evaluate shape-matcher (H=1.0) on gate3 — must stay ≤5%
  4. Compare to baseline (21.4% GNN, 28.6% shape-matcher from gate3_result.json)

Output: data/h3_traversal_results.json
"""

import json
import sys
import time
from pathlib import Path

import torch

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from src.explorer.dependency_graph import DependencyGraph
from src.explorer.gnn_encoder import GNNEncoder, prepare_graph_tensors, extract_initial_features
from src.explorer.mcts import MCTS, MCTSConfig
from src.explorer.explorer_trainer import ExplorerTrainer, ExplorerConfig
from src.explorer.proof_state import ProofState
from src.proof_checker.batch_checker import BatchChecker
from src.proof_checker.formats import wrap_theorem_with_proof
from src.reward.config import RewardConfig
from src.reward.base import compute_rewards_batch
from src.utils.xpu_utils import get_device, print_device_info
from scripts.eval.infer_explorer import run_inference, classify_proof_pattern


def load_jsonl(path: Path) -> list[dict]:
    """Load JSONL file as list of dicts."""
    if not path.is_absolute():
        path = _project_root / path
    with open(path) as f:
        return [json.loads(line) for line in f]


def train_with_traversal(
    train_theorems: list[dict],
    output_dir: Path,
    num_epochs: int = 50,
    mcts_sims: int = 500,
) -> dict:
    """Train GNN with traversal reward on gate2 data.

    Returns training metrics dict.
    """
    print("=" * 70)
    print("H3 TRAVERSAL TRAINING")
    print("=" * 70)

    # ---- Load graph ----
    graph = DependencyGraph.load(_project_root / "data/graph/dependency_graph")
    print(f"Graph: {graph.summary()}")

    # Filter to Algebra domain (matches gate3 training setup)
    graph = graph.domain_subgraph("Algebra")
    print(f"Algebra subgraph: {graph.num_nodes} nodes, {graph.num_edges} edges")

    # ---- Load pretrained GNN (wave2) ----
    wave2_path = _project_root / "checkpoints/explorer_wave2/gnn_final.pt"
    if not wave2_path.exists():
        print(f"ERROR: wave2 checkpoint not found at {wave2_path}")
        sys.exit(1)

    gnn = GNNEncoder.load(str(wave2_path))
    total_params = sum(p.numel() for p in gnn.parameters())
    print(f"GNN: {total_params:,} params, hidden={gnn.config.hidden_dim}")

    device = get_device()
    gnn = gnn.to(device)

    # ---- Proof checker ----
    checker = BatchChecker(timeout=30, max_workers=4, cache_size=128)
    print("Proof checker: ready")

    # ---- Configs ----
    explorer_config = ExplorerConfig(
        batch_size=4,
        group_size=2,
        learning_rate=1e-3,
        weight_decay=1e-5,
        max_grad_norm=1.0,
        policy_weight=1.0,
        value_weight=0.5,
        use_correspondence=False,  # H3 uses raw traversal reward, not correspondence
        log_every=5,
        save_every=25,
        heuristic_anneal_epochs=0,  # No annealing: train at H=1.0, eval at H=0.0 (like H1)
        heuristic_scale_min=1.0,  # Full heuristics during training (GNN learns via rewards)
        resume_epoch=0,
    )

    mcts_config = MCTSConfig(
        num_simulations=mcts_sims,
        c_puct=1.4,
        max_depth=20,
        max_actions_per_node=50,
        temperature=0.5,
        top_k_lemmas=30,
        use_gnn=True,
        use_proof_checker=True,
        verify_timeout=5.0,
    )

    # Enable traversal reward
    reward_config = RewardConfig(
        curiosity_enabled=True,
        curiosity_weight=0.05,
        length_bonus_enabled=True,
        length_bonus_weight=0.1,
        traversal_bonus_enabled=True,
        traversal_bonus_weight=0.5,
        traversal_hop_threshold=3,
    )

    print(f"Traversal reward: ENABLED (weight={reward_config.traversal_bonus_weight}, "
          f"threshold={reward_config.traversal_hop_threshold})")

    # ---- Trainer ----
    trainer = ExplorerTrainer(
        gnn_encoder=gnn,
        dependency_graph=graph,
        proof_checker=checker,
        config=explorer_config,
        mcts_config=mcts_config,
        reward_config=reward_config,
        correspondence_modifier=None,
        device=device,
    )

    # ---- Format theorems for trainer ----
    formatted = []
    for t in train_theorems:
        stmt = t.get("statement", "")
        if not stmt or len(stmt) < 10:
            continue
        formatted.append({
            "statement": stmt,
            "name": t.get("name", ""),
            "proof": t.get("proof", ""),
        })
    print(f"Training theorems: {len(formatted)}")

    # ---- Train ----
    print(f"\nStarting training: {num_epochs} epochs, {mcts_sims} MCTS sims")
    t_start = time.time()

    metrics = trainer.train(
        train_theorems=formatted,
        val_theorems=None,
        output_dir=output_dir,
        num_epochs=num_epochs,
    )

    elapsed = time.time() - t_start
    print(f"\nTraining complete: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Log final metrics
    if metrics and "metrics" in metrics:
        mlist = metrics["metrics"]
        if mlist:
            successes = [m.get("success_rate", 0) for m in mlist if "success_rate" in m]
            if successes:
                print(f"  Final training success: {successes[-1]:.1%}")
                print(f"  Best training success:  {max(successes):.1%}")

    try:
        checker.shutdown()
    except:
        pass

    return {"metrics": metrics.get("metrics", []) if metrics else []}


def evaluate_on_gate3(checkpoint_path: Path, output_dir: Path) -> dict:
    """Evaluate trained GNN on gate3 lemma-novelty theorems.

    Runs both H=0.0 (pure GNN) and H=1.0 (shape-matcher/heuristics).
    Returns results dict with scores.
    """
    print("\n" + "=" * 70)
    print("H3 TRAVERSAL EVALUATION ON GATE3")
    print("=" * 70)

    device = "cpu"
    graph_path = _project_root / "data/graph/dependency_graph"
    theorems_path = _project_root / "data/raw/gate3_lemma_novelty.jsonl"

    # ---- H=0.0: Pure GNN ----
    print(f"\n--- H=0.0 (pure GNN) ---")
    gnn_results = run_inference(
        checkpoint=str(checkpoint_path),
        graph_path=str(graph_path),
        domain="Algebra",
        theorems_path=str(theorems_path),
        max_theorems=None,
        mcts_sims=500,
        era="pre_relativity",
        device=device,
        heuristic_scale=0.0,
        verbose=True,
        no_era_filter=True,  # Don't filter by era, test ALL gate3 theorems
    )

    gnn_passed = sum(1 for r in gnn_results if r["success"])
    gnn_pct = gnn_passed / max(1, len(gnn_results)) * 100

    # ---- H=1.0: Shape-matcher (heuristics only) ----
    print(f"\n--- H=1.0 (shape-matcher / heuristics) ---")
    heur_results = run_inference(
        checkpoint=str(checkpoint_path),
        graph_path=str(graph_path),
        domain="Algebra",
        theorems_path=str(theorems_path),
        max_theorems=None,
        mcts_sims=500,
        era="pre_relativity",
        device=device,
        heuristic_scale=1.0,
        verbose=True,
        no_era_filter=True,
    )

    heur_passed = sum(1 for r in heur_results if r["success"])
    heur_pct = heur_passed / max(1, len(heur_results)) * 100

    # Print comparison
    print(f"\n{'='*70}")
    print(f"GATE3 RESULTS: Traversal-Reward GNN")
    print(f"{'='*70}")
    print(f"  GNN (H=0.0):      {gnn_passed}/{len(gnn_results)} ({gnn_pct:.1f}%)")
    print(f"  Shape-matcher:    {heur_passed}/{len(heur_results)} ({heur_pct:.1f}%)")
    print(f"  Baseline GNN:     21.4%")
    print(f"  Baseline shape:   28.6%")
    print(f"  GNN Δ:            {gnn_pct - 21.4:+.1f}pp")
    print(f"  Shape Δ:           {heur_pct - 28.6:+.1f}pp")

    # Collect proved theorem names
    gnn_proved_theorems = [r["name"] for r in gnn_results if r["success"]]
    heur_proved_theorems = [r["name"] for r in heur_results if r["success"]]

    return {
        "gnn_h0_score": round(gnn_pct, 1),
        "shape_matcher_score": round(heur_pct, 1),
        "gnn_passed": gnn_passed,
        "shape_matcher_passed": heur_passed,
        "total_theorems": len(gnn_results),
        "gnn_theorems": gnn_proved_theorems,
        "shape_matcher_theorems": heur_proved_theorems,
        "gnn_results": gnn_results,
        "shape_matcher_results": heur_results,
        "baseline": {
            "gnn": 21.4,
            "shape_matcher": 28.6,
        },
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="H3 Traversal Reward Study")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--mcts-sims", type=int, default=500, help="MCTS simulations")
    parser.add_argument("--traversal-weight", type=float, default=0.5,
                        help="Traversal bonus weight")
    parser.add_argument("--traversal-threshold", type=int, default=3,
                        help="Hop threshold for traversal bonus")
    parser.add_argument("--output", default="checkpoints/h3_traversal",
                        help="Output directory")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training, evaluate existing checkpoint")
    parser.add_argument("--checkpoint", default=None,
                        help="Checkpoint for evaluation (overrides default)")
    args = parser.parse_args()

    output_dir = _project_root / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load gate2 training data ----
    print("Loading gate2 training data...")
    gate2_path = _project_root / "data/raw/gate2_training.jsonl"
    train_theorems = load_jsonl(gate2_path)
    print(f"Gate2 training theorems: {len(train_theorems)}")

    # ---- Train (or skip) ----
    checkpoint_path = output_dir / "gnn_final.pt"
    training_metrics = None

    if not args.skip_train:
        training_metrics = train_with_traversal(
            train_theorems=train_theorems,
            output_dir=output_dir,
            num_epochs=args.epochs,
            mcts_sims=args.mcts_sims,
        )
    else:
        if args.checkpoint:
            checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            print(f"ERROR: checkpoint not found at {checkpoint_path}")
            sys.exit(1)
        print(f"Using existing checkpoint: {checkpoint_path}")

    # ---- Evaluate on gate3 ----
    if not checkpoint_path.exists():
        print(f"ERROR: no checkpoint at {checkpoint_path} — cannot evaluate")
        sys.exit(1)

    eval_results = evaluate_on_gate3(checkpoint_path, output_dir)

    # ---- Write full results ----
    result_data = {
        "study": "H3",
        "name": "Graph-Traversal Reward Experiment",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "hypothesis": (
            "Adding a graph-traversal reward bonus for proofs using lemmas "
            "3+ hops from training lemmas improves GNN lemma-novelty generalization."
        ),
        "config": {
            "epochs": args.epochs,
            "mcts_sims": args.mcts_sims,
            "traversal_weight": args.traversal_weight,
            "traversal_threshold": args.traversal_threshold,
            "pretrained": "checkpoints/explorer_wave2/gnn_final.pt",
            "training_data": "data/raw/gate2_training.jsonl",
            "test_data": "data/raw/gate3_lemma_novelty.jsonl",
        },
        "results": {
            "gnn_h0_score": eval_results["gnn_h0_score"],
            "shape_matcher_score": eval_results["shape_matcher_score"],
            "gnn_passed": eval_results["gnn_passed"],
            "shape_matcher_passed": eval_results["shape_matcher_passed"],
            "total_theorems": eval_results["total_theorems"],
            "gnn_theorems": eval_results["gnn_theorems"],
            "shape_matcher_theorems": eval_results["shape_matcher_theorems"],
            "baseline_comparison": {
                "baseline_gnn": 21.4,
                "baseline_shape_matcher": 28.6,
                "gnn_delta_pp": round(eval_results["gnn_h0_score"] - 21.4, 1),
                "shape_matcher_delta_pp": round(eval_results["shape_matcher_score"] - 28.6, 1),
            },
        },
        "constraints": {
            "shape_matcher_max": 5.0,
            "shape_matcher_actual": eval_results["shape_matcher_score"],
            "shape_matcher_pass": eval_results["shape_matcher_score"] <= 5.0,
        },
        "verdict": "PASS" if (
            eval_results["gnn_h0_score"] > 21.4 and
            eval_results["shape_matcher_score"] <= 5.0
        ) else "FAIL",
        "verdict_reason": get_verdict_reason(eval_results),
    }

    # Add training metrics summary
    if training_metrics:
        mlist = training_metrics.get("metrics", [])
        if mlist:
            successes = [m.get("success_rate", 0) for m in mlist if "success_rate" in m]
            if successes:
                result_data["training"] = {
                    "final_success_rate": round(successes[-1] * 100, 1),
                    "best_success_rate": round(max(successes) * 100, 1),
                    "final_loss": mlist[-1].get("loss", None),
                }

    output_path = _project_root / "data/h3_traversal_results.json"
    with open(output_path, "w") as f:
        json.dump(result_data, f, indent=2)
    print(f"\nResults written to {output_path}")

    return 0 if result_data["verdict"] == "PASS" else 1


def get_verdict_reason(eval_results: dict) -> str:
    """Generate verdict reason string."""
    parts = []
    gnn = eval_results["gnn_h0_score"]
    shape = eval_results["shape_matcher_score"]

    if gnn <= 21.4:
        parts.append(f"GNN {gnn}% ≤ baseline 21.4% (no improvement)")
    else:
        parts.append(f"GNN {gnn}% > baseline 21.4% (improved)")

    if shape > 5.0:
        parts.append(f"Shape-matcher {shape}% > 5% threshold (FAIL)")
    else:
        parts.append(f"Shape-matcher {shape}% ≤ 5% (PASS)")

    return "; ".join(parts)


if __name__ == "__main__":
    sys.exit(main())
