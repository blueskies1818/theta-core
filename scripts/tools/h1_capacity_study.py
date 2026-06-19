#!/usr/bin/env python3
"""H1 STUDY: Measure lemma-novelty score vs GNN param count.

Trains identical architecture at 4 scales (1M, 5M, 10M, 50M params)
on same gate2 training data. Tests all on gate3 lemma-novelty set.
Outputs data/h1_capacity_results.json with score-vs-params curve.

Pauses if any scale OOMs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# === Scale configurations ===
# Each entry: (label, hidden_dim, num_layers, num_heads, approx_params)
SCALES = [
    ("1M",  256,  3, 8),
    ("5M",  544,  3, 8),   # 544 divisible by 8 (head_dim=68)
    ("10M", 768,  3, 8),   # 768 divisible by 8 (head_dim=96)
    ("50M", 1536, 4, 8),   # 1536 divisible by 8 (head_dim=192)
]

# Training budget (same for all scales)
TRAIN_STEPS = 7
TRAIN_BATCH = 2
TRAIN_MCTS_SIMS = 50

# Evaluation budget
EVAL_MCTS_SIMS = 100

# Shared paths
GATE2_TRAINING = "data/raw/gate2_training.jsonl"
GATE3_TEST = "data/raw/gate3_lemma_novelty.jsonl"
GRAPH_PATH = "data/graph/dependency_graph"
DOMAIN = "Algebra"
RESULTS_PATH = "data/h1_capacity_results.json"


def run_cmd(cmd: list[str], desc: str, timeout: int = 7200) -> subprocess.CompletedProcess:
    """Run a subprocess command, printing output as it comes."""
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"  {' '.join(cmd[:6])}{' ...' if len(cmd) > 6 else ''}")
    print(f"{'='*60}")

    t0 = time.time()
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "12"
    env["MKL_NUM_THREADS"] = "12"
    env["OPENBLAS_NUM_THREADS"] = "12"
    proc = subprocess.Popen(
        cmd,
        cwd=_PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    output_lines = []
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        print(f"  {line}")
        output_lines.append(line)

    proc.wait(timeout=timeout)
    elapsed = time.time() - t0

    if proc.returncode != 0:
        print(f"\n  *** {desc} FAILED (exit code {proc.returncode}, {elapsed:.0f}s) ***")
    else:
        print(f"\n  --- {desc} OK ({elapsed:.0f}s) ---")

    return subprocess.CompletedProcess(
        args=cmd, returncode=proc.returncode, stdout="\n".join(output_lines), stderr=""
    )


def parse_training_result(stdout: str) -> dict:
    """Parse train_explorer.py output for key metrics."""
    result = {"train_time_s": 0, "final_success": 0.0, "best_success": 0.0, "final_loss": 0.0}

    for line in stdout.split("\n"):
        if "Total time:" in line:
            try:
                result["train_time_s"] = float(line.split("Total time:")[1].split("s")[0].strip())
            except (ValueError, IndexError):
                pass
        if "Final success:" in line:
            try:
                pct = line.split("Final success:")[1].strip().rstrip("%")
                result["final_success"] = float(pct) / 100.0
            except (ValueError, IndexError):
                pass
        if "Best success:" in line:
            try:
                pct = line.split("Best success:")[1].strip().rstrip("%")
                result["best_success"] = float(pct) / 100.0
            except (ValueError, IndexError):
                pass
        if "Final loss:" in line:
            try:
                result["final_loss"] = float(line.split("Final loss:")[1].strip())
            except (ValueError, IndexError):
                pass

    return result


def parse_infer_result(stdout: str) -> dict:
    """Parse infer_explorer.py output for gate3 score."""
    result = {"score": 0.0, "passed": 0, "total": 0, "eval_time_s": 0, "details": []}

    # Look for the result line: "Result: X/Y (Z%)"
    for line in stdout.split("\n"):
        if line.startswith("  Result:"):
            # e.g., "  Result: 3/14 (21%) in 245s"
            parts = line.split()
            if len(parts) >= 2:
                ratio = parts[1]  # "3/14"
                try:
                    p, t = ratio.split("/")
                    result["passed"] = int(p)
                    result["total"] = int(t)
                    result["score"] = result["passed"] / max(1, result["total"])
                except (ValueError, ZeroDivisionError):
                    pass
            # Time
            for part in parts:
                if part.endswith("s"):
                    try:
                        result["eval_time_s"] = float(part[:-1])
                    except ValueError:
                        pass
                    break

        # Collect per-theorem results
        if line.strip().startswith("[") and "]" in line and ("✓" in line or "✗" in line):
            # Extract theorem name and success
            try:
                bracket_end = line.index("]")
                after_bracket = line[bracket_end:].strip()
                status = after_bracket[0]  # ✓ or ✗
                parts = line[bracket_end:].split()
                name = parts[1] if len(parts) > 1 else "?"
                result["details"].append({
                    "name": name,
                    "success": status == "✓",
                })
            except (ValueError, IndexError):
                pass

    return result


def compute_params(hidden_dim: int, num_layers: int, num_heads: int) -> int:
    """Estimate GNNEncoder parameter count."""
    D = hidden_dim
    L = num_layers
    H = num_heads
    return (4 * L + 5) * D * D + (3 * L + 8) * D + 4 * L * D // H


def main():
    parser = argparse.ArgumentParser(
        description="H1 Capacity Study: lemma-novelty score vs GNN param count"
    )
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training, run evaluation only (requires existing checkpoints)")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip evaluation, run training only")
    parser.add_argument("--scale", default=None,
                        choices=["1M", "5M", "10M", "50M"],
                        help="Run only a specific scale")
    args = parser.parse_args()

    print("=" * 70)
    print("H1 CAPACITY STUDY: Lemma-Novelty Score vs GNN Param Count")
    print("=" * 70)
    print(f"Training: {TRAIN_STEPS} steps × {TRAIN_BATCH} batch × {TRAIN_MCTS_SIMS} MCTS sims")
    print(f"Eval:     {EVAL_MCTS_SIMS} MCTS sims on {GATE3_TEST}")
    print(f"Graph:    {GRAPH_PATH} ({DOMAIN} domain)")
    print()

    # Pre-compute all configs
    configs = []
    for label, hidden_dim, num_layers, num_heads in SCALES:
        if args.scale and label != args.scale:
            continue
        params = compute_params(hidden_dim, num_layers, num_heads)
        configs.append({
            "label": label,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "params": params,
            "checkpoint_dir": f"checkpoints/h1_{label}",
        })
        print(f"  Scale {label}: hidden={hidden_dim}, layers={num_layers}, "
              f"heads={num_heads} → {params:,} params")

    if not configs:
        print("ERROR: No scales selected. Check --scale flag.")
        sys.exit(1)

    results_data = {"scales": [], "config": {
        "train_steps": TRAIN_STEPS,
        "train_batch": TRAIN_BATCH,
        "train_mcts_sims": TRAIN_MCTS_SIMS,
        "eval_mcts_sims": EVAL_MCTS_SIMS,
        "graph_domain": DOMAIN,
        "training_data": GATE2_TRAINING,
        "eval_data": GATE3_TEST,
    }}

    # === TRAINING PHASE ===
    if not args.skip_training:
        print("\n" + "=" * 70)
        print("PHASE 1: TRAINING")
        print("=" * 70)

        for cfg in configs:
            label = cfg["label"]
            print(f"\n{'─'*60}")
            print(f"  Training scale {label} ({cfg['params']:,} params)")
            print(f"{'─'*60}")

            checkpoint_file = f"{cfg['checkpoint_dir']}/gnn_final.pt"

            # Check if checkpoint already exists
            if (Path(_PROJECT_ROOT) / checkpoint_file).exists():
                print(f"  Checkpoint already exists: {checkpoint_file} — skipping training")
                cfg["train_result"] = {"train_time_s": 0, "note": "used existing checkpoint"}
                continue

            cmd = [
                sys.executable, "scripts/training/train_explorer.py",
                "--domain", DOMAIN,
                "--theorems", GATE2_TRAINING,
                "--max-theorems", str(TRAIN_BATCH * TRAIN_STEPS),
                "--steps", str(TRAIN_STEPS),
                "--batch-size", str(TRAIN_BATCH),
                "--group-size", "1",
                "--mcts-sims", str(TRAIN_MCTS_SIMS),
                "--no-eval",
                "--no-correspondence",
                "--heuristic-scale-min", "1.0",
                "--heuristic-anneal-epochs", "0",
                "--output", cfg["checkpoint_dir"],
                "--hidden-dim", str(cfg["hidden_dim"]),
                "--num-layers", str(cfg["num_layers"]),
                "--num-heads", str(cfg["num_heads"]),
                "--device", "cpu",
            ]

            proc = run_cmd(cmd, f"Training {label} ({cfg['params']:,} params)", timeout=7200)

            if proc.returncode != 0:
                print(f"\n  *** Scale {label} training FAILED. Checking for OOM... ***")
                stdout = proc.stdout or ""
                if "memory" in stdout.lower() or "oom" in stdout.lower():
                    print(f"  *** OOM detected for {label}. Pausing as instructed. ***")
                    results_data["scales"].append({
                        "label": label,
                        "params": cfg["params"],
                        "hidden_dim": cfg["hidden_dim"],
                        "num_layers": cfg["num_layers"],
                        "num_heads": cfg["num_heads"],
                        "status": "OOM",
                        "error": "Out of memory during training",
                    })
                    # Save partial results
                    output_path = _PROJECT_ROOT / RESULTS_PATH
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(output_path, "w") as f:
                        json.dump(results_data, f, indent=2)
                    print(f"\nPartial results saved to {RESULTS_PATH}.")
                    print("Pausing as instructed — remaining scales not attempted.")
                    sys.exit(1)
                else:
                    print(f"  Non-OOM failure. Continuing with remaining scales.")
                    cfg["train_result"] = {"status": "failed", "error": proc.stdout[-500:] if proc.stdout else "unknown"}
                    continue

            # Find the saved checkpoint
            ckpt_dir = _PROJECT_ROOT / cfg["checkpoint_dir"]
            # train_explorer saves as gnn_final.pt or gnn_step_*.pt
            saved_checkpoints = sorted(ckpt_dir.glob("gnn_*.pt"))
            if not saved_checkpoints:
                # Also check for gnn_final.pt
                if (ckpt_dir / "gnn_final.pt").exists():
                    saved_checkpoints = [ckpt_dir / "gnn_final.pt"]
                else:
                    # Check if explorer trainer saved differently
                    saved_checkpoints = sorted(ckpt_dir.glob("*.pt"))

            if saved_checkpoints:
                cfg["checkpoint_path"] = str(saved_checkpoints[-1].relative_to(_PROJECT_ROOT))
                print(f"  Checkpoint saved: {cfg['checkpoint_path']}")
            else:
                print(f"  WARNING: No checkpoint found in {cfg['checkpoint_dir']}")

            cfg["train_result"] = parse_training_result(proc.stdout or "")
            cfg["train_result"]["params"] = cfg["params"]

    # === EVALUATION PHASE ===
    if not args.skip_eval:
        print("\n" + "=" * 70)
        print("PHASE 2: EVALUATION ON GATE 3 LEMMA-NOVELTY")
        print("=" * 70)

        for cfg in configs:
            label = cfg["label"]
            checkpoint_path = cfg.get("checkpoint_path")

            if not checkpoint_path:
                # Try to find checkpoint
                ckpt_dir = _PROJECT_ROOT / cfg["checkpoint_dir"]
                candidates = sorted(ckpt_dir.glob("gnn_*.pt"))
                if candidates:
                    checkpoint_path = str(candidates[-1].relative_to(_PROJECT_ROOT))
                elif (ckpt_dir / "gnn_final.pt").exists():
                    checkpoint_path = f"{cfg['checkpoint_dir']}/gnn_final.pt"
                else:
                    print(f"  Scale {label}: No checkpoint found — skipping evaluation")
                    cfg["eval_result"] = {"status": "skipped", "reason": "no checkpoint"}
                    continue

            print(f"\n{'─'*60}")
            print(f"  Evaluating scale {label} ({cfg['params']:,} params)")
            print(f"  Checkpoint: {checkpoint_path}")
            print(f"{'─'*60}")

            cmd = [
                sys.executable, "scripts/eval/infer_explorer.py",
                "--checkpoint", checkpoint_path,
                "--graph", GRAPH_PATH,
                "--domain", DOMAIN,
                "--theorems", GATE3_TEST,
                "--mcts-sims", str(EVAL_MCTS_SIMS),
                "--device", "cpu",
                "--heuristic-scale", "0.0",  # Pure GNN, no heuristics
                "--no-era-filter",
                "--repeat", "1",
            ]

            proc = run_cmd(cmd, f"Eval {label}", timeout=7200)

            if proc.returncode != 0:
                cfg["eval_result"] = {"status": "failed", "error": proc.stdout[-500:] if proc.stdout else "unknown"}
                continue

            eval_result = parse_infer_result(proc.stdout or "")
            eval_result["label"] = label
            eval_result["params"] = cfg["params"]
            eval_result["checkpoint"] = checkpoint_path

            cfg["eval_result"] = eval_result
            print(f"\n  Scale {label} Gate 3 score: {eval_result['score']:.1%} "
                  f"({eval_result['passed']}/{eval_result['total']})")

        # === BUILD RESULTS ===
        print("\n" + "=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)

        for cfg in configs:
            label = cfg["label"]
            eval_r = cfg.get("eval_result", {})
            train_r = cfg.get("train_result", {})

            scale_entry = {
                "label": label,
                "params": cfg["params"],
                "hidden_dim": cfg["hidden_dim"],
                "num_layers": cfg["num_layers"],
                "num_heads": cfg["num_heads"],
                "gate3_score": eval_r.get("score", 0.0),
                "gate3_passed": eval_r.get("passed", 0),
                "gate3_total": eval_r.get("total", 0),
                "train_time_s": train_r.get("train_time_s", 0),
                "eval_time_s": eval_r.get("eval_time_s", 0),
                "train_final_success": train_r.get("final_success", 0.0),
                "train_best_success": train_r.get("best_success", 0.0),
                "train_final_loss": train_r.get("final_loss", 0.0),
                "theorem_details": eval_r.get("details", []),
            }

            results_data["scales"].append(scale_entry)

            score = eval_r.get("score", 0.0)
            print(f"  {label:>4s} ({cfg['params']:>8,} params): "
                  f"Gate3={score:.1%} ({eval_r.get('passed',0)}/{eval_r.get('total',0)}) | "
                  f"Train: {train_r.get('final_success',0):.1%} success, "
                  f"{train_r.get('train_time_s',0):.0f}s")

    # === SAVE ===
    output_path = _PROJECT_ROOT / RESULTS_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results_data, f, indent=2)

    print(f"\n{'='*70}")
    print(f"H1 study complete. Results saved to {RESULTS_PATH}")
    print(f"{'='*70}")

    # Print recommended next action
    scores = [(e["label"], e.get("gate3_score", 0.0)) for e in results_data["scales"]]
    if len(scores) >= 2:
        print("\nScore vs params curve:")
        for label, score in scores:
            bar = "█" * int(score * 50)
            print(f"  {label:>4s}: {bar} {score:.1%}")


if __name__ == "__main__":
    main()
