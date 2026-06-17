#!/usr/bin/env python3
"""Path A: Best-first search with step-level dense rewards.

Runs contrastive+best-first with dense reward tracking on gate2
training data and gate3 evaluation data. Records per-step reward
components: (a) step validity, (b) goal proximity, (c) completion bonus.

Usage:
    python scripts/train_patha_dense.py

Output:
    data/patha_dense_results.json
"""

import sys, json, time, argparse
from pathlib import Path
from collections import Counter

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

import torch
import torch.nn.functional as F

from src.contrastive.encoder import (
    CharTokenizer,
    ContrastiveConfig,
    ContrastiveDualEncoder,
)
from src.explorer.best_first_search import BestFirstSearch, BestFirstConfig
from src.explorer.proof_state import ProofState
from src.proof_checker.batch_checker import BatchChecker
from src.proof_checker.formats import wrap_theorem_with_proof
from src.reward.dense_rewards import (
    DenseRewardConfig,
    DenseRewardTracker,
    DenseRewardBestFirstSearch,
    DenseTrajectory,
    summarize_trajectories,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_encoder(checkpoint_path: str) -> tuple:
    """Load Path C contrastive encoder and tokenizer."""
    torch.set_num_threads(4)
    encoder = ContrastiveDualEncoder.load(checkpoint_path)
    encoder.eval()
    tokenizer = CharTokenizer(max_len=256)
    return encoder, tokenizer


def load_lemma_data(
    lemmas_path: str, cache_path: str | None = None
) -> tuple[list[str], torch.Tensor]:
    """Load lemma names and embeddings."""
    with open(lemmas_path) as f:
        lemma_names = json.load(f)

    if cache_path and Path(cache_path).exists():
        print(f"Loading cached embeddings from {cache_path}...")
        data = torch.load(cache_path, map_location="cpu", weights_only=False)
        return data["lemma_names"], data["embeddings"]

    return lemma_names, None


def load_theorems(theorems_path: str, max_theorems: int | None = None) -> list[dict]:
    """Load theorems from JSONL file."""
    with open(theorems_path) as f:
        theorems = [json.loads(line) for line in f]
    if max_theorems:
        theorems = theorems[:max_theorems]
    return theorems


# ---------------------------------------------------------------------------
# Dense reward evaluation
# ---------------------------------------------------------------------------

def classify_tactic(tactic_str: str) -> str:
    """Classify a single tactic into its primary category."""
    if not tactic_str:
        return "unknown"
    s = tactic_str.strip().lower()
    if s.startswith("apply"):
        return "apply"
    elif s.startswith("rw") or s.startswith("rewrite"):
        return "rw"
    elif s.startswith("exact"):
        return "exact"
    elif s.startswith("intro"):
        return "intro"
    elif s.startswith("have"):
        return "have"
    elif s.startswith("calc"):
        return "calc"
    elif s.startswith("refine"):
        return "refine"
    elif s.startswith("cases"):
        return "cases"
    elif s in ("ring", "simp", "linarith", "nlinarith", "field_simp",
               "positivity", "norm_num", "omega"):
        return s
    else:
        return "other"


def classify_proof_pattern(proof_steps: list[str]) -> str:
    """Classify a proof into its primary pattern."""
    if not proof_steps:
        return "empty"
    tactic_types = set()
    for step in proof_steps:
        tactic_types.add(classify_tactic(step))
    if len(tactic_types) >= 2:
        return "multi"
    return list(tactic_types)[0] if tactic_types else "unknown"


def run_dense_reward_search(
    dense_search: DenseRewardBestFirstSearch,
    theorems: list[dict],
    checker: BatchChecker,
    verbose: bool = False,
):
    """Run best-first search with dense rewards on all theorems.

    Returns:
        (results, summary) — per-theorem result dicts and aggregate stats.
    """
    results = []
    trajectories = []
    t_start = time.time()

    for i, t in enumerate(theorems):
        stmt = t["statement"]
        name = t["name"]
        era = t.get("era", "unknown")
        zone = t.get("frontier_zone", "unknown")
        ground_truth = t.get("proof", "?")
        domain = t.get("domain", "unknown")

        t0 = time.time()
        steps, final_state, tracker = dense_search.search(
            stmt, theorem_name=name, verbose=False
        )
        search_time = time.time() - t0

        # Build proof code for verification
        proof_text = ProofState._render_proof(steps)

        # Post-hoc Lean verification
        if not steps:
            ok = False
            err = "no proof found"
        else:
            full_code = wrap_theorem_with_proof(stmt, proof_text)
            check_results = checker.check_batch([full_code])
            ok = check_results[0].success
            err = check_results[0].errors[0][:200] if check_results[0].errors else ""

        # Record completion in tracker (AFTER post-hoc verification)
        tracker.record_completion(success=ok)
        trajectory = tracker.to_trajectory()
        trajectories.append(trajectory)

        steps_str = [s.to_lean() for s in steps[:10]]
        pattern = classify_proof_pattern(steps_str) if ok else "failed"

        result = {
            "name": name,
            "era": era,
            "zone": zone,
            "domain": domain,
            "success": ok,
            "error": err,
            "proof_steps": steps_str,
            "num_steps": len(steps),
            "ground_truth": ground_truth,
            "search_time_s": search_time,
            "pattern": pattern,
            "dense_reward": {
                "total": trajectory.total_reward,
                "completion_bonus": trajectory.completion_bonus,
                "num_valid_steps": trajectory.num_valid_steps,
                "num_invalid_steps": trajectory.num_invalid_steps,
                "per_step": [
                    {
                        "index": s.step_index,
                        "tactic": s.tactic,
                        "is_valid": s.is_valid,
                        "step_validity": s.step_validity,
                        "goal_proximity": s.goal_proximity,
                        "cumulative": s.cumulative,
                    }
                    for s in trajectory.steps
                ],
            },
        }
        results.append(result)

        status = "\u2713" if ok else "\u2717"
        print(
            f"  [{i+1:2d}/{len(theorems)}] {status} {name:35s} "
            f"[{pattern:10s}] reward={trajectory.total_reward:.3f} "
            f"({len(steps)} steps, {search_time:.1f}s)"
        )
        if ok:
            print(f"         Proof: {steps_str}")

    elapsed = time.time() - t_start
    passed = sum(1 for r in results if r["success"])

    # Build trajectory summary
    summary = summarize_trajectories(trajectories)

    print(f"\nDense-reward result: {passed}/{len(results)} "
          f"({passed/max(1,len(results))*100:.0f}%) in {elapsed:.0f}s")
    print(f"  Mean total reward:   {summary['mean_total_reward']:.3f}")
    print(f"  Mean validity/step:  {summary['mean_validity_per_step']:.3f}")
    print(f"  Mean proximity/step: {summary['mean_proximity_per_step']:.4f}")
    print(f"  Multi-step proofs:   {summary['multi_step_proofs']}")

    return results, summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Path A: Best-first search with step-level dense rewards"
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/contrastive/lemma_encoder.pt",
        help="Path to contrastive encoder checkpoint",
    )
    parser.add_argument(
        "--lemmas",
        default="data/all_unique_lemmas.json",
        help="Path to unique lemma names JSON",
    )
    parser.add_argument(
        "--lemma-cache",
        default="data/lemma_embeddings.pt",
        help="Path to cached lemma embeddings",
    )
    parser.add_argument(
        "--gate2",
        default="data/raw/gate2_training.jsonl",
        help="Path to gate2 training theorems",
    )
    parser.add_argument(
        "--gate3",
        default="data/raw/gate3_v2.jsonl",
        help="Path to gate3 evaluation theorems",
    )
    parser.add_argument(
        "--max-gate2",
        type=int,
        default=None,
        help="Max gate2 theorems to test (None=all)",
    )
    parser.add_argument(
        "--max-gate3",
        type=int,
        default=None,
        help="Max gate3 theorems to test (None=all)",
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
        "--use-proof-checker",
        action="store_true",
        default=False,
        help="Verify candidates with Lean during expansion (SLOW)",
    )
    parser.add_argument(
        "--proximity-weight",
        type=float,
        default=0.2,
        help="Weight for goal proximity reward",
    )
    parser.add_argument(
        "--completion-bonus",
        type=float,
        default=1.0,
        help="Bonus for verified complete proofs",
    )
    parser.add_argument(
        "--output",
        default="data/patha_dense_results.json",
        help="Output JSON file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    args = parser.parse_args()

    # Resolve paths
    ckpt_path = _project_root / args.checkpoint
    lemmas_path = _project_root / args.lemmas
    lemma_cache_path = _project_root / args.lemma_cache
    gate2_path = _project_root / args.gate2
    gate3_path = _project_root / args.gate3
    output_path = _project_root / args.output

    print("=" * 70)
    print("PATH A: DENSE REWARD BEST-FIRST SEARCH")
    print("=" * 70)
    print(f"Encoder:     {ckpt_path}")
    print(f"Gate2 train: {gate2_path}")
    print(f"Gate3 eval:  {gate3_path}")
    print(f"Reward:      step_validity=0.1, proximity_w={args.proximity_weight}, "
          f"completion_bonus={args.completion_bonus}")

    # ---- Load encoder ----
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        return 1

    encoder, tokenizer = load_encoder(str(ckpt_path))
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"Encoder:     {n_params:,} params, hidden={encoder.config.hidden_dim}")

    # ---- Load lemma data ----
    if not lemmas_path.exists():
        print(f"ERROR: Lemma file not found: {lemmas_path}")
        return 1

    lemma_names, cached_embeddings = load_lemma_data(
        str(lemmas_path), str(lemma_cache_path) if lemma_cache_path.exists() else None
    )
    print(f"Lemmas:      {len(lemma_names)} unique")

    if cached_embeddings is not None:
        lemma_embeddings = cached_embeddings
        print(f"Using cached embeddings: {lemma_embeddings.shape}")
    else:
        print("ERROR: No cached lemma embeddings found. Run compare_best_first.py first.")
        return 1

    # ---- Setup search ----
    bf_config = BestFirstConfig(
        max_depth=20,
        max_expansions=args.max_expansions,
        top_k_lemmas=args.top_k,
        depth_penalty=0.05,
        use_proof_checker=args.use_proof_checker,
        verify_timeout=5.0,
        num_threads=4,
    )

    reward_config = DenseRewardConfig(
        step_validity=0.1,
        goal_proximity_enabled=True,
        goal_proximity_weight=args.proximity_weight,
        goal_proximity_temperature=0.5,
        completion_bonus=args.completion_bonus,
    )

    checker = BatchChecker(timeout=30, max_workers=1, cache_size=128)

    bf_search = BestFirstSearch(
        encoder=encoder,
        tokenizer=tokenizer,
        lemma_names=lemma_names,
        lemma_embeddings=lemma_embeddings,
        config=bf_config,
        proof_checker=checker if args.use_proof_checker else None,
    )

    dense_search = DenseRewardBestFirstSearch(
        bf_search,
        reward_config,
        encoder=encoder,
        tokenizer=tokenizer,
    )

    # ---- Run on gate2 (training/calibration) ----
    print()
    print("-" * 70)
    print("GATE 2: Training with dense rewards")
    print("-" * 70)

    if not gate2_path.exists():
        print(f"WARNING: Gate2 file not found: {gate2_path}. Skipping.")
        gate2_results = []
        gate2_summary = {}
    else:
        gate2_theorems = load_theorems(str(gate2_path), args.max_gate2)
        print(f"Theorems:    {len(gate2_theorems)} loaded")
        print()
        gate2_results, gate2_summary = run_dense_reward_search(
            dense_search, gate2_theorems, checker, verbose=args.verbose
        )

    # ---- Run on gate3 (evaluation) ----
    print()
    print("-" * 70)
    print("GATE 3: Evaluation with dense rewards")
    print("-" * 70)

    if not gate3_path.exists():
        print(f"WARNING: Gate3 file not found: {gate3_path}. Skipping.")
        gate3_results = []
        gate3_summary = {}
    else:
        gate3_theorems = load_theorems(str(gate3_path), args.max_gate3)
        print(f"Theorems:    {len(gate3_theorems)} loaded")
        print()
        gate3_results, gate3_summary = run_dense_reward_search(
            dense_search, gate3_theorems, checker, verbose=args.verbose
        )

    # ---- Comparison ----
    gate2_passed = sum(1 for r in gate2_results if r["success"])
    gate3_passed = sum(1 for r in gate3_results if r["success"])
    gate2_total = max(1, len(gate2_results))
    gate3_total = max(1, len(gate3_results))

    print()
    print("=" * 70)
    print("DENSE REWARD RESULTS")
    print("=" * 70)
    print(f"  Gate 2 (train): {gate2_passed}/{gate2_total} "
          f"({gate2_passed/gate2_total*100:.0f}%)")
    print(f"  Gate 3 (eval):  {gate3_passed}/{gate3_total} "
          f"({gate3_passed/gate3_total*100:.0f}%)")

    # Gate 3 multi-step breakdown
    gate3_multi = [r for r in gate3_results
                   if r["success"] and r["pattern"] == "multi"]
    gate3_single = [r for r in gate3_results
                    if r["success"] and r["pattern"] != "multi"]
    print(f"  Gate3 multi-step: {len(gate3_multi)}")
    print(f"  Gate3 single-step: {len(gate3_single)}")

    if gate3_multi:
        print(f"\n  Gate3 multi-step proofs:")
        for r in gate3_multi:
            print(f"    \u2713 {r['name']:<35s} [{r['pattern']}] "
                  f"→ {r['proof_steps']}")
            print(f"      ground_truth: {r['ground_truth']}")

    # ---- Save results ----
    output_data = {
        "task": "Pivot Step 4: Path A — Dense reward best-first search",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": {
            "checkpoint": str(ckpt_path.relative_to(_project_root)),
            "architecture": "CharCNN dual-encoder with dense reward best-first",
            "params": n_params,
        },
        "config": {
            "max_expansions": args.max_expansions,
            "top_k_lemmas": args.top_k,
            "depth_penalty": 0.05,
            "num_lemmas": len(lemma_names),
            "use_proof_checker": args.use_proof_checker,
            "dense_reward": {
                "step_validity": 0.1,
                "goal_proximity_weight": args.proximity_weight,
                "goal_proximity_enabled": True,
                "completion_bonus": args.completion_bonus,
            },
        },
        "gate2_results": {
            "num_theorems": len(gate2_results),
            "passed": gate2_passed,
            "rate": gate2_passed / gate2_total,
            "summary": gate2_summary if gate2_summary else {},
            "per_theorem": gate2_results,
        },
        "gate3_results": {
            "num_theorems": len(gate3_results),
            "passed": gate3_passed,
            "rate": gate3_passed / gate3_total,
            "multi_step_passed": len(gate3_multi),
            "multi_step_rate": len(gate3_multi) / gate3_total,
            "single_step_passed": len(gate3_single),
            "summary": gate3_summary if gate3_summary else {},
            "per_theorem": gate3_results,
        },
        "comparison": {
            "gate2_passed": gate2_passed,
            "gate2_rate": gate2_passed / gate2_total,
            "gate3_passed": gate3_passed,
            "gate3_rate": gate3_passed / gate3_total,
            "gate3_multi_step": len(gate3_multi),
            "note": (
                "Path A adds step-level dense rewards on top of Path C "
                "(contrastive+best-first). Rewards: (a) +0.1 per valid tactic, "
                "(b) goal proximity via embedding distance, (c) +1.0 completion bonus. "
                "Gate2 = training/calibration, Gate3 = held-out evaluation."
            ),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
