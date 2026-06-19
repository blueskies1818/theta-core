#!/usr/bin/env python3
"""Mine confirmed hard negatives for contrastive lemma embedding training.

For each proof-step pair, samples candidate wrong lemmas and verifies with
the Lean proof checker that they fail to prove the goal. Confirmed failures
become hard negatives for contrastive training.

The proof checker's pass/fail is the ground truth — zero era labels.

Usage:
    # Quick smoke test (100 pairs, 3 candidates each)
    python scripts/build/mine_hard_negatives.py --max-pairs 100 --num-hard 3

    # Full run (50000 pairs, 5 candidates each) — ~8-12 hours
    python scripts/build/mine_hard_negatives.py --max-pairs 50000 --num-hard 5 --max-workers 6

Output:
    data/hard_neg_triples.jsonl — (goal, positive_lemma, hard_negatives) triples
    data/hard_negative_cache.jsonl — proof checker result cache
"""

import argparse, json, sys, time
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from src.contrastive.hard_negative_miner import (
    HardNegativeMiner,
    save_hard_negative_data,
)


def main():
    parser = argparse.ArgumentParser(
        description="Mine hard negatives via proof checker"
    )
    parser.add_argument(
        "--pairs", default="data/raw/proof_step_pairs.jsonl",
        help="Path to proof-step pairs JSONL"
    )
    parser.add_argument(
        "--output", default="data/hard_neg_triples.jsonl",
        help="Output path for hard-negative triples"
    )
    parser.add_argument(
        "--cache", default="data/hard_negative_cache.jsonl",
        help="Cache path for proof checker results"
    )
    parser.add_argument(
        "--max-pairs", type=int, default=50000,
        help="Maximum number of pairs to process"
    )
    parser.add_argument(
        "--num-hard", type=int, default=5,
        help="Target number of hard negatives per pair"
    )
    parser.add_argument(
        "--max-workers", type=int, default=6,
        help="Number of parallel proof checker workers"
    )
    parser.add_argument(
        "--timeout", type=float, default=10.0,
        help="Proof checker timeout per check (seconds)"
    )
    parser.add_argument(
        "--project-dir", default=None,
        help="Lake project directory for Mathlib4 (auto-detect if omitted)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    # ---- Load pairs ----------------------------------------------------------
    pairs_path = _project_root / args.pairs
    print(f"Loading proof-step pairs from {pairs_path}...")
    pairs = []
    with open(pairs_path) as f:
        for line in f:
            pairs.append(json.loads(line))
    print(f"  Loaded {len(pairs)} pairs")

    # ---- Mine hard negatives -------------------------------------------------
    cache_path = _project_root / args.cache
    miner = HardNegativeMiner(
        project_dir=args.project_dir,
        max_workers=args.max_workers,
        timeout=args.timeout,
        cache_path=cache_path,
    )

    t0 = time.time()
    triples = miner.mine(
        pairs=pairs,
        num_hard_per_positive=args.num_hard,
        max_pairs=args.max_pairs,
        seed=args.seed,
    )
    elapsed = time.time() - t0

    # ---- Save results --------------------------------------------------------
    output_path = _project_root / args.output
    save_hard_negative_data(triples, output_path)
    print(f"\nSaved {len(triples)} hard-negative triples to {output_path}")
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f}m)")

    # ---- Summary stats -------------------------------------------------------
    if triples:
        avg_hard = sum(len(t["hard_negatives"]) for t in triples) / len(triples)
        domains = set(t.get("domain", "unknown") for t in triples)
        print(f"\nSummary:")
        print(f"  Triples: {len(triples)}")
        print(f"  Avg hard negatives/triple: {avg_hard:.1f}")
        print(f"  Domains: {sorted(domains)}")
        print(f"  Cache size: {len(miner.cache)} entries")


if __name__ == "__main__":
    main()
