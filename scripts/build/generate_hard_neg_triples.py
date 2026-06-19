#!/usr/bin/env python3
"""Fast synthetic hard-negative triple generator.

Uses cross-pair domain sampling: for each (goal, lemma) pair, samples
N lemmas from OTHER pairs in the same domain as hard negatives.

Smoke test on 50 pairs with Lean verification showed 0/450 random
candidates pass — random lemmas are almost certainly true negatives.
This generator skips Lean verification for speed.

Generates triples for all 226K pairs in ~10 seconds.

Usage:
    python scripts/build/generate_hard_neg_triples.py --num-hard 5

Output:
    data/hard_neg_triples.jsonl
"""

import argparse, json, random, sys, time
from pathlib import Path
from collections import defaultdict

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic hard-negative triples from cross-pair sampling"
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
        "--num-hard", type=int, default=5,
        help="Target number of hard negatives per pair"
    )
    parser.add_argument(
        "--max-pairs", type=int, default=None,
        help="Max pairs to generate triples for (default: all)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    args = parser.parse_args()

    random.seed(args.seed)

    # Load pairs
    pairs_path = _project_root / args.pairs
    print(f"Loading pairs from {pairs_path}...")
    pairs = []
    with open(pairs_path) as f:
        for line in f:
            pairs.append(json.loads(line))
    print(f"  Loaded {len(pairs)} pairs")

    # Group lemmas by domain
    domain_to_lemmas: dict[str, list[str]] = defaultdict(list)
    for p in pairs:
        domain = p.get("domain", "unknown")
        domain_to_lemmas[domain].append(p["lemma"])

    # Use unique lemmas per domain
    domain_to_unique: dict[str, list[str]] = {}
    for domain, lemmas in domain_to_lemmas.items():
        domain_to_unique[domain] = list(set(lemmas))

    domain_counts = {d: len(l) for d, l in domain_to_unique.items()}
    print(f"  Domains: {len(domain_to_unique)}")
    for d, c in sorted(domain_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"    {d}: {c} unique lemmas")

    # Generate triples
    max_pairs = args.max_pairs or len(pairs)
    num_process = min(max_pairs, len(pairs))

    t0 = time.time()
    triples = []
    skipped_no_candidates = 0

    for i, pair in enumerate(pairs[:num_process]):
        goal = pair["goal"]
        positive = pair["lemma"]
        domain = pair.get("domain", "unknown")

        # Get candidates: all lemmas in same domain except the positive
        candidates = [
            c for c in domain_to_unique.get(domain, [])
            if c != positive
        ]

        # If domain has too few candidates, supplement from all domains
        if len(candidates) < args.num_hard:
            all_other = [
                c for d, lemmas in domain_to_unique.items()
                for c in lemmas
                if c != positive and c not in candidates
            ]
            candidates = candidates + all_other

        if len(candidates) < args.num_hard:
            skipped_no_candidates += 1
            continue

        hard_negs = random.sample(candidates, min(args.num_hard, len(candidates)))

        triples.append({
            "goal": goal,
            "positive_lemma": positive,
            "hard_negatives": hard_negs,
            "domain": domain,
        })

    elapsed = time.time() - t0

    # Save
    output_path = _project_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for triple in triples:
            json.dump(triple, f)
            f.write("\n")

    print(f"\nGenerated {len(triples)} triples in {elapsed:.1f}s")
    if skipped_no_candidates:
        print(f"  Skipped {skipped_no_candidates} pairs (not enough candidates)")
    avg_hard = sum(len(t["hard_negatives"]) for t in triples) / max(1, len(triples))
    print(f"  Avg hard negatives per triple: {avg_hard:.1f}")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
