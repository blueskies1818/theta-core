#!/usr/bin/env python3
"""Collect self-play training data for template generator.

Uses the SelfPlayExpressionGenerator to produce (quantities, domain, expression)
triples from pre-1905 quantity symbols only. Filters for dimensional validity
and domain consistency.

Output: JSONL file with {quantities: [...], domain: str, expression: str} records.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_project_root))

from src.physics.expression_generator import (
    SelfPlayExpressionGenerator,
    PRE_1905_QUANTITY_DIMS,
    DOMAIN_LABELS,
)


def generate_training_data(
    n_examples: int = 12_000,
    levels: list[int] | None = None,
    seed: int = 42,
    output_path: str | Path = "data/self_play_training.jsonl",
) -> list[dict]:
    """Generate training examples from self-play expression generator.

    Args:
        n_examples: Target number of examples to generate.
        levels: Complexity levels to use (default [1, 2, 3]).
        seed: Random seed for reproducibility.
        output_path: Path for JSONL output.

    Returns:
        List of training example dicts.
    """
    if levels is None:
        levels = [1, 2, 3]

    generator = SelfPlayExpressionGenerator(
        seed=seed,
        include_hidden_vars=False,  # No hidden vars for template training
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    examples: list[dict] = []
    written = 0

    # Distribute examples across levels: L1=40%, L2=35%, L3=25%
    level_weights = {1: 0.40, 2: 0.35, 3: 0.25}
    targets = {
        lv: int(n_examples * level_weights.get(lv, 0.33))
        for lv in levels
    }

    print(f"Target examples per level: {targets}")
    print(f"Total target: {sum(targets.values())}")

    # Track domain counts for balancing
    domain_counts: dict[str, int] = {}
    total_generated = 0

    iteration = 0
    max_iterations = n_examples * 3  # safety limit

    while sum(domain_counts.values()) < sum(targets.values()) and iteration < max_iterations:
        iteration += 1

        # Cycle through levels
        level = levels[iteration % len(levels)]

        # Check if this level has reached its target
        level_done = all(
            sum(1 for e in examples if e.get("complexity_level") == lv) >= targets[lv]
            for lv in levels
        )
        if level_done:
            break

        try:
            gen_result = generator.generate(level)
        except Exception:
            continue

        total_generated += 1

        # Extract quantities (symbols only)
        quantity_symbols = sorted(gen_result.quantities_dict.keys())

        # Only use pre-1905 quantities
        if any(q not in PRE_1905_QUANTITY_DIMS for q in quantity_symbols):
            continue

        # Must have at least 2 quantities for meaningful template
        if len(quantity_symbols) < 2:
            continue

        domain = gen_result.domain_label
        expr = gen_result.expression_str

        # Skip empty expressions
        if not expr or not expr.strip():
            continue

        record = {
            "quantities": quantity_symbols,
            "domain": domain,
            "expression": expr,
            "complexity_level": level,
        }

        examples.append(record)
        domain_counts[domain] = domain_counts.get(domain, 0) + 1

        # Progress
        if total_generated % 1000 == 0:
            print(
                f"  Generated {total_generated}, "
                f"kept {len(examples)}, "
                f"domains: {dict(sorted(domain_counts.items()))}"
            )

    # Write JSONL
    with open(output_path, "w") as f:
        for ex in examples:
            json.dump(ex, f)
            f.write("\n")

    print(f"\nCollected {len(examples)} training examples")
    print(f"Domain distribution: {dict(sorted(domain_counts.items()))}")
    print(f"Saved to: {output_path}")
    return examples


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect self-play training data for template generator"
    )
    parser.add_argument(
        "--n-examples", type=int, default=12_000,
        help="Target number of examples (default: 12000)",
    )
    parser.add_argument(
        "--levels", type=int, nargs="+", default=[1, 2, 3],
        help="Complexity levels (default: 1 2 3)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--output", type=str, default="data/self_play_training.jsonl",
        help="Output JSONL path",
    )
    args = parser.parse_args()

    output_path = _project_root / args.output
    generate_training_data(
        n_examples=args.n_examples,
        levels=args.levels,
        seed=args.seed,
        output_path=output_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
