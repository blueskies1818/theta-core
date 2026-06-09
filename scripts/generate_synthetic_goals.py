#!/usr/bin/env python3
"""Generate synthetic (goal, lemma) pairs with clean variable patterns.

The pretraining data from Mathlib4 proofs uses complex goal expressions
(∑, integrals, type annotations, etc.). Physics theorem goals are clean
(a + b = b + a, S1 + S2 = S2 + S1). The GoalEncoder sees almost zero
overlap in keyword patterns between training and evaluation, causing 0%
accuracy despite high per-lemma training accuracy.

This script generates clean synthetic goals for fundamental lemmas so the
GoalEncoder learns to handle physics-style goal patterns.
"""

import json
import random
from pathlib import Path

# Variable name pools for generating diverse goals
VAR_NAMES = [
    ("a", "b", "c"), ("x", "y", "z"), ("u", "v", "w"), ("p", "q", "r"),
    ("A", "B", "C"), ("X", "Y", "Z"), ("U", "V", "W"), ("P", "Q", "R"),
    ("S1", "S2", "S3"), ("T1", "T2", "T3"), ("m", "n", "k"),
    ("T", "T_env"), ("T_hot", "T_cold"), ("m1", "m2"), ("q1", "q2"),
    ("V1", "V2"), ("E1", "E2"), ("B1", "B2"), ("p1", "p2"),
    ("Φ", "Ψ"), ("σ", "τ"), ("ν", "μ"), ("α", "β"),
]

# Fundamental lemmas with template goals in clean variable notation
# Each template uses {0}, {1}, {2} as variable placeholders
LEMMA_TEMPLATES = {
    "add_comm": [
        "{0} + {1} = {1} + {0}",
    ],
    "mul_comm": [
        "{0} * {1} = {1} * {0}",
    ],
    "add_assoc": [
        "{0} + {1} + {2} = {0} + ({1} + {2})",
        "({0} + {1}) + {2} = {0} + ({1} + {2})",
    ],
    "mul_assoc": [
        "{0} * {1} * {2} = {0} * ({1} * {2})",
        "({0} * {1}) * {2} = {0} * ({1} * {2})",
    ],
    "add_zero": [
        "{0} + 0 = {0}",
    ],
    "zero_add": [
        "0 + {0} = {0}",
    ],
    "mul_one": [
        "{0} * 1 = {0}",
    ],
    "one_mul": [
        "1 * {0} = {0}",
    ],
    "sub_self": [
        "{0} - {0} = 0",
    ],
    "sub_eq_add_neg": [
        "{0} - {1} = {0} + (-{1})",
    ],
    "eq_comm": [
        "({0} = {1}) ↔ ({1} = {0})",
    ],
    "rfl": [
        "{0} = {0}",
    ],
    "Eq.refl": [
        "{0} = {0}",
    ],
    "Eq.symm": [
        "({0} = {1}) → ({1} = {0})",
    ],
    "neg_add": [
        "-({0} + {1}) = (-{0}) + (-{1})",
    ],
    "div_eq_mul_inv": [
        "{0} / {1} = {0} * ({1})⁻¹",
    ],
    "div_one": [
        "{0} / 1 = {0}",
    ],
    "one_div": [
        "1 / {0} = {0}⁻¹",
    ],
    "sq": [
        "{0} ^ 2 = {0} * {0}",
    ],
    "pow_succ": [
        "{0} ^ ({1} + 1) = {0} ^ {1} * {0}",
    ],
    "mul_zero": [
        "{0} * 0 = 0",
    ],
    "zero_mul": [
        "0 * {0} = 0",
    ],
    "mul_add": [
        "{0} * ({1} + {2}) = {0} * {1} + {0} * {2}",
    ],
    "add_mul": [
        "({0} + {1}) * {2} = {0} * {2} + {1} * {2}",
    ],
    "left_distrib": [
        "{0} * ({1} + {2}) = {0} * {1} + {0} * {2}",
    ],
    "right_distrib": [
        "({0} + {1}) * {2} = {0} * {2} + {1} * {2}",
    ],
    "sub_add_cancel": [
        "{0} - {1} + {1} = {0}",
    ],
    "add_sub_cancel": [
        "{0} + {1} - {1} = {0}",
    ],
    "neg_mul": [
        "(-{0}) * {1} = -({0} * {1})",
    ],
    "mul_neg": [
        "{0} * (-{1}) = -({0} * {1})",
    ],
    "neg_neg": [
        "-(-{0}) = {0}",
    ],
    "abs_of_nonneg": [
        "|{0}| = {0}",
    ],
    "le_refl": [
        "{0} ≤ {0}",
    ],
    "le_trans": [
        "{0} ≤ {1} ∧ {1} ≤ {2} → {0} ≤ {2}",
    ],
    "lt_of_lt_of_le": [
        "{0} < {1} ∧ {1} ≤ {2} → {0} < {2}",
    ],
    "Nat.add_comm": [
        "({0} : ℕ) + {1} = {1} + {0}",
    ],
    "Nat.mul_comm": [
        "({0} : ℕ) * {1} = {1} * {0}",
    ],
    "Int.add_comm": [
        "({0} : ℤ) + {1} = {1} + {0}",
    ],
    "ne_eq": [
        "({0} ≠ {1}) ↔ ¬({0} = {1})",
    ],
    "exists_ne": [
        "∃ x, x ≠ {0}",
    ],
}


def generate_pairs(duplicates_per_template: int = 5) -> list[dict]:
    """Generate synthetic (goal, lemma) pairs with diverse variable names."""
    pairs = []
    rng = random.Random(42)  # deterministic seed

    # Filter VAR_NAMES to ensure enough variables for each template
    for lemma_name, templates in LEMMA_TEMPLATES.items():
        for template in templates:
            # Count needed variables
            n_vars = 3 if "{2}" in template else (2 if "{1}" in template else 1)
            # Filter to variable sets with enough names
            valid_sets = [vs for vs in VAR_NAMES if len(vs) >= n_vars]

            for _ in range(duplicates_per_template):
                var_set = rng.choice(valid_sets)
                vars_used = list(var_set[:n_vars])
                goal = template.format(*vars_used)
                pairs.append({"goal": goal, "lemma": lemma_name})

    rng.shuffle(pairs)
    return pairs


def main():
    project_root = Path(__file__).resolve().parent.parent
    output_path = project_root / "data" / "raw" / "synthetic_goals.jsonl"

    pairs = generate_pairs(duplicates_per_template=8)
    print(f"Generated {len(pairs)} synthetic goal-lemma pairs")

    # Append to existing pairs
    merge_path = project_root / "data" / "raw" / "proof_step_pairs.jsonl"
    existing_count = 0
    with open(merge_path) as f:
        for line in f:
            existing_count += 1

    with open(merge_path, "a") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    print(f"Appended to {merge_path} (was {existing_count} pairs)")

    # Verify
    verify_count = 0
    with open(merge_path) as f:
        for line in f:
            verify_count += 1
    print(f"Total pairs now: {verify_count}")

    # Print samples
    print("\nSample synthetic pairs:")
    for p in pairs[:15]:
        print(f"  goal: {p['goal']:45s}  →  {p['lemma']}")


if __name__ == "__main__":
    main()
