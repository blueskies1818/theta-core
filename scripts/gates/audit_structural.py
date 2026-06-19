#!/usr/bin/env python3
"""Gate 2: Structural Independence Audit.

Builds a shape-matcher baseline: for each test theorem, finds the most
structurally similar training theorem and applies its proof tactic.
If the shape-matcher scores above random chance, the test set is leaked.

Usage:
    python scripts/gates/audit_structural.py --train data/raw/training_combined.jsonl \
                                       --test data/raw/physics_theorems_post1905.jsonl \
                                       [--threshold 0.05]
"""

import sys, json, argparse, re, random
from pathlib import Path
from collections import Counter

# ── Theorem shape extraction ────────────────────────────────────────

def extract_shape(theorem: dict) -> dict:
    """Extract structural features from a theorem, ignoring surface text."""
    statement = theorem.get("statement", "")
    proof = theorem.get("proof", theorem.get("ground_truth", ""))

    # Count structural elements
    hypotheses = statement.count("→") + statement.count("->")
    universal_quants = len(re.findall(r"∀|forall", statement))
    existential_quants = len(re.findall(r"∃|exists", statement))

    # Identify the type signature structure
    # e.g., "a + b = b + a" has pattern: _ + _ = _ + _
    structure = re.sub(r"[a-zA-Z0-9_]+", "_", statement)

    # Tactic usage in ground truth proof
    tactics = re.findall(r"\b(rfl|simp|linarith|ring|field_simp|rw|apply|exact|intro|cases|have|calc|norm_num|positivity|nlinarith|omega|native_decide)\b", proof)
    tactic_count = len(tactics)
    unique_tactics = set(tactics)

    # Goal type classification
    goal_types = []
    if "=" in statement:
        goal_types.append("equality")
    if "≤" in statement or "≤" in statement or "<" in statement or ">" in statement:
        goal_types.append("inequality")
    if "→" in statement or "->" in statement:
        goal_types.append("implication")
    if "∀" in statement or "forall" in statement:
        goal_types.append("universal")
    if "/" in statement:
        goal_types.append("division")
    if "¬" in statement or "not" in statement.lower():
        goal_types.append("negation")

    return {
        "name": theorem.get("name", ""),
        "hypotheses": hypotheses,
        "universal_quants": universal_quants,
        "existential_quants": existential_quants,
        "structure": structure,
        "tactics": sorted(unique_tactics),
        "tactic_count": tactic_count,
        "goal_types": sorted(goal_types),
        "proof": proof.strip(),
    }


def shape_similarity(a: dict, b: dict) -> float:
    """Score how structurally similar two theorem shapes are. 0-1 scale."""
    score = 0.0
    total = 0.0

    # Same number of hypotheses
    total += 1
    if a["hypotheses"] == b["hypotheses"]:
        score += 1

    # Same universal quant count
    total += 1
    if a["universal_quants"] == b["universal_quants"]:
        score += 1

    # Same goal types (Jaccard)
    total += 1
    if a["goal_types"] and b["goal_types"]:
        intersection = set(a["goal_types"]) & set(b["goal_types"])
        union = set(a["goal_types"]) | set(b["goal_types"])
        score += len(intersection) / len(union) if union else 0

    # Same tactic set (Jaccard)
    total += 1
    if a["tactics"] and b["tactics"]:
        intersection = set(a["tactics"]) & set(b["tactics"])
        union = set(a["tactics"]) | set(b["tactics"])
        score += len(intersection) / len(union) if union else 0

    # Same tactic count (exact)
    total += 1
    if a["tactic_count"] == b["tactic_count"]:
        score += 1

    return score / total if total > 0 else 0


# ── Shape-matcher ───────────────────────────────────────────────────

class ShapeMatcher:
    """Match test theorems to closest training theorem by shape."""

    def __init__(self, train_theorems: list[dict]):
        self.train_shapes = [extract_shape(t) for t in train_theorems]

    def match(self, test_theorem: dict) -> tuple[dict, float, str]:
        """Return (closest_train_shape, similarity_score, suggested_proof)."""
        test_shape = extract_shape(test_theorem)
        best_score = -1
        best_shape = None

        for train_shape in self.train_shapes:
            sim = shape_similarity(test_shape, train_shape)
            if sim > best_score:
                best_score = sim
                best_shape = train_shape

        suggested_proof = best_shape["proof"] if best_shape else "?"
        return best_shape, best_score, suggested_proof


# ── Random baseline ─────────────────────────────────────────────────

def random_baseline(train_theorems: list[dict], num_samples: int = 1000) -> float:
    """Estimate chance-level shape matching accuracy."""
    shapes = [extract_shape(t) for t in train_theorems]
    if len(shapes) < 2:
        return 0.0

    scores = []
    for _ in range(num_samples):
        a, b = random.sample(shapes, 2)
        scores.append(shape_similarity(a, b))

    return sum(scores) / len(scores)


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Gate 2: Structural Independence Audit")
    parser.add_argument("--train", default="data/raw/training_combined.jsonl",
                        help="Training theorems JSONL")
    parser.add_argument("--test", default="data/raw/physics_theorems_post1905.jsonl",
                        help="Test theorems JSONL")
    parser.add_argument("--threshold", type=float, default=0.05,
                        help="Tolerance above random (default: 0.05 = 5%)")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    train_path = project_root / args.train
    test_path = project_root / args.test

    print("=" * 60)
    print("GATE 2: STRUCTURAL INDEPENDENCE AUDIT")
    print("=" * 60)
    print()

    # Load data
    if not train_path.exists():
        print(f"ERROR: Training file not found: {train_path}")
        sys.exit(1)
    if not test_path.exists():
        print(f"ERROR: Test file not found: {test_path}")
        sys.exit(1)

    with open(train_path) as f:
        train_theorems = [json.loads(line) for line in f if line.strip()]
    with open(test_path) as f:
        test_theorems = [json.loads(line) for line in f if line.strip()]

    print(f"Training theorems: {len(train_theorems)}")
    print(f"Test theorems: {len(test_theorems)}")
    print()

    # Compute random baseline
    random_acc = random_baseline(train_theorems, num_samples=2000)
    print(f"Random shape-matching baseline: {random_acc:.4f} "
          f"(avg similarity between random training pairs)")
    print()

    # Run shape-matcher
    matcher = ShapeMatcher(train_theorems)
    results = []

    for test_thm in test_theorems:
        closest_shape, similarity, suggested_proof = matcher.match(test_thm)
        actual_proof = test_thm.get("proof", test_thm.get("ground_truth", ""))

        # Check if the suggested proof tactic matches the actual proof tactic
        suggested_tactic = re.findall(r"\b(simp|linarith|ring|field_simp|rfl|rw|apply|exact)\b",
                                      suggested_proof)
        actual_tactic = re.findall(r"\b(simp|linarith|ring|field_simp|rfl|rw|apply|exact)\b",
                                   actual_proof)

        tactic_match = bool(suggested_tactic and actual_tactic and
                          suggested_tactic[0] == actual_tactic[0])

        results.append({
            "test": test_thm.get("name", "?"),
            "closest_train": closest_shape["name"] if closest_shape else "?",
            "similarity": similarity,
            "suggested_proof": suggested_proof,
            "actual_proof": actual_proof,
            "tactic_match": tactic_match,
            "leaked": tactic_match and similarity > 0.6,
        })

    # Analyze
    tactic_matches = sum(1 for r in results if r["tactic_match"])
    leaked_count = sum(1 for r in results if r["leaked"])
    match_rate = tactic_matches / len(results) if results else 0

    # Print per-theorem results
    print("--- Per-Theorem Shape Matching ---")
    print(f"{'Test Theorem':<40} {'Closest Train':<35} {'Sim':>6} {'Match':>6}")
    print("-" * 90)
    for r in results:
        flag = " LEAK" if r["leaked"] else ""
        print(f"{r['test']:<40} {r['closest_train']:<35} {r['similarity']:>5.3f}  "
              f"{'YES' if r['tactic_match'] else 'no':>5}{flag}")

    print()
    print("--- Summary ---")
    print(f"Shape-matcher tactic match rate: {match_rate:.2%} ({tactic_matches}/{len(results)})")
    print(f"Structurally leaked theorems:    {leaked_count}/{len(results)}")
    print(f"Random baseline:                 {random_acc:.4f}")
    print()

    # Gate verdict
    threshold_upper = random_acc + args.threshold
    print(f"Threshold: shape-matcher ≤ {threshold_upper:.4f} (random + {args.threshold:.0%})")

    if match_rate <= threshold_upper:
        if match_rate <= random_acc:
            verdict = "PASS"
            print(f"RESULT: PASS — Shape-matcher at random level ({match_rate:.2%} ≤ {random_acc:.4f})")
            print("Test theorems are structurally independent of training data.")
        else:
            verdict = "MARGINAL"
            print(f"RESULT: MARGINAL — Shape-matcher slightly above random ({match_rate:.2%} vs {random_acc:.4f})")
            print("Consider redesigning the weakest theorems and retesting.")
    else:
        verdict = "FAIL"
        print(f"RESULT: FAIL — Shape-matcher significantly above random ({match_rate:.2%} > {threshold_upper:.4f})")
        print()
        print("Structurally leaked theorems:")
        for r in results:
            if r["leaked"]:
                print(f"  {r['test']}: matched {r['closest_train']} "
                      f"(sim={r['similarity']:.3f}, proof={r['suggested_proof'][:60]})")
        print()
        print("Redesign these theorems so their proofs require genuinely different strategies.")

    if args.json:
        output = {
            "gate": 2,
            "verdict": verdict,
            "random_baseline": random_acc,
            "shape_matcher_rate": match_rate,
            "threshold": threshold_upper,
            "tactic_matches": tactic_matches,
            "total": len(results),
            "leaked": leaked_count,
            "per_theorem": results,
        }
        print("\n" + json.dumps(output, indent=2))

    sys.exit(0 if verdict in ("PASS", "MARGINAL") else 1)


if __name__ == "__main__":
    main()
