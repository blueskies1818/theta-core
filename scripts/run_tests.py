#!/usr/bin/env python3
"""Single entry-point for testing theta-core on custom physics scenarios.

A researcher only needs:
  1. A JSON file with Observation scenarios (see docs/HOW_TO_CREATE_TESTS.md)
  2. Run: python scripts/run_tests.py path/to/scenarios.json

The system automatically selects the best discovery pipeline (beam search,
simple search, or grouped-quantity detector) and reports what it found.

USAGE:
  python scripts/run_tests.py data/frontier_tests.json
  python scripts/run_tests.py data/frontier_tests.json --threshold 0.85
  python scripts/run_tests.py data/frontier_tests.json --verbose

The system makes NO changes to itself — it uses the core engine as-is.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.physics.dimensions import Dimension
from src.physics.evaluator import ExpressionEvaluator
from src.physics.observations import Observation, ObservationDatabase
from src.physics.search import auto_discover, SearchResult


def run_tests(
    db_path: str,
    *,
    discovery_threshold: float = 0.90,
    verbose: bool = False,
) -> dict:
    """Run auto_discover on all scenarios in a JSON database.

    Returns a dict with per-scenario results and summary statistics.
    """
    db = ObservationDatabase(db_path)
    evaluator = ExpressionEvaluator()
    results = {}

    print(f"Loading {len(db)} scenarios from {db_path}")
    print(f"Discovery threshold: {discovery_threshold}")
    print(f"{'='*60}")

    for obs in db:
        if verbose:
            print(f"\n  {obs.id}: {obs.name}")
            print(f"    Known invariant: {obs.known_invariant or 'none'}")
            print(f"    Quantities: {list(obs.quantities.keys())}")
            print(f"    Timesteps: {len(obs.timesteps)}")

        # Build quantity → Dimension mapping
        quantities = {
            name: Dimension.named(dim)
            for name, dim in obs.quantities.items()
        }

        # Run discovery
        t0 = time.time()
        result = auto_discover(
            quantities=quantities,
            observations=[obs],
            known_invariant=obs.known_invariant,
            discovery_threshold=discovery_threshold,
            beam_expansions=2000,
        )
        elapsed = time.time() - t0

        discovered = result.is_discovery
        found_expr = result.expression or "NONE"

        # Determine status
        if discovered:
            if obs.known_invariant and _expressions_equivalent(
                found_expr, obs.known_invariant
            ):
                status = "MATCH"
            else:
                status = "FOUND"
        else:
            status = "MISS"

        print(f"  {obs.id}: {status}  found={found_expr}  score={result.score:.4f}")

        if verbose:
            print(
                f"    Ground truth: {ground_truth_score:.4f}  "
                f"Found: {found_expr}  Score: {result.score:.4f}"
            )
            print(
                f"    Expansions: {result.expansions}  "
                f"Time: {elapsed:.1f}s  Result: {status}"
            )

        results[obs.id] = {
            "scenario": obs.name,
            "known_invariant": obs.known_invariant,
            "found_expression": found_expr,
            "found_score": result.score,
            "discovered": discovered,
            "status": status,
            "expansions": result.expansions,
            "time_seconds": round(elapsed, 3),
        }

    # Summary
    total = len(results)
    discovered = sum(1 for r in results.values() if r["discovered"])
    matched = sum(1 for r in results.values() if r["status"] == "MATCH")
    found = sum(1 for r in results.values() if r["status"] == "FOUND")
    missed = sum(1 for r in results.values() if r["status"] == "MISS")

    print(f"\n{'='*60}")
    print(f"  RESULTS: {discovered}/{total} discovered")
    print(f"    Exact match:  {matched}")
    print(f"    Found (alt):  {found}")
    print(f"    Not found:    {missed}")
    print(f"{'='*60}")

    return results


def _expressions_equivalent(a: str, b: str) -> bool:
    """Check if two expressions are equivalent (commutativity of * only)."""
    a = a.replace(" ", "")
    b = b.replace(" ", "")
    if a == b:
        return True
    # Handle multiplication commutativity: a*b == b*a
    if "*" in a and "*" in b and "+" not in a and "-" not in a:
        parts_a = sorted(a.split("*"))
        parts_b = sorted(b.split("*"))
        return parts_a == parts_b
    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run theta-core discovery on custom physics scenarios",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_tests.py data/my_scenarios.json
  python scripts/run_tests.py data/my_scenarios.json --threshold 0.85
  python scripts/run_tests.py data/my_scenarios.json --verbose --output results.json
        """,
    )
    parser.add_argument(
        "database", type=str, help="Path to JSON scenario database"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.90,
        help="Discovery constancy threshold (default: 0.90)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show per-scenario details",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Export results to JSON file",
    )
    args = parser.parse_args()

    db_path = Path(args.database)
    if not db_path.exists():
        print(f"Error: {db_path} not found")
        sys.exit(1)

    results = run_tests(
        str(db_path),
        discovery_threshold=args.threshold,
        verbose=args.verbose,
    )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results exported to {args.output}")


if __name__ == "__main__":
    main()
