#!/usr/bin/env python3
"""Generate synthetic observation scenarios from physical simulators.

Usage:
    python scripts/build/generate_observations.py              # All domains
    python scripts/build/generate_observations.py --domain em  # EM only
    python scripts/build/generate_observations.py --dry-run    # Validate only

Output:
    data/observations/mechanics_synthetic.json
    data/observations/em_synthetic.json
    data/observations/thermal_synthetic.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on path
_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))

from src.physics.observations import ObservationDatabase
from src.physics.simulators import (
    generate_all_mechanics,
    generate_all_electromagnetism,
    generate_all_thermodynamics,
)


OUTPUT_DIR = _project_root / "data" / "observations"
OUTPUT_FILES = {
    "mechanics": OUTPUT_DIR / "mechanics_synthetic.json",
    "em": OUTPUT_DIR / "em_synthetic.json",
    "thermal": OUTPUT_DIR / "thermal_synthetic.json",
}


def save_scenarios(
    scenarios: list[dict],
    output_path: Path,
    dry_run: bool = False,
) -> int:
    """Save scenarios to JSON, validate them, return count."""
    if dry_run:
        # Validate without writing
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        tmp_path = OUTPUT_DIR / "_tmp_validate.json"
        with open(tmp_path, "w") as f:
            json.dump(scenarios, f, indent=2)

        try:
            db = ObservationDatabase(tmp_path)
            issues = db.validate()
            tmp_path.unlink()
            if issues:
                print(f"  VALIDATION ISSUES:")
                for issue in issues:
                    print(f"    - {issue}")
            else:
                print(f"  All {len(scenarios)} scenarios valid ✓")
        except Exception as e:
            tmp_path.unlink(missing_ok=True)
            print(f"  VALIDATION ERROR: {e}")
            return 0
    else:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(scenarios, f, indent=2)

        # Validate
        db = ObservationDatabase(output_path)
        issues = db.validate()
        if issues:
            print(f"  WARNING: Validation issues found:")
            for issue in issues:
                print(f"    - {issue}")

        print(f"  Saved to {output_path}")

    return len(scenarios)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic physics observations")
    parser.add_argument(
        "--domain",
        choices=["mechanics", "em", "thermal", "all"],
        default="all",
        help="Which domain to generate (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate scenarios without writing output files",
    )
    parser.add_argument(
        "--min-scenarios",
        type=int,
        default=50,
        help="Minimum scenarios required per domain (default: 50)",
    )
    args = parser.parse_args()

    domains = ["mechanics", "em", "thermal"] if args.domain == "all" else [args.domain]

    total_count = 0
    all_ok = True

    for domain in domains:
        print(f"\n{'='*60}")
        print(f"Generating {domain} scenarios...")

        if domain == "mechanics":
            scenarios = generate_all_mechanics()
        elif domain == "em":
            scenarios = generate_all_electromagnetism()
        else:
            scenarios = generate_all_thermodynamics()

        count = save_scenarios(scenarios, OUTPUT_FILES[domain], dry_run=args.dry_run)
        total_count += count

        if count < args.min_scenarios:
            print(f"  ERROR: Only {count} scenarios generated, need >= {args.min_scenarios}")
            all_ok = False
        else:
            print(f"  Generated {count} scenarios (>= {args.min_scenarios} required) ✓")

    print(f"\n{'='*60}")
    print(f"Total: {total_count} scenarios across {len(domains)} domain(s)")
    if not all_ok:
        print("Some domains failed to meet minimum scenario count!")
        sys.exit(1)
    print("Done ✓")


if __name__ == "__main__":
    main()
