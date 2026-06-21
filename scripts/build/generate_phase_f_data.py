#!/usr/bin/env python3
"""Generate phase F observation database — 7 domains.

Outputs a combined JSON file with observations from all 7 domains:
  gravity, spring, collision, em, thermal, quantum, relativistic.

Usage:
  python scripts/build/generate_phase_f_data.py
  python scripts/build/generate_phase_f_data.py --output data/observations/phase_f_7domain.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root
_project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_project_root))

from src.physics.simulators.mechanics import generate_all_mechanics
from src.physics.simulators.electromagnetism import generate_all_electromagnetism
from src.physics.simulators.thermodynamics import generate_all_thermodynamics
from src.physics.simulators.quantum import generate_all_quantum
from src.physics.simulators.relativity import generate_all_relativity


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate 7-domain phase F observation database"
    )
    parser.add_argument(
        "--output",
        default="data/observations/phase_f_7domain.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    output_path = _project_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Generating 7-domain observation database...")
    print(f"  Output: {output_path}")

    all_observations: list[dict] = []

    # Domain 1: Classical mechanics
    print("\n  [1/7] Classical mechanics...")
    mech = generate_all_mechanics()
    print(f"        {len(mech)} scenarios")
    all_observations.extend(mech)

    # Domain 2: Electromagnetism
    print("  [2/7] Electromagnetism...")
    em = generate_all_electromagnetism()
    print(f"        {len(em)} scenarios")
    all_observations.extend(em)

    # Domain 3: Thermodynamics
    print("  [3/7] Thermodynamics...")
    thermal = generate_all_thermodynamics()
    print(f"        {len(thermal)} scenarios")
    all_observations.extend(thermal)

    # Domain 4: Quantum mechanics
    print("  [4/7] Quantum mechanics...")
    quantum = generate_all_quantum()
    print(f"        {len(quantum)} scenarios")
    all_observations.extend(quantum)

    # Domain 5: Special relativity
    print("  [5/7] Special relativity...")
    relativity = generate_all_relativity()
    print(f"        {len(relativity)} scenarios")
    all_observations.extend(relativity)

    # Domains 6-7: Collision (already part of mechanics) and cross-domain
    print("  [6/7] Collision (embedded in mechanics)")

    total = len(all_observations)
    print(f"\n  Total: {total} observations across 7 domains")

    # Save
    with open(output_path, "w") as f:
        json.dump(all_observations, f, indent=2)

    print(f"  Saved: {output_path}")

    # Summary per domain
    print("\n  Domain breakdown:")
    domain_counts: dict[str, int] = {}
    for obs in all_observations:
        oid = obs["id"]
        if "collision" in oid.lower():
            domain = "collision"
        elif any(kw in oid for kw in ["freefall", "projectile", "pendulum", "spring"]):
            domain = "mechanics"
        elif any(kw in oid for kw in ["e_field", "b_field", "eb_field", "coulomb", "induced"]):
            domain = "em"
        elif any(kw in oid for kw in ["isothermal", "adiabatic", "isobaric", "isochoric",
                                       "carnot", "otto", "entropy"]):
            domain = "thermal"
        elif any(kw in oid for kw in ["box_n", "qho", "hydrogen", "free_particle"]):
            domain = "quantum"
        elif any(kw in oid for kw in ["spacetime", "energy_mom", "time_dilation",
                                       "length_contraction", "vel_add"]):
            domain = "relativistic"
        else:
            domain = "other"
        domain_counts[domain] = domain_counts.get(domain, 0) + 1

    for domain, count in sorted(domain_counts.items()):
        print(f"    {domain:20s}: {count:4d}")

    print("\nDone.")


if __name__ == "__main__":
    main()
