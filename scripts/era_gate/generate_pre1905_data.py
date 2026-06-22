"""Generate pre-1905 training observation database.

Uses ONLY pre-1905 physics simulators:
  - Classical mechanics (free fall, projectile, pendulum, spring, collision)
  - Classical EM (E field, B field, E×B drift, Coulomb)
  - Classical thermodynamics (isothermal, adiabatic, isobaric, isochoric, Carnot, entropy)

NO quantum, NO relativity. Galilean + U(1) symmetries only.
"""

import json
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.physics.simulators.mechanics import generate_all_mechanics
from src.physics.simulators.electromagnetism import generate_all_electromagnetism
from src.physics.simulators.thermodynamics import generate_all_thermodynamics


def _domain_tag(obs: dict) -> str:
    """Infer domain from observation id."""
    oid = obs["id"]
    if any(prefix in oid for prefix in ["freefall", "projectile", "pendulum", "spring", "collision"]):
        return "mechanics"
    if any(prefix in oid for prefix in ["e_field", "b_field", "eb_combined", "coulomb"]):
        return "electromagnetism"
    if any(prefix in oid for prefix in ["isothermal", "adiabatic", "isobaric", "isochoric", "carnot", "entropy"]):
        return "thermodynamics"
    return "unknown"


def generate_pre1905_database(output_path: str = "data/observations/pre1905_training.json") -> dict:
    """Generate the combined pre-1905 training database.

    Returns summary dict with counts per domain.
    """
    all_obs = []

    print("Generating mechanics scenarios...")
    mechanics = generate_all_mechanics()
    for obs in mechanics:
        obs["domain"] = "mechanics"
        obs["era"] = "pre-1905"
    all_obs.extend(mechanics)
    print(f"  → {len(mechanics)} mechanics scenarios")

    print("Generating electromagnetism scenarios...")
    em = generate_all_electromagnetism()
    for obs in em:
        obs["domain"] = "electromagnetism"
        obs["era"] = "pre-1905"
    all_obs.extend(em)
    print(f"  → {len(em)} electromagnetism scenarios")

    print("Generating thermodynamics scenarios...")
    thermo = generate_all_thermodynamics()
    for obs in thermo:
        obs["domain"] = "thermodynamics"
        obs["era"] = "pre-1905"
    all_obs.extend(thermo)
    print(f"  → {len(thermo)} thermodynamics scenarios")

    # Save
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(all_obs, f, indent=2)

    # Summary
    domains = {}
    for obs in all_obs:
        d = obs.get("domain", "unknown")
        domains[d] = domains.get(d, 0) + 1

    summary = {
        "total_scenarios": len(all_obs),
        "domains": domains,
        "output_path": str(output),
        "era": "pre-1905",
        "symmetry_groups": ["Galilean (time, space, rotation)", "U(1) gauge"],
        "excluded": ["Poincaré", "SU(2)", "Lorentz", "quantum operators"],
    }

    print(f"\nTotal: {len(all_obs)} pre-1905 scenarios")
    for d, c in sorted(domains.items()):
        print(f"  {d}: {c}")
    print(f"Saved to: {output}")

    return summary


if __name__ == "__main__":
    generate_pre1905_database()
