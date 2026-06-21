#!/usr/bin/env python3
"""Generate symmetry analysis results for all observation databases.

Output: data/symmetry_results.json
"""
import json
from pathlib import Path

from src.physics.observations import ObservationDatabase
from src.physics.symmetry import (
    SymmetryDetector,
    SymmetryPipeline,
    NoetherDerivation,
)

OBSERVATION_FILES = [
    "data/observations/phase1_falling.json",
    "data/observations/phase2_extended.json",
    "data/observations/mechanics_synthetic.json",
    "data/observations/em_synthetic.json",
    "data/observations/thermal_synthetic.json",
]

def main():
    detector = SymmetryDetector()
    results = {}

    for path_str in OBSERVATION_FILES:
        path = Path(path_str)
        if not path.exists():
            continue
        db = ObservationDatabase(path)

        # Determine Lagrangian from observation content
        for obs in db:
            detection = detector.detect(obs)

            # Pick Lagrangian based on quantities
            qty_set = set(obs.quantities.keys())
            if "g" in qty_set and "k" in qty_set:
                lagrangian_key = "gravity_spring"
            elif "k" in qty_set:
                lagrangian_key = "spring"
            elif "vx" in qty_set and "g" in qty_set:
                lagrangian_key = "projectile"
            elif "q" in qty_set or "E" in qty_set:
                lagrangian_key = "free_fall"  # fallback for EM
            else:
                lagrangian_key = "free_fall"

            pipeline = SymmetryPipeline(
                detector=detector,
                lagrangian=lagrangian_key,
            )
            result = pipeline.run(obs)

            results[obs.id] = {
                "scenario": obs.name,
                "description": obs.description,
                "quantities": dict(obs.quantities),
                "known_invariant": obs.known_invariant,
                "symmetry_detection": {
                    "active_symmetries": result.detection.symmetry_names,
                    "evidence": {
                        result.detection.symmetry_names[i]: result.detection.evidence.get(gen, "")
                        for i, gen in enumerate(result.detection.active_symmetries)
                    },
                    "confidence": {
                        result.detection.symmetry_names[i]: result.detection.confidence.get(gen, 0.0)
                        for i, gen in enumerate(result.detection.active_symmetries)
                    },
                },
                "groups_matched": result.groups_matched,
                "derived_invariants": result.expressions,
            }

    Path("data").mkdir(exist_ok=True)
    with open("data/symmetry_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Written {len(results)} results to data/symmetry_results.json")


if __name__ == "__main__":
    main()
