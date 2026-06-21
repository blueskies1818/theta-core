"""Add external_forces, is_conservative, phase_regions metadata to phase2_extended.json."""
import json
from pathlib import Path

PATH = Path("data/observations/phase2_extended.json")

with open(PATH) as f:
    scenarios = json.load(f)

# External force detection by scenario id pattern
force_patterns = {
    "freefall_air_resistance": ["air_resistance"],
    "projectile_linear_drag": ["air_resistance", "drag"],
    "projectile_quadratic_drag": ["air_resistance", "drag"],
    "spring_damped": ["damping"],
    "spring_heavily_damped": ["damping"],
    "spring_critically_damped": ["damping"],
    "spring_medium_damped": ["damping"],
    "spring_forced": ["external_driving_force"],
    "spring_coupled": ["internal_transfer"],  # not truly external but energy moves
    "collision_inelastic": ["inelastic_deformation", "heat_loss"],
    "pendulum_air_resistance": ["air_resistance"],
    "mass_spring_damped_gravity": ["damping"],
    "incline_.*_friction": ["friction"],
}

import re

def detect_external_forces(scenario):
    sid = scenario["id"]
    for pattern, forces in force_patterns.items():
        if re.search(pattern, sid):
            return forces
    # Check parameter names
    params = scenario.get("parameters", {})
    forces = []
    if "mu" in params and params.get("mu", 0) > 0:
        forces.append("friction")
    if "beta" in params:
        forces.append("damping")
    if "k" in params and "drag" in sid.lower() and params.get("k", 0) > 0:
        forces.append("drag")
    if "F0" in params:
        forces.append("external_driving_force")
    if "E" in params and "field" in sid.lower():
        pass  # EM fields are conservative
    return forces if forces else None

def determine_conservative(scenario, external_forces):
    """Determine if a scenario is conservative.
    
    Conservative: known_invariant is not None AND no external forces that
    break conservation.
    """
    if scenario.get("known_invariant") is not None and not external_forces:
        return True
    if external_forces:
        return False
    if scenario.get("known_invariant") is not None:
        return True
    return False

def get_phase_regions(scenario):
    sid = scenario["id"]
    if "collision" in sid:
        # Find the collision event time from the velocity discontinuity
        ts = scenario["timesteps"]
        if len(ts) >= 4:
            # Collision happens between timesteps 1 and 2 (by convention in our data)
            t_collision = ts[2]["t"]  # after collision
            return [
                {"label": "before_collision", "t_range": [ts[0]["t"], t_collision - 0.01]},
                {"label": "after_collision", "t_range": [t_collision, ts[-1]["t"]]},
            ]
    return None

for s in scenarios:
    # Add external forces
    ext_forces = detect_external_forces(s)
    s["external_forces"] = ext_forces
    
    # Determine conservation
    s["is_conservative"] = determine_conservative(s, ext_forces)
    
    # Add phase regions for collision scenarios
    pr = get_phase_regions(s)
    if pr:
        s["phase_regions"] = pr

# Stats
conservative = sum(1 for s in scenarios if s["is_conservative"])
nonconservative = sum(1 for s in scenarios if not s["is_conservative"])
with_forces = sum(1 for s in scenarios if s["external_forces"])
with_phases = sum(1 for s in scenarios if s.get("phase_regions"))

print(f"Updated {len(scenarios)} scenarios:")
print(f"  Conservative: {conservative}")
print(f"  Non-conservative: {nonconservative}")
print(f"  With external_forces: {with_forces}")
print(f"  With phase_regions: {with_phases}")

with open(PATH, "w") as f:
    json.dump(scenarios, f, indent=2)
print(f"Wrote {PATH}")
