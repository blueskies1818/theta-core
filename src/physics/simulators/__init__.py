"""Physics observation simulators.

Generate synthetic observation scenarios from known classical physical
equations. These simulators are programmatic (no AI) and produce data
compatible with the ObservationDatabase format for self-play training.

Modules:
    mechanics         — Classical mechanics (free fall, projectile, pendulum, spring, collision)
    electromagnetism  — Classical EM (E field, B field, E×B, Coulomb, induced EMF)
    thermodynamics    — Classical thermodynamics (ideal gas processes, Carnot, entropy)
"""

from src.physics.simulators.mechanics import generate_all_mechanics
from src.physics.simulators.electromagnetism import generate_all_electromagnetism
from src.physics.simulators.thermodynamics import generate_all_thermodynamics

__all__ = [
    "generate_all_mechanics",
    "generate_all_electromagnetism",
    "generate_all_thermodynamics",
]
