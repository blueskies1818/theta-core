"""Physics observation simulators.

Generate synthetic observation scenarios from known physical
equations. These simulators are programmatic (no AI) and produce data
compatible with the ObservationDatabase format for self-play training.

Modules:
    mechanics         — Classical mechanics (free fall, projectile, pendulum, spring, collision)
    electromagnetism  — Classical EM (E field, B field, E×B, Coulomb, induced EMF)
    thermodynamics    — Classical thermodynamics (ideal gas processes, Carnot, entropy)
    quantum           — Quantum mechanics (particle-in-box, harmonic oscillator, hydrogen atom)
    relativity        — Special relativity (spacetime interval, energy-momentum, Lorentz transforms)
"""

from src.physics.simulators.mechanics import generate_all_mechanics
from src.physics.simulators.electromagnetism import generate_all_electromagnetism
from src.physics.simulators.thermodynamics import generate_all_thermodynamics
from src.physics.simulators.quantum import generate_all_quantum
from src.physics.simulators.relativity import generate_all_relativity

__all__ = [
    "generate_all_mechanics",
    "generate_all_electromagnetism",
    "generate_all_thermodynamics",
    "generate_all_quantum",
    "generate_all_relativity",
]
