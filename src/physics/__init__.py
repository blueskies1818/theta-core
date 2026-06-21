"""Physics expression system — grammar, dimensions, generator, observations, evaluator.

Phase A: Type system, expression grammar, and breadth-first combinatorial
expression builder for self-play physics discovery.

Phase B: Observation database loader and constancy-based expression evaluator.

Phase C: Self-play search loop for invariant discovery.

Phase D: Lean proof generation for discovered conservation laws.
         Symmetry-driven invariant derivation (Noether's theorem).
"""

from src.physics.dimensions import Dimension
from src.physics.grammar import Expression
from src.physics.generator import ExpressionGenerator
from src.physics.observations import Observation, ObservationDatabase
from src.physics.evaluator import Evaluator, ExpressionEvaluator, score_expression
from src.physics.lean_prover import (
    PhysicsScenario,
    LeanTheorem,
    SCENARIOS,
    generate_theorem,
    generate_all_theorems,
    write_lean_file,
    verify_theorem,
    verify_scenario,
    verify_all,
    save_verified_theorem,
    verified_theorems_dir,
)
from src.physics.symmetry import (
    SymmetryGroup,
    SymmetryDetection,
    SymmetryDetector,
    NoetherDerivation,
    ConservedQuantity,
    SymmetryResult,
    SymmetryPipeline,
    SymmetryClassifier,
    GeneratorKind,
    Lagrangian,
    PREBUILT_GROUPS,
    build_galilean_group,
    build_u1_group,
    build_su2_group,
    build_symmetry_training_data,
    train_symmetry_classifier,
    run_symmetry_smoke_test,
)

__all__ = [
    "Dimension",
    "Expression",
    "ExpressionGenerator",
    "Observation",
    "ObservationDatabase",
    "Evaluator",
    "ExpressionEvaluator",
    "score_expression",
    "PhysicsScenario",
    "LeanTheorem",
    "SCENARIOS",
    "generate_theorem",
    "generate_all_theorems",
    "write_lean_file",
    "verify_theorem",
    "verify_scenario",
    "verify_all",
    "save_verified_theorem",
    "verified_theorems_dir",
    # Symmetry module
    "SymmetryGroup",
    "SymmetryDetection",
    "SymmetryDetector",
    "NoetherDerivation",
    "ConservedQuantity",
    "SymmetryResult",
    "SymmetryPipeline",
    "SymmetryClassifier",
    "GeneratorKind",
    "Lagrangian",
    "PREBUILT_GROUPS",
    "build_galilean_group",
    "build_u1_group",
    "build_su2_group",
    "build_symmetry_training_data",
    "train_symmetry_classifier",
    "run_symmetry_smoke_test",
]
