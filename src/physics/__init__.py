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
from src.physics.auto_lean import (
    AutoLeanScenario,
    AutoLeanProver,
    ProofAttempt,
    TacticLibrary,
    build_mechanics_scenarios,
    build_em_scenarios,
    build_relativistic_scenarios,
    build_all_auto_scenarios,
    run_auto_proof_benchmark,
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
    build_diverse_symmetry_examples,
    train_symmetry_classifier,
    run_symmetry_smoke_test,
    # v4: Grouped quantity / metric detection
    GroupedQuantityDetector,
    GroupedQuantityResult,
)
from src.physics.symmetry_discovery import (
    SymmetryDiscoverer,
    CandidateGroup,
    GroupCandidate,
    DISCOVERY_GENERATORS,
    DISCOVERY_GENERATOR_POOL,
    generate_candidate_groups,
    candidate_to_symmetry_group,
    generate_discovery_training_data,
    build_discovery_training_scenarios,
    train_symmetry_discoverer,
    evaluate_discovery,
    run_symmetry_discovery_pipeline,
    run_discovery_smoke_test,
    run_discovery_on_database,
    save_discovery_results,
    KNOWN_GROUP_NAMES,
    KNOWN_GENERATOR_SETS,
    SymmetryScorer,
    DiscoveryResult,
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
    # auto_lean module
    "AutoLeanScenario",
    "AutoLeanProver",
    "ProofAttempt",
    "TacticLibrary",
    "build_mechanics_scenarios",
    "build_em_scenarios",
    "build_relativistic_scenarios",
    "build_all_auto_scenarios",
    "run_auto_proof_benchmark",
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
    "build_diverse_symmetry_examples",
    "train_symmetry_classifier",
    "run_symmetry_smoke_test",
    # v4: Grouped quantity / metric detection
    "GroupedQuantityDetector",
    "GroupedQuantityResult",
    # Symmetry discovery module
    "SymmetryDiscoverer",
    "CandidateGroup",
    "GroupCandidate",
    "DISCOVERY_GENERATORS",
    "DISCOVERY_GENERATOR_POOL",
    "generate_candidate_groups",
    "candidate_to_symmetry_group",
    "generate_discovery_training_data",
    "build_discovery_training_scenarios",
    "train_symmetry_discoverer",
    "evaluate_discovery",
    "run_symmetry_discovery_pipeline",
    "run_discovery_on_database",
    "run_discovery_smoke_test",
    "save_discovery_results",
    "KNOWN_GROUP_NAMES",
    "KNOWN_GENERATOR_SETS",
    "SymmetryScorer",
    "DiscoveryResult",
]
