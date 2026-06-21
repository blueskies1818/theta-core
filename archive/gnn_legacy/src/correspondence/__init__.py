"""Correspondence layer — the bridge between mathematics and physical reality.

Phase 2 modules. The correspondence layer encodes known physics into
machine-readable form to guide the explorer's search. It is the COMPASS
for the GNN+MCTS architecture — without it, the explorer searches blindly.

Modules:
    limits.py         — Experimental constraints (Pressure 2 in the 3-pressure hierarchy)
    frontier.py       — Three-zone frontier map (Established/Uncertain/Breakdown)
    failure_points.py — Known failure coordinates (Planck, singularities, divergences)

Architecture:
    The frontier map divides mathematical-physical space into zones.
    Failure points are the anchors of breakdown zones — specific conditions
    where current theories fail. Together they provide:
    1. A compass (which territories to explore — pull toward breakdown)
    2. Negative waypoints (where known theories fail — the problem to solve)
    3. Reward shaping (zone-based multipliers for the GRPO training loop)

Integration:
    src/explorer/explorer_trainer.py  — reward modification from zone + failure scoring
    src/explorer/structure_generator.py — structures classified into zones
    src/correspondence/reward_integration.py — plugs frontier map + failure coords
                                                into the GRPO training loop
    src/correspondence/limits.py             — experimental constraints per zone

    Phase 3 will add:
    src/correspondence/experimental_db.py    — measurement database
    src/correspondence/conservation_check.py — Noether formalization
"""

from src.correspondence.limits import (
    CorrespondenceResult,
    ExperimentalConstraint,
    ExperimentalDomain,
    LimitRegime,
)

from src.correspondence.frontier import (
    BoundaryCondition,
    BoundaryType,
    FrontierMap,
    FrontierZone,
    ZoneType,
    build_standard_frontier_map,
    frontier_map_from_dict,
    frontier_map_to_yaml,
    load_frontier_map,
)

from src.correspondence.failure_points import (
    FailureCoordinateSystem,
    FailurePoint,
    FailureRegime,
    FailureSeverity,
    build_standard_failure_coordinates,
    failure_coordinates_from_dict,
    failure_coordinates_to_yaml,
    load_failure_coordinates,
)

from src.correspondence.reward_integration import (
    CorrespondenceRewardModifier,
    create_default_modifier,
)

__all__ = [
    # limits
    "CorrespondenceResult",
    "ExperimentalConstraint",
    "ExperimentalDomain",
    "LimitRegime",
    # frontier
    "BoundaryCondition",
    "BoundaryType",
    "FrontierMap",
    "FrontierZone",
    "ZoneType",
    "build_standard_frontier_map",
    "frontier_map_from_dict",
    "frontier_map_to_yaml",
    "load_frontier_map",
    # failure points
    "FailureCoordinateSystem",
    "FailurePoint",
    "FailureRegime",
    "FailureSeverity",
    "build_standard_failure_coordinates",
    "failure_coordinates_from_dict",
    "failure_coordinates_to_yaml",
    "load_failure_coordinates",
    # reward integration
    "CorrespondenceRewardModifier",
    "create_default_modifier",
]
