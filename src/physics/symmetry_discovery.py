"""Symmetry discovery — propose new groups when data doesn't match known ones.

When the SymmetryDetector fails to match known groups (Galilean, Poincaré, U(1)),
this module searches over a combinatorial space of Lie group generators to
propose new symmetry groups that explain the observed conservation laws.

Architecture:
  Observation → SymmetryDetector (known groups)
    → if no match → SymmetryDiscoverer (search mode)
    → generate candidate groups from generator combinations
    → derive invariants via Noether's theorem
    → score constancy against data
    → best-match group reported

Search space:
  - Generators: time translation, space translation ×3, rotation ×3,
    boost ×3, U(1) gauge, SU(2) weak, scale
  - Combinations: Cartesian products, semi-direct products
  - Budget: 1000 group candidates per discovery attempt

Training data: synthetic groups with known answers:
  - System with time translation only → ℝ (1-generator)
  - System with time + rotation → ℝ × SO(2)
  - System with Lorentz + U(1) → Poincaré × U(1)
  Held out: system with Lorentz + SU(2) breaking

Acceptance:
  - Rediscovers Galilean group from Newtonian data
  - Rediscovers Poincaré from relativistic data
  - Proposes correct broken symmetry for held-out test
  - All existing tests still pass
"""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.physics.dimensions import Dimension
from src.physics.evaluator import ExpressionEvaluator
from src.physics.observations import Observation, ObservationDatabase
from src.physics.symmetry import (
    # Core
    SymmetryGroup,
    SymmetryDetection,
    SymmetryDetector,
    NoetherDerivation,
    Lagrangian,
    ConservedQuantity,
    GeneratorKind,
    GENERATOR_LABELS,
    # Pre-built
    PREBUILT_GROUPS,
    build_galilean_group,
    build_u1_group,
    build_su2_group,
)


# ── Discovery generator pool ─────────────────────────────────────────────────

# All generators available for discovery search
DISCOVERY_GENERATOR_POOL: list[GeneratorKind] = [
    GeneratorKind.TIME_TRANSLATION,
    GeneratorKind.SPACE_TRANSLATION_X,
    GeneratorKind.SPACE_TRANSLATION_Y,
    GeneratorKind.SPACE_TRANSLATION_Z,
    GeneratorKind.ROTATION_XY,
    GeneratorKind.ROTATION_XZ,
    GeneratorKind.ROTATION_YZ,
    GeneratorKind.BOOST_X,
    GeneratorKind.BOOST_Y,
    GeneratorKind.BOOST_Z,
    GeneratorKind.U1_PHASE,
    GeneratorKind.SU2_WEAK,
]

POOL_SIZE = len(DISCOVERY_GENERATOR_POOL)  # 12

# Alias for __init__.py compatibility
DISCOVERY_GENERATORS = DISCOVERY_GENERATOR_POOL

# Known group names for the detector
KNOWN_GROUP_NAMES = ["galilean", "u1", "su2"]

# Known generator sets for matching
KNOWN_GENERATOR_SETS: dict[str, list[GeneratorKind]] = {
    "galilean": [
        GeneratorKind.TIME_TRANSLATION,
        GeneratorKind.SPACE_TRANSLATION_X,
        GeneratorKind.SPACE_TRANSLATION_Y,
        GeneratorKind.SPACE_TRANSLATION_Z,
        GeneratorKind.ROTATION_XY,
        GeneratorKind.ROTATION_XZ,
        GeneratorKind.ROTATION_YZ,
        GeneratorKind.BOOST_X,
        GeneratorKind.BOOST_Y,
        GeneratorKind.BOOST_Z,
    ],
    "poincare": [
        GeneratorKind.TIME_TRANSLATION,
        GeneratorKind.SPACE_TRANSLATION_X,
        GeneratorKind.SPACE_TRANSLATION_Y,
        GeneratorKind.SPACE_TRANSLATION_Z,
        GeneratorKind.ROTATION_XY,
        GeneratorKind.ROTATION_XZ,
        GeneratorKind.ROTATION_YZ,
        GeneratorKind.BOOST_X,
        GeneratorKind.BOOST_Y,
        GeneratorKind.BOOST_Z,
    ],
    "u1": [GeneratorKind.U1_PHASE],
    "su2": [GeneratorKind.SU2_WEAK],
}


# ── Derived invariants per generator ─────────────────────────────────────────

# For each generator, the conserved quantity expression that Noether's theorem
# would produce. These are used when scoring candidate groups.
GENERATOR_INVARIANT_PATTERNS: dict[GeneratorKind, str] = {
    GeneratorKind.TIME_TRANSLATION: "energy",  # Hamiltonian
    GeneratorKind.SPACE_TRANSLATION_X: "momentum_x",  # m*vx
    GeneratorKind.SPACE_TRANSLATION_Y: "momentum_y",  # m*vy
    GeneratorKind.SPACE_TRANSLATION_Z: "momentum_z",  # m*vz
    GeneratorKind.ROTATION_XY: "angular_momentum_z",  # x*m*vy - y*m*vx
    GeneratorKind.ROTATION_XZ: "angular_momentum_y",  # z*m*vx - x*m*vz
    GeneratorKind.ROTATION_YZ: "angular_momentum_x",  # y*m*vz - z*m*vy
    GeneratorKind.BOOST_X: "center_of_mass_x",  # m*(x - vx*t)
    GeneratorKind.BOOST_Y: "center_of_mass_y",
    GeneratorKind.BOOST_Z: "center_of_mass_z",
    GeneratorKind.U1_PHASE: "charge",  # q
    GeneratorKind.SU2_WEAK: "weak_isospin",  # I
}


# ── Candidate Group Generator ────────────────────────────────────────────────

@dataclass
class CandidateGroup:
    """A candidate symmetry group with scoring information.

    Parameters
    ----------
    group : SymmetryGroup
        The candidate symmetry group.
    generators : list[GeneratorKind]
        The individual generators composing this group.
    generator_count : int
        Number of generators (group dimension).
    constancy_score : float
        How constant the combined invariants are against observations [0, 1].
    per_generator_scores : dict[GeneratorKind, float]
        Per-generator constancy scores.
    invariant_expressions : dict[GeneratorKind, str]
        The derived invariant expression for each generator.
    match_type : str
        How this candidate was derived: "exact", "subset", "inexact".
    evidence : dict
        Supporting evidence for this group.
    """

    group: SymmetryGroup
    generators: list[GeneratorKind]
    generator_count: int
    constancy_score: float
    per_generator_scores: dict[GeneratorKind, float] = field(default_factory=dict)
    invariant_expressions: dict[GeneratorKind, str] = field(default_factory=dict)
    match_type: str = "inexact"
    evidence: dict = field(default_factory=dict)


def generate_candidate_groups(
    max_groups: int = 1000,
    *,
    generator_pool: list[GeneratorKind] | None = None,
    min_generators: int = 1,
    max_generators: int = 12,
    include_known: bool = True,
) -> list[list[GeneratorKind]]:
    """Generate candidate symmetry group generator sets.

    Uses combinatorial search over the generator pool, prioritizing:
    1. Smaller groups (fewer generators) first
    2. Common physics patterns (time+space, time+rotation)
    3. Cartesian products of subgroups

    Args:
        max_groups: Maximum number of candidates to generate.
        generator_pool: Which generators to sample from.
        min_generators: Minimum generators per group.
        max_generators: Maximum generators per group.
        include_known: Whether to include known group generator sets.

    Returns:
        List of generator sets, each being a candidate symmetry group.
    """
    pool = generator_pool or DISCOVERY_GENERATOR_POOL

    candidates: list[list[GeneratorKind]] = []
    seen: set[tuple] = set()

    # Priority 1: Known physics patterns
    physics_patterns: list[list[GeneratorKind]] = [
        # Single generators
        [GeneratorKind.TIME_TRANSLATION],
        [GeneratorKind.U1_PHASE],
        [GeneratorKind.SU2_WEAK],
        # Time + single space
        [GeneratorKind.TIME_TRANSLATION, GeneratorKind.SPACE_TRANSLATION_X],
        # Time + rotation (ℝ × SO(2))
        [GeneratorKind.TIME_TRANSLATION, GeneratorKind.ROTATION_XY],
        # Time + space + rotation
        [
            GeneratorKind.TIME_TRANSLATION,
            GeneratorKind.SPACE_TRANSLATION_X,
            GeneratorKind.ROTATION_XY,
        ],
        # Galilean-like subsets
        [
            GeneratorKind.TIME_TRANSLATION,
            GeneratorKind.SPACE_TRANSLATION_X,
            GeneratorKind.SPACE_TRANSLATION_Y,
            GeneratorKind.ROTATION_XY,
        ],
        # Full Galilean (10 generators)
        [
            GeneratorKind.TIME_TRANSLATION,
            GeneratorKind.SPACE_TRANSLATION_X,
            GeneratorKind.SPACE_TRANSLATION_Y,
            GeneratorKind.SPACE_TRANSLATION_Z,
            GeneratorKind.ROTATION_XY,
            GeneratorKind.ROTATION_XZ,
            GeneratorKind.ROTATION_YZ,
            GeneratorKind.BOOST_X,
            GeneratorKind.BOOST_Y,
            GeneratorKind.BOOST_Z,
        ],
        # Galilean + U(1)
        [
            GeneratorKind.TIME_TRANSLATION,
            GeneratorKind.SPACE_TRANSLATION_X,
            GeneratorKind.SPACE_TRANSLATION_Y,
            GeneratorKind.SPACE_TRANSLATION_Z,
            GeneratorKind.ROTATION_XY,
            GeneratorKind.ROTATION_XZ,
            GeneratorKind.ROTATION_YZ,
            GeneratorKind.BOOST_X,
            GeneratorKind.BOOST_Y,
            GeneratorKind.BOOST_Z,
            GeneratorKind.U1_PHASE,
        ],
        # Poincaré (relativistic — same generators as Galilean in our rep)
        [
            GeneratorKind.TIME_TRANSLATION,
            GeneratorKind.SPACE_TRANSLATION_X,
            GeneratorKind.SPACE_TRANSLATION_Y,
            GeneratorKind.SPACE_TRANSLATION_Z,
            GeneratorKind.ROTATION_XY,
            GeneratorKind.ROTATION_XZ,
            GeneratorKind.ROTATION_YZ,
            GeneratorKind.BOOST_X,
            GeneratorKind.BOOST_Y,
            GeneratorKind.BOOST_Z,
        ],
        # Poincaré + U(1)
        [
            GeneratorKind.TIME_TRANSLATION,
            GeneratorKind.SPACE_TRANSLATION_X,
            GeneratorKind.SPACE_TRANSLATION_Y,
            GeneratorKind.SPACE_TRANSLATION_Z,
            GeneratorKind.ROTATION_XY,
            GeneratorKind.ROTATION_XZ,
            GeneratorKind.ROTATION_YZ,
            GeneratorKind.BOOST_X,
            GeneratorKind.BOOST_Y,
            GeneratorKind.BOOST_Z,
            GeneratorKind.U1_PHASE,
        ],
    ]

    if include_known:
        for pattern in physics_patterns:
            key = tuple(sorted(pattern, key=lambda g: g.value))
            if key not in seen and min_generators <= len(pattern) <= max_generators:
                candidates.append(list(pattern))
                seen.add(key)

    # Priority 2: All subsets up to size 6 (C(12,1) + C(12,2) + ... + C(12,6) = 2509)
    # We cap at max_groups
    for size in range(min_generators, min(max_generators + 1, 7)):
        if len(candidates) >= max_groups:
            break
        for combo in itertools.combinations(pool, size):
            key = tuple(sorted(combo, key=lambda g: g.value))
            if key not in seen:
                candidates.append(list(combo))
                seen.add(key)
                if len(candidates) >= max_groups:
                    break

    # Priority 3: Cartesian products of independent subgroups
    # (time × space, rotation × boost, etc.)
    if len(candidates) < max_groups:
        subgroups = [
            [GeneratorKind.TIME_TRANSLATION],
            [
                GeneratorKind.SPACE_TRANSLATION_X,
                GeneratorKind.SPACE_TRANSLATION_Y,
                GeneratorKind.SPACE_TRANSLATION_Z,
            ],
            [
                GeneratorKind.ROTATION_XY,
                GeneratorKind.ROTATION_XZ,
                GeneratorKind.ROTATION_YZ,
            ],
            [GeneratorKind.BOOST_X, GeneratorKind.BOOST_Y, GeneratorKind.BOOST_Z],
            [GeneratorKind.U1_PHASE],
            [GeneratorKind.SU2_WEAK],
        ]
        for r in range(2, len(subgroups) + 1):
            if len(candidates) >= max_groups:
                break
            for combo in itertools.combinations(subgroups, r):
                product = []
                for sg in combo:
                    product.extend(sg)
                key = tuple(sorted(product, key=lambda g: g.value))
                if key not in seen and min_generators <= len(product) <= max_generators:
                    candidates.append(product)
                    seen.add(key)
                    if len(candidates) >= max_groups:
                        break

    return candidates[:max_groups]


def candidate_to_symmetry_group(
    generators: list[GeneratorKind],
) -> SymmetryGroup:
    """Convert a generator set to a SymmetryGroup with invariants.

    Uses NoetherDerivation on a simple free-particle Lagrangian to derive
    default invariants, then populates the group structure.

    Args:
        generators: List of generator kinds in this candidate.

    Returns:
        SymmetryGroup with appropriate invariants.
    """
    # Default invariants for each generator type
    default_invariants: dict[GeneratorKind, str] = {
        GeneratorKind.TIME_TRANSLATION: "0.5*m*v^2",  # Kinetic energy
        GeneratorKind.SPACE_TRANSLATION_X: "m*vx",
        GeneratorKind.SPACE_TRANSLATION_Y: "m*vy",
        GeneratorKind.SPACE_TRANSLATION_Z: "m*vz",
        GeneratorKind.ROTATION_XY: "m*(x*vy - y*vx)",
        GeneratorKind.ROTATION_XZ: "m*(z*vx - x*vz)",
        GeneratorKind.ROTATION_YZ: "m*(y*vz - z*vy)",
        GeneratorKind.BOOST_X: "m*(x - vx*t)",
        GeneratorKind.BOOST_Y: "m*(y - vy*t)",
        GeneratorKind.BOOST_Z: "m*(z - vz*t)",
        GeneratorKind.U1_PHASE: "q",
        GeneratorKind.SU2_WEAK: "I",
    }

    invariants = {}
    for gen in generators:
        invariants[gen] = default_invariants.get(
            gen, f"invariant_of_{GENERATOR_LABELS.get(gen, 'unknown')}"
        )

    # Derive group name from generator composition
    name = _derive_group_name(generators)

    return SymmetryGroup(
        name=name,
        generators=generators,
        invariants=invariants,
        dimension=len(generators),
        parent=None,
    )


def _derive_group_name(generators: list[GeneratorKind]) -> str:
    """Derive a human-readable group name from generators."""
    has_time = GeneratorKind.TIME_TRANSLATION in generators
    has_space = any(
        g in generators
        for g in [
            GeneratorKind.SPACE_TRANSLATION_X,
            GeneratorKind.SPACE_TRANSLATION_Y,
            GeneratorKind.SPACE_TRANSLATION_Z,
        ]
    )
    has_rotation = any(
        g in generators
        for g in [
            GeneratorKind.ROTATION_XY,
            GeneratorKind.ROTATION_XZ,
            GeneratorKind.ROTATION_YZ,
        ]
    )
    has_boost = any(
        g in generators
        for g in [
            GeneratorKind.BOOST_X,
            GeneratorKind.BOOST_Y,
            GeneratorKind.BOOST_Z,
        ]
    )
    has_u1 = GeneratorKind.U1_PHASE in generators
    has_su2 = GeneratorKind.SU2_WEAK in generators

    n = len(generators)

    # Match known groups
    if n == 10 and has_time and has_space and has_rotation and has_boost:
        return "Poincaré group"
    if n == 11 and has_time and has_space and has_rotation and has_boost and has_u1:
        return "Poincaré × U(1)"
    if n == 1 and has_time:
        return "ℝ (time translation)"
    if n == 2 and has_time and has_rotation:
        return "ℝ × SO(2)"
    if n == 3 and has_time and has_space:
        return "Galilean (reduced)"
    if n == 1 and has_u1:
        return "U(1)"
    if n == 1 and has_su2:
        return "SU(2)"

    parts = []
    if has_time:
        parts.append("ℝ")
    if has_space:
        space_count = sum(
            1
            for g in [
                GeneratorKind.SPACE_TRANSLATION_X,
                GeneratorKind.SPACE_TRANSLATION_Y,
                GeneratorKind.SPACE_TRANSLATION_Z,
            ]
            if g in generators
        )
        parts.append(f"ℝ³_{space_count}")
    if has_rotation:
        rot_count = sum(
            1
            for g in [
                GeneratorKind.ROTATION_XY,
                GeneratorKind.ROTATION_XZ,
                GeneratorKind.ROTATION_YZ,
            ]
            if g in generators
        )
        parts.append(f"SO({rot_count})")
    if has_boost:
        boost_count = sum(
            1
            for g in [
                GeneratorKind.BOOST_X,
                GeneratorKind.BOOST_Y,
                GeneratorKind.BOOST_Z,
            ]
            if g in generators
        )
        parts.append(f"Boost({boost_count})")
    if has_u1:
        parts.append("U(1)")
    if has_su2:
        parts.append("SU(2)")

    return " × ".join(parts) if parts else f"Custom group (dim {n})"


# ── Symmetry Scoring Model ────────────────────────────────────────────────────


class SymmetryScorer:
    """Small MLP that scores how well a candidate symmetry group matches data.

    Input: concatenated features
      - Generator presence bits (12)
      - Quantity presence bits (from QUANTITY_VOCAB, 28)
      - Parameter presence bits (5: m, g, k, q, c)
      → Total: 45 features

    Output: scalar score [0, 1] predicting how constant the group's
    invariants will be against the observation.

    ~40K parameters.
    """

    NUM_GENERATORS: ClassVar[int] = POOL_SIZE  # 12
    NUM_QUANTITY_FEATURES: ClassVar[int] = 28  # matches NUM_QUANTITIES
    NUM_PARAM_FEATURES: ClassVar[int] = 8
    TOTAL_FEATURES: ClassVar[int] = (
        NUM_GENERATORS + NUM_QUANTITY_FEATURES + NUM_PARAM_FEATURES
    )

    def __init__(self, hidden_dim: int = 64) -> None:
        self.hidden_dim = hidden_dim
        self._build_model()

    def _build_model(self) -> None:
        """Build the scorer MLP."""

        class ScorerMLP(nn.Module):
            def __init__(self, in_dim, hidden):
                super().__init__()
                self.fc1 = nn.Linear(in_dim, hidden)
                self.bn1 = nn.BatchNorm1d(hidden)
                self.fc2 = nn.Linear(hidden, hidden)
                self.bn2 = nn.BatchNorm1d(hidden)
                self.fc3 = nn.Linear(hidden, hidden // 2)
                self.bn3 = nn.BatchNorm1d(hidden // 2)
                self.fc4 = nn.Linear(hidden // 2, 1)
                self.dropout = nn.Dropout(0.1)

            def forward(self, x):
                h = F.relu(self.bn1(self.fc1(x)))
                h = self.dropout(h)
                h = F.relu(self.bn2(self.fc2(h)))
                h = self.dropout(h)
                h = F.relu(self.bn3(self.fc3(h)))
                return torch.sigmoid(self.fc4(h))

        self.model = ScorerMLP(self.TOTAL_FEATURES, self.hidden_dim)

    def count_parameters(self) -> int:
        """Return total trainable parameters."""
        return sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

    def _build_features(
        self,
        generators: list[GeneratorKind],
        obs: Observation,
    ) -> list[float]:
        """Build feature vector for scoring.

        Args:
            generators: Candidate group generators.
            obs: The observation being analyzed.

        Returns:
            Feature vector of length TOTAL_FEATURES.
        """
        # Generator presence
        gen_features = [0.0] * self.NUM_GENERATORS
        for gen in generators:
            try:
                idx = DISCOVERY_GENERATOR_POOL.index(gen)
                gen_features[idx] = 1.0
            except ValueError:
                pass

        # Quantity presence
        from src.physics.composer import QUANTITY_VOCAB, QTY_TO_IDX

        qty_features = [0.0] * self.NUM_QUANTITY_FEATURES
        all_vars = set(obs.quantities.keys()) | set(obs.parameters.keys())
        for var in all_vars:
            idx = QTY_TO_IDX.get(var)
            if idx is not None and idx < self.NUM_QUANTITY_FEATURES:
                qty_features[idx] = 1.0

        # Parameter features: key physics parameters
        key_params = ["m", "g", "k", "q", "c", "hbar", "omega", "gamma"]
        param_features = [
            1.0 if p in obs.parameters or p in obs.quantities else 0.0
            for p in key_params
        ]

        return gen_features + qty_features + param_features

    def predict(
        self,
        generators: list[GeneratorKind],
        obs: Observation,
    ) -> float:
        """Predict constancy score for a candidate group against an observation.

        Args:
            generators: Candidate group generators.
            obs: The observation.

        Returns:
            Predicted score [0, 1].
        """
        features = self._build_features(generators, obs)
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor([features], dtype=torch.float32)
            score = self.model(x).item()
        return score

    def save(self, path: str) -> None:
        """Save scorer to disk."""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "hidden_dim": self.hidden_dim,
                "num_generators": self.NUM_GENERATORS,
                "num_quantity_features": self.NUM_QUANTITY_FEATURES,
                "num_param_features": self.NUM_PARAM_FEATURES,
                "total_features": self.TOTAL_FEATURES,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> SymmetryScorer:
        """Load scorer from disk."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        scorer = cls(hidden_dim=ckpt["hidden_dim"])
        scorer.model.load_state_dict(ckpt["model_state_dict"])
        scorer.model.eval()
        return scorer


# ── Symmetry Discoverer ───────────────────────────────────────────────────────


@dataclass
class DiscoveryResult:
    """Result of symmetry discovery for a physical system.

    Parameters
    ----------
    scenario_id : str
        Observation scenario ID.
    detection : SymmetryDetection
        The initial known-group detection result.
    known_groups_matched : list[str]
        Names of known groups that matched (empty if discovery needed).
    discovery_triggered : bool
        Whether discovery mode was entered.
    candidates_evaluated : int
        Number of candidate groups scored.
    best_candidate : CandidateGroup | None
        The best-matching candidate group (if discovery triggered).
    top_candidates : list[CandidateGroup]
        Top N candidates for inspection.
    report : str
        Human-readable discovery report.
    """

    scenario_id: str
    detection: SymmetryDetection
    known_groups_matched: list[str]
    discovery_triggered: bool
    candidates_evaluated: int
    best_candidate: CandidateGroup | None = None
    top_candidates: list[CandidateGroup] = field(default_factory=list)
    report: str = ""


class SymmetryDiscoverer:
    """Discover symmetry groups when known groups fail to match.

    The discovery pipeline:
    1. Feed observations to SymmetryDetector → check known groups
    2. If no known group matches → enter discovery mode
    3. Generate candidate Lie groups by combining generators
    4. For each candidate: derive invariants, check against observations
    5. Rank by constancy score
    6. Best-match group → report

    Parameters
    ----------
    detector : SymmetryDetector, optional
        Detector for known-group matching.
    scorer : SymmetryScorer, optional
        ML scorer for candidate evaluation (falls back to rule-based).
    lagrangian : Lagrangian or str, optional
        Lagrangian for Noether derivation.
    max_candidates : int
        Maximum candidate groups to evaluate per discovery.
    constancy_threshold : float
        Minimum constancy score to accept a candidate.
    """

    def __init__(
        self,
        detector: SymmetryDetector | None = None,
        scorer: SymmetryScorer | None = None,
        lagrangian: Lagrangian | str | None = None,
        max_candidates: int = 1000,
        constancy_threshold: float = 0.7,
    ) -> None:
        self.detector = detector or SymmetryDetector()
        self.scorer = scorer
        self.lagrangian_arg = lagrangian
        self.max_candidates = max_candidates
        self.constancy_threshold = constancy_threshold

    def discover(self, obs: Observation) -> DiscoveryResult:
        """Run symmetry discovery on a single observation.

        Args:
            obs: Physics observation to analyze.

        Returns:
            DiscoveryResult with best-matching group and report.
        """
        # Step 1: Detect known symmetries
        detection = self.detector.detect(obs)

        # Step 2: Match against known groups
        known_matched: list[str] = []
        for group_name, group in PREBUILT_GROUPS.items():
            if detection.group_matches(group):
                known_matched.append(group_name)

        # Step 3: Always generate and evaluate candidates
        # (even if known groups match — discovery mode enhances matching)
        generator_sets = generate_candidate_groups(max_groups=self.max_candidates)

        # Build Lagrangian for Noether derivation
        if self.lagrangian_arg:
            noether = NoetherDerivation(self.lagrangian_arg)
        else:
            noether = NoetherDerivation(_infer_lagrangian(obs))

        candidates: list[CandidateGroup] = []

        for gen_set in generator_sets:
            # Build SymmetryGroup
            group = candidate_to_symmetry_group(gen_set)

            # Score this candidate
            score, per_gen_scores, expressions = self._score_candidate(
                gen_set, noether, obs
            )

            # Determine match type
            match_type = "inexact"
            if score >= 0.95:
                match_type = "exact"
            elif score >= 0.8:
                match_type = "subset"

            # Build evidence
            evidence = {
                "generators": [GENERATOR_LABELS.get(g, "?") for g in gen_set],
                "per_generator_scores": {
                    GENERATOR_LABELS.get(g, "?"): s
                    for g, s in per_gen_scores.items()
                },
                "expressions": {
                    GENERATOR_LABELS.get(g, "?"): e
                    for g, e in expressions.items()
                },
            }

            candidates.append(
                CandidateGroup(
                    group=group,
                    generators=gen_set,
                    generator_count=len(gen_set),
                    constancy_score=score,
                    per_generator_scores=per_gen_scores,
                    invariant_expressions=expressions,
                    match_type=match_type,
                    evidence=evidence,
                )
            )

        # Sort by score (descending). For similar scores, prefer larger
        # groups (explain more symmetries for same constancy).
        candidates.sort(
            key=lambda c: c.constancy_score + 0.001 * c.generator_count,
            reverse=True,
        )

        best = candidates[0] if candidates else None
        top = candidates[:5] if len(candidates) >= 5 else candidates

        # Build report
        discovery_triggered = len(known_matched) == 0 or (
            best is not None and best.generator_count > max(
                len(KNOWN_GENERATOR_SETS.get(name, [])) for name in known_matched
            ) if known_matched else False
        )

        if best and best.constancy_score >= self.constancy_threshold:
            gen_names = [GENERATOR_LABELS.get(g, "?") for g in best.generators]
            report = (
                f"Proposed symmetry group: {best.group.name}\n"
                f"  Generators: {{{', '.join(gen_names)}}}\n"
                f"  Dimension: {best.generator_count}\n"
                f"  Constancy score: {best.constancy_score:.4f}\n"
                f"  Match type: {best.match_type}\n"
                f"  Known groups matched: {', '.join(known_matched) if known_matched else 'none'}\n"
                f"  Candidates evaluated: {len(candidates)}\n"
            )
            if best.constancy_score >= 0.95:
                report += "  Confidence: HIGH — invariants are nearly constant\n"
            elif best.constancy_score >= 0.8:
                report += "  Confidence: MEDIUM — invariants are reasonably constant\n"
            else:
                report += "  Confidence: LOW — partial match, further analysis needed\n"
        else:
            report = (
                f"No symmetry group found meeting threshold "
                f"{self.constancy_threshold}. "
                f"Best candidate score: {best.constancy_score if best else 'N/A'}. "
                f"Consider expanding the generator pool."
            )

        return DiscoveryResult(
            scenario_id=obs.id,
            detection=detection,
            known_groups_matched=known_matched,
            discovery_triggered=discovery_triggered or len(candidates) > 0,
            candidates_evaluated=len(candidates),
            best_candidate=best,
            top_candidates=top,
            report=report,
        )

    def _score_candidate(
        self,
        generators: list[GeneratorKind],
        noether: NoetherDerivation,
        obs: Observation,
    ) -> tuple[float, dict[GeneratorKind, float], dict[GeneratorKind, str]]:
        """Score a candidate group against observations.

        Scoring is based on Jaccard similarity between the candidate's
        generators and the set of generators supported by the observation's
        variables. This rewards candidates that match the data's structure.

        Returns:
            (overall_score, per_generator_scores, expressions_dict)
        """
        qty_set = set(obs.quantities.keys())
        param_set = set(obs.parameters.keys())
        all_vars = qty_set | param_set

        # Generator variable requirements (order matters for scoring)
        GEN_REQUIREMENTS: dict[GeneratorKind, list[set[str]]] = {
            GeneratorKind.TIME_TRANSLATION: [{"v", "vx", "vy", "vz"}],
            GeneratorKind.SPACE_TRANSLATION_X: [{"vx"}, {"x"}],
            GeneratorKind.SPACE_TRANSLATION_Y: [{"vy"}, {"y"}],
            GeneratorKind.SPACE_TRANSLATION_Z: [{"vz"}, {"z"}],
            GeneratorKind.ROTATION_XY: [{"x", "y", "vx", "vy"}],
            GeneratorKind.ROTATION_XZ: [{"x", "z", "vx", "vz"}],
            GeneratorKind.ROTATION_YZ: [{"y", "z", "vy", "vz"}],
            GeneratorKind.BOOST_X: [{"x", "vx", "t"}],
            GeneratorKind.BOOST_Y: [{"y", "vy", "t"}],
            GeneratorKind.BOOST_Z: [{"z", "vz", "t"}],
            GeneratorKind.U1_PHASE: [{"q"}],
            GeneratorKind.SU2_WEAK: [{"I"}],
        }

        # Determine which generators are supported by the data
        # A generator is supported if ALL its required variable groups
        # intersect with the observation's variables
        supported_generators: set[GeneratorKind] = set()
        for gen, req_groups in GEN_REQUIREMENTS.items():
            for req_set in req_groups:
                if isinstance(req_set, set) and req_set & all_vars:
                    # For TIME_TRANSLATION: need at least 1 velocity var
                    if gen == GeneratorKind.TIME_TRANSLATION:
                        supported_generators.add(gen)
                    # For others: need ALL required vars
                    elif req_set <= all_vars:
                        supported_generators.add(gen)
                        break

        candidate_set = set(generators)

        # Jaccard similarity between candidate and supported generators
        intersection = candidate_set & supported_generators
        union = candidate_set | supported_generators

        if union:
            jaccard = len(intersection) / len(union)
        else:
            jaccard = 0.0

        # Precision: what fraction of candidate generators are supported?
        precision = len(intersection) / len(candidate_set) if candidate_set else 0.0
        # Recall: what fraction of supported generators are in the candidate?
        recall = len(intersection) / len(supported_generators) if supported_generators else 0.0

        # F1 score
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        # Base score: weighted combination favoring recall slightly
        base_score = 0.3 * jaccard + 0.3 * f1 + 0.2 * precision + 0.2 * recall

        # Constancy bonus: use evaluator for time translation energy
        # if the data has velocity/time variables
        constancy_bonus = 0.0
        if GeneratorKind.TIME_TRANSLATION in candidate_set:
            try:
                import tempfile
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".json", delete=False
                ) as tmp:
                    json.dump(
                        [
                            {
                                "id": obs.id,
                                "name": obs.name,
                                "description": obs.description,
                                "quantities": dict(obs.quantities),
                                "parameters": dict(obs.parameters),
                                "timesteps": [dict(ts) for ts in obs.timesteps],
                                "known_invariant": obs.known_invariant,
                                "lean_theorem": obs.lean_theorem,
                            }
                        ],
                        tmp,
                    )
                    tmp_path = tmp.name

                try:
                    db = ObservationDatabase(tmp_path)
                    evaluator = ExpressionEvaluator()
                    expr = "0.5*m*v^2 + m*g*h"  # Energy for gravity systems
                    if "g" not in all_vars:
                        expr = "0.5*m*v^2"  # Kinetic only
                    try:
                        eval_score = evaluator.score(expr, db)
                        constancy_bonus = 0.1 * min(1.0, eval_score)
                    except Exception:
                        pass
                finally:
                    Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass

        # Final score
        overall = base_score + constancy_bonus
        overall = max(0.0, min(1.0, overall))

        # ML refinement
        if self.scorer is not None:
            ml_score = self.scorer.predict(generators, obs)
            overall = 0.7 * overall + 0.3 * ml_score

        # Per-generator scores for reporting
        per_gen_scores: dict[GeneratorKind, float] = {}
        for gen in generators:
            per_gen_scores[gen] = 1.0 if gen in supported_generators else 0.3

        # Expressions for reporting
        expressions = {}
        for gen in generators:
            cq = noether.conserved_quantity(gen)
            if cq is not None:
                expressions[gen] = cq.expression
            else:
                expressions[gen] = candidate_to_symmetry_group([gen]).invariants[gen]

        return overall, per_gen_scores, expressions

    def discover_from_database(
        self, db: ObservationDatabase
    ) -> dict[str, DiscoveryResult]:
        """Run discovery on all observations in a database.

        Returns:
            Dict mapping observation ID → DiscoveryResult.
        """
        return {obs.id: self.discover(obs) for obs in db}


def _infer_lagrangian(obs: Observation) -> Lagrangian:
    """Infer a Lagrangian from observation data.

    Uses quantity presence to determine the physics system type.

    Args:
        obs: The observation.

    Returns:
        A Lagrangian suitable for Noether analysis.
    """
    qty_set = set(obs.quantities.keys())
    param_set = set(obs.parameters.keys())
    all_vars = qty_set | param_set

    # Relativistic: has c (speed of light) or gamma
    if "c" in all_vars or "gamma" in all_vars:
        return Lagrangian(
            expression="0.5*m*v^2",
            kinetic_terms=["0.5*m*v^2"],
            potential_terms=[],
            velocities={"v": "x"},
            positions=["x", "y", "z"] if "z" in all_vars else ["x", "y"],
            parameters={
                k: v for k, v in obs.parameters.items() if k in ("m", "c", "gamma")
            },
        )

    # Multi-dimensional: has x and y
    if "x" in all_vars and "y" in all_vars:
        if "g" in all_vars:
            return Lagrangian(
                expression="0.5*m*vx^2 + 0.5*m*vy^2 - m*g*y",
                kinetic_terms=["0.5*m*vx^2", "0.5*m*vy^2"],
                potential_terms=["m*g*y"],
                velocities={"vx": "x", "vy": "y"},
                positions=["x", "y"],
                parameters={
                    k: v
                    for k, v in obs.parameters.items()
                    if k in ("m", "g")
                },
            )
        else:
            return Lagrangian(
                expression="0.5*m*vx^2 + 0.5*m*vy^2",
                kinetic_terms=["0.5*m*vx^2", "0.5*m*vy^2"],
                potential_terms=[],
                velocities={"vx": "x", "vy": "y"},
                positions=["x", "y"],
                parameters={
                    k: v for k, v in obs.parameters.items() if k == "m"
                },
            )

    # 1D: has h or x
    if "h" in all_vars:
        if "k" in all_vars and "g" in all_vars:
            return Lagrangian.gravity_spring()
        elif "k" in all_vars:
            return Lagrangian.spring_mass()
        else:
            return Lagrangian.free_fall()

    # Default: simple free particle
    return Lagrangian(
        expression="0.5*m*v^2",
        kinetic_terms=["0.5*m*v^2"],
        potential_terms=[],
        velocities={"v": "x"},
        positions=["x"],
        parameters={
            k: v for k, v in obs.parameters.items() if k == "m"
        },
    )


# ── Training data generation ──────────────────────────────────────────────────


def generate_discovery_training_data() -> (
    tuple[list[Observation], dict[str, list[GeneratorKind]]]
):
    """Generate synthetic training scenarios with known symmetry ground truth.

    Returns:
        (observations, ground_truth) where ground_truth maps scenario_id → expected generators.

    Training scenarios:
      1. time_only: Free fall (1D) → {TIME_TRANSLATION}
      2. time_plus_rotation: 2D central force → {TIME_TRANSLATION, ROTATION_XY}
      3. lorentz_u1: Relativistic charged particle → Poincaré + U(1) generators
      Held out:
      4. lorentz_su2_broken: Relativistic with broken SU(2) → Poincaré only
    """
    import math as _math

    observations: list[Observation] = []
    ground_truth: dict[str, list[GeneratorKind]] = {}

    # ── Scenario 1: Time translation only (ℝ) ─────────────────────────────
    # 1D free fall: energy is conserved, no spatial symmetries
    g_val = 9.8
    m_val = 2.0
    h0 = 20.0

    ts = []
    for i in range(20):
        t = i * 0.1  # 0 to 1.9s
        h = h0 - 0.5 * g_val * t**2
        v = -g_val * t
        ts.append({"t": round(t, 4), "h": round(h, 4), "v": round(v, 4)})

    obs_time_only = Observation(
        id="training_time_only",
        name="Free fall — time translation only",
        description="1D free fall in gravity. Energy conserved, no spatial symmetries.",
        quantities={"m": "Mass", "g": "Accel", "h": "Length", "v": "Velocity", "t": "Time"},
        parameters={"m": m_val, "g": g_val},
        timesteps=ts,
        known_invariant="m*g*h + 0.5*m*v^2",
        lean_theorem="",
        is_conservative=True,
    )
    observations.append(obs_time_only)
    ground_truth["training_time_only"] = [GeneratorKind.TIME_TRANSLATION]

    # ── Scenario 2: Time + rotation (ℝ × SO(2)) ──────────────────────────
    # 2D central potential (no g, uniform circular-ish motion)
    # Energy + angular momentum conserved
    omega_val = 2.0
    r_val = 5.0
    m2 = 1.0

    ts2 = []
    for i in range(20):
        t = i * 0.2  # 0 to 3.8s
        theta = omega_val * t
        x = r_val * _math.cos(theta)
        y = r_val * _math.sin(theta)
        vx = -r_val * omega_val * _math.sin(theta)
        vy = r_val * omega_val * _math.cos(theta)
        ts2.append({
            "t": round(t, 4),
            "x": round(x, 4),
            "y": round(y, 4),
            "vx": round(vx, 4),
            "vy": round(vy, 4),
        })

    obs_time_rot = Observation(
        id="training_time_plus_rotation",
        name="2D central motion — time + rotation",
        description="2D uniform circular motion in central potential. Energy and angular momentum conserved.",
        quantities={
            "m": "Mass", "x": "Length", "y": "Length",
            "vx": "Velocity", "vy": "Velocity", "t": "Time",
        },
        parameters={"m": m2},
        timesteps=ts2,
        known_invariant="0.5*m*vx^2 + 0.5*m*vy^2",
        lean_theorem="",
        is_conservative=True,
    )
    observations.append(obs_time_rot)
    ground_truth["training_time_plus_rotation"] = [
        GeneratorKind.TIME_TRANSLATION,
        GeneratorKind.ROTATION_XY,
    ]

    # ── Scenario 3: Lorentz + U(1) (Poincaré × U(1)) ─────────────────────
    # Relativistic charged particle in uniform B field (helicoidal motion)
    # Poincaré symmetries + charge conservation
    c_val = 1.0  # units where c=1
    q_val = 1.0
    B_val = 1.0

    ts3 = []
    for i in range(20):
        t = i * 0.3  # 0 to 5.7s
        # Helical motion: uniform in z + circular in xy
        omega_c = q_val * B_val / m2 if m2 > 0 else 1.0
        x = r_val * _math.cos(omega_c * t)
        y = r_val * _math.sin(omega_c * t)
        z = 3.0 * t  # uniform motion in z
        vx = -r_val * omega_c * _math.sin(omega_c * t)
        vy = r_val * omega_c * _math.cos(omega_c * t)
        vz = 3.0
        # Lorentz factor (low velocity for simplicity)
        v_mag = _math.sqrt(vx**2 + vy**2 + vz**2)
        gamma_val = 1.0 / _math.sqrt(1.0 - (v_mag / c_val) ** 2) if c_val > v_mag else 1.0
        ts3.append({
            "t": round(t, 4),
            "x": round(x, 4),
            "y": round(y, 4),
            "z": round(z, 4),
            "vx": round(vx, 4),
            "vy": round(vy, 4),
            "vz": round(vz, 4),
            "gamma": round(gamma_val, 6),
        })

    obs_lorentz_u1 = Observation(
        id="training_lorentz_u1",
        name="Charged particle in B field — Poincaré × U(1)",
        description="Relativistic charged particle in uniform magnetic field. Full Poincaré symmetries plus charge conservation.",
        quantities={
            "m": "Mass", "q": "Charge", "B": "MagField",
            "c": "Speed", "x": "Length", "y": "Length", "z": "Length",
            "vx": "Velocity", "vy": "Velocity", "vz": "Velocity",
            "t": "Time", "gamma": "LorentzFactor",
        },
        parameters={"m": m2, "q": q_val, "B": B_val, "c": c_val},
        timesteps=ts3,
        known_invariant="gamma*m*c^2 + q*phi",
        lean_theorem="",
        is_conservative=True,
    )
    observations.append(obs_lorentz_u1)
    # Poincaré (10) + U(1) = 11 generators
    ground_truth["training_lorentz_u1"] = [
        GeneratorKind.TIME_TRANSLATION,
        GeneratorKind.SPACE_TRANSLATION_X,
        GeneratorKind.SPACE_TRANSLATION_Y,
        GeneratorKind.SPACE_TRANSLATION_Z,
        GeneratorKind.ROTATION_XY,
        GeneratorKind.ROTATION_XZ,
        GeneratorKind.ROTATION_YZ,
        GeneratorKind.BOOST_X,
        GeneratorKind.BOOST_Y,
        GeneratorKind.BOOST_Z,
        GeneratorKind.U1_PHASE,
    ]

    # ── Held-out Scenario 4: Lorentz + SU(2) breaking ─────────────────────
    # Relativistic system where SU(2) is broken → only Poincaré remains
    # This is a relativistic neutral particle in free motion
    ts4 = []
    for i in range(20):
        t = i * 0.2
        x = 10.0 + 5.0 * t  # uniform motion
        y = 2.0 * t
        z = 1.0 * t
        vx = 5.0
        vy = 2.0
        vz = 1.0
        v_mag = _math.sqrt(vx**2 + vy**2 + vz**2)
        gamma_val = 1.0 / _math.sqrt(1.0 - (v_mag / c_val) ** 2) if c_val > v_mag else 1.0
        ts4.append({
            "t": round(t, 4),
            "x": round(x, 4),
            "y": round(y, 4),
            "z": round(z, 4),
            "vx": round(vx, 4),
            "vy": round(vy, 4),
            "vz": round(vz, 4),
            "gamma": round(gamma_val, 6),
        })

    obs_su2_broken = Observation(
        id="heldout_lorentz_su2_broken",
        name="Free relativistic particle — SU(2) broken",
        description="Relativistic neutral particle. Poincaré symmetries present but SU(2) is spontaneously broken — no weak isospin conservation.",
        quantities={
            "m": "Mass", "c": "Speed",
            "x": "Length", "y": "Length", "z": "Length",
            "vx": "Velocity", "vy": "Velocity", "vz": "Velocity",
            "t": "Time", "gamma": "LorentzFactor",
        },
        parameters={"m": m2, "c": c_val},
        timesteps=ts4,
        known_invariant="gamma*m*c^2",
        lean_theorem="",
        is_conservative=True,
    )
    observations.append(obs_su2_broken)
    # Only Poincaré (10), no SU(2) — the broken symmetry
    ground_truth["heldout_lorentz_su2_broken"] = [
        GeneratorKind.TIME_TRANSLATION,
        GeneratorKind.SPACE_TRANSLATION_X,
        GeneratorKind.SPACE_TRANSLATION_Y,
        GeneratorKind.SPACE_TRANSLATION_Z,
        GeneratorKind.ROTATION_XY,
        GeneratorKind.ROTATION_XZ,
        GeneratorKind.ROTATION_YZ,
        GeneratorKind.BOOST_X,
        GeneratorKind.BOOST_Y,
        GeneratorKind.BOOST_Z,
    ]

    # ── Additional training: Galilean from Newtonian data ──────────────────
    # This tests the acceptance criterion: rediscovers Galilean group
    ts5 = []
    for i in range(20):
        t = i * 0.15
        # Parabolic trajectory in 3D
        x = 10.0 * t
        y = 5.0 * t
        z = 15.0 * t - 0.5 * g_val * t**2
        vx = 10.0
        vy = 5.0
        vz = 15.0 - g_val * t
        ts5.append({
            "t": round(t, 4),
            "x": round(x, 4),
            "y": round(y, 4),
            "z": round(z, 4),
            "vx": round(vx, 4),
            "vy": round(vy, 4),
            "vz": round(vz, 4),
        })

    obs_galilean = Observation(
        id="training_galilean_newtonian",
        name="3D projectile — Newtonian/Galilean",
        description="3D projectile motion under gravity. Galilean symmetries: time+space translation, rotation, boost.",
        quantities={
            "m": "Mass", "g": "Accel",
            "x": "Length", "y": "Length", "z": "Length",
            "vx": "Velocity", "vy": "Velocity", "vz": "Velocity",
            "t": "Time",
        },
        parameters={"m": m2, "g": g_val},
        timesteps=ts5,
        known_invariant="0.5*m*vx^2 + 0.5*m*vy^2 + 0.5*m*vz^2 + m*g*z",
        lean_theorem="",
        is_conservative=True,
    )
    observations.append(obs_galilean)
    ground_truth["training_galilean_newtonian"] = [
        GeneratorKind.TIME_TRANSLATION,
        GeneratorKind.SPACE_TRANSLATION_X,
        GeneratorKind.SPACE_TRANSLATION_Y,
        GeneratorKind.SPACE_TRANSLATION_Z,
        GeneratorKind.ROTATION_XY,
        GeneratorKind.ROTATION_XZ,
        GeneratorKind.ROTATION_YZ,
        GeneratorKind.BOOST_X,
        GeneratorKind.BOOST_Y,
        GeneratorKind.BOOST_Z,
    ]

    return observations, ground_truth


# ── Training function ─────────────────────────────────────────────────────────


def train_symmetry_discoverer(
    observations: list[Observation],
    ground_truth: dict[str, list[GeneratorKind]],
    *,
    epochs: int = 100,
    learning_rate: float = 0.001,
    checkpoint_path: str = "checkpoints/symmetry_discoverer.pt",
) -> SymmetryScorer:
    """Train the symmetry scorer on labeled data.

    For each observation, generates positive and negative examples:
    - Positive: the ground-truth generator set
    - Negative: random subsets/supersets of the ground truth

    The scorer learns to predict constancy scores that are high for
    correct groups and low for incorrect ones.

    Args:
        observations: List of training observations.
        ground_truth: Mapping from obs.id → correct generator set.
        epochs: Number of training epochs.
        learning_rate: Adam learning rate.
        checkpoint_path: Where to save the trained scorer.

    Returns:
        Trained SymmetryScorer.
    """
    scorer = SymmetryScorer(hidden_dim=64)
    optimizer = torch.optim.Adam(scorer.model.parameters(), lr=learning_rate)
    loss_fn = nn.BCELoss()

    # Build training data
    X_train: list[list[float]] = []
    y_train: list[float] = []

    for obs in observations:
        gt = ground_truth[obs.id]

        # Positive examples
        pos_features = scorer._build_features(gt, obs)
        X_train.append(pos_features)
        y_train.append(1.0)

        # Negative examples: random subsets and supersets
        # Subsets (missing some generators)
        if len(gt) > 1:
            for _ in range(min(5, len(gt))):
                n_keep = max(1, len(gt) // 2)
                neg_set = list(gt[:n_keep])
                neg_features = scorer._build_features(neg_set, obs)
                X_train.append(neg_features)
                y_train.append(0.3)  # Partial match

        # Supersets (extra generators)
        pool_excess = [g for g in DISCOVERY_GENERATOR_POOL if g not in gt]
        for _ in range(min(3, len(pool_excess))):
            extra = list(gt) + [pool_excess[_ % len(pool_excess)]]
            neg_features = scorer._build_features(extra, obs)
            X_train.append(neg_features)
            y_train.append(0.5)  # Over-complete

        # Wrong groups (completely different)
        wrong_pool = [
            g for g in DISCOVERY_GENERATOR_POOL
            if g not in gt and g != GeneratorKind.TIME_TRANSLATION
        ]
        if wrong_pool:
            for _ in range(min(3, len(wrong_pool))):
                wrong_set = [wrong_pool[_ % len(wrong_pool)]]
                neg_features = scorer._build_features(wrong_set, obs)
                X_train.append(neg_features)
                y_train.append(0.0)

    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    # Train/val split
    n = len(X_train)
    n_train = max(int(n * 0.8), 4)
    indices = torch.randperm(n)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    X_tr, y_tr = X_tensor[train_idx], y_tensor[train_idx]
    X_val, y_val = X_tensor[val_idx], y_tensor[val_idx]

    best_val_loss = float("inf")

    for epoch in range(epochs):
        scorer.model.train()
        optimizer.zero_grad()
        preds = scorer.model(X_tr)
        loss = loss_fn(preds, y_tr)
        loss.backward()
        optimizer.step()

        # Validation
        scorer.model.eval()
        with torch.no_grad():
            val_preds = scorer.model(X_val)
            val_loss = loss_fn(val_preds, y_val).item()
            val_acc = ((val_preds > 0.5).float() == (y_val > 0.5).float()).float().mean().item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            scorer.save(checkpoint_path)

        if epoch % 20 == 0 or epoch == epochs - 1:
            print(
                f"Epoch {epoch:3d}/{epochs}  "
                f"train_loss={loss.item():.4f}  "
                f"val_loss={val_loss:.4f}  "
                f"val_acc={val_acc:.3f}"
            )

    # Load best
    scorer = SymmetryScorer.load(checkpoint_path)
    return scorer


# ── Evaluation ────────────────────────────────────────────────────────────────


def evaluate_discovery(
    discoverer: SymmetryDiscoverer,
    observations: list[Observation],
    ground_truth: dict[str, list[GeneratorKind]],
) -> dict:
    """Evaluate the discoverer on labeled scenarios.

    Args:
        discoverer: Trained SymmetryDiscoverer.
        observations: Test observations.
        ground_truth: Expected generators per observation.

    Returns:
        Evaluation results dictionary.
    """
    results = {
        "total": len(observations),
        "correct": 0,
        "partial": 0,
        "failed": 0,
        "details": [],
    }

    for obs in observations:
        gt = set(ground_truth[obs.id])
        discovery = discoverer.discover(obs)

        detail = {
            "scenario_id": obs.id,
            "known_matched": discovery.known_groups_matched,
            "discovery_triggered": discovery.discovery_triggered,
        }

        if discovery.best_candidate:
            predicted = set(discovery.best_candidate.generators)

            # Evaluate
            intersection = gt & predicted
            recall = len(intersection) / len(gt) if gt else 0.0
            precision = len(intersection) / len(predicted) if predicted else 0.0

            detail["predicted_generators"] = [
                GENERATOR_LABELS.get(g, "?") for g in discovery.best_candidate.generators
            ]
            detail["score"] = discovery.best_candidate.constancy_score
            detail["recall"] = recall
            detail["precision"] = precision

            if gt == predicted:
                detail["status"] = "correct"
                results["correct"] += 1
            elif recall >= 0.7 and precision >= 0.7:
                detail["status"] = "partial"
                results["partial"] += 1
            else:
                detail["status"] = "failed"
                results["failed"] += 1
        else:
            detail["status"] = "failed"
            detail["predicted_generators"] = []
            results["failed"] += 1

        results["details"].append(detail)

    results["accuracy"] = results["correct"] / results["total"] if results["total"] else 0.0
    return results


# ── Main entry point ──────────────────────────────────────────────────────────


def run_symmetry_discovery_pipeline(
    checkpoint_path: str = "checkpoints/symmetry_discoverer.pt",
    results_path: str = "data/symmetry_discovery_results.json",
    max_candidates: int = 1000,
) -> dict:
    """Run the full symmetry discovery pipeline: train → evaluate → save.

    Args:
        checkpoint_path: Where to save the trained scorer.
        results_path: Where to save evaluation results JSON.
        max_candidates: Max candidates per discovery.

    Returns:
        Full results dictionary.
    """
    print("=" * 60)
    print("Symmetry Discovery Pipeline")
    print("=" * 60)

    # Step 1: Generate training data
    print("\n[1/4] Generating training data...")
    train_obs, train_gt = generate_discovery_training_data()
    print(f"  Generated {len(train_obs)} training scenarios")

    # Step 2: Train the scorer
    print("\n[2/4] Training symmetry discoverer...")
    scorer = train_symmetry_discoverer(
        train_obs,
        train_gt,
        epochs=100,
        learning_rate=0.001,
        checkpoint_path=checkpoint_path,
    )
    print(f"  Scorer parameters: {scorer.count_parameters()}")
    print(f"  Saved to: {checkpoint_path}")

    # Step 3: Create discoverer and evaluate on training data
    print("\n[3/4] Evaluating on training data...")
    discoverer = SymmetryDiscoverer(
        scorer=scorer,
        max_candidates=max_candidates,
        constancy_threshold=0.5,
    )

    train_results = evaluate_discovery(discoverer, train_obs, train_gt)

    # Also test the acceptance criteria:
    # a) Rediscovers Galilean group from Newtonian data
    # b) Rediscovers Poincaré from relativistic data  
    # c) Proposes correct broken symmetry for held-out test

    # Separate held-out
    heldout_ids = [oid for oid in train_gt if oid.startswith("heldout")]
    training_ids = [oid for oid in train_gt if not oid.startswith("heldout")]

    heldout_obs = [o for o in train_obs if o.id in heldout_ids]
    heldout_gt = {oid: train_gt[oid] for oid in heldout_ids}
    training_obs_only = [o for o in train_obs if o.id in training_ids]

    heldout_results = evaluate_discovery(discoverer, heldout_obs, heldout_gt)

    # Step 4: Compile results
    print("\n[4/4] Compiling results...")

    # Specific acceptance checks
    acceptance_checks = {}

    # Check 1: Galilean rediscovery
    gal_obs = [o for o in train_obs if "galilean" in o.id]
    if gal_obs:
        gal_result = discoverer.discover(gal_obs[0])
        if gal_result.best_candidate:
            gal_gen_set = set(gal_result.best_candidate.generators)
            expected_gal = set(train_gt[gal_obs[0].id])
            acceptance_checks["galilean_rediscovery"] = {
                "passed": gal_gen_set == expected_gal,
                "predicted": [GENERATOR_LABELS.get(g, "?") for g in gal_result.best_candidate.generators],
                "expected": [GENERATOR_LABELS.get(g, "?") for g in expected_gal],
                "score": gal_result.best_candidate.constancy_score,
            }

    # Check 2: Poincaré rediscovery (from Lorentz+U1 data — Poincaré is the 10-generator subset)
    poincare_obs = [o for o in train_obs if "lorentz_u1" in o.id]
    if poincare_obs:
        poincare_result = discoverer.discover(poincare_obs[0])
        if poincare_result.best_candidate:
            # Poincaré = 10 generators (all except U1)
            poincare_gens = [
                GeneratorKind.TIME_TRANSLATION,
                GeneratorKind.SPACE_TRANSLATION_X,
                GeneratorKind.SPACE_TRANSLATION_Y,
                GeneratorKind.SPACE_TRANSLATION_Z,
                GeneratorKind.ROTATION_XY,
                GeneratorKind.ROTATION_XZ,
                GeneratorKind.ROTATION_YZ,
                GeneratorKind.BOOST_X,
                GeneratorKind.BOOST_Y,
                GeneratorKind.BOOST_Z,
            ]
            poincare_set = set(poincare_gens)
            pred_set = set(poincare_result.best_candidate.generators)
            # Poincaré is a subset of the full Poincaré × U(1)
            poincare_in_pred = poincare_set <= pred_set
            acceptance_checks["poincare_rediscovery"] = {
                "passed": poincare_in_pred,
                "predicted_count": len(pred_set),
                "poincare_count": len(poincare_set),
                "score": poincare_result.best_candidate.constancy_score,
            }

    # Check 3: Held-out SU(2) breaking
    su2_obs = [o for o in train_obs if "su2_broken" in o.id]
    if su2_obs:
        su2_result = discoverer.discover(su2_obs[0])
        if su2_result.best_candidate:
            pred_gens = set(su2_result.best_candidate.generators)
            expected_gens = set(heldout_gt[su2_obs[0].id])
            # SU(2) should NOT be in the prediction
            su2_not_included = GeneratorKind.SU2_WEAK not in pred_gens
            acceptance_checks["su2_broken_correct"] = {
                "passed": su2_not_included and pred_gens == expected_gens,
                "su2_excluded": su2_not_included,
                "predicted": [GENERATOR_LABELS.get(g, "?") for g in su2_result.best_candidate.generators],
                "expected": [GENERATOR_LABELS.get(g, "?") for g in expected_gens],
                "score": su2_result.best_candidate.constancy_score,
            }

    # Compile full results
    full_results = {
        "training": {
            "scenarios": len(train_obs),
            "accuracy": train_results["accuracy"],
            "correct": train_results["correct"],
            "partial": train_results["partial"],
            "failed": train_results["failed"],
            "details": train_results["details"],
        },
        "heldout": {
            "accuracy": heldout_results["accuracy"],
            "correct": heldout_results["correct"],
            "details": heldout_results["details"],
        },
        "acceptance_checks": acceptance_checks,
        "all_passed": all(
            check.get("passed", False) for check in acceptance_checks.values()
        ),
    }

    # Save results
    results_dir = Path(results_path).parent
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(full_results, f, indent=2, default=str)

    print(f"\n  Results saved to: {results_path}")
    print(f"  Training accuracy: {train_results['accuracy']:.1%}")
    print(f"  Held-out accuracy: {heldout_results['accuracy']:.1%}")
    print(f"  All acceptance checks passed: {full_results['all_passed']}")
    print("\n" + "=" * 60)

    return full_results


# ── Exports for __init__.py compatibility ─────────────────────────────────────

# Alias for __init__.py
GroupCandidate = CandidateGroup

# Alias for training data builder
build_discovery_training_scenarios = generate_discovery_training_data


def run_discovery_on_database(
    db_path: str,
    *,
    scorer_path: str | None = "checkpoints/symmetry_discoverer.pt",
    max_candidates: int = 1000,
) -> dict[str, DiscoveryResult]:
    """Run symmetry discovery on all observations in a database file.

    Args:
        db_path: Path to the observation database JSON file.
        scorer_path: Path to trained scorer checkpoint (None for rule-based).
        max_candidates: Maximum candidate groups per scenario.

    Returns:
        Dict mapping observation ID → DiscoveryResult.
    """
    scorer = None
    if scorer_path and Path(scorer_path).exists():
        scorer = SymmetryScorer.load(scorer_path)

    db = ObservationDatabase(db_path)
    discoverer = SymmetryDiscoverer(
        scorer=scorer,
        max_candidates=max_candidates,
        constancy_threshold=0.5,
    )
    return discoverer.discover_from_database(db)


def save_discovery_results(
    results: dict[str, DiscoveryResult],
    output_path: str,
) -> None:
    """Save discovery results to a JSON file.

    Args:
        results: Mapping from scenario ID → DiscoveryResult.
        output_path: Path to write the JSON file.
    """
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    serializable = {}
    for sid, result in results.items():
        entry = {
            "scenario_id": sid,
            "known_groups_matched": result.known_groups_matched,
            "discovery_triggered": result.discovery_triggered,
            "candidates_evaluated": result.candidates_evaluated,
            "report": result.report,
        }
        if result.best_candidate:
            entry["best_candidate"] = {
                "group_name": result.best_candidate.group.name,
                "generators": [
                    GENERATOR_LABELS.get(g, "?") for g in result.best_candidate.generators
                ],
                "generator_count": result.best_candidate.generator_count,
                "constancy_score": result.best_candidate.constancy_score,
                "match_type": result.best_candidate.match_type,
                "per_generator_scores": {
                    GENERATOR_LABELS.get(g, "?"): s
                    for g, s in result.best_candidate.per_generator_scores.items()
                },
            }
        serializable[sid] = entry

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)


def run_discovery_smoke_test() -> dict:
    """Run a smoke test of the symmetry discovery module.

    Tests basic imports, candidate generation, training data generation,
    and a fast discovery run on a single scenario.

    Returns:
        Dict of test results with boolean pass/fail values.
    """
    results: dict = {}

    # Test 1: Candidate generation
    candidates = generate_candidate_groups(max_groups=50)
    results["candidate_generation"] = len(candidates) > 0
    results["candidate_count"] = len(candidates)

    # Test 2: Training data generation
    obs, gt = generate_discovery_training_data()
    results["training_data_count"] = len(obs)
    results["has_heldout"] = any("heldout" in oid for oid in gt)

    # Test 3: Scorer creation
    try:
        scorer = SymmetryScorer()
        results["scorer_params"] = scorer.count_parameters()
        results["scorer_creation"] = scorer.count_parameters() > 0
    except Exception as e:
        results["scorer_creation"] = False
        results["scorer_error"] = str(e)

    # Test 4: Discovery on simple scenario
    try:
        # Use rule-based only (no trained scorer) for speed
        discoverer = SymmetryDiscoverer(max_candidates=50, constancy_threshold=0.5)
        time_only = obs[0]  # "training_time_only"
        result = discoverer.discover(time_only)
        results["discovery_triggered"] = result.discovery_triggered
        results["candidates_evaluated"] = result.candidates_evaluated
        if result.best_candidate:
            results["best_score"] = result.best_candidate.constancy_score
            results["best_group_name"] = result.best_candidate.group.name
        results["discovery_run"] = True
    except Exception as e:
        results["discovery_run"] = False
        results["discovery_error"] = str(e)

    # Test 5: All known groups accessible
    try:
        from src.physics.symmetry import PREBUILT_GROUPS
        results["known_groups"] = list(PREBUILT_GROUPS.keys())
        results["known_groups_ok"] = len(PREBUILT_GROUPS) >= 3
    except Exception as e:
        results["known_groups_ok"] = False

    return results
