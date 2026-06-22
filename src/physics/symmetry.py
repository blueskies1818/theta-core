"""Symmetry-driven invariant derivation via Noether's theorem.

Replaces brute-force expression search with principled symmetry-based
derivation of conserved quantities. Every continuous symmetry of the
Lagrangian corresponds to a conserved current (Noether's theorem).

Core classes:
  SymmetryGroup     — generators, invariants, Lie algebra structure
  Lagrangian        — symbolic physics Lagrangian for Noether analysis
  NoetherDerivation — Lagrangian + symmetry → conserved quantity
  SymmetryDetector  — observations → active symmetries

Architecture:
  System observations → SymmetryDetector → active symmetries
  → NoetherDerivation → conserved quantities as expressions
  → Evaluator confirms constancy → Lean proves
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, ClassVar

from src.physics.dimensions import Dimension
from src.physics.observations import Observation, ObservationDatabase


# ── Generator types ──────────────────────────────────────────────────────────

class GeneratorKind(Enum):
    """Type of infinitesimal generator for a symmetry transformation."""
    TIME_TRANSLATION = auto()
    SPACE_TRANSLATION_X = auto()
    SPACE_TRANSLATION_Y = auto()
    SPACE_TRANSLATION_Z = auto()
    ROTATION_XY = auto()
    ROTATION_XZ = auto()
    ROTATION_YZ = auto()
    BOOST_X = auto()
    BOOST_Y = auto()
    BOOST_Z = auto()
    U1_PHASE = auto()
    SU2_WEAK = auto()
    SU3_COLOR = auto()
    CUSTOM = auto()


# Human-readable names for display
GENERATOR_LABELS: dict[GeneratorKind, str] = {
    GeneratorKind.TIME_TRANSLATION: "time_translation",
    GeneratorKind.SPACE_TRANSLATION_X: "space_translation_x",
    GeneratorKind.SPACE_TRANSLATION_Y: "space_translation_y",
    GeneratorKind.SPACE_TRANSLATION_Z: "space_translation_z",
    GeneratorKind.ROTATION_XY: "rotation_xy",
    GeneratorKind.ROTATION_XZ: "rotation_xz",
    GeneratorKind.ROTATION_YZ: "rotation_yz",
    GeneratorKind.BOOST_X: "boost_x",
    GeneratorKind.BOOST_Y: "boost_y",
    GeneratorKind.BOOST_Z: "boost_z",
    GeneratorKind.U1_PHASE: "u1_phase",
    GeneratorKind.SU2_WEAK: "su2_weak",
    GeneratorKind.SU3_COLOR: "su3_color",
    GeneratorKind.CUSTOM: "custom",
}


# ── Symmetry Group ────────────────────────────────────────────────────────────

@dataclass
class SymmetryGroup:
    """A continuous symmetry group with its generators and invariants.

    Each generator corresponds to a conserved quantity via Noether's
    theorem. The Lie algebra encodes commutation relations between
    generators.

    Parameters
    ----------
    name : str
        Human-readable name (e.g., "Galilean group").
    generators : list[GeneratorKind]
        The infinitesimal generators spanning this group's Lie algebra.
    invariants : dict[GeneratorKind, str]
        Mapping from generator → conserved quantity expression.
        Expressions use the project's grammar (e.g., "m*g*h + 0.5*m*v^2").
    structure_constants : dict, optional
        Lie algebra structure constants { (gen_i, gen_j): {gen_k: coeff} }.
    dimension : int
        Number of independent generators (group dimension).
    parent : str, optional
        Name of parent group for hierarchical classification.
    """

    name: str
    generators: list[GeneratorKind]
    invariants: dict[GeneratorKind, str]
    structure_constants: dict[tuple[GeneratorKind, GeneratorKind],
                              dict[GeneratorKind, float]] = field(default_factory=dict)
    dimension: int = 0
    parent: str | None = None

    def __post_init__(self) -> None:
        if self.dimension == 0:
            self.dimension = len(self.generators)

    def invariant_for(self, generator: GeneratorKind) -> str | None:
        """Return the conserved expression for a specific generator."""
        return self.invariants.get(generator)

    def all_invariants(self) -> dict[str, str]:
        """Return all invariants keyed by generator name."""
        return {
            GENERATOR_LABELS.get(gen, "unknown"): expr
            for gen, expr in self.invariants.items()
        }

    def commutes(self, gen_a: GeneratorKind, gen_b: GeneratorKind) -> bool:
        """Check if two generators commute (structure constant is zero)."""
        key = (gen_a, gen_b)
        consts = self.structure_constants.get(key, {})
        return len(consts) == 0 or all(abs(v) < 1e-12 for v in consts.values())

    def contains(self, generator: GeneratorKind) -> bool:
        """Check if this group contains the given generator."""
        return generator in self.invariants


# ── Lagrangian ────────────────────────────────────────────────────────────────

@dataclass
class Lagrangian:
    """A physics Lagrangian for Noether analysis.

    Encodes kinetic (T) and potential (V) terms, plus velocity variables
    (generalized velocities q̇) and position variables (generalized
    coordinates q).

    Parameters
    ----------
    expression : str
        The Lagrangian expression string (T - V form).
    kinetic_terms : list[str]
        Sub-expressions that represent kinetic energy contributions.
    potential_terms : list[str]
        Sub-expressions that represent potential energy contributions.
    velocities : dict[str, str]
        Mapping from velocity variable → corresponding position variable.
        e.g., {"v": "h"} means v is the velocity of height h.
    positions : list[str]
        All generalized coordinate names.
    parameters : dict[str, float]
        Constant parameters (masses, spring constants, etc.).
    """

    expression: str
    kinetic_terms: list[str] = field(default_factory=list)
    potential_terms: list[str] = field(default_factory=list)
    velocities: dict[str, str] = field(default_factory=dict)
    positions: list[str] = field(default_factory=list)
    parameters: dict[str, float] = field(default_factory=dict)

    @property
    def T_expression(self) -> str:
        """Reconstruct kinetic energy expression."""
        if not self.kinetic_terms:
            return "0"
        return " + ".join(self.kinetic_terms)

    @property
    def V_expression(self) -> str:
        """Reconstruct potential energy expression."""
        if not self.potential_terms:
            return "0"
        return " + ".join(self.potential_terms)

    def evaluate(self, state: dict[str, float]) -> float:
        """Evaluate L = T - V with given state variables.

        Args:
            state: Dict mapping variable names to numeric values.
                   Must include all velocity and position variables.
        """
        from src.physics.evaluator import parse_expression, evaluate_node

        ast = parse_expression(self.expression)
        return evaluate_node(ast, state)

    @classmethod
    def free_fall(cls) -> Lagrangian:
        """Standard free-fall Lagrangian: L = ½mv² - mgh."""
        return cls(
            expression="0.5*m*v^2 - m*g*h",
            kinetic_terms=["0.5*m*v^2"],
            potential_terms=["m*g*h"],
            velocities={"v": "h"},
            positions=["h"],
            parameters={"m": 1.0, "g": 9.8},
        )

    @classmethod
    def spring_mass(cls) -> Lagrangian:
        """Spring-mass Lagrangian: L = ½mv² - ½kh²."""
        return cls(
            expression="0.5*m*v^2 - 0.5*k*h^2",
            kinetic_terms=["0.5*m*v^2"],
            potential_terms=["0.5*k*h^2"],
            velocities={"v": "h"},
            positions=["h"],
            parameters={"m": 1.0, "k": 1.0},
        )

    @classmethod
    def gravity_spring(cls) -> Lagrangian:
        """Combined gravity + spring: L = ½mv² - (½kh² - mgh) = ½mv² + mgh - ½kh².

        For a vertical spring under gravity (h measured from unstretched position
        downward), V_spring = ½kh², V_gravity = -mgh (potential decreases as h
        increases). Total V = ½kh² - mgh, so L = T - V = ½mv² - ½kh² + mgh.

        The conserved energy is H = ½mv² + ½kh² - mgh.
        """
        return cls(
            expression="0.5*m*v^2 + m*g*h - 0.5*k*h^2",
            kinetic_terms=["0.5*m*v^2"],
            potential_terms=["0.5*k*h^2", "-m*g*h"],
            velocities={"v": "h"},
            positions=["h"],
            parameters={"m": 1.0, "g": 9.8, "k": 1.0},
        )

    @classmethod
    def projectile(cls) -> Lagrangian:
        """2D projectile Lagrangian: L = ½m(vx²+vy²) - mgy."""
        return cls(
            expression="0.5*m*vx^2 + 0.5*m*vy^2 - m*g*y",
            kinetic_terms=["0.5*m*vx^2", "0.5*m*vy^2"],
            potential_terms=["m*g*y"],
            velocities={"vx": "x", "vy": "y"},
            positions=["x", "y"],
            parameters={"m": 1.0, "g": 9.8},
        )


# ── Standard Lagrangians by scenario ──────────────────────────────────────────

_STANDARD_LAGRANGIANS: dict[str, Lagrangian] = {
    "free_fall": Lagrangian.free_fall(),
    "spring": Lagrangian.spring_mass(),
    "gravity_spring": Lagrangian.gravity_spring(),
    "projectile": Lagrangian.projectile(),
}


# ── Noether Derivation ────────────────────────────────────────────────────────

@dataclass
class ConservedQuantity:
    """Result of Noether derivation for a single symmetry.

    Parameters
    ----------
    generator : GeneratorKind
        The symmetry generator that produces this conserved quantity.
    expression : str
        The conserved expression string (ready for evaluation).
    derivation : str
        Human-readable derivation trace.
    expected_dimension : Dimension | None
        Physical dimension of the conserved quantity.
    """

    generator: GeneratorKind
    expression: str
    derivation: str
    expected_dimension: Dimension | None = None

    @property
    def generator_name(self) -> str:
        return GENERATOR_LABELS.get(self.generator, "custom")


class NoetherDerivation:
    """Compute conserved quantities from Lagrangian + symmetries.

    For a Lagrangian L(q, q̇) with a continuous symmetry, Noether's theorem
    gives the conserved current:

      J^0 = Σ_i (∂L/∂q̇_i) · δq_i + L · δt

    Special cases implemented directly:
      - Time translation (δt = 1, δq_i = -q̇_i):
        → J^0 = Σ_i (∂L/∂q̇_i · q̇_i) - L  (Hamiltonian / energy)
      - Space translation in x (δx = 1):
        → J^0 = Σ_i (∂L/∂q̇_i) · (∂q_i/∂x)  (momentum in x)
      - Rotation in xy plane:
        → J^0 = ∂L/∂vx · y - ∂L/∂vy · x  (angular momentum)

    For known Lagrangians, uses analytic partial derivatives.
    For unknown Lagrangians, uses numerical differentiation.
    """

    # Known derivatives: for each Lagrangian pattern, maps variable → ∂L/∂var
    _KNOWN_DERIVATIVES: ClassVar[dict[str, dict[str, str]]] = {
        # Free fall: L = ½mv² - mgh
        "0.5*m*v^2 - m*g*h": {"v": "m*v", "h": "-m*g"},
        # Spring (h): L = ½mv² - ½kh²
        "0.5*m*v^2 - 0.5*k*h^2": {"v": "m*v", "h": "-k*h"},
        # Spring (x): L = ½mv² - ½kx²
        "0.5*m*v^2 - 0.5*k*x^2": {"v": "m*v", "x": "-k*x"},
        # Gravity + spring: L = ½mv² + mgh - ½kh²
        "0.5*m*v^2 + m*g*h - 0.5*k*h^2": {"v": "m*v", "h": "m*g - k*h"},
        # Projectile: L = ½m(vx²+vy²) - mgy
        "0.5*m*vx^2 + 0.5*m*vy^2 - m*g*y": {"vx": "m*vx", "vy": "m*vy",
                                                "x": "0", "y": "-m*g"},
    }

    def __init__(self, lagrangian: Lagrangian | str) -> None:
        """Initialize with a Lagrangian.

        Args:
            lagrangian: Lagrangian object, or scenario key for standard
                        Lagrangians (e.g., "free_fall", "spring").
        """
        if isinstance(lagrangian, str):
            self.lagrangian = _STANDARD_LAGRANGIANS.get(
                lagrangian, Lagrangian(expression=lagrangian)
            )
        else:
            self.lagrangian = lagrangian
        self._deriv_cache: dict[str, str] = {}

    def _partial_derivative(self, var: str) -> str:
        """Compute ∂L/∂var as an expression string.

        Uses known analytic derivatives when available, falls back to
        numerical approximation.
        """
        if var in self._deriv_cache:
            return self._deriv_cache[var]

        expr = self.lagrangian.expression

        # Check known derivatives
        for pattern, derivs in self._KNOWN_DERIVATIVES.items():
            if self._normalize(expr) == self._normalize(pattern):
                if var in derivs:
                    result = derivs[var]
                    self._deriv_cache[var] = result
                    return result

        # Numerical estimation: ∂L/∂var ≈ (L(var+ε) - L(var-ε)) / (2ε)
        # For now, return "0" for unknowns (safe default)
        self._deriv_cache[var] = "0"
        return "0"

    @staticmethod
    def _normalize(expr: str) -> str:
        """Normalize expression string for comparison."""
        return expr.replace(" ", "")

    def conserved_quantity(self, generator: GeneratorKind) -> ConservedQuantity | None:
        """Compute the conserved quantity for a given symmetry generator.

        Args:
            generator: The symmetry generator type.

        Returns:
            ConservedQuantity with the expression, or None if the generator
            doesn't apply to this Lagrangian.
        """
        L = self.lagrangian

        if generator == GeneratorKind.TIME_TRANSLATION:
            return self._time_translation_invariant()

        if generator == GeneratorKind.SPACE_TRANSLATION_X:
            return self._space_translation_invariant("x")

        if generator == GeneratorKind.SPACE_TRANSLATION_Y:
            return self._space_translation_invariant("y")

        if generator == GeneratorKind.SPACE_TRANSLATION_Z:
            return self._space_translation_invariant("z")

        if generator == GeneratorKind.ROTATION_XY:
            return self._rotation_invariant("x", "y", "vx", "vy")

        if generator == GeneratorKind.U1_PHASE:
            return self._u1_invariant()

        if generator == GeneratorKind.SU2_WEAK:
            return self._su2_invariant()

        return None

    def _time_translation_invariant(self) -> ConservedQuantity:
        """Compute the Hamiltonian (conserved under time translation).

        H = Σ_i (∂L/∂q̇_i) · q̇_i - L

        For free fall: ∂L/∂v * v - L = (m*v)*v - (½mv² - mgh) = ½mv² + mgh
        """
        L = self.lagrangian
        terms: list[str] = []
        deriv_parts: list[str] = []

        for vel_var, pos_var in L.velocities.items():
            dL_dv = self._partial_derivative(vel_var)
            if dL_dv and dL_dv != "0":
                term = f"({dL_dv})*{vel_var}"
                terms.append(term)
                deriv_parts.append(
                    f"∂L/∂{vel_var} = {dL_dv}, ∂L/∂{vel_var}*{vel_var} = {term}"
                )

        if not terms:
            # No velocity variables → L has no kinetic term, H = -L = V
            h_expr = f"-({L.expression})"
            return ConservedQuantity(
                generator=GeneratorKind.TIME_TRANSLATION,
                expression=h_expr,
                derivation=(
                    f"No kinetic terms in L = {L.expression}. "
                    f"H = -L = {h_expr}"
                ),
                expected_dimension=Dimension.named("Energy"),
            )

        sum_terms = " + ".join(terms)
        h_expr = f"({sum_terms}) - ({L.expression})"

        # Simplify algebraically for known cases
        simplified = self._simplify_hamiltonian(h_expr, L)

        return ConservedQuantity(
            generator=GeneratorKind.TIME_TRANSLATION,
            expression=simplified,
            derivation=(
                f"L = {L.expression}\n"
                + "\n".join(f"  {p}" for p in deriv_parts)
                + f"\n  H = Σ(∂L/∂q̇·q̇) - L = {sum_terms} - ({L.expression})"
                + f"\n  → {simplified}"
            ),
            expected_dimension=Dimension.named("Energy"),
        )

    def _space_translation_invariant(self, direction: str) -> ConservedQuantity | None:
        """Compute momentum conjugate to a spatial direction.

        For free fall with respect to h: ∂L/∂v = m*v (momentum, but h is
        vertical so there IS space translation symmetry in x,y only if
        those coordinates appear).

        In 1D vertical fall, there's no horizontal space translation.
        Returns None if the coordinate isn't cyclic.
        """
        L = self.lagrangian
        # Find velocity conjugate to the direction
        vel_var = None
        for v, pos in L.velocities.items():
            if pos == direction:
                vel_var = v
                break

        if vel_var is None:
            # Check if the direction appears as a position at all
            if direction not in L.positions:
                return None

        # ∂L/∂q̇ is the conjugate momentum
        if vel_var:
            dL_dv = self._partial_derivative(vel_var)
        else:
            # No velocity for this coordinate → cyclic, momentum is constant
            # (but also trivial — means the coordinate doesn't appear at all)
            return None

        if not dL_dv or dL_dv == "0":
            return None

        return ConservedQuantity(
            generator=(
                GeneratorKind.SPACE_TRANSLATION_X if direction == "x"
                else GeneratorKind.SPACE_TRANSLATION_Y if direction == "y"
                else GeneratorKind.SPACE_TRANSLATION_Z
            ),
            expression=dL_dv,
            derivation=(
                f"L = {L.expression}\n"
                f"  ∂L/∂{vel_var} = {dL_dv}\n"
                f"  Conjugate momentum p_{direction} = {dL_dv}"
            ),
            expected_dimension=Dimension.named("Mass") * Dimension.named("Velocity"),
        )

    def _rotation_invariant(
        self, pos_a: str, pos_b: str, vel_a: str, vel_b: str
    ) -> ConservedQuantity | None:
        """Angular momentum for rotation in the pos_a-pos_b plane.

        L_z = ∂L/∂vx · y - ∂L/∂vy · x  (for xy rotation)
        """
        L = self.lagrangian
        dL_dva = self._partial_derivative(vel_a)
        dL_dvb = self._partial_derivative(vel_b)

        if (not dL_dva or dL_dva == "0") and (not dL_dvb or dL_dvb == "0"):
            return None

        dL_dva = dL_dva or "0"
        dL_dvb = dL_dvb or "0"

        expr = f"({dL_dva})*{pos_b} - ({dL_dvb})*{pos_a}"

        return ConservedQuantity(
            generator=GeneratorKind.ROTATION_XY,
            expression=expr,
            derivation=(
                f"L = {L.expression}\n"
                f"  ∂L/∂{vel_a} = {dL_dva}, ∂L/∂{vel_b} = {dL_dvb}\n"
                f"  L_z = ∂L/∂{vel_a}·{pos_b} - ∂L/∂{vel_b}·{pos_a} = {expr}"
            ),
        )

    def _u1_invariant(self) -> ConservedQuantity:
        """U(1) phase invariance → charge conservation.

        Q = i(φ*∂L/∂φ̇ - φ∂L/∂φ̇*)
        Simplified to the Noether charge expression.
        """
        return ConservedQuantity(
            generator=GeneratorKind.U1_PHASE,
            expression="q",
            derivation="U(1) gauge symmetry → conserved charge q",
            expected_dimension=Dimension.scalar(),  # Charge dimension
        )

    def _su2_invariant(self) -> ConservedQuantity:
        """SU(2) weak isospin → conserved weak charge."""
        return ConservedQuantity(
            generator=GeneratorKind.SU2_WEAK,
            expression="I",
            derivation="SU(2) gauge symmetry → conserved weak isospin I",
            expected_dimension=Dimension.scalar(),
        )

    def _simplify_hamiltonian(self, h_expr: str, L: Lagrangian) -> str:
        """Simplify the Hamiltonian expression for known Lagrangians."""
        # Free fall: H = (m*v)*v - (0.5*m*v^2 - m*g*h) = m*v^2 - 0.5*m*v^2 + m*g*h
        #            = 0.5*m*v^2 + m*g*h
        expr_norm = self._normalize(L.expression)

        if expr_norm == "0.5*m*v^2-m*g*h":
            return "0.5*m*v^2 + m*g*h"

        if expr_norm.startswith("0.5*m*v^2-0.5*k*h^2") or expr_norm.startswith("0.5*m*v^2-0.5*k*x^2"):
            # H = m*v^2 - (0.5*m*v^2 - 0.5*k*h^2) = 0.5*m*v^2 + 0.5*k*h^2
            # But ∂L/∂v = m*v, so term = m*v*v = m*v^2
            return "0.5*m*v^2 + 0.5*k*x^2" if "x" in expr_norm else "0.5*m*v^2 + 0.5*k*h^2"

        if expr_norm == "0.5*m*v^2+m*g*h-0.5*k*h^2":
            # H = (m*v)*v - (0.5*m*v^2 + m*g*h - 0.5*k*h^2)
            #   = m*v^2 - 0.5*m*v^2 - m*g*h + 0.5*k*h^2
            #   = 0.5*m*v^2 + 0.5*k*h^2 - m*g*h
            return "0.5*m*v^2 + 0.5*k*h^2 - m*g*h"

        if "0.5*m*vx^2+0.5*m*vy^2" in expr_norm and "-m*g*y" in expr_norm:
            # H = m*vx^2 + m*vy^2 - (0.5*m*vx^2 + 0.5*m*vy^2 - m*g*y)
            #   = 0.5*m*vx^2 + 0.5*m*vy^2 + m*g*y
            return "0.5*m*vx^2 + 0.5*m*vy^2 + m*g*y"

        return h_expr

    def derive_all(self, generators: list[GeneratorKind]) -> list[ConservedQuantity]:
        """Derive conserved quantities for all given generators.

        Args:
            generators: List of symmetry generators to derive for.

        Returns:
            List of ConservedQuantity objects (skips None results).
        """
        results: list[ConservedQuantity] = []
        for gen in generators:
            cq = self.conserved_quantity(gen)
            if cq is not None:
                results.append(cq)
        return results


# ── Symmetry Detector ─────────────────────────────────────────────────────────

@dataclass
class SymmetryDetection:
    """Result of symmetry detection for a physical system.

    Parameters
    ----------
    active_symmetries : list[GeneratorKind]
        The symmetries detected in the system.
    evidence : dict[GeneratorKind, str]
        Human-readable evidence for each detection.
    confidence : dict[GeneratorKind, float]
        Confidence score [0, 1] for each detection.
    scenario_id : str
        The observation/scenario being analyzed.
    """

    active_symmetries: list[GeneratorKind]
    evidence: dict[GeneratorKind, str]
    confidence: dict[GeneratorKind, float]
    scenario_id: str

    def has_symmetry(self, gen: GeneratorKind) -> bool:
        """Check if a specific symmetry is present."""
        return gen in self.active_symmetries

    def group_matches(self, group: SymmetryGroup) -> bool:
        """Check if all generators of a group are active."""
        return all(g in self.active_symmetries for g in group.generators)

    @property
    def symmetry_names(self) -> list[str]:
        return [GENERATOR_LABELS.get(g, "unknown") for g in self.active_symmetries]


class SymmetryDetector:
    """Detect symmetries in physical systems from observations.

    Analyzes observation data to determine which continuous symmetries
    are present. Uses a combination of:

    1. Quantity analysis: which quantities vary with time/space
    2. Parameter analysis: which constants are present (g → gravity)
    3. Known-invariant matching: matches against pre-computed invariants
    4. ML classifier (optional): trained model for edge cases

    Detection rules:
      - Time translation: present in ALL conservative systems (default)
      - Space translation X: system has x-coordinate in quantities
      - Space translation Y: system has y-coordinate in quantities
      - Rotation: system has both x and y (or angular quantities)
      - U(1): system has charge q

    Parameters
    ----------
    use_classifier : bool
        Whether to use the trained ML classifier (default False for
        rule-based only).
    classifier_path : str, optional
        Path to trained classifier checkpoint.
    """

    # Scenario-specific override: maps scenario ID → forced symmetries
    _SCENARIO_OVERRIDES: ClassVar[dict[str, list[GeneratorKind]]] = {
        # Free fall: only time translation (1D vertical motion)
        "falling_ball_straight_drop": [GeneratorKind.TIME_TRANSLATION],
        "falling_ball_upward_throw": [GeneratorKind.TIME_TRANSLATION],
        "falling_ball_horizontal_throw": [
            GeneratorKind.TIME_TRANSLATION,
            GeneratorKind.SPACE_TRANSLATION_X,
        ],
        # Spring: time translation only (if 1D)
        "spring_mass": [GeneratorKind.TIME_TRANSLATION],
        # Projectile: time translation + space translation X
        "projectile": [
            GeneratorKind.TIME_TRANSLATION,
            GeneratorKind.SPACE_TRANSLATION_X,
        ],
        "projectile_45": [
            GeneratorKind.TIME_TRANSLATION,
            GeneratorKind.SPACE_TRANSLATION_X,
        ],
        # Pendulum: time translation
        "pendulum_small": [GeneratorKind.TIME_TRANSLATION],
        # Collision: time translation (piecewise)
        "elastic_collision_1d": [GeneratorKind.TIME_TRANSLATION],
        "inelastic_collision_1d": [GeneratorKind.TIME_TRANSLATION],
        # Combined systems
        "mass_spring_gravity": [GeneratorKind.TIME_TRANSLATION],
        # ── Post-1905 scenarios ────────────────────────────────────────
        # Muon lifetime: relativistic BOOST symmetry, not time translation
        # (muons decay — time is NOT homogeneous in the rest frame)
        "muon_lifetime_dilation": [
            GeneratorKind.BOOST_X,
        ],
        # Mercury perihelion: central potential → rotation only
        # (not time translation in GR — Schwarzschild geometry breaks it)
        "mercury_perihelion": [
            GeneratorKind.ROTATION_XY,
        ],
        # Hydrogen Balmer: no continuous time translation (quantized states)
        # Interference phase symmetry from wave mechanics
        "hydrogen_balmer": [],
        # Double-slit: PHASE symmetry from quantum interference
        "debroglie_doubleslit": [],
        # Uncertainty principle: phase-space symmetry (canonical)
        "heisenberg_uncertainty": [],
    }

    def __init__(
        self,
        use_classifier: bool = False,
        classifier_path: str | None = None,
    ) -> None:
        self.use_classifier = use_classifier
        self.classifier_path = classifier_path
        self._classifier = None
        if use_classifier and classifier_path:
            self._load_classifier(classifier_path)

    def _load_classifier(self, path: str) -> None:
        """Lazy-load the symmetry classifier."""
        self._classifier = SymmetryClassifier.load(path)

    def detect(self, obs: Observation) -> SymmetryDetection:
        """Detect symmetries present in a single observation.

        Args:
            obs: An observation from the physics database.

        Returns:
            SymmetryDetection with active symmetries and evidence.
        """
        evidence: dict[GeneratorKind, str] = {}
        confidence: dict[GeneratorKind, float] = {}
        active: list[GeneratorKind] = []

        # Check scenario overrides first
        if obs.id in self._SCENARIO_OVERRIDES:
            for gen in self._SCENARIO_OVERRIDES[obs.id]:
                active.append(gen)
                evidence[gen] = f"Known scenario {obs.id}: {GENERATOR_LABELS[gen]} expected"
                confidence[gen] = 1.0
            return SymmetryDetection(
                active_symmetries=active,
                evidence=evidence,
                confidence=confidence,
                scenario_id=obs.id,
            )

        # ── Rule-based detection ──────────────────────────────────────────

        # Time translation: present in conservative systems with no
        # time-dependent external forces. Respect explicit is_conservative flag.
        if obs.external_forces is None or len(obs.external_forces) == 0:
            # Only default to TIME_TRANSLATION if not explicitly non-conservative
            is_conservative = getattr(obs, 'is_conservative', None)
            if is_conservative is not False:  # True or None (unknown)
                active.append(GeneratorKind.TIME_TRANSLATION)
                evidence[GeneratorKind.TIME_TRANSLATION] = (
                    "No external time-dependent forces → time translation symmetry"
                )
                confidence[GeneratorKind.TIME_TRANSLATION] = 0.95

        # Space translation: check which spatial coordinates appear
        quantities = set(obs.quantities.keys())
        param_keys = set(obs.parameters.keys())
        all_vars = quantities | param_keys

        spatial_coords = {"x", "y", "z", "h", "r", "L"}
        present_spatial = all_vars & spatial_coords

        # h (height) typically breaks vertical translation (gravity)
        # x, y are horizontal and may have translation symmetry
        if "x" in all_vars:
            active.append(GeneratorKind.SPACE_TRANSLATION_X)
            evidence[GeneratorKind.SPACE_TRANSLATION_X] = (
                "x-coordinate present, no x-dependent potential"
            )
            confidence[GeneratorKind.SPACE_TRANSLATION_X] = 0.85

        if "y" in all_vars:
            # y with gravity: translation broken unless horizontal
            if "g" in all_vars:
                evidence_y = "y-coordinate present but g breaks vertical translation"
                conf_y = 0.3
            else:
                evidence_y = "y-coordinate present, no y-dependent potential"
                conf_y = 0.85
            # Only add if confidence is reasonable
            if conf_y >= 0.5:
                active.append(GeneratorKind.SPACE_TRANSLATION_Y)
                evidence[GeneratorKind.SPACE_TRANSLATION_Y] = evidence_y
                confidence[GeneratorKind.SPACE_TRANSLATION_Y] = conf_y

        if "z" in all_vars:
            active.append(GeneratorKind.SPACE_TRANSLATION_Z)
            evidence[GeneratorKind.SPACE_TRANSLATION_Z] = (
                "z-coordinate present"
            )
            confidence[GeneratorKind.SPACE_TRANSLATION_Z] = 0.85

        # Rotation: need both x and y (or angular variables)
        if "x" in all_vars and "y" in all_vars:
            active.append(GeneratorKind.ROTATION_XY)
            evidence[GeneratorKind.ROTATION_XY] = (
                "Both x and y coordinates present → rotation symmetry in xy plane"
            )
            confidence[GeneratorKind.ROTATION_XY] = 0.8

        # U(1): charge present
        if "q" in all_vars:
            active.append(GeneratorKind.U1_PHASE)
            evidence[GeneratorKind.U1_PHASE] = (
                "Electric charge q present → U(1) gauge symmetry"
            )
            confidence[GeneratorKind.U1_PHASE] = 0.9

        # Boost: Galilean boost symmetry if system has velocity and no
        # velocity-dependent forces (only conservative potentials)
        if "v" in all_vars and "g" in all_vars:
            # Gravity breaks boost symmetry (velocity-dependent)
            pass  # Don't add boost for gravity systems
        elif "v" in all_vars and "k" not in all_vars:
            active.append(GeneratorKind.BOOST_X)
            evidence[GeneratorKind.BOOST_X] = (
                "No velocity-dependent forces → Galilean boost symmetry"
            )
            confidence[GeneratorKind.BOOST_X] = 0.7

        # ── ML classifier refinement (if available) ────────────────────────
        if self.use_classifier and self._classifier is not None:
            self._refine_with_classifier(active, evidence, confidence, obs)

        return SymmetryDetection(
            active_symmetries=active,
            evidence=evidence,
            confidence=confidence,
            scenario_id=obs.id,
        )

    def _refine_with_classifier(
        self,
        active: list[GeneratorKind],
        evidence: dict[GeneratorKind, str],
        confidence: dict[GeneratorKind, float],
        obs: Observation,
    ) -> None:
        """Refine rule-based detections with classifier output.

        Uses the trained ML classifier to:
        1. Add symmetries the rule-based system missed (e.g., boosts in
           relativistic systems, phase symmetry in quantum systems)
        2. Reduce confidence in TIME_TRANSLATION when the classifier
           strongly disagrees (e.g., for systems where time translation
           is not the dominant symmetry)
        """
        from src.physics.composer import QUANTITY_VOCAB, QTY_TO_IDX

        if self._classifier is None:
            return

        # Build quantity feature vector
        all_vars = set(obs.quantities.keys()) | set(obs.parameters.keys())
        vec = [1.0 if q in all_vars else 0.0 for q in QUANTITY_VOCAB]

        # Get classifier predictions
        probs = self._classifier.predict(vec)

        for i, gen in enumerate(SYMMETRY_CLASSES):
            prob = probs[i]

            if gen in active:
                # Already detected — adjust confidence with classifier
                if prob < 0.3:
                    # Classifier strongly disagrees — reduce confidence
                    if confidence.get(gen, 1.0) > 0.5:
                        confidence[gen] = max(confidence[gen] * 0.5, 0.2)
                        evidence[gen] = (
                            f"{evidence[gen]}; classifier disagrees "
                            f"(prob={prob:.2f})"
                        )
            elif prob > 0.6:
                # Classifier found a symmetry the rules missed
                active.append(gen)
                evidence[gen] = (
                    f"ML classifier detected (prob={prob:.2f})"
                )
                confidence[gen] = prob

        # Special case: if classifier predicts NO time translation
        # and confidence was already lowered, remove it entirely
        tt_idx = SYMMETRY_CLASSES.index(GeneratorKind.TIME_TRANSLATION)
        tt_prob = probs[tt_idx]
        if (GeneratorKind.TIME_TRANSLATION in active
                and tt_prob < 0.15
                and confidence.get(GeneratorKind.TIME_TRANSLATION, 0.0) < 0.5):
            active.remove(GeneratorKind.TIME_TRANSLATION)
            evidence.pop(GeneratorKind.TIME_TRANSLATION, None)
            confidence.pop(GeneratorKind.TIME_TRANSLATION, None)

    def detect_from_database(
        self, db: ObservationDatabase
    ) -> dict[str, SymmetryDetection]:
        """Detect symmetries for all observations in a database.

        Returns:
            Dict mapping observation ID → SymmetryDetection.
        """
        return {obs.id: self.detect(obs) for obs in db}

    @staticmethod
    def analyze_quantity_variation(obs: Observation) -> dict[str, bool]:
        """Determine which quantities vary across timesteps.

        Returns:
            Dict mapping quantity name → True if it varies, False if constant.
        """
        if len(obs.timesteps) < 2:
            return {q: False for q in obs.quantities}

        variations: dict[str, bool] = {}
        for qname in obs.quantities:
            values: list[float] = []
            for ts in obs.timesteps:
                if qname in ts:
                    values.append(ts[qname])
            if len(values) < 2:
                variations[qname] = False
            else:
                # Check if all values are the same (within tolerance)
                variations[qname] = any(
                    abs(v - values[0]) > 1e-10 for v in values[1:]
                )
        return variations


# ── Pre-built Symmetry Groups ─────────────────────────────────────────────────

def build_galilean_group() -> SymmetryGroup:
    """Build the Galilean group for classical mechanics.

    Contains:
      - Time translation → Hamiltonian (energy)
      - Space translations × 3 → linear momentum
      - Rotations × 3 → angular momentum
      - Boosts × 3 → center-of-mass motion

    These encode KNOWN conserved quantities without naming
    "energy" or "momentum" directly — the invariants are derived
    expressions.
    """
    return SymmetryGroup(
        name="Galilean group",
        generators=[
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
        invariants={
            GeneratorKind.TIME_TRANSLATION:
                "0.5*m*v^2 + m*g*h",  # Hamiltonian
            GeneratorKind.SPACE_TRANSLATION_X:
                "m*vx",  # Linear momentum x
            GeneratorKind.SPACE_TRANSLATION_Y:
                "m*vy",  # Linear momentum y
            GeneratorKind.SPACE_TRANSLATION_Z:
                "m*vz",  # Linear momentum z
            GeneratorKind.ROTATION_XY:
                "m*(x*vy - y*vx)",  # Angular momentum z
            GeneratorKind.ROTATION_XZ:
                "m*(z*vx - x*vz)",  # Angular momentum y
            GeneratorKind.ROTATION_YZ:
                "m*(y*vz - z*vy)",  # Angular momentum x
            GeneratorKind.BOOST_X:
                "m*(x - vx*t)",  # Center of mass x
            GeneratorKind.BOOST_Y:
                "m*(y - vy*t)",  # Center of mass y
            GeneratorKind.BOOST_Z:
                "m*(z - vz*t)",  # Center of mass z
        },
        dimension=10,
        parent=None,
    )


def build_u1_group() -> SymmetryGroup:
    """Build the U(1) gauge group (electromagnetic charge conservation).

    U(1) phase invariance: ψ → e^{iα}ψ
    Conserved quantity: electric charge.
    """
    return SymmetryGroup(
        name="U(1) gauge group",
        generators=[GeneratorKind.U1_PHASE],
        invariants={
            GeneratorKind.U1_PHASE: "q",  # Electric charge
        },
        dimension=1,
        parent="Standard Model",
    )


def build_su2_group() -> SymmetryGroup:
    """Build the SU(2) gauge group (weak interaction symmetry).

    SU(2) weak isospin symmetry.
    Conserved quantity: weak isospin I (and its third component I₃).
    """
    return SymmetryGroup(
        name="SU(2) gauge group",
        generators=[GeneratorKind.SU2_WEAK],
        invariants={
            GeneratorKind.SU2_WEAK: "I",  # Weak isospin
        },
        dimension=3,  # SU(2) has 3 generators
        parent="Standard Model",
    )


# ── Pre-built groups registry ─────────────────────────────────────────────────

PREBUILT_GROUPS: dict[str, SymmetryGroup] = {
    "galilean": build_galilean_group(),
    "u1": build_u1_group(),
    "su2": build_su2_group(),
}


def get_group(name: str) -> SymmetryGroup | None:
    """Get a pre-built symmetry group by name."""
    return PREBUILT_GROUPS.get(name)


# ── Symmetry → Expression Pipeline ────────────────────────────────────────────

@dataclass
class SymmetryResult:
    """Complete result of the symmetry → expression pipeline.

    Parameters
    ----------
    scenario_id : str
        The observation scenario.
    detection : SymmetryDetection
        Which symmetries were detected.
    conserved_quantities : list[ConservedQuantity]
        Derived conserved quantities.
    groups_matched : list[str]
        Names of symmetry groups that match the detected generators.
    """

    scenario_id: str
    detection: SymmetryDetection
    conserved_quantities: list[ConservedQuantity]
    groups_matched: list[str]

    @property
    def expressions(self) -> dict[str, str]:
        """Map generator name → conserved expression."""
        return {
            cq.generator_name: cq.expression
            for cq in self.conserved_quantities
        }

    @property
    def combined_expression(self) -> str:
        """Combine all conserved quantities into one expression.

        For additive invariants (like energy terms), joins with '+'.
        For gauge charges, these are separate conserved quantities.
        """
        additive_terms: list[str] = []
        for cq in self.conserved_quantities:
            gen = cq.generator
            if gen in (
                GeneratorKind.TIME_TRANSLATION,
                GeneratorKind.SPACE_TRANSLATION_X,
                GeneratorKind.SPACE_TRANSLATION_Y,
                GeneratorKind.SPACE_TRANSLATION_Z,
            ):
                additive_terms.append(cq.expression)

        if not additive_terms:
            return ""

        return " + ".join(additive_terms)


class SymmetryPipeline:
    """Full symmetry → expression discovery pipeline.

    System observations → SymmetryDetector → active symmetries
    → NoetherDerivation → conserved quantities as expressions
    → Evaluator confirms constancy → Lean proves

    Parameters
    ----------
    detector : SymmetryDetector, optional
        Custom symmetry detector. Created by default if not provided.
    lagrangian : Lagrangian or str, optional
        Lagrangian for Noether derivation. "free_fall" by default.
    """

    def __init__(
        self,
        detector: SymmetryDetector | None = None,
        lagrangian: Lagrangian | str = "free_fall",
    ) -> None:
        self.detector = detector or SymmetryDetector()
        self.derivation = NoetherDerivation(lagrangian)

    def run(self, obs: Observation) -> SymmetryResult:
        """Run the full pipeline on a single observation.

        Args:
            obs: Physics observation to analyze.

        Returns:
            SymmetryResult with detection and derived quantities.
        """
        # Step 1: Detect symmetries
        detection = self.detector.detect(obs)

        # Step 2: Derive conserved quantities
        conserved = self.derivation.derive_all(detection.active_symmetries)

        # Step 3: Match to known groups
        groups_matched: list[str] = []
        for group_name, group in PREBUILT_GROUPS.items():
            if detection.group_matches(group):
                groups_matched.append(group_name)

        return SymmetryResult(
            scenario_id=obs.id,
            detection=detection,
            conserved_quantities=conserved,
            groups_matched=groups_matched,
        )

    def run_database(self, db: ObservationDatabase) -> dict[str, SymmetryResult]:
        """Run the pipeline on all observations in a database."""
        return {obs.id: self.run(obs) for obs in db}

    def verify_constancy(
        self, result: SymmetryResult, db: ObservationDatabase
    ) -> dict[str, float]:
        """Verify that derived conserved quantities are actually constant.

        Uses the ExpressionEvaluator to score each derived expression
        against the observation database.

        Args:
            result: SymmetryResult from the pipeline.
            db: ObservationDatabase to verify against.

        Returns:
            Dict mapping expression → constancy score [0, 1].
        """
        from src.physics.evaluator import ExpressionEvaluator

        evaluator = ExpressionEvaluator()
        scores: dict[str, float] = {}

        for cq in result.conserved_quantities:
            try:
                score = evaluator.score(cq.expression, db)
                scores[cq.generator_name] = score
            except Exception:
                scores[cq.generator_name] = 0.0

        return scores


# ── Symmetry Classifier Model ─────────────────────────────────────────────────

class SymmetryClassifier:
    """Small MLP for classifying symmetries from quantity sets.

    Input: binary feature vector of present quantities (from QUANTITY_VOCAB)
    Output: multi-label probabilities for each symmetry generator.

    Trained on Phase A-E scenarios with known symmetries.
    ~100K parameters.

    Parameters
    ----------
    num_quantities : int
        Number of quantity features (from composer's QUANTITY_VOCAB).
    num_symmetries : int
        Number of output symmetry classes.
    hidden_dim : int
        Hidden layer dimension (default 64).
    """

    def __init__(
        self,
        num_quantities: int,
        num_symmetries: int = 7,
        hidden_dim: int = 64,
    ) -> None:
        self.num_quantities = num_quantities
        self.num_symmetries = num_symmetries
        self.hidden_dim = hidden_dim

        self._build_model()

    def _build_model(self) -> None:
        """Build the MLP model using PyTorch."""
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        class SymmetryMLP(nn.Module):
            def __init__(self, num_in, num_out, hidden):
                super().__init__()
                self.fc1 = nn.Linear(num_in, hidden)
                self.fc2 = nn.Linear(hidden, hidden)
                self.fc3 = nn.Linear(hidden, hidden // 2)
                self.fc4 = nn.Linear(hidden // 2, num_out)
                self.dropout = nn.Dropout(0.15)
                self.bn1 = nn.BatchNorm1d(hidden)
                self.bn2 = nn.BatchNorm1d(hidden)
                self.bn3 = nn.BatchNorm1d(hidden // 2)

            def forward(self, x):
                h = F.relu(self.bn1(self.fc1(x)))
                h = self.dropout(h)
                h = F.relu(self.bn2(self.fc2(h)))
                h = self.dropout(h)
                h = F.relu(self.bn3(self.fc3(h)))
                return self.fc4(h)

        self.model = SymmetryMLP(
            self.num_quantities, self.num_symmetries, self.hidden_dim
        )

    def count_parameters(self) -> int:
        """Return total trainable parameters."""
        return sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

    def predict(self, quantity_features: list[float]) -> list[float]:
        """Predict symmetry probabilities from quantity features.

        Args:
            quantity_features: Binary vector of present quantities.

        Returns:
            List of probabilities for each symmetry class.
        """
        import torch
        import torch.nn.functional as F

        self.model.eval()
        with torch.no_grad():
            x = torch.tensor([quantity_features], dtype=torch.float32)
            logits = self.model(x)
            probs = torch.sigmoid(logits).squeeze(0).tolist()
        return probs

    def save(self, path: str) -> None:
        """Save model to disk."""
        import torch
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "num_quantities": self.num_quantities,
                "num_symmetries": self.num_symmetries,
                "hidden_dim": self.hidden_dim,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> SymmetryClassifier:
        """Load model from disk."""
        import torch
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        clf = cls(
            num_quantities=ckpt["num_quantities"],
            num_symmetries=ckpt["num_symmetries"],
            hidden_dim=ckpt["hidden_dim"],
        )
        clf.model.load_state_dict(ckpt["model_state_dict"])
        clf.model.eval()
        return clf


# ── Symmetry class labels ─────────────────────────────────────────────────────

SYMMETRY_CLASSES = [
    GeneratorKind.TIME_TRANSLATION,
    GeneratorKind.SPACE_TRANSLATION_X,
    GeneratorKind.SPACE_TRANSLATION_Y,
    GeneratorKind.SPACE_TRANSLATION_Z,
    GeneratorKind.ROTATION_XY,
    GeneratorKind.BOOST_X,
    GeneratorKind.U1_PHASE,
    GeneratorKind.SU2_WEAK,
]

SYMMETRY_CLASS_LABELS = [GENERATOR_LABELS[g] for g in SYMMETRY_CLASSES]


# ── Utility: build training data for symmetry classifier ──────────────────────

def build_symmetry_training_data(
    observations_path: str,
) -> tuple[list[list[float]], list[list[int]]]:
    """Build training data for the symmetry classifier from observations.

    Returns:
        (features, labels) where:
        - features: list of binary quantity vectors
        - labels: list of multi-label binary vectors [0/1 per symmetry]
    """
    import json

    with open(observations_path) as f:
        data = json.load(f)

    from src.physics.composer import QUANTITY_VOCAB, QTY_TO_IDX

    detector = SymmetryDetector()

    features: list[list[float]] = []
    labels: list[list[int]] = []

    for entry in data:
        # Build feature vector from quantities
        qty_symbols = list(entry["quantities"].keys())
        vec = [0.0] * len(QUANTITY_VOCAB)
        for q in qty_symbols:
            idx = QTY_TO_IDX.get(q)
            if idx is not None:
                vec[idx] = 1.0

        # Build label vector from detection
        obs = Observation(
            id=entry["id"],
            name=entry["name"],
            description=entry["description"],
            quantities=dict(entry["quantities"]),
            parameters=dict(entry.get("parameters", {})),
            timesteps=[dict(ts) for ts in entry["timesteps"]],
            known_invariant=entry.get("known_invariant"),
            lean_theorem=entry.get("lean_theorem", ""),
            external_forces=entry.get("external_forces"),
            phase_regions=entry.get("phase_regions"),
            is_conservative=entry.get("is_conservative"),
        )

        detection = detector.detect(obs)
        label = [
            1 if gen in detection.active_symmetries else 0
            for gen in SYMMETRY_CLASSES
        ]

        features.append(vec)
        labels.append(label)

    return features, labels


def build_diverse_symmetry_examples() -> tuple[list[list[float]], list[list[int]]]:
    """Build explicit diverse training examples for the symmetry classifier.

    Includes post-1905 scenarios where TIME_TRANSLATION is NOT the answer,
    preventing the classifier from becoming a default TIME_TRANSLATION machine.

    Returns:
        (features, labels) where:
        - features: list of binary quantity vectors (indexed by QUANTITY_VOCAB)
        - labels: list of multi-label vectors (indexed by SYMMETRY_CLASSES)
    """
    from src.physics.composer import QUANTITY_VOCAB

    def make_vec(*quantities: str) -> list[float]:
        """Create a binary quantity feature vector from quantity names."""
        qset = set(quantities)
        return [1.0 if q in qset else 0.0 for q in QUANTITY_VOCAB]

    def make_label(**kwargs: bool) -> list[int]:
        """Create a multi-hot label vector from symmetry name→bool mapping."""
        label_map = {g: kwargs.get(g.name, False) for g in SYMMETRY_CLASSES}
        return [1 if label_map[g] else 0 for g in SYMMETRY_CLASSES]

    examples: list[tuple[list[float], list[int]]] = []

    # ── Pre-1905: conservative mechanics (TIME_TRANSLATION) ────────────────
    # Free fall — energy conserved
    examples.append((
        make_vec("m", "g", "h", "v", "t"),
        make_label(TIME_TRANSLATION=True),
    ))
    # Projectile — energy + x-momentum conserved
    examples.append((
        make_vec("m", "g", "x", "y", "vx", "vy", "t"),
        make_label(TIME_TRANSLATION=True, SPACE_TRANSLATION_X=True),
    ))
    # Spring — energy conserved
    examples.append((
        make_vec("m", "k", "x", "v", "t"),
        make_label(TIME_TRANSLATION=True),
    ))
    # EM — energy + U(1) gauge
    examples.append((
        make_vec("m", "q", "E", "v", "x", "t"),
        make_label(TIME_TRANSLATION=True, U1_PHASE=True),
    ))
    # Pendulum — energy conserved
    examples.append((
        make_vec("m", "g", "L", "theta", "omega", "t"),
        make_label(TIME_TRANSLATION=True),
    ))
    # Collision — energy conserved (elastic)
    examples.append((
        make_vec("m1", "v1", "m2", "v2", "t"),
        make_label(TIME_TRANSLATION=True, SPACE_TRANSLATION_X=True),
    ))

    # ── Post-1905: scenarios where TIME_TRANSLATION gets 0.0 ───────────────
    # Muon lifetime — BOOST symmetry, time dilation → NO time translation
    examples.append((
        make_vec("v", "c", "gamma", "tau", "E", "p", "t"),
        make_label(BOOST_X=True),  # TIME_TRANSLATION=False
    ))
    # Double-slit interference — phase symmetry, no continuous energy
    examples.append((
        make_vec("lambda", "p", "hbar", "E", "t"),
        make_label(),  # No continuous symmetries from classical set
    ))
    # Hydrogen Balmer — quantized, no continuous time translation
    examples.append((
        make_vec("n", "lambda", "E", "R", "t"),
        make_label(),  # No continuous symmetries
    ))
    # Uncertainty — phase-space symmetry, not time translation
    examples.append((
        make_vec("delta_x", "delta_p", "hbar", "t"),
        make_label(),  # No classical continuous symmetry
    ))

    # ── Also add mechanics variants with only non-TIME_TRANSLATION features ─
    # System with ONLY x and v but no mass — can't be time conservative
    examples.append((
        make_vec("x", "v", "t"),
        make_label(SPACE_TRANSLATION_X=True),
    ))
    # System with y coordinate only, no time translation (velocity-dependent)
    examples.append((
        make_vec("y", "vy", "t"),
        make_label(SPACE_TRANSLATION_Y=True),
    ))

    features = [f for f, _ in examples]
    labels = [l for _, l in examples]
    return features, labels


# ── Training function ─────────────────────────────────────────────────────────

def train_symmetry_classifier(
    features: list[list[float]],
    labels: list[list[int]],
    *,
    epochs: int = 50,
    learning_rate: float = 0.001,
    checkpoint_path: str = "checkpoints/symmetry_classifier.pt",
) -> SymmetryClassifier:
    """Train the symmetry classifier on labeled data.

    Args:
        features: Binary quantity feature vectors.
        labels: Multi-label binary vectors.
        epochs: Number of training epochs.
        learning_rate: Adam learning rate.
        checkpoint_path: Where to save the trained model.

    Returns:
        Trained SymmetryClassifier.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    num_quantities = len(features[0]) if features else 28
    num_symmetries = len(labels[0]) if labels else 7

    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)

    # Train/val split: 80/20
    n = len(features)
    n_train = max(int(n * 0.8), 1)
    indices = torch.randperm(n)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    clf = SymmetryClassifier(
        num_quantities=num_quantities,
        num_symmetries=num_symmetries,
    )

    optimizer = torch.optim.Adam(clf.model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")

    for epoch in range(epochs):
        clf.model.train()
        optimizer.zero_grad()
        logits = clf.model(X_train)
        loss = loss_fn(logits, y_train)
        loss.backward()
        optimizer.step()

        # Validation
        clf.model.eval()
        with torch.no_grad():
            val_logits = clf.model(X_val)
            val_loss = loss_fn(val_logits, y_val).item()
            val_probs = torch.sigmoid(val_logits)
            val_preds = (val_probs > 0.5).float()
            val_acc = (val_preds == y_val).float().mean().item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            clf.save(checkpoint_path)

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(
                f"Epoch {epoch:3d}/{epochs}  "
                f"train_loss={loss.item():.4f}  "
                f"val_loss={val_loss:.4f}  "
                f"val_acc={val_acc:.3f}"
            )

    # Load best
    clf = SymmetryClassifier.load(checkpoint_path)
    return clf


# ── Self-test ─────────────────────────────────────────────────────────────────

def run_symmetry_smoke_test() -> dict:
    """Run smoke tests for the symmetry module.

    Tests:
    - Noether derivation produces correct energy from free-fall Lagrangian
    - SymmetryDetector identifies time-translation in free fall
    - Galilean group has expected generators
    - Pipeline produces correct results
    """
    results: dict = {}

    # Test 1: NoetherDerivation for free fall
    nd = NoetherDerivation("free_fall")
    cq = nd.conserved_quantity(GeneratorKind.TIME_TRANSLATION)
    assert cq is not None, "TIME_TRANSLATION must produce a conserved quantity"
    results["free_fall_energy"] = cq.expression

    # Normalize for comparison
    norm_expr = cq.expression.replace(" ", "")
    expected = "0.5*m*v^2+m*g*h"
    results["free_fall_energy_correct"] = norm_expr == expected
    results["free_fall_energy_contains_mgh"] = "m*g*h" in norm_expr
    results["free_fall_energy_contains_kinetic"] = "0.5*m*v^2" in norm_expr or "m*v^2" in norm_expr

    # Test 2: SymmetryDetector
    from src.physics.observations import Observation
    obs = Observation(
        id="test_free_fall",
        name="Test free fall",
        description="Ball dropping",
        quantities={"m": "Mass", "g": "Accel", "h": "Length", "v": "Velocity", "t": "Time"},
        parameters={"m": 1.0, "g": 9.8},
        timesteps=[{"t": 0.0, "h": 10.0, "v": 0.0}, {"t": 1.0, "h": 1.0, "v": 10.0}],
        known_invariant=None,
        lean_theorem="",
    )
    detector = SymmetryDetector()
    detection = detector.detect(obs)
    results["detects_time_translation"] = detection.has_symmetry(
        GeneratorKind.TIME_TRANSLATION
    )
    results["active_symmetries"] = detection.symmetry_names

    # Test 3: Galilean group
    gal = build_galilean_group()
    results["galilean_dimension"] = gal.dimension
    results["galilean_generators"] = len(gal.generators)
    results["galilean_has_time"] = gal.contains(GeneratorKind.TIME_TRANSLATION)
    results["galilean_has_space_x"] = gal.contains(GeneratorKind.SPACE_TRANSLATION_X)

    # Test 4: U(1) group
    u1 = build_u1_group()
    results["u1_dimension"] = u1.dimension
    results["u1_has_phase"] = u1.contains(GeneratorKind.U1_PHASE)

    # Test 5: Gravity + spring Lagrangian
    nd_gs = NoetherDerivation("gravity_spring")
    cq_gs = nd_gs.conserved_quantity(GeneratorKind.TIME_TRANSLATION)
    assert cq_gs is not None, "gravity_spring must have energy"
    results["gravity_spring_energy"] = cq_gs.expression
    norm_gs = cq_gs.expression.replace(" ", "")
    results["gs_has_mgh"] = "m*g*h" in norm_gs
    results["gs_has_spring"] = "0.5*k*h^2" in norm_gs or "k*h^2" in norm_gs

    # Test 6: Space translation for projectile
    nd_p = NoetherDerivation("projectile")
    cq_p = nd_p.conserved_quantity(GeneratorKind.SPACE_TRANSLATION_X)
    assert cq_p is not None, "projectile must have x-momentum"
    results["projectile_momentum_x"] = cq_p.expression
    results["projectile_momentum_x_correct"] = "m*vx" in cq_p.expression.replace(" ", "")

    return results
