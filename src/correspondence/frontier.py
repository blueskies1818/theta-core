"""Formal frontier map — mathematical territory zoning for the explorer.

Phase 2.5 of the ROADMAP. The frontier map divides mathematical-physical
space into three zones that guide the explorer's search:

    ESTABLISHED  — Proven theorems + experimentally confirmed regimes
    UNCERTAIN    — Competing theories, limited data, open problems
    BREAKDOWN    — Known infinities, singularities, theory failure points

This is the explorer's COMPASS. Without it, the GNN+MCTS searches blindly —
it can find proofs but doesn't know which proofs matter. The frontier map
provides reward shaping that pulls the explorer toward the breakdown zone:
the boundary between known physics and unknown territory.

Zone boundaries are formal conditions, not just descriptions. They reference:
- Lean 4 theorems that can be checked by the proof checker
- Energy/length scales that define physical regime boundaries
- Gauge group structures that define the Standard Model domain
- Curvature invariants that define gravitational regimes

Integration points:
- src/explorer/structure_generator.py  — structures are scored against the map
- src/explorer/explorer_trainer.py      — reward shaping uses zone weights
- src/correspondence/limits.py          — experimental constraints per zone
- src/correspondence/failure_points.py  — specific failure coordinates
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Zone types
# ---------------------------------------------------------------------------


class ZoneType(Enum):
    """The three frontier zones.

    ESTABLISHED: Proven mathematically AND confirmed experimentally.
        Examples: Euclidean geometry, Newtonian gravity regime, Maxwell's equations,
        Standard Model gauge group SU(3)×SU(2)×U(1), Einstein field equations
        in the weak-field limit.

    UNCERTAIN: Competing mathematical frameworks, limited experimental data.
        Examples: Quantum gravity approaches (string theory, LQG, asymptotic safety),
        dark matter models, inflation models, high-energy SUSY extensions,
        the hierarchy problem landscape.

    BREAKDOWN: Where current theories produce mathematical contradictions.
        Examples: Planck-scale divergences, black hole singularities (Penrose-Hawking),
        Big Bang t=0, Landau poles in QED, non-renormalizable QFT divergences,
        the GR-QFT incompatibility at the Planck scale.
    """

    ESTABLISHED = "established"
    UNCERTAIN = "uncertain"
    BREAKDOWN = "breakdown"


# ---------------------------------------------------------------------------
# Boundary types
# ---------------------------------------------------------------------------


class BoundaryType(Enum):
    """Types of boundary conditions that define zone edges."""

    ENERGY_SCALE = "energy_scale"          # Boundary defined by energy threshold
    LENGTH_SCALE = "length_scale"          # Boundary defined by length threshold
    CURVATURE = "curvature"                # Boundary defined by curvature invariant
    GAUGE_GROUP = "gauge_group"            # Boundary defined by symmetry structure
    THEOREM_DEPENDENT = "theorem_dependent"  # Boundary defined by formal theorem
    TOPOLOGICAL = "topological"            # Boundary defined by topological invariant
    CONSERVATION = "conservation"          # Boundary defined by conservation law


@dataclass
class BoundaryCondition:
    """A formal condition that defines a zone boundary.

    These are machine-readable conditions that can, in principle, be evaluated
    against a candidate mathematical structure.

    Attributes:
        name: Human-readable name (e.g., "Planck energy scale")
        boundary_type: The kind of boundary
        formal_expression: Lean 4 expression or mathematical formula
        value: Threshold value if applicable (e.g., 1.22e19 for Planck energy in GeV)
        description: Human explanation
        checkable: Whether this boundary can currently be checked by our proof checker
        required_theorems: Lean 4 theorem names needed to evaluate this boundary
    """

    name: str
    boundary_type: BoundaryType
    formal_expression: str
    value: float | None = None
    description: str = ""
    checkable: bool = False
    required_theorems: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "boundary_type": self.boundary_type.value,
            "formal_expression": self.formal_expression,
            "value": self.value,
            "description": self.description,
            "checkable": self.checkable,
            "required_theorems": self.required_theorems,
        }


# ---------------------------------------------------------------------------
# Frontier zone
# ---------------------------------------------------------------------------


@dataclass
class FrontierZone:
    """A named region in the mathematical frontier map.

    Each zone has a type (Established/Uncertain/Breakdown), formal boundary
    conditions that define its edges, and a reward multiplier for guiding
    exploration.

    The reward_multiplier shapes the explorer's behavior:
    - ESTABLISHED zones get low multipliers (< 1.0): the explorer shouldn't
      spend time re-proving known results unless necessary for a larger goal
    - UNCERTAIN zones get moderate multipliers (~1.0): legitimate exploration
    - BREAKDOWN zones get HIGH multipliers (> 1.0): this is where we want
      the explorer to focus — the frontier where new physics might exist
    """

    name: str
    zone_type: ZoneType
    description: str = ""

    # Boundary conditions defining this zone's edges
    boundary_conditions: list[BoundaryCondition] = field(default_factory=list)

    # Reward multiplier for structures in this zone
    # > 1.0 = incentivize exploration, < 1.0 = de-prioritize
    reward_multiplier: float = 1.0

    # Formal theorems or physical laws that characterize this zone
    characterizing_theorems: list[str] = field(default_factory=list)

    # Known structures / theories in this zone (for reference)
    known_structures: list[str] = field(default_factory=list)

    # Open problems specific to this zone
    open_problems: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "zone_type": self.zone_type.value,
            "description": self.description,
            "boundary_conditions": [bc.to_dict() for bc in self.boundary_conditions],
            "reward_multiplier": self.reward_multiplier,
            "characterizing_theorems": self.characterizing_theorems,
            "known_structures": self.known_structures,
            "open_problems": self.open_problems,
        }


# ---------------------------------------------------------------------------
# Frontier map
# ---------------------------------------------------------------------------


@dataclass
class FrontierMap:
    """The complete frontier map — all zones and their boundaries.

    This is the machine-readable map of mathematical-physical space.
    It answers the question: "Where are we, and where should we go next?"

    Usage:
        frontier = FrontierMap.load("configs/frontier_map.yaml")
        zone = frontier.classify(structure)
        reward_multiplier = zone.reward_multiplier
    """

    zones: list[FrontierZone] = field(default_factory=list)

    # Metadata
    version: str = "0.1.0"
    description: str = ""

    def get_zone(self, name: str) -> FrontierZone | None:
        """Get a zone by name."""
        for z in self.zones:
            if z.name == name:
                return z
        return None

    def get_zones_by_type(self, zone_type: ZoneType) -> list[FrontierZone]:
        """Get all zones of a given type."""
        return [z for z in self.zones if z.zone_type == zone_type]

    def classify(
        self, structure: "MathematicalStructure | None" = None, **conditions
    ) -> FrontierZone | None:
        """Classify a structure or set of conditions into a zone.

        This is the primary API: given a candidate structure, determine
        which frontier zone it belongs to. Phase 2 uses simple rule-based
        matching against boundary conditions; Phase 3 will add numerical
        scoring from experimental data.

        Args:
            structure: A MathematicalStructure from the structure generator
            **conditions: Manual conditions (energy_scale, curvature, etc.)

        Returns:
            The best-matching FrontierZone, or None if unclassifiable
        """
        if structure is None and not conditions:
            return None

        # Check breakdown zones first — they take priority
        for zone in self.get_zones_by_type(ZoneType.BREAKDOWN):
            if self._matches_zone(structure, zone, **conditions):
                return zone

        # Then established — if it definitely matches known physics
        for zone in self.get_zones_by_type(ZoneType.ESTABLISHED):
            if self._matches_zone(structure, zone, **conditions):
                return zone

        # Default to uncertain — the frontier between known and unknown
        for zone in self.get_zones_by_type(ZoneType.UNCERTAIN):
            if self._matches_zone(structure, zone, **conditions):
                return zone

        # If nothing matches, return a generic uncertain zone
        return self._default_uncertain_zone()

    def _matches_zone(
        self,
        structure: "MathematicalStructure | None",
        zone: FrontierZone,
        **conditions,
    ) -> bool:
        """Check if a structure matches a zone's boundary conditions.

        Phase 2 implementation: keyword + domain matching against boundary
        descriptions. Phase 3 will upgrade to formal theorem checking.
        """
        if not zone.boundary_conditions:
            # Zones without boundary conditions match everything of their type
            return True

        match_count = 0
        attempted_count = 0  # Only count boundary conditions we actually test
        for bc in zone.boundary_conditions:
            result, attempted = self._check_boundary(structure, bc, **conditions)
            if result:
                match_count += 1
            if attempted:
                attempted_count += 1

        if attempted_count == 0:
            # No boundary conditions were testable with the given inputs
            # — match if structure has components that look relevant
            if structure is not None:
                # First: check boundary names against component text
                for bc in zone.boundary_conditions:
                    bc_terms = set(bc.name.lower().replace("_", " ").split())
                    for comp in structure.components:
                        comp_text = (comp.name + " " + comp.expression).lower()
                        if bc.name.lower() in comp_text:
                            return True
                        for term in bc_terms:
                            if len(term) > 2 and term in comp_text:
                                return True
                # Second: check structure domain against zone name
                domain_text = structure.domain.value.lower()
                zone_text = zone.name.lower()
                if domain_text in zone_text or zone_text in domain_text:
                    return True
                # Third: check boundary descriptions against component expressions
                for bc in zone.boundary_conditions:
                    bc_desc_words = set(bc.description.lower().split())
                    for comp in structure.components:
                        comp_text = (comp.name + " " + comp.expression).lower()
                        hits = sum(1 for w in bc_desc_words if len(w) > 3 and w in comp_text)
                        if hits >= 2:
                            return True
                return False
            return False

        # If explicit conditions were tested, any match is sufficient
        # (e.g., if we test energy_scale=1e19, any zone with that scale matches)
        if conditions and attempted_count > 0:
            return match_count >= 1

        # Otherwise require majority of testable conditions to match
        return match_count >= max(1, attempted_count // 2 + 1)

    @staticmethod
    def _check_boundary(
        structure: "MathematicalStructure | None",
        boundary: BoundaryCondition,
        **conditions,
    ) -> tuple[bool, bool]:
        """Check a single boundary condition against a structure or conditions.

        Returns:
            (matched, attempted) where attempted=True means we explicitly tested
            this boundary against provided conditions (not just structure matching).
        """
        # Check explicit conditions first (highest priority)
        if boundary.boundary_type == BoundaryType.ENERGY_SCALE:
            if "energy_scale" in conditions:
                val = conditions["energy_scale"]
                if boundary.value is not None:
                    return (val >= boundary.value, True)
                # Boundary condition with no threshold — can't match on value alone
                return (False, True)

        if boundary.boundary_type == BoundaryType.LENGTH_SCALE:
            if "length_scale" in conditions:
                val = conditions["length_scale"]
                if boundary.value is not None:
                    return (val <= boundary.value, True)
                # Boundary condition with no threshold — can't match on value alone
                return (False, True)

        if boundary.boundary_type == BoundaryType.CURVATURE:
            if "curvature" in conditions:
                return (True, True)

        if boundary.boundary_type == BoundaryType.GAUGE_GROUP:
            if structure is not None:
                # Check if structure references the gauge group
                for comp in structure.components:
                    if boundary.formal_expression.lower() in comp.expression.lower():
                        return (True, True)
                    if boundary.name.lower() in comp.name.lower():
                        return (True, True)
            if "gauge_group" in conditions:
                # Normalize: remove spacing around ×
                expected = boundary.formal_expression.replace(" ", "").replace("×", "×")
                actual = str(conditions["gauge_group"]).replace(" ", "").replace("×", "×")
                if expected.lower() == actual.lower():
                    return (True, True)
                # Partial match: e.g. "SU(3)" matches "SU(3)_C × SU(2)_L × U(1)_Y"
                key_parts = actual.split("×") if "×" in actual else [actual]
                expected_parts = expected.split("×") if "×" in expected else [expected]
                if any(kp.strip().lower() in expected.lower() for kp in key_parts):
                    return (True, True)

        # Theorem-dependent boundary: check if structure depends on the theorem
        if boundary.boundary_type == BoundaryType.THEOREM_DEPENDENT:
            if structure is not None:
                for thm_name in boundary.required_theorems:
                    # Search through dependency values (structure.dependencies
                    # is dict[str, list[str]] — component → theorem names)
                    for dep_list in structure.dependencies.values():
                        if any(thm_name in d for d in dep_list):
                            return (True, True)
                        if thm_name in dep_list:
                            return (True, True)
                    # Also check component names and expressions
                    for comp in structure.components:
                        if thm_name.lower() in comp.expression.lower():
                            return (True, True)
                        if thm_name.lower() in comp.name.lower():
                            return (True, True)

        # Conservation boundary type
        if boundary.boundary_type == BoundaryType.CONSERVATION:
            if "conservation" in conditions:
                return (True, True)

        # Topological boundary type
        if boundary.boundary_type == BoundaryType.TOPOLOGICAL:
            if "topological" in conditions:
                return (True, True)

        # For name-based matching (fallback, from structure only)
        if structure is not None:
            bc_lower = boundary.name.lower()
            # Also extract key terms from the boundary name for broader matching
            bc_terms = set(bc_lower.replace("_", " ").split())
            for comp in structure.components:
                comp_text = (comp.name + " " + comp.expression).lower()
                if bc_lower in comp_text:
                    return (True, True)
                # Check if any significant boundary term appears in component
                for term in bc_terms:
                    if len(term) > 2 and term in comp_text:
                        return (True, True)

        return (False, False)

    @staticmethod
    def _default_uncertain_zone() -> FrontierZone:
        """Return a generic 'unclassified' uncertain zone."""
        return FrontierZone(
            name="unknown_territory",
            zone_type=ZoneType.UNCERTAIN,
            description="Unclassified mathematical territory — newly explored region",
            reward_multiplier=0.8,  # Slightly de-prioritize until we know more
        )

    def get_reward_multiplier(
        self, structure: "MathematicalStructure | None" = None, **conditions
    ) -> float:
        """Get the reward multiplier for a structure or condition set.

        This is the primary integration point with the reward pipeline.
        The explorer trainer calls this to weight rewards based on territory.
        """
        zone = self.classify(structure, **conditions)
        if zone is None:
            return 1.0
        return zone.reward_multiplier

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "description": self.description,
            "zones": [z.to_dict() for z in self.zones],
        }

    def summary(self) -> str:
        """Human-readable summary of the frontier map."""
        established = self.get_zones_by_type(ZoneType.ESTABLISHED)
        uncertain = self.get_zones_by_type(ZoneType.UNCERTAIN)
        breakdown = self.get_zones_by_type(ZoneType.BREAKDOWN)

        lines = [
            f"Frontier Map v{self.version}",
            f"  {len(self.zones)} zones total",
            f"  ESTABLISHED:  {len(established)} zones — " +
            ", ".join(z.name for z in established),
            f"  UNCERTAIN:    {len(uncertain)} zones — " +
            ", ".join(z.name for z in uncertain),
            f"  BREAKDOWN:    {len(breakdown)} zones — " +
            ", ".join(z.name for z in breakdown),
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Factory: build the standard frontier map
# ---------------------------------------------------------------------------


def build_standard_frontier_map() -> FrontierMap:
    """Build the standard frontier map with known physics zones.

    This encodes the current state of theoretical physics as a frontier map:
    - Established: GR (weak-field), Standard Model, QED, thermodynamics
    - Uncertain: quantum gravity approaches, dark matter/energy, inflation
    - Breakdown: Planck regime, singularities, non-renormalizable divergences

    Each zone has formal boundary conditions that can be progressively
    upgraded from keyword matching to theorem checking as the system matures.

    Returns:
        A FrontierMap with all standard zones populated
    """
    zones: list[FrontierZone] = []

    # =========================================================================
    # ESTABLISHED ZONES
    # =========================================================================

    # --- General Relativity (weak-field, classical) ---
    zones.append(FrontierZone(
        name="gr_classical",
        zone_type=ZoneType.ESTABLISHED,
        description=(
            "Classical general relativity in the weak-to-moderate field regime. "
            "Einstein field equations hold. Curvature is finite and well-behaved. "
            "Experimentally confirmed by: perihelion precession, gravitational "
            "lensing, Shapiro delay, binary pulsar orbital decay, GW150914 waveform "
            "(inspiral through ringdown)."
        ),
        boundary_conditions=[
            BoundaryCondition(
                name="einstein_field_equations",
                boundary_type=BoundaryType.THEOREM_DEPENDENT,
                formal_expression="R_μν - (1/2)R g_μν + Λ g_μν = (8πG/c⁴) T_μν",
                description="Structure must satisfy or generalize the Einstein field equations",
                required_theorems=["einstein_field_equations", "ricci_tensor", "hilbert_action"],
            ),
            BoundaryCondition(
                name="weak_field_limit",
                boundary_type=BoundaryType.CURVATURE,
                formal_expression="|Φ| << c²",
                description="Newtonian potential much less than c² — weak field",
            ),
            BoundaryCondition(
                name="classical_regime",
                boundary_type=BoundaryType.LENGTH_SCALE,
                formal_expression="L >> l_Planck ≈ 1.616e-35 m",
                value=1.616e-35,
                description="Length scales much larger than Planck length",
            ),
        ],
        reward_multiplier=0.3,  # Low: this is well-trodden territory
        characterizing_theorems=[
            "einstein_field_equations",
            "birkhoff_theorem",
            "no_hair_theorem",
            "positive_energy_theorem",
        ],
        known_structures=[
            "Einstein-Hilbert action",
            "Schwarzschild metric",
            "Kerr metric",
            "FLRW metric",
            "Friedmann equations",
        ],
        open_problems=[
            "cosmic_censorship_hypothesis",
            "final_state_conjecture",
        ],
    ))

    # --- Standard Model of Particle Physics ---
    zones.append(FrontierZone(
        name="standard_model",
        zone_type=ZoneType.ESTABLISHED,
        description=(
            "The Standard Model of particle physics: SU(3)×SU(2)×U(1) gauge theory. "
            "Experimentally confirmed up to ~13 TeV (LHC). All predicted particles "
            "observed including the Higgs boson (2012)."
        ),
        boundary_conditions=[
            BoundaryCondition(
                name="su3_su2_u1_gauge_group",
                boundary_type=BoundaryType.GAUGE_GROUP,
                formal_expression="SU(3)_C × SU(2)_L × U(1)_Y",
                description="Structure must contain or generalize the Standard Model gauge group",
                required_theorems=["gauge_theory", "yang_mills_theory", "higgs_mechanism"],
            ),
            BoundaryCondition(
                name="higgs_mechanism",
                boundary_type=BoundaryType.THEOREM_DEPENDENT,
                formal_expression="Spontaneous symmetry breaking: SU(2)_L × U(1)_Y → U(1)_EM",
                description="Electroweak symmetry breaking via Higgs mechanism",
                required_theorems=["higgs_mechanism", "spontaneous_symmetry_breaking"],
            ),
            BoundaryCondition(
                name="particle_content",
                boundary_type=BoundaryType.CONSERVATION,
                formal_expression=(
                    "3 generations: (u,d), (c,s), (t,b) quarks; "
                    "e, μ, τ leptons; ν_e, ν_μ, ν_τ neutrinos; "
                    "γ, W±, Z, g, H bosons"
                ),
                description="Known particle content with correct quantum numbers",
            ),
        ],
        reward_multiplier=0.3,
        characterizing_theorems=[
            "noether_theorem",
            "ward_identities",
            "lsz_reduction",
            "optical_theorem",
            "cpt_theorem",
        ],
        known_structures=[
            "Standard Model Lagrangian",
            "QCD Lagrangian",
            "Electroweak Lagrangian",
            "Yukawa couplings",
        ],
        open_problems=[
            "strong_cp_problem",
            "hierarchy_problem",
            "neutrino_masses",
            "baryon_asymmetry",
        ],
    ))

    # --- Quantum Electrodynamics ---
    zones.append(FrontierZone(
        name="qed",
        zone_type=ZoneType.ESTABLISHED,
        description=(
            "Quantum Electrodynamics: U(1) gauge theory. The most precisely tested "
            "theory in physics. Electron g-factor matches theory to 12 decimal places."
        ),
        boundary_conditions=[
            BoundaryCondition(
                name="u1_gauge_theory",
                boundary_type=BoundaryType.GAUGE_GROUP,
                formal_expression="U(1)_EM",
                description="QED is a U(1) gauge theory",
                required_theorems=["gauge_invariance", "ward_takahashi_identity"],
            ),
        ],
        reward_multiplier=0.2,  # Very low: this territory is fully mapped
        characterizing_theorems=[
            "ward_takahashi_identity",
            "lsz_reduction_formula",
            "optical_theorem",
            "spin_statistics_theorem",
        ],
        known_structures=[
            "QED Lagrangian: ψ̄(iγ^μ D_μ - m)ψ - (1/4)F_μν F^μν",
            "Dirac equation",
            "Maxwell's equations",
        ],
        open_problems=[
            "landau_pole",  # Actually a breakdown — see breakdown zone
        ],
    ))

    # --- Thermodynamics & Statistical Mechanics ---
    zones.append(FrontierZone(
        name="thermodynamics",
        zone_type=ZoneType.ESTABLISHED,
        description=(
            "Equilibrium thermodynamics and statistical mechanics. "
            "Laws of thermodynamics are among the most universal in physics."
        ),
        boundary_conditions=[
            BoundaryCondition(
                name="second_law",
                boundary_type=BoundaryType.CONSERVATION,
                formal_expression="dS ≥ 0 for isolated systems",
                description="Second law of thermodynamics",
            ),
        ],
        reward_multiplier=0.1,  # Lowest: foundational, few new discoveries here
        characterizing_theorems=[
            "zeroth_law_thermodynamics",
            "first_law_thermodynamics",
            "second_law_thermodynamics",
            "fluctuation_dissipation_theorem",
            "boltzmann_h_theorem",
        ],
        known_structures=[
            "Canonical ensemble",
            "Grand canonical ensemble",
            "Microcanonical ensemble",
            "Boltzmann distribution",
            "Gibbs free energy",
        ],
        open_problems=[
            "arrow_of_time",
            "thermalization_quantum_systems",
        ],
    ))

    # =========================================================================
    # UNCERTAIN ZONES
    # =========================================================================

    # --- Quantum Gravity ---
    zones.append(FrontierZone(
        name="quantum_gravity",
        zone_type=ZoneType.UNCERTAIN,
        description=(
            "The quantum gravity regime: where general relativity and quantum "
            "mechanics must both be accounted for. No experimentally confirmed "
            "theory exists. Competing approaches include string theory, loop "
            "quantum gravity, asymptotic safety, causal dynamical triangulations, "
            "and causal set theory."
        ),
        boundary_conditions=[
            BoundaryCondition(
                name="planck_scale_energy",
                boundary_type=BoundaryType.ENERGY_SCALE,
                formal_expression="E ~ M_Planck ≈ 1.22 × 10^19 GeV",
                value=1.22e19,
                description="Energy scale where quantum gravity effects become dominant",
            ),
            BoundaryCondition(
                name="planck_length",
                boundary_type=BoundaryType.LENGTH_SCALE,
                formal_expression="L ~ l_Planck ≈ 1.616 × 10^-35 m",
                value=1.616e-35,
                description="Length scale where spacetime is expected to be quantized",
            ),
        ],
        reward_multiplier=2.0,  # HIGH: this is where new physics lives
        characterizing_theorems=[
            "bekenstein_hawking_entropy",
            "holographic_principle",
            "ads_cft_correspondence",
            "weinberg_witten_theorem",
        ],
        known_structures=[
            "String theory action (Polyakov, Nambu-Goto)",
            "Loop quantum gravity Hamiltonian constraint",
            "Asymptotic safety: non-Gaussian fixed point",
            "Einstein-Hilbert action + R², R_μνR^μν terms",
        ],
        open_problems=[
            "quantum_gravity_uv_completion",
            "black_hole_information_paradox",
            "quantum_measurement_problem_in_curved_spacetime",
            "time_problem_in_canonical_quantum_gravity",
        ],
    ))

    # --- Dark Matter ---
    zones.append(FrontierZone(
        name="dark_matter",
        zone_type=ZoneType.UNCERTAIN,
        description=(
            "Dark matter: ~27% of the universe's energy budget. Gravitational "
            "effects confirmed (galaxy rotation curves, bullet cluster, CMB "
            "anisotropies) but particle nature unknown. Candidates span from "
            "WIMPs (GeV-TeV) to axions (μeV) to primordial black holes."
        ),
        boundary_conditions=[
            BoundaryCondition(
                name="non_relativistic_cold_dark_matter",
                boundary_type=BoundaryType.THEOREM_DEPENDENT,
                formal_expression="Ω_CDM h² ≈ 0.12 from Planck 2018",
                description="Cold dark matter density from CMB",
            ),
        ],
        reward_multiplier=1.5,
        characterizing_theorems=[
            "virial_theorem",
            "jeans_instability",
            "collisionless_boltzmann_equation",
        ],
        known_structures=[
            "WIMP freeze-out",
            "Axion potential: V(a) = m_a² f_a² (1 - cos(a/f_a))",
            "Sterile neutrino mixing",
            "Self-interacting dark matter",
        ],
        open_problems=[
            "dark_matter_particle_identity",
            "core_cusp_problem",
            "missing_satellite_problem",
            "too_big_to_fail_problem",
        ],
    ))

    # --- Dark Energy / Cosmological Constant ---
    zones.append(FrontierZone(
        name="dark_energy",
        zone_type=ZoneType.UNCERTAIN,
        description=(
            "Dark energy: ~68% of the universe's energy budget. Drives accelerated "
            "cosmic expansion. The cosmological constant problem — the 120-orders-of-"
            "magnitude discrepancy between QFT vacuum energy and observed Λ — remains "
            "the largest quantitative mismatch in physics."
        ),
        boundary_conditions=[
            BoundaryCondition(
                name="cosmological_constant_problem",
                boundary_type=BoundaryType.THEOREM_DEPENDENT,
                formal_expression="ρ_vac(QFT) / ρ_vac(obs) ≈ 10^120",
                description="The cosmological constant problem",
            ),
        ],
        reward_multiplier=2.0,  # HIGH: solving this is a major goal
        characterizing_theorems=[
            "friedmann_equations",
            "positive_energy_theorem",
            "weinberg_no_go_theorem",
        ],
        known_structures=[
            "Cosmological constant Λ",
            "Quintessence: scalar field with w > -1",
            "Phantom dark energy: w < -1",
            "Modified gravity: f(R), DGP, massive gravity",
        ],
        open_problems=[
            "cosmological_constant_problem",
            "coincidence_problem",
            "dark_energy_equation_of_state",
        ],
    ))

    # --- Inflation ---
    zones.append(FrontierZone(
        name="inflation",
        zone_type=ZoneType.UNCERTAIN,
        description=(
            "Cosmological inflation: early-universe exponential expansion. Solves "
            "horizon, flatness, and monopole problems. Predicts nearly scale-invariant "
            "primordial power spectrum. Specific model (scalar field potential) not yet "
            "identified. B-mode polarization not yet detected."
        ),
        boundary_conditions=[
            BoundaryCondition(
                name="slow_roll_inflation",
                boundary_type=BoundaryType.THEOREM_DEPENDENT,
                formal_expression="ε = (M_Pl²/2)(V'/V)² << 1, η = M_Pl²(V''/V) << 1",
                description="Slow-roll inflation conditions",
            ),
        ],
        reward_multiplier=1.2,
        characterizing_theorems=[
            "mukhanov_sasaki_equation",
            "weinberg_theorem_adiabatic",
            "consistency_relation_tensor_scalar",
        ],
        known_structures=[
            "Single-field slow-roll: V(φ) = (1/2)m²φ²",
            "Starobinsky: f(R) = R + R²/(6M²)",
            "Higgs inflation: non-minimal coupling ξ",
            "Axion monodromy",
        ],
        open_problems=[
            "inflaton_potential",
            "initial_singularity",
            "eternal_inflation_measure",
            "transplanckian_problem",
        ],
    ))

    # =========================================================================
    # BREAKDOWN ZONES
    # =========================================================================

    # --- Planck-scale breakdown ---
    zones.append(FrontierZone(
        name="planck_breakdown",
        zone_type=ZoneType.BREAKDOWN,
        description=(
            "The Planck scale: where general relativity and quantum field theory "
            "become mutually incompatible. Spacetime is expected to lose its smooth "
            "manifold structure. All perturbative approaches fail. The breakdown is "
            "fundamental — not a failure of technique but a signal that our current "
            "mathematical framework (smooth manifolds + quantum fields on them) is "
            "incomplete."
        ),
        boundary_conditions=[
            BoundaryCondition(
                name="planck_energy",
                boundary_type=BoundaryType.ENERGY_SCALE,
                formal_expression="E ≥ E_Planck = sqrt(ħc⁵/G) ≈ 1.22 × 10^19 GeV",
                value=1.22e19,
                description="Energies at or above the Planck scale",
            ),
            BoundaryCondition(
                name="planck_length_boundary",
                boundary_type=BoundaryType.LENGTH_SCALE,
                formal_expression="L ≤ l_Planck = sqrt(ħG/c³) ≈ 1.616 × 10^-35 m",
                value=1.616e-35,
                description="Lengths at or below the Planck length",
            ),
            BoundaryCondition(
                name="quantum_gravity_regime",
                boundary_type=BoundaryType.CURVATURE,
                formal_expression=(
                    "R_μνρσ R^μνρσ ~ l_Planck^-4, or equivalently, "
                    "when the Compton wavelength ~ Schwarzschild radius"
                ),
                description="Curvature invariants at Planckian values",
            ),
        ],
        reward_multiplier=3.0,  # HIGHEST: this is THE target zone
        characterizing_theorems=[
            "penrose_hawking_singularity_theorems",
            "bekenstein_bound",
            "holographic_entropy_bound",
            "no_global_symmetries_quantum_gravity",
        ],
        known_structures=[
            "GR + QFT (mutually incompatible here)",
            "String theory (UV-complete, landscape problem)",
            "Loop quantum gravity (discrete spacetime)",
            "Asymptotic safety (non-perturbative renormalizability)",
        ],
        open_problems=[
            "quantum_gravity_uv_completion",
            "spacetime_singularity_resolution",
            "black_hole_information_paradox",
            "holographic_principle_proof",
            "background_independence",
        ],
    ))

    # --- Black hole singularities ---
    zones.append(FrontierZone(
        name="black_hole_singularity",
        zone_type=ZoneType.BREAKDOWN,
        description=(
            "Black hole singularities: regions where spacetime curvature becomes "
            "infinite. Penrose-Hawking singularity theorems prove these are generic "
            "in GR, not artifacts of symmetry. Geodesic incompleteness means time "
            "ends for infalling observers — a pathological prediction that signals "
            "GR's incompleteness."
        ),
        boundary_conditions=[
            BoundaryCondition(
                name="penrose_hawking_conditions",
                boundary_type=BoundaryType.THEOREM_DEPENDENT,
                formal_expression=(
                    "Trapped surface + null energy condition + generic condition "
                    "+ global hyperbolicity (or appropriate causality) → singularity"
                ),
                description="Penrose-Hawking singularity theorem conditions",
                required_theorems=[
                    "penrose_singularity_theorem",
                    "hawking_singularity_theorem",
                    "penrose_hawking_singularity_theorems",
                ],
            ),
            BoundaryCondition(
                name="curvature_blowup",
                boundary_type=BoundaryType.CURVATURE,
                formal_expression="Kretschmann scalar K = R_μνρσ R^μνρσ → ∞",
                description="Curvature invariants diverge at the singularity",
            ),
        ],
        reward_multiplier=2.5,  # HIGH: solving singularities is a major goal
        characterizing_theorems=[
            "penrose_1965_singularity_theorem",
            "hawking_penrose_1970",
            "no_hair_theorem",
            "cosmic_censorship_conjecture",
        ],
        known_structures=[
            "Schwarzschild singularity (r=0, spacelike)",
            "Kerr singularity (r=0, ring, timelike)",
            "Reissner-Nordström singularity",
            "BKL chaotic singularity",
        ],
        open_problems=[
            "singularity_resolution_quantum_gravity",
            "cosmic_censorship_proof",
            "mass_inflation_instability",
            "final_state_conjecture",
        ],
    ))

    # --- Big Bang t=0 ---
    zones.append(FrontierZone(
        name="big_bang_singularity",
        zone_type=ZoneType.BREAKDOWN,
        description=(
            "The Big Bang singularity at t=0: the FLRW scale factor a(t) → 0, "
            "density ρ → ∞, and curvature R → ∞. This is the beginning of time "
            "in classical GR — a boundary to spacetime itself. The singularity "
            "theorems guarantee this under reasonable assumptions (Hawking-Penrose)."
        ),
        boundary_conditions=[
            BoundaryCondition(
                name="flrw_initial_singularity",
                boundary_type=BoundaryType.THEOREM_DEPENDENT,
                formal_expression=(
                    "a(t) → 0 as t → 0, ρ ∝ a^{-3(1+w)} → ∞, "
                    "R ∝ a^{-2} (ä/a + (ȧ/a)²) → ∞"
                ),
                description="FLRW cosmological singularity conditions",
                required_theorems=[
                    "hawking_penrose_singularity",
                    "borde_guth_vilenkin_theorem",
                ],
            ),
        ],
        reward_multiplier=2.5,
        characterizing_theorems=[
            "hawking_penrose_singularity_theorems",
            "borde_guth_vilenkin_theorem",
            "weyl_curvature_hypothesis",
        ],
        known_structures=[
            "FLRW metric with a(t) → 0",
            "Mixmaster universe (BKL)",
            "Hartle-Hawking no-boundary proposal",
            "Vilenkin tunneling proposal",
        ],
        open_problems=[
            "initial_singularity_resolution",
            "arrow_of_time_origin",
            "low_entropy_initial_condition",
            "quantum_cosmology_interpretation",
        ],
    ))

    # --- Non-renormalizable QFT divergences ---
    zones.append(FrontierZone(
        name="qft_divergence",
        zone_type=ZoneType.BREAKDOWN,
        description=(
            "Non-renormalizable divergences in quantum field theory. Perturbative "
            "quantum gravity (graviton spin-2 field on flat spacetime) has infinite "
            "UV divergences at two loops that cannot be absorbed by renormalizing "
            "a finite number of parameters. This is a mathematical breakdown of the "
            "perturbative approach — not necessarily of the underlying theory."
        ),
        boundary_conditions=[
            BoundaryCondition(
                name="non_renormalizable_interaction",
                boundary_type=BoundaryType.ENERGY_SCALE,
                formal_expression=(
                    "Operator dimension > 4 → coupling has negative mass dimension "
                    "→ perturbation theory breaks down at E ~ Λ"
                ),
                description="Non-renormalizable operators cause perturbative breakdown",
                value=None,  # Depends on the specific operator
            ),
            BoundaryCondition(
                name="landau_pole",
                boundary_type=BoundaryType.ENERGY_SCALE,
                formal_expression=(
                    "Running coupling g(μ) diverges at finite scale μ = Λ_Landau"
                ),
                description="Landau pole: coupling blows up at finite energy",
            ),
        ],
        reward_multiplier=2.0,
        characterizing_theorems=[
            "power_counting_theorem",
            "weinberg_theorem_low_energy",
            "wilsonian_renormalization_group",
        ],
        known_structures=[
            "Fermi theory (4-fermion, dim-6, non-renormalizable)",
            "GR as EFT (non-renormalizable at 2 loops)",
            "QED Landau pole (triviality)",
            "Higgs triviality bound",
        ],
        open_problems=[
            "asymptotic_safety_evidence",
            "uv_completion_eft",
            "non_perturbative_quantum_gravity",
        ],
    ))

    # --- GR-QFT incompatibility ---
    zones.append(FrontierZone(
        name="gr_qft_incompatibility",
        zone_type=ZoneType.BREAKDOWN,
        description=(
            "The fundamental incompatibility between General Relativity and Quantum "
            "Field Theory. GR describes spacetime as a dynamical geometry; QFT "
            "describes matter as operator-valued distributions on a fixed background. "
            "At the Planck scale, quantum fluctuations of the metric become of order "
            "unity — you cannot treat spacetime as a fixed background, and you cannot "
            "quantize a theory where the stage itself is part of the play."
        ),
        boundary_conditions=[
            BoundaryCondition(
                name="planck_scale_where_gr_and_qft_clash",
                boundary_type=BoundaryType.ENERGY_SCALE,
                formal_expression="E ≥ M_Planck ≈ 1.22 × 10^19 GeV",
                value=1.22e19,
                description="Energy where quantum gravitational effects are O(1)",
            ),
            BoundaryCondition(
                name="metric_fluctuations_order_unity",
                boundary_type=BoundaryType.CURVATURE,
                formal_expression="δg_μν / g_μν ~ 1",
                description="Metric fluctuations become order unity at Planck scale",
            ),
        ],
        reward_multiplier=3.0,  # HIGHEST: this is the core problem
        characterizing_theorems=[
            "weinberg_witten_theorem",
            "marolf_theorem",
            "penrose_hawking_singularity_theorems",
        ],
        known_structures=[
            "GR + Standard Model (mutually incompatible)",
            "String theory (unified, but background-dependent formulation)",
            "LQG (background-independent, but classical limit unclear)",
            "Causal dynamical triangulations (non-perturbative, numerical)",
        ],
        open_problems=[
            "theory_of_everything",
            "unification_gr_qft",
            "background_independence",
            "problem_of_time",
        ],
    ))

    return FrontierMap(
        zones=zones,
        version="0.1.0",
        description=(
            "Standard frontier map encoding the current state of theoretical physics. "
            "Zones are defined by formal boundary conditions that can be evaluated "
            "against candidate mathematical structures. The map guides the explorer "
            "toward breakdown zones where new physics must exist."
        ),
    )


# ---------------------------------------------------------------------------
# YAML serialization
# ---------------------------------------------------------------------------


def frontier_map_to_yaml(frontier_map: FrontierMap) -> str:
    """Serialize a FrontierMap to YAML string."""
    import yaml

    return yaml.dump(frontier_map.to_dict(), default_flow_style=False, sort_keys=False)


def frontier_map_from_dict(data: dict[str, Any]) -> FrontierMap:
    """Deserialize a FrontierMap from a dictionary (loaded from YAML)."""
    zones = []
    for zdata in data.get("zones", []):
        bcs = []
        for bc_data in zdata.get("boundary_conditions", []):
            bcs.append(BoundaryCondition(
                name=bc_data["name"],
                boundary_type=BoundaryType(bc_data["boundary_type"]),
                formal_expression=bc_data.get("formal_expression", ""),
                value=bc_data.get("value"),
                description=bc_data.get("description", ""),
                checkable=bc_data.get("checkable", False),
                required_theorems=bc_data.get("required_theorems", []),
            ))

        zones.append(FrontierZone(
            name=zdata["name"],
            zone_type=ZoneType(zdata["zone_type"]),
            description=zdata.get("description", ""),
            boundary_conditions=bcs,
            reward_multiplier=zdata.get("reward_multiplier", 1.0),
            characterizing_theorems=zdata.get("characterizing_theorems", []),
            known_structures=zdata.get("known_structures", []),
            open_problems=zdata.get("open_problems", []),
        ))

    return FrontierMap(
        zones=zones,
        version=data.get("version", "0.1.0"),
        description=data.get("description", ""),
    )


def load_frontier_map(path: str = "configs/frontier_map.yaml") -> FrontierMap:
    """Load a FrontierMap from a YAML file.

    If the file doesn't exist, builds and returns the standard map.
    """
    import os
    import yaml
    from pathlib import Path

    filepath = Path(path)
    if filepath.exists():
        with open(filepath) as f:
            data = yaml.safe_load(f)
        return frontier_map_from_dict(data)

    # Fall back to standard map
    return build_standard_frontier_map()
