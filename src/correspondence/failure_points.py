"""Formal failure coordinates — known points where current theories break.

Phase 2.7 of the ROADMAP. Encodes the exact mathematical conditions where
GR, QFT, and the Standard Model produce infinities, singularities, or
unphysical predictions.

These are NEGATIVE WAYPOINTS for the explorer:
- A structure that reproduces the same failure gets a penalty
- A structure that remains finite and consistent at these points gets a bonus
- This directly incentivizes solving the known problems rather than
  replicating the known failures

Each failure point has:
1. A formal mathematical condition (what goes wrong and where)
2. The relevant physical scale (energy, length, curvature)
3. Which theories fail there
4. The severity (how fundamental the failure is)
5. A reward modifier (how much to penalize reproducing it / reward solving it)

Integration:
- src/explorer/explorer_trainer.py — reward modification based on failure behavior
- src/correspondence/frontier.py — failure points are BREAKDOWN zone anchors
- src/correspondence/limits.py — experimental constraints related to failures
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Failure severity
# ---------------------------------------------------------------------------


class FailureSeverity(Enum):
    """How fundamental a failure is.

    CATASTROPHIC — Theory produces literal infinities (singularities, divergences)
        Example: GR predicts geodesic incompleteness inside black holes.
        A successful successor theory must resolve this.

    PATHOLOGICAL — Theory makes a mathematically valid but physically wrong prediction
        Example: QFT vacuum energy predicts Λ ~ 10^120 × observed.
        A successor theory must explain the mismatch.

    INCOMPLETE — Theory is silent or provides no prediction
        Example: Standard Model doesn't explain dark matter.
        A successor theory should provide a mechanism.

    TENSION — Theory makes predictions that disagree with other well-tested theories
        Example: GR and QFT are mutually incompatible at high energies.
        A successor theory must reconcile both.
    """

    CATASTROPHIC = "catastrophic"
    PATHOLOGICAL = "pathological"
    INCOMPLETE = "incomplete"
    TENSION = "tension"


# ---------------------------------------------------------------------------
# Failure regime types
# ---------------------------------------------------------------------------


class FailureRegime(Enum):
    """The physical regime where the failure occurs."""

    PLANCK_SCALE = "planck_scale"
    SINGULARITY = "singularity"
    QUANTUM_GRAVITY = "quantum_gravity"
    HIGH_ENERGY_QFT = "high_energy_qft"
    COSMOLOGICAL = "cosmological"
    THERMODYNAMIC = "thermodynamic"
    INFORMATIONAL = "informational"


# ---------------------------------------------------------------------------
# Failure point
# ---------------------------------------------------------------------------


@dataclass
class FailurePoint:
    """A specific coordinate where current theories fail.

    This is the formal encoding of a known problem. The explorer evaluates
    every candidate structure at this coordinate. Behavior at failure points
    directly impacts reward:

    - Structure fails here too → NEGATIVE reward contribution
    - Structure is undefined here → MILDLY NEGATIVE (incompleteness)
    - Structure remains finite here → POSITIVE reward contribution
    - Structure predicts a resolution → STRONG POSITIVE reward

    Attributes:
        name: Human-readable identifier
        regime: The physical regime where this failure occurs
        severity: How fundamental the failure is
        description: What goes wrong
        formal_condition: Mathematical condition for the failure
        energy_scale_gev: Characteristic energy in GeV (if applicable)
        length_scale_m: Characteristic length in meters (if applicable)
        failing_theories: Which current theories fail at this point
        reward_solve_bonus: Extra reward if structure resolves this failure
        reward_reproduce_penalty: Penalty if structure reproduces this failure
        related_theorems: Lean 4 theorem names encoding this failure
        resolution_candidates: Known proposed resolutions (for reference)
    """

    name: str
    regime: FailureRegime
    severity: FailureSeverity
    description: str = ""
    formal_condition: str = ""
    energy_scale_gev: float | None = None
    length_scale_m: float | None = None
    failing_theories: list[str] = field(default_factory=list)
    reward_solve_bonus: float = 0.0
    reward_reproduce_penalty: float = 0.0
    related_theorems: list[str] = field(default_factory=list)
    resolution_candidates: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "regime": self.regime.value,
            "severity": self.severity.value,
            "description": self.description,
            "formal_condition": self.formal_condition,
            "energy_scale_gev": self.energy_scale_gev,
            "length_scale_m": self.length_scale_m,
            "failing_theories": self.failing_theories,
            "reward_solve_bonus": self.reward_solve_bonus,
            "reward_reproduce_penalty": self.reward_reproduce_penalty,
            "related_theorems": self.related_theorems,
            "resolution_candidates": self.resolution_candidates,
        }


# ---------------------------------------------------------------------------
# Failure coordinate system
# ---------------------------------------------------------------------------


@dataclass
class FailureCoordinateSystem:
    """The complete set of known failure coordinates.

    This is a searchable registry of every known point where current
    physics breaks down. The explorer uses this to:
    1. Check candidate structures against each failure point
    2. Compute reward modifications based on failure behavior
    3. Track which failures remain unresolved

    The coordinate system is versioned — as new experimental anomalies
    are discovered (Phase 3), new failure points can be added.
    """

    failure_points: list[FailurePoint] = field(default_factory=list)
    version: str = "0.1.0"
    description: str = ""

    def get_by_regime(self, regime: FailureRegime) -> list[FailurePoint]:
        """Get all failure points in a given physical regime."""
        return [fp for fp in self.failure_points if fp.regime == regime]

    def get_by_severity(self, severity: FailureSeverity) -> list[FailurePoint]:
        """Get all failure points of a given severity."""
        return [fp for fp in self.failure_points if fp.severity == severity]

    def get_catastrophic(self) -> list[FailurePoint]:
        """Get only the catastrophic failure points (literal infinities)."""
        return self.get_by_severity(FailureSeverity.CATASTROPHIC)

    def get_planck_scale(self) -> list[FailurePoint]:
        """Get failure points at the Planck scale (THE target regime)."""
        return self.get_by_regime(FailureRegime.PLANCK_SCALE)

    def estimate_reward_modifier(
        self,
        structure_name: str,
        resolved_failures: set[str],
        reproduced_failures: set[str],
    ) -> float:
        """Compute net reward modifier from failure point behavior.

        Args:
            structure_name: Name of the candidate structure
            resolved_failures: Set of failure point names the structure resolves
            reproduced_failures: Set of failure point names the structure reproduces

        Returns:
            Net reward modifier (additive): positive for resolving, negative for
            reproducing, zero for neutral.
        """
        modifier = 0.0

        for fp in self.failure_points:
            if fp.name in resolved_failures:
                modifier += fp.reward_solve_bonus
            elif fp.name in reproduced_failures:
                modifier -= fp.reward_reproduce_penalty

        return modifier

    def summary(self) -> str:
        """Human-readable summary of the failure coordinate system."""
        by_regime: dict[str, int] = {}
        by_severity: dict[str, int] = {}
        for fp in self.failure_points:
            by_regime[fp.regime.value] = by_regime.get(fp.regime.value, 0) + 1
            by_severity[fp.severity.value] = by_severity.get(fp.severity.value, 0) + 1

        lines = [
            f"Failure Coordinate System v{self.version}",
            f"  {len(self.failure_points)} failure points total",
            "",
            "  By regime:",
        ]
        for regime, count in sorted(by_regime.items()):
            lines.append(f"    {regime}: {count}")
        lines.append("")
        lines.append("  By severity:")
        for severity, count in sorted(by_severity.items()):
            lines.append(f"    {severity}: {count}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "description": self.description,
            "failure_points": [fp.to_dict() for fp in self.failure_points],
        }


# ---------------------------------------------------------------------------
# Factory: build the standard failure coordinate system
# ---------------------------------------------------------------------------


def build_standard_failure_coordinates() -> FailureCoordinateSystem:
    """Build the standard set of known failure coordinates.

    These encode the known breakdown locations in current theoretical physics.
    Each point is a specific condition where our best theories fail.

    Reward values are calibrated so that:
    - Resolving ALL catastrophic failures ≈ +10.0 bonus (very strong signal)
    - Resolving a single major failure ≈ +2.0 to +3.0
    - Reproducing a known failure ≈ -1.0 to -2.0 (significant penalty)
    - The net effect guides the explorer toward solving, not copying
    """

    points: list[FailurePoint] = []

    # =========================================================================
    # CATASTROPHIC FAILURES (literal infinities)
    # =========================================================================

    points.append(FailurePoint(
        name="planck_scale_divergence",
        regime=FailureRegime.PLANCK_SCALE,
        severity=FailureSeverity.CATASTROPHIC,
        description=(
            "At the Planck scale (E ~ 10^19 GeV, L ~ 10^-35 m), quantum "
            "fluctuations of the spacetime metric become of order unity. "
            "GR predicts smooth geometry; QFT predicts violent quantum "
            "fluctuations. Neither framework works. Perturbative quantum "
            "gravity has non-renormalizable divergences at two loops."
        ),
        formal_condition=(
            "Compton wavelength λ_C = ħ/(m c) equals Schwarzschild radius "
            "r_s = 2Gm/c² → m = m_Planck = sqrt(ħc/G) ≈ 2.176×10^-8 kg ≈ 1.22×10^19 GeV/c²"
        ),
        energy_scale_gev=1.22e19,
        length_scale_m=1.616e-35,
        failing_theories=["General Relativity", "Quantum Field Theory", "Standard Model"],
        reward_solve_bonus=3.0,
        reward_reproduce_penalty=1.5,
        related_theorems=[
            "penrose_hawking_singularity_theorems",
            "weinberg_witten_theorem",
            "power_counting_theorem",
        ],
        resolution_candidates=[
            "String theory: finite UV completion via extended objects",
            "Loop quantum gravity: discrete spacetime at Planck scale",
            "Asymptotic safety: non-perturbative UV fixed point",
            "Causal set theory: discrete causal structure",
        ],
    ))

    points.append(FailurePoint(
        name="black_hole_singularity",
        regime=FailureRegime.SINGULARITY,
        severity=FailureSeverity.CATASTROPHIC,
        description=(
            "Inside every black hole, GR predicts a spacetime singularity — "
            "a region of infinite curvature where time ends. Penrose (1965) "
            "proved this is generic: any collapsing star that forms a trapped "
            "surface must develop a singularity. Geodesic incompleteness means "
            "the theory cannot predict what happens to infalling observers."
        ),
        formal_condition=(
            "Existence of a trapped surface + null energy condition R_ab k^a k^b ≥ 0 "
            "+ generic condition + no closed timelike curves → incomplete causal geodesic"
        ),
        energy_scale_gev=None,  # Scale depends on black hole mass
        length_scale_m=None,     # Curvature singularity, not a length scale
        failing_theories=["General Relativity"],
        reward_solve_bonus=2.5,
        reward_reproduce_penalty=1.5,
        related_theorems=[
            "penrose_1965_singularity_theorem",
            "hawking_penrose_1970",
            "cosmic_censorship_conjecture",
        ],
        resolution_candidates=[
            "Quantum gravity: singularity replaced by Planck-scale structure",
            "Loop quantum gravity: bounce at Planck density",
            "String theory: fuzzball proposal for black hole interiors",
            "Regular black holes: Bardeen, Hayward, etc. (avoidance, not resolution)",
        ],
    ))

    points.append(FailurePoint(
        name="big_bang_initial_singularity",
        regime=FailureRegime.COSMOLOGICAL,
        severity=FailureSeverity.CATASTROPHIC,
        description=(
            "At t=0 in standard cosmology, the FLRW scale factor a(t) → 0, "
            "density ρ → ∞, and curvature R → ∞. The Borde-Guth-Vilenkin "
            "theorem shows that any universe that is on average expanding "
            "must be past-geodesically incomplete — the singularity is not "
            "an artifact of symmetry assumptions."
        ),
        formal_condition=(
            "FLRW metric: ds² = -dt² + a(t)² [dr²/(1-kr²) + r² dΩ²]. "
            "a(t) → 0 as t → 0. ρ ∝ a^{-3(1+w)} → ∞. "
            "R ∝ (ä/a + (ȧ/a)² + k/a²) → ∞."
        ),
        energy_scale_gev=1.22e19,  # At Planck time ~10^-43 s
        length_scale_m=1.616e-35,
        failing_theories=["General Relativity", "Standard Model of Cosmology (ΛCDM)"],
        reward_solve_bonus=2.5,
        reward_reproduce_penalty=1.5,
        related_theorems=[
            "hawking_penrose_singularity_theorems",
            "borde_guth_vilenkin_theorem",
        ],
        resolution_candidates=[
            "Hartle-Hawking no-boundary proposal: Euclidean signature at origin",
            "Loop quantum cosmology: big bounce from previous contraction",
            "String gas cosmology: Hagedorn phase avoids singularity",
            "Emergent spacetime: time emerges from a timeless fundamental state",
        ],
    ))

    points.append(FailurePoint(
        name="perturbative_quantum_gravity_non_renormalizable",
        regime=FailureRegime.QUANTUM_GRAVITY,
        severity=FailureSeverity.CATASTROPHIC,
        description=(
            "Treating GR as a perturbative QFT of spin-2 gravitons on flat "
            "spacetime produces non-renormalizable divergences at two-loop order. "
            "An infinite number of counterterms are needed to absorb UV divergences. "
            "This means perturbative quantum GR has no predictive power at high energies."
        ),
        formal_condition=(
            "Einstein-Hilbert action S = (1/16πG) ∫ d⁴x √(-g) R. "
            "Expand g_μν = η_μν + κ h_μν where κ = sqrt(32πG). "
            "Two-loop counterterm: R_μνρσ R_ρσαβ R_αβ^μν. "
            "Requires infinite counterterms → non-renormalizable by power counting. "
            "G has mass dimension -2 → each loop adds two powers of external momentum."
        ),
        energy_scale_gev=1.22e19,
        length_scale_m=None,
        failing_theories=["General Relativity (perturbative)", "Effective Field Theory of Gravity"],
        reward_solve_bonus=2.0,
        reward_reproduce_penalty=1.0,
        related_theorems=[
            "power_counting_theorem",
            "weinberg_theorem_low_energy",
        ],
        resolution_candidates=[
            "String theory: UV finite due to extended nature of strings",
            "Asymptotic safety: non-perturbative UV fixed point",
            "Hořava-Lifshitz gravity: anisotropic scaling improves UV behavior",
            "Causal dynamical triangulations: non-perturbative lattice approach",
        ],
    ))

    # =========================================================================
    # PATHOLOGICAL FAILURES (mathematically valid, physically wrong)
    # =========================================================================

    points.append(FailurePoint(
        name="cosmological_constant_problem",
        regime=FailureRegime.COSMOLOGICAL,
        severity=FailureSeverity.PATHOLOGICAL,
        description=(
            "QFT predicts vacuum energy density ρ_vac ~ M_Planck^4 ≈ 10^76 GeV^4. "
            "Observed dark energy density is ρ_Λ ≈ 10^-47 GeV^4. "
            "This 120-orders-of-magnitude mismatch is the largest quantitative "
            "failure in all of physics. Even with supersymmetry above the TeV "
            "scale, the mismatch is still ~60 orders of magnitude."
        ),
        formal_condition=(
            "ρ_vac(QFT) = (1/2) Σ_i (-1)^F ∫₀^Λ_UV d³k/(2π)³ sqrt(k² + m_i²) ~ Λ_UV⁴. "
            "For Λ_UV = M_Planck: ρ_vac ≈ 10^76 GeV⁴. "
            "Observed: ρ_Λ = Ω_Λ ρ_crit = 0.6889 × (2.775×10^11 h² eV⁴) ≈ 10^-47 GeV⁴."
        ),
        energy_scale_gev=2.3e-42,  # Dark energy scale (meV)
        length_scale_m=None,
        failing_theories=["Quantum Field Theory", "Standard Model", "General Relativity"],
        reward_solve_bonus=3.0,  # Solving the biggest quantitative failure in physics
        reward_reproduce_penalty=1.0,
        related_theorems=[
            "weinberg_no_go_theorem",
            "positive_energy_theorem",
        ],
        resolution_candidates=[
            "Anthropic landscape: our vacuum is one of many (string theory)",
            "Unimodular gravity: CC decouples from field equations",
            "Degravitation: massive gravity filters out vacuum energy",
            "Dynamical dark energy: CC relaxes to small value via scalar field",
        ],
    ))

    points.append(FailurePoint(
        name="hierarchy_problem",
        regime=FailureRegime.HIGH_ENERGY_QFT,
        severity=FailureSeverity.PATHOLOGICAL,
        description=(
            "The Higgs boson mass (125 GeV) is 17 orders of magnitude smaller "
            "than the Planck scale. In QFT, scalar masses receive quantum "
            "corrections δm_H² ~ Λ_UV² from every particle it couples to. "
            "Explaining m_H << M_Planck requires extreme fine-tuning unless "
            "new physics (SUSY, compositeness, extra dimensions) appears at "
            "the TeV scale."
        ),
        formal_condition=(
            "m_H² = m_H0² + δm_H². "
            "δm_H² ~ (y_t²/(16π²)) Λ_UV² + (g²/(64π²)) Λ_UV² + ... "
            "For Λ_UV = M_Planck ≈ 1.22×10^19 GeV, δm_H² ≈ 10^36 GeV². "
            "But m_H²(observed) ≈ (125 GeV)² ≈ 1.56×10^4 GeV². "
            "Cancellation requires fine-tuning to 1 part in 10^32."
        ),
        energy_scale_gev=1.25e2,  # Higgs mass
        length_scale_m=None,
        failing_theories=["Standard Model (as a complete theory)"],
        reward_solve_bonus=2.0,
        reward_reproduce_penalty=1.0,
        related_theorems=[
            "naturalness_principle",
            "wilsonian_renormalization_group",
        ],
        resolution_candidates=[
            "SUSY: fermion/boson loop cancellations above TeV scale",
            "Composite Higgs: Higgs is a bound state, form factor at TeV",
            "Large extra dimensions: fundamental Planck scale lowered to TeV",
            "Relaxion mechanism: dynamical relaxation of Higgs mass during inflation",
        ],
    ))

    points.append(FailurePoint(
        name="strong_cp_problem",
        regime=FailureRegime.HIGH_ENERGY_QFT,
        severity=FailureSeverity.PATHOLOGICAL,
        description=(
            "QCD allows a CP-violating term (θ/32π²) G_μν^a G̃^{a μν} in the "
            "Lagrangian. The parameter θ is experimentally constrained to "
            "|θ| < 10^-10 from neutron electric dipole moment limits. "
            "QFT provides no explanation for why θ is so small — it's a "
            "dimensionless parameter that could naturally be O(1)."
        ),
        formal_condition=(
            "L_QCD ⊃ (θ/32π²) G_μν^a G̃^{a μν}. "
            "d_n ≈ 5.2×10^-16 θ e·cm (neutron EDM). "
            "Experiment: |d_n| < 1.8×10^-26 e·cm → |θ| < 10^-10."
        ),
        energy_scale_gev=None,  # Low-energy phenomenon
        length_scale_m=None,
        failing_theories=["Quantum Chromodynamics", "Standard Model"],
        reward_solve_bonus=1.5,
        reward_reproduce_penalty=0.5,
        related_theorems=[
            "index_theorem",
            "chiral_anomaly",
        ],
        resolution_candidates=[
            "Peccei-Quinn mechanism: θ promoted to dynamical axion field",
            "Nelson-Barr mechanism: CP violation is spontaneous",
            "Massless up quark solution (disfavored by lattice QCD)",
        ],
    ))

    # =========================================================================
    # INCOMPLETE FAILURES (theory is silent)
    # =========================================================================

    points.append(FailurePoint(
        name="dark_matter_identity",
        regime=FailureRegime.COSMOLOGICAL,
        severity=FailureSeverity.INCOMPLETE,
        description=(
            "~27% of the universe's energy density is in the form of dark matter. "
            "Its gravitational effects are precisely measured (CMB, galaxy rotation "
            "curves, gravitational lensing, bullet cluster) but its particle nature "
            "is unknown. The Standard Model has no candidate particle with the right "
            "properties. Sterile neutrinos, axions, WIMPs, and primordial black holes "
            "are all viable."
        ),
        formal_condition=(
            "Ω_CDM h² = 0.11933 ± 0.00091 (Planck 2018). "
            "Dark matter must be: cold or warm (not hot), collisionless or weakly "
            "self-interacting, stable on cosmological timescales, produced in the "
            "early universe with the correct relic abundance."
        ),
        energy_scale_gev=None,  # Unknown — depends on particle mass
        length_scale_m=None,
        failing_theories=["Standard Model of Particle Physics"],
        reward_solve_bonus=2.0,
        reward_reproduce_penalty=0.0,  # No penalty — the Standard Model is simply silent
        related_theorems=[
            "virial_theorem",
            "freeze_out_mechanism",
        ],
        resolution_candidates=[
            "WIMP: weak-scale particle with freeze-out relic abundance",
            "Axion: solves strong CP + provides dark matter via misalignment",
            "Sterile neutrino: keV-scale, produced via oscillation",
            "Primordial black holes: formed before BBN",
            "Modified gravity: MOND, TeVeS (alters gravity, not particle)",
        ],
    ))

    points.append(FailurePoint(
        name="neutrino_masses",
        regime=FailureRegime.HIGH_ENERGY_QFT,
        severity=FailureSeverity.INCOMPLETE,
        description=(
            "Neutrino oscillations prove neutrinos have mass (Δm²_21 ≈ 7.5×10^-5 eV², "
            "|Δm²_31| ≈ 2.5×10^-3 eV²). But the Standard Model predicts massless "
            "neutrinos because it has no right-handed neutrino fields. The smallness "
            "of neutrino masses (m_ν < 0.1 eV) compared to charged fermions (m_e = 0.511 MeV, "
            "m_t = 173 GeV) requires explanation."
        ),
        formal_condition=(
            "Standard Model: L ⊃ -y_ν L̄ H ν_R (requires ν_R, not in minimal SM). "
            "Seesaw: m_ν ~ y² v²/M_R. For m_ν ~ 0.1 eV, y ~ 1 → M_R ~ 10^14 GeV. "
            "Alternative: radiative mass generation at loop level."
        ),
        energy_scale_gev=1.0e-10,  # Neutrino mass scale (0.1 eV)
        length_scale_m=None,
        failing_theories=["Standard Model (minimal)"],
        reward_solve_bonus=1.0,
        reward_reproduce_penalty=0.0,
        related_theorems=[
            "seesaw_mechanism",
            "lepton_number_violation",
        ],
        resolution_candidates=[
            "Type I seesaw: heavy right-handed neutrinos at GUT scale",
            "Type II seesaw: heavy SU(2)_L triplet scalar",
            "Type III seesaw: heavy SU(2)_L triplet fermions",
            "Radiative: loop-suppressed masses (Zee, Ma models)",
        ],
    ))

    points.append(FailurePoint(
        name="baryon_asymmetry",
        regime=FailureRegime.COSMOLOGICAL,
        severity=FailureSeverity.INCOMPLETE,
        description=(
            "The universe contains matter but almost no antimatter: "
            "η = (n_B - n_Ḇ)/n_γ ≈ 6×10^-10. The Standard Model has all three "
            "Sakharov conditions (baryon number violation via sphalerons, C and CP "
            "violation in the CKM matrix, out-of-equilibrium via electroweak phase "
            "transition), but the CP violation is too small by ~10 orders of magnitude "
            "and the phase transition is a crossover, not first-order, at the measured "
            "Higgs mass."
        ),
        formal_condition=(
            "Sakharov conditions: (1) B violation, (2) C and CP violation, "
            "(3) departure from thermal equilibrium. "
            "Observed: η_CMB = (6.104 ± 0.058)×10^-10 (Planck 2018). "
            "SM prediction: too small by factor ~10^-10."
        ),
        energy_scale_gev=1.0e2,  # Electroweak scale
        length_scale_m=None,
        failing_theories=["Standard Model of Particle Physics", "Standard Cosmology"],
        reward_solve_bonus=1.5,
        reward_reproduce_penalty=0.5,
        related_theorems=[
            "sakharov_conditions",
            "electroweak_baryogenesis",
        ],
        resolution_candidates=[
            "Leptogenesis: lepton asymmetry → baryon asymmetry via sphalerons",
            "Electroweak baryogenesis with BSM CP violation (needs 1st-order PT)",
            "GUT baryogenesis: heavy boson decays at GUT scale",
            "Affleck-Dine mechanism: scalar condensate in SUSY",
        ],
    ))

    # =========================================================================
    # TENSION FAILURES (predictions contradict)
    # =========================================================================

    points.append(FailurePoint(
        name="gr_qft_mutual_incompatibility",
        regime=FailureRegime.QUANTUM_GRAVITY,
        severity=FailureSeverity.TENSION,
        description=(
            "GR and QFT are built on contradictory mathematical foundations. "
            "GR: spacetime is a dynamical manifold, background-independent. "
            "QFT: fields are operator-valued distributions on fixed Minkowski background. "
            "In GR, time is part of the dynamical geometry; in QM, time is an external "
            "parameter. These frameworks cannot both be correct in their current "
            "formulations — yet both are experimentally confirmed in their domains."
        ),
        formal_condition=(
            "GR action: S = (1/16πG)∫ d⁴x √-g (R - 2Λ). "
            "QFT action: S = ∫ d⁴x L_matter(φ, ∂φ) on η_μν. "
            "No consistent quantum theory exists where the metric is simultaneously "
            "a dynamical field AND a fixed background. The problem of time: Wheeler-"
            "DeWitt equation Ĥ|ψ⟩ = 0 has no time evolution parameter."
        ),
        energy_scale_gev=1.22e19,
        length_scale_m=1.616e-35,
        failing_theories=["General Relativity", "Quantum Field Theory (jointly)"],
        reward_solve_bonus=3.0,  # This is what the whole project is about
        reward_reproduce_penalty=1.5,
        related_theorems=[
            "weinberg_witten_theorem",
            "marolf_theorem",
            "problem_of_time",
        ],
        resolution_candidates=[
            "String theory: QFT and GR unified in a single framework",
            "Loop quantum gravity: quantize geometry, matter lives on spin networks",
            "Causal dynamical triangulations: sum over geometries, non-perturbative",
            "Emergent gravity: spacetime is not fundamental (entropic, holographic)",
        ],
    ))

    points.append(FailurePoint(
        name="black_hole_information_paradox",
        regime=FailureRegime.INFORMATIONAL,
        severity=FailureSeverity.TENSION,
        description=(
            "Hawking radiation (1974) implies black holes evaporate into thermal "
            "radiation that carries no information about what fell in. If the black "
            "hole completely evaporates, information is destroyed — violating quantum "
            "mechanical unitarity. If information is preserved, it requires a "
            "modification of either QFT in curved spacetime, the no-hair theorem, "
            "or the understanding of black hole evaporation."
        ),
        formal_condition=(
            "Hawking temperature: T_H = ħc³/(8πGM k_B). "
            "Bekenstein-Hawking entropy: S_BH = k_B c³ A/(4Għ). "
            "Unitarity requires: ρ_final = S ρ_initial S† for some S matrix. "
            "But Hawking's calculation gives: ρ_final = Tr_env |ψ⟩⟨ψ| (mixed state). "
            "Page curve: S_vN must follow S_BH(t) for information preservation."
        ),
        energy_scale_gev=None,  # Any black hole mass
        length_scale_m=None,
        failing_theories=["Quantum Field Theory in Curved Spacetime", "General Relativity"],
        reward_solve_bonus=2.5,
        reward_reproduce_penalty=1.0,
        related_theorems=[
            "bekenstein_hawking_entropy",
            "holographic_principle",
            "page_theorem",
        ],
        resolution_candidates=[
            "AdS/CFT: black hole evaporation is unitary in dual CFT",
            "Island formula: entanglement wedge reconstruction",
            "Fuzzball proposal: black hole microstates are horizonless",
            "Firewall: information preserved but horizon is not smooth",
            "Remnant: evaporation stops at Planck scale",
        ],
    ))

    return FailureCoordinateSystem(
        failure_points=points,
        version="0.1.0",
        description=(
            "Standard failure coordinate system encoding known breakdown points in "
            "theoretical physics. Each point represents a specific condition where "
            "GR, QFT, or the Standard Model fails. Candidate structures are evaluated "
            "at these coordinates — resolving failures earns reward; reproducing them "
            "incurs penalty."
        ),
    )


# ---------------------------------------------------------------------------
# YAML serialization
# ---------------------------------------------------------------------------


def failure_coordinates_to_yaml(fcs: FailureCoordinateSystem) -> str:
    """Serialize a FailureCoordinateSystem to YAML string."""
    import yaml

    return yaml.dump(fcs.to_dict(), default_flow_style=False, sort_keys=False)


def failure_coordinates_from_dict(data: dict[str, Any]) -> FailureCoordinateSystem:
    """Deserialize a FailureCoordinateSystem from a dictionary."""
    points = []
    for fp_data in data.get("failure_points", []):
        points.append(FailurePoint(
            name=fp_data["name"],
            regime=FailureRegime(fp_data["regime"]),
            severity=FailureSeverity(fp_data["severity"]),
            description=fp_data.get("description", ""),
            formal_condition=fp_data.get("formal_condition", ""),
            energy_scale_gev=fp_data.get("energy_scale_gev"),
            length_scale_m=fp_data.get("length_scale_m"),
            failing_theories=fp_data.get("failing_theories", []),
            reward_solve_bonus=fp_data.get("reward_solve_bonus", 0.0),
            reward_reproduce_penalty=fp_data.get("reward_reproduce_penalty", 0.0),
            related_theorems=fp_data.get("related_theorems", []),
            resolution_candidates=fp_data.get("resolution_candidates", []),
        ))

    return FailureCoordinateSystem(
        failure_points=points,
        version=data.get("version", "0.1.0"),
        description=data.get("description", ""),
    )


def load_failure_coordinates(
    path: str = "configs/failure_coordinates.yaml",
) -> FailureCoordinateSystem:
    """Load failure coordinates from a YAML file.

    If the file doesn't exist, builds and returns the standard system.
    """
    import os
    import yaml
    from pathlib import Path

    filepath = Path(path)
    if filepath.exists():
        with open(filepath) as f:
            data = yaml.safe_load(f)
        return failure_coordinates_from_dict(data)

    return build_standard_failure_coordinates()
