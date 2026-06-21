"""Temporal gating for explorer training — era-gated discovery tracking.

When training with a historical era cutoff (e.g., `--era pre_relativity` for ≤1904),
the explorer should be rewarded for discovering physics concepts that were unknown
at that time. This incentivizes the architecture to rediscover what physicists
actually found — special relativity, quantum mechanics, GR, etc.

Architecture:
    EraTracker knows what was known at each cutoff year.
    For a given proof, it scans for "future" physics concepts and awards
    discovery bonuses proportional to how far ahead of the era they are.

This is the strongest validation of the theta-core architecture:
    Train on pre-1905 data → does the explorer find special relativity?
    Train on pre-1915 data → does it find GR?
    Train on pre-1925 data → does it find quantum mechanics?

Usage:
    from src.correspondence.era_tracker import EraTracker, ERA_CUTOFFS

    tracker = EraTracker("pre_relativity")  # ≤1904
    discoveries = tracker.scan_proof(proof_text)
    # → ["photoelectric_quantization", "lorentz_transformations", ...]
    bonus = tracker.compute_discovery_bonus(discoveries)
    # → +2.5 for finding 3 future concepts
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# =============================================================================
# Era cutoffs — identical to src/data/physical/constants.py
# =============================================================================

ERA_CUTOFFS: dict[str, int] = {
    "classical": 1860,          # Pre-Maxwell
    "classical_crisis": 1900,   # Pre-Planck/Einstein — blackbody problem unsolved
    "pre_relativity": 1904,     # Pre-special relativity
    "pre_gr": 1914,             # Pre-general relativity
    "old_quantum": 1925,        # Pre-Heisenberg/Schrödinger
    "pre_qed": 1946,            # Pre-Lamb shift, g-2
    "pre_sm": 1965,             # Pre-electroweak, pre-QCD
    "sm_construction": 1975,    # Pre-τ, pre-bottom, pre-W/Z discovery
    "sm_confirmed": 1995,       # Pre-top, pre-neutrino oscillations
    "precision_era": 2010,      # Pre-LHC, pre-Planck
    "modern": 2026,             # Everything
}

ERA_NAMES: list[str] = list(ERA_CUTOFFS.keys())


# =============================================================================
# Discovery catalog — physics concepts organized by discovery era
# =============================================================================

@dataclass
class DiscoveryConcept:
    """A physics concept associated with a historical discovery period.

    Each concept has keywords for detection in proof text, the year it was
    discovered, and a significance weight for bonus computation.
    """

    name: str
    year: int
    keywords: list[str]
    description: str
    significance: float = 1.0  # How important this discovery is (1.0 = baseline)

    def matches(self, text: str) -> bool:
        """Check if any keywords for this concept appear in the text."""
        text_lower = text.lower()
        return any(kw.lower() in text_lower for kw in self.keywords)


# Discovery timeline — what physics was discovered and when.
# Organized so that training with an era cutoff rewards finding concepts
# from subsequent eras.

DISCOVERY_TIMELINE: list[DiscoveryConcept] = [
    # ── 1905–1915: Relativity + Old Quantum ──
    DiscoveryConcept(
        "special_relativity", 1905,
        ["lorentz transformation", "time dilation", "length contraction",
         "relativity of simultaneity", "minkowski spacetime", "e=mc",
         "constancy of light speed", "no aether", "michelson-morley"],
        "Special relativity: spacetime unified, c is invariant",
        significance=2.0,
    ),
    DiscoveryConcept(
        "light_quanta", 1905,
        ["photon", "light quantum", "photoelectric", "quantized light",
         "planck distribution", "blackbody spectrum", "uv catastrophe resolved",
         "energy quantum", "quantum of radiation"],
        "Light quanta (photons): electromagnetic radiation is quantized",
        significance=1.5,
    ),
    DiscoveryConcept(
        "brownian_motion_atomism", 1905,
        ["brownian motion", "atomistic", "molecular kinetics",
         "avogadro number", "statistical fluctuation", "random walk diffusion"],
        "Brownian motion: direct evidence for atoms via statistical mechanics",
        significance=1.0,
    ),
    DiscoveryConcept(
        "general_relativity", 1915,
        ["einstein field equation", "curved spacetime", "geodesic equation",
         "ricci tensor", "riemann curvature", "equivalence principle",
         "gravitational redshift", "perihelion precession", "gravitational lensing",
         "schwarzschild metric"],
        "General relativity: gravity as spacetime curvature",
        significance=2.5,
    ),
    DiscoveryConcept(
        "bohr_model_atom", 1913,
        ["bohr model", "quantized orbit", "rydberg constant", "balmer series",
         "hydrogen spectrum", "atomic energy level", "quantum jump",
         "orbital quantization"],
        "Bohr model: quantized electron orbits explain atomic spectra",
        significance=1.2,
    ),

    # ── 1925–1935: Quantum Mechanics ──
    DiscoveryConcept(
        "quantum_mechanics", 1925,
        ["schrödinger equation", "wave function", "heisenberg uncertainty",
         "matrix mechanics", "quantum state", "probability amplitude",
         "superposition principle", "born rule", "quantum measurement",
         "commutation relation"],
        "Quantum mechanics: wave-particle duality, uncertainty principle",
        significance=3.0,
    ),
    DiscoveryConcept(
        "spin_and_statistics", 1925,
        ["electron spin", "pauli exclusion", "fermi-dirac statistics",
         "bose-einstein statistics", "spin statistics theorem",
         "stern-gerlach", "spinor"],
        "Spin and quantum statistics: fermions vs bosons",
        significance=1.5,
    ),
    DiscoveryConcept(
        "antimatter", 1932,
        ["positron", "antiparticle", "dirac sea", "antimatter",
         "pair production", "dirac equation", "negative energy"],
        "Antimatter: Dirac equation predicts positrons",
        significance=1.5,
    ),
    DiscoveryConcept(
        "quantum_field_theory", 1930,
        ["quantized field", "second quantization", "creation operator",
         "annihilation operator", "fock space", "vacuum fluctuation",
         "quantum electrodynamics", "lamb shift"],
        "Quantum field theory: fields are the fundamental entities",
        significance=2.0,
    ),

    # ── 1945–1965: QED + Nuclear ──
    DiscoveryConcept(
        "renormalization", 1947,
        ["renormalization", "renormalized", "counterterm", "running coupling",
         "beta function", "anomalous dimension", "regularization scheme"],
        "Renormalization: taming infinities in QFT",
        significance=2.0,
    ),
    DiscoveryConcept(
        "electroweak_unification", 1961,
        ["electroweak", "weinberg angle", "spontaneous symmetry breaking",
         "goldstone boson", "higgs mechanism", "w boson", "z boson",
         "neutral current", "su(2)×u(1)"],
        "Electroweak unification: weak and electromagnetic forces unified",
        significance=2.5,
    ),
    DiscoveryConcept(
        "quark_model", 1964,
        ["quark", "hadron", "color charge", "gluon", "quantum chromodynamics",
         "asymptotic freedom", "confinement", "su(3) gauge", "qcd"],
        "Quark model and QCD: strong force from SU(3) gauge theory",
        significance=2.0,
    ),

    # ── 1965–1995: Standard Model ──
    DiscoveryConcept(
        "cp_violation", 1964,
        ["cp violation", "kaon decay", "ckm matrix", "matter antimatter",
         "baryon asymmetry", "sakharov condition", "cp symmetry"],
        "CP violation: matter-antimatter asymmetry",
        significance=1.5,
    ),
    DiscoveryConcept(
        "neutrino_mass", 1998,
        ["neutrino oscillation", "neutrino mass", "atmospheric neutrino",
         "solar neutrino problem", "mixing angle", "majorana mass",
         "see-saw mechanism"],
        "Neutrino mass and oscillations: beyond the Standard Model",
        significance=1.8,
    ),

    # ── 1995–2026: Cosmology + Frontiers ──
    DiscoveryConcept(
        "dark_energy_acceleration", 1998,
        ["dark energy", "cosmic acceleration", "cosmological constant",
         "lambda cdm", "equation of state", "quintessence",
         "supernova cosmology", "vacuum energy density"],
        "Dark energy: accelerating cosmic expansion",
        significance=2.0,
    ),
    DiscoveryConcept(
        "gravitational_waves", 2015,
        ["gravitational wave", "binary black hole", "ligo", "inspiral",
         "ringdown", "chirp mass", "gw150914", "gw170817", "kilonova"],
        "Gravitational waves: direct detection by LIGO",
        significance=1.5,
    ),
    DiscoveryConcept(
        "black_hole_imaging", 2019,
        ["event horizon telescope", "black hole shadow", "photon sphere",
         "eht", "m87", "accretion disk image"],
        "Black hole imaging: Event Horizon Telescope",
        significance=1.0,
    ),
    DiscoveryConcept(
        "hierarchy_problem", 1979,
        ["hierarchy problem", "naturalness", "fine tuning",
         "higgs mass correction", "supersymmetry", "compositeness",
         "extra dimension"],
        "Hierarchy problem: why is the Higgs so light?",
        significance=2.5,
    ),
    DiscoveryConcept(
        "quantum_gravity_frontier", 1980,
        ["quantum gravity", "string theory", "loop quantum gravity",
         "planck scale completion", "holographic principle", "ads/cft",
         "spacetime foam", "non-perturbative gravity"],
        "Quantum gravity: the ultimate frontier",
        significance=3.0,
    ),
]


# =============================================================================
# Era tracker
# =============================================================================


@dataclass
class EraTracker:
    """Tracks discoveries relative to a historical era cutoff.

    Given a cutoff year, classifies physics concepts as:
    - known: discovered before or during the cutoff year
    - discoverable: discovered after the cutoff year (these get bonuses)

    The tracker is used during training to:
    1. Shape rewards: proofs engaging with discoverable concepts get bonuses
    2. Report progress: which future physics is the explorer touching?
    3. Validate architecture: does training on pre-era data produce post-era discoveries?
    """

    era_name: str
    cutoff_year: int

    # Pre-computed concept classifications
    known_concepts: list[DiscoveryConcept] = field(default_factory=list)
    discoverable_concepts: list[DiscoveryConcept] = field(default_factory=list)
    discoverable_by_name: dict[str, DiscoveryConcept] = field(default_factory=dict)

    # Per-era tracking
    discovery_counts: dict[str, int] = field(default_factory=dict)
    total_discoveries: int = 0
    proofs_scanned: int = 0

    def __post_init__(self) -> None:
        # Classify all concepts relative to this era cutoff
        for concept in DISCOVERY_TIMELINE:
            if concept.year <= self.cutoff_year:
                self.known_concepts.append(concept)
            else:
                self.discoverable_concepts.append(concept)
                self.discoverable_by_name[concept.name] = concept
                self.discovery_counts[concept.name] = 0

    # ------------------------------------------------------------------
    # Discovery scanning
    # ------------------------------------------------------------------

    def scan_proof(self, proof_text: str) -> list[str]:
        """Scan a proof for discoverable physics concepts.

        Args:
            proof_text: The generated proof (or theorem statement + proof).

        Returns:
            List of concept names discovered in this proof.
        """
        if not proof_text:
            return []

        discovered = []
        for concept in self.discoverable_concepts:
            if concept.matches(proof_text):
                discovered.append(concept.name)
                self.discovery_counts[concept.name] += 1

        if discovered:
            self.total_discoveries += 1
        self.proofs_scanned += 1

        return discovered

    def scan_batch(self, proofs: list[str], statements: list[str]) -> list[list[str]]:
        """Scan a batch of proofs for era discoveries.

        Only scans the PROOF text, not the theorem statement. The theorem
        statement is the problem — we want to know if the explorer's
        SOLUTION spontaneously uses post-era physics concepts.

        Returns:
            List of discovery lists, one per proof.
        """
        results = []
        for proof in proofs:
            results.append(self.scan_proof(proof))
        return results

    # ------------------------------------------------------------------
    # Discovery bonuses
    # ------------------------------------------------------------------

    def compute_discovery_bonus(self, discovered_concepts: list[str]) -> float:
        """Compute a reward bonus for discovering future physics concepts.

        The bonus scales with:
        - Number of concepts discovered
        - Significance of each concept
        - How far ahead of the era each concept is (temporal distance bonus)

        Args:
            discovered_concepts: List of concept names found in a proof.

        Returns:
            Scalar bonus to add to the proof's reward.
        """
        if not discovered_concepts:
            return 0.0

        bonus = 0.0
        for name in discovered_concepts:
            concept = self.discoverable_by_name.get(name)
            if concept is None:
                continue
            # Base significance
            base = concept.significance * 0.5
            # Temporal distance bonus: farther ahead = bigger bonus
            years_ahead = concept.year - self.cutoff_year
            temporal = min(1.0, years_ahead / 100.0)  # Cap at 100 years
            bonus += base * (1.0 + temporal)

        # Diminishing returns: first discovery is worth most
        return bonus / (1.0 + 0.3 * (len(discovered_concepts) - 1))

    def apply_discovery_bonuses(
        self,
        rewards: "torch.Tensor",
        proofs: list[str],
        statements: list[str],
        bonus_scale: float = 1.0,
    ) -> "torch.Tensor":
        """Apply discovery bonuses to a batch of rewards.

        Args:
            rewards: [batch_size] tensor of base rewards.
            proofs: List of proof strings.
            statements: List of theorem statements.
            bonus_scale: Scaling factor for discovery bonuses.

        Returns:
            Modified rewards tensor with discovery bonuses added.
        """
        import torch

        modified = rewards.clone()
        discoveries = self.scan_batch(proofs, statements)

        for i, disc in enumerate(discoveries):
            if disc:
                bonus = self.compute_discovery_bonus(disc)
                modified[i] = modified[i] + bonus_scale * bonus

        return modified

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_top_discoveries(self, top_n: int = 10) -> list[tuple[str, int, float]]:
        """Return the most-discovered concepts with their significance.

        Returns:
            List of (concept_name, count, significance) sorted by count desc.
        """
        ranked = []
        for name, count in self.discovery_counts.items():
            if count > 0:
                concept = self.discoverable_by_name.get(name)
                sig = concept.significance if concept else 1.0
                ranked.append((name, count, sig))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked[:top_n]

    def get_discovery_rate(self) -> float:
        """Fraction of proofs that touched any discoverable concept."""
        if self.proofs_scanned == 0:
            return 0.0
        return self.total_discoveries / self.proofs_scanned

    def get_era_summary(self) -> str:
        """Human-readable summary of discoveries in this era."""
        top = self.get_top_discoveries(5)
        if not top:
            return "(no discoveries yet)"

        lines = []
        for name, count, sig in top:
            concept = self.discoverable_by_name.get(name)
            year = concept.year if concept else "?"
            lines.append(f"{name} ({year}): {count}× [sig={sig:.1f}]")
        return ", ".join(lines)

    def reset_counts(self) -> None:
        """Reset per-epoch counters (keep era classification)."""
        for key in self.discovery_counts:
            self.discovery_counts[key] = 0
        self.total_discoveries = 0
        self.proofs_scanned = 0


# =============================================================================
# Factory
# =============================================================================


def create_era_tracker(era_name: str) -> EraTracker:
    """Create an EraTracker for the given era name.

    Args:
        era_name: One of the keys in ERA_CUTOFFS (e.g., "pre_relativity").

    Returns:
        EraTracker configured for that era.

    Raises:
        ValueError: If era_name is not recognized.
    """
    if era_name not in ERA_CUTOFFS:
        raise ValueError(
            f"Unknown era '{era_name}'. Choose from: {list(ERA_CUTOFFS.keys())}"
        )

    return EraTracker(
        era_name=era_name,
        cutoff_year=ERA_CUTOFFS[era_name],
    )
