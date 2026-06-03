"""Physical constants and experimental measurements with temporal gating.

Every data point carries a discovery_year for chronology-gated evaluation.
Filter to pre-1905 and ask: does the system discover special relativity?
Filter to pre-1915 and ask: does it find GR?

Sources:
    PDG 2024 (Particle Data Group)      — pdg.lbl.gov
    CODATA 2022 (NIST)                   — physics.nist.gov/cuu/Constants/
    Planck 2018 (cosmology)              — arxiv.org/abs/1807.06209
    NIST ASD (atomic spectra)            — physics.nist.gov/PhysRefData/ASD/
    AME 2020 (nuclear masses)            — www-nds.iaea.org/amdc/
    NuFIT 5.3 (neutrino oscillations)    — nu-fit.org
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# =============================================================================
# Data model
# =============================================================================


class PhysicalDomain(Enum):
    """Top-level physical domain for organizing measurements."""
    FUNDAMENTAL_CONSTANTS = "fundamental_constants"
    PARTICLE_PROPERTIES = "particle_properties"
    ATOMIC_SPECTRA = "atomic_spectra"
    COSMOLOGICAL = "cosmological"
    NUCLEAR = "nuclear"
    NEUTRINO = "neutrino"
    GRAVITATIONAL = "gravitational"
    ELECTROMAGNETIC = "electromagnetic"
    THERMODYNAMIC = "thermodynamic"
    CONDENSED_MATTER = "condensed_matter"
    QUANTUM_FOUNDATIONS = "quantum_foundations"


@dataclass
class PhysicalConstant:
    """A fundamental physical constant with its measured value and uncertainty.

    Attributes:
        name: Human-readable name (e.g., "speed of light in vacuum")
        symbol: Standard symbol (e.g., "c")
        value: Measured value in SI units
        uncertainty: 1σ uncertainty (same units as value)
        units: SI unit string
        discovery_year: Year of first reliable measurement
        refined_year: Year of most recent precision measurement
        source: Authoritative source
        domain: Physical domain classification
        description: What this constant means and why it's important
        relative_uncertainty: Fractional uncertainty (value/uncertainty)
    """

    name: str
    symbol: str
    value: float
    uncertainty: float
    units: str
    discovery_year: int
    refined_year: int = 2024
    source: str = "CODATA 2022"
    domain: PhysicalDomain = PhysicalDomain.FUNDAMENTAL_CONSTANTS
    description: str = ""

    @property
    def relative_uncertainty(self) -> float:
        return abs(self.uncertainty / self.value) if self.value != 0 else float("inf")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "symbol": self.symbol,
            "value": self.value,
            "uncertainty": self.uncertainty,
            "units": self.units,
            "discovery_year": self.discovery_year,
            "source": self.source,
            "domain": self.domain.value,
        }


@dataclass
class ParticleProperty:
    """Properties of a known particle.

    Every Standard Model particle plus key hadrons and resonances.
    """

    name: str                    # e.g., "electron", "Z boson", "Ω⁻"
    symbol: str                  # e.g., "e⁻", "Z⁰", "Ω⁻"
    mass_mev: float              # Mass in MeV/c²
    mass_uncertainty_mev: float  # 1σ mass uncertainty
    charge_e: float              # Electric charge in units of e
    spin: float                  # Spin in units of ħ
    lifetime_s: float | None     # Mean lifetime in seconds (None = stable)
    width_mev: float | None      # Decay width in MeV (None = stable)
    generation: int              # 0 = gauge boson, 1/2/3 = fermion generation
    particle_type: str           # "quark", "lepton", "gauge_boson", "scalar", "hadron"
    discovery_year: int
    source: str = "PDG 2024"
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "symbol": self.symbol,
            "mass_mev": self.mass_mev, "charge_e": self.charge_e,
            "spin": self.spin, "lifetime_s": self.lifetime_s,
            "discovery_year": self.discovery_year,
        }


@dataclass
class SpectralLine:
    """A measured atomic or molecular spectral line."""

    element: str                 # e.g., "H", "He", "Na"
    ionization: int              # 0 = neutral, 1 = singly ionized, etc.
    transition: str              # e.g., "1s-2p", "Ly-α", "Hα"
    wavelength_nm: float         # Wavelength in nanometers
    frequency_hz: float          # Frequency in Hz
    series: str                  # "Lyman", "Balmer", "Paschen", etc.
    discovery_year: int
    source: str = "NIST ASD"
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "element": self.element, "transition": self.transition,
            "wavelength_nm": self.wavelength_nm, "discovery_year": self.discovery_year,
        }


@dataclass
class CosmologicalParameter:
    """A measured cosmological parameter."""

    name: str
    symbol: str
    value: float
    uncertainty: float            # ±1σ
    discovery_year: int
    source: str = "Planck 2018"
    description: str = ""
    depends_on_model: bool = True  # True if value depends on ΛCDM assumption

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "symbol": self.symbol,
            "value": self.value, "uncertainty": self.uncertainty,
            "discovery_year": self.discovery_year,
        }


@dataclass
class NuclearProperty:
    """Nuclear binding energy, magic number, or decay property."""

    isotope: str                  # e.g., "⁴He", "⁵⁶Fe", "²³⁵U"
    z: int                        # Proton number
    n: int                        # Neutron number
    mass_excess_kev: float        # Mass excess in keV
    binding_energy_per_nucleon_kev: float
    half_life_s: float | None     # None = stable
    discovery_year: int
    source: str = "AME 2020"
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "isotope": self.isotope, "z": self.z, "n": self.n,
            "binding_energy_per_nucleon_kev": self.binding_energy_per_nucleon_kev,
            "half_life_s": self.half_life_s, "discovery_year": self.discovery_year,
        }


@dataclass
class AnomalyMeasurement:
    """A measurement that deviates from Standard Model / ΛCDM prediction.

    These are the CURRENT OPEN PROBLEMS — the system should resolve them.
    """

    name: str
    observed_value: str            # Human-readable measured value
    expected_value: str            # Human-readable SM/ΛCDM prediction
    significance_sigma: float      # Tension in σ
    discovery_year: int            # Year anomaly was first noted
    latest_year: int               # Year of most recent measurement
    source: str
    description: str = ""
    is_resolved: bool = False      # True if later measurements closed the anomaly

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "observed_value": self.observed_value,
            "expected_value": self.expected_value,
            "significance_sigma": self.significance_sigma,
            "discovery_year": self.discovery_year,
        }


# =============================================================================
# Era cutoffs for temporal gating
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


# =============================================================================
# Section 1: Fundamental Constants (CODATA 2022)
# =============================================================================

def _build_fundamental_constants() -> list[PhysicalConstant]:
    """Build the registry of fundamental physical constants.

    Values from CODATA 2022 (NIST). Ordered roughly chronologically
    by first measurement.
    """
    return [
        # ---- Speed of light -------------------------------------------------
        PhysicalConstant(
            name="speed of light in vacuum",
            symbol="c",
            value=299792458.0,
            uncertainty=0.0,  # Exact (defined)
            units="m/s",
            discovery_year=1676,
            refined_year=1983,
            description="Finite speed of light first measured by Rømer from Jupiter's moons (1676). "
                        "Fizeau (1849) first terrestrial measurement. Michelson refined through 1920s. "
                        "Defined as exact in 1983 via the meter definition.",
        ),

        # ---- Gravitational constant -----------------------------------------
        PhysicalConstant(
            name="Newtonian gravitational constant",
            symbol="G",
            value=6.67430e-11,
            uncertainty=0.00015e-11,
            units="m³/(kg·s²)",
            discovery_year=1798,
            refined_year=2022,
            description="First measured by Cavendish (1798) using a torsion balance. "
                        "Least precisely known fundamental constant (~2.2×10⁻⁵ relative uncertainty) "
                        "due to the weakness of gravity.",
        ),

        # ---- Planck constant ------------------------------------------------
        PhysicalConstant(
            name="Planck constant",
            symbol="h",
            value=6.62607015e-34,
            uncertainty=0.0,  # Exact (defined)
            units="J·s",
            discovery_year=1900,
            refined_year=2019,
            description="Introduced by Planck (1900) to explain the blackbody spectrum — "
                        "the founding act of quantum mechanics. Defined as exact in 2019.",
        ),
        PhysicalConstant(
            name="reduced Planck constant",
            symbol="ħ",
            value=1.054571817e-34,
            uncertainty=0.0,
            units="J·s",
            discovery_year=1900,
            refined_year=2019,
            description="ħ = h/(2π). The quantum of angular momentum.",
        ),

        # ---- Elementary charge ----------------------------------------------
        PhysicalConstant(
            name="elementary charge",
            symbol="e",
            value=1.602176634e-19,
            uncertainty=0.0,  # Exact (defined)
            units="C",
            discovery_year=1909,
            refined_year=2019,
            description="Charge of the electron/proton. First measured by Millikan's "
                        "oil drop experiment (1909). Defined as exact in 2019.",
        ),

        # ---- Electron mass --------------------------------------------------
        PhysicalConstant(
            name="electron mass",
            symbol="m_e",
            value=9.1093837015e-31,
            uncertainty=0.0000000028e-31,
            units="kg",
            discovery_year=1897,
            refined_year=2022,
            description="J.J. Thomson measured e/m for cathode rays (1897), discovering "
                        "the electron as a particle. Mass itself determined later from e/m + e.",
        ),
        PhysicalConstant(
            name="electron mass (energy units)",
            symbol="m_e c²",
            value=0.51099895000,
            uncertainty=0.00000000015,
            units="MeV",
            discovery_year=1897,
            refined_year=2022,
        ),

        # ---- Proton mass ----------------------------------------------------
        PhysicalConstant(
            name="proton mass",
            symbol="m_p",
            value=1.67262192369e-27,
            uncertainty=0.00000000051e-27,
            units="kg",
            discovery_year=1919,
            refined_year=2022,
            description="Rutherford identified the proton as the hydrogen nucleus (1919). "
                        "Mass measured via mass spectrometry (Aston) and Penning traps.",
        ),
        PhysicalConstant(
            name="proton mass (energy units)",
            symbol="m_p c²",
            value=938.27208816,
            uncertainty=0.00000029,
            units="MeV",
            discovery_year=1919,
            refined_year=2022,
        ),

        # ---- Fine-structure constant ----------------------------------------
        PhysicalConstant(
            name="fine-structure constant",
            symbol="α",
            value=7.2973525693e-3,
            uncertainty=0.0000000011e-3,
            units="dimensionless",
            discovery_year=1916,
            refined_year=2022,
            description="α = e²/(4πε₀ħc) ≈ 1/137.036. Introduced by Sommerfeld (1916) "
                        "to explain fine structure splitting in hydrogen. The fundamental "
                        "coupling constant of electromagnetism.",
        ),

        # ---- Boltzmann constant ---------------------------------------------
        PhysicalConstant(
            name="Boltzmann constant",
            symbol="k_B",
            value=1.380649e-23,
            uncertainty=0.0,  # Exact (defined)
            units="J/K",
            discovery_year=1872,
            refined_year=2019,
            description="Links temperature to energy. Introduced by Boltzmann (1872) in "
                        "statistical mechanics. Defined as exact in 2019.",
        ),

        # ---- Avogadro constant ----------------------------------------------
        PhysicalConstant(
            name="Avogadro constant",
            symbol="N_A",
            value=6.02214076e23,
            uncertainty=0.0,  # Exact (defined)
            units="mol⁻¹",
            discovery_year=1811,
            refined_year=2019,
            description="Number of particles per mole. First proposed by Avogadro (1811). "
                        "Perrin (1909) measured via Brownian motion. Defined as exact in 2019.",
        ),

        # ---- Magnetic constants ---------------------------------------------
        PhysicalConstant(
            name="Bohr magneton",
            symbol="μ_B",
            value=9.2740100783e-24,
            uncertainty=0.0000000028e-24,
            units="J/T",
            discovery_year=1913,
            refined_year=2022,
            description="μ_B = eħ/(2m_e). Natural unit of magnetic moment for electrons.",
        ),
        PhysicalConstant(
            name="nuclear magneton",
            symbol="μ_N",
            value=5.0507837461e-27,
            uncertainty=0.0000000015e-27,
            units="J/T",
            discovery_year=1930,
            refined_year=2022,
            description="μ_N = eħ/(2m_p). Natural unit of magnetic moment for nucleons.",
        ),

        # ---- Rydberg constant -----------------------------------------------
        PhysicalConstant(
            name="Rydberg constant",
            symbol="R_∞",
            value=10973731.568157,
            uncertainty=0.000012,
            units="m⁻¹",
            discovery_year=1888,
            refined_year=2022,
            description="R_∞ = m_e e⁴/(8ε₀²h³c). Central to all hydrogen spectroscopy. "
                        "Rydberg (1888) found the empirical formula; Bohr (1913) derived it.",
        ),

        # ---- Stefan-Boltzmann constant --------------------------------------
        PhysicalConstant(
            name="Stefan-Boltzmann constant",
            symbol="σ",
            value=5.670374419e-8,
            uncertainty=0.000000021e-8,
            units="W/(m²·K⁴)",
            discovery_year=1879,
            refined_year=2022,
            description="σ = 2π⁵k_B⁴/(15h³c²). Total radiated power per unit area from "
                        "a blackbody. Stefan (1879) found T⁴ law empirically; Boltzmann (1884) "
                        "derived it from thermodynamics.",
        ),

        # ---- Vacuum permittivity / permeability -----------------------------
        PhysicalConstant(
            name="vacuum electric permittivity",
            symbol="ε₀",
            value=8.8541878128e-12,
            uncertainty=0.0000000013e-12,
            units="F/m",
            discovery_year=1861,
            refined_year=2022,
            description="ε₀ = 1/(μ₀c²). From Maxwell's equations (1861) — determines the "
                        "strength of the electrostatic force. Related to α, e, ħ, c.",
        ),

        # ---- Planck units (derived, not measured) ---------------------------
        PhysicalConstant(
            name="Planck mass",
            symbol="m_Planck",
            value=2.176434e-8,
            uncertainty=0.000024e-8,
            units="kg",
            discovery_year=1899,
            refined_year=2022,
            description="m_P = √(ħc/G). Mass scale where quantum gravity effects become O(1). "
                        "Planck (1899) introduced these natural units.",
        ),
        PhysicalConstant(
            name="Planck mass (energy units)",
            symbol="m_Planck c²",
            value=1.220890e19,
            uncertainty=0.000014e19,
            units="GeV",
            discovery_year=1899,
            refined_year=2022,
        ),
        PhysicalConstant(
            name="Planck length",
            symbol="l_Planck",
            value=1.616255e-35,
            uncertainty=0.000018e-35,
            units="m",
            discovery_year=1899,
            refined_year=2022,
            description="l_P = √(ħG/c³). Length scale where spacetime is expected to be quantized.",
        ),
        PhysicalConstant(
            name="Planck time",
            symbol="t_Planck",
            value=5.391247e-44,
            uncertainty=0.000060e-44,
            units="s",
            discovery_year=1899,
            refined_year=2022,
            description="t_P = √(ħG/c⁵). Time scale where quantum gravity dominates.",
        ),
    ]


# =============================================================================
# Section 2: Particle Properties (PDG 2024)
# =============================================================================

def _build_particle_properties() -> list[ParticleProperty]:
    """Build the registry of known particles.

    Ordered by discovery year. This is the complete Standard Model particle
    content plus key hadrons for historical completeness.
    """
    return [
        # ---- Gauge bosons ---------------------------------------------------
        ParticleProperty(
            name="photon", symbol="γ",
            mass_mev=0.0, mass_uncertainty_mev=0.0,
            charge_e=0, spin=1, lifetime_s=None, width_mev=None,
            generation=0, particle_type="gauge_boson",
            discovery_year=1905,
            description="Quantum of the electromagnetic field. Einstein's 1905 photoelectric "
                        "effect paper established light quanta.",
        ),
        ParticleProperty(
            name="gluon", symbol="g",
            mass_mev=0.0, mass_uncertainty_mev=0.0,
            charge_e=0, spin=1, lifetime_s=None, width_mev=None,
            generation=0, particle_type="gauge_boson",
            discovery_year=1979,
            description="Gauge boson of QCD. Three-jet events at PETRA (1979) provided "
                        "direct evidence. Carries color charge.",
        ),
        ParticleProperty(
            name="W boson", symbol="W±",
            mass_mev=80360.0, mass_uncertainty_mev=16.0,
            charge_e=1, spin=1, lifetime_s=3.0e-25, width_mev=2085.0,
            generation=0, particle_type="gauge_boson",
            discovery_year=1983,
            description="Charged weak gauge boson. Discovered at UA1/UA2 (CERN, 1983). "
                        "W mass currently shows tension between CDF (80433 MeV) and ATLAS/LHCb.",
        ),
        ParticleProperty(
            name="Z boson", symbol="Z⁰",
            mass_mev=91187.6, mass_uncertainty_mev=2.1,
            charge_e=0, spin=1, lifetime_s=2.6e-25, width_mev=2495.2,
            generation=0, particle_type="gauge_boson",
            discovery_year=1983,
            description="Neutral weak gauge boson. Discovered at UA1/UA2 (CERN, 1983). "
                        "LEP measured its properties to extraordinary precision.",
        ),
        ParticleProperty(
            name="Higgs boson", symbol="H⁰",
            mass_mev=125200.0, mass_uncertainty_mev=110.0,
            charge_e=0, spin=0, lifetime_s=1.6e-22, width_mev=3.7,
            generation=0, particle_type="scalar",
            discovery_year=2012,
            description="Scalar boson of the Higgs mechanism. Discovered by ATLAS + CMS (2012). "
                        "The only fundamental scalar particle known. Width too narrow to measure "
                        "directly; indirect constraint from off-shell production.",
        ),

        # ---- Leptons --------------------------------------------------------
        ParticleProperty(
            name="electron", symbol="e⁻",
            mass_mev=0.51099895000, mass_uncertainty_mev=0.00000000015,
            charge_e=-1, spin=0.5, lifetime_s=None, width_mev=None,
            generation=1, particle_type="lepton",
            discovery_year=1897,
            description="Discovered by J.J. Thomson via cathode ray deflection (1897). "
                        "Stable. Anomalous magnetic moment is the most precisely tested "
                        "prediction in physics (12 significant digits).",
        ),
        ParticleProperty(
            name="electron neutrino", symbol="ν_e",
            mass_mev=0.0, mass_uncertainty_mev=0.0,  # Mass < 0.8 eV, but SM predicts 0
            charge_e=0, spin=0.5, lifetime_s=None, width_mev=None,
            generation=1, particle_type="lepton",
            discovery_year=1956,
            description="Detected by Reines and Cowan (1956) using reactor antineutrinos. "
                        "Now known to have mass < 0.8 eV from KATRIN. Oscillates into ν_μ, ν_τ.",
        ),
        ParticleProperty(
            name="muon", symbol="μ⁻",
            mass_mev=105.6583745, mass_uncertainty_mev=0.0000024,
            charge_e=-1, spin=0.5, lifetime_s=2.1969811e-6, width_mev=2.996e-16,
            generation=2, particle_type="lepton",
            discovery_year=1936,
            description="Discovered by Anderson and Neddermeyer (1936) in cosmic rays. "
                        "Famous 'who ordered that?' quote. g-2 currently shows ~5σ tension.",
        ),
        ParticleProperty(
            name="muon neutrino", symbol="ν_μ",
            mass_mev=0.0, mass_uncertainty_mev=0.0,
            charge_e=0, spin=0.5, lifetime_s=None, width_mev=None,
            generation=2, particle_type="lepton",
            discovery_year=1962,
            description="Discovered at BNL (1962) — first demonstration that ν_μ ≠ ν_e.",
        ),
        ParticleProperty(
            name="tau", symbol="τ⁻",
            mass_mev=1776.86, mass_uncertainty_mev=0.12,
            charge_e=-1, spin=0.5, lifetime_s=2.903e-13, width_mev=2.265e-9,
            generation=3, particle_type="lepton",
            discovery_year=1975,
            description="Discovered by Perl at SLAC (1975). The only lepton heavy enough "
                        "to decay hadronically. Third generation completes the SM.",
        ),
        ParticleProperty(
            name="tau neutrino", symbol="ν_τ",
            mass_mev=0.0, mass_uncertainty_mev=0.0,
            charge_e=0, spin=0.5, lifetime_s=None, width_mev=None,
            generation=3, particle_type="lepton",
            discovery_year=2000,
            description="Directly detected by DONUT at Fermilab (2000). 4 events. "
                        "Last Standard Model particle discovered before the Higgs.",
        ),

        # ---- Quarks ---------------------------------------------------------
        ParticleProperty(
            name="up quark", symbol="u",
            mass_mev=2.16, mass_uncertainty_mev=0.49,
            charge_e=2/3, spin=0.5, lifetime_s=None, width_mev=None,
            generation=1, particle_type="quark",
            discovery_year=1968,
            description="Inferred from deep inelastic scattering at SLAC (1968). "
                        "Constituent of protons (uud) and neutrons (udd). Confined — "
                        "never observed as a free particle.",
        ),
        ParticleProperty(
            name="down quark", symbol="d",
            mass_mev=4.67, mass_uncertainty_mev=0.48,
            charge_e=-1/3, spin=0.5, lifetime_s=None, width_mev=None,
            generation=1, particle_type="quark",
            discovery_year=1968,
        ),
        ParticleProperty(
            name="strange quark", symbol="s",
            mass_mev=93.4, mass_uncertainty_mev=8.6,
            charge_e=-1/3, spin=0.5, lifetime_s=None, width_mev=None,
            generation=2, particle_type="quark",
            discovery_year=1947,
            description="'Strangeness' discovered in cosmic ray events (1947) — "
                        "particles produced strongly but decaying weakly. Gell-Mann "
                        "and Nishijima explained this with a new quantum number (1953).",
        ),
        ParticleProperty(
            name="charm quark", symbol="c",
            mass_mev=1270.0, mass_uncertainty_mev=20.0,
            charge_e=2/3, spin=0.5, lifetime_s=None, width_mev=None,
            generation=2, particle_type="quark",
            discovery_year=1974,
            description="Discovered simultaneously at BNL (Ting) and SLAC (Richter) as "
                        "the J/ψ meson (1974). The 'November Revolution' — confirmed "
                        "the GIM mechanism and the quark model.",
        ),
        ParticleProperty(
            name="bottom quark", symbol="b",
            mass_mev=4180.0, mass_uncertainty_mev=30.0,
            charge_e=-1/3, spin=0.5, lifetime_s=None, width_mev=None,
            generation=3, particle_type="quark",
            discovery_year=1977,
            description="Discovered at Fermilab (Lederman, 1977) via the Υ meson (bḃ bound state). "
                        "Third generation confirmed. Long lifetime enabled B factories.",
        ),
        ParticleProperty(
            name="top quark", symbol="t",
            mass_mev=172500.0, mass_uncertainty_mev=700.0,
            charge_e=2/3, spin=0.5, lifetime_s=5e-25, width_mev=1320.0,
            generation=3, particle_type="quark",
            discovery_year=1995,
            description="Discovered at the Tevatron (CDF + DØ, 1995). Heaviest known "
                        "fundamental particle — mass ~184× the proton. Lifetime too short "
                        "to form hadrons (decays before hadronization).",
        ),

        # ---- Key light hadrons (historical importance) ----------------------
        ParticleProperty(
            name="pion (±)", symbol="π±",
            mass_mev=139.57039, mass_uncertainty_mev=0.00018,
            charge_e=1, spin=0, lifetime_s=2.6033e-8, width_mev=2.5284e-14,
            generation=0, particle_type="hadron",
            discovery_year=1947,
            description="Yukawa's predicted meson (1935), discovered by Powell (1947) "
                        "in cosmic ray emulsion. Lightest hadron — mediates the long-range "
                        "nucleon-nucleon force.",
        ),
        ParticleProperty(
            name="pion (0)", symbol="π⁰",
            mass_mev=134.9768, mass_uncertainty_mev=0.0005,
            charge_e=0, spin=0, lifetime_s=8.52e-17, width_mev=7.73e-6,
            generation=0, particle_type="hadron",
            discovery_year=1950,
        ),
        ParticleProperty(
            name="kaon (±)", symbol="K±",
            mass_mev=493.677, mass_uncertainty_mev=0.016,
            charge_e=1, spin=0, lifetime_s=1.2380e-8, width_mev=5.317e-14,
            generation=0, particle_type="hadron",
            discovery_year=1947,
            description="First strange particle discovered. K_L → π⁺π⁻ decay revealed "
                        "CP violation (Cronin, Fitch, 1964).",
        ),
        ParticleProperty(
            name="proton", symbol="p",
            mass_mev=938.27208816, mass_uncertainty_mev=0.00000029,
            charge_e=1, spin=0.5, lifetime_s=None, width_mev=None,
            generation=0, particle_type="hadron",
            discovery_year=1919,
            description="Rutherford identified the hydrogen nucleus as a fundamental "
                        "constituent (1919). Stable to >10³⁴ years (Super-K limit).",
        ),
        ParticleProperty(
            name="neutron", symbol="n",
            mass_mev=939.56542052, mass_uncertainty_mev=0.00000054,
            charge_e=0, spin=0.5, lifetime_s=878.4, width_mev=7.49e-28,
            generation=0, particle_type="hadron",
            discovery_year=1932,
            description="Discovered by Chadwick (1932). Stable inside nuclei, decays via "
                        "beta decay when free: n → p + e⁻ + ν̄_e. Mass slightly larger "
                        "than proton — the reason free neutrons decay.",
        ),
        ParticleProperty(
            name="omega minus", symbol="Ω⁻",
            mass_mev=1672.45, mass_uncertainty_mev=0.29,
            charge_e=-1, spin=1.5, lifetime_s=8.21e-11, width_mev=8.01e-12,
            generation=0, particle_type="hadron",
            discovery_year=1964,
            description="Predicted by Gell-Mann's SU(3) quark model before discovery. "
                        "Detection at BNL (1964) was a dramatic confirmation of the "
                        "eightfold way and quark classification.",
        ),
    ]


# =============================================================================
# Section 3: Atomic Spectral Lines (NIST ASD)
# =============================================================================

def _build_spectral_lines() -> list[SpectralLine]:
    """Build the registry of key atomic spectral lines.

    Focus on hydrogen (most fundamental) and key historical lines.
    Full NIST ASD tables have >100,000 lines; we select the ~50 most
    important for physical theory validation.
    """
    c_ms = 299792458.0  # Exact

    lines = [
        # ---- Hydrogen: Lyman series (UV) ------------------------------------
        SpectralLine("H", 0, "Ly-α  (1s-2p)",  121.5668, c_ms/(121.5668e-9), "Lyman",
                     discovery_year=1906,
                     description="First Lyman line. Discovered by Lyman (1906-1914)."),
        SpectralLine("H", 0, "Ly-β  (1s-3p)",  102.5722, c_ms/(102.5722e-9), "Lyman",
                     discovery_year=1906),
        SpectralLine("H", 0, "Ly-γ  (1s-4p)",   97.2537, c_ms/(97.2537e-9), "Lyman", 1906),
        SpectralLine("H", 0, "Ly-limit (1s-∞)",  91.1753, c_ms/(91.1753e-9), "Lyman", 1906),

        # ---- Hydrogen: Balmer series (visible) — the most historically important ----
        SpectralLine("H", 0, "Hα  (2p-3d)",  656.279, c_ms/(656.279e-9), "Balmer",
                     discovery_year=1885,
                     description="First Balmer line. Balmer (1885) found the empirical "
                                 "formula λ = 364.56 × n²/(n²-4) nm. This was the "
                                 "Rosetta Stone for atomic theory."),
        SpectralLine("H", 0, "Hβ  (2p-4d)",  486.135, c_ms/(486.135e-9), "Balmer", 1885),
        SpectralLine("H", 0, "Hγ  (2p-5d)",  434.0472, c_ms/(434.0472e-9), "Balmer", 1885),
        SpectralLine("H", 0, "Hδ  (2p-6d)",  410.1734, c_ms/(410.1734e-9), "Balmer", 1885),

        # ---- Hydrogen: Paschen series (IR) ----------------------------------
        SpectralLine("H", 0, "Pα  (3-4)", 1875.10, c_ms/(1875.10e-9), "Paschen",
                     discovery_year=1908,
                     description="Paschen series discovered 1908."),
        SpectralLine("H", 0, "Pβ  (3-5)", 1281.81, c_ms/(1281.81e-9), "Paschen", 1908),

        # ---- Hydrogen: Fine structure ---------------------------------------
        SpectralLine("H", 0, "Hα fine: 2p_3/2 → 3d_5/2", 656.272, c_ms/(656.272e-9),
                     "Balmer-fine", discovery_year=1916,
                     description="Fine structure splitting — key evidence for spin and "
                                 "Sommerfeld's relativistic correction. α emerges from the splitting."),
        SpectralLine("H", 0, "Hα fine: 2s_1/2 → 3p_3/2", 656.272, c_ms/(656.272e-9),
                     "Balmer-fine", 1916),

        # ---- Hydrogen: 21 cm hyperfine line ---------------------------------
        SpectralLine("H", 0, "21 cm hyperfine (F=1→F=0)", 2.1026611e8, 1420.40575177e6,
                     "hyperfine", discovery_year=1951,
                     description="Hyperfine transition of neutral hydrogen. Detected by "
                                 "Ewen and Purcell (1951). Predicted by Van de Hulst (1944). "
                                 "Fundamental to radio astronomy. Wavelength = 21.106 cm."),

        # ---- Helium ---------------------------------------------------------
        SpectralLine("He", 0, "He I  (1s2p ¹P→1s² ¹S)", 58.4334,
                     c_ms/(58.4334e-9), "principal", discovery_year=1868,
                     description="Helium discovered in solar spectrum (Janssen/Lockyer, 1868) "
                                 "before it was found on Earth."),
        SpectralLine("He", 0, "He I 587.6 nm (1s3d→1s2p)", 587.562,
                     c_ms/(587.562e-9), "visible", 1868),

        # ---- Sodium (historically important doublet) ------------------------
        SpectralLine("Na", 0, "Na D₁ (3p_1/2 → 3s_1/2)", 589.592,
                     c_ms/(589.592e-9), "principal", discovery_year=1814,
                     description="Fraunhofer D lines. One of the earliest spectral lines "
                                 "catalogued. The doublet separation reveals spin-orbit coupling."),
        SpectralLine("Na", 0, "Na D₂ (3p_3/2 → 3s_1/2)", 588.995,
                     c_ms/(588.995e-9), "principal", 1814),

        # ---- Mercury (historically important) -------------------------------
        SpectralLine("Hg", 0, "Hg 253.7 nm (6p→6s)", 253.652,
                     c_ms/(253.652e-9), "principal", discovery_year=1860,
                     description="Key line in early spectroscopy and Franck-Hertz experiment."),
    ]

    return lines


# =============================================================================
# Section 4: Cosmological Parameters (Planck 2018)
# =============================================================================

def _build_cosmological_parameters() -> list[CosmologicalParameter]:
    """Build the registry of measured cosmological parameters.

    Primarily from Planck 2018 (TT,TE,EE + lowE + lensing).
    Earlier measurements (WMAP, COBE, Penzias-Wilson) noted where historically
    important.
    """
    return [
        # ---- Hubble constant (CMB-derived) ----------------------------------
        CosmologicalParameter(
            name="Hubble constant (Planck, ΛCDM)",
            symbol="H₀",
            value=67.36,
            uncertainty=0.54,
            discovery_year=1929,
            source="Planck 2018",
            description="Hubble discovered cosmic expansion (1929) with H₀ ≈ 500 km/s/Mpc. "
                        "Modern CMB value is 67.4 km/s/Mpc. Tensions with local measurements "
                        "(SH0ES: H₀ = 73.0 ± 1.0) at 5σ — a major open problem.",
        ),
        CosmologicalParameter(
            name="Hubble constant (SH0ES, local distance ladder)",
            symbol="H₀ (local)",
            value=73.04,
            uncertainty=1.04,
            discovery_year=2001,
            source="Riess et al. 2022 (SH0ES)",
            description="Local measurement from Cepheid-calibrated SNIa. 5σ tension with "
                        "Planck CMB value. If real, requires new physics beyond ΛCDM.",
        ),

        # ---- Density parameters ---------------------------------------------
        CosmologicalParameter(
            name="total matter density parameter",
            symbol="Ω_m",
            value=0.3153,
            uncertainty=0.0073,
            discovery_year=1998,
            source="Planck 2018",
            description="Ω_m = Ω_b + Ω_CDM ≈ 0.315. Total matter fraction of the "
                        "critical density.",
        ),
        CosmologicalParameter(
            name="baryon density parameter",
            symbol="Ω_b h²",
            value=0.02237,
            uncertainty=0.00015,
            discovery_year=1998,
            source="Planck 2018",
            description="Baryon density from CMB acoustic peaks + BBN. Consistent with "
                        "primordial deuterium abundance. Tight constraint on baryonic matter.",
        ),
        CosmologicalParameter(
            name="cold dark matter density parameter",
            symbol="Ω_c h²",
            value=0.1200,
            uncertainty=0.0012,
            discovery_year=1998,
            source="Planck 2018",
            description="Cold dark matter density. ~5× the baryon density. The particle "
                        "nature remains unknown.",
        ),
        CosmologicalParameter(
            name="dark energy density parameter",
            symbol="Ω_Λ",
            value=0.6847,
            uncertainty=0.0073,
            discovery_year=1998,
            source="Planck 2018",
            description="Dark energy dominates the current energy budget (~68%). "
                        "The cosmological constant problem: ρ_Λ(obs) ≈ 10⁻⁴⁷ GeV⁴, "
                        "ρ_vac(QFT) ≈ 10⁷⁶ GeV⁴. 120 orders of magnitude.",
        ),

        # ---- Primordial power spectrum --------------------------------------
        CosmologicalParameter(
            name="scalar spectral index",
            symbol="n_s",
            value=0.9649,
            uncertainty=0.0042,
            discovery_year=2001,
            source="Planck 2018",
            description="Scale dependence of primordial scalar fluctuations. "
                        "n_s < 1 (slightly red) at >8σ — rules out exact scale invariance. "
                        "Predicted by slow-roll inflation.",
        ),
        CosmologicalParameter(
            name="amplitude of scalar fluctuations",
            symbol="ln(10¹⁰ A_s)",
            value=3.044,
            uncertainty=0.014,
            discovery_year=2001,
            source="Planck 2018",
        ),

        # ---- Reionization ---------------------------------------------------
        CosmologicalParameter(
            name="reionization optical depth",
            symbol="τ",
            value=0.0544,
            uncertainty=0.0073,
            discovery_year=2003,
            source="Planck 2018",
            description="Optical depth to reionization. First stars turned on at z ~ 7.7. "
                        "Measured from CMB polarization at large angular scales.",
        ),

        # ---- Matter fluctuation amplitude -----------------------------------
        CosmologicalParameter(
            name="RMS matter fluctuation at 8 Mpc/h",
            symbol="σ_8",
            value=0.8111,
            uncertainty=0.0060,
            discovery_year=2000,
            source="Planck 2018",
            description="Normalization of the matter power spectrum. Tension with weak "
                        "lensing surveys (DES, KiDS measure lower σ_8). The S_8 tension.",
        ),
        CosmologicalParameter(
            name="S_8 parameter (CMB)",
            symbol="S_8 = σ_8(Ω_m/0.3)^0.5",
            value=0.832,
            uncertainty=0.013,
            discovery_year=2015,
            source="Planck 2018",
            description="S_8 from CMB = 0.832. Weak lensing surveys typically measure "
                        "S_8 ≈ 0.76-0.78. 2-3σ tension.",
        ),

        # ---- CMB temperature ------------------------------------------------
        CosmologicalParameter(
            name="CMB temperature today",
            symbol="T₀",
            value=2.72548,
            uncertainty=0.00057,
            discovery_year=1965,
            source="COBE FIRAS (Fixsen et al. 2009)",
            description="Penzias and Wilson (1965) discovered the CMB at ~3.5 K. "
                        "COBE FIRAS measured a perfect blackbody at 2.725 K — residuals "
                        "< 50 ppm of the peak. The most perfect blackbody in nature.",
            depends_on_model=False,
        ),

        # ---- Baryon acoustic oscillation scale ------------------------------
        CosmologicalParameter(
            name="sound horizon at drag epoch",
            symbol="r_drag",
            value=147.09,
            uncertainty=0.26,
            discovery_year=2005,
            source="Planck 2018",
            description="BAO standard ruler — the distance sound waves traveled in "
                        "the primordial plasma before recombination. Imprinted as a "
                        "preferred 150 Mpc separation of galaxies.",
        ),

        # ---- Age of universe ------------------------------------------------
        CosmologicalParameter(
            name="age of the universe",
            symbol="t₀",
            value=13.797,
            uncertainty=0.023,
            discovery_year=2003,
            source="Planck 2018",
            description="Age in billions of years. Consistent with globular cluster ages "
                        "and white dwarf cooling ages. In ΛCDM.",
        ),
    ]


# =============================================================================
# Section 5: Nuclear Properties (AME 2020)
# =============================================================================

def _build_nuclear_properties() -> list[NuclearProperty]:
    """Key nuclear properties: binding energies for BBN nuclei and notable isotopes."""
    return [
        # Light nuclei (BBN-relevant)
        NuclearProperty("²H", 1, 1, 13135.7, 1112.3, None, 1931,
                        description="Deuteron. Binding energy 2.22 MeV. Fragile — "
                                    "no bound excited states. BBN abundance sensitive to η."),
        NuclearProperty("³H", 1, 2, 14949.8, 2827.3, 3.888e8, 1934,
                        description="Triton. Beta decays to ³He with 12.33 year half-life."),
        NuclearProperty("³He", 2, 1, 14931.2, 2572.7, None, 1934),
        NuclearProperty("⁴He", 2, 2, 2424.9, 7073.9, None, 1868,
                        description="Alpha particle. Most tightly bound light nucleus. "
                                    "~25% of baryonic mass produced in BBN."),
        NuclearProperty("⁷Li", 3, 4, 14908.0, 5606.3, None, 1932,
                        description="BBN predicts ⁷Li/H ~ 5×10⁻¹⁰ — but observed in "
                                    "metal-poor stars is ~3× lower. The 'lithium problem.'"),
        NuclearProperty("⁷Be", 4, 3, 15769.0, 5373.5, 4.6e6, 1932),

        # Mid-weight nuclei (binding energy curve)
        NuclearProperty("¹²C", 6, 6, 0.0, 7680.1, None, 1919,
                        description="Carbon-12. Atomic mass standard (by definition: 12 u). "
                                    "Produced by triple-alpha process in stars. Hoyle state "
                                    "at 7.65 MeV enables resonant production."),
        NuclearProperty("¹⁶O", 8, 8, -4737.0, 7982.2, None, 1919),
        NuclearProperty("²⁰Ne", 10, 10, -7041.9, 8033.5, None, 1919),
        NuclearProperty("²⁴Mg", 12, 12, -13933.1, 8359.6, None, 1919),
        NuclearProperty("⁴⁰Ca", 20, 20, -34847.0, 8567.5, None, 1920),
        NuclearProperty("⁵⁶Fe", 26, 30, -60605.2, 8790.4, None, 1920,
                        description="Iron-56. Near the peak of the binding energy curve. "
                                    "Most stable nucleus — no energy from fusion or fission. "
                                    "This is why stellar cores end as iron."),
        NuclearProperty("⁶²Ni", 28, 34, -66741.0, 8794.6, None, 1948,
                        description="Nickel-62. Actually the most tightly bound nucleus "
                                    "(highest binding energy per nucleon), though ⁵⁶Fe "
                                    "is more abundant due to nuclear reaction pathways."),

        # Heavy nuclei (fission-relevant)
        NuclearProperty("²³⁵U", 92, 143, 40918.0, 7601.1, 2.22e16, 1935,
                        description="Uranium-235. Fissile with thermal neutrons. "
                                    "Half-life 704 million years. Primary fuel for "
                                    "nuclear reactors and weapons."),
        NuclearProperty("²³⁸U", 92, 146, 47307.0, 7584.2, 1.41e17, 1935,
                        description="Uranium-238. Fertile (breeds to ²³⁹Pu). "
                                    "Half-life 4.47 billion years — used for dating."),

        # Magic number nuclei
        NuclearProperty("⁴⁸Ca", 20, 28, -44222.0, 8690.3, 2.0e27, 1949,
                        description="Doubly magic (Z=20, N=28). Anomalously long "
                                    "half-life for a neutron-rich nucleus. Tests of "
                                    "shell model."),
        NuclearProperty("²⁰⁸Pb", 82, 126, -21750.0, 7876.2, None, 1920,
                        description="Lead-208. Heaviest stable nucleus. Doubly magic "
                                    "(Z=82, N=126). Endpoint of many radioactive decay chains."),
    ]


# =============================================================================
# Section 6: Neutrino Oscillation Parameters (NuFIT 5.3)
# =============================================================================

def _build_neutrino_parameters() -> list[PhysicalConstant]:
    """Neutrino oscillation parameters. Ordered by discovery.

    These are the only direct evidence for physics beyond the Standard Model
    (neutrino masses require either right-handed neutrinos, a seesaw mechanism,
    or loop-level mass generation).
    """
    return [
        PhysicalConstant(
            name="solar mass splitting",
            symbol="Δm²_21",
            value=7.41e-5,
            uncertainty=0.21e-5,
            units="eV²",
            discovery_year=2002,
            refined_year=2023,
            source="NuFIT 5.3",
            domain=PhysicalDomain.NEUTRINO,
            description="Mass-squared difference between ν₂ and ν₁. Measured by "
                        "SNO + KamLAND from solar neutrino flavor transformation.",
        ),
        PhysicalConstant(
            name="atmospheric mass splitting (normal ordering)",
            symbol="|Δm²_31|",
            value=2.511e-3,
            uncertainty=0.027e-3,
            units="eV²",
            discovery_year=1998,
            refined_year=2023,
            source="NuFIT 5.3",
            domain=PhysicalDomain.NEUTRINO,
            description="Mass-squared difference between ν₃ and ν₁. Measured by "
                        "Super-Kamiokande atmospheric neutrinos (1998) and confirmed "
                        "by accelerator (MINOS, T2K) and reactor experiments.",
        ),
        PhysicalConstant(
            name="solar mixing angle",
            symbol="sin² θ_12",
            value=0.307,
            uncertainty=0.013,
            units="dimensionless",
            discovery_year=2002,
            refined_year=2023,
            source="NuFIT 5.3",
            domain=PhysicalDomain.NEUTRINO,
            description="Mixing angle between ν_e and the ν₂ mass eigenstate. "
                        "Large (~33° — unlike quark mixing which is small).",
        ),
        PhysicalConstant(
            name="atmospheric mixing angle",
            symbol="sin² θ_23",
            value=0.545,
            uncertainty=0.021,
            units="dimensionless",
            discovery_year=1998,
            refined_year=2023,
            source="NuFIT 5.3",
            domain=PhysicalDomain.NEUTRINO,
            description="Mixing angle for ν_μ→ν_τ transitions. Near-maximal (~47°). "
                        "Whether θ_23 < 45° or > 45° (octant degeneracy) is unknown.",
        ),
        PhysicalConstant(
            name="reactor mixing angle",
            symbol="sin² θ_13",
            value=0.02203,
            uncertainty=0.00058,
            units="dimensionless",
            discovery_year=2012,
            refined_year=2023,
            source="NuFIT 5.3",
            domain=PhysicalDomain.NEUTRINO,
            description="Measured by Daya Bay, RENO, Double Chooz (2012). "
                        "Non-zero at >5σ — essential for CP violation in the lepton sector. "
                        "Surprisingly large (~8.5°).",
        ),
        PhysicalConstant(
            name="CP-violating phase (normal ordering)",
            symbol="δ_CP",
            value=232.0,
            uncertainty=36.0,
            units="degrees",
            discovery_year=2020,
            refined_year=2023,
            source="NuFIT 5.3",
            domain=PhysicalDomain.NEUTRINO,
            description="Dirac CP phase in PMNS matrix. Early hints of δ_CP ~ 270° "
                        "(near-maximal CP violation). T2K and NOvA tension. DUNE and "
                        "Hyper-Kamiokande aim to measure this definitively.",
        ),
        PhysicalConstant(
            name="effective Majorana mass upper limit",
            symbol="m_ββ",
            value=0.036,
            uncertainty=0.0,  # Upper limit
            units="eV",
            discovery_year=2023,
            source="KamLAND-Zen 800",
            domain=PhysicalDomain.NEUTRINO,
            description="Upper limit on neutrinoless double beta decay effective mass. "
                        "If observed, would prove neutrinos are Majorana particles — "
                        "their own antiparticles. KamLAND-Zen 800: m_ββ < 0.036-0.156 eV "
                        "(range from nuclear matrix element uncertainty).",
        ),
        PhysicalConstant(
            name="direct neutrino mass limit (tritium endpoint)",
            symbol="m_ν (effective)",
            value=0.8,
            uncertainty=0.0,  # Upper limit
            units="eV",
            discovery_year=2022,
            source="KATRIN",
            domain=PhysicalDomain.NEUTRINO,
            description="KATRIN 2022: m_ν < 0.8 eV (90% CL) from tritium beta decay "
                        "endpoint. Final sensitivity goal: 0.2 eV.",
        ),
    ]


# =============================================================================
# Section 7: Anomalies and Tensions (current open problems)
# =============================================================================

def _build_anomalies() -> list[AnomalyMeasurement]:
    """Current experimental anomalies — the open problems.

    These are the strongest signals that the Standard Model + ΛCDM is incomplete.
    A successful structure should RESOLVE these, not reproduce them.
    """
    return [
        AnomalyMeasurement(
            name="Hubble tension (SH0ES vs Planck)",
            observed_value="H₀ = 73.04 ± 1.04 km/s/Mpc (local, SH0ES 2022)",
            expected_value="H₀ = 67.36 ± 0.54 km/s/Mpc (CMB, Planck 2018)",
            significance_sigma=5.0,
            discovery_year=2016,
            latest_year=2022,
            source="Riess et al. 2022; Planck 2018",
            description="The most significant tension in modern cosmology. Local "
                        "distance ladder measurements consistently give higher H₀ "
                        "than the CMB-inferred value. If real, requires new physics "
                        "(early dark energy, extra relativistic species, modified gravity).",
        ),
        AnomalyMeasurement(
            name="S_8 tension (weak lensing vs CMB)",
            observed_value="S_8 = 0.766 ± 0.020 (KiDS-1000); 0.759 ± 0.025 (DES Y3)",
            expected_value="S_8 = 0.832 ± 0.013 (Planck 2018)",
            significance_sigma=2.5,
            discovery_year=2018,
            latest_year=2023,
            source="KiDS-1000, DES Y3, Planck 2018",
            description="Weak lensing surveys measure lower matter clustering than "
                        "predicted by Planck ΛCDM. Lower significance than H₀ tension "
                        "but consistent across multiple independent surveys.",
        ),
        AnomalyMeasurement(
            name="muon anomalous magnetic moment (g-2)",
            observed_value="a_μ(exp) = 116592059.0 ± 22.0 × 10⁻¹¹ (FNAL 2023)",
            expected_value="a_μ(SM) = 116591810.0 ± 43.0 × 10⁻¹¹ (WP 2020)",
            significance_sigma=5.1,
            discovery_year=2001,
            latest_year=2023,
            source="Muon g-2 Collaboration (FNAL); Theory Initiative White Paper 2020",
            description="The muon g-2 has been anomalous since BNL E821 (2001). "
                        "FNAL Run 1-3 confirms the anomaly. Significance depends on "
                        "the theory prediction — lattice QCD results may reduce the tension. "
                        "If real: evidence for new particles at the TeV scale.",
        ),
        AnomalyMeasurement(
            name="W boson mass (CDF 2022)",
            observed_value="m_W = 80433.5 ± 9.4 MeV (CDF 2022)",
            expected_value="m_W = 80357 ± 6 MeV (SM global fit)",
            significance_sigma=7.0,
            discovery_year=2022,
            latest_year=2022,
            source="CDF Collaboration (Science 2022); PDG 2024",
            description="CDF's high-precision W mass measurement is 7σ above the SM. "
                        "However: ATLAS (80360 ± 16 MeV) and LHCb are consistent with SM. "
                        "Likely a systematic effect in the CDF measurement, but unresolved.",
        ),
        AnomalyMeasurement(
            name="lithium problem (BBN ⁷Li abundance)",
            observed_value="⁷Li/H = (1.6 ± 0.3)×10⁻¹⁰ (metal-poor halo stars)",
            expected_value="⁷Li/H = (5.2 ± 0.7)×10⁻¹⁰ (BBN prediction from CMB η)",
            significance_sigma=5.0,
            discovery_year=1982,
            latest_year=2020,
            source="Sbordone et al. 2010; Planck 2018 BBN",
            description="BBN predicts ~3× more lithium-7 than observed in the oldest "
                        "stars. Could be astrophysical (stellar depletion) or new physics "
                        "(modified BBN, decaying particles).",
        ),
        AnomalyMeasurement(
            name="neutron lifetime puzzle (beam vs bottle)",
            observed_value="τ_n = 888.0 ± 2.0 s (bottle); 877.7 ± 0.7 s (beam)",
            expected_value="Should agree — same physics",
            significance_sigma=4.0,
            discovery_year=2005,
            latest_year=2021,
            source="UCNτ collaboration; PDG 2024",
            description="Two measurement methods disagree by ~10 seconds (4σ). "
                        "Bottle: count surviving neutrons. Beam: count decay protons. "
                        "Could indicate an unobserved neutron decay channel.",
        ),
        AnomalyMeasurement(
            name="XENON1T electronic recoil excess",
            observed_value="Excess at 2.4 ± 0.1 keV (285 events vs 232 expected)",
            expected_value="Background-only hypothesis",
            significance_sigma=3.3,
            discovery_year=2020,
            latest_year=2020,
            source="XENON1T Collaboration (2020)",
            description="Low-energy electronic recoil excess. Could be solar axions, "
                        "a neutrino magnetic moment, or tritium contamination. "
                        "XENONnT will resolve.",
            is_resolved=False,
        ),
        AnomalyMeasurement(
            name="Cosmological constant problem",
            observed_value="ρ_Λ ≈ 10⁻⁴⁷ GeV⁴ (observed dark energy density)",
            expected_value="ρ_vac ≈ 10⁷⁶ GeV⁴ (QFT vacuum energy, Λ_UV = M_Planck)",
            significance_sigma=120.0,  # Orders of magnitude, not standard deviations
            discovery_year=1998,
            latest_year=2026,
            source="Weinberg 1989; Martin 2012",
            description="The largest quantitative mismatch in all of physics. "
                        "Not a statistical tension — a 120-order-of-magnitude failure "
                        "of QFT vacuum energy prediction. Any theory of quantum gravity "
                        "must explain why Λ is small but non-zero.",
        ),
    ]


# =============================================================================
# Section 8: CKM Matrix Elements (PDG 2024)
# =============================================================================

@dataclass
class FlavorPhysics:
    """A flavor physics measurement — CKM/PMNS element or CPV observable."""
    name: str
    symbol: str
    value: float
    uncertainty: float
    units: str = "dimensionless"
    discovery_year: int = 2000
    source: str = "PDG 2024"
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "symbol": self.symbol,
            "value": self.value, "uncertainty": self.uncertainty,
            "discovery_year": self.discovery_year,
        }


def _build_flavor_physics() -> list[FlavorPhysics]:
    """CKM matrix magnitudes, CP violation parameters, and key mixing observables."""
    return [
        # ---- CKM matrix magnitudes (PDG 2024) --------------------------------
        FlavorPhysics("CKM |V_ud|", "|V_ud|", 0.97373, 0.00031, discovery_year=1963,
                      description="Up-down coupling. Most precisely measured CKM element. "
                                  "From superallowed nuclear beta decays."),
        FlavorPhysics("CKM |V_us|", "|V_us|", 0.2243, 0.0008, discovery_year=1964,
                      description="Up-strange coupling. From kaon semileptonic decays."),
        FlavorPhysics("CKM |V_ub|", "|V_ub|", 0.00382, 0.00020, discovery_year=1990,
                      description="Up-bottom coupling. Small — explains long B lifetime. "
                                  "From B→πℓν and B→X_uℓν."),
        FlavorPhysics("CKM |V_cd|", "|V_cd|", 0.221, 0.004, discovery_year=1976,
                      description="Charm-down coupling. From neutrino dimuon production."),
        FlavorPhysics("CKM |V_cs|", "|V_cs|", 0.975, 0.006, discovery_year=1980,
                      description="Charm-strange coupling. From D→Kℓν and W→cs̄."),
        FlavorPhysics("CKM |V_cb|", "|V_cb|", 0.0408, 0.0014, discovery_year=1990,
                      description="Charm-bottom coupling. From B→D*ℓν."),
        FlavorPhysics("CKM |V_td|", "|V_td|", 0.0086, 0.0002, discovery_year=2006,
                      description="Top-down coupling. Smallest measured CKM element. "
                                  "From B⁰-B̄⁰ mixing Δm_d."),
        FlavorPhysics("CKM |V_ts|", "|V_ts|", 0.0415, 0.0009, discovery_year=2006,
                      description="Top-strange coupling. From B_s-B̄_s mixing Δm_s."),
        FlavorPhysics("CKM |V_tb|", "|V_tb|", 1.014, 0.029, discovery_year=2009,
                      description="Top-bottom coupling. Near unity — top decays almost "
                                  "exclusively to b. From single top production at Tevatron."),

        # ---- CKM CP violation ------------------------------------------------
        FlavorPhysics("Jarlskog invariant", "J_CP", 3.08e-5, 0.17e-5, discovery_year=1985,
                      description="J = Im(V_ud V_cs V*_us V*_cd). Parameterization-independent "
                                  "measure of CP violation in the quark sector. Non-zero at >100σ. "
                                  "But ~10⁻⁹ too small to explain baryon asymmetry."),
        FlavorPhysics("CKM angle β (from B⁰→J/ψ K_S)", "sin 2β", 0.699, 0.017,
                      discovery_year=2001,
                      description="First B-factory result from BaBar+Belle (2001). "
                                  "Confirmed CKM mechanism of CP violation."),
        FlavorPhysics("CKM angle α (from B→ππ, ρπ, ρρ)", "α", 85.2, 4.8, units="degrees",
                      discovery_year=2003,
                      description="Measured from time-dependent CP asymmetry in b→u decays."),
        FlavorPhysics("CKM angle γ (from B→DK)", "γ", 65.9, 3.5, units="degrees",
                      discovery_year=2010,
                      description="Least precisely known CKM angle. Measured from tree-level "
                                  "B→DK interference. LHCb + Belle II target <1° precision."),

        # ---- Neutral meson mixing --------------------------------------------
        FlavorPhysics("B⁰-B̄⁰ mass difference", "Δm_d", 0.5065, 0.0019, units="ps⁻¹",
                      discovery_year=1987,
                      description="ARGUS (1987) discovered B⁰-B̄⁰ mixing — surprisingly large. "
                                  "Allowed prediction of m_t before top discovery."),
        FlavorPhysics("B_s-B̄_s mass difference", "Δm_s", 17.765, 0.006, units="ps⁻¹",
                      discovery_year=2006,
                      description="CDF (2006) measured B_s mixing frequency. Very rapid "
                                  "oscillations — requires excellent time resolution."),
        FlavorPhysics("K_L-K_S mass difference", "Δm_K", 3.484e-12, 0.006e-12, units="MeV",
                      discovery_year=1964,
                      description="The tiny mass difference (~5×10⁻¹⁵ m_K) that enabled "
                                  "CP violation discovery. Predicted the charm quark (GIM mechanism, 1970)."),
        FlavorPhysics("CP violation in kaons (indirect)", "|ε_K|", 2.228e-3, 0.011e-3,
                      discovery_year=1964,
                      description="Indirect CP violation parameter in the neutral kaon system. "
                                  "Cronin + Fitch (1964): K_L→π⁺π⁻ = 2×10⁻³ branching ratio."),
        FlavorPhysics("Direct CPV in kaons", "Re(ε'/ε)", 1.66e-3, 0.23e-3,
                      discovery_year=1999,
                      description="Direct CP violation in K→ππ. NA48 + KTeV confirmed at "
                                  ">10σ. Distinguishes direct from indirect CPV."),
    ]


# =============================================================================
# Section 9: More Hadrons and Resonances (PDG 2024)
# =============================================================================

def _build_more_hadrons() -> list[ParticleProperty]:
    """Vector mesons, heavy-light mesons, and key resonances."""
    return [
        # Vector mesons (spin-1, quark-antiquark)
        ParticleProperty("rho(770)", "ρ",
                         mass_mev=775.26, mass_uncertainty_mev=0.25,
                         charge_e=0, spin=1, lifetime_s=4.4e-24, width_mev=149.1,
                         generation=0, particle_type="hadron",
                         discovery_year=1961,
                         description="Lightest vector meson (uū-dd̄). Dominates low-energy "
                                     "ππ scattering. Key to vector meson dominance model."),
        ParticleProperty("omega(782)", "ω",
                         mass_mev=782.66, mass_uncertainty_mev=0.13,
                         charge_e=0, spin=1, lifetime_s=7.8e-23, width_mev=8.49,
                         generation=0, particle_type="hadron",
                         discovery_year=1961,
                         description="Isoscalar vector meson (uū+dd̄). Narrow width due to "
                                     "OZI suppression — prefers π⁰γ over 3π."),
        ParticleProperty("phi(1020)", "φ",
                         mass_mev=1019.461, mass_uncertainty_mev=0.016,
                         charge_e=0, spin=1, lifetime_s=1.5e-22, width_mev=4.249,
                         generation=0, particle_type="hadron",
                         discovery_year=1962,
                         description="Nearly pure ss̄ state. OZI-suppressed decay to ρπ. "
                                     "Strangeness content confirmed from decay pattern."),
        ParticleProperty("eta", "η",
                         mass_mev=547.862, mass_uncertainty_mev=0.017,
                         charge_e=0, spin=0, lifetime_s=5.0e-19, width_mev=1.31e-3,
                         generation=0, particle_type="hadron",
                         discovery_year=1961,
                         description="Pseudoscalar octet member (uū+dd̄-2ss̄)/√6. "
                                     "η-η' mixing reveals U(1)_A anomaly — connection to "
                                     "the strong CP problem."),
        ParticleProperty("eta prime", "η'",
                         mass_mev=957.78, mass_uncertainty_mev=0.06,
                         charge_e=0, spin=0, lifetime_s=3.4e-21, width_mev=0.194,
                         generation=0, particle_type="hadron",
                         discovery_year=1964,
                         description="Pseudoscalar singlet. Abnormally heavy due to U(1)_A "
                                     "anomaly — would be a Nambu-Goldstone boson otherwise. "
                                     "The η-η' mass splitting is a window into non-perturbative QCD."),

        # ---- Heavy mesons (charm, bottom) — flavor physics workhorses --------
        ParticleProperty("J/psi(1S)", "J/ψ",
                         mass_mev=3096.900, mass_uncertainty_mev=0.006,
                         charge_e=0, spin=1, lifetime_s=7.1e-21, width_mev=0.0926,
                         generation=0, particle_type="hadron",
                         discovery_year=1974,
                         description="The November Revolution particle. cc̄ bound state. "
                                     "Extremely narrow width — OZI suppression + below "
                                     "DD̄ threshold. Proof of charm quark."),
        ParticleProperty("Upsilon(1S)", "Υ(1S)",
                         mass_mev=9460.30, mass_uncertainty_mev=0.26,
                         charge_e=0, spin=1, lifetime_s=1.2e-20, width_mev=0.05402,
                         generation=0, particle_type="hadron",
                         discovery_year=1977,
                         description="bb̄ bound state. Lederman at Fermilab. Narrow — below "
                                     "BB̄ threshold. Bottom quark confirmed."),
        ParticleProperty("D⁰ meson", "D⁰",
                         mass_mev=1864.84, mass_uncertainty_mev=0.05,
                         charge_e=0, spin=0, lifetime_s=4.101e-13, width_mev=1.60e-9,
                         generation=0, particle_type="hadron",
                         discovery_year=1976,
                         description="Lightest charmed meson (cū). D⁰-D̄⁰ mixing observed "
                                     "in 2007 — first up-type quark mixing evidence."),
        ParticleProperty("B⁰ meson", "B⁰",
                         mass_mev=5279.66, mass_uncertainty_mev=0.12,
                         charge_e=0, spin=0, lifetime_s=1.519e-12, width_mev=4.33e-10,
                         generation=0, particle_type="hadron",
                         discovery_year=1983,
                         description="db̄ bound state. Long lifetime enables precision CP "
                                     "violation studies at B factories. Time-dependent "
                                     "B⁰→J/ψ K_S asymmetry measures sin 2β."),
        ParticleProperty("B_s meson", "B_s⁰",
                         mass_mev=5366.88, mass_uncertainty_mev=0.14,
                         charge_e=0, spin=0, lifetime_s=1.520e-12, width_mev=4.33e-10,
                         generation=0, particle_type="hadron",
                         discovery_year=1992,
                         description="sb̄ bound state. Rapid B_s-B̄_s oscillations (Δm_s=17.8 ps⁻¹) "
                                     "probe high-mass scales. width difference ΔΓ_s tests HQET."),

        # ---- Baryons ---------------------------------------------------------
        ParticleProperty("Lambda (uds)", "Λ⁰",
                         mass_mev=1115.683, mass_uncertainty_mev=0.006,
                         charge_e=0, spin=0.5, lifetime_s=2.632e-10, width_mev=2.50e-12,
                         generation=0, particle_type="hadron",
                         discovery_year=1950,
                         description="Lightest strange baryon. Discovery of 'V particles' "
                                     "in cosmic rays launched strange particle physics."),
        ParticleProperty("Sigma plus (uus)", "Σ⁺",
                         mass_mev=1189.37, mass_uncertainty_mev=0.07,
                         charge_e=1, spin=0.5, lifetime_s=8.018e-11, width_mev=8.21e-12,
                         generation=0, particle_type="hadron",
                         discovery_year=1953),
        ParticleProperty("Xi minus (dss)", "Ξ⁻",
                         mass_mev=1321.71, mass_uncertainty_mev=0.07,
                         charge_e=-1, spin=0.5, lifetime_s=1.639e-10, width_mev=4.02e-12,
                         generation=0, particle_type="hadron",
                         discovery_year=1952,
                         description="Cascade baryon. Two-step decay: Ξ⁻→Λ⁰π⁻, then Λ⁰→pπ⁻. "
                                     "'Cascade' particle in cosmic ray emulsion."),
    ]


# =============================================================================
# Section 10: Thermodynamic Reference Data
# =============================================================================

def _build_thermodynamic_data() -> list[PhysicalConstant]:
    """Key thermodynamic measurements: specific heats, latent heats, phase transitions."""
    return [
        PhysicalConstant("specific heat capacity of water (liquid, 25°C)", "c_p(H₂O)",
                         4.1813, 0.0003, "J/(g·K)", discovery_year=1843, refined_year=1956,
                         domain=PhysicalDomain.THERMODYNAMIC,
                         description="Joule's mechanical equivalent of heat (1843) was the "
                                     "first precision measurement. Water's high heat capacity "
                                     "is anomalous — due to hydrogen bonding."),
        PhysicalConstant("specific heat capacity of iron", "c_p(Fe)",
                         0.449, 0.002, "J/(g·K)", discovery_year=1819, refined_year=1930,
                         domain=PhysicalDomain.THERMODYNAMIC),
        PhysicalConstant("specific heat capacity of aluminum", "c_p(Al)",
                         0.897, 0.001, "J/(g·K)", discovery_year=1819, refined_year=1930,
                         domain=PhysicalDomain.THERMODYNAMIC),
        PhysicalConstant("specific heat capacity of copper", "c_p(Cu)",
                         0.385, 0.001, "J/(g·K)", discovery_year=1819, refined_year=1930,
                         domain=PhysicalDomain.THERMODYNAMIC),

        PhysicalConstant("latent heat of fusion (water)", "L_f(H₂O)",
                         334.0, 0.5, "J/g", discovery_year=1762, refined_year=1900,
                         domain=PhysicalDomain.THERMODYNAMIC,
                         description="Joseph Black (1762) discovered latent heat. Water's "
                                     "high latent heat is critical for climate regulation."),
        PhysicalConstant("latent heat of vaporization (water at 100°C)", "L_v(H₂O)",
                         2260.0, 2.0, "J/g", discovery_year=1762, refined_year=1900,
                         domain=PhysicalDomain.THERMODYNAMIC),

        PhysicalConstant("triple point temperature of water", "T_tp(H₂O)",
                         273.16, 0.0, "K", discovery_year=1954, refined_year=2019,
                         domain=PhysicalDomain.THERMODYNAMIC,
                         description="Defined as exactly 273.16 K until 2019 redefinition. "
                                     "Now fixed at this value by the kelvin definition."),
        PhysicalConstant("boiling point of water at 1 atm", "T_boil(H₂O)",
                         373.1339, 0.002, "K", discovery_year=1742, refined_year=1954,
                         domain=PhysicalDomain.THERMODYNAMIC),

        PhysicalConstant("absolute zero (from gas extrapolation)", "T=0",
                         -273.15, 0.01, "°C", discovery_year=1787, refined_year=1900,
                         domain=PhysicalDomain.THERMODYNAMIC,
                         description="Charles (1787) and Gay-Lussac (1802) found gases "
                                     "contract linearly with temperature, extrapolating to "
                                     "zero volume at -273°C. Kelvin (1848) proposed absolute scale."),

        PhysicalConstant("ideal gas constant", "R",
                         8.314462618, 0.000000024, "J/(mol·K)", discovery_year=1834,
                         refined_year=2019, domain=PhysicalDomain.THERMODYNAMIC,
                         description="R = N_A k_B. Clapeyron (1834) combined Boyle, Charles, "
                                     "and Gay-Lussac's laws into PV = nRT."),

        PhysicalConstant("standard molar volume (ideal gas at STP)", "V_m",
                         22.41396954, 0.00000018, "L/mol", discovery_year=1811,
                         refined_year=2019, domain=PhysicalDomain.THERMODYNAMIC,
                         description="Avogadro's law (1811): equal volumes contain equal "
                                     "numbers of molecules at same T and P."),

        PhysicalConstant("critical temperature of water", "T_c(H₂O)",
                         647.096, 0.001, "K", discovery_year=1822, refined_year=1995,
                         domain=PhysicalDomain.THERMODYNAMIC,
                         description="Cagniard de la Tour (1822) discovered the critical point. "
                                     "Above T_c, liquid and vapor phases become indistinguishable."),
        PhysicalConstant("critical pressure of water", "P_c(H₂O)",
                         22.064, 0.001, "MPa", discovery_year=1822, refined_year=1995,
                         domain=PhysicalDomain.THERMODYNAMIC),

        # Phase transition temperatures (elements)
        PhysicalConstant("melting point of iron", "T_melt(Fe)",
                         1811.0, 1.0, "K", discovery_year=-2000, refined_year=1950,
                         domain=PhysicalDomain.THERMODYNAMIC),
        PhysicalConstant("boiling point of iron", "T_boil(Fe)",
                         3134.0, 2.0, "K", discovery_year=-2000, refined_year=1950,
                         domain=PhysicalDomain.THERMODYNAMIC),
        PhysicalConstant("melting point of tungsten", "T_melt(W)",
                         3695.0, 2.0, "K", discovery_year=1781, refined_year=1950,
                         domain=PhysicalDomain.THERMODYNAMIC,
                         description="Highest melting point of any element. Critical for "
                                     "incandescent filaments and fusion reactor walls."),
    ]


# =============================================================================
# Section 11: GR Solar System and Precision Tests
# =============================================================================

@dataclass
class GRTestObservation:
    """A precision test of general relativity."""
    name: str
    measured_value: str           # Human-readable result
    gr_prediction: str            # What GR predicts
    ppn_parameter: str            # Which PPN parameter is constrained
    constraint_value: float       # Numerical constraint
    constraint_uncertainty: float
    discovery_year: int
    source: str = "Will 2014 (Living Reviews in Relativity)"
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "measured_value": self.measured_value,
            "gr_prediction": self.gr_prediction,
            "ppn_parameter": self.ppn_parameter,
            "discovery_year": self.discovery_year,
        }


def _build_gr_tests() -> list[GRTestObservation]:
    """Precision experimental tests of general relativity."""
    return [
        GRTestObservation(
            name="Mercury perihelion precession (excess)",
            measured_value="42.9799 ± 0.0009 arcsec/century (MESSENGER)",
            gr_prediction="42.98 arcsec/century",
            ppn_parameter="(2+2γ-β)/3",
            constraint_value=42.9799, constraint_uncertainty=0.0009,
            discovery_year=1859,
            description="Leverrier (1859) discovered the excess. Einstein (1915): 'For a few "
                        "days I was beside myself with joyous excitement.' The most historically "
                        "important anomaly in physics — led directly to GR."),
        GRTestObservation(
            name="Light deflection by the Sun (VLBI)",
            measured_value="γ = 0.9998 ± 0.0003 (Cassini, 2003)",
            gr_prediction="γ = 1.0 exactly",
            ppn_parameter="γ",
            constraint_value=0.9998, constraint_uncertainty=0.0003,
            discovery_year=1919,
            description="Eddington (1919) measured 1.98±0.16 arcsec at Sobral. Modern VLBI "
                        "measurements of quasar positions near the Sun give γ = 0.9998 ± 0.0003. "
                        "Cassini Shapiro delay gives γ-1 = (2.1 ± 2.3)×10⁻⁵."),
        GRTestObservation(
            name="Shapiro time delay (Cassini)",
            measured_value="γ-1 = (2.1 ± 2.3)×10⁻⁵",
            gr_prediction="γ = 1.0 → γ-1 = 0",
            ppn_parameter="γ",
            constraint_value=2.1e-5, constraint_uncertainty=2.3e-5,
            discovery_year=1964,
            description="Shapiro (1964) predicted radar signals would be delayed passing near "
                        "the Sun. Cassini (2003) measured the round-trip time to <0.002% — "
                        "the most precise Solar System GR test."),
        GRTestObservation(
            name="Lunar Laser Ranging — equivalence principle",
            measured_value="η = (0.6 ± 5.7)×10⁻¹⁴ (Nordtvedt parameter)",
            gr_prediction="η = 0 exactly (strong equivalence principle)",
            ppn_parameter="η (Nordtvedt)",
            constraint_value=0.6e-14, constraint_uncertainty=5.7e-14,
            discovery_year=1969,
            description="LLR tests whether the Earth and Moon fall toward the Sun at the "
                        "same rate despite different gravitational self-energy. Confirms "
                        "the strong equivalence principle to ~10⁻¹³."),
        GRTestObservation(
            name="Lunar Laser Ranging — dG/dt",
            measured_value="Ġ/G = (2.1 ± 7.4)×10⁻¹³ yr⁻¹",
            gr_prediction="Ġ/G = 0 (G is constant in GR)",
            ppn_parameter="Ġ/G",
            constraint_value=2.1e-13, constraint_uncertainty=7.4e-13,
            discovery_year=1980,
            description="Tests whether the gravitational constant varies with time. "
                        "GR predicts it is constant. Some modified gravity theories "
                        "(Brans-Dicke, scalar-tensor) predict variation."),
        GRTestObservation(
            name="Gravity Probe B — frame dragging",
            measured_value="geodetic: -6601.8 ± 18.3 mas/yr; frame-dragging: -37.2 ± 7.2 mas/yr",
            gr_prediction="geodetic: -6606.1 mas/yr; frame-dragging: -39.2 mas/yr",
            ppn_parameter="γ (geodetic), Lense-Thirring (frame-dragging)",
            constraint_value=37.2, constraint_uncertainty=7.2,
            discovery_year=2011,
            description="GP-B (2004-2011) measured two GR effects with gyroscopes in orbit: "
                        "geodetic precession (spacetime curvature) and frame-dragging "
                        "(Earth's rotation dragging spacetime). Confirmed to ~19%."),
        GRTestObservation(
            name="Binary pulsar orbital decay (PSR B1913+16)",
            measured_value="Ṗ_b = (-2.423 ± 0.001)×10⁻¹²",
            gr_prediction="Ṗ_b = -2.402×10⁻¹² (quadrupole formula)",
            ppn_parameter="Quadrupole formula test",
            constraint_value=-2.423e-12, constraint_uncertainty=0.001e-12,
            discovery_year=1974,
            description="Hulse-Taylor binary pulsar (1974). Orbital decay matches GR's "
                        "gravitational wave emission prediction to 0.2%. Nobel Prize 1993. "
                        "First indirect evidence for gravitational waves."),
        GRTestObservation(
            name="Double pulsar PSR J0737-3039",
            measured_value="5 independent GR tests in single system, all consistent to <0.05%",
            gr_prediction="All 5 tests consistent with GR",
            ppn_parameter="Multiple",
            constraint_value=0.0, constraint_uncertainty=0.0005,
            discovery_year=2003,
            description="The only known double pulsar. Five independent GR tests: "
                        "periastron advance, Shapiro delay, orbital decay, gravitational "
                        "redshift, and relativistic spin precession. All match GR."),
        GRTestObservation(
            name="Event Horizon Telescope — M87* shadow",
            measured_value="Shadow diameter = 42 ± 3 μas; mass = (6.5 ± 0.7)×10⁹ M⊙",
            gr_prediction="Shadow size consistent with Kerr metric for given mass",
            ppn_parameter="Kerr metric test",
            constraint_value=42.0, constraint_uncertainty=3.0,
            discovery_year=2019,
            description="EHT (2019) imaged the black hole shadow in M87. Size, shape, and "
                        "brightness asymmetry consistent with GR's Kerr metric. Rules out "
                        "many alternative gravity theories."),
    ]


# =============================================================================
# Section 12: Equivalence Principle Tests
# =============================================================================

def _build_equivalence_principle_tests() -> list[PhysicalConstant]:
    """Experimental tests of the weak, Einstein, and strong equivalence principles."""
    return [
        PhysicalConstant("Eötvös experiment (wood vs platinum)", "η(Eötvös)",
                         5.0e-9, 1.0e-9, "dimensionless", discovery_year=1890,
                         domain=PhysicalDomain.GRAVITATIONAL,
                         description="Eötvös (1890) showed gravitational and inertial mass "
                                     "are equal to ~5×10⁻⁹. The founding experiment of the "
                                     "equivalence principle — Einstein called it 'the fundamental "
                                     "fact' that inspired GR."),
        PhysicalConstant("Eöt-Wash torsion balance (Be-Ti)", "η(Eöt-Wash, Be-Ti)",
                         1.0e-13, 1.5e-13, "dimensionless", discovery_year=1999,
                         refined_year=2008,
                         domain=PhysicalDomain.GRAVITATIONAL,
                         description="Eöt-Wash group (Seattle). Torsion balance comparing "
                                     "beryllium and titanium test masses. WEP verified to "
                                     "~10⁻¹³ — one of the most precise null results in physics."),
        PhysicalConstant("Eöt-Wash torsion balance (Be-Al)", "η(Eöt-Wash, Be-Al)",
                         1.8e-13, 2.8e-13, "dimensionless", discovery_year=2008,
                         domain=PhysicalDomain.GRAVITATIONAL),
        PhysicalConstant("MICROSCOPE satellite (Ti-Pt)", "η(MICROSCOPE)",
                         1.5e-15, 0.0, "dimensionless (upper limit)", discovery_year=2017,
                         refined_year=2022,
                         domain=PhysicalDomain.GRAVITATIONAL,
                         description="CNES/ONERA satellite mission (2016-2018). Tested WEP "
                                     "in space with Ti and Pt test masses. Final result (2022): "
                                     "η < 1.5×10⁻¹⁵ — 100× better than ground-based. "
                                     "No violation found."),
        PhysicalConstant("Lunar laser ranging — Nordtvedt effect", "η(LLR)",
                         4.4e-4, 4.5e-4, "dimensionless", discovery_year=1976,
                         refined_year=2012,
                         domain=PhysicalDomain.GRAVITATIONAL,
                         description="Tests the STRONG equivalence principle: does gravitational "
                                     "self-energy contribute to inertial mass differently from "
                                     "gravitational mass? GR says no (η=0). LLR confirms η ≈ 0."),
    ]


# =============================================================================
# Section 13: Direct Detection Limits (DM, Axions, Proton Decay, nEDM)
# =============================================================================

def _build_direct_detection_limits() -> list[PhysicalConstant]:
    """Current experimental limits on BSM signatures from direct detection experiments."""
    return [
        PhysicalConstant("WIMP-nucleon cross-section (spin-independent, 90% CL)",
                         "σ_SI(WIMP)", 9.2e-48, 0.0, "cm² (upper limit at 36 GeV/c²)",
                         discovery_year=2023, source="LZ 2023",
                         domain=PhysicalDomain.CONDENSED_MATTER,
                         description="LUX-ZEPLIN (LZ): world's most sensitive WIMP search. "
                                     "5.5 tonnes of liquid xenon, 1 km underground at SURF. "
                                     "No signal — excludes WIMP miracle cross-sections."),
        PhysicalConstant("WIMP-nucleon cross-section (XENONnT, 90% CL)",
                         "σ_SI(XENONnT)", 2.6e-47, 0.0, "cm² (upper limit)",
                         discovery_year=2023, source="XENONnT 2023",
                         domain=PhysicalDomain.CONDENSED_MATTER),
        PhysicalConstant("Axion-photon coupling (ADMX, 90% CL)",
                         "g_aγγ(ADMX)", 1.0e-15, 0.0, "GeV⁻¹ (upper limit in 2-4 μeV range)",
                         discovery_year=2021, source="ADMX 2021",
                         domain=PhysicalDomain.CONDENSED_MATTER,
                         description="Axion Dark Matter eXperiment. Searches for axions "
                                     "converting to photons in a microwave cavity in a strong "
                                     "magnetic field. Covers DFSZ model axion masses 2-4 μeV."),
        PhysicalConstant("Neutrinoless double beta decay half-life (¹³⁶Xe, 90% CL)",
                         "T_1/2(0νββ, ¹³⁶Xe)", 2.3e26, 0.0, "years (lower limit)",
                         discovery_year=2023, source="KamLAND-Zen 800",
                         domain=PhysicalDomain.NUCLEAR,
                         description="KamLAND-Zen 800 (2023): most sensitive 0νββ search. "
                                     "Upper limit on m_ββ < 0.036-0.156 eV. If 0νββ is observed, "
                                     "neutrinos are Majorana particles (their own antiparticles) "
                                     "and lepton number is violated."),
        PhysicalConstant("Proton decay (p→e⁺π⁰, partial lifetime, 90% CL)",
                         "τ(p→e⁺π⁰)", 2.4e34, 0.0, "years (lower limit)",
                         discovery_year=2020, source="Super-Kamiokande",
                         domain=PhysicalDomain.PARTICLE_PROPERTIES,
                         description="Super-Kamiokande (2020): no proton decay observed after "
                                     ">20 years. Lifetimes > 10³⁴ years for most channels. "
                                     "Rules out minimal SU(5) GUT. Constrains SUSY GUTs."),
        PhysicalConstant("Neutron electric dipole moment (90% CL)",
                         "|d_n|", 1.8e-26, 0.0, "e·cm (upper limit)",
                         discovery_year=2020, source="PSI nEDM",
                         domain=PhysicalDomain.PARTICLE_PROPERTIES,
                         description="PSI nEDM experiment: |d_n| < 1.8×10⁻²⁶ e·cm. "
                                     "EDM would violate T and P (and CP by CPT). Constrains "
                                     "CP-violating phases in BSM theories — SUSY θ parameters "
                                     "must be < 10⁻¹⁰ unless CP is spontaneously broken."),
    ]


# =============================================================================
# Section 14: Periodic Table — Key Element Properties
# =============================================================================

@dataclass
class ElementProperty:
    """Key physical properties of a chemical element."""
    symbol: str
    name: str
    atomic_number: int
    atomic_mass: float             # in atomic mass units (u)
    ionization_energy_ev: float    # First ionization energy
    electron_affinity_ev: float | None   # None = unstable negative ion
    electronegativity: float | None      # Pauling scale
    density_g_cm3: float | None
    melting_point_k: float | None
    boiling_point_k: float | None
    discovery_year: int
    source: str = "NIST/CRC Handbook"
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol, "name": self.name,
            "atomic_number": self.atomic_number,
            "ionization_energy_ev": self.ionization_energy_ev,
            "electronegativity": self.electronegativity,
            "discovery_year": self.discovery_year,
        }


def _build_element_properties() -> list[ElementProperty]:
    """Key elements spanning the periodic table."""
    return [
        ElementProperty("H", "hydrogen", 1, 1.008, 13.598, 0.754, 2.20,
                        8.99e-5, 13.99, 20.27, 1766,
                        description="Lightest, most abundant element. ~75% of baryonic mass. "
                                    "Hydrogen spectrum was the key to quantum mechanics."),
        ElementProperty("He", "helium", 2, 4.0026, 24.587, None, None,
                        1.79e-4, 0.95, 4.22, 1868,
                        description="Discovered in the solar spectrum before found on Earth. "
                                    "Highest first ionization energy. Inert — closed 1s shell."),
        ElementProperty("Li", "lithium", 3, 6.94, 5.392, 0.618, 0.98,
                        0.534, 453.65, 1603.0, 1817,
                        description="Third element. ⁷Li abundance from BBN is ~3× lower than "
                                    "predicted — the 'lithium problem.'"),
        ElementProperty("C", "carbon", 6, 12.011, 11.260, 1.263, 2.55,
                        2.267, 3915.0, 3915.0, -8000,  # Prehistoric
                        description="Basis of organic chemistry. Triple-alpha Hoyle state "
                                    "at 7.65 MeV enables stellar production — a famous example "
                                    "of anthropic fine-tuning (or environmental selection)."),
        ElementProperty("N", "nitrogen", 7, 14.007, 14.534, -0.07, 3.04,
                        1.251e-3, 63.15, 77.36, 1772),
        ElementProperty("O", "oxygen", 8, 15.999, 13.618, 1.461, 3.44,
                        1.429e-3, 54.36, 90.20, 1774,
                        description="Most abundant element in Earth's crust. Produced in "
                                    "massive stars via helium burning and dispersed by supernovae."),
        ElementProperty("Na", "sodium", 11, 22.990, 5.139, 0.548, 0.93,
                        0.971, 370.94, 1156.0, 1807,
                        description="Fraunhofer D lines (589.0, 589.6 nm) were among the "
                                    "first spectral lines catalogued. Alkali metal."),
        ElementProperty("Al", "aluminum", 13, 26.982, 5.986, 0.433, 1.61,
                        2.70, 933.47, 2792.0, 1825),
        ElementProperty("Si", "silicon", 14, 28.085, 8.152, 1.389, 1.90,
                        2.329, 1687.0, 3538.0, 1824,
                        description="Semiconductor foundation. Band gap 1.12 eV. "
                                    "Silicon-silicon bond energy 222 kJ/mol."),
        ElementProperty("Fe", "iron", 26, 55.845, 7.902, 0.151, 1.83,
                        7.874, 1811.0, 3134.0, -2500,  # ~2500 BCE
                        description="Peak of the binding energy curve — most stable nucleus. "
                                    "This is why stellar cores end as iron. The iron peak in "
                                    "cosmic abundances is a direct probe of nuclear astrophysics."),
        ElementProperty("Cu", "copper", 29, 63.546, 7.726, 1.236, 1.90,
                        8.96, 1357.77, 2835.0, -8000),
        ElementProperty("Ag", "silver", 47, 107.868, 7.576, 1.302, 1.93,
                        10.49, 1234.93, 2435.0, -4000),
        ElementProperty("Au", "gold", 79, 196.967, 9.226, 2.309, 2.54,
                        19.30, 1337.33, 3129.0, -4000,
                        description="Heaviest stable mono-isotopic element accessible to "
                                    "ancient civilizations. Produced by neutron star mergers "
                                    "(confirmed by GW170817 kilonova)."),
        ElementProperty("Hg", "mercury", 80, 200.59, 10.438, None, 2.00,
                        13.534, 234.32, 629.88, -1500,
                        description="Only liquid metal at room temperature. Franck-Hertz "
                                    "experiment (1914) used mercury vapor."),
        ElementProperty("Pb", "lead", 82, 207.2, 7.417, 0.364, 2.33,
                        11.34, 600.61, 2022.0, -4000,
                        description="Heaviest stable element (²⁰⁸Pb is doubly magic: Z=82, "
                                    "N=126). Endpoint of uranium/thorium decay chains."),
        ElementProperty("U", "uranium", 92, 238.029, 6.194, None, 1.38,
                        19.1, 1405.3, 4404.0, 1789,
                        description="Heaviest naturally occurring element. Fissile ²³⁵U "
                                    "(0.72% natural abundance) discovered by Hahn/Strassmann/Meitner (1938)."),
    ]


# =============================================================================
# Build all registries
# =============================================================================

ALL_CONSTANTS: list[PhysicalConstant] = _build_fundamental_constants()
ALL_PARTICLES: list[ParticleProperty] = _build_particle_properties()
ALL_SPECTRAL_LINES: list[SpectralLine] = _build_spectral_lines()
ALL_COSMOLOGY: list[CosmologicalParameter] = _build_cosmological_parameters()
ALL_NUCLEAR: list[NuclearProperty] = _build_nuclear_properties()
ALL_NEUTRINO: list[PhysicalConstant] = _build_neutrino_parameters()
ALL_ANOMALIES: list[AnomalyMeasurement] = _build_anomalies()
ALL_FLAVOR: list[FlavorPhysics] = _build_flavor_physics()
ALL_MORE_HADRONS: list[ParticleProperty] = _build_more_hadrons()
ALL_THERMODYNAMIC: list[PhysicalConstant] = _build_thermodynamic_data()
ALL_GR_TESTS: list[GRTestObservation] = _build_gr_tests()
ALL_EQUIVALENCE: list[PhysicalConstant] = _build_equivalence_principle_tests()
ALL_DIRECT_LIMITS: list[PhysicalConstant] = _build_direct_detection_limits()
ALL_ELEMENTS: list[ElementProperty] = _build_element_properties()


# =============================================================================
# Temporal gating
# =============================================================================

def get_data_up_to_year(year: int) -> dict[str, list[Any]]:
    """Return all data discovered up to and including the given year.

    This is the core temporal gating function. Given a cutoff year,
    returns only the experimental data that was known at that time.

    Args:
        year: Cutoff year (inclusive). Data from year 0 to year is included.

    Returns:
        Dict mapping domain name to list of data objects known by that year.
    """
    def _filter(data_list: list[Any]) -> list[Any]:
        return [d for d in data_list if d.discovery_year <= year]

    return {
        "constants": _filter(ALL_CONSTANTS),
        "particles": _filter(ALL_PARTICLES),
        "spectral_lines": _filter(ALL_SPECTRAL_LINES),
        "cosmology": _filter(ALL_COSMOLOGY),
        "nuclear": _filter(ALL_NUCLEAR),
        "neutrino": _filter(ALL_NEUTRINO),
        "anomalies": _filter(ALL_ANOMALIES),
        "flavor": _filter(ALL_FLAVOR),
        "more_hadrons": _filter(ALL_MORE_HADRONS),
        "thermodynamic": _filter(ALL_THERMODYNAMIC),
        "gr_tests": _filter(ALL_GR_TESTS),
        "equivalence": _filter(ALL_EQUIVALENCE),
        "direct_limits": _filter(ALL_DIRECT_LIMITS),
        "elements": _filter(ALL_ELEMENTS),
    }


def get_all_data_up_to_year(year: int) -> list[Any]:
    """Flat list of all data objects known up to the given year."""
    data = get_data_up_to_year(year)
    result: list[Any] = []
    for items in data.values():
        result.extend(items)
    return result


def print_timeline_summary() -> None:
    """Print a summary of what data was available at each era cutoff."""
    for era_name, year in ERA_CUTOFFS.items():
        data = get_data_up_to_year(year)
        total = sum(len(v) for v in data.values())
        n_particles = len(data["particles"])
        n_anomalies = len(data["anomalies"])
        print(
            f"  {era_name:25s} (≤{year:5d}): "
            f"{total:3d} total, {n_particles:2d} particles, "
            f"{n_anomalies:2d} anomalies"
        )


# =============================================================================
# Quick self-test
# =============================================================================

if __name__ == "__main__":
    print(f"Physical constants module loaded.")
    print(f"  Fundamental constants: {len(ALL_CONSTANTS)}")
    print(f"  Particle properties:   {len(ALL_PARTICLES)}")
    print(f"  Spectral lines:        {len(ALL_SPECTRAL_LINES)}")
    print(f"  Cosmological params:   {len(ALL_COSMOLOGY)}")
    print(f"  Nuclear properties:    {len(ALL_NUCLEAR)}")
    print(f"  Neutrino parameters:   {len(ALL_NEUTRINO)}")
    print(f"  Anomalies:             {len(ALL_ANOMALIES)}")
    total = (len(ALL_CONSTANTS) + len(ALL_PARTICLES) + len(ALL_SPECTRAL_LINES)
             + len(ALL_COSMOLOGY) + len(ALL_NUCLEAR) + len(ALL_NEUTRINO)
             + len(ALL_ANOMALIES))
    print(f"  TOTAL:                 {total} entries")
    print()
    print("Timeline summary (data available at each era cutoff):")
    print_timeline_summary()
    print()

    # Verify temporal gating: pre-1900 should have no quantum particles
    pre_1900 = get_data_up_to_year(1899)
    particles_pre_1900 = [p.name for p in pre_1900["particles"]]
    assert "electron" in particles_pre_1900, "Electron should be known by 1900"
    assert "photon" not in particles_pre_1900, "Photon was not known by 1900"
    print("✓ Temporal gating: electron known pre-1900, photon not")
    print("✓ Self-test passed")
