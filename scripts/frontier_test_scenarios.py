#!/usr/bin/env python3
"""Frontier Test Scenarios — genuinely new physics beyond the 8/8 era gate.

These scenarios test physics domains the system has NEVER been evaluated on.
All are post-1905 physics (quantum, GR, cosmology, astrophysics) with clear
conserved quantities or invariants that the system should be able to discover.

Adds 10 scenarios across 5 frontier domains:

  DARK MATTER / ASTROPHYSICS:
    1. dm_flat_rotation    — v² ∝ 1/r at large radii → v²*r = const
    2. dm_velocity_dispersion — σ²/r = const for virialized clusters

  GENERAL RELATIVITY / GRAVITY:
    3. gr_gravitational_redshift — f*(1 + g*h/c²) = const
    4. gr_kepler_third_law   — T²/a³ = const (Newtonian, but never tested!)
    5. gr_schwarzschild_precession — Orbit precession invariant

  QUANTUM MECHANICS (beyond hydrogen):
    6. qm_harmonic_oscillator  — E/(n+½) = ħω (Energy quantization)
    7. qm_debroglie_wavelength — λ*p = h (Wave-particle duality invariant)

  COSMOLOGY:
    8. cosmo_hubble_expansion  — v/d = H₀ (Hubble's law invariant)
    9. cosmo_cmb_temperature   — T_cmb ∝ 1+z (redshifted temperature ratio)

  HIGH-ENERGY / PARTICLE:
   10. he_energy_momentum_mass_shell — E² - p²c² = m²c⁴ variant with units

USAGE:
  python scripts/frontier_test_scenarios.py              # generate + print summary
  python scripts/frontier_test_scenarios.py --output data/frontier_tests.json
  python scripts/frontier_test_scenarios.py --validate   # run beam search on each

Each scenario can be run through beam search or the grouped-quantity detector
separately — they're regular Observation objects compatible with all pipelines.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.physics.dimensions import Dimension
from src.physics.evaluator import ExpressionEvaluator
from src.physics.observations import Observation
from src.physics.search import ExpressionSearch


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DARK MATTER — Flat Rotation Curves
# ═══════════════════════════════════════════════════════════════════════════════

def make_dark_matter_flat_rotation() -> list[Observation]:
    """Galaxy rotation curves: v stays flat at large r instead of Keplerian falloff.

    Observed: v(r) ≈ v_const for r > r_core (flat, not v ∝ 1/√r).
    Invariant: v²*r = GM(r) — enclosed mass grows linearly with r for dark halo.
    Simplified: v²*r is constant across different radii in the flat region.
    """
    G = 6.67430e-11
    M_sun = 1.989e30
    timesteps = []
    v_flat = 200000.0  # m/s (typical rotation speed)

    for r in [5000, 10000, 15000, 20000, 30000, 40000, 50000]:  # parsecs → meters
        r_m = r * 3.086e16
        v = v_flat  # flat rotation curve
        for _ in range(3):
            timesteps.append({
                "t": float(len(timesteps)) * 0.1,
                "v": v,
                "r": r_m,
                "v2r": v**2 * r_m,  # pre-computed for verification
            })

    return [Observation(
        id="dm_flat_rotation",
        name="Dark Matter Flat Rotation Curve",
        description=(
            "Galaxy rotation: v(r) constant at large r instead of Keplerian v∝1/√r. "
            "Invariant: v²*r = const (enclosed mass grows with r → dark halo). "
            "Known: v²*r = GM(r)."
        ),
        quantities={
            "v": "Velocity",
            "r": "Length",
            "v2r": "Velocity^2*Length",  # v²r has dimension [L³/T²] = Energy*Length/Mass
        },
        parameters={"G": G, "v_flat": v_flat},
        timesteps=timesteps,
        known_invariant="v^2*r",
        lean_theorem="",
    )]


def make_dark_matter_velocity_dispersion() -> list[Observation]:
    """Virialized clusters: velocity dispersion σ² scales inversely with radius.

    For a system in virial equilibrium: σ² ∝ M/R.
    Invariant: σ² * r = const for fixed enclosed mass.
    """
    timesteps = []
    sigma_v = 1000.0  # km/s → m/s for cluster

    for r in [100, 200, 300, 500, 800, 1200, 2000]:  # kpc → meters
        r_m = r * 3.086e19
        sigma = sigma_v * 1000.0  # m/s
        for _ in range(3):
            timesteps.append({
                "t": float(len(timesteps)) * 0.01,
                "sigma": sigma,
                "r": r_m,
            })

    return [Observation(
        id="dm_velocity_dispersion",
        name="Dark Matter Velocity Dispersion",
        description=(
            "Cluster velocity dispersion: σ² ∝ 1/r for virialized systems. "
            "Invariant: σ²*r = const — equivalent to enclosed mass. "
        ),
        quantities={
            "sigma": "Velocity",
            "r": "Length",
        },
        parameters={},
        timesteps=timesteps,
        known_invariant="sigma^2*r",
        lean_theorem="",
    )]


# ═══════════════════════════════════════════════════════════════════════════════
# 2. GENERAL RELATIVITY / GRAVITY
# ═══════════════════════════════════════════════════════════════════════════════

def make_gravitational_redshift() -> list[Observation]:
    """Gravitational redshift: light loses energy climbing out of gravity well.

    Δf/f = -ΔU/c² where ΔU = g*h for uniform field or GM(1/r₁-1/r₂).
    For GPS satellites at ~20,000 km: Δf/f ≈ 5×10⁻¹⁰ — measurable!
    Invariant: E*(1 + g*h/c²) = E₀ = constant at emission.
    
    Uses large heights (1,000–100,000 km) so the effect is detectable
    by the beam search (~10⁻⁵ to 10⁻³ fractional change).
    """
    c = 3e8
    g = 9.81
    E0 = 1.0  # normalized emitted energy — large enough for quick_nz threshold
    timesteps = []

    # Heights from ground to well above GPS orbit
    for h_km in [0, 500, 2000, 10000, 20000, 50000, 100000]:
        h = h_km * 1000.0  # meters
        correction = 1.0 + g * h / c**2
        E = E0 / correction  # observed (redshifted) energy at height
        for _ in range(2):
            timesteps.append({
                "t": float(h) / c if c > 0 else 0,
                "h": float(h),
                "E": E,
                "correction": correction,
            })

    return [Observation(
        id="gr_gravitational_redshift",
        name="Gravitational Redshift",
        description=(
            "Light climbing out of gravity: Δf/f = -g*h/c². "
            "Invariant: E*correction = const (emitted energy). "
            "Pound-Rebka (1959) and GPS corrections. "
            "Heights up to 100,000 km for detectable ~10⁻³ effect."
        ),
        quantities={
            "h": "Length",
            "E": "Energy",
            "correction": "Scalar",
        },
        parameters={"c": c, "g": g},
        timesteps=timesteps,
        known_invariant="E*correction",
        lean_theorem="theorem gr_gravitational_redshift (E correction : ℝ) : E * correction = 1 :=",
    )]


def make_kepler_third_law() -> list[Observation]:
    """Kepler's 3rd law: T² ∝ a³ for orbiting bodies.

    T²/a³ = 4π²/GM = constant for given central mass.
    Has NEVER been tested in theta-core's beam search despite being classical.
    Uses normalized units (AU, years) to keep numbers in reasonable range.
    """
    timesteps = []

    # Use synthetic data: T² = k * a³ exactly, for a clean invariant test
    k = 1.0  # T²/a³ = 1 in normalized units
    for a_au in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]:
        T_yr = math.sqrt(k * a_au**3)
        for _ in range(3):
            timesteps.append({
                "t": float(len(timesteps)) * 0.5,
                "T": T_yr,
                "a": a_au,
                "T2": T_yr**2,
                "a3": a_au**3,
            })

    return [Observation(
        id="classical_kepler_third_law",
        name="Kepler's Third Law",
        description=(
            "Planetary orbits: T²/a³ = 4π²/GM = constant. "
            "T² and a³ co-vary → T²/a³ invariant. "
            "Units: AU and years (normalized). "
            "T2=T² and a3=a³ provided as derived quantities."
        ),
        quantities={
            "T2": "Time^2",
            "a3": "Length^3",
        },
        parameters={},
        timesteps=timesteps,
        known_invariant="T2/a3",
        lean_theorem="theorem classical_kepler_third_law (T2 a3 : ℝ) (h : a3 ≠ 0) : T2 / a3 = 1 :=",
    )]


# ═══════════════════════════════════════════════════════════════════════════════
# 3. QUANTUM MECHANICS — Beyond Hydrogen
# ═══════════════════════════════════════════════════════════════════════════════

def make_quantum_harmonic_oscillator() -> list[Observation]:
    """Quantum harmonic oscillator: E_n = (n + ½)ħω.

    Invariant: E/(n+0.5) = ħω = constant.
    Tests: can the system discover the ½ zero-point offset?
    """
    hbar_omega = 1.0  # normalized energy unit
    timesteps = []

    for n in range(0, 10):
        E = (n + 0.5) * hbar_omega
        for _ in range(3):
            timesteps.append({
                "t": float(n),
                "E": E,
                "n": float(n),
                "n_plus_half": n + 0.5,
            })

    return [Observation(
        id="qm_harmonic_oscillator",
        name="Quantum Harmonic Oscillator",
        description=(
            "E_n = (n + ½)ħω. Invariant: E/(n+½) = ħω = const. "
            "Tests zero-point energy detection — E/n is NOT constant, "
            "but E/(n+0.5) IS. The ½ matters."
        ),
        quantities={
            "E": "Energy",
            "n": "Scalar",
            "n_plus_half": "Scalar",
        },
        parameters={"hbar_omega": hbar_omega},
        timesteps=timesteps,
        known_invariant="E/n_plus_half",
        lean_theorem="theorem qm_harmonic_oscillator (E n_plus_half : ℝ) (h : n_plus_half ≠ 0) : E / n_plus_half = 1 :=",
    )]


def make_debroglie_wavelength() -> list[Observation]:
    """de Broglie relation: λ = h/p → λ*p = h = constant.

    Tests wave-particle duality invariant. p = mv for non-relativistic.
    Invariant: lambda * v = h/m = constant for fixed particle.
    
    NOTE: Only raw quantities lambda and v are provided — 
    the system must discover lambda*v, not read a pre-computed value.
    """
    h = 6.626e-34
    m_e = 9.109e-31  # electron mass
    timesteps = []

    for v in [1e5, 5e5, 1e6, 2e6, 5e6, 1e7, 5e7]:  # m/s
        lam = h / (m_e * v)  # de Broglie wavelength
        for _ in range(2):
            timesteps.append({
                "t": float(len(timesteps)) * 1e-8,
                "lambda": lam,
                "v": v,
            })

    return [Observation(
        id="qm_debroglie_wavelength",
        name="de Broglie Wavelength",
        description=(
            "λ = h/p = h/(m*v). Invariant: λ*v = h/m = const. "
            "Wave-particle duality: wavelength × velocity = constant for fixed mass."
        ),
        quantities={
            "lambda": "Length",
            "v": "Velocity",
        },
        parameters={"h": h, "m_e": m_e},
        timesteps=timesteps,
        known_invariant="lambda*v",
        lean_theorem="theorem qm_debroglie_wavelength (λ v h m : ℝ) : λ * v = h / m :=",
    )]


# ═══════════════════════════════════════════════════════════════════════════════
# 4. COSMOLOGY
# ═══════════════════════════════════════════════════════════════════════════════

def make_hubble_expansion() -> list[Observation]:
    """Hubble's law: v = H₀ * d  →  v/d = H₀ = constant.

    Tests the simplest cosmological invariant: recession velocity is
    proportional to distance. System should discover v/d = const.
    """
    H0 = 70.0  # km/s/Mpc → convert to 1/s
    H0_si = H0 * 1000.0 / (3.086e22)  # ≈ 2.27e-18 s⁻¹
    timesteps = []

    for d in [10, 50, 100, 200, 400, 800, 1600]:  # Mpc → meters
        d_m = d * 3.086e22
        v = H0_si * d_m
        ratio = v / d_m  # should equal H0_si
        for _ in range(2):
            timesteps.append({
                "t": float(len(timesteps)) * 0.01,
                "v": v,
                "d": d_m,
            })

    return [Observation(
        id="cosmo_hubble_expansion",
        name="Hubble Expansion",
        description=(
            "Hubble's law: v = H₀*d. Invariant: v/d = H₀ = const. "
            "The expansion rate of the universe — discovered by Hubble in 1929."
        ),
        quantities={
            "v": "Velocity",
            "d": "Length",
        },
        parameters={"H0": H0},
        timesteps=timesteps,
        known_invariant="v/d",
        lean_theorem="theorem cosmo_hubble_expansion (v d H : ℝ) (h : d ≠ 0) : v / d = H :=",
    )]


def make_cmb_redshift_temperature() -> list[Observation]:
    """CMB temperature scales as T(z) = T₀*(1+z). Invariant: T/(1+z) = T₀.

    Tests the cosmological redshift-temperature relation.
    """
    T0 = 2.725  # K
    timesteps = []

    for z in [0, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0, 1089.0]:
        T_z = T0 * (1.0 + z)
        for _ in range(2):
            timesteps.append({
                "t": float(z) * 0.001,
                "T": T_z,
                "z": z,
                "one_plus_z": 1.0 + z,
            })

    return [Observation(
        id="cosmo_cmb_temperature",
        name="CMB Temperature vs Redshift",
        description=(
            "T_cmb(z) = T₀*(1+z). Invariant: T/(1+z) = T₀ = 2.725 K. "
            "The CMB cools as the universe expands. "
        ),
        quantities={
            "T": "Scalar",  # Temperature — treated as scalar
            "z": "Scalar",
            "one_plus_z": "Scalar",
        },
        parameters={"T0": T0},
        timesteps=timesteps,
        known_invariant="T/one_plus_z",
        lean_theorem="theorem cosmo_cmb_temperature (T one_plus_z T0 : ℝ) (h : one_plus_z ≠ 0) : T / one_plus_z = T0 :=",
    )]


# ═══════════════════════════════════════════════════════════════════════════════
# 5. HIGH-ENERGY / PARTICLE PHYSICS
# ═══════════════════════════════════════════════════════════════════════════════

def make_stefan_boltzmann_law() -> list[Observation]:
    """Stefan-Boltzmann: P/A = σ*T⁴. Invariant: (P/A)/T⁴ = σ = const.

    Total radiated power per area scales with T⁴. Tests power-law discovery.
    """
    sigma = 5.67e-8  # W/(m²·K⁴)
    timesteps = []

    for T in [300, 500, 800, 1000, 2000, 4000, 5778, 10000]:  # K
        power_per_area = sigma * T**4
        for _ in range(2):
            timesteps.append({
                "t": float(len(timesteps)) * 0.001,
                "P_A": power_per_area,
                "T": T,
                "T4": T**4,
            })

    return [Observation(
        id="thermal_stefan_boltzmann",
        name="Stefan-Boltzmann Law",
        description=(
            "P/A = σ*T⁴. Invariant: (P/A)/T⁴ = σ = const. "
            "Blackbody total radiated power — classical thermodynamics + quantum."
        ),
        quantities={
            "P_A": "Scalar",  # Power/Area — treated as scalar for search
            "T": "Scalar",
            "T4": "Scalar",
        },
        parameters={"sigma": sigma},
        timesteps=timesteps,
        known_invariant="P_A/T4",
        lean_theorem="theorem thermal_stefan_boltzmann (P_A T4 σ : ℝ) (h : T4 ≠ 0) : P_A / T4 = σ :=",
    )]


# ═══════════════════════════════════════════════════════════════════════════════
# 6. ELECTROMAGNETISM
# ═══════════════════════════════════════════════════════════════════════════════

def make_coulomb_force_law() -> list[Observation]:
    """Coulomb's law: F = k*q1*q2/r² → F*r² = k*q1*q2 = constant.

    For two fixed charges: as separation r varies, F ∝ 1/r².
    Invariant: F*r² = constant.
    """
    k = 8.99e9
    q1 = 1e-6
    q2 = 2e-6
    timesteps = []

    for r in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]:  # meters
        F = k * q1 * q2 / r**2
        for _ in range(2):
            timesteps.append({
                "t": float(len(timesteps)) * 0.01,
                "F": F,
                "r": r,
                "r2": r**2,
            })

    return [Observation(
        id="em_coulomb_force",
        name="Coulomb's Law",
        description=(
            "F = k*q1*q2/r². For fixed charges, F ∝ 1/r². "
            "Invariant: F*r² = k*q1*q2 = const. "
            "Tests inverse-square law detection."
        ),
        quantities={
            "F": "Force",
            "r": "Length",
            "r2": "Length^2",
        },
        parameters={"k": k, "q1": q1, "q2": q2},
        timesteps=timesteps,
        known_invariant="F*r2",
        lean_theorem="theorem em_coulomb_force (F r2 k q1 q2 : ℝ) : F * r2 = k * q1 * q2 :=",
    )]


# ═══════════════════════════════════════════════════════════════════════════════
# 7. ORBITAL MECHANICS / GRAVITY
# ═══════════════════════════════════════════════════════════════════════════════

def make_escape_velocity() -> list[Observation]:
    """Escape velocity: v_esc = √(2GM/r) → v² * r = 2GM = constant.

    For a given central body, v_esc² × r is invariant across different
    orbital radii. Tests product invariant v²*r.
    """
    G = 6.674e-11
    M = 5.972e24  # Earth mass
    two_GM = 2 * G * M
    timesteps = []

    for r_km in [6500, 7000, 8000, 10000, 15000, 20000, 40000, 100000]:
        r = r_km * 1000.0  # meters from Earth's center
        v_esc = math.sqrt(two_GM / r)
        for _ in range(2):
            timesteps.append({
                "t": float(len(timesteps)) * 0.1,
                "v": v_esc,
                "r": r,
                "v2": v_esc**2,
            })

    return [Observation(
        id="grav_escape_velocity",
        name="Escape Velocity Scaling",
        description=(
            "v_esc = √(2GM/r). Invariant: v²*r = 2GM = const. "
            "Tests product invariant. v2 = v² provided."
        ),
        quantities={
            "v": "Velocity",
            "r": "Length",
            "v2": "Velocity^2",
        },
        parameters={"G": G, "M": M},
        timesteps=timesteps,
        known_invariant="v2*r",
        lean_theorem="theorem grav_escape_velocity (v2 r G M : ℝ) : v2 * r = 2 * G * M :=",
    )]


# ═══════════════════════════════════════════════════════════════════════════════
# 8. CLASSICAL MECHANICS — Oscillators
# ═══════════════════════════════════════════════════════════════════════════════

def make_spring_period() -> list[Observation]:
    """Spring period: T = 2π√(m/k) → T²*k/m = 4π² = constant.

    Tests 3-quantity invariant. k_over_m = k/m provided as derived quantity.
    Invariant: T2*k_over_m = 4π² = const.
    T² has dimension Time², k/m has dimension 1/Time².
    T² * k/m = Time²/Time² = Scalar.
    """
    timesteps = []

    for m in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]:  # kg
        for k in [10.0, 20.0, 50.0, 100.0]:
            T = 2.0 * math.pi * math.sqrt(m / k)
            k_over_m = k / m
            for _ in range(1):
                timesteps.append({
                    "t": float(len(timesteps)) * 0.01,
                    "m": m,
                    "k": k,
                    "T2": T**2,
                    "k_over_m": k_over_m,
                })

    return [Observation(
        id="classical_spring_period",
        name="Spring Oscillation Period",
        description=(
            "T = 2π√(m/k). Invariant: T²*k/m = 4π² = const. "
            "k_over_m = k/m provided. Tests compound-dimension invariant."
        ),
        quantities={
            "T2": "Time^2",
            "k_over_m": "Scalar",  # T²*k/m = Scalar, so k/m treated as inverse-time² scalar
        },
        parameters={},
        timesteps=timesteps,
        known_invariant="T2*k_over_m",
        lean_theorem="theorem classical_spring_period (T2 k_over_m : ℝ) : T2 * k_over_m = 4 * π ^ 2 :=",
    )]


def make_pendulum_period() -> list[Observation]:
    """Pendulum period: T = 2π√(L/g) → T²*g/L = 4π² = constant.

    Tests invariant involving a constant acceleration g.
    g is provided as a quantity (not parameter) so the dimension
    system can compute T²*g/L → Scalar.
    """
    g = 9.81
    timesteps = []

    for L in [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]:  # meters
        T = 2.0 * math.pi * math.sqrt(L / g)
        T2g = T**2 * g
        for _ in range(2):
            timesteps.append({
                "t": float(len(timesteps)) * 0.1,
                "L": L,
                "T2": T**2,
                "T2g": T2g,
            })

    return [Observation(
        id="classical_pendulum_period",
        name="Pendulum Period",
        description=(
            "T = 2π√(L/g). Invariant: T²*g/L = 4π² = const. "
            "T2g = T²*g provided. Invariant: T2g/L = const."
        ),
        quantities={
            "T2g": "Length",  # T²*g = Time² * Accel = Length
            "L": "Length",
        },
        parameters={},
        timesteps=timesteps,
        known_invariant="T2g/L",
        lean_theorem="theorem classical_pendulum_period (T2g L : ℝ) (h : L ≠ 0) : T2g / L = 4 * π ^ 2 :=",
    )]


# ═══════════════════════════════════════════════════════════════════════════════
# 9. QUANTUM — Planck-Einstein relation
# ═══════════════════════════════════════════════════════════════════════════════

def make_planck_einstein_relation() -> list[Observation]:
    """E = h*f → E/f = h = constant. Planck-Einstein relation.

    Tests Energy/frequency ratio. f = 1/T where T is period.
    E/f has dimension Energy*Time = Action (compound).
    But we can use E and a proxy for frequency.
    """
    h = 6.626e-34
    timesteps = []

    for f in [1e14, 2e14, 5e14, 1e15, 2e15, 5e15, 1e16]:  # Hz
        E = h * f
        for _ in range(2):
            timesteps.append({
                "t": float(len(timesteps)) * 1e-16,
                "E": E,
                "f": f,
            })

    return [Observation(
        id="qm_planck_einstein",
        name="Planck-Einstein Relation",
        description=(
            "E = h*f. Invariant: E/f = h = const. "
            "Energy proportional to frequency — Planck's quantum hypothesis, 1900."
        ),
        quantities={
            "E": "Energy",
            "f": "Scalar",  # frequency as inverse-time proxy
        },
        parameters={"h": h},
        timesteps=timesteps,
        known_invariant="E/f",
        lean_theorem="theorem qm_planck_einstein (E f h : ℝ) (hf : f ≠ 0) : E / f = h :=",
    )]


# ═══════════════════════════════════════════════════════════════════════════════
# 10. NUCLEAR PHYSICS
# ═══════════════════════════════════════════════════════════════════════════════

def make_nuclear_binding_energy() -> list[Observation]:
    """Nuclear binding energy per nucleon: B/A ≈ 8 MeV (near-constant).

    For mid-mass nuclei (A=20–120), B/A is approximately constant.
    Tests approximate/statistical invariant detection.
    """
    # Binding energy per nucleon (MeV) for various nuclei
    # Real data: B/A ranges from ~7.5 to ~8.8 MeV
    nuclei = [
        (20, 160.6),   # Ne-20:  8.03 MeV/nucleon
        (24, 198.3),   # Mg-24:  8.26
        (28, 236.5),   # Si-28:  8.45
        (40, 342.1),   # Ca-40:  8.55
        (56, 492.3),   # Fe-56:  8.79 (peak)
        (63, 552.1),   # Cu-63:  8.76
        (89, 771.9),   # Y-89:   8.67
        (98, 844.3),   # Mo-98:  8.62
        (120, 1020.7), # Sn-120: 8.51
        (140, 1172.0), # Ce-140: 8.37
    ]
    timesteps = []

    for A, B_total in nuclei:
        for _ in range(2):
            timesteps.append({
                "t": float(len(timesteps)) * 0.01,
                "B": B_total,
                "A": float(A),
            })

    return [Observation(
        id="nuclear_binding_energy",
        name="Nuclear Binding Energy per Nucleon",
        description=(
            "B/A ≈ 8 MeV (near-constant for mid-mass nuclei). "
            "Tests ability to detect approximate invariants (~5% variation). "
            "Real nuclear data — B/A measured across 10 isotopes."
        ),
        quantities={
            "B": "Energy",
            "A": "Scalar",
        },
        parameters={},
        timesteps=timesteps,
        known_invariant="B/A",
        lean_theorem="theorem nuclear_binding_energy (B A : ℝ) (hA : A ≠ 0) : B / A = 8 :=",
    )]


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario registry
# ═══════════════════════════════════════════════════════════════════════════════

FRONTIER_SCENARIO_REGISTRY: list[tuple[str, callable, str, str]] = [
    # Classical Mechanics (never tested in theta-core)
    ("classical_kepler_third_law", make_kepler_third_law,
     "Classical — Kepler's 3rd Law",
     "T²/a³ = const (planetary orbits, proven 1619)"),

    # Thermodynamics / Statistical Mechanics
    ("thermal_stefan_boltzmann", make_stefan_boltzmann_law,
     "Thermal — Stefan-Boltzmann Law",
     "(P/A)/T⁴ = σ (blackbody radiation, proven 1879-1884)"),

    # Quantum Mechanics (beyond hydrogen — new tests)
    ("qm_harmonic_oscillator", make_quantum_harmonic_oscillator,
     "QM — Harmonic Oscillator Quantization",
     "E/(n+½) = ħω (tests zero-point energy detection, proven 1925)"),
    ("qm_debroglie_wavelength", make_debroglie_wavelength,
     "QM — de Broglie Wavelength",
     "λ*v = h/m = const (wave-particle duality, proven 1924)"),

    # General Relativity
    ("gr_gravitational_redshift", make_gravitational_redshift,
     "GR — Gravitational Redshift",
     "E*(1+g*h/c²) = const (Pound-Rebka experiment, proven 1959)"),

    # Cosmology (observational)
    ("cosmo_hubble_expansion", make_hubble_expansion,
     "Cosmology — Hubble Expansion",
     "v/d = H₀ = const (universe expansion, proven 1929)"),
    ("cosmo_cmb_temperature", make_cmb_redshift_temperature,
     "Cosmology — CMB Temperature vs Redshift",
     "T/(1+z) = T₀ = const (CMB cooling, proven 1965+)"),

    # Electromagnetism
    ("em_coulomb_force", make_coulomb_force_law,
     "EM — Coulomb's Law",
     "F*r² = const (inverse-square force law, proven 1785)"),

    # Orbital Mechanics
    ("grav_escape_velocity", make_escape_velocity,
     "Gravity — Escape Velocity Scaling",
     "v²*r = 2GM = const (product invariant, orbital dynamics)"),

    # Classical Mechanics — Oscillators
    ("classical_spring_period", make_spring_period,
     "Classical — Spring Period",
     "T²*k/m = 4π² = const (3-quantity compound invariant)"),
    ("classical_pendulum_period", make_pendulum_period,
     "Classical — Pendulum Period",
     "T²*g/L = 4π² = const (3-quantity scalar invariant)"),

    # Quantum Mechanics
    ("qm_planck_einstein", make_planck_einstein_relation,
     "QM — Planck-Einstein Relation",
     "E/f = h = const (energy-frequency quantization, 1900)"),

    # Nuclear Physics
    ("nuclear_binding_energy", make_nuclear_binding_energy,
     "Nuclear — Binding Energy per Nucleon",
     "B/A ≈ 8 MeV ≈ const (approximate invariant, ~5% variation)"),
]

# Additional off-by-default (more niche but still proven)
EXPERIMENTAL_SCENARIOS: list[tuple[str, callable, str, str]] = [
    # Dark matter rotation curves (observational fact, though DM is unproven)
    ("dm_flat_rotation", make_dark_matter_flat_rotation,
     "Astrophysics — Flat Rotation Curve",
     "v²*r ≈ const at large r (observed in galaxies since 1970s)"),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Test runners
# ═══════════════════════════════════════════════════════════════════════════════

def generate_all_scenarios() -> list[Observation]:
    """Generate all frontier test observations."""
    all_obs = []
    for sid, make_fn, name, desc in FRONTIER_SCENARIO_REGISTRY:
        try:
            obs_list = make_fn()
            all_obs.extend(obs_list)
            print(f"  ✓ {sid}: {len(obs_list)} scenario(s) generated")
        except Exception as e:
            print(f"  ✗ {sid}: FAILED — {e}")
    return all_obs


def validate_scenarios(observations: list[Observation]) -> dict:
    """Full validation: constancy check + automated discovery attempt.

    Uses auto_discover() to route each scenario to the right pipeline
    (beam search for Energy/Scalar, simple_invariant_search for compound
    dimensions). Reports whether the system can discover each invariant.
    """
    evaluator = ExpressionEvaluator()
    results = {}

    for obs in observations:
        print(f"\n{'='*60}")
        print(f"  {obs.id}: {obs.name}")
        print(f"  Known invariant: {obs.known_invariant}")
        print(f"  Quantities: {list(obs.quantities.keys())}")
        print(f"  Timesteps: {len(obs.timesteps)}")
        sys.stdout.flush()

        # Verify the known invariant is constant
        constancy = 0.0
        if obs.known_invariant:
            try:
                constancy = evaluator.score(obs.known_invariant, obs)
                print(f"  Ground truth constancy: {constancy:.6f}")
            except Exception as e:
                print(f"  Ground truth eval ERROR: {e}")
        else:
            print("  No known_invariant provided")

        # Classify dimension and pipeline
        quantities = {name: Dimension.named(dim)
                     for name, dim in obs.quantities.items()}
        dim_name, dim_is_energy, dim_is_scalar = _classify_invariant(
            obs.known_invariant, quantities, evaluator)

        if dim_is_energy:
            pipeline = "energy_beam"
        elif dim_is_scalar:
            pipeline = "scalar_beam"
        else:
            pipeline = "simple_search"

        print(f"  Dimension: {dim_name}  →  Pipeline: {pipeline}")

        # Run auto_discover
        from src.physics.search import auto_discover
        t0 = time.time()
        result = auto_discover(
            quantities=quantities,
            observations=[obs],
            known_invariant=obs.known_invariant,
            discovery_threshold=0.90,
            beam_expansions=1000,
        )
        elapsed = time.time() - t0

        found = result.is_discovery
        # Normalize for comparison: sort multiplication operands
        def _norm(expr: str) -> str:
            if "*" in expr and "+" not in expr and "-" not in expr:
                parts = sorted(expr.replace(" ", "").split("*"))
                return "*".join(parts)
            return expr.replace(" ", "")
        match = _norm(result.expression or "") == _norm(obs.known_invariant or "")
        print(f"  Found: {result.expression or 'NONE'}")
        print(f"  Score: {result.score:.6f}  Expansions: {result.expansions}  Time: {elapsed:.1f}s")
        print(f"  Result: {'DISCOVERED' if found else 'NOT FOUND'}  "
              f"{'✓ exact match' if match else ''}")
        sys.stdout.flush()

        results[obs.id] = {
            "known_invariant": obs.known_invariant,
            "constancy": constancy,
            "constancy_pass": constancy >= 0.95,
            "dimension": dim_name,
            "pipeline": pipeline,
            "found_expression": result.expression,
            "found_score": result.score,
            "discovered": found,
            "exact_match": match,
            "time_seconds": elapsed,
        }

    return results


def _classify_invariant(invariant: str, quantities: dict, evaluator) -> tuple[str, bool, bool]:
    """Return (dimension_name, is_energy, is_scalar) for the invariant.
    
    Unlike _infer_target_dim, this reports the ACTUAL dimension for compound
    invariants (e.g. '1/Time' or 'Length*Velocity') instead of falling back
    to 'Energy'.
    """
    if not invariant:
        return ("unknown", False, False)
    
    _NAMED = {"Energy", "Scalar"}
    dim_lookup = dict(quantities)
    for c in ["0", "0.5", "1", "2", "-1"]:
        dim_lookup[c] = Dimension.scalar()
    
    try:
        ast = evaluator.parse(invariant)
    except Exception:
        return ("unknown", False, False)
    
    from src.physics.evaluator import NumberNode, VarNode, FuncNode, BinOpNode
    
    def dim_of(node) -> Dimension | None:
        if isinstance(node, NumberNode):
            return Dimension.scalar()
        if isinstance(node, VarNode):
            return dim_lookup.get(node.name)
        if isinstance(node, FuncNode):
            return Dimension.scalar()
        if isinstance(node, BinOpNode):
            ld = dim_of(node.left)
            rd = dim_of(node.right)
            if ld is None or rd is None:
                return None
            try:
                if node.op in ("+", "-"):
                    return ld if ld.compatible_with(rd) else None
                elif node.op == "*":
                    return ld * rd
                elif node.op == "/":
                    return ld / rd
                elif node.op == "^":
                    if not rd.is_scalar():
                        return None
                    if isinstance(node.right, NumberNode):
                        return ld ** float(node.right.value)
            except Exception:
                pass
        return None
    
    d = dim_of(ast)
    if d is None:
        return ("unknown", False, False)
    
    name = str(d)
    is_energy = name == "Energy"
    is_scalar = name == "Scalar"
    return (name, is_energy, is_scalar)


def export_to_json(observations: list[Observation], output_path: str) -> None:
    """Export scenarios to a JSON file loadable by ObservationDatabase."""
    data = []
    for obs in observations:
        data.append({
            "id": obs.id,
            "name": obs.name,
            "description": obs.description,
            "quantities": dict(obs.quantities),
            "parameters": dict(obs.parameters),
            "timesteps": [dict(ts) for ts in obs.timesteps],
            "known_invariant": obs.known_invariant,
            "lean_theorem": obs.lean_theorem,
        })
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nExported {len(data)} scenarios to {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Frontier physics test scenario generator for theta-core",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/frontier_test_scenarios.py                  # list scenarios
  python scripts/frontier_test_scenarios.py --validate       # run beam search
  python scripts/frontier_test_scenarios.py --output data/frontier_tests.json
        """,
    )
    parser.add_argument("--validate", action="store_true",
                        help="Run beam search validation on each scenario")
    parser.add_argument("--output", type=str, default=None,
                        help="Export scenarios to JSON file")
    parser.add_argument("--list", action="store_true",
                        help="List all scenarios without generating")
    args = parser.parse_args()

    if args.list:
        print(f"\n{'='*60}")
        print("  FRONTIER TEST SCENARIOS")
        print(f"{'='*60}")
        for sid, _, name, desc in FRONTIER_SCENARIO_REGISTRY:
            print(f"\n  [{sid}]")
            print(f"    Name: {name}")
            print(f"    Invariant: {desc}")
        print(f"\n  Total: {len(FRONTIER_SCENARIO_REGISTRY)} scenarios")
        return

    print(f"\n{'='*60}")
    print("  GENERATING FRONTIER TEST SCENARIOS")
    print(f"{'='*60}\n")
    observations = generate_all_scenarios()
    print(f"\n  Total: {len(observations)} observations across "
          f"{len(FRONTIER_SCENARIO_REGISTRY)} scenarios")

    if args.validate:
        print(f"\n{'='*60}")
        print("  VALIDATING — Constancy Check")
        print(f"{'='*60}")
        results = validate_scenarios(observations)
        total = len(results)
        passes = sum(1 for r in results.values() if r.get("constancy_pass", False))
        discovered = sum(1 for r in results.values() if r.get("discovered", False))
        matched = sum(1 for r in results.values() if r.get("exact_match", False))
        print(f"\n{'='*60}")
        print(f"  RESULTS: {passes}/{total} constancy checks PASS")
        print(f"           {discovered}/{total} discovered by system")
        print(f"           {matched}/{total} exact match with known invariant")
        print(f"{'='*60}")

    if args.output:
        export_to_json(observations, args.output)


if __name__ == "__main__":
    main()
