#!/usr/bin/env python3
"""Spacetime ERA GATE — Grouped quantity metric discovery.

Tests the grouped quantity system with configurable era cutoff:
1. Train pre-cutoff only (domains, symmetries, scenarios from eras <= cutoff)
2. Show post-cutoff scenarios with hidden physics
3. System detects co-varying groups and discovers invariants

RUN:
  python scripts/spacetime_era_gate.py --era-cutoff 1905
  python scripts/spacetime_era_gate.py --era-cutoff 1920
  python scripts/spacetime_era_gate.py --era-cutoff 1950
  python scripts/spacetime_era_gate.py --era-cutoff 1970

OUTPUTS:
  checkpoints/grouped_quantity_detector_era{cutoff}.pt
  data/era_gate_{cutoff}_results.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.physics.dimensions import Dimension
from src.physics.evaluator import ExpressionEvaluator
from src.physics.observations import Observation
from src.physics.hidden_variables import (
    GroupedQuantityDetector,
    GroupedMetricProposer,
    GroupedMetricSearch,
    GroupedMetricDiscoveryResult,
    MetricProposal,
    load_grouped_metric_proposer,
    run_grouped_metric_discovery,
)
from src.physics.symmetry import run_spacetime_era_gate

# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════

CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "grouped_quantity_detector.pt"
RESULTS_PATH = PROJECT_ROOT / "data" / "spacetime_era_gate_results.json"
DISCOVERY_THRESHOLD = 0.90
SEED = 42

# ═══════════════════════════════════════════════════════════════════════════
# Era Domain Mapping — domains available at each physics era cutoff
# ═══════════════════════════════════════════════════════════════════════════

ERA_DOMAIN_MAP: dict[int, set[str]] = {
    1905: {"gravity", "spring", "collision", "classical_em", "thermodynamics"},
    1920: {"special_relativity", "early_quantum_bohr"},
    1950: {"qed", "nuclear_physics", "dirac_equation"},
    1970: {"standard_model", "qcd", "electroweak"},
    9999: {"higgs", "neutrino_oscillations", "dark_matter"},
}

# Human-readable era cutoff labels
ERA_CUTOFF_LABELS: dict[int, str] = {
    1905: "Classical (pre-1905)",
    1920: "Classical + SR + Early Quantum (pre-1920)",
    1950: "+ QED + Nuclear + Dirac (pre-1950)",
    1970: "+ Standard Model + QCD + Electroweak (pre-1970)",
    9999: "Today: + Higgs + Neutrinos + Dark Matter",
}


def get_domains_for_cutoff(cutoff_year: int) -> set[str]:
    """Return the set of domain names available at or before a cutoff."""
    domains: set[str] = set()
    for era_year, era_domains in sorted(ERA_DOMAIN_MAP.items()):
        if era_year <= cutoff_year:
            domains.update(era_domains)
    return domains


def get_cutoff_label(cutoff_year: int) -> str:
    """Human-readable label for an era cutoff year."""
    if cutoff_year in ERA_CUTOFF_LABELS:
        return ERA_CUTOFF_LABELS[cutoff_year]
    return f"Custom cutoff {cutoff_year}"


# Mapping of each test scenario to its physics discovery era
# Mapping of each test scenario to its physics discovery era
# Used to filter which scenarios are "post-cutoff" for evaluation
SCENARIO_ERA_MAP: dict[str, int] = {
    # 1905: Special Relativity
    "muon_time_dilation": 1905,
    "velocity_addition": 1905,
    "relativistic_momentum": 1905,
    "length_contraction": 1905,
    "doppler_shift": 1905,
    "twin_paradox": 1905,
    "mass_energy": 1905,
    "spacetime_interval": 1905,
    # 1950: QED + Dirac + Compton
    "qed_fine_structure": 1950,
    "compton_scattering": 1950,
    "dirac_spinor_norm": 1950,
    # 1970: QCD + Electroweak
    "qcd_asymptotic_freedom": 1970,
    "electroweak_mixing": 1970,
    # Today: Higgs + Neutrinos
    "higgs_mechanism": 9999,
    "neutrino_oscillation": 9999,
}


# ═══════════════════════════════════════════════════════════════════════════
# Post-1905 Test Scenario Generators
# ═══════════════════════════════════════════════════════════════════════════

def make_muon_time_dilation() -> list[Observation]:
    """Muon lifetime — time dilation: (c*t)^2 - x^2 is invariant."""
    c = 3e8
    tau0 = 2.2e-6
    timesteps = []
    velocities = [0, 0.3e8, 0.6e8, 0.9e8, 0.99e8, 0.995e8, 0.999e8]
    for v in velocities:
        if v < 1:
            v = 0
        gamma = 1.0 / math.sqrt(1.0 - (v / c) ** 2) if v < c else 10.0
        t_lab = gamma * tau0
        x_lab = v * t_lab
        for _ in range(3):
            timesteps.append({
                "t": t_lab, "x": x_lab, "v": v,
                "c": c, "tau": tau0, "gamma": gamma,
            })
    return [Observation(
        id="muon_time_dilation",
        name="Muon time dilation",
        description="Atmospheric muons: t = gamma*tau0, x = v*t. Invariant: (c*t)^2 - x^2",
        quantities={"t": "Time", "x": "Length", "c": "Velocity",
                    "v": "Velocity", "tau": "Time", "gamma": "Scalar"},
        parameters={"c": c, "tau0": tau0},
        timesteps=timesteps,
        known_invariant="(c*t)^2 - x^2",
        lean_theorem="",
    )]


def make_special_relativity_velocity_addition() -> list[Observation]:
    """Relativistic velocity addition: speeds don't add as v1+v2."""
    c = 3e8
    timesteps = []
    for v1 in [0.3e8, 0.6e8, 0.8e8, 0.9e8]:
        for v2 in [0.3e8, 0.6e8, 0.8e8]:
            v_rel = (v1 + v2) / (1.0 + v1 * v2 / c**2)
            t_obs = 1.0 / math.sqrt(1.0 - v_rel**2 / c**2) if v_rel < c else 10.0
            x_obs = v_rel * t_obs
            for _ in range(2):
                timesteps.append({
                    "t": t_obs, "x": x_obs, "v1": v1, "v2": v2, "c": c,
                })
    return [Observation(
        id="relativistic_velocity_addition",
        name="Velocity addition",
        description="v_rel = (v1+v2)/(1+v1*v2/c^2). Invariant: (c*t)^2 - x^2",
        quantities={"t": "Time", "x": "Length", "c": "Velocity",
                    "v1": "Velocity", "v2": "Velocity"},
        parameters={"c": c},
        timesteps=timesteps,
        known_invariant="(c*t)^2 - x^2",
        lean_theorem="",
    )]


def make_relativistic_momentum() -> list[Observation]:
    """p = gamma*m*v, E = gamma*m*c^2. Invariant: E^2 - (p*c)^2 = (m*c^2)^2."""
    c = 3e8
    m = 1.0
    timesteps = []
    for v in [0, 0.3e8, 0.6e8, 0.9e8, 0.99e8]:
        gamma = 1.0 / math.sqrt(1.0 - v**2 / c**2) if v < c else 10.0
        p = gamma * m * v
        E = gamma * m * c**2
        t_equiv = E / (m * c**2)
        x_equiv = p * t_equiv / E if E > 0 else 0
        for _ in range(3):
            timesteps.append({
                "t": t_equiv, "x": x_equiv, "v": v, "c": c,
                "p": p, "E": E, "m": m, "gamma": gamma,
            })
    return [Observation(
        id="relativistic_momentum",
        name="Relativistic momentum",
        description="p=gamma*m*v, E=gamma*m*c^2. Invariant: E^2 - (p*c)^2 = (m*c^2)^2",
        quantities={"t": "Time", "x": "Length", "c": "Velocity",
                    "v": "Velocity", "m": "Mass", "E": "Energy",
                    "p": "Momentum", "gamma": "Scalar"},
        parameters={"c": c, "m": m},
        timesteps=timesteps,
        known_invariant="E^2 - (p*c)^2",
        lean_theorem="",
    )]


def make_length_contraction() -> list[Observation]:
    """Length contraction: L = L0/gamma. Invariant: (c*t)^2 - x^2."""
    c = 3e8
    L0 = 10.0
    timesteps = []
    for v in [0, 0.3e8, 0.6e8, 0.9e8, 0.99e8]:
        gamma = 1.0 / math.sqrt(1.0 - v**2 / c**2) if v < c else 10.0
        L = L0 / gamma
        t_cross = L / v if v > 0 else float("inf")
        x_cross = L
        for _ in range(3):
            timesteps.append({
                "t": t_cross if t_cross < 1e-3 else 1e-7,
                "x": x_cross, "v": v, "c": c, "gamma": gamma,
                "L0": L0, "L": L,
            })
    return [Observation(
        id="length_contraction",
        name="Length contraction",
        description="L = L0/gamma. Invariant: (c*t)^2 - x^2",
        quantities={"t": "Time", "x": "Length", "c": "Velocity",
                    "v": "Velocity", "gamma": "Scalar"},
        parameters={"c": c, "L0": L0},
        timesteps=timesteps,
        known_invariant="(c*t)^2 - x^2",
        lean_theorem="",
    )]


def make_doppler_shift() -> list[Observation]:
    """Relativistic Doppler: f_obs = f0*sqrt((1+beta)/(1-beta))."""
    c = 3e8
    f0 = 1e15
    timesteps = []
    for v in [0.1e8, 0.3e8, 0.6e8, 0.9e8, -0.1e8, -0.3e8, -0.6e8]:
        beta = v / c
        if abs(beta) < 1.0:
            gamma = 1.0 / math.sqrt(1.0 - beta**2)
            t_equiv = gamma * 1.0
            x_equiv = v * t_equiv
            for _ in range(2):
                timesteps.append({
                    "t": t_equiv, "x": x_equiv, "v": v, "c": c, "gamma": gamma,
                    "f0": f0,
                })
    return [Observation(
        id="relativistic_doppler",
        name="Relativistic Doppler shift",
        description="Doppler: f_obs = f0*sqrt((1+beta)/(1-beta))."
                    " Invariant: (c*t)^2 - x^2",
        quantities={"t": "Time", "x": "Length", "c": "Velocity",
                    "v": "Velocity", "gamma": "Scalar"},
        parameters={"c": c, "f0": f0},
        timesteps=timesteps,
        known_invariant="(c*t)^2 - x^2",
        lean_theorem="",
    )]


def make_twin_paradox() -> list[Observation]:
    """Twin paradox: traveling twin ages less. Invariant: proper time."""
    c = 3e8
    timesteps = []
    for v in [0.3e8, 0.6e8, 0.9e8, 0.99e8]:
        gamma = 1.0 / math.sqrt(1.0 - v**2 / c**2) if v < c else 10.0
        t_earth = 10.0
        t_ship = t_earth / gamma
        x_ship = v * t_earth
        for _ in range(3):
            timesteps.append({
                "t": t_ship, "x": x_ship, "v": v, "c": c, "gamma": gamma,
                "t_earth": t_earth,
            })
    return [Observation(
        id="twin_paradox",
        name="Twin paradox",
        description="Traveler's proper time < Earth time. Invariant: (c*t)^2 - x^2",
        quantities={"t": "Time", "x": "Length", "c": "Velocity",
                    "v": "Velocity", "gamma": "Scalar"},
        parameters={"c": c, "t_earth": 10.0},
        timesteps=timesteps,
        known_invariant="(c*t)^2 - x^2",
        lean_theorem="",
    )]


def make_mass_energy_equivalence() -> list[Observation]:
    """E = m*c^2. Invariant: E/m = c^2 rest frame, (c*t)^2 - x^2 in motion."""
    c = 3e8
    timesteps = []
    masses = [1.0, 2.0, 5.0, 10.0]
    for m0 in masses:
        E0 = m0 * c**2
        for v in [0, 0.3e8, 0.6e8]:
            gamma = 1.0 / math.sqrt(1.0 - v**2 / c**2) if v < c else 10.0
            E = gamma * E0
            p = gamma * m0 * v
            t_equiv = E / E0
            x_equiv = p * t_equiv / E if E > 0 else 0
            for _ in range(2):
                timesteps.append({
                    "t": t_equiv, "x": x_equiv, "v": v, "c": c, "gamma": gamma,
                    "m": m0, "E": E, "p": p,
                })
    return [Observation(
        id="mass_energy_equivalence",
        name="E = m*c^2",
        description="Mass-energy equivalence. Invariant: E^2 - (p*c)^2 = (m*c^2)^2",
        quantities={"t": "Time", "x": "Length", "c": "Velocity",
                    "v": "Velocity", "m": "Mass", "E": "Energy",
                    "p": "Momentum", "gamma": "Scalar"},
        parameters={"c": c},
        timesteps=timesteps,
        known_invariant="E^2 - (p*c)^2",
        lean_theorem="",
    )]


def make_spacetime_interval_invariance() -> list[Observation]:
    """Direct test: ds^2 = (c*dt)^2 - dx^2 is invariant across frames."""
    c = 3e8
    timesteps = []
    events = [
        (0, 0, 0),
        (0.5e-6, 100, 0.5e8),
        (1.0e-6, 200, 0.8e8),
        (2.0e-6, 500, 0.9e8),
        (5.0e-6, 1400, 0.99e8),
    ]
    for t, x, v in events:
        gamma = 1.0 / math.sqrt(1.0 - v**2 / c**2) if v < c else 10.0
        for _ in range(3):
            timesteps.append({
                "t": t if t > 0 else 0.1e-6,
                "x": x, "v": v, "c": c, "gamma": gamma,
            })
    return [Observation(
        id="spacetime_interval",
        name="Spacetime interval invariance",
        description="ds^2 = (c*dt)^2 - dx^2 is Lorentz invariant",
        quantities={"t": "Time", "x": "Length", "c": "Velocity",
                    "v": "Velocity", "gamma": "Scalar"},
        parameters={"c": c},
        timesteps=timesteps,
        known_invariant="(c*t)^2 - x^2",
        lean_theorem="",
    )]


# ═══════════════════════════════════════════════════════════════════════════
# Post-1920 Test Scenarios (1950-era physics)
# Tested when cutoff >= 1920 — QED, Dirac, Compton
# ═══════════════════════════════════════════════════════════════════════════

def make_qed_fine_structure() -> list[Observation]:
    """Fine structure constant: alpha = e^2/(4*pi*eps0*hbar*c) — ratio invariant."""
    alpha = 1.0 / 137.036
    hbar_c = 197.3269804  # MeV*fm
    timesteps = []
    for e_val in [0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 1.5]:
        coupling = e_val * math.sqrt(4 * math.pi * alpha)
        for _ in range(3):
            timesteps.append({
                "t": float(len(timesteps) * 0.01),
                "e": e_val, "alpha": alpha,
                "coupling": coupling,
            })
    return [Observation(
        id="qed_fine_structure",
        name="QED Fine Structure Constant",
        description="alpha = e^2/(hbar*c). Invariant: e^2/hbar_c = constant",
        quantities={"alpha": "Scalar", "e": "Scalar", "coupling": "Scalar"},
        parameters={"hbar_c": hbar_c},
        timesteps=timesteps,
        known_invariant="e^2 / hbar_c",
        lean_theorem="",
    )]


def make_compton_scattering() -> list[Observation]:
    """Compton: dlambda = lambda_c*(1-cos(theta)). Invariant: dlambda/(1-cos(theta))."""
    lambda_c = 2.426e-12
    timesteps = []
    for theta in [0.2, 0.5, 1.0, 1.5, 2.0, 2.5, math.pi - 0.1]:
        dlambda = lambda_c * (1 - math.cos(theta))
        for _ in range(3):
            timesteps.append({
                "t": float(len(timesteps) * 0.01),
                "dlambda": dlambda, "theta": theta,
                "lambda_c": lambda_c,
                "one_minus_cos": 1.0 - math.cos(theta),
            })
    return [Observation(
        id="compton_scattering",
        name="Compton Scattering",
        description="dlambda = lambda_c*(1-cos(theta))."
                    " Invariant: dlambda/(1-cos(theta)) = lambda_c",
        quantities={"dlambda": "Length", "theta": "Scalar",
                    "one_minus_cos": "Scalar"},
        parameters={"lambda_c": lambda_c},
        timesteps=timesteps,
        known_invariant="dlambda / one_minus_cos",
        lean_theorem="",
    )]


def make_dirac_spinor_norm() -> list[Observation]:
    """Dirac spinor probability density: psi_dagger*psi = rho (constant)."""
    rho = 1.0
    timesteps = []
    for psi_re in [-0.8, -0.4, 0.0, 0.3, 0.6, 0.9]:
        psi_im = math.sqrt(max(0.001, rho - psi_re**2))
        for _ in range(3):
            timesteps.append({
                "t": float(len(timesteps) * 0.01),
                "psi_re": psi_re, "psi_im": psi_im,
                "norm": psi_re**2 + psi_im**2,
            })
    return [Observation(
        id="dirac_spinor_norm",
        name="Dirac Spinor Norm",
        description="psi_dagger*psi = |psi_re|^2 + |psi_im|^2 = rho."
                    " Invariant: sum of squares",
        quantities={"psi_re": "Scalar", "psi_im": "Scalar", "norm": "Scalar"},
        parameters={"rho": rho},
        timesteps=timesteps,
        known_invariant="psi_re^2 + psi_im^2",
        lean_theorem="",
    )]


# ═══════════════════════════════════════════════════════════════════════════
# Post-1950 Test Scenarios (1970-era physics)
# Tested when cutoff >= 1950 — QCD, Electroweak
# ═══════════════════════════════════════════════════════════════════════════

def make_qcd_asymptotic_freedom() -> list[Observation]:
    """QCD: alpha_s(Q) ~ 1/log(Q/Lambda_QCD). Invariant: alpha_s * log(Q/Lambda)."""
    Lambda = 0.217  # GeV
    timesteps = []
    for Q in [2.0, 5.0, 10.0, 20.0, 50.0, 100.0]:
        alpha_s = 1.0 / math.log(Q / Lambda)
        for _ in range(3):
            timesteps.append({
                "t": float(len(timesteps) * 0.01),
                "alpha_s": alpha_s, "Q": Q,
                "log_Q": math.log(Q / Lambda),
            })
    return [Observation(
        id="qcd_asymptotic_freedom",
        name="QCD Asymptotic Freedom",
        description="alpha_s(Q) ~ 1/log(Q/Lambda)."
                    " Invariant: alpha_s * log(Q/Lambda) ~ 1",
        quantities={"alpha_s": "Scalar", "Q": "Scalar", "log_Q": "Scalar"},
        parameters={"Lambda": Lambda},
        timesteps=timesteps,
        known_invariant="alpha_s * log_Q",
        lean_theorem="",
    )]


def make_electroweak_mixing() -> list[Observation]:
    """Electroweak: sin^2(theta_W) = g'^2/(g^2 + g'^2)."""
    theta_w = 0.4908  # sin^2(theta_W) ~ 0.231
    tan_w = math.tan(theta_w)
    timesteps = []
    for g in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        gp = g * tan_w
        ratio = gp / g
        for _ in range(3):
            timesteps.append({
                "t": float(len(timesteps) * 0.01),
                "g": g, "gp": gp,
                "ratio": ratio,
            })
    return [Observation(
        id="electroweak_mixing",
        name="Electroweak Mixing Angle",
        description="gp/g = tan(theta_W). Invariant: gp/g = constant",
        quantities={"g": "Scalar", "gp": "Scalar", "ratio": "Scalar"},
        parameters={"theta_w": theta_w},
        timesteps=timesteps,
        known_invariant="gp / g",
        lean_theorem="",
    )]


# ═══════════════════════════════════════════════════════════════════════════
# Post-1970 / Today Test Scenarios
# Tested when cutoff >= 1970 — Higgs, Neutrinos
# ═══════════════════════════════════════════════════════════════════════════

def make_higgs_mechanism() -> list[Observation]:
    """Higgs potential: V(phi) = lambda*(|phi|^2 - v^2/2)^2."""
    v = 246.0  # GeV
    lam = 0.13
    timesteps = []
    for phi in [0, 50, 100, 150, 200, 246, 300, 400]:
        V = lam * (phi**2 - v**2 / 2)**2
        for _ in range(2):
            timesteps.append({
                "t": float(len(timesteps) * 0.01),
                "phi": phi, "V": V,
                "phi_sq": phi**2,
                "v_sq_half": v**2 / 2,
            })
    return [Observation(
        id="higgs_mechanism",
        name="Higgs Mechanism",
        description="V(phi) = lambda*(phi^2 - v^2/2)^2."
                    " Invariant: phi^2 - v^2/2",
        quantities={"phi": "Scalar", "V": "Scalar",
                    "phi_sq": "Scalar", "v_sq_half": "Scalar"},
        parameters={"v": v, "lam": lam},
        timesteps=timesteps,
        known_invariant="phi^2 - v^2/2",
        lean_theorem="",
    )]


def make_neutrino_oscillation() -> list[Observation]:
    """Neutrino oscillation phase: phi = dm^2*L/(4*E)."""
    dm2 = 2.5e-3  # eV^2
    timesteps = []
    for E in [1.0, 2.0, 3.0, 5.0, 8.0, 10.0]:  # GeV
        for L in [10, 100, 300, 735, 1000]:  # km
            phase = dm2 * L / (4 * E)
            prob = math.sin(2 * 0.15)**2 * math.sin(phase)**2
            for _ in range(1):
                timesteps.append({
                    "t": float(len(timesteps) * 0.01),
                    "E": E, "L": L,
                    "phase": phase, "prob": prob,
                })
    return [Observation(
        id="neutrino_oscillation",
        name="Neutrino Oscillation",
        description="P(nu_e->nu_mu) = sin^2(2*theta)*sin^2(dm^2*L/(4*E))."
                    " Invariant: phase * E / L = dm^2/4",
        quantities={"E": "Scalar", "L": "Scalar",
                    "phase": "Scalar", "prob": "Scalar"},
        parameters={"dm2": dm2},
        timesteps=timesteps,
        known_invariant="phase * E / L",
        lean_theorem="",
    )]


# ═══════════════════════════════════════════════════════════════════════════
# All test scenarios — dynamically filtered by era cutoff
# ═══════════════════════════════════════════════════════════════════════════

ALL_SCENARIOS: list[tuple[str, Any, str, str]] = [
    # 1905-era: Special Relativity
    ("muon_time_dilation", make_muon_time_dilation, "Muon time dilation",
     "Time dilation: (c*t)^2-x^2"),
    ("velocity_addition", make_special_relativity_velocity_addition,
     "Velocity addition", "v_rel formula -> (c*t)^2-x^2"),
    ("relativistic_momentum", make_relativistic_momentum,
     "Relativistic momentum", "E^2-(p*c)^2 invariant"),
    ("length_contraction", make_length_contraction, "Length contraction",
     "L=L0/gamma -> (c*t)^2-x^2"),
    ("doppler_shift", make_doppler_shift, "Doppler shift",
     "Relativistic Doppler -> (c*t)^2-x^2"),
    ("twin_paradox", make_twin_paradox, "Twin paradox",
     "Twin aging -> (c*t)^2-x^2"),
    ("mass_energy", make_mass_energy_equivalence, "E=mc^2",
     "Mass-energy -> (c*t)^2-x^2"),
    ("spacetime_interval", make_spacetime_interval_invariance,
     "Spacetime interval", "ds^2 invariance"),
    # 1950-era: QED, Dirac, Compton
    ("qed_fine_structure", make_qed_fine_structure,
     "QED Fine Structure", "alpha=e^2/(hbar*c) invariant"),
    ("compton_scattering", make_compton_scattering,
     "Compton Scattering", "dlambda/(1-cos(theta))=lambda_c"),
    ("dirac_spinor_norm", make_dirac_spinor_norm,
     "Dirac Spinor Norm", "psi_dagger*psi=rho invariant"),
    # 1970-era: QCD, Electroweak
    ("qcd_asymptotic_freedom", make_qcd_asymptotic_freedom,
     "QCD Asymptotic Freedom", "alpha_s*log(Q/Lambda)~1"),
    ("electroweak_mixing", make_electroweak_mixing,
     "Electroweak Mixing", "gp/g=tan(theta_W) invariant"),
    # Today-era: Higgs, Neutrinos
    ("higgs_mechanism", make_higgs_mechanism,
     "Higgs Mechanism", "phi^2-v^2/2 invariant"),
    ("neutrino_oscillation", make_neutrino_oscillation,
     "Neutrino Oscillation", "phase*E/L=const invariant"),
]


# ═══════════════════════════════════════════════════════════════════════════
# Main test
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ScenarioResult:
    scenario_id: str
    scenario_name: str
    description: str
    group_detected: bool = False
    detected_groups: list[list[str]] = field(default_factory=list)
    best_metric: str | None = None
    best_expression: str | None = None
    best_constancy: float = 0.0
    discovered: bool = False
    spacetime_verified: bool = False
    errors: list[str] = field(default_factory=list)
    timing_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "scenario_name": self.scenario_name,
            "description": self.description,
            "group_detected": self.group_detected,
            "detected_groups": self.detected_groups,
            "best_metric": self.best_metric,
            "best_expression": self.best_expression,
            "best_constancy": self.best_constancy,
            "discovered": self.discovered,
            "spacetime_verified": self.spacetime_verified,
            "errors": self.errors,
            "timing_seconds": self.timing_seconds,
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Spacetime ERA GATE — configurable era knowledge gate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Era cutoffs:\n"
            "  1905  Classical (gravity, spring, EM, thermo) — test 1905+ SR\n"
            "  1920  + Special Relativity + Early Quantum — test 1950+\n"
            "  1950  + QED + Nuclear + Dirac — test 1970+\n"
            "  1970  + Standard Model + QCD + Electroweak — test Today+\n"
            "  9999  Today: + Higgs + Neutrinos + Dark Matter"
        ),
    )
    parser.add_argument(
        "--era-cutoff", type=int, default=1905,
        help="Physics era cutoff year (default: 1905). "
             "Training includes domains from eras <= cutoff. "
             "Testing includes scenarios from eras >= cutoff.",
    )
    args = parser.parse_args()
    cutoff_year: int = args.era_cutoff

    # Compute domain-aware paths
    # Backward compat: cutoff=1905 uses original checkpoint
    if cutoff_year == 1905:
        checkpoint_path = CHECKPOINT_PATH
        results_path = RESULTS_PATH
    else:
        checkpoint_path = (
            PROJECT_ROOT / "checkpoints"
            / f"grouped_quantity_detector_era{cutoff_year}.pt"
        )
        results_path = (
            PROJECT_ROOT / "data" / f"era_gate_{cutoff_year}_results.json"
        )
    training_domains = get_domains_for_cutoff(cutoff_year)
    cutoff_label = get_cutoff_label(cutoff_year)

    print("=" * 70)
    print("SPACETIME ERA GATE — Grouped Quantity Metric Discovery")
    print(f"Era cutoff: {cutoff_year} — {cutoff_label}")
    print(f"Training domains (pre-{cutoff_year}): {sorted(training_domains)}")
    print("=" * 70)
    print()

    # Filter test scenarios: test if era >= cutoff, exclude if era < cutoff
    test_scenarios = [
        (sid, fn, name, desc)
        for sid, fn, name, desc in ALL_SCENARIOS
        if SCENARIO_ERA_MAP.get(sid, 1905) >= cutoff_year
    ]
    excluded_scenarios = [
        (sid, name)
        for sid, fn, name, desc in ALL_SCENARIOS
        if SCENARIO_ERA_MAP.get(sid, 1905) < cutoff_year
    ]

    if excluded_scenarios:
        print(f"Training-absorbed scenarios (pre-{cutoff_year}):")
        for sid, name in excluded_scenarios:
            print(f"  - {name} (era {SCENARIO_ERA_MAP.get(sid, 1905)})")
        print()

    print(f"Test scenarios (era >= {cutoff_year}): {len(test_scenarios)}")

    # Load trained proposer
    print(f"\nLoading proposer from {checkpoint_path}...")
    proposer = None
    if checkpoint_path.exists():
        try:
            proposer = load_grouped_metric_proposer(str(checkpoint_path), device="cpu")
            print(f"  Loaded with {proposer.count_parameters()} parameters")
        except RuntimeError as e:
            print(f"  Checkpoint architecture mismatch: {e}")
            print("  Will train fresh model.")
    if proposer is None:
        print("  No valid checkpoint found — training from scratch")
        from src.physics.hidden_variables import train_grouped_metric_proposer
        proposer = train_grouped_metric_proposer(
            epochs=150, lr=0.003, device="cpu",
            checkpoint_path=str(checkpoint_path),
            era_cutoff=cutoff_year,
        )

    # Run test scenarios
    all_results: list[ScenarioResult] = []

    if not test_scenarios:
        print(f"\n  No post-{cutoff_year} test scenarios available.")
        print(f"  All known scenarios are pre-{cutoff_year} (training domain).")
    else:
        for scenario_id, make_fn, name, desc in test_scenarios:
            print(f"\n{'─' * 60}")
            print(f"Scenario: {name}")
            print(f"  {desc}")
            t0 = time.time()

            try:
                observations = make_fn()
                quantity_dict = {}
                for obs in observations:
                    for qname, qdim in obs.quantities.items():
                        if qname not in quantity_dict:
                            quantity_dict[qname] = Dimension.named(qdim)

                result = run_spacetime_era_gate(
                    observations=observations,
                    quantity_dict=quantity_dict,
                    proposer=proposer,
                    discovery_threshold=DISCOVERY_THRESHOLD,
                )
            except Exception as e:
                result = {"accepted": False, "error": str(e), "all_results": []}

            elapsed = time.time() - t0
            sr = ScenarioResult(
                scenario_id=scenario_id,
                scenario_name=name,
                description=desc,
                timing_seconds=elapsed,
            )

            if "error" in result:
                sr.errors.append(str(result["error"]))
                print(f"  ERROR: {result['error']}")
            else:
                sr.best_constancy = result.get("best_constancy", 0)
                sr.best_expression = result.get("best_expression")
                sr.best_metric = result.get("best_metric")
                sr.discovered = result.get("spacetime_discovered", False)
                sr.spacetime_verified = result.get("accepted", False)

                for r in result.get("all_results", []):
                    sr.detected_groups.append(r.get("group", []))
                    if any("t" in g and "x" in g for g in sr.detected_groups):
                        sr.group_detected = True

                print(f"  Groups detected: {sr.detected_groups}")
                print(f"  Best metric: {sr.best_metric}")
                print(f"  Best expression: {sr.best_expression}")
                print(f"  Best constancy: {sr.best_constancy:.4f}")
                print(f"  Spacetime verified: {sr.spacetime_verified}")
                print(f"  Time: {elapsed:.2f}s")

            all_results.append(sr)

    # Summary
    print(f"\n{'=' * 70}")
    print("RESULTS SUMMARY")
    print(f"  Era cutoff: {cutoff_year} ({cutoff_label})")
    print(f"  Training domains ({len(training_domains)}):"
          f" {sorted(training_domains)}")
    print(f"{'=' * 70}")
    verified = sum(1 for r in all_results if r.spacetime_verified)
    discovered_any = sum(1 for r in all_results if r.discovered)
    group_detected = sum(1 for r in all_results if r.group_detected)
    print(f"  Spacetime verified: {verified}/{len(all_results)}")
    print(f"  Group detected: {group_detected}/{len(all_results)}")
    print(f"  Any discovery: {discovered_any}/{len(all_results)}")

    for sr in all_results:
        status = "✓" if sr.spacetime_verified else "✗"
        print(f"  {status} {sr.scenario_id:<25s} const={sr.best_constancy:.3f}  "
              f"expr={sr.best_expression or 'N/A':<30s}")

    # Save results
    output = {
        "experiment": "SPACETIME_ERA_GATE_CONFIGURABLE",
        "description": (
            f"Pre-{cutoff_year} training -> post-{cutoff_year} metric discovery"
        ),
        "era_cutoff": cutoff_year,
        "cutoff_label": cutoff_label,
        "training_domains": sorted(training_domains),
        "test_scenarios_count": len(test_scenarios),
        "excluded_scenarios": [
            {"id": sid, "name": name, "era": SCENARIO_ERA_MAP.get(sid, 1905)}
            for sid, name in excluded_scenarios
        ],
        "not_taught": "Domain physics from eras after the cutoff",
        "proposer_params": proposer.count_parameters(),
        "discovery_threshold": DISCOVERY_THRESHOLD,
        "total_scenarios": len(all_results),
        "spacetime_verified": verified,
        "scenarios": [r.to_dict() for r in all_results],
    }

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
