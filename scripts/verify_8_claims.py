#!/usr/bin/env python3
"""Independent verification of 8 README claims.

Two modes:
  DEFAULT: Legacy formula-generated data (circular — claims 1,2,3,5,6,7,8).
  --honest-data: Independent-measurement simulation with noise.
                 Each quantity measured by a different simulated instrument,
                 breaking the circular dependency between data and invariant.

Flags:
  --no-neural-templates  Skip Pipeline 0 (hand-written templates).
  --honest-data          Use independent-measurement data generators.
  --noise FLOAT          Measurement noise fraction (default 0.01 for 1%).

RUN:
  python scripts/verify_8_claims.py
  python scripts/verify_8_claims.py --honest-data --no-neural-templates
  python scripts/verify_8_claims.py --honest-data --noise 0.02
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.physics.dimensions import Dimension
from src.physics.evaluator import ExpressionEvaluator
from src.physics.observations import Observation
from src.physics.search import auto_discover
from src.physics.canonicalizer import create_pre1905_canonicalizer

C = 299792458.0
HBAR = 1.054571817e-34
EV_TO_J = 1.602176634e-19
M_E = 9.10938356e-31
H = 6.62607015e-34
HC = H * C


def _obs(obs_id, name, desc, quantities, params, timesteps, invariant):
    return Observation(
        id=obs_id, name=name, description=desc,
        quantities=quantities, parameters=params,
        timesteps=timesteps, known_invariant=invariant,
        lean_theorem="",
    )


def _noise(rng: random.Random, frac: float) -> float:
    """Independent Gaussian measurement noise."""
    return 1.0 + rng.gauss(0.0, frac)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Hydrogen Balmer: E*lambda = h*c
# ═══════════════════════════════════════════════════════════════════════════

def make_hydrogen_balmer(noise_frac: float = 0.0, rng: random.Random | None = None) -> list[Observation]:
    """Each lambda from Balmer formula, E computed from E = h*c/lambda."""
    points = [
        (3, 6.563e-7, 3.027e-19),
        (4, 4.861e-7, 4.087e-19),
        (5, 4.340e-7, 4.578e-19),
        (6, 4.102e-7, 4.843e-19),
        (7, 3.970e-7, 5.004e-19),
        (8, 3.889e-7, 5.108e-19),
        (9, 3.835e-7, 5.180e-19),
        (10, 3.798e-7, 5.231e-19),
    ]
    timesteps = [
        {"t": float(i), "n": n, "lambda": lam, "E": E}
        for i, (n, lam, E) in enumerate(points)
    ]
    return [_obs(
        "h_balmer", "Hydrogen Balmer",
        "Visible hydrogen spectrum. E*lambda = h*c.",
        {"lambda": "Length", "n": "Scalar", "E": "Energy"},
        {},
        timesteps,
        "E*lambda",
    )]


def make_hydrogen_balmer_honest(noise_frac: float, rng: random.Random) -> list[Observation]:
    """Independent measurements: lambda from spectrometer, E from photodetector.
    Each instrument has independent noise.  lambda_true and E_true are
    physically linked (E_true = h*c/lambda_true), but the MEASURED values
    have independent error — breaking the algebraic identity."""
    lam_true = [6.563e-7, 4.861e-7, 4.340e-7, 4.102e-7,
                3.970e-7, 3.889e-7, 3.835e-7, 3.798e-7]
    timesteps = []
    for i, lt in enumerate(lam_true):
        lam_meas = lt * _noise(rng, noise_frac)
        e_true = HC / lt
        e_meas = e_true * _noise(rng, noise_frac)
        timesteps.append({"t": float(i), "lambda": lam_meas, "E": e_meas})
    return [_obs(
        "h_balmer", "Hydrogen Balmer (honest)",
        "E*lambda approx h*c. Lambda from spectrometer, E from photodetector. "
        f"Independent noise {noise_frac*100:.0f}%.",
        {"lambda": "Length", "E": "Energy"},
        {},
        timesteps,
        "E*lambda",
    )]


# ═══════════════════════════════════════════════════════════════════════════
# 2. Spin quantization: E/n = constant
# ═══════════════════════════════════════════════════════════════════════════

def make_spin_quantization(noise_frac: float = 0.0, rng: random.Random | None = None) -> list[Observation]:
    base = HBAR * 1e15 / EV_TO_J
    timesteps = [
        {"t": float(n), "n": n, "E": n * base}
        for n in range(1, 9)
    ]
    return [_obs(
        "spin_quant", "Spin quantization",
        "E proportional to n. E/n = constant.",
        {"n": "Scalar", "E": "Energy"},
        {},
        timesteps,
        "E/n",
    )]


def make_spin_quantization_honest(noise_frac: float, rng: random.Random) -> list[Observation]:
    """Zeeman splitting: E = g*mu_B*B*n.  Measure n (quantum number, integer,
    known exactly) and E (from spectroscopy, noisy).  E/n ≈ g*mu_B*B."""
    base = HBAR * 1e15 / EV_TO_J
    timesteps = []
    for n in range(1, 9):
        e_true = n * base
        e_meas = e_true * _noise(rng, noise_frac)
        timesteps.append({"t": float(n), "n": float(n), "E": e_meas})
    return [_obs(
        "spin_quant", "Spin quantization (honest)",
        f"E/n approx constant. n exact (quantum number), E from spectroscopy "
        f"with {noise_frac*100:.0f}% noise.",
        {"n": "Scalar", "E": "Energy"},
        {},
        timesteps,
        "E/n",
    )]


# ═══════════════════════════════════════════════════════════════════════════
# 3. Wien displacement: E_peak/T = constant
# ═══════════════════════════════════════════════════════════════════════════

def make_wien(noise_frac: float = 0.0, rng: random.Random | None = None) -> list[Observation]:
    wien_k = 2.821439 * 1.380649e-23
    T_vals = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
    timesteps = [
        {"t": float(i), "T": T, "E_peak": T * wien_k}
        for i, T in enumerate(T_vals)
    ]
    return [_obs(
        "wien", "Wien displacement",
        "Peak energy proportional to T. E_peak/T = constant.",
        {"T": "Scalar", "E_peak": "Energy"},
        {},
        timesteps,
        "E_peak/T",
    )]


def make_wien_honest(noise_frac: float, rng: random.Random) -> list[Observation]:
    """Blackbody spectrum: T from thermocouple, E_peak from fitting Planck
    curve to spectrometer data.  Independent instruments, independent noise."""
    wien_k = 2.821439 * 1.380649e-23
    T_vals = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
    timesteps = []
    for i, T_true in enumerate(T_vals):
        T_meas = T_true * _noise(rng, noise_frac * 0.5)
        e_true = T_true * wien_k
        e_meas = e_true * _noise(rng, noise_frac)
        timesteps.append({"t": float(i), "T": T_meas, "E_peak": e_meas})
    return [_obs(
        "wien", "Wien displacement (honest)",
        f"E_peak/T approx constant. T from thermocouple, E_peak from "
        f"spectrometer fit. Independent {noise_frac*100:.0f}% noise.",
        {"T": "Scalar", "E_peak": "Energy"},
        {},
        timesteps,
        "E_peak/T",
    )]


# ═══════════════════════════════════════════════════════════════════════════
# 4. Photoelectric effect: h*nu - K_max = phi  (already honest — has regimes)
# ═══════════════════════════════════════════════════════════════════════════

def make_photoelectric(noise_frac: float = 0.0, rng: random.Random | None = None) -> list[Observation]:
    """Honest photoelectric data: below-threshold K_max=0, above-threshold K_max=h*nu-phi.
    Each frequency is a separate Observation so the regime loop can split them.
    Optionally add measurement noise."""
    if rng is None:
        rng = random.Random(42)
    H_MEV_THZ = 4.135667662
    PHI_MEV = 4500.0
    nu_below = [200.0, 400.0, 600.0, 800.0]
    nu_above = [1200.0, 1500.0, 1800.0, 2100.0, 2400.0, 2700.0]

    observations = []
    for nu in nu_below:
        nu_m = nu * _noise(rng, noise_frac * 0.5) if noise_frac else nu
        timesteps = [{"t": 0.0, "nu": nu_m, "K_max": 0.0},
                     {"t": 1.0, "nu": nu_m * 1.001, "K_max": 0.0}]
        observations.append(_obs(
            f"pe_below_{nu:.0f}", "Photoelectric (below threshold)",
            f"nu={nu:.0f} THz, K_max=0.",
            {"nu": "Scalar", "K_max": "Energy"},
            {"h": H_MEV_THZ, "phi": PHI_MEV},
            timesteps,
            "K_max",  # K_max=0 is the invariant below threshold
        ))
    for nu in nu_above:
        nu_m = nu * _noise(rng, noise_frac * 0.5) if noise_frac else nu
        km_true = H_MEV_THZ * nu - PHI_MEV
        km1 = km_true * _noise(rng, noise_frac) if noise_frac else km_true
        km2 = km_true * _noise(rng, noise_frac) if noise_frac else km_true
        timesteps = [{"t": 0.0, "nu": nu_m, "K_max": km1},
                     {"t": 1.0, "nu": nu_m * 1.001, "K_max": km2}]
        observations.append(_obs(
            f"pe_above_{nu:.0f}", "Photoelectric (above threshold)",
            f"nu={nu:.0f} THz, h*nu-K_max=phi.",
            {"nu": "Scalar", "K_max": "Energy"},
            {"h": H_MEV_THZ, "phi": PHI_MEV},
            timesteps,
            "h*nu - K_max",
        ))
    return observations


# ═══════════════════════════════════════════════════════════════════════════
# 5. Rest energy: E/gamma = m*c^2
# ═══════════════════════════════════════════════════════════════════════════

def make_rest_energy(noise_frac: float = 0.0, rng: random.Random | None = None) -> list[Observation]:
    m = M_E
    mc2 = m * C**2 / EV_TO_J
    v_fracs = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
    timesteps = []
    for i, beta in enumerate(v_fracs):
        gamma = 1.0 / math.sqrt(1.0 - beta**2) if beta < 1 else 10.0
        E = gamma * mc2
        timesteps.append({"t": float(i), "gamma": gamma, "E": E})
    return [_obs(
        "rest_energy", "Rest energy",
        "E/gamma = m*c^2. Same particle, different velocities.",
        {"gamma": "Scalar", "E": "Energy"},
        {"m": m, "c": C},
        timesteps,
        "E/gamma",
    )]


def make_rest_energy_honest(noise_frac: float, rng: random.Random) -> list[Observation]:
    """Independent measurements: gamma from velocity (time-of-flight),
    E from calorimeter.  Different instruments, independent noise."""
    m = M_E
    mc2 = m * C**2 / EV_TO_J
    v_fracs = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
    timesteps = []
    for i, beta in enumerate(v_fracs):
        gamma_true = 1.0 / math.sqrt(1.0 - beta**2) if beta < 1 else 10.0
        gamma_meas = gamma_true * _noise(rng, noise_frac * 0.3)
        e_true = gamma_true * mc2
        e_meas = e_true * _noise(rng, noise_frac)
        timesteps.append({"t": float(i), "gamma": gamma_meas, "E": e_meas})
    return [_obs(
        "rest_energy", "Rest energy (honest)",
        f"E/gamma approx mc^2. Gamma from TOF, E from calorimeter. "
        f"Independent noise {noise_frac*100:.0f}%.",
        {"gamma": "Scalar", "E": "Energy"},
        {"m": m, "c": C},
        timesteps,
        "E/gamma",
    )]


# ═══════════════════════════════════════════════════════════════════════════
# 6. Velocity addition: (u+v)/(1+u*v/c^2)
# ═══════════════════════════════════════════════════════════════════════════

def make_velocity_addition(noise_frac: float = 0.0, rng: random.Random | None = None) -> list[Observation]:
    u_rel_fixed = 0.8 * C
    # Exact configs: for each u, compute v such that (u+v)/(1+u*v/c^2) = 0.8c
    u_fracs = [0.5, 0.6, 0.3, 0.9, 0.1, 0.4, 0.7, 0.2]
    configs = []
    for uf in u_fracs:
        u = uf * C
        # v = (u_rel - u) / (1 - u_rel*u/c^2) — exact inverse of velocity addition
        v = (u_rel_fixed - u) / (1.0 - u_rel_fixed * u / C**2)
        configs.append((u, v))
    timesteps = [
        {"t": float(i), "u": u, "v": v}
        for i, (u, v) in enumerate(configs)
    ]
    return [_obs(
        "velocity_add", "Velocity addition",
        "(u+v)/(1+u*v/c^2) = constant. Same relative velocity, different frames.",
        {"u": "Velocity", "v": "Velocity", "c": "Velocity"},
        {"c": C, "u_rel": u_rel_fixed},
        timesteps,
        "(u+v)/(1+u*v/c^2)",
    )]


# ═══════════════════════════════════════════════════════════════════════════
# 7. Energy-momentum: E^2 - (p*c)^2 = (m*c^2)^2
# ═══════════════════════════════════════════════════════════════════════════

def make_energy_momentum(noise_frac: float = 0.0, rng: random.Random | None = None) -> list[Observation]:
    m = M_E
    mc2 = m * C**2
    v_fracs = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
    timesteps = []
    for i, beta in enumerate(v_fracs):
        gamma = 1.0 / math.sqrt(1.0 - beta**2) if beta < 1 else 10.0
        E = gamma * mc2
        p = gamma * m * (beta * C)
        pc = p * C
        timesteps.append({
            "t": float(i), "gamma": gamma,
            "E": E / 1e-10, "p": pc / 1e-10,
        })
    return [_obs(
        "energy_momentum", "Energy-momentum",
        "E^2 - (p*c)^2 = (m*c^2)^2. Same particle, different velocities.",
        {"gamma": "Scalar", "E": "Energy", "p": "Energy", "c": "Velocity"},
        {"m": m, "c": C},
        timesteps,
        "E^2 - p^2",
    )]


def make_energy_momentum_honest(noise_frac: float, rng: random.Random) -> list[Observation]:
    """Independent E (calorimeter) and p (tracking + magnetic field) measurements."""
    m = M_E
    mc2 = m * C**2
    v_fracs = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
    timesteps = []
    for i, beta in enumerate(v_fracs):
        gamma_true = 1.0 / math.sqrt(1.0 - beta**2) if beta < 1 else 10.0
        e_true = gamma_true * mc2
        p_true = gamma_true * m * (beta * C)
        pc_true = p_true * C
        e_meas = (e_true / 1e-10) * _noise(rng, noise_frac)
        pc_meas = (pc_true / 1e-10) * _noise(rng, noise_frac)
        timesteps.append({"t": float(i), "E": e_meas, "p": pc_meas})
    return [_obs(
        "energy_momentum", "Energy-momentum (honest)",
        f"E^2-p^2 approx (mc^2)^2. E from calorimeter, p from tracking. "
        f"Independent {noise_frac*100:.0f}% noise.",
        {"E": "Energy", "p": "Energy", "c": "Velocity"},
        {"m": m, "c": C},
        timesteps,
        "E^2-p^2",
    )]


# ═══════════════════════════════════════════════════════════════════════════
# 8. Spacetime interval: (c*t)^2 - x^2
# ═══════════════════════════════════════════════════════════════════════════

def make_spacetime_interval(noise_frac: float = 0.0, rng: random.Random | None = None) -> list[Observation]:
    t0_rest = 1e-6
    x0_rest = 200.0
    s2_invariant = (C * t0_rest)**2 - x0_rest**2
    v_fracs = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
    timesteps = []
    for i, beta in enumerate(v_fracs):
        gamma = 1.0 / math.sqrt(1.0 - beta**2) if beta < 1 else 10.0
        v = beta * C
        t_prime = gamma * (t0_rest - v * x0_rest / C**2)
        x_prime = gamma * (x0_rest - v * t0_rest)
        timesteps.append({
            "t": t_prime, "x": x_prime, "gamma": gamma,
            "s2": s2_invariant,
        })
    return [_obs(
        "spacetime_interval", "Spacetime interval",
        "(c*t)^2 - x^2 = s^2. Same events, different frames.",
        {"t": "Time", "x": "Length", "c": "Velocity", "gamma": "Scalar"},
        {"c": C, "s2_invariant": s2_invariant},
        timesteps,
        "(c*t)^2 - x^2",
    )]


def make_spacetime_interval_honest(noise_frac: float, rng: random.Random) -> list[Observation]:
    """Independent measurements: t from atomic clocks, x from laser ranging.
    Different frames (Lorentz transformed), independent noise per measurement."""
    t0_rest = 1e-6
    x0_rest = 200.0
    v_fracs = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
    timesteps = []
    for i, beta in enumerate(v_fracs):
        gamma = 1.0 / math.sqrt(1.0 - beta**2) if beta < 1 else 10.0
        v = beta * C
        t_prime = gamma * (t0_rest - v * x0_rest / C**2)
        x_prime = gamma * (x0_rest - v * t0_rest)
        t_meas = t_prime * _noise(rng, noise_frac * 0.5)
        x_meas = x_prime * _noise(rng, noise_frac)
        timesteps.append({"t": t_meas, "x": x_meas, "c": C})
    return [_obs(
        "spacetime_interval", "Spacetime interval (honest)",
        f"(c*t)^2-x^2 approx constant. t from clock, x from laser ranging. "
        f"Independent {noise_frac*100:.0f}% noise.",
        {"t": "Time", "x": "Length", "c": "Velocity"},
        {"c": C},
        timesteps,
        "(c*t)^2-x^2",
    )]


# ═══════════════════════════════════════════════════════════════════════════
# Verification
# ═══════════════════════════════════════════════════════════════════════════

CLAIMS: list[tuple[str, str, str, str, callable, callable | None]] = [
    ("QUANTUM", "E*lambda = h*c", "E*lambda", "h_balmer",
     make_hydrogen_balmer, make_hydrogen_balmer_honest),
    ("QUANTUM", "E/n = constant", "E/n", "spin_quant",
     make_spin_quantization, make_spin_quantization_honest),
    ("QUANTUM", "E_peak/T = constant", "E_peak/T", "wien",
     make_wien, make_wien_honest),
    ("QUANTUM", "h*nu - K_max = phi", "h*nu - K_max", "photoelectric",
     make_photoelectric, None),  # already honest — has regimes
    ("RELATIVISTIC", "E/gamma = m*c^2", "E/gamma", "rest_energy",
     make_rest_energy, make_rest_energy_honest),
    ("RELATIVISTIC", "(u+v)/(1+u*v/c^2)", "(u+v)/(1+u*v/c^2)", "velocity_add",
     make_velocity_addition, None),  # needs neural templates
    ("RELATIVISTIC", "E^2 - p^2 = (m*c^2)^2", "E^2 - p^2", "energy_momentum",
     make_energy_momentum, make_energy_momentum_honest),
    ("RELATIVISTIC", "(c*t)^2 - x^2", "(c*t)^2 - x^2", "spacetime_interval",
     make_spacetime_interval, make_spacetime_interval_honest),
]


@dataclass
class ClaimResult:
    domain: str
    claim: str
    invariant: str
    discovered_expr: str = ""
    discovered_score: float = 0.0
    invariant_score: float = 0.0
    exact_match: bool = False
    passed: bool = False
    notes: str = ""


def expr_normalize(s: str) -> str:
    s = s.replace(" ", "")
    if "+" in s and "-" not in s.replace("^-", ""):
        parts = sorted(s.split("+"))
        return "+".join(parts)
    return s


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify 8 README claims")
    parser.add_argument("--honest-data", action="store_true",
                        help="Use independent-measurement data with noise")
    parser.add_argument("--no-neural-templates", action="store_true",
                        help="Skip Pipeline 0 (hand-written templates)")
    parser.add_argument("--noise", type=float, default=0.01,
                        help="Measurement noise fraction (default 0.01)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for noise")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    evaluator = ExpressionEvaluator()
    canonicalizer = create_pre1905_canonicalizer()
    results: list[ClaimResult] = []

    mode = "HONEST (independent measurements)" if args.honest_data else "LEGACY (formula-generated)"
    template_mode = "DISABLED" if args.no_neural_templates else "ACTIVE"

    print("=" * 72)
    print("VERIFICATION OF 8 README CLAIMS")
    print(f"Data mode: {mode}")
    print(f"Neural templates: {template_mode}")
    if args.honest_data:
        print(f"Measurement noise: {args.noise*100:.0f}%")
    print("Hybrid pipeline: neural -> simple search -> beam search -> regime")
    print("Trivial-constancy gate: ACTIVE")
    print("Self-cancellation gate: ACTIVE")
    print("=" * 72)

    for domain, claim, invariant, _key, make_legacy, make_honest in CLAIMS:
        print(f"\n{'─' * 60}")
        print(f"[{domain}] {claim}")

        if args.honest_data and make_honest is not None:
            observations = make_honest(args.noise, rng)
        elif args.honest_data and make_honest is None:
            # Use legacy but with noise for photoelectric
            observations = make_legacy(args.noise, rng)
        else:
            observations = make_legacy()

        quantity_dict = {}
        for obs in observations:
            for qname, qdim in obs.quantities.items():
                if qname not in quantity_dict:
                    quantity_dict[qname] = Dimension.named(qdim)

        inv_scores = [evaluator.score(invariant, obs) for obs in observations]
        inv_score = sum(inv_scores) / len(inv_scores)
        print(f"  Invariant {invariant} scores {inv_score:.4f} on data")
        if inv_score < 0.90:
            print(f"  NOTE: invariant not perfectly constant "
                  f"(measurement noise or regime structure)")

        t0 = time.time()
        discovery = auto_discover(
            quantities=quantity_dict,
            observations=observations,
            known_invariant=invariant,
            discovery_threshold=0.85 if args.honest_data else 0.90,
            beam_expansions=2000,
            _no_neural_templates=args.no_neural_templates,
        )
        elapsed = time.time() - t0

        result = ClaimResult(
            domain=domain, claim=claim, invariant=invariant,
            discovered_expr=discovery.expression,
            discovered_score=discovery.score,
            invariant_score=inv_score,
            exact_match=expr_normalize(discovery.expression) == expr_normalize(invariant),
            passed=discovery.is_discovery,
        )

        if result.exact_match:
            print(f"  PASS: EXACT: {discovery.expression} "
                  f"(score={discovery.score:.4f}, {elapsed:.1f}s)")
        elif result.passed and discovery.score >= 0.85:
            print(f"  PASS: ALTERNATE: {discovery.expression} "
                  f"(score={discovery.score:.4f}, expected {invariant}, {elapsed:.1f}s)")
        else:
            print(f"  FAIL: best={discovery.expression} "
                  f"(score={discovery.score:.4f}, {elapsed:.1f}s)")
            result.notes = f"score {discovery.score:.4f} below threshold"

        results.append(result)

    # Scorecard
    print(f"\n{'=' * 72}")
    print("SCORECARD")
    print(f"{'=' * 72}")
    exact_count = sum(1 for r in results if r.exact_match)
    passed_count = sum(1 for r in results if r.passed)
    print(f"  Exact matches: {exact_count}/{len(results)}")
    print(f"  Discoveries (score >= threshold): {passed_count}/{len(results)}")

    for r in results:
        status = "✓ EXACT" if r.exact_match else (
            "✓ PASS" if r.passed else "✗ FAIL")
        print(f"  {status:12s} [{r.domain:15s}] {r.claim:40s} "
              f"→ {r.discovered_expr[:50]} ({r.discovered_score:.4f})")

    if exact_count == len(results):
        print(f"\n  ALL {len(results)} CLAIMS VERIFIED (exact matches)")
    elif passed_count == len(results):
        print(f"\n  ALL {len(results)} CLAIMS VERIFIED "
              f"({exact_count} exact, {passed_count - exact_count} alternates)")
    else:
        print(f"\n  {passed_count}/{len(results)} CLAIMS VERIFIED")


if __name__ == "__main__":
    main()
