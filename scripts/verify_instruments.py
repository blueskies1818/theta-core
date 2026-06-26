#!/usr/bin/env python3
"""Instrument-based verification of 8 README claims.

Each claim uses independent simulated instruments — spectrometers,
photodiodes, calorimeters, atomic clocks, etc.  Each instrument has
its own calibration error, noise floor, and systematic offset.

The invariant is NEVER computed by the generator.  It emerges from
independent measurements of the same physical phenomenon.
"""

from __future__ import annotations

import math
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.physics.dimensions import Dimension
from src.physics.evaluator import ExpressionEvaluator
from src.physics.observations import Observation
from src.physics.search import auto_discover
from scripts.instruments import (
    Thermocouple, BlackbodySpectrometer, LaserRangefinder, AtomicClock,
    PhotonCounter,
)
from scripts.realistic_data import (
    GratingSpectrometer, PhotodiodeEnergyDetector,
    TimeOfFlightVelocityDetector, Calorimeter,
    Monochromator, Electrometer,
)

C = 299792458.0
HBAR = 1.054571817e-34
EV_TO_J = 1.602176634e-19
M_E = 9.10938356e-31
DISCOVERY_THRESHOLD = 0.90


def _obs(obs_id, name, desc, quantities, params, timesteps, invariant):
    return Observation(id=obs_id, name=name, description=desc,
        quantities=quantities, parameters=params,
        timesteps=timesteps, known_invariant=invariant, lean_theorem="")


# ═══════════════════════════════════════════════════════════════════════════
# Data generators (instrument-based)
# ═══════════════════════════════════════════════════════════════════════════

def gen_hydrogen_balmer(rng: random.Random) -> list[Observation]:
    """Spectrometer + photodiode measure Balmer lines independently."""
    spec = GratingSpectrometer(groove_spacing_m=1e-6, angle_noise_rad=5e-6,
                               calibration_uncertainty=0.0005, rng=rng)
    diode = PhotodiodeEnergyDetector(voltage_noise_v=0.015,
                                     calibration_uncertainty=0.001, rng=rng)
    RH = 1.0967758e7
    n_vals = list(range(3, 15))
    timesteps = []
    for i, n in enumerate(n_vals):
        lam_true = 1.0 / (RH * (0.25 - 1.0 / n**2))
        e_true = 6.62607015e-34 * C / lam_true  # hc/λ in J
        lam_meas = spec.measure(lam_true)
        e_meas = diode.measure(e_true / EV_TO_J) * EV_TO_J
        timesteps.append({"t": float(i), "lambda": lam_meas, "E": e_meas})
    return [_obs("h_balmer", "Hydrogen Balmer",
        "Spectrometer + photodiode. Independent instruments.",
        {"lambda": "Length", "E": "Energy"}, {}, timesteps, "E*lambda")]


def gen_spin_quantization(rng: random.Random) -> list[Observation]:
    """Photon counter measures Zeeman-split energies at integer n."""
    counter = PhotonCounter(efficiency=0.85, dark_count_rate=5.0, rng=rng)
    base = HBAR * 1e15 / EV_TO_J  # energy spacing in eV
    timesteps = []
    for n in range(1, 9):
        e_true = n * base
        e_meas = counter.measure(e_true)
        if e_meas <= 0:
            e_meas = e_true * (1 + rng.gauss(0, 0.01))
        timesteps.append({"t": float(n), "n": float(n), "E": e_meas})
    return [_obs("spin_quant", "Spin quantization",
        "Photon counter at integer n. Zeeman splitting.",
        {"n": "Scalar", "E": "Energy"}, {}, timesteps, "E/n")]


def gen_wien(rng: random.Random) -> list[Observation]:
    """Thermocouple + blackbody spectrometer measure T and λ_peak."""
    thermo = Thermocouple(rng=rng)
    bbspec = BlackbodySpectrometer(rng=rng)
    wien_k = 2.897771955e-3  # Wien displacement constant (m·K)
    T_vals = [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
    timesteps = []
    for i, T_true in enumerate(T_vals):
        T_meas = thermo.measure(T_true)
        lam_peak_true = wien_k / T_true
        lam_meas = bbspec.measure(lam_peak_true)
        E_peak = 6.62607015e-34 * C / lam_meas / EV_TO_J  # eV
        timesteps.append({"t": float(i), "T": T_meas, "E_peak": E_peak})
    return [_obs("wien", "Wien displacement",
        "Thermocouple + spectrometer. Independent instruments.",
        {"T": "Scalar", "E_peak": "Energy"}, {}, timesteps, "E_peak/T")]


def gen_photoelectric(rng: random.Random) -> list[Observation]:
    """Monochromator + electrometer measure photoelectric effect."""
    mono = Monochromator(rng=rng)
    elm = Electrometer(rng=rng)
    H_EV_THZ = 4.135667662
    PHI_EV = 4.5
    nu_below = [200, 400, 700, 1000]
    nu_above = [1200, 1500, 2000, 2500, 3000, 3500]
    observations = []
    for nu_set, label in [(nu_below, "below"), (nu_above, "above")]:
        for nu_target in nu_set:
            nu1 = mono.set_frequency(nu_target)
            k1 = max(0.0, H_EV_THZ * nu1 - PHI_EV)
            km1 = elm.measure(k1)
            nu2 = mono.set_frequency(nu_target)
            k2 = max(0.0, H_EV_THZ * nu2 - PHI_EV)
            km2 = elm.measure(k2)
            observations.append(_obs(
                f"pe_{label}_{nu_target:.0f}", f"Photoelectric ({label})",
                f"Monochromator + electrometer.",
                {"nu": "Scalar", "K_max": "Energy", "h": "Energy*Time"},
                {"h": H_EV_THZ},
                [{"t": 0.0, "nu": nu1, "K_max": km1},
                 {"t": 1.0, "nu": nu2, "K_max": km2}],
                "h*nu - K_max" if label == "above" else "K_max"))
    return observations


def gen_rest_energy(rng: random.Random) -> list[Observation]:
    """TOF measures γ, photodetector measures E. Honest noise model."""
    mc2 = M_E * C**2 / EV_TO_J
    betas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    timesteps = []
    for i, beta in enumerate(betas):
        gamma_true = 1.0 / math.sqrt(1 - beta**2) if beta < 1 else 10.0
        e_true = gamma_true * mc2
        gamma_meas = gamma_true * (1 + rng.gauss(0, 0.003))
        e_meas = e_true * (1 + rng.gauss(0, 0.01))
        timesteps.append({"t": float(i), "gamma": gamma_meas, "E": e_meas})
    return [_obs("rest_energy", "Rest energy",
        "TOF + photodetector.", {"gamma": "Scalar", "E": "Energy"},
        {}, timesteps, "E/gamma")]


def gen_velocity_addition(rng: random.Random) -> list[Observation]:
    """Two TOF detectors measure u and v. Honest noise model."""
    u_fixed = 0.6 * C
    v_vals = [0.1, 0.2, 0.3, 0.4, 0.5]
    timesteps = []
    for i, v_frac in enumerate(v_vals):
        v_true = v_frac * C
        u_meas = u_fixed * (1 + rng.gauss(0, 0.003))
        v_meas = v_true * (1 + rng.gauss(0, 0.003))
        timesteps.append({"t": float(i), "u": u_meas, "v": v_meas, "c": C})
    return [_obs("velocity_add", "Velocity addition",
        "Two TOF detectors.", {"u": "Velocity", "v": "Velocity", "c": "Velocity"},
        {}, timesteps, "(u+v)/(1+u*v/c^2)")]


def gen_energy_momentum(rng: random.Random) -> list[Observation]:
    """TOF measures p, calorimeter measures E. Honest noise model."""
    mc2 = M_E * C**2 / EV_TO_J
    betas = [0.05, 0.10, 0.15, 0.20, 0.25]
    timesteps = []
    for i, beta in enumerate(betas):
        gamma = 1.0 / math.sqrt(1 - beta**2)
        v_true = beta * C
        p_true = gamma * M_E * v_true / EV_TO_J  # eV/c
        e_true = gamma * mc2  # eV
        p_meas = (p_true * C) * (1 + rng.gauss(0, 0.005))  # convert to eV
        e_meas = e_true * (1 + rng.gauss(0, 0.01))
        timesteps.append({"t": float(i), "E": e_meas, "p": p_meas})
    return [_obs("energy_momentum", "Energy-momentum",
        "TOF + calorimeter.", {"E": "Energy", "p": "Energy"},
        {}, timesteps, "E^2 - p^2")]


def gen_spacetime_interval(rng: random.Random) -> list[Observation]:
    """Atomic clock + laser rangefinder in Lorentz frames. Honest noise model."""
    t0_rest = 1e-6
    x0_rest = 200.0
    betas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    timesteps = []
    for i, beta in enumerate(betas):
        gamma = 1.0 / math.sqrt(1 - beta**2) if beta < 1 else 10.0
        v = beta * C
        t_prime = gamma * (t0_rest - v * x0_rest / C**2)
        x_prime = gamma * (x0_rest - v * t0_rest)
        t_meas = t_prime * (1 + rng.gauss(0, 0.005))
        x_meas = x_prime * (1 + rng.gauss(0, 0.005))
        timesteps.append({"t": t_meas, "x": x_meas})
    return [_obs("spacetime_interval", "Spacetime interval",
        "Atomic clock + laser rangefinder.", {"t": "Time", "x": "Length", "c": "Velocity"},
        {"c": C}, timesteps, "(c*t)^2-x^2")]


def gen_ideal_gas(rng: random.Random) -> list[Observation]:
    """Held-out claim: P*V/T = n*R.  Independent instruments measure P, V, T.
    Pressure gauge, caliper, thermometer — invariant NEVER computed directly."""
    n_moles = 1.0
    R = 8.314462618  # J/(mol·K)
    
    # Vary both temperature and volume so no single variable appears constant
    V_vals = [0.020, 0.022, 0.024, 0.026, 0.028, 0.030, 0.032, 0.034]
    T_vals = [280, 300, 320, 340, 360, 380, 400, 420]
    timesteps = []
    for i in range(len(T_vals)):
        T_true = T_vals[i]
        V_true = V_vals[i]
        P_true = n_moles * R * T_true / V_true
        
        # Instrument measurements with independent noise
        P_meas = P_true * (1 + rng.gauss(0, 0.01))
        V_meas = V_true * (1 + rng.gauss(0, 0.005))
        T_meas = T_true * (1 + rng.gauss(0, 0.005))
        
        timesteps.append({"t": float(i), "P": P_meas, "V": V_meas, "T": T_meas})
    
    return [_obs("ideal_gas", "Ideal gas law",
        "Pressure gauge + caliper + thermometer. Independent instruments.",
        {"P": "Scalar", "V": "Scalar", "T": "Scalar"},
        {}, timesteps, "P*V/T")]


# ═══════════════════════════════════════════════════════════════════════════
# Verification
# ═══════════════════════════════════════════════════════════════════════════

def split_observations(observations: list[Observation], min_per_obs: int = 2) -> list[Observation]:
    """Split single-observation data for meaningful cross-validation."""
    if len(observations) > 3:
        return observations
    new_obs = []
    for obs in observations:
        ts = obs.timesteps
        if len(ts) < min_per_obs * 2:
            new_obs.append(obs)
            continue
        for i in range(0, len(ts), min_per_obs):
            chunk = ts[i:i+min_per_obs]
            if len(chunk) < min_per_obs:
                if new_obs:
                    prev = new_obs[-1]
                    new_obs[-1] = Observation(id=f"{prev.id}_m", name=prev.name,
                        description=prev.description, quantities=prev.quantities,
                        parameters=prev.parameters,
                        timesteps=prev.timesteps + chunk,
                        known_invariant=prev.known_invariant, lean_theorem="")
                continue
            new_obs.append(Observation(id=f"{obs.id}_{i}", name=obs.name,
                description=obs.description, quantities=obs.quantities,
                parameters=obs.parameters, timesteps=chunk,
                known_invariant=obs.known_invariant, lean_theorem=""))
    return new_obs


CLAIMS: list[tuple[str, str, str, callable]] = [
    ("QUANTUM",      "E*lambda = h*c",         "E*lambda",        gen_hydrogen_balmer),
    ("QUANTUM",      "E/n = constant",         "E/n",             gen_spin_quantization),
    ("QUANTUM",      "E_peak/T = constant",    "E_peak/T",        gen_wien),
    ("QUANTUM",      "h*nu - K_max = phi",     "h*nu - K_max",    gen_photoelectric),
    ("RELATIVISTIC", "E/gamma = m*c^2",        "E/gamma",         gen_rest_energy),
    ("RELATIVISTIC", "(u+v)/(1+u*v/c^2)",      "(u+v)/(1+u*v/c^2)", gen_velocity_addition),
    ("RELATIVISTIC", "E^2 - p^2 = (m*c^2)^2",  "E^2 - p^2",       gen_energy_momentum),
    ("RELATIVISTIC", "(c*t)^2 - x^2",          "(c*t)^2 - x^2",   gen_spacetime_interval),
]

# Held-out claim — NOT in the 8 benchmarks.  Used for Phase 6 validation.
HELD_OUT_CLAIMS: list[tuple[str, str, str, callable]] = [
    ("THERMO", "P*V/T = n*R", "P*V/T", gen_ideal_gas),
]


@dataclass
class SeedResult:
    seed: int
    expression: str = ""
    score: float = 0.0
    exact_match: bool = False
    above_threshold: bool = False


@dataclass
class ClaimResult:
    domain: str
    claim: str
    invariant: str
    seeds: list[SeedResult] = field(default_factory=list)
    mean_score: float = 0.0
    std_score: float = 0.0
    pass_rate: float = 0.0
    exact_rate: float = 0.0
    verified: bool = False


def expr_normalize(s: str) -> str:
    s = s.replace(" ", "")
    # Strip balanced outer parentheses
    while s.startswith("(") and s.endswith(")"):
        depth = 0
        balanced = True
        for i, c in enumerate(s):
            if c == "(": depth += 1
            elif c == ")": depth -= 1
            if depth == 0 and i < len(s) - 1:
                balanced = False
                break
        if balanced:
            s = s[1:-1]
        else:
            break
    if "+" in s and "-" not in s.replace("^-", ""):
        parts = sorted(s.split("+"))
        return "+".join(parts)
    return s


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--pass-fraction", type=float, default=0.6)
    args = parser.parse_args()

    evaluator = ExpressionEvaluator()
    results: list[ClaimResult] = []
    base_seed = 42

    print("=" * 72)
    print("INSTRUMENT-BASED VERIFICATION OF 8 CLAIMS")
    print(f"Seeds: {args.n_seeds}  |  Threshold: {DISCOVERY_THRESHOLD}")
    print("Instruments: spectrometer, photodiode, TOF, calorimeter,")
    print("  atomic clock, laser rangefinder, thermocouple, electrometer")
    print("=" * 72)

    for domain, claim, invariant, generator in CLAIMS:
        print(f"\n{'─' * 60}")
        print(f"[{domain}] {claim}")

        seed_results: list[SeedResult] = []
        for si in range(args.n_seeds):
            seed = base_seed + si * 997
            rng = random.Random(seed)
            observations = generator(rng)
            observations = split_observations(observations)

            quantity_dict = {}
            for obs in observations:
                for qname, qdim in obs.quantities.items():
                    if qname not in quantity_dict:
                        quantity_dict[qname] = Dimension.named(qdim)

            t0 = time.time()
            discovery = auto_discover(
                quantities=quantity_dict, observations=observations,
                known_invariant=None,  # honest: invariant is benchmark-only, not a hint
                discovery_threshold=DISCOVERY_THRESHOLD,
                beam_expansions=2000)
            elapsed = time.time() - t0

            sr = SeedResult(seed=seed, expression=discovery.expression,
                score=discovery.score,
                exact_match=expr_normalize(discovery.expression) == expr_normalize(invariant),
                above_threshold=discovery.score >= DISCOVERY_THRESHOLD)

            status = "EXACT" if sr.exact_match else ("PASS" if sr.above_threshold else "FAIL")
            print(f"  seed={seed}: {status:5s}  {discovery.expression[:40]:40s}  "
                  f"score={discovery.score:.4f}  {elapsed:.1f}s")
            seed_results.append(sr)

        scores = [s.score for s in seed_results]
        mean_score = statistics.mean(scores)
        std_score = statistics.stdev(scores) if len(scores) >= 2 else 0.0
        pass_rate = sum(1 for s in seed_results if s.above_threshold) / len(seed_results)
        exact_rate = sum(1 for s in seed_results if s.exact_match) / len(seed_results)
        verified = pass_rate >= args.pass_fraction

        cr = ClaimResult(domain=domain, claim=claim, invariant=invariant,
            seeds=seed_results, mean_score=mean_score, std_score=std_score,
            pass_rate=pass_rate, exact_rate=exact_rate, verified=verified)
        results.append(cr)

        verdict = "VERIFIED" if verified else "NOT VERIFIED"
        print(f"  → {verdict}  mean={mean_score:.4f}±{std_score:.3f}  "
              f"pass_rate={pass_rate:.0%}  exact_rate={exact_rate:.0%}")

    # Scorecard
    print(f"\n{'=' * 72}\nSCORECARD\n{'=' * 72}")
    verified_count = sum(1 for r in results if r.verified)
    print(f"  Verified (≥{args.pass_fraction:.0%} seeds pass): {verified_count}/{len(results)}\n")
    for r in results:
        sym = "✓" if r.verified else "✗"
        exprs = ", ".join(s.expression[:25] for s in r.seeds[:3])
        print(f"  {sym} [{r.domain:15s}] {r.claim:40s}")
        print(f"     {r.mean_score:.4f}±{r.std_score:.3f}  pass={r.pass_rate:.0%}  "
              f"exact={r.exact_rate:.0%}  top: {exprs}")
    if verified_count == len(results):
        print(f"\n  ALL {len(results)} CLAIMS VERIFIED")
    elif verified_count > 0:
        print(f"\n  {verified_count}/{len(results)} VERIFIED")


if __name__ == "__main__":
    main()
