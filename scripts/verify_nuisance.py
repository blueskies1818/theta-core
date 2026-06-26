#!/usr/bin/env python3
"""Sequential verification with nuisance variables and accumulating memory.

Each claim has 2-3 nuisance variables that appear in the data but are
NOT part of the invariant.  The system must learn which variables matter
through accumulating semantic memory across discoveries.

Claims are processed in order.  Memory persists.  Earlier discoveries
teach the system which variable pairings are meaningful.  Later claims
benefit from this learned knowledge.
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
from src.memory import load_memory, save_memory, reset_memory, memory_summary

C = 299792458.0
HBAR = 1.054571817e-34
EV_TO_J = 1.602176634e-19
M_E = 9.10938356e-31
THRESHOLD = 0.90


def _obs(obs_id, name, desc, quantities, params, timesteps, invariant):
    return Observation(id=obs_id, name=name, description=desc,
        quantities=quantities, parameters=params,
        timesteps=timesteps, known_invariant=invariant, lean_theorem="")


def _split(obs_list, min_per=4):  # min 4 timesteps — prevents nuisance coincidences
    """Split single observations for cross-validation."""
    if len(obs_list) > 3:
        return obs_list
    new = []
    for obs in obs_list:
        ts = obs.timesteps
        if len(ts) < min_per * 2:
            new.append(obs); continue
        for i in range(0, len(ts), min_per):
            chunk = ts[i:i+min_per]
            if len(chunk) < min_per:
                if new:
                    prev = new[-1]
                    new[-1] = Observation(id=f"{prev.id}_m", name=prev.name,
                        description=prev.description, quantities=prev.quantities,
                        parameters=prev.parameters, timesteps=prev.timesteps + chunk,
                        known_invariant=prev.known_invariant, lean_theorem="")
                continue
            new.append(Observation(id=f"{obs.id}_{i}", name=obs.name,
                description=obs.description, quantities=obs.quantities,
                parameters=obs.parameters, timesteps=chunk,
                known_invariant=obs.known_invariant, lean_theorem=""))
    return new


# ═══════════════════════════════════════════════════════════════════════════
# Data generators with nuisance variables
# ═══════════════════════════════════════════════════════════════════════════

def gen_balmer(rng: random.Random) -> list[Observation]:
    """Balmer: E*lambda = hc.  Nuisance: magnetic field B, temperature T."""
    RH = 1.0967758e7
    n_vals = list(range(3, 15))
    timesteps = []
    for i, n in enumerate(n_vals):
        lam_true = 1.0 / (RH * (0.25 - 1.0 / n**2))
        e_true = 6.62607015e-34 * C / lam_true
        lam = lam_true * (1 + rng.gauss(0, 0.005))
        E = e_true * (1 + rng.gauss(0, 0.01))
        B = rng.uniform(0.5, 2.0)  # Tesla, varies randomly
        T = rng.uniform(290, 310)  # Kelvin, lab temperature
        timesteps.append({"t": float(i), "lambda": lam, "E": E, "B": B, "T": T})
    return [_obs("balmer", "Hydrogen Balmer", "Nuisance: B, T",
        {"lambda": "Length", "E": "Energy", "B": "Scalar", "T": "Scalar"},
        {}, timesteps, "E*lambda")]


def gen_spin(rng: random.Random) -> list[Observation]:
    """Spin quantization: E/n = constant.  Nuisance: B (field), theta (angle)."""
    base = HBAR * 1e15 / EV_TO_J
    timesteps = []
    for n in range(1, 9):
        E = n * base * (1 + rng.gauss(0, 0.01))
        B = rng.uniform(0.1, 5.0)
        theta = rng.uniform(0, math.pi/2)
        timesteps.append({"t": float(n), "n": float(n), "E": E, "B": B, "theta": theta})
    return [_obs("spin", "Spin quantization", "Nuisance: B, theta",
        {"n": "Scalar", "E": "Energy", "B": "Scalar", "theta": "Scalar"},
        {}, timesteps, "E/n")]


def gen_wien(rng: random.Random) -> list[Observation]:
    """Wien: E_peak/T = constant.  Nuisance: R (distance), A (area)."""
    wien_k = 2.897771955e-3
    T_vals = [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
    timesteps = []
    for i, T_true in enumerate(T_vals):
        T = T_true * (1 + rng.gauss(0, 0.005))
        lam = wien_k / T_true * (1 + rng.gauss(0, 0.005))
        E_peak = 6.62607015e-34 * C / lam / EV_TO_J
        R = rng.uniform(0.5, 2.0)
        A = rng.uniform(0.01, 0.1)
        timesteps.append({"t": float(i), "T": T, "E_peak": E_peak, "R": R, "A": A})
    return [_obs("wien", "Wien displacement", "Nuisance: R, A",
        {"T": "Scalar", "E_peak": "Energy", "R": "Scalar", "A": "Scalar"},
        {}, timesteps, "E_peak/T")]


def gen_photoelectric(rng: random.Random) -> list[Observation]:
    """Photoelectric: h*nu-K_max = phi.  Nuisance: I (intensity), d (distance)."""
    H_EV_THZ = 4.135667662
    PHI = 4.5
    nu_below = [200, 400, 700, 1000]
    nu_above = [1200, 1500, 2000, 2500, 3000, 3500]
    observations = []
    for nu_set, label in [(nu_below, "below"), (nu_above, "above")]:
        for nu_target in nu_set:
            nu1 = nu_target * (1 + rng.gauss(0, 0.002))
            nu2 = nu_target * (1 + rng.gauss(0, 0.002))
            K1 = max(0, H_EV_THZ * nu1 - PHI) * (1 + rng.gauss(0, 0.003))
            K2 = max(0, H_EV_THZ * nu2 - PHI) * (1 + rng.gauss(0, 0.003))
            I = rng.uniform(0.1, 10)
            d = rng.uniform(0.05, 0.5)
            observations.append(_obs(f"pe_{label}_{nu_target:.0f}", f"PE ({label})",
                "Nuisance: I, d",
                {"nu": "Scalar", "K_max": "Energy", "h": "Energy*Time", "I": "Scalar", "d": "Scalar"},
                {"h": H_EV_THZ},
                [{"t": 0.0, "nu": nu1, "K_max": K1, "I": I, "d": d},
                 {"t": 1.0, "nu": nu2, "K_max": K2, "I": I * 1.001, "d": d}],
                "h*nu - K_max" if label == "above" else "K_max"))
    return observations


def gen_rest_energy(rng: random.Random) -> list[Observation]:
    """E/gamma = mc².  Nuisance: d (distance), theta (angle)."""
    mc2 = M_E * C**2 / EV_TO_J
    betas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    timesteps = []
    for i, beta in enumerate(betas):
        gamma = 1.0 / math.sqrt(1 - beta**2)
        E = gamma * mc2 * (1 + rng.gauss(0, 0.01))
        gamma_m = gamma * (1 + rng.gauss(0, 0.003))
        d = rng.uniform(1, 100)
        theta = rng.uniform(0, 2*math.pi)
        timesteps.append({"t": float(i), "gamma": gamma_m, "E": E, "d": d, "theta": theta})
    return [_obs("rest_energy", "Rest energy", "Nuisance: d, theta",
        {"gamma": "Scalar", "E": "Energy", "d": "Scalar", "theta": "Scalar"},
        {}, timesteps, "E/gamma")]


def gen_velocity_add(rng: random.Random) -> list[Observation]:
    """(u+v)/(1+uv/c²).  Nuisance: m1, m2, t."""
    u_fixed = 0.6 * C
    v_vals = [0.1, 0.2, 0.3, 0.4, 0.5]
    timesteps = []
    for i, v_frac in enumerate(v_vals):
        u = u_fixed * (1 + rng.gauss(0, 0.003))
        v = v_frac * C * (1 + rng.gauss(0, 0.003))
        m1 = rng.uniform(1e-30, 1e-27)
        m2 = rng.uniform(1e-30, 1e-27)
        t = rng.uniform(1e-9, 1e-6)
        timesteps.append({"t": t, "u": u, "v": v, "c": C, "m1": m1, "m2": m2})
    return [_obs("vel_add", "Velocity addition", "Nuisance: m1, m2",
        {"u": "Velocity", "v": "Velocity", "c": "Velocity",
         "m1": "Scalar", "m2": "Scalar"}, {},
        timesteps, "(u+v)/(1+u*v/c^2)")]


def gen_energy_momentum(rng: random.Random) -> list[Observation]:
    """E²-p² = (mc²)².  Nuisance: x, t, d."""
    mc2 = M_E * C**2 / EV_TO_J
    betas = [0.05, 0.10, 0.15, 0.20, 0.25]
    timesteps = []
    for i, beta in enumerate(betas):
        gamma = 1.0 / math.sqrt(1 - beta**2)
        v = beta * C
        p = gamma * M_E * v / EV_TO_J * C * (1 + rng.gauss(0, 0.005))
        E = gamma * mc2 * (1 + rng.gauss(0, 0.01))
        x = rng.uniform(0, 10)
        t_val = rng.uniform(0, 1e-6)
        d = rng.uniform(1, 100)
        timesteps.append({"t": t_val, "E": E, "p": p, "x": x, "d": d})
    return [_obs("em", "Energy-momentum", "Nuisance: x, d",
        {"E": "Energy", "p": "Energy", "x": "Scalar", "d": "Scalar"},
        {}, timesteps, "E^2 - p^2")]


def gen_spacetime(rng: random.Random) -> list[Observation]:
    """(c*t)²-x².  Nuisance: v, E."""
    t0, x0 = 1e-6, 200.0
    betas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    timesteps = []
    for i, beta in enumerate(betas):
        gamma = 1.0 / math.sqrt(1 - beta**2) if beta < 1 else 10.0
        v = beta * C
        t = gamma * (t0 - v * x0 / C**2) * (1 + rng.gauss(0, 0.005))
        x = gamma * (x0 - v * t0) * (1 + rng.gauss(0, 0.005))
        E = rng.uniform(1e5, 1e7)  # nuisance energy
        v_nuis = rng.uniform(0, 0.9*C)  # nuisance velocity
        timesteps.append({"t": t, "x": x, "c": C, "E": E, "v": v_nuis})
    return [_obs("spacetime", "Spacetime interval", "Nuisance: v, E",
        {"t": "Time", "x": "Length", "c": "Velocity", "v": "Scalar", "E": "Scalar"},
        {}, timesteps, "(c*t)^2-x^2")]


# ═══════════════════════════════════════════════════════════════════════════
# Sequential verification
# ═══════════════════════════════════════════════════════════════════════════

CLAIMS = [
    ("QUANTUM",      "E*lambda = h*c",         "E*lambda",        gen_balmer,
     {"lambda": "Length", "E": "Energy", "B": "Scalar", "T": "Scalar"}),
    ("QUANTUM",      "E/n = constant",         "E/n",             gen_spin,
     {"n": "Scalar", "E": "Energy", "B": "Scalar", "theta": "Scalar"}),
    ("QUANTUM",      "E_peak/T = constant",    "E_peak/T",        gen_wien,
     {"T": "Scalar", "E_peak": "Energy", "R": "Scalar", "A": "Scalar"}),
    ("QUANTUM",      "h*nu - K_max = phi",     "h*nu - K_max",    gen_photoelectric,
     {"nu": "Scalar", "K_max": "Energy", "h": "Energy*Time", "I": "Scalar", "d": "Scalar"}),
    ("RELATIVISTIC", "E/gamma = m*c^2",        "E/gamma",         gen_rest_energy,
     {"gamma": "Scalar", "E": "Energy", "d": "Scalar", "theta": "Scalar"}),
    ("RELATIVISTIC", "(u+v)/(1+u*v/c^2)",      "(u+v)/(1+u*v/c^2)", gen_velocity_add,
     {"u": "Velocity", "v": "Velocity", "c": "Velocity", "m1": "Scalar", "m2": "Scalar"}),
    ("RELATIVISTIC", "E^2 - p^2 = (m*c^2)^2",  "E^2 - p^2",       gen_energy_momentum,
     {"E": "Energy", "p": "Energy", "x": "Scalar", "d": "Scalar"}),
    ("RELATIVISTIC", "(c*t)^2 - x^2",          "(c*t)^2 - x^2",   gen_spacetime,
     {"t": "Time", "x": "Length", "c": "Velocity", "v": "Scalar", "E": "Scalar"}),
]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-memory", action="store_true",
                        help="Disable semantic memory (baseline)")
    parser.add_argument("--reset-memory", action="store_true",
                        help="Reset memory before starting")
    parser.add_argument("--n-seeds", type=int, default=3)
    args = parser.parse_args()

    if args.reset_memory:
        reset_memory()
    evaluator = ExpressionEvaluator()
    base_seed = 42

    print("=" * 72)
    print("SEQUENTIAL VERIFICATION WITH NUISANCE VARIABLES")
    print(f"Memory: {'DISABLED' if args.no_memory else 'ACCUMULATING'}")
    print(f"Seeds per claim: {args.n_seeds}")
    print("=" * 72)

    total_pass = 0
    total_claims = 0

    for domain, claim, invariant, generator, qty_types in CLAIMS:
        total_claims += 1
        print(f"\n{'─' * 60}")
        print(f"[{domain}] {claim} ({len(qty_types)} quantities)")
        if not args.no_memory:
            mem = load_memory()
            dc = mem.get("discovery_count", 0)
            co = len(mem.get("co_occurrence", {}))
            print(f"  Memory: {dc} discoveries, {co} co-occurrence pairs")

        seeds_pass = 0
        for si in range(args.n_seeds):
            seed = base_seed + si * 997
            rng = random.Random(seed)
            obs = generator(rng)
            obs = _split(obs)
            quantities = {k: Dimension.named(v) for k, v in qty_types.items()}

            t0 = time.time()
            discovery = auto_discover(quantities, obs, known_invariant=None,
                                      discovery_threshold=THRESHOLD, beam_expansions=2000)
            elapsed = time.time() - t0

            status = "PASS" if discovery.score >= THRESHOLD else "FAIL"
            exact = "(EXACT)" if discovery.expression.replace(" ","") == invariant.replace(" ","") else ""
            if discovery.score >= THRESHOLD:
                seeds_pass += 1
            print(f"  seed={seed}: {status:4s} {exact:7s} {discovery.expression[:35]:35s} "
                  f"score={discovery.score:.4f} ({elapsed:.1f}s)")

        rate = seeds_pass / args.n_seeds
        print(f"  → {'PASS' if rate >= 0.6 else 'FAIL'} rate={rate:.0%}")

    print(f"\n{'=' * 72}")
    print(f"Memory state after all claims:")
    print(memory_summary(load_memory()))


if __name__ == "__main__":
    main()
