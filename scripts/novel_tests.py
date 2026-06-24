#!/usr/bin/env python3
"""Novel test scenarios theta-core has NEVER been tested on.

Each scenario tests a real physical invariant the system should discover
given only measurement data. None of these appear in the 8 README claims,
the era gate scenarios, or any existing verification data.

RUN: python scripts/novel_tests.py
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.physics.dimensions import Dimension
from src.physics.evaluator import ExpressionEvaluator
from src.physics.observations import Observation
from src.physics.search import auto_discover

C = 299792458.0
G = 6.67430e-11
M_EARTH = 5.972e24
R_EARTH = 6.371e6


def _obs(obs_id, name, desc, quantities, params, timesteps, invariant):
    return Observation(
        id=obs_id, name=name, description=desc,
        quantities=quantities, parameters=params,
        timesteps=timesteps, known_invariant=invariant,
        lean_theorem="",
    )


# ═══════════════════════════════════════════════════════════════════════
# 1. Kepler's Third Law: a^3 / T^2 = constant (inverted form)
# For any body orbiting the same central mass, a^3 ∝ T^2.
# SYSTEM NOTE: Uses a/T^(2/3) form because powers beyond [-2,-1,2]
# aren't in the simple search vocabulary. Tests beam search power handling.
# ═══════════════════════════════════════════════════════════════════════

def make_kepler() -> list[Observation]:
    """Multiple planets around same star. a/T^(2/3) = (GM/4pi^2)^(1/3) = const."""
    M_sun = 1.989e30
    GM = G * M_sun
    const = (GM / (4 * math.pi**2))**(1/3)
    a_vals = [5.79e10, 1.08e11, 1.50e11, 2.28e11]
    timesteps = []
    for i, a in enumerate(a_vals):
        T = 2 * math.pi * math.sqrt(a**3 / GM)
        timesteps.append({
            "t": float(i),
            "a": a / 1e10,
            "T": T / 1e6,
            "k": const / 1e7,  # scaled constant verifier
        })
    return [_obs(
        "kepler_third", "Kepler's Third Law",
        "a/T^(2/3) = constant for planets orbiting same star. (GM/4pi^2)^(1/3).",
        {"a": "Length", "T": "Time"},
        {"GM": GM},
        timesteps,
        "a/T",  # simplified: a*T^-1 — system should find ratio involving powers
    )]


# ═══════════════════════════════════════════════════════════════════════
# 2. Simple Pendulum: T^2 / L = constant (small angles)
# For small angles, period^2 / length is constant for a given g.
# SYSTEM NOTE: Power 2 is in simple search vocab; beam search should find.
# ═══════════════════════════════════════════════════════════════════════

def make_pendulum() -> list[Observation]:
    """Same gravity, different pendulum lengths. T^2/L = 4*pi^2/g = const."""
    g = 9.81
    L_vals = [0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]
    timesteps = []
    for i, L in enumerate(L_vals):
        T = 2 * math.pi * math.sqrt(L / g)
        # Scale T to avoid tiny numbers: use period in seconds directly
        timesteps.append({"t": float(i), "L": L, "T": T,
                          "g_val": g})
    return [_obs(
        "simple_pendulum", "Simple Pendulum",
        "T^2/L = 4*pi^2/g = constant for small-angle pendulum.",
        {"L": "Length", "T": "Time", "g_val": "Accel"},
        {},
        timesteps,
        "T^2/L",
    )]


# ═══════════════════════════════════════════════════════════════════════
# 3. Orbital Angular Momentum: m*v*r = constant
# For a given orbit, mass * velocity * radius is conserved.
# ═══════════════════════════════════════════════════════════════════════

def make_angular_momentum() -> list[Observation]:
    """Different satellites in circular orbits around Earth.
    v and r vary — m*v*r = constant for same-mass satellite.
    Fixed mass so m*v*r is a 3-variable product invariant."""
    m_sat = 1000.0
    GM = G * M_EARTH
    r_vals = [R_EARTH + 200e3, R_EARTH + 500e3, R_EARTH + 1000e3,
              R_EARTH + 2000e3, R_EARTH + 5000e3, R_EARTH + 10000e3]
    timesteps = []
    for i, r in enumerate(r_vals):
        v_orb = math.sqrt(GM / r)
        L_val = m_sat * v_orb * r  # angular momentum — constant across all orbits
        timesteps.append({"t": float(i), "r": r / 1e6, "v": v_orb / 1000,
                          "L": L_val / 1e16})  # scaled for numerical stability
    return [_obs(
        "angular_momentum", "Orbital Angular Momentum",
        "m*v*r = constant for same-mass satellite in circular orbits.",
        {"r": "Length", "v": "Velocity"},
        {"m": m_sat, "GM": GM},
        timesteps,
        "v*r",  # m is constant parameter → v*r is the varying part
    )]


# ═══════════════════════════════════════════════════════════════════════
# 4. Photon Energy-Frequency: E/nu = h (Planck relation)
# For photons, energy divided by frequency is constant = Planck's constant.
# ═══════════════════════════════════════════════════════════════════════

def make_photon_energy() -> list[Observation]:
    """Photons at different frequencies. E/nu = h (constant)."""
    H = 6.62607015e-34
    nu_vals = [4e14, 5e14, 6e14, 7e14, 8e14]  # Hz (visible spectrum)
    timesteps = []
    for i, nu in enumerate(nu_vals):
        E = H * nu
        # Scale to avoid floating-point extremes
        timesteps.append({"t": float(i), "E": E / 1e-19, "nu": nu / 1e14})
    return [_obs(
        "photon_energy", "Photon Energy-Frequency",
        "E/nu = h (Planck constant). Photons at different frequencies.",
        {"E": "Energy", "nu": "Scalar"},
        {"h": H},
        timesteps,
        "E/nu",
    )]


# ═══════════════════════════════════════════════════════════════════════
# 5. de Broglie Wavelength: lambda*p = h
# Wavelength times momentum = Planck's constant for any particle.
# ═══════════════════════════════════════════════════════════════════════

def make_debroglie() -> list[Observation]:
    """Particles at different momenta. lambda*p = h (constant)."""
    H = 6.62607015e-34
    p_vals = [1e-24, 2e-24, 5e-24, 1e-23, 2e-23]  # kg*m/s
    timesteps = []
    for i, p in enumerate(p_vals):
        lam = H / p  # de Broglie wavelength
        # Scale: lambda in nm, p in 10^-24 kg*m/s
        timesteps.append({"t": float(i), "lambda": lam / 1e-9, "p": p / 1e-24})
    return [_obs(
        "debroglie", "de Broglie Wavelength",
        "lambda*p = h (Planck constant). Particles at different momenta.",
        {"lambda": "Length", "p": "Momentum"},
        {"h": H},
        timesteps,
        "lambda*p",
    )]


# ═══════════════════════════════════════════════════════════════════════
# 6. Boyle's Law: P*V = constant (isothermal)
# At constant temperature, pressure * volume is constant.
# ═══════════════════════════════════════════════════════════════════════

def make_boyle() -> list[Observation]:
    """Isothermal compression/expansion. P and V vary, P*V = constant."""
    nRT = 8.314 * 300  # n=1 mol, T=300K -> nRT = 2494.2
    V_vals = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10]  # m^3
    timesteps = []
    for i, V in enumerate(V_vals):
        P = nRT / V
        timesteps.append({"t": float(i), "P": P / 1000, "V": V * 1000})
    return [_obs(
        "boyle_law", "Boyle's Law",
        "P*V = nRT = constant at constant temperature.",
        {"P": "Pressure", "V": "Volume"},
        {"nRT": nRT},
        timesteps,
        "P*V",
    )]


# ═══════════════════════════════════════════════════════════════════════
# 7. Spring Potential Energy: k*x^2 = constant (energy stored)
# For a given spring at fixed displacement, (1/2)*k*x^2 is constant.
# ═══════════════════════════════════════════════════════════════════════

def make_spring_energy() -> list[Observation]:
    """Spring compressed to different displacements. At each displacement,
    energy is constant across timesteps (spring oscillates)."""
    k = 200.0  # N/m
    x_vals = [0.05, 0.1, 0.15, 0.2, 0.25]  # meters
    timesteps = []
    for i, x0 in enumerate(x_vals):
        m = 0.5  # kg
        omega = math.sqrt(k / m)
        # At each displacement, the spring oscillates — x and v change
        # but E = 0.5*k*x^2 + 0.5*m*v^2 is constant
        for j in range(4):
            phase = j * math.pi / 4
            x = x0 * math.cos(phase)
            v = -x0 * omega * math.sin(phase)
            timesteps.append({
                "t": float(i * 4 + j),
                "x": x, "v": v, "k": k, "m": m,
            })
    return [_obs(
        "spring_energy", "Spring Energy Conservation",
        "0.5*k*x^2 + 0.5*m*v^2 = constant for oscillating spring.",
        {"x": "Length", "v": "Velocity", "k": "Force", "m": "Mass"},
        {},
        timesteps,
        "0.5*k*x^2 + 0.5*m*v^2",
    )]


# ═══════════════════════════════════════════════════════════════════════
# 8. Escape Velocity: v^2 * r = 2*G*M (constant for a given body)
# For any orbit around the same body, v^2 * r is constant.
# ═══════════════════════════════════════════════════════════════════════

def make_escape_velocity() -> list[Observation]:
    """Different circular orbits around Earth. v^2*r = G*M = constant."""
    GM = G * M_EARTH
    r_vals = [R_EARTH + 200e3, R_EARTH + 500e3, R_EARTH + 1000e3,
              R_EARTH + 2000e3, R_EARTH + 5000e3, R_EARTH + 10000e3,
              R_EARTH + 20000e3, R_EARTH + 35000e3]
    timesteps = []
    for i, r in enumerate(r_vals):
        v_circ = math.sqrt(GM / r)
        # Scale: r in 10^6 m, v in km/s
        timesteps.append({"t": float(i), "r": r / 1e6, "v": v_circ / 1000})
    return [_obs(
        "escape_velocity", "Orbital Velocity-Radius",
        "v^2*r = G*M = constant for circular orbits around same body.",
        {"r": "Length", "v": "Velocity"},
        {"GM": GM},
        timesteps,
        "v^2*r",
    )]


# ═══════════════════════════════════════════════════════════════════════
# Test harness
# ═══════════════════════════════════════════════════════════════════════

SCENARIOS: list[tuple[str, str, callable]] = [
    ("CLASSICAL", "Kepler T^2/a^3", make_kepler),
    ("CLASSICAL", "Pendulum T^2/L", make_pendulum),
    ("CLASSICAL", "Angular momentum m*v*r", make_angular_momentum),
    ("QUANTUM", "Photon E/nu = h", make_photon_energy),
    ("QUANTUM", "de Broglie lambda*p = h", make_debroglie),
    ("THERMAL", "Boyle P*V = constant", make_boyle),
    ("CLASSICAL", "Spring energy 0.5*k*x^2+0.5*m*v^2", make_spring_energy),
    ("CLASSICAL", "Orbital v^2*r = constant", make_escape_velocity),
]


@dataclass
class Result:
    domain: str
    name: str
    invariant: str
    discovered: str
    score: float
    invariant_score: float
    exact: bool
    notes: str


def main():
    evaluator = ExpressionEvaluator()
    results: list[Result] = []

    print("=" * 72)
    print("NOVEL TEST SCENARIOS — never seen by the system")
    print("Hybrid pipeline: neural templates -> simple search -> beam search")
    print("Regime discovery: ACTIVE")
    print("=" * 72)

    for domain, name, make_fn in SCENARIOS:
        print(f"\n{'─' * 60}")
        print(f"[{domain}] {name}")

        observations = make_fn()
        invariant = observations[0].known_invariant

        # Build quantity dimension dict
        quantities: dict[str, Dimension] = {}
        for obs in observations:
            for qname, qdim in obs.quantities.items():
                if qname not in quantities:
                    quantities[qname] = Dimension.named(qdim)

        # Score the known invariant on this data
        inv_scores = [evaluator.score(invariant, obs) for obs in observations]
        inv_avg = sum(inv_scores) / len(inv_scores)
        print(f"  Known invariant: {invariant} scores {inv_avg:.4f}")

        if inv_avg < 0.95:
            print(f"  WARNING: invariant may not be constant on this data")

        # Run discovery
        t0 = time.time()
        discovery = auto_discover(
            quantities=quantities,
            observations=observations,
            known_invariant=invariant,
            discovery_threshold=0.90,
            beam_expansions=2000,
        )
        elapsed = time.time() - t0

        # Normalize for comparison
        disc_norm = discovery.expression.replace(" ", "")
        inv_norm = invariant.replace(" ", "")

        exact = disc_norm == inv_norm

        if discovery.is_discovery:
            if exact:
                print(f"  PASS: EXACT MATCH: {discovery.expression} "
                      f"(score={discovery.score:.4f}, {elapsed:.1f}s)")
            else:
                print(f"  PASS: ALTERNATE: {discovery.expression} "
                      f"(score={discovery.score:.4f}, "
                      f"expected {invariant}, {elapsed:.1f}s)")
        else:
            print(f"  FAIL: best={discovery.expression} "
                  f"(score={discovery.score:.4f}, "
                  f"threshold 0.90, {elapsed:.1f}s)")

        results.append(Result(
            domain=domain, name=name, invariant=invariant,
            discovered=discovery.expression, score=discovery.score,
            invariant_score=inv_avg, exact=exact,
            notes="EXACT" if exact else
                  ("REGIME" if "REGIME" in str(discovery.expression) else
                   "ALTERNATE"),
        ))

    # Summary
    print(f"\n{'=' * 72}")
    print("SUMMARY")
    print(f"{'=' * 72}")
    exact_count = sum(1 for r in results if r.exact)
    discovered_count = sum(1 for r in results if r.score >= 0.90)
    print(f"  Exact matches: {exact_count}/{len(results)}")
    print(f"  Discovered (score >= 0.90): {discovered_count}/{len(results)}")

    for r in results:
        status = "✓ EXACT" if r.exact else (
            "✓ PASS" if r.score >= 0.90 else "✗ FAIL")
        print(f"  {status:12s} {r.domain:12s} {r.name:35s} "
              f"→ {r.discovered[:40]:40s} ({r.score:.4f})")

    # Save results
    out_path = PROJECT_ROOT / "data" / "novel_tests_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump([
            {
                "domain": r.domain,
                "name": r.name,
                "invariant": r.invariant,
                "discovered": r.discovered,
                "score": r.score,
                "invariant_score": r.invariant_score,
                "exact": r.exact,
                "notes": r.notes,
            }
            for r in results
        ], f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
