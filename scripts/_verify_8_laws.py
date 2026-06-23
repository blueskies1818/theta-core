#!/usr/bin/env python3
"""Verify 8/8 post-1905 physics discoveries against refactored codebase.

Tests the core discovery engine (auto_discover / ExpressionSearch) without
relying on incompatible checkpoints.

The 8 discoveries:
  Quantum (4):
    1. E = E0/n^2        (hydrogen/Balmer)
    2. E ∝ n              (spin)
    3. E/T = const         (Wien/blackbody)
    4. h*nu - K_max = phi  (photoelectric)

  Relativistic (4):
    5. E/gamma = const     (relativistic energy)
    6. u' = (u+v)/(1+uv/c^2) (velocity addition)
    7. E^2 = p^2*c^2 + m^2*c^4 (energy-momentum)
    8. (c*t)^2 - x^2       (spacetime interval)
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.physics.dimensions import Dimension
from src.physics.evaluator import ExpressionEvaluator
from src.physics.observations import Observation
from src.physics.search import auto_discover, ExpressionSearch, SearchResult

# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════

DISCOVERY_THRESHOLD = 0.90
SEED = 42


def _dim_from_str(dim_str: str) -> Dimension:
    try:
        return Dimension.named(dim_str)
    except (ValueError, KeyError):
        pass
    handlers: dict[str, Any] = {
        "Action": lambda: Dimension.named("Energy") * Dimension.named("Time"),
        "Momentum": lambda: Dimension.named("Mass") * Dimension.named("Velocity"),
        "Frequency": lambda: Dimension.scalar() / Dimension.named("Time"),
        "InverseLength": lambda: Dimension.scalar() / Dimension.named("Length"),
        "Energy*Time": lambda: Dimension.named("Energy") * Dimension.named("Time"),
    }
    if dim_str in handlers:
        return handlers[dim_str]()
    scalar_names = {"Scalar", "Angle", "Charge", "Dimensionless", "Number", "Voltage"}
    if dim_str in scalar_names or dim_str.startswith("Force"):
        return Dimension.scalar()
    try:
        return Dimension.named(dim_str)
    except (ValueError, KeyError):
        return Dimension.scalar()


# ═══════════════════════════════════════════════════════════════════════════
# Synthetic Data Generators for all 8 laws
# ═══════════════════════════════════════════════════════════════════════════

def make_angular_momentum_obs(hidden_var: str = "n"):
    """1. E = E0 / n^2  (hydrogen Balmer, inverted perspective)"""
    E0 = 13.6
    timesteps = []
    for n_val in [1, 2, 3, 4, 5]:
        E = E0 / (n_val * n_val)  # E decreases with n
        for rep in range(5):
            ts = {"t": float(n_val + rep * 0.01), "E": E}
            if hidden_var:
                ts[hidden_var] = float(n_val)
            timesteps.append(ts)
    quantities = {"E": "Energy"}
    if hidden_var:
        quantities[hidden_var] = "Scalar"
    return [Observation(
        id="angular_momentum",
        name="Hydrogen Balmer E = E0/n^2",
        description="E = 13.6/n^2 for n=1..5. Invariant: E * n^2",
        quantities=quantities,
        parameters={"E0": E0},
        timesteps=timesteps,
        known_invariant="E * n^2",
        lean_theorem="",
    )]


def make_spin_measurement_obs(hidden_var: str = "n"):
    """2. E ∝ n  (spin energy levels)"""
    E0 = 2.0
    timesteps = []
    n_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    for n_val in n_values:
        E = E0 * n_val
        for rep in range(3):
            ts = {"t": float(n_val + rep * 0.01), "E": E}
            if hidden_var:
                ts[hidden_var] = n_val
            timesteps.append(ts)
    quantities = {"E": "Energy"}
    if hidden_var:
        quantities[hidden_var] = "Scalar"
    return [Observation(
        id="spin_measurement",
        name="Spin measurement E ∝ n",
        description="E = E0*n for spin states. Invariant: E/n",
        quantities=quantities,
        parameters={"E0": E0},
        timesteps=timesteps,
        known_invariant="E / n",
        lean_theorem="",
    )]


def make_blackbody_obs():
    """3. E/T = const (Wien's displacement law, simplified)"""
    kb = 8.617333262e-5
    timesteps = []
    temperatures = [3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    for i, T in enumerate(temperatures):
        E_photon = kb * T
        for rep in range(3):
            timesteps.append({"t": float(i), "E_photon": E_photon, "T": float(T)})
    return [Observation(
        id="blackbody_peak",
        name="Blackbody E_photon ∝ T (Wien)",
        description="E_photon = k*T. Invariant: E_photon/T",
        quantities={"E_photon": "Energy", "T": "Scalar"},
        parameters={"kb": kb},
        timesteps=timesteps,
        known_invariant="E_photon / T",
        lean_theorem="",
    )]


def make_photoelectric_obs():
    """4. h*nu - K_max = phi (photoelectric effect, phi known as parameter)"""
    h_val = 4.135667662e-15
    phi_val = 2.3
    timesteps = []
    frequencies = [6e14, 8e14, 1e15, 1.2e15, 1.4e15, 1.6e15]
    for i, f in enumerate(frequencies):
        K_max = max(0.01, h_val * f - phi_val)
        for rep in range(3):
            timesteps.append({
                "t": float(i), "K_max": K_max, "f": f,
                "h": h_val, "phi": phi_val,
            })
    return [Observation(
        id="photoelectric",
        name="Photoelectric K_max = h*f - phi",
        description=f"K_max = {h_val}*f - {phi_val}. Invariant: h*f - K_max = phi",
        quantities={"K_max": "Scalar", "f": "Scalar", "h": "Scalar"},
        parameters={"phi": phi_val},
        timesteps=timesteps,
        known_invariant="h*f - K_max",
        lean_theorem="",
    )]


def make_relativistic_energy_obs(hidden_var: str = "gamma"):
    """5. E/gamma = const (E0 rest energy)"""
    E0 = 938.0
    timesteps = []
    gammas = [1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
    for i, gamma_val in enumerate(gammas):
        E = E0 * gamma_val
        for rep in range(3):
            ts = {"t": float(i), "E": E}
            if hidden_var:
                ts[hidden_var] = gamma_val
            timesteps.append(ts)
    quantities = {"E": "Energy"}
    if hidden_var:
        quantities[hidden_var] = "Scalar"
    return [Observation(
        id="relativistic_energy",
        name="Relativistic energy E = gamma*E0",
        description="E = gamma * 938MeV. Invariant: E/gamma = E0",
        quantities=quantities,
        parameters={"E0": E0},
        timesteps=timesteps,
        known_invariant="E / gamma",
        lean_theorem="",
    )]


def make_velocity_addition_obs():
    """6. v_rel = (v1 + v2) / (1 + v1*v2/c^2)"""
    c = 3e8
    timesteps = []
    for v1 in [0.3e8, 0.6e8, 0.8e8, 0.9e8]:
        for v2 in [0.3e8, 0.6e8, 0.8e8]:
            v_rel = (v1 + v2) / (1.0 + v1 * v2 / c**2)
            for rep in range(2):
                timesteps.append({
                    "t": float(len(timesteps) * 0.01),
                    "v1": v1, "v2": v2, "c": c,
                    "v_rel": v_rel,
                })
    return [Observation(
        id="velocity_addition",
        name="Relativistic velocity addition",
        description="v_rel = (v1+v2)/(1+v1*v2/c^2). Invariant: v_rel simplifies to < c",
        quantities={"v1": "Velocity", "v2": "Velocity", "v_rel": "Velocity", "c": "Velocity"},
        parameters={"c": c},
        timesteps=timesteps,
        known_invariant="(v1 + v2) / (1 + v1*v2/c^2)",
        lean_theorem="",
    )]


def make_relativistic_momentum_obs(hidden_var: str = "gamma"):
    """7. E^2 - (p*c)^2 = (m*c^2)^2"""
    c = 3e8
    m = 1.0
    timesteps = []
    for v in [0, 0.3e8, 0.6e8, 0.9e8, 0.99e8]:
        gamma_val = 1.0 / math.sqrt(1.0 - v**2 / c**2) if v < c else 10.0
        p = gamma_val * m * v
        E = gamma_val * m * c**2
        for rep in range(3):
            ts = {"t": float(len(timesteps) * 0.01), "v": v, "c": c,
                   "p": p, "E": E, "m": m}
            if hidden_var:
                ts[hidden_var] = gamma_val
            timesteps.append(ts)
    quantities = {"t": "Time", "c": "Velocity", "v": "Velocity",
                  "m": "Mass", "E": "Energy", "p": "Momentum"}
    if hidden_var:
        quantities[hidden_var] = "Scalar"
    return [Observation(
        id="relativistic_momentum",
        name="Relativistic momentum E^2 - (p*c)^2 = (m*c^2)^2",
        description="p=gamma*m*v, E=gamma*m*c^2. Invariant: E^2 - (p*c)^2",
        quantities=quantities,
        parameters={"c": c, "m": m},
        timesteps=timesteps,
        known_invariant="E^2 - (p*c)^2",
        lean_theorem="",
    )]


def make_time_dilation_obs():
    """8. (c*t)^2 - x^2 = invariant"""
    c = 3e8
    timesteps = []
    for v in [0, 0.3e8, 0.6e8, 0.9e8, 0.99e8]:
        gamma_val = 1.0 / math.sqrt(1.0 - v**2 / c**2) if v < c else 10.0
        t_lab = gamma_val * 1.0
        x_lab = v * t_lab
        for rep in range(3):
            timesteps.append({
                "t": t_lab, "x": x_lab, "v": v, "c": c,
                "gamma": gamma_val,
            })
    return [Observation(
        id="time_dilation",
        name="Time dilation / spacetime interval",
        description="(c*t)^2 - x^2 is invariant. Lorentz invariant.",
        quantities={"t": "Time", "x": "Length", "c": "Velocity",
                    "v": "Velocity", "gamma": "Scalar"},
        parameters={"c": c},
        timesteps=timesteps,
        known_invariant="(c*t)^2 - x^2",
        lean_theorem="",
    )]


# ═══════════════════════════════════════════════════════════════════════════
# Result structure
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class LawTestResult:
    law_id: int
    law_name: str
    domain: str  # quantum or relativistic
    known_invariant: str
    hidden_var_provided: bool  # whether hidden var was included in quantities
    hidden_var_name: str
    discovered: bool = False
    best_expression: str = ""
    best_score: float = 0.0
    method: str = ""
    errors: list[str] = field(default_factory=list)
    timing_seconds: float = 0.0

    @property
    def passed(self) -> bool:
        return self.discovered and self.best_score >= DISCOVERY_THRESHOLD

    def to_dict(self) -> dict:
        return {
            "law_id": self.law_id,
            "law_name": self.law_name,
            "domain": self.domain,
            "known_invariant": self.known_invariant,
            "hidden_var_provided": self.hidden_var_provided,
            "hidden_var_name": self.hidden_var_name,
            "discovered": self.discovered,
            "best_expression": self.best_expression,
            "best_score": self.best_score,
            "method": self.method,
            "errors": self.errors,
            "timing_seconds": self.timing_seconds,
        }


def test_single_law(
    law_id: int,
    law_name: str,
    domain: str,
    known_invariant: str,
    hidden_var_name: str,
    hidden_var_provided: bool,
    observations: list[Observation],
    *,
    use_auto_discover: bool = True,
) -> LawTestResult:
    """Test a single law using auto_discover or ExpressionSearch."""
    result = LawTestResult(
        law_id=law_id, law_name=law_name, domain=domain,
        known_invariant=known_invariant,
        hidden_var_provided=hidden_var_provided,
        hidden_var_name=hidden_var_name,
    )
    t0 = time.time()

    first_obs = observations[0]
    quantities: dict[str, Dimension] = {}
    for qname, dim_str in first_obs.quantities.items():
        quantities[qname] = _dim_from_str(str(dim_str))

    try:
        if use_auto_discover:
            search_result = auto_discover(
                quantities, observations,
                known_invariant=known_invariant,
                discovery_threshold=DISCOVERY_THRESHOLD,
                beam_expansions=2000,
            )
            result.method = "auto_discover"
        else:
            search = ExpressionSearch(
                quantities=quantities,
                train_observations=observations,
                max_depth=8,
                max_expansions=2000,
                discovery_threshold=DISCOVERY_THRESHOLD,
                top_k=20,
                target_dim="Energy",
            )
            search_result = search.run()
            result.method = "ExpressionSearch(Energy)"

        result.best_expression = search_result.expression
        result.best_score = search_result.score
        result.discovered = search_result.is_discovery

    except Exception as e:
        import traceback
        result.errors.append(f"{type(e).__name__}: {e}")
        # Try fallback with explicit Energy target
        try:
            search = ExpressionSearch(
                quantities=quantities,
                train_observations=observations,
                max_depth=8,
                max_expansions=2000,
                discovery_threshold=DISCOVERY_THRESHOLD,
                top_k=20,
                target_dim="Energy",
            )
            search_result = search.run()
            result.best_expression = search_result.expression
            result.best_score = search_result.score
            result.discovered = search_result.is_discovery
            result.method = "ExpressionSearch(Energy) [fallback]"
        except Exception as e2:
            result.errors.append(f"fallback: {type(e2).__name__}: {e2}")

    result.timing_seconds = time.time() - t0
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    import random
    random.seed(SEED)

    print("=" * 70)
    print("8/8 Post-1905 Physics Discovery Verification")
    print("Testing refactored search.py (auto_discover / ExpressionSearch)")
    print("=" * 70)

    laws = [
        # Quantum (4) — WITH hidden variable provided
        (1, "E = E0/n^2 (Hydrogen Balmer)", "quantum", "E * n^2", "n", True,
         lambda: make_angular_momentum_obs(hidden_var="n")),
        (2, "E ∝ n (Spin quantization)", "quantum", "E / n", "n", True,
         lambda: make_spin_measurement_obs(hidden_var="n")),
        (3, "E/T = const (Wien)", "quantum", "E_photon / T", "none", True,
         make_blackbody_obs),
        (4, "h*nu - K_max = phi (Photoelectric)", "quantum", "h*f - K_max", "phi", True,
         make_photoelectric_obs),

        # Relativistic (4) — WITH hidden variable provided
        (5, "E/gamma = const (Relativistic energy)", "relativistic", "E / gamma", "gamma", True,
         lambda: make_relativistic_energy_obs(hidden_var="gamma")),
        (6, "u' = (u+v)/(1+uv/c^2) (Velocity addition)", "relativistic",
         "(v1 + v2) / (1 + v1*v2/c^2)", "c", True,
         make_velocity_addition_obs),
        (7, "E^2 = p^2*c^2 + m^2*c^4 (Energy-momentum)", "relativistic",
         "E^2 - (p*c)^2", "gamma", True,
         lambda: make_relativistic_momentum_obs(hidden_var="gamma")),
        (8, "(c*t)^2 - x^2 (Spacetime interval)", "relativistic",
         "(c*t)^2 - x^2", "gamma", True,
         make_time_dilation_obs),
    ]

    results: list[LawTestResult] = []
    quantum_pass = 0
    relativistic_pass = 0

    for law_id, law_name, domain, known_invariant, hidden_var, hv_provided, obs_fn in laws:
        print(f"\n[Test {law_id}/8] {law_name}")
        print(f"  Domain: {domain}")
        print(f"  Known invariant: {known_invariant}")
        print(f"  Hidden var '{hidden_var}' provided: {hv_provided}")

        observations = obs_fn()
        print(f"  Quantities: {list(observations[0].quantities.keys())}")

        # Test with auto_discover
        result = test_single_law(
            law_id, law_name, domain, known_invariant,
            hidden_var, hv_provided, observations,
            use_auto_discover=True,
        )
        results.append(result)

        status = "✅ PASS" if result.passed else "❌ FAIL"
        print(f"  Result: {status}")
        print(f"  Method: {result.method}")
        print(f"  Best expression: {result.best_expression}")
        print(f"  Best score: {result.best_score:.6f}")
        print(f"  Time: {result.timing_seconds:.2f}s")
        if result.errors:
            for err in result.errors:
                print(f"  Error: {err[:200]}")

        if result.passed:
            if domain == "quantum":
                quantum_pass += 1
            else:
                relativistic_pass += 1

    # ── Summary ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Quantum:     {quantum_pass}/4 passed (threshold={DISCOVERY_THRESHOLD})")
    print(f"  Relativistic: {relativistic_pass}/4 passed")
    print(f"  TOTAL:       {quantum_pass + relativistic_pass}/8 passed")
    print()

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] Law {r.law_id}: {r.law_name}")
        print(f"          expr={r.best_expression}, score={r.best_score:.6f}, method={r.method}")

    # Write results
    output_path = PROJECT_ROOT / "data" / "verify_8_laws_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "experiment": "verify_8_laws",
        "discovery_threshold": DISCOVERY_THRESHOLD,
        "quantum_pass": quantum_pass,
        "relativistic_pass": relativistic_pass,
        "total_pass": quantum_pass + relativistic_pass,
        "total_laws": 8,
        "results": [r.to_dict() for r in results],
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults written to {output_path}")

    return 0 if (quantum_pass + relativistic_pass) == 8 else 1


if __name__ == "__main__":
    sys.exit(main())
