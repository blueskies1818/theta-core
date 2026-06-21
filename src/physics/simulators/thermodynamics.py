"""Thermodynamics observation simulator.

Generates Observation-compatible dicts for classical thermodynamics:
1. Ideal gas processes (isothermal, adiabatic, isobaric, isochoric)
2. Heat engine cycles (Carnot, simplified Otto)
3. Entropy changes

All equations are classical thermodynamics (pre-1905).
Known invariants are recorded for acceptance testing only.
"""

from __future__ import annotations

import math
from typing import Any


def _make_obs(
    obs_id: str,
    name: str,
    description: str,
    quantities: dict[str, str],
    parameters: dict[str, float],
    timesteps: list[dict[str, float]],
    known_invariant: str | None = None,
    lean_theorem: str = "",
    external_forces: list[str] | None = None,
    phase_regions: list[dict] | None = None,
    is_conservative: bool | None = None,
) -> dict[str, Any]:
    return {
        "id": obs_id,
        "name": name,
        "description": description,
        "quantities": quantities,
        "parameters": parameters,
        "timesteps": timesteps,
        "known_invariant": known_invariant,
        "lean_theorem": lean_theorem,
        "external_forces": external_forces,
        "phase_regions": phase_regions,
        "is_conservative": is_conservative,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 1. Ideal gas: Isothermal process (T = constant)
# ═══════════════════════════════════════════════════════════════════════════


def simulate_isothermal(
    n: float = 1.0,
    R: float = 8.314,
    T: float = 300.0,
    P1: float = 100000.0,
    V1: float = 0.0249,
    V2: float = 0.05,
    n_steps: int = 20,
) -> dict[str, Any]:
    """Isothermal expansion/compression of ideal gas.

    PV = nRT = constant
    For each volume V between V1 and V2:
        P = nRT / V
        T = constant
        W = nRT * ln(V2/V1)  (total work)
        Q = W (since ΔU = 0 for isothermal ideal gas)

    Invariant: P*V = constant (= nRT)
    """
    PV_const = n * R * T

    timesteps: list[dict[str, float]] = []
    for i in range(n_steps):
        frac = i / (n_steps - 1)
        V = V1 + (V2 - V1) * frac
        P = PV_const / V
        W_done = PV_const * math.log(V / V1)
        timesteps.append({
            "t": round(i, 2),
            "V": round(V, 8),
            "P": round(P, 2),
            "T": round(T, 2),
            "W": round(W_done, 2),
        })

    desc = f"Isothermal: n={n}mol, T={T}K, V: {V1:.4f}→{V2:.4f}m³"
    return _make_obs(
        obs_id=f"isothermal_n{n}_T{T}_V{V1}_{V2}",
        name=f"Isothermal process (T={T}K)",
        description=desc,
        quantities={
            "n": "Scalar", "R": "Energy",
            "P": "Force/Length^2", "V": "Length^3",
            "T": "Scalar", "W": "Energy",
            "t": "Time",
        },
        parameters={"n": n, "R": R, "T": T},
        timesteps=timesteps,
        known_invariant="P*V",
        lean_theorem="",
        is_conservative=True,
    )


def generate_isothermal_scenarios() -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    configs = [
        (1.0, 8.314, 300.0, 100000.0, 0.0249, 0.05),     # Standard expansion
        (1.0, 8.314, 400.0, 100000.0, 0.0332, 0.06),     # Hotter
        (1.0, 8.314, 200.0, 100000.0, 0.0166, 0.03),     # Colder
        (2.0, 8.314, 300.0, 100000.0, 0.0498, 0.08),     # More moles
        (1.0, 8.314, 300.0, 200000.0, 0.0125, 0.03),     # Higher pressure start
        (0.5, 8.314, 300.0, 100000.0, 0.0125, 0.03),     # Half mole
        (1.0, 8.314, 350.0, 150000.0, 0.0194, 0.04),     # Medium hot
        (1.0, 8.314, 250.0, 80000.0, 0.0260, 0.05),      # Medium cold
        (2.0, 8.314, 500.0, 100000.0, 0.0831, 0.12),     # Hot, more moles
        (3.0, 8.314, 300.0, 100000.0, 0.0748, 0.10),     # Triple moles
    ]
    for n, R, T, P1, V1, V2 in configs:
        scenarios.append(simulate_isothermal(n=n, R=R, T=T, P1=P1, V1=V1, V2=V2))
    return scenarios


# ═══════════════════════════════════════════════════════════════════════════
# 2. Ideal gas: Adiabatic process (PV^γ = constant)
# ═══════════════════════════════════════════════════════════════════════════


def simulate_adiabatic(
    n: float = 1.0,
    R: float = 8.314,
    T1: float = 300.0,
    P1: float = 100000.0,
    V1: float = 0.0249,
    V2: float = 0.05,
    gamma: float = 1.4,  # Cp/Cv for diatomic gas
    n_steps: int = 20,
) -> dict[str, Any]:
    """Adiabatic expansion/compression of ideal gas.

    PV^γ = constant
    TV^(γ-1) = constant

    P1*V1^γ = P2*V2^γ
    For each V: P = P1*(V1/V)^γ
                T = T1*(V1/V)^(γ-1)

    Invariants: P*(V^gamma) = constant, T*V^(gamma-1) = constant
    We'll use the cleaner: P*V^1.4 (for gamma=1.4)
    """
    gamma_str = str(gamma).replace(".", "_")
    PVg_const = P1 * (V1 ** gamma)
    TVg_const = T1 * (V1 ** (gamma - 1))

    timesteps: list[dict[str, float]] = []
    for i in range(n_steps):
        frac = i / (n_steps - 1)
        V = V1 + (V2 - V1) * frac
        P = PVg_const / (V ** gamma)
        T = TVg_const / (V ** (gamma - 1))
        timesteps.append({
            "t": round(i, 2),
            "V": round(V, 8),
            "P": round(P, 2),
            "T": round(T, 2),
        })

    desc = f"Adiabatic: γ={gamma}, V: {V1:.4f}→{V2:.4f}m³"
    return _make_obs(
        obs_id=f"adiabatic_gamma{gamma_str}_V{V1}_{V2}",
        name=f"Adiabatic process (γ={gamma})",
        description=desc,
        quantities={
            "n": "Scalar", "R": "Energy",
            "P": "Force/Length^2", "V": "Length^3",
            "T": "Scalar",
            "t": "Time",
        },
        parameters={"n": n, "R": R, "gamma": gamma},
        timesteps=timesteps,
        known_invariant=f"P*V^{gamma}",
        lean_theorem="",
        is_conservative=True,
    )


def generate_adiabatic_scenarios() -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    configs = [
        (1.0, 8.314, 300.0, 100000.0, 0.0249, 0.05, 1.4),    # Standard (diatomic)
        (1.0, 8.314, 400.0, 100000.0, 0.0332, 0.06, 1.4),    # Hotter start
        (1.0, 8.314, 300.0, 200000.0, 0.0125, 0.04, 1.4),    # Higher pressure
        (1.0, 8.314, 300.0, 100000.0, 0.0249, 0.04, 1.67),   # Monatomic (γ=5/3)
        (2.0, 8.314, 300.0, 100000.0, 0.0498, 0.08, 1.4),    # More moles
        (1.0, 8.314, 300.0, 100000.0, 0.03, 0.015, 1.4),     # Compression
        (1.0, 8.314, 500.0, 150000.0, 0.0277, 0.05, 1.4),    # Hot, high P
        (0.5, 8.314, 300.0, 50000.0, 0.0249, 0.06, 1.4),     # Half mole, low P
        (1.0, 8.314, 300.0, 100000.0, 0.02, 0.01, 1.4),      # Strong compression
        (2.0, 8.314, 400.0, 200000.0, 0.0332, 0.05, 1.67),   # Double mole, hot, high P, monatomic
    ]
    for n, R, T1, P1, V1, V2, gamma in configs:
        scenarios.append(simulate_adiabatic(n=n, R=R, T1=T1, P1=P1, V1=V1, V2=V2, gamma=gamma))
    return scenarios


# ═══════════════════════════════════════════════════════════════════════════
# 3. Ideal gas: Isobaric process (P = constant)
# ═══════════════════════════════════════════════════════════════════════════


def simulate_isobaric(
    n: float = 1.0,
    R: float = 8.314,
    P: float = 100000.0,
    T1: float = 300.0,
    T2: float = 500.0,
    n_steps: int = 20,
) -> dict[str, Any]:
    """Isobaric heating/cooling of ideal gas.

    V/T = nR/P = constant
    V1 = n*R*T1/P, V2 = n*R*T2/P

    Invariant: V/T = constant
    """
    const_VT = n * R / P

    timesteps: list[dict[str, float]] = []
    for i in range(n_steps):
        frac = i / (n_steps - 1)
        T = T1 + (T2 - T1) * frac
        V = const_VT * T
        Q = n * (5/2) * R * (T - T1)  # Cp = (5/2)R for diatomic, Q = n*Cp*ΔT
        W = P * (V - const_VT * T1)
        timesteps.append({
            "t": round(i, 2),
            "T": round(T, 2),
            "V": round(V, 8),
            "P": round(P, 2),
            "Q": round(Q, 2),
            "W": round(W, 2),
        })

    desc = f"Isobaric: P={P}Pa, T: {T1}→{T2}K"
    return _make_obs(
        obs_id=f"isobaric_P{P}_T{T1}_{T2}",
        name=f"Isobaric process (P={P}Pa)",
        description=desc,
        quantities={
            "n": "Scalar", "R": "Energy",
            "P": "Force/Length^2", "V": "Length^3",
            "T": "Scalar", "Q": "Energy", "W": "Energy",
            "t": "Time",
        },
        parameters={"n": n, "R": R, "P": P},
        timesteps=timesteps,
        known_invariant="V/T",
        lean_theorem="",
        is_conservative=True,
    )


def generate_isobaric_scenarios() -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    configs = [
        (1.0, 8.314, 100000.0, 300.0, 500.0),     # Standard heating
        (1.0, 8.314, 100000.0, 200.0, 400.0),     # Cold to medium
        (2.0, 8.314, 100000.0, 300.0, 600.0),     # More moles
        (1.0, 8.314, 200000.0, 300.0, 450.0),     # Higher pressure
        (0.5, 8.314, 50000.0, 300.0, 500.0),      # Half mole, low pressure
        (1.0, 8.314, 100000.0, 500.0, 300.0),     # Cooling
        (2.0, 8.314, 150000.0, 400.0, 700.0),     # High-T heating
        (1.0, 8.314, 80000.0, 250.0, 550.0),      # Wide range
        (3.0, 8.314, 100000.0, 300.0, 400.0),     # Triple moles
        (0.5, 8.314, 100000.0, 600.0, 350.0),     # Cooling, half mole
    ]
    for n, R, P, T1, T2 in configs:
        scenarios.append(simulate_isobaric(n=n, R=R, P=P, T1=T1, T2=T2))
    return scenarios


# ═══════════════════════════════════════════════════════════════════════════
# 4. Ideal gas: Isochoric process (V = constant)
# ═══════════════════════════════════════════════════════════════════════════


def simulate_isochoric(
    n: float = 1.0,
    R: float = 8.314,
    V: float = 0.0249,
    T1: float = 300.0,
    T2: float = 500.0,
    n_steps: int = 20,
) -> dict[str, Any]:
    """Isochoric heating/cooling of ideal gas.

    P/T = nR/V = constant
    P1 = n*R*T1/V, P2 = n*R*T2/V

    Invariant: P/T = constant
    No work done (W = 0).
    """
    const_PT = n * R / V

    timesteps: list[dict[str, float]] = []
    for i in range(n_steps):
        frac = i / (n_steps - 1)
        T = T1 + (T2 - T1) * frac
        P = const_PT * T
        Q = n * (3/2) * R * (T - T1)  # Cv = (3/2)R for monatomic
        timesteps.append({
            "t": round(i, 2),
            "T": round(T, 2),
            "P": round(P, 2),
            "V": round(V, 8),
            "Q": round(Q, 2),
        })

    desc = f"Isochoric: V={V}m³, T: {T1}→{T2}K"
    return _make_obs(
        obs_id=f"isochoric_V{V}_T{T1}_{T2}",
        name=f"Isochoric process (V={V}m³)",
        description=desc,
        quantities={
            "n": "Scalar", "R": "Energy",
            "P": "Force/Length^2", "V": "Length^3",
            "T": "Scalar", "Q": "Energy",
            "t": "Time",
        },
        parameters={"n": n, "R": R, "V": V},
        timesteps=timesteps,
        known_invariant="P/T",
        lean_theorem="",
        is_conservative=True,
    )


def generate_isochoric_scenarios() -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    configs = [
        (1.0, 8.314, 0.0249, 300.0, 500.0),     # Standard
        (1.0, 8.314, 0.05, 300.0, 400.0),       # Larger volume
        (2.0, 8.314, 0.0249, 300.0, 600.0),     # More moles
        (1.0, 8.314, 0.01, 200.0, 500.0),       # Small volume
        (0.5, 8.314, 0.0249, 300.0, 450.0),     # Half mole
        (1.0, 8.314, 0.03, 350.0, 650.0),       # Medium volume, wide range
        (2.0, 8.314, 0.02, 300.0, 700.0),       # Double mole, high T range
        (1.0, 8.314, 0.015, 250.0, 400.0),      # Small V
        (3.0, 8.314, 0.04, 300.0, 500.0),       # Triple mole
        (0.5, 8.314, 0.05, 400.0, 600.0),       # Half mole, large V
    ]
    for n, R, V, T1, T2 in configs:
        scenarios.append(simulate_isochoric(n=n, R=R, V=V, T1=T1, T2=T2))
    return scenarios


# ═══════════════════════════════════════════════════════════════════════════
# 5. Carnot cycle
# ═══════════════════════════════════════════════════════════════════════════


def simulate_carnot_cycle(
    n: float = 1.0,
    R: float = 8.314,
    Th: float = 600.0,
    Tc: float = 300.0,
    V1: float = 0.01,
    V2: float = 0.02,
    gamma: float = 1.4,
    n_steps: int = 40,
) -> dict[str, Any]:
    """Simulate a Carnot cycle.

    4 steps:
    1. Isothermal expansion at Th: V1 → V2
       P = nRTh/V, Qh = nRTh*ln(V2/V1), W1 = Qh
    2. Adiabatic expansion: Th → Tc
       TV^(γ-1) = constant → V3 = V2*(Th/Tc)^(1/(γ-1))
    3. Isothermal compression at Tc: V3 → V4
       P = nRTc/V, Qc = nRTc*ln(V4/V3), W3 = Qc (negative)
    4. Adiabatic compression: Tc → Th
       TV^(γ-1) = constant → V4 = V1*(Tc/Th)^(1/(γ-1))

    Efficiency: η = 1 - Tc/Th = W_net / Qh

    Invariant: efficiency = 1 - Tc/Th
    """
    # Step 1: Isothermal expansion at Th
    V3 = V2 * (Th / Tc) ** (1 / (gamma - 1))
    V4 = V1 * (Tc / Th) ** (1 / (gamma - 1))

    Qh = n * R * Th * math.log(V2 / V1)
    Qc = n * R * Tc * math.log(V4 / V3)
    W_net = Qh + Qc  # Qc is negative
    efficiency = W_net / Qh if Qh > 0 else 0.0

    PVg_h_const = n * R * Th * (V2 ** (gamma - 1))
    PVg_c_const = n * R * Tc * (V4 ** (gamma - 1))

    timesteps: list[dict[str, float]] = []
    steps_per_phase = n_steps // 4
    total_steps = steps_per_phase * 4

    for i in range(total_steps):
        phase = i // steps_per_phase
        frac = (i % steps_per_phase) / max(steps_per_phase - 1, 1)

        if phase == 0:
            # Isothermal expansion at Th
            V = V1 + (V2 - V1) * frac
            T = Th
            P = n * R * Th / V
        elif phase == 1:
            # Adiabatic expansion Th→Tc
            V = V2 + (V3 - V2) * frac
            T = PVg_h_const / (V ** (gamma - 1))
            P = PVg_h_const / (V ** gamma)
        elif phase == 2:
            # Isothermal compression at Tc
            V = V3 + (V4 - V3) * frac
            T = Tc
            P = n * R * Tc / V
        else:
            # Adiabatic compression Tc→Th
            V = V4 + (V1 - V4) * frac
            T = PVg_c_const / (V ** (gamma - 1))
            P = PVg_c_const / (V ** gamma)

        timesteps.append({
            "t": round(i, 2),
            "V": round(V, 8),
            "P": round(P, 2),
            "T": round(T, 2),
        })

    gamma_str = str(gamma).replace(".", "_")
    desc = f"Carnot cycle: Th={Th}K, Tc={Tc}K, η={efficiency:.4f}"
    return _make_obs(
        obs_id=f"carnot_Th{Th}_Tc{Tc}_g{gamma_str}",
        name=f"Carnot cycle (Th={Th}K, Tc={Tc}K)",
        description=desc,
        quantities={
            "n": "Scalar", "R": "Energy",
            "P": "Force/Length^2", "V": "Length^3",
            "T": "Scalar",
            "t": "Time",
        },
        parameters={"n": n, "R": R, "Th": Th, "Tc": Tc, "gamma": gamma,
                     "Qh": round(Qh, 2), "Qc": round(Qc, 2),
                     "W_net": round(W_net, 2), "efficiency": round(efficiency, 6)},
        timesteps=timesteps,
        known_invariant=None,  # Carnot is about efficiency, not a simple PVT invariant
        lean_theorem="",
        is_conservative=True,
    )


def generate_carnot_scenarios() -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    configs = [
        (1.0, 8.314, 600.0, 300.0, 0.01, 0.02, 1.4),    # Standard (η=0.5)
        (1.0, 8.314, 800.0, 300.0, 0.01, 0.02, 1.4),    # Higher Th (η=0.625)
        (1.0, 8.314, 500.0, 300.0, 0.01, 0.02, 1.4),    # Lower Th (η=0.4)
        (1.0, 8.314, 600.0, 200.0, 0.01, 0.02, 1.4),    # Lower Tc (η=0.667)
        (1.0, 8.314, 600.0, 300.0, 0.02, 0.03, 1.67),   # Monatomic
    ]
    for n, R, Th, Tc, V1, V2, gamma in configs:
        scenarios.append(simulate_carnot_cycle(n=n, R=R, Th=Th, Tc=Tc, V1=V1, V2=V2, gamma=gamma))
    return scenarios


# ═══════════════════════════════════════════════════════════════════════════
# 6. Entropy change (isothermal)
# ═══════════════════════════════════════════════════════════════════════════


def simulate_entropy_change(
    n: float = 1.0,
    R: float = 8.314,
    T: float = 300.0,
    V1: float = 0.01,
    V2: float = 0.05,
    n_steps: int = 20,
) -> dict[str, Any]:
    """Isothermal entropy change.

    ΔS = Q/T = nR*ln(V2/V1)
    For each step: ΔS(i) = nR*ln(V(i)/V1)

    Invariant: ΔS/ln(V/V1) = nR = constant
    """
    delta_S_total = n * R * math.log(V2 / V1)

    timesteps: list[dict[str, float]] = []
    for i in range(n_steps):
        frac = i / (n_steps - 1)
        V = V1 + (V2 - V1) * frac
        delta_S = n * R * math.log(V / V1)
        Q = T * delta_S
        timesteps.append({
            "t": round(i, 2),
            "V": round(V, 8),
            "delta_S": round(delta_S, 4),
            "Q": round(Q, 2),
            "T": round(T, 2),
        })

    desc = f"Entropy change: T={T}K, V: {V1}→{V2}m³, ΔS={delta_S_total:.4f}J/K"
    return _make_obs(
        obs_id=f"entropy_T{T}_V{V1}_{V2}",
        name=f"Entropy change (T={T}K)",
        description=desc,
        quantities={
            "n": "Scalar", "R": "Energy",
            "V": "Length^3", "T": "Scalar",
            "Q": "Energy", "delta_S": "Energy",
            "t": "Time",
        },
        parameters={"n": n, "R": R, "T": T, "delta_S_total": round(delta_S_total, 4)},
        timesteps=timesteps,
        known_invariant="delta_S / log(V)",
        lean_theorem="",
        is_conservative=True,
    )


def generate_entropy_scenarios() -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    configs = [
        (1.0, 8.314, 300.0, 0.01, 0.05),     # Standard
        (1.0, 8.314, 400.0, 0.01, 0.03),     # Hotter
        (2.0, 8.314, 300.0, 0.01, 0.04),     # More moles
        (1.0, 8.314, 200.0, 0.02, 0.08),     # Colder, larger expansion
        (0.5, 8.314, 300.0, 0.01, 0.06),     # Half mole
        (1.0, 8.314, 500.0, 0.01, 0.07),     # Very hot
        (2.0, 8.314, 350.0, 0.005, 0.03),    # Double mole, small start
        (1.0, 8.314, 300.0, 0.02, 0.10),     # Large expansion
        (3.0, 8.314, 300.0, 0.01, 0.03),     # Triple mole
        (0.5, 8.314, 450.0, 0.015, 0.06),    # Half mole, hot
    ]
    for n, R, T, V1, V2 in configs:
        scenarios.append(simulate_entropy_change(n=n, R=R, T=T, V1=V1, V2=V2))
    return scenarios


# ═══════════════════════════════════════════════════════════════════════════
# 7. General ideal gas: PV/T = nR = constant
# ═══════════════════════════════════════════════════════════════════════════


def simulate_ideal_gas_varied(
    n: float = 1.0,
    R: float = 8.314,
    n_steps: int = 30,
) -> dict[str, Any]:
    """Simulate an ideal gas undergoing a generic non-adiabatic, non-isothermal
    path to verify PV/T = nR.

    We use a path: V(T) = V0 + alpha*(T - T0) + beta*(T - T0)^2
    This is arbitrary — the point is PV/T should still equal nR.
    """
    T0, V0 = 300.0, 0.0249
    alpha = 0.0001
    beta = 0.000001
    const_PVT = n * R

    timesteps: list[dict[str, float]] = []
    for i in range(n_steps):
        T = T0 + i * 10.0
        V = V0 + alpha * (T - T0) + beta * (T - T0)**2
        P = const_PVT * T / V
        PVT = P * V / T
        timesteps.append({
            "t": round(i, 2),
            "T": round(T, 2),
            "V": round(V, 8),
            "P": round(P, 2),
        })

    desc = f"Ideal gas varied path: n={n}mol"
    return _make_obs(
        obs_id=f"ideal_gas_varied_n{n}",
        name=f"Ideal gas varied path (n={n})",
        description=desc,
        quantities={
            "n": "Scalar", "R": "Energy",
            "P": "Force/Length^2", "V": "Length^3",
            "T": "Scalar",
            "t": "Time",
        },
        parameters={"n": n, "R": R},
        timesteps=timesteps,
        known_invariant="P*V/T",
        lean_theorem="",
        is_conservative=True,
    )


def generate_ideal_gas_scenarios() -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    for n in [1.0, 2.0, 0.5]:
        for R in [8.314]:
            scenarios.append(simulate_ideal_gas_varied(n=n, R=R))
    return scenarios


# ═══════════════════════════════════════════════════════════════════════════
# Aggregation
# ═══════════════════════════════════════════════════════════════════════════


def generate_all_thermodynamics() -> list[dict[str, Any]]:
    """Generate all thermal observation scenarios."""
    scenarios: list[dict[str, Any]] = []
    scenarios.extend(generate_isothermal_scenarios())
    scenarios.extend(generate_adiabatic_scenarios())
    scenarios.extend(generate_isobaric_scenarios())
    scenarios.extend(generate_isochoric_scenarios())
    scenarios.extend(generate_carnot_scenarios())
    scenarios.extend(generate_entropy_scenarios())
    scenarios.extend(generate_ideal_gas_scenarios())
    return scenarios
