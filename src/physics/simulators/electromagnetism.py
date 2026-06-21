"""Electromagnetism observation simulator.

Generates Observation-compatible dicts for classical EM scenarios:
1. Charged particle in uniform E field
2. Charged particle in uniform B field (circular motion)
3. Combined E+B fields (E×B drift)
4. Two-charge Coulomb system
5. Induced EMF (changing magnetic flux)

All equations are classical EM (pre-1905 Maxwell/Coulomb/Lorentz).
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
# 1. Charged particle in uniform E field
# ═══════════════════════════════════════════════════════════════════════════


def simulate_e_field(
    q: float = 1.0,
    m: float = 1.0,
    E: float = 1.0,
    x0: float = 0.0,
    y0: float = 0.0,
    vx0: float = 0.0,
    vy0: float = 5.0,
    n_steps: int = 30,
    t_max: float = 5.0,
) -> dict[str, Any]:
    """Charged particle in uniform electric field along +x.

    a_x = q*E/m  (constant)
    a_y = 0

    vx(t) = vx0 + (q*E/m)*t
    x(t)  = x0 + vx0*t + 0.5*(q*E/m)*t²
    vy(t) = vy0
    y(t)  = y0 + vy0*t

    Invariant: 0.5*m*(vx²+vy²) - q*E*x = constant
    (equivalent to ½mv² + qV with V = -Ex)
    """
    a_x = q * E / m

    timesteps: list[dict[str, float]] = []
    for i in range(n_steps):
        t = t_max * i / (n_steps - 1)
        x = x0 + vx0 * t + 0.5 * a_x * t**2
        y = y0 + vy0 * t
        vx = vx0 + a_x * t
        vy = vy0
        timesteps.append({
            "t": round(t, 6),
            "x": round(x, 6),
            "y": round(y, 6),
            "vx": round(vx, 6),
            "vy": round(vy, 6),
        })

    # Include vx0/vy0 in ID for uniqueness
    vx_str = str(vx0).replace(".", "_").replace("-", "n")
    vy_str = str(vy0).replace(".", "_").replace("-", "n")
    desc = f"Charged particle in E field: q={q}C, m={m}kg, E={E}N/C"
    return _make_obs(
        obs_id=f"e_field_q{q}_m{m}_E{E}_vx{vx_str}_vy{vy_str}",
        name=f"Uniform E field (q={q}, m={m}, E={E})",
        description=desc,
        quantities={
            "q": "Scalar", "m": "Mass", "E": "Force",
            "x": "Length", "y": "Length",
            "vx": "Velocity", "vy": "Velocity",
            "t": "Time",
        },
        parameters={"q": q, "m": m, "E": E, "x0": x0, "y0": y0},
        timesteps=timesteps,
        known_invariant="0.5*m*(vx^2 + vy^2) - q*E*x",
        lean_theorem="",
        is_conservative=True,
    )


def generate_e_field_scenarios() -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    configs = [
        (1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 5.0),     # Standard
        (2.0, 1.0, 1.0, 0.0, 0.0, 0.0, 3.0),     # Double charge
        (1.0, 2.0, 1.0, 0.0, 0.0, 0.0, 5.0),     # Double mass
        (-1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 5.0),    # Negative charge
        (1.0, 1.0, 3.0, 0.0, 0.0, 2.0, 2.0),     # Stronger field with init vx
        (1.0, 1.0, 0.5, 0.0, 0.0, 3.0, 0.0),     # Weaker field, initial vx
        (1.0, 0.5, 2.0, 0.0, 0.0, 0.0, 4.0),     # Light particle
        (-2.0, 1.0, 2.0, 0.0, 0.0, 1.0, 3.0),    # Double negative
        (1.0, 1.0, 5.0, 0.0, 0.0, -2.0, 5.0),    # Strong field, negative vx
        (1.0, 1.0, 1.0, 1.0, 2.0, 4.0, 0.0),     # Non-zero start
    ]
    for q, m, E, x0, y0, vx0, vy0 in configs:
        scenarios.append(simulate_e_field(q=q, m=m, E=E, x0=x0, y0=y0, vx0=vx0, vy0=vy0))
    return scenarios


# ═══════════════════════════════════════════════════════════════════════════
# 2. Charged particle in uniform B field (circular motion)
# ═══════════════════════════════════════════════════════════════════════════


def simulate_b_field(
    q: float = 1.0,
    m: float = 1.0,
    B: float = 1.0,
    x0: float = 0.0,
    y0: float = 0.0,
    vx0: float = 5.0,
    vy0: float = 0.0,
    n_steps: int = 40,
    n_periods: float = 2.0,
) -> dict[str, Any]:
    """Charged particle in uniform B field (along +z, into page).

    Circular motion in x-y plane:
    omega_c = |q|*B/m  (cyclotron frequency)
    radius r = m*v_perp / (|q|*B)

    For positive q, B along +z:
    vx(t) = vx0*cos(omega_c*t) - vy0*sin(omega_c*t)
    vy(t) = vx0*sin(omega_c*t) + vy0*cos(omega_c*t)

    x(t) = x0 + vx0*sin(omega_c*t)/omega_c + vy0*cos(omega_c*t)/omega_c - vy0/omega_c
    y(t) = y0 - vx0*cos(omega_c*t)/omega_c + vy0*sin(omega_c*t)/omega_c + vx0/omega_c

    Invariant: 0.5*m*(vx²+vy²) = constant (B does no work)
    """
    omega_c = abs(q) * B / m
    period = 2 * math.pi / omega_c
    t_max = n_periods * period

    # Guard against division by zero
    if omega_c < 1e-12:
        omega_c = 1e-12

    timesteps: list[dict[str, float]] = []
    for i in range(n_steps):
        t = t_max * i / (n_steps - 1)

        if q > 0:
            vx = vx0 * math.cos(omega_c * t) - vy0 * math.sin(omega_c * t)
            vy = vx0 * math.sin(omega_c * t) + vy0 * math.cos(omega_c * t)
            x = x0 + (vx0 * math.sin(omega_c * t) + vy0 * math.cos(omega_c * t) - vy0) / omega_c
            y = y0 + (-vx0 * math.cos(omega_c * t) + vy0 * math.sin(omega_c * t) + vx0) / omega_c
        else:
            # Negative charge rotates opposite
            vx = vx0 * math.cos(omega_c * t) + vy0 * math.sin(omega_c * t)
            vy = -vx0 * math.sin(omega_c * t) + vy0 * math.cos(omega_c * t)
            x = x0 + (vx0 * math.sin(omega_c * t) - vy0 * math.cos(omega_c * t) + vy0) / omega_c
            y = y0 + (vx0 * math.cos(omega_c * t) + vy0 * math.sin(omega_c * t) - vx0) / omega_c

        timesteps.append({
            "t": round(t, 6),
            "x": round(x, 6),
            "y": round(y, 6),
            "vx": round(vx, 6),
            "vy": round(vy, 6),
        })

    # Include key ICs in ID for uniqueness
    vx_str = str(vx0).replace(".", "_").replace("-", "n")
    vy_str = str(vy0).replace(".", "_").replace("-", "n")
    desc = f"Charged particle in B field: q={q}C, m={m}kg, B={B}T"
    return _make_obs(
        obs_id=f"b_field_q{q}_m{m}_B{B}_vx{vx_str}_vy{vy_str}",
        name=f"Uniform B field (q={q}, m={m}, B={B})",
        description=desc,
        quantities={
            "q": "Scalar", "m": "Mass", "B": "Force/Velocity",
            "x": "Length", "y": "Length",
            "vx": "Velocity", "vy": "Velocity",
            "t": "Time",
        },
        parameters={"q": q, "m": m, "B": B, "x0": x0, "y0": y0},
        timesteps=timesteps,
        known_invariant="0.5*m*(vx^2 + vy^2)",
        lean_theorem="",
        is_conservative=True,
    )


def generate_b_field_scenarios() -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    configs = [
        (1.0, 1.0, 1.0, 0.0, 0.0, 5.0, 0.0),    # Standard
        (1.0, 1.0, 2.0, 0.0, 0.0, 3.0, 0.0),    # Stronger B, slower
        (-1.0, 1.0, 1.0, 0.0, 0.0, 5.0, 0.0),   # Negative charge
        (1.0, 0.5, 1.0, 0.0, 0.0, 5.0, 0.0),    # Lighter
        (2.0, 1.0, 1.0, 0.0, 0.0, 5.0, 0.0),    # Double charge
        (1.0, 2.0, 1.0, 0.0, 0.0, 5.0, 0.0),    # Heavier (larger radius)
        (1.0, 1.0, 0.5, 0.0, 0.0, 5.0, 3.0),    # Weaker B, with vy
        (1.0, 1.0, 3.0, 0.0, 0.0, 10.0, 0.0),   # Strong B, fast
        (-2.0, 1.0, 2.0, 0.0, 0.0, 4.0, 2.0),   # Double neg charge
        (1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0),    # Non-zero start
    ]
    for q, m, B, x0, y0, vx0, vy0 in configs:
        scenarios.append(simulate_b_field(q=q, m=m, B=B, x0=x0, y0=y0, vx0=vx0, vy0=vy0))
    return scenarios


# ═══════════════════════════════════════════════════════════════════════════
# 3. Combined E+B fields (E×B drift)
# ═══════════════════════════════════════════════════════════════════════════


def _rk4_step_e_b(
    x: float, y: float, vx: float, vy: float,
    q: float, m: float, E: float, B: float, dt: float,
) -> tuple[float, float, float, float]:
    """Single RK4 step for E (along y) + B (along z) fields.

    Equations:
      dvx/dt = (q/m)*(vy*B)        (v×B force, x-component)
      dvy/dt = (q/m)*(E - vx*B)    (E force + v×B force, y-component)
      dx/dt  = vx
      dy/dt  = vy
    """
    def derivatives(_vx: float, _vy: float) -> tuple[float, float]:
        ax = (q / m) * (_vy * B)
        ay = (q / m) * (E - _vx * B)
        return ax, ay

    # k1
    ax1, ay1 = derivatives(vx, vy)
    k1_vx = ax1 * dt
    k1_vy = ay1 * dt
    k1_x = vx * dt
    k1_y = vy * dt

    # k2
    ax2, ay2 = derivatives(vx + 0.5 * k1_vx, vy + 0.5 * k1_vy)
    k2_vx = ax2 * dt
    k2_vy = ay2 * dt
    k2_x = (vx + 0.5 * k1_vx) * dt
    k2_y = (vy + 0.5 * k1_vy) * dt

    # k3
    ax3, ay3 = derivatives(vx + 0.5 * k2_vx, vy + 0.5 * k2_vy)
    k3_vx = ax3 * dt
    k3_vy = ay3 * dt
    k3_x = (vx + 0.5 * k2_vx) * dt
    k3_y = (vy + 0.5 * k2_vy) * dt

    # k4
    ax4, ay4 = derivatives(vx + k3_vx, vy + k3_vy)
    k4_vx = ax4 * dt
    k4_vy = ay4 * dt
    k4_x = (vx + k3_vx) * dt
    k4_y = (vy + k3_vy) * dt

    x_new = x + (k1_x + 2*k2_x + 2*k3_x + k4_x) / 6
    y_new = y + (k1_y + 2*k2_y + 2*k3_y + k4_y) / 6
    vx_new = vx + (k1_vx + 2*k2_vx + 2*k3_vx + k4_vx) / 6
    vy_new = vy + (k1_vy + 2*k2_vy + 2*k3_vy + k4_vy) / 6

    return x_new, y_new, vx_new, vy_new


def simulate_e_b_combined(
    q: float = 1.0,
    m: float = 1.0,
    E: float = 1.0,
    B: float = 1.0,
    x0: float = 0.0,
    y0: float = 5.0,
    vx0: float = 0.0,
    vy0: float = 0.0,
    n_steps: int = 100,
    t_max: float = 20.0,
) -> dict[str, Any]:
    """Charged particle in crossed E (along y) + B (along z) fields.

    E×B drift velocity: v_drift = E/B along +x
    Cycloid/trochoid motion superimposed on drift.

    Invariant: 0.5*m*(vx²+vy²) - q*E*y = constant (B does no work).
    """
    dt = t_max / (n_steps - 1)
    x, y = x0, y0
    vx, vy = vx0, vy0

    timesteps: list[dict[str, float]] = []
    for i in range(n_steps):
        t = i * dt
        timesteps.append({
            "t": round(t, 6),
            "x": round(x, 6),
            "y": round(y, 6),
            "vx": round(vx, 6),
            "vy": round(vy, 6),
        })
        x, y, vx, vy = _rk4_step_e_b(x, y, vx, vy, q, m, E, B, dt)

    desc = f"E×B drift: q={q}C, m={m}kg, E={E}N/C, B={B}T"
    return _make_obs(
        obs_id=f"eb_combined_q{q}_m{m}_E{E}_B{B}",
        name=f"E×B drift (q={q}, m={m}, E={E}, B={B})",
        description=desc,
        quantities={
            "q": "Scalar", "m": "Mass",
            "E": "Force", "B": "Force/Velocity",
            "x": "Length", "y": "Length",
            "vx": "Velocity", "vy": "Velocity",
            "t": "Time",
        },
        parameters={"q": q, "m": m, "E": E, "B": B, "x0": x0, "y0": y0},
        timesteps=timesteps,
        known_invariant="0.5*m*(vx^2 + vy^2) - q*E*y",
        lean_theorem="",
        is_conservative=True,
    )


def generate_eb_combined_scenarios() -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    configs = [
        (1.0, 1.0, 1.0, 1.0),           # Standard
        (1.0, 1.0, 2.0, 1.0),           # Stronger E
        (1.0, 1.0, 1.0, 2.0),           # Stronger B
        (-1.0, 1.0, 1.0, 1.0),          # Negative charge
        (1.0, 2.0, 1.0, 1.0),           # Heavier
        (2.0, 1.0, 1.0, 1.0),           # Double charge
        (1.0, 1.0, 0.5, 3.0),           # Weak E, strong B
        (1.0, 1.0, 3.0, 0.5),           # Strong E, weak B
        (1.0, 0.5, 2.0, 1.0),           # Light
        (-2.0, 1.0, 1.0, 2.0),          # Double neg, strong B
    ]
    for q, m, E, B in configs:
        scenarios.append(simulate_e_b_combined(q=q, m=m, E=E, B=B))
    return scenarios


# ═══════════════════════════════════════════════════════════════════════════
# 4. Two-charge Coulomb system (1D)
# ═══════════════════════════════════════════════════════════════════════════


def _verlet_step_coulomb(
    x1: float, x2: float, v1: float, v2: float,
    m1: float, m2: float, q1: float, q2: float, k: float, dt: float,
) -> tuple[float, float, float, float]:
    """Velocity Verlet step for two-charge 1D Coulomb interaction.

    Energy-conserving symplectic integrator.
    F1 = k*q1*q2 * (x1 - x2) / |x1 - x2|³
    F2 = -F1
    """
    def force(_x1: float, _x2: float) -> tuple[float, float]:
        r = abs(_x2 - _x1)
        if r < 1e-10:
            return 0.0, 0.0
        F1 = k * q1 * q2 * (_x1 - _x2) / (r ** 3)
        return F1, -F1

    # Current forces
    F1_curr, F2_curr = force(x1, x2)

    # Half-step velocity
    v1_half = v1 + 0.5 * (F1_curr / m1) * dt
    v2_half = v2 + 0.5 * (F2_curr / m2) * dt

    # Full-step position
    x1_new = x1 + v1_half * dt
    x2_new = x2 + v2_half * dt

    # New forces
    F1_new, F2_new = force(x1_new, x2_new)

    # Half-step velocity to full
    v1_new = v1_half + 0.5 * (F1_new / m1) * dt
    v2_new = v2_half + 0.5 * (F2_new / m2) * dt

    return x1_new, x2_new, v1_new, v2_new


def simulate_coulomb(
    q1: float = 1.0,
    q2: float = -1.0,
    m1: float = 1.0,
    m2: float = 1.0,
    r0: float = 2.0,
    v1i: float = 0.0,
    v2i: float = 0.0,
    k: float = 1.0,  # Coulomb constant
    n_steps: int = 500,
    t_max: float = 1.0,
) -> dict[str, Any]:
    """Two-charge 1D Coulomb interaction.

    Charges aligned on x-axis. x1(0) = 0, x2(0) = r0.

    Uses velocity Verlet (symplectic) and stops before close approach
    to maintain energy conservation accuracy.

    Invariant (opposite charges, attractive):
      0.5*m1*v1² + 0.5*m2*v2² + k*q1*q2/r = constant
      where r = |x2 - x1|
    """
    dt = t_max / (n_steps - 1)

    x1, x2 = 0.0, r0
    v1, v2 = v1i, v2i

    timesteps: list[dict[str, float]] = []
    for i in range(n_steps):
        t = i * dt
        r = x2 - x1
        timesteps.append({
            "t": round(t, 6),
            "x1": round(x1, 6),
            "x2": round(x2, 6),
            "v1": round(v1, 6),
            "v2": round(v2, 6),
            "r": round(abs(r), 6),
        })

        x1, x2, v1, v2 = _verlet_step_coulomb(x1, x2, v1, v2, m1, m2, q1, q2, k, dt)

        # Stop before charges get too close (force singularity)
        if abs(x2 - x1) < 0.05:
            break

    desc = f"Coulomb: q1={q1}, q2={q2}, m1={m1}, m2={m2}, r0={r0}"
    return _make_obs(
        obs_id=f"coulomb_q1{q1}_q2{q2}_m1{m1}_m2{m2}_r0{r0}",
        name=f"Coulomb system (q1={q1}, q2={q2})",
        description=desc,
        quantities={
            "q1": "Scalar", "q2": "Scalar",
            "m1": "Mass", "m2": "Mass",
            "k": "Force*Length^2",
            "x1": "Length", "x2": "Length",
            "v1": "Velocity", "v2": "Velocity",
            "t": "Time",
        },
        parameters={"q1": q1, "q2": q2, "m1": m1, "m2": m2, "k": k, "r0": r0},
        timesteps=timesteps,
        known_invariant="0.5*m1*v1^2 + 0.5*m2*v2^2 + k*q1*q2 / (abs(x2 - x1))",
        lean_theorem="",
        is_conservative=True,
    )


def generate_coulomb_scenarios() -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    configs = [
        (1.0, -1.0, 1.0, 1.0, 2.0),     # Opposite charges, equal mass
        (2.0, -1.0, 1.0, 1.0, 2.0),     # Asymmetric charge
        (1.0, -1.0, 2.0, 1.0, 3.0),     # Asymmetric mass
        (-1.0, -1.0, 1.0, 1.0, 2.0),    # Like charges (repel)
        (1.0, 1.0, 1.0, 1.0, 2.0),      # Like charges (repel)
        (1.0, -2.0, 1.0, 0.5, 1.0),     # Strong attraction, light
        (3.0, -1.0, 1.0, 3.0, 5.0),     # Far apart
        (1.0, -1.0, 1.0, 1.0, 5.0),     # Large separation
        (2.0, -3.0, 2.0, 1.0, 3.0),     # Mixed
        (-2.0, 1.0, 0.5, 2.0, 4.0),     # Mixed signs
    ]
    for q1, q2, m1, m2, r0 in configs:
        scenarios.append(simulate_coulomb(q1=q1, q2=q2, m1=m1, m2=m2, r0=r0))
    return scenarios


# ═══════════════════════════════════════════════════════════════════════════
# 5. Induced EMF — changing magnetic flux
# ═══════════════════════════════════════════════════════════════════════════


def simulate_induced_emf(
    B0: float = 1.0,
    alpha: float = 0.2,
    A: float = 1.0,
    R: float = 1.0,
    n_steps: int = 30,
    t_max: float = 10.0,
) -> dict[str, Any]:
    """Induced EMF from linearly changing magnetic field through fixed loop.

    B(t) = B0 + alpha*t
    Phi(t) = B(t) * A
    epsilon(t) = -dPhi/dt = -alpha * A

    Induced current: I(t) = epsilon / R

    Power dissipated: P(t) = epsilon² / R = (alpha*A)² / R
    Total energy dissipated over time: E_diss = P * t

    Invariant: epsilon is constant, energy input = energy dissipated.
    """
    epsilon_val = -alpha * A

    timesteps: list[dict[str, float]] = []
    for i in range(n_steps):
        t = t_max * i / (n_steps - 1)
        B = B0 + alpha * t
        Phi = B * A
        I_val = epsilon_val / R
        P = epsilon_val * I_val
        E_diss = P * t
        timesteps.append({
            "t": round(t, 6),
            "B": round(B, 6),
            "Phi": round(Phi, 6),
            "epsilon": round(epsilon_val, 6),
            "I": round(I_val, 6),
        })

    desc = f"Induced EMF: changing B field, B0={B0}T, alpha={alpha}T/s, A={A}m²"
    return _make_obs(
        obs_id=f"induced_emf_B0{B0}_alpha{alpha}_A{A}",
        name=f"Induced EMF (B0={B0}, α={alpha}, A={A})",
        description=desc,
        quantities={
            "B": "Force/Velocity",
            "Phi": "Energy",
            "epsilon": "Energy",
            "I": "Scalar",
            "t": "Time",
        },
        parameters={"B0": B0, "alpha": alpha, "A": A, "R": R},
        timesteps=timesteps,
        known_invariant="abs(epsilon)",
        lean_theorem="",
        is_conservative=True,
    )


def generate_induced_emf_scenarios() -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    configs = [
        (1.0, 0.2, 1.0, 1.0),     # Standard
        (2.0, 0.5, 1.0, 1.0),     # Stronger field, faster change
        (1.0, 0.1, 2.0, 1.0),     # Larger area
        (0.5, 0.3, 1.5, 2.0),     # Mixed
        (1.0, 0.5, 0.5, 1.0),     # Small area, fast change
        (3.0, 0.1, 2.0, 0.5),     # Strong B, large area
        (0.2, 0.05, 3.0, 1.0),    # Weak B, large area
        (1.0, 0.3, 1.0, 2.0),     # Higher resistance
        (1.5, 0.4, 1.0, 0.5),     # Lower resistance
        (0.1, 1.0, 1.0, 1.0),     # Fast changing, weak start
    ]
    for B0, alpha, A, R in configs:
        scenarios.append(simulate_induced_emf(B0=B0, alpha=alpha, A=A, R=R))
    return scenarios


# ═══════════════════════════════════════════════════════════════════════════
# Aggregation
# ═══════════════════════════════════════════════════════════════════════════


def generate_all_electromagnetism() -> list[dict[str, Any]]:
    """Generate all EM observation scenarios."""
    scenarios: list[dict[str, Any]] = []
    scenarios.extend(generate_e_field_scenarios())
    scenarios.extend(generate_b_field_scenarios())
    scenarios.extend(generate_eb_combined_scenarios())
    scenarios.extend(generate_coulomb_scenarios())
    scenarios.extend(generate_induced_emf_scenarios())
    return scenarios
