"""Special relativity observation simulator.

Generates Observation-compatible dicts for relativistic scenarios:
1. Time dilation — moving clock runs slow
2. Length contraction — moving rod is shortened
3. Velocity addition — relativistic composition law
4. Energy-momentum invariant — E² = (pc)² + (mc²)²
5. Spacetime interval — (cΔt)² - Δx² invariant
6. Relativistic Doppler shift
7. Proper time along worldline
8. Lorentz boost of 4-vectors

All equations are special relativity (1905 Einstein/Minkowski).
Known invariants are recorded for acceptance testing only.

Symmetries:
  - Poincaré group (translations + rotations + boosts)
  - Lorentz invariance — spacetime interval invariant
  - Proper time — worldline scalar
"""

from __future__ import annotations

import math
from typing import Any


# Physical constants (SI units)
C = 299792458.0         # Speed of light (m/s)
M_E = 9.10938356e-31    # Electron mass (kg)
M_P = 1.67262192369e-27 # Proton mass (kg)


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
    """Build a single observation dict."""
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


def _gamma(v: float) -> float:
    """Lorentz factor: γ = 1/sqrt(1 - v²/c²)."""
    beta = v / C
    if beta >= 1.0:
        return float("inf")
    return 1.0 / math.sqrt(1.0 - beta**2)


def _beta(v: float) -> float:
    """v/c (dimensionless velocity)."""
    return v / C


# ═══════════════════════════════════════════════════════════════════════════
# 1. Time dilation
# ═══════════════════════════════════════════════════════════════════════════


def simulate_time_dilation(
    v: float = 0.6 * C,
    n_steps: int = 20,
    dt_proper: float = 1.0,
) -> dict[str, Any]:
    """Simulate time dilation for a moving clock.

    In rest frame S: moving clock at velocity v.
    Proper time in clock's rest frame S': Δτ
    Observed time in S: Δt = γ Δτ

    γ = 1/sqrt(1 - v²/c²) ≥ 1

    At each timestep, the proper time along the moving worldline advances
    by dt_proper, while the lab-frame observes dt_lab = γ * dt_proper.

    Invariant: Δt/Δτ = γ = constant for uniform velocity.
    Equivalently: Δt² - (vΔt/c)² = Δτ² (spacetime interval).

    Parameters
    ----------
    v : float
        Velocity of moving clock (m/s).
    n_steps : int
        Number of timesteps.
    dt_proper : float
        Proper time increment per step (s).
    """
    gam = _gamma(v)

    timesteps: list[dict[str, float]] = []
    for i in range(1, n_steps + 1):  # Start from 1 to avoid tau=0
        tau = i * dt_proper
        t_lab = gam * tau
        x_lab = v * t_lab
        timesteps.append({
            "t": round(t_lab, 8),
            "tau": round(tau, 6),
            "t_lab": round(t_lab, 8),
            "x_lab": round(x_lab, 4),
            "gamma": round(gam, 6),
        })

    beta_val = _beta(v)
    desc = f"Time dilation: v={beta_val:.3f}c, γ={gam:.4f}"
    return _make_obs(
        obs_id=f"time_dilation_v{beta_val:.3f}c",
        name=f"Time dilation (v={beta_val:.3f}c)",
        description=desc,
        quantities={
            "v": "Velocity", "c": "Velocity",
            "tau": "Time", "t": "Time",
            "x": "Length", "gamma": "Scalar",
        },
        parameters={"v": v, "c": C, "gamma": round(gam, 6), "beta": round(beta_val, 6)},
        timesteps=timesteps,
        known_invariant="(c*t)^2 - x^2",
        lean_theorem="",
        is_conservative=True,
    )


def generate_time_dilation_scenarios() -> list[dict[str, Any]]:
    """Generate diverse time dilation scenarios."""
    scenarios: list[dict[str, Any]] = []

    for v_frac in [0.1, 0.3, 0.5, 0.6, 0.8, 0.9, 0.95, 0.99, 0.2, 0.7]:
        v = v_frac * C
        scenarios.append(simulate_time_dilation(v=v, n_steps=30))

    return scenarios


# ═══════════════════════════════════════════════════════════════════════════
# 2. Length contraction
# ═══════════════════════════════════════════════════════════════════════════


def simulate_length_contraction(
    v: float = 0.8 * C,
    L0: float = 10.0,
    n_steps: int = 20,
) -> dict[str, Any]:
    """Simulate length contraction.

    Rod of proper length L₀ at rest in S'.
    Observed from S (moving at v relative to rod): L = L₀ / γ

    The rod's endpoints in S': x'₁ = 0, x'₂ = L₀
    In S (Lorentz transform): x₁ = vt, x₂ = vt + L₀/γ

    Observed length L = x₂ - x₁ = L₀/γ ≤ L₀

    Invariants:
      - Spacetime interval between simultaneous endpoint measurements
        (Δt = 0): Δs² = L₀² (invariant)
      - Proper length L₀ is the maximum length

    Parameters
    ----------
    v : float
        Relative velocity (m/s).
    L0 : float
        Proper length of rod (m).
    n_steps : int
        Number of timesteps.
    """
    gam = _gamma(v)
    L_obs = L0 / gam

    timesteps: list[dict[str, float]] = []
    for i in range(n_steps):
        t = i * 1e-8
        x1 = v * t
        x2 = v * t + L_obs
        timesteps.append({
            "t": round(t, 12),
            "x1": round(x1, 8),
            "x2": round(x2, 8),
            "L_obs": round(L_obs, 8),
        })

    beta_val = _beta(v)
    desc = f"Length contraction: v={beta_val:.3f}c, γ={gam:.4f}, L₀={L0}m → L={L_obs:.4f}m"
    return _make_obs(
        obs_id=f"length_contraction_v{beta_val:.3f}c_L{L0}",
        name=f"Length contraction (v={beta_val:.3f}c, L₀={L0}m)",
        description=desc,
        quantities={
            "v": "Velocity", "c": "Velocity",
            "L0": "Length", "L": "Length",
            "x": "Length", "t": "Time", "gamma": "Scalar",
        },
        parameters={"v": v, "c": C, "L0": L0, "gamma": round(gam, 6),
                     "L_obs": round(L_obs, 6)},
        timesteps=timesteps,
        known_invariant="L * gamma",
        lean_theorem="",
        is_conservative=True,
    )


def generate_length_contraction_scenarios() -> list[dict[str, Any]]:
    """Generate diverse length contraction scenarios."""
    scenarios: list[dict[str, Any]] = []

    configs = [
        (0.5 * C, 10.0),
        (0.8 * C, 10.0),
        (0.9 * C, 10.0),
        (0.95 * C, 5.0),
        (0.99 * C, 1.0),
        (0.3 * C, 100.0),
        (0.6 * C, 50.0),
        (0.7 * C, 20.0),
        (0.85 * C, 15.0),
        (0.1 * C, 1000.0),
    ]

    for v, L0 in configs:
        scenarios.append(simulate_length_contraction(v=v, L0=L0, n_steps=30))

    return scenarios


# ═══════════════════════════════════════════════════════════════════════════
# 3. Relativistic velocity addition
# ═══════════════════════════════════════════════════════════════════════════


def simulate_velocity_addition(
    u_prime: float = 0.5 * C,
    v: float = 0.5 * C,
    n_steps: int = 20,
    t_max: float = 1e-7,
) -> dict[str, Any]:
    """Simulate relativistic velocity addition.

    Frame S' moves at velocity v relative to S.
    Object moves at velocity u' in S'.
    Velocity in S: u = (u' + v) / (1 + u'v/c²)

    Galilean: u = u' + v = c → exceeds c
    Relativistic: u = c if either u' = c or v = c

    Invariant: (1 + u'v/c²) relates velocities across frames.
    The proper velocity (rapidity) adds linearly: η = arctanh(u/c).

    Parameters
    ----------
    u_prime : float
        Object velocity in moving frame S' (m/s).
    v : float
        Relative velocity between frames (m/s).
    n_steps : int
        Number of timesteps.
    t_max : float
        Total time simulated in S.
    """
    u = (u_prime + v) / (1.0 + u_prime * v / C**2)
    u_galilean = u_prime + v

    eta_u_prime = math.atanh(min(u_prime / C, 0.999999))
    eta_v = math.atanh(min(v / C, 0.999999))
    eta_u = eta_u_prime + eta_v  # rapidity addition is linear
    u_from_rapidity = C * math.tanh(eta_u)

    timesteps: list[dict[str, float]] = []
    for i in range(n_steps):
        t = t_max * i / (n_steps - 1) if n_steps > 1 else t_max / 2
        x_lab = u * t
        x_galilean = u_galilean * t
        timesteps.append({
            "t": round(t, 12),
            "x_rel": round(x_lab, 8),
            "x_gal": round(x_galilean, 8),
            "u": round(u, 4),
            "u_gal": round(u_galilean, 4),
        })

    beta_up = _beta(u_prime)
    beta_v = _beta(v)
    beta_u = _beta(u)
    desc = f"Velocity addition: u'={beta_up:.3f}c + v={beta_v:.3f}c → u={beta_u:.3f}c"
    return _make_obs(
        obs_id=f"velocity_add_u{beta_up:.3f}_v{beta_v:.3f}",
        name=f"Velocity addition (u'={beta_up:.3f}c, v={beta_v:.3f}c)",
        description=desc,
        quantities={
            "u": "Velocity", "v": "Velocity",
            "c": "Velocity", "x": "Length",
            "t": "Time",
        },
        parameters={"u_prime": u_prime, "v": v, "c": C,
                     "u": round(u, 4), "u_galilean": round(u_galilean, 4),
                     "eta_sum": round(eta_u, 6)},
        timesteps=timesteps,
        known_invariant="(u+v) / (1+u*v/c^2)",
        lean_theorem="",
        is_conservative=True,
    )


def generate_velocity_addition_scenarios() -> list[dict[str, Any]]:
    """Generate diverse velocity addition scenarios."""
    scenarios: list[dict[str, Any]] = []

    configs = [
        (0.5 * C, 0.5 * C),     # u'+v=c Galilean, <c relativistic
        (0.9 * C, 0.9 * C),     # Near-c both
        (0.3 * C, 0.7 * C),     # Asymmetric
        (0.99 * C, 0.5 * C),    # One near-c
        (0.1 * C, 0.1 * C),     # Slow (approximately Galilean)
        (0.6 * C, 0.6 * C),     # Symmetric medium
        (0.8 * C, 0.2 * C),     # High-low
        (0.95 * C, 0.3 * C),    # Very high + medium
        (0.4 * C, 0.4 * C),     # Medium symmetric
        (0.7 * C, 0.7 * C),     # High symmetric
    ]

    for up, v in configs:
        scenarios.append(simulate_velocity_addition(u_prime=up, v=v, n_steps=30))

    return scenarios


# ═══════════════════════════════════════════════════════════════════════════
# 4. Energy-momentum invariant
# ═══════════════════════════════════════════════════════════════════════════


def simulate_energy_momentum(
    m: float = M_E,
    v_fractions: list[float] | None = None,
    n_steps: int = 30,
) -> dict[str, Any]:
    """Simulate relativistic energy-momentum relation.

    For a particle of rest mass m and velocity v:
      E = γ m c²
      p = γ m v
      E² = (pc)² + (mc²)²   ← invariant across all frames

    As particle accelerates, E and p change, but E² - (pc)² = (mc²)² is constant.
    This is the Poincaré invariant — the mass shell condition.

    Parameters
    ----------
    m : float
        Rest mass (kg).
    v_fractions : list[float] | None
        List of v/c fractions to sample. Defaults to uniform 0.05 to 0.999.
    n_steps : int
        Number of velocity samples.
    """
    if v_fractions is None:
        v_fractions = [0.05 + 0.94 * i / (n_steps - 1) for i in range(n_steps)]

    mc2 = m * C**2
    mc2_MeV = mc2 / 1.602176634e-13  # Convert J to MeV

    timesteps: list[dict[str, float]] = []
    for i, beta_val in enumerate(v_fractions):
        v = beta_val * C
        gam = _gamma(v)
        E = gam * mc2
        p = gam * m * v
        invariant = E**2 - (p * C)**2  # should equal (mc²)²
        timesteps.append({
            "t": round(beta_val * 1e-8, 12),
            "idx": float(i),
            "beta": round(beta_val, 6),
            "gamma": round(gam, 6),
            "E": round(E / 1.602176634e-13, 6),  # MeV
            "p": round(p * C / 1.602176634e-13, 6),  # MeV/c → MeV
            "E2_p2c2": round(invariant / 1.602176634e-13**2, 6),  # MeV²
        })

    desc = f"Energy-momentum: m={m:.2e}kg, mc²={mc2_MeV:.3f}MeV"
    return _make_obs(
        obs_id=f"energy_momentum_m{m:.2e}",
        name=f"E² = (pc)² + (mc²)² (m={mc2_MeV:.3f}MeV)",
        description=desc,
        quantities={
            "m": "Mass", "c": "Velocity",
            "E": "Energy", "p": "Momentum",
            "v": "Velocity", "gamma": "Scalar",
        },
        parameters={"m": m, "c": C, "mc2_MeV": round(mc2_MeV, 3)},
        timesteps=timesteps,
        known_invariant="E^2 - (p*c)^2",
        lean_theorem="",
        is_conservative=True,
    )


def generate_energy_momentum_scenarios() -> list[dict[str, Any]]:
    """Generate diverse energy-momentum scenarios."""
    scenarios: list[dict[str, Any]] = []

    masses = [M_E, M_P, 2 * M_E, 0.5 * M_P, M_E * 10]
    for m in masses:
        scenarios.append(simulate_energy_momentum(m=m, n_steps=30))

    return scenarios


# ═══════════════════════════════════════════════════════════════════════════
# 5. Spacetime interval
# ═══════════════════════════════════════════════════════════════════════════


def simulate_spacetime_interval(
    v: float = 0.6 * C,
    n_events: int = 10,
    t_max: float = 1e-6,
) -> dict[str, Any]:
    """Simulate spacetime interval between events.

    Event A at (t=0, x=0) in frame S.
    Event B at (t, x=vt) — on the worldline of a particle moving at v.

    Spacetime interval: Δs² = (cΔt)² - Δx² = (c² - v²)Δt² = (cΔt/γ)²

    For timelike separation (v < c): Δs² > 0, Δs = cΔτ
    For lightlike separation (v = c): Δs² = 0
    For spacelike separation: Δs² < 0

    Invariant: Δs² is Lorentz-invariant (same in all inertial frames).

    Parameters
    ----------
    v : float
        Velocity of particle (m/s).
    n_events : int
        Number of events along worldline.
    t_max : float
        Maximum coordinate time (s).
    """
    gam = _gamma(v)

    timesteps: list[dict[str, float]] = []
    for i in range(n_events):
        t = t_max * (i + 1) / n_events
        x = v * t
        s2 = (C * t)**2 - x**2  # invariant
        s_val = math.sqrt(abs(s2))
        tau = t / gam  # proper time
        timesteps.append({
            "t": round(t, 12),
            "x": round(x, 8),
            "s2": round(s2, 4),
            "s": round(s_val, 8),
            "tau": round(tau, 12),
            "gamma": round(gam, 6),
        })

    beta_val = _beta(v)
    desc = f"Spacetime interval: v={beta_val:.3f}c, γ={gam:.4f}"
    return _make_obs(
        obs_id=f"spacetime_interval_v{beta_val:.3f}c",
        name=f"Spacetime interval (v={beta_val:.3f}c)",
        description=desc,
        quantities={
            "c": "Velocity", "t": "Time", "x": "Length",
            "v": "Velocity", "tau": "Time",
            "s2": "Length^2",
        },
        parameters={"v": v, "c": C, "gamma": round(gam, 6)},
        timesteps=timesteps,
        known_invariant="(c*t)^2 - x^2",
        lean_theorem="",
        is_conservative=True,
    )


def generate_spacetime_interval_scenarios() -> list[dict[str, Any]]:
    """Generate diverse spacetime interval scenarios."""
    scenarios: list[dict[str, Any]] = []

    for v_frac in [0.1, 0.3, 0.5, 0.6, 0.8, 0.9, 0.95, 0.99, 0.2, 0.4]:
        scenarios.append(simulate_spacetime_interval(v=v_frac * C, n_events=15))

    return scenarios


# ═══════════════════════════════════════════════════════════════════════════
# 6. Relativistic Doppler shift
# ═══════════════════════════════════════════════════════════════════════════


def simulate_doppler_shift(
    v: float = 0.5 * C,
    f0: float = 5e14,
    n_steps: int = 20,
) -> dict[str, Any]:
    """Simulate relativistic Doppler shift.

    Source emitting at proper frequency f₀, moving with velocity v.

    Longitudinal Doppler (along line of sight):
      f_obs = f₀ sqrt((1-β)/(1+β))  for receding source
      f_obs = f₀ sqrt((1+β)/(1-β))  for approaching source

    Transverse Doppler (pure time dilation):
      f_obs = f₀ / γ  (observed at closest approach)

    Invariant: f_obs * f_receding * f_approaching relation.
    The frequency transforms as the time component of the wave 4-vector.

    Parameters
    ----------
    v : float
        Source velocity magnitude (m/s).
    f0 : float
        Proper frequency (Hz).
    n_steps : int
        Number of timesteps.
    """
    beta_val = _beta(v)
    gam = _gamma(v)

    f_receding = f0 * math.sqrt((1 - beta_val) / (1 + beta_val))
    f_approaching = f0 * math.sqrt((1 + beta_val) / (1 - beta_val))
    f_transverse = f0 / gam

    # Simulate a moving source passing observer
    # At each time, compute observer-frame frequency based on geometry
    timesteps: list[dict[str, float]] = []
    b = 1.0  # impact parameter (arbitrary units)
    for i in range(n_steps):
        t_frac = -1.0 + 2.0 * i / (n_steps - 1)  # -1 to 1
        x_src = v * t_frac * 1e-6
        # Line-of-sight velocity component
        cos_theta = x_src / math.sqrt(x_src**2 + b**2)
        v_los = v * cos_theta  # positive = receding convention
        beta_los = v_los / C
        # Relativistic Doppler
        if abs(beta_los) < 0.9999:
            f_obs = f0 * math.sqrt((1 - beta_los) / (1 + beta_los))
        else:
            f_obs = f0
        timesteps.append({
            "t": round(t_frac * 1e-6, 12),
            "x": round(x_src, 8),
            "v_los": round(v_los, 4),
            "f_obs": round(f_obs / 1e14, 6),
            "gamma": round(gam, 6),
        })

    desc = f"Doppler shift: v={beta_val:.3f}c, f₀={f0/1e14:.1f}×10¹⁴Hz"
    return _make_obs(
        obs_id=f"doppler_v{beta_val:.3f}c_f{f0/1e14:.1f}",
        name=f"Relativistic Doppler (v={beta_val:.3f}c)",
        description=desc,
        quantities={
            "v": "Velocity", "c": "Velocity",
            "f": "1/Time", "t": "Time",
            "gamma": "Scalar",
        },
        parameters={"v": v, "c": C, "f0": f0, "gamma": round(gam, 6),
                     "f_receding": round(f_receding / 1e14, 6),
                     "f_approaching": round(f_approaching / 1e14, 6),
                     "f_transverse": round(f_transverse / 1e14, 6)},
        timesteps=timesteps,
        known_invariant="f / sqrt((1-beta)/(1+beta))",
        lean_theorem="",
        is_conservative=True,
    )


def generate_doppler_scenarios() -> list[dict[str, Any]]:
    """Generate diverse Doppler shift scenarios."""
    scenarios: list[dict[str, Any]] = []

    for v_frac, f0 in [
        (0.5, 5e14),
        (0.8, 5e14),
        (0.3, 5e14),
        (0.9, 3e14),
        (0.95, 5e14),
        (0.1, 5e14),
        (0.6, 4e14),
        (0.7, 6e14),
        (0.99, 5e14),
        (0.2, 3e14),
    ]:
        scenarios.append(simulate_doppler_shift(v=v_frac * C, f0=f0, n_steps=30))

    return scenarios


# ═══════════════════════════════════════════════════════════════════════════
# 7. Lorentz boost of 4-vectors
# ═══════════════════════════════════════════════════════════════════════════


def simulate_lorentz_boost(
    v: float = 0.6 * C,
    n_transforms: int = 8,
) -> dict[str, Any]:
    """Simulate Lorentz boost of 4-vectors between inertial frames.

    4-vector X^μ = (ct, x, y, z)
    Boost along x with velocity v:
      ct' = γ(ct - βx)
      x'  = γ(x - βct)
      y'  = y
      z'  = z

    Invariant: (ct')² - x'² - y'² - z'² = (ct)² - x² - y² - z²
    This is the Lorentz/Poincaré group invariant.

    Parameters
    ----------
    v : float
        Boost velocity (m/s).
    n_transforms : int
        Number of 4-vectors to transform.
    """
    beta_val = _beta(v)
    gam = _gamma(v)

    timesteps: list[dict[str, float]] = []
    for i in range(n_transforms):
        t = i * 1e-8
        x = 100.0 + i * 500.0
        y = 0.0
        z = 0.0

        # Lorentz boost
        ct = C * t
        ct_prime = gam * (ct - beta_val * x)
        x_prime = gam * (x - beta_val * ct)
        t_prime = ct_prime / C
        y_prime = y
        z_prime = z

        # Invariant
        s2 = ct**2 - x**2 - y**2 - z**2
        s2_prime = ct_prime**2 - x_prime**2 - y_prime**2 - z_prime**2

        timesteps.append({
            "t": round(t, 12),
            "x": round(x, 4),
            "t_prime": round(t_prime, 12),
            "x_prime": round(x_prime, 4),
            "s2": round(s2, 4),
            "s2_prime": round(s2_prime, 4),
        })

    beta_str = str(round(beta_val, 3)).replace(".", "_")
    desc = f"Lorentz boost: β={beta_val:.3f}, γ={gam:.4f}"
    return _make_obs(
        obs_id=f"lorentz_boost_beta{beta_str}",
        name=f"Lorentz boost (β={beta_val:.3f})",
        description=desc,
        quantities={
            "c": "Velocity", "t": "Time", "x": "Length",
            "v": "Velocity", "gamma": "Scalar",
            "t_prime": "Time", "x_prime": "Length",
        },
        parameters={"v": v, "c": C, "beta": round(beta_val, 6), "gamma": round(gam, 6)},
        timesteps=timesteps,
        known_invariant="(c*t)^2 - x^2",
        lean_theorem="",
        is_conservative=True,
    )


def generate_lorentz_boost_scenarios() -> list[dict[str, Any]]:
    """Generate diverse Lorentz boost scenarios."""
    scenarios: list[dict[str, Any]] = []

    for v_frac in [0.1, 0.3, 0.5, 0.6, 0.8, 0.9, 0.95, 0.99]:
        scenarios.append(simulate_lorentz_boost(v=v_frac * C, n_transforms=10))

    return scenarios


# ═══════════════════════════════════════════════════════════════════════════
# 8. Proper time along worldline
# ═══════════════════════════════════════════════════════════════════════════


def simulate_proper_time(
    v_func: str = "constant",
    v_param: float = 0.6,
    n_steps: int = 30,
    t_total: float = 1e-6,
) -> dict[str, Any]:
    """Simulate proper time along various worldlines.

    Proper time: τ = ∫ dt sqrt(1 - v²(t)/c²) = ∫ dt/γ(t)

    For constant velocity: τ = t/γ
    For accelerating worldline: τ is shorter than coordinate time.

    Invariant: τ is Lorentz-invariant scalar (all observers agree on elapsed
    proper time along a given worldline).

    Parameters
    ----------
    v_func : str
        "constant" or "accelerating"
    v_param : float
        Velocity parameter (fraction of c for constant, acceleration factor for accelerating).
    n_steps : int
        Number of timesteps.
    t_total : float
        Total coordinate time (s).
    """
    dt = t_total / n_steps
    tau = 0.0

    timesteps: list[dict[str, float]] = []
    for i in range(n_steps):
        t = i * dt + dt / 2  # midpoint

        if v_func == "constant":
            v = v_param * C
        else:
            # Constant proper acceleration: v = c tanh(aτ/c)
            # Approximate: α = v_param, v(t) = α*t (non-relativistic ramp)
            alpha = v_param * C / t_total
            v = min(alpha * t, 0.99 * C)

        gam = _gamma(v)
        dtau = dt / gam
        tau += dtau

        timesteps.append({
            "t": round(t, 12),
            "v": round(v, 4),
            "gamma": round(gam, 6),
            "tau": round(tau, 12),
        })

    desc = f"Proper time: {v_func} velocity, v_param={v_param}"
    return _make_obs(
        obs_id=f"proper_time_{v_func}_v{v_param}",
        name=f"Proper time ({v_func}, param={v_param})",
        description=desc,
        quantities={
            "v": "Velocity", "c": "Velocity",
            "t": "Time", "tau": "Time",
            "gamma": "Scalar",
        },
        parameters={"v_param": v_param, "c": C,
                     "tau_total": round(tau, 12)},
        timesteps=timesteps,
        known_invariant="tau * gamma",
        lean_theorem="",
        is_conservative=True,
    )


def generate_proper_time_scenarios() -> list[dict[str, Any]]:
    """Generate diverse proper time scenarios."""
    scenarios: list[dict[str, Any]] = []

    for v_frac in [0.1, 0.3, 0.5, 0.6, 0.8, 0.9, 0.95, 0.99]:
        scenarios.append(simulate_proper_time(v_func="constant", v_param=v_frac, n_steps=30))

    # Accelerating case
    scenarios.append(simulate_proper_time(v_func="accelerating", v_param=0.5, n_steps=30))
    scenarios.append(simulate_proper_time(v_func="accelerating", v_param=0.8, n_steps=30))

    return scenarios


# ═══════════════════════════════════════════════════════════════════════════
# Aggregation
# ═══════════════════════════════════════════════════════════════════════════


def generate_all_relativity() -> list[dict[str, Any]]:
    """Generate all relativistic observation scenarios."""
    scenarios: list[dict[str, Any]] = []
    scenarios.extend(generate_time_dilation_scenarios())
    scenarios.extend(generate_length_contraction_scenarios())
    scenarios.extend(generate_velocity_addition_scenarios())
    scenarios.extend(generate_energy_momentum_scenarios())
    scenarios.extend(generate_spacetime_interval_scenarios())
    scenarios.extend(generate_doppler_scenarios())
    scenarios.extend(generate_lorentz_boost_scenarios())
    scenarios.extend(generate_proper_time_scenarios())
    return scenarios
