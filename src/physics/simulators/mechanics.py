"""Classical mechanics observation simulator.

Generates Observation-compatible dicts from known physical equations
for free fall, projectile, pendulum, spring, and elastic collision.

All equations are pre-1905 classical mechanics. No quantum, no relativity.
Known invariants are recorded for acceptance testing only — not injected
into the discovery model.
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


# ═══════════════════════════════════════════════════════════════════════════
# 1. Free Fall
# ═══════════════════════════════════════════════════════════════════════════


def simulate_freefall(
    g: float = 9.8,
    h0: float = 10.0,
    v0: float = 0.0,
    m: float = 1.0,
    n_steps: int = 20,
    t_max: float | None = None,
) -> dict[str, Any]:
    """Simulate free fall under constant gravity.

    h(t) = h0 + v0*t - 0.5*g*t^2
    v(t) = v0 - g*t   (positive = upward convention)

    If v0 > 0 (thrown upward), t_max defaults to time to return to h=0.
    If v0 == 0 (dropped), t_max defaults to time to hit ground.
    """
    # Compute time to hit ground
    if t_max is None:
        if v0 >= 0:
            # Quadratic: h0 + v0*t - 0.5*g*t^2 = 0
            # t = (v0 + sqrt(v0^2 + 2*g*h0)) / g
            t_max = (v0 + math.sqrt(v0**2 + 2 * g * h0)) / g
        else:
            # Thrown downward: h0 + v0*t - 0.5*g*t^2 = 0
            t_max = (-v0 + math.sqrt(v0**2 + 2 * g * h0)) / g
        # At least 0.1s
        if t_max < 0.1:
            t_max = 1.0

    timesteps: list[dict[str, float]] = []
    for i in range(n_steps):
        t = t_max * i / (n_steps - 1)
        h = h0 + v0 * t - 0.5 * g * t**2
        v = v0 - g * t
        timesteps.append({"t": round(t, 6), "h": round(h, 6), "v": round(v, 6)})

    desc = f"Free fall: m={m}kg, g={g}m/s², h0={h0}m, v0={v0}m/s"
    return _make_obs(
        obs_id=f"freefall_g{g}_h{h0}_v{v0}",
        name=f"Free fall (g={g}, h0={h0}, v0={v0})",
        description=desc,
        quantities={"m": "Mass", "g": "Accel", "h": "Length", "v": "Velocity", "t": "Time"},
        parameters={"m": m, "g": g, "h0": h0, "v0": v0},
        timesteps=timesteps,
        known_invariant="m*g*h + 0.5*m*v^2",
        lean_theorem="",
        is_conservative=True,
    )


def generate_freefall_scenarios() -> list[dict[str, Any]]:
    """Generate a diverse set of free-fall scenarios."""
    scenarios: list[dict[str, Any]] = []

    configs = [
        # (g, h0, v0, m)
        (9.8, 10.0, 0.0, 1.0),     # Earth drop
        (9.8, 20.0, 0.0, 2.0),     # Heavy ball from higher
        (9.8, 5.0, 10.0, 1.0),     # Thrown upward
        (9.8, 0.0, 15.0, 1.0),     # From ground upward
        (9.8, 15.0, -5.0, 1.0),    # Thrown downward
        (1.62, 10.0, 0.0, 1.0),    # Moon drop
        (3.72, 10.0, 0.0, 1.0),    # Mars drop
        (9.8, 50.0, 0.0, 0.5),     # Light ball from high
        (9.8, 30.0, 20.0, 3.0),    # Heavy ball thrown up high
        (9.8, 10.0, 5.0, 1.0),     # Moderate throw
        (9.8, 2.0, 2.0, 1.0),      # Short throw from low
        (9.8, 100.0, 0.0, 1.0),    # High drop
        (25.0, 10.0, 0.0, 1.0),    # High-g planet
        (1.0, 10.0, 0.0, 1.0),     # Low-g asteroid
        (9.8, 8.0, 12.0, 1.5),     # Medium ball
        (9.8, 12.0, -3.0, 0.8),    # Light thrown down
        (9.8, 6.0, 8.0, 2.5),      # Heavy throw
        (9.8, 25.0, 25.0, 1.0),    # Hard upward throw
    ]

    for g, h0, v0, m in configs:
        scenarios.append(simulate_freefall(g=g, h0=h0, v0=v0, m=m, n_steps=20))

    return scenarios


# ═══════════════════════════════════════════════════════════════════════════
# 2. Projectile Motion
# ═══════════════════════════════════════════════════════════════════════════


def simulate_projectile(
    g: float = 9.8,
    v0: float = 20.0,
    theta_deg: float = 45.0,
    m: float = 1.0,
    n_steps: int = 20,
) -> dict[str, Any]:
    """Simulate projectile motion.

    x(t) = v0*cos(theta)*t
    y(t) = v0*sin(theta)*t - 0.5*g*t^2
    vx(t) = v0*cos(theta)
    vy(t) = v0*sin(theta) - g*t
    """
    theta = math.radians(theta_deg)
    vx0 = v0 * math.cos(theta)
    vy0 = v0 * math.sin(theta)

    # Time of flight: y = 0 => vy0*t - 0.5*g*t^2 = 0 => t = 2*vy0/g
    t_max = 2 * vy0 / g if vy0 > 0 else 0

    timesteps: list[dict[str, float]] = []
    for i in range(n_steps):
        t = t_max * i / (n_steps - 1) if t_max > 0 else i * 0.1
        x = vx0 * t
        y = vy0 * t - 0.5 * g * t**2
        vx = vx0
        vy = vy0 - g * t
        timesteps.append({
            "t": round(t, 6),
            "x": round(x, 6),
            "y": round(y, 6),
            "vx": round(vx, 6),
            "vy": round(vy, 6),
        })

    desc = f"Projectile: v0={v0}m/s, θ={theta_deg}°, m={m}kg"
    return _make_obs(
        obs_id=f"projectile_g{g}_v{v0}_theta{theta_deg}",
        name=f"Projectile motion (v0={v0}, θ={theta_deg}°)",
        description=desc,
        quantities={
            "m": "Mass", "g": "Accel",
            "x": "Length", "y": "Length",
            "vx": "Velocity", "vy": "Velocity",
            "t": "Time",
        },
        parameters={"m": m, "g": g, "v0": v0, "theta_deg": theta_deg},
        timesteps=timesteps,
        known_invariant="m*g*y + 0.5*m*(vx^2 + vy^2)",
        lean_theorem="",
        is_conservative=True,
    )


def generate_projectile_scenarios() -> list[dict[str, Any]]:
    """Generate diverse projectile scenarios."""
    scenarios: list[dict[str, Any]] = []

    configs = [
        (9.8, 20.0, 45.0, 1.0),    # Standard
        (9.8, 30.0, 30.0, 1.0),    # Shallower, faster
        (9.8, 15.0, 60.0, 1.0),    # Steeper, slower
        (9.8, 25.0, 15.0, 1.0),    # Very shallow
        (9.8, 20.0, 75.0, 1.0),    # Very steep
        (9.8, 50.0, 45.0, 2.0),    # Fast, heavy
        (1.62, 10.0, 45.0, 1.0),   # Moon
        (3.72, 15.0, 45.0, 1.0),   # Mars
        (9.8, 10.0, 45.0, 0.5),    # Light
        (9.8, 40.0, 45.0, 3.0),    # Heavy, fast
    ]

    for g, v0, theta_deg, m in configs:
        scenarios.append(simulate_projectile(g=g, v0=v0, theta_deg=theta_deg, m=m, n_steps=30))

    return scenarios


# ═══════════════════════════════════════════════════════════════════════════
# 3. Pendulum (small angle approximation)
# ═══════════════════════════════════════════════════════════════════════════


def simulate_pendulum(
    L: float = 1.0,
    theta0_deg: float = 30.0,
    m: float = 1.0,
    g: float = 9.8,
    n_steps: int = 30,
    n_periods: float = 2.0,
) -> dict[str, Any]:
    """Simulate a simple pendulum (small-angle approx).

    theta(t) = theta0 * cos(omega * t)  where omega = sqrt(g/L)
    h = L * (1 - cos(theta))
    v = L * |dtheta/dt| = L * omega * |theta0 * sin(omega*t)|

    Energy: m*g*h + 0.5*m*v^2 is constant.
    """
    omega = math.sqrt(g / L)
    theta0 = math.radians(theta0_deg)
    period = 2 * math.pi / omega
    t_max = n_periods * period

    timesteps: list[dict[str, float]] = []
    for i in range(n_steps):
        t = t_max * i / (n_steps - 1)
        theta = theta0 * math.cos(omega * t)
        h = L * (1.0 - math.cos(theta))
        v = L * omega * abs(theta0 * math.sin(omega * t))
        timesteps.append({
            "t": round(t, 6),
            "theta": round(theta, 6),
            "h": round(h, 6),
            "v": round(v, 6),
        })

    desc = f"Pendulum: L={L}m, θ0={theta0_deg}°, m={m}kg"
    return _make_obs(
        obs_id=f"pendulum_L{L}_theta{theta0_deg}",
        name=f"Pendulum (L={L}m, θ0={theta0_deg}°)",
        description=desc,
        quantities={
            "m": "Mass", "g": "Accel", "L": "Length",
            "h": "Length", "v": "Velocity", "t": "Time",
        },
        parameters={"m": m, "g": g, "L": L, "theta0_rad": round(theta0, 6)},
        timesteps=timesteps,
        known_invariant="m*g*h + 0.5*m*v^2",
        lean_theorem="",
        is_conservative=True,
    )


def generate_pendulum_scenarios() -> list[dict[str, Any]]:
    """Generate diverse pendulum scenarios."""
    scenarios: list[dict[str, Any]] = []

    configs = [
        (1.0, 10.0, 1.0, 9.8),     # Small angle
        (1.0, 20.0, 1.0, 9.8),     # Medium angle
        (2.0, 15.0, 1.0, 9.8),     # Long pendulum
        (0.5, 10.0, 1.0, 9.8),     # Short pendulum
        (1.5, 25.0, 2.0, 9.8),     # Heavy
        (1.0, 5.0, 1.0, 9.8),      # Very small angle
        (3.0, 10.0, 0.5, 9.8),     # Very long, light
        (0.3, 15.0, 0.5, 9.8),     # Very short
    ]

    for L, theta0, m, g in configs:
        scenarios.append(simulate_pendulum(L=L, theta0_deg=theta0, m=m, g=g, n_steps=40))

    return scenarios


# ═══════════════════════════════════════════════════════════════════════════
# 4. Spring-Mass System
# ═══════════════════════════════════════════════════════════════════════════


def simulate_spring(
    k: float = 10.0,
    m: float = 1.0,
    A: float = 0.5,
    phi: float = 0.0,
    n_steps: int = 30,
    n_periods: float = 2.0,
) -> dict[str, Any]:
    """Simulate an undamped spring-mass system.

    x(t) = A * cos(omega*t + phi)  where omega = sqrt(k/m)
    v(t) = -omega * A * sin(omega*t + phi)

    Energy: 0.5*k*x^2 + 0.5*m*v^2 is constant.
    """
    omega = math.sqrt(k / m)
    period = 2 * math.pi / omega
    t_max = n_periods * period

    timesteps: list[dict[str, float]] = []
    for i in range(n_steps):
        t = t_max * i / (n_steps - 1)
        x = A * math.cos(omega * t + phi)
        v = -omega * A * math.sin(omega * t + phi)
        timesteps.append({
            "t": round(t, 6),
            "x": round(x, 6),
            "v": round(v, 6),
        })

    desc = f"Spring: k={k}N/m, m={m}kg, A={A}m"
    return _make_obs(
        obs_id=f"spring_k{k}_m{m}_A{A}",
        name=f"Spring-mass (k={k}, m={m}, A={A})",
        description=desc,
        quantities={
            "m": "Mass", "k": "Force/Length",
            "x": "Length", "v": "Velocity", "t": "Time",
        },
        parameters={"m": m, "k": k, "A": A},
        timesteps=timesteps,
        known_invariant="0.5*k*x^2 + 0.5*m*v^2",
        lean_theorem="",
        is_conservative=True,
    )


def generate_spring_scenarios() -> list[dict[str, Any]]:
    """Generate diverse spring scenarios."""
    scenarios: list[dict[str, Any]] = []

    configs = [
        (10.0, 1.0, 0.5),     # Standard
        (20.0, 1.0, 0.3),     # Stiffer
        (5.0, 2.0, 0.5),      # Softer, heavier
        (10.0, 0.5, 0.5),     # Lighter mass
        (50.0, 1.0, 0.2),     # Very stiff
        (2.0, 1.0, 1.0),      # Very soft, large amplitude
        (15.0, 3.0, 0.4),     # Heavy, medium stiffness
        (10.0, 1.0, 0.1),     # Small amplitude
    ]

    for k, m, A in configs:
        scenarios.append(simulate_spring(k=k, m=m, A=A, n_steps=40))

    return scenarios


# ═══════════════════════════════════════════════════════════════════════════
# 5. Elastic Collision (1D)
# ═══════════════════════════════════════════════════════════════════════════


def simulate_collision_1d(
    m1: float = 1.0,
    m2: float = 1.0,
    v1i: float = 2.0,
    v2i: float = -1.0,
    n_steps_before: int = 10,
    n_steps_after: int = 10,
) -> dict[str, Any]:
    """Simulate a 1D elastic collision between two masses.

    Before collision: masses approach each other at constant velocity.
    After collision: v1' = ((m1-m2)v1i + 2*m2*v2i) / (m1+m2)
                     v2' = ((m2-m1)v2i + 2*m1*v1i) / (m1+m2)

    This is piecewise: separate evaluation regions before/after collision.
    """
    # Post-collision velocities
    v1f = ((m1 - m2) * v1i + 2 * m2 * v2i) / (m1 + m2)
    v2f = ((m2 - m1) * v2i + 2 * m1 * v1i) / (m1 + m2)

    # Time of collision: t=0 is collision. Timesteps go from -1 to +1.
    t_before = -1.0
    t_collision = 0.0
    t_after = 1.0

    timesteps: list[dict[str, float]] = []

    # Before collision: positions approach collision point
    x_collision_1 = 0.0
    x_collision_2 = 1.0  # separation at collision

    for i in range(n_steps_before):
        t = t_before + (t_collision - t_before) * i / (n_steps_before - 1)
        x1 = x_collision_1 + v1i * t
        x2 = x_collision_2 + v2i * t
        v1 = v1i
        v2 = v2i
        timesteps.append({
            "t": round(t, 6),
            "x1": round(x1, 6),
            "x2": round(x2, 6),
            "v1": round(v1, 6),
            "v2": round(v2, 6),
        })

    # Collision point
    timesteps.append({
        "t": 0.0,
        "x1": x_collision_1,
        "x2": x_collision_2,
        "v1": v1i,
        "v2": v2i,
    })

    # After collision
    for i in range(1, n_steps_after + 1):
        t = t_collision + (t_after - t_collision) * i / n_steps_after
        x1 = x_collision_1 + v1f * t
        x2 = x_collision_2 + v2f * t
        v1 = v1f
        v2 = v2f
        timesteps.append({
            "t": round(t, 6),
            "x1": round(x1, 6),
            "x2": round(x2, 6),
            "v1": round(v1, 6),
            "v2": round(v2, 6),
        })

    desc = f"1D elastic collision: m1={m1}kg, m2={m2}kg, v1i={v1i}m/s, v2i={v2i}m/s"
    return _make_obs(
        obs_id=f"collision_m1_{m1}_m2_{m2}_v1i_{v1i}_v2i_{v2i}",
        name=f"Elastic collision (m1={m1}, m2={m2})",
        description=desc,
        quantities={
            "m1": "Mass", "m2": "Mass",
            "x1": "Length", "x2": "Length",
            "v1": "Velocity", "v2": "Velocity",
            "t": "Time",
        },
        parameters={"m1": m1, "m2": m2, "v1i": v1i, "v2i": v2i},
        timesteps=timesteps,
        known_invariant="0.5*m1*v1^2 + 0.5*m2*v2^2",
        lean_theorem="",
        phase_regions=[
            {"label": "before", "t_range": [-1.0, 0.0]},
            {"label": "after", "t_range": [0.001, 1.0]},
        ],
        is_conservative=True,
    )


def generate_collision_scenarios() -> list[dict[str, Any]]:
    """Generate diverse 1D elastic collision scenarios."""
    scenarios: list[dict[str, Any]] = []

    configs = [
        (1.0, 1.0, 2.0, -1.0),      # Equal mass
        (1.0, 2.0, 2.0, 0.0),       # Light hits heavy at rest
        (2.0, 1.0, 2.0, 0.0),       # Heavy hits light at rest
        (1.0, 3.0, 3.0, -1.0),      # Light hits heavy head-on
        (1.0, 1.0, 5.0, 0.0),       # Equal mass, fast vs stationary
        (1.0, 1.0, 3.0, -3.0),      # Equal mass, symmetric
        (0.5, 2.0, 4.0, -2.0),      # Very light vs heavy
        (3.0, 0.5, 1.0, 0.0),       # Heavy vs very light
        (1.0, 4.0, 2.0, -0.5),      # Various
        (2.0, 2.0, 1.0, -2.0),      # Equal heavy masses
    ]

    for m1, m2, v1i, v2i in configs:
        scenarios.append(simulate_collision_1d(m1=m1, m2=m2, v1i=v1i, v2i=v2i))

    return scenarios


# ═══════════════════════════════════════════════════════════════════════════
# Aggregation
# ═══════════════════════════════════════════════════════════════════════════


def generate_all_mechanics() -> list[dict[str, Any]]:
    """Generate all classical mechanics observation scenarios."""
    scenarios: list[dict[str, Any]] = []
    scenarios.extend(generate_freefall_scenarios())
    scenarios.extend(generate_projectile_scenarios())
    scenarios.extend(generate_pendulum_scenarios())
    scenarios.extend(generate_spring_scenarios())
    scenarios.extend(generate_collision_scenarios())
    return scenarios
