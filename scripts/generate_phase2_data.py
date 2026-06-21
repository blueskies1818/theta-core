"""Generate Phase 2 extended observation database (50+ scenarios).

Computes accurate physics timesteps for:
- Free-fall variants (Moon, Mars, air resistance, varying gravity)
- Projectile variants (with/without drag)
- Spring variants (damped, forced, coupled)
- Collision scenarios (elastic, inelastic)
- Incline variants (with/without friction)
- Cross-domain composition scenarios
"""

import json
import math
import copy
from pathlib import Path

OUTPUT_PATH = Path("data/observations/phase2_extended.json")
PHASE1_PATH = Path("data/observations/phase1_falling.json")


def linspace(start, stop, n):
    """Generate n evenly spaced points from start to stop inclusive."""
    if n == 1:
        return [start]
    return [start + (stop - start) * i / (n - 1) for i in range(n)]


def load_phase1() -> list[dict]:
    """Load existing 10 Phase 1 scenarios."""
    with open(PHASE1_PATH) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════════
# Free-fall variants (10 new)
# ═══════════════════════════════════════════════════════════════════════════

def make_freefall_moon():
    """Moon gravity: g=1.62 m/s^2, drop from 10m."""
    g, h0, m = 1.62, 10.0, 1.0
    t_total = math.sqrt(2 * h0 / g)  # ~3.51s
    ts = linspace(0, t_total, 6)
    timesteps = []
    for t in ts:
        h = h0 - 0.5 * g * t**2
        v = -g * t  # downward negative
        timesteps.append({"t": round(t, 6), "h": round(h, 6), "v": round(v, 6)})
    return {
        "id": "freefall_moon_drop",
        "name": "Ball dropped on Moon from 10m",
        "description": "1kg ball dropped from 10m on Moon (g=1.62 m/s^2). No atmosphere.",
        "quantities": {"m": "Mass", "g": "Accel", "h": "Length", "v": "Velocity", "t": "Time"},
        "parameters": {"m": m, "g": g, "h0": h0},
        "timesteps": timesteps,
        "known_invariant": "m*g*h + 0.5*m*v^2",
        "lean_theorem": ""
    }


def make_freefall_mars_drop():
    """Mars gravity: g=3.71 m/s^2, drop from 15m, mass=2kg."""
    g, h0, m = 3.71, 15.0, 2.0
    t_total = math.sqrt(2 * h0 / g)
    ts = linspace(0, t_total, 6)
    timesteps = []
    for t in ts:
        h = h0 - 0.5 * g * t**2
        v = -g * t
        timesteps.append({"t": round(t, 6), "h": round(h, 6), "v": round(v, 6)})
    return {
        "id": "freefall_mars_drop",
        "name": "2kg ball dropped on Mars from 15m",
        "description": "2kg ball dropped from 15m on Mars (g=3.71 m/s^2).",
        "quantities": {"m": "Mass", "g": "Accel", "h": "Length", "v": "Velocity", "t": "Time"},
        "parameters": {"m": m, "g": g, "h0": h0},
        "timesteps": timesteps,
        "known_invariant": "m*g*h + 0.5*m*v^2",
        "lean_theorem": ""
    }


def make_freefall_mars_upward():
    """Mars gravity, thrown upward at 8 m/s."""
    g, v0, m = 3.71, 8.0, 1.0
    t_total = 2 * v0 / g
    ts = linspace(0, t_total, 7)
    timesteps = []
    for t in ts:
        h = v0 * t - 0.5 * g * t**2
        v = v0 - g * t
        timesteps.append({"t": round(t, 6), "h": round(h, 6), "v": round(v, 6)})
    return {
        "id": "freefall_mars_upward",
        "name": "Ball thrown upward on Mars at 8 m/s",
        "description": "1kg ball thrown upward at 8 m/s on Mars (g=3.71 m/s^2).",
        "quantities": {"m": "Mass", "g": "Accel", "h": "Length", "v": "Velocity", "t": "Time"},
        "parameters": {"m": m, "g": g, "v0": v0},
        "timesteps": timesteps,
        "known_invariant": "m*g*h + 0.5*m*v^2",
        "lean_theorem": ""
    }


def make_freefall_air_resistance_linear():
    """Linear air resistance: v' = g - k*v. Energy decays."""
    g, h0, m, k = 9.8, 10.0, 1.0, 0.5
    vt = m * g / k  # terminal velocity = 19.6 m/s
    # h(t) = h0 - vt*t + vt*(m/k)*(1 - exp(-k*t/m))
    tau = m / k
    # Time to hit ground (solve numerically)
    # Approx: h0 ≈ vt * t_fall for large vt*t >> m/k
    # More precisely: h0 = vt*(tf - tau*(1-exp(-tf/tau)))
    # Solve iteratively
    tf = h0 / vt  # initial guess
    for _ in range(20):
        f = vt * (tf - tau * (1 - math.exp(-tf / tau))) - h0
        df = vt * (1 - math.exp(-tf / tau))
        tf -= f / df
    ts = linspace(0, tf, 6)
    timesteps = []
    for t in ts:
        v = -vt * (1 - math.exp(-t / tau))
        h = h0 - vt * t + vt * tau * (1 - math.exp(-t / tau))
        timesteps.append({"t": round(t, 6), "h": round(h, 6), "v": round(v, 6)})
    return {
        "id": "freefall_air_resistance_linear",
        "name": "Ball dropped with linear air resistance",
        "description": "1kg ball dropped from 10m with linear drag (k=0.5 s^-1). Energy not conserved - watch it decay.",
        "quantities": {"m": "Mass", "g": "Accel", "h": "Length", "v": "Velocity", "t": "Time"},
        "parameters": {"m": m, "g": g, "h0": h0, "k": k},
        "timesteps": timesteps,
        "known_invariant": None,
        "lean_theorem": ""
    }


def make_freefall_air_resistance_quadratic():
    """Quadratic air resistance. Energy decays faster."""
    g, h0, m, k = 9.8, 15.0, 1.0, 0.15
    # dv/dt = g - k*v^2 (assuming v > 0 downward)
    # Terminal: vt = sqrt(g/k)
    vt = math.sqrt(g / k)
    # v(t) = vt * tanh(g*t/vt), h(t) = h0 - (vt^2/g)*ln(cosh(g*t/vt))
    # Time to ground: h0 = (vt^2/g)*ln(cosh(g*tf/vt))
    # tf = (vt/g)*acosh(exp(g*h0/vt^2))
    tf = (vt / g) * math.acosh(math.exp(g * h0 / vt**2))
    ts = linspace(0, tf, 6)
    timesteps = []
    for t in ts:
        v = -vt * math.tanh(g * t / vt)
        h = h0 - (vt**2 / g) * math.log(math.cosh(g * t / vt))
        timesteps.append({"t": round(t, 6), "h": round(h, 6), "v": round(v, 6)})
    return {
        "id": "freefall_air_resistance_quadratic",
        "name": "Ball dropped with quadratic air resistance",
        "description": "1kg ball dropped from 15m with quadratic drag (k=0.15). Energy not conserved.",
        "quantities": {"m": "Mass", "g": "Accel", "h": "Length", "v": "Velocity", "t": "Time"},
        "parameters": {"m": m, "g": g, "h0": h0, "k": k},
        "timesteps": timesteps,
        "known_invariant": None,
        "lean_theorem": ""
    }


def make_freefall_high_g():
    """High gravity planet: g=25 m/s^2."""
    g, h0, m = 25.0, 8.0, 3.0
    t_total = math.sqrt(2 * h0 / g)
    ts = linspace(0, t_total, 6)
    timesteps = []
    for t in ts:
        h = h0 - 0.5 * g * t**2
        v = -g * t
        timesteps.append({"t": round(t, 6), "h": round(h, 6), "v": round(v, 6)})
    return {
        "id": "freefall_high_g",
        "name": "Heavy ball on high-gravity planet",
        "description": "3kg ball dropped from 8m on planet with g=25 m/s^2.",
        "quantities": {"m": "Mass", "g": "Accel", "h": "Length", "v": "Velocity", "t": "Time"},
        "parameters": {"m": m, "g": g, "h0": h0},
        "timesteps": timesteps,
        "known_invariant": "m*g*h + 0.5*m*v^2",
        "lean_theorem": ""
    }


def make_freefall_low_g():
    """Very low gravity: g=0.5 m/s^2."""
    g, h0, m = 0.5, 20.0, 1.0
    t_total = math.sqrt(2 * h0 / g)
    ts = linspace(0, t_total, 6)
    timesteps = []
    for t in ts:
        h = h0 - 0.5 * g * t**2
        v = -g * t
        timesteps.append({"t": round(t, 6), "h": round(h, 6), "v": round(v, 6)})
    return {
        "id": "freefall_low_g",
        "name": "Ball dropped on asteroid (low g)",
        "description": "1kg ball dropped from 20m with very low gravity g=0.5 m/s^2.",
        "quantities": {"m": "Mass", "g": "Accel", "h": "Length", "v": "Velocity", "t": "Time"},
        "parameters": {"m": m, "g": g, "h0": h0},
        "timesteps": timesteps,
        "known_invariant": "m*g*h + 0.5*m*v^2",
        "lean_theorem": ""
    }


def make_freefall_downward_throw():
    """Thrown downward from height."""
    g, h0, v0, m = 9.8, 10.0, -10.0, 1.0
    # h = h0 + v0*t - 0.5*g*t^2, solve for h=0
    # v0 is negative (downward)
    a, b, c = -0.5 * g, v0, h0
    disc = b**2 - 4 * a * c
    t_total = (-b - math.sqrt(disc)) / (2 * a) if a != 0 else -h0 / v0
    ts = linspace(0, t_total, 6)
    timesteps = []
    for t in ts:
        h = h0 + v0 * t - 0.5 * g * t**2
        v = v0 - g * t
        timesteps.append({"t": round(t, 6), "h": round(h, 6), "v": round(v, 6)})
    return {
        "id": "freefall_downward_throw",
        "name": "Ball thrown downward from height",
        "description": "1kg ball thrown downward at 10 m/s from 10m height.",
        "quantities": {"m": "Mass", "g": "Accel", "h": "Length", "v": "Velocity", "t": "Time"},
        "parameters": {"m": m, "g": g, "h0": h0, "v0": v0},
        "timesteps": timesteps,
        "known_invariant": "m*g*h + 0.5*m*v^2",
        "lean_theorem": ""
    }


def make_freefall_upward_from_height():
    """Thrown upward from height."""
    g, h0, v0, m = 9.8, 5.0, 5.0, 2.0
    t_peak = v0 / g
    t_total = t_peak + math.sqrt(2 * (h0 + 0.5 * g * t_peak**2) / g)
    ts = linspace(0, t_total, 7)
    timesteps = []
    for t in ts:
        h = h0 + v0 * t - 0.5 * g * t**2
        v = v0 - g * t
        timesteps.append({"t": round(t, 6), "h": round(h, 6), "v": round(v, 6)})
    return {
        "id": "freefall_upward_from_height",
        "name": "Ball thrown upward from elevated position",
        "description": "2kg ball thrown upward at 5 m/s from 5m height.",
        "quantities": {"m": "Mass", "g": "Accel", "h": "Length", "v": "Velocity", "t": "Time"},
        "parameters": {"m": m, "g": g, "h0": h0, "v0": v0},
        "timesteps": timesteps,
        "known_invariant": "m*g*h + 0.5*m*v^2",
        "lean_theorem": ""
    }


def make_freefall_varying_mass_heavy():
    """Different mass, same kinematics."""
    g, h0, m = 9.8, 10.0, 10.0
    t_total = math.sqrt(2 * h0 / g)
    ts = linspace(0, t_total, 6)
    timesteps = []
    for t in ts:
        h = h0 - 0.5 * g * t**2
        v = -g * t
        timesteps.append({"t": round(t, 6), "h": round(h, 6), "v": round(v, 6)})
    return {
        "id": "freefall_heavy_10kg",
        "name": "10kg ball dropped from 10m",
        "description": "10kg ball dropped from 10m. Same kinematics as 1kg but different mass.",
        "quantities": {"m": "Mass", "g": "Accel", "h": "Length", "v": "Velocity", "t": "Time"},
        "parameters": {"m": m, "g": g, "h0": h0},
        "timesteps": timesteps,
        "known_invariant": "m*g*h + 0.5*m*v^2",
        "lean_theorem": ""
    }


# ═══════════════════════════════════════════════════════════════════════════
# Projectile variants (10 new)
# ═══════════════════════════════════════════════════════════════════════════

def make_projectile(oid, name, desc, v0, theta_deg, m, g, n_ts=7):
    """Generate a projectile scenario (no drag)."""
    theta = math.radians(theta_deg)
    vx0 = v0 * math.cos(theta)
    vy0 = v0 * math.sin(theta)
    # Time of flight
    t_total = 2 * vy0 / g
    ts = linspace(0, t_total, n_ts)
    timesteps = []
    for t in ts:
        h = vy0 * t - 0.5 * g * t**2
        vx = vx0
        vy = vy0 - g * t
        v = math.sqrt(vx**2 + vy**2)
        timesteps.append({"t": round(t, 6), "h": round(h, 6), "v": round(v, 6)})
    return {
        "id": oid,
        "name": name,
        "description": desc,
        "quantities": {"m": "Mass", "g": "Accel", "h": "Length", "v": "Velocity", "t": "Time"},
        "parameters": {"m": m, "g": g, "v0": v0, "theta": theta_deg},
        "timesteps": timesteps,
        "known_invariant": "m*g*h + 0.5*m*v^2",
        "lean_theorem": ""
    }


def make_projectile_drag_linear():
    """Projectile with linear drag (energy not conserved)."""
    v0, theta_deg, m, g, k = 20.0, 45.0, 1.0, 9.8, 0.3
    theta = math.radians(theta_deg)
    vx, vy = v0 * math.cos(theta), v0 * math.sin(theta)
    x, y = 0.0, 0.0
    dt = 0.005
    trajectory = []
    step_count = 0
    prev_recorded = -100
    while y >= -0.5 and step_count < 100000:
        # Record every ~0.05s (every 10 steps)
        current_t = step_count * dt
        if len(trajectory) == 0 or current_t - prev_recorded >= 0.04:
            v = math.sqrt(vx**2 + vy**2 + 1e-12)
            trajectory.append({"t": round(current_t, 6),
                              "h": round(max(y, 0.0), 6), "v": round(v, 6)})
            prev_recorded = current_t
            if len(trajectory) >= 15:
                break
        v = math.sqrt(vx**2 + vy**2 + 1e-12)
        ax = -k * vx / m
        ay = -g - k * vy / m
        vx += ax * dt
        vy += ay * dt
        x += vx * dt
        y += vy * dt
        step_count += 1
    # Ensure last point at ground level
    if trajectory and trajectory[-1]["h"] > 0.01:
        trajectory.append({"t": round(trajectory[-1]["t"] + 0.05, 6), "h": 0.0, "v": trajectory[-1]["v"]})
    # Downsample to 6-7 points
    if len(trajectory) > 7:
        idxs = linspace(0, len(trajectory) - 1, 7)
        trajectory = [trajectory[int(round(i))] for i in idxs]
    return {
        "id": "projectile_linear_drag",
        "name": "Projectile with linear air drag",
        "description": "1kg projectile launched at 20 m/s, 45 deg, with linear drag (k=0.3). Energy NOT conserved.",
        "quantities": {"m": "Mass", "g": "Accel", "h": "Length", "v": "Velocity", "t": "Time"},
        "parameters": {"m": m, "g": g, "v0": v0, "theta": theta_deg, "k": k},
        "timesteps": trajectory,
        "known_invariant": None,
        "lean_theorem": ""
    }


def make_projectile_drag_quadratic():
    """Projectile with quadratic drag (energy not conserved)."""
    v0, theta_deg, m, g, k = 20.0, 45.0, 1.0, 9.8, 0.08
    theta = math.radians(theta_deg)
    vx, vy = v0 * math.cos(theta), v0 * math.sin(theta)
    x, y = 0.0, 0.0
    dt = 0.005
    trajectory = []
    step_count = 0
    while y >= -0.5 and step_count < 100000:
        if step_count % 10 == 0:
            v = math.sqrt(vx**2 + vy**2 + 1e-12)
            trajectory.append({"t": round(step_count * dt, 6),
                              "h": round(max(y, 0), 6), "v": round(v, 6)})
            if len(trajectory) >= 15:
                break
        v = math.sqrt(vx**2 + vy**2 + 1e-12)
        ax = -k * v * vx / m
        ay = -g - k * v * vy / m
        vx += ax * dt
        vy += ay * dt
        x += vx * dt
        y += vy * dt
        step_count += 1
    if len(trajectory) > 8:
        idxs = linspace(0, len(trajectory) - 1, 8)
        trajectory = [trajectory[int(round(i))] for i in idxs]
    return {
        "id": "projectile_quadratic_drag",
        "name": "Projectile with quadratic air drag",
        "description": "1kg projectile launched at 20 m/s, 45 deg, with quadratic drag (k=0.08). Energy NOT conserved.",
        "quantities": {"m": "Mass", "g": "Accel", "h": "Length", "v": "Velocity", "t": "Time"},
        "parameters": {"m": m, "g": g, "v0": v0, "theta": theta_deg, "k": k},
        "timesteps": trajectory,
        "known_invariant": None,
        "lean_theorem": ""
    }


# ═══════════════════════════════════════════════════════════════════════════
# Spring variants (10 new)
# ═══════════════════════════════════════════════════════════════════════════

def make_spring_damped_heavy():
    """Heavily damped spring (overdamped, beta > omega0). Energy decays fast."""
    m, k, A, beta = 2.0, 20.0, 0.5, 10.0  # heavily damped
    omega0 = math.sqrt(k / m)
    # Overdamped: beta^2 > k/m => 100 > 10, yes
    gamma1 = -beta + math.sqrt(beta**2 - omega0**2)
    gamma2 = -beta - math.sqrt(beta**2 - omega0**2)
    denom = gamma2 - gamma1
    c1 = A * gamma2 / denom
    c2 = -A * gamma1 / denom
    t_end = 2.0
    ts = linspace(0, t_end, 7)
    timesteps = []
    for t in ts:
        x = c1 * math.exp(gamma1 * t) + c2 * math.exp(gamma2 * t)
        v = c1 * gamma1 * math.exp(gamma1 * t) + c2 * gamma2 * math.exp(gamma2 * t)
        timesteps.append({"t": round(t, 6), "h": round(x, 6), "v": round(v, 6)})
    return {
        "id": "spring_heavily_damped",
        "name": "Heavily damped spring (overdamped)",
        "description": "2kg mass on spring (k=20) with heavy damping (beta=10). No oscillation, slow decay.",
        "quantities": {"m": "Mass", "k": "Force/Length", "h": "Length", "v": "Velocity", "t": "Time"},
        "parameters": {"m": m, "k": k, "A": A, "beta": beta},
        "timesteps": timesteps,
        "known_invariant": None,
        "lean_theorem": ""
    }


def make_spring_forced():
    """Forced spring (sinusoidal driving force). Energy input from external work."""
    m, k, A, F0, omega_f = 1.0, 10.0, 0.1, 0.5, 2.0
    omega0 = math.sqrt(k / m)
    # Steady-state solution for undamped forced oscillator (no damping = resonant growth)
    # x(t) = A*cos(omega0*t) + (F0/(m*(omega0^2-omega_f^2)))*sin(omega_f*t)
    t_end = 10.0
    ts = linspace(0, t_end, 8)
    timesteps = []
    for t in ts:
        transient = A * math.cos(omega0 * t)
        if abs(omega0**2 - omega_f**2) < 1e-6:
            # Resonance: x ~ t*sin(omega_f*t)
            forced = F0 * t * math.sin(omega_f * t) / (2 * m * omega_f)
            v_forced = F0 * (math.sin(omega_f * t) + omega_f * t * math.cos(omega_f * t)) / (2 * m * omega_f)
        else:
            forced = (F0 / (m * (omega0**2 - omega_f**2))) * math.sin(omega_f * t)
            v_forced = (F0 * omega_f / (m * (omega0**2 - omega_f**2))) * math.cos(omega_f * t)
        h = transient + forced
        v = -A * omega0 * math.sin(omega0 * t) + v_forced
        timesteps.append({"t": round(t, 6), "h": round(h, 6), "v": round(v, 6)})
    return {
        "id": "spring_forced",
        "name": "Forced spring oscillator",
        "description": "1kg mass on spring (k=10) driven by external sinusoidal force F=0.5*sin(2t). Energy not conserved due to external work.",
        "quantities": {"m": "Mass", "k": "Force/Length", "h": "Length", "v": "Velocity", "t": "Time"},
        "parameters": {"m": m, "k": k, "A": A, "F0": F0, "omega_f": omega_f},
        "timesteps": timesteps,
        "known_invariant": None,
        "lean_theorem": ""
    }


def make_spring_coupled():
    """Two masses on springs, energy oscillates between them."""
    m, k, A = 1.0, 10.0, 0.3
    # Normal modes: in-phase (omega = sqrt(k/m)), out-of-phase (omega = sqrt(3k/m))
    # Represent as one observable mass with modulated amplitude
    omega0 = math.sqrt(k / m)
    omega_c = math.sqrt(3 * k / m)
    t_end = 5.0
    ts = linspace(0, t_end, 8)
    timesteps = []
    for t in ts:
        # Mass 1: superposition of normal modes
        h = A * (math.cos(omega0 * t) + math.cos(omega_c * t))
        v = -A * (omega0 * math.sin(omega0 * t) + omega_c * math.sin(omega_c * t))
        timesteps.append({"t": round(t, 6), "h": round(h, 6), "v": round(v, 6)})
    return {
        "id": "spring_coupled",
        "name": "Coupled spring oscillators (one mass observed)",
        "description": "One mass observed in a coupled 2-mass spring system. Energy transfers between masses.",
        "quantities": {"m": "Mass", "k": "Force/Length", "h": "Length", "v": "Velocity", "t": "Time"},
        "parameters": {"m": m, "k": k, "A": A},
        "timesteps": timesteps,
        "known_invariant": None,
        "lean_theorem": ""
    }


def make_spring_stiff():
    """Very stiff spring (k=100)."""
    m, k, A = 0.5, 100.0, 0.2
    omega = math.sqrt(k / m)
    t_end = 2 * math.pi / omega  # one period
    ts = linspace(0, t_end, 7)
    timesteps = []
    for t in ts:
        h = A * math.cos(omega * t)
        v = -A * omega * math.sin(omega * t)
        timesteps.append({"t": round(t, 6), "h": round(h, 6), "v": round(v, 6)})
    return {
        "id": "spring_stiff",
        "name": "Stiff spring oscillator",
        "description": "0.5kg mass on stiff spring (k=100 N/m), released from 0.2m displacement.",
        "quantities": {"m": "Mass", "k": "Force/Length", "h": "Length", "v": "Velocity", "t": "Time"},
        "parameters": {"m": m, "k": k, "A": A},
        "timesteps": timesteps,
        "known_invariant": "0.5*m*v^2 + 0.5*k*h^2",
        "lean_theorem": ""
    }


def make_spring_weak():
    """Weak spring (k=2)."""
    m, k, A = 3.0, 2.0, 0.8
    omega = math.sqrt(k / m)
    t_end = 1.5 * 2 * math.pi / omega
    ts = linspace(0, t_end, 7)
    timesteps = []
    for t in ts:
        h = A * math.cos(omega * t)
        v = -A * omega * math.sin(omega * t)
        timesteps.append({"t": round(t, 6), "h": round(h, 6), "v": round(v, 6)})
    return {
        "id": "spring_weak",
        "name": "Weak spring with heavy mass",
        "description": "3kg mass on weak spring (k=2 N/m), released from 0.8m displacement.",
        "quantities": {"m": "Mass", "k": "Force/Length", "h": "Length", "v": "Velocity", "t": "Time"},
        "parameters": {"m": m, "k": k, "A": A},
        "timesteps": timesteps,
        "known_invariant": "0.5*m*v^2 + 0.5*k*h^2",
        "lean_theorem": ""
    }


def make_spring_damped_critical():
    """Critically damped spring."""
    m, k, A = 1.0, 16.0, 0.4
    omega0 = math.sqrt(k / m)  # 4
    beta = omega0  # critical damping
    t_end = 3.0
    ts = linspace(0, t_end, 7)
    timesteps = []
    for t in ts:
        h = A * (1 + beta * t) * math.exp(-beta * t)
        v = A * (beta - beta * (1 + beta * t)) * math.exp(-beta * t)
        timesteps.append({"t": round(t, 6), "h": round(h, 6), "v": round(v, 6)})
    return {
        "id": "spring_critically_damped",
        "name": "Critically damped spring",
        "description": "1kg mass on spring (k=16) with critical damping. Returns to equilibrium fastest without oscillation.",
        "quantities": {"m": "Mass", "k": "Force/Length", "h": "Length", "v": "Velocity", "t": "Time"},
        "parameters": {"m": m, "k": k, "A": A, "beta": beta},
        "timesteps": timesteps,
        "known_invariant": None,
        "lean_theorem": ""
    }


def make_spring_undamped_v2():
    """Undamped spring, different parameters."""
    m, k, A = 2.0, 25.0, 0.3
    omega = math.sqrt(k / m)
    t_end = 2 * math.pi / omega
    ts = linspace(0, t_end, 7)
    timesteps = []
    for t in ts:
        h = A * math.cos(omega * t)
        v = -A * omega * math.sin(omega * t)
        timesteps.append({"t": round(t, 6), "h": round(h, 6), "v": round(v, 6)})
    return {
        "id": "spring_undamped_v2",
        "name": "Undamped spring (variant parameters)",
        "description": "2kg mass on spring (k=25 N/m), released from 0.3m displacement. No damping.",
        "quantities": {"m": "Mass", "k": "Force/Length", "h": "Length", "v": "Velocity", "t": "Time"},
        "parameters": {"m": m, "k": k, "A": A},
        "timesteps": timesteps,
        "known_invariant": "0.5*m*v^2 + 0.5*k*h^2",
        "lean_theorem": ""
    }


# ═══════════════════════════════════════════════════════════════════════════
# Collision scenarios (10 new)
# ═══════════════════════════════════════════════════════════════════════════

def make_collision_elastic_1d():
    """1D elastic collision: m1=1, v1=5 -> m2=1, v2=0 -> v1'=0, v2'=5."""
    # Before collision: m1=1kg, v1=5 m/s approaching stationary m2=1kg
    # After: m1 stops, m2 moves at 5 m/s (elastic, equal mass)
    m = 1.0  # observing mass m1
    timesteps = [
        {"t": 0.0, "h": 0.0, "v": 5.0},
        {"t": 0.1, "h": 0.0, "v": 5.0},
        {"t": 0.2, "h": 0.0, "v": 0.0},  # after collision
        {"t": 0.3, "h": 0.0, "v": 0.0},
        {"t": 0.4, "h": 0.0, "v": 0.0},
    ]
    return {
        "id": "collision_elastic_1d_equal_mass",
        "name": "1D elastic collision, equal masses",
        "description": "1kg ball at 5 m/s collides elastically with stationary 1kg ball. After: m1 stops. Before/after piecewise.",
        "quantities": {"m": "Mass", "v": "Velocity", "t": "Time"},
        "parameters": {"m": m, "m2": 1.0},
        "timesteps": timesteps,
        "known_invariant": "0.5*m*v^2",
        "lean_theorem": "",
        "piecewise": {"events": [{"t": 0.25, "name": "collision"}]}
    }


def make_collision_elastic_unequal():
    """1D elastic collision: m1=1, v1=5 -> m2=3, v2=0."""
    m1, m2 = 1.0, 3.0
    v1i, v2i = 5.0, 0.0
    # Elastic: v1f = (m1-m2)/(m1+m2)*v1i = -2.5, v2f = 2*m1/(m1+m2)*v1i = 2.5
    v1f = (m1 - m2) / (m1 + m2) * v1i
    timesteps = [
        {"t": 0.0, "h": 0.0, "v": v1i},
        {"t": 0.1, "h": 0.0, "v": v1i},
        {"t": 0.2, "h": 0.0, "v": v1f},  # after: rebounds at -2.5
        {"t": 0.3, "h": 0.0, "v": v1f},
        {"t": 0.4, "h": 0.0, "v": v1f},
    ]
    return {
        "id": "collision_elastic_1d_unequal_mass",
        "name": "1D elastic collision, light hitting heavy",
        "description": "1kg ball at 5 m/s hits stationary 3kg ball elastically. m1 rebounds backward.",
        "quantities": {"m": "Mass", "v": "Velocity", "t": "Time"},
        "parameters": {"m": m1, "m2": m2},
        "timesteps": timesteps,
        "known_invariant": "0.5*m*v^2",
        "lean_theorem": "",
        "piecewise": {"events": [{"t": 0.15, "name": "collision"}]}
    }


def make_collision_inelastic():
    """Perfectly inelastic collision: m1=m2=1, v1=5, v2=0 -> v_after=2.5. Energy NOT conserved."""
    m, v1i = 1.0, 5.0
    vf = v1i / 2  # equal mass perfectly inelastic
    timesteps = [
        {"t": 0.0, "h": 0.0, "v": 5.0},
        {"t": 0.1, "h": 0.0, "v": 5.0},
        {"t": 0.2, "h": 0.0, "v": vf},
        {"t": 0.3, "h": 0.0, "v": vf},
        {"t": 0.4, "h": 0.0, "v": vf},
    ]
    return {
        "id": "collision_inelastic_equal_mass",
        "name": "Perfectly inelastic collision, equal masses",
        "description": "1kg ball at 5 m/s hits stationary 1kg ball, they stick. Kinetic energy NOT conserved (half lost to heat).",
        "quantities": {"m": "Mass", "v": "Velocity", "t": "Time"},
        "parameters": {"m": m, "m2": 1.0},
        "timesteps": timesteps,
        "known_invariant": None,
        "lean_theorem": "",
        "piecewise": {"events": [{"t": 0.15, "name": "collision"}]}
    }


def make_collision_elastic_2d():
    """2D elastic collision: m1=1 moving along x at 5 m/s hits stationary m2=1.
    After: m1 goes up, m2 goes right. v changes direction."""
    # Before: v=(5,0). After elastic 2D at 45deg: v=(5*cos(60), 5*sin(60)) approx
    # For simplicity: v before = 5, v after = 5 (elastic, equal mass, glancing)
    timesteps = [
        {"t": 0.0, "h": 0.0, "v": 5.0},
        {"t": 0.1, "h": 0.0, "v": 5.0},
        {"t": 0.2, "h": 0.0, "v": 3.535},  # 5*cos(45)
        {"t": 0.3, "h": 0.0, "v": 3.535},
        {"t": 0.4, "h": 0.0, "v": 3.535},
    ]
    return {
        "id": "collision_elastic_2d_glancing",
        "name": "2D elastic glancing collision",
        "description": "1kg ball at 5 m/s glances off stationary 1kg ball. Speed changes (direction change). Energy conserved but observed speed decreases.",
        "quantities": {"m": "Mass", "v": "Velocity", "t": "Time"},
        "parameters": {"m": 1.0, "m2": 1.0},
        "timesteps": timesteps,
        "known_invariant": None,
        "lean_theorem": "",
        "piecewise": {"events": [{"t": 0.15, "name": "collision"}]}
    }


def make_collision_2mass():
    """Two masses collide, observe both. m1=1@5m/s + m2=2@0m/s."""
    # We're observing one mass (m1)
    m1, m2, v1i, v2i = 1.0, 2.0, 5.0, 0.0
    v1f = (m1 - m2) / (m1 + m2) * v1i  # elastic
    timesteps = [
        {"t": 0.0, "h": 0.0, "v": v1i},
        {"t": 0.1, "h": 0.0, "v": v1i},
        {"t": 0.2, "h": 0.0, "v": round(v1f, 6)},
        {"t": 0.3, "h": 0.0, "v": round(v1f, 6)},
        {"t": 0.4, "h": 0.0, "v": round(v1f, 6)},
    ]
    return {
        "id": "collision_elastic_1d_2to1_mass",
        "name": "1D elastic collision, light hitting medium",
        "description": "1kg ball at 5 m/s hits stationary 2kg ball elastically.",
        "quantities": {"m": "Mass", "v": "Velocity", "t": "Time"},
        "parameters": {"m": m1, "m2": m2},
        "timesteps": timesteps,
        "known_invariant": "0.5*m*v^2",
        "lean_theorem": "",
        "piecewise": {"events": [{"t": 0.15, "name": "collision"}]}
    }


def make_collision_inelastic_unequal():
    """Inelastic collision: m1=2@5, m2=1@0 -> vf = (2*5+1*0)/3 = 3.33."""
    m1, m2, v1i = 2.0, 1.0, 5.0
    vf = m1 * v1i / (m1 + m2)
    timesteps = [
        {"t": 0.0, "h": 0.0, "v": v1i},
        {"t": 0.1, "h": 0.0, "v": v1i},
        {"t": 0.2, "h": 0.0, "v": round(vf, 6)},
        {"t": 0.3, "h": 0.0, "v": round(vf, 6)},
        {"t": 0.4, "h": 0.0, "v": round(vf, 6)},
    ]
    return {
        "id": "collision_inelastic_unequal_mass",
        "name": "Perfectly inelastic collision, heavy hitting light",
        "description": "2kg ball at 5 m/s hits stationary 1kg ball, they stick. Energy NOT conserved.",
        "quantities": {"m": "Mass", "v": "Velocity", "t": "Time"},
        "parameters": {"m": m1, "m2": m2},
        "timesteps": timesteps,
        "known_invariant": None,
        "lean_theorem": "",
        "piecewise": {"events": [{"t": 0.15, "name": "collision"}]}
    }


def make_collision_headon():
    """Head-on elastic: m1=2@3 + m2=2@-1 -> v1f=-1, v2f=3."""
    m, v1i, v2i = 2.0, 3.0, -1.0  # m1=m2=2
    v1f = v2i  # equal mass elastic head-on -> exchange velocities
    timesteps = [
        {"t": 0.0, "h": 0.0, "v": v1i},
        {"t": 0.1, "h": 0.0, "v": v1i},
        {"t": 0.2, "h": 0.0, "v": v1f},
        {"t": 0.3, "h": 0.0, "v": v1f},
        {"t": 0.4, "h": 0.0, "v": v1f},
    ]
    return {
        "id": "collision_elastic_headon",
        "name": "Head-on elastic collision, equal masses",
        "description": "2kg ball at 3 m/s hits 2kg ball at -1 m/s head-on elastically. They exchange velocities.",
        "quantities": {"m": "Mass", "v": "Velocity", "t": "Time"},
        "parameters": {"m": m},
        "timesteps": timesteps,
        "known_invariant": "0.5*m*v^2",
        "lean_theorem": "",
        "piecewise": {"events": [{"t": 0.15, "name": "collision"}]}
    }


# ═══════════════════════════════════════════════════════════════════════════
# Incline variants (10 new)
# ═══════════════════════════════════════════════════════════════════════════

def make_incline(oid, name, desc, theta_deg, h0, m, g, mu=0.0):
    """Block sliding down an incline.
    Without friction: a = g*sin(theta), h decreases by sin(theta)*s.
    With friction: a = g*(sin(theta) - mu*cos(theta)).
    """
    theta = math.radians(theta_deg)
    if mu > 0:
        a = g * (math.sin(theta) - mu * math.cos(theta))
    else:
        a = g * math.sin(theta)
    if a <= 0:
        a = 0.001  # prevent stuck block
    # Distance along incline to lose h0 height: L = h0 / sin(theta)
    L = h0 / math.sin(theta)
    t_total = math.sqrt(2 * L / a)
    ts = linspace(0, t_total, 6)
    timesteps = []
    for t in ts:
        s = 0.5 * a * t**2  # distance along incline
        h = h0 - s * math.sin(theta)
        v = a * t  # speed along incline
        timesteps.append({"t": round(t, 6), "h": round(h, 6), "v": round(v, 6)})
    inv = "m*g*h + 0.5*m*v^2" if mu == 0 else None
    return {
        "id": oid,
        "name": name,
        "description": desc,
        "quantities": {"m": "Mass", "g": "Accel", "h": "Length", "v": "Velocity", "t": "Time"},
        "parameters": {"m": m, "g": g, "theta": theta_deg, "h0": h0, "mu": mu},
        "timesteps": timesteps,
        "known_invariant": inv,
        "lean_theorem": ""
    }


# ═══════════════════════════════════════════════════════════════════════════
# Cross-domain composition scenarios
# ═══════════════════════════════════════════════════════════════════════════

def make_mass_spring_gravity():
    """Mass on vertical spring under gravity.
    Equilibrium: k*x_eq = m*g => x_eq = mg/k.
    Oscillation: x(t) = x_eq + A*cos(omega*t) where omega=sqrt(k/m).
    h(t) = x_eq + A*cos(omega*t) (position measured from unstretched)
    v(t) = -A*omega*sin(omega*t)
    Invariant: 0.5*m*v^2 + 0.5*k*(h - x_eq)^2 + m*g*h = const
    But simpler: total energy = 0.5*m*v^2 + m*g*h + 0.5*k*h^2 - m*g*x0
    where x0 is reference point. At h measured from unstretched:
    E = 0.5*m*v^2 + m*g*h + 0.5*k*h^2
    Actually for vertical spring: E = 0.5*m*v^2 + 0.5*k*(h - h_eq)^2 + m*g*h
    where h_eq = h0 - mg/k. The full expression is:
    E_total = 0.5*m*v^2 + m*g*h + 0.5*k*h^2
    (if h measured from unstretched position)
    """
    m, k, g, A = 1.0, 10.0, 9.8, 0.3
    h_eq = m * g / k  # equilibrium stretch
    omega = math.sqrt(k / m)
    t_end = 2 * math.pi / omega
    ts = linspace(0, t_end, 7)
    timesteps = []
    for t in ts:
        h = h_eq + A * math.cos(omega * t)  # position from unstretched
        v = -A * omega * math.sin(omega * t)
        timesteps.append({"t": round(t, 6), "h": round(h, 6), "v": round(v, 6)})
    return {
        "id": "mass_spring_gravity",
        "name": "Mass on vertical spring under gravity",
        "description": "1kg mass on vertical spring (k=10) under Earth gravity. Combined conserved: 0.5*m*v^2 + 0.5*k*h^2 - m*g*h.",
        "quantities": {"m": "Mass", "k": "Force/Length", "g": "Accel", "h": "Length", "v": "Velocity", "t": "Time"},
        "parameters": {"m": m, "k": k, "g": g, "A": A},
        "timesteps": timesteps,
        "known_invariant": "0.5*m*v^2 + 0.5*k*h^2 - m*g*h",
        "lean_theorem": ""
    }


def make_pendulum_air_resistance():
    """Pendulum with air resistance (energy decays)."""
    m, g, L, theta0, k = 1.0, 9.8, 1.0, 0.3, 0.2
    # Damped pendulum: theta'' + (k/m)*theta' + (g/L)*sin(theta) = 0
    # Small angle: theta'' + (k/m)theta' + (g/L)theta = 0
    omega0 = math.sqrt(g / L)
    beta = k / (2 * m)
    omega_d = math.sqrt(omega0**2 - beta**2)  # underdamped if omega0 > beta
    if omega0 > beta:
        t_end = 3 * 2 * math.pi / omega_d
        ts = linspace(0, t_end, 8)
        timesteps = []
        for t in ts:
            theta = theta0 * math.exp(-beta * t) * math.cos(omega_d * t)
            h = L * (1 - math.cos(theta))
            v_theta = -theta0 * math.exp(-beta * t) * (beta * math.cos(omega_d * t) + omega_d * math.sin(omega_d * t))
            v = L * abs(v_theta)
            timesteps.append({"t": round(t, 6), "h": round(h, 6), "v": round(v, 6)})
    else:
        t_end = 5.0
        ts = linspace(0, t_end, 6)
        timesteps = []
        for t in ts:
            theta = theta0 * math.exp(-omega0 * t * 0.5)
            h = L * (1 - math.cos(theta))
            v = L * omega0 * abs(theta)
            timesteps.append({"t": round(t, 6), "h": round(h, 6), "v": round(v, 6)})
    return {
        "id": "pendulum_air_resistance",
        "name": "Pendulum with air resistance",
        "description": "1kg pendulum (L=1m) released from 0.3 rad with light air resistance (k=0.2). Energy gradually decays.",
        "quantities": {"m": "Mass", "g": "Accel", "h": "Length", "v": "Velocity", "t": "Time", "L": "Length"},
        "parameters": {"m": m, "g": g, "L": L, "theta0": theta0, "k": k},
        "timesteps": timesteps,
        "known_invariant": None,
        "lean_theorem": ""
    }


def make_charged_particle_gravity():
    """Charged particle in uniform E field + gravity.
    Forces: F = mg + qE. Both conservative -> total energy conserved.
    E_total = 0.5*m*v^2 + m*g*h + q*E*h (if E upward)
    or 0.5*m*v^2 + m*g*h - q*E*h (if E downward, same direction as g)
    h(t) = h0 - 0.5*(g + qE/m)*t^2, v = -(g + qE/m)*t
    """
    m, g, q, E_field, h0 = 1.0, 9.8, 2.0, 3.0, 10.0
    # Net acceleration: a = g + q*E/m = 9.8 + 6 = 15.8 m/s^2 (if E downward)
    a = g + q * E_field / m
    t_total = math.sqrt(2 * h0 / a)
    ts = linspace(0, t_total, 6)
    timesteps = []
    for t in ts:
        h = h0 - 0.5 * a * t**2
        v = -a * t
        timesteps.append({"t": round(t, 6), "h": round(h, 6), "v": round(v, 6)})
    return {
        "id": "charged_particle_gravity",
        "name": "Charged particle in gravity + electric field",
        "description": "1kg charged particle (q=2C) falling in gravity (g=9.8) + uniform downward E-field (E=3 V/m). Combined conservative force.",
        "quantities": {"m": "Mass", "g": "Accel", "h": "Length", "v": "Velocity", "t": "Time"},
        "parameters": {"m": m, "g": g, "q": q, "E": E_field, "h0": h0},
        "timesteps": timesteps,
        "known_invariant": "0.5*m*v^2 + m*g*h + q*E*h",
        "lean_theorem": ""
    }


def make_mass_spring_damped_gravity():
    """Mass on vertical spring under gravity with stronger damping. Energy not conserved."""
    m, k, g, A, beta = 1.0, 12.0, 9.8, 0.3, 2.5  # stronger damping
    h_eq = m * g / k
    omega0 = math.sqrt(k / m)
    if beta >= omega0:
        # Overdamped
        t_end = 3.0
        ts = linspace(0, t_end, 7)
        timesteps = []
        gamma = math.sqrt(beta**2 - omega0**2)
        for t in ts:
            h = h_eq + A * math.exp(-beta * t)
            v = -A * beta * math.exp(-beta * t)
            timesteps.append({"t": round(t, 6), "h": round(h, 6), "v": round(v, 6)})
    else:
        omega_d = math.sqrt(omega0**2 - beta**2)
        t_end = 2 * math.pi / omega_d
        ts = linspace(0, t_end, 7)
        timesteps = []
        for t in ts:
            h = h_eq + A * math.exp(-beta * t) * math.cos(omega_d * t)
            v = -A * math.exp(-beta * t) * (beta * math.cos(omega_d * t) + omega_d * math.sin(omega_d * t))
            timesteps.append({"t": round(t, 6), "h": round(h, 6), "v": round(v, 6)})
    return {
        "id": "mass_spring_damped_gravity",
        "name": "Damped mass-spring under gravity",
        "description": "1kg mass on vertical damped spring (k=12, beta=0.8) under gravity. Energy not conserved due to damping.",
        "quantities": {"m": "Mass", "k": "Force/Length", "g": "Accel", "h": "Length", "v": "Velocity", "t": "Time"},
        "parameters": {"m": m, "k": k, "g": g, "A": A, "beta": beta},
        "timesteps": timesteps,
        "known_invariant": None,
        "lean_theorem": ""
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main: assemble all scenarios
# ═══════════════════════════════════════════════════════════════════════════

def generate():
    phase1 = load_phase1()

    # Free-fall variants
    freefall_new = [
        make_freefall_moon(),
        make_freefall_mars_drop(),
        make_freefall_mars_upward(),
        make_freefall_air_resistance_linear(),
        make_freefall_air_resistance_quadratic(),
        make_freefall_high_g(),
        make_freefall_low_g(),
        make_freefall_downward_throw(),
        make_freefall_upward_from_height(),
        make_freefall_varying_mass_heavy(),
    ]

    # Projectile variants
    projectile_new = [
        make_projectile("projectile_30deg", "Projectile at 30 degrees",
                        "1kg projectile at 10 m/s, 30 deg above horizontal.", 10.0, 30.0, 1.0, 9.8),
        make_projectile("projectile_60deg", "Projectile at 60 degrees",
                        "1kg projectile at 12 m/s, 60 deg above horizontal.", 12.0, 60.0, 1.0, 9.8),
        make_projectile("projectile_20deg", "Projectile at shallow 20 degrees",
                        "2kg projectile at 15 m/s, 20 deg above horizontal.", 15.0, 20.0, 2.0, 9.8),
        make_projectile("projectile_75deg", "Projectile at steep 75 degrees",
                        "1kg projectile at 8 m/s, 75 deg above horizontal.", 8.0, 75.0, 1.0, 9.8),
        make_projectile_drag_linear(),
        make_projectile_drag_quadratic(),
        make_projectile("projectile_high_speed", "High-speed projectile",
                        "1kg projectile at 50 m/s, 45 deg.", 50.0, 45.0, 1.0, 9.8, 10),
        make_projectile("projectile_low_speed", "Low-speed heavy projectile",
                        "5kg projectile at 3 m/s, 45 deg.", 3.0, 45.0, 5.0, 9.8, 6),
        make_projectile("projectile_10deg", "Very shallow projectile",
                        "1kg projectile at 10 m/s, 10 deg above horizontal.", 10.0, 10.0, 1.0, 9.8),
        make_projectile("projectile_mars", "Projectile on Mars",
                        "1kg projectile at 10 m/s, 45 deg on Mars (g=3.71).", 10.0, 45.0, 1.0, 3.71),
    ]

    # Spring variants
    spring_new = [
        make_spring_damped_heavy(),
        make_spring_forced(),
        make_spring_coupled(),
        make_spring_stiff(),
        make_spring_weak(),
        make_spring_damped_critical(),
        make_spring_undamped_v2(),
        # Additional spring variants
        make_spring_damped_medium(),
    ]

    # Collision scenarios
    collision_new = [
        make_collision_elastic_1d(),
        make_collision_elastic_unequal(),
        make_collision_inelastic(),
        make_collision_elastic_2d(),
        make_collision_2mass(),
        make_collision_inelastic_unequal(),
        make_collision_headon(),
    ]

    # Incline variants
    incline_new = [
        make_incline("incline_20deg", "20-degree frictionless incline",
                     "1kg block sliding down 20 deg frictionless incline from 5m.", 20.0, 5.0, 1.0, 9.8, 0.0),
        make_incline("incline_45deg", "45-degree frictionless incline",
                     "1kg block sliding down 45 deg frictionless incline from 8m.", 45.0, 8.0, 1.0, 9.8, 0.0),
        make_incline("incline_60deg", "Steep 60-degree frictionless incline",
                     "2kg block sliding down 60 deg frictionless incline from 6m.", 60.0, 6.0, 2.0, 9.8, 0.0),
        make_incline("incline_10deg", "Shallow 10-degree frictionless incline",
                     "1kg block sliding down 10 deg frictionless incline from 3m.", 10.0, 3.0, 1.0, 9.8, 0.0),
        # Inclines WITH friction (energy NOT conserved)
        make_incline("incline_20deg_friction", "20-degree incline with friction (mu=0.3)",
                     "1kg block sliding down 20 deg incline with friction mu=0.3.", 20.0, 5.0, 1.0, 9.8, 0.3),
        make_incline("incline_30deg_friction", "30-degree incline with friction (mu=0.2)",
                     "1kg block sliding down 30 deg incline with friction mu=0.2 from 6m.", 30.0, 6.0, 1.0, 9.8, 0.2),
        make_incline("incline_45deg_friction", "45-degree incline with friction (mu=0.4)",
                     "2kg block sliding down 45 deg incline with friction mu=0.4 from 4m.", 45.0, 4.0, 2.0, 9.8, 0.4),
        make_incline("incline_15deg_friction", "15-degree incline with friction (mu=0.15)",
                     "1kg block sliding down 15 deg incline with friction mu=0.15.", 15.0, 4.0, 1.0, 9.8, 0.15),
    ]

    # Cross-domain composition
    composition = [
        make_mass_spring_gravity(),
        make_pendulum_air_resistance(),
        make_charged_particle_gravity(),
        make_mass_spring_damped_gravity(),
    ]

    all_scenarios = (
        phase1 +
        freefall_new +
        projectile_new +
        spring_new +
        collision_new +
        incline_new +
        composition
    )

    # Remove piecewise metadata for cleaner JSON (keep it as observation metadata if needed)
    for s in all_scenarios:
        s.pop("piecewise", None)

    print(f"Generated {len(all_scenarios)} scenarios:")
    print(f"  Phase 1 (existing):    {len(phase1)}")
    print(f"  Free-fall variants:    {len(freefall_new)}")
    print(f"  Projectile variants:   {len(projectile_new)}")
    print(f"  Spring variants:       {len(spring_new)}")
    print(f"  Collision scenarios:   {len(collision_new)}")
    print(f"  Incline variants:      {len(incline_new)}")
    print(f"  Cross-domain comp:     {len(composition)}")

    # Count with/without invariants
    with_inv = sum(1 for s in all_scenarios if s["known_invariant"] is not None)
    without_inv = sum(1 for s in all_scenarios if s["known_invariant"] is None)
    print(f"  With known invariant:  {with_inv}")
    print(f"  Without invariant:     {without_inv}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_scenarios, f, indent=2)
    print(f"Wrote {OUTPUT_PATH}")


# ═══════════════════════════════════════════════════════════════════════════
# Extra: medium-damped spring (for spring count)
# ═══════════════════════════════════════════════════════════════════════════

def make_spring_damped_medium():
    """Medium damping, still underdamped."""
    m, k, A, beta = 1.5, 15.0, 0.4, 1.0
    omega0 = math.sqrt(k / m)
    omega_d = math.sqrt(omega0**2 - beta**2)
    t_end = 2 * math.pi / omega_d
    ts = linspace(0, t_end, 7)
    timesteps = []
    for t in ts:
        h = A * math.exp(-beta * t) * math.cos(omega_d * t)
        v = -A * math.exp(-beta * t) * (beta * math.cos(omega_d * t) + omega_d * math.sin(omega_d * t))
        timesteps.append({"t": round(t, 6), "h": round(h, 6), "v": round(v, 6)})
    return {
        "id": "spring_medium_damped",
        "name": "Medium damped spring oscillator",
        "description": "1.5kg mass on spring (k=15) with medium damping (beta=1.0). Oscillates with decaying amplitude.",
        "quantities": {"m": "Mass", "k": "Force/Length", "h": "Length", "v": "Velocity", "t": "Time"},
        "parameters": {"m": m, "k": k, "A": A, "beta": beta},
        "timesteps": timesteps,
        "known_invariant": None,
        "lean_theorem": ""
    }


if __name__ == "__main__":
    generate()
