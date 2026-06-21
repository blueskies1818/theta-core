"""Quantum mechanics observation simulator.

Generates Observation-compatible dicts for quantum systems:
1. Particle in infinite square well — quantized energy levels
2. Quantum harmonic oscillator — equally spaced energy levels
3. Hydrogen atom — Rydberg energy levels
4. Probability current (1D free particle)
5. Expectation values in superposition states
6. Free wave-packet evolution
7. U(1) phase symmetry — probability conservation

All observables: energy eigenvalues (stationary states), probability
currents, expectation values. Known invariants are recorded for
acceptance testing only — not injected into the discovery model.

Symmetries:
  - U(1) phase rotation → probability conservation (|ψ|² invariant)
  - Time translation    → energy quantization (E_n constant)
"""

from __future__ import annotations

import math
from typing import Any


# Physical constants (SI units)
HBAR = 1.054571817e-34   # Reduced Planck constant (J·s)
M_E = 9.10938356e-31     # Electron mass (kg)
E_CHARGE = 1.602176634e-19  # Elementary charge (C)
EPSILON_0 = 8.854187817e-12  # Vacuum permittivity (F/m)
K_E = 1.0 / (4 * math.pi * EPSILON_0)  # Coulomb constant
EV_TO_J = 1.602176634e-19  # 1 eV in Joules


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
# 1. Particle in infinite square well
# ═══════════════════════════════════════════════════════════════════════════


def simulate_particle_in_box(
    L: float = 1e-9,
    m: float = M_E,
    n_max: int = 5,
    n_steps: int = 20,
) -> dict[str, Any]:
    """Simulate particle in 1D infinite square well (0 to L).

    Energy eigenvalues: E_n = n² π² ℏ² / (2 m L²)

    Stationary state wavefunction: ψ_n(x) = sqrt(2/L) sin(nπx/L)
    Probability density: |ψ_n(x)|² = (2/L) sin²(nπx/L)

    Invariant: E_n is constant for each quantum number n.
    The pattern E_n ∝ n² is the key discovery.

    Parameters
    ----------
    L : float
        Box width (m). Default 1nm.
    m : float
        Particle mass (kg). Default electron mass.
    n_max : int
        Number of energy levels to compute.
    n_steps : int
        Number of spatial positions to sample.
    """
    E1 = (math.pi ** 2) * (HBAR ** 2) / (2 * m * (L ** 2))
    E1_ev = E1 / EV_TO_J

    timesteps: list[dict[str, float]] = []
    for i in range(n_steps):
        # Spatial positions across the well
        x = L * i / (n_steps - 1) if n_steps > 1 else 0.5 * L
        probs: dict[str, float] = {"t": round(i * 1e-15, 20), "x": round(x, 12)}
        for n_val in range(1, n_max + 1):
            psi_n = math.sqrt(2.0 / L) * math.sin(n_val * math.pi * x / L)
            probs[f"psi{n_val}_sq"] = round(psi_n ** 2, 12)
        timesteps.append({"x": round(x, 12)} | probs)

    # Compute energy levels
    energy_levels = {}
    for n_val in range(1, n_max + 1):
        energy_levels[f"E{n_val}"] = round(n_val ** 2 * E1_ev, 6)

    desc = f"Particle in 1D box: L={L:.1e}m, m={m:.2e}kg"
    return _make_obs(
        obs_id=f"particle_in_box_L{L:.1e}_m{m:.1e}",
        name=f"Infinite square well (L={L:.1e}m)",
        description=desc,
        quantities={
            "L": "Length", "m": "Mass",
            "hbar": "Energy*Time",
            "E": "Energy", "n": "Scalar",
            "x": "Length",
        },
        parameters={"L": L, "m": m, "hbar": HBAR, "E1_ev": round(E1_ev, 6)} | energy_levels,
        timesteps=timesteps,
        known_invariant="E / n^2",
        lean_theorem="",
        is_conservative=True,
    )


def generate_particle_in_box_scenarios() -> list[dict[str, Any]]:
    """Generate diverse particle-in-box scenarios."""
    scenarios: list[dict[str, Any]] = []

    configs = [
        (1e-9, M_E),          # 1nm box, electron
        (2e-9, M_E),          # 2nm box
        (0.5e-9, M_E),        # 0.5nm box
        (1e-9, 2 * M_E),      # Heavier particle
        (1e-9, 0.5 * M_E),    # Lighter particle
        (5e-9, M_E),          # Wide box
        (0.2e-9, M_E),        # Very narrow box
        (3e-9, 3 * M_E),      # Wide, heavy
        (0.8e-9, 1.5 * M_E),  # Medium
        (1.5e-9, M_E),        # 1.5nm
    ]

    for L, m in configs:
        scenarios.append(simulate_particle_in_box(L=L, m=m, n_max=5, n_steps=30))

    return scenarios


# ═══════════════════════════════════════════════════════════════════════════
# 2. Quantum harmonic oscillator
# ═══════════════════════════════════════════════════════════════════════════


def simulate_harmonic_oscillator(
    omega: float = 1e15,
    m: float = M_E,
    n_max: int = 5,
    n_steps: int = 30,
    x_range: float | None = None,
) -> dict[str, Any]:
    """Simulate quantum harmonic oscillator.

    Energy eigenvalues: E_n = ℏ ω (n + ½)
    Ground state: E₀ = ½ ℏ ω

    Characteristic length: a = sqrt(ℏ/(mω))

    Invariant: E_n - E_{n-1} = ℏω (constant spacing)
    Equivalently: (E - 0.5*hbar*omega) / n = hbar*omega (per quantum)

    Parameters
    ----------
    omega : float
        Angular frequency (rad/s).
    m : float
        Particle mass (kg).
    n_max : int
        Number of energy levels.
    n_steps : int
        Number of position samples.
    x_range : float | None
        Position range for sampling (±x_range). Default: 5*a.
    """
    a0 = math.sqrt(HBAR / (m * omega))  # characteristic length
    if x_range is None:
        x_range = 5.0 * a0

    E0 = 0.5 * HBAR * omega
    E0_ev = E0 / EV_TO_J

    # Compute energy levels
    energy_levels = {}
    for n_val in range(n_max):
        energy_levels[f"E{n_val}"] = round((n_val + 0.5) * HBAR * omega / EV_TO_J, 6)

    # Hermite polynomial evaluations for wavefunctions (using recurrence)
    # H_0 = 1, H_1 = 2x, H_{n+1} = 2x H_n - 2n H_{n-1}
    def hermite(n: int, x: float) -> float:
        if n == 0:
            return 1.0
        if n == 1:
            return 2.0 * x
        h_prev = 1.0
        h_curr = 2.0 * x
        for k in range(1, n):
            h_next = 2.0 * x * h_curr - 2.0 * k * h_prev
            h_prev = h_curr
            h_curr = h_next
        return h_curr

    timesteps: list[dict[str, float]] = []
    for i in range(n_steps):
        x = -x_range + 2 * x_range * i / (n_steps - 1) if n_steps > 1 else 0.0
        xi = x / a0
        row = {"t": round(i * 1e-15, 20), "x": round(x, 12)}
        for n_val in range(min(n_max, 4)):  # Show first 4 states
            norm = 1.0 / math.sqrt(2**n_val * math.factorial(n_val) * a0 * math.sqrt(math.pi))
            psi_n = norm * hermite(n_val, xi) * math.exp(-0.5 * xi**2)
            row[f"psi{n_val}_sq"] = round(psi_n ** 2, 12)
        timesteps.append(row)

    desc = f"Quantum harmonic oscillator: ω={omega:.1e}rad/s, m={m:.2e}kg"
    return _make_obs(
        obs_id=f"harmonic_osc_omega{omega:.1e}_m{m:.1e}",
        name=f"Harmonic oscillator (ω={omega:.1e})",
        description=desc,
        quantities={
            "omega": "1/Time", "m": "Mass",
            "hbar": "Energy*Time",
            "E": "Energy", "n": "Scalar",
            "x": "Length",
        },
        parameters={"omega": omega, "m": m, "hbar": HBAR, "E0_ev": round(E0_ev, 6)} | energy_levels,
        timesteps=timesteps,
        known_invariant="E / (hbar*omega)",
        lean_theorem="",
        is_conservative=True,
    )


def generate_harmonic_oscillator_scenarios() -> list[dict[str, Any]]:
    """Generate diverse harmonic oscillator scenarios."""
    scenarios: list[dict[str, Any]] = []

    configs = [
        (1e15, M_E),           # Visible light frequency, electron
        (2e15, M_E),           # Higher frequency
        (5e14, M_E),           # Lower frequency
        (1e15, 2 * M_E),       # Heavier
        (1e15, 0.5 * M_E),     # Lighter
        (3e15, M_E),           # UV frequency
        (1e14, M_E),           # IR frequency
        (1.5e15, 1.5 * M_E),   # Medium both
        (2e15, 2 * M_E),       # High freq, heavy
        (5e14, 0.5 * M_E),     # Low freq, light
    ]

    for omega, m in configs:
        scenarios.append(simulate_harmonic_oscillator(omega=omega, m=m, n_max=5, n_steps=40))

    return scenarios


# ═══════════════════════════════════════════════════════════════════════════
# 3. Hydrogen atom energy levels
# ═══════════════════════════════════════════════════════════════════════════


def simulate_hydrogen_atom(
    Z: int = 1,
    n_max: int = 6,
    n_steps: int = 20,
) -> dict[str, Any]:
    """Simulate hydrogen-like atom energy levels.

    Energy eigenvalues: E_n = -13.6 eV * Z² / n²

    For hydrogen (Z=1): -13.6, -3.4, -1.51, -0.85, -0.54, -0.378 eV

    Invariant: E_n * n² / Z² = -13.6 eV (Rydberg constant in eV)

    Parameters
    ----------
    Z : int
        Atomic number (Z=1 for hydrogen).
    n_max : int
        Maximum principal quantum number.
    n_steps : int
        Number of radial position samples.
    """
    RYDBERG_EV = 13.605693  # Rydberg energy in eV

    energy_levels = {}
    for n_val in range(1, n_max + 1):
        energy_levels[f"E{n_val}"] = round(-RYDBERG_EV * Z**2 / n_val**2, 6)

    # Radial probability density for ground state (n=1, l=0)
    a0 = 5.29177210903e-11  # Bohr radius (m)
    a0_scaled = a0 / Z  # scaled Bohr radius

    timesteps: list[dict[str, float]] = []
    for i in range(n_steps):
        r = a0_scaled * (0.05 + 5.0 * i / (n_steps - 1)) if n_steps > 1 else a0_scaled
        # Ground state radial probability: P(r) = (4r²/a³) * exp(-2r/a)
        r_a = r / a0_scaled
        prob_1s = (4.0 * r_a**2) * math.exp(-2.0 * r_a)
        timesteps.append({
            "t": round(i * 1e-16, 20),
            "r": round(r, 14),
            "prob_1s": round(prob_1s, 12),
        })

    desc = f"Hydrogen atom: Z={Z}, energy levels up to n={n_max}"
    return _make_obs(
        obs_id=f"hydrogen_Z{Z}",
        name=f"Hydrogen atom (Z={Z})",
        description=desc,
        quantities={
            "Z": "Scalar", "n": "Scalar",
            "E": "Energy", "a0": "Length",
            "r": "Length",
        },
        parameters={"Z": Z, "Rydberg_eV": RYDBERG_EV, "a0": a0} | energy_levels,
        timesteps=timesteps,
        known_invariant="E * n^2",
        lean_theorem="",
        is_conservative=True,
    )


def generate_hydrogen_atom_scenarios() -> list[dict[str, Any]]:
    """Generate hydrogen-like atom scenarios."""
    scenarios: list[dict[str, Any]] = []

    for Z in [1, 2, 3]:
        scenarios.append(simulate_hydrogen_atom(Z=Z, n_max=6, n_steps=30))

    return scenarios


# ═══════════════════════════════════════════════════════════════════════════
# 4. Probability current (1D free particle)
# ═══════════════════════════════════════════════════════════════════════════


def simulate_probability_current(
    k: float = 1e10,
    m: float = M_E,
    n_steps: int = 20,
    t_max: float = 1e-14,
) -> dict[str, Any]:
    """Simulate 1D free particle probability current.

    Plane wave: ψ(x,t) = A exp(i(kx - ωt))
    ω = ℏk²/(2m)

    Probability density: ρ = |ψ|² = |A|² (constant — U(1) symmetry)
    Probability current: j = (ℏk/m) |A|² (constant)

    Invariant: j / ρ = ℏk/m (constant velocity)
    Global U(1) phase: ψ → e^{iα}ψ leaves ρ, j invariant.

    Parameters
    ----------
    k : float
        Wave number (rad/m).
    m : float
        Particle mass (kg).
    n_steps : int
        Number of timesteps.
    t_max : float
        Total time simulated (s).
    """
    omega = HBAR * k**2 / (2 * m)
    v_phase = omega / k
    v_group = HBAR * k / m  # = j/ρ

    A = 1.0  # amplitude

    timesteps: list[dict[str, float]] = []
    for i in range(n_steps):
        t = t_max * i / (n_steps - 1) if n_steps > 1 else 0.0
        # Sample at origin x=0 for clarity
        x_val = 0.0
        phase = k * x_val - omega * t
        psi_real = A * math.cos(phase)
        psi_imag = A * math.sin(phase)
        rho = A**2
        j = (HBAR * k / m) * A**2
        timesteps.append({
            "t": round(t, 16),
            "psi_real": round(psi_real, 6),
            "psi_imag": round(psi_imag, 6),
            "rho": round(rho, 6),
            "j": round(j, 12),
        })

    desc = f"Free particle probability current: k={k:.1e}/m, m={m:.2e}kg"
    return _make_obs(
        obs_id=f"prob_current_k{k:.1e}_m{m:.1e}",
        name=f"Probability current (k={k:.1e}/m)",
        description=desc,
        quantities={
            "k": "1/Length", "m": "Mass",
            "hbar": "Energy*Time",
            "rho": "Scalar", "j": "1/Time",
            "t": "Time",
        },
        parameters={"k": k, "m": m, "hbar": HBAR,
                     "v_group": round(v_group, 6), "omega": round(omega, 4)},
        timesteps=timesteps,
        known_invariant="j",
        lean_theorem="",
        is_conservative=True,
    )


def generate_probability_current_scenarios() -> list[dict[str, Any]]:
    """Generate diverse free-particle probability current scenarios."""
    scenarios: list[dict[str, Any]] = []

    configs = [
        (1e10, M_E),          # Standard
        (2e10, M_E),          # Higher k
        (5e9, M_E),           # Lower k
        (1e10, 2 * M_E),      # Heavier
        (1e10, 0.5 * M_E),    # Lighter
        (3e10, M_E),          # Very high k
        (1e9, M_E),           # Very low k
        (1.5e10, 1.5 * M_E),  # Medium both
        (2e10, 2 * M_E),      # High k, heavy
        (5e9, 0.5 * M_E),     # Low k, light
    ]

    for k, m in configs:
        scenarios.append(simulate_probability_current(k=k, m=m, n_steps=30))

    return scenarios


# ═══════════════════════════════════════════════════════════════════════════
# 5. Expectation values in superposition states
# ═══════════════════════════════════════════════════════════════════════════


def simulate_expectation_values(
    omega: float = 1e15,
    m: float = M_E,
    c0: float = 0.8,
    c1: float = 0.6,
    n_steps: int = 30,
) -> dict[str, Any]:
    """Simulate expectation values for a superposition of harmonic oscillator states.

    State: |ψ⟩ = c0|0⟩ + c1|1⟩ (superposition of ground + first excited)
    Normalization: |c0|² + |c1|² = 1

    ⟨x⟩(t) oscillates at frequency ω (coherent oscillation)
    ⟨p⟩(t) oscillates at frequency ω (90° out of phase)

    ⟨E⟩ = |c0|² E₀ + |c1|² E₁ = constant (time-independent)

    Invariant: ⟨E⟩ is constant despite oscillating ⟨x⟩ and ⟨p⟩.
    This demonstrates energy quantization from U(1) time translation symmetry.

    Parameters
    ----------
    omega : float
        Oscillator frequency (rad/s).
    m : float
        Particle mass (kg).
    c0, c1 : float
        Superposition coefficients (auto-normalized).
    n_steps : int
        Number of timesteps.
    """
    # Normalize
    norm = math.sqrt(c0**2 + c1**2)
    c0 /= norm
    c1 /= norm

    a0 = math.sqrt(HBAR / (m * omega))
    E0 = 0.5 * HBAR * omega
    E1 = 1.5 * HBAR * omega

    # Matrix elements: ⟨0|x|1⟩ = ⟨1|x|0⟩ = a0/√2
    x01 = a0 / math.sqrt(2.0)
    p01 = -HBAR / (a0 * math.sqrt(2.0))  # ⟨0|p|1⟩ = -⟨1|p|0⟩

    period = 2 * math.pi / omega
    t_max = 2 * period

    timesteps: list[dict[str, float]] = []
    for i in range(n_steps):
        t = t_max * i / (n_steps - 1) if n_steps > 1 else 0.0
        # ⟨x⟩(t) = 2 Re(c0* c1 ⟨0|x|1⟩ e^{-iωt})
        exp_x = c0 * c1 * x01 * math.cos(omega * t)
        # ⟨p⟩(t) = 2 Re(c0* c1 ⟨0|p|1⟩ e^{-iωt})
        exp_p = c0 * c1 * p01 * math.sin(omega * t)

        E_expect = c0**2 * E0 + c1**2 * E1
        timesteps.append({
            "t": round(t, 16),
            "x_exp": round(2 * exp_x, 12),
            "p_exp": round(2 * exp_p, 20),
            "E": round(E_expect / EV_TO_J, 6),
        })

    desc = f"Superposition: c0={c0:.3f}|0⟩ + c1={c1:.3f}|1⟩, ω={omega:.1e}"
    return _make_obs(
        obs_id=f"expectation_omega{omega:.1e}_c0{c0}_c1{c1}",
        name=f"Superposition expectations (ω={omega:.1e})",
        description=desc,
        quantities={
            "omega": "1/Time", "m": "Mass",
            "hbar": "Energy*Time",
            "x": "Length", "p": "Momentum",
            "E": "Energy", "t": "Time",
        },
        parameters={"omega": omega, "m": m, "hbar": HBAR, "c0": round(c0, 4), "c1": round(c1, 4),
                     "E_exp_eV": round((c0**2 * E0 + c1**2 * E1) / EV_TO_J, 6)},
        timesteps=timesteps,
        known_invariant="E",
        lean_theorem="",
        is_conservative=True,
    )


def generate_expectation_scenarios() -> list[dict[str, Any]]:
    """Generate diverse superposition scenarios."""
    scenarios: list[dict[str, Any]] = []

    configs = [
        (1e15, M_E, 0.8, 0.6),       # Standard
        (1e15, M_E, 1.0, 0.0),       # Pure ground (no oscillation)
        (1e15, M_E, 0.0, 1.0),       # Pure excited
        (2e15, M_E, 0.6, 0.8),       # Higher freq, complementary weights
        (1e15, 2 * M_E, 0.7, 0.7141), # Heavier
        (5e14, M_E, 0.5, 0.866),     # Lower freq, equal probability
        (3e15, M_E, 0.3, 0.954),     # High freq, mostly excited
        (1e15, 0.5 * M_E, 0.9, 0.436), # Light, mostly ground
    ]

    for omega, m, c0, c1 in configs:
        scenarios.append(simulate_expectation_values(
            omega=omega, m=m, c0=c0, c1=c1, n_steps=40,
        ))

    return scenarios


# ═══════════════════════════════════════════════════════════════════════════
# 6. Free wave-packet evolution
# ═══════════════════════════════════════════════════════════════════════════


def simulate_wave_packet(
    k0: float = 1e10,
    sigma_x: float = 1e-9,
    m: float = M_E,
    n_steps: int = 30,
    t_max: float = 1e-13,
) -> dict[str, Any]:
    """Simulate a Gaussian wave-packet spreading in free space.

    Initial: ψ(x,0) = (2πσ²)^{-1/4} exp(-x²/(4σ²)) exp(ik₀x)

    Width evolution: σ(t) = σ₀ sqrt(1 + (ℏt/(2mσ₀²))²)

    Probability is conserved: ∫|ψ|²dx = 1 (U(1) symmetry)
    Uncertainty product: Δx·Δp ≥ ℏ/2

    Invariants:
      - Total probability = 1 (conserved by U(1) phase symmetry)
      - Center velocity = ℏk₀/m (constant)
      - Energy expectation = ℏ²k₀²/(2m) + ℏ²/(8mσ₀²) (constant)

    Parameters
    ----------
    k0 : float
        Central wave number (rad/m).
    sigma_x : float
        Initial width (m).
    m : float
        Particle mass (kg).
    n_steps : int
        Number of timesteps.
    t_max : float
        Total time simulated (s).
    """
    v_center = HBAR * k0 / m
    sigma_t_scale = HBAR / (2 * m * sigma_x**2)
    E_kin = HBAR**2 * k0**2 / (2 * m)
    E_spread = HBAR**2 / (8 * m * sigma_x**2)
    E_total = E_kin + E_spread

    timesteps: list[dict[str, float]] = []
    for i in range(n_steps):
        t = t_max * i / (n_steps - 1) if n_steps > 1 else 0.0
        sigma_t = sigma_x * math.sqrt(1.0 + (sigma_t_scale * t)**2)
        x_center = v_center * t
        prob_total = 1.0  # conserved by U(1)
        delta_x = sigma_t
        delta_p = HBAR / (2 * sigma_t)  # Heisenberg minimum

        timesteps.append({
            "t": round(t, 16),
            "sigma_t": round(sigma_t, 14),
            "x_center": round(x_center, 12),
            "prob": round(prob_total, 6),
            "delta_x": round(delta_x, 14),
            "delta_p": round(delta_p, 20),
        })

    desc = f"Wave-packet: k0={k0:.1e}/m, σ₀={sigma_x:.1e}m, m={m:.2e}kg"
    return _make_obs(
        obs_id=f"wave_packet_k{k0:.1e}_sig{sigma_x:.1e}_m{m:.1e}",
        name=f"Wave-packet (k₀={k0:.1e}/m, σ₀={sigma_x:.1e}m)",
        description=desc,
        quantities={
            "k0": "1/Length", "sigma": "Length",
            "m": "Mass", "hbar": "Energy*Time",
            "prob": "Scalar", "E": "Energy",
            "t": "Time", "delta_x": "Length", "delta_p": "Momentum",
        },
        parameters={"k0": k0, "sigma_x": sigma_x, "m": m, "hbar": HBAR,
                     "E_total_eV": round(E_total / EV_TO_J, 6),
                     "v_center": round(v_center, 4)},
        timesteps=timesteps,
        known_invariant="prob",
        lean_theorem="",
        is_conservative=True,
    )


def generate_wave_packet_scenarios() -> list[dict[str, Any]]:
    """Generate diverse wave-packet scenarios."""
    scenarios: list[dict[str, Any]] = []

    configs = [
        (1e10, 1e-9, M_E),           # Standard
        (2e10, 1e-9, M_E),           # Higher k0
        (1e10, 2e-9, M_E),           # Wider initial packet
        (1e10, 5e-10, M_E),          # Narrower
        (1e10, 1e-9, 2 * M_E),       # Heavier
        (5e9, 1e-9, M_E),            # Lower k0
        (3e10, 5e-10, M_E),          # High k, narrow
        (1e10, 3e-9, 0.5 * M_E),     # Wide, light
    ]

    for k0, sigma_x, m in configs:
        scenarios.append(simulate_wave_packet(k0=k0, sigma_x=sigma_x, m=m, n_steps=40))

    return scenarios


# ═══════════════════════════════════════════════════════════════════════════
# Aggregation
# ═══════════════════════════════════════════════════════════════════════════


def generate_all_quantum() -> list[dict[str, Any]]:
    """Generate all quantum mechanics observation scenarios."""
    scenarios: list[dict[str, Any]] = []
    scenarios.extend(generate_particle_in_box_scenarios())
    scenarios.extend(generate_harmonic_oscillator_scenarios())
    scenarios.extend(generate_hydrogen_atom_scenarios())
    scenarios.extend(generate_probability_current_scenarios())
    scenarios.extend(generate_expectation_scenarios())
    scenarios.extend(generate_wave_packet_scenarios())
    return scenarios
