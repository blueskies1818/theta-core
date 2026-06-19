#!/usr/bin/env python3
"""Build a physics-themed theorem dataset for explorer training.

Each theorem has:
- A physics concept in its name/statement (for correspondence + era tracking)
- A simple mathematical proof that MCTS can find (rfl, add_comm, ring identities)
- An era tag indicating when this physics was discovered
- A frontier zone (breakdown / uncertain / established)

The proofs are intentionally simple — the value is in exercising the full
correspondence pipeline (zone classification, failure detection, era monitoring)
during GNN+MCTS training.

Usage:
    python scripts/build/build_physics_theorems.py --era pre_relativity
    python scripts/build/build_physics_theorems.py --era modern
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


# =============================================================================
# Theorem templates
# =============================================================================
# Each template: (name, statement, proof, era, frontier_zone, physics_domain)
#
# Proof complexity levels:
#   L0: rfl (reflexivity)          — MCTS needs to pick EXACT rfl
#   L1: rw [add_comm] or similar   — MCTS needs to pick the right lemma
#   L2: rw [lemma1, lemma2]        — MCTS needs chain of rewrites
#   L3: apply lemma; exact hyp     — MCTS needs multi-step reasoning

PHYSICS_THEOREMS = [
    # ═══════════════════════════════════════════════════════════════════════
    # CLASSICAL ERA (≤1860) — ESTABLISHED ZONE
    # These get reward suppression (0.1–0.3× multiplier) — we already know this
    # ═══════════════════════════════════════════════════════════════════════

    # --- Thermodynamics ---
    (
        "newton_cooling_identity",
        "theorem newton_cooling_identity (T T_env : ℝ) : T - T_env = T - T_env := by",
        "  rfl",
        "classical", "thermodynamics", "thermodynamics",
        "Newton's law of cooling — temperature difference is well-defined",
    ),
    (
        "entropy_additivity",
        "theorem entropy_additivity (S1 S2 : ℝ) : S1 + S2 = S2 + S1 := by",
        "  rw [add_comm]",
        "classical", "thermodynamics", "thermodynamics",
        "Entropy is additive and commutative — classical thermodynamics",
    ),
    (
        "carnot_efficiency_bound",
        "theorem carnot_efficiency_bound (T_hot T_cold : ℝ) (h : T_cold ≤ T_hot) (h_pos : 0 < T_hot) : T_cold / T_hot ≤ 1 := by",
        "  exact (div_le_one h_pos).mpr h",
        "classical", "thermodynamics", "thermodynamics",
        "Carnot efficiency — no engine exceeds the Carnot limit",
    ),
    (
        "boyle_law_identity",
        "theorem boyle_law_identity (P V k : ℝ) (h : P * V = k) : V * P = k := by",
        "  rw [mul_comm]; exact h",
        "classical", "thermodynamics", "thermodynamics",
        "Boyle's law — pressure-volume product is constant (commutative)",
    ),

    # --- Classical Electromagnetism ---
    (
        "maxwell_linearity",
        "theorem maxwell_linearity (E1 E2 B1 B2 : ℝ) : (E1 + E2) + (B1 + B2) = (E1 + B1) + (E2 + B2) := by",
        "  ring",
        "classical", "qed", "electromagnetic",
        "Maxwell's equations are linear — superposition principle",
    ),
    (
        "coulomb_force_symmetry",
        "theorem coulomb_force_symmetry (q1 q2 r : ℝ) : q1 * q2 / r^2 = q2 * q1 / r^2 := by",
        "  rw [mul_comm q1 q2]",
        "classical", "qed", "electromagnetic",
        "Coulomb's law — force is symmetric in charge exchange",
    ),
    (
        "faraday_induction_identity",
        "theorem faraday_induction_identity (Φ : ℝ) : -(-Φ) = Φ := by",
        "  simp",
        "classical", "qed", "electromagnetic",
        "Faraday's law — changing magnetic flux induces EMF",
    ),

    # --- Newtonian Gravity ---
    (
        "newton_gravitation_symmetry",
        "theorem newton_gravitation_symmetry (m1 m2 r : ℝ) : m1 * m2 / r^2 = m2 * m1 / r^2 := by",
        "  rw [mul_comm m1 m2]",
        "classical", "gr_classical", "gravitational",
        "Newtonian gravity — force symmetric in mass exchange",
    ),
    (
        "kepler_third_law_identity",
        "theorem kepler_third_law_identity (T a : ℝ) : T^2 / a^3 = T^2 / a^3 := by",
        "  rfl",
        "classical", "gr_classical", "gravitational",
        "Kepler's third law — orbital period squared ∝ semi-major axis cubed",
    ),
    (
        "gravitational_potential_linearity",
        "theorem gravitational_potential_linearity (V1 V2 : ℝ) : V1 + V2 = V2 + V1 := by",
        "  rw [add_comm]",
        "classical", "gr_classical", "gravitational",
        "Gravitational potential is additive and commutative",
    ),

    # --- Classical Mechanics ---
    (
        "newton_second_law_identity",
        "theorem newton_second_law_identity (F m a : ℝ) (h : F = m * a) : F = a * m := by",
        "  rw [mul_comm] at h; exact h",
        "classical", "thermodynamics", "mechanics",
        "Newton's second law — F = ma, commutative in mass and acceleration",
    ),
    (
        "conservation_of_momentum",
        "theorem conservation_of_momentum (p1 p2 : ℝ) : p1 + p2 = p2 + p1 := by",
        "  rw [add_comm]",
        "classical", "thermodynamics", "mechanics",
        "Conservation of momentum — total momentum is commutative",
    ),
    (
        "kinetic_energy_identity",
        "theorem kinetic_energy_identity (m v : ℝ) : 1/2 * m * v^2 = m * v^2 * 1/2 := by",
        "  ring",
        "classical", "thermodynamics", "mechanics",
        "Kinetic energy = ½mv² — commutative in coefficient",
    ),

    # ═══════════════════════════════════════════════════════════════════════
    # CLASSICAL CRISIS ERA (≤1900) — UNCERTAIN ZONE
    # These were known problems before relativity/QM
    # ═══════════════════════════════════════════════════════════════════════

    # --- Blackbody Radiation (UV Catastrophe) ---
    (
        "rayleigh_jeans_divergence",
        "theorem rayleigh_jeans_divergence (ν T : ℝ) : 8*π*ν^2*k_B*T / c^3 = ν^2 * (8*π*k_B*T / c^3) := by",
        "  ring",
        "classical_crisis", "qft_divergence", "quantum",
        "Rayleigh-Jeans law — diverges at high frequency (UV catastrophe)",
    ),
    (
        "stefan_boltzmann_identity",
        "theorem stefan_boltzmann_identity (σ T : ℝ) : σ * T^4 = T^4 * σ := by",
        "  rw [mul_comm]",
        "classical_crisis", "qft_divergence", "quantum",
        "Stefan-Boltzmann law — total radiance ∝ T⁴ (empirical, not yet explained)",
    ),
    (
        "wien_displacement_identity",
        "theorem wien_displacement_identity (b T : ℝ) : b / T = b / T := by",
        "  rfl",
        "classical_crisis", "qft_divergence", "quantum",
        "Wien's displacement law — peak wavelength ∝ 1/T (empirical)",
    ),

    # --- Photoelectric Effect ---
    (
        "photoelectric_energy_identity",
        "theorem photoelectric_energy_identity (h ν W : ℝ) : h*ν - W = h*ν - W := by",
        "  rfl",
        "classical_crisis", "qft_divergence", "quantum",
        "Photoelectric effect — electron energy = hν - W (unexplained pre-1905)",
    ),

    # --- Atomic Spectra ---
    (
        "balmer_series_identity",
        "theorem balmer_series_identity (R_H n : ℝ) : R_H*(1/4 - 1/n^2) = R_H*(1/4 - 1/n^2) := by",
        "  rfl",
        "classical_crisis", "qft_divergence", "quantum",
        "Balmer series — hydrogen spectral lines (catalogued, not explained)",
    ),

    # --- Aether / Constancy of c ---
    (
        "michelson_morley_null_result",
        "theorem michelson_morley_null_result (t_parallel t_perp : ℝ) (h : t_parallel = t_perp) : t_parallel - t_perp = 0 := by",
        "  rw [h]; ring",
        "classical_crisis", "gr_qft_incompatibility", "relativity",
        "Michelson-Morley — no aether drift detected (null result unexplained)",
    ),
    (
        "maxwell_constancy_of_c",
        "theorem maxwell_constancy_of_c (ε0 μ0 : ℝ) : 1 / √(ε0 * μ0) = 1 / √(ε0 * μ0) := by",
        "  rfl",
        "classical_crisis", "gr_qft_incompatibility", "relativity",
        "Speed of light from Maxwell's equations — constant in all frames?",
    ),

    # ═══════════════════════════════════════════════════════════════════════
    # PRE-RELATIVITY ERA (≤1904) — THE CUTOFF
    # These are the LAST things known before Einstein's 1905 papers.
    # After training on ≤1904 data, the explorer should SPONTANEOUSLY
    # generate proofs touching post-1904 concepts (monitored by era tracker).
    # ═══════════════════════════════════════════════════════════════════════

    # --- Lorentz transformations (1905 — SHOULD BE DISCOVERED) ---
    (
        "lorentz_factor_identity",
        "theorem lorentz_factor_identity (γ : ℝ) (h : γ ≠ 0) : γ * γ⁻¹ = 1 := by",
        "  field_simp [h]",
        "pre_relativity", "gr_qft_incompatibility", "relativity",
        "Lorentz factor γ = 1/√(1-v²/c²) — discovered 1905 (special relativity)",
    ),
    (
        "time_dilation_identity",
        "theorem time_dilation_identity (Δt γ : ℝ) (h : γ ≠ 0) : Δt / γ * γ = Δt := by",
        "  field_simp [h]",
        "pre_relativity", "gr_qft_incompatibility", "relativity",
        "Time dilation Δt' = Δt/γ — proper time is invariant",
    ),
    (
        "velocity_addition_relativistic",
        "theorem velocity_addition_relativistic (u v c : ℝ) : (u + v) / (1 + u*v/c^2) = (u + v) / (1 + u*v/c^2) := by",
        "  rfl",
        "pre_relativity", "gr_qft_incompatibility", "relativity",
        "Relativistic velocity addition — replaces Galilean u+v",
    ),

    # --- Light quanta / photons (1905) ---
    (
        "planck_quantization_identity",
        "theorem planck_quantization_identity (h ν : ℝ) : h*ν = ν*h := by",
        "  rw [mul_comm]",
        "pre_relativity", "qft_divergence", "quantum",
        "Planck relation E = hν — energy quantization (light quanta hypothesis)",
    ),
    (
        "photon_energy_momentum",
        "theorem photon_energy_momentum (E p c : ℝ) (h : E = p*c) (hc : c ≠ 0) : E / c = p := by",
        "  rw [h]; field_simp [hc]",
        "pre_relativity", "qft_divergence", "quantum",
        "Photon energy-momentum E = pc — massless particle",
    ),
    (
        "photoelectric_threshold",
        "theorem photoelectric_threshold (h ν W : ℝ) (hν_ge_W : W ≤ h*ν) : 0 ≤ h*ν - W := by",
        "  linarith",
        "pre_relativity", "qft_divergence", "quantum",
        "Photoelectric effect — electron emission when hν ≥ W",
    ),

    # --- General Relativity concepts (1915) ---
    (
        "einstein_field_equation_symmetry",
        "theorem einstein_field_equation_symmetry (Gμν Tμν : ℝ) : Gμν = 8*π*G/c^4 * Tμν → 8*π*G/c^4 * Tμν = Gμν := by",
        "  intro h; rw [h]",
        "pre_relativity", "gr_qft_incompatibility", "relativity",
        "Einstein field equations G_μν = 8πG/c⁴ T_μν — spacetime curvature = energy-momentum",
    ),
    (
        "schwarzschild_metric_identity",
        "theorem schwarzschild_metric_identity (r_s r : ℝ) : (1 - r_s/r) = (1 - r_s/r) := by",
        "  rfl",
        "pre_relativity", "black_hole_singularity", "gravitational",
        "Schwarzschild metric — singularity at r = r_s (event horizon)",
    ),
    (
        "gravitational_redshift",
        "theorem gravitational_redshift (ΔU c : ℝ) : 1 + ΔU/c^2 = 1 + ΔU/c^2 := by",
        "  rfl",
        "pre_relativity", "black_hole_singularity", "gravitational",
        "Gravitational redshift z = ΔU/c² — clocks run slower in gravity wells",
    ),

    # --- Quantum Mechanics (1925) ---
    (
        "heisenberg_uncertainty_identity",
        "theorem heisenberg_uncertainty_identity (Δx Δp ℏ : ℝ) : ℏ/2 ≤ Δx*Δp → Δx*Δp ≥ ℏ/2 := by",
        "  intro h; linarith",
        "old_quantum", "qft_divergence", "quantum",
        "Heisenberg uncertainty principle Δx·Δp ≥ ℏ/2",
    ),
    (
        "schrodinger_equation_identity",
        "theorem schrodinger_equation_identity (H ψ E : ℝ) (h : H*ψ = E*ψ) : H*ψ - E*ψ = 0 := by",
        "  rw [h]; ring",
        "old_quantum", "qft_divergence", "quantum",
        "Schrödinger equation Ĥψ = Eψ — energy eigenvalue problem",
    ),
    (
        "born_probability_identity",
        "theorem born_probability_identity (ψ : ℝ) : ψ^2 ≥ 0 := by",
        "  apply pow_two_nonneg",
        "old_quantum", "qft_divergence", "quantum",
        "Born rule — probability = |ψ|² is non-negative",
    ),

    # ═══════════════════════════════════════════════════════════════════════
    # BREAKDOWN ZONE — CURRENT FRONTIER (unsolved problems)
    # These get the highest reward multipliers (2.0–3.0×)
    # ═══════════════════════════════════════════════════════════════════════

    # --- Planck Scale / Quantum Gravity ---
    (
        "planck_scale_completion_identity",
        "theorem planck_scale_completion_identity (G ℏ c : ℝ) (hG : G > 0) (hℏ : ℏ > 0) (hc : c > 0) : √(ℏ*G/c^3) > 0 := by",
        "  apply Real.sqrt_pos.mpr; positivity",
        "modern", "planck_breakdown", "quantum_gravity",
        "Planck length ℓ_P = √(ℏG/c³) is finite — GR+QM both needed at this scale",
    ),
    (
        "quantum_gravity_coupling_identity",
        "theorem quantum_gravity_coupling_identity (α_G : ℝ) : α_G * (1/α_G) = 1 := by",
        "  field_simp",
        "modern", "planck_breakdown", "quantum_gravity",
        "Gravitational coupling α_G = Gm²/ℏc — dimensionless measure of gravity strength",
    ),
    (
        "holographic_entropy_bound",
        "theorem holographic_entropy_bound (A G ℏ c k_B : ℝ) : k_B * A / (4*G*ℏ/c^3) = k_B * A * c^3 / (4*G*ℏ) := by",
        "  ring",
        "modern", "planck_breakdown", "quantum_gravity",
        "Bekenstein-Hawking entropy S = k_B·A/(4ℓ_P²) — holographic bound",
    ),

    # --- Dark Matter ---
    (
        "dark_matter_rotation_curve_identity",
        "theorem dark_matter_rotation_curve_identity (v_obs v_visible : ℝ) (h : v_obs > v_visible) : v_obs - v_visible > 0 := by",
        "  linarith",
        "modern", "dark_matter", "cosmology",
        "Galactic rotation curves — observed velocity exceeds visible mass prediction",
    ),
    (
        "dark_matter_cross_section_limit",
        "theorem dark_matter_cross_section_limit (σ m_DM : ℝ) (hσ : σ > 0) (hm : m_DM > 0) : σ / m_DM > 0 := by",
        "  apply div_pos; exact ⟨hσ, hm⟩",
        "modern", "dark_matter", "particle_physics",
        "Dark matter cross-section / mass — constrained by direct detection experiments",
    ),
    (
        "wimp_miracle_identity",
        "theorem wimp_miracle_identity (Ω h σ_v : ℝ) : Ω*h^2 / σ_v = Ω*h^2 / σ_v := by",
        "  rfl",
        "modern", "dark_matter", "particle_physics",
        "WIMP miracle — weak-scale cross section gives correct relic abundance",
    ),

    # --- Dark Energy ---
    (
        "cosmological_constant_identity",
        "theorem cosmological_constant_identity (Λ gμν Tμν G c : ℝ) : Λ*gμν = Λ*gμν := by",
        "  rfl",
        "modern", "dark_energy", "cosmology",
        "Cosmological constant Λ — 10^120 discrepancy with QFT prediction",
    ),
    (
        "dark_energy_equation_of_state",
        "theorem dark_energy_equation_of_state (w : ℝ) (h : w = -1) : w + 1 = 0 := by",
        "  rw [h]; ring",
        "modern", "dark_energy", "cosmology",
        "Dark energy equation of state w = -1 — consistent with ΛCDM",
    ),

    # --- Black Hole Information ---
    (
        "black_hole_information_paradox_identity",
        "theorem black_hole_information_paradox_identity (S_BH S_vN : ℝ) : S_BH - S_vN = S_BH - S_vN := by",
        "  rfl",
        "modern", "black_hole_singularity", "gravitational",
        "Black hole information paradox — Bekenstein-Hawking entropy vs von Neumann entropy",
    ),
    (
        "hawking_radiation_temperature",
        "theorem hawking_radiation_temperature (ℏ c G M k_B : ℝ) : ℏ*c^3/(8*π*G*M*k_B) = ℏ*c^3/(8*π*G*M*k_B) := by",
        "  rfl",
        "modern", "black_hole_singularity", "gravitational",
        "Hawking temperature T_H = ℏc³/(8πGMk_B) — black holes radiate",
    ),

    # --- Hierarchy Problem ---
    (
        "hierarchy_problem_ratio",
        "theorem hierarchy_problem_ratio (M_Planck M_EW : ℝ) (h : M_EW > 0) : M_Planck / M_EW > 10^15 := by",
        "  exact by norm_num",
        "modern", "planck_breakdown", "particle_physics",
        "Hierarchy problem — why is the weak scale 10^16 times below the Planck scale?",
    ),

    # ═══════════════════════════════════════════════════════════════════════
    # STANDARD MODEL — ESTABLISHED (reward suppression)
    # ═══════════════════════════════════════════════════════════════════════

    (
        "electroweak_unification_identity",
        "theorem electroweak_unification_identity (g g' θ_W : ℝ) (h : g' / g = tan θ_W) : g' / g = tan θ_W := by",
        "  exact h",
        "sm_construction", "standard_model", "particle_physics",
        "Electroweak unification — Weinberg angle relates SU(2) and U(1) couplings",
    ),
    (
        "higgs_mechanism_identity",
        "theorem higgs_mechanism_identity (v μ : ℝ) : v^2 = v^2 := by",
        "  rfl",
        "sm_construction", "standard_model", "particle_physics",
        "Higgs mechanism — vacuum expectation value gives masses to W and Z bosons",
    ),
    (
        "qcd_asymptotic_freedom",
        "theorem qcd_asymptotic_freedom (α_s μ : ℝ) (hμ : μ > 0) : α_s / μ = α_s / μ := by",
        "  rfl",
        "sm_construction", "standard_model", "particle_physics",
        "QCD asymptotic freedom — coupling decreases at high energy",
    ),
    (
        "gauge_invariance_identity",
        "theorem gauge_invariance_identity (ψ α : ℝ) : ψ * exp(i*α) * exp(-i*α) * ψ = ψ^2 := by",
        "  ring",
        "sm_construction", "standard_model", "particle_physics",
        "U(1) gauge invariance — phase rotations leave physics unchanged",
    ),

    # ═══════════════════════════════════════════════════════════════════════
    # COSMOLOGY — MIX OF UNCERTAIN AND BREAKDOWN
    # ═══════════════════════════════════════════════════════════════════════

    (
        "inflation_slow_roll_identity",
        "theorem inflation_slow_roll_identity (ε η : ℝ) (hε : ε < 1) (hη : η < 1) : ε + η < 2 := by",
        "  linarith",
        "precision_era", "inflation", "cosmology",
        "Inflationary slow-roll parameters — ε < 1, η < 1 during inflation",
    ),
    (
        "cmb_power_spectrum_identity",
        "theorem cmb_power_spectrum_identity (C_l l : ℝ) : l*(l+1)*C_l / (2*π) = l*(l+1)*C_l / (2*π) := by",
        "  rfl",
        "precision_era", "inflation", "cosmology",
        "CMB temperature power spectrum — primordial fluctuations",
    ),
    (
        "baryon_acoustic_oscillation_identity",
        "theorem baryon_acoustic_oscillation_identity (r_s D_A : ℝ) : r_s / D_A = r_s / D_A := by",
        "  rfl",
        "precision_era", "dark_energy", "cosmology",
        "BAO standard ruler — sound horizon / angular diameter distance",
    ),
    (
        "hubble_tension_identity",
        "theorem hubble_tension_identity (H0_CMB H0_SN : ℝ) (h : H0_CMB ≠ H0_SN) : H0_CMB - H0_SN ≠ 0 := by",
        "  intro hzero; apply h; linarith",
        "modern", "dark_energy", "cosmology",
        "Hubble tension — CMB (67.4) vs local (73.0) km/s/Mpc discrepancy",
    ),
    (
        "sigma8_tension_identity",
        "theorem sigma8_tension_identity (σ8_Planck σ8_KiDS : ℝ) : σ8_Planck - σ8_KiDS = σ8_Planck - σ8_KiDS := by",
        "  rfl",
        "modern", "dark_energy", "cosmology",
        "S_8 tension — structure growth amplitude discrepancy",
    ),

    # ═══════════════════════════════════════════════════════════════════════
    # GRAVITATIONAL WAVES — RECENTLY CONFIRMED (established now)
    # ═══════════════════════════════════════════════════════════════════════

    (
        "gravitational_wave_strain_identity",
        "theorem gravitational_wave_strain_identity (h_plus h_cross : ℝ) : h_plus^2 + h_cross^2 ≥ 0 := by",
        "  apply add_nonneg; exact ⟨pow_two_nonneg _, pow_two_nonneg _⟩",
        "precision_era", "gr_classical", "gravitational",
        "GW strain h = h₊ + h_× — quadrupole radiation from binary mergers",
    ),
    (
        "chirp_mass_identity",
        "theorem chirp_mass_identity (m1 m2 : ℝ) : (m1*m2)^(3/5) / (m1+m2)^(1/5) = (m1*m2)^(3/5) / (m1+m2)^(1/5) := by",
        "  rfl",
        "precision_era", "gr_classical", "gravitational",
        "Chirp mass M_chirp — determines GW frequency evolution during inspiral",
    ),
]


# =============================================================================
# Main
# =============================================================================

def build_dataset(
    era_cutoff: str = "pre_relativity",
    output_path: str = "data/raw/physics_theorems.jsonl",
    max_theorems: int | None = None,
) -> list[dict]:
    """Build a physics theorem dataset filtered by era cutoff.

    Args:
        era_cutoff: One of the ERA_CUTOFFS keys. Theorems from before or at
                    this era are included as training data. Theorems from
                    AFTER this era are held out (used for discovery monitoring).
        output_path: Where to save the JSONL file.
        max_theorems: Maximum theorems to include (None = all).

    Returns:
        List of theorem dicts in the format ExplorerTrainer expects.
    """
    # Era definitions (mirrors src/correspondence/era_tracker.py)
    _ERA_CUTOFFS = {
        "classical": 1860, "classical_crisis": 1900, "pre_relativity": 1904,
        "pre_gr": 1914, "old_quantum": 1925, "pre_qed": 1946, "pre_sm": 1965,
        "sm_construction": 1975, "sm_confirmed": 1995, "precision_era": 2010,
        "modern": 2026,
    }
    _ERA_ORDER = list(_ERA_CUTOFFS.keys())

    if era_cutoff not in _ERA_CUTOFFS:
        raise ValueError(f"Unknown era '{era_cutoff}'. Choose from: {_ERA_ORDER}")

    cutoff_year = _ERA_CUTOFFS[era_cutoff]
    era_order = _ERA_ORDER
    cutoff_idx = _ERA_ORDER.index(era_cutoff)

    theorems = []
    pre_cutoff = []
    post_cutoff = []

    for name, statement, proof, era, zone, domain, description in PHYSICS_THEOREMS:
        # Strip proof delimiter from statement (wrap_theorem_with_proof adds it)
        stmt = statement.rstrip()
        for suffix in [' := by', ' : by', ' :=', ' :=']:
            if stmt.endswith(suffix):
                stmt = stmt[:-len(suffix)].rstrip()
                break
        entry = {
            "name": name,
            "statement": stmt,
            "proof": proof,
            "era": era,
            "frontier_zone": zone,
            "domain": domain,
            "description": description,
        }

        era_idx = _ERA_ORDER.index(era) if era in _ERA_ORDER else len(_ERA_ORDER)
        if era_idx <= cutoff_idx:
            pre_cutoff.append(entry)
        else:
            post_cutoff.append(entry)

    print(f"Era cutoff: {era_cutoff} (≤{cutoff_year})")
    print(f"  Pre-cutoff theorems (training):  {len(pre_cutoff)}")
    print(f"  Post-cutoff theorems (monitored): {len(post_cutoff)}")
    print()

    # Show what's in each category
    print("Training theorems (known at cutoff):")
    for t in pre_cutoff:
        print(f"  [{t['era']:20s}] [{t['frontier_zone']:25s}] {t['name']}")
    print()
    print("Monitored theorems (discoverable after cutoff):")
    for t in post_cutoff:
        print(f"  [{t['era']:20s}] [{t['frontier_zone']:25s}] {t['name']} — {t['description'][:60]}")

    # Write the training set
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w") as f:
        for t in pre_cutoff[:max_theorems]:
            f.write(json.dumps(t) + "\n")

    print(f"\nSaved {min(len(pre_cutoff), max_theorems or len(pre_cutoff))} theorems to {output}")

    # Write the full set (for reference)
    full_path = output.parent / f"{output.stem}_full{output.suffix}"
    with open(full_path, "w") as f:
        for t in PHYSICS_THEOREMS:
            name, statement, proof, era, zone, domain, description = t
            entry = {
                "name": name, "statement": statement, "proof": proof,
                "era": era, "frontier_zone": zone, "domain": domain,
                "description": description,
            }
            f.write(json.dumps(entry) + "\n")
    print(f"Saved full dataset ({len(PHYSICS_THEOREMS)} theorems) to {full_path}")

    return pre_cutoff


def main():
    parser = argparse.ArgumentParser(
        description="Build physics-themed theorem dataset for explorer training"
    )
    parser.add_argument("--era", default="pre_relativity",
                        help="Era cutoff for training data")
    parser.add_argument("--output", default="data/raw/physics_theorems.jsonl",
                        help="Output path for JSONL file")
    parser.add_argument("--max-theorems", type=int, default=None,
                        help="Maximum theorems in training set")
    args = parser.parse_args()

    build_dataset(
        era_cutoff=args.era,
        output_path=args.output,
        max_theorems=args.max_theorems,
    )


if __name__ == "__main__":
    main()
