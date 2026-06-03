#!/usr/bin/env python3
"""Temporal Gating Evaluation: Can the explorer rediscover physics from history?

Phase 2 closure evaluation. The core hypothesis:

    If we restrict the model to only the data available at a given point in
    history, will it prioritize the same problems that led to the next
    generation of physics theories?

Tests:
  1. Data inventory by era — what constants/particles/spectra were known?
  2. Open problem detection — what anomalies were visible but unexplained?
  3. Frontier classification — does the correspondence modifier correctly
     identify era-appropriate frontier zones?
  4. Prioritization analysis — given era-N data, does the reward landscape
     pull the explorer toward era-N+1 discoveries?

Usage:
    python scripts/eval_temporal_gating.py [--era YEAR] [--compare ERAS] [--full-timeline]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.physical import (
    get_data_up_to_year,
    get_all_data_up_to_year,
    ERA_CUTOFFS,
)
from src.correspondence.frontier import (
    ZoneType,
    build_standard_frontier_map,
)
from src.correspondence.failure_points import (
    FailureSeverity,
    build_standard_failure_coordinates,
)
from src.correspondence.reward_integration import CorrespondenceRewardModifier


# ═══════════════════════════════════════════════════════════════════════════════
# Historical Physics Knowledge Graph
# ═══════════════════════════════════════════════════════════════════════════════

# For each era, what was KNOWN (established theory + data) and what was
# UNKNOWN (the open problems that led to the next revolution)?

ERA_KNOWLEDGE = {
    "classical (≤1860)": {
        "known_theory": [
            "Newtonian mechanics (F=ma, universal gravitation)",
            "Maxwell's equations (unified E&M, predicted EM waves at c)",
            "Thermodynamics (Carnot, Clausius — 1st and 2nd laws)",
            "Wave optics (Young, Fresnel — interference, diffraction)",
            "Chemical elements (Mendeleev table taking shape)",
        ],
        "known_data": [
            "Speed of light (Fizeau 1849, Foucault 1862)",
            "Gravitational constant G (Cavendish 1798)",
            "Spectral lines (Fraunhofer 1814 — solar absorption lines)",
            "Gas laws (Boyle, Charles, Gay-Lussac, Avogadro)",
        ],
        "open_problems": [
            "Mercury perihelion precession (observed but unexplained by Newton)",
            "Blackbody radiation spectrum (no theoretical model fits)",
            "Nature of light (wave in WHAT medium? — aether problem)",
            "Atomic hypothesis (chemical evidence only — no direct proof)",
            "Cathode rays (observed but not understood as electrons)",
        ],
        "next_discovery": "Special relativity (1905) + photon (1905) + QM (1920s)",
        "should_prioritize": [
            "constancy of c across frames → special relativity",
            "blackbody spectrum → Planck's law → quantum hypothesis",
            "cathode ray properties → electron discovery → atomic models",
        ],
    },
    "classical crisis (≤1900)": {
        "known_theory": [
            "Newtonian mechanics + Maxwell's equations (fully developed)",
            "Statistical mechanics (Boltzmann, Gibbs)",
            "Lorentz electron theory (but no relativity yet)",
        ],
        "known_data": [
            "Electron (Thomson 1897 — m/e ratio measured)",
            "Zeeman effect (1896 — spectral line splitting in B field)",
            "Michelson-Morley null result (1887 — no aether drift)",
            "Rydberg formula for hydrogen (1888 — empirical fit only)",
            "X-rays (Röntgen 1895)",
            "Radioactivity (Becquerel 1896)",
        ],
        "open_problems": [
            "UV catastrophe — Rayleigh-Jeans diverges at short λ",
            "Michelson-Morley — no aether detected",
            "Photoelectric effect — unexplained by wave theory",
            "Radioactivity — unknown energy source",
            "Atomic spectra — unexplained discrete lines",
        ],
        "next_discovery": "Quantum hypothesis (Planck 1900) + SR (Einstein 1905)",
        "should_prioritize": [
            "blackbody → energy quantization → Planck's constant",
            "null aether result → Lorentz transformations → SR",
            "photoelectric effect → light quanta → photon",
        ],
    },
    "pre-relativity (≤1904)": {
        "known_theory": [
            "Planck's blackbody law (1900 — quantum hypothesis, no full theory)",
            "Lorentz transformations (but treated as mathematical trick)",
            "Maxwell's equations (fully established)",
            "Statistical mechanics (Boltzmann, Gibbs — mature field)",
        ],
        "known_data": [
            "Electron charge (Millikan started oil drop experiments)",
            "Alpha/beta/gamma radiation distinguished (Rutherford)",
            "Balmer series for hydrogen (visible lines fitted)",
            "Stefan-Boltzmann constant (blackbody total power)",
        ],
        "open_problems": [
            "What IS Planck's quantum? (mathematical trick or physical reality?)",
            "Why is c constant in all frames? (relativity of simultaneity)",
            "What is the structure of the atom? (plum pudding vs nuclear)",
            "Why do elements have discrete spectra?",
        ],
        "next_discovery": "Special relativity (1905) + photon (1905) + Bohr atom (1913)",
        "should_prioritize": [
            "constancy of c → relativity postulates → SR",
            "light quanta → photoelectric explanation → photon",
            "atomic structure → Rutherford/Bohr model",
        ],
    },
    "pre-GR (≤1914)": {
        "known_theory": [
            "Special relativity (1905 — fully accepted by 1914)",
            "Old quantum theory (Bohr atom 1913 — partial success)",
            "Maxwell's equations (now relativistic — covariant formulation)",
        ],
        "known_data": [
            "Geiger-Marsden scattering (1911 — nuclear atom confirmed)",
            "X-ray diffraction (von Laue 1912 — crystal structure)",
            "Isotopes (Soddy 1913 — same element, different mass)",
            "Milky Way structure (Shapley started mapping)",
        ],
        "open_problems": [
            "Gravity is NOT Lorentz invariant — Newton's gravity violates SR",
            "Mercury's perihelion still unexplained (43 arcsec/century excess)",
            "Bohr model fails for helium and multi-electron atoms",
            "No quantum theory — just ad hoc quantization rules",
        ],
        "next_discovery": "General relativity (1915) + full QM (1925-1927)",
        "should_prioritize": [
            "equivalence principle → curved spacetime → GR",
            "perihelion precession → Einstein field equations",
            "wave-particle duality → full quantum mechanics",
        ],
    },
    "modern (≤2026)": {
        "known_theory": [
            "Standard Model of particle physics (SU(3)×SU(2)×U(1))",
            "General Relativity (ΛCDM cosmology)",
            "Quantum Field Theory (renormalization, effective field theories)",
            "Neutrino oscillations (PMNS matrix, massive neutrinos)",
            "Higgs mechanism (electroweak symmetry breaking confirmed 2012)",
        ],
        "known_data": [
            "LHC Run 2 — SM confirmed to ~1% at EW scale, no BSM particles",
            "Planck 2018 — ΛCDM fits CMB exquisitely (6-parameter model)",
            "LIGO/Virgo — ~90 GW events, BH and NS mergers",
            "Dark matter — 27% of universe, particle identity unknown",
            "Dark energy — 68% of universe, nature unknown",
            "Neutrino masses — Δm² measured, absolute scale unknown",
        ],
        "open_problems": [
            "H₀ tension — 5σ disagreement between CMB and local measurements",
            "Dark matter identity — no WIMP detection, no axion signal",
            "Cosmological constant problem — 10¹²⁰ discrepancy with QFT",
            "Quantum gravity — GR and QFT incompatible at Planck scale",
            "Baryon asymmetry — why is there matter at all?",
            "Hierarchy problem — why is Higgs mass so light?",
            "Strong CP problem — why is θ_QCD < 10⁻¹⁰?",
            "Black hole information paradox",
        ],
        "next_discovery": "Quantum gravity? Dark matter particle? New force?",
        "should_prioritize": [
            "planck scale UV completion → quantum gravity",
            "dark matter particle detection → new physics beyond SM",
            "H₀ tension resolution → new cosmology or systematics",
            "GR-QFT interface → effective field theory of gravity",
        ],
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# Era-appropriate theorem statements that the explorer would encounter
# ═══════════════════════════════════════════════════════════════════════════════

ERA_THEOREMS = {
    "classical (≤1860)": [
        ("newton_gravitation", "theorem universal_gravitation_inverse_square : "
         "F = G * M * m / r^2 for all point masses in Newtonian mechanics"),
        ("maxwell_equations", "theorem maxwell_faraday_law : "
         "a changing magnetic field induces an electric field, ∇ × E = -∂B/∂t"),
        ("carnot_efficiency", "theorem carnot_cycle_maximum_efficiency : "
         "η = 1 - T_c/T_h for any reversible heat engine"),
        ("wave_theory_light", "theorem huygens_fresnel_principle : "
         "light propagates as a wave with interference and diffraction"),
        ("mercury_anomaly", "theorem mercury_perihelion_precession : "
         "observed precession 5600 arcsec/century, Newton predicts 5557 — "
         "43 arcsec/century excess unexplained"),
        ("blackbody_problem", "theorem blackbody_radiation_spectrum : "
         "the Rayleigh-Jeans law diverges at short wavelengths, "
         "the ultraviolet catastrophe of classical thermodynamics"),
        ("aether_problem", "theorem electromagnetic_wave_propagation : "
         "Maxwell's equations predict EM waves at speed c, but relative to "
         "what frame? The luminiferous aether is undetected"),
    ],
    "classical crisis (≤1900)": [
        ("electron_discovery", "theorem cathode_ray_particle_nature : "
         "cathode rays are charged particles with m/e ~ 1/2000 of hydrogen — "
         "a subatomic particle, the electron"),
        ("michelson_morley", "theorem michelson_morley_experiment : "
         "no detectable aether drift to within 1/40th of expected fringe shift, "
         "despite Earth's orbital velocity of 30 km/s"),
        ("rydberg_formula", "theorem rydberg_hydrogen_spectrum : "
         "1/λ = R_H (1/n₁² - 1/n₂²) fits all hydrogen lines empirically "
         "but has no theoretical justification"),
        ("radioactivity", "theorem becquerel_radioactivity : "
         "uranium salts emit penetrating radiation spontaneously — "
         "energy source unknown, violates energy conservation?"),
        ("photoelectric", "theorem photoelectric_effect_anomaly : "
         "electron emission depends on light frequency not intensity, "
         "contradicting classical wave theory"),
    ],
    "pre-relativity (≤1904)": [
        ("planck_quantum", "theorem planck_blackbody_distribution : "
         "E = hν resolves the UV catastrophe but the physical meaning "
         "of energy quantization is unclear"),
        ("lorentz_transform", "theorem lorentz_fitzgerald_contraction : "
         "length contraction L = L₀√(1-v²/c²) explains the null aether result "
         "but is treated as an ad hoc mathematical fix"),
        ("atomic_structure", "theorem thomson_plum_pudding_atom : "
         "electrons embedded in positive sphere — but cannot explain "
         "Rutherford scattering or spectral lines"),
    ],
    "pre-GR (≤1914)": [
        ("special_relativity", "theorem einstein_special_relativity : "
         "laws of physics are the same in all inertial frames and the speed "
         "of light in vacuum is constant — Maxwell's equations are Lorentz invariant"),
        ("bohr_atom", "theorem bohr_hydrogen_atom : "
         "electron in quantized circular orbits explains Balmer series but "
         "fails for helium and cannot explain chemical bonding"),
        ("equivalence_principle", "theorem equivalence_principle_gravity : "
         "gravitational mass equals inertial mass to high precision — "
         "the foundation for a new theory of gravity extending SR"),
        ("perihelion_problem", "theorem mercury_anomalous_precession : "
         "43 arcsec/century excess cannot be explained by Newtonian gravity "
         "with known solar system masses — requires new gravitational theory"),
    ],
    "modern (≤2026)": [
        ("planck_breakdown", "theorem planck_scale_breakdown : "
         "at the Planck scale M_P ~ 10^19 GeV, general relativity becomes "
         "non-renormalizable — quantum field theory and GR are incompatible"),
        ("dark_matter", "theorem dark_matter_identity : "
         "27% of the universe's energy density is non-baryonic dark matter "
         "with no detected particle — requires physics beyond the Standard Model"),
        ("h0_tension", "theorem hubble_constant_tension : "
         "CMB-inferred H₀ = 67.4 ± 0.5 and local H₀ = 73.0 ± 1.0 differ at "
         "5σ — either systematics or new physics in the early universe"),
        ("cosmological_constant", "theorem cosmological_constant_problem : "
         "Λ_obs / Λ_QFT ~ 10^-120 — the worst prediction in the history "
         "of physics requires a resolution mechanism"),
        ("hierarchy", "theorem gauge_hierarchy_problem : "
         "the Higgs boson mass at 125 GeV is 16 orders of magnitude below "
         "the Planck scale — why is the electroweak scale so light?"),
        ("black_hole_info", "theorem black_hole_information_paradox : "
         "Hawking radiation implies pure states evolve to mixed states, "
         "violating unitarity of quantum mechanics"),
    ],
}


# ═══════════════════════════════════════════════════════════════════════════════
# Test 1: Data Inventory by Era
# ═══════════════════════════════════════════════════════════════════════════════

def print_data_inventory(era_year: int, era_name: str):
    """Print what experimental data was available in a given era."""
    data = get_data_up_to_year(era_year)

    sections = [
        ("Fundamental constants", "constants"),
        ("Particles", "particles"),
        ("Spectral lines", "spectral_lines"),
        ("Cosmological parameters", "cosmology"),
        ("Nuclear properties", "nuclear"),
        ("Neutrino parameters", "neutrino"),
        ("Flavor physics", "flavor"),
        ("Hadrons (extended)", "more_hadrons"),
        ("Thermodynamic data", "thermodynamic"),
        ("GR tests", "gr_tests"),
        ("Equivalence principle tests", "equivalence"),
        ("Direct detection limits", "direct_limits"),
        ("Elements", "elements"),
        ("ANOMALIES", "anomalies"),
    ]

    total = 0
    for label, key in sections:
        items = data.get(key, [])
        total += len(items)
        if items:
            if key == "particles":
                names = [getattr(p, "name", str(p)) for p in items[:8]]
                extra = f" — {', '.join(names)}"
                if len(items) > 8:
                    extra += f", ... (+{len(items)-8} more)"
            elif key == "constants":
                syms = [getattr(c, "symbol", str(c)) for c in items[:10]]
                extra = f" — {', '.join(syms)}"
                if len(items) > 10:
                    extra += f", ... (+{len(items)-10} more)"
            elif key == "anomalies":
                names = [getattr(a, "name", str(a)) for a in items]
                extra = f" — {', '.join(names)}"
            else:
                extra = ""
            print(f"  {label:<30} {len(items):>4}{extra}")

    print(f"  {'─' * 30} {'─' * 4}")
    print(f"  {'TOTAL':<30} {total:>4}")

    return total


# ═══════════════════════════════════════════════════════════════════════════════
# Test 2: Open Problem Detection
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_open_problems(era_year: int, era_name: str) -> list[dict]:
    """Identify open physics problems visible given the data in an era.

    This is what the explorer should prioritize — the anomalies and gaps
    that the current theory cannot explain.
    """
    data = get_data_up_to_year(era_year)
    problems = []

    available_constants = {c.symbol: c for c in data["constants"]}
    available_particles = {p.symbol: p for p in data["particles"]}

    # Pattern 1: Known constants that imply missing theory
    if "c" in available_constants and era_year < 1905:
        problems.append({
            "problem": "Constancy of speed of light",
            "evidence": f"c = {available_constants['c'].value:.2e} m/s (measured {available_constants['c'].discovery_year})",
            "gap": "No relativity — c should depend on observer frame in Newtonian physics",
            "resolution": "Special relativity (Einstein 1905)",
            "priority": 10,
        })

    if "G" in available_constants and era_year < 1915:
        problems.append({
            "problem": "Newton's gravity is not Lorentz invariant",
            "evidence": f"G = {available_constants['G'].value:.2e} (measured {available_constants['G'].discovery_year})",
            "gap": "Gravity propagates instantaneously in Newtonian theory — violates SR",
            "resolution": "General relativity (Einstein 1915)",
            "priority": 9,
        })

    # Pattern 2: Missing particles that theory needs
    if "e⁻" in available_particles and "γ" not in available_particles:
        p_elec = available_particles["e⁻"]
        problems.append({
            "problem": "No photon concept",
            "evidence": f"Electron discovered {p_elec.discovery_year}",
            "gap": "EM radiation treated as continuous wave — no explanation for photoelectric effect",
            "resolution": "Photon / light quanta (Einstein 1905)",
            "priority": 9,
        })

    if "γ" in available_particles and era_year < 1925:
        problems.append({
            "problem": "Wave-particle duality unresolved",
            "evidence": "Photon known, electron known",
            "gap": "Both show wave and particle behavior — no unified framework",
            "resolution": "Quantum mechanics (Schrödinger/Heisenberg 1925-1927)",
            "priority": 8,
        })

    # Pattern 3: Spectral lines without quantum theory
    if data["spectral_lines"] and era_year < 1913:
        n_lines = len(data["spectral_lines"])
        problems.append({
            "problem": f"Atomic spectra unexplained ({n_lines} lines catalogued)",
            "evidence": "Balmer series empirically fitted — no physical model",
            "gap": "Classical physics predicts continuous spectrum, not discrete lines",
            "resolution": "Bohr atom (1913) → quantum mechanics",
            "priority": 8,
        })

    # Pattern 4: Thermal radiation without quantum
    if era_year < 1900:
        if "σ" in available_constants:  # Stefan-Boltzmann
            sb = available_constants["σ"]
            problems.append({
                "problem": "Blackbody spectrum (UV catastrophe)",
                "evidence": f"Stefan-Boltzmann constant σ measured {sb.discovery_year}",
                "gap": "Rayleigh-Jeans law diverges at short wavelengths — infinite energy",
                "resolution": "Planck distribution (Planck 1900) — first quantum hypothesis",
                "priority": 10,
            })

    # Pattern 5: Anomalies (modern)
    for anomaly in data.get("anomalies", []):
        name = getattr(anomaly, "name", str(anomaly))
        sigma = getattr(anomaly, "significance_sigma", 0)
        problems.append({
            "problem": name,
            "evidence": f"Tension at {sigma}σ",
            "gap": "Standard Model / ΛCDM prediction disagrees with measurement",
            "resolution": "New physics beyond current theory",
            "priority": min(10, max(5, int(sigma))),
        })

    return sorted(problems, key=lambda p: p["priority"], reverse=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Test 3: Frontier Classification by Era
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_era_classification(modifier: CorrespondenceRewardModifier, era_name: str):
    """Classify era-appropriate theorems against the frontier map.

    Shows what reward multipliers the explorer sees for theorems at each era.
    """
    theorems = ERA_THEOREMS.get(era_name, [])
    if not theorems:
        return None

    print(f"\n  Theorem classification and reward shaping:")
    print(f"  {'Theorem':<40} {'Zone':<24} {'Type':<12} {'Mult':>6} {'Reward':>8}")
    print(f"  {'─'*40} {'─'*24} {'─'*12} {'─'*6} {'─'*8}")

    results = []
    for theo_id, statement in theorems:
        zone = modifier._classify_proof(statement, "", None, None)
        zone_name = zone.name if zone else "unclassified"
        zone_type = zone.zone_type.value if zone else "?"
        mult = zone.reward_multiplier if zone else 1.0

        # Simulate: base reward 1.5, apply zone multiplier
        base_r = 1.5
        zone_effect = 1.0 + modifier.zone_multiplier_scale * (mult - 1.0)
        mod_r = base_r * zone_effect

        # Also check failure points
        resolved, reproduced = modifier._check_failure_points(statement, "")
        fail_mod = modifier.failure_coords.estimate_reward_modifier(
            statement, resolved, reproduced
        )
        failure_effect = (
            modifier.failure_bonus_scale * max(0, fail_mod)
            - modifier.failure_penalty_scale * max(0, -fail_mod)
        )
        final_r = mod_r + failure_effect

        direction = "↑" if final_r > base_r else ("↓" if final_r < base_r else "→")
        res_str = f"+resolved:{','.join(resolved)}" if resolved else ""
        rep_str = f"-reproduced:{','.join(reproduced)}" if reproduced else ""
        note = " ".join(filter(None, [res_str, rep_str]))

        print(f"  {direction} {theo_id:<38} {zone_name:<24} {zone_type:<12} {mult:>5.1f}× {final_r:>7.3f}  {note}")

        results.append({
            "theorem": theo_id,
            "zone": zone_name,
            "zone_type": zone_type,
            "multiplier": mult,
            "base_reward": base_r,
            "final_reward": final_r,
            "resolved": list(resolved),
            "reproduced": list(reproduced),
        })

    # Summary statistics
    multipliers = [r["multiplier"] for r in results]
    final_rewards = [r["final_reward"] for r in results]
    bd_count = sum(1 for r in results if r["zone_type"] == "BREAKDOWN")
    est_count = sum(1 for r in results if r["zone_type"] == "ESTABLISHED")
    unc_count = sum(1 for r in results if r["zone_type"] == "UNCERTAIN")
    unk_count = sum(1 for r in results if r["zone_type"] == "?")

    print(f"\n  Summary: {len(results)} theorems — "
          f"BD={bd_count} UNC={unc_count} EST={est_count} UNK={unk_count} | "
          f"Reward range: [{min(final_rewards):.2f}, {max(final_rewards):.2f}], "
          f"Mean multiplier: {sum(multipliers)/len(multipliers):.2f}×")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Test 4: Cross-Era Prioritization
# ═══════════════════════════════════════════════════════════════════════════════

def cross_era_prioritization(modifier: CorrespondenceRewardModifier):
    """Test: if we take theorems from era N, does the reward landscape point
    toward era N+1 discoveries?

    Key metric: the ratio of (next-era theorem reward) / (current-era theorem reward).
    If ratio > 1, the explorer is correctly incentivized to explore forward.
    """
    print(f"\n  Cross-era prioritization analysis:")
    print(f"  {'Era Transition':<35} {'Current Mean':>12} {'Next Mean':>12} {'Ratio':>8} {'Signal':>10}")
    print(f"  {'─'*35} {'─'*12} {'─'*12} {'─'*8} {'─'*10}")

    era_names = list(ERA_THEOREMS.keys())

    for i in range(len(era_names) - 1):
        current_era = era_names[i]
        next_era = era_names[i + 1]

        # Compute mean reward for current era theorems
        current_theorems = ERA_THEOREMS[current_era]
        current_rewards = []
        for theo_id, statement in current_theorems:
            zone = modifier._classify_proof(statement, "", None, None)
            mult = zone.reward_multiplier if zone else 1.0
            zone_effect = 1.0 + modifier.zone_multiplier_scale * (mult - 1.0)
            current_rewards.append(1.5 * zone_effect)

        # Compute mean reward for next era theorems
        next_theorems = ERA_THEOREMS[next_era]
        next_rewards = []
        for theo_id, statement in next_theorems:
            zone = modifier._classify_proof(statement, "", None, None)
            mult = zone.reward_multiplier if zone else 1.0
            zone_effect = 1.0 + modifier.zone_multiplier_scale * (mult - 1.0)
            next_rewards.append(1.5 * zone_effect)

        current_mean = sum(current_rewards) / len(current_rewards)
        next_mean = sum(next_rewards) / len(next_rewards) if next_rewards else current_mean
        ratio = next_mean / current_mean if current_mean > 0 else 1.0

        signal = ("PULL FORWARD" if ratio > 1.05 else
                  "NEUTRAL" if ratio > 0.95 else
                  "PULL BACKWARD")

        transition = f"{current_era.split('(')[0].strip()} → {next_era.split('(')[0].strip()}"
        print(f"  {transition:<35} {current_mean:>12.3f} {next_mean:>12.3f} {ratio:>7.2f}× {signal:>10}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Temporal Gating Evaluation — can the explorer rediscover physics?"
    )
    parser.add_argument("--era", type=int, default=None,
                       help="Focus on a specific era year (overrides --all)")
    parser.add_argument("--all", action="store_true",
                       help="Run full timeline analysis across all eras")
    parser.add_argument("--compare", type=int, nargs=2, default=None,
                       metavar=("YEAR1", "YEAR2"),
                       help="Compare two specific years (e.g., --compare 1904 1914)")
    args = parser.parse_args()

    print("╔" + "=" * 72 + "╗")
    print("║  Temporal Gating Evaluation                                          ║")
    print("║  Can the explorer rediscover physics from historical constraints?    ║")
    print("╚" + "=" * 72 + "╝")

    # Load the modifier for classification
    modifier = CorrespondenceRewardModifier(
        frontier_map=build_standard_frontier_map(),
        failure_coords=build_standard_failure_coordinates(),
    )

    eras_to_analyze = []

    if args.era is not None:
        # Find which era name corresponds to this year
        for name, cutoff in ERA_CUTOFFS.items():
            if cutoff == args.era:
                eras_to_analyze.append((name, cutoff))
                break
        if not eras_to_analyze:
            # Use the closest era
            eras_to_analyze.append((f"≤{args.era}", args.era))

    elif args.compare:
        for year in args.compare:
            for name, cutoff in ERA_CUTOFFS.items():
                if cutoff == year:
                    eras_to_analyze.append((name, cutoff))
                    break
            else:
                eras_to_analyze.append((f"≤{year}", year))

    elif args.all:
        # All era cutoffs
        eras_to_analyze = list(ERA_CUTOFFS.items())

    else:
        # Default: show key transition points
        key_eras = [
            ("classical", 1860),
            ("classical crisis", 1900),
            ("pre-relativity", 1904),
            ("pre-GR", 1914),
            ("modern", 2026),
        ]
        eras_to_analyze = [
            (name, year) for name, year in key_eras
            if name in ERA_CUTOFFS
        ]

    # ═══════════════════════════════════════════════════════════════════════
    # For each era, run the full analysis
    # ═══════════════════════════════════════════════════════════════════════

    all_era_results = {}

    for era_name, era_year in eras_to_analyze:
        print(f"\n{'='*72}")
        print(f"ERA: {era_name} (≤{era_year})")
        print(f"{'='*72}")

        # Historical context
        knowledge = ERA_KNOWLEDGE.get(era_name, {})
        if knowledge:
            print(f"\n  Historical context:")
            print(f"    Known theory: {knowledge.get('known_theory', [])[0] if knowledge.get('known_theory') else 'N/A'}")
            next_disc = knowledge.get("next_discovery", "")
            if next_disc:
                print(f"    Next discovery to make: {next_disc}")

        # 1. Data inventory
        print(f"\n  ── Data Inventory ──")
        n_total = print_data_inventory(era_year, era_name)

        # 2. Open problems
        print(f"\n  ── Open Problems ──")
        problems = analyze_open_problems(era_year, era_name)
        if problems:
            for p in problems[:8]:  # Top 8
                bar = "█" * p["priority"]
                print(f"  [{bar}] {p['problem']}")
                print(f"       Evidence: {p['evidence']}")
                print(f"       Gap: {p['gap']}")
                print(f"       → {p['resolution']}")
        else:
            print(f"  (No open problems detected — theory is complete at this level)")

        # 3. Should prioritize
        if knowledge.get("should_prioritize"):
            print(f"\n  ── Explorer Should Prioritize ──")
            for i, goal in enumerate(knowledge["should_prioritize"], 1):
                print(f"  {i}. {goal}")

        # 4. Frontier classification
        print(f"\n  ── Frontier Classification ──")
        classifications = analyze_era_classification(modifier, era_name)

        all_era_results[era_name] = {
            "year": era_year,
            "total_data": n_total,
            "problems": len(problems),
            "classifications": classifications,
        }

    # ═══════════════════════════════════════════════════════════════════════
    # Cross-era prioritization
    # ═══════════════════════════════════════════════════════════════════════

    if len(eras_to_analyze) > 1:
        print(f"\n{'='*72}")
        print(f"CROSS-ERA PRIORITIZATION ANALYSIS")
        print(f"{'='*72}")
        print(f"\n  Question: Does the reward landscape pull the explorer forward in time?")
        print(f"  If the mean reward for era N+1 theorems > era N theorems, the explorer")
        print(f"  is incentivized to discover the next generation of physics.")
        cross_era_prioritization(modifier)

    # ═══════════════════════════════════════════════════════════════════════
    # Final assessment
    # ═══════════════════════════════════════════════════════════════════════

    print(f"\n{'='*72}")
    print(f"ASSESSMENT")
    print(f"{'='*72}")

    print(f"""
  The temporal gating infrastructure enables the strongest validation of
  the theta-core architecture:

  1. TRAIN on era N data (restrict constants, particles, anomalies)
  2. Let the explorer search the theorem space
  3. OBSERVE what it prioritizes
  4. COMPARE to what physicists actually discovered in era N+1

  If the explorer consistently prioritizes the same problems that led to
  the next revolution (blackbody → QM, constancy of c → SR, equivalence
  principle → GR), that is evidence the architecture does genuine science.

  Current status:
  - 192 physical constants encoded with discovery years
  - 11 era cutoffs from classical (≤1860) to modern (≤2026)
  - {len(ERA_THEOREMS)} eras with theorem evaluation sets
  - Frontier map + failure coordinates for reward shaping
  - CorrespondenceRewardModifier wired into explorer trainer

  Ready for full temporal gating evaluation with a trained GNN+MCTS.
""")

    print(f"  Next steps:")
    print(f"    1. Train GNN on era N dependency graph subgraph")
    print(f"    2. Run MCTS exploration with era N data constraint")
    print(f"    3. Rank discovered theorems by reward")
    print(f"    4. Compare top-ranked to historical era N+1 discoveries")
    print(f"    5. Metric: Precision@K of historical discoveries in top-K")
    print(f"    6. Repeat for all era transitions")
    print(f"    7. Baseline: untrained GNN (random search) vs trained GNN")

    return 0


if __name__ == "__main__":
    sys.exit(main())
