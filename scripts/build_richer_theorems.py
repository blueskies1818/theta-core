#!/usr/bin/env python3
"""Build a richer theorem set for training the GNN+MCTS explorer.

Generates theorems that exercise proof patterns beyond single-tactic proofs:
  Level 2 — hypothesis usage, symmetry, chained rewrite
  Level 3 — transitivity, combined tactics, contraposition
  Level 4 — case analysis, induction (future)

Each theorem is a valid Lean 4 statement with a known proof. The training
loop uses these as GRPO training examples — MCTS searches for proofs and
the proof checker validates them.

Usage:
    python scripts/build_richer_theorems.py
    python scripts/build_richer_theorems.py --append data/raw/physics_theorems_pre1905.jsonl
"""

import argparse, json
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent


def make_theorem(name: str, statement: str, proof: str, era: str = "classical",
                 zone: str = "mathematical", domain: str = "algebra",
                 description: str = "") -> dict:
    """Create a theorem entry."""
    return {
        "name": name,
        "statement": statement,
        "proof": proof,
        "era": era,
        "frontier_zone": zone,
        "domain": domain,
        "description": description,
    }


def build_level2_hypothesis_theorems() -> list[dict]:
    """Theorems requiring hypothesis usage (single hypothesis).

    These teach the GNN to look at hypotheses, not just global lemmas.
    """
    thms = []

    # --- Hypothesis rewrite ---
    thms.append(make_theorem(
        "hyp_rewrite_eq",
        "theorem hyp_rewrite_eq (a b : ℝ) (h : a = b) : b = a",
        "  rw [h]",
        era="classical", zone="mathematical", domain="algebra",
        description="Hypothesis rewrite: given h: a=b, prove b=a"
    ))

    thms.append(make_theorem(
        "hyp_rewrite_add",
        "theorem hyp_rewrite_add (a b c : ℝ) (h : a = b) : a + c = b + c",
        "  rw [h]",
        era="classical", zone="mathematical", domain="algebra",
        description="Hypothesis rewrite: given h: a=b, prove a+c = b+c"
    ))

    thms.append(make_theorem(
        "hyp_rewrite_mul",
        "theorem hyp_rewrite_mul (a b c : ℝ) (h : a = b) : a * c = b * c",
        "  rw [h]",
        era="classical", zone="mathematical", domain="algebra",
        description="Hypothesis rewrite: given h: a=b, prove a*c = b*c"
    ))

    # --- Hypothesis exact ---
    thms.append(make_theorem(
        "hyp_exact_eq",
        "theorem hyp_exact_eq (a b : ℝ) (h : a = b) : a = b",
        "  exact h",
        era="classical", zone="mathematical", domain="algebra",
        description="Hypothesis exact: given h: a=b, exact h proves a=b"
    ))

    thms.append(make_theorem(
        "hyp_exact_ineq",
        "theorem hyp_exact_ineq (a b : ℝ) (h : a ≤ b) : a ≤ b",
        "  exact h",
        era="classical", zone="mathematical", domain="algebra",
        description="Hypothesis exact: given h: a≤b, exact h proves a≤b"
    ))

    # --- Symmetry ---
    thms.append(make_theorem(
        "hyp_symm_eq",
        "theorem hyp_symm_eq (a b : ℝ) (h : a = b) : b = a",
        "  exact h.symm",
        era="classical", zone="mathematical", domain="algebra",
        description="Symmetry: h: a=b gives h.symm: b=a"
    ))

    # --- Rewrite then exact ---
    thms.append(make_theorem(
        "hyp_rewrite_then_exact",
        "theorem hyp_rewrite_then_exact (a b : ℝ) (h : a = b) : b = a",
        "  rw [h]",
        era="classical", zone="mathematical", domain="algebra",
        description="Rewrite then exact: rw [h] proves b=a (already rfl)"
    ))

    # --- Hypothesis at location ---
    thms.append(make_theorem(
        "hyp_rewrite_at_left",
        "theorem hyp_rewrite_at_left (a b : ℝ) (h : a = b) : b + a = a + b",
        "  rw [h]",
        era="classical", zone="mathematical", domain="algebra",
        description="Rewrite at position: rw [h] changes all a to b, then rfl"
    ))

    return thms


def build_level3_multi_step_theorems() -> list[dict]:
    """Theorems requiring multiple hypotheses and tactic chains.

    These teach the GNN to combine tactics for multi-step reasoning.
    """
    thms = []

    # --- Transitivity (2-step chain) ---
    thms.append(make_theorem(
        "transitivity_eq",
        "theorem transitivity_eq (a b c : ℝ) (h₁ : a = b) (h₂ : b = c) : a = c",
        "  rw [h₁, h₂]",
        era="classical", zone="mathematical", domain="algebra",
        description="Transitivity: a=b and b=c implies a=c via rw [h₁, h₂]"
    ))

    thms.append(make_theorem(
        "transitivity_ineq",
        "theorem transitivity_ineq (a b c : ℝ) (h₁ : a ≤ b) (h₂ : b ≤ c) : a ≤ c",
        "  linarith",
        era="classical", zone="mathematical", domain="algebra",
        description="Inequality transitivity: a≤b and b≤c gives a≤c via linarith"
    ))

    thms.append(make_theorem(
        "transitivity_lt",
        "theorem transitivity_lt (a b c : ℝ) (h₁ : a < b) (h₂ : b < c) : a < c",
        "  linarith",
        era="classical", zone="mathematical", domain="algebra",
        description="Strict inequality transitivity via linarith"
    ))

    # --- Combined rewrite + ring ---
    thms.append(make_theorem(
        "hyp_rewrite_then_ring",
        "theorem hyp_rewrite_then_ring (a b : ℝ) (h : a = b) : (a + b)^2 = 4 * a * b + (a - b)^2",
        "  rw [h] at *; ring",
        era="classical", zone="mathematical", domain="algebra",
        description="Rewrite hypothesis then ring: substitute b for a and expand"
    ))

    thms.append(make_theorem(
        "hyp_rewrite_then_linarith",
        "theorem hyp_rewrite_then_linarith (x y z : ℝ) (h : x = y + z) : x - z = y",
        "  rw [h]; ring",
        era="classical", zone="mathematical", domain="algebra",
        description="Rewrite then ring: substitute and simplify expression"
    ))

    # --- Combined field_simp + ring ---
    thms.append(make_theorem(
        "field_simp_then_ring",
        "theorem field_simp_then_ring (a b : ℝ) (hb : b ≠ 0) : a / b + 1 = (a + b) / b",
        "  field_simp [hb]",
        era="classical", zone="mathematical", domain="algebra",
        description="field_simp: rational expression simplification (ring not needed)"
    ))

    # --- Contraposition ---
    thms.append(make_theorem(
        "contrapositive_eq",
        "theorem contrapositive_eq (a b : ℝ) (h : a = b → a ≤ b) : a > b → a ≠ b",
        "  intro hgt heq; apply not_lt.mpr (h heq); exact hgt",
        era="classical", zone="mathematical", domain="algebra",
        description="Contrapositive reasoning: if a=b → a≤b, then a>b → a≠b"
    ))

    # --- Intro + apply (implication chain) ---
    thms.append(make_theorem(
        "intro_apply_chain",
        "theorem intro_apply_chain (P Q R : Prop) (hPQ : P → Q) (hQR : Q → R) : P → R",
        "  intro hP; apply hQR; apply hPQ; exact hP",
        era="classical", zone="mathematical", domain="algebra",
        description="Implication chain: P→Q and Q→R gives P→R via intro/apply"
    ))

    # --- Have + exact (intermediate lemma) ---
    thms.append(make_theorem(
        "have_intermediate",
        "theorem have_intermediate (a b c : ℝ) (h : a + b = c) : a = c - b",
        "  linarith",
        era="classical", zone="mathematical", domain="algebra",
        description="Linear arithmetic: linarith handles a+b=c → a=c-b"
    ))

    # --- Calc block ---
    thms.append(make_theorem(
        "calc_transitivity",
        "theorem calc_transitivity (a b c d : ℝ) (h₁ : a = b) (h₂ : b = c) (h₃ : c = d) : a = d",
        "  calc\n    a = b := h₁\n    _ = c := h₂\n    _ = d := h₃",
        era="classical", zone="mathematical", domain="algebra",
        description="Calc chain: a=b=c=d via structured calculation"
    ))

    return thms


def build_level3_physics_bridged_theorems() -> list[dict]:
    """Physics theorems where the proof uses hypotheses (more realistic).

    These bridge the gap between trivial identities and genuine physics proofs.
    The hypotheses encode physical constraints (conservation, bounds, etc.).
    """
    thms = []

    # --- Conservation of energy (algebraic form) ---
    thms.append(make_theorem(
        "energy_conservation_algebraic",
        "theorem energy_conservation_algebraic (E₁ E₂ ΔE : ℝ) (h : E₂ = E₁ + ΔE) (hz : ΔE = 0) : E₂ = E₁",
        "  rw [hz, add_zero] at h; exact h",
        era="classical", zone="thermodynamics", domain="physics",
        description="Energy conservation: ΔE=0 implies E₁=E₂ (classical)"
    ))

    thms.append(make_theorem(
        "energy_nonconservation_algebraic",
        "theorem energy_nonconservation_algebraic (E₁ E₂ ΔE : ℝ) (h : E₂ = E₁ + ΔE) (hpos : ΔE > 0) : E₂ > E₁",
        "  linarith",
        era="pre_relativity", zone="qft_divergence", domain="physics",
        description="Energy non-conservation: ΔE>0 at quantum scale (uncertainty — pre-1905 concept, formalized 1927)"
    ))

    # --- Galilean velocity addition (classical) ---
    thms.append(make_theorem(
        "velocity_addition_galilean",
        "theorem velocity_addition_galilean (u v w : ℝ) (h : w = u + v) : u = w - v",
        "  linarith",
        era="classical", zone="mechanics", domain="physics",
        description="Galilean velocity addition: w=u+v implies u=w-v"
    ))

    # --- Relativistic velocity bound ---
    thms.append(make_theorem(
        "velocity_bound_relativistic",
        "theorem velocity_bound_relativistic (v c : ℝ) (h : v < c) (hc : c > 0) : v / c < 1",
        "  exact (div_lt_one hc).mpr h",
        era="pre_relativity", zone="gr_qft_incompatibility", domain="physics",
        description="Relativistic bound: v<c implies v/c<1 (requires c>0 hypothesis)"
    ))

    # --- Ideal gas law (algebraic manipulation) ---
    thms.append(make_theorem(
        "ideal_gas_pressure_ratio",
        "theorem ideal_gas_pressure_ratio (P V n R T : ℝ) (h : P * V = n * R * T) (hV : V ≠ 0) : P = n * R * T / V",
        "  field_simp [hV]; exact h",
        era="classical", zone="thermodynamics", domain="physics",
        description="Ideal gas law: PV=nRT solved for P via field_simp and hypothesis"
    ))

    # --- Blackbody radiation (bounds) ---
    thms.append(make_theorem(
        "blackbody_energy_density_bound",
        "theorem blackbody_energy_density_bound (u T : ℝ) (hu : 0 ≤ u) (hT : T > 0) : u / (T^4) ≥ 0",
        "  apply div_nonneg hu; positivity",
        era="classical_crisis", zone="qft_divergence", domain="physics",
        description="Blackbody energy density: u/T^4≥0 (finite at all T>0, pre-Planck)"
    ))

    # --- Photoelectric threshold (quantum) ---
    thms.append(make_theorem(
        "photoelectric_threshold_condition",
        "theorem photoelectric_threshold_condition (E_photon W : ℝ) (hE : E_photon ≥ W) : E_photon - W ≥ 0",
        "  linarith",
        era="pre_relativity", zone="qft_divergence", domain="physics",
        description="Photoelectric threshold: electron emission iff photon energy ≥ work function"
    ))

    # --- Gravitational potential superposition ---
    thms.append(make_theorem(
        "gravitational_superposition",
        "theorem gravitational_superposition (V₁ V₂ V : ℝ) (h : V = V₁ + V₂) : V - V₁ = V₂",
        "  linarith",
        era="classical", zone="gr_classical", domain="physics",
        description="Gravitational potential superposition: V=V₁+V₂ implies V-V₁=V₂"
    ))

    return thms


def build_level4_multi_tactic_theorems() -> list[dict]:
    """Theorems requiring 2-3 distinct tactics in sequence.

    These are the core multi-step training data for Phase 2 curriculum learning.
    Each proof chains multiple tactics: rewrite → ring, intro → apply → linarith,
    have → exact, field_simp → ring, etc.

    The GNN must learn to select lemmas for intermediate states, not just the
    initial goal — the key capability missing from Wave 1.
    """
    thms = []

    # --- Chain: rewrite then ring (substitute + normalize) ---
    thms.append(make_theorem(
        "multi_rewrite_ring",
        "theorem multi_rewrite_ring (a b x : ℝ) (h : x = a + b) : x^2 = a^2 + 2*a*b + b^2",
        "  rw [h]; ring",
        era="classical", zone="mathematical", domain="algebra",
        description="2-step: rw [h] substitutes x, ring expands polynomial"
    ))

    thms.append(make_theorem(
        "multi_rewrite_ring2",
        "theorem multi_rewrite_ring2 (a b c : ℝ) (h : c = a - b) : (a + b)^2 = c^2 + 4*a*b",
        "  rw [h]; ring",
        era="classical", zone="mathematical", domain="algebra",
        description="2-step: rw then ring on quadratic identity"
    ))

    # --- Chain: intro then apply then linarith (implication with arithmetic) ---
    thms.append(make_theorem(
        "multi_intro_apply_linarith",
        "theorem multi_intro_apply_linarith (x y z : ℝ) (h : x + y ≤ z) : x ≤ z - y",
        "  linarith",
        era="classical", zone="mathematical", domain="algebra",
        description="Linear arithmetic from inequality hypothesis"
    ))

    thms.append(make_theorem(
        "multi_intro_linarith_chain",
        "theorem multi_intro_linarith_chain (a b c d : ℝ) (h1 : a ≤ b) (h2 : c ≤ d) : a + c ≤ b + d",
        "  linarith",
        era="classical", zone="mathematical", domain="algebra",
        description="Inequality addition: given two ≤, prove sum via linarith"
    ))

    # --- Chain: have intermediate lemma then exact ---
    thms.append(make_theorem(
        "multi_have_exact",
        "theorem multi_have_exact (a b : ℝ) (h : a = b) : 2*a = 2*b",
        "  have h2 : 2*a = 2*b := by rw [h]; exact h2",
        era="classical", zone="mathematical", domain="algebra",
        description="2-step: have intermediate lemma, then use it"
    ))

    # --- Chain: rewrite at hypothesis then linarith ---
    thms.append(make_theorem(
        "multi_rewrite_at_linarith",
        "theorem multi_rewrite_at_linarith (x y z : ℝ) (h : x = y + z) : x ≤ y + z",
        "  rw [h]",
        era="classical", zone="mathematical", domain="algebra",
        description="Rewrite then exact: x=y+z rewrites to y+z≤y+z which is rfl"
    ))

    # --- Chain: field_simp then ring (rational to polynomial) ---
    thms.append(make_theorem(
        "multi_field_ring",
        "theorem multi_field_ring (a b : ℝ) (hb : b ≠ 0) : (a + b)/b = a/b + 1",
        "  field_simp [hb]; ring",
        era="classical", zone="mathematical", domain="algebra",
        description="2-step: field_simp clears denominator, ring normalizes numerator"
    ))

    thms.append(make_theorem(
        "multi_field_ring2",
        "theorem multi_field_ring2 (x y : ℝ) (hx : x ≠ 0) : (x + y)/x - y/x = 1",
        "  field_simp [hx]; ring",
        era="classical", zone="mathematical", domain="algebra",
        description="2-step: field_simp denominators, ring simplifies to 1"
    ))

    # --- Chain: apply lemma then exact term ---
    thms.append(make_theorem(
        "multi_apply_exact",
        "theorem multi_apply_exact (a b : ℝ) (h : a > 0) (h2 : b > 0) : a + b > 0",
        "  positivity",
        era="classical", zone="mathematical", domain="algebra",
        description="Positivity: sum of two positive numbers is positive"
    ))

    thms.append(make_theorem(
        "multi_apply_exact2",
        "theorem multi_apply_exact2 (a b : ℝ) (ha : a > 0) (hb : b > 0) : a*b > 0",
        "  positivity",
        era="classical", zone="mathematical", domain="algebra",
        description="Positivity: product of two positive numbers is positive"
    ))

    # --- Chain: rewrite list then simp ---
    thms.append(make_theorem(
        "multi_rewrite_simp",
        "theorem multi_rewrite_simp (a b : ℝ) (h : a = b) : a + a = b + b",
        "  rw [h]",
        era="classical", zone="mathematical", domain="algebra",
        description="Rewrite then rfl: substitute and get b+b=b+b"
    ))

    # --- Chain: intro intro apply (nested implications) ---
    thms.append(make_theorem(
        "multi_intro_intro_apply",
        "theorem multi_intro_intro_apply (a b c : ℝ) (h : a = b) : a = c → b = c",
        "  intro hac; rw [← h, hac]",
        era="classical", zone="mathematical", domain="algebra",
        description="2-step: intro implication, rewrite with hypothesis"
    ))

    # --- Hard: 3 distinct steps ---
    thms.append(make_theorem(
        "multi_three_step",
        "theorem multi_three_step (a b c d : ℝ) (h : a = b + c) (h2 : c = d) : a - b = d",
        "  rw [h2] at h; rw [h]; ring",
        era="classical", zone="mathematical", domain="algebra",
        description="3-step: rewrite hypothesis, substitute, ring simplify"
    ))

    thms.append(make_theorem(
        "multi_three_step2",
        "theorem multi_three_step2 (x y : ℝ) (h : x + y = 0) : x^2 = y^2",
        "  have hx : x = -y := by linarith; rw [hx]; ring",
        era="classical", zone="mathematical", domain="algebra",
        description="3-step: have intermediate (linarith), rewrite, ring"
    ))

    return thms


def main():
    parser = argparse.ArgumentParser(description="Build richer theorem dataset")
    parser.add_argument("--append", default=None,
                        help="Append to existing file instead of overwriting")
    parser.add_argument("--output", default="data/raw/richer_theorems.jsonl",
                        help="Output file")
    args = parser.parse_args()

    all_theorems = []
    all_theorems.extend(build_level2_hypothesis_theorems())
    all_theorems.extend(build_level3_multi_step_theorems())
    all_theorems.extend(build_level3_physics_bridged_theorems())
    all_theorems.extend(build_level4_multi_tactic_theorems())

    out_path = _project_root / args.output

    if args.append:
        append_path = Path(args.append)
        if not append_path.is_absolute():
            append_path = _project_root / append_path
        if append_path.exists():
            with open(append_path) as f:
                for line in f:
                    try:
                        all_theorems.insert(0, json.loads(line))
                    except json.JSONDecodeError:
                        continue

    with open(out_path, "w") as f:
        for t in all_theorems:
            f.write(json.dumps(t) + "\n")

    # Summary
    by_level = {"level2": 0, "level3_math": 0, "level3_physics": 0}
    for t in all_theorems:
        name = t["name"]
        if name.startswith("hyp_") or name.startswith("transitivity") or "symm" in name:
            pass  # counts below
    print(f"Generated {len(all_theorems)} theorems:")
    print(f"  Level 2 (hypothesis usage):  {len(build_level2_hypothesis_theorems())}")
    print(f"  Level 3 (multi-step math):   {len(build_level3_multi_step_theorems())}")
    print(f"  Level 3 (physics bridged):   {len(build_level3_physics_bridged_theorems())}")
    print(f"  Level 4 (multi-tactic 2-3 step): {len(build_level4_multi_tactic_theorems())}")
    print(f"  Total new theorems:          {len(build_level2_hypothesis_theorems()) + len(build_level3_multi_step_theorems()) + len(build_level3_physics_bridged_theorems()) + len(build_level4_multi_tactic_theorems())}")
    print(f"\nWritten to {out_path}")
    print(f"\nProof patterns covered:")
    patterns = set()
    for t in all_theorems:
        proof = t["proof"].strip()
        if "rw [" in proof:
            if "; rw" in proof or "; ring" in proof or "; linarith" in proof:
                patterns.add("rw + tactic (chain)")
            else:
                patterns.add("rw (rewrite)")
        if "exact " in proof and "intro" not in proof:
            patterns.add("exact (assumption)")
        if "intro " in proof:
            patterns.add("intro (implication)")
        if "apply " in proof:
            patterns.add("apply (forward)")
        if "linarith" in proof:
            patterns.add("linarith (linear)")
        if "ring" in proof:
            patterns.add("ring (polynomial)")
        if "field_simp" in proof:
            patterns.add("field_simp (rational)")
        if "have " in proof:
            patterns.add("have (intermediate)")
        if "calc" in proof:
            patterns.add("calc (structured)")
        if "positivity" in proof:
            patterns.add("positivity (sign)")
        if ".symm" in proof:
            patterns.add(".symm (symmetry)")
    for pat in sorted(patterns):
        print(f"    {pat}")


if __name__ == "__main__":
    main()
