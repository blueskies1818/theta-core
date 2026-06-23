#!/usr/bin/env python3
"""Build and test Lean dimensional-constancy proofs for the 5 exact-match claims.

Claims:
  1. E*lambda = h*c        (Hydrogen Balmer)
  2. E/n = constant        (Spin quantization)
  3. E_peak/T = constant   (Wien displacement)
  4. E/gamma = m*c^2       (Rest energy)
  5. (c*t)^2 - x^2 = inv   (Spacetime interval)

Each proof shows the expression is dimensionally constant using the
defining physical relationship as a hypothesis.
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.proof_checker.lean_interface import LeanProofChecker
from src.proof_checker.formats import ProofResult


def make_claim_scenarios():
    """Return list of (name, lean_code) for each claim."""
    scenarios = []

    # ── Claim 1: E*lambda = h*c ──────────────────────────────────────
    # Hypothesis: E * lambda = h * c
    # Simple: the invariant IS the hypothesis
    claim1_code = """theorem dim_const_E_lambda (E lambda h c : ℝ) (h_rel : E * lambda = h * c) : E * lambda = h * c := by
  exact h_rel"""
    scenarios.append(("E*lambda", claim1_code))

    # ── Claim 2: E/n = constant ──────────────────────────────────────
    # Hypothesis: E = n * const  →  E/n = const
    claim2_code = """theorem dim_const_E_div_n (E n const : ℝ) (h_def : E = n * const) (h_nz : n ≠ 0) : E / n = const := by
  rw [h_def]
  field_simp [h_nz]"""
    scenarios.append(("E/n", claim2_code))

    # ── Claim 3: E_peak/T = constant ─────────────────────────────────
    # Hypothesis: E_peak = const * T  →  E_peak/T = const
    claim3_code = """theorem dim_const_E_peak_div_T (E_peak T const : ℝ) (h_wien : E_peak = const * T) (h_Tz : T ≠ 0) : E_peak / T = const := by
  rw [h_wien]
  field_simp [h_Tz]"""
    scenarios.append(("E_peak/T", claim3_code))

    # ── Claim 4: E/gamma = m*c^2 ─────────────────────────────────────
    # Hypothesis: E = gamma * m * c^2  →  E/gamma = m*c^2
    claim4_code = """theorem dim_const_E_div_gamma (E gamma m c : ℝ) (h_rel : E = gamma * (m * c ^ 2)) (h_gz : gamma ≠ 0) : E / gamma = m * c ^ 2 := by
  rw [h_rel]
  field_simp [h_gz]"""
    scenarios.append(("E/gamma", claim4_code))

    # ── Claim 5: (c*t)^2 - x^2 = invariant ───────────────────────────
    # Lorentz transform: t' = gamma*(t - v*x/c^2), x' = gamma*(x - v*t)
    # Hypothesis: gamma^2*(c^2 - v^2) = c^2
    # Prove: (c*t')^2 - x'^2 = (c*t)^2 - x^2
    claim5_code = """theorem dim_const_spacetime_interval (c t x t' x' gamma v : ℝ)
    (h_t' : t' = gamma * (t - v * x / c ^ 2))
    (h_x' : x' = gamma * (x - v * t))
    (h_gamma : gamma ^ 2 * (c ^ 2 - v ^ 2) = c ^ 2)
    (h_cz : c ≠ 0) :
    (c * t') ^ 2 - x' ^ 2 = (c * t) ^ 2 - x ^ 2 := by
  rw [h_t', h_x']
  have h_gamma_id : gamma ^ 2 * (c ^ 2 - v ^ 2) = c ^ 2 := h_gamma
  have h_expr : (c * (gamma * (t - v * x / c ^ 2))) ^ 2 - (gamma * (x - v * t)) ^ 2
             = (gamma ^ 2 * (c ^ 2 - v ^ 2) / c ^ 2) * ((c * t) ^ 2 - x ^ 2) := by
    field_simp [h_cz]
    ring
  rw [h_expr]
  have h_factor : gamma ^ 2 * (c ^ 2 - v ^ 2) / c ^ 2 = 1 := by
    rw [h_gamma_id]
    field_simp [h_cz]
  rw [h_factor]
  ring"""
    scenarios.append(("(c*t)^2-x^2", claim5_code))

    return scenarios


def main():
    checker = LeanProofChecker(timeout=30.0)
    scenarios = make_claim_scenarios()

    print("=" * 60)
    print("DIMENSIONAL CONSTANCY LEAN PROOF VERIFICATION")
    print("=" * 60)

    passed = 0
    failed = 0
    results = []

    for name, code in scenarios:
        print(f"\n--- {name} ---")
        result = checker.check(code)
        status = "PASS" if result.success else "FAIL"
        print(f"  {status}")
        if not result.success:
            for err in result.errors[:3]:
                print(f"  Error: {err}")
            failed += 1
        else:
            passed += 1
        results.append((name, result))

    print(f"\n{'=' * 60}")
    print(f"RESULTS: {passed}/{len(scenarios)} proofs verified")
    if failed > 0:
        print(f"FAILED: {failed}")
    print(f"{'=' * 60}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
