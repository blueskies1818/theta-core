#!/usr/bin/env python3
"""Create and verify Gate 3 lemma-novelty test theorems.
Each theorem uses a lemma NEVER seen in training proofs.
"""
import subprocess, json, re, sys
from pathlib import Path

PROJECT = Path("/home/blueman1818/Projects/theta-core")
LAKE_DIR = PROJECT / "proof_checker_env"

LEAN_PREAMBLE = """import Mathlib
open Polynomial
open Real
"""

def verify_theorem(name, statement, proof):
    stmt = statement.strip()
    stmt = re.sub(r'\s*:=.*$', '', stmt)
    stmt = re.sub(r'^(lemma|theorem)\s+\S+\s+', 'example ', stmt, count=1)
    proof = proof.strip()
    if '\n' in proof:
        lines = [l.strip() for l in proof.split('\n') if l.strip()]
        indented = '\n'.join(f"  {line}" for line in lines)
        code = f"{LEAN_PREAMBLE}\n{stmt} := by\n{indented}"
    elif proof.split()[0].rstrip(':') in ('by', 'intro', 'intros', 'apply', 'exact', 'refine',
            'rcases', 'rw', 'rwa', 'erw', 'simp', 'simpa', 'have', 'calc', 'linarith', 'nlinarith',
            'omega', 'ring', 'ring_nf', 'field_simp', 'norm_num', 'positivity', 'native_decide', 'trivial'):
        code = f"{LEAN_PREAMBLE}\n{stmt} := {proof}" if proof.startswith('by ') else f"{LEAN_PREAMBLE}\n{stmt} := by {proof}"
    else:
        code = f"{LEAN_PREAMBLE}\n{stmt} := {proof}"
    try:
        proc = subprocess.run(["lake", "env", "lean", "--stdin"], input=code, capture_output=True,
                              text=True, timeout=30, cwd=str(LAKE_DIR))
        if proc.returncode == 0:
            return True, ""
        err = (proc.stderr or proc.stdout or "").strip()
        err_lines = [l for l in err.split('\n') if l.strip() and 'error' in l.lower()]
        return False, err_lines[0][:200] if err_lines else err[:200]
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except Exception as e:
        return False, str(e)[:200]

theorems = [
    # --- Simple evaluations ---
    {
        "name": "poly_eval_C_add_X",
        "statement": "theorem poly_eval_C_add_X (a x : ℝ) : (Polynomial.C a + Polynomial.X).eval x = a + x",
        "proof": "simp",
        "era": "classical", "frontier_zone": "algebra", "domain": "Algebra",
        "description": "Evaluation of C a + X at x gives a + x — requires eval_add, eval_X lemmas",
    },
    {
        "name": "poly_mul_X_add_C",
        "statement": "theorem poly_mul_X_add_C (p : Polynomial ℝ) (a x : ℝ) : (p * (Polynomial.X + Polynomial.C a)).eval x = p.eval x * (x + a)",
        "proof": "simp [mul_add]",
        "era": "classical", "frontier_zone": "algebra", "domain": "Algebra",
        "description": "Evaluation of p*(X+C a) distributes — requires eval_mul lemma",
    },
    {
        "name": "poly_eval_mul_X_sub_C",
        "statement": "theorem poly_eval_mul_X_sub_C (p : Polynomial ℝ) (a : ℝ) : (p * (Polynomial.X - Polynomial.C a)).eval a = 0",
        "proof": "simp",
        "era": "classical", "frontier_zone": "algebra", "domain": "Algebra",
        "description": "p*(X-C a) evaluates to zero at a — requires eval_sub lemma",
    },
    # --- Derivatives ---
    {
        "name": "poly_derivative_X_pow",
        "statement": "theorem poly_derivative_X_pow (n : ℕ) : (Polynomial.X ^ n).derivative = (n : Polynomial ℝ) * Polynomial.X ^ (n - 1)",
        "proof": "simpa using Polynomial.derivative_X_pow n",
        "era": "classical", "frontier_zone": "algebra", "domain": "Algebra",
        "description": "Derivative of X^n is n*X^(n-1) — requires derivative_X_pow lemma",
    },
    {
        "name": "poly_eval_derivative_C_mul",
        "statement": "theorem poly_eval_derivative_C_mul (p : Polynomial ℝ) (c : ℝ) : (Polynomial.C c * p).derivative = Polynomial.C c * p.derivative",
        "proof": "simp",
        "era": "classical", "frontier_zone": "algebra", "domain": "Algebra",
        "description": "Derivative of scalar multiple — requires derivative_C_mul lemma",
    },
    {
        "name": "poly_derivative_add",
        "statement": "theorem poly_derivative_add (p q : Polynomial ℝ) : (p + q).derivative = p.derivative + q.derivative",
        "proof": "simp",
        "era": "classical", "frontier_zone": "algebra", "domain": "Algebra",
        "description": "Derivative of sum is sum of derivatives — requires derivative_add lemma",
    },
    # --- Degree properties ---
    {
        "name": "poly_degree_X_pow",
        "statement": "theorem poly_degree_X_pow (n : ℕ) : (Polynomial.X ^ n : Polynomial ℝ).natDegree = n",
        "proof": "simp",
        "era": "classical", "frontier_zone": "algebra", "domain": "Algebra",
        "description": "Degree of X^n is n — requires natDegree_X_pow lemma",
    },
    {
        "name": "poly_degree_C_mul_X_pow",
        "statement": "theorem poly_degree_C_mul_X_pow (c : ℝ) (n : ℕ) (hc : c ≠ 0) : (Polynomial.C c * Polynomial.X ^ n).natDegree = n",
        "proof": "simpa using Polynomial.natDegree_C_mul_X_pow n c hc",
        "era": "classical", "frontier_zone": "algebra", "domain": "Algebra",
        "description": "Degree of c*X^n is n for nonzero c — requires natDegree_C_mul_X_pow lemma",
    },
    {
        "name": "poly_degree_mul_X",
        "statement": "theorem poly_degree_mul_X (p : Polynomial ℝ) (hp : p ≠ 0) : (p * Polynomial.X).natDegree = p.natDegree + 1",
        "proof": "simpa [Polynomial.natDegree_X] using Polynomial.natDegree_mul (p := p) (q := Polynomial.X) hp (by simpa using Polynomial.X_ne_zero)",
        "era": "classical", "frontier_zone": "algebra", "domain": "Algebra",
        "description": "Multiplying by X increases degree by 1 — requires natDegree_mul lemma",
    },
    {
        "name": "poly_degree_add_eq_left_of_degree_lt",
        "statement": "theorem poly_degree_add_eq_left_of_degree_lt (p q : Polynomial ℝ) (h : q.degree < p.degree) : (p + q).degree = p.degree",
        "proof": "simpa using Polynomial.degree_add_eq_left_of_degree_lt h",
        "era": "classical", "frontier_zone": "algebra", "domain": "Algebra",
        "description": "Degree of sum when one dominates — requires degree_add_eq_left_of_degree_lt",
    },
    {
        "name": "poly_degree_sub_eq_left_of_degree_lt",
        "statement": "theorem poly_degree_sub_eq_left_of_degree_lt (p q : Polynomial ℝ) (h : q.degree < p.degree) : (p - q).degree = p.degree",
        "proof": "simpa using Polynomial.degree_sub_eq_left_of_degree_lt h",
        "era": "classical", "frontier_zone": "algebra", "domain": "Algebra",
        "description": "Degree of difference when one dominates — requires degree_sub_eq_left_of_degree_lt",
    },
    # --- Monic properties ---
    {
        "name": "poly_monic_X_sub_C",
        "statement": "theorem poly_monic_X_sub_C (a : ℝ) : (Polynomial.X - Polynomial.C a).Monic",
        "proof": "simpa using Polynomial.monic_X_sub_C a",
        "era": "classical", "frontier_zone": "algebra", "domain": "Algebra",
        "description": "X - C a is monic — requires monic_X_sub_C lemma",
    },
    {
        "name": "poly_monic_X_pow_add",
        "statement": "theorem poly_monic_X_pow_add (n : ℕ) (p : Polynomial ℝ) (hp : p.degree < n) : (Polynomial.X ^ n + p).Monic",
        "proof": "exact Polynomial.monic_X_pow_add hp",
        "era": "classical", "frontier_zone": "algebra", "domain": "Algebra",
        "description": "X^n + lower degree term is monic — requires monic_X_pow_add lemma",
    },
    # --- Map properties ---
    {
        "name": "poly_map_id",
        "statement": "theorem poly_map_id (p : Polynomial ℝ) : p.map (RingHom.id ℝ) = p",
        "proof": "simp",
        "era": "classical", "frontier_zone": "algebra", "domain": "Algebra",
        "description": "Identity map preserves polynomial — requires map_id lemma",
    },
]

verified = []
failed = []
for thm in theorems:
    ok, err = verify_theorem(thm["name"], thm["statement"], thm["proof"])
    status = "PASS" if ok else f"FAIL"
    print(f"  [{status}] {thm['name']}")
    if not ok:
        print(f"          Error: {err[:120]}")
    if ok:
        verified.append(thm)
    else:
        failed.append((thm["name"], err))

print(f"\nVerified: {len(verified)}/{len(theorems)}")

if verified:
    output_path = PROJECT / "data/raw/gate3_lemma_novelty.jsonl"
    with open(output_path, "w") as f:
        for thm in verified:
            out = {k: v for k, v in thm.items()}
            f.write(json.dumps(out) + "\n")
    print(f"Wrote {len(verified)} theorems to {output_path}")
    sys.exit(0)
else:
    print("No theorems verified!")
    sys.exit(1)
