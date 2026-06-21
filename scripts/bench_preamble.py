#!/usr/bin/env python3
"""Benchmark lighter preamble for faster lean checks."""
import sys, time
sys.path.insert(0, '/home/blueman1818/Projects/theta-core')

from src.proof_checker.lean_interface import _capture_lake_env
from src.proof_checker.formats import ProofResult, parse_lean_error, wrap_lean_code

# Lighter preamble
LIGHT_PREAMBLE = """import Mathlib.Tactic
open Real
open Set
open Nat
"""

FULL_PREAMBLE = """import Mathlib
open Real
open Set
open Filter
open Function
open Nat
"""

import subprocess
from pathlib import Path

project_dir = Path('/home/blueman1818/Projects/theta-core/proof_checker_env')
lake_env = _capture_lake_env(project_dir)

test_cases = [
    ('ring', 'example : (a + b)^2 = a^2 + 2*a*b + b^2 := by ring'),
    ('field_simp', 'example (a b : ℚ) (h : b ≠ 0) : a / b + 1 = (a + b) / b := by field_simp [h]; ring'),
    ('simp', 'example : 0 + x = x := by simp'),
    ('apply', 'example (h : a = b) : b = a := by apply Eq.symm; exact h'),
    ('intro', 'example : a = a := by intro h; exact rfl'),
]

def check_with_preamble(preamble, name, code, env):
    wrapped = wrap_lean_code(code, preamble=preamble)
    t0 = time.time()
    result = subprocess.run(
        ['lean', '--stdin'],
        input=wrapped,
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(project_dir),
        env=env,
    )
    elapsed = time.time() - t0
    success = result.returncode == 0
    err = result.stderr[:100] if not success else ''
    print(f"  {name:12s} success={success} time={elapsed:.2f}s {err}")

print("=== Full preamble (import Mathlib) ===")
for name, code in test_cases:
    check_with_preamble(FULL_PREAMBLE, name, code, lake_env)

print("\n=== Light preamble (import Mathlib.Tactic) ===")
for name, code in test_cases:
    check_with_preamble(LIGHT_PREAMBLE, name, code, lake_env)
