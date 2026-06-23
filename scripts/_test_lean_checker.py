#!/usr/bin/env python3
"""Quick test of Lean proof checker."""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.proof_checker.lean_interface import LeanProofChecker

checker = LeanProofChecker(timeout=15.0)

# Test 1: trivial proof
test1 = """theorem test_trivial (a b : ℝ) (h : a = b) : a = b := by
  exact h"""
r1 = checker.check(test1)
print(f"Test 1 (trivial): success={r1.success}, errors={r1.errors}")

# Test 2: ring proof
test2 = """theorem test_ring (a b : ℝ) : (a + b)^2 = a^2 + 2*a*b + b^2 := by
  ring"""
r2 = checker.check(test2)
print(f"Test 2 (ring): success={r2.success}, errors={r2.errors}")

# Test 3: field_simp
test3 = """theorem test_field (a : ℝ) (h : a ≠ 0) : a / a = 1 := by
  field_simp [h]"""
r3 = checker.check(test3)
print(f"Test 3 (field_simp): success={r3.success}, errors={r3.errors}")

# Test 4: nlinarith
test4 = """theorem test_nlinarith (a : ℝ) : a^2 >= 0 := by
  nlinarith"""
r4 = checker.check(test4)
print(f"Test 4 (nlinarith): success={r4.success}, errors={r4.errors}")
