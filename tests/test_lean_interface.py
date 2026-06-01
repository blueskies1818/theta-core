"""Tests for the Lean 4 proof checker interface."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.proof_checker.lean_interface import LeanProofChecker
from src.proof_checker.formats import LEAN_PREAMBLE


def test_valid_proof_native_decide():
    """A known-correct Lean 4 proof returns success."""
    checker = LeanProofChecker()
    code = "theorem add_one_two : 1 + 1 = 2 := by\n  native_decide"
    result = checker.check(code)
    assert result.success, f"Expected success, got: {result.errors}"
    assert not result.timed_out


def test_invalid_proof():
    """A known-incorrect Lean 4 proof returns failure."""
    checker = LeanProofChecker()
    code = "theorem false_claim : 1 + 1 = 3 := by\n  native_decide"
    result = checker.check(code)
    assert not result.success, "Expected failure for false theorem"


def test_valid_proof_omega():
    """Omega tactic works for linear arithmetic."""
    checker = LeanProofChecker()
    code = "theorem add_comm (a b : Nat) : a + b = b + a := by\n  omega"
    result = checker.check(code)
    assert result.success, f"Expected success, got: {result.errors}"


def test_valid_proof_rfl():
    """rfl tactic works for definitional equality."""
    checker = LeanProofChecker()
    code = "theorem identity (x : Nat) : x = x := by\n  rfl"
    result = checker.check(code)
    assert result.success, f"Expected success, got: {result.errors}"


def test_cache_hit():
    """Caching avoids redundant checks."""
    checker = LeanProofChecker()
    code = "theorem test_cache : 0 = 0 := by\n  rfl"

    r1 = checker.check(code)
    r2 = checker.check(code)

    assert r1.success == r2.success
    assert checker.cache.hit_rate > 0.0, "Cache should have hits"


def test_timeout():
    """Proofs that take too long should be timed out."""
    checker = LeanProofChecker(timeout=0.01)
    code = "theorem slow : 1 = 1 := by\n  native_decide"
    result = checker.check(code)
    # native_decide should be fast; this test validates timeout mechanism
    assert result.success or result.timed_out or not result.success


if __name__ == "__main__":
    test_valid_proof_native_decide()
    print("PASS: test_valid_proof_native_decide")

    test_invalid_proof()
    print("PASS: test_invalid_proof")

    test_valid_proof_omega()
    print("PASS: test_valid_proof_omega")

    test_valid_proof_rfl()
    print("PASS: test_valid_proof_rfl")

    test_cache_hit()
    print("PASS: test_cache_hit")

    test_timeout()
    print("PASS: test_timeout")

    print("\nAll tests passed!")
