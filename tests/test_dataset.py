"""Tests for data pipeline components."""

import sys
import json
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.mathlib_extractor import (
    extract_theorems_from_file,
    save_theorems_jsonl,
    load_theorems_jsonl,
    format_theorem_proof,
)

# A minimal Lean 4 file with known theorems
SAMPLE_LEAN_FILE = """
import Mathlib

theorem add_comm (a b : Nat) : a + b = b + a := by
  omega

theorem simple : 1 + 1 = 2 := by
  native_decide

lemma zero_add (n : Nat) : 0 + n = n := by
  omega

example : 2 + 2 = 4 := by
  native_decide
"""


def test_extract_theorems():
    """Extract theorems from a sample Lean file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".lean", delete=False) as f:
        f.write(SAMPLE_LEAN_FILE)
        tmp_path = f.name

    try:
        theorems = extract_theorems_from_file(Path(tmp_path))
        # Should find: add_comm, simple, zero_add, and one example
        assert len(theorems) >= 3, f"Expected >=3 theorems, got {len(theorems)}"

        # Check theorem structure
        for t in theorems:
            assert "statement" in t
            assert "proof" in t
            assert "name" in t
    finally:
        Path(tmp_path).unlink()


def test_save_and_load_jsonl():
    """Round-trip theorems through JSONL."""
    theorems = [
        {"name": "test1", "statement": "1+1=2", "proof": "rfl"},
        {"name": "test2", "statement": "x=x", "proof": "rfl"},
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        tmp_path = f.name

    try:
        save_theorems_jsonl(theorems, Path(tmp_path))
        loaded = load_theorems_jsonl(Path(tmp_path))
        assert len(loaded) == len(theorems)
        assert loaded[0]["name"] == "test1"
    finally:
        Path(tmp_path).unlink()


def test_format_theorem_proof():
    """Format theorem dict as training string."""
    theorem = {"name": "test", "statement": "1+1=2", "proof": "rfl"}
    formatted = format_theorem_proof(theorem)
    assert "Theorem:" in formatted
    assert "Proof:" in formatted
    assert "1+1=2" in formatted
    assert "rfl" in formatted


if __name__ == "__main__":
    test_extract_theorems()
    print("PASS: test_extract_theorems")

    test_save_and_load_jsonl()
    print("PASS: test_save_and_load_jsonl")

    test_format_theorem_proof()
    print("PASS: test_format_theorem_proof")

    print("\nAll tests passed!")
