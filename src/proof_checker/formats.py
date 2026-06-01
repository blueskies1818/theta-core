"""Lean 4 code formatting and template wrapping.

The model generates proof bodies without import statements.
The checker wraps them with the necessary preambles before verification.
"""

from dataclasses import dataclass
from typing import Optional

# Light preamble using only core Lean 4 (no Mathlib dependency).
# Used for testing and simple theorem proving.
LEAN_PREAMBLE_LIGHT = """open Nat
open Int
"""

# Full preamble with Mathlib4 imports.
# Used when mathlib4 is available for richer mathematical domains.
LEAN_PREAMBLE_MATHLIB = """import Mathlib
open Real
open Set
open Filter
open Function
open Nat
"""

# Default preamble for Phase 1. Switch to LEAN_PREAMBLE_MATHLIB
# after cloning and building mathlib4.
LEAN_PREAMBLE = LEAN_PREAMBLE_LIGHT


@dataclass
class ProofResult:
    """Result of a single proof check."""

    success: bool
    errors: list[str]
    num_tokens: int
    timed_out: bool = False
    check_time_ms: float = 0.0


def wrap_lean_code(generated_code: str, include_preamble: bool = True) -> str:
    """Wrap model output with necessary imports for checking.

    Strips any existing import/open statements from the generated code
    and prepends the standard preamble.
    """
    code = _strip_existing_imports(generated_code)
    if include_preamble:
        return LEAN_PREAMBLE + "\n" + code
    return code


def wrap_theorem_with_proof(theorem_statement: str, proof_body: str) -> str:
    """Combine a theorem statement and proof body into a checkable Lean block.

    The proof_body should be the tactic block content (after ':= by').
    """
    statement = theorem_statement.strip()
    proof = proof_body.strip()

    if not statement.endswith(":="):
        if ":" in statement:
            statement = statement.rstrip() + " := by"
        else:
            statement = statement.rstrip() + " := by"
    elif not statement.endswith("by") and statement.endswith(":="):
        statement += " by"

    return f"{statement}\n  {proof}"


def extract_proof_body(full_generation: str) -> str:
    """Extract the proof part from a model-generated completion.

    Returns everything after ':= by' or the full text if no marker found.
    """
    markers = [" := by\n", " := by ", ":=\nby\n", ":= by\n"]
    for marker in markers:
        if marker in full_generation:
            return full_generation.split(marker, 1)[1].strip()
    return full_generation.strip()


def parse_lean_error(error_text: str) -> list[str]:
    """Parse Lean error output into a list of individual error messages."""
    if not error_text:
        return []

    errors = []
    for line in error_text.split("\n"):
        line = line.strip()
        if line and ("error" in line.lower() or "unknown" in line.lower()):
            errors.append(line)

    if not errors:
        errors = [error_text.strip()[:500]]

    return errors


def _strip_existing_imports(code: str) -> str:
    """Remove existing import/open lines from generated code."""
    lines = code.split("\n")
    filtered = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("open "):
            continue
        filtered.append(line)
    return "\n".join(filtered)
