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
# Uses Mathlib.Tactic instead of full Mathlib for 2x faster check times
# (1.1s vs 2.1s per unique check).  Covers ring, field_simp, simp, linarith,
# nlinarith, positivity, norm_num, rw, apply, exact, intro, calc, and more.
LEAN_PREAMBLE_MATHLIB = """import Mathlib.Tactic
open Real
open Set
open Function
open Nat
"""

# Default preamble. Uses Mathlib4 now that the Lake project is built (Phase 1.1).
# Fall back to LEAN_PREAMBLE_LIGHT when running without a Lake project (bare lean).
LEAN_PREAMBLE = LEAN_PREAMBLE_MATHLIB


@dataclass
class ProofResult:
    """Result of a single proof check."""

    success: bool
    errors: list[str]
    num_tokens: int
    timed_out: bool = False
    check_time_ms: float = 0.0


def wrap_lean_code(
    generated_code: str,
    include_preamble: bool = True,
    preamble: str | None = None,
) -> str:
    """Wrap model output with necessary imports for checking.

    Strips any existing import/open statements from the generated code
    and prepends the standard preamble.

    Args:
        generated_code: The raw model output (theorem + proof).
        include_preamble: If False, no preamble is prepended.
        preamble: Override the default preamble. If None, uses LEAN_PREAMBLE.
    """
    code = _strip_existing_imports(generated_code)
    if include_preamble:
        pre = preamble if preamble is not None else LEAN_PREAMBLE
        return pre + "\n" + code
    return code


def wrap_theorem_with_proof(theorem_statement: str, proof_body: str) -> str:
    """Combine a theorem statement and proof body into a checkable Lean block.

    Uses ':=' for term proofs and ':= by' for tactic proofs.
    Lean 4 accepts both:
        example : x = x := rfl              (term)
        example : x = x := by rfl           (tactic, single-line)
        example : x = x := by               (tactic, multi-line)
          intro h; exact h

    Named declarations (lemma/theorem) are converted to 'example' to avoid
    conflicts with theorems already in the Mathlib environment.
    """
    import re

    statement = theorem_statement.strip()
    proof = proof_body.strip()

    if not proof:
        return f"{statement} := sorry"

    # Strip existing proof body (:= ...) from the statement.
    # Some statements end with := proof_term already.
    statement = _strip_existing_proof(statement)

    # Convert "lemma name ..." / "theorem name ..." → "example ..."
    statement = re.sub(
        r'^(lemma|theorem)\s+\S+\s+', 'example ',
        statement, count=1,
    )

    # Ensure the statement ends with ':=' for proof delimiter
    if not statement.endswith(":="):
        if statement.rstrip().endswith(" by"):
            statement = statement.rstrip()[:-3].rstrip()
        if ":" in statement and not statement.rstrip().endswith(":="):
            statement = statement.rstrip() + " :="
        else:
            statement = statement.rstrip() + " :="

    # Detect tactic-style proofs
    is_tactic = _is_tactic_proof(proof)

    if is_tactic:
        if proof.startswith("by "):
            return f"{statement} {proof}"
        elif "\n" in proof:
            # Multi-line: indent each line under `by`.
            # Strip existing indentation first (proof may already be indented
            # from ProofState._render_proof).
            lines = [line.strip() for line in proof.split("\n") if line.strip()]
            indented = "\n".join(f"  {line}" for line in lines)
            return f"{statement} by\n{indented}"
        else:
            return f"{statement} by {proof}"
    else:
        # Term proofs use :=
        if "\n" in proof:
            return f"{statement}\n  {proof}"
        else:
            return f"{statement} {proof}"


def _strip_existing_proof(statement: str) -> str:
    """Remove any existing proof (:= term or := by ...) from a statement.

    Some theorem statements in the JSONL already include the proof, e.g.:
        lemma foo : x = y := rfl
        lemma bar : P → Q := by intro h; exact h

    Returns the statement up to the final ':' before any existing proof.
    """
    import re

    # Find the last ':=' that introduces a proof (not binder type annotation).
    # Strategy: find ':=' not inside parentheses/brackets/braces.
    depth = 0
    last_colon_eq = -1
    i = 0
    while i < len(statement) - 1:
        ch = statement[i]
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == ":" and statement[i + 1] == "=" and depth == 0:
            last_colon_eq = i
            break
        i += 1

    if last_colon_eq >= 0:
        statement = statement[:last_colon_eq].rstrip()

    return statement


def _is_tactic_proof(proof: str) -> bool:
    """Detect if a proof body uses tactic style (needs ':= by' wrapper).

    Tactic proofs start with keywords like apply, exact, intro, rw, simp.
    Term proofs are expressions like `rfl`, `add_comm a b`, `h ▸ h2`.
    """
    _tactic_keywords = {
        "apply", "exact", "refine", "intro", "intros", "rcases", "rintro",
        "rw", "rwa", "erw", "simp", "simpa", "simp_rw", "dsimp",
        "cases", "case", "induction",
        "constructor", "left", "right", "split",
        "have", "let", "show", "suffices",
        "calc", "convert", "gcongr",
        "by_contra", "exfalso", "push_neg",
        "obtain", "set", "choose",
        "positivity", "linarith", "nlinarith", "omega", "norm_num", "norm_cast",
        "field_simp", "ring", "ring_nf",
        "native_decide", "dec_trivial",
        "repeat", "try", "all_goals", "any_goals",
        "filter_upwards", "specialize", "generalize",
        "apply_rules", "solve_by_elim",
        "conv", "conv_lhs", "conv_rhs",
        "infer_instance", "assumption",
        "done", "skip",
    }
    first_word = proof.split()[0] if proof else ""
    first_word = first_word.rstrip(":")

    # Single-word terms like "rfl", "trivial", "sorry"
    if first_word in {"rfl", "trivial", "sorry"}:
        return False

    # Multi-line: check if most non-empty lines start with tactic keywords
    if "\n" in proof:
        lines = [l.strip() for l in proof.split("\n") if l.strip()]
        if not lines:
            return False
        tactic_count = sum(
            1 for l in lines
            if l.split()[0].rstrip(":") in _tactic_keywords
        )
        if tactic_count >= len(lines) * 0.5:
            return True
        # Bullet points indicate tactic proofs
        if any(l.strip().startswith(("·", "•")) for l in lines):
            return True

    return first_word in _tactic_keywords


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
