"""
Error-Guided Lemma Search — Use Lean failures as a compass.

Instead of predicting the right lemma blindly, we try a lemma,
capture Lean's error message, classify the error, and use the
error type to redirect the search toward the correct lemma.

Architecture:
  1. Start with goal-only encoder's top-10 retrieved lemmas
  2. For each lemma: try the proof, capture Lean's stdout/stderr
  3. Parse error message → classify into error type
  4. Error type determines NEXT lemma to try:
     - "could not unify" → switch tactic (apply→rewrite→exact→calc)
     - "made no progress" → try lemma's graph neighbors
     - "linarith failed" → escalate to nlinarith → ring → field_simp
     - "unknown identifier" → add import, retry
  5. Loop: max 5 error-guided retries per theorem
  6. Total Lean budget: 5 initial + 5 retries = 10 checks per theorem

ERA-SAFETY: Lean error messages are about proof engine mechanics
(unification, tactic application, namespacing). Zero physics content.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from src.explorer.proof_state import ProofState, Tactic, TacticType
from src.proof_checker.formats import ProofResult, wrap_theorem_with_proof
from src.proof_checker.lean_interface import LeanProofChecker


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

class ErrorType(Enum):
    """Lean error categories that guide search redirection."""

    COULD_NOT_UNIFY = "could_not_unify"
    """Wrong lemma TYPE — try a different tactic strategy (apply→rewrite→calc)."""

    MADE_NO_PROGRESS = "made_no_progress"
    """Lemma doesn't fire — try graph neighbors (similar lemmas, same domain)."""

    LINARITH_FAILED = "linarith_failed"
    """Not linear — escalate to nlinarith, ring, or field_simp."""

    UNKNOWN_IDENTIFIER = "unknown_identifier"
    """Lemma exists but not imported — add import and retry."""

    TYPE_MISMATCH = "type_mismatch"
    """Type error (argument count, binder mismatch) — try with different args."""

    NO_GOALS = "no_goals"
    """Tactic has no effect — goal is already solved or tactic doesn't apply."""

    SYNTAX_ERROR = "syntax_error"
    """Malformed tactic or expression."""

    UNKNOWN_ERROR = "unknown_error"
    """Unclassified error — move to next lemma."""

    SUCCESS = "success"
    """Proof succeeded."""


# Regex patterns for Lean error messages
_ERROR_PATTERNS: list[tuple[re.Pattern, ErrorType]] = [
    (
        re.compile(
            r"could\s*(not|n't)\s*unify|"
            r"application\s+type\s+mismatch|"
            r"type\s+mismatch.*application|"
            r"has\s+type.*but\s+is\s+expected\s+to\s+have\s+type|"
            r"invalid\s+rewrite\s+argument|"
            r"cannot\s+be\s+applied\s+to",
            re.IGNORECASE,
        ),
        ErrorType.COULD_NOT_UNIFY,
    ),
    (
        re.compile(
            r"made\s+no\s+progress|"
            r"tactic\s+\w+\s+made\s+no\s+progress|"
            r"no\s+progress|"
            r"did\s+not\s+change\s+the\s+goal|"
            r"unsolved\s+goals",
            re.IGNORECASE,
        ),
        ErrorType.MADE_NO_PROGRESS,
    ),
    (
        re.compile(
            r"linarith\s+failed\s+to\s+find\s+a\s+contradiction|"
            r"linarith\s+failed|"
            r"ring\s+failed|"
            r"nlinarith\s+failed",
            re.IGNORECASE,
        ),
        ErrorType.LINARITH_FAILED,
    ),
    (
        re.compile(
            r"unknown\s+identifier|"
            r"unknown\s+constant|"
            r"not\s+found|"
            r"not\s+available\s+in\s+the\s+current\s+environment",
            re.IGNORECASE,
        ),
        ErrorType.UNKNOWN_IDENTIFIER,
    ),
    (
        re.compile(
            r"function\s+expected|"
            r"incorrect\s+number\s+of\s+arguments|"
            r"too\s+many\s+arguments|"
            r"too\s+few\s+arguments|"
            r"invalid\s+field|"
            r"don't\s+know\s+how\s+to\s+synthesize",
            re.IGNORECASE,
        ),
        ErrorType.TYPE_MISMATCH,
    ),
    (
        re.compile(
            r"no\s+goals\s+to\s+be\s+solved|"
            r"no\s+goals\s+left|"
            r"tactic\s+has\s+no\s+effect",
            re.IGNORECASE,
        ),
        ErrorType.NO_GOALS,
    ),
    (
        re.compile(
            r"syntax\s+error|"
            r"unexpected\s+token|"
            r"expected\s+'\w+'",
            re.IGNORECASE,
        ),
        ErrorType.SYNTAX_ERROR,
    ),
]


def classify_error(errors: list[str]) -> ErrorType:
    """Classify Lean error messages into a search-redirecting error type.

    Args:
        errors: List of error strings from Lean proof checker.

    Returns:
        ErrorType indicating what kind of search redirection to perform.
    """
    if not errors:
        return ErrorType.SUCCESS

    combined = " ".join(errors).lower()

    for pattern, error_type in _ERROR_PATTERNS:
        if pattern.search(combined):
            return error_type

    return ErrorType.UNKNOWN_ERROR


def classify_result(result: ProofResult) -> ErrorType:
    """Classify a ProofResult into an ErrorType."""
    if result.success:
        return ErrorType.SUCCESS
    return classify_error(result.errors)


# ---------------------------------------------------------------------------
# Error-guided search redirection
# ---------------------------------------------------------------------------

# Tactic escalation chain for arithmetic failures
_ARITHMETIC_ESCALATION: list[tuple[str, TacticType]] = [
    ("linarith", TacticType.LINARITH),
    ("nlinarith", TacticType.NLINARITH),
    ("ring", TacticType.RING),
    ("field_simp", TacticType.FIELD_SIMP),
    ("simp", TacticType.SIMP),
]

# Tactic rotation for unification failures: try different application styles
_UNIFICATION_TACTICS: list[TacticType] = [
    TacticType.APPLY,
    TacticType.REWRITE,
    TacticType.EXACT,
    TacticType.CALC,
    TacticType.REFINE,
    TacticType.HAVE,
]

# Additional imports that may help
_IMPORT_ADDITIONS: list[str] = [
    "open Real",
    "open Set",
    "open Function",
    "open Nat",
    "open Int",
    "open Complex",
    "open Polynomial",
    "open Matrix",
]


@dataclass
class ErrorGuideConfig:
    """Configuration for error-guided search."""

    # Maximum number of initial lemmas to try (from encoder top-K)
    max_initial_lemmas: int = 5

    # Maximum number of error-guided retries per lemma
    max_retries: int = 5

    # Total Lean checks per theorem: initials + retries
    max_total_checks: int = 10

    # Top-K from goal-only encoder retrieval
    retrieval_top_k: int = 10

    # Number of graph neighbors to try when redirected
    max_neighbors: int = 5

    # Timeout per individual Lean check (seconds)
    check_timeout: float = 15.0

    # Maximum search time per theorem (seconds)
    max_theorem_time: float = 120.0

    # Whether to try structural tactics first (fast path)
    structural_first: bool = True

    # Number of CPU threads for PyTorch
    num_threads: int = 4


@dataclass
class _LemmaAttempt:
    """Record a single lemma-check attempt with its result."""

    lemma: str
    tactic_type: TacticType
    proof_text: str
    result: ProofResult
    error_type: ErrorType


# ---------------------------------------------------------------------------
# Error-guided search
# ---------------------------------------------------------------------------


class ErrorGuidedSearch:
    """Proof search guided by Lean error messages.

    Instead of a full best-first search tree, this uses a linear try-and-redirect
    strategy: try a lemma, capture the error, and let the error type determine
    which lemma to try next.

    Usage:
        egs = ErrorGuidedSearch(
            encoder=goal_only_encoder,
            vocab=vocab,
            proof_checker=checker,
            graph=dep_graph,
            lemma_index=lemma_index,
            lemma_index_to_name=idx_to_name,
        )
        proof, stats = egs.search(theorem_statement)
    """

    def __init__(
        self,
        encoder,  # GoalOnlyEncoder
        vocab: dict[str, int],
        proof_checker: LeanProofChecker,
        graph=None,  # DependencyGraph (networkx)
        lemma_index: dict[str, int] | None = None,
        lemma_index_to_name: dict[int, str] | None = None,
        config: ErrorGuideConfig | None = None,
    ):
        self.encoder = encoder
        self.vocab = vocab
        self.proof_checker = proof_checker
        self.graph = graph
        self.lemma_index = lemma_index or {}
        self.lemma_index_to_name = lemma_index_to_name or {}
        self.config = config or ErrorGuideConfig()

        torch.set_num_threads(self.config.num_threads)

    # ------------------------------------------------------------------
    # Main search
    # ------------------------------------------------------------------

    def search(
        self,
        theorem_statement: str,
        top_lemmas: list[tuple[str, float]] | None = None,
        verbose: bool = False,
    ) -> tuple[list[Tactic] | None, dict]:
        """Search for a proof with error-guided redirection.

        Args:
            theorem_statement: The theorem to prove (Lean 4 statement).
            top_lemmas: Pre-retrieved top lemmas with scores. If None, uses
                        self._retrieve_lemmas (must be overridden or provided).
            verbose: Print progress.

        Returns:
            (proof_steps, stats) — None if no proof found. Stats dict contains
            attempt history, error types encountered, and timing.
        """
        config = self.config
        t_start = time.time()
        stats: dict = {
            "theorem": theorem_statement[:120],
            "attempts": [],
            "total_checks": 0,
            "error_types_seen": [],
            "search_time_s": 0.0,
            "success": False,
            "proof_steps": [],
        }

        # Phase 0: Structural tactics first (fast path)
        if config.structural_first:
            result = self._try_structural_tactics(theorem_statement, stats)
            if result is not None:
                stats["success"] = True
                stats["proof_steps"] = [t.to_lean() for t in result]
                stats["search_time_s"] = time.time() - t_start
                return result, stats

        if top_lemmas is None:
            if verbose:
                print("  [ErrorGuided] No lemmas provided, skipping lemma search")
            stats["search_time_s"] = time.time() - t_start
            return None, stats

        # Phase 1-2: Try initial lemmas with error-guided retries
        tried_lemmas: set[str] = set()
        initial_count = min(config.max_initial_lemmas, len(top_lemmas))

        for i in range(initial_count):
            if stats["total_checks"] >= config.max_total_checks:
                break
            if time.time() - t_start > config.max_theorem_time:
                break

            lemma_name, lemma_score = top_lemmas[i]

            if verbose:
                print(f"  [{i+1}/{initial_count}] Trying: {lemma_name} "
                      f"(score={lemma_score:.3f})")

            result = self._try_with_error_guidance(
                theorem_statement,
                lemma_name,
                stats,
                tried_lemmas,
                verbose=verbose,
            )
            if result is not None:
                stats["success"] = True
                stats["proof_steps"] = [t.to_lean() for t in result]
                stats["search_time_s"] = time.time() - t_start
                return result, stats

        stats["search_time_s"] = time.time() - t_start
        return None, stats

    # ------------------------------------------------------------------
    # Error-guided retry loop
    # ------------------------------------------------------------------

    def _try_with_error_guidance(
        self,
        theorem_statement: str,
        starting_lemma: str,
        stats: dict,
        tried_lemmas: set[str],
        verbose: bool = False,
    ) -> list[Tactic] | None:
        """Try a lemma, capture error, redirect search.

        Retry loop: try lemma → get error → classify → redirect → retry.
        Max config.max_retries error-guided redirects per starting lemma.

        Returns proof steps if found, None otherwise.
        """
        config = self.config
        current_lemma = starting_lemma
        current_tactic = TacticType.APPLY  # Start with apply
        retry_count = 0

        while retry_count <= config.max_retries:
            if stats["total_checks"] >= config.max_total_checks:
                return None

            # Build and check the proof
            tactics, result = self._check_single_lemma(
                theorem_statement, current_lemma, current_tactic
            )

            stats["total_checks"] += 1

            error_type = classify_result(result)

            attempt_record = _LemmaAttempt(
                lemma=current_lemma,
                tactic_type=current_tactic,
                proof_text=tactics[0].to_lean() if tactics else "?",
                result=result,
                error_type=error_type,
            )
            stats["attempts"].append({
                "lemma": current_lemma,
                "tactic": current_tactic.value,
                "success": result.success,
                "error_type": error_type.value,
                "error_preview": result.errors[0][:120] if result.errors else "",
            })
            stats["error_types_seen"].append(error_type.value)

            if result.success:
                if verbose:
                    print(f"    ✓ PROVED with {current_lemma} [{current_tactic.value}] "
                          f"after {retry_count} retries")
                return tactics

            if verbose:
                errors = result.errors
                err_snippet = errors[0][:100] if errors else "?"
                print(f"    ✗ {error_type.value}: {err_snippet}")

            # Redirect based on error type
            next_action = self._redirect(
                error_type=error_type,
                current_lemma=current_lemma,
                current_tactic=current_tactic,
                tried_lemmas=tried_lemmas,
            )

            if next_action is None:
                return None

            current_lemma, current_tactic = next_action

            if current_lemma in tried_lemmas and retry_count > 0:
                if verbose:
                    print(f"    → Already tried {current_lemma}, giving up")
                return None

            tried_lemmas.add(current_lemma)
            retry_count += 1

        return None

    # ------------------------------------------------------------------
    # Single lemma check
    # ------------------------------------------------------------------

    def _check_single_lemma(
        self,
        theorem_statement: str,
        lemma_name: str,
        tactic_type: TacticType,
    ) -> tuple[list[Tactic], ProofResult]:
        """Build a one-step proof with the given lemma and check it with Lean.

        Returns (tactics, result).
        """
        tactic = Tactic(tactic_type, lemma=lemma_name)
        proof_body = tactic.to_lean()
        code = wrap_theorem_with_proof(theorem_statement, proof_body)
        result = self.proof_checker.check(code, timeout=self.config.check_timeout)
        return [tactic], result

    # ------------------------------------------------------------------
    # Error redirection
    # ------------------------------------------------------------------

    def _redirect(
        self,
        error_type: ErrorType,
        current_lemma: str,
        current_tactic: TacticType,
        tried_lemmas: set[str],
    ) -> tuple[str, TacticType] | None:
        """Determine next lemma and tactic based on error type.

        Returns (next_lemma, next_tactic) or None if no redirection possible.
        """
        if error_type == ErrorType.COULD_NOT_UNIFY or error_type == ErrorType.TYPE_MISMATCH:
            return self._redirect_unification(current_lemma, current_tactic)

        elif error_type == ErrorType.MADE_NO_PROGRESS:
            return self._redirect_neighbor(current_lemma, tried_lemmas)

        elif error_type == ErrorType.LINARITH_FAILED:
            return self._redirect_arithmetic_escalation(current_lemma, tried_lemmas)

        elif error_type == ErrorType.UNKNOWN_IDENTIFIER:
            # Try with import addition — for now, try the lemma as `exact` instead
            # (some identifiers resolve under different tactic contexts)
            if current_tactic == TacticType.APPLY:
                return current_lemma, TacticType.EXACT
            elif current_tactic == TacticType.EXACT:
                return current_lemma, TacticType.REWRITE
            else:
                return self._redirect_neighbor(current_lemma, tried_lemmas)

        elif error_type == ErrorType.NO_GOALS:
            # Goal already solved or tactic doesn't apply — try exact
            if current_tactic != TacticType.EXACT:
                return current_lemma, TacticType.EXACT
            return None

        elif error_type == ErrorType.SYNTAX_ERROR:
            # Try as rewrite or exact instead
            if current_tactic == TacticType.APPLY:
                return current_lemma, TacticType.EXACT
            return None

        else:
            # Unknown error: try a neighbor
            return self._redirect_neighbor(current_lemma, tried_lemmas)

    def _redirect_unification(
        self, lemma: str, current_tactic: TacticType
    ) -> tuple[str, TacticType] | None:
        """Unification failed → rotate through tactic types."""
        tactics = _UNIFICATION_TACTICS
        try:
            idx = tactics.index(current_tactic)
            next_idx = (idx + 1) % len(tactics)
            return lemma, tactics[next_idx]
        except ValueError:
            return lemma, TacticType.REWRITE

    def _redirect_neighbor(
        self, lemma: str, tried_lemmas: set[str]
    ) -> tuple[str, TacticType] | None:
        """Lemma made no progress → try graph neighbors."""
        neighbors = self._get_lemma_neighbors(lemma)
        for neighbor in neighbors[:self.config.max_neighbors]:
            if neighbor not in tried_lemmas:
                return neighbor, TacticType.APPLY
        return None

    def _redirect_arithmetic_escalation(
        self, lemma: str, tried_lemmas: set[str]
    ) -> tuple[str, TacticType] | None:
        """Linarith failed → escalate up the arithmetic chain."""
        # Map the current tactic to the escalation chain
        current_tactic_value = None
        for i, (_, tt) in enumerate(_ARITHMETIC_ESCALATION):
            if lemma in ("linarith", "nlinarith", "ring", "field_simp", "simp"):
                if tt == TacticType.LINARITH and "linarith" in lemma:
                    current_tactic_value = i
                    break
                elif tt == TacticType.NLINARITH and "nlinarith" in lemma:
                    current_tactic_value = i
                    break
                elif tt == TacticType.RING and "ring" in lemma:
                    current_tactic_value = i
                    break
                elif tt == TacticType.FIELD_SIMP and "field_simp" in lemma:
                    current_tactic_value = i
                    break
                elif tt == TacticType.SIMP and "simp" in lemma:
                    current_tactic_value = i
                    break

        if current_tactic_value is not None:
            next_idx = current_tactic_value + 1
        else:
            # Start from beginning
            next_idx = 1  # Skip linarith, go to nlinarith

        if next_idx < len(_ARITHMETIC_ESCALATION):
            name, tt = _ARITHMETIC_ESCALATION[next_idx]
            return name, tt

        # All arithmetic tactics failed — try neighbor of the original problem
        return self._redirect_neighbor(lemma, tried_lemmas)

    # ------------------------------------------------------------------
    # Structural tactics (fast path)
    # ------------------------------------------------------------------

    def _try_structural_tactics(
        self, theorem_statement: str, stats: dict
    ) -> list[Tactic] | None:
        """Try built-in tactics (ring, field_simp, linarith) before lemma search.

        Enhanced with error-guided hypothesis usage:
        1. Parse equality hypotheses (h : x = expr) from theorem statement
        2. Try rw [hyp] before ring/nlinarith
        3. If nlinarith fails, try nlinarith [hyp1, hyp2] with all hypotheses
        4. If linarith fails, escalate to nlinarith
        """
        goal_lower = theorem_statement.lower()

        # Parse equality/inequality hypotheses from the theorem statement
        # Pattern: (h : expr = expr) or (h : expr ≥ expr) etc.
        import re

        eq_hyps: list[str] = []  # Names of equality hypotheses
        ineq_hyps: list[str] = []  # Names of inequality hypotheses
        all_hyps: list[str] = []  # All hypothesis names

        # Match hypothesis binders: (name : type)
        hyp_pattern = re.compile(
            r'\(\s*(\w+)\s*:\s*([^)]+?)\s*\)'
        )
        for m in hyp_pattern.finditer(theorem_statement):
            name = m.group(1)
            ty = m.group(2)
            all_hyps.append(name)
            if "=" in ty and "≠" not in ty and "→" not in ty and "∀" not in ty:
                eq_hyps.append(name)
            if any(op in ty for op in ("≤", "≥", "<", ">")):
                ineq_hyps.append(name)

        # Determine which structural tactics to try based on goal content
        has_arith = any(op in theorem_statement
                       for op in ("+", "*", "-", "^", "="))
        has_div = "/" in theorem_statement or "⁻¹" in theorem_statement
        has_ineq = any(op in theorem_statement
                       for op in ("≤", "≥", "<", ">"))
        has_implication = "→" in theorem_statement or "∀" in theorem_statement

        def _check_tactic(tactic: Tactic, label: str) -> bool:
            if stats["total_checks"] >= self.config.max_total_checks:
                return False
            proof_body = tactic.to_lean()
            code = wrap_theorem_with_proof(theorem_statement, proof_body)
            result = self.proof_checker.check(code, timeout=self.config.check_timeout)
            stats["total_checks"] += 1
            error_type = classify_result(result)
            stats["attempts"].append({
                "lemma": label,
                "tactic": tactic.tactic_type.value,
                "success": result.success,
                "error_type": error_type.value,
                "error_preview": result.errors[0][:120] if result.errors else "",
            })
            return result.success

        # Phase 1: Try rewriting with equality hypotheses, then arithmetic
        if eq_hyps and has_arith and not has_implication:
            for hyp_name in eq_hyps[:3]:
                rw_tactic = Tactic(TacticType.REWRITE, hypothesis=hyp_name)
                if _check_tactic(rw_tactic, f"rw[{hyp_name}]"):
                    return [rw_tactic]

                # rw then ring
                rw_ring = Tactic(TacticType.REWRITE, hypothesis=hyp_name)
                # We can't do multi-step in single check, so skip
                # But we CAN try ring after rw by wrapping them
                # Actually, Lean accepts multi-step proofs. Let's try:
                try:
                    combo_body = f"rw [{hyp_name}]; ring"
                    code = wrap_theorem_with_proof(theorem_statement, combo_body)
                    if stats["total_checks"] < self.config.max_total_checks:
                        result = self.proof_checker.check(code, timeout=self.config.check_timeout)
                        stats["total_checks"] += 1
                        error_type = classify_result(result)
                        stats["attempts"].append({
                            "lemma": f"rw[{hyp_name}];ring",
                            "tactic": "combo",
                            "success": result.success,
                            "error_type": error_type.value,
                            "error_preview": result.errors[0][:120] if result.errors else "",
                        })
                        if result.success:
                            return [
                                Tactic(TacticType.REWRITE, hypothesis=hyp_name),
                                Tactic(TacticType.RING),
                            ]
                except Exception:
                    pass

                # rw then nlinarith
                try:
                    combo_body = f"rw [{hyp_name}]; nlinarith"
                    code = wrap_theorem_with_proof(theorem_statement, combo_body)
                    if stats["total_checks"] < self.config.max_total_checks:
                        result = self.proof_checker.check(code, timeout=self.config.check_timeout)
                        stats["total_checks"] += 1
                        error_type = classify_result(result)
                        stats["attempts"].append({
                            "lemma": f"rw[{hyp_name}];nlinarith",
                            "tactic": "combo",
                            "success": result.success,
                            "error_type": error_type.value,
                            "error_preview": result.errors[0][:120] if result.errors else "",
                        })
                        if result.success:
                            return [
                                Tactic(TacticType.REWRITE, hypothesis=hyp_name),
                                Tactic(TacticType.NLINARITH),
                            ]
                except Exception:
                    pass

                # rw then field_simp then ring
                if has_div:
                    try:
                        combo_body = f"rw [{hyp_name}]; field_simp; ring"
                        # Only if has_div
                        code = wrap_theorem_with_proof(theorem_statement, combo_body)
                        if stats["total_checks"] < self.config.max_total_checks:
                            result = self.proof_checker.check(code, timeout=self.config.check_timeout)
                            stats["total_checks"] += 1
                            error_type = classify_result(result)
                            stats["attempts"].append({
                                "lemma": f"rw[{hyp_name}];field_simp;ring",
                                "tactic": "combo",
                                "success": result.success,
                                "error_type": error_type.value,
                                "error_preview": result.errors[0][:120] if result.errors else "",
                            })
                            if result.success:
                                return [
                                    Tactic(TacticType.REWRITE, hypothesis=hyp_name),
                                    Tactic(TacticType.FIELD_SIMP),
                                    Tactic(TacticType.RING),
                                ]
                    except Exception:
                        pass

        # Phase 2: Try structural tactics with hypothesis arguments
        if has_arith and not has_implication:
            # Try nlinarith with all hypotheses
            if all_hyps:
                nl_hyp_tac = Tactic(TacticType.NLINARITH, args=all_hyps[:5])
                if _check_tactic(nl_hyp_tac, f"nlinarith[{','.join(all_hyps[:3])}]"):
                    return [nl_hyp_tac]

            # Try nlinarith bare
            if _check_tactic(Tactic(TacticType.RING), "ring"):
                return [Tactic(TacticType.RING)]
            if _check_tactic(Tactic(TacticType.NLINARITH), "nlinarith"):
                return [Tactic(TacticType.NLINARITH)]

        if has_div and not has_implication:
            if _check_tactic(Tactic(TacticType.FIELD_SIMP), "field_simp"):
                return [Tactic(TacticType.FIELD_SIMP)]

        if has_ineq and not has_implication:
            if all_hyps:
                l_hyp_tac = Tactic(TacticType.LINARITH, args=all_hyps[:5])
                if _check_tactic(l_hyp_tac, f"linarith[{','.join(all_hyps[:3])}]"):
                    return [l_hyp_tac]
            if _check_tactic(Tactic(TacticType.LINARITH), "linarith"):
                return [Tactic(TacticType.LINARITH)]
            # Escalate linarith→nlinarith
            if all_hyps:
                nl_hyp_tac2 = Tactic(TacticType.NLINARITH, args=all_hyps[:5])
                if _check_tactic(nl_hyp_tac2, f"nlinarith[{','.join(all_hyps[:3])}]"):
                    return [nl_hyp_tac2]
            if _check_tactic(Tactic(TacticType.NLINARITH), "nlinarith"):
                return [Tactic(TacticType.NLINARITH)]

        if _check_tactic(Tactic(TacticType.SIMP), "simp"):
            return [Tactic(TacticType.SIMP)]

        return None

    # ------------------------------------------------------------------
    # Graph neighbor lookup
    # ------------------------------------------------------------------

    def _get_lemma_neighbors(self, lemma_name: str) -> list[str]:
        """Get lemma neighbors from the dependency graph.

        Returns lemmas that are one hop away in the dependency graph
        (both dependencies and dependents), sorted by relevance.
        """
        if self.graph is None or not self.lemma_index:
            return []

        idx = self.lemma_index.get(lemma_name)
        if idx is None:
            return []

        neighbors: list[str] = []

        try:
            # Get graph neighborhood
            neighborhood = self.graph.get_neighborhood(
                str(idx), radius=1, direction="both"
            )
            for nid in neighborhood:
                name = self.lemma_index_to_name.get(int(nid))
                if name and name != lemma_name:
                    neighbors.append(name)
                if len(neighbors) >= self.config.max_neighbors * 2:
                    break
        except Exception:
            pass

        # Also try direct networkx neighbors
        try:
            if hasattr(self.graph, 'graph') and self.graph.graph is not None:
                g = self.graph.graph
                node = str(idx)
                if g.has_node(node):
                    for pred in list(g.predecessors(node))[:5]:
                        name = self.lemma_index_to_name.get(int(pred))
                        if name and name != lemma_name and name not in neighbors:
                            neighbors.append(name)
                    for succ in list(g.successors(node))[:5]:
                        name = self.lemma_index_to_name.get(int(succ))
                        if name and name != lemma_name and name not in neighbors:
                            neighbors.append(name)
        except Exception:
            pass

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for n in neighbors:
            if n not in seen:
                seen.add(n)
                unique.append(n)
                if len(unique) >= self.config.max_neighbors:
                    break

        return unique

    # ------------------------------------------------------------------
    # Lemma retrieval (to be called by external eval script)
    # ------------------------------------------------------------------

    def retrieve_top_lemmas(
        self,
        theorem_statement: str,
        goal_embeddings: torch.Tensor,
        lemma_names: list[str],
        k: int | None = None,
    ) -> list[tuple[str, float]]:
        """Retrieve top-K lemmas using cosine similarity with goal embedding.

        Uses a pre-computed goal embedding and an index of lemma
        embeddings for fast retrieval.

        Args:
            theorem_statement: The theorem text.
            goal_embeddings: Pre-computed goal embedding [hidden_dim].
            lemma_names: All possible lemma names (same order as lemma_embeddings).
            k: Number of top lemmas to return.

        Returns:
            List of (lemma_name, score) sorted by score descending.
        """
        k = k or self.config.retrieval_top_k

        # Encode goal
        goal_emb = self._encode_goal(theorem_statement)

        # Retrieve lemmas will be done by the caller who has the pre-computed
        # index embeddings. This method just does cosine similarity.
        # Actually, let me compute it here using goal_embeddings from caller.
        if goal_emb is not None:
            goal_emb = F.normalize(goal_emb, dim=-1)
            scores = (goal_emb.unsqueeze(0) @ goal_embeddings.T).squeeze(0)  # type: ignore
            k = min(k, len(lemma_names))
            top_scores, top_indices = torch.topk(scores, k)
            return [(lemma_names[idx.item()], top_scores[i].item())
                    for i, idx in enumerate(top_indices)]
        return []

    def _encode_goal(self, goal_text: str) -> torch.Tensor | None:
        """Encode a goal using the goal-only encoder."""
        try:
            from src.retrieval.goal_only_encoder import _tokenize_batch
            device = next(self.encoder.parameters()).device
            with torch.no_grad():
                batch_ids = _tokenize_batch([goal_text], self.vocab, 128).to(device)
                emb = self.encoder(batch_ids)
                return emb.squeeze(0)
        except Exception:
            return None
