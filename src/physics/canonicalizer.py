"""Expression canonicalizer — prefers physically-sensible expression forms.

Learns token-level patterns from pre-1905 textbook invariants via
bigram overlap scoring.  Era-gated: training uses only pre-1905
canonical forms; evaluation applies to any era.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


def _tokenize(expr: str) -> list[str]:
    """Tokenize an expression into a sequence of tokens.

    Examples:
        "m*g*h + 0.5*m*v^2" → ["m","*","g","*","h","+","0.5","*","m","*","v","^","2"]
        "(c*t)^2 - x^2" → ["(","c","*","t",")","^","2","-","x","^","2"]
    """
    tokens = []
    i = 0
    s = expr.replace(" ", "")
    while i < len(s):
        # Multi-char tokens
        if s[i:i+3] == "^-2":
            tokens.append("^-2")
            i += 3
            continue
        if s[i:i+2] == "^-":
            tokens.append("^")
            tokens.append("-")
            i += 1
            continue
        # Numbers (including decimals)
        if s[i].isdigit() or (s[i] == "." and i+1 < len(s) and s[i+1].isdigit()):
            j = i
            while j < len(s) and (s[j].isdigit() or s[j] == "."):
                j += 1
            tokens.append(s[i:j])
            i = j
            continue
        # Variable names (multi-char)
        if s[i].isalpha() or s[i] == "_":
            j = i
            while j < len(s) and (s[j].isalnum() or s[j] == "_"):
                j += 1
            tokens.append(s[i:j])
            i = j
            continue
        # Single char operators/parens
        tokens.append(s[i])
        i += 1
    return tokens


def _bigrams(tokens: list[str]) -> set[tuple[str, str]]:
    """Extract adjacent bigrams from token sequence."""
    return {(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)}


class ExpressionCanonicalizer:
    """Scores expressions by similarity to canonical physics formula patterns.

    Trained on pre-1905 invariants via token bigram overlap.
    """

    def __init__(self, canonical_expressions: list[str]) -> None:
        # Build bigram frequency map from all canonical expressions
        self._bigram_freq: dict[tuple[str, str], int] = {}
        self._token_freq: dict[str, int] = {}
        self._total_bigrams = 0

        for expr in canonical_expressions:
            tokens = _tokenize(expr)
            for t in tokens:
                self._token_freq[t] = self._token_freq.get(t, 0) + 1
            for bg in _bigrams(tokens):
                self._bigram_freq[bg] = self._bigram_freq.get(bg, 0) + 1
                self._total_bigrams += 1

        # Pre-compute common physical patterns from token frequencies
        self._energy_tokens = {"m", "g", "h", "v", "E", "0.5", "k", "x"}
        self._potential_first = self._compute_ordering_preference()

    def _compute_ordering_preference(self) -> dict[str, float]:
        """Compute preferred position for key tokens based on bigram order."""
        # From training data, learn: m should come before g, E before lambda, etc.
        pref: dict[str, float] = {}
        for (a, b), count in self._bigram_freq.items():
            if a in self._energy_tokens and b in self._energy_tokens:
                key = f"{a}<{b}"
                pref[key] = pref.get(key, 0) + count
        return pref

    def score(self, expression: str) -> float:
        """Score an expression by similarity to canonical patterns.

        Returns 0.0-1.0.  Higher = more canonical.
        """
        if not expression:
            return 0.0

        tokens = _tokenize(expression)
        if not tokens:
            return 0.0

        # Compute bigram overlap with training data
        expr_bigrams = _bigrams(tokens)
        if not expr_bigrams:
            return 0.5

        # Weighted overlap: rare training bigrams get higher weight
        overlap_score = 0.0
        total_weight = 0.0
        for bg in expr_bigrams:
            train_count = self._bigram_freq.get(bg, 0)
            if train_count > 0:
                # Inverse frequency: rare correct patterns are more distinctive
                weight = 1.0 / (1.0 + train_count)
                overlap_score += weight
                total_weight += weight

        if total_weight == 0:
            return 0.1  # no overlap with any canonical bigrams

        base_score = overlap_score / len(expr_bigrams)

        # Boost: prefer energy-quantity first ordering
        boost = 0.0
        for i in range(len(tokens)-1):
            if tokens[i] in self._energy_tokens and tokens[i+1] in self._energy_tokens:
                key = f"{tokens[i]}<{tokens[i+1]}"
                if key in self._potential_first:
                    boost += 0.01

        # Penalize: division by zero or self-cancellation patterns
        penalty = 0.0
        if re.search(r"\b(\w+)/\1\b", expression.replace(" ", "")):
            penalty += 0.3
        if re.search(r"\b(\w+)\+0\*", expression.replace(" ", "")):
            penalty += 0.3

        return max(0.0, min(1.0, base_score + boost - penalty))


def select_best_expression(
    candidates: list[tuple[str, float]],
    canonicalizer: ExpressionCanonicalizer,
    constancy_weight: float = 0.7,
) -> str:
    """Select the best expression combining constancy and canonical form."""
    if not candidates:
        return ""

    best_expr = candidates[0][0]
    best_score = -1.0

    for expr, constancy in candidates:
        form_score = canonicalizer.score(expr)
        combined = constancy_weight * constancy + (1 - constancy_weight) * form_score
        if combined > best_score:
            best_score = combined
            best_expr = expr

    return best_expr


def create_pre1905_canonicalizer() -> ExpressionCanonicalizer:
    """Create a canonicalizer trained on pre-1905 textbook invariants."""
    project_root = Path(__file__).resolve().parent.parent.parent
    pre1905_files = [
        project_root / "data/observations/mechanics_synthetic.json",
        project_root / "data/observations/em_synthetic.json",
        project_root / "data/observations/thermal_synthetic.json",
    ]

    invariants: list[str] = []
    for path in pre1905_files:
        if path.exists():
            with open(path) as f:
                for obs in json.load(f):
                    inv = obs.get("known_invariant")
                    if inv:
                        invariants.append(inv)

    return ExpressionCanonicalizer(list(set(invariants)))
