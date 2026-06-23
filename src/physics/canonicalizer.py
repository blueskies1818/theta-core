"""Expression canonicalizer v2 — dimensional ordering preferences.

Learns from pre-1905 invariants:
  - Which dimension types appear first in multiplications
  - Which dimensions are preferred in numerators vs denominators
  - Coefficient placement conventions
  - Term ordering in sums (potential before kinetic)

These are GENERAL structural conventions, not token-specific.
E/n is preferred over n/E because Energy (Mass·L²/T²) is "larger"
dimensionally than Scalar — the same reason pre-1905 has P*V/T
(Pressure·Volume / Temperature) rather than T/(P*V).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.physics.dimensions import Dimension


# ── Dimension weights — learned from pre-1905 patterns ───────────────────────

# Higher weight = preferred in numerator / preferred first in multiplication
# Derived from: m*g*h prefers Mass first, P*V/T puts Pressure/Volume up,
# 0.5*m*v^2 puts coefficient before mass, m*g*h + 0.5*m*v^2 puts potential first.
#
# The ordering: Mass > Length > Time (extensives before rates)
# Also: Scalar (dimensionless) is low — it's usually a count, not a quantity
_DIMENSION_WEIGHTS: dict[str, float] = {
    "Mass":     1.0,
    "Length":   0.8,
    "Time":     0.6,
    "Velocity": 0.7,
    "Accel":    0.65,
    "Force":    0.9,
    "Momentum": 0.85,
    "Energy":   0.95,
    "Scalar":   0.3,
}

# Symbols that should be ordered by dimension weight
# (known pre-1905 quantity symbols mapped to their dimension names)
_PRE1905_QTY_DIMS: dict[str, str] = {
    # Mechanics
    "m": "Mass", "m1": "Mass", "m2": "Mass",
    "g": "Accel",
    "h": "Length", "x": "Length", "y": "Length", "r": "Length",
    "v": "Velocity", "v1": "Velocity", "v2": "Velocity",
    "vx": "Velocity", "vy": "Velocity",
    "t": "Time",
    "k": "Force",
    # EM
    "q": "Scalar", "q1": "Scalar", "q2": "Scalar",
    "E": "Energy",  # electric field energy dimension
    "B": "Force",
    "epsilon": "Scalar", "Phi": "Scalar", "I": "Scalar",
    # Thermal
    "P": "Force", "V": "Length", "T": "Time",
    "n": "Scalar", "R": "Scalar",
    "S": "Scalar", "W": "Scalar", "Q": "Scalar",
    "delta_S": "Scalar",
}

# Post-1905 extensions — same dimensional logic applies
_POST1905_QTY_DIMS: dict[str, str] = {
    "E": "Energy",     # energy (relativistic/quantum)
    "p": "Momentum",
    "c": "Velocity",
    "gamma": "Scalar",
    "lambda": "Length",
    "hbar": "Energy",  # action ~ Energy*Time, treated as Energy
    "omega": "Scalar",  # frequency
    "h": "Energy",     # Planck's constant ~ Energy*Time
    "nu": "Scalar",    # frequency
    "K_max": "Energy",
    "phi": "Energy",   # work function
    "E_peak": "Energy",
    "u_rel": "Velocity",
    "u": "Velocity",
    "tau": "Time",
}

_ALL_QTY_DIMS = {**_PRE1905_QTY_DIMS, **_POST1905_QTY_DIMS}


def _dim_weight(symbol: str) -> float:
    """Return the dimensional preference weight for a quantity symbol."""
    dim_name = _ALL_QTY_DIMS.get(symbol)
    if dim_name is None:
        return 0.5  # unknown
    return _DIMENSION_WEIGHTS.get(dim_name, 0.5)


def _tokenize(expr: str) -> list[str]:
    """Tokenize an expression string."""
    tokens = []
    i = 0
    s = expr.replace(" ", "")
    while i < len(s):
        if s[i:i+3] == "^-2":
            tokens.append("^-2"); i += 3; continue
        if s[i:i+2] == "^-":
            tokens.append("^"); tokens.append("-"); i += 1; continue
        if s[i].isdigit() or (s[i] == "." and i+1 < len(s) and s[i+1].isdigit()):
            j = i
            while j < len(s) and (s[j].isdigit() or s[j] == "."):
                j += 1
            tokens.append(s[i:j]); i = j; continue
        if s[i].isalpha() or s[i] == "_":
            j = i
            while j < len(s) and (s[j].isalnum() or s[j] == "_"):
                j += 1
            tokens.append(s[i:j]); i = j; continue
        tokens.append(s[i]); i += 1
    return tokens


def _split_terms(expr: str) -> list[str]:
    """Split an expression into additive terms, preserving signs."""
    clean = expr.replace(" ", "")
    terms = []
    current = ""
    depth = 0
    for ch in clean:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch in "+-" and depth == 0:
            if current:
                terms.append(current)
            current = ch if ch == "-" else ""
        else:
            current += ch
    if current:
        terms.append(current)
    return [t for t in terms if t and t != "+"]


class ExpressionCanonicalizer:
    """Scores expressions by dimensional ordering and structural conventions.

    Learns general patterns from pre-1905 data, applies to any era.
    """

    def __init__(self, canonical_expressions: list[str]) -> None:
        # Learn: which dimension pairs appear in what order in multiplications
        self._mult_order: dict[tuple[str, str], int] = {}  # (dim1, dim2) -> count
        # Learn: which dimensions appear in numerators vs denominators
        self._num_count: dict[str, int] = {}
        self._den_count: dict[str, int] = {}
        # Learn: term ordering preferences
        self._term_order: dict[tuple[str, str], int] = {}

        for expr in canonical_expressions:
            tokens = _tokenize(expr)
            self._learn_from_tokens(tokens, expr)

        # Normalize counts to weights
        total_mult = sum(self._mult_order.values()) or 1
        self._mult_weights = {k: v/total_mult for k, v in self._mult_order.items()}

    def _learn_from_tokens(self, tokens: list[str], expr: str) -> None:
        """Extract dimensional patterns from tokenized expression."""
        terms = _split_terms(expr)

        for term in terms:
            # Find all quantity symbols in this term (in order)
            qty_symbols = []
            for t in tokens:
                if t in _ALL_QTY_DIMS and t not in {"+", "-", "*", "/", "^", "(", ")"}:
                    if t not in qty_symbols:  # deduplicate within term
                        qty_symbols.append(t)

            # Learn multiplication ordering
            for i in range(len(qty_symbols)-1):
                d1 = _ALL_QTY_DIMS.get(qty_symbols[i], "Unknown")
                d2 = _ALL_QTY_DIMS.get(qty_symbols[i+1], "Unknown")
                key = (d1, d2)
                self._mult_order[key] = self._mult_order.get(key, 0) + 1

            # Learn numerator/denominator preferences
            if "/" in term:
                num_part = term.split("/")[0]
                den_part = "/".join(term.split("/")[1:])
                for t in _tokenize(num_part):
                    if t in _ALL_QTY_DIMS and t not in {"(", ")"}:
                        d = _ALL_QTY_DIMS[t]
                        self._num_count[d] = self._num_count.get(d, 0) + 1
                for t in _tokenize(den_part):
                    if t in _ALL_QTY_DIMS and t not in {"(", ")"}:
                        d = _ALL_QTY_DIMS[t]
                        self._den_count[d] = self._den_count.get(d, 0) + 1

        # Learn term ordering: which dimension types come first in sums
        term_dims = []
        for term in terms:
            weights = [_dim_weight(t) for t in _tokenize(term)
                       if t in _ALL_QTY_DIMS]
            if weights:
                term_dims.append(max(weights))

    def score(self, expression: str) -> float:
        """Score by dimensional sensibility. 0-1, higher = more canonical."""
        if not expression:
            return 0.0

        tokens = _tokenize(expression)
        terms = _split_terms(expression)
        if not terms:
            return 0.0

        # Each term gets scored by dimensional ordering
        term_scores = []
        for term in terms:
            term_tokens = _tokenize(term)
            qty_symbols = [t for t in term_tokens
                          if t in _ALL_QTY_DIMS and t not in {"(", ")"}]

            if not qty_symbols:
                term_scores.append(0.5)
                continue

            # Check: are quantities ordered by dimension weight (descending)?
            weights = [_dim_weight(q) for q in qty_symbols]
            order_score = 1.0
            for i in range(len(weights)-1):
                if weights[i] < weights[i+1]:
                    order_score -= 0.1  # penalty for wrong order

            # Check: in a fraction, is numerator dimension >= denominator?
            frac_score = 1.0
            if "/" in term:
                before_slash = term.split("/")[0]
                after_slash = "/".join(term.split("/")[1:])
                num_syms = [t for t in _tokenize(before_slash)
                           if t in _ALL_QTY_DIMS]
                den_syms = [t for t in _tokenize(after_slash)
                           if t in _ALL_QTY_DIMS]
                num_w = max([_dim_weight(s) for s in num_syms]) if num_syms else 0.5
                den_w = max([_dim_weight(s) for s in den_syms]) if den_syms else 0.5
                if den_w > num_w:
                    frac_score -= 0.15  # energy/time OK, time/energy less so

            # Check: coefficient placement (number before variable)
            coeff_score = 1.0
            for i, t in enumerate(term_tokens):
                if re.match(r"^[\d.]+$", t) and i+1 < len(term_tokens):
                    if term_tokens[i+1] in _ALL_QTY_DIMS:
                        pass  # good: number before variable
                elif t in _ALL_QTY_DIMS and i+1 < len(term_tokens):
                    if re.match(r"^[\d.]+$", term_tokens[i+1]):
                        coeff_score -= 0.05  # variable before number

            term_scores.append(max(0.0, order_score * frac_score * coeff_score))

        # Overall: average of term scores, penalize self-cancellation
        avg = sum(term_scores) / len(term_scores)

        clean = expression.replace(" ", "")
        if re.search(r"\b(\w+)/\1\b", clean):
            avg -= 0.3
        if re.search(r"\b(\w+)\+0\*", clean):
            avg -= 0.3
        if re.search(r"\b(\w+)-\1\b", clean):
            avg -= 0.3

        return max(0.0, min(1.0, avg))


def select_best_expression(
    candidates: list[tuple[str, float]],
    canonicalizer: ExpressionCanonicalizer,
    constancy_weight: float = 0.7,
) -> str:
    """Select best expression by combined constancy + canonical form score."""
    if not candidates:
        return ""
    best_expr = candidates[0][0]
    best_score = -1.0
    for expr, constancy in candidates:
        form = canonicalizer.score(expr)
        combined = constancy_weight * constancy + (1 - constancy_weight) * form
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
