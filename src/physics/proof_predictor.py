"""Auto-prover v2: ML-based proof prediction trained on synthetic algebra.

Replaces the deterministic lookup table with a learning system. The prover
pattern-matches from synthetic math training data, NOT scenario-specific
pre-written proofs.

Architecture:
  1. SyntheticProofGenerator — generates 10,000+ algebraic proofs with tactics
  2. GoalFingerprinter — extracts structural features from Lean goal expressions
  3. ProofPredictor — ~4K param MLP that predicts best first tactic
  4. Training loop with cross-entropy loss
  5. Evaluation on held-out physics proofs

Output:
  - Model checkpoint: checkpoints/proof_predictor.pt (~20KB)
  - Results: data/proof_predictor_results.json
"""

from __future__ import annotations

import json
import math
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ═══════════════════════════════════════════════════════════════════════════════
# Tactic vocabulary — ordered by priority (lower index = preferred first attempt)
# ═══════════════════════════════════════════════════════════════════════════════

TACTIC_VOCAB = [
    "ring",
    "ring_nf",
    "field_simp; ring",
    "nlinarith",
    "linarith",
    "rw [h]; ring",
    "rw [h]; field_simp; ring",
    "expand; ring; nlinarith",
    "simp; ring",
    "norm_num; ring",
    "ring_nf; nlinarith",
    "field_simp; nlinarith",
]

TACTIC_TO_IDX = {t: i for i, t in enumerate(TACTIC_VOCAB)}
IDX_TO_TACTIC = {i: t for t, i in TACTIC_TO_IDX.items()}
NUM_TACTICS = len(TACTIC_VOCAB)

# ═══════════════════════════════════════════════════════════════════════════════
# Goal structure feature extraction
# ═══════════════════════════════════════════════════════════════════════════════

def _count_terms(expr: str) -> int:
    """Count top-level additive terms."""
    # Split on + and - but not inside parens
    depth = 0
    count = 1
    for ch in expr:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        elif depth == 0 and ch in '+-' and not (ch == '-' and expr.index(ch) == 0):
            # Check position — is this a leading minus on the whole expression?
            # Actually just count separators
            pass
    # Better approach: tokenize
    tokens = re.findall(r'[+\-]', expr)
    # Count only top-level + and -
    depth = 0
    top_level_ops = 0
    for i, ch in enumerate(expr):
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        elif depth == 0 and ch in '+-':
            # Skip leading minus/unary
            if i > 0 and expr[i-1] not in '(^*/+-':
                top_level_ops += 1
    return top_level_ops + 1


def _count_paren_depth(expr: str) -> int:
    """Maximum nesting depth of parentheses."""
    depth = 0
    max_depth = 0
    for ch in expr:
        if ch == '(':
            depth += 1
            max_depth = max(max_depth, depth)
        elif ch == ')':
            depth -= 1
    return max_depth


def _has_algebraic_division(expr: str) -> bool:
    """Check for algebraic division (variable in denominator), not numeric fractions."""
    # Find all '/' occurrences
    for m in re.finditer(r'/', expr):
        # Check what comes after the '/'
        after = expr[m.end():].lstrip()
        # If what follows is a variable or expression with variables, it's algebraic
        if after and (after[0].isalpha() or after[0] in '([{'):
            return True
    return False


def _count_algebraic_fractions(expr: str) -> int:
    """Count algebraic fractions (not numeric fractions like 1/2)."""
    count = 0
    for m in re.finditer(r'/', expr):
        pos = m.end()
        # Skip whitespace
        while pos < len(expr) and expr[pos] in ' \t':
            pos += 1
        if pos < len(expr) and (expr[pos].isalpha() or expr[pos] in '([{'):
            count += 1
    return count


def extract_features(goal_expr: str, kinematic_subs: Optional[dict[str, str]] = None,
                     has_hypothesis: bool = False) -> dict[str, float]:
    """Extract structural fingerprint from a proof goal expression.

    Args:
        goal_expr: The expression to prove conserved (e.g., "m * g * h + (1/2) * m * v ^ 2")
        kinematic_subs: Optional kinematic substitution rules
        has_hypothesis: Whether a hypothesis is present in the theorem

    Returns:
        Dict of 24 feature floats normalized to [0, 1] range where possible.
    """
    # Track pre-substitution features for expansion ratio
    pre_expr = goal_expr
    pre_paren_depth = _count_paren_depth(pre_expr)
    pre_length = len(pre_expr)
    pre_ops = sum(1 for ch in pre_expr if ch in '+-*/^')

    # Apply substitutions if present to get the actual Lean goal
    working_expr = goal_expr
    if kinematic_subs:
        for var in sorted(kinematic_subs, key=len, reverse=True):
            working_expr = re.sub(r'\b' + re.escape(var) + r'\b',
                                  f"({kinematic_subs[var]})", working_expr)

    # Feature extraction
    features: dict[str, float] = {}

    # --- Boolean/indicator features ---
    features["has_trig"] = 1.0 if re.search(r'\b(sin|cos|tan)\b', working_expr) else 0.0
    features["has_algebraic_div"] = 1.0 if _has_algebraic_division(working_expr) else 0.0
    features["has_power"] = 1.0 if '^' in working_expr else 0.0
    features["has_sqrt"] = 1.0 if 'sqrt' in working_expr else 0.0
    features["has_hypothesis"] = 1.0 if has_hypothesis else 0.0
    features["has_squared_sum"] = 1.0 if re.search(r'\^2', working_expr) else 0.0

    # --- Numeric features ---

    # Count variables (lowercase identifiers that are not keywords)
    keywords = {'sin', 'cos', 'tan', 'sqrt', 'exp', 'log', 'abs',
                'theta', 'omega', 'alpha', 'beta', 'gamma', 'delta', 'pi'}
    var_matches = re.findall(r'\b([a-zA-Z_]\w*)\b', working_expr)
    variables = set(v for v in var_matches if v not in keywords and not v.isdigit())
    features["num_variables"] = min(len(variables), 20) / 20.0

    # Term count
    features["term_count"] = min(_count_terms(working_expr), 20) / 20.0

    # Parenthesis depth
    post_paren_depth = _count_paren_depth(working_expr)
    features["paren_depth"] = min(post_paren_depth, 10) / 10.0

    # Substitution expansion ratio (how much paren depth increased)
    if kinematic_subs and pre_paren_depth > 0:
        features["sub_expansion"] = min(post_paren_depth / max(pre_paren_depth, 1), 5.0) / 5.0
    else:
        features["sub_expansion"] = 0.0

    # Operator counts
    features["num_add"] = min(working_expr.count('+'), 10) / 10.0
    features["num_mult"] = min(working_expr.count('*'), 10) / 10.0

    # Power statistics
    power_exponents = [int(m) for m in re.findall(r'\^(\d+)', working_expr)]
    features["max_power"] = min(max(power_exponents) if power_exponents else 0, 10) / 10.0

    # Total operations
    total_ops = sum(1 for ch in working_expr if ch in '+-*/^')
    features["total_ops"] = min(total_ops, 30) / 30.0

    # Expression length
    features["expr_length"] = min(len(working_expr), 200) / 200.0

    # Number of parentheses pairs
    features["num_parens"] = min(working_expr.count('('), 15) / 15.0

    # Algebraic fraction count (not numeric like 1/2)
    features["num_alg_fractions"] = min(_count_algebraic_fractions(working_expr), 10) / 10.0

    # Has numeric coefficients (like 1/2, 0.5)
    features["has_numeric_coeff"] = 1.0 if re.search(r'\b\d+\.?\d*\b', working_expr) else 0.0

    # Detect nlinarith patterns (nonlinear polynomial equations with multiple power terms)
    features["has_nonlinear"] = 1.0 if (features["has_power"] > 0 and
                                         features["max_power"] > 0.2) else 0.0

    # Detect ring-friendly patterns (purely polynomial, no algebraic division, no trig)
    features["is_pure_polynomial"] = 1.0 if (
        features["has_algebraic_div"] == 0.0 and
        features["has_trig"] == 0.0 and
        features["has_sqrt"] == 0.0
    ) else 0.0

    # Has any substitution been applied (vs original expression)
    features["has_subs"] = 1.0 if kinematic_subs else 0.0

    return features


# Ordered feature list for tensor conversion
FEATURE_NAMES = [
    "has_trig", "has_algebraic_div", "has_power", "has_sqrt",
    "has_hypothesis", "has_squared_sum",
    "num_variables", "term_count", "paren_depth", "sub_expansion",
    "num_add", "num_mult", "max_power", "total_ops",
    "expr_length", "num_parens", "num_alg_fractions",
    "has_numeric_coeff", "has_nonlinear", "is_pure_polynomial",
    "has_subs",
]
NUM_FEATURES = len(FEATURE_NAMES)


def features_to_tensor(features: dict[str, float]) -> torch.Tensor:
    """Convert feature dict to normalized tensor."""
    return torch.tensor([features.get(k, 0.0) for k in FEATURE_NAMES],
                        dtype=torch.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# Synthetic proof data generation
# ═══════════════════════════════════════════════════════════════════════════════

# Variable pools for different algebraic domains
_VAR_POOL = ["x", "y", "z", "a", "b", "c", "d", "u", "v", "w"]
_TRIG_VARS = ["theta", "alpha", "beta"]
_COEFF_POOL = ["2", "3", "4", "5", "-1", "-2"]


def _random_vars(n: int, with_trig: bool = False) -> list[str]:
    """Pick n random variables."""
    pool = list(_VAR_POOL)
    if with_trig:
        pool = pool + _TRIG_VARS
    return random.sample(pool, min(n, len(pool)))


def _rand_coeff() -> str:
    return random.choice(_COEFF_POOL + ["1"])


def _maybe_coeff(var: str) -> str:
    """Possibly prefix with a coefficient."""
    if random.random() < 0.4:
        return f"{_rand_coeff()}*{var}"
    return var


def _rand_sign() -> str:
    return random.choice(["+", "-"])


def _wrap_paren(s: str) -> str:
    if s.startswith('('):
        return s
    if '+' in s or '-' in s:
        return f"({s})"
    return s


# ═══════════════ Category A: Polynomial simplification (ring) ═══════════════

def gen_poly_simp() -> tuple[str, str, dict]:
    """Generate a polynomial simplification needing ring."""
    x = random.choice(_VAR_POOL)
    # (ax² + bx + c) - (dx² + ex + f) = (a-d)x² + (b-e)x + (c-f)
    a, b, c = random.randint(1, 9), random.randint(1, 9), random.randint(1, 9)
    d, e, f = random.randint(1, 9), random.randint(1, 9), random.randint(1, 9)
    lhs = f"({a}*{x}^2 + {b}*{x} + {c}) - ({d}*{x}^2 + {e}*{x} + {f})"
    rhs = f"{(a-d)}*{x}^2 + {(b-e)}*{x} + {(c-f)}"
    goal = f"{lhs} = {rhs}"
    features = extract_features(goal)
    return goal, "ring", features


def gen_poly_expand() -> tuple[str, str, dict]:
    """Generate a polynomial expansion needing ring_nf."""
    x, y = random.sample(_VAR_POOL, 2)
    a = random.randint(1, 4)
    b = random.randint(1, 4)
    lhs = f"({x} + {y})^{a + b}"
    # Expanded form
    if a + b == 2:
        rhs = f"{x}^2 + 2*{x}*{y} + {y}^2"
    elif a + b == 3:
        rhs = f"{x}^3 + 3*{x}^2*{y} + 3*{x}*{y}^2 + {y}^3"
    elif a + b == 4:
        rhs = f"{x}^4 + 4*{x}^3*{y} + 6*{x}^2*{y}^2 + 4*{x}*{y}^3 + {y}^4"
    else:
        rhs = f"{x}^5 + 5*{x}^4*{y} + 10*{x}^3*{y}^2 + 10*{x}^2*{y}^3 + 5*{x}*{y}^4 + {y}^5"
    goal = f"{lhs} = {rhs}"
    features = extract_features(goal)
    return goal, "ring_nf", features


def gen_poly_ring_basic() -> tuple[str, str, dict]:
    """Simple ring: factoring/distributing."""
    x, y, z = random.sample(_VAR_POOL, 3)
    a, b = random.randint(1, 5), random.randint(1, 5)
    lhs = f"{a}*({x} + {y}) + {b}*({x} - {y})"
    rhs = f"({a+b})*{x} + ({a-b})*{y}"
    goal = f"{lhs} = {rhs}"
    features = extract_features(goal)
    return goal, "ring", features


# ═══════════════ Category B: Field algebra (field_simp; ring) ═══════════════

def gen_field_basic() -> tuple[str, str, dict]:
    """Fraction algebra: a/(b+c) + d/(b+c) = (a+d)/(b+c)."""
    a, b, c, d = random.sample(_VAR_POOL, 4)
    lhs = f"{a}/({b} + {c}) + {d}/({b} + {c})"
    rhs = f"({a} + {d})/({b} + {c})"
    goal = f"{lhs} = {rhs}"
    features = extract_features(goal)
    return goal, "field_simp; ring", features


def gen_field_complex() -> tuple[str, str, dict]:
    """Complex fraction: (a/b) + (c/d) = (ad + bc)/(bd)."""
    a, b, c, d = random.sample(_VAR_POOL, 4)
    lhs = f"{a}/{b} + {c}/{d}"
    rhs = f"({a}*{d} + {b}*{c})/({b}*{d})"
    goal = f"{lhs} = {rhs}"
    features = extract_features(goal)
    return goal, "field_simp; ring", features


def gen_field_mult() -> tuple[str, str, dict]:
    """Multiplication with fractions: (a/b)*(c/d) = (a*c)/(b*d)."""
    a, b, c, d = random.sample(_VAR_POOL, 4)
    lhs = f"({a}/{b})*({c}/{d})"
    rhs = f"({a}*{c})/({b}*{d})"
    goal = f"{lhs} = {rhs}"
    features = extract_features(goal)
    return goal, "field_simp; ring", features


# ═══════════════ Category C: Trig identities (rw[h]; ring) ═══════════════

def gen_trig_id() -> tuple[str, str, dict]:
    """Trig identity: sin² + cos² = 1."""
    t = random.choice(_TRIG_VARS)
    lhs = f"sin({t})^2 + cos({t})^2"
    rhs = "1"
    goal = f"{lhs} = {rhs}"
    features = extract_features(goal, has_hypothesis=True)
    # This needs rw[h] where h is the trig identity, then ring
    return goal, "rw [h]; ring", features


def gen_trig_poly() -> tuple[str, str, dict]:
    """Polynomial with trig substitutions."""
    t = random.choice(_TRIG_VARS)
    a, b = random.randint(1, 5), random.randint(1, 5)
    lhs = f"{a}*sin({t})^2 + {a}*cos({t})^2"
    rhs = str(a)
    goal = f"{lhs} = {rhs}"
    features = extract_features(goal, has_hypothesis=True)
    return goal, "rw [h]; ring", features


def gen_trig_complex() -> tuple[str, str, dict]:
    """More complex trig polynomial."""
    t = random.choice(_TRIG_VARS)
    a = random.randint(1, 4)
    lhs = f"({a}*sin({t})^2 + {a}*cos({t})^2)*{a}"
    rhs = str(a * a)
    goal = f"{lhs} = {rhs}"
    features = extract_features(goal, has_hypothesis=True)
    return goal, "rw [h]; ring", features


# ═══════════════ Category D: nlinarith ═══════════════

def gen_nlinarith_simple() -> tuple[str, str, dict]:
    """Nonlinear arithmetic needing nlinarith."""
    x, y = random.sample(_VAR_POOL, 2)
    a, b = random.randint(1, 5), random.randint(1, 5)
    lhs = f"({x} + {y})^2"
    rhs = f"{x}^2 + 2*{x}*{y} + {y}^2"
    goal = f"{lhs} = {rhs}"
    features = extract_features(goal)
    return goal, "nlinarith", features


def gen_nlinarith_diff() -> tuple[str, str, dict]:
    """Difference of squares."""
    x, y = random.sample(_VAR_POOL, 2)
    lhs = f"{x}^2 - {y}^2"
    rhs = f"({x} + {y})*({x} - {y})"
    goal = f"{lhs} = {rhs}"
    features = extract_features(goal)
    return goal, "nlinarith", features


# ═══════════════ Category E: linarith ═══════════════

def gen_linarith() -> tuple[str, str, dict]:
    """Linear arithmetic needing linarith."""
    x, y, z = random.sample(_VAR_POOL, 3)
    a, b, c = random.randint(1, 5), random.randint(1, 5), random.randint(1, 5)
    lhs = f"{a}*{x} + {b}*{y}"
    rhs = f"{c}*{x} + {c}*{y} + ({a-c})*{x} + ({b-c})*{y}"
    goal = f"{lhs} = {rhs}"
    features = extract_features(goal)
    return goal, "linarith", features


# ═══════════════ Category F: ring_nf; nlinarith ═══════════════

def gen_ring_nf_nlinarith() -> tuple[str, str, dict]:
    """Expansion then nonlinear arithmetic."""
    x, y = random.sample(_VAR_POOL, 2)
    a = random.randint(2, 4)
    lhs = f"({x} + {y})^{a} - ({x}^{a} + {y}^{a})"
    # (x+y)^a - x^a - y^a has mixed terms
    # e.g., a=3: 3x²y + 3xy²
    if a == 2:
        rhs = f"2*{x}*{y}"
    elif a == 3:
        rhs = f"3*{x}^2*{y} + 3*{x}*{y}^2"
    else:
        rhs = f"4*{x}^3*{y} + 6*{x}^2*{y}^2 + 4*{x}*{y}^3"
    goal = f"{lhs} = {rhs}"
    features = extract_features(goal)
    return goal, "ring_nf; nlinarith", features


# ═══════════════ Category G: field_simp; nlinarith ═══════════════

def gen_field_nlinarith() -> tuple[str, str, dict]:
    """Field simplification then nonlinear arithmetic."""
    a, b, c = random.sample(_VAR_POOL, 3)
    lhs = f"({a}/{b})^2 + ({a}/{c})^2"
    rhs = f"({a}^2*({b}^2 + {c}^2))/({b}^2*{c}^2)"
    goal = f"{lhs} = {rhs}"
    features = extract_features(goal)
    return goal, "field_simp; nlinarith", features


# ═══════════════ Category H: expand; ring; nlinarith (multi-step) ═══════════════

def gen_expand_ring_nlinarith() -> tuple[str, str, dict]:
    """Multi-step: expand product then ring/nlinarith."""
    x, y, z = random.sample(_VAR_POOL, 3)
    a = random.randint(1, 3)
    b = random.randint(1, 3)
    lhs = f"({x} + {y})*({x}^{a} + {y}^{a})"
    if a == 1:
        rhs = f"{x}^2 + 2*{x}*{y} + {y}^2"
    elif a == 2:
        rhs = f"{x}^3 + {x}^2*{y} + {x}*{y}^2 + {y}^3"
    else:
        rhs = f"{x}^4 + {x}^3*{y} + {x}*{y}^3 + {y}^4"
    goal = f"{lhs} = {rhs}"
    features = extract_features(goal)
    return goal, "expand; ring; nlinarith", features


# ═══════════════ Category I: norm_num; ring ═══════════════

def gen_norm_num_ring() -> tuple[str, str, dict]:
    """Numeric normalization then ring."""
    x, y = random.sample(_VAR_POOL, 2)
    a, b, c = random.randint(1, 9), random.randint(1, 9), random.randint(1, 9)
    lhs = f"{a}*({b}*{x} + {c}*{y})"
    rhs = f"{a*b}*{x} + {a*c}*{y}"
    goal = f"{lhs} = {rhs}"
    features = extract_features(goal)
    return goal, "norm_num; ring", features


def gen_norm_num_ring_complex() -> tuple[str, str, dict]:
    """More complex numeric normalization."""
    x, y = random.sample(_VAR_POOL, 2)
    a = random.randint(2, 6)
    lhs = f"({a}*{x} + {a}*{y})^{a}"
    # ring alone would work here; norm_num handles the numeric coefficients
    goal = f"{lhs} = {lhs}"  # identity, just to verify expansion
    # Actually generate a real target
    if a == 2:
        rhs = f"{a*a}*{x}^2 + {2*a*a}*{x}*{y} + {a*a}*{y}^2"
    elif a == 3:
        rhs = f"{a**3}*{x}^3 + {3*a**3}*{x}^2*{y} + {3*a**3}*{x}*{y}^2 + {a**3}*{y}^3"
    else:
        rhs = lhs  # fallback
    goal = f"{lhs} = {rhs}"
    features = extract_features(goal)
    return goal, "norm_num; ring", features


# ═══════════════ Category J: simp; ring ═══════════════

def gen_simp_ring() -> tuple[str, str, dict]:
    """Simplification then ring."""
    x, y = random.sample(_VAR_POOL, 2)
    a, b = random.randint(1, 5), random.randint(1, 5)
    lhs = f"({a}*{x} + 0) + ({b}*{y}*1)"
    rhs = f"{a}*{x} + {b}*{y}"
    goal = f"{lhs} = {rhs}"
    features = extract_features(goal)
    return goal, "simp; ring", features


# ═══════════════ Category K: polynomial with numeric fractions (ring) ══════════

def gen_poly_numeric_frac() -> tuple[str, str, dict]:
    """Polynomial with numeric coefficients like (1/2) — still needs ring."""
    x, y = random.sample(_VAR_POOL, 2)
    num, den = random.choice([(1, 2), (1, 3), (2, 3), (3, 4), (1, 4), (3, 2)])
    a, b = random.randint(1, 5), random.randint(1, 5)
    c, d = random.randint(1, 5), random.randint(1, 5)
    lhs = f"({num}/{den})*{a}*{x}^2 + ({num}/{den})*{b}*{x} + ({num}/{den})*{c}"
    rhs = f"({num}/{den})*({a}*{x}^2 + {b}*{x} + {c})"
    goal = f"{lhs} = {rhs}"
    features = extract_features(goal)
    return goal, "ring", features


def gen_poly_power_ring() -> tuple[str, str, dict]:
    """Polynomial with powers but no trig/hypothesis — needs ring."""
    x, y = random.sample(_VAR_POOL, 2)
    a, b = random.randint(1, 5), random.randint(1, 5)
    lhs = f"({a}*{x} + {b}*{y})^2"
    rhs = f"{a*a}*{x}^2 + {2*a*b}*{x}*{y} + {b*b}*{y}^2"
    goal = f"{lhs} = {rhs}"
    features = extract_features(goal)
    return goal, "ring", features


def gen_poly_multi_power_ring() -> tuple[str, str, dict]:
    """Multi-variable polynomial with powers — needs ring_nf."""
    x, y, z = random.sample(_VAR_POOL, 3)
    a, b = random.randint(2, 4), random.randint(1, 3)
    lhs = f"({x} + {y} + {z})^2"
    rhs = f"{x}^2 + {y}^2 + {z}^2 + 2*{x}*{y} + 2*{x}*{z} + 2*{y}*{z}"
    goal = f"{lhs} = {rhs}"
    features = extract_features(goal)
    return goal, "ring_nf", features


def gen_ring_with_constants() -> tuple[str, str, dict]:
    """Ring problem with numeric constants mixed in."""
    x, y = random.sample(_VAR_POOL, 2)
    c1, c2 = random.randint(2, 8), random.randint(2, 8)
    lhs = f"({c1}*{x} + {c2})*({c1}*{x} - {c2})"
    rhs = f"{c1*c1}*{x}^2 - {c2*c2}"
    goal = f"{lhs} = {rhs}"
    features = extract_features(goal)
    return goal, "ring", features


def gen_linarith_with_fractions() -> tuple[str, str, dict]:
    """Linear arithmetic with numeric fractions — still linarith."""
    x, y = random.sample(_VAR_POOL, 2)
    a, b = random.randint(1, 6), random.randint(1, 6)
    lhs = f"({a}/{b})*{x} + ({a}/{b})*{y}"
    rhs = f"({a}/{b})*({x} + {y})"
    goal = f"{lhs} = {rhs}"
    features = extract_features(goal)
    return goal, "linarith", features


# ═══════════════════════════════════════════════════════════════════════════════
# Generator registry — maps generation functions to tactic categories
# ═══════════════════════════════════════════════════════════════════════════════

GENERATORS: list[tuple[Any, str, float]] = [
    # (function, tactic_name, weight) — weights control distribution
    # 2-step (40% of data): ring, ring_nf, field_simp; ring
    (gen_poly_simp, "ring", 8.0),
    (gen_poly_expand, "ring_nf", 6.0),
    (gen_poly_ring_basic, "ring", 6.0),
    (gen_poly_power_ring, "ring", 5.0),
    (gen_poly_multi_power_ring, "ring_nf", 4.0),
    (gen_ring_with_constants, "ring", 4.0),
    (gen_poly_numeric_frac, "ring", 4.0),
    (gen_field_basic, "field_simp; ring", 4.0),
    (gen_field_complex, "field_simp; ring", 4.0),
    (gen_field_mult, "field_simp; ring", 3.0),
    (gen_norm_num_ring, "norm_num; ring", 3.0),
    (gen_simp_ring, "simp; ring", 2.0),
    # 3-step (30%): nlinarith combos, trig combos
    (gen_trig_id, "rw [h]; ring", 5.0),
    (gen_trig_poly, "rw [h]; ring", 4.0),
    (gen_trig_complex, "rw [h]; ring", 3.0),
    (gen_nlinarith_simple, "nlinarith", 5.0),
    (gen_nlinarith_diff, "nlinarith", 5.0),
    (gen_ring_nf_nlinarith, "ring_nf; nlinarith", 5.0),
    (gen_field_nlinarith, "field_simp; nlinarith", 3.0),
    # 1-step (20%): linarith
    (gen_linarith, "linarith", 8.0),
    (gen_linarith_with_fractions, "linarith", 4.0),
    (gen_norm_num_ring_complex, "norm_num; ring", 3.0),
    # 4+ step (10%): expand, then ring/nlinarith
    (gen_expand_ring_nlinarith, "expand; ring; nlinarith", 6.0),
]

# Normalize weights
_total_weight = sum(w for _, _, w in GENERATORS)


@dataclass
class TrainingExample:
    """A single training example for the proof predictor."""
    goal_expr: str
    tactic: str
    tactic_idx: int
    features: dict[str, float]
    feature_tensor: torch.Tensor


class SyntheticProofGenerator:
    """Generate synthetic algebraic proofs for training.

    Generates 10,000+ random algebraic proofs programmatically across
    multiple categories: polynomials, trig identities, fraction algebra,
    expansions, multi-step proofs.

    Split: 40% 2-step, 30% 3-step, 20% 1-step, 10% 4+ step.
    """

    def __init__(self, seed: int = 42):
        random.seed(seed)

    def generate(self, n: int = 10_500) -> list[TrainingExample]:
        """Generate n synthetic training examples.

        Returns:
            List of TrainingExample objects.
        """
        examples: list[TrainingExample] = []
        seen_goals: set[str] = set()

        attempts = 0
        while len(examples) < n and attempts < n * 3:
            attempts += 1
            gen_fn, tactic, _ = random.choices(
                GENERATORS, weights=[w for _, _, w in GENERATORS], k=1
            )[0]
            try:
                goal, _, features = gen_fn()
            except (ValueError, IndexError, KeyError):
                continue

            # Deduplicate
            goal_key = goal.replace(" ", "")
            if goal_key in seen_goals:
                continue
            seen_goals.add(goal_key)

            tactic_idx = TACTIC_TO_IDX.get(tactic, 0)
            feat_tensor = features_to_tensor(features)
            examples.append(TrainingExample(
                goal_expr=goal,
                tactic=tactic,
                tactic_idx=tactic_idx,
                features=features,
                feature_tensor=feat_tensor,
            ))

        return examples


# ═══════════════════════════════════════════════════════════════════════════════
# Proof predictor model (~4K parameters)
# ═══════════════════════════════════════════════════════════════════════════════

class ProofPredictor(nn.Module):
    """Small MLP: goal structure fingerprint → best tactic prediction.

    Architecture: 20 → 64 → 32 → 12 (3,964 parameters)
    Input: 20 structural features from goal expression
    Output: 12-class tactic prediction with confidence scores

    The model learns structural patterns: e.g., division → field_simp,
    trig → rw[h]; ring, pure polynomial → ring.
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(NUM_FEATURES, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, NUM_TACTICS)
        self.dropout = nn.Dropout(0.1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [batch, 20] feature tensor

        Returns:
            [batch, 12] raw logits for tactic classes
        """
        h = F.relu(self.bn1(self.fc1(x)))
        h = self.dropout(h)
        h = F.relu(self.bn2(self.fc2(h)))
        h = self.dropout(h)
        return self.fc3(h)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return predicted tactic indices."""
        logits = self.forward(x)
        return torch.argmax(logits, dim=-1)

    def predict_topk(self, x: torch.Tensor, k: int = 3) -> torch.Tensor:
        """Return top-k predicted tactic indices."""
        logits = self.forward(x)
        return torch.topk(logits, k=min(k, NUM_TACTICS), dim=-1).indices

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities."""
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ═══════════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrainingResult:
    model: ProofPredictor
    train_acc: float
    val_acc: float
    epochs: int
    train_losses: list[float]
    val_accuracies: list[float]
    best_epoch: int


def train_model(
    model: ProofPredictor,
    train_examples: list[TrainingExample],
    val_examples: list[TrainingExample] | None = None,
    num_epochs: int = 50,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-5,
    early_stop_patience: int = 10,
    verbose: bool = True,
) -> TrainingResult:
    """Train the proof predictor on synthetic examples.

    Args:
        model: ProofPredictor instance
        train_examples: Training examples
        val_examples: Validation examples (if None, uses 10% of train)
        num_epochs: Max training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        weight_decay: L2 regularization
        early_stop_patience: Stop if val acc doesn't improve for N epochs
        verbose: Print progress

    Returns:
        TrainingResult with model and metrics.
    """
    if val_examples is None:
        split = int(len(train_examples) * 0.9)
        random.shuffle(train_examples)
        val_examples = train_examples[split:]
        train_examples = train_examples[:split]

    # Prepare tensors
    X_train = torch.stack([ex.feature_tensor for ex in train_examples])
    y_train = torch.tensor([ex.tactic_idx for ex in train_examples], dtype=torch.long)
    X_val = torch.stack([ex.feature_tensor for ex in val_examples])
    y_val = torch.tensor([ex.tactic_idx for ex in val_examples], dtype=torch.long)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.CrossEntropyLoss()

    train_losses: list[float] = []
    val_accuracies: list[float] = []
    best_val_acc = -1.0
    best_epoch = -1
    best_state = None
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0

        # Shuffle training data
        perm = torch.randperm(len(X_train))
        X_shuffled = X_train[perm]
        y_shuffled = y_train[perm]

        for i in range(0, len(X_train), batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            y_batch = y_shuffled[i:i + batch_size]

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(num_batches, 1)
        train_losses.append(avg_loss)

        # Validation
        model.eval()
        with torch.no_grad():
            predictions = model.predict(X_val)
            val_acc = (predictions == y_val).float().mean().item()
        val_accuracies.append(val_acc)

        # Early stopping
        if val_acc > best_val_acc + 0.001:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and epoch % 5 == 0:
            print(f"Epoch {epoch:3d}/{num_epochs} | loss={avg_loss:.4f} | val_acc={val_acc:.4f}")

        if patience_counter >= early_stop_patience:
            if verbose:
                print(f"Early stopping at epoch {epoch} (patience={early_stop_patience})")
            break

    # Restore best model
    if best_state:
        model.load_state_dict(best_state)

    # Compute final training accuracy
    model.eval()
    with torch.no_grad():
        train_preds = model.predict(X_train)
        train_acc = (train_preds == y_train).float().mean().item()

    return TrainingResult(
        model=model,
        train_acc=train_acc,
        val_acc=best_val_acc,
        epochs=best_epoch + 1,
        train_losses=train_losses,
        val_accuracies=val_accuracies,
        best_epoch=best_epoch,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation on physics proofs
# ═══════════════════════════════════════════════════════════════════════════════

# Held-out physics proof test cases (the model has NEVER seen these)
PHYSICS_TEST_CASES = [
    # Energy conservation — free fall
    {
        "name": "energy_free_fall",
        "goal": "m * g * h + (1/2) * m * v ^ 2",
        "subs": {"v": "g * t", "h": "h0 - (1/2) * g * t ^ 2"},
        "has_hypothesis": False,
        "expected_tactics": ["ring", "ring_nf", "nlinarith"],  # any of these works
    },
    # Energy conservation — free fall with initial velocity
    {
        "name": "energy_free_fall_v0",
        "goal": "m * g * h + (1/2) * m * v ^ 2",
        "subs": {"v": "v0 - g * t", "h": "h0 + v0 * t - (1/2) * g * t ^ 2"},
        "has_hypothesis": False,
        "expected_tactics": ["ring", "ring_nf", "nlinarith"],
    },
    # Energy conservation — projectile
    {
        "name": "energy_projectile",
        "goal": "(1/2) * m * (vx ^ 2 + vy ^ 2) + m * g * y",
        "subs": {
            "vx": "v0 * cos(theta)",
            "vy": "v0 * sin(theta) - g * t",
            "y": "h0 + v0 * sin(theta) * t - (1/2) * g * t ^ 2",
        },
        "has_hypothesis": True,
        "expected_tactics": ["rw [h]; ring", "rw [h]; field_simp; ring",
                            "rw [h]; field_simp; ring", "ring_nf; nlinarith"],
    },
    # EM conservation — charged particle in E-field
    {
        "name": "em_kinetic_potential",
        "goal": "(1/2) * m * v ^ 2 + q * E * x",
        "subs": {"v": "q * E * t / m", "x": "(1/2) * q * E * t ^ 2 / m"},
        "has_hypothesis": True,
        "expected_tactics": ["rw [h]; ring", "field_simp; ring", "rw [h]; field_simp; ring"],
    },
    # EM conservation — general field
    {
        "name": "em_general",
        "goal": "(1/2) * m * v ^ 2 + q * Phi",
        "subs": {"v": "v0 + q * E * t / m"},
        "has_hypothesis": True,
        "expected_tactics": ["rw [h]; ring", "field_simp; ring", "rw [h]; field_simp; ring"],
    },
    # Relativistic invariant — E² - p²c²
    {
        "name": "relativistic_invariant",
        "goal": "E ^ 2 - (p * c) ^ 2",
        "subs": {
            "E": "gamma * m * c ^ 2",
            "p": "gamma * m * v",
        },
        "has_hypothesis": True,
        "expected_tactics": ["rw [h]; ring", "expand; ring; nlinarith",
                            "ring_nf; nlinarith", "ring_nf"],
    },
    # Energy conservation — pendulum (trigonometric)
    {
        "name": "energy_pendulum",
        "goal": "m * g * L * (1 - cos(theta)) + m * g * L * (cos(theta) - cos(theta0))",
        "subs": {},
        "has_hypothesis": True,
        "expected_tactics": ["rw [h]; ring", "ring", "simp; ring"],
    },
    # Spring energy — trig
    {
        "name": "energy_spring_trig",
        "goal": "(1/2) * m * (A * omega * cos(omega * t)) ^ 2 + (1/2) * k * (A * sin(omega * t)) ^ 2",
        "subs": {},
        "has_hypothesis": True,
        "expected_tactics": ["rw [h]; ring", "rw [h]; ring"],
    },
    # Simple polynomial identity (physics-related)
    {
        "name": "work_energy",
        "goal": "(1/2) * m * (v ^ 2 - v0 ^ 2)",
        "subs": {},
        "has_hypothesis": True,
        "expected_tactics": ["rw [h]; ring", "ring", "nlinarith"],
    },
    # Generic algebraic conservation: m * a + 0.5 * m * b^2 with subs
    {
        "name": "algebraic_conservation_generic",
        "goal": "m * a * x + (1/2) * m * b ^ 2",
        "subs": {"b": "a * t", "x": "x0 - (1/2) * a * t ^ 2"},
        "has_hypothesis": False,
        "expected_tactics": ["ring", "ring_nf", "nlinarith"],
    },
]


def evaluate_physics_proofs(
    model: ProofPredictor,
    test_cases: list[dict] | None = None,
) -> dict:
    """Evaluate the model on held-out physics proofs.

    Returns:
        Dict with per-case results and aggregate metrics.
    """
    if test_cases is None:
        test_cases = PHYSICS_TEST_CASES

    device = next(model.parameters()).device
    model.eval()

    results: list[dict] = []
    first_correct = 0
    top3_correct = 0

    for case in test_cases:
        # Extract features
        features = extract_features(
            case["goal"],
            kinematic_subs=case.get("subs"),
            has_hypothesis=case.get("has_hypothesis", False),
        )
        feat_tensor = features_to_tensor(features).unsqueeze(0).to(device)

        with torch.no_grad():
            # Get top-3 predictions
            topk_indices = model.predict_topk(feat_tensor, k=3)[0]
            topk_tactics = [IDX_TO_TACTIC[int(idx.item())] for idx in topk_indices]
            probs = model.predict_proba(feat_tensor)[0]

        # Check correctness
        expected = case["expected_tactics"]
        first_hit = topk_tactics[0] in expected
        top3_hit = any(t in expected for t in topk_tactics)

        if first_hit:
            first_correct += 1
        if top3_hit:
            top3_correct += 1

        results.append({
            "name": case["name"],
            "first_prediction": topk_tactics[0],
            "first_correct": first_hit,
            "top3_predictions": topk_tactics,
            "top3_correct": top3_hit,
            "confidence_top1": probs[topk_indices[0]].item(),
            "feature_summary": {
                k: features[k] for k in FEATURE_NAMES[:8]
            },
        })

    total = len(test_cases)
    first_accuracy = first_correct / total if total > 0 else 0.0
    top3_accuracy = top3_correct / total if total > 0 else 0.0

    return {
        "total_test_cases": total,
        "first_tactic_accuracy": first_accuracy,
        "top3_tactic_accuracy": top3_accuracy,
        "first_tactic_correct": first_correct,
        "top3_correct": top3_correct,
        "per_case_results": results,
        "passes_first_threshold": first_accuracy >= 0.60,
        "passes_top3_threshold": top3_accuracy >= 0.85,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def main(
    num_synthetic: int = 10_500,
    output_dir: str | Path = ".",
    seed: int = 42,
) -> dict:
    """Run the full auto-prover v2 pipeline.

    1. Generate synthetic training data
    2. Train the ProofPredictor model
    3. Evaluate on held-out physics proofs
    4. Save model checkpoint and results

    Returns:
        Dict with training and evaluation metrics.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("AUTO-PROVER v2: ML-based proof prediction")
    print("=" * 70)

    # ── Step 1: Generate synthetic training data ──
    print(f"\n[1/4] Generating {num_synthetic} synthetic algebraic proofs...")
    t0 = time.time()
    gen = SyntheticProofGenerator(seed=seed)
    examples = gen.generate(n=num_synthetic)
    gen_time = time.time() - t0
    print(f"  Generated {len(examples)} unique examples in {gen_time:.1f}s")

    # Print tactic distribution
    tactic_counts: dict[str, int] = {}
    for ex in examples:
        tactic_counts[ex.tactic] = tactic_counts.get(ex.tactic, 0) + 1
    print("  Tactic distribution:")
    for tactic, count in sorted(tactic_counts.items(), key=lambda x: -x[1]):
        print(f"    {tactic:30s}: {count:5d} ({100*count/len(examples):.1f}%)")

    # ── Step 2: Train model ──
    print(f"\n[2/4] Training ProofPredictor (~4K params)...")
    model = ProofPredictor()
    print(f"  Model parameters: {model.count_parameters():,}")

    t0 = time.time()
    result = train_model(
        model, examples,
        num_epochs=50,
        batch_size=256,
        learning_rate=0.001,
        early_stop_patience=10,
        verbose=True,
    )
    train_time = time.time() - t0
    print(f"  Training completed in {train_time:.1f}s")
    print(f"  Best epoch: {result.best_epoch + 1}")
    print(f"  Train accuracy: {result.train_acc:.4f}")
    print(f"  Validation accuracy: {result.val_acc:.4f}")

    # ── Step 3: Evaluate on held-out physics proofs ──
    print(f"\n[3/4] Evaluating on {len(PHYSICS_TEST_CASES)} held-out physics proofs...")
    eval_results = evaluate_physics_proofs(result.model, PHYSICS_TEST_CASES)

    print(f"  First-tactic accuracy:  {eval_results['first_tactic_accuracy']:.2%}")
    print(f"  3-tactic accuracy:      {eval_results['top3_tactic_accuracy']:.2%}")
    print(f"  First-tactic threshold (60%): {'✓ PASS' if eval_results['passes_first_threshold'] else '✗ FAIL'}")
    print(f"  3-tactic threshold (85%):     {'✓ PASS' if eval_results['passes_top3_threshold'] else '✗ FAIL'}")

    print("\n  Per-case results:")
    for case in eval_results["per_case_results"]:
        status1 = "✓" if case["first_correct"] else "✗"
        status3 = "✓" if case["top3_correct"] else "✗"
        print(f"    {status1} {status3}  {case['name']:35s} → {case['first_prediction']}")

    # ── Step 4: Save outputs ──
    print(f"\n[4/4] Saving outputs...")
    model_path = output_dir / "proof_predictor.pt"
    torch.save({
        "model_state_dict": result.model.state_dict(),
        "feature_names": FEATURE_NAMES,
        "tactic_vocab": TACTIC_VOCAB,
        "num_features": NUM_FEATURES,
        "num_tactics": NUM_TACTICS,
        "train_accuracy": result.train_acc,
        "val_accuracy": result.val_acc,
        "num_train_examples": len(examples),
    }, model_path)
    model_size = model_path.stat().st_size
    print(f"  Model saved to {model_path} ({model_size:,} bytes)")

    results_path = output_dir / "proof_predictor_results.json"
    full_results = {
        "training": {
            "num_synthetic_examples": len(examples),
            "train_accuracy": result.train_acc,
            "val_accuracy": result.val_acc,
            "best_epoch": result.best_epoch,
            "training_time_seconds": round(train_time, 1),
            "generation_time_seconds": round(gen_time, 1),
            "model_parameters": result.model.count_parameters(),
            "model_size_bytes": model_size,
            "tactic_distribution": tactic_counts,
        },
        "evaluation": eval_results,
        "acceptance": {
            "first_tactic_accuracy": eval_results["first_tactic_accuracy"],
            "first_tactic_threshold_0.60": eval_results["passes_first_threshold"],
            "top3_tactic_accuracy": eval_results["top3_tactic_accuracy"],
            "top3_tactic_threshold_0.85": eval_results["passes_top3_threshold"],
        },
    }
    with open(results_path, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"  Results saved to {results_path}")

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)

    return full_results


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience API for integration with existing codebase
# ═══════════════════════════════════════════════════════════════════════════════

class ProofPredictorWrapper:
    """Thin wrapper for loading and using a trained ProofPredictor.

    Usage:
        predictor = ProofPredictorWrapper("checkpoints/proof_predictor.pt")
        top_tactic = predictor.predict_best(goal="m*g*h + 0.5*m*v^2", ...)
        top3 = predictor.predict_top3(goal="m*g*h + 0.5*m*v^2", ...)
    """

    def __init__(self, checkpoint_path: str | Path | None = None):
        self.model = ProofPredictor()
        if checkpoint_path:
            self.load(checkpoint_path)
        self.model.eval()

    def load(self, checkpoint_path: str | Path):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    def predict_best(
        self,
        goal: str,
        kinematic_subs: dict[str, str] | None = None,
        has_hypothesis: bool = False,
    ) -> tuple[str, float]:
        """Predict the single best tactic.

        Returns:
            (tactic_name, confidence) tuple.
        """
        features = extract_features(goal, kinematic_subs, has_hypothesis)
        feat_tensor = features_to_tensor(features).unsqueeze(0)
        with torch.no_grad():
            probs = self.model.predict_proba(feat_tensor)[0]
            best_idx = int(torch.argmax(probs).item())
        return IDX_TO_TACTIC[best_idx], probs[best_idx].item()

    def predict_top3(
        self,
        goal: str,
        kinematic_subs: dict[str, str] | None = None,
        has_hypothesis: bool = False,
    ) -> list[tuple[str, float]]:
        """Predict top-3 tactics.

        Returns:
            List of (tactic_name, confidence) tuples ordered by confidence.
        """
        features = extract_features(goal, kinematic_subs, has_hypothesis)
        feat_tensor = features_to_tensor(features).unsqueeze(0)
        with torch.no_grad():
            probs = self.model.predict_proba(feat_tensor)[0]
            topk = torch.topk(probs, k=3)
        return [(IDX_TO_TACTIC[int(idx.item())], prob.item())
                for idx, prob in zip(topk.indices, topk.values)]


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
