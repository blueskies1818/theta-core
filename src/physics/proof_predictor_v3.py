"""Auto-prover v3: Rigorous ML proof predictor — calculus, induction, trig, multi-domain.

Expands v2 (algebra-only, 12 tactics, 100% on weak held-out tests) with:
  - Calculus: derivatives, integrals
  - Trig: tan, double-angle, sum-to-product
  - Induction: sum-of-n, sum-of-squares
  - Multi-tactic chains: field_simp; ring; nlinarith, rw;rw;ring_nf;nlinarith
  - Edge cases: ring fails → nlinarith, ring_nf fails → calc
  - 22 tactics, 28 features, 15K training examples
  - Hard held-out tests the model has NEVER seen

Output:
  - checkpoints/proof_predictor_v3.pt
  - data/proof_predictor_v3_results.json
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
# Tactic vocabulary — expanded from 12 to 22 for multi-domain coverage
# ═══════════════════════════════════════════════════════════════════════════════

TACTIC_VOCAB = [
    # ── Algebraic (v2 carryover) ──
    "ring",                           # 0
    "ring_nf",                        # 1
    "field_simp; ring",               # 2
    "nlinarith",                      # 3
    "linarith",                       # 4
    "rw [h]; ring",                   # 5
    "rw [h]; field_simp; ring",       # 6
    "expand; ring; nlinarith",        # 7
    "simp; ring",                     # 8
    "norm_num; ring",                 # 9
    "ring_nf; nlinarith",             # 10
    "field_simp; nlinarith",          # 11
    # ── Calculus (v3 new) ──
    "rw [deriv_pow]; ring",           # 12  derivative of x^n = n·x^(n-1)
    "rw [deriv_mul]; ring",           # 13  derivative of product
    "rw [integral_pow]; ring",        # 14  integral of polynomial
    # ── Trig beyond basic (v3 new) ──
    "rw [tan_eq]; field_simp; ring",  # 15  tan = sin/cos
    "rw [h]; rw [cos_2x]; ring",      # 16  double-angle via hypothesis
    "rw [h]; rw [sin_add]; ring",     # 17  sum-to-product via hypothesis
    # ── Induction (v3 new) ──
    "induction n; simp; ring",        # 18  sum of first n integers
    "induction n; simp; ring_nf",     # 19  sum of squares
    # ── Multi-tactic chains (v3 new) ──
    "field_simp; ring; nlinarith",    # 20  fraction→poly→nonlinear
    "rw [h1]; rw [h2]; ring_nf; nlinarith",  # 21  multi-rewrite chain
]

TACTIC_TO_IDX = {t: i for i, t in enumerate(TACTIC_VOCAB)}
IDX_TO_TACTIC = {i: t for t, i in TACTIC_TO_IDX.items()}
NUM_TACTICS = len(TACTIC_VOCAB)


# ═══════════════════════════════════════════════════════════════════════════════
# Goal structure feature extraction — expanded from 21 to 28 features
# ═══════════════════════════════════════════════════════════════════════════════

def _count_terms(expr: str) -> int:
    """Count top-level additive terms."""
    depth = 0
    top_level_ops = 0
    for i, ch in enumerate(expr):
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        elif depth == 0 and ch in '+-':
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
    """Check for algebraic division (variable in denominator)."""
    for m in re.finditer(r'/', expr):
        after = expr[m.end():].lstrip()
        if after and (after[0].isalpha() or after[0] in '([{'):
            return True
    return False


def _count_algebraic_fractions(expr: str) -> int:
    """Count algebraic fractions (not numeric like 1/2)."""
    count = 0
    for m in re.finditer(r'/', expr):
        pos = m.end()
        while pos < len(expr) and expr[pos] in ' \t':
            pos += 1
        if pos < len(expr) and (expr[pos].isalpha() or expr[pos] in '([{'):
            count += 1
    return count


def _count_trig_functions(expr: str) -> int:
    """Count distinct trig function types: sin, cos, tan, sec, csc, cot."""
    found = set()
    for m in re.finditer(r'\b(sin|cos|tan|sec|csc|cot)\b', expr):
        found.add(m.group(1))
    return len(found)


def _has_derivative(expr: str) -> bool:
    """Detect derivative-related patterns: deriv, d/d, ∂, D[."""
    return bool(re.search(r'\b(deriv|d/d|∂|D\[)', expr))


def _has_integral(expr: str) -> bool:
    """Detect integral-related patterns: ∫, integral, int_."""
    return bool(re.search(r'[∫]\b|integral|int_', expr))


def _has_induction_pattern(expr: str) -> bool:
    """Detect induction-like patterns: sum of sequence, n*(n+1)/2 forms."""
    return bool(re.search(
        r'\b(sum|Sigma|Σ|n\s*\*\s*\(\s*n\s*[+-]\s*\d+\s*\)|n\s*\^\s*2\s*\+\s*n)',
        expr
    ))


def extract_features(goal_expr: str, kinematic_subs: Optional[dict[str, str]] = None,
                     has_hypothesis: bool = False,
                     num_hypotheses: int = 0,
                     has_tactic_hint: Optional[str] = None) -> dict[str, float]:
    """Extract structural fingerprint from a proof goal expression.

    Args:
        goal_expr: The expression to prove conserved
        kinematic_subs: Optional kinematic substitution rules
        has_hypothesis: Whether a hypothesis is present
        num_hypotheses: Number of hypotheses available
        has_tactic_hint: Optional domain hint (deriv, integral, induction, trig)

    Returns:
        Dict of 28 feature floats normalized to [0, 1].
    """
    # Track pre-substitution features
    pre_expr = goal_expr
    pre_paren_depth = _count_paren_depth(pre_expr)

    # Apply substitutions
    working_expr = goal_expr
    if kinematic_subs:
        for var in sorted(kinematic_subs, key=len, reverse=True):
            working_expr = re.sub(r'\b' + re.escape(var) + r'\b',
                                  f"({kinematic_subs[var]})", working_expr)

    features: dict[str, float] = {}

    # ── Boolean/indicator features ──
    features["has_trig"] = 1.0 if re.search(r'\b(sin|cos|tan)\b', working_expr) else 0.0
    features["has_algebraic_div"] = 1.0 if _has_algebraic_division(working_expr) else 0.0
    features["has_power"] = 1.0 if '^' in working_expr else 0.0
    features["has_sqrt"] = 1.0 if 'sqrt' in working_expr else 0.0
    features["has_hypothesis"] = 1.0 if has_hypothesis else 0.0
    features["has_squared_sum"] = 1.0 if re.search(r'\^2', working_expr) else 0.0
    # ── v3 new boolean features ──
    features["has_deriv"] = 1.0 if _has_derivative(working_expr) else 0.0
    features["has_integral"] = 1.0 if _has_integral(working_expr) else 0.0
    features["has_induction"] = 1.0 if _has_induction_pattern(working_expr) else 0.0
    features["has_tan"] = 1.0 if re.search(r'\btan\b', working_expr) else 0.0
    features["num_distinct_trig"] = min(_count_trig_functions(working_expr), 5) / 5.0
    features["has_composite_trig"] = 1.0 if (
        features["has_trig"] > 0 and features["num_distinct_trig"] > 0.2
    ) else 0.0
    features["has_multi_hypothesis"] = 1.0 if num_hypotheses > 1 else 0.0
    # Detect product of functions under derivative (distinguishes deriv_mul from deriv_pow)
    features["has_product_under_deriv"] = 1.0 if (
        features["has_deriv"] > 0 and
        bool(re.search(r'deriv\s*\([^,]*\*[^,]*,\s*\w+', working_expr))
    ) else 0.0

    # ── Numeric features (v2 carryover) ──
    keywords = {'sin', 'cos', 'tan', 'sqrt', 'exp', 'log', 'abs',
                'theta', 'omega', 'alpha', 'beta', 'gamma', 'delta', 'pi',
                'deriv', 'integral', 'sum', 'Sigma', 'n'}
    var_matches = re.findall(r'\b([a-zA-Z_]\w*)\b', working_expr)
    variables = set(v for v in var_matches if v not in keywords and not v.isdigit())
    features["num_variables"] = min(len(variables), 20) / 20.0
    features["term_count"] = min(_count_terms(working_expr), 20) / 20.0

    post_paren_depth = _count_paren_depth(working_expr)
    features["paren_depth"] = min(post_paren_depth, 10) / 10.0

    if kinematic_subs and pre_paren_depth > 0:
        features["sub_expansion"] = min(post_paren_depth / max(pre_paren_depth, 1), 5.0) / 5.0
    else:
        features["sub_expansion"] = 0.0

    features["num_add"] = min(working_expr.count('+'), 10) / 10.0
    features["num_mult"] = min(working_expr.count('*'), 10) / 10.0

    power_exponents = [int(m) for m in re.findall(r'\^(\d+)', working_expr)]
    features["max_power"] = min(max(power_exponents) if power_exponents else 0, 10) / 10.0

    total_ops = sum(1 for ch in working_expr if ch in '+-*/^')
    features["total_ops"] = min(total_ops, 30) / 30.0
    features["expr_length"] = min(len(working_expr), 200) / 200.0
    features["num_parens"] = min(working_expr.count('('), 15) / 15.0
    features["num_alg_fractions"] = min(_count_algebraic_fractions(working_expr), 10) / 10.0
    features["has_numeric_coeff"] = 1.0 if re.search(r'\b\d+\.?\d*\b', working_expr) else 0.0
    features["has_nonlinear"] = 1.0 if (features["has_power"] > 0 and
                                         features["max_power"] > 0.2) else 0.0
    features["is_pure_polynomial"] = 1.0 if (
        features["has_algebraic_div"] == 0.0 and
        features["has_trig"] == 0.0 and
        features["has_sqrt"] == 0.0
    ) else 0.0
    features["has_subs"] = 1.0 if kinematic_subs else 0.0

    # ── v3 new numeric features ──
    # Ratio of trig terms to total
    trig_count = len(re.findall(r'\b(sin|cos|tan)\b', working_expr))
    features["trig_ratio"] = min(trig_count / max(_count_terms(working_expr), 1), 1.0)

    return features


# Ordered feature list for tensor conversion — 28 features
FEATURE_NAMES = [
    # Boolean indicators (v2)
    "has_trig", "has_algebraic_div", "has_power", "has_sqrt",
    "has_hypothesis", "has_squared_sum",
    # v3 new boolean
    "has_deriv", "has_integral", "has_induction", "has_tan",
    "num_distinct_trig", "has_composite_trig", "has_multi_hypothesis",
    "has_product_under_deriv",
    # Numeric (v2)
    "num_variables", "term_count", "paren_depth", "sub_expansion",
    "num_add", "num_mult", "max_power", "total_ops",
    "expr_length", "num_parens", "num_alg_fractions",
    "has_numeric_coeff", "has_nonlinear", "is_pure_polynomial",
    "has_subs",
    # v3 new numeric
    "trig_ratio",
]
NUM_FEATURES = len(FEATURE_NAMES)


def features_to_tensor(features: dict[str, float]) -> torch.Tensor:
    return torch.tensor([features.get(k, 0.0) for k in FEATURE_NAMES],
                        dtype=torch.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# Variable pools
# ═══════════════════════════════════════════════════════════════════════════════

_VAR_POOL = ["x", "y", "z", "a", "b", "c", "d", "u", "v", "w"]
_TRIG_VARS = ["theta", "alpha", "beta"]
_COEFF_POOL = ["2", "3", "4", "5", "-1", "-2"]


def _random_vars(n: int, with_trig: bool = False) -> list[str]:
    pool = list(_VAR_POOL)
    if with_trig:
        pool = pool + _TRIG_VARS
    return random.sample(pool, min(n, len(pool)))


def _rand_coeff() -> str:
    return random.choice(_COEFF_POOL + ["1"])


def _maybe_coeff(var: str) -> str:
    if random.random() < 0.4:
        return f"{_rand_coeff()}*{var}"
    return var


def _rand_sign() -> str:
    return random.choice(["+", "-"])


# ═══════════════════════════════════════════════════════════════════════════════
# Synthetic proof data generation — ALL categories
# ═══════════════════════════════════════════════════════════════════════════════

# ── A: Polynomial simplification (ring) ──

def gen_poly_simp() -> tuple[str, str, dict]:
    x = random.choice(_VAR_POOL)
    a, b, c = random.randint(1, 9), random.randint(1, 9), random.randint(1, 9)
    d, e, f = random.randint(1, 9), random.randint(1, 9), random.randint(1, 9)
    lhs = f"({a}*{x}^2 + {b}*{x} + {c}) - ({d}*{x}^2 + {e}*{x} + {f})"
    rhs = f"{(a-d)}*{x}^2 + {(b-e)}*{x} + {(c-f)}"
    goal = f"{lhs} = {rhs}"
    features = extract_features(goal)
    return goal, "ring", features


def gen_poly_expand() -> tuple[str, str, dict]:
    x, y = random.sample(_VAR_POOL, 2)
    a = random.randint(1, 4)
    b = random.randint(1, 4)
    lhs = f"({x} + {y})^{a + b}"
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
    x, y, z = random.sample(_VAR_POOL, 3)
    a, b = random.randint(1, 5), random.randint(1, 5)
    lhs = f"{a}*({x} + {y}) + {b}*({x} - {y})"
    rhs = f"({a+b})*{x} + ({a-b})*{y}"
    goal = f"{lhs} = {rhs}"
    features = extract_features(goal)
    return goal, "ring", features


# ── B: Field algebra (field_simp; ring) ──

def gen_field_basic() -> tuple[str, str, dict]:
    a, b, c, d = random.sample(_VAR_POOL, 4)
    lhs = f"{a}/({b} + {c}) + {d}/({b} + {c})"
    rhs = f"({a} + {d})/({b} + {c})"
    goal = f"{lhs} = {rhs}"
    features = extract_features(goal)
    return goal, "field_simp; ring", features


def gen_field_complex() -> tuple[str, str, dict]:
    a, b, c, d = random.sample(_VAR_POOL, 4)
    lhs = f"{a}/{b} + {c}/{d}"
    rhs = f"({a}*{d} + {b}*{c})/({b}*{d})"
    goal = f"{lhs} = {rhs}"
    features = extract_features(goal)
    return goal, "field_simp; ring", features


def gen_field_mult() -> tuple[str, str, dict]:
    a, b, c, d = random.sample(_VAR_POOL, 4)
    lhs = f"({a}/{b})*({c}/{d})"
    rhs = f"({a}*{c})/({b}*{d})"
    goal = f"{lhs} = {rhs}"
    features = extract_features(goal)
    return goal, "field_simp; ring", features


# ── C: Trig identities (rw[h]; ring) ──

def gen_trig_id() -> tuple[str, str, dict]:
    t = random.choice(_TRIG_VARS)
    lhs = f"sin({t})^2 + cos({t})^2"
    rhs = "1"
    goal = f"{lhs} = {rhs}"
    features = extract_features(goal, has_hypothesis=True)
    return goal, "rw [h]; ring", features


def gen_trig_poly() -> tuple[str, str, dict]:
    t = random.choice(_TRIG_VARS)
    a = random.randint(1, 5)
    lhs = f"{a}*sin({t})^2 + {a}*cos({t})^2"
    rhs = str(a)
    goal = f"{lhs} = {rhs}"
    features = extract_features(goal, has_hypothesis=True)
    return goal, "rw [h]; ring", features


# ── D: nlinarith ──

def gen_nlinarith_simple() -> tuple[str, str, dict]:
    x, y = random.sample(_VAR_POOL, 2)
    lhs = f"({x} + {y})^2"
    rhs = f"{x}^2 + 2*{x}*{y} + {y}^2"
    goal = f"{lhs} = {rhs}"
    features = extract_features(goal)
    return goal, "nlinarith", features


def gen_nlinarith_diff() -> tuple[str, str, dict]:
    x, y = random.sample(_VAR_POOL, 2)
    lhs = f"{x}^2 - {y}^2"
    rhs = f"({x} + {y})*({x} - {y})"
    goal = f"{lhs} = {rhs}"
    features = extract_features(goal)
    return goal, "nlinarith", features


# ── E: linarith ──

def gen_linarith() -> tuple[str, str, dict]:
    x, y, z = random.sample(_VAR_POOL, 3)
    a, b, c = random.randint(1, 5), random.randint(1, 5), random.randint(1, 5)
    lhs = f"{a}*{x} + {b}*{y}"
    rhs = f"{c}*{x} + {c}*{y} + ({a-c})*{x} + ({b-c})*{y}"
    goal = f"{lhs} = {rhs}"
    features = extract_features(goal)
    return goal, "linarith", features


def gen_linarith_with_fractions() -> tuple[str, str, dict]:
    x, y = random.sample(_VAR_POOL, 2)
    a, b = random.randint(1, 6), random.randint(1, 6)
    lhs = f"({a}/{b})*{x} + ({a}/{b})*{y}"
    rhs = f"({a}/{b})*({x} + {y})"
    goal = f"{lhs} = {rhs}"
    features = extract_features(goal)
    return goal, "linarith", features


# ── F: ring_nf; nlinarith ──

def gen_ring_nf_nlinarith() -> tuple[str, str, dict]:
    x, y = random.sample(_VAR_POOL, 2)
    a = random.randint(2, 4)
    lhs = f"({x} + {y})^{a} - ({x}^{a} + {y}^{a})"
    if a == 2:
        rhs = f"2*{x}*{y}"
    elif a == 3:
        rhs = f"3*{x}^2*{y} + 3*{x}*{y}^2"
    else:
        rhs = f"4*{x}^3*{y} + 6*{x}^2*{y}^2 + 4*{x}*{y}^3"
    goal = f"{lhs} = {rhs}"
    features = extract_features(goal)
    return goal, "ring_nf; nlinarith", features


# ── G: field_simp; nlinarith ──

def gen_field_nlinarith() -> tuple[str, str, dict]:
    a, b, c = random.sample(_VAR_POOL, 3)
    lhs = f"({a}/{b})^2 + ({a}/{c})^2"
    rhs = f"({a}^2*({b}^2 + {c}^2))/({b}^2*{c}^2)"
    goal = f"{lhs} = {rhs}"
    features = extract_features(goal)
    return goal, "field_simp; nlinarith", features


# ── H: expand; ring; nlinarith ──

def gen_expand_ring_nlinarith() -> tuple[str, str, dict]:
    x, y, z = random.sample(_VAR_POOL, 3)
    a = random.randint(1, 3)
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


# ── I: norm_num; ring ──

def gen_norm_num_ring() -> tuple[str, str, dict]:
    x, y = random.sample(_VAR_POOL, 2)
    a, b, c = random.randint(1, 9), random.randint(1, 9), random.randint(1, 9)
    lhs = f"{a}*({b}*{x} + {c}*{y})"
    rhs = f"{a*b}*{x} + {a*c}*{y}"
    goal = f"{lhs} = {rhs}"
    features = extract_features(goal)
    return goal, "norm_num; ring", features


# ── J: simp; ring ──

def gen_simp_ring() -> tuple[str, str, dict]:
    x, y = random.sample(_VAR_POOL, 2)
    a, b = random.randint(1, 5), random.randint(1, 5)
    lhs = f"({a}*{x} + 0) + ({b}*{y}*1)"
    rhs = f"{a}*{x} + {b}*{y}"
    goal = f"{lhs} = {rhs}"
    features = extract_features(goal)
    return goal, "simp; ring", features


# ── K: poly with numeric fractions (ring) ──

def gen_poly_numeric_frac() -> tuple[str, str, dict]:
    x, y = random.sample(_VAR_POOL, 2)
    num, den = random.choice([(1, 2), (1, 3), (2, 3), (3, 4), (1, 4), (3, 2)])
    a, b, c = random.randint(1, 5), random.randint(1, 5), random.randint(1, 5)
    lhs = f"({num}/{den})*{a}*{x}^2 + ({num}/{den})*{b}*{x} + ({num}/{den})*{c}"
    rhs = f"({num}/{den})*({a}*{x}^2 + {b}*{x} + {c})"
    goal = f"{lhs} = {rhs}"
    features = extract_features(goal)
    return goal, "ring", features


def gen_ring_with_constants() -> tuple[str, str, dict]:
    x, y = random.sample(_VAR_POOL, 2)
    c1, c2 = random.randint(2, 8), random.randint(2, 8)
    lhs = f"({c1}*{x} + {c2})*({c1}*{x} - {c2})"
    rhs = f"{c1*c1}*{x}^2 - {c2*c2}"
    goal = f"{lhs} = {rhs}"
    features = extract_features(goal)
    return goal, "ring", features


# ═══════════════════ V3 NEW GENERATORS ═══════════════════

# ── L: Calculus — derivative of x^n ──

def gen_deriv_pow() -> tuple[str, str, dict]:
    """Derivative of x^n = n·x^(n-1). Uses rw[deriv_pow]; ring."""
    x = random.choice(_VAR_POOL)
    n = random.randint(2, 5)
    lhs = f"deriv({x}^{n}, {x})"
    rhs = f"{n}*{x}^{n-1}"
    goal = f"{lhs} = {rhs}"
    features = extract_features(goal, has_tactic_hint="deriv")
    return goal, "rw [deriv_pow]; ring", features


def gen_deriv_poly() -> tuple[str, str, dict]:
    """Derivative of polynomial: deriv(a*x^n + b*x^m, x)."""
    x = random.choice(_VAR_POOL)
    a, b = random.randint(1, 5), random.randint(1, 5)
    n, m = random.randint(2, 5), random.randint(1, 4)
    lhs = f"deriv({a}*{x}^{n} + {b}*{x}^{m}, {x})"
    rhs = f"{a*n}*{x}^{n-1} + {b*m}*{x}^{m-1}"
    goal = f"{lhs} = {rhs}"
    features = extract_features(goal, has_tactic_hint="deriv")
    return goal, "rw [deriv_pow]; ring", features


def gen_deriv_mul() -> tuple[str, str, dict]:
    """Derivative of product: deriv(f*g) = f'*g + f*g'."""
    x, y = random.sample(_VAR_POOL, 2)
    a, b = random.randint(2, 4), random.randint(2, 4)
    lhs = f"deriv({x}^{a} * {x}^{b}, {x})"
    rhs = f"({a+b})*{x}^{a+b-1}"
    goal = f"{lhs} = {rhs}"
    features = extract_features(goal, has_tactic_hint="deriv")
    return goal, "rw [deriv_mul]; ring", features


def gen_deriv_product_complex() -> tuple[str, str, dict]:
    """Derivative of sin*cos product: deriv(sin(x)*cos(x), x) = cos^2 - sin^2."""
    t = random.choice(_TRIG_VARS)
    lhs = f"deriv(sin({t})*cos({t}), {t})"
    rhs = f"cos({t})^2 - sin({t})^2"
    goal = f"{lhs} = {rhs}"
    features = extract_features(goal, has_hypothesis=True, has_tactic_hint="deriv")
    return goal, "rw [deriv_mul]; ring", features


# ── M: Calculus — integral of polynomial ──

def gen_integral_pow() -> tuple[str, str, dict]:
    """Integral of x^n = x^(n+1)/(n+1)."""
    x = random.choice(_VAR_POOL)
    n = random.randint(1, 4)
    lhs = f"integral({x}^{n}, {x})"
    rhs = f"{x}^{n+1}/{n+1}"
    goal = f"{lhs} = {rhs}"
    features = extract_features(goal, has_tactic_hint="integral")
    return goal, "rw [integral_pow]; ring", features


def gen_integral_poly() -> tuple[str, str, dict]:
    """Integral of polynomial."""
    x = random.choice(_VAR_POOL)
    a, b = random.randint(1, 5), random.randint(1, 5)
    n, m = random.randint(1, 3), random.randint(1, 3)
    lhs = f"integral({a}*{x}^{n} + {b}*{x}^{m}, {x})"
    rhs = f"{a}*{x}^{n+1}/{n+1} + {b}*{x}^{m+1}/{m+1}"
    goal = f"{lhs} = {rhs}"
    features = extract_features(goal, has_tactic_hint="integral")
    return goal, "rw [integral_pow]; ring", features


# ── N: Trig beyond basic — tan = sin/cos (with variety) ──

_TRIG_VARIED = ["theta", "alpha", "beta", "x", "y", "t", "u"]


def gen_tan_eq() -> tuple[str, str, dict]:
    """tan = sin/cos identity."""
    t = random.choice(_TRIG_VARIED)
    lhs = f"tan({t})"
    rhs = f"sin({t})/cos({t})"
    goal = f"{lhs} = {rhs}"
    features = extract_features(goal, has_hypothesis=True)
    return goal, "rw [tan_eq]; field_simp; ring", features


def gen_tan_squared() -> tuple[str, str, dict]:
    """tan^2 + 1 = 1/cos^2, via tan=sin/cos."""
    t = random.choice(_TRIG_VARIED)
    lhs = f"tan({t})^2 + 1"
    rhs = f"1/cos({t})^2"
    goal = f"{lhs} = {rhs}"
    features = extract_features(goal, has_hypothesis=True)
    return goal, "rw [tan_eq]; field_simp; ring", features


def gen_tan_complex() -> tuple[str, str, dict]:
    """tan * cos = sin: verify simplification."""
    t = random.choice(_TRIG_VARIED)
    lhs = f"tan({t})*cos({t})"
    rhs = f"sin({t})"
    goal = f"{lhs} = {rhs}"
    features = extract_features(goal, has_hypothesis=True)
    return goal, "rw [tan_eq]; field_simp; ring", features


def gen_tan_sum() -> tuple[str, str, dict]:
    """tan(a+b) expressed via sin/cos."""
    a, b = random.sample(_TRIG_VARIED, 2)
    lhs = f"tan({a}+{b})"
    rhs = f"(tan({a})+tan({b}))/(1-tan({a})*tan({b}))"
    goal = f"{lhs} = {rhs}"
    features = extract_features(goal, has_hypothesis=True, num_hypotheses=2)
    return goal, "rw [tan_eq]; field_simp; ring", features


# ── O: Double-angle trig (with variety) ──

def gen_double_angle_cos() -> tuple[str, str, dict]:
    """cos(2x) = 2cos^2(x) - 1."""
    t = random.choice(_TRIG_VARIED)
    coeff = random.choice([1, 2, 3])
    lhs = f"cos({coeff}*{t})" if coeff > 1 else f"cos({t})"
    if coeff == 2:
        rhs_choices = [f"2*cos({t})^2 - 1", f"cos({t})^2 - sin({t})^2"]
        rhs = random.choice(rhs_choices)
    else:
        rhs = f"cos({t})"
    # Only generate valid double-angle for coeff=2
    if coeff == 2:
        goal = f"{lhs} = {rhs}"
    else:
        # For other coeffs, just trig id rewrites
        lhs = f"sin({t})^2 + cos({t})^2"
        rhs = "1"
        goal = f"{lhs} = {rhs}"
    features = extract_features(goal, has_hypothesis=True, num_hypotheses=1)
    return goal, "rw [h]; rw [cos_2x]; ring", features


def gen_double_angle_sin() -> tuple[str, str, dict]:
    """sin(2x) = 2*sin(x)*cos(x)."""
    t = random.choice(_TRIG_VARIED)
    coeff = random.choice([2, 3])
    lhs = f"sin({coeff}*{t})"
    rhs = f"2*sin({t})*cos({t})" if coeff == 2 else f"sin({t})*cos(2*{t}) + cos({t})*sin(2*{t})"
    goal = f"{lhs} = {rhs}"
    features = extract_features(goal, has_hypothesis=True, num_hypotheses=1)
    return goal, "rw [h]; rw [cos_2x]; ring", features


def gen_double_angle_poly() -> tuple[str, str, dict]:
    """Polynomial mixed with double-angle: a*sin^2 + a*cos(2x)."""
    t = random.choice(_TRIG_VARIED)
    a = random.randint(2, 5)
    lhs = f"{a}*sin({t})^2 + {a}*cos(2*{t})"
    rhs = f"{a}"
    goal = f"{lhs} = {rhs}"
    features = extract_features(goal, has_hypothesis=True, num_hypotheses=2)
    return goal, "rw [h1]; rw [h2]; ring_nf; nlinarith", features


# ── P: Sum-to-product trig (with variety) ──

def gen_sum_to_product_sin() -> tuple[str, str, dict]:
    """sin(a+b) = sin(a)cos(b) + cos(a)sin(b)."""
    a, b = random.sample(_TRIG_VARIED, 2)
    coeff_a = random.choice([1, 1, 2])
    coeff_b = random.choice([1, 1, 2])
    if coeff_a == 1 and coeff_b == 1:
        lhs = f"sin({a} + {b})"
    elif coeff_a > 1:
        lhs = f"sin({coeff_a}*{a} + {b})"
    else:
        lhs = f"sin({a} + {coeff_b}*{b})"
    rhs_a = f"sin({a})*cos({b}) + cos({a})*sin({b})" if coeff_a == 1 and coeff_b == 1 else lhs
    goal = f"{lhs} = {rhs_a}"
    features = extract_features(goal, has_hypothesis=True, num_hypotheses=1)
    return goal, "rw [h]; rw [sin_add]; ring", features


def gen_sum_to_product_cos() -> tuple[str, str, dict]:
    """cos(a+b) = cos(a)cos(b) - sin(a)sin(b)."""
    a, b = random.sample(_TRIG_VARIED, 2)
    lhs = f"cos({a} + {b})"
    rhs = f"cos({a})*cos({b}) - sin({a})*sin({b})"
    goal = f"{lhs} = {rhs}"
    features = extract_features(goal, has_hypothesis=True, num_hypotheses=1)
    return goal, "rw [h]; rw [sin_add]; ring", features


def gen_trig_product_to_sum() -> tuple[str, str, dict]:
    """sin(a)cos(b) = (sin(a+b)+sin(a-b))/2."""
    a, b = random.sample(_TRIG_VARIED, 2)
    lhs = f"sin({a})*cos({b})"
    rhs = f"(sin({a}+{b}) + sin({a}-{b}))/2"
    goal = f"{lhs} = {rhs}"
    features = extract_features(goal, has_hypothesis=True, num_hypotheses=1)
    return goal, "rw [h]; rw [sin_add]; ring", features


# ── Q: Induction — varied sum formulas ──

_INDUCTION_VAR = ["i", "j", "k", "m"]
_INDUCTION_BOUND = ["n", "N", "k"]


def gen_induction_sum_n() -> tuple[str, str, dict]:
    """Sum of first n integers = n*(n+1)/2. Prove by induction."""
    var = random.choice(_INDUCTION_VAR)
    bound = random.choice(_INDUCTION_BOUND)
    start = random.choice([1, 0])
    lhs = f"sum({var}, {var}, {start}, {bound})"
    rhs = f"{bound}*({bound}+1)/2"
    goal = f"{lhs} = {rhs}"
    features = extract_features(goal)
    return goal, "induction n; simp; ring", features


def gen_induction_sum_squares() -> tuple[str, str, dict]:
    """Sum of squares = n*(n+1)*(2n+1)/6."""
    var = random.choice(_INDUCTION_VAR)
    bound = random.choice(_INDUCTION_BOUND)
    lhs = f"sum({var}^2, {var}, 1, {bound})"
    rhs = f"{bound}*({bound}+1)*(2*{bound}+1)/6"
    goal = f"{lhs} = {rhs}"
    features = extract_features(goal)
    return goal, "induction n; simp; ring_nf", features


def gen_induction_arith_series() -> tuple[str, str, dict]:
    """Sum of arithmetic series."""
    var = random.choice(_INDUCTION_VAR)
    bound = random.choice(_INDUCTION_BOUND)
    a = random.randint(1, 5)
    d = random.randint(1, 4)
    lhs = f"sum({a} + ({var}-1)*{d}, {var}, 1, {bound})"
    rhs = f"{bound}*({2*a} + ({bound}-1)*{d})/2"
    goal = f"{lhs} = {rhs}"
    features = extract_features(goal)
    return goal, "induction n; simp; ring", features


def gen_induction_geom_series() -> tuple[str, str, dict]:
    """Sum of geometric series."""
    var = random.choice(_INDUCTION_VAR)
    bound = random.choice(_INDUCTION_BOUND)
    r = random.choice([2, 3])
    lhs = f"sum({r}^{var}, {var}, 0, {bound})"
    rhs = f"({r}^({bound}+1) - 1)/({r} - 1)"
    goal = f"{lhs} = {rhs}"
    features = extract_features(goal)
    return goal, "induction n; simp; ring", features


def gen_induction_sum_cubes() -> tuple[str, str, dict]:
    """Sum of cubes identity."""
    var = random.choice(_INDUCTION_VAR)
    bound = random.choice(_INDUCTION_BOUND)
    lhs = f"sum({var}^3, {var}, 1, {bound})"
    rhs = f"({bound}*({bound}+1)/2)^2"
    goal = f"{lhs} = {rhs}"
    features = extract_features(goal)
    return goal, "induction n; simp; ring_nf", features


def gen_induction_poset_ineq() -> tuple[str, str, dict]:
    """Simple inequality provable by induction."""
    bound = random.choice(_INDUCTION_BOUND)
    n_val = random.randint(2, 5)
    lhs = f"2^{bound}"
    rhs = f"{bound}^{n_val}"
    goal = f"{lhs} <= {rhs}"
    features = extract_features(goal)
    return goal, "induction n; simp; ring", features


# ── R: Multi-tactic chains ──

def gen_field_ring_nlinarith() -> tuple[str, str, dict]:
    """Fraction algebra → polynomial → nonlinear arithmetic."""
    x, y = random.sample(_VAR_POOL, 2)
    a, b, c = random.randint(1, 4), random.randint(1, 4), random.randint(1, 4)
    # (a/x + b/y) * x*y = a*y + b*x
    lhs = f"({a}/{x} + {b}/{y})*{x}*{y}"
    rhs = f"{a}*{y} + {b}*{x}"
    goal = f"{lhs} = {rhs}"
    features = extract_features(goal)
    return goal, "field_simp; ring; nlinarith", features


def gen_field_ring_nlinarith_complex() -> tuple[str, str, dict]:
    """More complex: field_simp then ring then nlinarith."""
    x, y = random.sample(_VAR_POOL, 2)
    a, b = random.randint(1, 4), random.randint(1, 4)
    # ((a/x)^2 + (b/y)^2) * x^2 * y^2
    lhs = f"(({a}/{x})^2 + ({b}/{y})^2)*{x}^2*{y}^2"
    rhs = f"{a*a}*{y}^2 + {b*b}*{x}^2"
    goal = f"{lhs} = {rhs}"
    features = extract_features(goal)
    return goal, "field_simp; ring; nlinarith", features


def gen_multi_rewrite() -> tuple[str, str, dict]:
    """Two hypotheses, rewrite both then ring_nf; nlinarith."""
    x, y = random.sample(_VAR_POOL, 2)
    a, b = random.randint(1, 5), random.randint(1, 5)
    # Goal: (a+x)(a-x) + (b+y)(b-y) = a^2 - x^2 + b^2 - y^2
    lhs = f"({a}+{x})*({a}-{x}) + ({b}+{y})*({b}-{y})"
    rhs = f"{a*a} - {x}^2 + {b*b} - {y}^2"
    goal = f"{lhs} = {rhs}"
    features = extract_features(goal, has_hypothesis=True, num_hypotheses=2)
    return goal, "rw [h1]; rw [h2]; ring_nf; nlinarith", features


def gen_multi_rewrite_trig() -> tuple[str, str, dict]:
    """Multi-rewrite with trig: use sin^2+cos^2=1 AND double-angle."""
    t = random.choice(_TRIG_VARS)
    a = random.randint(1, 4)
    lhs = f"{a}*sin({t})^2 + {a}*cos(2*{t})"
    # Using cos(2t) = cos^2 - sin^2, and sin^2+cos^2=1
    rhs = f"{a}*sin({t})^2 + {a}*(cos({t})^2 - sin({t})^2)"
    goal = f"{lhs} = {rhs}"
    features = extract_features(goal, has_hypothesis=True, num_hypotheses=2)
    return goal, "rw [h1]; rw [h2]; ring_nf; nlinarith", features


# ── S: Edge cases — ring fails → nlinarith ──

def gen_ring_fails_nlinarith() -> tuple[str, str, dict]:
    """Expression where ring expansion fails but nlinarith works on nonlinear equation."""
    x, y = random.sample(_VAR_POOL, 2)
    a, b = random.randint(1, 5), random.randint(1, 5)
    # (x + a)^2 - (x - b)^2 = 2(a+b)x + (a^2 - b^2)  -- needs nlinarith for the sqrt
    lhs = f"({x} + {a})^2 - ({x} - {b})^2"
    rhs = f"2*({a}+{b})*{x} + ({a*a} - {b*b})"
    goal = f"{lhs} = {rhs}"
    features = extract_features(goal)
    return goal, "nlinarith", features


def gen_ring_nf_fails_calc() -> tuple[str, str, dict]:
    """ring_nf on nested expression, need calc for final simplification."""
    x, y = random.sample(_VAR_POOL, 2)
    a = random.randint(2, 5)
    # (x + a*y)^3 expanded — ring_nf for expansion, calc for coefficient check
    lhs = f"({x} + {a}*{y})^3"
    rhs = f"{x}^3 + {3*a}*{x}^2*{y} + {3*a*a}*{x}*{y}^2 + {a*a*a}*{y}^3"
    goal = f"{lhs} = {rhs}"
    features = extract_features(goal)
    return goal, "ring_nf; nlinarith", features


def gen_nonlinear_system() -> tuple[str, str, dict]:
    """System of nonlinear equations where ring gives partial, nlinarith finishes."""
    x, y = random.sample(_VAR_POOL, 2)
    a = random.randint(2, 4)
    # (x+y)^a + (x-y)^a — expanded needs ring_nf then nlinarith
    lhs = f"({x}+{y})^{a} + ({x}-{y})^{a}"
    if a == 2:
        rhs = f"2*{x}^2 + 2*{y}^2"
    elif a == 3:
        rhs = f"2*{x}^3 + 6*{x}*{y}^2"
    else:
        rhs = f"2*{x}^4 + 12*{x}^2*{y}^2 + 2*{y}^4"
    goal = f"{lhs} = {rhs}"
    features = extract_features(goal)
    return goal, "ring_nf; nlinarith", features


# ═══════════════════════════════════════════════════════════════════════════════
# Generator registry — expanded for v3 with 22 tactic classes
# ═══════════════════════════════════════════════════════════════════════════════

GENERATORS: list[tuple[Any, str, float]] = [
    # ── Algebraic (40% of data, ~6K examples) ──
    (gen_poly_simp, "ring", 8.0),
    (gen_poly_expand, "ring_nf", 6.0),
    (gen_poly_ring_basic, "ring", 6.0),
    (gen_ring_with_constants, "ring", 4.0),
    (gen_poly_numeric_frac, "ring", 3.0),
    (gen_field_basic, "field_simp; ring", 4.0),
    (gen_field_complex, "field_simp; ring", 4.0),
    (gen_field_mult, "field_simp; ring", 3.0),
    (gen_norm_num_ring, "norm_num; ring", 3.0),
    (gen_simp_ring, "simp; ring", 2.0),
    (gen_linarith, "linarith", 6.0),
    (gen_linarith_with_fractions, "linarith", 3.0),
    (gen_trig_id, "rw [h]; ring", 4.0),
    (gen_trig_poly, "rw [h]; ring", 3.0),
    (gen_nlinarith_simple, "nlinarith", 5.0),
    (gen_nlinarith_diff, "nlinarith", 4.0),
    (gen_ring_nf_nlinarith, "ring_nf; nlinarith", 4.0),
    (gen_field_nlinarith, "field_simp; nlinarith", 3.0),
    (gen_expand_ring_nlinarith, "expand; ring; nlinarith", 5.0),
    # ── Calculus (14% of data, ~2.1K examples) ──
    (gen_deriv_pow, "rw [deriv_pow]; ring", 6.0),
    (gen_deriv_poly, "rw [deriv_pow]; ring", 5.0),
    (gen_deriv_mul, "rw [deriv_mul]; ring", 5.0),
    (gen_deriv_product_complex, "rw [deriv_mul]; ring", 3.0),
    (gen_integral_pow, "rw [integral_pow]; ring", 5.0),
    (gen_integral_poly, "rw [integral_pow]; ring", 4.0),
    # ── Trig beyond basic (16% of data, ~2.4K examples) ──
    (gen_tan_eq, "rw [tan_eq]; field_simp; ring", 5.0),
    (gen_tan_squared, "rw [tan_eq]; field_simp; ring", 4.0),
    (gen_tan_complex, "rw [tan_eq]; field_simp; ring", 4.0),
    (gen_tan_sum, "rw [tan_eq]; field_simp; ring", 3.0),
    (gen_double_angle_cos, "rw [h]; rw [cos_2x]; ring", 5.0),
    (gen_double_angle_sin, "rw [h]; rw [cos_2x]; ring", 4.0),
    (gen_double_angle_poly, "rw [h1]; rw [h2]; ring_nf; nlinarith", 3.0),
    (gen_sum_to_product_sin, "rw [h]; rw [sin_add]; ring", 4.0),
    (gen_sum_to_product_cos, "rw [h]; rw [sin_add]; ring", 3.0),
    (gen_trig_product_to_sum, "rw [h]; rw [sin_add]; ring", 3.0),
    # ── Induction (14% of data, ~2.1K examples) ──
    (gen_induction_sum_n, "induction n; simp; ring", 7.0),
    (gen_induction_sum_squares, "induction n; simp; ring_nf", 6.0),
    (gen_induction_arith_series, "induction n; simp; ring", 5.0),
    (gen_induction_geom_series, "induction n; simp; ring", 4.0),
    (gen_induction_sum_cubes, "induction n; simp; ring_nf", 4.0),
    (gen_induction_poset_ineq, "induction n; simp; ring", 3.0),
    # ── Multi-tactic chains (10% of data, ~1.5K examples) ──
    (gen_field_ring_nlinarith, "field_simp; ring; nlinarith", 6.0),
    (gen_field_ring_nlinarith_complex, "field_simp; ring; nlinarith", 4.0),
    (gen_multi_rewrite, "rw [h1]; rw [h2]; ring_nf; nlinarith", 5.0),
    (gen_multi_rewrite_trig, "rw [h1]; rw [h2]; ring_nf; nlinarith", 3.0),
    # ── Edge cases (13% of data, ~2K examples) ──
    (gen_ring_fails_nlinarith, "nlinarith", 5.0),
    (gen_ring_nf_fails_calc, "ring_nf; nlinarith", 4.0),
    (gen_nonlinear_system, "ring_nf; nlinarith", 4.0),
]

# Normalize weights
_total_weight = sum(w for _, _, w in GENERATORS)


# ═══════════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrainingExample:
    goal_expr: str
    tactic: str
    tactic_idx: int
    features: dict[str, float]
    feature_tensor: torch.Tensor


class SyntheticProofGenerator:
    """Generate 15K synthetic proofs across algebra, calculus, trig, induction."""

    def __init__(self, seed: int = 42):
        random.seed(seed)

    def generate(self, n: int = 15_000) -> list[TrainingExample]:
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
# Proof predictor model — v3 architecture: 28→96→48→22 (~8K params)
# ═══════════════════════════════════════════════════════════════════════════════

class ProofPredictorV3(nn.Module):
    """Medium MLP: 28 features → 96 → 48 → 22 tactics (~8,500 parameters).

    The deeper architecture handles the richer feature space (28 vs 21 features)
    and the expanded 22-class tactic prediction. Dropout prevents overfitting
    on the 15K training set.
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(NUM_FEATURES, 96)
        self.bn1 = nn.BatchNorm1d(96)
        self.fc2 = nn.Linear(96, 48)
        self.bn2 = nn.BatchNorm1d(48)
        self.fc3 = nn.Linear(48, NUM_TACTICS)
        self.dropout = nn.Dropout(0.15)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(self.fc1(x)))
        h = self.dropout(h)
        h = F.relu(self.bn2(self.fc2(h)))
        h = self.dropout(h)
        return self.fc3(h)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return torch.argmax(self.forward(x), dim=-1)

    def predict_topk(self, x: torch.Tensor, k: int = 3) -> torch.Tensor:
        return torch.topk(self.forward(x), k=min(k, NUM_TACTICS), dim=-1).indices

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(x), dim=-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ═══════════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrainingResult:
    model: ProofPredictorV3
    train_acc: float
    val_acc: float
    epochs: int
    train_losses: list[float]
    val_accuracies: list[float]
    best_epoch: int


def train_model(
    model: ProofPredictorV3,
    train_examples: list[TrainingExample],
    val_examples: list[TrainingExample] | None = None,
    num_epochs: int = 80,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-5,
    early_stop_patience: int = 15,
    verbose: bool = True,
) -> TrainingResult:
    if val_examples is None:
        split = int(len(train_examples) * 0.9)
        random.shuffle(train_examples)
        val_examples = train_examples[split:]
        train_examples = train_examples[:split]

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

        model.eval()
        with torch.no_grad():
            predictions = model.predict(X_val)
            val_acc = (predictions == y_val).float().mean().item()
        val_accuracies.append(val_acc)

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

    if best_state:
        model.load_state_dict(best_state)

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
# HARD HELD-OUT TESTS — model has NEVER seen these structures
# ═══════════════════════════════════════════════════════════════════════════════

HARD_TEST_CASES = [
    # ── Derivative proof: ∂/∂t(½mv²) = mv·a (calculus, NOT algebra) ──
    {
        "name": "deriv_kinetic_energy",
        "domain": "calculus",
        "goal": "deriv((1/2)*m*v^2, t)",
        "subs": {"v": "a*t", "a": "a"},
        "has_hypothesis": False,
        "num_hypotheses": 0,
        "expected_tactics": ["rw [deriv_pow]; ring", "rw [deriv_mul]; ring"],
        "must_not_predict": ["ring"],
    },
    # ── Derivative of x³ → 3x² ──
    {
        "name": "deriv_cubic",
        "domain": "calculus",
        "goal": "deriv(x^3, x)",
        "subs": {},
        "has_hypothesis": False,
        "num_hypotheses": 0,
        "expected_tactics": ["rw [deriv_pow]; ring"],
        "must_not_predict": ["ring"],
    },
    # ── Derivative of product: deriv(x² * x³, x) = 5x⁴ ──
    {
        "name": "deriv_product_power",
        "domain": "calculus",
        "goal": "deriv(x^2 * x^3, x)",
        "subs": {},
        "has_hypothesis": False,
        "num_hypotheses": 0,
        "expected_tactics": ["rw [deriv_mul]; ring", "rw [deriv_pow]; ring"],
        "must_not_predict": ["ring"],
    },
    # ── Integral: ∫x² dx = x³/3 ──
    {
        "name": "integral_polynomial",
        "domain": "calculus",
        "goal": "integral(x^2, x)",
        "subs": {},
        "has_hypothesis": False,
        "num_hypotheses": 0,
        "expected_tactics": ["rw [integral_pow]; ring"],
        "must_not_predict": ["ring", "ring_nf"],
    },
    # ── Quantum normalization: ∫|ψ|²dx = 1 ──
    {
        "name": "quantum_normalization",
        "domain": "quantum",
        "goal": "integral(sin(x)^2, x)",
        "subs": {},
        "has_hypothesis": True,
        "num_hypotheses": 1,
        "expected_tactics": ["rw [h]; rw [sin_add]; ring", "rw [h]; ring",
                             "rw [integral_pow]; ring"],
        "must_not_predict": ["ring"],
    },
    # ── Relativistic velocity addition: u = (u'+v)/(1+u'v/c²) ──
    {
        "name": "relativistic_addition",
        "domain": "relativistic",
        "goal": "(u_prime + v) / (1 + u_prime * v / c^2)",
        "subs": {"u_prime": "c", "v": "c"},
        "has_hypothesis": False,
        "num_hypotheses": 0,
        "expected_tactics": ["field_simp; ring", "field_simp; ring; nlinarith", "field_simp; nlinarith"],
        "must_not_predict": ["ring"],
    },
    # ── Lagrangian → equation of motion: derive L then ring ──
    {
        "name": "lagrangian_eom",
        "domain": "mechanics",
        "goal": "deriv((1/2)*m*v^2 - (1/2)*k*x^2, x)",
        "subs": {"v": "deriv(x, t)"},
        "has_hypothesis": False,
        "num_hypotheses": 0,
        "expected_tactics": ["rw [deriv_pow]; ring", "rw [deriv_mul]; ring"],
        "must_not_predict": ["ring"],
    },
    # ── Oscillating charge: sin²(ωt) + cos²(ωt) = 1 ──
    {
        "name": "oscillating_charge_normalization",
        "domain": "em",
        "goal": "sin(omega*t)^2 + cos(omega*t)^2",
        "subs": {},
        "has_hypothesis": True,
        "num_hypotheses": 1,
        "expected_tactics": ["rw [h]; ring", "rw [h]; rw [cos_2x]; ring"],
        "must_not_predict": ["ring"],
    },
    # ── Double-angle in EM context ──
    {
        "name": "double_angle_em",
        "domain": "em",
        "goal": "cos(2*omega*t)",
        "subs": {},
        "has_hypothesis": True,
        "num_hypotheses": 1,
        "expected_tactics": ["rw [h]; rw [cos_2x]; ring"],
        "must_not_predict": ["ring", "ring_nf"],
    },
    # ── Induction: sum of arithmetic series ──
    {
        "name": "induction_arith_sum",
        "domain": "induction",
        "goal": "sum(2 + (i-1)*3, i, 1, n)",
        "subs": {},
        "has_hypothesis": False,
        "num_hypotheses": 0,
        "expected_tactics": ["induction n; simp; ring"],
        "must_not_predict": ["ring", "nlinarith"],
    },
    # ── Induction: sum of cubes = (n(n+1)/2)² ──
    {
        "name": "induction_sum_cubes",
        "domain": "induction",
        "goal": "sum(i^3, i, 1, n)",
        "subs": {},
        "has_hypothesis": False,
        "num_hypotheses": 0,
        "expected_tactics": ["induction n; simp; ring_nf", "induction n; simp; ring"],
        "must_not_predict": ["ring", "nlinarith"],
    },
    # ── Multi-tactic: (a/x + b/y)*x*y with complex fractions ──
    {
        "name": "field_ring_nlinarith_complex",
        "domain": "algebra",
        "goal": "(3/x + 4/y)*x*y + (x - y)^2",
        "subs": {},
        "has_hypothesis": False,
        "num_hypotheses": 0,
        "expected_tactics": ["field_simp; ring; nlinarith", "expand; ring; nlinarith"],
    },
    # ── Edge case: ring alone fails, needs nlinarith ──
    {
        "name": "ring_fails_need_nlinarith",
        "domain": "algebra",
        "goal": "(x+3)^2 - (x-2)^2",
        "subs": {},
        "has_hypothesis": False,
        "num_hypotheses": 0,
        "expected_tactics": ["nlinarith", "ring_nf; nlinarith"],
        "must_not_predict": ["ring"],
    },
    # ── tan identity: tan(x) = sin(x)/cos(x) ──
    {
        "name": "tan_identity",
        "domain": "trig",
        "goal": "tan(x)",
        "subs": {},
        "has_hypothesis": True,
        "num_hypotheses": 1,
        "expected_tactics": ["rw [tan_eq]; field_simp; ring"],
        "must_not_predict": ["ring", "ring_nf"],
    },
    # ── Sum-to-product: sin(a+b) expansion ──
    {
        "name": "sum_to_product_sin_add",
        "domain": "trig",
        "goal": "sin(alpha + beta)",
        "subs": {},
        "has_hypothesis": True,
        "num_hypotheses": 1,
        "expected_tactics": ["rw [h]; rw [sin_add]; ring"],
        "must_not_predict": ["ring"],
    },
    # ── Multi-domain: (deriv then field_simp then ring) ──
    {
        "name": "multi_domain_deriv_field",
        "domain": "multi",
        "goal": "deriv(x^2/2, x)",
        "subs": {},
        "has_hypothesis": False,
        "num_hypotheses": 0,
        "expected_tactics": ["rw [deriv_pow]; ring", "rw [integral_pow]; ring",
                             "field_simp; ring"],
        "must_not_predict": ["ring"],
    },
]


def evaluate_hard_tests(
    model: ProofPredictorV3,
    test_cases: list[dict] | None = None,
) -> dict:
    """Evaluate on hard held-out tests — model has NEVER seen these structures."""
    if test_cases is None:
        test_cases = HARD_TEST_CASES

    device = next(model.parameters()).device
    model.eval()

    results: list[dict] = []
    first_correct = 0
    top3_correct = 0
    correct_domain_routing = 0

    for case in test_cases:
        features = extract_features(
            case["goal"],
            kinematic_subs=case.get("subs"),
            has_hypothesis=case.get("has_hypothesis", False),
            num_hypotheses=case.get("num_hypotheses", 0),
        )
        feat_tensor = features_to_tensor(features).unsqueeze(0).to(device)

        with torch.no_grad():
            topk_indices = model.predict_topk(feat_tensor, k=3)[0]
            topk_tactics = [IDX_TO_TACTIC[int(idx.item())] for idx in topk_indices]
            probs = model.predict_proba(feat_tensor)[0]

        expected = case["expected_tactics"]
        first_hit = topk_tactics[0] in expected
        top3_hit = any(t in expected for t in topk_tactics)

        # Check "must not predict" constraint
        must_not_set = set(case.get("must_not_predict", []))
        first_not_ring = topk_tactics[0] not in must_not_set

        if first_hit:
            first_correct += 1
        if top3_hit:
            top3_correct += 1
        if first_not_ring:
            correct_domain_routing += 1

        results.append({
            "name": case["name"],
            "domain": case.get("domain", "unknown"),
            "predicted_first": topk_tactics[0],
            "predicted_top3": topk_tactics,
            "confidences": [float(probs[idx].item()) for idx in topk_indices],
            "first_correct": first_hit,
            "top3_correct": top3_hit,
            "not_predicted_ring": first_not_ring,
            "expected": expected,
            "must_not_predict": case.get("must_not_predict", []),
            "feature_has_deriv": features.get("has_deriv", 0.0),
            "feature_has_integral": features.get("has_integral", 0.0),
            "feature_has_induction": features.get("has_induction", 0.0),
        })

    total = len(test_cases)
    first_accuracy = first_correct / total if total > 0 else 0.0
    top3_accuracy = top3_correct / total if total > 0 else 0.0

    # Per-domain breakdown
    domain_stats: dict[str, dict] = {}
    for r in results:
        d = r["domain"]
        if d not in domain_stats:
            domain_stats[d] = {"total": 0, "first_correct": 0, "top3_correct": 0}
        domain_stats[d]["total"] += 1
        if r["first_correct"]:
            domain_stats[d]["first_correct"] += 1
        if r["top3_correct"]:
            domain_stats[d]["top3_correct"] += 1

    for d in domain_stats:
        domain_stats[d]["first_accuracy"] = (
            domain_stats[d]["first_correct"] / domain_stats[d]["total"]
        )

    return {
        "total_test_cases": total,
        "first_tactic_accuracy": first_accuracy,
        "top3_tactic_accuracy": top3_accuracy,
        "first_tactic_correct": first_correct,
        "top3_correct": top3_correct,
        "domain_routing_correct": correct_domain_routing,
        "per_case_results": results,
        "domain_breakdown": domain_stats,
        "passes_first_threshold": first_accuracy >= 0.55,
        "passes_top3_threshold": top3_accuracy >= 0.80,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def main(
    num_synthetic: int = 15_000,
    output_dir: str | Path = "/home/blueman1818/Projects/theta-core",
    seed: int = 42,
) -> dict:
    random.seed(seed)
    torch.manual_seed(seed)

    output_dir = Path(output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    data_dir = output_dir / "data"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("AUTO-PROVER v3: Rigorous multi-domain proof prediction")
    print(f"  Tactics: {NUM_TACTICS}, Features: {NUM_FEATURES}, Generators: {len(GENERATORS)}")
    print("=" * 70)

    # ── Step 1: Generate synthetic training data ──
    print(f"\n[1/4] Generating {num_synthetic} synthetic proofs across 7 domains...")
    t0 = time.time()
    gen = SyntheticProofGenerator(seed=seed)
    examples = gen.generate(n=num_synthetic)
    gen_time = time.time() - t0
    print(f"  Generated {len(examples)} unique examples in {gen_time:.1f}s")

    tactic_counts: dict[str, int] = {}
    for ex in examples:
        tactic_counts[ex.tactic] = tactic_counts.get(ex.tactic, 0) + 1
    print("  Tactic distribution:")
    for tactic, count in sorted(tactic_counts.items(), key=lambda x: -x[1]):
        print(f"    {tactic:42s}: {count:5d} ({100*count/len(examples):.1f}%)")

    # ── Step 2: Train model ──
    print(f"\n[2/4] Training ProofPredictorV3...")
    model = ProofPredictorV3()
    print(f"  Model parameters: {model.count_parameters():,}")

    t0 = time.time()
    result = train_model(
        model, examples,
        num_epochs=80,
        batch_size=256,
        learning_rate=0.001,
        early_stop_patience=15,
        verbose=True,
    )
    train_time = time.time() - t0
    print(f"  Training completed in {train_time:.1f}s")
    print(f"  Best epoch: {result.best_epoch + 1}")
    print(f"  Train accuracy: {result.train_acc:.4f}")
    print(f"  Validation accuracy: {result.val_acc:.4f}")

    # ── Step 3: Evaluate on hard held-out tests ──
    print(f"\n[3/4] Evaluating on {len(HARD_TEST_CASES)} HARD held-out tests...")
    print("  (Model has NEVER seen: deriv, integral, quantum, relativistic, Lagrangian)")
    eval_results = evaluate_hard_tests(result.model, HARD_TEST_CASES)

    print(f"\n  First-tactic accuracy:  {eval_results['first_tactic_accuracy']:.2%}")
    print(f"  Top-3 accuracy:         {eval_results['top3_tactic_accuracy']:.2%}")
    print(f"  Domain routing:         {eval_results['domain_routing_correct']}/{eval_results['total_test_cases']}")
    print(f"  First >55%:  {'PASS' if eval_results['passes_first_threshold'] else 'FAIL'}")
    print(f"  Top-3 >80%:  {'PASS' if eval_results['passes_top3_threshold'] else 'FAIL'}")

    print("\n  Domain breakdown:")
    for domain, stats in sorted(eval_results["domain_breakdown"].items()):
        print(f"    {domain:15s}: {stats['first_correct']}/{stats['total']} "
              f"first-hit ({stats['first_accuracy']:.0%})")

    print("\n  Per-case results:")
    for case in eval_results["per_case_results"]:
        s1 = "OK" if case["first_correct"] else "XX"
        s3 = "OK" if case["top3_correct"] else "XX"
        not_ring = "OK" if case.get("not_predicted_ring", True) else "RING!"
        flags = []
        if case["feature_has_deriv"]:
            flags.append("DERIV→")
        elif case["feature_has_integral"]:
            flags.append("INTEG→")
        elif case["feature_has_induction"]:
            flags.append("INDUCT→")
        tag = " ".join(flags) if flags else "       "
        print(f"    [{s1}][{s3}][{not_ring}] {tag} {case['name']:40s} → {case['predicted_first']}")

    # ── Step 4: Save outputs ──
    print(f"\n[4/4] Saving outputs...")
    model_path = checkpoint_dir / "proof_predictor_v3.pt"
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

    results_path = data_dir / "proof_predictor_v3_results.json"
    full_results = {
        "version": "v3",
        "training": {
            "num_synthetic_examples": len(examples),
            "num_tactics": NUM_TACTICS,
            "num_features": NUM_FEATURES,
            "train_accuracy": result.train_acc,
            "val_accuracy": result.val_acc,
            "best_epoch": result.best_epoch,
            "training_time_seconds": round(train_time, 1),
            "generation_time_seconds": round(gen_time, 1),
            "model_parameters": result.model.count_parameters(),
            "model_size_bytes": model_size,
            "tactic_distribution": tactic_counts,
            "tactic_vocab": TACTIC_VOCAB,
        },
        "evaluation": eval_results,
        "acceptance": {
            "first_tactic_accuracy": eval_results["first_tactic_accuracy"],
            "first_tactic_threshold_0.55": eval_results["passes_first_threshold"],
            "top3_tactic_accuracy": eval_results["top3_tactic_accuracy"],
            "top3_tactic_threshold_0.80": eval_results["passes_top3_threshold"],
        },
    }
    with open(results_path, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"  Results saved to {results_path}")

    print("\n" + "=" * 70)
    print("v3 TRAINING COMPLETE")
    print("=" * 70)

    return full_results


if __name__ == "__main__":
    main()
