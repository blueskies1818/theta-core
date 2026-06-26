"""
Expression mutation engine — structural exploration beyond templates.

Generates mutated variants of an expression by applying purely
structural operations. All mutations are grammar-constrained:
each variant is parseable and dimension-checked by the evaluator.

Zero physics knowledge — mutations are just operator swaps, power
adjustments, function wrapping, and variable extension/contraction.
"""

from __future__ import annotations

import re
from typing import Callable

# ════════════════════════════════════════════════════════
# Mutation rules
# ════════════════════════════════════════════════════════

def mutate_expression(expr: str, all_quantities: list[str]) -> list[str]:
    """Generate structurally-mutated variants of an expression.

    Args:
        expr: A valid expression string, e.g. "a*b" or "sin(x)/sin(y)"
        all_quantities: All available quantity names

    Returns:
        List of mutated expression strings (may contain duplicates,
        caller should deduplicate).
    """
    mutations: list[str] = []

    # Extract variables used in the expression
    used_vars = _extract_vars(expr, all_quantities)
    unused_vars = [q for q in all_quantities if q not in used_vars]

    # 1. Operator mutation: swap the outermost binary operator
    mutations.extend(_mutate_operators(expr))

    # 2. Power mutation: add/remove powers
    mutations.extend(_mutate_powers(expr))

    # 3. Function wrapping
    mutations.extend(_mutate_functions(expr))

    # 4. Negation
    mutations.extend(_mutate_negation(expr))

    # 5. Reciprocal
    mutations.extend(_mutate_reciprocal(expr))

    # 6. Variable extension: add unused vars
    if unused_vars:
        mutations.extend(_mutate_extend(expr, unused_vars))

    # 7. Variable contraction: remove vars from multi-var expressions
    if len(used_vars) >= 2:
        mutations.extend(_mutate_contract(expr, used_vars, all_quantities))

    # 8. Nested operator mutation: try mutating inner operations
    mutations.extend(_mutate_inner(expr))

    # 9. Factor out: if expr is a sum/difference, try product/ratio forms
    mutations.extend(_mutate_sum_to_product(expr))

    # Filter: only return valid, non-trivial expressions
    result = []
    seen = set()
    for m in mutations:
        m = m.strip()
        if m and m != expr and m not in seen:
            # Basic sanity: must have at least one variable from quantities
            mvars = _extract_vars(m, all_quantities)
            if mvars:
                seen.add(m)
                result.append(m)

    return result


# ════════════════════════════════════════════════════════
# Internal mutation operators
# ════════════════════════════════════════════════════════

def _extract_vars(expr: str, all_quantities: list[str]) -> list[str]:
    """Extract variable names present in the expression."""
    tokens = set(re.findall(r'\b[a-zA-Z_]\w*\b', expr))
    # Filter math function names
    tokens -= {"sin", "cos", "sqrt", "exp", "log", "abs", "tan"}
    return [q for q in all_quantities if q in tokens]


def _mutate_operators(expr: str) -> list[str]:
    """Swap binary operators at the top level."""
    results = []
    ops_to_try = [("*", "/"), ("+", "-")]

    for op_pair in ops_to_try:
        a, b = op_pair
        if a in expr:
            # Replace only at top-level (not inside parens or functions)
            # Find the operator position by balancing parens
            pos = _find_top_level_op(expr, a)
            if pos >= 0:
                left = expr[:pos]
                right = expr[pos+1:]
                results.append(f"{left}{b}{right}")
        if b in expr:
            pos = _find_top_level_op(expr, b)
            if pos >= 0:
                left = expr[:pos]
                right = expr[pos+1:]
                results.append(f"{left}{a}{right}")

    return results


def _find_top_level_op(expr: str, op: str) -> int:
    """Find position of operator at top level (not inside parens)."""
    depth = 0
    for i, c in enumerate(expr):
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
        elif depth == 0 and c == op:
            # Make sure we matched the full operator (not part of ** or similar)
            if op in "+-":
                # Don't match exponent sign
                if i > 0 and expr[i-1] in 'eE':
                    continue
            return i
    return -1


def _mutate_powers(expr: str) -> list[str]:
    """Add or remove power of 2, -1."""
    results = []

    # Add square
    results.append(f"({expr})^2")

    # Add inverse
    results.append(f"({expr})^-1")

    # If expr already has ^2 at top level, try removing it
    if expr.endswith("^2"):
        inner = expr[:-2]
        if inner.startswith("(") and inner.endswith(")"):
            results.append(inner[1:-1])
        else:
            results.append(inner)

    # Try sqrt equivalent: if expr is (X)^2, try sqrt(X)
    m = re.match(r'^\((.+)\)\^2$', expr)
    if m:
        results.append(f"sqrt({m.group(1)})")

    return results


def _mutate_functions(expr: str) -> list[str]:
    """Wrap expression in transcendental functions."""
    funcs = ["sqrt", "sin", "cos", "exp", "log"]
    results = []
    for f in funcs:
        results.append(f"{f}({expr})")
    return results


def _mutate_negation(expr: str) -> list[str]:
    """Add or remove negation."""
    results = []
    if expr.startswith("-"):
        results.append(expr[1:])
    else:
        results.append(f"-({expr})")
    return results


def _mutate_reciprocal(expr: str) -> list[str]:
    """Reciprocal of the expression."""
    return [f"1/({expr})"]


def _mutate_extend(expr: str, unused_vars: list[str]) -> list[str]:
    """Extend expression by adding an unused variable via an operator."""
    results = []
    ops = ["*", "/", "+", "-"]
    for v in unused_vars:
        for op in ops:
            results.append(f"({expr}){op}{v}")
            results.append(f"{v}{op}({expr})")
    return results


def _mutate_contract(expr: str, used_vars: list[str],
                     all_quantities: list[str]) -> list[str]:
    """Contract expression by removing one variable.

    Simplistic approach: if expression is a binary op at top level,
    return each operand separately.
    """
    results = []
    for op in ["+", "-", "*", "/"]:
        pos = _find_top_level_op(expr, op)
        if pos >= 0:
            left = expr[:pos].strip()
            right = expr[pos+1:].strip()
            # Strip outer parens if present
            if left.startswith("(") and left.endswith(")"):
                left = left[1:-1]
            if right.startswith("(") and right.endswith(")"):
                right = right[1:-1]
            results.append(left)
            results.append(right)
            break
    return results


def _mutate_inner(expr: str) -> list[str]:
    """Try applying function wrapping or operator mutation to inner sub-expressions.

    For expression like (a*b)/c, try wrapping the inner part: sqrt(a*b)/c, etc.
    """
    results = []
    # Find sub-expressions in parentheses
    for m in re.finditer(r'\(([^()]+)\)', expr):
        inner = m.group(1)
        # Try function wrapping the inner expression
        for f in ["sqrt", "sin", "cos", "exp", "log"]:
            results.append(expr[:m.start()] + f"{f}({inner})" + expr[m.end():])

        # Try power on the inner expression
        results.append(expr[:m.start()] + f"({inner})^2" + expr[m.end():])

    return results


def _mutate_sum_to_product(expr: str) -> list[str]:
    """For sum/difference expressions, try product/ratio of the terms.

    a+b -> a*b, a/b, b/a
    """
    results = []
    for op, alt_ops in [("+", ["*", "/"]), ("-", ["*", "/"])]:
        pos = _find_top_level_op(expr, op)
        if pos >= 0:
            left = expr[:pos].strip()
            right = expr[pos+1:].strip()
            if left.startswith("(") and left.endswith(")"):
                left = left[1:-1]
            if right.startswith("(") and right.endswith(")"):
                right = right[1:-1]
            for aop in alt_ops:
                results.append(f"{left}{aop}{right}")
                if aop == "/":
                    results.append(f"{right}/{left}")
            break
    return results


# ════════════════════════════════════════════════════════
# Mutation-based search
# ════════════════════════════════════════════════════════

def mutation_search(
    seed_expressions: list[tuple[float, str]],
    all_quantities: list[str],
    evaluate_fn: Callable[[str], float],
    max_mutations: int = 200,
    depth: int = 3,
) -> tuple[str, float]:
    """Search for invariants by mutating promising seed expressions.

    Args:
        seed_expressions: List of (score, expression) pairs to mutate from
        all_quantities: Available quantity names
        evaluate_fn: Function that scores an expression (0-1)
        max_mutations: Maximum total mutations to evaluate
        depth: How many rounds of mutation to perform

    Returns:
        (best_expression, best_score)
    """
    best_expr = ""
    best_score = 0.0
    seen: set[str] = set()
    total_evaluated = 0

    # Start with the seed expressions
    current_generation = seed_expressions[:]

    for round_num in range(depth):
        if total_evaluated >= max_mutations:
            break

        next_generation: list[tuple[float, str]] = []

        for score, expr in current_generation:
            if total_evaluated >= max_mutations:
                break

            # Generate mutations
            mutated = mutate_expression(expr, all_quantities)

            for m_expr in mutated:
                if total_evaluated >= max_mutations:
                    break
                if m_expr in seen:
                    continue
                seen.add(m_expr)

                # Evaluate mutation
                try:
                    m_score = evaluate_fn(m_expr)
                except Exception:
                    continue

                total_evaluated += 1

                if m_score > best_score:
                    best_score = m_score
                    best_expr = m_expr

                # Keep promising mutations for next round
                if m_score > 0.5:  # threshold for further mutation
                    next_generation.append((m_score, m_expr))

        # Sort by score for next round
        next_generation.sort(key=lambda x: -x[0])
        current_generation = next_generation[:20]  # beam width

    return best_expr, best_score
