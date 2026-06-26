"""Tree-building beam search with learned guidance.

Builds expression trees bottom-up by composing sub-expressions with
operators.  Scales with complexity — any expression tree expressible
with binary operators is reachable.

Three layers of guidance prevent degenerate explosions:
  1. Search-time gates: var-set overlap, subset, dimension mismatch
  2. Quality pruning: only compose if constituents score above threshold
  3. Dimensional diversity: keep top-K per dimension, not global top-K
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any


def _is_composite(expr: str) -> bool:
    return any(op in expr for op in "+-") or (
        any(op in expr for op in "*/^") and len(expr) > 3
    )


def _parenthesize(expr: str) -> str:
    if _is_composite(expr):
        return f"({expr})"
    return expr


def _vars_of(expr: str, cache: dict[str, set[str]]) -> set[str]:
    """Extract variable names from expression, with caching."""
    if expr in cache:
        return cache[expr]
    vs = set(re.findall(r'\b[a-zA-Z_]\w*\b', expr))
    vs -= {"sin", "cos", "sqrt", "exp", "log", "abs", "tan"}
    # Remove pure numbers
    vs = {v for v in vs if not v.replace('.', '').isdigit()}
    cache[expr] = vs
    return vs


def tree_beam_search(
    seeds: list[str],
    quantities: dict[str, Any],
    observations: list,
    evaluator,
    *,
    discovery_threshold: float = 0.90,
    max_depth: int = 4,
    top_k_per_dim: int = 8,
    max_expansions: int = 5000,
    min_quality: float = 0.3,
    quality_ratio: float = 0.5,
) -> str | None:
    """Build expression trees by composing sub-expressions.

    Parameters:
        seeds: Initial sub-expressions to compose (from proposer + combos)
        quantities: Dict mapping symbol names to Dimension objects
        observations: Physical observations for scoring
        evaluator: ExpressionEvaluator instance
        discovery_threshold: Minimum score for a discovery
        max_depth: Maximum composition depth (1 = no composition)
        top_k_per_dim: Beam width per unique dimension
        max_expansions: Hard limit on total combinations tried
        min_quality: Minimum score for a term to be used in composition
        quality_ratio: Child must score at least this fraction of parent
                       to be worth composing further
    """
    from src.physics.dimensions import Dimension

    scalar_dim = Dimension.scalar()
    dim_cache: dict[str, Dimension | None] = {}
    score_cache: dict[str, float] = {}
    var_cache: dict[str, set[str]] = {}

    # ── Dimensions ──────────────────────────────────────────────────
    def _dim(expr: str) -> Dimension | None:
        if expr in dim_cache:
            return dim_cache[expr]
        try:
            ast = evaluator.parse(expr)
        except Exception:
            dim_cache[expr] = None
            return None
        from src.physics.evaluator import NumberNode, VarNode, FuncNode, BinOpNode

        def _d(node) -> Dimension | None:
            if isinstance(node, NumberNode):
                return scalar_dim
            if isinstance(node, VarNode):
                d = quantities.get(node.name)
                return d if d else scalar_dim
            if isinstance(node, FuncNode):
                return scalar_dim
            if isinstance(node, BinOpNode):
                ld, rd = _d(node.left), _d(node.right)
                if ld is None or rd is None:
                    return None
                try:
                    if node.op in ("+", "-"):
                        if isinstance(node.left, NumberNode):
                            return rd
                        if isinstance(node.right, NumberNode):
                            return ld
                        return ld if ld.compatible_with(rd) else None
                    elif node.op == "*":
                        return ld * rd
                    elif node.op == "/":
                        return ld / rd
                    elif node.op == "^":
                        if isinstance(node.right, NumberNode):
                            return ld ** float(node.right.value)
                except Exception:
                    pass
            return None

        result = _d(ast)
        dim_cache[expr] = result
        return result

    # ── Scoring ─────────────────────────────────────────────────────
    def _score(expr: str) -> float:
        if expr in score_cache:
            return score_cache[expr]
        try:
            s = sum(evaluator.score(expr, o) for o in observations) / len(observations)
        except Exception:
            s = 0.0
        score_cache[expr] = s
        return s

    # ── Initialize beam ─────────────────────────────────────────────
    # Group terms by dimension string for diversity
    beam_by_dim: dict[str, list[tuple[str, float]]] = defaultdict(list)
    all_terms: list[str] = []
    seen: set[str] = set()
    expansions = 0

    for seed in seeds:
        if seed in seen:
            continue
        seen.add(seed)
        s = _score(seed)
        if s < min_quality:
            continue
        d = _dim(seed)
        if d is None:
            continue
        dim_key = str(d)
        beam_by_dim[dim_key].append((seed, s))
        all_terms.append(seed)

    # Keep top-K per dimension
    for dim_key in beam_by_dim:
        beam_by_dim[dim_key].sort(key=lambda x: -x[1])
        beam_by_dim[dim_key] = beam_by_dim[dim_key][:top_k_per_dim]

    best_expr = ""
    best_score = 0.0
    # Track best from initialization
    for terms in beam_by_dim.values():
        for expr, s in terms:
            if s > best_score:
                best_score, best_expr = s, expr

    _OPS = ["+", "-", "*", "/", "^"]

    # ── Beam search loop ────────────────────────────────────────────
    for depth in range(2, max_depth + 1):
        # Don't break early when a seed crosses threshold — a composed
        # expression might be a better canonical form (e.g., (c*t)^2-x^2
        # is better than c/t even though c/t scores higher alone).
        # Keep composing to find the deeper structure.

        new_by_dim: dict[str, list[tuple[str, float]]] = defaultdict(list)

        # Get flat list of current beam terms with their scores
        beam_terms: list[tuple[str, float, str]] = []  # (expr, score, dim_key)
        for dim_key, terms in beam_by_dim.items():
            for expr, s in terms:
                beam_terms.append((expr, s, dim_key))

        for expr_a, score_a, dim_a in beam_terms:
            dim_a_obj = _dim(expr_a)
            if dim_a_obj is None:
                continue
            vars_a = _vars_of(expr_a, var_cache)

            for expr_b in all_terms:
                score_b = _score(expr_b)
                if score_b < min_quality:
                    continue
                dim_b_obj = _dim(expr_b)
                if dim_b_obj is None:
                    continue
                vars_b = _vars_of(expr_b, var_cache)

                # ── Quality pruning: don't compose two weak terms ──
                if score_a < min_quality or score_b < min_quality:
                    continue

                for op in _OPS:
                    expansions += 1
                    if expansions > max_expansions:
                        break

                    # ── Dimension check ────────────────────────────
                    if op in ("+", "-"):
                        if not dim_a_obj.compatible_with(dim_b_obj):
                            continue
                    elif op == "^":
                        if not dim_b_obj.is_scalar():
                            continue

                    # ── Variable-set gates ─────────────────────────
                    if op in ("+", "-", "*", "/"):
                        if vars_a and vars_b and vars_a == vars_b:
                            continue  # X/X, X+X/X
                        if vars_a and vars_b:
                            if vars_a.issubset(vars_b) or vars_b.issubset(vars_a):
                                continue  # (a+b)/a, a/(a-b), etc.
                            overlap = vars_a & vars_b
                            if overlap and op in ("*", "/"):
                                continue  # (h*nu)^2/(K_max*nu) — partial cancel

                    # Build both orderings
                    for left_first in [True, False]:
                        if left_first:
                            left, right = _parenthesize(expr_a), _parenthesize(expr_b)
                        else:
                            left, right = _parenthesize(expr_b), _parenthesize(expr_a)

                        child = f"{left}{op}{right}"
                        if child in seen:
                            continue
                        seen.add(child)

                        # ── Beam guider: skip if model predicts low value ──
                        try:
                            from src.math.beam_guider import should_explore
                            if not should_explore(left, right, op, threshold=0.2):
                                continue
                        except Exception:
                            pass  # guider not available, explore all

                        dim_child = _dim(child)
                        if dim_child is None:
                            continue

                        s = _score(child)

                        # ── Quality pruning: must beat parent scores ──
                        parent_avg = (score_a + score_b) / 2
                        if s < parent_avg * quality_ratio:
                            continue  # composition made things worse

                        dim_key = str(dim_child)
                        new_by_dim[dim_key].append((child, s))

                        if s > best_score:
                            best_score, best_expr = s, child

                if expansions > max_expansions:
                    break
            if expansions > max_expansions:
                break

        if not new_by_dim:
            break

        # ── Update beam: keep top-K per dimension ───────────────────
        for dim_key, candidates in new_by_dim.items():
            # Merge with existing
            existing = beam_by_dim.get(dim_key, [])
            combined = existing + candidates
            # Deduplicate by expression
            unique: dict[str, float] = {}
            for expr, s in combined:
                if expr not in unique or s > unique[expr]:
                    unique[expr] = s
            sorted_candidates = sorted(unique.items(), key=lambda x: -x[1])
            beam_by_dim[dim_key] = sorted_candidates[:top_k_per_dim]

            # Add new terms to all_terms pool (for future composition)
            for expr, s in sorted_candidates[:top_k_per_dim]:
                if s >= min_quality and expr not in all_terms:
                    all_terms.append(expr)

        if expansions > max_expansions:
            break

    if best_score < discovery_threshold:
        return None

    return best_expr
