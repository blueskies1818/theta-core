"""Targeted beam search for physics invariants.

Uses a beam-search approach: keep top-K expressions, expand children
from those, repeat. Supports dimension-specific search (Energy by default)
and dimension-agnostic mode (target_dim=None).

Also provides simple_invariant_search for compound-dimension invariants
and auto_discover for automatic pipeline routing.
"""

from __future__ import annotations

import heapq
import math
import re
import time
from dataclasses import dataclass
from itertools import permutations
from typing import Iterator

from src.physics.dimensions import Dimension, DimensionError
from src.physics.evaluator import ExpressionEvaluator
from src.physics.grammar import Expression
from src.physics.observations import Observation


_BINARY_OPS = ["+", "-", "*", "/", "^"]
_SCALAR_CONSTANTS = ["0", "0.5", "1", "2", "-1"]
_THREE_QTY_TEMPLATES = [
    "{a}*{b}/{c}", "{a}^2/{b}", "{a}^2/{b}^2",
    "{a}*{b}*{c}", "{a}/({b}*{c})", "{a}^2/{b}^3",
]
_SIMPLE_OPS = ["*", "/"]
_SIMPLE_POWERS = [2, -1, -2]


@dataclass
class SearchResult:
    expression: str
    score: float
    depth: int
    expansions: int
    train_constancies: list[float]
    test_constancies: list[float] | None = None

    @property
    def is_discovery(self) -> bool:
        return self.score >= 0.95


class ExpressionSearch:
    """Beam search for physics invariants.

    Key insight: invariants of a given dimension are sums of terms
    of that same dimension. Generate candidates of the target dimension,
    combine via addition, score constancy.

    Parameters
    ----------
    target_dim : str | None
        "Energy" (default), "Scalar", or None for dimension-agnostic.
    """

    def __init__(
        self,
        quantities: dict[str, Dimension],
        train_observations: list[Observation],
        *,
        scalar_constants: list[str] | None = None,
        max_depth: int = 10,
        max_expansions: int = 20_000,
        depth_discount: float = 0.95,
        top_k: int = 50,
        discovery_threshold: float = 0.95,
        min_discovery_depth: int = 2,
        target_dim: str | None = "Energy",
        time_limit: float = 30.0,
        **kwargs,
    ) -> None:
        self.quantities = quantities
        self.train_observations = train_observations
        self.scalar_constants = scalar_constants or _SCALAR_CONSTANTS
        self.max_depth = max_depth
        self.max_expansions = max_expansions
        self.depth_discount = depth_discount
        self.top_k = top_k
        self.discovery_threshold = discovery_threshold
        self.min_discovery_depth = min_discovery_depth
        self.time_limit = time_limit

        self._evaluator = ExpressionEvaluator()
        self.target_dim = (
            Dimension.named(target_dim) if target_dim is not None else None
        )

        scalar_dim = Dimension.scalar()
        self._dim_lookup: dict[str, Dimension] = dict(quantities)
        for c in self.scalar_constants:
            self._dim_lookup[c] = scalar_dim

        self._seen: set[str] = set()
        self._scored: dict[str, float] = {}
        self._expansion_count: int = 0
        self._best_expr: str = ""
        self._best_score: float = 0.0
        self._best_depth: int = 0
        self._expansions_at_last_improvement: int = 0
        self._early_stopping_patience: int = 2000

    def _score_expression(self, expr_str: str) -> float:
        if not self.train_observations:
            return 0.0
        if expr_str in self._scored:
            return self._scored[expr_str]
        try:
            scores = [
                self._evaluator.score(expr_str, obs)
                for obs in self.train_observations
            ]
        except Exception:
            return 0.0
        avg = sum(scores) / len(scores)
        self._scored[expr_str] = avg
        return avg

    def per_observation_scores(
        self, expr_str: str, observations: list[Observation]
    ) -> list[float]:
        return [self._evaluator.score(expr_str, obs) for obs in observations]

    def _dimension_of(self, expr_str: str) -> Dimension | None:
        """Compute and cache dimension of an expression string."""
        if expr_str in self._dim_lookup:
            return self._dim_lookup[expr_str]
        try:
            ast = self._evaluator.parse(expr_str)
        except Exception:
            return None
        from src.physics.evaluator import (
            NumberNode, VarNode, FuncNode, BinOpNode,
        )

        def dim_of(node) -> Dimension | None:
            if isinstance(node, NumberNode):
                return Dimension.scalar()
            if isinstance(node, VarNode):
                return self._dim_lookup.get(node.name)
            if isinstance(node, FuncNode):
                return Dimension.scalar()
            if isinstance(node, BinOpNode):
                ld = dim_of(node.left)
                rd = dim_of(node.right)
                if ld is None or rd is None:
                    return None
                try:
                    if node.op in ("+", "-"):
                        # Literal numbers are dimensionless and compatible with anything
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
                        if not rd.is_scalar():
                            return None
                        if isinstance(node.right, NumberNode):
                            return ld ** float(node.right.value)
                except DimensionError:
                    pass
            return None

        result = dim_of(ast)
        if result is not None:
            self._dim_lookup[expr_str] = result
        return result

    def _dim_ok(self, dim: Dimension | None) -> bool:
        if dim is None:
            return False
        if self.target_dim is None:
            return True
        return dim == self.target_dim

    def run(self) -> SearchResult:
        target = self.target_dim
        beam_width = self.top_k
        start_time = time.monotonic()

        beam: list[tuple[str, float, int]] = []
        all_starts = set(self.quantities.keys()) | set(self.scalar_constants)
        for name in all_starts:
            if name in self._dim_lookup:
                s = self._score_expression(name)
                beam.append((name, s, 1))

        max_beam_depth = min(self.max_depth - 1, 10)
        for gen_depth in range(2, max_beam_depth + 1):
            if self._expansion_count >= self.max_expansions:
                break
            if self.time_limit and time.monotonic() - start_time > self.time_limit:
                break

            new_candidates: dict[str, float] = {}
            leaf_names = list(self._dim_lookup.keys())
            for expr_str, prev_score, expr_depth in beam:
                self._expansion_count += 1
                if self._expansion_count >= self.max_expansions:
                    break
                for leaf_name in leaf_names:
                    for op in _BINARY_OPS:
                        for left_str, right_str in [
                            (expr_str, leaf_name), (leaf_name, expr_str),
                        ]:
                            child = self._build_str(left_str, op, right_str)
                            if child and child not in self._seen:
                                self._seen.add(child)
                                score = self._score_expression(child)
                                new_candidates[child] = score

            scored = sorted(
                [(e, s) for e, s in new_candidates.items()
                 if self._is_nontrivial(e, gen_depth)],
                key=lambda x: -x[1],
            )
            beam = [
                (expr, score, gen_depth)
                for expr, score in scored[:beam_width]
            ]
            # Boost dynamic-quantity expressions: keep some lower-scoring
            # expressions that contain dynamic quantities so they don't get
            # displaced by constants (needed for n^2 → E/n^2 discovery)
            dyn_candidates = [
                (e, s) for e, s in new_candidates.items()
                if self._is_nontrivial(e, gen_depth)
                and any(dq in e for dq in self._get_dynamic_quantities())
                and (e, s) not in set(scored[:beam_width])
            ]
            if dyn_candidates:
                dyn_candidates.sort(key=lambda x: -x[1])
                for expr, score in dyn_candidates[:max(5, beam_width // 5)]:
                    if (expr, score, gen_depth) not in beam:
                        beam.append((expr, score, gen_depth))
                beam = beam[:beam_width + 5]  # allow slight expansion

            for c in self.scalar_constants:
                if c in self._scored and not any(b[0] == c for b in beam):
                    beam.append((c, self._scored[c], 1))

            for expr_str, score, d in beam:
                dim = self._dimension_of(expr_str)
                if (
                    self._dim_ok(dim)
                    and score >= self._best_score
                    and d >= self.min_discovery_depth
                    and self._is_nontrivial(expr_str, d)
                ):
                    if self._better_than_best(expr_str, score):
                        self._best_score = score
                        self._best_expr = expr_str
                        self._best_depth = d
                        self._expansions_at_last_improvement = (
                            self._expansion_count
                        )

            if (
                self._expansion_count - self._expansions_at_last_improvement
                >= self._early_stopping_patience
            ):
                break

        # Phase 2: Combine same-dimension terms
        if self.time_limit and time.monotonic() - start_time > self.time_limit:
            return SearchResult(
                expression=self._best_expr,
                score=self._best_score,
                depth=self._best_depth,
                expansions=self._expansion_count,
                train_constancies=(
                    self.per_observation_scores(
                        self._best_expr, self.train_observations
                    )
                    if self._best_expr
                    else []
                ),
            )

        target_terms: list[tuple[str, float, int]] = []
        for expr_str, score in self._scored.items():
            dim = self._dimension_of(expr_str)
            if self._dim_ok(dim) and self._is_nontrivial(expr_str, 3):
                target_terms.append((expr_str, score, 3))
                # Check original term as a candidate for best expression
                # (terms may have been scored but not in beam during search)
                if self._better_than_best(expr_str, score):
                    self._best_score = score
                    self._best_expr = expr_str
                    self._best_depth = 3
                    self._expansions_at_last_improvement = (
                        self._expansion_count
                    )

        for expr_str, _, d in list(target_terms):
            # If expression already scores perfectly, skip scaling
            # (2*perfect, 0.5*perfect, -1*perfect are mathematically identical)
            if self._scored.get(expr_str, 0) >= 0.9999:
                continue
            for c in self.scalar_constants:
                if c in ("0", "1"):
                    continue
                child = self._build_str(c, "*", expr_str)
                if child and child not in self._seen:
                    self._seen.add(child)
                    s = self._score_expression(child)
                    self._scored[child] = s
                    cd = self._dimension_of(child)
                    if (
                        self._dim_ok(cd)
                        and s >= self._best_score
                        and self._is_nontrivial(child, d + 1)
                    ):
                        if self._better_than_best(child, s):
                            self._best_score = s
                            self._best_expr = child
                            self._best_depth = d + 1
                            self._expansions_at_last_improvement = (
                                self._expansion_count
                            )
                    target_terms.append((child, s, d + 1))

        target_terms.sort(key=lambda x: -x[1])

        for i, (a_str, a_score, a_depth) in enumerate(target_terms[:100]):
            # Early exit: if we already have a discovery, stop searching
            if self._best_score >= self.discovery_threshold:
                break
            # Time-limit check: return best found so far
            if self.time_limit and time.monotonic() - start_time > self.time_limit:
                break
            for j, (b_str, b_score, b_depth) in enumerate(target_terms[:100]):
                if i > j:
                    continue
                self._expansion_count += 1
                if self._expansion_count >= self.max_expansions:
                    break

                sum_str = f"{a_str}+{b_str}"
                if sum_str not in self._seen:
                    self._seen.add(sum_str)
                    score = self._score_expression(sum_str)
                    depth = max(a_depth, b_depth) + 1
                    if (
                        score >= self._best_score
                        and depth >= self.min_discovery_depth
                        and self._is_nontrivial(sum_str, depth)
                    ):
                        if self._better_than_best(sum_str, score):
                            self._best_score = score
                            self._best_expr = sum_str
                            self._best_depth = depth
                            self._expansions_at_last_improvement = (
                                self._expansion_count
                            )
                    if score >= self.discovery_threshold:
                        break

                rp = f"({b_str})" if "+" in b_str or "-" in b_str else b_str
                sum_str = f"{a_str}+0.5*{rp}"
                if sum_str not in self._seen:
                    self._seen.add(sum_str)
                    score = self._score_expression(sum_str)
                    depth = max(a_depth, b_depth) + 2
                    if (
                        score >= self._best_score
                        and depth >= self.min_discovery_depth
                        and self._is_nontrivial(sum_str, depth)
                    ):
                        if self._better_than_best(sum_str, score):
                            self._best_score = score
                            self._best_expr = sum_str
                            self._best_depth = depth
                            self._expansions_at_last_improvement = (
                                self._expansion_count
                            )
                    if score >= self.discovery_threshold:
                        break

                lp = f"({a_str})" if "+" in a_str or "-" in a_str else a_str
                sum_str = f"0.5*{lp}+{b_str}"
                if sum_str not in self._seen:
                    self._seen.add(sum_str)
                    score = self._score_expression(sum_str)
                    depth = max(a_depth, b_depth) + 2
                    if (
                        score >= self._best_score
                        and depth >= self.min_discovery_depth
                        and self._is_nontrivial(sum_str, depth)
                    ):
                        if self._better_than_best(sum_str, score):
                            self._best_score = score
                            self._best_expr = sum_str
                            self._best_depth = depth
                            self._expansions_at_last_improvement = (
                                self._expansion_count
                            )
                    if score >= self.discovery_threshold:
                        break

            if self._best_score >= self.discovery_threshold:
                break

        return SearchResult(
            expression=self._best_expr,
            score=self._best_score,
            depth=self._best_depth,
            expansions=self._expansion_count,
            train_constancies=(
                self.per_observation_scores(
                    self._best_expr, self.train_observations
                )
                if self._best_expr
                else []
            ),
        )

    def run_with_snapshots(self) -> Iterator[tuple[int, SearchResult]]:
        result = self.run()
        yield result.expansions, result

    def _get_dynamic_quantities(self) -> set[str]:
        dyn = set()
        for obs in self.train_observations:
            for ts in obs.timesteps:
                dyn.update(ts.keys())
        return dyn & set(self.quantities.keys())

    def _is_nontrivial(self, expr_str: str, depth: int) -> bool:
        if depth < 2:
            return False
        dyn = self._get_dynamic_quantities()
        count = sum(
            1 for name in dyn
            if re.search(r"\b" + re.escape(name) + r"\b", expr_str)
        )
        if count < 1:
            return False
        if not self._quick_nz(expr_str):
            return False
        if "^0" in expr_str:
            return False
        if re.search(r"\b(\w+)/\1\b", expr_str):
            return False
        if re.search(r"\b(\w+)-\1\b", expr_str):
            return False
        # Multiplication by zero → trivially constant (0)
        if re.search(r"(?:^|\+|-|\*)0\*(?!\d)", expr_str):
            return False
        if re.search(r"\*0(?:$|\+|-|\)|/|\*)", expr_str):
            return False
        # Cross-cancellation: a variable divided by itself elsewhere
        # e.g., x*.../x, hbar/x*omega*x.  Does NOT flag n*n/n (same term).
        clean = expr_str.replace("(", "").replace(")", "")
        if "/" in clean:
            # Split on / to find divisor sections.  A divisor is the first
            # variable token after each /.  Only flag if the same variable
            # appears OUTSIDE its own divisor section (as a separate factor).
            sections = clean.split("/")
            for si in range(1, len(sections)):
                section = sections[si]
                tokens = re.findall(r"\b[a-zA-Z_]\w*\b", section)
                if tokens:
                    divisor = tokens[0]
                    # Check OTHER sections
                    other = "/".join(sections[:si] + sections[si+1:])
                    if re.search(r"\b" + re.escape(divisor) + r"\b", other):
                        return False
                    # Also check if divisor reappears later in SAME section
                    # (catches 1/v*1*v where v*1*v has v after the divisor)
                    if len(tokens) > 1 and divisor in tokens[1:]:
                        return False
        # Parameter-only expression (no dynamic quantities used)
        # e.g., c^2, h*h, R*2 — trivially constant
        all_vars = set(re.findall(r"\b[a-zA-Z_]\w*\b", expr_str))
        all_vars -= {"sin", "cos", "sqrt", "exp", "log", "abs"}  # function names
        dyn_set = set(dyn)
        if all_vars and not (all_vars & dyn_set):
            return False
        # Reordered self-cancellation: a*b-b*a (same factors, different order)
        if "-" in expr_str:
            parts = expr_str.split("-", 1)
            if len(parts) == 2:
                left_vars = set(re.findall(r"\b[a-zA-Z_]\w*\b", parts[0]))
                right_vars = set(re.findall(r"\b[a-zA-Z_]\w*\b", parts[1]))
                if left_vars and left_vars == right_vars:
                    return False
        # Generalized self-subtraction: k*a*b - c*a*b where k == c
        # (coefficient before same set of variables)
        m = re.match(r"^([\d.]+)\*(.+)-(\1)\*\2$", expr_str.replace(" ", ""))
        if m:
            return False
        return True

    def _quick_nz(self, expr_str: str) -> bool:
        if not self.train_observations:
            return True
        from src.physics.evaluator import evaluate_node
        for obs in self.train_observations[:3]:
            try:
                ast = self._evaluator.parse(expr_str)
                for ts in obs.timesteps:
                    val = evaluate_node(ast, {**obs.parameters, **ts})
                    if isinstance(val, (int, float)) and abs(val) > 1e-12:
                        return True
            except Exception:
                continue
        return False

    def _build_str(self, left: str, op: str, right: str) -> str | None:
        child = f"{left}{op}{right}"
        if child in self._seen:
            return None
        ld = self._dimension_of(left)
        rd = self._dimension_of(right)
        if ld is None or rd is None:
            return None
        try:
            if op in ("+", "-"):
                if not ld.compatible_with(rd):
                    return None
            elif op == "^":
                if not rd.is_scalar():
                    return None
        except DimensionError:
            return None
        return child

    def _count_terms(self, expr_str: str) -> int:
        """Count independent additive terms at top level.

        E^2-p^2 → 2,  E/gamma → 1,  E → 1.
        Multi-term invariants (sums of distinct sub-expressions) are
        structurally more interesting than single-term ratios/products.
        """
        depth = 0
        count = 1
        for c in expr_str:
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
            elif c in ('+', '-') and depth == 0:
                count += 1
        return count

    def _better_than_best(self, expr_str: str, score: float) -> bool:
        """Compare expression against current best with multi-term tiebreaker.

        Heuristic: a multi-term invariant (E^2-p^2) is more fundamental
        than a single-term consequence (E/gamma), even when both are
        perfectly constant.  When scores are tied, prefer more terms.
        If terms also tie, prefer shorter.
        """
        if score > self._best_score:
            return True
        if abs(score - self._best_score) < 1e-9:
            new_terms = self._count_terms(expr_str)
            old_terms = self._count_terms(self._best_expr)
            if new_terms > old_terms:
                return True
            if new_terms == old_terms and len(expr_str) < len(self._best_expr):
                return True
        return False

    def _expr_depth(self, expr_str: str) -> int:
        return 1 + sum(1 for c in expr_str if c in "+-*/^")

    @property
    def best_expression(self) -> str:
        return self._best_expr

    @property
    def best_score(self) -> float:
        return self._best_score

    @property
    def expansion_count(self) -> int:
        return self._expansion_count

    @property
    def discovered(self) -> bool:
        return (
            self._best_score >= self.discovery_threshold
            and self._best_depth >= self.min_discovery_depth
        )


# ════════════════════════════════════════════════════════════
# Simple invariant search — for compound-dimension invariants
# ════════════════════════════════════════════════════════════


def _count_terms_module(expr_str: str) -> int:
    """Count independent additive terms at top level (module-level helper)."""
    if not expr_str:
        return 0
    depth = 0
    count = 1
    for c in expr_str:
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
        elif c in ('+', '-') and depth == 0:
            count += 1
    return count


def simple_invariant_search(
    quantities: dict[str, Dimension],
    observations: list[Observation],
    *,
    max_pairs: int = 200,
    discovery_threshold: float = 0.90,
) -> SearchResult:
    """Search for simple invariants: ratios, products, and powers.

    Handles compound-dimension invariants (v/d, λ*v, T²/a³) that
    ExpressionSearch cannot find. O(n²) — very fast.
    """
    evaluator = ExpressionEvaluator()
    qnames = list(quantities.keys())
    best_expr = ""
    best_score = 0.0
    best_depth = 1
    expansions = 0
    train_constancies: list[float] = []

    _DEPTH_BONUS = 1.001

    def _ed(expr_str: str) -> int:
        return 1 + sum(1 for c in expr_str if c in "+-*/^")

    def _adj_score(raw: float, d: int) -> float:
        return raw * _DEPTH_BONUS if d >= 2 else raw

    def _is_trivial(expr_str: str) -> bool:
        if "^0" in expr_str:
            return True
        if re.search(r"\b(\w+)/\1\b", expr_str):
            return True
        if re.search(r"\b(\w+)-\1\b", expr_str):
            return True
        return False

    def _count_terms(expr_str: str) -> int:
        depth = 0
        count = 1
        for c in expr_str:
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
            elif c in ('+', '-') and depth == 0:
                count += 1
        return count

    def _better_than(expr: str, raw: float, d: int) -> bool:
        # Compare raw constancy first (not adjusted — avoids amplifying
        # floating-point noise through the depth multiplier).
        if raw > best_score + 1e-9:
            return True
        if abs(raw - best_score) < 1e-6:
            # Raw scores tied.  Multi-term invariants are more fundamental.
            new_terms = _count_terms(expr)
            old_terms = _count_terms(best_expr) if best_expr else 0
            if new_terms > old_terms:
                return True
            if new_terms == old_terms:
                if d > best_depth:
                    return True
                if d == best_depth and len(expr) < len(best_expr):
                    return True
        return False

    # Single quantities
    for name in qnames:
        if _is_trivial(name):
            continue
        expansions += 1
        d = _ed(name)
        try:
            scores = [evaluator.score(name, obs) for obs in observations]
            avg = sum(scores) / len(scores) if scores else 0.0
        except Exception:
            avg = 0.0
        if _better_than(name, avg, d):
            best_score, best_expr, best_depth = avg, name, d
            train_constancies = scores

    # Pairs with *, /
    for i, a in enumerate(qnames):
        for j, b in enumerate(qnames):
            if i == j:
                continue
            for op in _SIMPLE_OPS:
                expr = f"{a}{op}{b}"
                if _is_trivial(expr):
                    continue
                expansions += 1
                if expansions > max_pairs:
                    break
                d = _ed(expr)
                try:
                    scores = [
                        evaluator.score(expr, obs) for obs in observations
                    ]
                    avg = sum(scores) / len(scores) if scores else 0.0
                except Exception:
                    avg = 0.0
                if _better_than(expr, avg, d):
                    best_score, best_expr, best_depth = avg, expr, d
                    train_constancies = scores
            if expansions > max_pairs:
                break
        if expansions > max_pairs:
            break

    # Powers
    for name in qnames:
        dim = quantities.get(name)
        if dim is None or dim.is_scalar():
            continue
        for p in _SIMPLE_POWERS:
            expr = f"{name}^{p}" if p > 0 else f"{name}^{p}"
            if _is_trivial(expr):
                continue
            expansions += 1
            if expansions > max_pairs:
                break
            d = _ed(expr)
            try:
                scores = [
                    evaluator.score(expr, obs) for obs in observations
                ]
                avg = sum(scores) / len(scores) if scores else 0.0
            except Exception:
                avg = 0.0
            if _better_than(expr, avg, d):
                best_score, best_expr, best_depth = avg, expr, d
                train_constancies = scores
        if expansions > max_pairs:
            break

    # Squared differences: q1^2-q2^2 for same-dimension quantities.
    # Catches multi-term invariants like E^2-p^2 (Lorentz scalar)
    # that simple ratios/products miss.
    for i, a in enumerate(qnames):
        dim_a = quantities.get(a)
        if dim_a is None or dim_a.is_scalar():
            continue
        for j, b in enumerate(qnames):
            if i >= j:
                continue
            dim_b = quantities.get(b)
            if dim_b is None or dim_a != dim_b:
                continue
            expr = f"{a}^2-{b}^2"
            if _is_trivial(expr):
                continue
            expansions += 1
            if expansions > max_pairs:
                break
            d = _ed(expr)
            try:
                scores = [
                    evaluator.score(expr, obs) for obs in observations
                ]
                avg = sum(scores) / len(scores) if scores else 0.0
            except Exception:
                avg = 0.0
            if _better_than(expr, avg, d):
                best_score, best_expr, best_depth = avg, expr, d
                train_constancies = scores
        if expansions > max_pairs:
            break

    # 3-quantity templates
    if len(qnames) >= 3:
        for a, b, c in permutations(qnames, 3):
            for template in _THREE_QTY_TEMPLATES:
                expr = template.format(a=a, b=b, c=c)
                if _is_trivial(expr):
                    continue
                expansions += 1
                if expansions > max_pairs:
                    break
                d = _ed(expr)
                try:
                    scores = [
                        evaluator.score(expr, obs) for obs in observations
                    ]
                    avg = sum(scores) / len(scores) if scores else 0.0
                except Exception:
                    avg = 0.0
                if _better_than(expr, avg, d):
                    best_score, best_expr, best_depth = avg, expr, d
                    train_constancies = scores
            if expansions > max_pairs:
                break

    return SearchResult(
        expression=best_expr,
        score=best_score,
        depth=best_depth,
        expansions=expansions,
        train_constancies=train_constancies,
    )


# ════════════════════════════════════════════════════════════
# Auto-routing discovery
# ════════════════════════════════════════════════════════════


def _expr_varies(
    expr_str: str,
    evaluator: ExpressionEvaluator,
    observations: list[Observation],
) -> bool:
    """Check if an expression's value varies across observations."""
    if not expr_str or not observations:
        return False
    try:
        vals = [
            evaluator.score(expr_str, obs) for obs in observations[:1]
        ]
        if not vals:
            return False
        vals2 = []
        for obs in observations:
            try:
                ast = evaluator.parse(expr_str)
                from src.physics.evaluator import evaluate_node
                for ts in obs.timesteps:
                    v = evaluate_node(ast, {**obs.parameters, **ts})
                    if isinstance(v, (int, float)):
                        vals2.append(v)
            except Exception:
                continue
        if len(vals2) < 2:
            return False
        mean = sum(vals2) / len(vals2)
        if abs(mean) < 1e-12:
            return False
        var = sum((v - mean) ** 2 for v in vals2) / len(vals2)
        return math.sqrt(var) / abs(mean) > 1e-6
    except Exception:
        return True


def _is_simple_ratio(expr: str) -> bool:
    if not expr:
        return False
    op_count = sum(1 for c in expr if c in "*/")
    sum_count = sum(1 for c in expr if c in "+-")
    return op_count <= 1 and sum_count == 0 and "^" not in expr


def _neural_template_search(
    quantities: dict[str, Dimension],
    observations: list[Observation],
    *,
    discovery_threshold: float = 0.90,
) -> SearchResult | None:
    """Use neural template generators to propose invariants for complex domains.

    The domain classifier identifies which physics domains are active.
    For each active domain with a trained template generator checkpoint,
    the generator maps the domain's quantity symbols to a candidate
    expression.  Candidates are scored and the best is returned.

    This is the neural half of the hybrid system — handles complex
    nested expressions (parentheses, grouped powers, nested fractions)
    that deterministic beam search cannot express.
    """
    try:
        from pathlib import Path
        import torch

        from src.physics.composer import (
            load_domain_classifier, load_domain_generator,
            QUANTITY_VOCAB, QTY_TO_IDX, DOMAINS, DOMAIN_QUANTITY_KEY,
            DOMAIN_QUANTITIES,
            quantities_to_tensor, detokenize_expression,
            assign_domain_labels,
        )

        checkpoint_dir = Path(__file__).parent.parent.parent / "checkpoints"

        # Detect which domains are active from the quantity symbols
        syms = set(quantities.keys())
        domain_labels = assign_domain_labels(list(syms))
        if isinstance(domain_labels, torch.Tensor):
            domain_labels = domain_labels.tolist()

        active_domains = [
            DOMAINS[i] for i, label in enumerate(domain_labels)
            if label == 1
        ]
        if not active_domains:
            return None

        evaluator = ExpressionEvaluator()
        best_expr = ""
        best_score = 0.0
        candidates: list[str] = []

        # For each active domain, try the neural template generator
        for domain in active_domains:
            gen_path = checkpoint_dir / f"{domain}_template.pt"
            if not gen_path.exists():
                continue
            try:
                generator = load_domain_generator(str(gen_path))
            except Exception:
                continue

            # Build source: domain-specific quantities present in the observation.
            # Filtering to domain-relevant quantities lets the generator
            # distinguish different invariant types — e.g.,
            # {c,t,x} → (c*t)^2-x^2  vs  {E,p,c,m} → E^2-(p*c)^2.
            domain_qty_names = sorted(
                q for q in DOMAIN_QUANTITIES.get(domain, [])
                if q in quantities
            )
            if not domain_qty_names:
                continue

            src_tensor = quantities_to_tensor(
                domain_qty_names, max_len=8
            ).unsqueeze(0)  # [1, src_len]

            with torch.no_grad():
                token_seqs = generator.generate(
                    src_tensor, max_len=32, temperature=0.0,
                )
            for seq in token_seqs:
                expr = detokenize_expression(seq)
                if expr and expr not in candidates:
                    candidates.append(expr)

        if not candidates:
            return None

        # Score all candidates, preferring canonical forms when constancy is close
        candidate_scores: list[tuple[str, float, float]] = []
        for expr in candidates:
            scores = [evaluator.score(expr, obs) for obs in observations]
            avg = sum(scores) / len(scores) if scores else 0.0
            candidate_scores.append((expr, avg, 0.0))

        if not candidate_scores:
            return None

        # If multiple candidates score similarly, prefer the most canonical form
        try:
            from src.physics.canonicalizer import create_pre1905_canonicalizer
            canonicalizer = create_pre1905_canonicalizer()
            for i, (expr, const, _) in enumerate(candidate_scores):
                c_score = canonicalizer.score(expr)
                candidate_scores[i] = (expr, const, c_score)
            # Combined score: 0.7 constancy + 0.3 canonical form
            candidate_scores.sort(
                key=lambda x: 0.7 * x[1] + 0.3 * x[2],
                reverse=True,
            )
        except Exception:
            pass  # canonicalizer unavailable, use constancy only

        best_expr, best_score, _ = candidate_scores[0]

        return SearchResult(
            expression=best_expr,
            score=best_score,
            depth=1,
            expansions=len(candidates),
            train_constancies=[
                evaluator.score(best_expr, obs) for obs in observations
            ],
        )
    except Exception:
        return None


def auto_discover(
    quantities: dict[str, Dimension],
    observations: list[Observation],
    known_invariant: str | None = None,
    *,
    discovery_threshold: float = 0.90,
    beam_expansions: int = 2000,
    _no_regime_split: bool = False,
    _no_neural_templates: bool = False,
) -> SearchResult:
    """Automatically select and run the best discovery pipeline.

    1. Classify invariant dimension from known_invariant
    2. Simple ratios/products → simple_invariant_search
    3. Energy multi-term → ExpressionSearch
    4. Compound/dimension-agnostic → ExpressionSearch (target_dim=None)
    5. (New) Regime discovery: if best global expression scores < 0.90,
       split observations into regimes and re-discover per regime.
    """
    evaluator = ExpressionEvaluator()
    best_result = SearchResult(
        expression="", score=0.0, depth=0, expansions=0, train_constancies=[]
    )

    target_dim = "Energy"
    if known_invariant:
        dim_lookup = dict(quantities)
        for c in ["0", "0.5", "1", "2", "-1"]:
            dim_lookup[c] = Dimension.scalar()

        try:
            ast = evaluator.parse(known_invariant)
            from src.physics.evaluator import (
                NumberNode, VarNode, BinOpNode,
            )

            def dim_of(node) -> Dimension | None:
                if isinstance(node, NumberNode):
                    return Dimension.scalar()
                if isinstance(node, VarNode):
                    d = dim_lookup.get(node.name)
                    if d is None:
                        return Dimension.scalar()
                    return d
                if isinstance(node, BinOpNode):
                    ld = dim_of(node.left)
                    rd = dim_of(node.right)
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

            d = dim_of(ast)
            if d is not None:
                dim_name = str(d)
                if dim_name == "Energy":
                    target_dim = "Energy"
                elif dim_name == "Scalar":
                    target_dim = "Scalar"
                else:
                    target_dim = "compound"
        except Exception:
            target_dim = None

    # Pipeline 0: Neural template generators (for complex invariants).
    # Domain-specific transformer decoders that learned to map quantity
    # sets to expression templates.  They handle the complex nested
    # formulas (parentheses, powers of groups, nested fractions) that
    # deterministic search cannot express.
    #
    # NOTE: When _no_neural_templates=True, this pipeline is skipped.
    # The system must rely on beam search + simple search alone.
    # This is the honest mode — templates may encode domain knowledge
    # from the developer that the system hasn't genuinely discovered.
    neural_result = None
    if not _no_neural_templates:
        neural_result = _neural_template_search(
            quantities, observations,
            discovery_threshold=discovery_threshold,
        )
    if neural_result is not None:
        if neural_result.score > best_result.score:
            best_result = neural_result
        # Apply canonical refinement to neural results
        refined = _refine_canonical(neural_result, evaluator, observations)
        if refined.score >= discovery_threshold:
            return refined

    # Pipeline 1: Simple search for simple ratios/products
    use_simple = target_dim in ("compound", "Scalar")
    if not use_simple and target_dim == "Energy" and known_invariant:
        if _is_simple_ratio(known_invariant):
            use_simple = True

    if use_simple:
        result = simple_invariant_search(
            quantities, observations,
            discovery_threshold=discovery_threshold,
        )
        refined = _refine_canonical(result, evaluator, observations)
        if refined.score >= discovery_threshold:
            return refined
        if result.score > best_result.score:
            best_result = result

    # Pipeline 2: Beam search
    search_target = (
        target_dim if target_dim in ("Energy", "Scalar") else None
    )
    search = ExpressionSearch(
        quantities=quantities,
        train_observations=observations,
        max_depth=8,
        max_expansions=beam_expansions,
        discovery_threshold=discovery_threshold,
        top_k=20,
        target_dim=search_target,
    )
    result = search.run()
    refined = _refine_canonical(result, evaluator, observations)
    if refined.score >= discovery_threshold:
        return refined
    if result.score > best_result.score:
        best_result = result

    # Pipeline 3: Grouped-quantity metric discovery (only as fallback)
    # Only run if no discovery yet AND there are multiple observations
    # (grouped detection needs across-observation variation patterns)
    if best_result.score < discovery_threshold and len(observations) >= 2:
        try:
            from src.physics.hidden_variables import (
                run_grouped_metric_discovery,
            )
            grouped_results = run_grouped_metric_discovery(
                quantities, observations,
                domain=getattr(observations[0], 'name', 'unknown'),
                discovery_threshold=discovery_threshold,
                use_mlp=False,  # rule-based only, no MLP training needed
            )
            for gr in grouped_results:
                if gr.best_constancy > best_result.score:
                    best_result = SearchResult(
                        expression=gr.best_invariant or "",
                        score=gr.best_constancy,
                        depth=0,
                        expansions=0,
                        train_constancies=[],
                    )
                if best_result.is_discovery:
                    break
        except Exception:
            pass  # grouped detection may fail gracefully

    # Pipeline 3.5: Regime-discovery fallback.
    # If no global invariant found and regime splitting is not suppressed,
    # attempt to split observations into regimes where different invariants
    # may hold (e.g., classical vs relativistic velocity regimes).
    if (
        not _no_regime_split
        and best_result.score < discovery_threshold
        and best_result.expression
    ):
        regime_result = _attempt_regime_discovery(
            quantities, observations,
            best_expr=best_result.expression,
            best_score=best_result.score,
            evaluator=evaluator,
            discovery_threshold=discovery_threshold,
            beam_expansions=beam_expansions,
        )
        if regime_result is not None and regime_result.score > best_result.score:
            best_result = regime_result

    # Final fallback: refine whatever we found
    best_result = _refine_canonical(best_result, evaluator, observations)

    return best_result


# ════════════════════════════════════════════════════════════
# Regime-discovery loop
# ════════════════════════════════════════════════════════════


def _attempt_regime_discovery(
    quantities: dict[str, Dimension],
    observations: list[Observation],
    best_expr: str,
    best_score: float,
    *,
    evaluator: ExpressionEvaluator | None = None,
    discovery_threshold: float = 0.90,
    beam_expansions: int = 2000,
    min_regime_size: int = 3,
    test_split_ratio: float = 0.3,
) -> SearchResult | None:
    """Attempt to split observations into regimes and re-discover per regime.

    When the best global expression scores below the discovery threshold,
    the invariant may only hold in a subset of observations — for example,
    classical mechanics holds at low velocities while relativistic
    corrections are needed at high velocities.  This function:

    1. Scores the best expression per-observation to find WHERE it breaks.
    2. Detects a threshold in per-observation residuals by trying each
       quantity as a sort key and finding the largest score gap.
    3. Splits observations into two regimes at the threshold.
    4. Re-runs the discovery pipeline on each regime independently.
    5. Verifies each discovered invariant on a held-out test split.
    6. Enforces anti-hacking: minimum 3 data points per regime,
       train/test generalization required.

    Parameters
    ----------
    quantities : dict[str, Dimension]
        Quantity dimension lookup.
    observations : list[Observation]
        All available observations.
    best_expr : str
        The best expression found by the global pipeline.
    best_score : float
        The global score of *best_expr*.
    evaluator : ExpressionEvaluator | None
        Reusable evaluator instance.
    discovery_threshold : float
        Minimum score for a discovery.
    beam_expansions : int
        Max expansions for sub-regime searches.
    min_regime_size : int
        Minimum observations per regime (anti-hacking).
    test_split_ratio : float
        Fraction of each regime's observations held out for verification.

    Returns
    -------
    SearchResult or None
        The best discovery across regimes, or None if regime splitting
        does not produce a verified invariant.
    """
    if evaluator is None:
        evaluator = ExpressionEvaluator()

    if not best_expr or best_score >= discovery_threshold:
        return None

    if len(observations) < 2 * (min_regime_size + 1):
        # Need enough data for train+test in each regime.
        return None

    # Step 1-2: Score per-observation and detect threshold.
    from src.physics.evaluator import find_regime_threshold

    split = find_regime_threshold(
        best_expr, observations, evaluator,
        min_regime_size=min_regime_size + 1,  # leave room for test split
    )
    if split is None:
        return None

    # Require a meaningful gap — at least 0.15 score difference.
    if split["gap"] < 0.15:
        return None

    regime_a = split["regime_a_obs"]
    regime_b = split["regime_b_obs"]

    # Step 3-4: Split each regime into train/test, re-run discovery on train.
    import random as _random
    _rng = _random.Random(42)  # deterministic split

    best_regime_result: SearchResult | None = None

    for regime_label, regime_obs in [("A", regime_a), ("B", regime_b)]:
        if len(regime_obs) < min_regime_size + 1:
            continue

        # Shuffle and split: train / test.
        shuffled = list(regime_obs)
        _rng.shuffle(shuffled)
        test_count = max(1, int(len(shuffled) * test_split_ratio))
        train_obs = shuffled[test_count:]
        test_obs = shuffled[:test_count]

        if len(train_obs) < min_regime_size or len(test_obs) < 1:
            continue

        # Run discovery on the regime's training split without further
        # regime recursion (avoid infinite loop).
        regime_result = auto_discover(
            quantities, train_obs,
            known_invariant=None,
            discovery_threshold=discovery_threshold,
            beam_expansions=beam_expansions,
            _no_regime_split=True,
        )

        if not regime_result.expression:
            continue

        # Step 5: Verify on the held-out test split.
        test_scores = evaluator.score_per_observation(
            regime_result.expression, test_obs,
        )
        test_mean = sum(test_scores) / len(test_scores) if test_scores else 0.0

        # Require the test score to also meet the discovery threshold.
        if test_mean < discovery_threshold:
            continue

        # Step 6: Anti-hacking — the train and test scores must be
        # consistent (no overfitting to the specific split).
        train_mean = sum(regime_result.train_constancies) / len(
            regime_result.train_constancies
        ) if regime_result.train_constancies else 0.0

        if abs(train_mean - test_mean) > 0.20:
            # Suspiciously large generalization gap — overfit artifact.
            continue

        # This regime produced a verified invariant.
        regime_result = SearchResult(
            expression=regime_result.expression,
            score=test_mean,  # use test score as the honest metric
            depth=regime_result.depth,
            expansions=regime_result.expansions,
            train_constancies=regime_result.train_constancies,
            test_constancies=test_scores,
        )

        if best_regime_result is None or (
            regime_result.score > best_regime_result.score
        ):
            best_regime_result = regime_result

    return best_regime_result


def _refine_canonical(
    result: SearchResult,
    evaluator: ExpressionEvaluator,
    observations: list[Observation],
) -> SearchResult:
    """Try dimensionally-canonical alternate forms of the discovered expression.

    If the pipeline found n/E at constancy 1.0, the canonicalizer knows
    E/n is the preferred form.  Try it — if it also scores well, use it.
    """
    if not result.expression or result.score < 0.90:
        return result

    try:
        from src.physics.canonicalizer import (
            create_pre1905_canonicalizer, _tokenize, _ALL_QTY_DIMS, _dim_weight,
            _split_terms,
        )
        canonicalizer = create_pre1905_canonicalizer()
        current_canon = canonicalizer.score(result.expression)

        # If already canonical, don't change
        if current_canon >= 0.95:
            return result

        # Generate alternates by trying dimension-ordered variants
        alternates: list[str] = []
        terms = _split_terms(result.expression)
        for term in terms:
            tokens = _tokenize(term)
            qty_symbols = [t for t in tokens if t in _ALL_QTY_DIMS]
            if len(qty_symbols) >= 2:
                weights = [(_dim_weight(q), q) for q in qty_symbols]
                # Check if weights are in descending order
                if any(weights[i][0] < weights[i+1][0] for i in range(len(weights)-1)):
                    # Propose the dimension-ordered version
                    ordered = sorted(weights, key=lambda x: -x[0])
                    ordered_str = term
                    for i in range(len(qty_symbols)):
                        ordered_str = ordered_str.replace(
                            qty_symbols[i], f"__TMP{i}__"
                        )
                    for i, (_, q_orig) in enumerate(weights):
                        ordered_str = ordered_str.replace(
                            f"__TMP{i}__", ordered[i][1]
                        )
                    alternates.append(
                        result.expression.replace(term, ordered_str)
                    )

        if not alternates:
            return result

        # Score alternates
        for alt in alternates:
            scores = [evaluator.score(alt, obs) for obs in observations]
            avg = sum(scores) / len(scores) if scores else 0.0
            alt_canon = canonicalizer.score(alt)
            # Accept if similar constancy but better canonical form.
            # Two tiers:
            # (1) near-identical constancy → any canonical improvement wins
            # (2) slightly worse constancy (within 0.01) → need stronger improvement
            if avg >= result.score - 1e-9 and alt_canon > current_canon:
                return SearchResult(
                    expression=alt,
                    score=avg,
                    depth=result.depth,
                    expansions=result.expansions,
                    train_constancies=scores,
                )
            if avg >= result.score - 0.01 and alt_canon > current_canon + 0.05:
                return SearchResult(
                    expression=alt,
                    score=avg,
                    depth=result.depth,
                    expansions=result.expansions,
                    train_constancies=scores,
                )

    except Exception:
        pass

    return result
