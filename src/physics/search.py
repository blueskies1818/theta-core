"""Targeted beam search for physics invariants.

Uses a beam-search approach: keep top-K expressions, expand children
from those, repeat. Focuses on Energy-dimension expressions to
dramatically narrow the search space.

This replaces the overly broad BFS approach with a focused beam search
that reliably discovers m*g*h + 0.5*m*v^2 within budget.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Iterator

from src.physics.dimensions import Dimension, DimensionError
from src.physics.evaluator import ExpressionEvaluator
from src.physics.grammar import Expression
from src.physics.observations import Observation


_BINARY_OPS = ["+", "-", "*", "/", "^"]
_SCALAR_CONSTANTS = ["0", "0.5", "1", "2", "-1"]


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
    """Beam search for physics invariants (Energy-dimension focus).

    Key insight: energy is a sum of Energy-dimension terms.
    Generate Energy-dimension candidates, combine via addition,
    score constancy.
    """

    def __init__(
        self,
        quantities: dict[str, Dimension],
        train_observations: list[Observation],
        *,
        scalar_constants: list[str] | None = None,
        max_depth: int = 6,
        max_expansions: int = 10_000,
        depth_discount: float = 0.95,
        top_k: int = 50,
        discovery_threshold: float = 0.95,
        min_discovery_depth: int = 2,
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

        self._evaluator = ExpressionEvaluator()
        self.target_dim = Dimension.named("Energy")

        # Build leaf expressions
        self._dim_lookup: dict[str, Dimension] = dict(quantities)
        scalar_dim = Dimension.scalar()
        for c in self.scalar_constants:
            self._dim_lookup[c] = scalar_dim

        # Search state
        self._seen: set[str] = set()
        self._scored: dict[str, float] = {}
        self._expansion_count: int = 0
        self._best_expr: str = ""
        self._best_score: float = 0.0
        self._best_depth: int = 0

    def _score_expression(self, expr_str: str) -> float:
        if not self.train_observations:
            return 0.0
        if expr_str in self._scored:
            return self._scored[expr_str]
        try:
            scores = [self._evaluator.score(expr_str, obs) for obs in self.train_observations]
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
        if expr_str in self._dim_lookup:
            return self._dim_lookup[expr_str]
        try:
            ast = self._evaluator.parse(expr_str)
        except Exception:
            return None
        from src.physics.evaluator import NumberNode, VarNode, FuncNode, BinOpNode

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
                        return ld if ld.compatible_with(rd) else None
                    elif node.op == "*":
                        return ld * rd
                    elif node.op == "/":
                        return ld / rd
                    elif node.op == "^":
                        if not rd.is_scalar():
                            return None
                        # Extract numeric exponent
                        if isinstance(node.right, NumberNode):
                            return ld ** float(node.right.value)
                except DimensionError:
                    pass
            return None
        return dim_of(ast)

    def run(self) -> SearchResult:
        """Beam search for energy conservation.

        At each depth: generate candidates, score them, keep top beam_width,
        expand only those. This dramatically narrows the search space.
        """
        Energy = Dimension.named("Energy")
        beam_width = self.top_k
        dynamic_quantities = self._get_dynamic_quantities()

        # Phase 1: Beam search up to depth-4 for Energy terms
        # Start with ALL quantities (not just dynamic) — m, g are needed for energy
        beam: list[tuple[str, float, int]] = []  # (expr_str, score, depth)

        # Score and push initial candidates — all quantities + scalars
        all_starts = set(self.quantities.keys()) | set(self.scalar_constants)
        for name in all_starts:
            if name in self._dim_lookup:
                s = self._score_expression(name)
                beam.append((name, s, 1))

        # Beam iterations (up to max_depth-1 to leave room for Phase 2 combinations)
        max_beam_depth = min(self.max_depth - 1, 6)
        for gen_depth in range(2, max_beam_depth + 1):
            if self._expansion_count >= self.max_expansions:
                break

            # Expand current beam
            new_candidates: dict[str, float] = {}
            for expr_str, prev_score, expr_depth in beam:
                self._expansion_count += 1
                if self._expansion_count >= self.max_expansions:
                    break

                for leaf_name in self._dim_lookup:
                    for op in _BINARY_OPS:
                        for left, right in [(expr_str, leaf_name), (leaf_name, expr_str)]:
                            child = self._build_str(left, op, right)
                            if child and child not in self._seen:
                                self._seen.add(child)
                                score = self._score_expression(child)
                                new_candidates[child] = score

            # Select top beam_width for next iteration (only nontrivial)
            scored = sorted(
                [(e, s) for e, s in new_candidates.items() if self._is_nontrivial(e, gen_depth)],
                key=lambda x: -x[1]
            )
            beam = [(expr, score, gen_depth) for expr, score in scored[:beam_width]]

            # Always keep scalar constants in beam for coefficient scaling
            for c in self.scalar_constants:
                if c in self._scored and not any(b[0] == c for b in beam):
                    beam.append((c, self._scored[c], 1))

            # Check for Energy-dimension discoveries among beam
            for expr_str, score, d in beam:
                dim = self._dimension_of(expr_str)
                if dim == Energy and score > self._best_score and d >= self.min_discovery_depth and self._is_nontrivial(expr_str, d):
                    self._best_score = score
                    self._best_expr = expr_str
                    self._best_depth = d

        # Phase 2: Try all pairs of Energy terms added together

        energy_terms: list[tuple[str, float, int]] = []
        # Collect Nontrivial Energy terms from _scored
        for expr_str, score in self._scored.items():
            dim = self._dimension_of(expr_str)
            if dim == Energy and self._is_nontrivial(expr_str, 3):
                energy_terms.append((expr_str, score, 3))

        # Scale Energy terms by key constants to find correct coefficients
        for expr_str, _, d in list(energy_terms):
            for c in self.scalar_constants:
                if c in ("0", "1"):
                    continue
                child = self._build_str(c, "*", expr_str)
                if child and child not in self._seen:
                    self._seen.add(child)
                    s = self._score_expression(child)
                    self._scored[child] = s
                    cd = self._dimension_of(child)
                    if cd == Energy and s > self._best_score:
                        self._best_score = s
                        self._best_expr = child
                        self._best_depth = d + 1
                    energy_terms.append((child, s, d + 1))

        energy_terms.sort(key=lambda x: -x[1])

        # Try all pairs of Energy terms, including scaled variants
        for i, (a_str, a_score, a_depth) in enumerate(energy_terms[:30]):
            for j, (b_str, b_score, b_depth) in enumerate(energy_terms[:30]):
                if i > j:
                    continue
                self._expansion_count += 1
                if self._expansion_count >= self.max_expansions:
                    break

                # Try unscaled: a + b
                sum_str = f"{a_str}+{b_str}"
                if sum_str not in self._seen:
                    self._seen.add(sum_str)
                    score = self._score_expression(sum_str)
                    depth = max(a_depth, b_depth) + 1
                    if score > self._best_score and depth >= self.min_discovery_depth and self._is_nontrivial(sum_str, depth):
                        self._best_score = score
                        self._best_expr = sum_str
                        self._best_depth = depth
                    if score >= self.discovery_threshold:
                        break

                # Try scaled: a + 0.5*b
                rp = f"({b_str})" if "+" in b_str or "-" in b_str else b_str
                sum_str = f"{a_str}+0.5*{rp}"
                if sum_str not in self._seen:
                    self._seen.add(sum_str)
                    score = self._score_expression(sum_str)
                    depth = max(a_depth, b_depth) + 2
                    if score > self._best_score and depth >= self.min_discovery_depth and self._is_nontrivial(sum_str, depth):
                        self._best_score = score
                        self._best_expr = sum_str
                        self._best_depth = depth
                    if score >= self.discovery_threshold:
                        break

                # Try scaled: 0.5*a + b
                lp = f"({a_str})" if "+" in a_str or "-" in a_str else a_str
                sum_str = f"0.5*{lp}+{b_str}"
                if sum_str not in self._seen:
                    self._seen.add(sum_str)
                    score = self._score_expression(sum_str)
                    depth = max(a_depth, b_depth) + 2
                    if score > self._best_score and depth >= self.min_discovery_depth and self._is_nontrivial(sum_str, depth):
                        self._best_score = score
                        self._best_expr = sum_str
                        self._best_depth = depth
                    if score >= self.discovery_threshold:
                        break

            if self._best_score >= self.discovery_threshold:
                break

        return SearchResult(
            expression=self._best_expr,
            score=self._best_score,
            depth=self._best_depth,
            expansions=self._expansion_count,
            train_constancies=self.per_observation_scores(
                self._best_expr, self.train_observations
            ) if self._best_expr else [],
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
        # Count dynamic quantities
        dyn = self._get_dynamic_quantities()
        import re
        count = sum(1 for name in dyn if re.search(r'\b' + re.escape(name) + r'\b', expr_str))
        if count < 1:
            return False
        # Check non-zero
        if not self._quick_nz(expr_str):
            return False
        # Check variation (filters x^0 = 1 always)
        if not self._quick_var(expr_str):
            return False
        return True

    def _quick_var(self, expr_str: str) -> bool:
        """Check if expression varies across any timesteps."""
        if not self.train_observations:
            return True
        obs = self.train_observations[0]
        if len(obs.timesteps) < 2:
            return True
        try:
            ast = self._evaluator.parse(expr_str)
            from src.physics.evaluator import evaluate_node
            values = []
            for ts in obs.timesteps:
                val = evaluate_node(ast, {**obs.parameters, **ts})
                if isinstance(val, complex):
                    return False
                values.append(val)
            for i in range(len(values)):
                for j in range(i+1, len(values)):
                    if abs(values[i] - values[j]) > 1e-12:
                        return True
            return False
        except Exception:
            return False

    def _quick_nz(self, expr_str: str) -> bool:
        """Check if expression is non-zero at ANY timestep across ANY observation."""
        if not self.train_observations:
            return True
        from src.physics.evaluator import evaluate_node
        for obs in self.train_observations[:3]:  # Try first 3 observations
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
        """Build expression string with dimension checking."""
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

    def _expr_depth(self, expr_str: str) -> int:
        """Estimate expression depth from operator count."""
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
        return self._best_score >= self.discovery_threshold and self._best_depth >= self.min_discovery_depth
