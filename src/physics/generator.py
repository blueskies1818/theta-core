"""Breadth-first combinatorial expression builder.

Given a set of quantities (with their physical dimensions), operations,
scalar constants, and a maximum depth, generates ALL dimensionally valid
expressions up to max_depth.
"""

from __future__ import annotations

import itertools
from collections import defaultdict

from src.physics.dimensions import Dimension, DimensionError
from src.physics.grammar import Expression


class ExpressionGenerator:
    """Breadth-first generator of dimensionally valid physics expressions.

    Parameters
    ----------
    quantities : dict[str, Dimension]
        Named quantities and their physical dimensions, e.g.
        ``{"m": Dimension.named("Mass"), "v": Dimension.named("Velocity"), ...}``
    operations : set[str]
        Subset of ``{"+", "-", "*", "/", "^"}`` to use.
    constants : dict[str, Dimension]
        Scalar or physical constants, e.g.
        ``{"0": Dimension.scalar(), "0.5": Dimension.scalar(), "g": Dimension.named("Accel")}``
    max_depth : int
        Maximum expression tree depth (leaf = depth 1).
    """

    def __init__(
        self,
        quantities: dict[str, Dimension],
        operations: set[str],
        constants: dict[str, Dimension],
        max_depth: int,
    ) -> None:
        self.quantities = quantities
        self.operations = operations
        self.constants = constants
        self.max_depth = max_depth

        # Pre-built leaf nodes (depth 1)
        self._leaf_nodes: dict[str, Expression] = {}
        for name, dim in quantities.items():
            self._leaf_nodes[name] = Expression.quantity(name, dim)
        for value, dim in constants.items():
            self._leaf_nodes[value] = Expression.constant(value, dim)

        # Generated expressions grouped by depth
        self._by_depth: dict[int, list[Expression]] = defaultdict(list)
        self._by_depth[1] = list(self._leaf_nodes.values())

        # All expressions keyed by string representation (dedup)
        self._seen: set[str] = set()
        for expr in self._by_depth[1]:
            self._seen.add(str(expr))

        # Generated flag
        self._generated: bool = False

    # ── generation ───────────────────────────────────────────────────────

    def generate(self) -> list[Expression]:
        """Run breadth-first generation up to max_depth.

        Returns all valid expressions at all depths (including leaves).
        """
        if self._generated:
            return self._all_expressions()

        for depth in range(2, self.max_depth + 1):
            self._generate_depth(depth)

        self._generated = True
        return self._all_expressions()

    def _generate_depth(self, depth: int) -> None:
        """Generate all valid expressions of exactly the given depth.

        An expression of depth D is formed by combining sub-expressions
        of depths (i, D-i) for i in 1..D-1.
        """
        new_exprs: list[Expression] = []

        for left_depth in range(1, depth):
            right_depth = depth - left_depth
            left_pool = self._by_depth.get(left_depth, [])
            right_pool = self._by_depth.get(right_depth, [])
            if not left_pool or not right_pool:
                continue
            for op_symbol in self.operations:
                new_exprs.extend(
                    self._try_combine(op_symbol, left_pool, right_pool)
                )

        self._by_depth[depth] = new_exprs

    def _try_combine(
        self,
        op_symbol: str,
        left_pool: list[Expression],
        right_pool: list[Expression],
    ) -> list[Expression]:
        """Try all left×right pairs with the given operator.

        Returns only dimensionally valid results not seen before.
        """
        results: list[Expression] = []
        for left, right in itertools.product(left_pool, right_pool):
            try:
                expr = Expression.build(op_symbol, left, right)
            except DimensionError:
                continue
            key = str(expr)
            if key not in self._seen:
                self._seen.add(key)
                results.append(expr)
        return results

    # ── query ────────────────────────────────────────────────────────────

    def expressions_at_depth(self, depth: int) -> list[Expression]:
        """Return all expressions of exactly the given depth."""
        if not self._generated:
            raise RuntimeError("Call generate() first")
        return self._by_depth.get(depth, [])

    def expressions_by_dimension(self, dim: Dimension) -> list[Expression]:
        """Return all generated expressions matching the given dimension."""
        if not self._generated:
            raise RuntimeError("Call generate() first")
        return [e for e in self._all_expressions() if e.dim == dim]

    def _all_expressions(self) -> list[Expression]:
        """Flatten all depths into a single list."""
        all_exprs: list[Expression] = []
        for d in range(1, self.max_depth + 1):
            all_exprs.extend(self._by_depth.get(d, []))
        return all_exprs

    def count(self) -> int:
        """Total number of valid expressions generated."""
        if not self._generated:
            raise RuntimeError("Call generate() first")
        return len(self._seen)

    def __len__(self) -> int:
        return self.count()

    def __iter__(self):
        if not self._generated:
            raise RuntimeError("Call generate() first")
        return iter(self._all_expressions())

    def __contains__(self, expr_str: str) -> bool:
        """Check if an expression string is in the generated set."""
        if not self._generated:
            raise RuntimeError("Call generate() first")
        return expr_str in self._seen


def run_smoke_test() -> dict:
    """Run the Phase A smoke test.

    Generates all valid depth-4 expressions from {m, g, h, v} with
    {+, *, /, ^} and checks that m*g*h and 0.5*m*v^2 are present,
    while m+g and m*v+h are rejected.
    """
    quantities = {
        "m": Dimension.named("Mass"),
        "g": Dimension.named("Accel"),
        "h": Dimension.named("Length"),
        "v": Dimension.named("Velocity"),
    }
    constants = {
        "0": Dimension.scalar(),
        "0.5": Dimension.scalar(),
        "1": Dimension.scalar(),
        "2": Dimension.scalar(),
    }
    ops = {"+", "*", "/", "^"}

    gen = ExpressionGenerator(
        quantities=quantities,
        operations=ops,
        constants=constants,
        max_depth=4,
    )
    gen.generate()

    results = {
        "total_expressions": gen.count(),
        "has_mgh": "m*g*h" in gen,
        "has_half_mv2": "0.5*m*v^2" in gen,
        "rejects_m_plus_g": "m+g" not in gen,
        "rejects_mv_plus_h": "m*v+h" not in gen,
        "depth_counts": {d: len(gen.expressions_at_depth(d)) for d in range(1, 5)},
    }

    return results
