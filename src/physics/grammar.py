"""Expression grammar with physical dimension type tracking.

Implements the EBNF grammar from Section 2.3 of the self-play physics
discovery plan.  Each expression node carries a computed physical dimension;
incompatible operations (e.g. adding Mass to Velocity) raise DimensionError.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from src.physics.dimensions import Dimension, DimensionError


# ── Operation table ──────────────────────────────────────────────────────────

class Op:
    """Operation descriptors with dimension rules."""

    __slots__ = ("symbol", "arity", "precedence")

    def __init__(self, symbol: str, arity: int = 2, precedence: int = 0) -> None:
        self.symbol = symbol
        self.arity = arity
        self.precedence = precedence

    def __repr__(self) -> str:
        return f"Op({self.symbol!r})"


# Binary operations
OP_ADD = Op("+", arity=2, precedence=1)
OP_SUB = Op("-", arity=2, precedence=1)
OP_MUL = Op("*", arity=2, precedence=2)
OP_DIV = Op("/", arity=2, precedence=2)
OP_POW = Op("^", arity=2, precedence=3)

# Unary operations (defined for future phases)
OP_DDT = Op("d/dt", arity=1, precedence=4)
OP_DDX = Op("d/dx", arity=1, precedence=4)
OP_SIN = Op("sin", arity=1, precedence=4)
OP_COS = Op("cos", arity=1, precedence=4)
OP_EXP = Op("exp", arity=1, precedence=4)
OP_LOG = Op("log", arity=1, precedence=4)
OP_SQRT = Op("sqrt", arity=1, precedence=4)

_BINARY_OPS: dict[str, Op] = {
    "+": OP_ADD,
    "-": OP_SUB,
    "*": OP_MUL,
    "/": OP_DIV,
    "^": OP_POW,
}

_UNARY_OPS: dict[str, Op] = {
    "d/dt": OP_DDT,
    "d/dx": OP_DDX,
    "sin": OP_SIN,
    "cos": OP_COS,
    "exp": OP_EXP,
    "log": OP_LOG,
    "sqrt": OP_SQRT,
}

ALL_OPS: dict[str, Op] = {**_BINARY_OPS, **_UNARY_OPS}


# ── Expression node ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Expression:
    """A node in the physics expression tree.

    Immutable (frozen=True) so expressions can be hashed and cached.
    The dimension is computed at construction time from the operator
    and children's dimensions.
    """

    op: Op | None  # None for leaf nodes (quantities / constants)
    children: tuple[Expression, ...] = ()
    dim: Dimension = field(default_factory=Dimension.scalar)

    # Leaf payloads (only one is set for leaf nodes)
    quantity_name: str | None = None
    constant_value: str | None = None  # stored as string to preserve exact repr

    _str_cache: str | None = field(default=None, repr=False, compare=False)

    # ── factories (leaves) ───────────────────────────────────────────────

    @classmethod
    def quantity(cls, name: str, dim: Dimension) -> Expression:
        """Create a quantity leaf node (e.g. 'm' with Mass dimension)."""
        return cls(
            op=None,
            children=(),
            dim=dim,
            quantity_name=name,
            constant_value=None,
        )

    @classmethod
    def constant(cls, value: str, dim: Dimension) -> Expression:
        """Create a constant leaf node (e.g. '0.5' with Scalar dimension)."""
        return cls(
            op=None,
            children=(),
            dim=dim,
            quantity_name=None,
            constant_value=value,
        )

    @classmethod
    def build(cls, op_symbol: str, left: Expression, right: Expression) -> Expression:
        """Build a binary operation node with dimension checking.

        Args:
            op_symbol: one of '+', '-', '*', '/', '^'
            left: left-hand expression
            right: right-hand expression

        Returns:
            New Expression with computed dimension.

        Raises:
            DimensionError: if the operation is dimensionally invalid
                (e.g. adding Mass + Velocity, or using non-scalar exponent).
        """
        op = _BINARY_OPS.get(op_symbol)
        if op is None:
            raise ValueError(
                f"Unknown binary operator {op_symbol!r}. "
                f"Available: {list(_BINARY_OPS)}"
            )

        if op_symbol == "+":
            left.dim.check_add(right.dim)
            result_dim = left.dim

        elif op_symbol == "-":
            left.dim.check_add(right.dim)
            result_dim = left.dim

        elif op_symbol == "*":
            result_dim = left.dim * right.dim

        elif op_symbol == "/":
            result_dim = left.dim / right.dim

        elif op_symbol == "^":
            # Exponent must be scalar
            right.dim.check_power_exponent()
            # The exponent value is stored in the constant; extract numeric value
            exponent = cls._extract_exponent(right)
            result_dim = left.dim ** exponent

        else:
            raise ValueError(f"Unhandled operator {op_symbol!r}")

        return cls(
            op=op,
            children=(left, right),
            dim=result_dim,
        )

    @staticmethod
    def _extract_exponent(expr: Expression) -> float:
        """Extract numeric value from a constant expression for power ops."""
        if expr.constant_value is not None:
            val = expr.constant_value
            try:
                # Handle fractions like "1/2", "½"
                if "/" in val:
                    num, den = val.split("/")
                    return float(num) / float(den)
                # Handle Unicode fractions
                unicode_fracs = {
                    "½": 0.5, "⅓": 1/3, "⅔": 2/3,
                    "¼": 0.25, "¾": 0.75,
                    "⅕": 0.2, "⅖": 0.4, "⅗": 0.6, "⅘": 0.8,
                    "⅙": 1/6, "⅚": 5/6,
                    "⅛": 0.125, "⅜": 0.375, "⅝": 0.625, "⅞": 0.875,
                }
                if val in unicode_fracs:
                    return unicode_fracs[val]
                return float(val)
            except (ValueError, ZeroDivisionError):
                pass  # fall through to error below
        raise DimensionError(
            f"Power exponent must be a scalar constant, got {expr}"
        )

    # ── queries ──────────────────────────────────────────────────────────

    def is_leaf(self) -> bool:
        """True if this is a leaf node (quantity or constant)."""
        return self.op is None

    def is_quantity(self) -> bool:
        """True if this is a quantity leaf."""
        return self.quantity_name is not None

    def is_constant(self) -> bool:
        """True if this is a constant leaf."""
        return self.constant_value is not None

    def depth(self) -> int:
        """Expression tree depth (leaf = 1)."""
        if self.is_leaf():
            return 1
        return 1 + max(c.depth() for c in self.children)

    # ── display ──────────────────────────────────────────────────────────

    def __str__(self) -> str:
        """Return the expression as a readable string with minimal parentheses."""
        if self._str_cache is not None:
            return self._str_cache

        if self.is_leaf():
            s = self.quantity_name or self.constant_value or "?"
            # Use __dict__ mutation on frozen dataclass for caching (safe here)
            object.__setattr__(self, "_str_cache", s)
            return s

        op = self.op
        assert op is not None  # non-leaf always has an op
        left, right = self.children[0], self.children[1]

        # Format left child
        left_str = str(left)
        if not left.is_leaf() and left.op is not None:
            if left.op.precedence < op.precedence:
                left_str = f"({left_str})"
            # Same precedence and non-associative (like ^, -)
            elif left.op.precedence == op.precedence:
                if op.symbol in ("-", "/", "^"):
                    left_str = f"({left_str})"

        # Format right child
        right_str = str(right)
        if not right.is_leaf() and right.op is not None:
            if right.op.precedence < op.precedence:
                right_str = f"({right_str})"
            elif right.op.precedence == op.precedence:
                # Right-associative: a^b^c = a^(b^c)
                # But for safety and readability, parenthesize
                if op.symbol in ("-", "/"):
                    right_str = f"({right_str})"

        s = f"{left_str}{op.symbol}{right_str}"
        object.__setattr__(self, "_str_cache", s)
        return s

    def __repr__(self) -> str:
        dim_str = str(self.dim)
        return f"Expr({self}, dim={dim_str})"

    # ── comparison / hashing (inherited from frozen dataclass) ───────────

    def __hash__(self) -> int:
        # Frozen dataclass gives us this but we need to exclude _str_cache
        return hash((
            self.op,
            self.children,
            self.dim,
            self.quantity_name,
            self.constant_value,
        ))
