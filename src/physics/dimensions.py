"""Physical dimension type system.

Base types: Scalar, Mass, Length, Time, Velocity, Accel, Force, Energy.
Supports dimension arithmetic (multiply, divide, power) and type-checking
for addition/subtraction compatibility.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar


# ── Named dimension definitions ──────────────────────────────────────────────

_NAMED_DIMENSIONS: dict[str, dict[str, float]] = {
    "Scalar":   {},
    "Mass":     {"Mass": 1.0},
    "Length":   {"Length": 1.0},
    "Time":     {"Time": 1.0},
    "Velocity": {"Length": 1.0, "Time": -1.0},
    "Accel":    {"Length": 1.0, "Time": -2.0},
    "Force":    {"Mass": 1.0, "Length": 1.0, "Time": -2.0},
    "Energy":   {"Mass": 1.0, "Length": 2.0, "Time": -2.0},
}

# Map each base dimension to its exponents across named types (for display)
_BASE_DIMENSIONS: list[str] = ["Mass", "Length", "Time"]
_BASE_SYMBOLS: dict[str, str] = {
    "Mass":   "kg",
    "Length": "m",
    "Time":   "s",
}


class Dimension:
    """Physical dimension as a vector of base-dimension exponents.

    Dimensions are immutable and hashable.  Compound dimensions arise from
    multiplication and division (e.g. ``Mass * Length / Time**2`` = Force).
    Addition/subtraction is only permitted between dimensions that match
    exactly — mixing e.g. ``Mass + Velocity`` raises a ``DimensionError``.

    Factory methods
    ---------------
    ``Dimension.scalar()`` — dimensionless.
    ``Dimension.named(name)`` — one of the pre-defined types (``"Mass"``,
    ``"Length"``, ``"Time"``, ``"Velocity"``, ``"Accel"``, ``"Force"``,
    ``"Energy"``, ``"Scalar"``).
    ``Dimension.from_exponents(d)`` — construct from raw exponent dict.

    Examples
    --------
    >>> L = Dimension.named("Length")
    >>> T = Dimension.named("Time")
    >>> V = L / T
    >>> V == Dimension.named("Velocity")
    True
    """

    __slots__ = ("_exp", "_hash")

    # Lookup from string name → pre-built singleton (avoids recomputation)
    _cache: ClassVar[dict[tuple, Dimension]] = {}

    def __init__(self, exp: dict[str, float] | None = None) -> None:
        if exp is None:
            exp = {}
        # Normalise: only track base dimensions; omit zeroes
        self._exp: dict[str, float] = {
            d: float(exp.get(d, 0)) for d in _BASE_DIMENSIONS
        }
        # Strip zeroes for canonical form
        self._exp = {k: v for k, v in self._exp.items() if v != 0}
        # Pre-compute hash for dict/set lookups
        self._hash: int = hash(tuple(sorted(self._exp.items())))

    # ── factories ────────────────────────────────────────────────────────

    @classmethod
    def scalar(cls) -> Dimension:
        """Return the dimensionless Scalar type."""
        return cls._cached({})

    @classmethod
    def named(cls, name: str) -> Dimension:
        """Return a pre-defined named dimension.

        Valid names: Scalar, Mass, Length, Time, Velocity, Accel, Force, Energy.
        Also supports compound names like ``"Force/Length"`` or ``"Mass*Length/Time^2"``
        by parsing and computing the resulting dimension.
        """
        if name in _NAMED_DIMENSIONS:
            return cls._cached(_NAMED_DIMENSIONS[name])

        # Try to parse compound dimension name like "Force/Length" or "Mass*Velocity"
        parsed = cls._parse_compound(name)
        if parsed is not None:
            return parsed

        raise ValueError(
            f"Unknown dimension name {name!r}. "
            f"Available: {list(_NAMED_DIMENSIONS)}"
        )

    @classmethod
    def _parse_compound(cls, name: str) -> Dimension | None:
        """Parse a compound dimension name like 'Force/Length' or 'Mass*Velocity^2'.

        Supports: * (multiply), / (divide), ^ (power), and named dimensions.
        Returns None if the string cannot be parsed.
        """
        import re

        name = name.strip()

        # Handle division: "A/B"
        if "/" in name:
            parts = name.split("/", 1)
            left = cls._parse_compound(parts[0])
            right = cls._parse_compound(parts[1])
            if left is not None and right is not None:
                return left / right
            return None

        # Handle multiplication: split on *
        # But be careful: "Mass*Length" or "Mass*Length^2"
        tokens: list[str] = []
        depth = 0
        current = ""
        for ch in name:
            if ch == "(":
                depth += 1
                current += ch
            elif ch == ")":
                depth -= 1
                current += ch
            elif ch == "*" and depth == 0:
                if current:
                    tokens.append(current.strip())
                current = ""
            else:
                current += ch
        if current:
            tokens.append(current.strip())

        if len(tokens) == 0:
            return None

        if len(tokens) == 1:
            # Single token: could be a named dimension, possibly with ^N
            token = tokens[0]
            # Match "Name^N" pattern
            m = re.match(r'^(\w+)\^(-?\d+)$', token)
            if m:
                base_name = m.group(1)
                exponent = float(m.group(2))
                if base_name in _NAMED_DIMENSIONS:
                    base = cls._cached(_NAMED_DIMENSIONS[base_name])
                    return base ** exponent
                return None
            # Just a named dimension
            if token in _NAMED_DIMENSIONS:
                return cls._cached(_NAMED_DIMENSIONS[token])
            return None

        # Multiple tokens: multiply them all
        result: Dimension | None = None
        for token in tokens:
            dim = cls._parse_compound(token)
            if dim is None:
                return None
            if result is None:
                result = dim
            else:
                result = result * dim
        return result

    @classmethod
    def from_exponents(cls, exp: dict[str, float]) -> Dimension:
        """Build a dimension from raw exponent dict (used by arithmetic)."""
        return cls._cached(exp)

    @classmethod
    def _cached(cls, exp: Mapping[str, float]) -> Dimension:
        """Return a cached singleton for the given exponent tuple."""
        key = tuple(sorted((k, v) for k, v in exp.items() if v != 0))
        if key not in cls._cache:
            obj = object.__new__(cls)
            obj._exp = dict(key)
            obj._hash = hash(key)
            cls._cache[key] = obj
        return cls._cache[key]

    # ── arithmetic ───────────────────────────────────────────────────────

    def __mul__(self, other: Dimension) -> Dimension:
        """Multiply dimensions: exponents add."""
        if not isinstance(other, Dimension):
            return NotImplemented
        combined: dict[str, float] = {}
        for d in _BASE_DIMENSIONS:
            v = self._exp.get(d, 0.0) + other._exp.get(d, 0.0)
            if v != 0:
                combined[d] = float(v)
        return self._cached(combined)

    def __truediv__(self, other: Dimension) -> Dimension:
        """Divide dimensions: exponents subtract."""
        if not isinstance(other, Dimension):
            return NotImplemented
        combined: dict[str, float] = {}
        for d in _BASE_DIMENSIONS:
            v = self._exp.get(d, 0.0) - other._exp.get(d, 0.0)
            if v != 0:
                combined[d] = float(v)
        return self._cached(combined)

    def __pow__(self, exponent: int | float) -> Dimension:
        """Raise dimension to a scalar power: exponents multiplied."""
        if not isinstance(exponent, (int, float)):
            return NotImplemented
        combined: dict[str, float] = {}
        for d, v in self._exp.items():
            result = v * exponent
            if result != 0:
                combined[d] = result
        return self._cached(combined)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Dimension):
            return NotImplemented
        return self._hash == other._hash and self._exp == other._exp

    def __hash__(self) -> int:
        return self._hash

    # ── queries ──────────────────────────────────────────────────────────

    def is_scalar(self) -> bool:
        """True if this dimension is dimensionless."""
        return len(self._exp) == 0

    def compatible_with(self, other: Dimension) -> bool:
        """Can *add* or *subtract* these dimensions?  (Must match exactly.)"""
        return self == other

    def check_add(self, other: Dimension) -> None:
        """Raise DimensionError if dimensions are incompatible for +/−."""
        if not self.compatible_with(other):
            raise DimensionError(
                f"Cannot add/subtract incompatible dimensions: "
                f"{self} + {other}"
            )

    def check_power_exponent(self) -> None:
        """Raise DimensionError if this dimension is non-scalar and is used
        as a power exponent.  Power exponent must be scalar."""
        if not self.is_scalar():
            raise DimensionError(
                f"Power exponent must be scalar, got {self}"
            )

    # ── display ──────────────────────────────────────────────────────────

    def _as_symbolic(self) -> str:
        """Render as symbolic kg·m²/s² form."""
        if self.is_scalar():
            return ""
        numer: list[str] = []
        denom: list[str] = []
        for d in _BASE_DIMENSIONS:
            e = self._exp.get(d, 0)
            if e == 0:
                continue
            sym = _BASE_SYMBOLS.get(d, d)
            if e == 1:
                numer.append(sym)
            elif e > 1:
                numer.append(f"{sym}{_superscript(e)}")
            elif e == -1:
                denom.append(sym)
            else:  # e < -1
                denom.append(f"{sym}{_superscript(-e)}")
        num_str = "·".join(numer) if numer else "1"
        if denom:
            den_str = "·".join(denom)
            return f"{num_str}/{den_str}" if len(denom) == 1 else f"{num_str}/({den_str})"
        return num_str

    def __str__(self) -> str:
        symbolic = self._as_symbolic()
        if not symbolic:
            return "Scalar"
        # Check against named dimensions for a friendlier label
        for name, exp in _NAMED_DIMENSIONS.items():
            if name == "Scalar":
                continue
            if self._exp == {k: v for k, v in exp.items() if v != 0}:
                return name
        return symbolic

    def __repr__(self) -> str:
        return f"Dimension({self})"

    # ── pickling support ─────────────────────────────────────────────────

    def __getstate__(self) -> dict:
        return {"exp": self._exp}

    def __setstate__(self, state: dict) -> None:
        # Re-build via cache
        restored = Dimension._cached(state["exp"])
        self._exp = restored._exp
        self._hash = restored._hash


def _superscript(n: int | float) -> str:
    """Render integer as Unicode superscript."""
    n_int = int(n)
    chars = "⁰¹²³⁴⁵⁶⁷⁸⁹"
    result = ""
    for d in str(abs(n_int)):
        result += chars[int(d)]
    if n_int < 0:
        result = "⁻" + result
    return result


class DimensionError(TypeError):
    """Raised when a dimensionally invalid operation is attempted
    (e.g. adding Mass + Velocity)."""
    pass
