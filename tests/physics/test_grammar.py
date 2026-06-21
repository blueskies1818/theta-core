"""Unit tests for physics expression grammar."""

import pytest
from src.physics.dimensions import Dimension, DimensionError
from src.physics.grammar import Expression


# ── Convenience dimensions ─────────────────────────────────────────

MASS = Dimension.named("Mass")
LENGTH = Dimension.named("Length")
TIME = Dimension.named("Time")
SCALAR = Dimension.scalar()
VELOCITY = Dimension.named("Velocity")
ACCEL = Dimension.named("Accel")
FORCE = Dimension.named("Force")
ENERGY = Dimension.named("Energy")


class TestExpressionConstruction:
    """Test creating leaf and binary expression nodes."""

    # ── leaves ──────────────────────────────────────────────────

    def test_quantity_mass(self):
        m = Expression.quantity("m", MASS)
        assert m.is_leaf()
        assert m.is_quantity()
        assert not m.is_constant()
        assert m.quantity_name == "m"
        assert m.dim == MASS

    def test_quantity_velocity(self):
        v = Expression.quantity("v", VELOCITY)
        assert v.dim == VELOCITY

    def test_quantity_length(self):
        h = Expression.quantity("h", LENGTH)
        assert h.dim == LENGTH

    def test_constant_numeric(self):
        c = Expression.constant("0.5", SCALAR)
        assert c.is_leaf()
        assert c.is_constant()
        assert not c.is_quantity()
        assert c.constant_value == "0.5"
        assert c.dim == SCALAR

    def test_constant_pi(self):
        pi = Expression.constant("π", SCALAR)
        assert pi.dim == SCALAR

    # ── binary: multiplication ──────────────────────────────────

    def test_build_mul_mass_accel(self):
        """m * g → Force (kg·m/s²)."""
        m = Expression.quantity("m", MASS)
        g = Expression.quantity("g", ACCEL)
        expr = Expression.build("*", m, g)
        assert expr.dim == FORCE

    def test_build_mul_force_length(self):
        """m*g * h → Energy (kg·m²/s²)."""
        m = Expression.quantity("m", MASS)
        g = Expression.quantity("g", ACCEL)
        mg = Expression.build("*", m, g)
        h = Expression.quantity("h", LENGTH)
        expr = Expression.build("*", mg, h)
        assert expr.dim == ENERGY

    def test_build_mul_scalar_velocity(self):
        """0.5 * v → Velocity (Scalar * Velocity = Velocity)."""
        half = Expression.constant("0.5", SCALAR)
        v = Expression.quantity("v", VELOCITY)
        expr = Expression.build("*", half, v)
        assert expr.dim == VELOCITY

    # ── binary: division ────────────────────────────────────────

    def test_build_div_length_time(self):
        """x / t → Velocity (m/s)."""
        x = Expression.quantity("x", LENGTH)
        t = Expression.quantity("t", TIME)
        expr = Expression.build("/", x, t)
        assert expr.dim == VELOCITY

    # ── binary: power ───────────────────────────────────────────

    def test_build_pow_velocity_squared(self):
        """v ^ 2 → m²/s²."""
        v = Expression.quantity("v", VELOCITY)
        two = Expression.constant("2", SCALAR)
        expr = Expression.build("^", v, two)
        assert expr.dim == Dimension.from_exponents({"Length": 2.0, "Time": -2.0})

    def test_build_pow_length_squared(self):
        """h ^ 2 → m²."""
        h = Expression.quantity("h", LENGTH)
        two = Expression.constant("2", SCALAR)
        expr = Expression.build("^", h, two)
        assert expr.dim == Dimension.from_exponents({"Length": 2.0})

    def test_build_pow_non_scalar_exponent_raises(self):
        """v ^ m → ERROR (exponent must be scalar)."""
        v = Expression.quantity("v", VELOCITY)
        m = Expression.quantity("m", MASS)
        with pytest.raises(DimensionError, match="scalar"):
            Expression.build("^", v, m)

    # ── binary: addition / subtraction ──────────────────────────

    def test_build_add_same_type(self):
        """Energy + Energy → Energy."""
        e1 = Expression.quantity("E", ENERGY)
        e2 = Expression.quantity("E", ENERGY)
        expr = Expression.build("+", e1, e2)
        assert expr.dim == ENERGY

    def test_build_add_incompatible_raises(self):
        """m + g → ERROR (Mass + Accel)."""
        m = Expression.quantity("m", MASS)
        g = Expression.quantity("g", ACCEL)
        with pytest.raises(DimensionError, match="incompatible"):
            Expression.build("+", m, g)

    def test_build_add_mass_velocity_raises(self):
        """m + v → ERROR (kg + m/s). Per spec: m+g is NOT."""
        m = Expression.quantity("m", MASS)
        v = Expression.quantity("v", VELOCITY)
        with pytest.raises(DimensionError):
            Expression.build("+", m, v)

    def test_build_sub_incompatible_raises(self):
        """v - h → ERROR."""
        v = Expression.quantity("v", VELOCITY)
        h = Expression.quantity("h", LENGTH)
        with pytest.raises(DimensionError):
            Expression.build("-", v, h)


class TestExpressionDimensions:
    """Test that dimension computation is correct for compound expressions."""

    def test_mass_times_velocity(self):
        """m * v → kg·m/s (momentum-type dimension)."""
        m = Expression.quantity("m", MASS)
        v = Expression.quantity("v", VELOCITY)
        expr = Expression.build("*", m, v)
        assert expr.dim == Dimension.from_exponents(
            {"Mass": 1.0, "Length": 1.0, "Time": -1.0}
        )

    def test_half_m_v_squared(self):
        """0.5 * m * v^2 → Energy."""
        half = Expression.constant("0.5", SCALAR)
        m = Expression.quantity("m", MASS)
        v = Expression.quantity("v", VELOCITY)
        two = Expression.constant("2", SCALAR)
        v2 = Expression.build("^", v, two)
        m_v2 = Expression.build("*", m, v2)
        expr = Expression.build("*", half, m_v2)
        assert expr.dim == ENERGY

    def test_m_g_h(self):
        """m * g * h → Energy."""
        m = Expression.quantity("m", MASS)
        g = Expression.quantity("g", ACCEL)
        h = Expression.quantity("h", LENGTH)
        mg = Expression.build("*", m, g)      # Force
        mgh = Expression.build("*", mg, h)    # Energy
        assert mgh.dim == ENERGY

    def test_m_g_h_plus_half_m_v_squared(self):
        """m*g*h + 0.5*m*v^2 → Energy."""
        m = Expression.quantity("m", MASS)
        g = Expression.quantity("g", ACCEL)
        h = Expression.quantity("h", LENGTH)
        v = Expression.quantity("v", VELOCITY)
        two = Expression.constant("2", SCALAR)
        half = Expression.constant("0.5", SCALAR)

        mgh = Expression.build("*", Expression.build("*", m, g), h)
        v2 = Expression.build("^", v, two)
        half_m_v2 = Expression.build("*", half, Expression.build("*", m, v2))
        total = Expression.build("+", mgh, half_m_v2)
        assert total.dim == ENERGY


class TestExpressionString:
    """Test string representation of expressions."""

    def test_quantity_str(self):
        assert str(Expression.quantity("m", MASS)) == "m"

    def test_constant_str(self):
        assert str(Expression.constant("0.5", SCALAR)) == "0.5"

    def test_mul_str(self):
        m = Expression.quantity("m", MASS)
        v = Expression.quantity("v", VELOCITY)
        expr = Expression.build("*", m, v)
        assert str(expr) == "m*v"

    def test_add_str(self):
        e1 = Expression.quantity("E", ENERGY)
        e2 = Expression.quantity("E", ENERGY)
        expr = Expression.build("+", e1, e2)
        assert str(expr) == "E+E"

    def test_pow_str(self):
        v = Expression.quantity("v", VELOCITY)
        two = Expression.constant("2", SCALAR)
        expr = Expression.build("^", v, two)
        assert str(expr) == "v^2"

    def test_nested_parens(self):
        """a*b + c: multiplication binds tighter, no extra parens needed."""
        a = Expression.quantity("a", SCALAR)
        b = Expression.quantity("b", SCALAR)
        c = Expression.quantity("c", SCALAR)
        ab = Expression.build("*", a, b)
        ab_plus_c = Expression.build("+", ab, c)
        # a*b+c — no parens needed since * binds tighter than +
        assert "(" not in str(ab_plus_c) or str(ab_plus_c) == "a*b+c"

    def test_fraction_exponent(self):
        """v^(1/2) should work."""
        v = Expression.quantity("v", VELOCITY)
        half = Expression.constant("1/2", SCALAR)
        expr = Expression.build("^", v, half)
        assert "v" in str(expr)


class TestExpressionProperties:
    """Test depth, equality, and hashing."""

    def test_leaf_depth(self):
        assert Expression.quantity("m", MASS).depth() == 1
        assert Expression.constant("0.5", SCALAR).depth() == 1

    def test_binary_depth_2(self):
        m = Expression.quantity("m", MASS)
        g = Expression.quantity("g", ACCEL)
        expr = Expression.build("*", m, g)
        assert expr.depth() == 2

    def test_depth_3(self):
        """m * g * h → depth 3 (max child = depth 2 + 1)."""
        m = Expression.quantity("m", MASS)
        g = Expression.quantity("g", ACCEL)
        h = Expression.quantity("h", LENGTH)
        mg = Expression.build("*", m, g)
        mgh = Expression.build("*", mg, h)
        assert mgh.depth() == 3

    def test_depth_4(self):
        """0.5 * m * v^2 → depth 4."""
        half = Expression.constant("0.5", SCALAR)
        m = Expression.quantity("m", MASS)
        v = Expression.quantity("v", VELOCITY)
        two = Expression.constant("2", SCALAR)
        v2 = Expression.build("^", v, two)       # depth 2
        m_v2 = Expression.build("*", m, v2)       # depth 3
        expr = Expression.build("*", half, m_v2)  # depth 4
        assert expr.depth() == 4

    def test_equality_equal(self):
        a = Expression.build("*",
            Expression.quantity("m", MASS),
            Expression.quantity("v", VELOCITY))
        b = Expression.build("*",
            Expression.quantity("m", MASS),
            Expression.quantity("v", VELOCITY))
        assert a == b
        assert hash(a) == hash(b)

    def test_equality_different(self):
        a = Expression.build("*",
            Expression.quantity("m", MASS),
            Expression.quantity("v", VELOCITY))
        b = Expression.build("*",
            Expression.quantity("m", MASS),
            Expression.quantity("g", ACCEL))
        assert a != b


class TestEdgeCases:
    """Edge case handling."""

    def test_unknown_operator(self):
        with pytest.raises(ValueError, match="Unknown binary operator"):
            Expression.build("%",
                Expression.quantity("m", MASS),
                Expression.quantity("v", VELOCITY))

    def test_scalar_times_scalar(self):
        """0.5 * 2 → Scalar."""
        expr = Expression.build("*",
            Expression.constant("0.5", SCALAR),
            Expression.constant("2", SCALAR))
        assert expr.dim == SCALAR
