"""Unit tests for physical dimension type system."""

import pytest
from src.physics.dimensions import Dimension, DimensionError


# ── Convenience module-level dimension constants ───────────────────

MASS = Dimension.named("Mass")
LENGTH = Dimension.named("Length")
TIME = Dimension.named("Time")
SCALAR = Dimension.scalar()
VELOCITY = Dimension.named("Velocity")
ACCEL = Dimension.named("Accel")
FORCE = Dimension.named("Force")
ENERGY = Dimension.named("Energy")


class TestDimensionConstants:
    """Verify predefined dimension constants."""

    def test_scalar_is_dimensionless(self):
        assert SCALAR.is_scalar()

    def test_mass_dimension(self):
        assert MASS == Dimension.from_exponents({"Mass": 1.0})

    def test_length_dimension(self):
        assert LENGTH == Dimension.from_exponents({"Length": 1.0})

    def test_time_dimension(self):
        assert TIME == Dimension.from_exponents({"Time": 1.0})

    def test_velocity_dimension(self):
        assert VELOCITY == Dimension.from_exponents({"Length": 1.0, "Time": -1.0})

    def test_acceleration_dimension(self):
        assert ACCEL == Dimension.from_exponents({"Length": 1.0, "Time": -2.0})

    def test_force_dimension(self):
        assert FORCE == Dimension.from_exponents(
            {"Mass": 1.0, "Length": 1.0, "Time": -2.0}
        )

    def test_energy_dimension(self):
        assert ENERGY == Dimension.from_exponents(
            {"Mass": 1.0, "Length": 2.0, "Time": -2.0}
        )

    def test_aliases(self):
        assert Dimension.named("Accel") == ACCEL


class TestDimensionArithmetic:
    """Test dimension algebra: *, /, **."""

    # ── addition / subtraction compatibility ───────────────────────

    def test_check_add_same_type(self):
        MASS.check_add(MASS)  # should not raise

    def test_check_add_incompatible_raises(self):
        with pytest.raises(DimensionError, match="incompatible"):
            MASS.check_add(LENGTH)

    def test_check_add_kg_plus_ms_raises(self):
        """kg + m/s = ERROR per spec."""
        with pytest.raises(DimensionError):
            MASS.check_add(VELOCITY)

    # ── multiplication ─────────────────────────────────────────────

    def test_mul_scalar_times_mass(self):
        assert SCALAR * MASS == MASS

    def test_mul_mass_times_accel(self):
        """Mass * Accel → Force: kg * m/s² = kg·m/s²."""
        assert MASS * ACCEL == FORCE

    def test_mul_force_times_length(self):
        """Force * Length → Energy: kg·m/s² * m = kg·m²/s²."""
        assert FORCE * LENGTH == ENERGY

    def test_mul_velocity_times_time(self):
        """Velocity * Time → Length: m/s * s = m."""
        assert VELOCITY * TIME == LENGTH

    def test_mul_mass_times_velocity_squared(self):
        """Mass * Velocity² → Energy: kg * (m²/s²) = kg·m²/s²."""
        v2 = VELOCITY * VELOCITY
        assert v2 == Dimension.from_exponents({"Length": 2.0, "Time": -2.0})
        assert MASS * v2 == ENERGY

    def test_mul_commutative(self):
        assert MASS * VELOCITY == VELOCITY * MASS

    # ── division ───────────────────────────────────────────────────

    def test_div_length_by_time(self):
        """Length / Time → Velocity: m / s = m/s."""
        assert LENGTH / TIME == VELOCITY

    def test_div_velocity_by_time(self):
        """Velocity / Time → Accel: m/s / s = m/s²."""
        assert VELOCITY / TIME == ACCEL

    def test_div_energy_by_force(self):
        """Energy / Force → Length: kg·m²/s² / (kg·m/s²) = m."""
        assert ENERGY / FORCE == LENGTH

    def test_div_energy_by_time(self):
        """Energy / Time: kg·m²/s² / s = kg·m²/s³ (compound, not built-in)."""
        result = ENERGY / TIME
        assert result == Dimension.from_exponents(
            {"Mass": 1.0, "Length": 2.0, "Time": -3.0}
        )

    # ── power ──────────────────────────────────────────────────────

    def test_power_length_squared(self):
        """Length² → m²."""
        assert LENGTH ** 2 == Dimension.from_exponents({"Length": 2.0})

    def test_power_velocity_squared(self):
        """Velocity² → m²/s²."""
        assert VELOCITY ** 2 == Dimension.from_exponents(
            {"Length": 2.0, "Time": -2.0}
        )

    def test_power_to_zero_is_scalar(self):
        assert MASS ** 0 == SCALAR
        assert VELOCITY ** 0 == SCALAR

    def test_power_scalar(self):
        assert SCALAR ** 5 == SCALAR

    # ── compound examples ──────────────────────────────────────────

    def test_energy_times_time_equals_mass_length_squared_over_time(self):
        """kg·m²/s² × s = kg·m²/s (energy × time)."""
        result = ENERGY * TIME
        assert result == Dimension.from_exponents(
            {"Mass": 1.0, "Length": 2.0, "Time": -1.0}
        )

    def test_energy_div_length(self):
        """kg·m²/s² / m = kg·m/s² = Force."""
        assert ENERGY / LENGTH == FORCE

    def test_velocity_squared_dimension(self):
        """(m/s)² = m²/s²."""
        v2 = VELOCITY ** 2
        assert v2 == Dimension.from_exponents({"Length": 2.0, "Time": -2.0})


class TestDimensionNaming:
    """Test human-readable dimension names."""

    def test_known_names(self):
        assert str(SCALAR) == "Scalar"
        assert str(MASS) == "Mass"
        assert str(LENGTH) == "Length"
        assert str(TIME) == "Time"
        assert str(VELOCITY) == "Velocity"
        assert str(ACCEL) == "Accel"
        assert str(FORCE) == "Force"
        assert str(ENERGY) == "Energy"

    def test_compound_name(self):
        """kg·m²/s³ should produce a symbolic name."""
        compound = Dimension.from_exponents(
            {"Mass": 1.0, "Length": 2.0, "Time": -3.0}
        )
        name = str(compound)
        assert "kg" in name or "m" in name

    def test_scalar_compound(self):
        assert str(Dimension.scalar()) == "Scalar"


class TestDimensionNamed:
    """Test Dimension.named() factory."""

    def test_named_known(self):
        assert Dimension.named("Scalar") == SCALAR
        assert Dimension.named("Mass") == MASS
        assert Dimension.named("Energy") == ENERGY

    def test_named_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown dimension"):
            Dimension.named("Momentum")


class TestDimensionHelpers:
    """Test is_scalar, compatible_with, check_power_exponent."""

    def test_scalar_check(self):
        assert SCALAR.is_scalar()
        assert not MASS.is_scalar()
        assert not VELOCITY.is_scalar()

    def test_compatible_same(self):
        assert MASS.compatible_with(MASS)
        assert ENERGY.compatible_with(ENERGY)

    def test_compatible_different(self):
        assert not MASS.compatible_with(LENGTH)

    def test_check_power_exponent_scalar(self):
        SCALAR.check_power_exponent()  # should not raise

    def test_check_power_exponent_non_scalar_raises(self):
        with pytest.raises(DimensionError, match="scalar"):
            MASS.check_power_exponent()


class TestDimensionImmutability:
    """Dimension hashing and equality."""

    def test_hashable(self):
        d = {SCALAR: "dimensionless", MASS: "kilogram"}
        assert d[SCALAR] == "dimensionless"

    def test_set_membership(self):
        dims = {SCALAR, MASS, LENGTH}
        assert VELOCITY not in dims
        assert MASS in dims

    def test_equality(self):
        a = Dimension.named("Force")
        b = Dimension.from_exponents(
            {"Mass": 1.0, "Length": 1.0, "Time": -2.0}
        )
        assert a == b
        assert hash(a) == hash(b)
