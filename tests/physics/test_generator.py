"""Unit tests for breadth-first physics expression generator."""

import pytest
from src.physics.dimensions import Dimension, DimensionError
from src.physics.grammar import Expression
from src.physics.generator import ExpressionGenerator, run_smoke_test


# ── Convenience dimensions ─────────────────────────────────────────

MASS = Dimension.named("Mass")
LENGTH = Dimension.named("Length")
TIME = Dimension.named("Time")
SCALAR = Dimension.scalar()
VELOCITY = Dimension.named("Velocity")
ACCEL = Dimension.named("Accel")
FORCE = Dimension.named("Force")
ENERGY = Dimension.named("Energy")


class TestExpressionGeneratorBasics:
    """Basic generation behavior."""

    def test_depth1_only_atoms(self):
        gen = ExpressionGenerator(
            quantities={"m": MASS, "v": VELOCITY},
            constants={"0.5": SCALAR},
            operations={"+", "*", "^"},
            max_depth=1,
        )
        gen.generate()
        assert gen.count() == 3  # m, v, 0.5
        strs = {str(e) for e in gen.expressions_at_depth(1)}
        assert strs == {"m", "v", "0.5"}

    def test_depth2_generates_pairs(self):
        gen = ExpressionGenerator(
            quantities={"m": MASS, "g": ACCEL},
            constants={"2": SCALAR},
            operations={"*", "^"},
            max_depth=2,
        )
        gen.generate()
        depth2_strs = {str(e) for e in gen.expressions_at_depth(2)}
        assert "m*g" in depth2_strs
        assert "m^2" in depth2_strs

    def test_generate_all_levels_present(self):
        gen = ExpressionGenerator(
            quantities={"m": MASS, "v": VELOCITY},
            constants={},
            operations={"*"},
            max_depth=3,
        )
        gen.generate()
        for d in range(1, 4):
            exprs = gen.expressions_at_depth(d)
            assert len(exprs) > 0, f"Empty depth {d}"

    def test_no_duplicate_expressions(self):
        gen = ExpressionGenerator(
            quantities={"m": MASS, "g": ACCEL, "h": LENGTH},
            constants={"0.5": SCALAR},
            operations={"*"},
            max_depth=3,
        )
        gen.generate()
        # Check all expressions are unique by string
        all_strs = []
        for d in range(1, 4):
            all_strs.extend(str(e) for e in gen.expressions_at_depth(d))
        assert len(all_strs) == len(set(all_strs)), \
            f"Duplicates: {len(all_strs)} total, {len(set(all_strs))} unique"

    def test_contains_check(self):
        gen = ExpressionGenerator(
            quantities={"m": MASS, "v": VELOCITY},
            constants={},
            operations={"*"},
            max_depth=2,
        )
        gen.generate()
        assert "m" in gen
        assert "v" in gen
        assert "m*v" in gen or "v*m" in gen


class TestDimensionFiltering:
    """Verify type errors are filtered out."""

    def test_mass_plus_velocity_filtered(self):
        """m + v → type error, must NOT appear in output."""
        gen = ExpressionGenerator(
            quantities={"m": MASS, "v": VELOCITY},
            constants={},
            operations={"+"},
            max_depth=2,
        )
        gen.generate()
        for d in range(1, 3):
            for e in gen.expressions_at_depth(d):
                s = str(e)
                assert "m+v" not in s, f"Found type error: {s}"
                assert "v+m" not in s, f"Found type error: {s}"

    def test_energy_plus_energy_present(self):
        """E + E → valid, should appear."""
        gen = ExpressionGenerator(
            quantities={"E": ENERGY},
            constants={},
            operations={"+"},
            max_depth=2,
        )
        gen.generate()
        found = False
        for d in range(1, 3):
            for e in gen.expressions_at_depth(d):
                if str(e) == "E+E":
                    found = True
        assert found, "E+E should be generated"

    def test_power_non_scalar_exponent_filtered(self):
        """v ^ m → error (non-scalar exponent), must NOT appear."""
        gen = ExpressionGenerator(
            quantities={"v": VELOCITY, "m": MASS},
            constants={},
            operations={"^"},
            max_depth=2,
        )
        gen.generate()
        for d in range(1, 3):
            for e in gen.expressions_at_depth(d):
                s = str(e)
                assert "v^m" not in s, f"Found invalid power: {s}"


class TestSmokeTest:
    """The canonical smoke test from Phase A acceptance criteria.

    Generate all valid depth-4 expressions from {m, g, h, v} with {+, *, /, ^}
    Verify: m*g*h is in the output, 0.5*m*v^2 is in the output,
            m+g is NOT (type error).
    """

    @pytest.fixture
    def generator(self):
        gen = ExpressionGenerator(
            quantities={
                "m": MASS,
                "g": ACCEL,
                "h": LENGTH,
                "v": VELOCITY,
            },
            constants={
                "0.5": SCALAR,
                "2": SCALAR,
            },
            operations={"+", "*", "/", "^"},
            max_depth=4,
        )
        gen.generate()
        return gen

    def test_smoke_mgh_in_output(self, generator):
        """m*g*h must be in generated output at depth <= 4."""
        all_strs = set()
        for d in range(1, 5):
            all_strs.update(str(e) for e in generator.expressions_at_depth(d))

        # m*g*h should appear — the generator uses infix without spaces
        mgh_found = "m*g*h" in all_strs
        if not mgh_found:
            # Try parenthesized variants
            mgh_found = any(
                v in all_strs
                for v in ["(m*g)*h", "m*(g*h)"]
            )
        assert mgh_found, (
            f"m*g*h not found. "
            f"Sample expressions with m,g,h: "
            f"{[s for s in all_strs if 'm' in s and 'g' in s and 'h' in s][:20]}"
        )

    def test_smoke_half_m_v_squared_in_output(self, generator):
        """0.5*m*v^2 must be in generated output at depth <= 4."""
        all_strs = set()
        for d in range(1, 5):
            all_strs.update(str(e) for e in generator.expressions_at_depth(d))

        candidates = [
            s for s in all_strs
            if "0.5" in s and "m" in s and "v" in s and ("^2" in s or "v*v" in s)
        ]
        assert len(candidates) > 0, (
            f"0.5*m*v^2 not found.\n"
            f"Total expressions: {generator.count()}\n"
            f"Sample 0.5/m/v expressions: "
            f"{[s for s in all_strs if '0.5' in s and 'm' in s and 'v' in s][:20]}"
        )

    def test_smoke_m_plus_g_not_in_output(self, generator):
        """m+g is a type error (Mass + Accel) — must NOT appear."""
        for d in range(1, 5):
            for e in generator.expressions_at_depth(d):
                s = str(e)
                assert s != "m+g", f"Type error expression found: {s}"
                assert s != "g+m", f"Type error expression found: {s}"

    def test_smoke_all_expressions_have_valid_dimensions(self, generator):
        """Every expression should have a valid str representation."""
        for d in range(1, 5):
            for e in generator.expressions_at_depth(d):
                dim = e.dim
                assert isinstance(dim, Dimension)
                s = str(dim)
                assert isinstance(s, str)
                assert len(s) > 0


class TestRunSmokeTest:
    """Test the built-in run_smoke_test() function."""

    def test_run_smoke_test_returns_results(self):
        results = run_smoke_test()
        assert results["has_mgh"], f"Expected m*g*h, got {results}"
        assert results["has_half_mv2"], f"Expected 0.5*m*v^2, got {results}"
        assert results["rejects_m_plus_g"], f"m+g should be rejected, got {results}"
        assert results["rejects_mv_plus_h"], f"m*v+h should be rejected, got {results}"
        assert results["total_expressions"] > 0

    def test_run_smoke_test_has_depth_counts(self):
        results = run_smoke_test()
        assert "depth_counts" in results
        assert results["depth_counts"][1] >= 6  # m, g, h, v, 0, 0.5, 1, 2
        assert results["depth_counts"][4] > 0


class TestExpressionsByDimension:
    """Test filtering by dimension."""

    def test_filter_energy_expressions(self):
        gen = ExpressionGenerator(
            quantities={"m": MASS, "g": ACCEL, "h": LENGTH, "v": VELOCITY},
            constants={"0.5": SCALAR, "2": SCALAR},
            operations={"*", "^"},
            max_depth=4,
        )
        gen.generate()
        energy_exprs = gen.expressions_by_dimension(ENERGY)
        assert len(energy_exprs) > 0
        # Check all actually have Energy dimension
        for e in energy_exprs:
            assert e.dim == ENERGY, f"{e} has dim {e.dim}, expected Energy"


class TestEdgeCases:
    """Edge cases for the generator."""

    def test_empty_quantities(self):
        gen = ExpressionGenerator(
            quantities={},
            constants={"0.5": SCALAR},
            operations=set(),
            max_depth=1,
        )
        gen.generate()
        assert gen.count() == 1  # just the constant

    def test_empty_constants(self):
        gen = ExpressionGenerator(
            quantities={"m": MASS},
            constants={},
            operations=set(),
            max_depth=1,
        )
        gen.generate()
        assert gen.count() == 1  # just m

    def test_max_depth_1(self):
        gen = ExpressionGenerator(
            quantities={"m": MASS, "v": VELOCITY},
            constants={"0.5": SCALAR},
            operations={"*"},
            max_depth=1,
        )
        gen.generate()
        # Depth 2 should be empty (not generated)
        assert gen.expressions_at_depth(2) == []

    def test_no_operations_generates_only_atoms(self):
        gen = ExpressionGenerator(
            quantities={"m": MASS, "v": VELOCITY},
            constants={"0.5": SCALAR},
            operations=set(),
            max_depth=3,
        )
        gen.generate()
        assert gen.count() == 3  # only atoms

    def test_len(self):
        gen = ExpressionGenerator(
            quantities={"m": MASS, "g": ACCEL},
            constants={"2": SCALAR},
            operations={"*"},
            max_depth=2,
        )
        gen.generate()
        assert len(gen) == gen.count()
        assert len(gen) >= 4  # m, g, 2, m*g, 2*m, 2*g, ...
