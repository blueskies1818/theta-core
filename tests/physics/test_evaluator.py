"""Tests for the physics expression evaluator."""

import math
from pathlib import Path

import pytest

from src.physics.evaluator import (
    ExpressionEvaluator,
    EvalError,
    ParseError,
    parse_expression,
    evaluate_node,
)
from src.physics.observations import Observation, ObservationDatabase


PHASE1_PATH = Path(__file__).parent.parent.parent / "data" / "observations" / "phase1_falling.json"


@pytest.fixture
def db() -> ObservationDatabase:
    return ObservationDatabase(PHASE1_PATH)


@pytest.fixture
def ev() -> ExpressionEvaluator:
    return ExpressionEvaluator()


# ── Parsing tests ────────────────────────────────────────────────────────

class TestParsing:
    """Tests for expression parsing."""

    @pytest.mark.parametrize("expr_str", [
        "m*g*h + 0.5*m*v^2",
        "m*g*h+0.5*m*v^2",
        "m * g * h + 0.5 * m * v ^ 2",
        "m",
        "42",
        "3.14",
        "v^2",
        "(m*v)",
        "-x",
        "+x",
        "sin(theta)",
        "cos(omega*t)",
        "sqrt(2*g*h)",
        "exp(-t)",
        "log(abs(x))",
    ])
    def test_parses_valid_expressions(self, expr_str: str) -> None:
        """Should parse without errors."""
        ast = parse_expression(expr_str)
        assert ast is not None

    @pytest.mark.parametrize("expr_str", [
        "",
        "m + ",
        ")m",
        "m*/v",
        "m**v",
        "m+*v",
    ])
    def test_rejects_invalid_expressions(self, expr_str: str) -> None:
        """Should raise ParseError for invalid expressions."""
        with pytest.raises(ParseError):
            parse_expression(expr_str)


# ── Evaluation tests ─────────────────────────────────────────────────────

class TestExpressionEvaluation:
    """Tests for AST evaluation against contexts."""

    def test_simple_arithmetic(self) -> None:
        """Evaluate basic math expressions."""
        ast = parse_expression("2 + 3 * 4")
        result = evaluate_node(ast, {})
        assert result == 14.0

    def test_variable_substitution(self) -> None:
        """Should substitute variable values from context."""
        ast = parse_expression("a + b * c")
        result = evaluate_node(ast, {"a": 1.0, "b": 2.0, "c": 3.0})
        assert result == 7.0

    def test_power_operator(self) -> None:
        """^ should exponentiate."""
        ast = parse_expression("x^2")
        result = evaluate_node(ast, {"x": 3.0})
        assert result == 9.0

    def test_undefined_variable(self) -> None:
        """Should raise EvalError for undefined variables."""
        ast = parse_expression("x + y")
        with pytest.raises(EvalError, match="Undefined"):
            evaluate_node(ast, {"x": 1.0})

    def test_division_by_zero(self) -> None:
        """Should raise EvalError on division by zero."""
        ast = parse_expression("1/x")
        with pytest.raises(EvalError, match="Division by zero"):
            evaluate_node(ast, {"x": 0.0})

    def test_sin_function(self) -> None:
        """Should evaluate sin()."""
        ast = parse_expression("sin(x)")
        result = evaluate_node(ast, {"x": math.pi / 2})
        assert abs(result - 1.0) < 1e-10

    def test_cos_function(self) -> None:
        """Should evaluate cos()."""
        ast = parse_expression("cos(x)")
        result = evaluate_node(ast, {"x": 0.0})
        assert abs(result - 1.0) < 1e-10

    def test_sqrt_function(self) -> None:
        """Should evaluate sqrt()."""
        ast = parse_expression("sqrt(x)")
        result = evaluate_node(ast, {"x": 16.0})
        assert result == 4.0

    def test_negative_domain_error(self) -> None:
        """sqrt(-1) should raise EvalError."""
        ast = parse_expression("sqrt(x)")
        with pytest.raises(EvalError):
            evaluate_node(ast, {"x": -1.0})

    def test_precedence_mult_after_pow(self) -> None:
        """0.5*m*v^2 should parse as 0.5 * m * (v^2)."""
        ast = parse_expression("0.5*m*v^2")
        # Evaluate with m=2, v=3 -> 0.5 * 2 * 9 = 9
        result = evaluate_node(ast, {"m": 2.0, "v": 3.0})
        assert result == 9.0


# ── Evaluator (scoring) tests ────────────────────────────────────────────

class TestEvaluator:
    """Tests for the full ExpressionEvaluator scoring pipeline."""

    def test_energy_invariant_scores_high_straight_drop(
        self, ev: ExpressionEvaluator, db: ObservationDatabase
    ) -> None:
        """ACCEPTANCE: m*g*h + 0.5*m*v^2 > 0.95 on straight drop."""
        obs = db.get("falling_ball_straight_drop")
        score = ev.score("m*g*h + 0.5*m*v^2", obs)
        assert score > 0.95, (
            f"Expected energy invariant score > 0.95, got {score:.6f}"
        )

    def test_random_expression_scores_low(
        self, ev: ExpressionEvaluator, db: ObservationDatabase
    ) -> None:
        """m*v varies linearly in straight_drop, scores < 0.6.

        m is constant (parameter), so m*v varies linearly from 0.
        For linear variation, std/μ ≈ 0.79 → constancy ≈ 0.56.
        """
        obs = db.get("falling_ball_straight_drop")
        score = ev.score("m*v", obs)
        assert score < 0.6, (
            f"Expected m*v score < 0.6, got {score:.6f}"
        )

    def test_energy_invariant_scores_high_across_all_falling(
        self, ev: ExpressionEvaluator, db: ObservationDatabase
    ) -> None:
        """Energy conserved across all falling ball scenarios."""
        falling_ids = [
            "falling_ball_straight_drop",
            "falling_ball_upward_throw",
            "falling_ball_varying_mass",
        ]
        for oid in falling_ids:
            obs = db.get(oid)
            score = ev.score("m*g*h + 0.5*m*v^2", obs)
            assert score > 0.95, (
                f"Expected energy score > 0.95 for {oid}, got {score:.6f}"
            )

    def test_energy_conserved_on_projectiles(
        self, ev: ExpressionEvaluator, db: ObservationDatabase
    ) -> None:
        """Energy conserved on projectile scenarios."""
        for oid in ["projectile_45deg", "projectile_90deg"]:
            obs = db.get(oid)
            score = ev.score("m*g*h + 0.5*m*v^2", obs)
            assert score > 0.95, (
                f"Expected energy score > 0.95 for {oid}, got {score:.6f}"
            )

    def test_energy_conserved_on_pendulum_small(
        self, ev: ExpressionEvaluator, db: ObservationDatabase
    ) -> None:
        """Energy approximately conserved on small-angle pendulum."""
        obs = db.get("pendulum_small_angle")
        score = ev.score("m*g*h + 0.5*m*v^2", obs)
        assert score > 0.90, (
            f"Expected energy score > 0.90 on small pendulum, got {score:.6f}"
        )

    def test_spring_energy_conserved_undamped(
        self, ev: ExpressionEvaluator, db: ObservationDatabase
    ) -> None:
        """Spring energy conserved on undamped spring."""
        obs = db.get("spring_undamped")
        score = ev.score("0.5*m*v^2 + 0.5*k*h^2", obs)
        assert score > 0.95, (
            f"Expected spring energy score > 0.95, got {score:.6f}"
        )

    def test_spring_energy_not_conserved_damped(
        self, ev: ExpressionEvaluator, db: ObservationDatabase
    ) -> None:
        """Spring energy NOT conserved on damped spring."""
        obs = db.get("spring_damped_light")
        score = ev.score("0.5*m*v^2 + 0.5*k*h^2", obs)
        assert score < 0.7, (
            f"Expected damped spring energy score < 0.7, got {score:.6f}"
        )

    def test_energy_conserved_on_incline(
        self, ev: ExpressionEvaluator, db: ObservationDatabase
    ) -> None:
        """Energy conserved on frictionless incline."""
        obs = db.get("sliding_block_incline")
        score = ev.score("m*g*h + 0.5*m*v^2", obs)
        assert score > 0.95, (
            f"Expected energy score > 0.95 on incline, got {score:.6f}"
        )

    def test_invalid_expression_returns_zero(
        self, ev: ExpressionEvaluator, db: ObservationDatabase
    ) -> None:
        """Unparseable expressions should score 0.0."""
        obs = db.get("falling_ball_straight_drop")
        score = ev.score("m + +", obs)
        assert score == 0.0

    def test_undefined_variable_returns_zero(
        self, ev: ExpressionEvaluator, db: ObservationDatabase
    ) -> None:
        """Expressions with undefined variables score 0.0."""
        obs = db.get("falling_ball_straight_drop")
        score = ev.score("z * q", obs)
        assert score == 0.0

    def test_score_across_all_observations(
        self, ev: ExpressionEvaluator, db: ObservationDatabase
    ) -> None:
        """Score against entire database returns mean constancy."""
        score = ev.score("m*g*h + 0.5*m*v^2", db)
        assert 0.0 <= score <= 1.0

    def test_constant_expression_scores_one(
        self, ev: ExpressionEvaluator, db: ObservationDatabase
    ) -> None:
        """A literal constant should score 1.0."""
        obs = db.get("falling_ball_straight_drop")
        score = ev.score("42", obs)
        assert abs(score - 1.0) < 1e-6

    def test_linear_with_time_scores_low(
        self, ev: ExpressionEvaluator, db: ObservationDatabase
    ) -> None:
        """Velocity alone varies, so score should be low."""
        obs = db.get("falling_ball_straight_drop")
        score = ev.score("v", obs)
        assert score < 0.6, f"Expected velocity alone score < 0.6, got {score:.6f}"

    def test_score_all_returns_per_obs_scores(
        self, ev: ExpressionEvaluator, db: ObservationDatabase
    ) -> None:
        """score_all returns list of per-observation scores."""
        scores = ev.score_all("m*g*h + 0.5*m*v^2", db)
        assert len(scores) == len(db)
        assert all(0.0 <= s <= 1.0 for s in scores)
