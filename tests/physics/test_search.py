"""Tests for the best-first expression search engine."""

from pathlib import Path

import pytest

from src.physics.dimensions import Dimension
from src.physics.observations import ObservationDatabase
from src.physics.search import ExpressionSearch, SearchResult

PHASE1_PATH = Path(__file__).parent.parent.parent / "data" / "observations" / "phase1_falling.json"


@pytest.fixture
def db() -> ObservationDatabase:
    return ObservationDatabase(PHASE1_PATH)


@pytest.fixture
def train_obs_gravitational(db: ObservationDatabase):
    """8 gravitational scenarios where m*g*h + 0.5*m*v^2 is conserved."""
    train_ids = [
        "falling_ball_straight_drop",
        "falling_ball_upward_throw",
        "falling_ball_varying_mass",
        "pendulum_small_angle",
        "pendulum_large_angle",
        "projectile_45deg",
        "projectile_90deg",
        "sliding_block_incline",
    ]
    return [db.get(oid) for oid in train_ids]


@pytest.fixture
def quantities(train_obs_gravitational):
    """Extract quantity dimensions from training observations."""
    quantities: dict[str, Dimension] = {}
    for obs in train_obs_gravitational:
        for name, dim_name in obs.quantities.items():
            if name not in quantities:
                quantities[name] = Dimension.named(dim_name)
    return quantities


# ── Search initialization ─────────────────────────────────────────────────────

class TestSearchInitialization:

    def test_creates_search_engine(self, quantities, train_obs_gravitational) -> None:
        search = ExpressionSearch(quantities, train_obs_gravitational)
        assert search is not None
        assert search.max_depth == 6
        assert search.max_expansions == 10_000

    def test_custom_parameters(self, quantities, train_obs_gravitational) -> None:
        search = ExpressionSearch(
            quantities, train_obs_gravitational,
            max_depth=4, max_expansions=500,
            depth_discount=0.9, discovery_threshold=0.9,
        )
        assert search.max_depth == 4
        assert search.max_expansions == 500
        assert search.depth_discount == 0.9
        assert search.discovery_threshold == 0.9

    def test_initial_best_is_zero(self, quantities, train_obs_gravitational) -> None:
        search = ExpressionSearch(quantities, train_obs_gravitational)
        assert search.best_score == 0.0
        assert search.best_expression == ""


# ── Scoring ───────────────────────────────────────────────────────────────────

class TestSearchScoring:

    def test_constant_scores_one(self, quantities, train_obs_gravitational) -> None:
        search = ExpressionSearch(quantities, train_obs_gravitational)
        score = search._score_expression("42")
        assert score > 0.99

    def test_energy_expression_scores_high(self, quantities, train_obs_gravitational) -> None:
        search = ExpressionSearch(quantities, train_obs_gravitational)
        score = search._score_expression("m*g*h + 0.5*m*v^2")
        assert score > 0.90, f"Expected > 0.90, got {score:.6f}"

    def test_v_scores_low(self, quantities, train_obs_gravitational) -> None:
        search = ExpressionSearch(quantities, train_obs_gravitational)
        score = search._score_expression("v")
        assert score < 0.65, f"Expected < 0.65, got {score:.6f}"


# ── Dimension computation ─────────────────────────────────────────────────────

class TestDimensionComputation:

    def test_scalar_leaf_dimension(self, quantities, train_obs_gravitational) -> None:
        search = ExpressionSearch(quantities, train_obs_gravitational)
        dim = search._dimension_of("42")
        assert dim is not None
        assert dim.is_scalar()

    def test_mass_dimension(self, quantities, train_obs_gravitational) -> None:
        search = ExpressionSearch(quantities, train_obs_gravitational)
        dim = search._dimension_of("m")
        assert dim is not None
        assert dim == Dimension.named("Mass")

    def test_add_incompatible_returns_none(self, quantities, train_obs_gravitational) -> None:
        search = ExpressionSearch(quantities, train_obs_gravitational)
        dim = search._dimension_of("m+v")
        assert dim is None


# ── Full search ───────────────────────────────────────────────────────────────

class TestFullSearch:

    def test_search_discovers_energy_within_budget(
        self, quantities, train_obs_gravitational
    ) -> None:
        """ACCEPTANCE: Search discovers energy conservation in < 20000 expansions."""
        search = ExpressionSearch(
            quantities, train_obs_gravitational,
            max_depth=6, max_expansions=20_000, discovery_threshold=0.95,
        )
        result = search.run()
        assert result.is_discovery, (
            f"Search did not discover invariant. "
            f"Best: {result.expression} score={result.score:.6f} "
            f"after {result.expansions} expansions"
        )
        assert result.expansions < 20_000
        assert result.score > 0.95

    def test_search_result_has_all_fields(
        self, quantities, train_obs_gravitational
    ) -> None:
        search = ExpressionSearch(quantities, train_obs_gravitational, max_expansions=200)
        result = search.run()
        assert isinstance(result.expression, str)
        assert len(result.expression) > 0
        assert isinstance(result.score, float)
        assert 0.0 <= result.score <= 1.0
        assert isinstance(result.depth, int)
        assert result.depth >= 1
        assert isinstance(result.expansions, int)
        assert result.expansions > 0
        assert isinstance(result.train_constancies, list)
        assert len(result.train_constancies) == len(train_obs_gravitational)

    def test_search_with_snapshots(self, quantities, train_obs_gravitational) -> None:
        search = ExpressionSearch(quantities, train_obs_gravitational, max_expansions=500)
        snapshots = list(search.run_with_snapshots())
        assert len(snapshots) > 0
        for count, result in snapshots:
            assert isinstance(count, int)
            assert isinstance(result, SearchResult)
            assert count == result.expansions
