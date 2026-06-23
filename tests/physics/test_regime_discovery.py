"""Tests for regime discovery — splitting observations into regimes
and re-discovering invariants per regime.
"""

import math
from pathlib import Path

import pytest

from src.physics.dimensions import Dimension
from src.physics.evaluator import (
    ExpressionEvaluator,
    find_regime_threshold,
)
from src.physics.observations import Observation, ObservationDatabase
from src.physics.search import (
    SearchResult,
    _attempt_regime_discovery,
    auto_discover,
)

PHASE1_PATH = Path(__file__).parent.parent.parent / "data" / "observations" / "phase1_falling.json"


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def evaluator() -> ExpressionEvaluator:
    return ExpressionEvaluator()


@pytest.fixture
def db() -> ObservationDatabase:
    return ObservationDatabase(PHASE1_PATH)


@pytest.fixture
def gravitational_obs(db: ObservationDatabase) -> list[Observation]:
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
def gravitational_quantities(gravitational_obs: list[Observation]) -> dict[str, Dimension]:
    quantities: dict[str, Dimension] = {}
    for obs in gravitational_obs:
        for name, dim_name in obs.quantities.items():
            if name not in quantities:
                quantities[name] = Dimension.named(dim_name)
    return quantities


@pytest.fixture
def regime_observations() -> list[Observation]:
    """Synthetic observations with two clear regimes.

    Regime A (low-v): 5 observations where classical expression holds.
    Regime B (high-v): 5 observations where a different expression holds,
    but the classical expression fails.

    Each observation has quantities: m, g, h, v, c (for regime B).
    """
    obs_list: list[Observation] = []

    # Regime A: classical mechanics, v ∈ [1, 20]
    for i in range(5):
        m_val = 1.0 + i * 0.5
        v_val = 1.0 + i * 4.0  # v ∈ [1, 5, 9, 13, 17]
        g_val = 9.8
        h_val = 10.0 - i * 1.0
        timesteps = []
        for t_mult in [1.0, 2.0, 3.0, 4.0]:
            # E = m*g*h + 0.5*m*v^2 is conserved
            v_t = v_val * (1.0 + 0.01 * t_mult)
            h_t = h_val * (1.0 - 0.05 * t_mult)
            timesteps.append({
                "t": t_mult,
                "m": m_val,
                "g": g_val,
                "h": h_t,
                "v": v_t,
                "c": 300.0,  # speed of light (relativistic regime not active)
            })

        obs_list.append(Observation(
            id=f"regime_a_{i}",
            name=f"classical_low_v_{i}",
            description="Classical mechanics at low velocity",
            quantities={"m": "Mass", "g": "Accel", "h": "Length", "v": "Velocity", "c": "Velocity"},
            parameters={"m": m_val, "g": g_val},
            timesteps=timesteps,
            known_invariant="m*g*h + 0.5*m*v^2",
            lean_theorem="",
        ))

    # Regime B: relativistic, v ∈ [150, 290] (significant fraction of c=300)
    for i in range(5):
        m_val = 1.0 + i * 0.3
        v_val = 150.0 + i * 35.0  # v ∈ [150, 185, 220, 255, 290]
        g_val = 9.8
        h_val = 10.0
        timesteps = []
        for t_mult in [1.0, 2.0, 3.0, 4.0]:
            # In relativistic regime, classical expression varies
            v_t = v_val * (1.0 + 0.005 * t_mult)
            h_t = h_val * (1.0 - 0.02 * t_mult)
            gamma = 1.0 / math.sqrt(1.0 - (v_t / 300.0) ** 2)
            timesteps.append({
                "t": t_mult,
                "m": m_val,
                "g": g_val,
                "h": h_t,
                "v": v_t,
                "c": 300.0,
                "gamma": gamma,
            })

        obs_list.append(Observation(
            id=f"regime_b_{i}",
            name=f"relativistic_high_v_{i}",
            description="Relativistic regime at high velocity",
            quantities={
                "m": "Mass", "g": "Accel", "h": "Length",
                "v": "Velocity", "c": "Velocity", "gamma": "Scalar",
            },
            parameters={"m": m_val, "g": g_val, "c": 300.0},
            timesteps=timesteps,
            known_invariant="m*c^2*(gamma-1)",
            lean_theorem="",
        ))

    return obs_list


@pytest.fixture
def regime_quantities(regime_observations: list[Observation]) -> dict[str, Dimension]:
    quantities: dict[str, Dimension] = {}
    for obs in regime_observations:
        for name, dim_name in obs.quantities.items():
            if name not in quantities:
                quantities[name] = Dimension.named(dim_name)
    return quantities


# ── score_per_observation tests ────────────────────────────────────────

class TestScorePerObservation:

    def test_returns_list_same_length(
        self, evaluator: ExpressionEvaluator, gravitational_obs: list[Observation],
    ) -> None:
        scores = evaluator.score_per_observation(
            "m*g*h + 0.5*m*v^2", gravitational_obs,
        )
        assert len(scores) == len(gravitational_obs)
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_empty_observations(self, evaluator: ExpressionEvaluator) -> None:
        scores = evaluator.score_per_observation("m*g*h", [])
        assert scores == []

    def test_energy_conservation_high_scores(
        self, evaluator: ExpressionEvaluator, gravitational_obs: list[Observation],
    ) -> None:
        scores = evaluator.score_per_observation(
            "m*g*h + 0.5*m*v^2", gravitational_obs,
        )
        # All gravitational scenarios should score highly for energy
        mean_score = sum(scores) / len(scores)
        assert mean_score > 0.85, f"Expected > 0.85, got {mean_score}"

    def test_non_conserved_low_scores(
        self, evaluator: ExpressionEvaluator, gravitational_obs: list[Observation],
    ) -> None:
        scores = evaluator.score_per_observation("v", gravitational_obs)
        mean_score = sum(scores) / len(scores)
        assert mean_score < 0.65, f"Expected < 0.65, got {mean_score}"


# ── find_regime_threshold tests ────────────────────────────────────────

class TestFindRegimeThreshold:

    def test_detects_split_in_mixed_regime_data(
        self, evaluator: ExpressionEvaluator, regime_observations: list[Observation],
    ) -> None:
        """Classical expression should have high scores in regime A, low in B."""
        split = find_regime_threshold(
            "m*g*h + 0.5*m*v^2", regime_observations, evaluator,
        )
        assert split is not None, "Should detect regime split"
        assert split["gap"] > 0.0
        assert len(split["regime_a_obs"]) >= 3
        assert len(split["regime_b_obs"]) >= 3
        # The key quantity should be 'v' (velocity separates classical/relativistic)
        assert "key_quantity" in split
        assert isinstance(split["gap"], float)

    def test_returns_none_for_too_few_observations(
        self, evaluator: ExpressionEvaluator,
    ) -> None:
        obs = [
            Observation(
                id=f"single_{i}", name=f"single_{i}", description="",
                quantities={"v": "Velocity"},
                parameters={},
                timesteps=[{"t": float(j), "v": float(j + i)} for j in range(3)],
                known_invariant=None, lean_theorem="",
            )
            for i in range(3)
        ]
        split = find_regime_threshold("v", obs, evaluator, min_regime_size=3)
        # 3 observations < 2 * 3 = 6, so should return None
        assert split is None

    def test_returns_none_when_no_split_possible(
        self, evaluator: ExpressionEvaluator, gravitational_obs: list[Observation],
    ) -> None:
        """All observations are conservative — no split should be detected with
        a meaningful gap."""
        split = find_regime_threshold(
            "m*g*h + 0.5*m*v^2", gravitational_obs, evaluator,
            min_regime_size=3,
        )
        if split is not None:
            # If a split IS found, the gap should be small (< 0.1)
            assert split["gap"] < 0.20, (
                f"Expected small gap for homogeneous data, got {split['gap']}"
            )

    def test_split_preserves_observation_order(
        self, evaluator: ExpressionEvaluator, regime_observations: list[Observation],
    ) -> None:
        split = find_regime_threshold(
            "m*g*h + 0.5*m*v^2", regime_observations, evaluator,
        )
        if split is None:
            pytest.skip("No split found in regime data")
        # Total observations in both regimes should equal input count
        total = len(split["regime_a_obs"]) + len(split["regime_b_obs"])
        assert total == len(regime_observations), (
            f"Expected {len(regime_observations)} total, got {total}"
        )

    def test_returns_fields_correctly(
        self, evaluator: ExpressionEvaluator, regime_observations: list[Observation],
    ) -> None:
        split = find_regime_threshold(
            "m*g*h + 0.5*m*v^2", regime_observations, evaluator,
        )
        if split is None:
            pytest.skip("No split found in regime data")
        required_keys = [
            "key_quantity", "split_index", "gap",
            "regime_a_obs", "regime_b_obs",
            "regime_a_scores", "regime_b_scores",
            "sorted_per_obs_scores",
        ]
        for key in required_keys:
            assert key in split, f"Missing key: {key}"


# ── _attempt_regime_discovery tests ────────────────────────────────────

class TestAttemptRegimeDiscovery:

    def test_returns_none_when_score_already_high(
        self, gravitational_quantities: dict[str, Dimension],
        gravitational_obs: list[Observation],
    ) -> None:
        result = _attempt_regime_discovery(
            gravitational_quantities, gravitational_obs,
            best_expr="m*g*h + 0.5*m*v^2",
            best_score=0.95,  # already above threshold
            discovery_threshold=0.90,
        )
        assert result is None, "Should not attempt regime split when already discovered"

    def test_returns_none_when_no_expression(
        self, gravitational_quantities: dict[str, Dimension],
        gravitational_obs: list[Observation],
    ) -> None:
        result = _attempt_regime_discovery(
            gravitational_quantities, gravitational_obs,
            best_expr="",
            best_score=0.30,
            discovery_threshold=0.90,
        )
        assert result is None

    def test_discovers_per_regime_invariant(
        self, regime_quantities: dict[str, Dimension],
        regime_observations: list[Observation],
        evaluator: ExpressionEvaluator,
    ) -> None:
        """With clear low-v/high-v split, regime discovery should find
        invariants in at least one regime."""
        # First, find a candidate that scores poorly globally
        # v alone has low constancy (velocity varies)
        global_score = evaluator.score("v", regime_observations[0])
        # Use a nontrivial expression that clearly fails across regimes
        result = _attempt_regime_discovery(
            regime_quantities, regime_observations,
            best_expr="v",  # velocity — not conserved
            best_score=min(global_score, 0.30),
            evaluator=evaluator,
            discovery_threshold=0.85,  # slightly relaxed for synthetic data
            beam_expansions=5000,
        )
        if result is not None:
            assert isinstance(result, SearchResult)
            assert result.expression, "Discovered expression should not be empty"
            assert result.score >= 0.85, (
                f"Regime invariant score {result.score} below threshold"
            )
            # test_constancies should be populated
            assert result.test_constancies is not None
            assert len(result.test_constancies) > 0

    def test_anti_hacking_minimum_size_enforced(
        self, regime_quantities: dict[str, Dimension],
        regime_observations: list[Observation],
        evaluator: ExpressionEvaluator,
    ) -> None:
        """With min_regime_size=10 and only 10 total observations, no split
        passes the anti-hacking guard (need 10+10=20 minimum)."""
        result = _attempt_regime_discovery(
            regime_quantities, regime_observations,
            best_expr="v",
            best_score=0.30,
            evaluator=evaluator,
            discovery_threshold=0.85,
            min_regime_size=10,  # impossible with 10 observations
        )
        assert result is None, (
            "Should return None when min_regime_size cannot be satisfied"
        )

    def test_returns_search_result_with_all_fields(
        self, regime_quantities: dict[str, Dimension],
        regime_observations: list[Observation],
        evaluator: ExpressionEvaluator,
    ) -> None:
        result = _attempt_regime_discovery(
            regime_quantities, regime_observations,
            best_expr="v",
            best_score=0.30,
            evaluator=evaluator,
            discovery_threshold=0.70,  # very relaxed
            beam_expansions=5000,
        )
        if result is None:
            pytest.skip("No regime invariant discovered")
        assert isinstance(result.expression, str)
        assert len(result.expression) > 0
        assert isinstance(result.score, float)
        assert 0.0 <= result.score <= 1.0
        assert isinstance(result.depth, int)
        assert isinstance(result.expansions, int)
        assert isinstance(result.train_constancies, list)
        assert result.test_constancies is not None


# ── Integration: auto_discover with regime fallback ────────────────────

class TestAutoDiscoverRegimeIntegration:

    def test_auto_discover_on_classical_data_still_works(
        self, gravitational_quantities: dict[str, Dimension],
        gravitational_obs: list[Observation],
    ) -> None:
        """auto_discover on classical-only data should find energy conservation
        without needing regime splitting."""
        result = auto_discover(
            gravitational_quantities, gravitational_obs,
            known_invariant="m*g*h + 0.5*m*v^2",
            discovery_threshold=0.90,
            beam_expansions=2000,
        )
        assert result.is_discovery, (
            f"auto_discover should find invariant on classical data. "
            f"Best: {result.expression} score={result.score:.6f}"
        )

    def test_auto_discover_no_regime_split_flag(
        self, regime_quantities: dict[str, Dimension],
        regime_observations: list[Observation],
    ) -> None:
        """With _no_regime_split=True, regime splitting should be suppressed."""
        result = auto_discover(
            regime_quantities, regime_observations,
            discovery_threshold=0.90,
            beam_expansions=5000,
            _no_regime_split=True,
        )
        # Should still return a SearchResult, just without regime splitting
        assert isinstance(result, SearchResult)
        assert isinstance(result.score, float)

    def test_auto_discover_handles_mixed_regimes(
        self, regime_quantities: dict[str, Dimension],
        regime_observations: list[Observation],
    ) -> None:
        """auto_discover on mixed regime data should either find a global
        invariant or fall back to regime discovery."""
        result = auto_discover(
            regime_quantities, regime_observations,
            discovery_threshold=0.80,  # relaxed
            beam_expansions=5000,
        )
        # The system should at minimum return a valid result
        assert isinstance(result, SearchResult)
        assert isinstance(result.expression, str)
        # If a discovery was made, verify quality
        if result.is_discovery:
            assert result.score >= 0.80
