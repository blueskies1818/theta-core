"""Tests for step-level dense reward system (Path A).

Tests DenseRewardTracker, DenseRewardConfig, step validity,
goal proximity, completion bonus, and trajectory summarization.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.reward.dense_rewards import (
    DenseRewardConfig,
    StepReward,
    DenseTrajectory,
    DenseRewardTracker,
    summarize_trajectories,
)


# ---------------------------------------------------------------------------
# DenseRewardConfig
# ---------------------------------------------------------------------------

def test_default_config():
    """Default config has expected values."""
    c = DenseRewardConfig()
    assert c.step_validity == 0.1
    assert c.goal_proximity_weight == 0.2
    assert c.completion_bonus == 1.0
    assert c.goal_proximity_enabled is True
    assert c.max_steps == 20


def test_custom_config():
    """Custom config overrides defaults."""
    c = DenseRewardConfig(
        step_validity=0.5,
        goal_proximity_weight=0.7,
        completion_bonus=5.0,
    )
    assert c.step_validity == 0.5
    assert c.goal_proximity_weight == 0.7
    assert c.completion_bonus == 5.0


# ---------------------------------------------------------------------------
# StepReward dataclass
# ---------------------------------------------------------------------------

def test_step_reward_creation():
    """StepReward fields are correctly set."""
    sr = StepReward(
        step_index=0,
        tactic="apply mul_comm",
        is_valid=True,
        step_validity=0.1,
        goal_proximity=0.05,
        cumulative=0.15,
    )
    assert sr.step_index == 0
    assert sr.tactic == "apply mul_comm"
    assert sr.is_valid is True
    assert sr.step_validity == 0.1
    assert sr.goal_proximity == 0.05
    assert sr.cumulative == 0.15


# ---------------------------------------------------------------------------
# DenseRewardTracker — step validity (a)
# ---------------------------------------------------------------------------

def test_step_validity_reward():
    """Valid steps get step_validity reward; invalid steps get 0."""
    config = DenseRewardConfig(
        step_validity=0.1,
        goal_proximity_enabled=False,  # disable proximity for clean test
    )
    tracker = DenseRewardTracker(config, theorem_statement="example : 1 = 1")

    r1 = tracker.record_step("apply mul_comm", is_valid=True)
    r2 = tracker.record_step("apply bad_lemma", is_valid=False)

    assert r1 == 0.1  # valid → +0.1
    assert r2 == 0.0  # invalid → 0.0
    assert tracker.cumulative_reward == 0.1
    assert tracker.num_steps == 2


def test_custom_step_validity():
    """Custom step_validity weight is respected."""
    config = DenseRewardConfig(
        step_validity=0.25,
        goal_proximity_enabled=False,
    )
    tracker = DenseRewardTracker(config, theorem_statement="example : 1 = 1")
    r = tracker.record_step("apply lemma_x", is_valid=True)
    assert r == 0.25


# ---------------------------------------------------------------------------
# DenseRewardTracker — goal proximity (b)
# ---------------------------------------------------------------------------

def test_goal_proximity_disabled():
    """When goal_proximity_enabled=False, proximity is always 0."""
    config = DenseRewardConfig(
        step_validity=0.1,
        goal_proximity_enabled=False,
    )
    tracker = DenseRewardTracker(config, theorem_statement="example : 1 = 1")
    r = tracker.record_step(
        "apply lemma_x", is_valid=True, current_goal_text="some goal"
    )
    assert r == 0.1  # no proximity bonus


def test_goal_proximity_without_encoder():
    """Without encoder/tokenizer, proximity is 0 (graceful degradation)."""
    config = DenseRewardConfig(
        step_validity=0.1,
        goal_proximity_enabled=True,
        goal_proximity_weight=0.2,
    )
    tracker = DenseRewardTracker(
        config, encoder=None, tokenizer=None, theorem_statement="example : 1 = 1"
    )
    r = tracker.record_step(
        "apply lemma_x", is_valid=True, current_goal_text="new goal"
    )
    assert r == 0.1  # proximity stays 0 without encoder


def test_goal_proximity_no_goal_text():
    """Empty goal text → proximity 0."""
    config = DenseRewardConfig(
        step_validity=0.1,
        goal_proximity_enabled=True,
        goal_proximity_weight=0.5,
    )
    tracker = DenseRewardTracker(config, theorem_statement="example : 1 = 1")
    r = tracker.record_step("apply lemma_x", is_valid=True, current_goal_text="")
    assert r == 0.1  # no goal text → no proximity


# ---------------------------------------------------------------------------
# DenseRewardTracker — completion bonus (c)
# ---------------------------------------------------------------------------

def test_completion_bonus_success():
    """Successful completion adds completion_bonus."""
    config = DenseRewardConfig(
        step_validity=0.1,
        goal_proximity_enabled=False,
        completion_bonus=1.0,
    )
    tracker = DenseRewardTracker(config, theorem_statement="example : 1 = 1")
    tracker.record_step("apply mul_comm", is_valid=True)
    tracker.record_step("rw [add_comm]", is_valid=True)

    bonus = tracker.record_completion(success=True)
    assert bonus == 1.0
    assert tracker._cumulative == 1.2  # 0.1+0.1+1.0


def test_completion_bonus_failure():
    """Failed completion adds no bonus."""
    config = DenseRewardConfig(
        step_validity=0.1,
        goal_proximity_enabled=False,
        completion_bonus=1.0,
    )
    tracker = DenseRewardTracker(config, theorem_statement="example : 1 = 1")
    tracker.record_step("apply mul_comm", is_valid=True)

    bonus = tracker.record_completion(success=False)
    assert bonus == 0.0
    assert tracker._cumulative == 0.1  # only step reward


# ---------------------------------------------------------------------------
# DenseRewardTracker — trajectory export
# ---------------------------------------------------------------------------

def test_to_trajectory():
    """to_trajectory() exports correct aggregate stats."""
    config = DenseRewardConfig(
        step_validity=0.1,
        goal_proximity_enabled=False,
        completion_bonus=1.0,
    )
    tracker = DenseRewardTracker(
        config,
        theorem_name="test_theorem",
        theorem_statement="example : 1 = 1",
    )
    tracker.record_step("intro h", is_valid=True)
    tracker.record_step("exact h", is_valid=True)
    tracker.record_completion(success=True)

    traj = tracker.to_trajectory()
    assert traj.theorem_name == "test_theorem"
    assert traj.proof_success is True
    assert traj.num_valid_steps == 2
    assert traj.num_invalid_steps == 0
    assert len(traj.steps) == 2
    assert traj.total_reward == pytest.approx(1.2)  # 0.1+0.1+1.0


def test_to_trajectory_with_failure():
    """Failed trajectory has correct stats."""
    config = DenseRewardConfig(
        step_validity=0.1,
        goal_proximity_enabled=False,
    )
    tracker = DenseRewardTracker(
        config,
        theorem_name="bad_theorem",
        theorem_statement="example : 1 = 0",
    )
    tracker.record_step("apply bad", is_valid=False)
    tracker.record_step("ring", is_valid=True)
    tracker.record_completion(success=False)

    traj = tracker.to_trajectory()
    assert traj.proof_success is False
    assert traj.num_valid_steps == 1
    assert traj.num_invalid_steps == 1
    assert traj.completion_bonus == 0.0
    assert traj.total_reward == pytest.approx(0.1)  # only one valid step


def test_record_step_after_completion_raises():
    """Recording a step after completion raises RuntimeError."""
    config = DenseRewardConfig()
    tracker = DenseRewardTracker(config, theorem_statement="example : 1 = 1")
    tracker.record_step("intro h", is_valid=True)
    tracker.record_completion(success=True)

    with pytest.raises(RuntimeError):
        tracker.record_step("exact h", is_valid=True)


# ---------------------------------------------------------------------------
# DenseRewardTracker — auto-completion in to_trajectory
# ---------------------------------------------------------------------------

def test_to_trajectory_auto_completes():
    """Calling to_trajectory() without explicit completion auto-completes as failure."""
    config = DenseRewardConfig(
        step_validity=0.1,
        goal_proximity_enabled=False,
    )
    tracker = DenseRewardTracker(config, theorem_statement="example : 1 = 1")
    tracker.record_step("rw [h]", is_valid=True)

    traj = tracker.to_trajectory()
    assert traj.proof_success is False
    assert traj.completion_bonus == 0.0
    assert traj.total_reward == 0.1


# ---------------------------------------------------------------------------
# summarize_trajectories
# ---------------------------------------------------------------------------

def test_summarize_empty():
    """Empty list returns zero values."""
    summary = summarize_trajectories([])
    assert summary["num_theorems"] == 0


def test_summarize_mixed():
    """Summary aggregates correctly across success and failure."""
    t1 = DenseTrajectory(
        theorem_name="thm1",
        proof_success=True,
        total_reward=1.5,
        completion_bonus=1.0,
        num_valid_steps=3,
        num_invalid_steps=0,
        steps=[
            StepReward(0, "intro h", True, 0.1, 0.0, 0.1),
            StepReward(1, "rw [h]", True, 0.1, 0.0, 0.2),
            StepReward(2, "exact h", True, 0.1, 0.0, 0.3),
        ],
    )
    t2 = DenseTrajectory(
        theorem_name="thm2",
        proof_success=False,
        total_reward=0.1,
        completion_bonus=0.0,
        num_valid_steps=1,
        num_invalid_steps=2,
        steps=[
            StepReward(0, "apply bad", False, 0.0, 0.0, 0.0),
            StepReward(1, "simp", True, 0.1, 0.0, 0.1),
            StepReward(2, "apply bad2", False, 0.0, 0.0, 0.1),
        ],
    )

    summary = summarize_trajectories([t1, t2])
    assert summary["num_theorems"] == 2
    assert summary["num_proved"] == 1
    assert summary["proof_rate"] == 0.5
    assert summary["multi_step_proofs"] == 1  # thm1 has 3 steps >= 2
    assert summary["mean_total_reward"] == pytest.approx(0.8)  # (1.5+0.1)/2
    assert summary["mean_completion_bonus"] == pytest.approx(1.0)
    assert summary["total_valid_steps"] == 4
    assert summary["total_invalid_steps"] == 2
    assert summary["mean_steps"] == pytest.approx(3.0)  # (3+3)/2


def test_summarize_all_success():
    """All-success batch has proof_rate=1.0."""
    t = DenseTrajectory(
        theorem_name="thm",
        proof_success=True,
        total_reward=2.2,
        completion_bonus=1.0,
        num_valid_steps=2,
        num_invalid_steps=0,
        steps=[
            StepReward(0, "intro h", True, 0.1, 0.0, 0.1),
            StepReward(1, "exact h", True, 0.1, 0.0, 0.2),
        ],
    )
    summary = summarize_trajectories([t, t, t])
    assert summary["proof_rate"] == 1.0
    assert summary["num_proved"] == 3


# ---------------------------------------------------------------------------
# Run main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
