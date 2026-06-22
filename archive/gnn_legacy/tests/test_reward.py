"""Tests for reward computation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.proof_checker.formats import ProofResult
from src.reward.base import compute_reward, compute_rewards_batch, compute_group_advantages
from src.reward.config import RewardConfig


def test_valid_proof_gets_positive_reward():
    """Valid proofs get positive reward."""
    result = ProofResult(success=True, errors=[], num_tokens=50)
    reward = compute_reward(result)
    assert reward > 0.0, f"Expected positive reward, got {reward}"


def test_invalid_proof_gets_zero():
    """Invalid proofs get zero reward."""
    result = ProofResult(success=False, errors=["error"], num_tokens=50)
    reward = compute_reward(result)
    assert reward == 0.0, f"Expected 0 reward, got {reward}"


def test_longer_proof_gets_lower_bonus():
    """With length bonus, longer proofs score lower."""
    config = RewardConfig(length_bonus_enabled=True)

    short = ProofResult(success=True, errors=[], num_tokens=50)
    long = ProofResult(success=True, errors=[], num_tokens=500)

    short_reward = compute_reward(short, config)
    long_reward = compute_reward(long, config)

    assert short_reward >= long_reward, (
        f"Short ({short_reward}) should be >= long ({long_reward})"
    )


def test_trivial_proof_rejected():
    """Very short proofs are rejected (anti-reward-hacking)."""
    result = ProofResult(success=True, errors=[], num_tokens=3)
    reward = compute_reward(result)
    assert reward == 0.0, f"Expected 0 for trivial proof, got {reward}"


def test_group_advantages():
    """Group-relative advantages center each group at zero."""
    rewards = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])
    advantages = compute_group_advantages(rewards, group_size=4)

    # First group: [0, 1, 0, 1], mean=0.5
    # Second group: [1, 0, 1, 1], mean=0.75
    assert advantages.shape == rewards.shape
    # Advantage of above-mean elements should be positive
    assert advantages[1] > 0  # 1.0 > 0.5
    assert advantages[0] < 0  # 0.0 < 0.5


def test_rewards_batch():
    """Batch reward computation returns correct shape."""
    results = [
        ProofResult(success=True, errors=[], num_tokens=50),
        ProofResult(success=False, errors=["x"], num_tokens=50),
        ProofResult(success=True, errors=[], num_tokens=100),
    ]
    rewards = compute_rewards_batch(results)
    assert rewards.shape == (3,)
    assert rewards[0] > 0
    assert rewards[1] == 0
    assert rewards[2] > 0


if __name__ == "__main__":
    test_valid_proof_gets_positive_reward()
    print("PASS: test_valid_proof_gets_positive_reward")

    test_invalid_proof_gets_zero()
    print("PASS: test_invalid_proof_gets_zero")

    test_longer_proof_gets_lower_bonus()
    print("PASS: test_longer_proof_gets_lower_bonus")

    test_trivial_proof_rejected()
    print("PASS: test_trivial_proof_rejected")

    test_group_advantages()
    print("PASS: test_group_advantages")

    test_rewards_batch()
    print("PASS: test_rewards_batch")

    print("\nAll tests passed!")
