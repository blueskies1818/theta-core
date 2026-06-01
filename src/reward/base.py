"""Reward computation for GRPO training.

Phase 1 simplified reward:
- Binary: 1.0 for valid proofs, 0.0 for invalid
- Optional length bonus for shorter valid proofs
"""

import torch

from src.proof_checker.formats import ProofResult
from src.reward.config import RewardConfig


def compute_reward(
    proof_result: ProofResult,
    config: RewardConfig | None = None,
) -> float:
    """Compute reward for a single proof.

    Args:
        proof_result: Output from LeanProofChecker.check().
        config: Reward hyperparameters.

    Returns:
        Scalar reward value.
    """
    if config is None:
        config = RewardConfig()

    # Anti-reward-hacking: reject trivially short proofs
    if proof_result.success and proof_result.num_tokens < config.min_proof_tokens:
        return config.invalid_proof

    if not proof_result.success:
        return config.invalid_proof

    base = config.valid_proof

    # Length bonus: shorter valid proofs score higher
    if config.length_bonus_enabled:
        n_tokens = proof_result.num_tokens
        excess = max(0, n_tokens - config.length_reference_tokens)
        bonus = max(0.0, 1.0 - excess * config.length_decay_rate)
        base += config.length_bonus_weight * bonus

    return base


def compute_rewards_batch(
    results: list[ProofResult],
    config: RewardConfig | None = None,
) -> torch.Tensor:
    """Compute rewards for a batch of proof results.

    Args:
        results: List of ProofResult from batch checking.
        config: Reward hyperparameters.

    Returns:
        Tensor of reward values, shape (len(results),).
    """
    rewards = [compute_reward(r, config) for r in results]
    return torch.tensor(rewards, dtype=torch.float32)


def compute_group_advantages(
    rewards: torch.Tensor,
    group_size: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute group-relative advantages for GRPO.

    For each group of K proofs for one theorem:
      advantage_i = (r_i - mean(r_group)) / (std(r_group) + eps)

    Args:
        rewards: Flat tensor of rewards, shape (num_prompts * group_size,).
        group_size: K, number of proofs per theorem.
        eps: Small constant for numerical stability.

    Returns:
        Advantage tensor of same shape as rewards.
    """
    num_groups = rewards.numel() // group_size
    reshaped = rewards.view(num_groups, group_size)

    group_mean = reshaped.mean(dim=1, keepdim=True)
    group_std = reshaped.std(dim=1, keepdim=True)

    advantages = (reshaped - group_mean) / (group_std + eps)

    return advantages.view(-1)
