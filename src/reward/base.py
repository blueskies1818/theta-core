"""Reward computation for GRPO training.

Phase 1 reward components:
- Binary: 1.0 for valid proofs, 0.0 for invalid
- Optional length bonus for shorter valid proofs
- Curiosity/exploration bonus (Phase 1.5): count-based bonus for novel proofs
  CRITICAL for preventing mode collapse during self-play.
"""

import hashlib
import re
import torch
from collections import Counter

from src.proof_checker.formats import ProofResult
from src.reward.config import RewardConfig

# Module-level proof signature counter.
# Tracks how many times each proof pattern has been generated.
# Persists across compute_reward calls within a training run.
_proof_signature_counter: Counter = Counter()


def _normalize_proof_for_signature(proof_text: str, max_chars: int = 200) -> str:
    """Normalize a proof for signature computation.

    Strips whitespace variations, comments, and trivial formatting
    differences so that semantically identical proofs share a signature.
    """
    # Collapse whitespace
    normalized = re.sub(r'\s+', ' ', proof_text.strip())
    # Remove Lean comments
    normalized = re.sub(r'--[^\n]*', '', normalized)
    normalized = re.sub(r'/-.*?-/', '', normalized, flags=re.DOTALL)
    # Truncate to signature length (catch the proof "shape")
    return normalized[:max_chars]


def _compute_proof_signature(proof_text: str, max_chars: int = 200) -> str:
    """Compute a hash-based signature for a proof.

    Uses SHA-256 truncated to 16 hex chars — fast, collision-resistant
    enough for proof deduplication.
    """
    normalized = _normalize_proof_for_signature(proof_text, max_chars)
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def compute_curiosity_bonus(
    proof_text: str,
    config: RewardConfig | None = None,
) -> float:
    """Compute count-based exploration bonus for a proof.

    bonus = curiosity_weight / sqrt(count(signature) + 1)

    Novel proofs get the full bonus. Frequently generated proofs
    get diminishing returns, discouraging mode collapse.
    """
    if config is None:
        config = RewardConfig()

    if not config.curiosity_enabled:
        return 0.0

    sig = _compute_proof_signature(
        proof_text, config.curiosity_signature_length
    )
    count = _proof_signature_counter[sig]

    bonus = config.curiosity_weight / (count + 1) ** 0.5
    return bonus


def record_proof_signature(
    proof_text: str,
    config: RewardConfig | None = None,
) -> None:
    """Record a generated proof's signature in the global counter.

    Call this AFTER computing the reward so the curiosity bonus
    is based on the count BEFORE this proof was added.
    """
    if config is None:
        config = RewardConfig()

    if not config.curiosity_enabled:
        return

    sig = _compute_proof_signature(
        proof_text, config.curiosity_signature_length
    )
    _proof_signature_counter[sig] += 1

    # Prune old signatures if over the max
    if len(_proof_signature_counter) > config.curiosity_max_tracked:
        # Keep the most common half
        most_common = _proof_signature_counter.most_common(
            config.curiosity_max_tracked // 2
        )
        _proof_signature_counter.clear()
        _proof_signature_counter.update(dict(most_common))


def get_curiosity_stats() -> dict:
    """Return statistics about the curiosity tracker for logging."""
    if not _proof_signature_counter:
        return {"unique_signatures": 0, "total_counts": 0, "max_count": 0}

    return {
        "unique_signatures": len(_proof_signature_counter),
        "total_counts": sum(_proof_signature_counter.values()),
        "max_count": max(_proof_signature_counter.values()),
    }


def compute_reward(
    proof_result: ProofResult,
    config: RewardConfig | None = None,
    proof_text: str | None = None,
) -> float:
    """Compute reward for a single proof.

    Args:
        proof_result: Output from LeanProofChecker.check().
        config: Reward hyperparameters.
        proof_text: The generated proof text (required for curiosity bonus).

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

    # Curiosity bonus: novel proofs get higher reward (Phase 1.5)
    if config.curiosity_enabled and proof_text:
        base += compute_curiosity_bonus(proof_text, config)

    return base


def compute_rewards_batch(
    results: list[ProofResult],
    config: RewardConfig | None = None,
    proof_texts: list[str] | None = None,
) -> torch.Tensor:
    """Compute rewards for a batch of proof results.

    Args:
        results: List of ProofResult from batch checking.
        config: Reward hyperparameters.
        proof_texts: Optional list of generated proof strings for curiosity bonus.

    Returns:
        Tensor of reward values, shape (len(results),).
    """
    texts = proof_texts or [None] * len(results)
    rewards = [
        compute_reward(r, config, proof_text=t)
        for r, t in zip(results, texts)
    ]
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
