"""Reward configuration dataclass."""

from dataclasses import dataclass, field


@dataclass
class RewardConfig:
    """Configuration for proof reward computation."""

    # Base reward for valid/invalid proofs
    valid_proof: float = 1.0
    invalid_proof: float = 0.0

    # Length bonus: shorter valid proofs get higher reward
    length_bonus_enabled: bool = True
    length_bonus_weight: float = 0.1
    length_reference_tokens: int = 100
    length_decay_rate: float = 0.002

    # Minimum complexity: reject trivially short proofs
    min_proof_tokens: int = 10
