"""Reward configuration dataclass.

Phase 1: binary proof-check reward + length bonus.
Phase 3+: add predictive compression, correspondence, curiosity bonuses.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class RewardConfig:
    """Configuration for proof reward computation.

    Phase 1 simplified reward:
    - Binary: 1.0 for valid proofs, 0.0 for invalid
    - Optional length bonus for shorter valid proofs
    - Minimum token threshold to reject trivial proofs

    Phase 3+ will add:
    - Predictive compression score against physical observation data
    - Correspondence requirement score (reduction to GR/QFT at limits)
    - Curiosity/exploration bonus
    - Simplicity penalty (Occam's razor formalized)
    """

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

    # Curiosity/exploration bonus (Phase 1.5)
    # CRITICAL: without this, model mode-collapses onto one proof pattern.
    # Count-based exploration bonus: bonus = weight / sqrt(count + 1)
    curiosity_enabled: bool = True
    curiosity_weight: float = 0.05
    curiosity_signature_length: int = 200
    curiosity_max_tracked: int = 100_000

    # ── H3 Traversal bonus ─────────────────────────────────────────────
    # Rewards proofs that use lemmas far from the training distribution in
    # the dependency graph. "Far" = 3+ hops from any training lemma.
    # This encourages the GNN to explore deeper parts of the graph during
    # training, which should improve lemma-novelty generalization.
    traversal_bonus_enabled: bool = False
    traversal_bonus_weight: float = 0.5
    traversal_hop_threshold: int = 3


def load_reward_config(path: Path | None = None) -> RewardConfig:
    """Load reward configuration from YAML file.

    Args:
        path: Path to reward_config.yaml. Defaults to configs/reward_config.yaml.

    Returns:
        RewardConfig populated from YAML.
    """
    if path is None:
        path = Path(__file__).parent.parent.parent / "configs" / "reward_config.yaml"

    with open(path) as f:
        data = yaml.safe_load(f) or {}

    return RewardConfig(**data)
