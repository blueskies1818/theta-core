"""Contrastive lemma embedding (Path C) — dual-encoder for proof-step retrieval.

Extended with hard-negative mining via proof-checker pass/fail."""
from src.contrastive.encoder import ContrastiveDualEncoder, ContrastiveConfig, CharTokenizer, CharCNNEncoder
from src.contrastive.hard_negative_loss import (
    compute_infonce_loss,
    compute_triplet_margin_loss,
    compute_combined_loss,
    compute_retrieval_accuracy,
)
from src.contrastive.hard_negative_miner import (
    HardNegativeMiner,
    HardNegativeCache,
    build_lemma_goal_proof_script,
    load_hard_negative_data,
    save_hard_negative_data,
)

__all__ = [
    "ContrastiveDualEncoder",
    "ContrastiveConfig",
    "CharTokenizer",
    "CharCNNEncoder",
    "compute_infonce_loss",
    "compute_triplet_margin_loss",
    "compute_combined_loss",
    "compute_retrieval_accuracy",
    "HardNegativeMiner",
    "HardNegativeCache",
    "build_lemma_goal_proof_script",
    "load_hard_negative_data",
    "save_hard_negative_data",
]
