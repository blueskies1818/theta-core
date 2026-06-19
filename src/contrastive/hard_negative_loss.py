"""Hard-negative contrastive loss functions.

Provides loss functions that combine:
  1. InfoNCE (in-batch soft negatives) — pulls positive pairs together
  2. Triplet margin loss (confirmed hard negatives) — pushes verified
     wrong lemmas away from goals they don't solve

The combined loss uses proof-checker pass/fail as ground truth for
hard negatives, zero era labels.

Loss = InfoNCE_loss + λ * HardNegative_loss

where HardNegative_loss uses confirmed negatives from the proof checker
to push wrong lemmas away from goals using a margin-based triplet loss.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# InfoNCE (in-batch contrastive)
# ---------------------------------------------------------------------------


def compute_infonce_loss(
    goal_emb: torch.Tensor,
    lemma_emb: torch.Tensor,
    temperature_inv: float,
) -> torch.Tensor:
    """Symmetric InfoNCE loss on in-batch pairs.

    Positive pairs are on the diagonal. All other pairs in the batch
    serve as soft negatives.

    Args:
        goal_emb: [B, D] L2-normalized goal embeddings.
        lemma_emb: [B, D] L2-normalized lemma embeddings.
        temperature_inv: 1 / temperature.

    Returns:
        Scalar loss (mean of goal→lemma and lemma→goal directions).
    """
    batch_size = goal_emb.size(0)
    logits = goal_emb @ lemma_emb.T * temperature_inv
    labels = torch.arange(batch_size, device=goal_emb.device)

    loss_g2l = F.cross_entropy(logits, labels)
    loss_l2g = F.cross_entropy(logits.T, labels)

    return (loss_g2l + loss_l2g) / 2.0


# ---------------------------------------------------------------------------
# Triplet margin loss (hard negatives)
# ---------------------------------------------------------------------------


def compute_triplet_margin_loss(
    goal_emb: torch.Tensor,
    positive_lemma_emb: torch.Tensor,
    hard_negative_emb: torch.Tensor,
    margin: float = 0.3,
    reduction: str = "mean",
) -> torch.Tensor:
    """Triplet margin loss with confirmed hard negatives.

    For each (goal, positive_lemma, hard_negative) triple:
        loss = max(0, margin - sim(goal, positive) + sim(goal, negative))

    This pushes confirmed-wrong lemmas away from the goal, while pulling
    the correct lemma closer.

    Args:
        goal_emb: [B, D] L2-normalized goal embeddings.
        positive_lemma_emb: [B, D] L2-normalized positive lemma embeddings.
        hard_negative_emb: [B, K, D] hard negative lemma embeddings.
                           K = number of hard negatives per goal.
                           (K can vary per batch item; use padding/looping.)
        margin: Triplet margin (default 0.3 for normalized embeddings).
        reduction: 'mean' or 'sum'.

    Returns:
        Scalar loss.
    """
    batch_size, hidden_dim = goal_emb.shape
    K = hard_negative_emb.size(1)

    if K == 0:
        return torch.tensor(0.0, device=goal_emb.device)

    # Sim(goal, positive): [B]
    pos_sim = (goal_emb * positive_lemma_emb).sum(dim=-1)  # [B]

    # Sim(goal, each hard negative): [B, K]
    # goal_emb: [B, D], hard_negative_emb: [B, K, D]
    neg_sim = torch.einsum("bd,bkd->bk", goal_emb, hard_negative_emb)  # [B, K]

    # Triplet loss per negative: max(0, margin - pos_sim + neg_sim)
    # pos_sim: [B] → [B, 1] → broadcast to [B, K]
    losses = F.relu(margin - pos_sim.unsqueeze(1) + neg_sim)  # [B, K]

    if reduction == "mean":
        return losses.mean()
    elif reduction == "sum":
        return losses.sum()
    else:
        return losses.mean()


# ---------------------------------------------------------------------------
# Combined loss (InfoNCE + Hard Negative Triplet)
# ---------------------------------------------------------------------------


def compute_combined_loss(
    goal_emb: torch.Tensor,
    positive_lemma_emb: torch.Tensor,
    hard_negative_emb: torch.Tensor | None,
    temperature_inv: float,
    hard_neg_weight: float = 0.5,
    margin: float = 0.3,
) -> dict[str, torch.Tensor]:
    """Combined InfoNCE + hard-negative triplet loss.

    Args:
        goal_emb: [B, D] L2-normalized goal embeddings.
        positive_lemma_emb: [B, D] L2-normalized positive lemma embeddings.
        hard_negative_emb: [B, K, D] hard negative embeddings, or None.
        temperature_inv: 1 / temperature for InfoNCE.
        hard_neg_weight: Weight for the hard negative triplet loss.
        margin: Triplet margin.

    Returns:
        Dict with 'total_loss', 'infonce_loss', 'hard_neg_loss'.
    """
    infonce_loss = compute_infonce_loss(goal_emb, positive_lemma_emb, temperature_inv)

    if hard_negative_emb is not None and hard_negative_emb.size(1) > 0:
        hard_neg_loss = compute_triplet_margin_loss(
            goal_emb, positive_lemma_emb, hard_negative_emb, margin=margin,
        )
    else:
        hard_neg_loss = torch.tensor(0.0, device=goal_emb.device)

    total_loss = infonce_loss + hard_neg_weight * hard_neg_loss

    return {
        "total_loss": total_loss,
        "infonce_loss": infonce_loss,
        "hard_neg_loss": hard_neg_loss,
    }


# ---------------------------------------------------------------------------
# Accuracy (top-1 retrieval)
# ---------------------------------------------------------------------------


def compute_retrieval_accuracy(
    goal_emb: torch.Tensor,
    lemma_emb: torch.Tensor,
) -> torch.Tensor:
    """Compute top-1 retrieval accuracy on in-batch pairs.

    The correct lemma for goal i is lemma i (diagonal).
    Returns fraction of goals where the correct lemma has the highest score.

    Args:
        goal_emb: [B, D] L2-normalized goal embeddings.
        lemma_emb: [B, D] L2-normalized lemma embeddings.

    Returns:
        Scalar accuracy (0-1).
    """
    batch_size = goal_emb.size(0)
    scores = goal_emb @ lemma_emb.T  # [B, B]
    labels = torch.arange(batch_size, device=goal_emb.device)

    _, pred_g2l = scores.max(dim=1)
    _, pred_l2g = scores.T.max(dim=1)

    acc_g2l = (pred_g2l == labels).float().mean()
    acc_l2g = (pred_l2g == labels).float().mean()

    return (acc_g2l + acc_l2g) / 2.0
