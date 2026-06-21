"""Goal-only encoder for self-supervised proof-step retrieval.

NO lemma encoder. NO GNN graph. NO import edges. NO contrastive.

Architecture:
  Goal text → tokenize → embed tokens → avg pool → 2-layer MLP → L2-norm
  ~484K params, 256-dim output

Training (InfoNCE):
  Goals that share ANY lemma form positive pairs.
  InfoNCE loss pushes same-lemma goals together, random goals apart.

Inference:
  New goal → encode → find 50 nearest training goals (cosine sim)
  → collect their correct lemmas → rank by frequency × similarity → top-30

Usage:
    from src.retrieval.goal_only_encoder import GoalOnlyEncoder, train_goal_only_encoder
"""

from __future__ import annotations

import math
import re
import json
import time
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# Tokenization (same as direct_lookup.py)
# ---------------------------------------------------------------------------

_MATH_SPLIT_RE = re.compile(
    r'[\s+\-*/^=()\[\]{}:.,;→←↔⇒⇔∀∃λ≤≥<>&|!~@#$%\\\\]+'
)


def tokenize_goal(goal_text: str) -> list[str]:
    """Tokenize a Lean goal expression into meaningful tokens."""
    parts = _MATH_SPLIT_RE.split(goal_text)
    tokens = []
    for part in parts:
        part = part.strip().lower()
        if len(part) >= 1:
            tokens.append(part)
    return tokens


# ---------------------------------------------------------------------------
# Goal-only Encoder
# ---------------------------------------------------------------------------


class GoalOnlyEncoder(nn.Module):
    """Self-supervised goal encoder — lemma information stays in training pairs.

    Architecture:
      - Token embedding: vocab_size × 128-dim
      - Average pooling over token embeddings
      - 2-layer MLP: 128 → 256 → 256 (with residual + LayerNorm)
      - L2 normalization

    ~484K params (vocab=3000): 3000×128 + 128×256 + 256 + 256×256 + 256
    """

    def __init__(
        self,
        vocab_size: int = 3000,
        token_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.token_dim = token_dim
        self.hidden_dim = hidden_dim

        # Token embedding + PAD token (index 0)
        self.token_embed = nn.Embedding(vocab_size + 1, token_dim, padding_idx=0)

        # MLP: 128 → 256 → 256
        self.proj1 = nn.Linear(token_dim, hidden_dim)
        self.proj2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.proj1.weight)
        nn.init.zeros_(self.proj1.bias)
        nn.init.xavier_uniform_(self.proj2.weight)
        nn.init.zeros_(self.proj2.bias)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Encode goal from token IDs.

        Args:
            token_ids: [B, max_len] padded token indices (0 = PAD)

        Returns:
            [B, hidden_dim] L2-normalized embeddings
        """
        # Token embeddings: [B, max_len, token_dim]
        emb = self.token_embed(token_ids)

        # Average pooling over non-padded tokens
        mask = (token_ids != 0).float().unsqueeze(-1)  # [B, max_len, 1]
        pooled = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        # MLP with residual
        h = self.proj1(pooled)
        h = self.norm1(F.gelu(h))
        h = self.dropout(h)
        h = self.proj2(h)
        h = self.norm2(h + self.proj2(pooled) if False else h)  # no residual for now
        h = F.gelu(h)

        return F.normalize(h, dim=-1)

    def encode_text(
        self, goal_texts: list[str], vocab: dict[str, int], max_len: int = 128
    ) -> torch.Tensor:
        """Convenience: tokenize and encode goal texts.

        Returns [B, hidden_dim] normalized embeddings.
        """
        device = next(self.parameters()).device
        batch_ids = _tokenize_batch(goal_texts, vocab, max_len)
        batch_ids = batch_ids.to(device)
        return self.forward(batch_ids)

    def save(self, path: Path | str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_dict = {
            "model_state_dict": self.state_dict(),
            "config": {
                "vocab_size": self.vocab_size,
                "token_dim": self.token_dim,
                "hidden_dim": self.hidden_dim,
            },
        }
        torch.save(save_dict, path)

    @classmethod
    def load(cls, path: Path | str) -> "GoalOnlyEncoder":
        state = torch.load(str(path), map_location="cpu", weights_only=False)
        cfg = state["config"]
        model = cls(
            vocab_size=cfg["vocab_size"],
            token_dim=cfg["token_dim"],
            hidden_dim=cfg["hidden_dim"],
        )
        model.load_state_dict(state["model_state_dict"])
        return model

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Vocabulary building
# ---------------------------------------------------------------------------


def build_vocabulary(
    goals: list[str], max_vocab: int = 3000
) -> dict[str, int]:
    """Build token→id vocabulary from goal texts.

    Tokens are ranked by document frequency. PAD is index 0 (implicit).
    Token indices start at 1.
    """
    df: dict[str, int] = {}
    for goal in goals:
        tokens = set(tokenize_goal(goal))
        for t in tokens:
            df[t] = df.get(t, 0) + 1

    sorted_tokens = sorted(df.items(), key=lambda x: -x[1])[:max_vocab]
    vocab: dict[str, int] = {}
    for i, (token, _) in enumerate(sorted_tokens):
        vocab[token] = i + 1  # 0 = PAD

    return vocab


def _tokenize_batch(
    goal_texts: list[str], vocab: dict[str, int], max_len: int = 128
) -> torch.Tensor:
    """Tokenize a batch of goal texts into padded tensor.

    Returns [B, max_len] LongTensor.
    """
    batch_ids = torch.zeros(len(goal_texts), max_len, dtype=torch.long)
    for i, text in enumerate(goal_texts):
        tokens = tokenize_goal(text)
        ids = [vocab[t] for t in tokens if t in vocab]
        ids = ids[:max_len]
        batch_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    return batch_ids


# ---------------------------------------------------------------------------
# Lemma-based data preparation
# ---------------------------------------------------------------------------


def prepare_lemma_groups(
    pairs_path: Path, max_pairs: int | None = None
) -> tuple[list[str], list[str], dict[str, list[int]]]:
    """Load proof-step pairs and build lemma→goal_index groups.

    Returns:
        goals: list of goal texts
        lemmas: list of lemma names (parallel to goals)
        lemma_to_indices: lemma_name → list of goal indices
    """
    goals: list[str] = []
    lemmas: list[str] = []
    lemma_to_indices: dict[str, list[int]] = defaultdict(list)

    with open(pairs_path) as f:
        for i, line in enumerate(f):
            if max_pairs and i >= max_pairs:
                break
            pair = json.loads(line)
            goal_text = pair["goal"]
            lemma_name = pair["lemma"]
            goals.append(goal_text)
            lemmas.append(lemma_name)
            lemma_to_indices[lemma_name].append(i)

    return goals, lemmas, dict(lemma_to_indices)


# ---------------------------------------------------------------------------
# InfoNCE loss with lemma-based positive pairing
# ---------------------------------------------------------------------------


def compute_info_nce_loss(
    embeddings: torch.Tensor,
    lemma_ids: torch.Tensor,
    temperature: float = 0.07,
) -> tuple[torch.Tensor, float, float]:
    """InfoNCE loss: pull same-lemma goals together, push apart others.

    For each goal i, positive = any goal j where lemma_ids[i] == lemma_ids[j].
    Negatives = all other goals in the batch.

    Args:
        embeddings: [B, D] L2-normalized goal embeddings.
        lemma_ids: [B] integer lemma group IDs (same lemma → same id).
        temperature: softmax temperature.

    Returns:
        (loss, mean_pos_sim, mean_neg_sim)
    """
    B = embeddings.size(0)
    if B < 2:
        return torch.tensor(0.0, device=embeddings.device), 0.0, 0.0

    # Cosine similarity matrix: [B, B]
    sim = embeddings @ embeddings.T  # already normalized

    # Positive mask: i and j share a lemma (and i ≠ j)
    pos_mask = (lemma_ids.unsqueeze(0) == lemma_ids.unsqueeze(1)) & (
        ~torch.eye(B, dtype=torch.bool, device=embeddings.device)
    )

    # Temperature scaling
    sim = sim / temperature

    # For numerical stability: subtract max per row
    sim_max = sim.detach().max(dim=1, keepdim=True).values
    sim = sim - sim_max

    # exp of similarities
    exp_sim = torch.exp(sim)

    # Denominator: sum over all except self
    denom = exp_sim.sum(dim=1) - torch.diag(exp_sim)

    # Numerator: sum over positive pairs
    num = (exp_sim * pos_mask.float()).sum(dim=1)

    # Only compute loss for rows with at least one positive
    has_positive = pos_mask.any(dim=1)
    if not has_positive.any():
        return torch.tensor(0.0, device=embeddings.device), 0.0, 0.0

    # InfoNCE: -log(num / denom)
    loss = -torch.log((num[has_positive] / denom[has_positive]).clamp(min=1e-8))
    loss = loss.mean()

    # Stats
    with torch.no_grad():
        pos_sim_vals = (embeddings @ embeddings.T)[pos_mask]
        pos_mean = pos_sim_vals.mean().item() if pos_sim_vals.numel() > 0 else 0.0

        neg_mask = ~pos_mask & ~torch.eye(B, dtype=torch.bool, device=embeddings.device)
        neg_sim_vals = (embeddings @ embeddings.T)[neg_mask]
        neg_mean = neg_sim_vals.mean().item() if neg_sim_vals.numel() > 0 else 0.0

    return loss, pos_mean, neg_mean


# ---------------------------------------------------------------------------
# Batch sampling with lemma-aware grouping
# ---------------------------------------------------------------------------


def sample_info_nce_batch(
    goals: list[str],
    lemmas: list[str],
    lemma_to_indices: dict[str, list[int]],
    batch_size: int,
    vocab: dict[str, int],
    max_len: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a batch where every goal has at least one positive.

    Strategy: sample lemmas that have ≥2 goals, then pick 2 goals per lemma.

    Returns:
        token_ids: [B, max_len] padded token indices
        lemma_ids: [B] integer group IDs (same index = same lemma)
    """
    # Get lemmas with ≥2 goals
    multi_goal_lemmas = [
        (lemma, indices)
        for lemma, indices in lemma_to_indices.items()
        if len(indices) >= 2
    ]

    if not multi_goal_lemmas:
        # Fallback: just random sample
        indices = random.sample(range(len(goals)), min(batch_size, len(goals)))
        token_ids = _tokenize_batch(
            [goals[i] for i in indices], vocab, max_len
        )
        return token_ids, torch.arange(len(indices))

    # Sample lemmas
    n_lemmas = min(batch_size // 2, len(multi_goal_lemmas))
    sampled_lemmas = random.sample(multi_goal_lemmas, n_lemmas)

    batch_goals: list[str] = []
    lemma_ids: list[int] = []

    for group_id, (lemma, indices) in enumerate(sampled_lemmas):
        # Pick 2 goals from this lemma
        picks = random.sample(indices, min(2, len(indices)))
        for idx in picks:
            batch_goals.append(goals[idx])
            lemma_ids.append(group_id)

    # Trim to batch_size
    if len(batch_goals) > batch_size:
        batch_goals = batch_goals[:batch_size]
        lemma_ids = lemma_ids[:batch_size]

    # Pad with random if too small
    while len(batch_goals) < batch_size:
        idx = random.randint(0, len(goals) - 1)
        batch_goals.append(goals[idx])
        lemma_ids.append(len(lemma_ids))  # unique group

    token_ids = _tokenize_batch(batch_goals, vocab, max_len)
    return token_ids, torch.tensor(lemma_ids, dtype=torch.long)


# ---------------------------------------------------------------------------
# MRR evaluation
# ---------------------------------------------------------------------------


def compute_mrr(
    encoder: GoalOnlyEncoder,
    vocab: dict[str, int],
    queries: list[str],
    query_lemmas: list[str],
    index_goals: list[str],
    index_lemmas: list[str],
    index_embeddings: torch.Tensor,
    k: int = 50,
    sample_size: int = 1000,
    max_len: int = 128,
) -> dict:
    """Compute Mean Reciprocal Rank for lemma retrieval.

    For each query, find top-k nearest index goals, check if correct
    lemma is among their lemmas.

    Returns:
        {"mrr": float, "recall@k": float, "n_evaluated": int}
    """
    if len(queries) > sample_size:
        indices = random.sample(range(len(queries)), sample_size)
        q_goals = [queries[i] for i in indices]
        q_lemmas = [query_lemmas[i] for i in indices]
    else:
        q_goals = queries
        q_lemmas = query_lemmas

    device = next(encoder.parameters()).device
    encoder.eval()

    reciprocal_ranks = []
    found_in_k = 0

    with torch.no_grad():
        # Encode queries in batches
        query_embs_list = []
        for i in range(0, len(q_goals), 256):
            batch = q_goals[i : i + 256]
            batch_ids = _tokenize_batch(batch, vocab, max_len).to(device)
            embs = encoder(batch_ids)
            query_embs_list.append(embs.cpu())
        query_embs = torch.cat(query_embs_list, dim=0)  # [Q, D]

        # Compute similarity to all index embeddings
        sim = query_embs @ index_embeddings.T  # [Q, N]

        # For each query, check where correct lemma appears
        for i in range(len(q_goals)):
            correct_lemma = q_lemmas[i]
            if correct_lemma not in index_lemmas:
                continue

            # Find rank of the first index goal with the correct lemma
            correct_mask = torch.tensor(
                [l == correct_lemma for l in index_lemmas], dtype=torch.bool
            )

            # Sort similarities descending
            sorted_indices = torch.argsort(sim[i], descending=True)

            # Find rank of first correct match
            for rank, idx in enumerate(sorted_indices[:k]):
                if correct_mask[idx]:
                    reciprocal_ranks.append(1.0 / (rank + 1))
                    found_in_k += 1
                    break

    if not reciprocal_ranks:
        return {"mrr": 0.0, f"recall@{k}": 0.0, "n_evaluated": len(q_goals)}

    return {
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks),
        f"recall@{k}": found_in_k / max(1, len(q_goals)),
        "n_evaluated": len(q_goals),
    }


# ---------------------------------------------------------------------------
# Inference: retrieve lemmas for a new goal
# ---------------------------------------------------------------------------


def retrieve_lemmas(
    encoder: GoalOnlyEncoder,
    vocab: dict[str, int],
    goal_text: str,
    index_goals: list[str],
    index_lemmas: list[str],
    index_embeddings: torch.Tensor,
    k: int = 50,
    top_n: int = 30,
    max_len: int = 128,
) -> list[tuple[str, float]]:
    """Retrieve and rank lemmas for a new goal.

    1. Encode goal → embedding
    2. Find k nearest training goals (cosine similarity)
    3. Collect their correct lemmas
    4. Rank by mean_similarity × log(1 + count)
    5. Return top-N normalized
    """
    device = next(encoder.parameters()).device
    encoder.eval()

    with torch.no_grad():
        batch_ids = _tokenize_batch([goal_text], vocab, max_len).to(device)
        query_emb = encoder(batch_ids)  # [1, D]

        # Cosine similarity to all index embeddings
        sim = (query_emb @ index_embeddings.T).squeeze(0)  # [N]

        # Top-k indices
        if k >= len(sim):
            top_indices = torch.argsort(sim, descending=True)
        else:
            top_indices = torch.topk(sim, k).indices

        # Collect lemma statistics
        lemma_sims: dict[str, list[float]] = defaultdict(list)
        for idx in top_indices.tolist():
            s = float(sim[idx])
            if s > 0:
                lemma_name = index_lemmas[idx]
                lemma_sims[lemma_name].append(s)

        # Score: mean_similarity × log(1 + count)
        scored = []
        for lemma_name, sims in lemma_sims.items():
            mean_sim = sum(sims) / len(sims)
            count_bonus = math.log(1 + len(sims))
            score = mean_sim * count_bonus
            scored.append((lemma_name, score))

        scored.sort(key=lambda x: -x[1])

        # Normalize scores to [0, 1]
        if scored:
            max_s = max(s for _, s in scored)
            if max_s > 0:
                scored = [(name, s / max_s) for name, s in scored]

        return scored[:top_n]
