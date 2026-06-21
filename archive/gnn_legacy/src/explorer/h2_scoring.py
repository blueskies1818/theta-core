"""
H2 Scoring Architectures for Lemma Retrieval (Phase 2-H2 Study).

Three alternative lemma scoring architectures benchmarked against
the baseline cosine-similarity approach:

(a) Two-Tower with Learned Bilinear Scoring
    - Separate goal/candidate projection towers
    - Learnable bilinear interaction matrix W
    - score = goal_proj @ W @ candidate_proj

(b) Cross-Attention Goal→Candidates
    - Multi-head cross-attention: goal as query, candidates as keys/values
    - Attention weights serve as relevance scores
    - Captures fine-grained interactions between goal and candidate features

(c) Graph-Filtered Retrieval (k-hop)
    - Narrows candidates to k-hop neighbors of goal-relevant nodes
    - Scores within narrowed set using cosine similarity
    - Uses structural dependency information to prune irrelevant candidates

All architectures operate on top of pre-computed GNN embeddings.
They are drop-in replacements for the scoring function used in MCTS proof search.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from src.explorer.dependency_graph import DependencyGraph


# =============================================================================
# (a) Two-Tower with Learned Bilinear Scoring
# =============================================================================


class TwoTowerBilinear(nn.Module):
    """Two-tower architecture with learned bilinear interaction scoring.

    Goal tower projects the goal embedding into a latent scoring space.
    Candidate tower projects candidate lemma embeddings into the same space.
    A bilinear weight matrix W captures goal→candidate feature interactions.

    score = goal_proj @ W @ candidate_proj  (scalar per candidate)

    Args:
        embed_dim: Dimensionality of input GNN embeddings.
        hidden_dim: Hidden dimension for projection towers (default: 2× embed_dim).
        dropout: Dropout rate for regularization.
        init_W_scale: Scale factor for bilinear W initialization.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
        init_W_scale: float = 0.02,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim * 2

        # Goal tower: embed_dim → hidden_dim → embed_dim (projection + normalization)
        self.goal_tower = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # Candidate tower: same architecture, separate weights
        self.candidate_tower = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # Bilinear interaction matrix: [embed_dim, embed_dim]
        # Initialized as near-identity (reduces to dot-product at init) plus noise
        self.W = nn.Parameter(
            torch.eye(embed_dim) + torch.randn(embed_dim, embed_dim) * init_W_scale
        )

        # Optional bias
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        goal_embedding: torch.Tensor,
        candidate_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Score candidates against a goal.

        Args:
            goal_embedding: [D] or [1, D] goal embedding.
            candidate_embeddings: [C, D] candidate lemma embeddings.

        Returns:
            [C] scores, higher = more relevant.
        """
        if goal_embedding.dim() == 1:
            goal_embedding = goal_embedding.unsqueeze(0)  # [1, D]

        # Project through towers
        goal_proj = self.goal_tower(goal_embedding)  # [1, D]
        cand_proj = self.candidate_tower(candidate_embeddings)  # [C, D]

        # Bilinear scoring: goal_proj @ W @ cand_proj^T
        # (1, D) @ (D, D) @ (D, C) → (1, C)
        scores = torch.matmul(
            torch.matmul(goal_proj, self.W), cand_proj.T
        ).squeeze(0)  # [C]

        return scores + self.bias

    def score_batch(
        self,
        goal_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Score candidates for multiple goals in batch.

        Args:
            goal_embeddings: [B, D] goal embeddings.
            candidate_embeddings: [C, D] candidate embeddings.

        Returns:
            [B, C] scores.
        """
        goal_proj = self.goal_tower(goal_embeddings)  # [B, D]
        cand_proj = self.candidate_tower(candidate_embeddings)  # [C, D]

        # (B, D) @ (D, D) @ (D, C) → (B, C)
        scores = torch.matmul(
            torch.matmul(goal_proj, self.W), cand_proj.T
        )  # [B, C]

        return scores + self.bias


# =============================================================================
# (b) Cross-Attention Goal→Candidates
# =============================================================================


class CrossAttentionScorer(nn.Module):
    """Cross-attention scorer: goal attends to candidates.

    Multi-head cross-attention where the goal embedding serves as the query
    and candidate embeddings serve as keys and values. Attention weights
    directly represent relevance scores for each candidate.

    Architecture:
        1. Project goal → query (Q), candidates → keys (K) and values (V)
        2. Compute scaled dot-product attention: softmax(Q @ K^T / √d)
        3. Average attention weights across heads → final scores

    Args:
        embed_dim: Dimensionality of input embeddings.
        num_heads: Number of attention heads.
        head_dim: Dimension per head (default: embed_dim // num_heads).
        dropout: Dropout rate.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        head_dim: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim or (embed_dim // num_heads)
        self.inner_dim = self.num_heads * self.head_dim

        # Query projection (from goal)
        self.q_proj = nn.Linear(embed_dim, self.inner_dim)

        # Key projection (from candidates)
        self.k_proj = nn.Linear(embed_dim, self.inner_dim)

        # Value projection (from candidates)
        self.v_proj = nn.Linear(embed_dim, self.inner_dim)

        # Output projection (applied to attention-weighted values)
        self.out_proj = nn.Linear(self.inner_dim, embed_dim)

        # Score projection: collapses per-head attention into scalar scores
        self.score_proj = nn.Linear(self.num_heads, 1)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        goal_embedding: torch.Tensor,
        candidate_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Score candidates against a goal via cross-attention.

        Args:
            goal_embedding: [D] or [1, D].
            candidate_embeddings: [C, D].

        Returns:
            [C] attention-based relevance scores.
        """
        if goal_embedding.dim() == 1:
            goal_embedding = goal_embedding.unsqueeze(0)  # [1, D]

        C = candidate_embeddings.size(0)

        # Project to Q, K, V
        q = self.q_proj(goal_embedding)  # [1, inner_dim]
        k = self.k_proj(candidate_embeddings)  # [C, inner_dim]
        v = self.v_proj(candidate_embeddings)  # [C, inner_dim]

        # Reshape to multi-head: [B/1, num_heads, head_dim] and [C, num_heads, head_dim]
        q = q.view(1, self.num_heads, self.head_dim)  # [1, H, d]
        k = k.view(C, self.num_heads, self.head_dim)   # [C, H, d]
        v = v.view(C, self.num_heads, self.head_dim)   # [C, H, d]

        # Transpose for attention: [H, C, d]
        k = k.transpose(0, 1)  # [H, C, d]
        v = v.transpose(0, 1)  # [H, C, d]

        # Attention scores: Q @ K^T / √d
        # [H, 1, d] @ [H, d, C] → [H, 1, C]
        attn_weights = torch.matmul(
            q.transpose(0, 1), k.transpose(1, 2)
        ) * self.scale  # [H, 1, C]

        # Softmax over candidates
        attn_weights = F.softmax(attn_weights, dim=-1)  # [H, 1, C]
        attn_weights = self.dropout(attn_weights)

        # Aggregate attention across heads to get per-candidate scores
        # [H, 1, C] → [1, H, C] → [1, C, H] → project → [1, C, 1] → [C]
        attn_weights = attn_weights.permute(1, 0, 2)  # [1, H, C]
        attn_weights = attn_weights.permute(0, 2, 1)  # [1, C, H]
        scores = self.score_proj(attn_weights).squeeze(-1).squeeze(0)  # [C]

        return scores

    def score_batch(
        self,
        goal_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Score candidates for multiple goals in batch.

        Args:
            goal_embeddings: [B, D].
            candidate_embeddings: [C, D].

        Returns:
            [B, C] scores.
        """
        B = goal_embeddings.size(0)
        C = candidate_embeddings.size(0)

        q = self.q_proj(goal_embeddings)  # [B, inner_dim]
        k = self.k_proj(candidate_embeddings)  # [C, inner_dim]
        v = self.v_proj(candidate_embeddings)  # [C, inner_dim]

        q = q.view(B, self.num_heads, self.head_dim)
        k = k.view(C, self.num_heads, self.head_dim)
        v = v.view(C, self.num_heads, self.head_dim)

        # [B, H, d], [C, H, d]
        k = k.transpose(0, 1)  # [H, C, d]
        v = v.transpose(0, 1)  # [H, C, d]

        # [H, B, d] @ [H, d, C] → [H, B, C]
        attn_weights = torch.matmul(
            q.transpose(0, 1), k.transpose(1, 2)
        ) * self.scale  # [H, B, C]

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # [H, B, C] → [B, H, C] → [B, C, H] → project → [B, C]
        attn_weights = attn_weights.permute(1, 0, 2)  # [B, H, C]
        attn_weights = attn_weights.permute(0, 2, 1)  # [B, C, H]
        scores = self.score_proj(attn_weights).squeeze(-1)  # [B, C]

        return scores


# =============================================================================
# (c) Graph-Filtered Retrieval (k-hop narrowing + cosine)
# =============================================================================


def graph_filtered_retrieval(
    goal_embedding: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    candidate_names: list[str],
    goal_node_ids: list[str],
    dependency_graph: "DependencyGraph",
    k_hops: int = 3,
    candidate_idx_map: dict[str, int] | None = None,
) -> torch.Tensor:
    """Score candidates using k-hop graph-filtered retrieval + cosine similarity.

    Pipeline:
    1. For goal-relevant graph nodes, find k-hop neighbors
    2. Narrow candidate set to k-hop neighbors
    3. Score using cosine similarity within the narrowed set
    4. Non-neighbor candidates get score = -inf (filtered out)

    Args:
        goal_embedding: [D] goal embedding.
        candidate_embeddings: [C, D] candidate lemma embeddings.
        candidate_names: [C] lemma names matching candidate_embeddings order.
        goal_node_ids: List of graph node IDs relevant to the goal.
        dependency_graph: The dependency graph for neighborhood lookup.
        k_hops: Number of hops for neighborhood expansion.
        candidate_idx_map: Pre-built map from lemma name → index in candidates.

    Returns:
        [C] scores. Non-neighbor candidates get -inf.
    """
    if goal_embedding.dim() == 1:
        goal_embedding = goal_embedding.unsqueeze(0)  # [1, D]

    # Step 1: Compute k-hop neighborhood
    neighborhood: set[str] = set()
    for node_id in goal_node_ids:
        neighbors = dependency_graph.get_neighborhood(
            node_id, radius=k_hops, direction="both"
        )
        neighborhood.update(neighbors)

    # Also include the goal nodes themselves
    neighborhood.update(goal_node_ids)

    if not neighborhood:
        # Fallback: if no neighbors found, return all scored by cosine
        goal_norm = F.normalize(goal_embedding, dim=-1)
        cand_norm = F.normalize(candidate_embeddings, dim=-1)
        return (goal_norm @ cand_norm.T).squeeze(0)

    # Step 2: Build mask for k-hop neighbors
    if candidate_idx_map is None:
        candidate_idx_map = {name: i for i, name in enumerate(candidate_names)}

    neighbor_indices: list[int] = []
    for i, name in enumerate(candidate_names):
        if name in neighborhood:
            neighbor_indices.append(i)

    if not neighbor_indices:
        # No candidates in neighborhood → return all -inf
        return torch.full(
            (candidate_embeddings.size(0),),
            float("-inf"),
            device=candidate_embeddings.device,
        )

    # Step 3: Score only k-hop neighbors via cosine similarity
    filtered_embs = candidate_embeddings[neighbor_indices]  # [K, D]
    goal_norm = F.normalize(goal_embedding, dim=-1)  # [1, D]
    cand_norm = F.normalize(filtered_embs, dim=-1)  # [K, D]

    filtered_scores = (goal_norm @ cand_norm.T).squeeze(0)  # [K]

    # Step 4: Create full score tensor with -inf for filtered-out candidates
    scores = torch.full(
        (candidate_embeddings.size(0),),
        float("-inf"),
        device=candidate_embeddings.device,
        dtype=filtered_scores.dtype,
    )
    scores[neighbor_indices] = filtered_scores

    return scores


# =============================================================================
# Baseline: Cosine Similarity (for comparison)
# =============================================================================


def cosine_similarity_scoring(
    goal_embedding: torch.Tensor,
    candidate_embeddings: torch.Tensor,
) -> torch.Tensor:
    """Baseline cosine similarity scoring.

    Args:
        goal_embedding: [D] or [1, D].
        candidate_embeddings: [C, D].

    Returns:
        [C] cosine similarity scores.
    """
    if goal_embedding.dim() == 1:
        goal_embedding = goal_embedding.unsqueeze(0)

    goal_norm = F.normalize(goal_embedding, dim=-1)
    cand_norm = F.normalize(candidate_embeddings, dim=-1)

    return (goal_norm @ cand_norm.T).squeeze(0)


# =============================================================================
# Utility: Create scoring modules from config
# =============================================================================


def create_two_tower(embed_dim: int = 256, **kwargs) -> TwoTowerBilinear:
    """Create a Two-Tower Bilinear scorer with default config."""
    return TwoTowerBilinear(embed_dim=embed_dim, **kwargs)


def create_cross_attention(embed_dim: int = 256, **kwargs) -> CrossAttentionScorer:
    """Create a Cross-Attention scorer with default config."""
    return CrossAttentionScorer(embed_dim=embed_dim, **kwargs)
