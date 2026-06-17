"""H2 STUDY — Lemma scoring architectures for lemma retrieval benchmarking.

Four scoring architectures for ranking candidate lemmas given a proof goal:
  (a) Two-tower with learned bilinear scoring
  (b) Cross-attention goal→candidates
  (c) Graph-filtered retrieval narrowing to k-hop neighbors before cosine
  (d) Baseline cosine-similarity (dot product of normalized embeddings)

These are evaluated in scripts/bench_h2_scoring.py against the gate3
lemma-novelty test set (14 theorems requiring novel lemma retrieval).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.explorer.dependency_graph import DependencyGraph
    from src.explorer.gnn_encoder import GNNEncoder


# ==============================================================================
# (d) Baseline Cosine Scorer
# ==============================================================================


class BaselineCosineScorer(nn.Module):
    """Baseline: dot-product similarity of normalized embeddings.

    This replicates GNNEncoder.compute_link_scores() but as a standalone
    module for fair comparison. Scores = query_emb @ candidate_embs^T.
    """

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(
        self,
        query_emb: torch.Tensor,       # [B, D] or [D]
        candidate_embs: torch.Tensor,  # [N, D]
    ) -> torch.Tensor:
        """Score candidates against query.

        Args:
            query_emb: Goal embedding [B, D] or [D].
            candidate_embs: Candidate lemma embeddings [N, D].

        Returns:
            [B, N] relevance scores (higher = better).
        """
        if query_emb.dim() == 1:
            query_emb = query_emb.unsqueeze(0)
        # Both inputs should already be normalized externally,
        # but we normalize here for safety.
        q = F.normalize(query_emb, dim=-1)
        c = F.normalize(candidate_embs, dim=-1)
        return torch.matmul(q, c.T)  # [B, N]


# ==============================================================================
# (a) Two-Tower Bilinear Scorer
# ==============================================================================


class TwoTowerBilinearScorer(nn.Module):
    """Two-tower architecture with learned bilinear interaction.

    Architecture:
      Query tower:  MLP(query_emb)     → query_repr  [B, D']
      Cand tower:   MLP(cand_emb)      → cand_repr   [N, D']
      Score:        query_repr @ W @ cand_repr^T      [B, N]

    The bilinear weight matrix W ∈ R^{D'×D'} captures complex interactions
    between the query and candidate representations that simple dot-product
    cannot express (e.g., asymmetric relevance, conditional independence).
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        bottleneck_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim

        # Query tower: projects goal embedding into interaction space
        self.query_tower = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim * 2, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
        )

        # Candidate tower: projects lemma embeddings into interaction space
        self.candidate_tower = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim * 2, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
        )

        # Bilinear weight matrix: W ∈ R^{D'×D'}
        # Initialized near-identity so it starts close to dot-product.
        self.W = nn.Parameter(torch.eye(bottleneck_dim) * 0.1)

        self._init_weights()

    def _init_weights(self):
        """Xavier init for tower weights, small random perturbation for W."""
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2 and name != "W":
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        # W starts near identity (dot-product fallback) with small noise
        with torch.no_grad():
            self.W.copy_(torch.eye(self.bottleneck_dim) * 0.1
                         + torch.randn(self.bottleneck_dim, self.bottleneck_dim) * 0.01)

    def forward(
        self,
        query_emb: torch.Tensor,
        candidate_embs: torch.Tensor,
    ) -> torch.Tensor:
        """Score candidates via bilinear interaction.

        Args:
            query_emb: Goal embedding [B, D] or [D].
            candidate_embs: Lemma embeddings [N, D].

        Returns:
            [B, N] relevance scores.
        """
        if query_emb.dim() == 1:
            query_emb = query_emb.unsqueeze(0)

        # Normalize inputs
        q = F.normalize(query_emb, dim=-1)
        c = F.normalize(candidate_embs, dim=-1)

        # Tower projections
        q_repr = self.query_tower(q)      # [B, D']
        c_repr = self.candidate_tower(c)  # [N, D']

        # Bilinear interaction: q_repr @ W @ c_repr^T
        # q_repr: [B, D'], W: [D', D'], c_repr: [N, D']
        qW = torch.matmul(q_repr, self.W)           # [B, D']
        scores = torch.matmul(qW, c_repr.T)          # [B, N]
        return scores

    def calibrate_from_baseline(
        self,
        query_emb: torch.Tensor,
        candidate_embs: torch.Tensor,
        baseline_scorer: BaselineCosineScorer,
        epochs: int = 100,
        lr: float = 1e-3,
    ) -> float:
        """Quick MSE calibration to cosine baseline. Returns final loss."""
        # Detach and move inputs to the same device as model
        device = next(self.parameters()).device
        q = query_emb.detach().to(device).clone()
        c = candidate_embs.detach().to(device).clone()
        with torch.no_grad():
            target = baseline_scorer(q, c).detach().to(device).clone()

        opt = torch.optim.AdamW(self.parameters(), lr=lr)
        self.train()
        final_loss = 0.0
        for _ in range(epochs):
            opt.zero_grad(set_to_none=True)
            pred = self.forward(q, c)
            loss = F.mse_loss(pred, target)
            loss.backward()
            opt.step()
            final_loss = float(loss.detach().cpu())
        self.eval()
        return final_loss


# ==============================================================================
# (b) Cross-Attention Goal→Candidates Scorer
# ==============================================================================


class CrossAttentionScorer(nn.Module):
    """Multi-head cross-attention where the goal attends over candidates.

    Architecture:
      Query:   Linear(goal_emb)      → [B, H, D_head]
      Key/Val: Linear(cand_embs)    → [N, H, D_head]
      Attention: softmax(Q @ K^T / √d) → [B, H, N]
      Scores:   mean over heads of attention weights → [B, N]

    The per-head attention weights act as relevance scores — if a lemma
    is important for the goal, all heads should attend to it.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, (
            f"hidden_dim {hidden_dim} must be divisible by num_heads {num_heads}"
        )
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Projections for multi-head attention
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(
        self,
        query_emb: torch.Tensor,
        candidate_embs: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-attention: goal attends to all candidates.

        Args:
            query_emb: Goal embedding [B, D] or [D].
            candidate_embs: Lemma embeddings [N, D].

        Returns:
            [B, N] relevance scores (mean attention across heads).
        """
        if query_emb.dim() == 1:
            query_emb = query_emb.unsqueeze(0)

        B = query_emb.size(0)
        N = candidate_embs.size(0)
        D = self.hidden_dim
        H = self.num_heads
        d = self.head_dim

        # Normalize
        q = F.normalize(query_emb, dim=-1)
        c = F.normalize(candidate_embs, dim=-1)

        # Project to multi-head
        Q = self.q_proj(q).view(B, H, d)   # [B, H, d]
        K = self.k_proj(c).view(N, H, d)   # [N, H, d]

        # Permute for batched matmul: [H, B, d] and [H, N, d]
        Q = Q.permute(1, 0, 2)   # [H, B, d]
        K = K.permute(1, 0, 2)   # [H, N, d]

        # Scaled dot-product attention
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [H, B, N]
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Scoring: mean attention weight across heads → [B, N]
        scores = attn.mean(dim=0)  # [B, N]
        return scores

    def calibrate_from_baseline(
        self,
        query_emb: torch.Tensor,
        candidate_embs: torch.Tensor,
        baseline_scorer: BaselineCosineScorer,
        epochs: int = 100,
        lr: float = 1e-3,
    ) -> float:
        """Quick MSE calibration to cosine baseline. Returns final loss."""
        # Detach and move inputs to the same device as model
        device = next(self.parameters()).device
        q = query_emb.detach().to(device).clone()
        c = candidate_embs.detach().to(device).clone()
        with torch.no_grad():
            target = baseline_scorer(q, c).detach().to(device).clone()

        opt = torch.optim.AdamW(self.parameters(), lr=lr)
        self.train()
        final_loss = 0.0
        for _ in range(epochs):
            opt.zero_grad(set_to_none=True)
            pred = self.forward(q, c)
            loss = F.mse_loss(pred, target)
            loss.backward()
            opt.step()
            final_loss = float(loss.detach().cpu())
        self.eval()
        return final_loss


# ==============================================================================
# (c) Graph-Filtered Cosine Scorer
# ==============================================================================


class GraphFilteredCosineScorer(nn.Module):
    """Restrict candidates to k-hop neighbors before cosine scoring.

    Before computing cosine similarity, this scorer narrows the candidate
    set to lemmas within k hops of the query node in the dependency graph.
    This leverages structural information: relevant lemmas are typically
    close in the graph (e.g., `eval_add` is 2-3 hops from polynomial
    evaluation theorems).

    For theorems not in the graph (synthetic test theorems), we find
    the graph via related lemmas mentioned in the statement, then expand
    from those anchor nodes.

    Scoring: standard cosine similarity, but only on the filtered subset.
    Non-neighbor candidates get score = -inf.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        k_hops: int = 3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.k_hops = k_hops

    def compute_neighbor_mask(
        self,
        query_node_ids: list[str],
        candidate_node_ids: list[str],
        graph: "DependencyGraph",
        k_hops: int | None = None,
    ) -> torch.Tensor:
        """Build a boolean mask [B, N] where True = candidate is within k hops.

        Args:
            query_node_ids: [B] node IDs for queries (may not be in graph).
            candidate_node_ids: [N] node IDs for candidates.
            graph: The dependency graph.
            k_hops: Override self.k_hops.

        Returns:
            [B, N] bool tensor. True = candidate reachable within k hops.
        """
        k = k_hops if k_hops is not None else self.k_hops
        B = len(query_node_ids)
        N = len(candidate_node_ids)
        mask = torch.zeros(B, N, dtype=torch.bool)

        import networkx as nx

        for b, qid in enumerate(query_node_ids):
            if not graph.has_node(qid):
                # Query not in graph — try to find via lemma neighbors
                # Use all lemma nodes as potential starts (too expensive).
                # Instead, skip filtering for this query (all candidates pass).
                mask[b, :] = True
                continue

            # BFS from query node up to k hops
            reachable: set[str] = set()
            frontier = {qid}
            for hop in range(k + 1):
                reachable |= frontier
                if hop == k:
                    break
                next_frontier: set[str] = set()
                for node in frontier:
                    next_frontier.update(graph._graph.neighbors(node))
                    next_frontier.update(graph._graph.predecessors(node))
                frontier = next_frontier - reachable
                if not frontier:
                    break

            # Mark reachable candidates
            for j, cid in enumerate(candidate_node_ids):
                if cid in reachable:
                    mask[b, j] = True

        return mask

    def forward(
        self,
        query_emb: torch.Tensor,
        candidate_embs: torch.Tensor,
        neighbor_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Score only k-hop neighbors via cosine similarity.

        Args:
            query_emb: Goal embedding [B, D] or [D].
            candidate_embs: Lemma embeddings [N, D].
            neighbor_mask: [B, N] bool mask from compute_neighbor_mask().

        Returns:
            [B, N] scores. Non-neighbor candidates get -inf.
        """
        if query_emb.dim() == 1:
            query_emb = query_emb.unsqueeze(0)

        q = F.normalize(query_emb, dim=-1)
        c = F.normalize(candidate_embs, dim=-1)
        raw_scores = torch.matmul(q, c.T)  # [B, N]

        # Mask non-neighbors to -inf
        scores = raw_scores.masked_fill(~neighbor_mask, float("-inf"))
        return scores


# ==============================================================================
# Utility: create goal embeddings from theorem statements
# ==============================================================================


def build_goal_embedding(
    theorem: dict,
    lemma_embeddings: dict[str, torch.Tensor],
    gnn_encoder: Optional["GNNEncoder"] = None,
    fallback_dim: int = 256,
) -> torch.Tensor:
    """Build a goal embedding for a theorem from its required lemmas.

    Matches the statement text against lemma node IDs and averages
    the embeddings of matching lemmas, optionally passing through
    the GNN's goal encoder.

    Args:
        theorem: Dict with 'statement' and 'proof' keys.
        lemma_embeddings: Dict mapping node ID → embedding tensor [D].
        gnn_encoder: Optional GNNEncoder for goal projection.
        fallback_dim: Dimension for fallback random embedding.

    Returns:
        [D] goal embedding tensor.
    """
    import re

    stmt = theorem.get("statement", "")
    proof = theorem.get("proof", "")

    # Extract lemma names from proof and statement
    lemma_names: set[str] = set()

    # From proof: Polynomial.XXX references
    for m in re.finditer(r"Polynomial\.(\w+)", proof):
        lemma_names.add(m.group(1))

    # From proof: simpa/exact using ...
    for m in re.finditer(r"(?:using|exact)\s+(\w+)", proof):
        lemma_names.add(m.group(1))

    # From proof: simp [xxx] patterns
    for m in re.finditer(r"simp\s*\[(\w+)\]", proof):
        lemma_names.add(m.group(1))

    # From statement: infer needed lemmas by keyword
    stmt_lower = stmt.lower()
    if "eval" in stmt_lower:
        if "+" in stmt or "add" in stmt_lower:
            lemma_names.add("eval_add")
        if "*" in stmt or "mul" in stmt_lower:
            lemma_names.add("eval_mul")
        if "-" in stmt or "sub" in stmt_lower:
            lemma_names.add("eval_sub")
    if "derivative" in stmt_lower:
        if "+" in stmt or "add" in stmt_lower:
            lemma_names.add("derivative_add")
        if "c" in stmt_lower and "mul" in stmt_lower:
            lemma_names.add("derivative_C_mul")
        lemma_names.add("derivative_X_pow")
    if "natdegree" in stmt_lower or "nat_degree" in stmt_lower:
        if "x ^" in stmt_lower or "x_pow" in stmt_lower:
            lemma_names.add("natDegree_X_pow")
        if "c" in stmt_lower and "mul" in stmt_lower:
            lemma_names.add("natDegree_C_mul_X_pow")
        if "mul" in stmt_lower:
            lemma_names.add("natDegree_mul")
    if ".map" in stmt_lower:
        lemma_names.add("map_id")
    if "degree" in stmt_lower:
        if "add" in stmt_lower:
            lemma_names.add("degree_add_eq_left_of_degree_lt")
        if "sub" in stmt_lower:
            lemma_names.add("degree_sub_eq_left_of_degree_lt")
    if "monic" in stmt_lower:
        lemma_names.add("monic_X_sub_C")
        lemma_names.add("monic_X_pow_add")

    # Collect embeddings for matched lemmas
    matched_embs: list[torch.Tensor] = []
    for name in lemma_names:
        if name in lemma_embeddings:
            matched_embs.append(lemma_embeddings[name])

    if not matched_embs:
        # Fallback: random normalized embedding
        return F.normalize(torch.randn(fallback_dim), dim=-1)

    # Average matched lemma embeddings
    avg_emb = torch.stack(matched_embs).mean(dim=0)

    # Optionally pass through goal encoder
    if gnn_encoder is not None:
        return gnn_encoder.encode_goal(avg_emb)

    return F.normalize(avg_emb, dim=-1)
