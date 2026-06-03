"""Graph Neural Network encoder for math dependency graphs (Phase 2.2).

Learns node embeddings over the math dependency graph built in Phase 2.1.
These embeddings replace the transformer in Phase 1 — instead of generating
proofs token-by-token, the GNN evaluates which lemmas are relevant to a
given proof state.

Architecture: Graph Attention Network (GAT) with edge-type embeddings.
Implemented in pure PyTorch (no PyG/DGL dependency) for portability.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.explorer.gnn_config import GNNConfig


# ---------------------------------------------------------------------------
# Memory-efficient GAT layer
# ---------------------------------------------------------------------------


class GATLayer(nn.Module):
    """Single Graph Attention layer with edge-type conditioning.

    For each directed edge (src→tgt) with type e:
    1. Compute attention: α = softmax_tgt(LeakyReLU(q_src · k_tgt) / √d)
    2. Message: m = v_tgt + edge_embedding[e]
    3. Aggregate: h'_src = Σ α * m (summed over outgoing edges)
    4. h_src = LayerNorm( h_src + OutProj(h'_src) )

    Edge types condition messages via learned embeddings — avoids
    materializing [E, in_dim, out_dim] weight tensors.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 8,
        num_edge_types: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.out_dim = out_dim

        assert out_dim % num_heads == 0, (
            f"out_dim {out_dim} must be divisible by num_heads {num_heads}"
        )

        # Q, K, V projections (per-head via reshaping)
        self.q_proj = nn.Linear(in_dim, out_dim, bias=False)
        self.k_proj = nn.Linear(in_dim, out_dim, bias=False)
        self.v_proj = nn.Linear(in_dim, out_dim, bias=False)

        # Edge-type embeddings: adds type-specific bias to messages
        # Small: [num_edge_types, head_dim] ≈ 4 × 32 = 128 params
        self.edge_emb = nn.Parameter(
            torch.zeros(num_edge_types, self.head_dim)
        )

        # Attention scale
        self.scale = self.head_dim ** -0.5

        # Output projection
        self.out_proj = nn.Linear(out_dim, out_dim)

        # Skip connection projection
        self.skip_proj = (
            nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        )

        # Layer norm
        self.norm = nn.LayerNorm(out_dim)

        # Activation
        self.act = {"gelu": nn.GELU(), "relu": nn.ReLU(), "silu": nn.SiLU()}[
            activation
        ]
        self.drop = nn.Dropout(dropout)

        # Negative slope for LeakyReLU in attention
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(
        self,
        x: torch.Tensor,
        sources: torch.Tensor,
        targets: torch.Tensor,
        edge_types: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node features [num_nodes, in_dim].
            sources: Source node indices [E] (dependents).
            targets: Target node indices [E] (dependencies).
            edge_types: Edge type indices [E].
            num_nodes: Total number of nodes.

        Returns:
            Updated node features [num_nodes, out_dim].
        """
        num_edges = sources.size(0)

        # Project all nodes
        q = self.q_proj(x)  # [N, out_dim]
        k = self.k_proj(x)  # [N, out_dim]
        v = self.v_proj(x)  # [N, out_dim]

        # Gather for edges — these are the only [E, ...] tensors
        q_src = q[sources]  # [E, out_dim]
        k_tgt = k[targets]  # [E, out_dim]
        v_tgt = v[targets]  # [E, out_dim]

        # Reshape to [E, num_heads, head_dim]
        q_src = q_src.view(num_edges, self.num_heads, self.head_dim)
        k_tgt = k_tgt.view(num_edges, self.num_heads, self.head_dim)
        v_tgt = v_tgt.view(num_edges, self.num_heads, self.head_dim)

        # Edge-type embedding: [E, head_dim] → broadcast over heads
        edge_feat = self.edge_emb[edge_types]  # [E, head_dim]
        edge_feat = edge_feat.unsqueeze(1)  # [E, 1, head_dim]

        # Attention scores: dot(q_src, k_tgt) per head
        attn = (q_src * k_tgt).sum(dim=-1)  # [E, num_heads]
        attn = attn * self.scale

        # Softmax per source node (outgoing edges)
        attn = _scattered_softmax(attn, sources, num_nodes, self.num_heads)

        attn = self.drop(attn)  # [E, num_heads]

        # Message: v_tgt + edge_feat, weighted by attention
        msg = v_tgt + edge_feat  # [E, num_heads, head_dim]
        msg = attn.unsqueeze(-1) * msg  # [E, num_heads, head_dim]
        msg = msg.view(num_edges, self.out_dim)  # [E, out_dim]

        # Aggregate by source node
        out = torch.zeros(num_nodes, self.out_dim, device=x.device)
        out = out.index_add(0, sources, msg)

        # Output projection + skip connection + norm + activation
        out = self.out_proj(out)
        out = self.drop(out)

        residual = self.skip_proj(x)
        out = self.act(self.norm(out + residual))

        return out


def _scattered_softmax(
    attn: torch.Tensor,
    sources: torch.Tensor,
    num_nodes: int,
    num_heads: int,
) -> torch.Tensor:
    """Softmax over attention scores, normalized per source node.

    For each source node, softmax over its outgoing edges.

    Args:
        attn: [E, num_heads] — raw attention scores.
        sources: [E] — source node index for each edge.
        num_nodes: total nodes.
        num_heads: number of attention heads.

    Returns:
        Normalized attention weights [E, num_heads].
    """
    # Per-source max (for numerical stability)
    attn_max = torch.full(
        (num_nodes, num_heads), float("-inf"), device=attn.device
    )
    attn_max = attn_max.index_reduce(
        0, sources, attn, reduce="amax", include_self=False
    )
    attn_shifted = attn - attn_max[sources]

    # Per-source sum
    attn_exp = torch.exp(attn_shifted)
    attn_sum = torch.zeros(num_nodes, num_heads, device=attn.device)
    attn_sum = attn_sum.index_add(0, sources, attn_exp)
    attn_sum = attn_sum.clamp(min=1e-12)

    return attn_exp / attn_sum[sources]


# ---------------------------------------------------------------------------
# GNN Encoder
# ---------------------------------------------------------------------------


class GNNEncoder(nn.Module):
    """Graph Neural Network encoder for math dependency graphs.

    Takes a dependency graph and produces node embeddings that capture
    each theorem's position in the logical web of mathematics.

    Usage with MCTS (Phase 2.3):
        embeddings = gnn(features, sources, targets, edge_types)
        scores = gnn.compute_link_scores(embeddings, state_nodes, candidates)
        # scores[state, candidate] = relevance of candidate lemma to state
    """

    def __init__(self, config: GNNConfig | None = None):
        super().__init__()
        self.config = config or GNNConfig()

        # Input projection
        self.input_proj = nn.Linear(self.config.input_dim, self.config.hidden_dim)

        # Message-passing layers
        self.layers = nn.ModuleList()
        for _ in range(self.config.num_layers):
            self.layers.append(
                GATLayer(
                    in_dim=self.config.hidden_dim,
                    out_dim=self.config.hidden_dim,
                    num_heads=self.config.num_heads,
                    num_edge_types=self.config.num_edge_types
                    if self.config.use_edge_types
                    else 1,
                    dropout=self.config.dropout,
                    activation=self.config.activation,
                )
            )

        self.out_norm = nn.LayerNorm(self.config.hidden_dim)
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(
        self,
        x: torch.Tensor,
        sources: torch.Tensor,
        targets: torch.Tensor,
        edge_types: torch.Tensor | None = None,
        num_nodes: int | None = None,
    ) -> torch.Tensor:
        """Encode all nodes through GNN layers.

        Args:
            x: [N, input_dim] initial node features.
            sources: [E] source indices.
            targets: [E] target indices.
            edge_types: [E] edge type codes, or None for all-zeros.
            num_nodes: total nodes (inferred if None).

        Returns:
            [N, hidden_dim] node embeddings.
        """
        if num_nodes is None:
            num_nodes = x.size(0)
        if edge_types is None:
            edge_types = torch.zeros(sources.size(0), dtype=torch.long, device=x.device)

        h = F.gelu(self.input_proj(x))

        for layer in self.layers:
            # Forward direction
            h = layer(h, sources, targets, edge_types, num_nodes)
            # Reverse direction (bidirectional message passing)
            if self.config.bidirectional:
                rev_types = edge_types + (self.config.num_edge_types // 2)
                rev_types = rev_types.clamp(0, self.config.num_edge_types - 1)
                h = layer(h, targets, sources, rev_types, num_nodes)

        return self.out_norm(h)

    # ------------------------------------------------------------------
    # Link prediction
    # ------------------------------------------------------------------

    def compute_link_scores(
        self,
        node_embeddings: torch.Tensor,
        query_indices: torch.Tensor,
        candidate_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Score candidates for queries: dot-product similarity.

        Returns [num_queries, num_candidates] — higher = stronger dependency.
        """
        q = node_embeddings[query_indices]  # [Q, D]
        c = node_embeddings[candidate_indices]  # [C, D]
        return torch.matmul(q, c.T)  # [Q, C]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": self.state_dict(), "config": self.config}, path)

    @classmethod
    def load(cls, path: Path | str, config: GNNConfig | None = None) -> "GNNEncoder":
        state = torch.load(path, map_location="cpu", weights_only=False)
        if config is None:
            config = state.get("config", GNNConfig())
        model = cls(config)
        model.load_state_dict(state["model_state_dict"])
        return model


# ---------------------------------------------------------------------------
# Initial feature extraction
# ---------------------------------------------------------------------------


def extract_initial_features(
    graph: "DependencyGraph",
    config: GNNConfig,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Extract initial node features for the GNN.

    Args:
        graph: The dependency graph.
        config: GNN config (used for init_features mode).
        device: Target device.

    Returns:
        [num_nodes, input_dim] feature tensor.
    """
    # Import here to avoid circular dependency
    from src.explorer.dependency_graph import DependencyGraph

    node_ids = sorted(graph.graph.nodes())
    num_nodes = len(node_ids)
    device = device or torch.device("cpu")

    if config.init_features == "random":
        features = F.normalize(torch.randn(num_nodes, config.input_dim), dim=1)

    elif config.init_features == "onehot":
        features = _onehot_features(graph, node_ids, config.input_dim)

    elif config.init_features == "transformer":
        features = _transformer_features(graph, node_ids, config)

    else:
        raise ValueError(f"Unknown init_features: {config.init_features}")

    return features.to(device)


def _onehot_features(
    graph: "DependencyGraph", node_ids: list[str], dim: int
) -> torch.Tensor:
    """Simple domain+type one-hot features."""
    from src.explorer.dependency_graph import NodeType

    domains = sorted(
        {graph.graph.nodes[n].get("domain", "Unknown") for n in node_ids}
    )
    num_nodes = len(node_ids)

    features = torch.zeros(num_nodes, dim)
    for i, nid in enumerate(node_ids):
        attrs = graph.get_node(nid)
        if attrs:
            dom = attrs.get("domain", "Unknown")
            if dom in domains and domains.index(dom) < dim:
                features[i, domains.index(dom)] = 1.0
            nt = attrs.get("node_type", NodeType.LEMMA)
            offset = min(len(domains), dim - 8)
            type_idx = list(NodeType).index(nt) if nt in NodeType else 0
            if offset + type_idx < dim:
                features[i, offset + type_idx] = 1.0

    return features


def _transformer_features(
    graph: "DependencyGraph",
    node_ids: list[str],
    config: GNNConfig,
) -> torch.Tensor:
    """Embed theorem statements using the Phase 1 SFT model."""
    from transformers import AutoModel, AutoTokenizer

    num_nodes = len(node_ids)

    try:
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
        model = AutoModel.from_pretrained(
            config.base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
        )
    except Exception as e:
        print(f"Warning: Could not load transformer ({e}). Using random init.")
        return F.normalize(torch.randn(num_nodes, config.input_dim), dim=1)

    statements = []
    for nid in node_ids:
        attrs = graph.get_node(nid)
        stmt = attrs.get("statement", "") if attrs else ""
        statements.append(stmt or nid)

    batch_size = 64
    all_embeddings = []

    with torch.no_grad():
        for i in range(0, len(statements), batch_size):
            batch = statements[i : i + batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding=True,
            )
            outputs = model(**inputs)
            emb = outputs.last_hidden_state.mean(dim=1)  # [B, hidden]
            all_embeddings.append(emb)

    embeddings = torch.cat(all_embeddings, dim=0)  # [N, embed_dim]

    if embeddings.size(1) != config.input_dim:
        proj = nn.Linear(embeddings.size(1), config.input_dim)
        embeddings = proj(embeddings)

    return embeddings


# ---------------------------------------------------------------------------
# Graph-to-tensor conversion
# ---------------------------------------------------------------------------


def prepare_graph_tensors(
    graph: "DependencyGraph",
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Convert DependencyGraph edges to tensors for GNN forward pass.

    Returns:
        sources: [E] LongTensor — source node integer indices.
        targets: [E] LongTensor — target node integer indices.
        edge_types: [E] LongTensor — integer edge type codes.
        num_nodes: int.
    """
    from src.explorer.dependency_graph import EdgeType

    if not graph._id_to_idx:
        graph._rebuild_indices()

    _etype_map = {
        EdgeType.USES_IN_PROOF: 0,
        EdgeType.USES_IN_STATEMENT: 1,
        EdgeType.GENERALIZES: 2,
        EdgeType.INSTANTIATES: 3,
    }

    src_list, tgt_list, et_list = [], [], []
    for u, v, attrs in graph._graph.edges(data=True):
        if u in graph._id_to_idx and v in graph._id_to_idx:
            src_list.append(graph._id_to_idx[u])
            tgt_list.append(graph._id_to_idx[v])
            et_list.append(_etype_map.get(attrs.get("type", EdgeType.USES_IN_PROOF), 0))

    device = device or torch.device("cpu")
    return (
        torch.tensor(src_list, dtype=torch.long, device=device),
        torch.tensor(tgt_list, dtype=torch.long, device=device),
        torch.tensor(et_list, dtype=torch.long, device=device),
        graph.num_nodes,
    )
