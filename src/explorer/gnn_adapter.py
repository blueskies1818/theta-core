"""Projection adapter that sits on top of a frozen GNN.

Freeze the full GNN. Train only this small head to re-weight
embedding dimensions for proof utility. Preserves graph topology
knowledge while learning which dimensions matter for proof-closing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GNNAdapterHead(nn.Module):
    """Two-layer projection head over frozen GNN embeddings.

    Input: 256-dim GNN node embedding
    Output: 256-dim proof-utility embedding (L2-normalized)

    ~132K params, designed to be the ONLY trainable component
    when paired with a frozen GNN backbone.
    """

    def __init__(self, input_dim: int = 256, hidden_dim: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.LayerNorm(input_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.proj:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [*, D] GNN embeddings → [*, D] adapted embeddings."""
        adapted = self.proj(x)
        return F.normalize(adapted, dim=-1)

    def count_params(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())
