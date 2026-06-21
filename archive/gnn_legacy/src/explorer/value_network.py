"""Value Network for proof state evaluation.

Given a partial proof state (goal + context), estimates the probability of
eventual successful proof completion. Used to prune unpromising branches
during best-first search.

Architecture:
  Goal text → GoalEncoder (frozen GNN) → goal_embedding (768-dim)
    → ValueHead MLP (768→256→1) → sigmoid → P(success)

The GNN encoder is frozen; only the value head is trained. Training uses
verified (state, outcome) pairs from proof-checker pass/fail ground truth.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueHead(nn.Module):
    """MLP head that maps a goal embedding to a scalar success probability.

    Simple 2-layer MLP with GELU activation and sigmoid output.
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, goal_embedding: torch.Tensor) -> torch.Tensor:
        """Predict success probability from goal embedding.

        Args:
            goal_embedding: [D] or [B, D] normalized goal embedding.

        Returns:
            Scalar probability in [0, 1].
        """
        logit = self.net(goal_embedding)
        return torch.sigmoid(logit).squeeze(-1)


class ValueNetwork(nn.Module):
    """Value network: estimates P(proof success | current proof state).

    Wraps a frozen GNN GoalEncoder and adds a trainable ValueHead.
    The GNN encoder handles goal text → goal embedding.
    The value head maps goal embedding → success probability.

    Usage:
        vn = ValueNetwork(gnn)  # gnn.goal_encoder is frozen
        p_success = vn.predict(goal_embedding)

    Training:
        vn.train()  # only value head trainable (encoder frozen)
        loss = F.binary_cross_entropy(vn(goal_embeddings), targets)
    """

    def __init__(
        self,
        gnn,  # GNNEncoder with goal_encoder
        hidden_dim: int = 256,
        dropout: float = 0.1,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        self.gnn = gnn

        # Detect encoder output dim
        if gnn.goal_encoder is not None:
            # GoalEncoder output dim = hidden_dim
            enc_dim = gnn.config.hidden_dim
        else:
            enc_dim = gnn.config.hidden_dim

        self.value_head = ValueHead(
            input_dim=enc_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        # Freeze GNN encoder
        if freeze_encoder:
            try:
                for param in self.gnn.parameters():
                    param.requires_grad = False
            except AttributeError:
                # Mock/stub GNN without parameters() — ignore
                pass
            # But ensure value head is trainable
            for param in self.value_head.parameters():
                param.requires_grad = True

        self.enc_dim = enc_dim

    def encode_goal(self, goal_text: str, **encode_kwargs) -> torch.Tensor:
        """Encode a goal text to an embedding using the GNN's pipeline.

        This is a convenience wrapper that delegates to the GNN's goal encoding
        pipeline (normalized text matching → keyword average → GoalEncoder).

        Args:
            goal_text: Raw goal text or normalized expression.
            **encode_kwargs: Passed through to the embedding function
                (e.g., node_embeddings, keyword maps).

        Returns:
            [enc_dim] normalized goal embedding.
        """
        # Delegate to the caller's embedding function — we expose this
        # as a hook so the search can inject its own embedding pipeline.
        # By default, return zeros (caller must provide context).
        return torch.zeros(self.enc_dim)

    def forward(self, goal_embedding: torch.Tensor) -> torch.Tensor:
        """Predict success probability from a pre-computed goal embedding.

        Args:
            goal_embedding: [B, D] or [D] normalized goal embedding.

        Returns:
            [B] or scalar probability in [0, 1].
        """
        return self.value_head(goal_embedding)

    def predict(self, goal_embedding: torch.Tensor) -> torch.Tensor:
        """Convenience: predict with no_grad.

        Args:
            goal_embedding: [D] or [B, D] goal embedding.

        Returns:
            Scalar probability.
        """
        with torch.no_grad():
            self.eval()
            return self.forward(goal_embedding)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path | str) -> None:
        """Save value head weights (GNN saved separately)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_dict = {
            "value_head_state_dict": self.value_head.state_dict(),
            "enc_dim": self.enc_dim,
            "hidden_dim": self.value_head.net[0].out_features,
        }
        torch.save(save_dict, path)

    @classmethod
    def load(
        cls,
        path: Path | str,
        gnn,  # GNNEncoder with goal_encoder
        freeze_encoder: bool = True,
    ) -> "ValueNetwork":
        """Load value head weights onto a given GNN encoder."""
        state = torch.load(str(path), map_location="cpu", weights_only=False)
        hidden_dim = state.get("hidden_dim", 256)
        vn = cls(gnn, hidden_dim=hidden_dim, freeze_encoder=freeze_encoder)
        vn.value_head.load_state_dict(state["value_head_state_dict"])
        return vn
