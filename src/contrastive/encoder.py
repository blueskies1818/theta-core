"""Contrastive dual-encoder model for proof-step lemma retrieval.

Architecture (Path C):
  goal_encoder: character-CNN → MLP → L2-normalized embedding
  lemma_encoder: character-CNN → MLP → L2-normalized embedding
  Loss: InfoNCE (in-batch contrastive) on (goal, correct_lemma) pairs.

This replaces the GNN+GoalEncoder cosine-similarity retrieval with
a learned relevance scoring function trained directly on proof-step pairs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ContrastiveConfig:
    """Configuration for the ContrastiveDualEncoder."""

    # -- Embedding dimensions ------------------------------------------------
    hidden_dim: int = 256
    """Dimension of the output embedding (shared for goal and lemma)."""

    # -- Tokenizer ----------------------------------------------------------
    vocab_size: int = 256
    """Size of the character-level vocabulary."""

    max_seq_len: int = 256
    """Maximum sequence length for tokenization."""

    char_embed_dim: int = 64
    """Dimension of character embeddings."""

    # -- CNN encoder --------------------------------------------------------
    cnn_filters: int = 128
    """Filters per kernel size in the character CNN."""

    cnn_kernel_sizes: tuple[int, ...] = (2, 3, 4, 5)
    """Kernel sizes for the character CNN."""

    cnn_dropout: float = 0.2
    """Dropout after CNN and in MLP layers."""

    # -- MLP projection -----------------------------------------------------
    mlp_expansion: int = 2
    """Expansion factor for the MLP projection head."""
    
    # -- Pooling ------------------------------------------------------------
    pooling: Literal["mean", "max", "attention"] = "attention"
    """How to pool character-CNN outputs into a fixed-size vector."""

    # -- Training -----------------------------------------------------------
    temperature: float = 0.07
    """Temperature for InfoNCE contrastive loss."""

    learning_rate: float = 3e-4
    """Learning rate for AdamW optimizer."""

    weight_decay: float = 1e-4
    """Weight decay for AdamW optimizer."""

    batch_size: int = 256
    """Training batch size (in-batch negatives scale with this)."""

    num_epochs: int = 30
    """Number of training epochs."""

    # -- Hardware -----------------------------------------------------------
    device: str = "cpu"
    """Device to run on (CPU only for this project)."""

    num_threads: int = 4
    """Number of PyTorch CPU threads."""

    # -- Evaluation ---------------------------------------------------------
    retrieval_k: int = 10
    """Top-K for retrieval evaluation."""


# ---------------------------------------------------------------------------
# Character CNN Encoder (shared architecture for goal and lemma)
# ---------------------------------------------------------------------------


class CharCNNEncoder(nn.Module):
    """Character-level CNN encoder for mathematical text.

    Tokenizes text at the character level (fixed vocabulary of 256 chars),
    embeds each character, applies parallel 1D convolutions with multiple
    kernel sizes, pools the features, and projects to a fixed-dim embedding.

    Designed for mixed math notation + natural language in proof goals
    and lemma identifiers.
    """

    def __init__(self, config: ContrastiveConfig):
        super().__init__()
        self.config = config

        # Character embedding
        self.char_embed = nn.Embedding(config.vocab_size, config.char_embed_dim)

        # Parallel 1D convolutions (use "same" padding for consistent output lengths)
        self.convs = nn.ModuleList([
            nn.Conv1d(
                config.char_embed_dim,
                config.cnn_filters,
                k,
                padding="same",
            )
            for k in config.cnn_kernel_sizes
        ])

        total_cnn_dim = config.cnn_filters * len(config.cnn_kernel_sizes)

        # Pooling: learnable attention or fixed pooling
        if config.pooling == "attention":
            self.attn_query = nn.Parameter(torch.randn(1, 1, total_cnn_dim) * 0.02)
        self.pooling = config.pooling

        # MLP projection head
        self.proj = nn.Sequential(
            nn.Linear(total_cnn_dim, config.hidden_dim * config.mlp_expansion),
            nn.GELU(),
            nn.Dropout(config.cnn_dropout),
            nn.Linear(config.hidden_dim * config.mlp_expansion, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        # Initialize char embeddings with small random values
        nn.init.normal_(self.char_embed.weight, std=0.1)

    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:
        """Encode character-level token IDs into embedding vectors.

        Args:
            char_ids: [B, L] LongTensor of character IDs (0-255).
                      Padded with 0 (null character).

        Returns:
            [B, hidden_dim] normalized embeddings.
        """
        # Embed characters: [B, L] → [B, L, char_embed_dim]
        x = self.char_embed(char_ids)

        # Transpose for Conv1d: [B, char_embed_dim, L]
        x = x.transpose(1, 2)

        # Apply convolutions: each conv produces [B, cnn_filters, L']
        conv_outputs = []
        for conv in self.convs:
            out = conv(x)  # [B, cnn_filters, L']
            out = F.gelu(out)
            conv_outputs.append(out)

        # Concatenate along filter dimension: [B, total_cnn_dim, L']
        x = torch.cat(conv_outputs, dim=1)

        # Pool over sequence dimension
        if self.pooling == "mean":
            x = x.mean(dim=-1)  # [B, total_cnn_dim]
        elif self.pooling == "max":
            x = x.max(dim=-1).values  # [B, total_cnn_dim]
        elif self.pooling == "attention":
            # Learnable query attention over positions
            # x: [B, total_cnn_dim, L]
            # attn_query: [1, 1, total_cnn_dim]
            attn_scores = (self.attn_query @ x).squeeze(1)  # [B, L]
            attn_weights = F.softmax(attn_scores, dim=-1)  # [B, L]
            x = (x * attn_weights.unsqueeze(1)).sum(dim=-1)  # [B, total_cnn_dim]

        # MLP projection
        x = self.proj(x)  # [B, hidden_dim]

        # L2 normalize
        return F.normalize(x, dim=-1)


# ---------------------------------------------------------------------------
# Contrastive Dual Encoder
# ---------------------------------------------------------------------------


class ContrastiveDualEncoder(nn.Module):
    """Dual-encoder for contrastive proof-step retrieval.

    Goal encoder: CharCNN → goal embedding
    Lemma encoder: CharCNN → lemma embedding
    Score(goal, lemma) = dot(goal_emb, lemma_emb) / temperature

    Trained with InfoNCE loss: maximize similarity between (goal, correct_lemma)
    while minimizing similarity with in-batch negatives.
    """

    def __init__(self, config: ContrastiveConfig | None = None):
        super().__init__()
        self.config = config or ContrastiveConfig()

        # Separate encoders for goals and lemmas
        self.goal_encoder = CharCNNEncoder(self.config)
        self.lemma_encoder = CharCNNEncoder(self.config)

        # Pre-compute temperature inverse for efficiency
        self._t_inv: float = 1.0 / self.config.temperature

    def encode_goal(self, goal_char_ids: torch.Tensor) -> torch.Tensor:
        """Encode goal text into embedding.

        Args:
            goal_char_ids: [B, L] character IDs.

        Returns:
            [B, hidden_dim] L2-normalized goal embeddings.
        """
        return self.goal_encoder(goal_char_ids)

    def encode_lemma(self, lemma_char_ids: torch.Tensor) -> torch.Tensor:
        """Encode lemma identifier into embedding.

        Args:
            lemma_char_ids: [B, L] character IDs.

        Returns:
            [B, hidden_dim] L2-normalized lemma embeddings.
        """
        return self.lemma_encoder(lemma_char_ids)

    def forward(
        self,
        goal_char_ids: torch.Tensor,
        lemma_char_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Full forward pass returning logits and InfoNCE loss.

        Args:
            goal_char_ids: [B, L_goal] character IDs for goals.
            lemma_char_ids: [B, L_lemma] character IDs for lemmas.

        Returns:
            Dict with:
                - 'logits': [B, B] similarity logits (diagonal = positive)
                - 'loss': scalar InfoNCE loss
                - 'goal_emb': [B, D] goal embeddings
                - 'lemma_emb': [B, D] lemma embeddings
        """
        goal_emb = self.encode_goal(goal_char_ids)  # [B, D]
        lemma_emb = self.encode_lemma(lemma_char_ids)  # [B, D]

        # Compute similarity logits: [B, D] @ [D, B] → [B, B]
        logits = goal_emb @ lemma_emb.T * self._t_inv

        return {
            "logits": logits,
            "loss": self.contrastive_loss(logits),
            "goal_emb": goal_emb,
            "lemma_emb": lemma_emb,
        }

    def contrastive_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """Symmetric InfoNCE loss.

        Args:
            logits: [B, B] similarity logits.
                    Diagonal elements are positive pairs.

        Returns:
            Scalar loss (mean of goal→lemma and lemma→goal).
        """
        batch_size = logits.size(0)
        labels = torch.arange(batch_size, device=logits.device)

        # Goal → lemma direction
        loss_g2l = F.cross_entropy(logits, labels)

        # Lemma → goal direction (symmetric)
        loss_l2g = F.cross_entropy(logits.T, labels)

        return (loss_g2l + loss_l2g) / 2.0

    def score(
        self,
        goal_char_ids: torch.Tensor,
        lemma_char_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Score goal-lemma pairs (for retrieval evaluation).

        Args:
            goal_char_ids: [Q, L] query goal character IDs.
            lemma_char_ids: [C, L] candidate lemma character IDs.

        Returns:
            [Q, C] similarity scores.
        """
        goal_emb = self.encode_goal(goal_char_ids)  # [Q, D]
        lemma_emb = self.encode_lemma(lemma_char_ids)  # [C, D]
        return goal_emb @ lemma_emb.T * self._t_inv

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path | str) -> None:
        """Save model weights and config to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = {
            "hidden_dim": self.config.hidden_dim,
            "vocab_size": self.config.vocab_size,
            "max_seq_len": self.config.max_seq_len,
            "char_embed_dim": self.config.char_embed_dim,
            "cnn_filters": self.config.cnn_filters,
            "cnn_kernel_sizes": list(self.config.cnn_kernel_sizes),
            "cnn_dropout": self.config.cnn_dropout,
            "mlp_expansion": self.config.mlp_expansion,
            "pooling": self.config.pooling,
            "temperature": self.config.temperature,
        }

        save_dict = {
            "model_state_dict": self.state_dict(),
            "config_dict": config_dict,
        }
        torch.save(save_dict, str(path))

    @classmethod
    def load(cls, path: Path | str) -> "ContrastiveDualEncoder":
        """Load model from disk."""
        state = torch.load(str(path), map_location="cpu", weights_only=False)

        cd = state.get("config_dict", {})
        config = ContrastiveConfig(
            hidden_dim=cd.get("hidden_dim", 256),
            vocab_size=cd.get("vocab_size", 256),
            max_seq_len=cd.get("max_seq_len", 256),
            char_embed_dim=cd.get("char_embed_dim", 64),
            cnn_filters=cd.get("cnn_filters", 128),
            cnn_kernel_sizes=tuple(cd.get("cnn_kernel_sizes", (2, 3, 4, 5))),
            cnn_dropout=cd.get("cnn_dropout", 0.2),
            mlp_expansion=cd.get("mlp_expansion", 2),
            pooling=cd.get("pooling", "attention"),
            temperature=cd.get("temperature", 0.07),
        )

        model = cls(config)
        model.load_state_dict(state["model_state_dict"])
        return model

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def goal_encoder_params(self) -> int:
        return sum(p.numel() for p in self.goal_encoder.parameters())

    @property
    def lemma_encoder_params(self) -> int:
        return sum(p.numel() for p in self.lemma_encoder.parameters())


# ---------------------------------------------------------------------------
# Tokenizer (character-level)
# ---------------------------------------------------------------------------


class CharTokenizer:
    """Character-level tokenizer for mathematical text.

    Maps each byte (0-255) to itself, with optional padding and truncation.
    Unlike subword tokenizers, this handles Greek letters, math symbols,
    and mixed notation without a predefined vocabulary.
    """

    PAD_ID = 0  # null byte
    UNK_ID = ord("?")  # fallback for out-of-range (shouldn't happen)

    def __init__(self, max_len: int = 256):
        self.max_len = max_len

    def encode(self, text: str) -> torch.Tensor:
        """Encode a single text string into character IDs.

        Args:
            text: Input text string.

        Returns:
            [max_len] LongTensor of character IDs (padded with 0).
        """
        # Truncate to max_len
        text = text[:self.max_len]

        # Encode as UTF-8 bytes, clamp to 1-255 (0 reserved for padding)
        byte_ids = []
        for ch in text:
            b = ord(ch)
            byte_ids.append(b % 256 if b < 65536 else self.UNK_ID)

        # Pad to max_len
        ids = byte_ids[:self.max_len]
        ids += [self.PAD_ID] * (self.max_len - len(ids))

        return torch.tensor(ids, dtype=torch.long)

    def encode_batch(self, texts: list[str]) -> torch.Tensor:
        """Encode a batch of text strings (vectorized, fast).

        Args:
            texts: List of text strings.

        Returns:
            [B, max_len] LongTensor of character IDs.
        """
        import numpy as np
        batch_size = len(texts)
        ids = np.zeros((batch_size, self.max_len), dtype=np.int64)
        for i, text in enumerate(texts):
            text = text[:self.max_len]
            # Fast: use frombuffer for single-string ASCII conversion
            # then clip to 1-255 range
            if text:
                arr = np.frombuffer(text.encode("utf-8", errors="replace"),
                                    dtype=np.uint8)
                n = min(len(arr), self.max_len)
                # Shift: 0 → reserved for padding, so use 1-255
                ids[i, :n] = np.clip(arr[:n].astype(np.int64), 1, 255)
        return torch.from_numpy(ids)

    @staticmethod
    def preprocess_lemma(lemma_name: str) -> str:
        """Normalize a lemma name for encoding.

        Splits CamelCase into words, removes module prefixes,
        and normalizes underscores.

        Examples:
            'intervalIntegral.integral_mul_const' → 'interval integral integral mul const'
            'mul_comm' → 'mul comm'
            'eVariationOn.subsingleton' → 'e variation on subsingleton'
        """
        # Remove module prefix (keep last component)
        short = lemma_name.split(".")[-1]

        # Split CamelCase
        import re
        words = re.sub(r"([A-Z])", r" \1", short).strip().lower()
        # Also split on underscores
        words = words.replace("_", " ")

        # Remove extra whitespace
        words = " ".join(words.split())

        return words

    @staticmethod
    def preprocess_goal(goal_text: str) -> str:
        """Normalize a goal expression for encoding.

        Strips excessive whitespace and normalizes Unicode.
        """
        # Normalize whitespace
        import re
        text = " ".join(goal_text.split())
        # Normalize common math Unicode variants
        text = text.replace("\u2264", "<=").replace("\u2265", ">=")
        return text
