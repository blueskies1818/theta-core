"""
Goal-Only Encoder — Lean, vocabulary-based architecture.

Key changes from v1:
  - Builds a proper vocabulary from training goals (most common math tokens)
  - No hash collisions — each learned token gets its own embedding
  - Smaller total params (~500K) via dim reduction
  - Better initialization strategy to prevent embedding collapse

Architecture:
  Goal text → tokenize → VocabEmbedding(4000×128=512K)
           → 1×TransformerEncoder(d=128, 4h, ffn=192)
           → mean pool → Linear(128→256) → L2Norm
"""
from __future__ import annotations

import math
import re
import json
from collections import Counter
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
_TOKENIZE_RE = re.compile(r"[^\s,:(){}\[\]]+")

# Math-related single chars that carry meaning (kept even though length 1)
_MATH_SINGLE_CHARS = set("+-*/^=<>≤≥∀∃∈⊆⊂∪∩∑∏∫∂√∞→←⇒⇔λμπσθαβγδε")

# Common single-char variable names in Lean (kept)
_LEAN_SINGLE_CHARS = set("αβγδεζηθικλμνξπρστυφχψω")


def tokenize_goal(goal_text: str, max_tokens: int = 64) -> list[str]:
    """Tokenize goal text, keeping only meaningful multi-char tokens
    and semantically significant single-char math symbols."""
    tokens = []
    for tok in _TOKENIZE_RE.findall(goal_text):
        tok = tok.strip().lower()
        # Skip empty, overly long, or pure whitespace
        if not tok or len(tok) > 50:
            continue
        # Keep multi-char tokens, and single-char symbols/variables
        if len(tok) >= 2:
            tokens.append(tok)
        elif tok in _MATH_SINGLE_CHARS:
            tokens.append(tok)
        elif tok in _LEAN_SINGLE_CHARS:
            tokens.append(tok)
        # Skip other single characters (variable names like 'x', 'f', 'a')
        if len(tokens) >= max_tokens:
            break
    return tokens


# ---------------------------------------------------------------------------
# Vocabulary builder
# ---------------------------------------------------------------------------


def build_vocab(goals: list[str], vocab_size: int = 4000) -> dict[str, int]:
    """Build vocabulary from most common tokens in goals."""
    counter = Counter()
    for goal in goals:
        for tok in tokenize_goal(goal):
            counter[tok] += 1
    # Keep top vocab_size-2 (reserve 0=PAD, 1=UNK)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for tok, _ in counter.most_common(vocab_size - 2):
        vocab[tok] = len(vocab)
    return vocab


VOCAB_CACHE: dict[int, dict[str, int]] = {}  # hash(goals) → vocab


# ---------------------------------------------------------------------------
# Token embedding with vocabulary
# ---------------------------------------------------------------------------


class VocabEmbedding(nn.Module):
    """Learned embedding table for math tokens."""

    def __init__(self, vocab: dict[str, int], embed_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.vocab = vocab
        self.embed_dim = embed_dim
        self.vocab_size = len(vocab)
        self.unk_idx = vocab.get("<UNK>", 1)
        self.embedding = nn.Embedding(self.vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        # Kaiming init for better gradient flow
        nn.init.kaiming_normal_(self.embedding.weight, mode="fan_out", nonlinearity="relu")
        # Zero out padding
        with torch.no_grad():
            self.embedding.weight[0].zero_()

    def forward(self, tokens: list[str], device: torch.device) -> torch.Tensor:
        if not tokens:
            return torch.zeros(0, self.embed_dim, device=device)
        indices = torch.tensor(
            [self.vocab.get(t, self.unk_idx) for t in tokens],
            dtype=torch.long,
            device=device,
        )
        return self.dropout(self.embedding(indices))


# ---------------------------------------------------------------------------
# Transformer encoder layer (pre-norm)
# ---------------------------------------------------------------------------


class TransformerLayer(nn.Module):
    def __init__(self, dim: int = 128, num_heads: int = 4, ffn_dim: int = 192, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Goal-Only Encoder
# ---------------------------------------------------------------------------


class GoalOnlyEncoder(nn.Module):
    """Encodes math goals into 256-dim embeddings. Vocabulary-based."""

    def __init__(
        self,
        vocab: dict[str, int],
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 1,
        num_heads: int = 4,
        ffn_dim: int = 192,
        max_tokens: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.max_tokens = max_tokens
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ffn_dim = ffn_dim
        self.vocab = vocab

        # Token embedding
        self.token_embed = VocabEmbedding(vocab, embed_dim, dropout)

        # Positional encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, max_tokens, embed_dim))
        nn.init.normal_(self.pos_embed, std=0.02)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(embed_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

        # Output
        self.out_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2 and "embedding" not in name:
                nn.init.xavier_uniform_(param, gain=0.5)  # reduced gain to prevent collapse
            elif "bias" in name:
                nn.init.zeros_(param)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _encode_text(self, goal_text: str, device: torch.device) -> torch.Tensor:
        tokens = tokenize_goal(goal_text, self.max_tokens)
        if not tokens:
            return torch.zeros(1, self.hidden_dim, device=device)

        tok_emb = self.token_embed(tokens, device)  # [T, E]
        T = min(tok_emb.size(0), self.max_tokens)
        tok_emb = tok_emb[:T]

        if self.num_layers > 0:
            # Transformer path
            pos = self.pos_embed[:, :T, :]
            x = tok_emb.unsqueeze(0) + pos  # [1, T, E]
            for layer in self.layers:
                x = layer(x)
            x = x.mean(dim=1)  # [1, E]
        else:
            # Bag-of-Words path (no transformer, just mean pool)
            # More stable, prevents embedding collapse
            x = tok_emb.mean(dim=0, keepdim=True)  # [1, E]

        x = self.out_proj(x)  # [1, H]
        return F.normalize(x, dim=-1)

    def forward(self, goals: list[str], device: Optional[torch.device] = None) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device
        embs = [self._encode_text(g, device) for g in goals]
        return torch.cat(embs, dim=0)

    def encode_single(self, goal_text: str, device: Optional[torch.device] = None) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device
        return self._encode_text(goal_text, device).squeeze(0)

    def save(self, path: str) -> None:
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        state = {
            "model_state_dict": self.state_dict(),
            "vocab": self.vocab,
            "config": {
                "embed_dim": self.embed_dim,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "ffn_dim": self.ffn_dim,
                "max_tokens": self.max_tokens,
                "dropout": 0.1,
                "vocab_size": len(self.vocab),
            },
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path: str) -> "GoalOnlyEncoder":
        state = torch.load(path, map_location="cpu", weights_only=False)
        cfg = state["config"]
        vocab = state["vocab"]
        model = cls(
            vocab=vocab,
            embed_dim=cfg["embed_dim"],
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
            num_heads=cfg["num_heads"],
            ffn_dim=cfg.get("ffn_dim", 192),
            max_tokens=cfg.get("max_tokens", 64),
            dropout=cfg.get("dropout", 0.1),
        )
        model.load_state_dict(state["model_state_dict"])
        return model


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_goals = ["a + b = b + a", "∀ x : ℝ, x + 0 = x", "lim_{n→∞} a_n = L"]
    vocab = build_vocab(test_goals, vocab_size=100)
    print(f"Vocab size: {len(vocab)}")
    model = GoalOnlyEncoder(vocab=vocab)
    print(f"Params: {model.count_parameters():,}")
    with torch.no_grad():
        emb = model(test_goals)
    print(f"Output: {emb.shape}, norms={emb.norm(dim=-1).tolist()}")
