#!/usr/bin/env python3
"""Train a seed scorer — predicts whether a sub-expression is a useful building block.

Given {symbols} + candidate_expression → score 0-1.

Training data:
  Positive: Sub-expressions from known invariant structures (product, ratio,
            squared-diff) generated with random symbols.
  Negative: Random symbol pairs that don't form invariants.

This is PURE STRUCTURAL learning — no physics knowledge. The model learns
that a*b is a useful seed pattern, not that E*lambda = hc.
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim

# ════════════════════════════════════════════════════════
# Config
# ════════════════════════════════════════════════════════

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "math_self_play"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

N_POSITIVE = 6000   # positive examples (structural patterns)
N_NEGATIVE = 6000   # negative examples (random pairs)
EPOCHS = 15
BATCH_SIZE = 64
LR = 0.001
SEED = 42
DEVICE = "xpu" if torch.xpu.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

# ════════════════════════════════════════════════════════
# Tokenization
# ════════════════════════════════════════════════════════

def build_vocab():
    tokens = ["<pad>", "<sos>", "<eos>", "(", ")", "+", "-", "*", "/", "^"]
    tokens.extend(["0", "0.5", "1", "2", "-1"])
    for c in "abcdefghijklmnopqrstuvwxyz":
        tokens.append(c)
    for c in "abcdefghijklmnopqrstuvwxyz":
        for d in "0123":
            tokens.append(f"{c}{d}")
    for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        tokens.append(c)
    token_map = {t: i for i, t in enumerate(tokens)}
    inv_map = {i: t for t, i in token_map.items()}
    return token_map, inv_map


def tokenize_expr(expr: str, token_map: dict, max_len: int = 32) -> list[int]:
    result = [token_map["<sos>"]]
    i = 0
    while i < len(expr) and len(result) < max_len - 1:
        matched = False
        for length in range(min(4, len(expr) - i), 0, -1):
            token = expr[i:i+length]
            if token in token_map:
                result.append(token_map[token])
                i += length
                matched = True
                break
        if not matched:
            result.append(token_map.get(expr[i], 0))
            i += 1
    result.append(token_map["<eos>"])
    while len(result) < max_len:
        result.append(token_map["<pad>"])
    return result[:max_len]


def tokenize_symbols(symbols: list[str], token_map: dict, max_len: int = 16) -> list[int]:
    result = [token_map["<sos>"]]
    for s in symbols:
        result.append(token_map.get(s, 0))
    result.append(token_map["<eos>"])
    while len(result) < max_len:
        result.append(token_map["<pad>"])
    return result[:max_len]


# ════════════════════════════════════════════════════════
# Training data
# ════════════════════════════════════════════════════════

def generate_positive_examples(token_map: dict, rng: random.Random, n: int) -> list[dict]:
    """Generate positive examples from structural patterns."""
    all_symbols = [t for t in token_map.keys()
                   if t not in {"<pad>", "<sos>", "<eos>", "(", ")", "+", "-", "*", "/", "^"}
                   and not t.replace('.', '').replace('-', '').isdigit()]

    data = []
    seen = set()

    while len(data) < n:
        n_syms = rng.randint(2, 3)
        symbols = rng.sample(all_symbols, n_syms)

        # Generate all structural sub-expressions
        candidates = []
        for s in symbols:
            candidates.append(s)
            candidates.append(f"{s}^2")

        for i, a in enumerate(symbols):
            for b in symbols[i+1:]:
                candidates.append(f"{a}*{b}")
                candidates.append(f"{a}/{b}")
                candidates.append(f"{a}+{b}")
                candidates.append(f"{a}-{b}")
                candidates.append(f"{b}-{a}")

        for expr in candidates:
            key = (tuple(sorted(symbols)), expr)
            if key in seen:
                continue
            seen.add(key)

            data.append({
                "symbols": symbols,
                "expression": expr,
                "label": 1.0,
                "src": tokenize_symbols(symbols, token_map),
                "tgt": tokenize_expr(expr, token_map),
            })

            if len(data) >= n:
                break

    return data


def generate_negative_examples(token_map: dict, rng: random.Random, n: int) -> list[dict]:
    """Generate negative examples — random symbol pairs unlikely to form invariants."""
    all_symbols = [t for t in token_map.keys()
                   if t not in {"<pad>", "<sos>", "<eos>", "(", ")", "+", "-", "*", "/", "^"}
                   and not t.replace('.', '').replace('-', '').isdigit()]

    data = []
    seen = set()

    while len(data) < n:
        n_syms = rng.randint(2, 4)
        symbols = rng.sample(all_symbols, n_syms)
        # Pick a random pair and random operator
        a, b = rng.sample(symbols, 2)
        op = rng.choice(["*", "/", "+", "-"])
        expr = f"{a}{op}{b}"

        key = (tuple(sorted(symbols)), expr)
        if key in seen:
            continue
        seen.add(key)

        data.append({
            "symbols": symbols,
            "expression": expr,
            "label": 0.0,
            "src": tokenize_symbols(symbols, token_map),
            "tgt": tokenize_expr(expr, token_map),
        })

    return data


# ════════════════════════════════════════════════════════
# Model
# ════════════════════════════════════════════════════════

class SeedScorer(nn.Module):
    """Scores a sub-expression candidate for relevance.

    Takes symbol tokens + expression tokens, outputs a scalar 0-1.
    """

    def __init__(self, vocab_size: int, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, max_seq_len: int = 64):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256,
            dropout=0.1, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, src_tokens: torch.Tensor, tgt_tokens: torch.Tensor) -> torch.Tensor:
        """Forward pass: concatenate src and tgt, encode, project to score."""
        batch_size = src_tokens.size(0)

        # Concatenate symbol tokens and expression tokens
        combined = torch.cat([src_tokens, tgt_tokens], dim=1)

        emb = self.embedding(combined) + self.pos_encoding[:, :combined.size(1), :]
        encoded = self.encoder(emb)

        # Pool: mean over non-padding tokens
        mask = (combined != 0).float().unsqueeze(-1)  # 0 = pad
        pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.output_proj(pooled).squeeze(-1)


# ════════════════════════════════════════════════════════
# Training
# ════════════════════════════════════════════════════════

def train():
    token_map, inv_map = build_vocab()
    vocab_size = len(token_map)
    rng = random.Random(SEED)

    print(f"Vocabulary: {vocab_size} tokens")
    print(f"Generating {N_POSITIVE} positive + {N_NEGATIVE} negative examples...")

    pos = generate_positive_examples(token_map, rng, N_POSITIVE)
    neg = generate_negative_examples(token_map, rng, N_NEGATIVE)
    data = pos + neg
    rng.shuffle(data)
    print(f"Total: {len(data)} examples ({sum(1 for d in data if d['label']>0.5)} pos, {sum(1 for d in data if d['label']<0.5)} neg)")

    # Show samples
    for d in data[:3]:
        label = "POS" if d['label'] > 0.5 else "NEG"
        print(f"  {label} {' '.join(d['symbols']):20s} → {d['expression']}")

    split = int(0.85 * len(data))
    train_data, val_data = data[:split], data[split:]

    model = SeedScorer(vocab_size=vocab_size, d_model=64, nhead=4, num_layers=2)
    model.to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCELoss()

    print(f"\nTraining {EPOCHS} epochs...")
    best_val = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        rng.shuffle(train_data)

        for i in range(0, len(train_data), BATCH_SIZE):
            batch = train_data[i:i+BATCH_SIZE]
            max_src = max(len(d["src"]) for d in batch)
            max_tgt = max(len(d["tgt"]) for d in batch)

            src = torch.zeros(len(batch), max_src, dtype=torch.long, device=DEVICE)
            tgt = torch.zeros(len(batch), max_tgt, dtype=torch.long, device=DEVICE)
            labels = torch.zeros(len(batch), device=DEVICE)

            for j, d in enumerate(batch):
                src[j, :len(d["src"])] = torch.tensor(d["src"])
                tgt[j, :len(d["tgt"])] = torch.tensor(d["tgt"])
                labels[j] = d["label"]

            optimizer.zero_grad()
            preds = model(src, tgt)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for i in range(0, len(val_data), BATCH_SIZE):
                batch = val_data[i:i+BATCH_SIZE]
                max_src = max(len(d["src"]) for d in batch)
                max_tgt = max(len(d["tgt"]) for d in batch)

                src = torch.zeros(len(batch), max_src, dtype=torch.long, device=DEVICE)
                tgt = torch.zeros(len(batch), max_tgt, dtype=torch.long, device=DEVICE)
                labels = torch.zeros(len(batch), device=DEVICE)

                for j, d in enumerate(batch):
                    src[j, :len(d["src"])] = torch.tensor(d["src"])
                    tgt[j, :len(d["tgt"])] = torch.tensor(d["tgt"])
                    labels[j] = d["label"]

                preds = model(src, tgt)
                loss = criterion(preds, labels)
                val_loss += loss.item()
                correct += ((preds > 0.5) == (labels > 0.5)).sum().item()

        avg_train = total_loss / max(1, len(train_data) / BATCH_SIZE)
        avg_val = val_loss / max(1, len(val_data) / BATCH_SIZE)
        val_acc = correct / len(val_data)

        print(f"  Epoch {epoch+1:2d}/{EPOCHS}  train_loss={avg_train:.4f}  val_loss={avg_val:.4f}  val_acc={val_acc:.2%}")

        if avg_val < best_val:
            best_val = avg_val

    # Save
    ckpt_path = CHECKPOINT_DIR / "seed_scorer.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "token_map": token_map,
        "vocab_size": vocab_size,
        "config": {"d_model": 64, "nhead": 4, "num_layers": 2, "max_seq_len": 64},
    }, ckpt_path)
    print(f"\nSaved to {ckpt_path}")

    # Quick test
    print("\nQuick test:")
    test_pairs = [
        (["E", "lambda"], "E*lambda", 1.0),   # should score high
        (["E", "lambda"], "E+lambda", 0.7),    # should score medium
        (["B", "T"], "B*T", 0.0),              # should score low (random)
        (["c", "t", "x"], "c*t", 1.0),         # should score high
        (["c", "t", "x"], "c/t", 0.8),         # should score medium
        (["x", "d"], "x^2", 0.5),              # should score medium
    ]
    for syms, expr, expected in test_pairs:
        src = torch.tensor([tokenize_symbols(syms, token_map)], device=DEVICE)
        tgt = torch.tensor([tokenize_expr(expr, token_map)], device=DEVICE)
        with torch.no_grad():
            score = model(src, tgt).item()
        status = "✓" if (score > 0.5) == (expected > 0.5) else "✗"
        print(f"  {status} {' '.join(syms):20s} {expr:15s} → {score:.4f} (expected ~{expected})")


if __name__ == "__main__":
    train()
