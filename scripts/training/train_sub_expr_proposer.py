#!/usr/bin/env python3
"""Retrain cross-symbol model on structural sub-expression prediction.

The model learns to propose useful sub-expressions from symbol sets.
Given {X, Y}, it should propose: X*Y, X/Y, X^2, Y^2, X+Y, X-Y.
Given {X, Y, Z}, it should also propose: X*Z, Y*Z.

This is pure structural pattern learning — NOT physics-specific.
The training uses random symbols from a pool, not physics quantities.
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim

# ════════════════════════════════════════════════════════════
# Config
# ════════════════════════════════════════════════════════════

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "math_self_play"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

SYMBOL_POOL_SIZE = 130
N_SYMBOLS_PER_EXAMPLE = 4  # train on up to 4 symbols
N_EXAMPLES = 50000
EPOCHS = 15
BATCH_SIZE = 64
LR = 0.001
SEED = 42
DEVICE = "cpu"

# ════════════════════════════════════════════════════════════
# Tokenization
# ════════════════════════════════════════════════════════════

def build_vocab():
    """Build token map for expressions."""
    tokens = ["<pad>", "<sos>", "<eos>", "(", ")", "+", "-", "*", "/", "^"]
    # Numbers
    tokens.extend(["0", "0.5", "1", "2", "-1"])
    # Single-char symbols
    for c in "abcdefghijklmnopqrstuvwxyz":
        tokens.append(c)
    # Multi-char symbols
    for c in "abcdefghijklmnopqrstuvwxyz":
        for d in "0123":
            tokens.append(f"{c}{d}")
    # Uppercase
    for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        tokens.append(c)
    
    token_map = {t: i for i, t in enumerate(tokens)}
    inv_map = {i: t for t, i in token_map.items()}
    return token_map, inv_map


def tokenize_expression(expr: str, token_map: dict[str, int], max_len: int = 32) -> list[int]:
    """Tokenize an expression string."""
    result = [token_map["<sos>"]]
    i = 0
    while i < len(expr) and len(result) < max_len - 1:
        # Try longest match first
        matched = False
        for length in range(min(4, len(expr) - i), 0, -1):
            token = expr[i:i+length]
            if token in token_map:
                result.append(token_map[token])
                i += length
                matched = True
                break
        if not matched:
            # Single char fallback
            if expr[i] in token_map:
                result.append(token_map[expr[i]])
            i += 1
    result.append(token_map["<eos>"])
    return result


def tokenize_symbols(symbols: list[str], token_map: dict[str, int], max_len: int = 16) -> list[int]:
    """Tokenize a list of symbol names as input."""
    result = [token_map["<sos>"]]
    for s in symbols:
        if s in token_map:
            result.append(token_map[s])
    result.append(token_map["<eos>"])
    # Pad
    while len(result) < max_len:
        result.append(token_map["<pad>"])
    return result[:max_len]


# ════════════════════════════════════════════════════════════
# Training data generation
# ════════════════════════════════════════════════════════════

OPS = ["*", "/", "+", "-", "^"]

def generate_sub_expressions(symbols: list[str]) -> list[str]:
    """Generate all useful sub-expressions for a symbol set."""
    exprs = []
    # Single symbols and squares
    for s in symbols:
        exprs.append(s)
        exprs.append(f"{s}^2")
    
    # Pairs: products, ratios
    for i, a in enumerate(symbols):
        for b in symbols[i+1:]:
            exprs.append(f"{a}*{b}")
            exprs.append(f"{b}*{a}")
            exprs.append(f"{a}/{b}")
            exprs.append(f"{b}/{a}")
            exprs.append(f"{a}+{b}")
            exprs.append(f"{a}-{b}")
            exprs.append(f"{b}-{a}")
    
    return exprs


def generate_training_data(n_examples: int, token_map: dict, rng: random.Random) -> list[dict]:
    """Generate (symbols → sub_expression) pairs."""
    # Symbol pool
    all_symbols = [t for t in token_map.keys() 
                   if t not in {"<pad>", "<sos>", "<eos>", "(", ")", "+", "-", "*", "/", "^"}
                   and not t.replace('.', '').replace('-', '').isdigit()]
    
    data = []
    seen = set()
    
    while len(data) < n_examples:
        n_syms = rng.randint(2, N_SYMBOLS_PER_EXAMPLE)
        symbols = rng.sample(all_symbols, n_syms)
        sub_exprs = generate_sub_expressions(symbols)
        
        for expr in sub_exprs:
            key = (tuple(sorted(symbols)), expr)
            if key in seen:
                continue
            seen.add(key)
            
            src_tokens = tokenize_symbols(symbols, token_map)
            tgt_tokens = tokenize_expression(expr, token_map)
            
            data.append({
                "symbols": symbols,
                "expression": expr,
                "src_tokens": src_tokens,
                "tgt_tokens": tgt_tokens,
            })
            
            if len(data) >= n_examples:
                break
    
    rng.shuffle(data)
    return data


# ════════════════════════════════════════════════════════════
# Model
# ════════════════════════════════════════════════════════════

class SubExpressionProposer(nn.Module):
    """Transformer that maps [symbols] → [sub_expression]."""
    
    def __init__(self, vocab_size: int, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 4, max_seq_len: int = 64):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=512,
            dropout=0.1, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=512,
            dropout=0.1, batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.output_proj = nn.Linear(d_model, vocab_size)
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor):
        """Training forward pass."""
        src_emb = self.embedding(src) + self.pos_encoding[:, :src.size(1), :]
        tgt_emb = self.embedding(tgt) + self.pos_encoding[:, :tgt.size(1), :]
        
        # Causal mask for decoder
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        
        memory = self.encoder(src_emb)
        output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        return self.output_proj(output)
    
    def generate(self, src: torch.Tensor, max_len: int = 32, temperature: float = 1.0,
                 top_k: int = 10) -> list[list[int]]:
        """Generate expression tokens from symbols."""
        token_map = getattr(self, '_token_map', None)
        if token_map is None:
            return []
        
        self.eval()
        with torch.no_grad():
            src_emb = self.embedding(src) + self.pos_encoding[:, :src.size(1), :]
            memory = self.encoder(src_emb)
            
            # Start with <sos>
            tgt_token = token_map.get("<sos>", 1)
            generated = [[tgt_token] for _ in range(src.size(0))]
            
            for _ in range(max_len - 1):
                tgt = torch.tensor([g[-1:] for g in generated], device=src.device, dtype=torch.long)
                tgt_emb = self.embedding(tgt) + self.pos_encoding[:, :tgt.size(1), :]
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(src.device)
                
                output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
                logits = self.output_proj(output[:, -1, :]) / temperature
                
                # Top-k sampling
                if top_k > 0:
                    top_k_vals, top_k_idx = torch.topk(logits, min(top_k, logits.size(-1)))
                    probs = torch.zeros_like(logits)
                    probs.scatter_(-1, top_k_idx, torch.softmax(top_k_vals, dim=-1))
                else:
                    probs = torch.softmax(logits, dim=-1)
                
                next_tokens = torch.multinomial(probs, 1).squeeze(-1).tolist()
                for i, t in enumerate(next_tokens):
                    generated[i].append(t)
                
                # Stop if all sequences have <eos>
                eos = token_map.get("<eos>", 2)
                if all(eos in g for g in generated):
                    break
            
            return generated


# ════════════════════════════════════════════════════════════
# Training
# ════════════════════════════════════════════════════════════

def train():
    token_map, inv_map = build_vocab()
    vocab_size = len(token_map)
    rng = random.Random(SEED)
    
    print(f"Vocabulary size: {vocab_size}")
    print(f"Generating {N_EXAMPLES} training examples...")
    
    data = generate_training_data(N_EXAMPLES, token_map, rng)
    print(f"Generated {len(data)} examples")
    
    # Show some examples
    for i in range(5):
        d = data[i]
        print(f"  {'+'.join(d['symbols']):20s} → {d['expression']}")
    
    # Train/val split
    split = int(0.9 * len(data))
    train_data = data[:split]
    val_data = data[split:]
    
    model = SubExpressionProposer(
        vocab_size=vocab_size, d_model=64, nhead=4, num_layers=3, max_seq_len=64,
    )
    model._token_map = token_map
    model.to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=token_map["<pad>"])
    
    print(f"\nTraining for {EPOCHS} epochs...")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        rng.shuffle(train_data)
        
        for i in range(0, len(train_data), BATCH_SIZE):
            batch = train_data[i:i+BATCH_SIZE]
            
            # Pad sequences
            max_src_len = max(len(d["src_tokens"]) for d in batch)
            max_tgt_len = max(len(d["tgt_tokens"]) for d in batch)
            
            src = torch.zeros(len(batch), max_src_len, dtype=torch.long, device=DEVICE)
            tgt_in = torch.zeros(len(batch), max_tgt_len, dtype=torch.long, device=DEVICE)
            tgt_out = torch.zeros(len(batch), max_tgt_len, dtype=torch.long, device=DEVICE)
            
            for j, d in enumerate(batch):
                src[j, :len(d["src_tokens"])] = torch.tensor(d["src_tokens"])
                tgt_in[j, :len(d["tgt_tokens"])] = torch.tensor(d["tgt_tokens"])
                # Output is shifted right (predict next token)
                tgt_out[j, :len(d["tgt_tokens"])-1] = torch.tensor(d["tgt_tokens"][1:])
                tgt_out[j, len(d["tgt_tokens"])-1] = token_map["<eos>"]
            
            optimizer.zero_grad()
            output = model(src, tgt_in)
            loss = criterion(output.reshape(-1, vocab_size), tgt_out.reshape(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / (len(train_data) / BATCH_SIZE)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i in range(0, len(val_data), BATCH_SIZE):
                batch = val_data[i:i+BATCH_SIZE]
                max_src_len = max(len(d["src_tokens"]) for d in batch)
                max_tgt_len = max(len(d["tgt_tokens"]) for d in batch)
                
                src = torch.zeros(len(batch), max_src_len, dtype=torch.long, device=DEVICE)
                tgt_in = torch.zeros(len(batch), max_tgt_len, dtype=torch.long, device=DEVICE)
                tgt_out = torch.zeros(len(batch), max_tgt_len, dtype=torch.long, device=DEVICE)
                
                for j, d in enumerate(batch):
                    src[j, :len(d["src_tokens"])] = torch.tensor(d["src_tokens"])
                    tgt_in[j, :len(d["tgt_tokens"])] = torch.tensor(d["tgt_tokens"])
                    tgt_out[j, :len(d["tgt_tokens"])-1] = torch.tensor(d["tgt_tokens"][1:])
                    tgt_out[j, len(d["tgt_tokens"])-1] = token_map["<eos>"]
                
                output = model(src, tgt_in)
                loss = criterion(output.reshape(-1, vocab_size), tgt_out.reshape(-1))
                val_loss += loss.item()
        
        avg_val = val_loss / max(1, len(val_data) / BATCH_SIZE)
        print(f"  Epoch {epoch+1:2d}/{EPOCHS}  train_loss={avg_loss:.4f}  val_loss={avg_val:.4f}")
    
    # Save checkpoint
    ckpt_path = CHECKPOINT_DIR / "sub_expr_proposer.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "token_map": token_map,
        "inv_map": inv_map,
        "vocab_size": vocab_size,
        "config": {"d_model": 128, "nhead": 4, "num_layers": 4, "max_seq_len": 64},
    }, ckpt_path)
    print(f"\nSaved checkpoint to {ckpt_path}")
    
    # Quick test: what does the model propose?
    print("\nQuick test:")
    test_sets = [
        ["a", "b"],
        ["x", "y", "z"],
        ["p", "q"],
        ["E", "lambda"],
        ["c", "t", "x"],
    ]
    for syms in test_sets:
        src = tokenize_symbols(syms, token_map)
        src_tensor = torch.tensor([src], dtype=torch.long, device=DEVICE)
        seqs = model.generate(src_tensor, max_len=16, temperature=0.8, top_k=5)
        
        proposals = []
        for seq in seqs:
            tokens = []
            for tid in seq:
                if tid in (token_map.get("<pad>", 0), token_map.get("<sos>", 1)):
                    continue
                if tid == token_map.get("<eos>", 2):
                    break
                tokens.append(inv_map.get(tid, "?"))
            expr = "".join(tokens)
            if expr:
                proposals.append(expr)
        
        print(f"  {'+'.join(syms):20s} → {', '.join(proposals[:6])}")


if __name__ == "__main__":
    train()
