"""Cross-symbol training with Chain-of-Thought (CoT) decoding.

Two-stage generation:
  Stage 1: [symbols] → pattern_type  (which structural pattern?)
  Stage 2: [symbols, pattern_type] → expression  (instantiate pattern)

This lets the model compose known abstract patterns (a²-b², (a+b)*(c+d))
with novel symbol assignments, discovering expressions it was never
explicitly trained to produce.
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

from src.math import (
    MathInvariantGenerator,
    _INVARIANT_BUILDERS,
    reset_symbol_pool,
    MathInvariant,
)

# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "math_self_play"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

VARIANTS_PER_PATTERN = 500
TOTAL_EXAMPLES = 100_000
EPOCHS = 30
BATCH_SIZE = 64
LR = 0.001
SEED = 42
DEVICE = "cpu"

# ── CoT config ──
COT_PATTERN_TOKEN = "<pat>"       # marker: pattern type follows
COT_EXPR_TOKEN = "<expr>"         # marker: expression follows
COT_NUM_PATTERN_SAMPLES = 8       # how many pattern types to sample in Stage 1


# ═══════════════════════════════════════════════════════════════════════════
# Cross-symbol data generation (CoT-aware)
# ═══════════════════════════════════════════════════════════════════════════

def generate_cross_symbol_dataset(
    rng: random.Random,
    n_total: int = TOTAL_EXAMPLES,
    variants_per_pattern: int = VARIANTS_PER_PATTERN,
) -> list[dict]:
    """Generate training data with pattern-type metadata for CoT."""
    data: list[dict] = []
    all_builders = []
    for type_name, builders in _INVARIANT_BUILDERS.items():
        for b in builders:
            all_builders.append((type_name, b))

    n_patterns = min(len(all_builders), n_total // variants_per_pattern)
    selected = rng.sample(all_builders, n_patterns)

    for pat_idx, (inv_type, builder) in enumerate(selected):
        for var_idx in range(variants_per_pattern):
            seed = pat_idx * variants_per_pattern + var_idx
            reset_symbol_pool(seed=seed)
            inv = builder()

            if not inv.expression or len(inv.variables) == 0:
                continue

            data.append({
                "expression": inv.expression,
                "identity_type": inv.identity_type,
                "variables": inv.variables,
                "complexity": inv.complexity,
                "pattern_id": pat_idx,
                "variant_id": var_idx,
            })

    rng.shuffle(data)
    return data[:n_total]


# ═══════════════════════════════════════════════════════════════════════════
# Template generator (cross-symbol aware, CoT-enabled)
# ═══════════════════════════════════════════════════════════════════════════

class CrossSymbolTemplateGenerator(nn.Module):
    """Template generator with two-stage CoT decoding.

    Trained to map: [symbols] → [pattern_type | expression]
    At inference, Stage 1 samples pattern types, Stage 2 generates expressions.
    """

    def __init__(
        self,
        vocab_size: int = 128,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        max_seq_len: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256,
            dropout=0.1, batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, vocab_size)

        self.pad_token = 0
        self.sos_token = 1
        self.eos_token = 2

    def forward(
        self, src: torch.Tensor, tgt: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        src_emb = self.embedding(src) + self.pos_encoding[:, :src.size(1), :]
        tgt_emb = self.embedding(tgt) + self.pos_encoding[:, :tgt.size(1), :]
        out = self.decoder(tgt_emb, src_emb, tgt_mask=tgt_mask)
        return self.output_proj(out)

    def generate(self, src: torch.Tensor, max_len: int = 32) -> list[list[int]]:
        self.eval()
        B = src.size(0)
        src_emb = self.embedding(src) + self.pos_encoding[:, :src.size(1), :]
        generated = torch.full((B, 1), self.sos_token, dtype=torch.long, device=src.device)
        with torch.no_grad():
            for _ in range(max_len):
                tgt_emb = self.embedding(generated) + self.pos_encoding[:, :generated.size(1), :]
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                    generated.size(1), device=src.device)
                out = self.decoder(tgt_emb, src_emb, tgt_mask=tgt_mask)
                logits = self.output_proj(out[:, -1:, :])
                next_token = logits.argmax(dim=-1)
                generated = torch.cat([generated, next_token], dim=1)
                if (next_token == self.eos_token).all():
                    break
        return generated.tolist()

    def generate_sampled(
        self, src: torch.Tensor, max_len: int = 32,
        temperature: float = 1.0, top_k: int = 10,
        num_samples: int = 8,
    ) -> list[list[int]]:
        self.eval()
        B = src.size(0)
        src_emb = self.embedding(src) + self.pos_encoding[:, :src.size(1), :]
        all_samples: list[list[int]] = []
        seen: set[str] = set()
        for _ in range(num_samples * 2):
            if len(all_samples) >= num_samples:
                break
            generated = torch.full((B, 1), self.sos_token, dtype=torch.long, device=src.device)
            with torch.no_grad():
                for _ in range(max_len):
                    tgt_emb = self.embedding(generated) + self.pos_encoding[:, :generated.size(1), :]
                    tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                        generated.size(1), device=src.device)
                    out = self.decoder(tgt_emb, src_emb, tgt_mask=tgt_mask)
                    logits = self.output_proj(out[:, -1:, :]) / temperature
                    top_k_vals, top_k_idx = torch.topk(logits, k=min(top_k, logits.size(-1)), dim=-1)
                    probs = torch.softmax(top_k_vals, dim=-1)
                    sampled = torch.multinomial(probs.squeeze(0), 1)
                    next_token = top_k_idx[:, :, sampled].squeeze(-1)
                    generated = torch.cat([generated, next_token.view(B, 1)], dim=1)
                    if (next_token == self.eos_token).all():
                        break
            token_key = tuple(generated[0].tolist())
            if token_key not in seen:
                seen.add(token_key)
                all_samples.append(generated[0].tolist())
        return all_samples

    # ── Two-stage CoT generation ────────────────────────────────────

    def generate_pattern_types(
        self, src: torch.Tensor, num_samples: int = 8,
        temperature: float = 1.5, top_k: int = 10,
    ) -> list[int]:
        """Stage 1: sample pattern-type tokens from the symbol input.

        Generates one token after SOS — this is the pattern type.
        Returns list of token IDs (one per sample, deduplicated).
        """
        self.eval()
        B = src.size(0)
        src_emb = self.embedding(src) + self.pos_encoding[:, :src.size(1), :]

        pattern_ids: list[int] = []
        seen: set[int] = set()

        for _ in range(num_samples * 3):
            if len(pattern_ids) >= num_samples:
                break
            # Start with SOS, generate just one token
            generated = torch.full((B, 1), self.sos_token, dtype=torch.long, device=src.device)
            with torch.no_grad():
                tgt_emb = self.embedding(generated) + self.pos_encoding[:, :generated.size(1), :]
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                    generated.size(1), device=src.device)
                out = self.decoder(tgt_emb, src_emb, tgt_mask=tgt_mask)
                logits = self.output_proj(out[:, -1:, :]) / temperature
                top_k_vals, top_k_idx = torch.topk(logits, k=min(top_k, logits.size(-1)), dim=-1)
                probs = torch.softmax(top_k_vals, dim=-1)
                sampled = torch.multinomial(probs.squeeze(0), 1)
                token_id = top_k_idx[0, 0, sampled].item()
                if token_id not in seen:
                    seen.add(token_id)
                    pattern_ids.append(token_id)

        return pattern_ids

    def generate_with_pattern(
        self, src: torch.Tensor, pattern_token: int,
        max_len: int = 32, temperature: float = 1.0, top_k: int = 10,
        num_samples: int = 4,
    ) -> list[list[int]]:
        """Stage 2: generate expression given a specific pattern type.

        The target starts with [SOS, pattern_token] and continues
        autoregressively to generate the expression.
        """
        self.eval()
        B = src.size(0)
        src_emb = self.embedding(src) + self.pos_encoding[:, :src.size(1), :]

        all_samples: list[list[int]] = []
        seen: set[str] = set()

        for _ in range(num_samples * 2):
            if len(all_samples) >= num_samples:
                break
            # Start with SOS + pattern_token
            generated = torch.tensor(
                [[self.sos_token, pattern_token]], dtype=torch.long, device=src.device)
            with torch.no_grad():
                for _ in range(max_len):
                    tgt_emb = self.embedding(generated) + self.pos_encoding[:, :generated.size(1), :]
                    tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                        generated.size(1), device=src.device)
                    out = self.decoder(tgt_emb, src_emb, tgt_mask=tgt_mask)
                    logits = self.output_proj(out[:, -1:, :]) / temperature
                    top_k_vals, top_k_idx = torch.topk(logits, k=min(top_k, logits.size(-1)), dim=-1)
                    probs = torch.softmax(top_k_vals, dim=-1)
                    sampled = torch.multinomial(probs.squeeze(0), 1)
                    next_token = top_k_idx[:, :, sampled].squeeze(-1)
                    generated = torch.cat([generated, next_token.view(B, 1)], dim=1)
                    if (next_token == self.eos_token).all():
                        break
            token_key = tuple(generated[0].tolist())
            if token_key not in seen:
                seen.add(token_key)
                all_samples.append(generated[0].tolist())

        return all_samples


# ═══════════════════════════════════════════════════════════════════════════
# Tokenization (CoT-aware)
# ═══════════════════════════════════════════════════════════════════════════

def build_tokenizer(data: list[dict]) -> tuple[dict[str, int], dict[int, str]]:
    """Build token vocab including pattern-type tokens for CoT."""
    tokens = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}

    # Collect all pattern types as tokens
    for item in data:
        ptype = item["identity_type"]
        if ptype not in tokens:
            tokens[ptype] = len(tokens)

    # Collect tokens from expressions and variables
    for item in data:
        expr = item["expression"]
        current = ""
        for c in expr:
            if c in "+-*/^()":
                if current:
                    if current not in tokens:
                        tokens[current] = len(tokens)
                    current = ""
                if c not in tokens:
                    tokens[c] = len(tokens)
            elif c == " ":
                if current:
                    if current not in tokens:
                        tokens[current] = len(tokens)
                    current = ""
            else:
                current += c
        if current and current not in tokens:
            tokens[current] = len(tokens)

    for fn in ["sin", "cos", "exp", "log", "sqrt", "abs", "tan",
               "gcd_val", "lcm_val"]:
        if fn not in tokens:
            tokens[fn] = len(tokens)

    for sym in ["c", "t", "x", "E", "p", "u", "v", "lambda", "n",
                "gamma", "T", "E_peak", "K_max", "nu", "m", "g", "h",
                "k", "P", "V", "R", "hbar", "omega", "a0", "b0", "c0",
                "d0", "e0", "f0"]:
        if sym not in tokens:
            tokens[sym] = len(tokens)

    for d in "0123456789.":
        if d not in tokens:
            tokens[d] = len(tokens)

    inv_tokens = {v: k for k, v in tokens.items()}
    return tokens, inv_tokens


def tokenize_expr(expr: str, token_map: dict[str, int]) -> list[int]:
    """Tokenize an expression string."""
    tokens = [token_map.get("<sos>", 1)]
    current = ""
    i = 0
    while i < len(expr):
        c = expr[i]
        if c in "+-*/^()":
            if current:
                t = token_map.get(current, token_map.get("<unk>", 3))
                tokens.append(t)
                current = ""
            t = token_map.get(c, token_map.get("<unk>", 3))
            tokens.append(t)
            i += 1
        elif c == " ":
            if current:
                t = token_map.get(current, token_map.get("<unk>", 3))
                tokens.append(t)
                current = ""
            i += 1
        elif c.isdigit() or c == ".":
            num = ""
            while i < len(expr) and (expr[i].isdigit() or expr[i] == "."):
                num += expr[i]
                i += 1
            t = token_map.get(num, token_map.get("<unk>", 3))
            tokens.append(t)
        else:
            current += c
            i += 1
    if current:
        t = token_map.get(current, token_map.get("<unk>", 3))
        tokens.append(t)
    tokens.append(token_map.get("<eos>", 2))
    return tokens


def tokenize_with_pattern(
    expr: str, pattern_type: str, token_map: dict[str, int],
) -> list[int]:
    """Tokenize expression with CoT pattern type prepended.

    Format: [sos, pattern_type, expr_tokens..., eos]
    """
    tokens = [token_map.get("<sos>", 1)]
    pt = token_map.get(pattern_type, token_map.get("<unk>", 3))
    tokens.append(pt)
    # Append expression tokens (without the extra SOS)
    expr_tokens = tokenize_expr(expr, token_map)
    tokens.extend(expr_tokens[1:])  # skip the SOS from tokenize_expr
    return tokens


def tokenize_variables(vars_list: list[str], token_map: dict[str, int],
                       max_len: int = 16) -> list[int]:
    tokens = [token_map.get(v, token_map.get("<unk>", 3)) for v in vars_list]
    while len(tokens) < max_len:
        tokens.append(token_map.get("<pad>", 0))
    return tokens[:max_len]


# ═══════════════════════════════════════════════════════════════════════════
# Training (CoT-aware)
# ═══════════════════════════════════════════════════════════════════════════

def train_cross_symbol_model(
    data: list[dict],
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
    device: str = DEVICE,
) -> tuple[CrossSymbolTemplateGenerator, dict, dict]:
    """Train the cross-symbol template generator with CoT."""
    token_map, inv_map = build_tokenizer(data)
    vocab_size = len(token_map)

    print(f"Vocab size: {vocab_size}  (incl. pattern types)")
    print(f"Training examples: {len(data)}")
    print(f"Pattern types: {sorted(k for k in token_map if k not in '<>0123456789+-*/^()' and len(k) > 3 and '_' in k)}")

    model = CrossSymbolTemplateGenerator(
        vocab_size=vocab_size, d_model=64, nhead=4, num_layers=3, max_seq_len=64,
    ).to(device)

    # Convert data to tensors
    src_seqs: list[torch.Tensor] = []
    tgt_seqs: list[torch.Tensor] = []
    max_tgt_len = 0

    for item in data:
        src = tokenize_variables(item["variables"], token_map)
        # CoT: target = [sos, pattern_type, expression..., eos]
        tgt = tokenize_with_pattern(item["expression"], item["identity_type"], token_map)
        src_seqs.append(torch.tensor(src, dtype=torch.long))
        tgt_seqs.append(torch.tensor(tgt, dtype=torch.long))
        max_tgt_len = max(max_tgt_len, len(tgt))

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=token_map.get("<pad>", 0))

    best_loss = float("inf")
    n_batches = (len(data) + batch_size - 1) // batch_size

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total_tokens = 0

        indices = list(range(len(data)))
        random.shuffle(indices)

        for bi in range(n_batches):
            batch_idx = indices[bi * batch_size:(bi + 1) * batch_size]
            if not batch_idx:
                continue

            # Pad to max length in this batch
            batch_max_tgt = max(len(tgt_seqs[i]) for i in batch_idx)
            batch_src = torch.stack([src_seqs[i] for i in batch_idx]).to(device)
            batch_tgt = torch.stack([
                torch.nn.functional.pad(tgt_seqs[i],
                    (0, batch_max_tgt - len(tgt_seqs[i])),
                    value=token_map.get("<pad>", 0))
                for i in batch_idx
            ]).to(device)

            tgt_input = batch_tgt[:, :-1]
            tgt_output = batch_tgt[:, 1:]

            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                tgt_input.size(1), device=device)

            logits = model(batch_src, tgt_input, tgt_mask=tgt_mask)
            loss = criterion(logits.reshape(-1, vocab_size), tgt_output.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

            # Accuracy
            pred = logits.argmax(dim=-1)
            mask = tgt_output != token_map.get("<pad>", 0)
            correct += (pred[mask] == tgt_output[mask]).sum().item()
            total_tokens += mask.sum().item()

        avg_loss = total_loss / n_batches
        acc = correct / max(total_tokens, 1)

        if avg_loss < best_loss:
            best_loss = avg_loss

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"  Epoch {epoch}/{epochs}: loss={avg_loss:.6f}, acc={acc:.4f}, "
                  f"best_loss={best_loss:.6f}")

    print(f"  Training complete")

    model.pattern_token_ids = {
        k: v for k, v in token_map.items()
        if k in set(item["identity_type"] for item in data)
    }

    return model, token_map, inv_map


# ═══════════════════════════════════════════════════════════════════════════
# Testing
# ═══════════════════════════════════════════════════════════════════════════

def test_structural_transfer(
    model: CrossSymbolTemplateGenerator,
    token_map: dict[str, int],
    inv_map: dict[int, str],
):
    """Test CoT two-stage generation on physics symbol sets."""
    print("\n=== STRUCTURAL TRANSFER TEST (CoT) ===")

    test_cases = [
        (["c", "t", "x"], "spacetime: (c*t)^2-x^2"),
        (["E", "p", "c"], "energy-momentum: E^2-p^2"),
        (["u", "v", "c"], "velocity-add: (u+v)/(1+u*v/c^2)"),
        (["lambda", "E"], "Balmer: E*lambda"),
        (["n", "E"], "spin: E/n"),
        (["gamma", "E"], "rest-energy: E/gamma"),
        (["T", "E_peak"], "Wien: E_peak/T"),
        (["K_max", "nu", "h"], "photoelectric: h*nu-K_max"),
    ]

    for symbols, description in test_cases:
        src = tokenize_variables(symbols, token_map)
        src_tensor = torch.tensor([src], dtype=torch.long)

        # Stage 1: sample pattern types
        pattern_ids = model.generate_pattern_types(
            src_tensor, num_samples=COT_NUM_PATTERN_SAMPLES,
            temperature=1.5, top_k=8,
        )

        # Decode pattern names
        pattern_names = [inv_map.get(pid, f"<{pid}>") for pid in pattern_ids]

        # Stage 2: generate expressions for top patterns
        all_exprs = []
        for pid in pattern_ids[:4]:  # top 4 patterns
            seqs = model.generate_with_pattern(
                src_tensor, pid, max_len=32,
                temperature=1.0, top_k=8, num_samples=2,
            )
            for seq in seqs:
                expr = decode_tokens(seq, None, inv_map, token_map)
                if expr and expr not in all_exprs:
                    all_exprs.append(expr)

        print(f"  {description}")
        print(f"    patterns: {pattern_names[:5]}")
        print(f"    exprs: {all_exprs[:5]}")


def decode_tokens(
    seq: list[int],
    _unused: None = None,
    inv_map: dict[int, str] | None = None,
    token_map: dict[str, int] | None = None,
) -> str | None:
    """Decode a token sequence to an expression string.

    Skips SOS and stops at EOS.  For CoT sequences, the pattern-type
    token after SOS is also skipped — only expression tokens are kept.
    """
    tokens = []
    skipped_first = False  # skip SOS
    for tid in seq:
        if token_map and tid in (token_map.get("<pad>", 0), token_map.get("<sos>", 1)):
            continue
        if token_map and tid == token_map.get("<eos>", 2):
            break
        token_str = inv_map.get(tid, "") if inv_map else ""
        # Skip pattern-type tokens (they contain underscores and are not operators)
        if "_" in token_str:
            continue
        tokens.append(token_str)
    expr = "".join(tokens)
    return expr if expr else None


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    rng = random.Random(SEED)
    torch.manual_seed(SEED)

    print("Generating cross-symbol CoT training data...")
    data = generate_cross_symbol_dataset(rng)
    print(f"  Generated {len(data)} examples")

    # Show pattern type distribution
    type_counts = defaultdict(int)
    for item in data:
        type_counts[item["identity_type"]] += 1
    print(f"  Pattern types: {len(type_counts)}")
    for pt, count in sorted(type_counts.items()):
        print(f"    {pt}: {count}")

    t0 = time.time()
    model, token_map, inv_map = train_cross_symbol_model(data)
    elapsed = time.time() - t0
    print(f"  Training complete in {elapsed:.1f}s")

    # Save checkpoint
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "token_map": token_map,
        "inv_map": inv_map,
        "vocab_size": len(token_map),
        "config": {"d_model": 64, "nhead": 4, "num_layers": 3, "max_seq_len": 64},
        "pattern_token_ids": model.pattern_token_ids,
    }
    path = CHECKPOINT_DIR / "cross_symbol_template.pt"
    torch.save(checkpoint, path)
    print(f"  Checkpoint saved to {path}")

    # Test CoT transfer
    test_structural_transfer(model, token_map, inv_map)
    print("\nDone.")


if __name__ == "__main__":
    main()
