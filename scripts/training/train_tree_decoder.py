#!/usr/bin/env python3
"""Train the tree-based expression decoder (Phase C proper).

Generates expression proposals as RPN action sequences — eliminates the
token-length bias that causes the flat grammar decoder to mode-collapse.

Architecture:
  Encoder: Transformer (maps input symbols → context)
  Decoder: Transformer (autoregressive action prediction with grammar mask)
  Output:  (action_type, param) at each step — always valid by construction.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim

from src.math.tree_decoder import (
    TreeDecoder, expr_to_rpn, rpn_to_expr,
    ACTION_SOS, ACTION_PUSH_VAR, ACTION_PUSH_CONST,
    ACTION_APPLY_ADD, ACTION_APPLY_SUB, ACTION_APPLY_MUL,
    ACTION_APPLY_DIV, ACTION_APPLY_POW, ACTION_DONE,
    NUM_ACTIONS, CONST_VALUES, MAX_VARS, NUM_CONSTS,
    ACTION_NAMES,
)

# ════════════════════════════════════════════════════
# Config
# ════════════════════════════════════════════════════

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "math_self_play"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

N_EXAMPLES = 50000
EPOCHS = 20
BATCH_SIZE = 64
LR = 0.0005
SEED = 42
DEVICE = "xpu" if torch.xpu.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
MAX_SEQ_LEN = 20  # max RPN actions per expression

# ════════════════════════════════════════════════════
# Symbol pool (matching training data from other phases)
# ════════════════════════════════════════════════════

ALL_QUANTITY_SYMBOLS = [
    # Single-char
    "E", "B", "T", "c", "x", "t", "y", "z", "u", "v", "w",
    "p", "q", "r", "s", "a", "b", "d", "e", "f", "g", "h",
    "i", "j", "k", "l", "m", "n", "o",
    # Multi-char (aliased to single-char for tokenization)
    "K_max", "nu", "lambda", "gamma",
    # Capitals
    "P", "V", "R", "I", "A", "C", "F", "G", "H", "K", "L",
    "M", "N", "O", "Q", "S", "U", "W", "X", "Y", "Z",
]

# ════════════════════════════════════════════════════
# Symbol ID encoding (consistent across training/inference)
# ════════════════════════════════════════════════════

def build_symbol_map():
    """Build a simple symbol-to-id mapping. ID 0 is reserved for padding."""
    smap = {}
    idx = 1
    for s in ALL_QUANTITY_SYMBOLS:
        smap[s] = idx
        idx += 1
    return smap

SYMBOL_MAP = build_symbol_map()
SYMBOL_COUNT = len(SYMBOL_MAP) + 1  # +1 for padding

# ════════════════════════════════════════════════════
# Training data generation
# ════════════════════════════════════════════════════

def generate_expression(symbols: list[str], rng: random.Random, max_depth: int = 2) -> str:
    """Generate a random valid expression from the given symbols.

    Generates diverse expression types: singles, squares, products,
    ratios, sums, differences, powers, and nested compositions.
    """
    n = len(symbols)

    # Singleton
    if n == 1 or rng.random() < 0.1:
        return symbols[0]

    a, b = rng.sample(symbols, 2) if n >= 2 else (symbols[0], symbols[0])

    # Choose expression type with balanced distribution
    r = rng.random()

    if r < 0.25:
        # Product
        return f"({a}*{b})"

    elif r < 0.40:
        # Ratio
        return f"({a}/{b})"

    elif r < 0.50:
        # Sum
        return f"({a}+{b})"

    elif r < 0.58:
        # Difference
        return f"({a}-{b})"

    elif r < 0.72:
        # Square of single var
        return f"({a}^2)"

    elif r < 0.85:
        # Square of product (if depth allows)
        return f"(({a}*{b})^2)"

    elif r < 0.95:
        # Power with other constant
        const = rng.choice(["0.5", "-1"])
        return f"({a}^{const})"

    else:
        # Nested: squared difference or deeper composition
        if n >= 3 and max_depth >= 2:
            remaining = [s for s in symbols if s != a and s != b]
            if remaining:
                c = rng.choice(remaining)
                return f"(({a}*{b})^2-{c}^2)"
            else:
                return f"(({a}*{b})^2-{b}^2)"
        else:
            return f"(({a}*{b})^2)"


def generate_training_data(n: int, rng: random.Random) -> list[dict]:
    """Generate n training examples: (symbol_ids, action_seq, param_seq)."""
    data = []
    seen = set()

    while len(data) < n:
        n_syms = rng.randint(1, 4)
        symbols = rng.sample(ALL_QUANTITY_SYMBOLS, n_syms)
        sym_key = tuple(sorted(symbols))

        expr = generate_expression(symbols, rng)
        key = (sym_key, expr)
        if key in seen:
            continue
        seen.add(key)

        # Convert to RPN actions
        var_indices = {s: i for i, s in enumerate(symbols)}
        try:
            actions = expr_to_rpn(expr, var_indices)
        except (ValueError, KeyError):
            continue

        if len(actions) > MAX_SEQ_LEN:
            continue

        # Encode symbol IDs
        sym_ids = [SYMBOL_MAP.get(s, 0) for s in symbols]
        # Pad to MAX_VARS
        while len(sym_ids) < MAX_VARS:
            sym_ids.append(0)

        action_seq = [ACTION_SOS] + [a for a, p in actions]
        param_seq = [0] + [p for a, p in actions]

        # Pad sequences to MAX_SEQ_LEN
        while len(action_seq) < MAX_SEQ_LEN:
            action_seq.append(0)
            param_seq.append(0)

        data.append({
            "symbols": symbols,
            "expression": expr,
            "symbol_ids": sym_ids[:MAX_VARS],
            "action_seq": action_seq[:MAX_SEQ_LEN],
            "param_seq": param_seq[:MAX_SEQ_LEN],
            "num_vars": n_syms,
            "num_actions": len(actions),
        })

    return data


# ════════════════════════════════════════════════════
# Training
# ════════════════════════════════════════════════════

def train():
    print(f"Device: {DEVICE}")
    print(f"Generating {N_EXAMPLES} training examples...")

    rng = random.Random(SEED)
    data = generate_training_data(N_EXAMPLES, rng)
    print(f"Generated {len(data)} examples")

    # Show sample distribution
    type_counts = {}
    for d in data[:100]:
        expr = d["expression"]
        if "^" in expr:
            if "*" in expr:
                t = "squared_product"
            else:
                t = "power"
        elif "*" in expr:
            t = "product"
        elif "/" in expr:
            t = "ratio"
        elif "+" in expr:
            t = "sum"
        elif "-" in expr:
            t = "difference"
        else:
            t = "single"
        type_counts[t] = type_counts.get(t, 0) + 1
    print(f"Type distribution (first 100): {type_counts}")

    for d in data[:5]:
        print(f"  {'+'.join(d['symbols']):25s} → {d['expression']:25s} "
              f"actions={d['num_actions']} (with SOS)")

    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]

    model = TreeDecoder(d_model=128, nhead=4, num_encoder_layers=3,
                         num_decoder_layers=3, max_seq_len=MAX_SEQ_LEN)
    model.to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Losses
    action_criterion = nn.CrossEntropyLoss(ignore_index=0)
    param_criterion = nn.CrossEntropyLoss(ignore_index=0)

    print(f"\nTraining {EPOCHS} epochs (batches of {BATCH_SIZE})...")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(EPOCHS):
        model.train()
        total_action_loss = 0.0
        total_param_loss = 0.0
        rng.shuffle(train_data)

        for i in range(0, len(train_data), BATCH_SIZE):
            batch = train_data[i:i + BATCH_SIZE]
            if len(batch) < 2:
                continue

            sym_ids = torch.tensor([d["symbol_ids"] for d in batch],
                                    dtype=torch.long, device=DEVICE)
            action_seq = torch.tensor([d["action_seq"] for d in batch],
                                       dtype=torch.long, device=DEVICE)
            param_seq = torch.tensor([d["param_seq"] for d in batch],
                                      dtype=torch.long, device=DEVICE)

            # Forward — teacher forcing
            action_logits, var_logits, const_logits = model(
                sym_ids, action_seq[:, :-1], param_seq[:, :-1])

            # Target: shift right
            tgt_action = action_seq[:, 1:]   # (B, seq_len-1)
            tgt_param = param_seq[:, 1:]

            # Action loss
            action_loss = action_criterion(
                action_logits.reshape(-1, NUM_ACTIONS),
                tgt_action.reshape(-1))

            # Param loss — only where action is PUSH_VAR or PUSH_CONST
            param_loss = torch.tensor(0.0, device=DEVICE)
            mask_var = (tgt_action == ACTION_PUSH_VAR)
            mask_const = (tgt_action == ACTION_PUSH_CONST)

            if mask_var.any():
                var_l = param_criterion(
                    var_logits.reshape(-1, MAX_VARS),
                    tgt_param.reshape(-1))
                param_loss = param_loss + var_l * 0.3  # lower weight on var selection

            if mask_const.any():
                const_l = param_criterion(
                    const_logits.reshape(-1, NUM_CONSTS),
                    tgt_param.reshape(-1))
                param_loss = param_loss + const_l * 0.3

            loss = action_loss + param_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_action_loss += action_loss.item()

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct_action = 0
        val_total_action = 0

        with torch.no_grad():
            for i in range(0, len(val_data), BATCH_SIZE):
                batch = val_data[i:i + BATCH_SIZE]
                if len(batch) < 2:
                    continue

                sym_ids = torch.tensor([d["symbol_ids"] for d in batch],
                                        dtype=torch.long, device=DEVICE)
                action_seq = torch.tensor([d["action_seq"] for d in batch],
                                           dtype=torch.long, device=DEVICE)
                param_seq = torch.tensor([d["param_seq"] for d in batch],
                                          dtype=torch.long, device=DEVICE)

                action_logits, var_logits, const_logits = model(
                    sym_ids, action_seq[:, :-1], param_seq[:, :-1])

                tgt_action = action_seq[:, 1:]
                action_loss = action_criterion(
                    action_logits.reshape(-1, NUM_ACTIONS),
                    tgt_action.reshape(-1))
                val_loss += action_loss.item()

                # Accuracy: argmax of action prediction
                pred_action = action_logits.argmax(dim=-1)
                mask = tgt_action != 0
                val_correct_action += (pred_action[mask] == tgt_action[mask]).sum().item()
                val_total_action += mask.sum().item()

        n_batches = max(1, len(train_data) // BATCH_SIZE)
        n_val_batches = max(1, len(val_data) // BATCH_SIZE)
        val_acc = val_correct_action / max(1, val_total_action) * 100

        print(f"  Epoch {epoch+1:2d}/{EPOCHS}  "
              f"action_loss={total_action_loss/n_batches:.4f}  "
              f"val_loss={val_loss/n_val_batches:.4f}  "
              f"val_acc={val_acc:.1f}%  "
              f"lr={scheduler.get_last_lr()[0]:.6f}")

    # Save
    ckpt_path = CHECKPOINT_DIR / "tree_decoder.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "symbol_map": SYMBOL_MAP,
        "config": {"d_model": 128, "nhead": 4, "num_encoder_layers": 3,
                    "num_decoder_layers": 3, "max_seq_len": MAX_SEQ_LEN},
    }, ckpt_path)
    print(f"\nSaved to {ckpt_path}")

    # ════════════════════════════════════════════════
    # Generation test
    # ════════════════════════════════════════════════

    print("\n" + "=" * 60)
    print("Generation Test")
    print("=" * 60)

    test_sets = [
        ["E", "lambda"],
        ["c", "t", "x"],
        ["K_max", "nu"],
        ["E", "p"],
        ["u", "v"],
        ["P", "V", "T"],
    ]

    for syms in test_sets:
        sym_ids = torch.tensor([[SYMBOL_MAP.get(s, 0) for s in syms] +
                                 [0] * (MAX_VARS - len(syms))],
                                device=DEVICE)
        results = model.generate(sym_ids, len(syms), var_names=syms,
                                  temperature=0.8, num_samples=8)
        # Show unique
        unique = list(dict.fromkeys(r for r in results if r))
        print(f"\n  {'+'.join(syms):20s} → {', '.join(unique[:6])}")

    print("\nDone.")


if __name__ == "__main__":
    train()
