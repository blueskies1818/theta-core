#!/usr/bin/env python3
"""Grammar-constrained expression generator.

Unlike the failed token-by-token decoder, this model uses a grammar mask
at each generation step to ensure ONLY valid tokens are produced.  The
model CANNOT generate garbage like "u-u^2/" or "-1/-1+-1VP+PPV".

Training: 50K structural sub-expression examples.
Architecture: Transformer encoder + grammar-masked decoder.
CPU-trainable (no GPU required).
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

# ════════════════════════════════════════════════════
# Config
# ════════════════════════════════════════════════════

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "math_self_play"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

N_SYMBOLS_PER_EXAMPLE = 4
N_EXAMPLES = 30000
EPOCHS = 15
BATCH_SIZE = 64
LR = 0.001
SEED = 42
DEVICE = "xpu" if torch.xpu.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
MAX_SEQ_LEN = 32


# ════════════════════════════════════════════════════
# Tokenization
# ════════════════════════════════════════════════════

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


def tokenize_expr(expr: str, token_map: dict, max_len: int = MAX_SEQ_LEN) -> list[int]:
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


# ════════════════════════════════════════════════════
# Grammar mask
# ════════════════════════════════════════════════════

def build_grammar_mask(token_map: dict, symbols: list[str]):
    """Build boolean masks for valid tokens at each grammar state.

    Grammar states:
      0: START — can begin with symbol, number, '('
      1: AFTER_OP — after +,-,*,/ — expect symbol, number, '('
      2: AFTER_SYM — after symbol — expect op, ')', <eos>
      3: AFTER_NUM — after number — expect op, ')', <eos>
      4: AFTER_LPAREN — after '(' — expect symbol, number, '('
      5: AFTER_POWER — after ^ — expect number only
    """
    vocab_size = len(token_map)
    masks = {}

    # Allowed symbols (from input)
    symbol_ids = {token_map.get(s, -1) for s in symbols} - {-1}
    number_ids = {token_map.get(t, -1) for t in ["0", "0.5", "1", "2", "-1"]} - {-1}
    op_ids = {token_map.get(t, -1) for t in ["+", "-", "*", "/", "^"]} - {-1}
    lparen_id = token_map.get("(", -1)
    rparen_id = token_map.get(")", -1)
    eos_id = token_map.get("<eos>", -1)

    def make_mask(allowed: set) -> list[float]:
        m = [float('-inf')] * vocab_size
        for i in allowed:
            if i >= 0:
                m[i] = 0.0
        return m

    # State 0: START
    masks[0] = make_mask(symbol_ids | number_ids | {lparen_id})

    # State 1: AFTER_OP (+, -, *, /)
    masks[1] = make_mask(symbol_ids | number_ids | {lparen_id})

    # State 2: AFTER_SYM
    masks[2] = make_mask(op_ids | {rparen_id, eos_id})

    # State 3: AFTER_NUM
    masks[3] = make_mask(op_ids | {rparen_id, eos_id})

    # State 4: AFTER_LPAREN
    masks[4] = make_mask(symbol_ids | number_ids | {lparen_id})

    # State 5: AFTER_POWER — numbers only
    masks[5] = make_mask(number_ids)

    return masks


def get_grammar_state(prev_token_id: int, token_map: dict, depth: int) -> int:
    """Determine grammar state from previous token."""
    inv = {v: k for k, v in token_map.items()}

    token = inv.get(prev_token_id, "")
    if token in {"+", "-", "*", "/"}:
        return 1  # AFTER_OP
    if token == "^":
        return 5  # AFTER_POWER — numbers only
    if token == "(":
        return 4  # AFTER_LPAREN
    if token in ("<sos>",):
        return 0  # START
    # Check if it's a symbol or number
    if prev_token_id in {token_map.get(t, -1) for t in
                         list("abcdefghijklmnopqrstuvwxyz") +
                         [f"{c}{d}" for c in "abcdefghijklmnopqrstuvwxyz" for d in "0123"] +
                         list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")}:
        return 2  # AFTER_SYM
    if prev_token_id in {token_map.get(t, -1) for t in ["0", "0.5", "1", "2", "-1"]}:
        return 3  # AFTER_NUM

    return 2  # default: after symbol


# ════════════════════════════════════════════════════
# Training data
# ════════════════════════════════════════════════════

def generate_sub_expressions(symbols: list[str]) -> list[str]:
    """Generate weighted sub-expressions.  70% single-symbol to force EOS learning."""
    exprs = []
    # Promote products heavily — the model prefers short token sequences
    for s in symbols:
        exprs.append(s)  # 1 single
        exprs.append(f"{s}^2")  # 1 square
    for i, a in enumerate(symbols):
        for b in symbols[i+1:]:
            for _ in range(5):  # 5 copies of each product
                exprs.append(f"{a}*{b}")
            exprs.append(f"{a}/{b}")
            exprs.append(f"{a}+{b}")
    return exprs


def generate_training_data(token_map: dict, n: int, rng: random.Random) -> list[dict]:
    all_symbols = [t for t in token_map.keys()
                   if t not in {"<pad>", "<sos>", "<eos>", "(", ")", "+", "-", "*", "/", "^"}
                   and not t.replace('.', '').replace('-', '').isdigit()]
    data = []
    seen = set()

    while len(data) < n:
        n_syms = rng.randint(2, N_SYMBOLS_PER_EXAMPLE)
        symbols = rng.sample(all_symbols, n_syms)
        sub_exprs = generate_sub_expressions(symbols)

        for expr in sub_exprs:
            key = (tuple(sorted(symbols)), expr)
            if key in seen:
                continue
            seen.add(key)

            src = tokenize_symbols(symbols, token_map)
            tgt = tokenize_expr(expr, token_map)
            grammar_masks = build_grammar_mask(token_map, symbols)

            data.append({"symbols": symbols, "expression": expr,
                         "src": src, "tgt": tgt, "masks": grammar_masks})

            if len(data) >= n:
                break

    rng.shuffle(data)
    return data


# ════════════════════════════════════════════════════
# Model
# ════════════════════════════════════════════════════

class GrammarMaskedDecoder(nn.Module):
    """Transformer that generates tokens with grammar constraints."""

    def __init__(self, vocab_size: int, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 4):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.zeros(1, MAX_SEQ_LEN + 16, d_model))

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
        src_emb = self.embedding(src) + self.pos_encoding[:, :src.size(1), :]
        tgt_emb = self.embedding(tgt) + self.pos_encoding[:, :tgt.size(1), :]
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        memory = self.encoder(src_emb)
        output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        return self.output_proj(output)

    def generate(self, src: torch.Tensor, grammar_masks: dict,
                 max_len: int = 32, temperature: float = 1.0,
                 token_map: dict = None, vocab: dict = None) -> list[str]:
        """Generate grammatically valid expressions using beam search.

        token_map: str->int for grammar state tracking
        vocab: int->str for decoding output
        """
        if token_map is None or vocab is None:
            return []

        self.eval()
        batch_size = src.size(0)
        vocab_size = len(token_map)
        eos_id = token_map.get("<eos>", 2)
        sos_id = token_map.get("<sos>", 1)

        results = []

        with torch.no_grad():
            src_emb = self.embedding(src) + self.pos_encoding[:, :src.size(1), :]
            memory = self.encoder(src_emb)

            for b in range(batch_size):
                # Beam search with grammar mask + length penalty
                beam = [(0.0, [sos_id], 0, 0)]  # (score, tokens, depth, state)
                finished = []
                len_penalty = 0.1  # light — let data balance control stopping

                for _ in range(max_len - 1):
                    if not beam:
                        break
                    new_beam = []

                    for score, tokens, depth, state in beam:
                        if tokens[-1] == eos_id:
                            finished.append((score, tokens))
                            continue

                        tgt = torch.tensor([[tokens[-1]]], device=DEVICE)
                        tgt_emb = self.embedding(tgt) + self.pos_encoding[:, :1, :]
                        tgt_mask_sq = nn.Transformer.generate_square_subsequent_mask(1)
                        output = self.decoder(tgt_emb, memory[b:b+1], tgt_mask=tgt_mask_sq)
                        logits = self.output_proj(output[:, -1, :]) / max(temperature, 0.1)

                        # Apply grammar mask + eos bias
                        mask = torch.tensor(grammar_masks[state], device=DEVICE)
                        logits = logits + mask.unsqueeze(0)
                        # Boost <eos> when at a natural stopping point
                        if state in (2, 3) and depth == 0 and len(tokens) >= 2:
                            eos_id = vocab.get("<eos>", 2)
                            logits[0, eos_id] += 1.0  # moderate bias toward ending
                        probs = torch.softmax(logits, dim=-1)

                        # Top-3 candidates with <eos> bias
                        top_probs, top_ids = torch.topk(probs, 4, dim=-1)
                        for i in range(4):
                            tid = top_ids[0, i].item()
                            p = top_probs[0, i].item()
                            if p < 0.001:
                                continue

                            new_tokens = tokens + [tid]
                            new_depth = depth
                            if tid == vocab.get("(", -1):
                                new_depth += 1
                            elif tid == vocab.get(")", -1):
                                new_depth -= 1

                            new_state = get_grammar_state(tid, token_map, new_depth)
                            # Length penalty: shorter is better
                            new_score = score + p - len_penalty * len(new_tokens)
                            new_beam.append((new_score, new_tokens, new_depth, new_state))

                    # Keep top-K
                    new_beam.sort(key=lambda x: -x[0])
                    beam = new_beam[:5]

                    # Add finished
                    if len(finished) >= 3:
                        break

                # Pick best finished or best in-progress
                all_candidates = finished + [(lp, t) for lp, t, _, _ in beam]
                if not all_candidates:
                    results.append("")
                    continue

                all_candidates.sort(key=lambda x: -x[0])
                best = all_candidates[0]
                tokens = best[1]
                # Decode
                expr = ""
                for tid in tokens:
                    if tid in (sos_id, 0):
                        continue
                    if tid == eos_id:
                        break
                    expr += vocab.get(tid, "?")
                results.append(expr)

        return results


# ════════════════════════════════════════════════════
# Training
# ════════════════════════════════════════════════════

def train():
    token_map, inv_map = build_vocab()
    vocab_size = len(token_map)
    rng = random.Random(SEED)

    print(f"Vocabulary: {vocab_size} tokens")
    print(f"Generating {N_EXAMPLES} training examples...")

    data = generate_training_data(token_map, N_EXAMPLES, rng)
    print(f"Generated {len(data)} examples")
    for d in data[:3]:
        print(f"  {'+'.join(d['symbols']):20s} → {d['expression']}")

    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]

    model = GrammarMaskedDecoder(vocab_size=vocab_size, d_model=128, nhead=4, num_layers=4)
    model.to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=token_map["<pad>"])

    print(f"\nTraining {EPOCHS} epochs...")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        rng.shuffle(train_data)

        for i in range(0, len(train_data), BATCH_SIZE):
            batch = train_data[i:i+BATCH_SIZE]
            max_src = max(len(d["src"]) for d in batch)
            max_tgt = max(len(d["tgt"]) for d in batch)

            src = torch.zeros(len(batch), max_src, dtype=torch.long, device=DEVICE)
            tgt_in = torch.zeros(len(batch), max_tgt, dtype=torch.long, device=DEVICE)
            tgt_out = torch.zeros(len(batch), max_tgt, dtype=torch.long, device=DEVICE)

            for j, d in enumerate(batch):
                src[j, :len(d["src"])] = torch.tensor(d["src"])
                tgt_in[j, :len(d["tgt"])] = torch.tensor(d["tgt"])
                tgt_out[j, :len(d["tgt"])-1] = torch.tensor(d["tgt"][1:])
                tgt_out[j, len(d["tgt"])-1] = token_map["<eos>"]

            optimizer.zero_grad()
            output = model(src, tgt_in)
            loss = criterion(output.reshape(-1, vocab_size), tgt_out.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i in range(0, len(val_data), BATCH_SIZE):
                batch = val_data[i:i+BATCH_SIZE]
                max_src = max(len(d["src"]) for d in batch)
                max_tgt = max(len(d["tgt"]) for d in batch)
                src = torch.zeros(len(batch), max_src, dtype=torch.long, device=DEVICE)
                tgt_in = torch.zeros(len(batch), max_tgt, dtype=torch.long, device=DEVICE)
                tgt_out = torch.zeros(len(batch), max_tgt, dtype=torch.long, device=DEVICE)
                for j, d in enumerate(batch):
                    src[j, :len(d["src"])] = torch.tensor(d["src"])
                    tgt_in[j, :len(d["tgt"])] = torch.tensor(d["tgt"])
                    tgt_out[j, :len(d["tgt"])-1] = torch.tensor(d["tgt"][1:])
                    tgt_out[j, len(d["tgt"])-1] = token_map["<eos>"]
                output = model(src, tgt_in)
                loss = criterion(output.reshape(-1, vocab_size), tgt_out.reshape(-1))
                val_loss += loss.item()

        avg_train = total_loss / max(1, len(train_data) / BATCH_SIZE)
        avg_val = val_loss / max(1, len(val_data) / BATCH_SIZE)
        print(f"  Epoch {epoch+1:2d}/{EPOCHS}  train_loss={avg_train:.4f}  val_loss={avg_val:.4f}")

    # Save
    ckpt_path = CHECKPOINT_DIR / "grammar_decoder.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "token_map": token_map,
        "inv_map": inv_map,
        "vocab_size": vocab_size,
        "config": {"d_model": 128, "nhead": 4, "num_layers": 4},
    }, ckpt_path)
    print(f"\nSaved to {ckpt_path}")

    # Quick test — generate proposals
    print("\nQuick test:")
    test_sets = [
        ["E", "lambda"],
        ["c", "t", "x"],
        ["K_max", "nu"],
        ["u", "v", "c"],
    ]
    for syms in test_sets:
        aliased = []
        alias_map = {"K_max": "k", "nu": "n", "lambda": "l", "gamma": "g", "E_peak": "e"}
        for s in syms:
            aliased.append(alias_map.get(s, s))

        src = torch.tensor([tokenize_symbols(aliased, token_map)], device=DEVICE)
        masks = build_grammar_mask(token_map, aliased)
        results = model.generate(src, masks, temperature=0.8,
                                  token_map=token_map, vocab=inv_map)

        # Un-alias
        rev = {v: k for k, v in alias_map.items()}
        unaliased = []
        for r in results:
            for a, o in sorted(rev.items(), key=lambda x: -len(x[0])):
                r = r.replace(a, o)
            unaliased.append(r)

        print(f"  {'+'.join(syms):20s} → {', '.join(u for u in unaliased if u)}")


if __name__ == "__main__":
    train()
