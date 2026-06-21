"""Expression sequence model for learning conservation-law patterns.

Small transformer encoder-decoder that maps a set of physical quantities
(+ scenario type) to a conserved expression string.

Architecture:
  - Encoder: 2-layer TransformerEncoder (128 dim, 4 heads)
    Input: ordered quantity symbols + scenario type token
  - Decoder: 2-layer TransformerDecoder (128 dim, 4 heads)
    Autoregressive generation of expression tokens
  - < 1M parameters, suitable for CPU training with ~50 examples
"""

from __future__ import annotations

import json
import math
import random
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Tokens ────────────────────────────────────────────────────────────────────

# Special tokens
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
SEP_TOKEN = "<sep>"  # separates quantity list from expression
UNK_TOKEN = "<unk>"

SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, SEP_TOKEN, UNK_TOKEN]
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
SEP_IDX = 3
UNK_IDX = 4

# Operators used in physics expressions
OPERATORS = ["+", "-", "*", "/", "^"]

# Constants that appear in training data
CONSTANTS = [
    "0", "0.5", "1", "2", "1/2",
    "g",  # gravitational constant
]

# Physical quantity symbols (all domains)
QUANTITY_SYMBOLS = [
    "m",   # mass
    "g",   # gravitational acceleration
    "h",   # height / displacement
    "v",   # velocity
    "t",   # time
    "k",   # spring constant
    "L",   # length (pendulum)
    "q",   # charge
    "E",   # electric field
    "x",   # generic displacement
    "y",   # generic coordinate
    "r",   # radius / distance
]

# Scenario type tokens
SCENARIO_TYPES = [
    "free_fall",
    "projectile",
    "pendulum",
    "spring",
    "collision",
    "incline",
    "gravity_spring",
    "em_gravity",
    "spring_friction",
    "gravity_spring_friction",
    "unknown",
]


class ExpressionTokenizer:
    """Tokenize/detokenize physics expressions to/from integer sequences.

    Vocabulary:
      [PAD=0, SOS=1, EOS=2, SEP=3, UNK=4,
       operators (+, -, *, /, ^),
       constants (0, 0.5, 1, 2, 1/2, g),
       quantities (m, g, h, v, t, k, L, q, E, x, y, r),
       scenario types (free_fall, projectile, ...)]
    """

    def __init__(self) -> None:
        self._build_vocab()
        self._build_reverse()

    def _build_vocab(self) -> None:
        """Build token → id mapping."""
        tokens: list[str] = []
        tokens.extend(SPECIAL_TOKENS)
        tokens.extend(OPERATORS)
        tokens.extend(CONSTANTS)
        tokens.extend(QUANTITY_SYMBOLS)
        tokens.extend(SCENARIO_TYPES)
        self._token_to_id: dict[str, int] = {t: i for i, t in enumerate(tokens)}
        self._id_to_token: dict[int, str] = {i: t for t, i in self._token_to_id.items()}

    def _build_reverse(self) -> None:
        """Already done in _build_vocab."""
        pass

    @property
    def vocab_size(self) -> int:
        return len(self._token_to_id)

    def encode(self, token_str: str) -> int:
        """Encode a single token string to its id."""
        return self._token_to_id.get(token_str, UNK_IDX)

    def decode(self, token_id: int) -> str:
        """Decode a single token id to its string."""
        return self._id_to_token.get(token_id, UNK_TOKEN)

    def tokenize_expression(self, expr_str: str) -> list[int]:
        """Tokenize an expression string like 'm*g*h + 0.5*m*v^2' into token IDs.

        Handles operators, quantities, constants, and numbers.
        """
        tokens: list[int] = []
        i = 0
        s = expr_str.strip()

        while i < len(s):
            # Skip whitespace
            if s[i].isspace():
                i += 1
                continue

            # Try multi-char tokens first: operators, then identifiers/numbers
            matched = False

            # Check multi-char constants/quantities first
            for length in (4, 3, 2, 1):
                if i + length <= len(s):
                    candidate = s[i:i + length]
                    if candidate in self._token_to_id:
                        tokens.append(self._token_to_id[candidate])
                        i += length
                        matched = True
                        break
            if matched:
                continue

            # Fallback: single character
            ch = s[i]
            if ch in self._token_to_id:
                tokens.append(self._token_to_id[ch])
            else:
                # Unknown character — could be a number like "3"
                tokens.append(UNK_IDX)
            i += 1

        return tokens

    def detokenize_expression(self, token_ids: list[int]) -> str:
        """Convert token IDs back to expression string.

        Stops at EOS token. Skips PAD, SOS, SEP.
        """
        parts: list[str] = []
        for tid in token_ids:
            if tid in (PAD_IDX, SOS_IDX, SEP_IDX):
                continue
            if tid == EOS_IDX:
                break
            tok = self._id_to_token.get(tid, UNK_TOKEN)
            if tok in SPECIAL_TOKENS:
                continue
            parts.append(tok)
        return " ".join(parts) if parts else ""

    def encode_quantities(self, quantity_symbols: list[str]) -> list[int]:
        """Encode a list of quantity symbols to token IDs."""
        return [self.encode(q) for q in quantity_symbols]

    def encode_scenario(self, scenario_type: str) -> int:
        """Encode a scenario type string to token ID."""
        return self.encode(scenario_type)

    def expression_to_tensor(
        self, expr_str: str, max_len: int = 64
    ) -> torch.Tensor:
        """Convert expression string to padded tensor [SOS, tokens..., EOS, PAD...]."""
        tokens = self.tokenize_expression(expr_str)
        ids = [SOS_IDX] + tokens + [EOS_IDX]
        if len(ids) > max_len:
            ids = ids[:max_len - 1] + [EOS_IDX]
        # Pad
        padded = ids + [PAD_IDX] * (max_len - len(ids))
        return torch.tensor(padded, dtype=torch.long)

    def quantities_to_tensor(
        self, quantity_symbols: list[str], max_len: int = 16
    ) -> torch.Tensor:
        """Convert quantity symbol list to padded tensor."""
        ids = self.encode_quantities(quantity_symbols)
        if len(ids) > max_len:
            ids = ids[:max_len]
        padded = ids + [PAD_IDX] * (max_len - len(ids))
        return torch.tensor(padded, dtype=torch.long)

    def save(self, path: str | Path) -> None:
        """Save vocabulary to JSON."""
        with open(path, "w") as f:
            json.dump({"token_to_id": self._token_to_id}, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> ExpressionTokenizer:
        """Load vocabulary from JSON."""
        tok = cls.__new__(cls)
        with open(path) as f:
            data = json.load(f)
        tok._token_to_id = data["token_to_id"]
        tok._id_to_token = {int(i): t for t, i in tok._token_to_id.items()}
        return tok


# ── Positional Encoding ───────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 128, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding. x: [batch, seq_len, d_model]."""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ── Model ─────────────────────────────────────────────────────────────────────

class ExpressionSequenceModel(nn.Module):
    """Transformer encoder-decoder for physics expression generation.

    Encoder: processes the set of available quantities + scenario type
    Decoder: generates expression tokens autoregressively

    Parameters
    ----------
    vocab_size : int
        Size of the token vocabulary.
    d_model : int
        Hidden dimension (default 128).
    nhead : int
        Number of attention heads (default 4).
    num_encoder_layers : int
        Number of transformer encoder layers (default 2).
    num_decoder_layers : int
        Number of transformer decoder layers (default 2).
    max_src_len : int
        Maximum source sequence length (quantities + scenario type).
    max_tgt_len : int
        Maximum target sequence length (expression tokens).
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        max_src_len: int = 16,
        max_tgt_len: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        # Shared embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)

        # Positional encodings
        self.src_pos_encoding = PositionalEncoding(d_model, max_src_len, dropout)
        self.tgt_pos_encoding = PositionalEncoding(d_model, max_tgt_len, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src: torch.Tensor,          # [batch, src_len]
        tgt: torch.Tensor,          # [batch, tgt_len]
        src_padding_mask: torch.Tensor | None = None,  # [batch, src_len]
        tgt_padding_mask: torch.Tensor | None = None,  # [batch, tgt_len]
        tgt_mask: torch.Tensor | None = None,          # [tgt_len, tgt_len]
    ) -> torch.Tensor:
        """Forward pass.

        Returns logits: [batch, tgt_len, vocab_size]
        """
        # Embed and add positional encoding
        src_emb = self.token_embedding(src) * math.sqrt(self.d_model)
        src_emb = self.src_pos_encoding(src_emb)

        tgt_emb = self.token_embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.tgt_pos_encoding(tgt_emb)

        # Encode source
        memory = self.encoder(
            src_emb,
            src_key_padding_mask=src_padding_mask,
        )

        # Decode target (autoregressive with causal mask)
        output = self.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
        )

        return self.output_proj(output)

    def encode_source(
        self,
        src: torch.Tensor,
        src_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode source quantities → memory tensor for generation."""
        src_emb = self.token_embedding(src) * math.sqrt(self.d_model)
        src_emb = self.src_pos_encoding(src_emb)
        return self.encoder(src_emb, src_key_padding_mask=src_padding_mask)

    def decode_step(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Single decode step. Returns logits for next token."""
        tgt_emb = self.token_embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.tgt_pos_encoding(tgt_emb)
        output = self.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.output_proj(output)

    def generate(
        self,
        src: torch.Tensor,
        src_padding_mask: torch.Tensor | None = None,
        max_len: int = 64,
        temperature: float = 1.0,
    ) -> list[list[int]]:
        """Generate expression sequences for a batch of source inputs.

        Returns list of token ID sequences (one per batch item), terminated by EOS.
        """
        batch_size = src.size(0)
        device = src.device

        memory = self.encode_source(src, src_padding_mask)

        # Start with SOS token
        generated = torch.full(
            (batch_size, 1), SOS_IDX, dtype=torch.long, device=device
        )

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            # Create causal mask for current sequence length
            tgt_len = generated.size(1)
            tgt_mask = torch.triu(
                torch.ones(tgt_len, tgt_len, device=device) * float("-inf"), diagonal=1
            )

            # Decode
            logits = self.decode_step(
                generated, memory, tgt_mask, memory_key_padding_mask=src_padding_mask
            )

            # Sample or greedy
            if temperature > 0:
                next_logits = logits[:, -1, :] / temperature
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                next_token = logits[:, -1, :].argmax(dim=-1)

            # Don't change finished sequences
            next_token = torch.where(finished, PAD_IDX, next_token)

            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)

            # Check for EOS
            finished = finished | (next_token == EOS_IDX)
            if finished.all():
                break

        # Convert to list of lists
        results: list[list[int]] = []
        for b in range(batch_size):
            seq = generated[b].tolist()
            # Trim at EOS
            if EOS_IDX in seq:
                seq = seq[:seq.index(EOS_IDX) + 1]
            results.append(seq)

        return results

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Training Data ─────────────────────────────────────────────────────────────

class ExpressionDataset(torch.utils.data.Dataset):
    """Dataset for training the expression model.

    Each item: (src_tensor, tgt_tensor) where
      src_tensor = [qty1_id, qty2_id, ..., scenario_id]
      tgt_tensor = [SOS, token1, token2, ..., EOS, PAD, ...]
    """

    def __init__(
        self,
        examples: list[dict],
        tokenizer: ExpressionTokenizer,
        src_max_len: int = 16,
        tgt_max_len: int = 64,
    ):
        self.tokenizer = tokenizer
        self.src_max_len = src_max_len
        self.tgt_max_len = tgt_max_len

        self.examples: list[tuple[torch.Tensor, torch.Tensor]] = []
        for ex in examples:
            src, tgt = self._encode_example(ex)
            self.examples.append((src, tgt))

    def _encode_example(
        self, ex: dict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a single example dict into tensors."""
        quantities = sorted(ex["quantities"].keys())
        scenario = ex.get("scenario_type", "unknown")
        expression = ex["expression"]

        # Source: quantities + scenario type
        src_ids = [self.tokenizer.encode(q) for q in quantities]
        src_ids.append(self.tokenizer.encode_scenario(scenario))
        if len(src_ids) > self.src_max_len:
            src_ids = src_ids[:self.src_max_len]
        src = torch.tensor(
            src_ids + [PAD_IDX] * (self.src_max_len - len(src_ids)),
            dtype=torch.long,
        )

        # Target: expression with SOS/EOS
        tgt = self.tokenizer.expression_to_tensor(expression, self.tgt_max_len)

        return src, tgt

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.examples[idx]


# ── Data Loading ──────────────────────────────────────────────────────────────

def extract_training_examples(
    observations_path: str | Path,
    discoveries_path: str | Path | None = None,
    augment: bool = True,
) -> list[dict]:
    """Extract training examples from observation databases and discovery files.

    Returns list of dicts with keys: quantities, scenario_type, conservative, expression

    If augment=True, generates expression variants with reordered terms
    to help the model learn commutativity of addition.
    """
    examples: list[dict] = []

    # Load observations
    with open(observations_path) as f:
        observations = json.load(f)

    for obs in observations:
        inv = obs.get("known_invariant")
        if inv is None:
            continue
        quantities = dict(obs["quantities"])
        # Fix: charged_particle_gravity has q, E in parameters but not quantities
        if obs["id"] == "charged_particle_gravity":
            quantities["q"] = "Charge"
            quantities["E"] = "Force/Charge"
        scenario_type = _infer_scenario_type(obs["id"], list(quantities.keys()))
        examples.append({
            "quantities": quantities,
            "scenario_type": scenario_type,
            "conservative": obs.get("is_conservative", True),
            "expression": inv,
        })

    # Load discoveries
    if discoveries_path is not None:
        with open(discoveries_path) as f:
            discoveries = json.load(f)

        for disc in discoveries.get("discoveries", []):
            expr = disc["expression"]
            # Infer quantities from the expression symbols
            qty_symbols = _extract_quantity_symbols(expr)
            qty_dict = _symbols_to_dimensions(qty_symbols)
            scenario_type = disc.get("conditional_pattern", "unknown")
            # Map descriptive scenario types
            scenario_type = _normalize_scenario_type(scenario_type, expr)

            # Only add if conservative_score > 0.7 (real discoveries)
            if disc.get("conservative_score", 0) > 0.7:
                examples.append({
                    "quantities": qty_dict,
                    "scenario_type": scenario_type,
                    "conservative": True,
                    "expression": expr,
                })

    # Data augmentation: generate expression variants with reordered terms
    if augment:
        augmented: list[dict] = []
        for ex in examples:
            expr = ex["expression"]
            variants = _generate_expression_variants(expr)
            for variant in variants:
                if variant != expr:
                    augmented.append({
                        "quantities": dict(ex["quantities"]),
                        "scenario_type": ex["scenario_type"],
                        "conservative": ex["conservative"],
                        "expression": variant,
                    })
        examples.extend(augmented)

    return examples


def _generate_expression_variants(expr: str) -> list[str]:
    """Generate term-reordered variants of a sum-of-terms expression.

    For example, 'm*g*h + 0.5*m*v^2' yields:
      ['m*g*h + 0.5*m*v^2', '0.5*m*v^2 + m*g*h']
    """
    # Split on top-level '+'
    terms = _split_sum_terms(expr)
    if len(terms) <= 1:
        return [expr]

    variants: list[str] = []
    # Generate permutations of terms
    from itertools import permutations
    seen: set[str] = set()
    for perm in permutations(terms):
        variant = " + ".join(perm)
        if variant not in seen:
            seen.add(variant)
            variants.append(variant)

    return variants if variants else [expr]


def _split_sum_terms(expr: str) -> list[str]:
    """Split expression on top-level '+' signs, respecting nesting."""
    terms: list[str] = []
    depth = 0
    current: list[str] = []
    i = 0
    while i < len(expr):
        ch = expr[i]
        if ch == '(':
            depth += 1
            current.append(ch)
        elif ch == ')':
            depth -= 1
            current.append(ch)
        elif ch == '+' and depth == 0:
            terms.append("".join(current).strip())
            current = []
        elif ch == '-' and depth == 0 and i > 0 and expr[i-1] not in ('+', '-', '*', '/', '^', '('):
            # Subtraction between terms: split here
            terms.append("".join(current).strip())
            current = ["-"]
        else:
            current.append(ch)
        i += 1
    if current:
        terms.append("".join(current).strip())
    return [t for t in terms if t]


def _infer_scenario_type(obs_id: str, quantity_symbols: list[str]) -> str:
    """Infer scenario type from observation ID and quantities."""
    obs_id_lower = obs_id.lower()
    syms = set(quantity_symbols)
    # Combined: gravity + spring + friction/damped
    if ("gravity" in obs_id_lower or "g" in syms) and ("spring" in obs_id_lower or "k" in syms) and ("damped" in obs_id_lower or "friction" in obs_id_lower):
        return "gravity_spring_friction"
    # Combined: spring + friction/damped (no gravity)
    if ("spring" in obs_id_lower or "k" in syms) and ("damped" in obs_id_lower or "friction" in obs_id_lower):
        return "spring_friction"
    if "spring" in obs_id_lower and "gravity" in obs_id_lower:
        return "gravity_spring"
    if "spring" in obs_id_lower:
        return "spring"
    if "collision" in obs_id_lower:
        return "collision"
    if "pendulum" in obs_id_lower:
        return "pendulum"
    if "projectile" in obs_id_lower:
        return "projectile"
    if "incline" in obs_id_lower:
        return "incline"
    if "charged" in obs_id_lower:
        return "em_gravity"
    if "freefall" in obs_id_lower or "falling" in obs_id_lower:
        return "free_fall"
    # Default: use quantity-based classification
    if "k" in syms and "g" in syms:
        return "gravity_spring"
    if "k" in syms:
        return "spring"
    if "g" in syms:
        return "free_fall"
    if "v" in syms and "m" in syms:
        return "collision"
    return "unknown"


def _extract_quantity_symbols(expr: str) -> list[str]:
    """Extract quantity variable names from an expression string."""
    import re
    # Match single-letter variables that are standard physics symbols
    # Exclude: numbers, operators, functions
    symbols = set()
    for token in re.findall(r'[a-zA-Z_]\w*', expr):
        # Skip known functions
        if token in ("sin", "cos", "sqrt", "exp", "log", "abs"):
            continue
        if len(token) == 1:  # Single-letter variables only
            symbols.add(token)
    return sorted(symbols)


def _symbols_to_dimensions(symbols: list[str]) -> dict[str, str]:
    """Map physics symbols to dimension type strings."""
    dim_map = {
        "m": "Mass",
        "g": "Accel",
        "h": "Length",
        "v": "Velocity",
        "t": "Time",
        "k": "Force/Length",
        "L": "Length",
        "q": "Charge",
        "E": "Force/Charge",
        "x": "Length",
        "y": "Length",
        "r": "Length",
    }
    return {s: dim_map.get(s, "Scalar") for s in symbols}


def _normalize_scenario_type(raw: str, expr: str) -> str:
    """Normalize scenario type strings."""
    raw_lower = raw.lower()
    if "gravity" in raw_lower and "spring" in raw_lower:
        return "gravity_spring"
    if "spring" in raw_lower:
        return "spring"
    if "em" in raw_lower or "charged" in raw_lower:
        return "em_gravity"
    if "gravity" in raw_lower or "partial" in raw_lower:
        return "free_fall"
    if "kinetic" in raw_lower:
        return "collision"
    # Fallback: check expression content
    if "k" in expr and "g" in expr:
        return "gravity_spring"
    if "k" in expr:
        return "spring"
    if "g" in expr:
        return "free_fall"
    return "unknown"


def create_train_test_split(
    examples: list[dict],
    test_scenario_types: set[str] | None = None,
    test_size: float = 0.3,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Split examples into train and test sets.

    If test_scenario_types is provided, all examples of those types go to test.
    Otherwise, random stratified split by scenario_type.
    """
    if test_scenario_types is not None:
        train = [ex for ex in examples
                 if ex["scenario_type"] not in test_scenario_types]
        test = [ex for ex in examples
                if ex["scenario_type"] in test_scenario_types]
        return train, test

    # Random stratified split
    random.seed(seed)
    by_type: dict[str, list[dict]] = {}
    for ex in examples:
        by_type.setdefault(ex["scenario_type"], []).append(ex)

    train, test = [], []
    for stype, exs in by_type.items():
        random.shuffle(exs)
        n_test = max(1, int(len(exs) * test_size))
        train.extend(exs[n_test:])
        test.extend(exs[:n_test])

    return train, test
