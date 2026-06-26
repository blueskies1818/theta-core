"""
Tree-based expression decoder — generates expressions as AST operations.

Uses Reverse Polish Notation (RPN) to build expressions bottom-up.
Operations: PUSH_VAR, PUSH_CONST, APPLY_ADD/SUB/MUL/DIV/POW, DONE.
Grammar mask at each step ensures only valid operations.

Architecture:
  Encoder: Transformer encoder maps input symbols → context embeddings.
  Decoder: Autoregressive transformer decoder predicts (action, param) pairs.
  Output is always a valid, parseable expression by construction.

No token-length bias — each operation has equal prediction cost regardless
of symbol name length.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ════════════════════════════════════════════════════
# Action vocabulary
# ════════════════════════════════════════════════════

ACTION_SOS = 0
ACTION_PUSH_VAR = 1
ACTION_PUSH_CONST = 2
ACTION_APPLY_ADD = 3
ACTION_APPLY_SUB = 4
ACTION_APPLY_MUL = 5
ACTION_APPLY_DIV = 6
ACTION_APPLY_POW = 7
ACTION_DONE = 8

NUM_ACTIONS = 9
NUM_CONSTS = 5  # 0, 0.5, 1, 2, -1
CONST_VALUES = [0, 0.5, 1, 2, -1]
MAX_VARS = 8  # max variables per symbol set

ACTION_NAMES = {
    ACTION_SOS: "SOS",
    ACTION_PUSH_VAR: "PUSH_VAR",
    ACTION_PUSH_CONST: "PUSH_CONST",
    ACTION_APPLY_ADD: "APPLY_ADD",
    ACTION_APPLY_SUB: "APPLY_SUB",
    ACTION_APPLY_MUL: "APPLY_MUL",
    ACTION_APPLY_DIV: "APPLY_DIV",
    ACTION_APPLY_POW: "APPLY_POW",
    ACTION_DONE: "DONE",
}


# ════════════════════════════════════════════════════
# RPN stack simulator (non-differentiable)
# ════════════════════════════════════════════════════

class RPNState:
    """Tracks stack state for grammar enforcement during generation."""

    def __init__(self, num_vars: int):
        self.stack: list[str] = []  # expression strings
        self.vars_used: set[int] = set()
        self.num_vars = num_vars
        self.done = False

    def clone(self) -> "RPNState":
        s = RPNState(self.num_vars)
        s.stack = list(self.stack)
        s.vars_used = set(self.vars_used)
        s.done = self.done
        return s

    def apply(self, action: int, param: int, var_names: list[str]) -> Optional[str]:
        """Apply an action. Returns the expression string if tree complete."""
        if action == ACTION_PUSH_VAR:
            if param >= len(var_names):
                raise ValueError(f"Var index {param} out of range for {var_names}")
            self.stack.append(var_names[param])
            self.vars_used.add(param)
            return None

        elif action == ACTION_PUSH_CONST:
            self.stack.append(str(CONST_VALUES[param]))
            return None

        elif action == ACTION_DONE:
            if len(self.stack) != 1:
                raise ValueError(f"DONE with stack size {len(self.stack)} != 1")
            self.done = True
            return self.stack[0]

        elif action in (ACTION_APPLY_ADD, ACTION_APPLY_SUB, ACTION_APPLY_MUL,
                         ACTION_APPLY_DIV, ACTION_APPLY_POW):
            if len(self.stack) < 2:
                raise ValueError(f"APPLY with stack size {len(self.stack)} < 2")
            right = self.stack.pop()
            left = self.stack.pop()
            op = {ACTION_APPLY_ADD: "+", ACTION_APPLY_SUB: "-",
                  ACTION_APPLY_MUL: "*", ACTION_APPLY_DIV: "/",
                  ACTION_APPLY_POW: "^"}[action]
            expr = f"({left}{op}{right})"
            self.stack.append(expr)
            return None

        raise ValueError(f"Unknown action: {action}")


def get_valid_actions(state: RPNState, num_vars: int) -> tuple[list[int], list[float], list[float]]:
    """
    Returns (valid_actions, action_mask, var_mask) for current state.

    action_mask: list[float] length NUM_ACTIONS, 0.0 for valid, -inf for invalid
    var_mask: list[float] length MAX_VARS, 0.0 for valid, -inf for invalid
    """
    action_mask = [float("-inf")] * NUM_ACTIONS
    var_mask = [float("-inf")] * MAX_VARS

    stack_size = len(state.stack)

    # PUSH_VAR: always possible for unused vars
    unused = [i for i in range(num_vars) if i not in state.vars_used]
    if unused:
        action_mask[ACTION_PUSH_VAR] = 0.0
        for i in unused:
            var_mask[i] = 0.0

    # PUSH_CONST: always possible
    action_mask[ACTION_PUSH_CONST] = 0.0

    # APPLY operations: need >= 2 on stack
    if stack_size >= 2:
        action_mask[ACTION_APPLY_ADD] = 0.0
        action_mask[ACTION_APPLY_SUB] = 0.0
        action_mask[ACTION_APPLY_MUL] = 0.0
        action_mask[ACTION_APPLY_DIV] = 0.0
        # POW only if right operand would be a constant (enforced in training data)
        action_mask[ACTION_APPLY_POW] = 0.0

    # DONE: only if exactly 1 on stack
    if stack_size == 1:
        action_mask[ACTION_DONE] = 0.0

    return (unused, action_mask, var_mask)


# ════════════════════════════════════════════════════
# Model
# ════════════════════════════════════════════════════

class TreeDecoder(nn.Module):
    """Tree-based autoregressive expression decoder."""

    def __init__(self, d_model: int = 128, nhead: int = 4,
                 num_encoder_layers: int = 3, num_decoder_layers: int = 3,
                 max_vars: int = MAX_VARS, num_consts: int = NUM_CONSTS,
                 max_seq_len: int = 24):
        super().__init__()
        self.d_model = d_model
        self.max_vars = max_vars
        self.num_consts = num_consts
        self.max_seq_len = max_seq_len
        self.num_actions = NUM_ACTIONS

        # Input embeddings
        self.symbol_embedding = nn.Embedding(256, d_model, padding_idx=0)
        self.symbol_pos_encoder = PositionalEncoding(d_model, max_len=16)

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Decoder
        self.action_embedding = nn.Embedding(NUM_ACTIONS, d_model, padding_idx=0)
        self.param_embedding = nn.Embedding(max(max_vars, num_consts) + 1, d_model, padding_idx=0)
        self.decoder_pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Output heads
        self.action_head = nn.Linear(d_model, NUM_ACTIONS)
        self.var_head = nn.Linear(d_model, max_vars)
        self.const_head = nn.Linear(d_model, num_consts)

        # Projection for decoder input (action embedding + param embedding → d_model)
        self.decoder_input_proj = nn.Linear(d_model * 2, d_model)

    def encode(self, symbol_ids: torch.Tensor) -> torch.Tensor:
        """Encode input symbols. symbol_ids: (B, num_vars)"""
        emb = self.symbol_embedding(symbol_ids)  # (B, num_vars, d_model)
        emb = self.symbol_pos_encoder(emb)
        memory = self.encoder(emb)
        return memory  # (B, num_vars, d_model)

    def decode_step(self, memory: torch.Tensor, tgt_emb: torch.Tensor,
                    tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Single decode step. tgt_emb: (B, seq_len, d_model)"""
        tgt_emb = self.decoder_pos_encoder(tgt_emb)
        output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        return output  # (B, seq_len, d_model)

    def forward(self, symbol_ids: torch.Tensor,
                action_seq: torch.Tensor,
                param_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass for training.

        Args:
            symbol_ids: (B, num_vars) — variable symbols
            action_seq: (B, seq_len) — action type at each step
            param_seq: (B, seq_len) — parameter at each step

        Returns:
            action_logits: (B, seq_len, NUM_ACTIONS)
            var_logits: (B, seq_len, max_vars)
            const_logits: (B, seq_len, num_consts)
        """
        memory = self.encode(symbol_ids)

        # Embed decoder inputs
        act_emb = self.action_embedding(action_seq)  # (B, seq_len, d_model)
        par_emb = self.param_embedding(param_seq)  # (B, seq_len, d_model)
        combined = torch.cat([act_emb, par_emb], dim=-1)  # (B, seq_len, 2*d_model)
        dec_input = self.decoder_input_proj(combined)  # (B, seq_len, d_model)

        # Causal mask
        seq_len = action_seq.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=action_seq.device)

        output = self.decode_step(memory, dec_input, tgt_mask)  # (B, seq_len, d_model)

        action_logits = self.action_head(output)
        var_logits = self.var_head(output)
        const_logits = self.const_head(output)

        return action_logits, var_logits, const_logits

    def generate(self, symbol_ids: torch.Tensor, num_vars: int,
                 var_names: list[str] | None = None,
                 temperature: float = 0.8, num_samples: int = 5) -> list[str]:
        """
        Generate expressions autoregressively with grammar mask.

        Args:
            symbol_ids: (1, num_vars) or (num_vars,) — input variable ids
            num_vars: number of variables in the input set
            var_names: actual variable name strings for output (defaults to v0, v1...)
            temperature: sampling temperature
            num_samples: number of independent samples to generate

        Returns:
            List of expression strings (deduplicated)
        """
        if symbol_ids.dim() == 1:
            symbol_ids = symbol_ids.unsqueeze(0)

        if var_names is None:
            var_names = [f"v{i}" for i in range(num_vars)]

        device = symbol_ids.device
        memory = self.encode(symbol_ids)  # (1, num_vars, d_model)

        results = []
        seen = set()

        for _ in range(num_samples * 3):  # extra attempts for diversity
            if len(results) >= num_samples:
                break

            state = RPNState(num_vars)
            action_seq = []  # list of (action, param) for decoder context

            for step in range(self.max_seq_len):
                # Build decoder input from actions so far
                if len(action_seq) == 0:
                    # Use SOS (start-of-sequence) token
                    act_ids = torch.full((1, 1), ACTION_SOS, dtype=torch.long, device=device)
                    par_ids = torch.zeros(1, 1, dtype=torch.long, device=device)
                else:
                    act_ids = torch.tensor([[a for a, p in action_seq]], device=device)
                    par_ids = torch.tensor([[p for a, p in action_seq]], device=device)

                act_emb = self.action_embedding(act_ids)
                par_emb = self.param_embedding(par_ids)
                combined = torch.cat([act_emb, par_emb], dim=-1)
                dec_input = self.decoder_input_proj(combined)

                tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                    dec_input.size(1), device=device)
                output = self.decode_step(memory, dec_input, tgt_mask)
                last_out = output[:, -1, :]  # (1, d_model)

                # Predict action
                action_logits = self.action_head(last_out)  # (1, NUM_ACTIONS)
                _, valid_action_mask, valid_var_mask = get_valid_actions(state, num_vars)

                # Apply action mask
                action_mask = torch.tensor([valid_action_mask], device=device)
                masked_logits = action_logits + action_mask

                # Sample action
                probs = F.softmax(masked_logits / temperature, dim=-1)
                action = torch.multinomial(probs, 1).item()

                # Get parameter if needed
                param = 0
                if action == ACTION_PUSH_VAR:
                    var_logits = self.var_head(last_out)  # (1, max_vars)
                    var_mask = torch.tensor([valid_var_mask], device=device)
                    masked_var = var_logits + var_mask
                    var_probs = F.softmax(masked_var / temperature, dim=-1)
                    param = torch.multinomial(var_probs, 1).item()

                elif action == ACTION_PUSH_CONST:
                    const_logits = self.const_head(last_out)
                    const_probs = F.softmax(const_logits / temperature, dim=-1)
                    param = torch.multinomial(const_probs, 1).item()

                # Try to apply action
                try:
                    result = state.apply(action, param, var_names)
                    action_seq.append((action, param))

                    if result is not None:
                        if result not in seen:
                            seen.add(result)
                            results.append(result)
                        break

                except ValueError:
                    # Invalid action — abort this sample
                    break

            # If we ran out of steps with no result, force DONE if stack has 1
            if not state.done and len(state.stack) == 1:
                expr = state.stack[0]
                if expr not in seen:
                    seen.add(expr)
                    results.append(expr)

        return results


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 64, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)


# ════════════════════════════════════════════════════
# Expression → RPN conversion (for training data)
# ════════════════════════════════════════════════════

def expr_to_rpn(expr: str, var_indices: dict[str, int]) -> list[tuple[int, int]]:
    """
    Convert an infix expression string to RPN action sequence.

    Args:
        expr: Expression like "(E*lambda)" or "((c*t)^2)"
        var_indices: Mapping from variable name to index, e.g., {"E": 0, "lambda": 1}

    Returns:
        List of (action, param) pairs, e.g.:
        "E*lambda" → [(PUSH_VAR, 0), (PUSH_VAR, 1), (APPLY_MUL, 0)]
        "(c*t)^2" → [(PUSH_VAR, 0), (PUSH_VAR, 1), (APPLY_MUL, 0),
                      (PUSH_CONST, 3), (APPLY_POW, 0)]
    """
    tokens = tokenize_infix(expr)
    rpn_tokens = infix_to_rpn(tokens)
    return rpn_to_actions(rpn_tokens, var_indices)


def tokenize_infix(expr: str) -> list[str]:
    """Tokenize an infix expression into symbols, operators, parens, numbers."""
    tokens = []
    i = 0
    while i < len(expr):
        if expr[i] in "()+-*/^":
            tokens.append(expr[i])
            i += 1
        elif expr[i].isdigit() or (expr[i] == '-' and i + 1 < len(expr) and expr[i + 1].isdigit()):
            j = i + 1
            while j < len(expr) and (expr[j].isdigit() or expr[j] == '.'):
                j += 1
            tokens.append(expr[i:j])
            i = j
        else:
            # Variable name — read until operator/paren
            j = i
            while j < len(expr) and expr[j] not in "()+-*/^":
                j += 1
            tokens.append(expr[i:j])
            i = j
    return tokens


def infix_to_rpn(tokens: list[str]) -> list[str]:
    """Convert infix tokens to Reverse Polish Notation using shunting-yard."""
    precedence = {"+": 1, "-": 1, "*": 2, "/": 2, "^": 3}
    output = []
    ops = []

    for token in tokens:
        if token == "(":
            ops.append(token)
        elif token == ")":
            while ops and ops[-1] != "(":
                output.append(ops.pop())
            if ops:
                ops.pop()  # remove "("
        elif token in precedence:
            while (ops and ops[-1] != "(" and
                   precedence.get(ops[-1], 0) >= precedence[token]):
                output.append(ops.pop())
            ops.append(token)
        else:
            # Operand (variable or number)
            output.append(token)

    while ops:
        output.append(ops.pop())

    return output


def rpn_to_actions(rpn: list[str], var_indices: dict[str, int]) -> list[tuple[int, int]]:
    """Convert RPN tokens to action sequence."""
    actions = []
    const_map = {str(v): i for i, v in enumerate(CONST_VALUES)}

    for token in rpn:
        if token in var_indices:
            actions.append((ACTION_PUSH_VAR, var_indices[token]))
        elif token in const_map:
            actions.append((ACTION_PUSH_CONST, const_map[token]))
        elif token == "+":
            actions.append((ACTION_APPLY_ADD, 0))
        elif token == "-":
            actions.append((ACTION_APPLY_SUB, 0))
        elif token == "*":
            actions.append((ACTION_APPLY_MUL, 0))
        elif token == "/":
            actions.append((ACTION_APPLY_DIV, 0))
        elif token == "^":
            actions.append((ACTION_APPLY_POW, 0))
        else:
            raise ValueError(f"Unknown RPN token: {token}")

    actions.append((ACTION_DONE, 0))
    return actions


def rpn_to_expr(rpn_actions: list[tuple[int, int]], var_names: list[str]) -> Optional[str]:
    """Convert RPN actions back to an infix expression string."""
    state = RPNState(len(var_names))
    for action, param in rpn_actions:
        try:
            result = state.apply(action, param, var_names)
            if result is not None:
                return result
        except ValueError:
            return None
    return None
