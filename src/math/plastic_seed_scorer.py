"""
Plastic Seed Scorer v2 — per-symbol-pair learnable memory.

Replaces global Hebbian matrix with a sparse key-value store.
Each (symbol_set, expression) pair gets its own plastic bias,
updated via outcome-weighted increments. No backprop, no labels.

W_p is a dict: {(frozenset(symbols), expr_hash) → bias}
Sparse, interpretable, zero cross-talk between unrelated pairs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import hashlib
import math

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ════════════════════════════════════════════════════════
# Plastic Seed Scorer v2
# ════════════════════════════════════════════════════════

class PlasticSeedScorer(nn.Module):
    """Seed scorer with sparse plastic memory.

    Frozen branch: pre-trained grammar scorer.
    Plastic branch: key-value store mapping (symbols, expr) → bias.
    """

    def __init__(self, frozen_scorer: nn.Module, d_model: int = 64,
                 plasticity_rate: float = 0.02):
        super().__init__()
        self.frozen = frozen_scorer
        self.d_model = d_model

        # Plastic memory: {(sym_key, expr_key) → bias}
        self.memory: dict[tuple, float] = {}

        # Plasticity rate
        self.eta = nn.Parameter(torch.tensor(plasticity_rate), requires_grad=False)

        # Last accessed key for update
        self._last_key: Optional[tuple] = None

    def _make_key(self, symbols: list[str], expression: str) -> tuple:
        """Create a structural key — learns at the FORM level, not token level.

        Key is (num_symbols, structural_pattern) so that:
        - E*lambda and K*nu share the same key (both are 2-var products)
        - Plastic generalizes across domains
        """
        n = len(symbols)
        pattern = self._structural_pattern(expression)
        return (n, pattern)

    @staticmethod
    def _structural_pattern(expr: str) -> str:
        """Extract structural pattern from expression.

        Returns abstract form like 'a*b', 'a/b', 'a*b/c', 'a^2*b'.
        """
        # Count operators at top level
        import re
        # Remove parentheses for pattern extraction
        clean = expr.replace('(', '').replace(')', '')

        # Count operators
        ops = re.findall(r'[+\-*/^]', clean)
        op_types = ''.join(sorted(set(ops)))

        # Count variables vs numbers
        tokens = re.findall(r'\b[a-zA-Z_]\w*\b|\d+\.?\d*', clean)
        funcs = {"sin", "cos", "sqrt", "exp", "log", "abs", "tan"}
        vars_only = [t for t in tokens if t not in funcs and not t.replace('.','').replace('-','').isdigit()]
        num_count = len(tokens) - len(vars_only)

        n_vars = len(set(vars_only))

        if op_types == '*' and n_vars == 2:
            return 'a*b'
        elif op_types == '/' and n_vars == 2:
            return 'a/b'
        elif op_types == '*' and '^' in clean and n_vars <= 2:
            return 'a^2*b' if num_count > 0 else 'a*b^2'
        elif '*' in op_types and '/' in op_types:
            return 'a*b/c'
        elif op_types == '+' and n_vars == 2:
            return 'a+b'
        elif op_types == '-' and n_vars == 2:
            return 'a-b'
        elif '^' in op_types and n_vars == 1:
            return 'a^2'
        else:
            return f'custom_{op_types}_{n_vars}'

    def forward(self, frozen_score: float, symbols: list[str],
                expression: str) -> float:
        """Compute plastic-adjusted score for a single candidate."""
        key = self._make_key(symbols, expression)
        self._last_key = key

        # Retrieve plastic bias (default 0)
        bias = self.memory.get(key, 0.0)

        # Apply bias with sigmoid squash to keep in [0, 1]
        adjusted = frozen_score + bias
        return 1.0 / (1.0 + math.exp(-5.0 * (adjusted - 0.5)))
    def update(self, outcome: float, symbols: Optional[list[str]] = None,
               expression: Optional[str] = None):
        """Update plastic bias.

        If symbols/expression provided, uses those directly.
        Otherwise falls back to _last_key from most recent forward().
        """
        if symbols is not None and expression is not None:
            key = self._make_key(symbols, expression)
        elif self._last_key is not None:
            key = self._last_key
        else:
            return
        current = self.memory.get(key, 0.0)

        # Hebbian-like: outcome-weighted increment
        delta = self.eta.item() * outcome
        self.memory[key] = current + delta

        # Decay: small forgetting factor prevents unbounded growth
        self.memory[key] *= 0.999

        self._last_key = None

    def reset(self):
        """Clear all plastic memory."""
        self.memory.clear()
        self._last_key = None

    def get_plastic_norm(self) -> float:
        """Return L2 norm of all plastic biases."""
        if not self.memory:
            return 0.0
        return math.sqrt(sum(v * v for v in self.memory.values()))

    def get_top_pairs(self, n: int = 10) -> list[tuple[tuple, float]]:
        """Return top-n plastic memory entries by bias."""
        sorted_items = sorted(self.memory.items(), key=lambda x: -abs(x[1]))
        return [(k, v) for k, v in sorted_items[:n]]


# ════════════════════════════════════════════════════════
# Integration wrapper — drop-in replacement for seed_scorer
# ════════════════════════════════════════════════════════

_plastic_model: Optional[PlasticSeedScorer] = None
_plastic_token_map: dict[str, int] = {}
_plastic_device = "xpu" if torch.xpu.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

# Frozen scorer wrapper (keeps the original API)
_frozen_scorer = None


class _FrozenWrapper:
    """Wraps the original seed scorer to expose a simple score method."""

    def __init__(self, model, token_map):
        self._model = model
        self._token_map = token_map
        self._alias = {"K_max": "k", "nu": "n", "lambda": "l", "gamma": "g",
                       "E_peak": "e", "hbar": "q", "omega": "w"}
        self._device = _plastic_device

    def score(self, symbols: list[str], expression: str) -> float:
        """Compute frozen score (no plastic)."""
        aliased_syms = [self._alias.get(s, s) for s in symbols]
        aliased_expr = expression
        for orig, alias in sorted(self._alias.items(), key=lambda x: -len(x[0])):
            aliased_expr = aliased_expr.replace(orig, alias)

        src = _make_tensor(aliased_syms, self._token_map, 16, 'symbols')
        tgt = _make_tensor_expr(aliased_expr, self._token_map, 32)

        with torch.no_grad():
            score = self._model(src, tgt).item()
        return score


def _make_tensor(symbols: list[str], token_map: dict, max_len: int, kind: str) -> torch.Tensor:
    """Create input tensor for model."""
    result = [token_map.get("<sos>", 1)]
    for s in symbols:
        result.append(token_map.get(s, 0))
    result.append(token_map.get("<eos>", 2))
    while len(result) < max_len:
        result.append(token_map.get("<pad>", 0))
    return torch.tensor([result[:max_len]], device=_plastic_device)


def _make_tensor_expr(expr: str, token_map: dict, max_len: int) -> torch.Tensor:
    """Tokenize expression for model input."""
    result = [token_map.get("<sos>", 1)]
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
    result.append(token_map.get("<eos>", 2))
    while len(result) < max_len:
        result.append(token_map.get("<pad>", 0))
    return torch.tensor([result[:max_len]], device=_plastic_device)


def _load_plastic():
    """Load plastic seed scorer."""
    global _plastic_model, _plastic_token_map, _frozen_scorer

    if _plastic_model is not None:
        return True

    ckpt_path = PROJECT_ROOT / "checkpoints" / "math_self_play" / "seed_scorer.pt"
    if not ckpt_path.exists():
        return False

    try:
        ckpt = torch.load(ckpt_path, map_location=_plastic_device, weights_only=False)
        from scripts.training.train_seed_scorer import SeedScorer

        _plastic_token_map = ckpt["token_map"]
        config = ckpt.get("config", {"d_model": 64, "nhead": 4, "num_layers": 2, "max_seq_len": 64})

        frozen = SeedScorer(vocab_size=ckpt["vocab_size"], **config)
        frozen.load_state_dict(ckpt["model_state_dict"])
        frozen.eval()
        frozen.to(_plastic_device)

        _frozen_scorer = _FrozenWrapper(frozen, _plastic_token_map)
        _plastic_model = PlasticSeedScorer(None, d_model=config["d_model"])
        return True
    except Exception as e:
        print(f"Plastic load error: {e}")
        return False


def score_seed(symbols: list[str], expression: str) -> float:
    """Score with plastic adaptation. Falls back to neutral if unloaded."""
    if not _load_plastic() or _frozen_scorer is None:
        return 0.5

    frozen_score = _frozen_scorer.score(symbols, expression)
    plastic_score = _plastic_model.forward(frozen_score, symbols, expression)
    return plastic_score


def update_plastic(outcome: float, symbols: list[str] | None = None,
                   expression: str | None = None):
    """Update plastic memory with explicit (symbols, expression, outcome)."""
    if _plastic_model is None:
        _load_plastic()
    if _plastic_model is not None:
        _plastic_model.update(outcome, symbols, expression)


def get_plastic_state() -> dict:
    """Return plastic model state."""
    if _plastic_model is None:
        return {"loaded": False, "norm": 0.0, "entries": 0}
    return {
        "loaded": True,
        "norm": _plastic_model.get_plastic_norm(),
        "entries": len(_plastic_model.memory),
        "eta": _plastic_model.eta.item(),
    }


def get_top_plastic_pairs(n: int = 10) -> list:
    """Return top plastic memory entries."""
    if _plastic_model is None:
        return []
    return _plastic_model.get_top_pairs(n)


def reset_plastic():
    """Clear plastic memory."""
    if _plastic_model is not None:
        _plastic_model.reset()
