"""Seed scorer — scores sub-expression candidates for relevance.

Loads checkpoint from checkpoints/math_self_play/seed_scorer.pt.
Provides a simple function: score_seed(symbols, expression) -> float.
"""

from __future__ import annotations

from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "math_self_play" / "seed_scorer.pt"

_model = None
_token_map: dict[str, int] = {}
_device = "xpu" if torch.xpu.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def _load():
    global _model, _token_map
    if _model is not None:
        return True
    if not CHECKPOINT_PATH.exists():
        return False
    try:
        ckpt = torch.load(CHECKPOINT_PATH, map_location=_device, weights_only=False)
        from scripts.training.train_seed_scorer import SeedScorer
        _token_map = ckpt["token_map"]
        config = ckpt.get("config", {"d_model": 64, "nhead": 4, "num_layers": 2, "max_seq_len": 64})
        _model = SeedScorer(vocab_size=ckpt["vocab_size"], **config)
        _model.load_state_dict(ckpt["model_state_dict"])
        _model.eval()
        return True
    except Exception:
        return False


def tokenize(expr: str, max_len: int = 32) -> list[int]:
    result = [_token_map.get("<sos>", 1)]
    i = 0
    while i < len(expr) and len(result) < max_len - 1:
        matched = False
        for length in range(min(4, len(expr) - i), 0, -1):
            token = expr[i:i+length]
            if token in _token_map:
                result.append(_token_map[token])
                i += length
                matched = True
                break
        if not matched:
            result.append(_token_map.get(expr[i], 0))
            i += 1
    result.append(_token_map.get("<eos>", 2))
    while len(result) < max_len:
        result.append(_token_map.get("<pad>", 0))
    return result[:max_len]


def tokenize_symbols(symbols: list[str], max_len: int = 16) -> list[int]:
    result = [_token_map.get("<sos>", 1)]
    for s in symbols:
        result.append(_token_map.get(s, 0))
    result.append(_token_map.get("<eos>", 2))
    while len(result) < max_len:
        result.append(_token_map.get("<pad>", 0))
    return result[:max_len]


def score_seed(symbols: list[str], expression: str) -> float:
    """Score a sub-expression candidate. Returns 0-1.

    Returns 0.5 (neutral) if model is not loaded.
    """
    if not _load():
        return 0.5

    # Alias multi-char symbols to single-char (matching training)
    _ALIAS = {"K_max": "k", "nu": "n", "lambda": "l", "gamma": "g",
              "E_peak": "e", "hbar": "q", "omega": "w"}
    aliased_syms = [_ALIAS.get(s, s) for s in symbols]
    aliased_expr = expression
    for orig, alias in sorted(_ALIAS.items(), key=lambda x: -len(x[0])):
        aliased_expr = aliased_expr.replace(orig, alias)

    src = torch.tensor([tokenize_symbols(aliased_syms)], device=_device)
    tgt = torch.tensor([tokenize(aliased_expr)], device=_device)

    with torch.no_grad():
        score = _model(src, tgt).item()
    return score


def score_seeds(symbols: list[str], expressions: list[str]) -> list[tuple[float, str]]:
    """Score multiple candidates at once. Returns sorted (score, expr) pairs."""
    if not _load():
        return [(0.5, e) for e in expressions]

    _ALIAS = {"K_max": "k", "nu": "n", "lambda": "l", "gamma": "g",
              "E_peak": "e", "hbar": "q", "omega": "w"}
    aliased_syms = [_ALIAS.get(s, s) for s in symbols]

    batch_size = len(expressions)
    max_src = 16
    max_tgt = 32

    src = torch.zeros(batch_size, max_src, dtype=torch.long, device=_device)
    tgt = torch.zeros(batch_size, max_tgt, dtype=torch.long, device=_device)

    for i, expr in enumerate(expressions):
        aliased_expr = expr
        for orig, alias in sorted(_ALIAS.items(), key=lambda x: -len(x[0])):
            aliased_expr = aliased_expr.replace(orig, alias)
        s = tokenize_symbols(aliased_syms)
        t = tokenize(aliased_expr)
        src[i, :len(s)] = torch.tensor(s)
        tgt[i, :len(t)] = torch.tensor(t)

    with torch.no_grad():
        scores = _model(src, tgt).tolist()

    return sorted(zip(scores, expressions), key=lambda x: -x[0])
