"""Beam guider — predicts whether composing two sub-expressions is productive.

Loads checkpoint from checkpoints/math_self_play/beam_guider.pt.
"""

from __future__ import annotations

from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "math_self_play" / "beam_guider.pt"

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
        from scripts.training.train_beam_guider import BeamGuider
        _token_map = ckpt["token_map"]
        config = ckpt.get("config", {"d_model": 64, "nhead": 4, "num_layers": 2})
        _model = BeamGuider(vocab_size=ckpt["vocab_size"], **config)
        _model.load_state_dict(ckpt["model_state_dict"])
        _model.eval()
        return True
    except Exception:
        return False


def tokenize_expr(expr: str, max_len: int = 32) -> list[int]:
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


def should_explore(left_expr: str, right_expr: str, operator: str, threshold: float = 0.2) -> bool:
    """Return True if this composition is worth evaluating against data."""
    if not _load():
        return True  # if no model, explore everything

    op_token = _token_map.get(operator, 0)
    if op_token == 0:
        return True

    left = torch.tensor([tokenize_expr(left_expr)], device=_device)
    right = torch.tensor([tokenize_expr(right_expr)], device=_device)
    op = torch.tensor([[op_token]], device=_device)

    with torch.no_grad():
        score = _model(left, right, op).item()

    return score >= threshold


def batch_should_explore(
    triples: list[tuple[str, str, str]], threshold: float = 0.2
) -> list[bool]:
    """Batch version for efficiency."""
    if not _load():
        return [True] * len(triples)

    max_len = 32
    batch_size = len(triples)

    left = torch.zeros(batch_size, max_len, dtype=torch.long, device=_device)
    right = torch.zeros(batch_size, max_len, dtype=torch.long, device=_device)
    op = torch.zeros(batch_size, 1, dtype=torch.long, device=_device)

    for i, (l_expr, r_expr, operator) in enumerate(triples):
        lt = tokenize_expr(l_expr)
        rt = tokenize_expr(r_expr)
        left[i, :len(lt)] = torch.tensor(lt)
        right[i, :len(rt)] = torch.tensor(rt)
        op[i, 0] = _token_map.get(operator, 0)

    with torch.no_grad():
        scores = _model(left, right, op).tolist()

    return [s >= threshold for s in scores]
