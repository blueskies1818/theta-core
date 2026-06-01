"""Checkpoint save/load utilities."""

from pathlib import Path
from typing import Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


def save_checkpoint(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    optimizer: Optional[torch.optim.Optimizer],
    step: int,
    save_dir: Path,
    metrics: Optional[dict] = None,
) -> Path:
    """Save a training checkpoint."""
    save_dir = Path(save_dir)
    checkpoint_dir = save_dir / f"checkpoint-{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)

    if optimizer is not None:
        torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")

    if metrics is not None:
        import json

        with open(checkpoint_dir / "metrics.json", "w") as f:
            json.dump(metrics, f)

    torch.save({"step": step}, checkpoint_dir / "training_state.pt")

    return checkpoint_dir


def load_checkpoint(
    model: PreTrainedModel,
    checkpoint_dir: Path,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> int:
    """Load a training checkpoint. Returns the step number."""
    checkpoint_dir = Path(checkpoint_dir)

    from transformers import AutoModelForCausalLM

    state = torch.load(checkpoint_dir / "training_state.pt", map_location="cpu")
    step = state["step"]

    if optimizer is not None and (checkpoint_dir / "optimizer.pt").exists():
        optimizer.load_state_dict(
            torch.load(checkpoint_dir / "optimizer.pt", map_location="cpu")
        )

    return step
