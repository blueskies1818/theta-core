#!/usr/bin/env python3
"""Train Hidden Variable Proposer on self-play data.

Phase E: Loads self-play training data and trains the HiddenVariableProposer
MLP (~3K+ params) from scratch. Saves checkpoint to
checkpoints/self_play_hidden_var.pt.

Usage:
    python scripts/training/train_hidden_var_selfplay.py
    python scripts/training/train_hidden_var_selfplay.py --epochs 500 --lr 0.001
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))

from src.physics.hidden_variables import (
    HiddenVariableProposer,
    NUM_SHAPES, NUM_VAR_TYPES, NUM_TRANSFORMS,
    NUM_HV_QUANTITIES, NUM_HV_DOMAINS,
    SHAPE_TO_IDX, VAR_TYPE_TO_IDX, TRANSFORM_TO_IDX,
    HV_DOMAIN_TO_IDX, HV_QTY_TO_IDX,
)


class SelfPlayProposer(nn.Module):
    """Small MLP proposer (~3K params) for hidden variable type prediction.

    Input: residual_signature(NUM_SHAPES + 4) + quantity_vector(NUM_HV_QUANTITIES)
           + domain_onehot(NUM_HV_DOMAINS)
    Output: var_type_logits(NUM_VAR_TYPES) + confidence
    """

    def __init__(self, hidden_dim: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        self.input_dim = NUM_SHAPES + 4 + NUM_HV_QUANTITIES + NUM_HV_DOMAINS
        self.output_dim = NUM_VAR_TYPES + 1

        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, self.output_dim)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        h = self.dropout(h)
        h = F.relu(self.fc2(h))
        h = self.dropout(h)
        return self.fc3(h)

    def predict(
        self, x: torch.Tensor, *, temperature: float = 0.15,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (var_type_probs, confidence) for batch."""
        output = self.forward(x)
        var_logits = output[:, :NUM_VAR_TYPES]
        conf_logits = output[:, NUM_VAR_TYPES]
        var_probs = F.softmax(var_logits / max(temperature, 1e-8), dim=-1)
        confidence = torch.sigmoid(conf_logits)
        return var_probs, confidence

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def load_selfplay_data(data_path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Load self-play training tensors from disk."""
    checkpoint = torch.load(data_path, map_location="cpu")
    return checkpoint["inputs"], checkpoint["targets"]


def train_proposer(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    proposer: SelfPlayProposer | None = None,
    epochs: int = 300,
    lr: float = 0.003,
    batch_size: int = 1024,
    device: str = "cpu",
    checkpoint_path: str | None = None,
    validation_split: float = 0.1,
) -> SelfPlayProposer:
    """Train self-play proposer on residual→var_type data."""
    if proposer is None:
        proposer = SelfPlayProposer()
    proposer.to(device)
    proposer.train()

    n = inputs.size(0)
    n_val = int(n * validation_split)
    n_train = n - n_val

    # Shuffle
    perm = torch.randperm(n)
    inputs = inputs[perm]
    targets = targets[perm]

    train_inputs = inputs[:n_train].to(device)
    train_targets = targets[:n_train].to(device)
    val_inputs = inputs[n_train:].to(device) if n_val > 0 else None
    val_targets = targets[n_train:].to(device) if n_val > 0 else None

    param_count = proposer.count_parameters()
    print(f"  Training SelfPlayProposer ({param_count:,d} params)")
    print(f"  Train: {n_train:,d}, Val: {n_val:,d}, Batch: {batch_size}")
    print(f"  Epochs: {epochs}, LR: {lr}, Device: {device}")

    optimizer = torch.optim.Adam(proposer.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    var_loss_fn = nn.CrossEntropyLoss()
    conf_loss_fn = nn.BCEWithLogitsLoss()

    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(epochs):
        proposer.train()
        total_loss = 0.0
        num_batches = 0

        for i in range(0, n_train, batch_size):
            batch_inputs = train_inputs[i:i + batch_size]
            batch_targets = train_targets[i:i + batch_size]

            optimizer.zero_grad()
            output = proposer(batch_inputs)

            var_logits = output[:, :NUM_VAR_TYPES]
            conf_logits = output[:, NUM_VAR_TYPES]

            var_targets = batch_targets[:, :NUM_VAR_TYPES].argmax(dim=-1)
            conf_targets = batch_targets[:, NUM_VAR_TYPES]

            loss = (var_loss_fn(var_logits, var_targets)
                    + 0.2 * conf_loss_fn(conf_logits, conf_targets))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(proposer.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(num_batches, 1)

        # Validation
        if val_inputs is not None and (epoch + 1) % 10 == 0:
            proposer.eval()
            with torch.no_grad():
                val_output = proposer(val_inputs)
                val_var_logits = val_output[:, :NUM_VAR_TYPES]
                val_var_targets = val_targets[:, :NUM_VAR_TYPES].argmax(dim=-1)
                val_acc = (val_var_logits.argmax(-1) == val_var_targets).float().mean()
                if val_acc > best_val_acc:
                    best_val_acc = val_acc.item()

            if (epoch + 1) % 50 == 0:
                elapsed = time.time() - start_time
                print(f"  epoch {epoch+1:4d}/{epochs}  "
                      f"loss={avg_loss:.4f}  val_acc={val_acc.item():.3f}  "
                      f"best_val_acc={best_val_acc:.3f}  "
                      f"elapsed={elapsed:.0f}s")
        elif (epoch + 1) % 50 == 0:
            elapsed = time.time() - start_time
            print(f"  epoch {epoch+1:4d}/{epochs}  "
                      f"loss={avg_loss:.4f}  elapsed={elapsed:.0f}s")

    # Final evaluation
    proposer.eval()
    with torch.no_grad():
        train_output = proposer(train_inputs)
        train_var = train_output[:, :NUM_VAR_TYPES].argmax(-1)
        train_tgt = train_targets[:, :NUM_VAR_TYPES].argmax(-1)
        train_acc = (train_var == train_tgt).float().mean().item()

        val_acc_str = "N/A"
        if val_inputs is not None:
            val_output = proposer(val_inputs)
            val_var = val_output[:, :NUM_VAR_TYPES].argmax(-1)
            val_tgt = val_targets[:, :NUM_VAR_TYPES].argmax(-1)
            val_acc_str = f"{((val_var == val_tgt).float().mean().item() * 100):.1f}%"

    elapsed = time.time() - start_time
    print(f"\n  Training complete in {elapsed:.0f}s")
    print(f"  Train var accuracy: {train_acc*100:.1f}%")
    print(f"  Val var accuracy:   {val_acc_str}")

    if checkpoint_path:
        save_path = Path(checkpoint_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": proposer.state_dict(),
            "input_dim": proposer.input_dim,
            "output_dim": proposer.output_dim,
            "num_shapes": NUM_SHAPES,
            "num_var_types": NUM_VAR_TYPES,
            "num_hv_quantities": NUM_HV_QUANTITIES,
            "num_hv_domains": NUM_HV_DOMAINS,
            "shape_to_idx": SHAPE_TO_IDX,
            "var_type_to_idx": VAR_TYPE_TO_IDX,
            "domain_to_idx": HV_DOMAIN_TO_IDX,
            "qty_to_idx": HV_QTY_TO_IDX,
            "train_accuracy": train_acc,
            "version": "self_play_v1",
        }, save_path)
        print(f"  Saved checkpoint to {save_path}")

    return proposer


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Train Hidden Variable Proposer on self-play data"
    )
    parser.add_argument(
        "--data", type=str,
        default=str(_project_root / "data" / "self_play_hidden_var_data.pt"),
        help="Path to training data"
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default=str(_project_root / "checkpoints" / "self_play_hidden_var.pt"),
        help="Output checkpoint path"
    )
    parser.add_argument(
        "--epochs", type=int, default=300,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=0.003,
        help="Learning rate"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1024,
        help="Batch size"
    )
    parser.add_argument(
        "--threads", type=int, default=6,
        help="Number of CPU threads (max 6 for training)"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device (cpu only per task specs)"
    )
    args = parser.parse_args()

    torch.set_num_threads(args.threads)

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: Data file not found: {data_path}")
        print("Run generate_hidden_var_selfplay_data.py first.")
        sys.exit(1)

    print(f"Loading self-play data from {data_path}...")
    inputs, targets = load_selfplay_data(str(data_path))
    print(f"  Loaded {inputs.size(0):,d} examples")
    print(f"  Input dim:  {inputs.size(1)}")
    print(f"  Target dim: {targets.size(1)}")
    print()

    # Show class distribution
    from src.physics.hidden_variables import VAR_TYPES, IDX_TO_VAR_TYPE
    var_targets = targets[:, :NUM_VAR_TYPES].argmax(dim=-1)
    for i, vt_name in enumerate(VAR_TYPES):
        count = (var_targets == i).sum().item()
        pct = count / var_targets.size(0) * 100
        print(f"  {vt_name}: {count:,d} ({pct:.1f}%)")
    print()

    proposer = train_proposer(
        inputs, targets,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=args.device,
        checkpoint_path=str(args.checkpoint),
    )

    # Quick sanity: test on a few synthetic inputs
    print("\n  Sanity check predictions...")
    proposer.eval()
    with torch.no_grad():
        test_input = torch.randn(4, proposer.input_dim)
        probs, conf = proposer.predict(test_input)
        print(f"  Var probs shape: {probs.shape}")
        print(f"  Confidence: {conf.tolist()}")


if __name__ == "__main__":
    main()
