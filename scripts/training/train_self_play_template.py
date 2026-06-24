#!/usr/bin/env python3
"""Train template generators from self-play expression data.

Trains per-domain DomainTemplateGenerator models on self-play generated
(quantities, domain, expression) triples. Replaces the observation-database
trained checkpoints with generators trained on structural knowledge.

Output:
  checkpoints/self_play_gravity_template.pt
  checkpoints/self_play_spring_template.pt
  checkpoints/self_play_em_template.pt
  checkpoints/self_play_thermal_template.pt
  checkpoints/self_play_template.pt  (combined archive)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_project_root))

from src.physics.composer import (
    DomainTemplateGenerator,
    DOMAIN_TEMPLATES,
    DOMAIN_QUANTITIES,
    quantities_to_tensor,
    expression_to_tensor,
    detokenize_expression,
    TEMPLATE_VOCAB_SIZE,
    TEMPLATE_PAD_IDX,
    TEMPLATE_SOS_IDX,
    TEMPLATE_EOS_IDX,
    TEMPLATE_UNK_IDX,
    save_domain_generator,
)
from src.physics.expression_generator import PRE_1905_QUANTITY_DIMS

# Pre-1905 domain mapping for training
PRE_1905_DOMAINS = ["gravity", "spring", "em", "thermal"]
PRE_1905_SYMBOLS = set(PRE_1905_QUANTITY_DIMS.keys())

# Map expression generator domain labels to composer domain labels
DOMAIN_MAP = {
    "gravity": "gravity",
    "spring": "spring",
    "thermal": "thermal",
    "gas_law": "thermal",
    "em": "em",
    "classical": "gravity",
    "mechanics": "gravity",
}


def _normalize_domain(raw_domain: str) -> str:
    """Map expression generator domain labels to standard composer domains."""
    return DOMAIN_MAP.get(raw_domain, "gravity")


class SelfPlayTemplateDataset(torch.utils.data.Dataset):
    """Dataset for training template generator from self-play data."""

    def __init__(
        self,
        examples: list[dict],
        domain: str,
        src_max_len: int = 8,
        tgt_max_len: int = 32,
    ):
        self.samples: list[tuple[torch.Tensor, torch.Tensor]] = []

        for ex in examples:
            ex_domain = _normalize_domain(ex.get("domain", ""))
            if ex_domain != domain:
                continue

            qty_symbols = ex.get("quantities", [])
            # Filter to pre-1905 symbols only
            qty_symbols = [q for q in qty_symbols if q in PRE_1905_SYMBOLS]
            if len(qty_symbols) < 2:
                continue

            expr = ex.get("expression", "")
            if not expr or not expr.strip():
                continue

            # Source: domain-relevant quantities (sorted for consistency)
            qty_sorted = sorted(qty_symbols)
            src = quantities_to_tensor(qty_sorted, max_len=src_max_len)

            # Target: expression with SOS/EOS
            tgt = expression_to_tensor(expr, max_len=tgt_max_len)

            self.samples.append((src, tgt))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]


def load_self_play_data(
    data_path: str | Path,
    max_per_domain: int | None = 2000,
) -> list[dict]:
    """Load self-play training data from JSONL file.

    Args:
        data_path: Path to JSONL data file.
        max_per_domain: Cap examples per domain for training speed.
            None means use all examples.
    """
    examples: list[dict] = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    if max_per_domain is not None:
        domain_counts: dict[str, int] = defaultdict(int)
        subsampled: list[dict] = []
        for ex in examples:
            domain = _normalize_domain(ex.get("domain", ""))
            if domain_counts[domain] < max_per_domain:
                subsampled.append(ex)
                domain_counts[domain] += 1
        examples = subsampled

    return examples


def train_domain_generator_from_self_play(
    examples: list[dict],
    domain: str,
    checkpoint_dir: str | Path,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 8,
    device: torch.device = torch.device("cpu"),
    d_model: int = 48,
    nhead: int = 2,
) -> tuple[DomainTemplateGenerator, dict]:
    """Train a domain template generator on self-play data.

    Returns (trained_model, training_stats).
    """
    print(f"\n{'=' * 60}")
    print(f"Training {domain.capitalize()} Template Generator (self-play)")
    print(f"=" * 60)

    dataset = SelfPlayTemplateDataset(examples, domain)
    n_examples = len(dataset)
    print(f"  Training examples: {n_examples}")

    if n_examples == 0:
        print(f"  WARNING: No training examples for domain '{domain}'")
        model = DomainTemplateGenerator(d_model=d_model, nhead=nhead)
        return model, {"error": "no_examples"}

    dataloader = DataLoader(
        dataset, batch_size=min(batch_size, len(dataset)), shuffle=True
    )

    model = DomainTemplateGenerator(d_model=d_model, nhead=nhead)
    n_params = model.count_parameters()
    print(f"  Parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss(ignore_index=TEMPLATE_PAD_IDX)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    stats: dict[str, list[float]] = {"train_loss": [], "train_acc": []}

    model.train()
    model.to(device)

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        n_batches = 0

        for src, tgt in dataloader:
            src = src.to(device)
            tgt = tgt.to(device)

            optimizer.zero_grad()

            # Teacher forcing: input tgt[:, :-1], predict tgt[:, 1:]
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            src_mask = src == TEMPLATE_PAD_IDX
            tgt_mask_pad = tgt_input == TEMPLATE_PAD_IDX
            tgt_causal = torch.triu(
                torch.ones(tgt_input.size(1), tgt_input.size(1), device=device)
                * float("-inf"),
                diagonal=1,
            )

            logits = model.forward(
                src, tgt_input,
                src_padding_mask=src_mask,
                tgt_padding_mask=tgt_mask_pad,
                tgt_mask=tgt_causal,
            )

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_output.reshape(-1),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

            # Accuracy
            preds = logits.argmax(dim=-1)
            mask = tgt_output != TEMPLATE_PAD_IDX
            correct = (preds[mask] == tgt_output[mask]).sum().item()
            total = mask.sum().item()
            epoch_correct += correct
            epoch_total += total
            n_batches += 1

        scheduler.step()

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_acc = epoch_correct / max(epoch_total, 1) if epoch_total > 0 else 0.0
        stats["train_loss"].append(avg_loss)
        stats["train_acc"].append(avg_acc)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}/{epochs} — "
                f"loss: {avg_loss:.4f}, acc: {avg_acc:.4f}",
                flush=True,
            )

    # Save
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / f"self_play_{domain}_template.pt"
    save_domain_generator(model, str(ckpt_path))
    print(f"  Saved: {ckpt_path}")

    # Evaluate: generate template and show
    model.eval()
    if n_examples > 0:
        sample = dataset[0]
        src = sample[0].unsqueeze(0).to(device)
        src_mask = (src == TEMPLATE_PAD_IDX)
        with torch.no_grad():
            seqs = model.generate(src, src_padding_mask=src_mask, max_len=32)
            generated = detokenize_expression(seqs[0])
        expected = DOMAIN_TEMPLATES.get(domain, "?")
        print(f"  Sample generation: {generated}")
        print(f"  Hardcoded template: {expected}")

    return model, stats


def save_combined_checkpoint(
    models: dict[str, DomainTemplateGenerator],
    checkpoint_dir: str | Path,
) -> None:
    """Save all self-play trained generators as a single checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    state = {
        f"{domain}_state_dict": model.state_dict()
        for domain, model in models.items()
    }
    state["domains"] = list(models.keys())
    state["description"] = "Self-play trained template generators (Phase D)"

    path = checkpoint_dir / "self_play_template.pt"
    torch.save(state, path)
    print(f"\nSaved combined checkpoint: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train template generators from self-play data"
    )
    parser.add_argument(
        "--data", type=str, default="data/self_play_training.jsonl",
        help="Path to self-play training data JSONL",
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Training epochs per domain (default: 50)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Batch size (default: 8)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--d-model", type=int, default=48,
        help="Hidden dimension (default: 48)",
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="checkpoints",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--domains", type=str, nargs="+",
        default=["gravity", "spring", "em", "thermal"],
        help="Domains to train (default: pre-1905 domains only)",
    )
    parser.add_argument(
        "--max-per-domain", type=int, default=2000,
        help="Max examples per domain for training speed (default: 2000, 0=unlimited)",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    data_path = _project_root / args.data
    checkpoint_dir = _project_root / args.checkpoint_dir
    device = torch.device("cpu")
    torch.set_num_threads(4)  # CPU only, limit threads

    print(f"Device: {device}", flush=True)
    print(f"Data: {data_path}", flush=True)
    print(f"Checkpoints: {checkpoint_dir}", flush=True)

    # Load training data
    print(f"\nLoading self-play training data...", flush=True)
    max_per = args.max_per_domain if args.max_per_domain > 0 else None
    examples = load_self_play_data(data_path, max_per_domain=max_per)
    print(f"Loaded {len(examples)} total examples", flush=True)

    # Show domain distribution
    domain_counts: dict[str, int] = defaultdict(int)
    for ex in examples:
        d = _normalize_domain(ex.get("domain", ""))
        domain_counts[d] += 1
    print(f"Domain distribution (normalized): {dict(sorted(domain_counts.items()))}")

    # Train per-domain generators
    models: dict[str, DomainTemplateGenerator] = {}
    all_stats: dict[str, dict] = {}

    for domain in args.domains:
        model, stats = train_domain_generator_from_self_play(
            examples=examples,
            domain=domain,
            checkpoint_dir=checkpoint_dir,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            device=device,
            d_model=args.d_model,
            nhead=2,
        )
        models[domain] = model
        all_stats[domain] = stats

    # Save combined checkpoint
    save_combined_checkpoint(models, checkpoint_dir)

    # Final summary
    print(f"\n{'=' * 60}")
    print("Training Summary")
    print(f"{'=' * 60}")
    total_params = sum(m.count_parameters() for m in models.values())
    print(f"Total parameters: {total_params:,}")
    print(f"Domains trained: {list(models.keys())}")
    for domain, stats in all_stats.items():
        if "error" in stats:
            print(f"  {domain}: SKIPPED ({stats['error']})")
        else:
            final_loss = stats["train_loss"][-1] if stats["train_loss"] else float("nan")
            final_acc = stats["train_acc"][-1] if stats["train_acc"] else 0.0
            print(f"  {domain}: final_loss={final_loss:.4f}, final_acc={final_acc:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
