#!/usr/bin/env python3
"""Train the quantum and relativistic domain template generators.

Uses the 7-domain observation database to extract domain-specific
examples and train DomainTemplateGenerator models (~50K params each).

Usage:
  python scripts/training/train_new_domains.py
  python scripts/training/train_new_domains.py --epochs 50 --lr 1e-3
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_project_root))

from src.physics.composer import (
    DomainTemplateGenerator,
    DOMAINS,
    DOMAIN_TEMPLATES,
    extract_domain_examples,
    quantities_to_tensor,
    expression_to_tensor,
    detokenize_expression,
    DOMAIN_QUANTITIES,
    TEMPLATE_VOCAB_SIZE,
    TEMPLATE_PAD_IDX,
    save_domain_generator,
)


class DomainTemplateDataset(torch.utils.data.Dataset):
    def __init__(self, examples, src_max_len=8, tgt_max_len=32):
        self.samples = []
        for ex in examples:
            qty_symbols = sorted(ex["quantities"].keys())
            src = quantities_to_tensor(qty_symbols, max_len=src_max_len)
            tgt = expression_to_tensor(ex["expression"], max_len=tgt_max_len)
            self.samples.append((src, tgt))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def train_domain_generator(
    observations_path: Path,
    domain: str,
    checkpoint_dir: Path,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 4,
    d_model: int = 40,
    nhead: int = 2,
):
    print(f"\n{'=' * 60}")
    print(f"Training {domain.capitalize()} Template Generator")
    print(f"{'=' * 60}")

    examples = extract_domain_examples(str(observations_path), domain)
    print(f"  Training examples: {len(examples)}")

    if len(examples) == 0:
        print(f"  WARNING: No training examples for '{domain}' — using fallback")
        model = DomainTemplateGenerator(d_model=d_model, nhead=nhead)
        return model, {"error": "no_examples"}

    # Show a sample to verify data
    sample = examples[0]
    print(f"  Sample: qties={sorted(sample['quantities'].keys())}, "
          f"expr={sample['expression']}")

    dataset = DomainTemplateDataset(examples)
    dataloader = DataLoader(
        dataset, batch_size=min(batch_size, len(dataset)), shuffle=True
    )

    model = DomainTemplateGenerator(d_model=d_model, nhead=nhead)
    n_params = model.count_parameters()
    print(f"  Parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss(ignore_index=TEMPLATE_PAD_IDX)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    stats = {"train_loss": [], "train_acc": []}

    model.train()
    device = torch.device("cpu")
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

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            src_mask = (src == TEMPLATE_PAD_IDX)
            tgt_mask_pad = (tgt_input == TEMPLATE_PAD_IDX)
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
            optimizer.step()

            epoch_loss += loss.item()
            preds = logits.argmax(dim=-1)
            mask = tgt_output != TEMPLATE_PAD_IDX
            correct = (preds[mask] == tgt_output[mask]).sum().item()
            total = mask.sum().item()
            epoch_correct += correct
            epoch_total += total
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_acc = epoch_correct / max(epoch_total, 1)
        stats["train_loss"].append(avg_loss)
        stats["train_acc"].append(avg_acc)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} — loss: {avg_loss:.4f}, acc: {avg_acc:.4f}")

    # Save
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / f"{domain}_template.pt"
    save_domain_generator(model, str(ckpt_path))
    print(f"  Saved: {ckpt_path}")

    # Evaluate
    model.eval()
    expected = DOMAIN_TEMPLATES.get(domain, "?")
    if examples:
        sample_qty = sorted(examples[0]["quantities"].keys())
        src = quantities_to_tensor(sample_qty, max_len=8).unsqueeze(0).to(device)
        src_mask = (src == TEMPLATE_PAD_IDX)
        with torch.no_grad():
            seqs = model.generate(src, src_padding_mask=src_mask, max_len=32)
            generated = detokenize_expression(seqs[0])
        print(f"  Generated: {generated}")
        print(f"  Expected:  {expected}")

    return model, stats


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Train quantum and relativistic template generators"
    )
    parser.add_argument(
        "--observations",
        default="data/observations/phase_f_7domain.json",
        help="Path to 7-domain observation database",
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Training epochs per domain"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--checkpoint-dir", default="checkpoints", help="Checkpoint output directory"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    observations_path = _project_root / args.observations
    checkpoint_dir = _project_root / args.checkpoint_dir

    print(f"Observations: {observations_path}")
    print(f"Checkpoints:  {checkpoint_dir}")
    print(f"Device:       CPU")

    # Train quantum domain
    quantum_gen, quantum_stats = train_domain_generator(
        observations_path=observations_path,
        domain="quantum",
        checkpoint_dir=checkpoint_dir,
        epochs=args.epochs,
        lr=args.lr,
    )

    # Train relativistic domain
    rel_gen, rel_stats = train_domain_generator(
        observations_path=observations_path,
        domain="relativistic",
        checkpoint_dir=checkpoint_dir,
        epochs=args.epochs,
        lr=args.lr,
    )

    print(f"\n{'=' * 60}")
    print("Training complete!")
    print(f"  Quantum final loss:      {quantum_stats.get('train_loss', [0])[-1]:.4f}")
    print(f"  Relativistic final loss: {rel_stats.get('train_loss', [0])[-1]:.4f}")


if __name__ == "__main__":
    main()
