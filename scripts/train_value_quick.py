#!/usr/bin/env python3
"""Quick train value network — uses GNN checkpoint directly, no graph needed.

Loads the GNN checkpoint, extracts the GoalEncoder, and trains the ValueHead
on keyword-based goal encodings from proof_step_pairs. This avoids the slow
graph-loading + GNN-embedding step that dominates the full training script.

Training signal:
  POSITIVE: goal from proof_step_pairs (proven theorem) → V=1.0
  NEGATIVE: randomly perturbed goals → V=0.0

Output: checkpoints/value_network.pt
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.explorer.gnn_encoder import GNNEncoder, GoalEncoder
from src.explorer.value_network import ValueNetwork


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def train(gnn_checkpoint: str, pairs_path: str, output_path: str,
          num_samples=5000, epochs=20, batch_size=256, lr=1e-3):
    """Train value head on goal encodings."""

    print("Loading GNN checkpoint...")
    gnn = GNNEncoder.load(gnn_checkpoint)
    gnn.eval()
    n_params = sum(p.numel() for p in gnn.parameters())
    print(f"  GNN: {n_params:,} params, hidden_dim={gnn.config.hidden_dim}")

    enc_dim = gnn.config.hidden_dim
    print(f"  Encoder output dim: {enc_dim}")

    print(f"\nLoading proof step pairs...")
    pairs = load_jsonl(Path(pairs_path))
    print(f"  Total pairs: {len(pairs):,}")

    # Sample unique goals
    print(f"Sampling {num_samples} unique goals...")
    goals = list(set(p["goal"] for p in pairs))
    random.seed(42)
    random.shuffle(goals)
    goals = goals[:num_samples]
    print(f"  {len(goals)} unique goals")

    # Generate simple keyword-based embeddings
    # Use the GoalEncoder directly on synthetic embeddings
    print("Generating goal encodings...")
    embeddings = []
    targets = []

    for i, goal_text in enumerate(goals):
        # Create a synthetic embedding from keyword hash
        # This gives us a deterministic per-goal embedding
        keywords = [w for w in goal_text.lower().replace("(", " ").replace(")", " ")
                     .replace(":", " ").replace(",", " ").split() if len(w) >= 2]
        if not keywords:
            keywords = ["unknown"]

        # Hash keywords into a fixed-size vector
        seed = hash(tuple(sorted(set(keywords)))) % (2**31)
        gen = torch.Generator()
        gen.manual_seed(seed)
        raw_emb = torch.randn(enc_dim, generator=gen)
        raw_emb = F.normalize(raw_emb, dim=0)

        # If GoalEncoder exists, project through it
        if gnn.goal_encoder is not None:
            with torch.no_grad():
                goal_emb = gnn.goal_encoder(raw_emb)
        else:
            goal_emb = raw_emb

        embeddings.append(goal_emb.cpu())
        targets.append(1.0)  # Positive: provable goal

        # Negative: perturb embedding
        noise = torch.randn(enc_dim) * 0.5
        neg_emb = goal_emb + noise
        neg_emb = F.normalize(neg_emb, dim=0)
        embeddings.append(neg_emb.cpu())
        targets.append(0.0)  # Negative: perturbed goal

        if (i + 1) % 1000 == 0:
            print(f"  Encoded {i+1}/{len(goals)} goals...")

    embeddings_t = torch.stack(embeddings)
    targets_t = torch.tensor(targets, dtype=torch.float32)

    n_pos = int(targets_t.sum().item())
    n_neg = len(targets_t) - n_pos
    print(f"\nTraining data: {n_pos} positive, {n_neg} negative ({n_pos/len(targets_t):.1%})")

    # Create value network
    vn = ValueNetwork(gnn, freeze_encoder=True)
    vn_params = sum(p.numel() for p in vn.value_head.parameters())
    print(f"Value head: {vn_params:,} trainable params")

    # Train/test split
    perm = torch.randperm(len(embeddings_t))
    split = int(0.8 * len(embeddings_t))
    train_emb = embeddings_t[perm[:split]]
    train_tgt = targets_t[perm[:split]]
    test_emb = embeddings_t[perm[split:]]
    test_tgt = targets_t[perm[split:]]

    print(f"Train: {len(train_emb)}, Test: {len(test_emb)}")

    # Training loop
    optimizer = torch.optim.AdamW(vn.value_head.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCELoss()

    print(f"\nTraining ({epochs} epochs, batch={batch_size})...")
    t_start = time.time()
    best_acc = 0.0

    for epoch in range(epochs):
        vn.train()
        total_loss = 0.0

        indices = torch.randperm(len(train_emb))
        for i in range(0, len(train_emb), batch_size):
            batch_idx = indices[i:i + batch_size]
            preds = vn(train_emb[batch_idx])
            loss = criterion(preds, train_tgt[batch_idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        # Eval
        vn.eval()
        with torch.no_grad():
            test_preds = vn(test_emb)
            test_acc = ((test_preds > 0.5) == test_tgt).float().mean().item()

        if test_acc > best_acc:
            best_acc = test_acc

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}  "
                  f"loss={total_loss/max(1,(len(train_emb)//batch_size)):.4f}  "
                  f"test_acc={test_acc:.3f}")

    elapsed = time.time() - t_start
    print(f"\nTraining complete: {elapsed:.1f}s, best test_acc={best_acc:.3f}")

    # Save
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    vn.save(output)
    print(f"Value network saved to: {output}")

    # Also save metadata
    meta_path = output.with_suffix(".train_metrics.json")
    with open(meta_path, "w") as f:
        json.dump({
            "gnn_checkpoint": gnn_checkpoint,
            "num_samples": num_samples,
            "epochs": epochs,
            "lr": lr,
            "best_test_acc": round(best_acc, 4),
            "elapsed_s": round(elapsed, 1),
        }, f, indent=2)
    print(f"Metrics saved to: {meta_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gnn-checkpoint", default="checkpoints/gnn/proof_step_pretrained.pt")
    parser.add_argument("--train-pairs", default="data/raw/proof_step_pairs.jsonl")
    parser.add_argument("--output", default="checkpoints/value_network.pt")
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train(
        gnn_checkpoint=args.gnn_checkpoint,
        pairs_path=args.train_pairs,
        output_path=args.output,
        num_samples=args.num_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
