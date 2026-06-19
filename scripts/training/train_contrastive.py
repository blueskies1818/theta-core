#!/usr/bin/env python3
"""Train contrastive dual-encoder on proof-step pairs (Path C).

Trains a dual-encoder (goal_encoder + lemma_encoder) with InfoNCE
contrastive loss on (goal, correct_lemma) pairs from Mathlib4 proofs.

This replaces the GNN+GoalEncoder cosine-similarity retrieval with
a learned relevance scoring function trained end-to-end.

Usage:
    python scripts/training/train_contrastive.py --epochs 30 --batch-size 256
"""

import argparse, json, random, sys, time
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from src.contrastive.encoder import (
    ContrastiveDualEncoder,
    ContrastiveConfig,
    CharTokenizer,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_pairs(data_path: Path) -> list[dict]:
    """Load proof-step pairs from JSONL."""
    pairs = []
    with open(data_path) as f:
        for line in f:
            d = json.loads(line)
            pairs.append(d)
    return pairs


def build_lemma_to_ids(
    pairs: list[dict],
    tokenizer: CharTokenizer,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Build a cache of pre-tokenized unique lemma character IDs.

    Returns:
        Dict mapping lemma name → [max_len] LongTensor of character IDs.
    """
    unique_lemmas = sorted(set(p["lemma"] for p in pairs))
    lemma_to_ids = {}

    for lemma in unique_lemmas:
        lemma_text = tokenizer.preprocess_lemma(lemma)
        lemma_to_ids[lemma] = tokenizer.encode(lemma_text).to(device)

    return lemma_to_ids


# ---------------------------------------------------------------------------
# InfoNCE computation
# ---------------------------------------------------------------------------


def compute_infonce(
    goal_emb: torch.Tensor,
    lemma_emb: torch.Tensor,
    temperature_inv: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute symmetric InfoNCE loss and per-sample accuracy.

    Args:
        goal_emb: [B, D] L2-normalized goal embeddings.
        lemma_emb: [B, D] L2-normalized lemma embeddings.
        temperature_inv: 1 / temperature.

    Returns:
        (loss, accuracy) where accuracy is fraction of correct top-1 retrievals.
    """
    batch_size = goal_emb.size(0)
    logits = goal_emb @ lemma_emb.T * temperature_inv  # [B, B]
    labels = torch.arange(batch_size, device=goal_emb.device)

    loss_g2l = F.cross_entropy(logits, labels)
    loss_l2g = F.cross_entropy(logits.T, labels)
    loss = (loss_g2l + loss_l2g) / 2.0

    # Compute top-1 accuracy (diagonal should be highest)
    _, pred_g2l = logits.max(dim=1)
    _, pred_l2g = logits.T.max(dim=1)
    acc_g2l = (pred_g2l == labels).float().mean()
    acc_l2g = (pred_l2g == labels).float().mean()
    acc = (acc_g2l + acc_l2g) / 2.0

    return loss, acc


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(args) -> dict:
    """Train the contrastive dual-encoder.

    Returns:
        Dict with training stats for evaluation.
    """
    # ---- Setup --------------------------------------------------------------
    torch.set_num_threads(args.num_threads)
    device = torch.device(args.device or "cpu")
    print(f"Device: {device} (threads: {args.num_threads})")

    # ---- Config -------------------------------------------------------------
    config = ContrastiveConfig(
        hidden_dim=args.hidden_dim,
        vocab_size=args.vocab_size,
        max_seq_len=args.max_seq_len,
        char_embed_dim=args.char_embed_dim,
        cnn_filters=args.cnn_filters,
        cnn_kernel_sizes=tuple(args.kernel_sizes),
        cnn_dropout=args.dropout,
        mlp_expansion=args.mlp_expansion,
        pooling=args.pooling,
        temperature=args.temperature,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
    )

    tokenizer = CharTokenizer(max_len=config.max_seq_len)

    # ---- Load data ----------------------------------------------------------
    data_path = _project_root / args.data
    pairs = load_pairs(data_path)
    print(f"Loaded {len(pairs)} proof-step pairs from {data_path}")

    # ---- Build lemma ID cache -----------------------------------------------
    print("Tokenizing unique lemmas...", end=" ", flush=True)
    lemma_to_ids = build_lemma_to_ids(pairs, tokenizer, device)
    print(f"{len(lemma_to_ids)} unique lemmas tokenized.")

    # ---- Create model -------------------------------------------------------
    model = ContrastiveDualEncoder(config).to(device)
    print(f"Model: {model.num_params:,} total params "
          f"(goal: {model.goal_encoder_params:,}, "
          f"lemma: {model.lemma_encoder_params:,})")

    # ---- Optimizer ----------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.num_epochs,
    )

    # ---- Shuffle indices for reproducible splits ----------------------------
    indices = list(range(len(pairs)))
    random.seed(args.seed)
    random.shuffle(indices)

    split = int(len(indices) * args.train_split)
    train_indices = indices[:split]
    val_indices = indices[split:split + min(args.val_size, len(indices) - split)]

    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}")

    # ---- Pre-tokenize all goals (batched, fast) -----------------------------
    print("Tokenizing goals...", end=" ", flush=True)
    goal_texts = [tokenizer.preprocess_goal(p["goal"]) for p in pairs]
    goal_ids = tokenizer.encode_batch(goal_texts).to(device)
    print(f"done ({goal_ids.shape[0]} goals, {goal_ids.shape[1]} max_len).")

    # ---- Pre-tokenize all pair lemmas (batched, fast) -----------------------
    print("Tokenizing pair lemmas...", end=" ", flush=True)
    lemma_texts = [tokenizer.preprocess_lemma(p["lemma"]) for p in pairs]
    pair_lemma_ids = tokenizer.encode_batch(lemma_texts).to(device)
    print("done.")

    # ---- Training -----------------------------------------------------------
    best_val_acc = 0.0
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_acc": [], "val_loss": []}
    t_start = time.time()

    for epoch in range(config.num_epochs):
        model.train()
        random.shuffle(train_indices)

        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0

        for batch_start in range(0, len(train_indices), config.batch_size):
            batch_idx = train_indices[batch_start:batch_start + config.batch_size]
            if len(batch_idx) < 2:
                continue

            # Gather goal and lemma character IDs (pre-tokenized)
            batch_goal_ids = goal_ids[batch_idx]  # [B, max_len]
            batch_lemma_ids = pair_lemma_ids[batch_idx]  # [B, max_len]

            # Forward pass
            goal_emb = model.encode_goal(batch_goal_ids)
            lemma_emb = model.encode_lemma(batch_lemma_ids)

            loss, acc = compute_infonce(goal_emb, lemma_emb, model._t_inv)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            num_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(1, num_batches)
        avg_acc = epoch_acc / max(1, num_batches)
        history["train_loss"].append(avg_loss)

        # ---- Validation ----------------------------------------------------
        val_loss, val_acc = evaluate(model, goal_ids, pair_lemma_ids,
                                      val_indices, config.batch_size, device,
                                      model._t_inv)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t_start
        if epoch % 2 == 0 or epoch == config.num_epochs - 1:
            print(f"Epoch {epoch:3d}/{config.num_epochs} | "
                  f"Train Loss: {avg_loss:.4f} Acc: {avg_acc:.3f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                  f"Time: {elapsed:.1f}s", flush=True)

        # ---- Checkpoint ----------------------------------------------------
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            output_path = _project_root / args.output
            model.save(output_path)
            print(f"  → Saved best model (val_acc={best_val_acc:.3f})")

        if val_loss < best_val_loss:
            best_val_loss = val_loss

    total_time = time.time() - t_start
    print(f"\nTraining complete: {config.num_epochs} epochs in {total_time:.1f}s")
    print(f"  Best val accuracy: {best_val_acc:.3f}")
    print(f"  Best val loss: {best_val_loss:.4f}")

    return {
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "total_time": total_time,
        "num_pairs": len(pairs),
        "num_unique_lemmas": len(lemma_to_ids),
        "history": history,
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(
    model: ContrastiveDualEncoder,
    goal_ids: torch.Tensor,
    pair_lemma_ids: torch.Tensor,
    indices: list[int],
    batch_size: int,
    device: torch.device,
    temperature_inv: float,
) -> tuple[float, float]:
    """Evaluate model on validation set.

    Returns:
        (avg_loss, avg_accuracy)
    """
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0

    for batch_start in range(0, len(indices), batch_size):
        batch_idx = indices[batch_start:batch_start + batch_size]
        if len(batch_idx) < 2:
            continue

        batch_goal_ids = goal_ids[batch_idx]
        batch_lemma_ids = pair_lemma_ids[batch_idx]

        goal_emb = model.encode_goal(batch_goal_ids)
        lemma_emb = model.encode_lemma(batch_lemma_ids)

        loss, acc = compute_infonce(goal_emb, lemma_emb, temperature_inv)
        total_loss += loss.item()
        total_acc += acc.item()
        num_batches += 1

    return total_loss / max(1, num_batches), total_acc / max(1, num_batches)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Train contrastive dual-encoder on proof-step pairs (Path C)"
    )
    # Data
    parser.add_argument("--data", default="data/raw/proof_step_pairs.jsonl")
    parser.add_argument("--output", default="checkpoints/contrastive/lemma_encoder.pt")
    parser.add_argument("--train-split", type=float, default=0.95)
    parser.add_argument("--val-size", type=int, default=5000)

    # Model
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--char-embed-dim", type=int, default=64)
    parser.add_argument("--cnn-filters", type=int, default=128)
    parser.add_argument("--kernel-sizes", type=int, nargs="+", default=[2, 3, 4, 5])
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--mlp-expansion", type=int, default=2)
    parser.add_argument("--pooling", choices=["mean", "max", "attention"],
                        default="attention")

    # Training
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--seed", type=int, default=42)

    # Hardware
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-threads", type=int, default=4)

    args = parser.parse_args()

    stats = train(args)

    # Save training stats alongside model
    stats_path = _project_root / "data/pathc_retrieval_results.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump({
            "best_val_acc": stats["best_val_acc"],
            "best_val_loss": stats["best_val_loss"],
            "total_time_s": stats["total_time"],
            "num_pairs": stats["num_pairs"],
            "num_unique_lemmas": stats["num_unique_lemmas"],
            "train_loss_final": stats["history"]["train_loss"][-1] if stats["history"]["train_loss"] else None,
            "val_acc_final": stats["history"]["val_acc"][-1] if stats["history"]["val_acc"] else None,
        }, f, indent=2)

    print(f"Stats saved to {stats_path}")


if __name__ == "__main__":
    main()
