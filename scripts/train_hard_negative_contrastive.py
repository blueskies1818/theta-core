#!/usr/bin/env python3
"""Train hard-negative contrastive dual-encoder on proof-step pairs.

Combines InfoNCE (in-batch soft negatives) with triplet margin loss on
confirmed hard negatives from the proof checker.

The model learns to:
  1. Pull correct (goal, lemma) pairs together (InfoNCE)
  2. Push confirmed-wrong lemmas away from goals (triplet margin)

This produces embeddings that encode proof utility, not just graph proximity.

Usage:
    # Train with default hard negatives
    python scripts/train_hard_negative_contrastive.py --epochs 30 --batch-size 128

    # Train without hard negatives (InfoNCE only — like original)
    python scripts/train_hard_negative_contrastive.py --epochs 30 --no-hard-negatives

Output:
    checkpoints/contrastive/hard_negative_encoder.pt
    data/hard_neg_training_stats.json
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


def load_pairs(data_path: Path) -> list[dict]:
    """Load proof-step pairs from JSONL."""
    pairs = []
    with open(data_path) as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs


def load_hard_neg_triples(data_path: Path) -> list[dict]:
    """Load hard-negative triples from JSONL."""
    triples = []
    with open(data_path) as f:
        for line in f:
            triples.append(json.loads(line))
    return triples


def compute_infonce(
    goal_emb: torch.Tensor,
    lemma_emb: torch.Tensor,
    temperature_inv: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric InfoNCE loss + accuracy."""
    from src.contrastive.hard_negative_loss import (
        compute_infonce_loss,
        compute_retrieval_accuracy,
    )
    loss = compute_infonce_loss(goal_emb, lemma_emb, temperature_inv)
    acc = compute_retrieval_accuracy(goal_emb, lemma_emb)
    return loss, acc


def compute_combined(
    goal_emb: torch.Tensor,
    pos_emb: torch.Tensor,
    hard_neg_emb: torch.Tensor | None,
    temperature_inv: float,
    hard_neg_weight: float,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combined InfoNCE + hard-negative triplet loss."""
    from src.contrastive.hard_negative_loss import (
        compute_combined_loss,
        compute_retrieval_accuracy,
    )
    losses = compute_combined_loss(
        goal_emb, pos_emb, hard_neg_emb,
        temperature_inv=temperature_inv,
        hard_neg_weight=hard_neg_weight,
        margin=margin,
    )
    acc = compute_retrieval_accuracy(goal_emb, pos_emb)
    return losses["total_loss"], losses["infonce_loss"], losses["hard_neg_loss"], acc


@torch.no_grad()
def evaluate(
    model: ContrastiveDualEncoder,
    goal_ids: torch.Tensor,
    pair_lemma_ids: torch.Tensor,
    hard_neg_ids: dict[int, torch.Tensor] | None,
    indices: list[int],
    batch_size: int,
    temperature_inv: float,
    hard_neg_weight: float,
    margin: float,
    device: torch.device,
    tokenizer: CharTokenizer,
) -> dict:
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0.0
    total_infonce = 0.0
    total_hard = 0.0
    total_acc = 0.0
    num_batches = 0

    for batch_start in range(0, len(indices), batch_size):
        batch_idx = indices[batch_start:batch_start + batch_size]
        if len(batch_idx) < 2:
            continue

        batch_goal_ids = goal_ids[batch_idx]
        batch_lemma_ids = pair_lemma_ids[batch_idx]

        # Hard negatives for this batch
        hn_batch = None
        if hard_neg_ids is not None:
            hn_list = []
            for idx in batch_idx:
                if idx in hard_neg_ids:
                    hn_list.append(hard_neg_ids[idx])
                else:
                    hn_list.append(torch.zeros(0, model.config.max_seq_len,
                                               dtype=torch.long, device=device))
            if any(h.shape[0] > 0 for h in hn_list):
                K = max(h.shape[0] for h in hn_list if h.shape[0] > 0)
                padded = torch.zeros(len(hn_list), K, model.config.max_seq_len,
                                     dtype=torch.long, device=device)
                for i, h in enumerate(hn_list):
                    if h.shape[0] > 0:
                        padded[i, :h.shape[0]] = h
                hn_batch = padded
                # Mask: only consider real hard negatives (not padding)
                valid_mask = (hn_batch != 0).any(dim=-1)  # [B, K]
            else:
                hn_batch = None

        output = model.forward_hard(
            batch_goal_ids, batch_lemma_ids, hn_batch,
            hard_neg_weight=hard_neg_weight, margin=margin,
        )

        total_loss += output["total_loss"].item()
        total_infonce += output["infonce_loss"].item()
        total_hard += output["hard_neg_loss"].item()
        total_acc += output["accuracy"].item()
        num_batches += 1

    n = max(1, num_batches)
    return {
        "loss": total_loss / n,
        "infonce_loss": total_infonce / n,
        "hard_neg_loss": total_hard / n,
        "accuracy": total_acc / n,
    }


def train(args) -> dict:
    """Train the hard-negative contrastive dual-encoder."""
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
    pairs_path = _project_root / args.data
    pairs = load_pairs(pairs_path)
    print(f"Loaded {len(pairs)} proof-step pairs from {pairs_path}")

    # ---- Load hard-negative triples -----------------------------------------
    hard_neg_triples = None
    pair_to_hard_negs: dict[str, list[str]] = {}
    if not args.no_hard_negatives:
        triples_path = _project_root / args.hard_neg_data
        if triples_path.exists():
            hard_neg_triples = load_hard_neg_triples(triples_path)
            for t in hard_neg_triples:
                pair_to_hard_negs[t["goal"]] = t["hard_negatives"]
            print(f"Loaded {len(hard_neg_triples)} hard-negative triples "
                  f"from {triples_path}")
        else:
            print(f"WARNING: Hard negative data not found at {triples_path}")
            print("  Training with InfoNCE only (no hard negatives).")

    # ---- Build lemma ID cache -----------------------------------------------
    print("Tokenizing unique lemmas...", end=" ", flush=True)
    # Collect all hard negative lemma names
    all_hn_lemmas: set[str] = set()
    for hn_list in pair_to_hard_negs.values():
        for hn in hn_list:
            all_hn_lemmas.add(hn)

    pair_lemmas = set(p["lemma"] for p in pairs)
    unique_lemmas = sorted(pair_lemmas | all_hn_lemmas)

    lemma_to_ids: dict[str, torch.Tensor] = {}
    for lemma in unique_lemmas:
        lemma_text = tokenizer.preprocess_lemma(lemma)
        lemma_to_ids[lemma] = tokenizer.encode(lemma_text).to(device)
    print(f"{len(lemma_to_ids)} unique lemmas tokenized.")

    # ---- Pre-tokenize goals and pair lemmas ---------------------------------
    print("Tokenizing goals and pair lemmas...", end=" ", flush=True)
    goal_texts = [tokenizer.preprocess_goal(p["goal"]) for p in pairs]
    goal_ids = tokenizer.encode_batch(goal_texts).to(device)
    lemma_texts = [tokenizer.preprocess_lemma(p["lemma"]) for p in pairs]
    pair_lemma_ids = tokenizer.encode_batch(lemma_texts).to(device)
    print(f"done ({goal_ids.shape[0]} goals).")

    # ---- Pre-tokenize hard negatives ----------------------------------------
    pair_idx_to_hn_ids: dict[int, torch.Tensor] = {}
    missing_count = 0
    if pair_to_hard_negs:
        print("Tokenizing hard negatives...", end=" ", flush=True)
        for i, pair in enumerate(pairs):
            goal = pair["goal"]
            if goal in pair_to_hard_negs:
                hn_lemmas = pair_to_hard_negs[goal]
                hn_ids = torch.zeros(len(hn_lemmas), config.max_seq_len,
                                     dtype=torch.long, device=device)
                for j, hn in enumerate(hn_lemmas):
                    if hn in lemma_to_ids:
                        hn_ids[j] = lemma_to_ids[hn]
                    else:
                        hn_text = tokenizer.preprocess_lemma(hn)
                        hn_ids[j] = tokenizer.encode(hn_text).to(device)
                pair_idx_to_hn_ids[i] = hn_ids
            else:
                missing_count += 1
        print(f"done. {len(pair_idx_to_hn_ids)} pairs have hard negatives "
              f"({missing_count} without).")

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

    # ---- Split indices ------------------------------------------------------
    indices = list(range(len(pairs)))
    random.seed(args.seed)
    random.shuffle(indices)

    split = int(len(indices) * args.train_split)
    train_indices = indices[:split]
    val_indices = indices[split:split + min(args.val_size, len(indices) - split)]

    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}")

    # ---- Training -----------------------------------------------------------
    best_val_acc = 0.0
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_acc": [], "val_loss": [],
               "infonce_loss": [], "hard_neg_loss": []}
    t_start = time.time()

    temperature_inv = 1.0 / config.temperature

    for epoch in range(config.num_epochs):
        model.train()
        random.shuffle(train_indices)

        epoch_loss = 0.0
        epoch_infonce = 0.0
        epoch_hard = 0.0
        epoch_acc = 0.0
        num_batches = 0

        for batch_start in range(0, len(train_indices), config.batch_size):
            batch_idx = train_indices[batch_start:batch_start + config.batch_size]
            if len(batch_idx) < 2:
                continue

            batch_goal_ids = goal_ids[batch_idx]
            batch_lemma_ids = pair_lemma_ids[batch_idx]

            # Gather hard negatives for this batch
            hn_batch = None
            if pair_idx_to_hn_ids:
                hn_list = []
                for idx in batch_idx:
                    if idx in pair_idx_to_hn_ids:
                        hn_list.append(pair_idx_to_hn_ids[idx])
                    else:
                        hn_list.append(torch.zeros(0, config.max_seq_len,
                                                   dtype=torch.long, device=device))

                has_hard = any(h.shape[0] > 0 for h in hn_list)
                if has_hard:
                    K = max(h.shape[0] for h in hn_list if h.shape[0] > 0)
                    padded = torch.zeros(len(hn_list), K, config.max_seq_len,
                                         dtype=torch.long, device=device)
                    for i, h in enumerate(hn_list):
                        if h.shape[0] > 0:
                            padded[i, :h.shape[0]] = h
                    hn_batch = padded

            output = model.forward_hard(
                batch_goal_ids, batch_lemma_ids, hn_batch,
                hard_neg_weight=args.hard_neg_weight,
                margin=args.margin,
            )

            optimizer.zero_grad()
            output["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += output["total_loss"].item()
            epoch_infonce += output["infonce_loss"].item()
            epoch_hard += output["hard_neg_loss"].item()
            epoch_acc += output["accuracy"].item()
            num_batches += 1

        scheduler.step()
        n = max(1, num_batches)
        avg_loss = epoch_loss / n
        avg_infonce = epoch_infonce / n
        avg_hard = epoch_hard / n
        avg_acc = epoch_acc / n

        history["train_loss"].append(avg_loss)
        history["infonce_loss"].append(avg_infonce)
        history["hard_neg_loss"].append(avg_hard)

        # ---- Validation ----------------------------------------------------
        val_results = evaluate(
            model, goal_ids, pair_lemma_ids, pair_idx_to_hn_ids,
            val_indices, config.batch_size,
            temperature_inv, args.hard_neg_weight, args.margin,
            device, tokenizer,
        )
        history["val_loss"].append(val_results["loss"])
        history["val_acc"].append(val_results["accuracy"])

        elapsed = time.time() - t_start
        if epoch % 2 == 0 or epoch == config.num_epochs - 1:
            print(f"Epoch {epoch:3d}/{config.num_epochs} | "
                  f"Loss: {avg_loss:.4f} (iNCE: {avg_infonce:.4f}, "
                  f"HN: {avg_hard:.4f}) | "
                  f"Acc: {avg_acc:.3f} | "
                  f"Val: {val_results['loss']:.4f} Acc: {val_results['accuracy']:.3f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                  f"Time: {elapsed:.1f}s", flush=True)

        # ---- Checkpoint ----------------------------------------------------
        if val_results["accuracy"] > best_val_acc:
            best_val_acc = val_results["accuracy"]
            output_path = _project_root / args.output
            model.save(output_path)
            print(f"  → Saved best model (val_acc={best_val_acc:.3f})")

        if val_results["loss"] < best_val_loss:
            best_val_loss = val_results["loss"]

    total_time = time.time() - t_start
    print(f"\nTraining complete: {config.num_epochs} epochs in {total_time:.1f}s")
    print(f"  Best val accuracy: {best_val_acc:.3f}")
    print(f"  Best val loss:     {best_val_loss:.4f}")

    return {
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "total_time": total_time,
        "num_pairs": len(pairs),
        "num_hard_neg_triples": len(hard_neg_triples) if hard_neg_triples else 0,
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train hard-negative contrastive dual-encoder"
    )
    # Data
    parser.add_argument("--data", default="data/raw/proof_step_pairs.jsonl")
    parser.add_argument("--hard-neg-data", default="data/hard_neg_triples.jsonl")
    parser.add_argument("--output", default="checkpoints/contrastive/hard_negative_encoder.pt")
    parser.add_argument("--train-split", type=float, default=0.95)
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--no-hard-negatives", action="store_true",
                        help="Train with InfoNCE only (no hard negatives)")

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
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--hard-neg-weight", type=float, default=0.5,
                        help="Weight for hard negative triplet loss")
    parser.add_argument("--margin", type=float, default=0.3,
                        help="Triplet margin")
    parser.add_argument("--seed", type=int, default=42)

    # Hardware
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-threads", type=int, default=4)

    args = parser.parse_args()
    stats = train(args)

    # Save training stats
    stats_path = _project_root / "data/hard_neg_training_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump({
            "best_val_acc": stats["best_val_acc"],
            "best_val_loss": stats["best_val_loss"],
            "total_time_s": stats["total_time"],
            "num_pairs": stats["num_pairs"],
            "num_hard_neg_triples": stats["num_hard_neg_triples"],
            "train_loss_final": stats["history"]["train_loss"][-1] if stats["history"]["train_loss"] else None,
            "val_acc_final": stats["history"]["val_acc"][-1] if stats["history"]["val_acc"] else None,
        }, f, indent=2)

    print(f"Stats saved to {stats_path}")


if __name__ == "__main__":
    main()
