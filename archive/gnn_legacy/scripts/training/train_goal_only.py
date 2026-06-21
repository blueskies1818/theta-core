#!/usr/bin/env python3
"""
Goal-Only GNN Training: Self-supervised clustering via Triplet Margin Loss.

GOALS ONLY. No lemmas. No graph. No BCE. No InfoNCE.

Training objective:
  anchor=goal_i, positive=other goal with same lemma, negative=random goal with different lemma
  Loss = max(0, margin - cos(anchor, positive) + cos(anchor, negative))

Key design decisions:
  - Only train on goals that have ≥1 OTHER goal sharing a lemma (no singletons)
  - Multiple negatives per anchor (default K=5)
  - Cosine embedding space, L2 normalized

Usage:
  # Smoke test
  python scripts/training/train_goal_only.py --smoke --epochs 3 --max-pairs 2000

  # Full training
  python scripts/training/train_goal_only.py --epochs 20 --threads 6
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))

from src.explorer.goal_only_encoder import (
    GoalOnlyEncoder,
    tokenize_goal,
    build_vocab,
)


# ---------------------------------------------------------------------------
# Data loading and lemma grouping
# ---------------------------------------------------------------------------


def load_pairs(data_path: Path, max_pairs: int | None = None) -> list[dict]:
    pairs = []
    with open(data_path) as f:
        for line in f:
            pair = json.loads(line)
            pairs.append(pair)
            if max_pairs and len(pairs) >= max_pairs:
                break
    return pairs


def build_goal_lemma_map(pairs: list[dict]) -> dict[str, list[str]]:
    """Build: goal_text → list of lemmas this goal is paired with."""
    goal_lemmas: dict[str, set[str]] = defaultdict(set)
    for pair in pairs:
        goal_lemmas[pair["goal"]].add(pair["lemma"])
    return {g: list(ls) for g, ls in goal_lemmas.items()}


def build_lemma_goal_map(goal_lemma_map: dict[str, list[str]]) -> dict[str, list[str]]:
    """Build: lemma → list of goal_texts that use it."""
    lemma_goals: dict[str, list[str]] = defaultdict(list)
    for goal, lemmas in goal_lemma_map.items():
        for lemma in lemmas:
            lemma_goals[lemma].append(goal)
    return dict(lemma_goals)


# ---------------------------------------------------------------------------
# Triplet margin loss
# ---------------------------------------------------------------------------


def compute_triplet_loss(
    anchor_embs: torch.Tensor,   # [B, D]
    positive_embs: torch.Tensor, # [B, D]
    negative_embs: torch.Tensor, # [B, K, D]
    margin: float = 0.3,
) -> tuple[torch.Tensor, float, float]:
    """Cosine similarity loss: maximize cos(a,p), minimize max cos(a,n).

    Loss = -mean(cos(a,p)) + mean(max_k cos(a,n_k))

    Uses direct cosine similarity gradients (no margin) for stronger signal.
    Prevents embedding collapse by explicitly pushing negatives away.

    Returns:
        loss, mean_pos_cos, mean_neg_cos
    """
    B = anchor_embs.size(0)

    # Cosine similarities (embeddings already normalized)
    pos_cos = (anchor_embs * positive_embs).sum(dim=-1)  # [B]

    # Hardest negative per anchor
    neg_cos_all = torch.bmm(
        anchor_embs.unsqueeze(1),      # [B, 1, D]
        negative_embs.transpose(1, 2)   # [B, D, K]
    ).squeeze(1)  # [B, K]
    neg_cos = neg_cos_all.max(dim=1).values  # [B]

    # Loss: pull positives together, push negatives apart
    # Add a small repulsion bonus to prevent collapse
    pos_loss = -pos_cos.mean()
    neg_loss = neg_cos.mean()
    
    # Variance bonus: penalize low variance → encourages spread
    # (prevents the "all embeddings are the same" collapse)
    anchor_var = anchor_embs.var(dim=0).mean()  # mean variance across dimensions
    var_bonus = 0.1 * torch.relu(0.01 - anchor_var)  # penalize if var < 0.01

    loss = pos_loss + neg_loss + var_bonus

    return loss, pos_cos.mean().item(), neg_cos.mean().item()


# ---------------------------------------------------------------------------
# Validation: MRR (Mean Reciprocal Rank)
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_mrr(
    model: GoalOnlyEncoder,
    val_pairs: list[dict],
    train_goals: list[str],
    train_embeddings: torch.Tensor,
    lemma_to_goals: dict[str, list[str]],
    goal_to_lemmas: dict[str, list[str]],  # goal_text → list of lemma names
    device: torch.device,
    k: int = 50,
) -> float:
    """Compute MRR: encode goal → top-K training goals → collect lemmas (freq×sim).

    For each validation pair (goal, correct_lemma):
      1. Encode goal
      2. Find top-K similar training goals by cosine similarity
      3. Collect their lemmas, weighted by frequency × similarity
      4. Rank correct_lemma among collected lemmas

    Returns MRR score.
    """
    model.eval()
    reciprocal_ranks = []

    for pair in val_pairs:
        goal_text = pair["goal"]
        correct_lemma = pair["lemma"]

        # Encode goal
        try:
            goal_emb = model.encode_single(goal_text, device)  # [D]
        except Exception:
            continue

        # Cosine similarity to all training goals
        sims = (goal_emb @ train_embeddings.T)  # [N]

        # Top-k
        topk_sims, topk_indices = torch.topk(sims, min(k, len(train_goals)))

        # Collect lemmas from top-k, weighted by cosine similarity
        lemma_scores: dict[str, float] = defaultdict(float)
        for i in range(len(topk_indices)):
            idx = topk_indices[i].item()
            sim = max(0.0, topk_sims[i].item())
            goal = train_goals[idx]
            for lemma in goal_to_lemmas.get(goal, []):
                lemma_scores[lemma] += sim

        # Rank correct lemma
        sorted_lemmas = sorted(lemma_scores.items(), key=lambda x: -x[1])
        rank = len(sorted_lemmas) + 1  # default: worst rank
        for r, (lem, sc) in enumerate(sorted_lemmas, 1):
            if lem == correct_lemma:
                rank = r
                break

        reciprocal_ranks.append(1.0 / rank)

    if not reciprocal_ranks:
        return 0.0
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


# ---------------------------------------------------------------------------
# Main training
# ---------------------------------------------------------------------------


def train(args):
    import builtins
    import functools
    _real_print = print
    builtins.print = functools.partial(_real_print, flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    torch.set_num_threads(args.threads)
    print(f"Device: {device}, Threads: {torch.get_num_threads()}")

    # ---- Load pairs ----
    print("\n--- Loading proof-step pairs ---")
    pairs_path = Path(args.pairs)
    all_pairs = load_pairs(pairs_path, args.max_pairs)
    print(f"Loaded {len(all_pairs)} pairs")

    # ---- Build goal <-> lemma maps ----
    print("\n--- Building goal/lemma maps ---")
    goal_lemma_map = build_goal_lemma_map(all_pairs)
    lemma_goal_map = build_lemma_goal_map(goal_lemma_map)
    
    # ---- Filter to frequent lemmas (stronger training signal) ----
    if args.min_lemma_frequency > 1:
        lemma_freq = {lem: len(goals) for lem, goals in lemma_goal_map.items()}
        frequent_lemmas = {lem for lem, f in lemma_freq.items() if f >= args.min_lemma_frequency}
        # Filter goal_lemma_map to only frequent lemmas
        goal_lemma_map_filt = {}
        for goal, lemmas in goal_lemma_map.items():
            freq_lems = [l for l in lemmas if l in frequent_lemmas]
            if freq_lems:
                goal_lemma_map_filt[goal] = freq_lems
        n_removed = len(goal_lemma_map) - len(goal_lemma_map_filt)
        print(f"\n  Lemma frequency filter (≥{args.min_lemma_frequency}):")
        print(f"    Frequent lemmas: {len(frequent_lemmas)} / {len(lemma_goal_map)}")
        print(f"    Goals kept: {len(goal_lemma_map_filt)} / {len(goal_lemma_map)} ({n_removed} removed)")
        goal_lemma_map = goal_lemma_map_filt
        # Rebuild lemma_goal_map from filtered data
        lemma_goal_map = build_lemma_goal_map(goal_lemma_map)

    # Keep only goals that share a lemma with ≥1 OTHER goal
    trainable_goals = []
    single_goals = []
    for goal, lemmas in goal_lemma_map.items():
        has_partner = False
        for lemma in lemmas:
            partners = lemma_goal_map.get(lemma, [])
            if len(partners) >= 2:
                has_partner = True
                break
        if has_partner:
            trainable_goals.append(goal)
        else:
            single_goals.append(goal)

    print(f"Total unique goals: {len(goal_lemma_map)}")
    print(f"Trainable (share lemma): {len(trainable_goals)}")
    print(f"Singleton (no partners): {len(single_goals)}")
    print(f"Unique lemmas: {len(lemma_goal_map)}")

    lemmas_with_multi = sum(1 for gs in lemma_goal_map.values() if len(gs) >= 2)
    print(f"Lemmas with ≥2 goals: {lemmas_with_multi}")

    if len(trainable_goals) < 100:
        print("\nERROR: Not enough trainable goals. Try increasing --max-pairs.")
        sys.exit(1)

    # ---- Train/val split ----
    random.seed(42)
    random.shuffle(trainable_goals)
    split = int(len(trainable_goals) * (1 - args.val_split))
    train_goals = trainable_goals[:split]
    val_goals_set = set(trainable_goals[split:])
    print(f"\nTrain goals: {len(train_goals)}, Val goals: {len(val_goals_set)}")

    # Build goal_to_lemmas for use in MRR and positive sampling
    goal_to_lemmas = goal_lemma_map  # goal → [lemma1, lemma2, ...]
    
    # For each training goal, find a positive partner goal (same lemma, different goal)
    train_goals_set = set(train_goals)
    positive_partners: dict[str, list[str]] = {}
    for goal in train_goals:
        partners = set()
        for lemma in goal_to_lemmas.get(goal, []):
            for other in lemma_goal_map.get(lemma, []):
                if other != goal and other in train_goals_set:
                    partners.add(other)
        positive_partners[goal] = list(partners)

    # Remove goals with no positive partners in training set
    train_goals = [g for g in train_goals if positive_partners.get(g)]
    print(f"Train goals with partners: {len(train_goals)}")

    # All goals for negative sampling (exclude validation goals)
    all_goals_for_neg = [g for g in goal_lemma_map if g not in val_goals_set]
    print(f"Negative pool size: {len(all_goals_for_neg)}")

    # ---- Validation pairs ----
    val_pairs = [p for p in all_pairs if p["goal"] in val_goals_set]

    # ---- Build vocabulary ----
    print("\n--- Building vocabulary ---")
    vocab = build_vocab(list(goal_lemma_map.keys()), vocab_size=args.vocab_size)
    print(f"Vocabulary size: {len(vocab)}")

    # ---- Initialize model ----
    print("\n--- Initializing GoalOnlyEncoder ---")
    model = GoalOnlyEncoder(
        vocab=vocab,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
        max_tokens=args.max_tokens,
        dropout=args.dropout,
    ).to(device)
    n_params = model.count_parameters()
    print(f"Parameters: {n_params:,}")

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # ---- Pre-encode training goals for MRR ----
    @torch.no_grad()
    def encode_all_goals(goals: list[str], chunk_size: int = 1000) -> torch.Tensor:
        model.eval()
        chunks = []
        for start in range(0, len(goals), chunk_size):
            chunk = goals[start:start + chunk_size]
            embs = model(chunk, device)
            chunks.append(embs)
        model.train()
        return F.normalize(torch.cat(chunks, dim=0), dim=-1)

    # ---- Stats ----
    best_mrr = 0.0
    best_epoch = 0

    print(f"\n{'='*60}")
    print(f"TRAINING (Triplet): {args.epochs} epochs, batch={args.batch_size}")
    print(f"  Margin: {args.margin}, LR: {args.learning_rate}")
    print(f"  Negatives per anchor: {args.num_negatives}")
    print(f"{'='*60}")

    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()

        epoch_losses = []
        epoch_pos_cos = []
        epoch_neg_cos = []

        # Shuffle
        epoch_train_goals = train_goals.copy()
        random.shuffle(epoch_train_goals)

        n_batches = max(1, len(epoch_train_goals) // args.batch_size)

        for batch_num in range(n_batches):
            start = batch_num * args.batch_size
            end = min(start + args.batch_size, len(epoch_train_goals))
            batch_goals = epoch_train_goals[start:end]

            if len(batch_goals) < 2:
                continue

            # Build anchors, positives, negatives
            anchor_goals = []
            positive_goals = []
            negative_goals = []  # list of lists [B][K]

            for goal in batch_goals:
                # Anchor
                anchor_goals.append(goal)

                # Positive: random partner with same lemma
                partners = positive_partners.get(goal, [])
                if partners:
                    pos = random.choice(partners)
                else:
                    pos = goal  # fallback (shouldn't happen with filter)
                positive_goals.append(pos)

                # Negatives: K random goals with DIFFERENT lemmas
                anchor_lemmas = set(goal_to_lemmas.get(goal, []))
                negs = []
                attempts = 0
                while len(negs) < args.num_negatives and attempts < args.num_negatives * 10:
                    neg = random.choice(all_goals_for_neg)
                    neg_lemmas = set(goal_to_lemmas.get(neg, []))
                    if not anchor_lemmas.intersection(neg_lemmas) and neg != goal:
                        negs.append(neg)
                    attempts += 1
                # Fill remaining with random goals (might share lemmas, that's ok for triplet)
                while len(negs) < args.num_negatives:
                    neg = random.choice(all_goals_for_neg)
                    if neg != goal:
                        negs.append(neg)
                negative_goals.append(negs)

            # Encode
            anchor_embs = model(anchor_goals, device)  # [B, D]
            positive_embs = model(positive_goals, device)  # [B, D]

            # Encode negatives: K separate forward passes
            neg_emb_list = []
            for k in range(args.num_negatives):
                neg_k_goals = [negs[k] for negs in negative_goals]
                neg_k_embs = model(neg_k_goals, device)  # [B, D]
                neg_emb_list.append(neg_k_embs)
            negative_embs = torch.stack(neg_emb_list, dim=1)  # [B, K, D]

            # Normalize
            anchor_embs = F.normalize(anchor_embs, dim=-1)
            positive_embs = F.normalize(positive_embs, dim=-1)
            negative_embs = F.normalize(negative_embs, dim=-1)

            # Triplet loss
            loss, pos_cos, neg_cos = compute_triplet_loss(
                anchor_embs, positive_embs, negative_embs,
                margin=args.margin,
            )

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            epoch_losses.append(loss.item())
            epoch_pos_cos.append(pos_cos)
            epoch_neg_cos.append(neg_cos)

            if (batch_num + 1) % 20 == 0:
                avg_l = sum(epoch_losses[-20:]) / min(20, len(epoch_losses))
                avg_p = sum(epoch_pos_cos[-20:]) / min(20, len(epoch_pos_cos))
                avg_n = sum(epoch_neg_cos[-20:]) / min(20, len(epoch_neg_cos))
                print(f"  Batch {batch_num+1}/{n_batches}  loss={avg_l:.4f}  "
                      f"pos_cos={avg_p:.3f}  neg_cos={avg_n:.3f}")

        # ---- Epoch summary ----
        avg_loss = sum(epoch_losses) / max(1, len(epoch_losses))
        avg_pos = sum(epoch_pos_cos) / max(1, len(epoch_pos_cos))
        avg_neg = sum(epoch_neg_cos) / max(1, len(epoch_neg_cos))
        elapsed = time.time() - t0

        # ---- Validation MRR (every epoch) ----
        mrr = 0.0
        if len(val_pairs) > 0 and len(train_goals) > 0:
            print(f"  Computing MRR...")
            train_embs = encode_all_goals(train_goals[:2000])  # Cap for speed
            val_sample = random.sample(val_pairs, min(100, len(val_pairs)))
            mrr = compute_mrr(
                model, val_sample,
                train_goals[:2000], train_embs,
                lemma_goal_map, goal_to_lemmas, device,
                k=args.mrr_k,
            )

        print(f"Epoch {epoch+1:3d}/{args.epochs}  "
              f"loss={avg_loss:.4f}  pos_cos={avg_pos:.3f}  neg_cos={avg_neg:.3f}  "
              f"MRR={mrr:.4f}  time={elapsed:.1f}s")

        # ---- Checkpoint ----
        if mrr > best_mrr:
            best_mrr = mrr
            best_epoch = epoch + 1
            ckpt_path = output_dir / "goal_only_encoder.pt"
            model.save(str(ckpt_path))
            print(f"  ✓ Saved best model (MRR={best_mrr:.4f}) to {ckpt_path}")

        scheduler.step()

        # ---- Smoke test early stop ----
        if args.smoke and mrr > 0.3:
            print(f"\n  Smoke test PASSED: MRR={mrr:.4f} > 0.3")
            break

    # ---- Final ----
    print(f"\n--- Training complete ---")
    print(f"Best MRR: {best_mrr:.4f} at epoch {best_epoch}")

    final_ckpt = output_dir / "goal_only_encoder.pt"
    if not final_ckpt.exists():
        model.save(str(final_ckpt))

    meta = {
        "n_params": n_params,
        "train_goals": len(train_goals),
        "val_goals": len(val_goals_set),
        "trainable_goals": len(trainable_goals),
        "best_mrr": best_mrr,
        "best_epoch": best_epoch,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    with open(output_dir / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return best_mrr


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Train Goal-Only GNN with triplet margin loss"
    )
    # Data
    parser.add_argument("--pairs", type=str,
                        default="data/raw/proof_step_pairs.jsonl")
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--output-dir", type=str,
                        default="checkpoints/gnn/goal_only")

    # Model
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=4000,
                        help="Vocabulary size for token embedding")
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ffn-dim", type=int, default=192)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Training
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--num-negatives", type=int, default=5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--mrr-k", type=int, default=50)
    parser.add_argument("--min-lemma-frequency", type=int, default=1,
                        help="Filter lemmas with fewer than N occurrences (1=no filter)")
    parser.add_argument("--smoke", action="store_true")

    args = parser.parse_args()

    # Resolve paths
    pairs_path = Path(args.pairs)
    if not pairs_path.is_absolute():
        pairs_path = _project_root / pairs_path
    args.pairs = str(pairs_path)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = _project_root / output_dir
    args.output_dir = str(output_dir)

    mrr = train(args)
    return 0 if mrr >= 0.0 else 1


if __name__ == "__main__":
    sys.exit(main())
