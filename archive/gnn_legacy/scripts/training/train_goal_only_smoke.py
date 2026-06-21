#!/usr/bin/env python3
"""
Goal-Only GNN — Smoke Test with combined signal.

Uses BOTH lemma-sharing AND token overlap as positive signal.
This dramatically reduces noise compared to lemma-only training.

Train on a small subset, verify MRR > 0.3.

Usage:
  python scripts/training/train_goal_only_smoke.py
"""
import json, sys, re, random, time
from collections import defaultdict, Counter
from pathlib import Path
import torch
import torch.nn.functional as F

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))

from src.explorer.goal_only_encoder import (
    GoalOnlyEncoder, build_vocab, tokenize_goal
)

# ── Config ──
PAIRS = "data/raw/proof_step_pairs.jsonl"
NUM_GOALS = 3000
MIN_LEMMA_FREQ = 3
MIN_TOKEN_OVERLAP = 0.3  # Jaccard for positive pair
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
OUTPUT = "checkpoints/gnn/goal_only/goal_only_encoder.pt"

def jaccard(set_a, set_b):
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)

def main():
    device = torch.device("cpu")
    torch.set_num_threads(4)

    # Load pairs, sample goals
    print("Loading pairs...")
    all_pairs = []
    with open(PAIRS) as f:
        for line in f:
            all_pairs.append(json.loads(line))
    print(f"  Total pairs: {len(all_pairs)}")

    # Build goal → lemmas
    goal_to_lemmas = defaultdict(set)
    for p in all_pairs:
        goal_to_lemmas[p['goal']].add(p['lemma'])
    all_goals = list(goal_to_lemmas.keys())
    print(f"  Unique goals: {len(all_goals)}")

    # Filter to frequent lemmas
    lemma_freq = Counter()
    for lemmas in goal_to_lemmas.values():
        lemma_freq.update(lemmas)
    freq_lemmas = {l for l, c in lemma_freq.items() if c >= MIN_LEMMA_FREQ}
    print(f"  Frequent lemmas (≥{MIN_LEMMA_FREQ}): {len(freq_lemmas)}")

    # Filter goals to those with frequent lemmas
    goals_filt = [g for g in all_goals
                  if goal_to_lemmas[g] & freq_lemmas]
    print(f"  Goals with frequent lemmas: {len(goals_filt)}")

    # Sample NUM_GOALS
    random.seed(42)
    if len(goals_filt) > NUM_GOALS:
        goals = random.sample(goals_filt, NUM_GOALS)
    else:
        goals = goals_filt
    print(f"  Sampled {len(goals)} goals")

    # Build token sets for each goal
    print("Building token sets...")
    goal_tokens = {}
    for g in goals:
        goal_tokens[g] = set(tokenize_goal(g))
    print(f"  Average tokens/goal: {sum(len(t) for t in goal_tokens.values())/max(1,len(goals)):.1f}")

    # Train/val split
    random.shuffle(goals)
    split = int(len(goals) * 0.8)
    train_goals = goals[:split]
    val_goals = goals[split:]
    print(f"  Train: {len(train_goals)}, Val: {len(val_goals)}")

    # Build positive pairs: goals with high token overlap AND shared lemma
    print("Building positive pairs...")
    train_set = set(train_goals)
    positive_pairs = []
    for i, g1 in enumerate(train_goals):
        t1 = goal_tokens[g1]
        l1 = goal_to_lemmas[g1] & freq_lemmas
        for g2 in train_goals:
            if g1 >= g2:  # only upper triangle, unique pairs
                continue
            t2 = goal_tokens[g2]
            l2 = goal_to_lemmas[g2] & freq_lemmas
            # Combined signal: token overlap OR lemma sharing
            jac = jaccard(t1, t2)
            shares_lemma = bool(l1 & l2)
            if jac >= MIN_TOKEN_OVERLAP or shares_lemma:
                positive_pairs.append((g1, g2))

    print(f"  Positive pairs: {len(positive_pairs)}")
    if len(positive_pairs) < 10:
        print("  ERROR: too few positive pairs")
        sys.exit(1)

    # Build vocab
    print("Building vocab...")
    vocab = build_vocab(goals, vocab_size=4000)
    print(f"  Vocab size: {len(vocab)}")

    # Model
    print("Initializing model...")
    model = GoalOnlyEncoder(
        vocab=vocab,
        embed_dim=128,
        hidden_dim=256,
        num_layers=0,  # BoW
        num_heads=4,
        ffn_dim=256,
        max_tokens=64,
        dropout=0.1,
    ).to(device)
    n_params = model.count_parameters()
    print(f"  Params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Pre-encode val goals
    print("Pre-encoding validation goals...")
    model.eval()
    with torch.no_grad():
        val_embs = model(val_goals, device)
        val_embs = F.normalize(val_embs, dim=-1)

    # Build val MRR gold: for each val goal, find correct lemmas
    val_gold_lemmas = {g: goal_to_lemmas[g] & freq_lemmas for g in val_goals}

    # Training
    print(f"\n{'='*50}")
    print(f"Training: {EPOCHS} epochs, batch={BATCH_SIZE}")
    print(f"{'='*50}")

    best_mrr = 0.0

    for epoch in range(EPOCHS):
        t0 = time.time()
        model.train()

        # Shuffle positive pairs
        random.shuffle(positive_pairs)

        epoch_loss = 0.0
        epoch_pos = 0.0
        epoch_neg = 0.0
        n_batches = 0

        for b_start in range(0, len(positive_pairs), BATCH_SIZE):
            batch_pairs = positive_pairs[b_start:b_start + BATCH_SIZE]
            if len(batch_pairs) < 2:
                continue

            # Anchor = first goal, Positive = second goal
            anchors = [p[0] for p in batch_pairs]
            positives = [p[1] for p in batch_pairs]

            # Negatives: random goals NOT in anchor's lemma set
            negatives = []
            for anchor in anchors:
                anchor_lemmas = goal_to_lemmas[anchor] & freq_lemmas
                neg = random.choice(train_goals)
                neg_lemmas = goal_to_lemmas[neg] & freq_lemmas
                attempts = 0
                while (neg_lemmas & anchor_lemmas) and attempts < 50:
                    neg = random.choice(train_goals)
                    neg_lemmas = goal_to_lemmas[neg] & freq_lemmas
                    attempts += 1
                negatives.append(neg)

            # Encode
            anc_embs = model(anchors, device)
            pos_embs = model(positives, device)
            neg_embs = model(negatives, device)

            # Normalize
            anc_embs = F.normalize(anc_embs, dim=-1)
            pos_embs = F.normalize(pos_embs, dim=-1)
            neg_embs = F.normalize(neg_embs, dim=-1)

            # Loss: maximize pos, minimize neg + variance penalty
            pos_cos = (anc_embs * pos_embs).sum(dim=-1).mean()
            neg_cos = (anc_embs * neg_embs).sum(dim=-1).mean()
            var = anc_embs.var(dim=0).mean()
            loss = -pos_cos + neg_cos + 0.01 * torch.relu(0.01 - var)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_pos += pos_cos.item()
            epoch_neg += neg_cos.item()
            n_batches += 1

        # Validation MRR
        model.eval()
        with torch.no_grad():
            train_embs = model(train_goals, device)
            train_embs = F.normalize(train_embs, dim=-1)

            # Build lemma map for training goals
            train_lemma_to_goals = defaultdict(list)
            for idx, g in enumerate(train_goals):
                for lemma in goal_to_lemmas[g] & freq_lemmas:
                    train_lemma_to_goals[lemma].append(idx)

        # Compute MRR
        mrr_sum = 0.0
        mrr_count = 0
        with torch.no_grad():
            for i, val_goal in enumerate(val_goals):
                correct_lemmas = val_gold_lemmas[val_goal]
                if not correct_lemmas:
                    continue

                val_emb = val_embs[i:i+1]
                sims = (val_emb @ train_embs.T).squeeze(0)
                topk_sims, topk_idx = torch.topk(sims, min(50, len(train_goals)))

                lemma_scores = defaultdict(float)
                for j in range(len(topk_idx)):
                    idx = topk_idx[j].item()
                    sim = max(0.0, topk_sims[j].item())
                    g = train_goals[idx]
                    for lemma in goal_to_lemmas[g] & freq_lemmas:
                        lemma_scores[lemma] += sim

                sorted_lemmas = sorted(lemma_scores.items(), key=lambda x: -x[1])
                rank = len(sorted_lemmas) + 1
                for r, (lem, _) in enumerate(sorted_lemmas, 1):
                    if lem in correct_lemmas:
                        rank = r
                        break
                mrr_sum += 1.0 / rank
                mrr_count += 1

        mrr = mrr_sum / max(1, mrr_count)
        elapsed = time.time() - t0

        avg_loss = epoch_loss / max(1, n_batches)
        avg_pos = epoch_pos / max(1, n_batches)
        avg_neg = epoch_neg / max(1, n_batches)

        print(f"Epoch {epoch+1:3d}/{EPOCHS}  loss={avg_loss:.4f}  "
              f"pos_cos={avg_pos:.3f}  neg_cos={avg_neg:.3f}  "
              f"MRR={mrr:.4f}  time={elapsed:.1f}s")

        if mrr > best_mrr:
            best_mrr = mrr
            model.save(OUTPUT)
            print(f"  ✓ Saved (MRR={best_mrr:.4f})")

        if mrr > 0.3:
            print(f"\n  SMOKE TEST PASSED: MRR={mrr:.4f} > 0.3!")
            break

    print(f"\nBest MRR: {best_mrr:.4f}")
    return 0 if best_mrr > 0.3 else 1

if __name__ == "__main__":
    sys.exit(main())
