#!/usr/bin/env python3
"""Smoke test for hard-negative training pipeline."""
import json, tempfile, os, sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

# ---- Create synthetic data ---------------------------------------------------
triples = [
    {"goal": "a + b = b + a", "positive_lemma": "add_comm",
     "hard_negatives": ["mul_comm", "sub_add"], "domain": "Algebra"},
    {"goal": "x ≤ x", "positive_lemma": "le_refl",
     "hard_negatives": ["lt_of_lt_of_le"], "domain": "Order"},
    {"goal": "(a * b) * c = a * (b * c)", "positive_lemma": "mul_assoc",
     "hard_negatives": ["add_assoc", "sub_eq"], "domain": "Algebra"},
    {"goal": "x + 0 = x", "positive_lemma": "add_zero",
     "hard_negatives": ["mul_one", "sub_self"], "domain": "Algebra"},
] * 25  # 100 triples

tmpdir = tempfile.mkdtemp()
triples_path = os.path.join(tmpdir, "triples.jsonl")
with open(triples_path, "w") as f:
    for t in triples:
        json.dump(t, f)
        f.write("\n")

# Also create matching pairs data
pairs_path = os.path.join(tmpdir, "pairs.jsonl")
pairs = []
for t in triples:
    pairs.append({"goal": t["goal"], "lemma": t["positive_lemma"], "domain": t["domain"]})
    for hn in t["hard_negatives"]:
        pairs.append({"goal": t["goal"] + " (neg)", "lemma": hn, "domain": t["domain"]})
with open(pairs_path, "w") as f:
    for p in pairs:
        json.dump(p, f)
        f.write("\n")

print(f"Created {len(triples)} triples at {triples_path}")
print(f"Created {len(pairs)} pairs at {pairs_path}")

# ---- Run training for 2 epochs ----------------------------------------------
from scripts.training.train_hard_negative_contrastive import train

class Args:
    data = pairs_path
    hard_neg_data = triples_path
    output = os.path.join(tmpdir, "model.pt")
    train_split = 0.8
    val_size = 20
    no_hard_negatives = False
    hidden_dim = 32
    vocab_size = 256
    max_seq_len = 64
    char_embed_dim = 16
    cnn_filters = 32
    kernel_sizes = [2, 3]
    dropout = 0.1
    mlp_expansion = 1
    pooling = "mean"
    epochs = 2
    batch_size = 16
    lr = 3e-4
    weight_decay = 1e-4
    temperature = 0.07
    hard_neg_weight = 0.5
    margin = 0.3
    seed = 42
    device = "cpu"
    num_threads = 2

args = Args()
stats = train(args)

print(f"\nSmoke test results:")
print(f"  Best val acc: {stats['best_val_acc']:.3f}")
print(f"  Best val loss: {stats['best_val_loss']:.4f}")
print(f"  Time: {stats['total_time']:.1f}s")
print(f"  Hard neg triples: {stats['num_hard_neg_triples']}")

# Verify no NaN
for epoch_loss in stats["history"]["train_loss"]:
    assert not (epoch_loss != epoch_loss), f"NaN in train loss: {epoch_loss}"
assert stats["best_val_acc"] > 0.0, "Accuracy should be > 0"

print("\nSMOKE TEST PASSED")
