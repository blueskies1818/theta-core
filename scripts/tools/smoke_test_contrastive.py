#!/usr/bin/env python3
"""Quick smoke test for contrastive training pipeline."""
import sys, time, json, random
from pathlib import Path
import torch

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from src.contrastive.encoder import ContrastiveDualEncoder, ContrastiveConfig, CharTokenizer

t0 = time.time()

# Load small subset
with open("/tmp/smoke_pairs.jsonl") as f:
    pairs = [json.loads(l) for l in f]
print(f"Loaded {len(pairs)} pairs in {time.time()-t0:.1f}s")

# Tiny config for fast smoke test
config = ContrastiveConfig(
    hidden_dim=64, char_embed_dim=16, cnn_filters=32,
    cnn_kernel_sizes=(2, 3), max_seq_len=128, batch_size=32,
    num_epochs=2, temperature=0.07,
)
tokenizer = CharTokenizer(max_len=config.max_seq_len)

# Use 500 pairs
subset = pairs[:500]

# Build lemma cache
unique_lemmas = sorted(set(p["lemma"] for p in subset))
lemma_to_ids = {}
for lemma in unique_lemmas:
    lemma_text = tokenizer.preprocess_lemma(lemma)
    lemma_to_ids[lemma] = tokenizer.encode(lemma_text)
print(f"Tokenized {len(unique_lemmas)} lemmas in {time.time()-t0:.1f}s")

# Tokenize goals
goal_ids = torch.zeros(len(subset), config.max_seq_len, dtype=torch.long)
for i, p in enumerate(subset):
    goal_text = tokenizer.preprocess_goal(p["goal"])
    goal_ids[i] = tokenizer.encode(goal_text)
print(f"Tokenized goals in {time.time()-t0:.1f}s")

# Build model
model = ContrastiveDualEncoder(config)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
print(f"Params: {model.num_params:,}")

# Train 2 epochs
indices = list(range(len(subset)))
for epoch in range(2):
    random.shuffle(indices)
    total_loss = 0.0
    n = 0
    for i in range(0, len(indices), 32):
        batch_idx = indices[i:i+32]
        batch_goal_ids = goal_ids[batch_idx]
        batch_lemma_names = [subset[j]["lemma"] for j in batch_idx]
        batch_lemma_ids = torch.stack([lemma_to_ids[n] for n in batch_lemma_names])

        out = model(batch_goal_ids, batch_lemma_ids)
        loss = out["loss"]

        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()
        n += 1

    avg_loss = total_loss / max(1, n)
    # Check that loss is not NaN and decreases
    assert not torch.isnan(torch.tensor(avg_loss)), f"NaN loss at epoch {epoch}"
    print(f"Epoch {epoch}: loss={avg_loss:.4f}")

# Verify save/load
model.save("/tmp/smoke_model.pt")
loaded = ContrastiveDualEncoder.load("/tmp/smoke_model.pt")
test_ids = goal_ids[:4]
test_output = loaded.encode_goal(test_ids)
assert test_output.shape == (4, 64), f"Wrong shape: {test_output.shape}"

print(f"Smoke test PASSED in {time.time()-t0:.1f}s")
print(f"Final loss: {avg_loss:.4f}")
