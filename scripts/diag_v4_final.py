"""Test: can the MLP learn the ANTI-correlation signal?"""
import torch, json, random, statistics, sys
sys.path.insert(0, '.')
import torch.nn as nn
import torch.nn.functional as F
from src.scoring.binary_scorer import FrozenGNNEncoder

print("Loading multitask_v3 (best signal, even if inverted)...")
enc = FrozenGNNEncoder('checkpoints/gnn/multitask_v3.pt', 'data/graph/dependency_graph_full')

with open('data/raw/proof_step_pairs.jsonl') as f:
    pairs = [json.loads(next(f)) for _ in range(2000)]
print(f"Loaded {len(pairs)} pairs")

goals = torch.stack([enc.encode_goal(p['goal']).cpu() for p in pairs])
lemmas = torch.stack([enc.encode_lemma(p['lemma']).cpu() for p in pairs])

n = len(pairs)
# Features: concat + product + diff
X_pos = torch.cat([goals, lemmas, goals * lemmas, goals - lemmas], dim=1)  # [N, 1024]
y_pos = torch.ones(n)

# Negatives
indices = list(range(n))
random.shuffle(indices)
for i, s in enumerate(indices):
    if s == i:
        nxt = (i + 1) % n
        indices[i], indices[nxt] = indices[nxt], indices[i]
X_neg = torch.cat([goals, lemmas[indices], goals * lemmas[indices], goals - lemmas[indices]], dim=1)
y_neg = torch.zeros(n)

X = torch.cat([X_pos, X_neg], dim=0)
y = torch.cat([y_pos, y_neg])

# Simple model: 1024 → 256 → 64 → 1
model = nn.Sequential(
    nn.Linear(1024, 256), nn.ReLU(), nn.Dropout(0.2),
    nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.2),
    nn.Linear(64, 1),
)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# Train/val split
n_train = int(n * 0.85)
train_idx = list(range(n_train)) + list(range(n, n + n_train))
val_idx = list(range(n_train, n)) + list(range(n + n_train, 2*n))
X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")

best_val_acc = 0
best_epoch = 0
for epoch in range(100):
    model.train()
    opt.zero_grad()
    logits = model(X_train).squeeze()
    loss = F.binary_cross_entropy_with_logits(logits, y_train)
    loss.backward()
    opt.step()
    
    model.eval()
    with torch.no_grad():
        val_logits = model(X_val).squeeze()
        val_probs = torch.sigmoid(val_logits)
        val_acc = ((val_probs > 0.5).float() == y_val).float().mean().item()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
    
    if epoch % 20 == 0:
        nv = n - n_train
        pos_acc = (val_probs[:nv] > 0.5).float().mean().item()
        neg_acc = (val_probs[nv:] <= 0.5).float().mean().item()
        print(f"Epoch {epoch:3d}: loss={loss.item():.4f}, val_acc={val_acc:.3f}, "
              f"pos={pos_acc:.3f}, neg={neg_acc:.3f}")

print(f"\nBest val_acc: {best_val_acc:.3f} at epoch {best_epoch} "
      f"{'PASS' if best_val_acc > 0.55 else 'FAIL (signal inverted/weak)'}")
