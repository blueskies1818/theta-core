"""Diagnostic v2: element-wise product features + BatchNorm MLP."""
import torch, json, random, statistics, sys
sys.path.insert(0, '.')
import torch.nn as nn
import torch.nn.functional as F
from src.scoring.binary_scorer import FrozenGNNEncoder

print("Loading multitask_v3...")
enc = FrozenGNNEncoder('checkpoints/gnn/multitask_v3.pt', 'data/graph/dependency_graph_full')

# More pairs for training
with open('data/raw/proof_step_pairs.jsonl') as f:
    pairs = [json.loads(next(f)) for _ in range(5000)]
print(f"Loaded {len(pairs)} pairs")

goals = torch.stack([enc.encode_goal(p['goal']).cpu() for p in pairs])
lemmas = torch.stack([enc.encode_lemma(p['lemma']).cpu() for p in pairs])

# Build balanced dataset: 1 positive + 1 negative per pair
n = len(pairs)
# Positives
X_pos = torch.cat([goals, lemmas, goals * lemmas], dim=1)  # [N, 768]
y_pos = torch.ones(n)

# Negatives: shuffle lemmas (avoid self)
indices = list(range(n))
random.shuffle(indices)
for i, s in enumerate(indices):
    if s == i:
        nxt = (i + 1) % n
        indices[i], indices[nxt] = indices[nxt], indices[i]
X_neg = torch.cat([goals, lemmas[indices], goals * lemmas[indices]], dim=1)
y_neg = torch.zeros(n)

X = torch.cat([X_pos, X_neg], dim=0)
y = torch.cat([y_pos, y_neg])

# MLP with BatchNorm: 768 → 256 → 128 → 1
class Scorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(768),
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 1),
        )
    def forward(self, x):
        return self.net(x)

model = Scorer()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

# Train/val split
n_train = int(n * 0.9)
# Interleave: positives first, then negatives
train_idx = list(range(n_train)) + list(range(n, n + n_train))
val_idx = list(range(n_train, n)) + list(range(n + n_train, 2*n))
X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]

print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")

for epoch in range(30):
    model.train()
    opt.zero_grad()
    logits = model(X_train).squeeze()
    loss = F.binary_cross_entropy_with_logits(logits, y_train)
    loss.backward()
    opt.step()
    
    model.eval()
    with torch.no_grad():
        val_logits = model(X_val).squeeze()
        val_loss = F.binary_cross_entropy_with_logits(val_logits, y_val)
        val_probs = torch.sigmoid(val_logits)
        val_acc = ((val_probs > 0.5).float() == y_val).float().mean().item()
        pos_acc = (val_probs[:n//10] > 0.5).float().mean().item()
        neg_acc = (val_probs[n//10:] <= 0.5).float().mean().item()
    
    if epoch % 5 == 0:
        print(f"Epoch {epoch:3d}: train_loss={loss.item():.4f}, val_loss={val_loss.item():.4f}, "
              f"val_acc={val_acc:.3f}, pos_acc={pos_acc:.3f}, neg_acc={neg_acc:.3f}")

model.eval()
with torch.no_grad():
    final_logits = model(X_val).squeeze()
    final_acc = ((torch.sigmoid(final_logits) > 0.5).float() == y_val).float().mean().item()
print(f"\nFinal val_acc: {final_acc:.3f} {'PASS' if final_acc > 0.55 else 'FAIL'}")
