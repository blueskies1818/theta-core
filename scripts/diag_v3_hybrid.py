"""Diagnostic v3: Domain + text features alongside GNN embeddings."""
import torch, json, random, statistics, sys, re
sys.path.insert(0, '.')
import torch.nn as nn
import torch.nn.functional as F
from src.scoring.binary_scorer import FrozenGNNEncoder

print("Loading multitask_v3...")
enc = FrozenGNNEncoder('checkpoints/gnn/multitask_v3.pt', 'data/graph/dependency_graph_full')

with open('data/raw/proof_step_pairs.jsonl') as f:
    pairs = [json.loads(next(f)) for _ in range(10000)]
print(f"Loaded {len(pairs)} pairs")

# Collect all domains
domains = sorted(set(p.get('domain', 'unknown') for p in pairs))
domain_to_idx = {d: i for i, d in enumerate(domains)}
n_domains = len(domains)
print(f"Domains: {n_domains} - {domains}")

# Build char-bigram features for lemma names
all_lemmas = {p['lemma'] for p in pairs}
char_bigrams = set()
for lm in all_lemmas:
    lm_lower = lm.lower()
    for i in range(len(lm_lower) - 1):
        char_bigrams.add(lm_lower[i:i+2])
bigram_to_idx = {bg: i for i, bg in enumerate(sorted(char_bigrams))}
n_bigrams = len(bigram_to_idx)
print(f"Char bigrams: {n_bigrams}")

# Encode GNN features
goals_gnn = torch.stack([enc.encode_goal(p['goal']).cpu() for p in pairs])  # [N, 256]
lemmas_gnn = torch.stack([enc.encode_lemma(p['lemma']).cpu() for p in pairs])  # [N, 256]

# Encode domain (one-hot)
def domain_vec(domain):
    v = torch.zeros(n_domains)
    if domain in domain_to_idx:
        v[domain_to_idx[domain]] = 1.0
    return v

# Encode bigrams (multi-hot, capped)
def bigram_vec(lemma_name):
    v = torch.zeros(min(200, n_bigrams))
    lm = lemma_name.lower()
    added = 0
    for i in range(len(lm) - 1):
        bg = lm[i:i+2]
        if bg in bigram_to_idx and added < 200:
            idx = bigram_to_idx[bg] % 200
            v[idx] = 1.0
            added += 1
    return v

domains_t = torch.stack([domain_vec(p.get('domain', 'unknown')) for p in pairs])
bigrams_t = torch.stack([bigram_vec(p['lemma']) for p in pairs])

# Features: [gnn_goal(256) || gnn_lemma(256) || goal*lemma(256) || domain(goal) || domain(lemma) || lemma_bigrams]
domain_dim = domains_t.size(1)
bigram_dim = min(200, n_bigrams)
feat_dim = 256 + 256 + 256 + n_domains * 2 + bigram_dim

print(f"Total feature dim: {feat_dim}")

n = len(pairs)
# Positives
dom_goal = domains_t
dom_lemma = domains_t
X_pos = torch.cat([
    goals_gnn, lemmas_gnn, goals_gnn * lemmas_gnn,
    dom_goal, dom_lemma, bigrams_t
], dim=1)
y_pos = torch.ones(n)

# Negatives: shuffle lemmas
indices = list(range(n))
random.shuffle(indices)
for i, s in enumerate(indices):
    if s == i:
        nxt = (i + 1) % n
        indices[i], indices[nxt] = indices[nxt], indices[i]

X_neg = torch.cat([
    goals_gnn, lemmas_gnn[indices], goals_gnn * lemmas_gnn[indices],
    dom_goal, domains_t[indices], bigrams_t[indices]
], dim=1)
y_neg = torch.zeros(n)

X = torch.cat([X_pos, X_neg], dim=0)
y = torch.cat([y_pos, y_neg])

# Simple model
class HybridScorer(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
    def forward(self, x):
        return self.net(x)

model = HybridScorer(feat_dim)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# Train/val split
n_train = int(n * 0.85)
train_idx = list(range(n_train)) + list(range(n, n + n_train))
val_idx = list(range(n_train, n)) + list(range(n + n_train, 2*n))
X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")

best_val_acc = 0
for epoch in range(50):
    model.train()
    opt.zero_grad()
    logits = model(X_train).squeeze()
    loss = F.binary_cross_entropy_with_logits(logits, y_train)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    
    model.eval()
    with torch.no_grad():
        val_logits = model(X_val).squeeze()
        val_loss = F.binary_cross_entropy_with_logits(val_logits, y_val)
        val_probs = torch.sigmoid(val_logits)
        val_acc = ((val_probs > 0.5).float() == y_val).float().mean().item()
        best_val_acc = max(best_val_acc, val_acc)
    
    if epoch % 10 == 0:
        nv = n - n_train
        pos_acc = (val_probs[:nv] > 0.5).float().mean().item()
        neg_acc = (val_probs[nv:] <= 0.5).float().mean().item()
        print(f"Epoch {epoch:3d}: train_loss={loss.item():.4f}, val_loss={val_loss.item():.4f}, "
              f"val_acc={val_acc:.3f}, pos={pos_acc:.3f}, neg={neg_acc:.3f}")

print(f"\nBest val_acc: {best_val_acc:.3f} {'PASS' if best_val_acc > 0.55 else 'FAIL'}")
