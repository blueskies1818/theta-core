"""Diagnostic: check if GNN embeddings contain discriminative signal."""
import sys, torch, json, random, statistics
sys.path.insert(0, '.')
from src.scoring.binary_scorer import FrozenGNNEncoder

print("Loading encoder...")
enc = FrozenGNNEncoder(
    'checkpoints/gnn/gate2_fullgraph_finetuned.pt',
    'data/graph/dependency_graph_full'
)

# Load pairs
with open('data/raw/proof_step_pairs.jsonl') as f:
    pairs = [json.loads(next(f)) for _ in range(500)]
print(f"Loaded {len(pairs)} pairs")

# Encode
goals = torch.stack([enc.encode_goal(p['goal']).cpu() for p in pairs])
lemmas = torch.stack([enc.encode_lemma(p['lemma']).cpu() for p in pairs])
print(f"Goals: {goals.shape}, Lemmas: {lemmas.shape}")

# Cosine similarity pos vs neg
pos_cos = []
neg_cos = []
for i in range(len(pairs)):
    pos_cos.append(torch.dot(goals[i], lemmas[i]).item())
    j = random.randrange(len(pairs))
    while j == i:
        j = random.randrange(len(pairs))
    neg_cos.append(torch.dot(goals[i], lemmas[j]).item())

print(f"Pos cos: mean={statistics.mean(pos_cos):.4f}, stdev={statistics.stdev(pos_cos):.4f}")
print(f"Neg cos: mean={statistics.mean(neg_cos):.4f}, stdev={statistics.stdev(neg_cos):.4f}")

# Simple linear classifier
X_pos = torch.cat([goals, lemmas], dim=1)
shuffled = list(range(500))
random.shuffle(shuffled)
# Avoid self-pairs
for i, s in enumerate(shuffled):
    if s == i:
        nxt = (i + 1) % 500
        shuffled[i], shuffled[nxt] = shuffled[nxt], shuffled[i]
X_neg = torch.cat([goals, lemmas[shuffled]], dim=1)
X = torch.cat([X_pos, X_neg], dim=0)
y = torch.cat([torch.ones(500), torch.zeros(500)])

model = torch.nn.Linear(512, 1)
opt = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    model.train()
    opt.zero_grad()
    logits = model(X).squeeze()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
    loss.backward()
    opt.step()
    
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        acc = ((probs > 0.5).float() == y).float().mean().item()
    
    if epoch % 10 == 0:
        pos_acc = (probs[:500] > 0.5).float().mean().item()
        neg_acc = (probs[500:] <= 0.5).float().mean().item()
        print(f"Epoch {epoch:3d}: loss={loss.item():.4f}, acc={acc:.3f}, "
              f"pos_acc={pos_acc:.3f}, neg_acc={neg_acc:.3f}")
    
    if acc > 0.55:
        print(f"  -> Exceeded 55% at epoch {epoch}!")
        break
else:
    print(f"Final epoch 100: loss={loss.item():.4f}, acc={acc:.3f}")
