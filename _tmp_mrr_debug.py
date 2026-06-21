# Quick MRR sanity check
import json, sys, random
from collections import defaultdict, Counter
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))

import torch
import torch.nn.functional as F
from src.explorer.goal_only_encoder import GoalOnlyEncoder, tokenize_goal

PAIRS = "data/raw/proof_step_pairs.jsonl"
MODEL = "checkpoints/gnn/goal_only/goal_only_encoder.pt"
NUM_GOALS = 1000
MIN_LEMMA_FREQ = 3

# Load model
model = GoalOnlyEncoder.load(MODEL)
model.eval()
device = torch.device("cpu")

# Load pairs, build same data as training
print("Loading...")
all_pairs = []
with open(PAIRS) as f:
    for line in f:
        all_pairs.append(json.loads(line))

goal_to_lemmas = defaultdict(set)
for p in all_pairs:
    goal_to_lemmas[p['goal']].add(p['lemma'])

# Filter frequent lemmas
lemma_freq = Counter()
for lemmas in goal_to_lemmas.values():
    lemma_freq.update(lemmas)
freq_lemmas = {l for l, c in lemma_freq.items() if c >= MIN_LEMMA_FREQ}
goals = [g for g in goal_to_lemmas if goal_to_lemmas[g] & freq_lemmas]
random.seed(42)
goals = random.sample(goals, NUM_GOALS)

split = int(len(goals) * 0.8)
train_goals = goals[:split]
val_goals = goals[split:]

print(f"Train: {len(train_goals)}, Val: {len(val_goals)}")

# Pre-encode
with torch.no_grad():
    train_embs = model(train_goals, device)
    train_embs = F.normalize(train_embs, dim=-1)
    val_embs = model(val_goals, device)
    val_embs = F.normalize(val_embs, dim=-1)

# MRR with direct lemma lookup
mrr_sum = 0
mrr_count = 0
lemma_hits = 0
total = 0

for i, val_goal in enumerate(val_goals):
    val_lemmas = goal_to_lemmas[val_goal] & freq_lemmas
    if not val_lemmas:
        continue
    
    # Find nearest training goals
    sims = (val_embs[i:i+1] @ train_embs.T).squeeze(0)
    topk_idx = torch.topk(sims, min(50, len(train_goals))).indices
    
    # Collect lemmas from top-k
    lemma_scores = defaultdict(float)
    for j, idx in enumerate(topk_idx):
        sim = max(0.0, sims[idx].item())
        g = train_goals[idx.item()]
        for lemma in goal_to_lemmas[g] & freq_lemmas:
            lemma_scores[lemma] += sim
    
    # Check rank of best correct lemma
    sorted_lemmas = sorted(lemma_scores.items(), key=lambda x: -x[1])
    best_rank = len(sorted_lemmas) + 1
    for rank, (lem, score) in enumerate(sorted_lemmas, 1):
        if lem in val_lemmas:
            best_rank = min(best_rank, rank)
            break
    
    mrr_sum += 1.0 / best_rank
    mrr_count += 1
    
    if best_rank <= 50:
        lemma_hits += 1
    total += 1

mrr = mrr_sum / max(1, mrr_count)
print(f"MRR: {mrr:.4f}")
print(f"Lemma hits (rank≤50): {lemma_hits}/{total} ({100*lemma_hits/max(1,total):.1f}%)")
print(f"Best possible MRR (if correct lemma always at rank 1): 1.0")

# Check: for each val goal, is its lemma in training goals' lemma sets?
lemma_in_train = 0
for val_goal in val_goals:
    val_lemmas = goal_to_lemmas[val_goal] & freq_lemmas
    train_lemmas = set()
    for g in train_goals:
        train_lemmas.update(goal_to_lemmas[g] & freq_lemmas)
    if val_lemmas & train_lemmas:
        lemma_in_train += 1
print(f"Val goals with correct lemma in training: {lemma_in_train}/{len(val_goals)}")
