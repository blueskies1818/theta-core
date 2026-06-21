import json, random
from collections import defaultdict, Counter

PAIRS = "data/raw/proof_step_pairs.jsonl"
NUM_SAMPLES = 10000

def jaccard(sa, sb):
    if not sa or not sb: return 0.0
    return len(sa & sb) / len(sa | sb)

# Load
pairs = []
with open(PAIRS) as f:
    for line in f:
        pairs.append(json.loads(line))

# Build goal -> lemmas
goal_lemmas = defaultdict(set)
for p in pairs:
    goal_lemmas[p['goal']].add(p['lemma'])

goals = list(goal_lemmas.keys())
print(f"Goals: {len(goals)}")

# Filter to goals with ≥2 tokens (using simple split)
import re
TOK_RE = re.compile(r"[^\s,:(){}\[\]]+")
def tokens(g):
    return [t.lower() for t in TOK_RE.findall(g) if len(t) >= 2]

goals_tok = [(g, set(tokens(g))) for g in goals]
goals_tok = [(g, t) for g, t in goals_tok if len(t) >= 2]
print(f"Goals with ≥2 tokens: {len(goals_tok)}")

# Sample pairs
random.seed(42)
if len(goals_tok) > NUM_SAMPLES:
    sampled = random.sample(goals_tok, NUM_SAMPLES)
else:
    sampled = goals_tok

# Compute correlation
token_ov = []
lemma_ov = []
for i in range(len(sampled)):
    for j in range(i+1, len(sampled)):
        g1, t1 = sampled[i]
        g2, t2 = sampled[j]
        tok_sim = jaccard(t1, t2)
        lem_sim = jaccard(goal_lemmas[g1], goal_lemmas[g2])
        token_ov.append(tok_sim)
        lemma_ov.append(lem_sim)
        if len(token_ov) >= 50000:
            break
    if len(token_ov) >= 50000:
        break

import statistics
print(f"\nComputed {len(token_ov)} pairs")
print(f"Token overlap  mean={statistics.mean(token_ov):.4f}  median={statistics.median(token_ov):.4f}  max={max(token_ov):.4f}")
print(f"Lemma overlap  mean={statistics.mean(lemma_ov):.4f}  median={statistics.median(lemma_ov):.4f}  max={max(lemma_ov):.4f}")

# Pearson correlation
mean_t = statistics.mean(token_ov)
mean_l = statistics.mean(lemma_ov)
num = sum((t - mean_t) * (l - mean_l) for t, l in zip(token_ov, lemma_ov))
den_t = sum((t - mean_t)**2 for t in token_ov) ** 0.5
den_l = sum((l - mean_l)**2 for l in lemma_ov) ** 0.5
r = num / (den_t * den_l) if den_t > 0 and den_l > 0 else 0
print(f"Pearson r: {r:.4f}")

# What fraction of pairs with token overlap > 0.3 also have lemma overlap > 0?
high_tok = sum(1 for t, l in zip(token_ov, lemma_ov) if t > 0.3 and l > 0)
total_high_tok = sum(1 for t in token_ov if t > 0.3)
print(f"\nPairs with token overlap > 0.3: {total_high_tok}")
print(f"  Of those, share lemma: {high_tok} ({100*high_tok/max(1,total_high_tok):.1f}%)")
