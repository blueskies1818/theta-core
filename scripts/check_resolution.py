#!/usr/bin/env python3
"""Check how many theorem names from proof_step_pairs resolve in the graph index."""
import json

# Load indices
with open('data/graph/dependency_graph_full_v2.index.json') as f:
    index = json.load(f)
with open('data/graph/dependency_graph_full_v2.lemma_index.json') as f:
    lemma_index = json.load(f)

# Load pairs
pairs = []
with open('data/raw/proof_step_pairs.jsonl') as f:
    for line in f:
        if line.strip():
            pairs.append(json.loads(line.strip()))

print(f"Index entries: {len(index)}")
print(f"Lemma index entries: {len(lemma_index)}")
print(f"Pairs: {len(pairs)}")

# Check resolution
goal_resolved = 0
lemma_resolved = 0
both_resolved = 0
goal_not_found = set()
lemma_not_found = set()

sample_missing_goals = []
sample_missing_lemmas = []

for p in pairs:
    name = p['name']
    lemma = p['lemma']
    
    g_resolved = name in index
    l_resolved = lemma in lemma_index
    
    if g_resolved:
        goal_resolved += 1
    elif len(sample_missing_goals) < 10:
        sample_missing_goals.append(name)
    
    if l_resolved:
        lemma_resolved += 1
    elif len(sample_missing_lemmas) < 10:
        sample_missing_lemmas.append(lemma)
    
    if g_resolved and l_resolved:
        both_resolved += 1

print(f"\nGoal (theorem name) resolved: {goal_resolved}/{len(pairs)} ({goal_resolved/len(pairs):.1%})")
print(f"Lemma resolved: {lemma_resolved}/{len(pairs)} ({lemma_resolved/len(pairs):.1%})")
print(f"Both resolved: {both_resolved}/{len(pairs)} ({both_resolved/len(pairs):.1%})")
print(f"\nSample missing goals: {sample_missing_goals[:10]}")
print(f"Sample missing lemmas: {sample_missing_lemmas[:10]}")
