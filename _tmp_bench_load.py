import json, sys, time
from collections import defaultdict
t0=time.time()
pairs=[]
with open('data/raw/proof_step_pairs.jsonl') as f:
    for line in f:
        pairs.append(json.loads(line))
print(f'Loaded {len(pairs)} pairs in {time.time()-t0:.1f}s')
goal_map=defaultdict(set)
for p in pairs:
    goal_map[p['goal']].add(p['lemma'])
print(f'Built goal map: {len(goal_map)} goals in {time.time()-t0:.1f}s')
