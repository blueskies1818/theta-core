import json, re, sys
from collections import Counter

tokens = Counter()
goals_sample = []
with open('data/raw/proof_step_pairs.jsonl') as f:
    for i, line in enumerate(f):
        if i >= 5000:
            break
        p = json.loads(line)
        goals_sample.append(p['goal'][:120])
        parts = re.split(r'[\s,:(){}\[\]]+', p['goal'])
        for part in parts:
            part = part.strip().lower()
            if 2 <= len(part) <= 50:
                tokens[part] += 1

print(f'Unique tokens: {len(tokens)}')
print('Top 30:', [(t, c) for t, c in tokens.most_common(30)])
print()
print('Sample goals:')
for g in goals_sample[:5]:
    print(f'  {g}')
