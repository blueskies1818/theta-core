import json, re, time
from collections import Counter

_TOKENIZE_RE = re.compile(r"[^\s,:(){}\[\]]+")

def tokenize_goal(goal_text, max_tokens=64):
    tokens = []
    for tok in _TOKENIZE_RE.findall(goal_text):
        tok = tok.strip().lower()
        if 1 <= len(tok) <= 50:
            tokens.append(tok)
            if len(tokens) >= max_tokens:
                break
    return tokens

t0 = time.time()
goals = []
with open('data/raw/proof_step_pairs.jsonl') as f:
    seen = set()
    for line in f:
        p = json.loads(line)
        g = p['goal']
        if g not in seen:
            seen.add(g)
            goals.append(g)
print(f'Loaded {len(goals)} unique goals in {time.time()-t0:.1f}s')

counter = Counter()
for i, goal in enumerate(goals):
    for tok in tokenize_goal(goal):
        counter[tok] += 1
    if i % 10000 == 0:
        print(f'  {i}/{len(goals)}...')

print(f'Built vocab: {len(counter)} tokens in {time.time()-t0:.1f}s')
print(f'Top 20: {counter.most_common(20)}')
