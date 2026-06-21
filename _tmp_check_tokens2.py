import json, re
from collections import Counter

_TOKENIZE_RE = re.compile(r"[^\s,:(){}\[\]]+")
_MATH_SINGLE_CHARS = set("+-*/^=<>≤≥∀∃∈⊆⊂∪∩∑∏∫∂√∞→←⇒⇔λμπσθαβγδε")
_LEAN_SINGLE_CHARS = set("αβγδεζηθικλμνξπρστυφχψω")

def tokenize_goal(goal_text, max_tokens=64):
    tokens = []
    for tok in _TOKENIZE_RE.findall(goal_text):
        tok = tok.strip().lower()
        if not tok or len(tok) > 50:
            continue
        if len(tok) >= 2:
            tokens.append(tok)
        elif tok in _MATH_SINGLE_CHARS:
            tokens.append(tok)
        elif tok in _LEAN_SINGLE_CHARS:
            tokens.append(tok)
        if len(tokens) >= max_tokens:
            break
    return tokens

counter = Counter()
goals = []
with open('data/raw/proof_step_pairs.jsonl') as f:
    seen = set()
    for line in f:
        p = json.loads(line)
        g = p['goal']
        if g not in seen:
            seen.add(g)
            goals.append(g)

for goal in goals:
    for tok in tokenize_goal(goal):
        counter[tok] += 1

print(f'Unique goals: {len(goals)}')
print(f'Unique tokens: {len(counter)}')
print(f'Top 30: {counter.most_common(30)}')
print(f'Sample goals:')
for g in goals[:5]:
    tokens = tokenize_goal(g)
    print(f'  [{len(tokens)} tokens] {g[:100]}')
    print(f'    tokens: {tokens[:20]}')
