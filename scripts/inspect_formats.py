#!/usr/bin/env python3
"""Inspect data formats for Path 1 redo."""
import json

# 1. Lemma index
with open('data/graph/dependency_graph_full_v2.lemma_index.json') as f:
    li = json.load(f)
print("=== lemma_index ===")
print(f"type: {type(li).__name__}")
if isinstance(li, dict):
    keys = list(li.keys())[:5]
    for k in keys:
        val = li[k]
        print(f"  {k}: {json.dumps(val)[:200]}")
    print(f"  total keys: {len(li)}")
elif isinstance(li, list):
    print(f"  first 3: {json.dumps(li[:3], indent=2)[:500]}")
    print(f"  total entries: {len(li)}")

# 2. Proof step pairs
print("\n=== proof_step_pairs (first 3) ===")
with open('data/raw/proof_step_pairs.jsonl') as f:
    for i, line in enumerate(f):
        if i >= 3:
            break
        obj = json.loads(line.strip())
        print(f"\n--- pair {i} ---")
        for k, v in obj.items():
            val_str = str(v)[:150]
            print(f"  {k}: {val_str}")

# 3. V2 stats
print("\n=== v2 stats ===")
with open('data/graph/dependency_graph_full_v2.stats.json') as f:
    print(json.dumps(json.load(f), indent=2))
