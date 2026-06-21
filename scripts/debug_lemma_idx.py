#!/usr/bin/env python3
"""Debug lemma_index format."""
import json

with open('data/graph/dependency_graph_full_v2.lemma_index.json') as f:
    lemma_index = json.load(f)

# Check value types
keys = list(lemma_index.keys())[:10]
for k in keys:
    v = lemma_index[k]
    print(f"  {k}: {v} (type={type(v).__name__})")

# Check unique value types
types = set()
for v in lemma_index.values():
    types.add(type(v).__name__)
    if len(types) > 3:
        break
print(f"\nValue types: {types}")
