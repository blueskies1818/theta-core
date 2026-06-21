#!/usr/bin/env python3
"""Inspect index format."""
import json

with open('data/graph/dependency_graph_full_v2.index.json') as f:
    idx = json.load(f)
print('index type:', type(idx).__name__)
print('total keys:', len(idx))
keys = list(idx.keys())[:5]
for k in keys:
    val_str = json.dumps(idx[k])[:200]
    print(f'  {k}: {val_str}')

# Also check enrichment summary
print("\n=== enrichment_summary ===")
with open('data/graph/dependency_graph_full_v2.enrichment_summary.json') as f:
    print(json.dumps(json.load(f), indent=2)[:500])
