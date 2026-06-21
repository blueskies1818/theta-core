#!/usr/bin/env python3
"""Verify lemma_index -> graph idx mapping."""
import json, sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from src.explorer.dependency_graph import DependencyGraph

graph = DependencyGraph.load("data/graph/dependency_graph_full_v3")
print(f"Graph: {graph.num_nodes} nodes")

with open("data/graph/dependency_graph_full_v3.lemma_index.json") as f:
    lemma_index = json.load(f)
print(f"Lemma index: {len(lemma_index)} entries")

# Test first 10 entries
errors = 0
for i, (name, idx) in enumerate(lemma_index.items()):
    node_id = graph.idx_to_node_id(idx)
    if node_id is None and i < 100:
        print(f"  ERROR: idx={idx} for '{name}' not in graph (idx_to_node_id returned None)")
        errors += 1
    if i >= 100:
        break

print(f"Errors in first 100: {errors}")

# Test with actual pairs
with open("data/raw/proof_step_pairs.jsonl") as f:
    from collections import Counter
    resolutions = Counter()
    for line in f:
        d = json.loads(line)
        lemma = d["lemma"]
        if lemma in lemma_index:
            idx = lemma_index[lemma]
            node_id = graph.idx_to_node_id(idx)
            if node_id is not None:
                resolutions["ok"] += 1
            else:
                resolutions["bad_idx"] += 1
        else:
            resolutions["not_in_index"] += 1

print(f"\nPair resolution:")
for k, v in resolutions.most_common():
    print(f"  {k}: {v}")
