#!/usr/bin/env python3
"""Quick check: how many proof-step pairs match graph lemmas."""
import json, sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from src.explorer.dependency_graph import DependencyGraph

graph = DependencyGraph.load("data/graph/dependency_graph_full_v3")
print(f"Graph: {graph.num_nodes} nodes, {graph.num_edges} edges")

# Build lemma index
lemma_to_idx = {}
for node_id in graph.node_ids:
    short_name = node_id.split(".")[-1] if "." in node_id else node_id
    idx = graph.node_id_to_idx(node_id)
    lemma_to_idx[node_id] = idx
    if short_name not in lemma_to_idx:
        lemma_to_idx[short_name] = idx
print(f"Lemma index entries: {len(lemma_to_idx)}")

# Check pairs
pairs_path = _project_root / "data/raw/proof_step_pairs.jsonl"
matched = 0
total = 0
with open(pairs_path) as f:
    for line in f:
        d = json.loads(line)
        total += 1
        if d["lemma"] in lemma_to_idx:
            matched += 1

print(f"Pairs matching graph lemmas: {matched}/{total} ({matched/total:.1%})")
print(f"Pairs NOT matching: {total - matched}")
