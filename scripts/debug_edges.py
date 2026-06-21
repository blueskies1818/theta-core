#!/usr/bin/env python3
"""Debug why edges aren't being added."""
import json, pickle, sys

with open('data/graph/dependency_graph_full_v2.nx.pkl', 'rb') as f:
    G = pickle.load(f)
with open('data/graph/dependency_graph_full_v2.index.json') as f:
    index = json.load(f)
with open('data/graph/dependency_graph_full_v2.lemma_index.json') as f:
    lemma_index = json.load(f)

# Check: what does a sample goal look like?
with open('data/raw/proof_step_pairs.jsonl') as f:
    pair = json.loads(f.readline().strip())

name = pair['name']
lemma = pair['lemma']
print(f"Name: '{name}'")
print(f"Lemma: '{lemma}'")

gid = index.get(name)
print(f"Goal ID from index: '{gid}'")
print(f"Goal ID in G: {gid in G}")

lid = lemma_index.get(lemma)
print(f"Lemma ID from lemma_index: '{lid}'")
print(f"Lemma ID in G: {lid in G}")

# Check some G nodes
nodes_sample = list(G.nodes())[:10]
print(f"\nFirst 10 G nodes: {nodes_sample}")

# Check what index values look like
keys = list(index.keys())[:5]
for k in keys:
    v = index[k]
    print(f"  index['{k}'] = '{v}', in G: {v in G}")
