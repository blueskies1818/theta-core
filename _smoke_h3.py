#!/usr/bin/env python3
"""Smoke test: verify graph loads and gate2 lemmas are resolvable."""
from src.explorer.dependency_graph import DependencyGraph
import json
from src.reward.base import _extract_lemmas_from_proof, build_training_lemma_set

g = DependencyGraph.load('data/graph/dependency_graph')
print(f'Nodes: {g.num_nodes}, Edges: {g.num_edges}')

with open('data/raw/gate2_training.jsonl') as f:
    lines = [json.loads(l) for l in f]
print(f'Gate2 training theorems: {len(lines)}')

lemmas = set()
for t in lines:
    lemmas.update(_extract_lemmas_from_proof(t.get('proof', '')))
print(f'Unique lemmas in gate2 proofs: {len(lemmas)}')
in_graph = sum(1 for l in lemmas if l in g.graph)
print(f'  In graph: {in_graph}')
not_in = [l for l in lemmas if l not in g.graph]
print(f'  Not in graph: {len(not_in)}')
if not_in:
    print(f'  Sample not found: {sorted(not_in)[:15]}')

# Build training lemma set
tls = build_training_lemma_set(lines, g)
print(f'\nTraining lemma set size: {len(tls)}')
print(f'Sample: {sorted(list(tls))[:10]}')
