#!/usr/bin/env python3
"""Inspect edge types in cooccur graph."""
import pickle
from collections import Counter

with open('data/graph/dependency_graph_full_cooccur.nx.pkl', 'rb') as f:
    G = pickle.load(f)

edges = list(G.edges(data=True))
print(f'Total edges: {len(edges)}')

rel_counts = Counter(e[2].get('relation', 'none') for e in edges)
print('Edge relations:', rel_counts.most_common(15))

# Show first 5 PROVED_BY and CO_OCCURS edges
for src, dst, data in edges:
    rel = data.get('relation', '')
    if rel in ('PROVED_BY', 'CO_OCCURS_IN_PROOF'):
        print(f'  {src} -> {dst} ({rel}) - {dict(data)}')
