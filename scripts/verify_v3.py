#!/usr/bin/env python3
"""Verify v3 graph loads and edge types work correctly."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.explorer.dependency_graph import DependencyGraph, EdgeType
from src.explorer.gnn_encoder import prepare_graph_tensors

# Load v3 graph via DependencyGraph wrapper
dg = DependencyGraph.load("data/graph/dependency_graph_full_v3")
print(f"Loaded: {dg.num_nodes} nodes, {dg.num_edges} edges")

# Check edge types
from collections import Counter
et_counts = Counter()
proof_attrs = 0
for u, v, attrs in dg._graph.edges(data=True):
    et = attrs.get("type", None)
    key = et.value if hasattr(et, 'value') else str(et)
    et_counts[key] += 1
    if attrs.get("proved_by") or attrs.get("cooccurs_in_proof"):
        proof_attrs += 1

print(f"\nEdge types (primary):")
for k, v in et_counts.most_common():
    print(f"  {k}: {v}")
print(f"\nEdges with proof attributes: {proof_attrs}")

# Test prepare_graph_tensors
sources, targets, edge_types, num_nodes = prepare_graph_tensors(dg)
print(f"\nTensor shapes:")
print(f"  sources:    {sources.shape}")
print(f"  targets:    {targets.shape}")
print(f"  edge_types: {edge_types.shape}")
print(f"  num_nodes:  {num_nodes}")

# Count virtual edge types
from collections import Counter as TCounter
virtual_et = TCounter()
for i in range(len(edge_types)):
    virtual_et[int(edge_types[i].item())] += 1
print(f"\nVirtual edge type distribution:")
et_names = {0: "USES_IN_PROOF", 1: "USES_IN_STATEMENT", 2: "GENERALIZES", 
            3: "INSTANTIATES", 4: "CO_OCCURS_IN_PROOF", 5: "PROVED_BY"}
for code, count in sorted(virtual_et.items()):
    print(f"  {code} ({et_names.get(code, '?')}): {count}")
