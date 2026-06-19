#!/usr/bin/env python3
"""Check if gate3-required lemmas exist in the dependency graph - v2."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.explorer.dependency_graph import DependencyGraph

graph = DependencyGraph.load(Path("data/graph/dependency_graph"))

# Gate 3 required lemmas (short names as found in the graph)
needed_short = [
    "derivative_X_pow",
    "natDegree_C_mul_X_pow", 
    "degree_add_eq_left_of_degree_lt",
    "degree_sub_eq_left_of_degree_lt",
    "monic_X_sub_C",
    "monic_X_pow_add",
    "natDegree_X_pow",
    "natDegree_mul",
    "eval_add",
    "eval_mul",
    "eval_X",
    "derivative_add",
    "derivative_C_mul",
]

print("=== Checking node existence in graph ===")
for name in needed_short:
    node = graph.get_node(name)
    if node:
        domain = node.get("domain", "?")
        print(f"  FOUND: {name:50s} domain={domain:15s}")
    else:
        print(f"  MISSING: {name}")

print(f"\n=== Graph stats ===")
stats = graph.get_statistics()
print(f"  Nodes: {stats['num_nodes']}")
print(f"  Edges: {stats['num_edges']}")
print(f"  Domains: {json.dumps(stats['nodes_by_domain'], indent=2)}")

# Check if any Polynomial-related theorems are in the graph
print("\n=== Polynomial-related nodes in graph ===")
poly_nodes = []
for nid in graph.node_ids:
    node = graph.get_node(nid)
    if node:
        sf = node.get("source_file", "")
        if "Polynomial/" in sf:
            poly_nodes.append(nid)
print(f"  Nodes from Polynomial/ files: {len(poly_nodes)}")
if poly_nodes:
    print(f"  Examples: {poly_nodes[:10]}")

# Check: how many unique source file directories are there
print("\n=== Unique source file directories (top 30) ===")
dirs = {}
for nid in graph.node_ids[:1000]:  # sample
    node = graph.get_node(nid)
    if node:
        sf = node.get("source_file", "")
        if "Mathlib/" in sf:
            parts = sf.split("Mathlib/")[1].split("/")
            if len(parts) >= 2:
                d = f"{parts[0]}/{parts[1]}"
            else:
                d = parts[0]
            dirs[d] = dirs.get(d, 0) + 1
for d, c in sorted(dirs.items(), key=lambda x: x[1], reverse=True):
    print(f"  {d}: {c}")
