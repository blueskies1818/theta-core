#!/usr/bin/env python3
"""Check if gate3-required lemmas exist in the dependency graph."""
import json
from pathlib import Path

graph_index = Path("data/graph/dependency_graph.index.json")
index = json.loads(graph_index.read_text())

# Gate 3 required lemmas (from the task spec)
needed = [
    "Polynomial.derivative_X_pow",
    "Polynomial.natDegree_C_mul_X_pow",
    "Polynomial.degree_add_eq_left_of_degree_lt",
    "Polynomial.degree_sub_eq_left_of_degree_lt",
    "Polynomial.monic_X_sub_C",
    "Polynomial.monic_X_pow_add",
    "Polynomial.natDegree_X_pow",
    "Polynomial.natDegree_mul",
    "Polynomial.eval_add",
    "Polynomial.eval_mul",
    "Polynomial.eval_X",
    "Polynomial.derivative_add",
    "Polynomial.derivative_C_mul",
]

print("=== Direct name lookup ===")
for name in needed:
    if name in index:
        print(f"  FOUND: {name}")
    else:
        print(f"  MISSING: {name}")

# Check for partial matches
print("\n=== Partial pattern search ===")
patterns = ["derivative_X_pow", "natDegree_C_mul_X", "degree_add_eq_left",
            "degree_sub_eq_left", "monic_X_sub", "monic_X_pow_add",
            "natDegree_X_pow", "natDegree_mul", "derivative_add", "derivative_C_mul"]
found_any = set()
for k in index:
    for p in patterns:
        if p in k:
            found_any.add(k)
for k in sorted(found_any):
    print(f"  {k}")

# Domain stats
print("\n=== Domain distribution (from index resolution) ===")
# Load the graph to check domains
import pickle
with open("data/graph/dependency_graph.nx.pkl", "rb") as f:
    nx_graph = pickle.load(f)

domains = {}
for n, attrs in nx_graph.nodes(data=True):
    d = attrs.get("domain", "Unknown")
    domains[d] = domains.get(d, 0) + 1

for d, c in sorted(domains.items(), key=lambda x: x[1], reverse=True):
    print(f"  {d}: {c}")

# Check if data/raw/mathlib4_theorems.jsonl has Polynomial content
print("\n=== Checking source theorems for Polynomial content ===")
polynomial_files = set()
theorems_path = Path("data/raw/mathlib4_theorems.jsonl")
count = 0
with open(theorems_path) as f:
    for line in f:
        t = json.loads(line)
        sf = t.get("source_file", "")
        if "Polynomial" in sf:
            polynomial_files.add(sf)
        count += 1

print(f"Total theorems: {count}")
print(f"Files mentioning Polynomial: {len(polynomial_files)}")
for fpath in sorted(polynomial_files)[:20]:
    domain = fpath.split("Mathlib/", 1)[1].split("/")[0] if "Mathlib/" in fpath else "?"
    print(f"  [{domain}] {fpath}")
