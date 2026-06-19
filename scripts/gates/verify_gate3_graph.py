#!/usr/bin/env python3
"""Verify that gate3-required lemmas exist in the rebuilt dependency graph.

Usage:
    python scripts/gates/verify_gate3_graph.py [--graph-path data/graph/dependency_graph_full]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.explorer.dependency_graph import DependencyGraph


# Gate 3 required lemmas — the lemmas that gate3 theorem proofs reference.
# Both fully-qualified names (for exact lookup) and short names (for alias lookup).
GATE3_REQUIRED_LEMMAS = [
    # Polynomial basic evaluation
    ("Polynomial.eval_add", "eval_add"),
    ("Polynomial.eval_mul", "eval_mul"),
    ("Polynomial.eval_X", "eval_X"),
    # Polynomial derivatives
    ("Polynomial.derivative_X_pow", "derivative_X_pow"),
    ("Polynomial.derivative_add", "derivative_add"),
    ("Polynomial.derivative_C_mul", "derivative_C_mul"),
    # Polynomial degrees
    ("Polynomial.natDegree_X_pow", "natDegree_X_pow"),
    ("Polynomial.natDegree_C_mul_X_pow", "natDegree_C_mul_X_pow"),
    ("Polynomial.natDegree_mul", "natDegree_mul"),
    ("Polynomial.degree_add_eq_left_of_degree_lt", "degree_add_eq_left_of_degree_lt"),
    ("Polynomial.degree_sub_eq_left_of_degree_lt", "degree_sub_eq_left_of_degree_lt"),
    # Polynomial monic
    ("Polynomial.monic_X_sub_C", "monic_X_sub_C"),
    ("Polynomial.monic_X_pow_add", "monic_X_pow_add"),
    # Additional polynomial lemmas referenced in gate3
    ("Polynomial.map_id", "map_id"),
    ("Polynomial.X_ne_zero", "X_ne_zero"),
]

GATE3_REQUIRED_THEOREMS = [
    "poly_eval_C_add_X",
    "poly_mul_X_add_C",
    "poly_eval_mul_X_sub_C",
    "poly_derivative_X_pow",
    "poly_eval_derivative_C_mul",
    "poly_derivative_add",
    "poly_degree_X_pow",
    "poly_degree_C_mul_X_pow",
    "poly_degree_mul_X",
    "poly_degree_add_eq_left_of_degree_lt",
    "poly_degree_sub_eq_left_of_degree_lt",
    "poly_monic_X_sub_C",
    "poly_monic_X_pow_add",
    "poly_map_id",
]


def main():
    parser = argparse.ArgumentParser(description="Verify gate3 graph requirements")
    parser.add_argument(
        "--graph-path",
        default="data/graph/dependency_graph_full",
        help="Path to the rebuilt dependency graph (base, without extension)",
    )
    args = parser.parse_args()

    graph_path = Path(args.graph_path)
    nx_path = graph_path.with_suffix(".nx.pkl")

    if not nx_path.exists():
        print(f"ERROR: Graph not found at {nx_path}")
        sys.exit(1)

    print(f"Loading graph from {nx_path}...")
    graph = DependencyGraph.load(graph_path)

    stats = graph.get_statistics()
    print(f"\nGraph: {stats['num_nodes']} nodes, {stats['num_edges']} edges")
    print(f"Domains: {len(stats['nodes_by_domain'])}")
    for domain, count in sorted(stats["nodes_by_domain"].items(), key=lambda x: x[1], reverse=True)[:20]:
        print(f"  {domain}: {count}")

    # Check lemmas
    print("\n=== Gate 3 Required Lemmas ===")
    missing = []
    found = []
    for full_name, short_name in GATE3_REQUIRED_LEMMAS:
        node = graph.get_node(full_name)
        if not node:
            node = graph.get_node(short_name)
        if node:
            domain = node.get("domain", "?")
            found.append((full_name, domain))
        else:
            # Try via alias resolution
            resolved = graph.resolve_name(full_name)
            if not resolved:
                resolved = graph.resolve_name(short_name)
            if resolved:
                node = graph.get_node(resolved)
                domain = node.get("domain", "?") if node else "?"
                found.append((full_name, f"{domain} (via alias)"))
            else:
                missing.append(full_name)

    for name, domain in found:
        print(f"  OK  {name:55s} [{domain}]")
    for name in missing:
        print(f"  MISS  {name}")

    # Check custom gate3 test theorems (these are NOT Mathlib4 lemmas —
    # they're synthetic theorems from gate3_lemma_novelty.jsonl that REFERENCE
    # Mathlib4 lemmas. They won't be in the dependency graph.)
    print("\n=== Gate 3 Test Theorems (synthetic, NOT expected in graph) ===")
    thm_missing = []
    for name in GATE3_REQUIRED_THEOREMS:
        node = graph.get_node(name)
        if node:
            domain = node.get("domain", "?")
            print(f"  FOUND {name:55s} [{domain}]")
        else:
            print(f"  absent {name:55s} (synthetic test theorem — expected)")
            thm_missing.append(name)

    # Check domain coverage for Polynomial, RingTheory, FieldTheory
    print("\n=== Domain Coverage Check ===")
    required_domains = [
        ("Algebra/Polynomial", True),    # Main polynomial theorems
        ("RingTheory", True),             # Ring theory
        ("RingTheory/Polynomial", True),  # Ring-theoretic polynomial lemmas
        ("FieldTheory", True),            # Field theory
        ("Algebra/MvPolynomial", True),   # Multivariate polynomials
        # These don't exist as Mathlib4 subdirs (polynomials live under Algebra/)
        ("FieldTheory/Polynomial", False), # Doesn't exist — polynomials in Algebra/
        ("Data/Polynomial", False),        # Doesn't exist — polynomials in Algebra/
    ]
    for d, _expected in required_domains:
        count = len(graph.get_node_ids_by_domain(d))
        status = "OK" if count > 0 else ("ABSENT" if not _expected else "MISSING")
        expected_mark = "(expected absent)" if not _expected else ""
        print(f"  {status:8s} {d:40s} {count:6d} nodes {expected_mark}")

    # Summary (only count lemma misses, not synthetic test theorems)
    print("\n=== Summary ===")
    if len(missing) == 0:
        print(f"ALL {len(found)} GATE3 LEMMAS FOUND!")
    else:
        print(f"WARNING: {len(missing)} lemmas missing out of {len(found) + len(missing)}")
    print(f"(Synthetic test theorems checked: {len(GATE3_REQUIRED_THEOREMS)} — not expected in graph)")
    return len(missing) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
