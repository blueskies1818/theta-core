#!/usr/bin/env python3
"""Build the math dependency graph from extracted Mathlib4 theorems.

This script reads the JSONL file produced in Phase 1.2 and constructs
a directed dependency graph where nodes are theorems/lemmas/definitions
and edges represent logical dependencies ("A uses B in its proof").

Output goes to data/graph/ for use by Phase 2.2 (GNN encoder) and
Phase 2.3 (MCTS proof search).

Usage:
    python scripts/build/build_dependency_graph.py                          # Full build
    python scripts/build/build_dependency_graph.py --max 10000              # Test build
    python scripts/build/build_dependency_graph.py --domain Analysis        # Domain subgraph
"""

import argparse
import sys
from pathlib import Path

# Ensure the project root is on sys.path
_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))

from src.explorer.graph_builder import build_and_save, build_dependency_graph
from src.explorer.dependency_graph import DependencyGraph


def main():
    parser = argparse.ArgumentParser(
        description="Build the math dependency graph from Mathlib4 theorems"
    )
    parser.add_argument(
        "--input",
        default="data/raw/mathlib4_theorems.jsonl",
        help="Path to extracted theorems JSONL",
    )
    parser.add_argument(
        "--output",
        default="data/graph/dependency_graph",
        help="Base path for output graph files",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        help="Maximum number of theorems to load (for testing)",
    )
    parser.add_argument(
        "--min-references",
        type=int,
        default=1,
        help="Minimum degree to keep a node (0=keep all isolated nodes)",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="If set, also extract and save a domain-specific subgraph",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Print statistics for an already-built graph and exit",
    )
    parser.add_argument(
        "--proof-step-pairs",
        type=str,
        default=None,
        help="Path to proof_step_pairs.jsonl for co-occurrence edge enrichment",
    )
    args = parser.parse_args()

    input_path = _project_root / args.input
    output_path = _project_root / args.output

    if args.stats_only:
        graph = DependencyGraph.load(output_path)
        stats = graph.get_statistics()
        print(graph.summary())
        for k, v in stats.items():
            print(f"  {k}: {v}")
        return

    if not input_path.exists():
        print(f"Error: theorems file not found: {input_path}")
        print("Run scripts/build/prepare_data.py first to extract theorems.")
        sys.exit(1)

    # ---- Full build ----
    # Resolve proof_step_pairs path if provided
    psp_path = None
    if args.proof_step_pairs:
        psp_path = _project_root / args.proof_step_pairs
        if not psp_path.exists():
            print(f"Warning: proof_step_pairs not found: {psp_path}")
            psp_path = None

    graph = build_and_save(
        theorems_path=input_path,
        output_path=output_path,
        max_theorems=args.max,
        min_references=args.min_references,
        proof_step_pairs_path=psp_path,
    )

    # Export PyG-compatible format for Phase 2.2
    pyg_data = graph.to_pyg_data()
    import torch

    torch.save(pyg_data, output_path.with_suffix(".pyg.pt"))
    print(f"PyG data saved to {output_path}.pyg.pt")

    # Export adjacency dict (human-readable)
    import json

    adj = graph.to_adjacency_dict()
    with open(output_path.with_suffix(".adjacency.json"), "w") as f:
        # Save summary only — full adjacency can be huge
        summary = {
            f"adjacency ({graph.num_nodes} nodes)": (
                f"{sum(len(v) for v in adj.values())} total edges"
            )
        }
        json.dump(summary, f, indent=2)

    # ---- Domain subgraph ----
    if args.domain:
        domain_graph = graph.domain_subgraph(args.domain)
        domain_output = output_path.parent / f"dependency_graph_{args.domain}"
        domain_graph.save(domain_output)
        print(f"\nDomain subgraph '{args.domain}': {domain_graph.summary()}")

    # ---- Diagnostic: print top nodes by degree ----
    print("\nTop 20 nodes by in-degree (most depended on):")
    in_deg = sorted(
        graph.graph.in_degree(), key=lambda x: x[1], reverse=True
    )
    for node_id, deg in in_deg[:20]:
        node = graph.get_node(node_id)
        domain = node.get("domain", "?") if node else "?"
        print(f"  {deg:5d}  {node_id}  [{domain}]")

    print("\nTop 20 nodes by out-degree (most dependencies):")
    out_deg = sorted(
        graph.graph.out_degree(), key=lambda x: x[1], reverse=True
    )
    for node_id, deg in out_deg[:20]:
        node = graph.get_node(node_id)
        domain = node.get("domain", "?") if node else "?"
        print(f"  {deg:5d}  {node_id}  [{domain}]")

    # Print generation distribution
    gens = graph.topological_generations()
    print(f"\nTopological generations: {len(gens)} total")
    for i, gen in enumerate(gens[:10]):
        print(f"  Gen {i}: {len(gen)} nodes")
    if len(gens) > 10:
        print(f"  ... Gen {len(gens)-1}: {len(gens[-1])} nodes")


if __name__ == "__main__":
    main()
