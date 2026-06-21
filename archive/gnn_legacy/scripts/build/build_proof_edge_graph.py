#!/usr/bin/env python3
"""
STEP 1: Build dependency_graph_full_v3 — add PROVED_BY and CO_OCCURS_IN_PROOF edges.

The lemma_index stores INTEGER indices → need idx_to_id mapping to string node IDs.
Since networkx DiGraph allows only one edge per (u,v) pair, we:
  - Store `proved_by=True` and `cooccurs_in_proof=True` as edge attributes
  - If no existing edge, create it with type=PROVED_BY (primary type)
  - If existing edge, add both attributes to it

The `prepare_graph_tensors()` function will be updated to emit extra rows
for edges with these attributes, so the GNN sees all edge types.

Input:
    data/graph/dependency_graph_full_v2
    data/raw/proof_step_pairs.jsonl

Output:
    data/graph/dependency_graph_full_v3  (~650K unique edges, ~215K with proof attrs)
"""
from __future__ import annotations

import json
import pickle
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.explorer.dependency_graph import EdgeType


def main():
    graph_v2 = Path("data/graph/dependency_graph_full_v2")
    lemma_index_path = Path("data/graph/dependency_graph_full_v2.lemma_index.json")
    pairs_path = Path("data/raw/proof_step_pairs.jsonl")
    output_path = Path("data/graph/dependency_graph_full_v3")

    print("=== Building v3 Graph with Proof Edges ===", file=sys.stderr)

    # Load v2 graph
    print("Loading v2 graph...", file=sys.stderr)
    with open(graph_v2.with_name(graph_v2.name + ".nx.pkl"), "rb") as f:
        G = pickle.load(f)
    with open(graph_v2.with_name(graph_v2.name + ".index.json")) as f:
        node_index = json.load(f)
    print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges", file=sys.stderr)

    # Build idx_to_id mapping
    sorted_node_ids = sorted(G.nodes())
    idx_to_id: dict[int, str] = {i: nid for i, nid in enumerate(sorted_node_ids)}
    print(f"  idx_to_id: {len(idx_to_id)} entries", file=sys.stderr)

    # Load lemma index (name → int index)
    print("Loading lemma index...", file=sys.stderr)
    with open(lemma_index_path) as f:
        lemma_index = json.load(f)
    print(f"  {len(lemma_index)} entries", file=sys.stderr)

    # Load pairs
    print("Loading proof-step pairs...", file=sys.stderr)
    pairs = []
    with open(pairs_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    print(f"  {len(pairs)} pairs", file=sys.stderr)

    # ── Add edges ──────────────────────────────────────────────────────
    print("\nAdding proof edges...", file=sys.stderr)

    new_edges = 0       # edges that didn't exist before
    augmented = 0       # existing edges that got proof attributes added
    goal_missing = 0
    lemma_missing = 0
    domain_edges: Counter = Counter()

    for i, p in enumerate(pairs):
        name = p["name"]
        lemma = p["lemma"]
        domain = p.get("domain", "Unknown")

        # Resolve goal via node_index (name → string node_id)
        goal_id = node_index.get(name)
        if goal_id is None or goal_id not in G:
            goal_missing += 1
            continue

        # Resolve lemma via lemma_index (name → int) → idx_to_id (int → string)
        lemma_idx = lemma_index.get(lemma)
        if lemma_idx is None:
            lemma_missing += 1
            continue
        lemma_id = idx_to_id.get(lemma_idx)
        if lemma_id is None or lemma_id not in G:
            lemma_missing += 1
            continue

        domain_edges[domain] += 1

        if G.has_edge(goal_id, lemma_id):
            # Edge already exists (import, generalize, etc.) — add proof attributes
            edge_data = G.edges[goal_id, lemma_id]
            edge_data["proved_by"] = True
            edge_data["cooccurs_in_proof"] = True
            augmented += 1
        else:
            # New edge with PROVED_BY as primary type
            G.add_edge(
                goal_id, lemma_id,
                type=EdgeType.PROVED_BY,
                proved_by=True,
                cooccurs_in_proof=True,
            )
            new_edges += 1

        if (i + 1) % 50000 == 0:
            print(f"  {i+1}/{len(pairs)}: {new_edges} new + {augmented} augmented", file=sys.stderr)

    print(f"\n=== Results ===", file=sys.stderr)
    print(f"  New PROVED_BY edges: {new_edges}", file=sys.stderr)
    print(f"  Existing edges augmented: {augmented}", file=sys.stderr)
    print(f"  Total proof edges: {new_edges + augmented}", file=sys.stderr)
    print(f"  Goal missing: {goal_missing}", file=sys.stderr)
    print(f"  Lemma missing: {lemma_missing}", file=sys.stderr)
    print(f"  Final graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges", file=sys.stderr)

    # Count edges with proof attributes
    proved_by_total = sum(1 for _, _, d in G.edges(data=True) if d.get("proved_by"))
    cooccur_total = sum(1 for _, _, d in G.edges(data=True) if d.get("cooccurs_in_proof"))
    print(f"  Edges with proved_by=True: {proved_by_total}", file=sys.stderr)
    print(f"  Edges with cooccurs_in_proof=True: {cooccur_total}", file=sys.stderr)

    # ── Compute stats ──────────────────────────────────────────────────
    print("\nComputing stats...", file=sys.stderr)

    in_deg = [d for _, d in G.in_degree()]
    out_deg = [d for _, d in G.out_degree()]

    edge_type_counts: Counter = Counter()
    node_type_counts: Counter = Counter()
    domain_counts: Counter = Counter()

    for nid, attrs in G.nodes(data=True):
        nt = attrs.get("node_type", "unknown")
        node_type_counts[nt.value if hasattr(nt, 'value') else str(nt)] += 1
        domain_counts[attrs.get("domain", "Unknown")] += 1

    for u, v, attrs in G.edges(data=True):
        et = attrs.get("type", None)
        key = et.value if hasattr(et, 'value') else str(et)
        edge_type_counts[key] += 1

    stats = {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "density": 2 * G.number_of_edges() / (G.number_of_nodes() * (G.number_of_nodes() - 1))
        if G.number_of_nodes() > 1 else 0,
        "avg_in_degree": sum(in_deg) / len(in_deg) if in_deg else 0,
        "avg_out_degree": sum(out_deg) / len(out_deg) if out_deg else 0,
        "max_in_degree": max(in_deg) if in_deg else 0,
        "max_out_degree": max(out_deg) if out_deg else 0,
        "new_proved_by_edges": new_edges,
        "augmented_edges": augmented,
        "total_proof_edges": new_edges + augmented,
        "total_virtual_edges": G.number_of_edges() + new_edges + augmented,
        "source_graph": str(graph_v2),
        "edge_types": dict(edge_type_counts.most_common()),
        "nodes_by_type": dict(node_type_counts.most_common()),
        "nodes_by_domain": dict(domain_counts.most_common(80)),
        "proof_edges_by_domain": dict(domain_edges.most_common(30)),
    }

    # ── Save ───────────────────────────────────────────────────────────
    print(f"\nSaving to {output_path}...", file=sys.stderr)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path.with_name(output_path.name + ".nx.pkl"), "wb") as f:
        pickle.dump(G, f)

    # Rebuild node_index from graph
    rebuilt_ni: dict[str, str] = {}
    for nid, attrs in G.nodes(data=True):
        name = attrs.get("name", nid)
        if name not in rebuilt_ni:
            rebuilt_ni[name] = nid
    with open(output_path.with_name(output_path.name + ".index.json"), "w") as f:
        json.dump(rebuilt_ni, f, indent=2)

    # Rebuild lemma_index for v3 (name → int index aligned with v3 sorted nodes)
    v3_sorted = sorted(G.nodes())
    v3_id_to_idx = {nid: i for i, nid in enumerate(v3_sorted)}
    updated_li: dict[str, int] = {}
    for name, idx in lemma_index.items():
        if idx < len(sorted_node_ids):
            old_nid = idx_to_id.get(idx)
            if old_nid and old_nid in v3_id_to_idx:
                updated_li[name] = v3_id_to_idx[old_nid]
    with open(output_path.with_name(output_path.name + ".lemma_index.json"), "w") as f:
        json.dump(updated_li, f, indent=2)

    with open(output_path.with_name(output_path.name + ".stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    summary = {
        "source": str(graph_v2),
        "pairs_processed": len(pairs),
        "goal_resolved": len(pairs) - goal_missing,
        "lemma_resolved": len(pairs) - lemma_missing,
        "new_edges": new_edges,
        "augmented_edges": augmented,
        "total_proof_edges": new_edges + augmented,
        "total_edges": G.number_of_edges(),
    }
    with open(output_path.with_name(output_path.name + ".build_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone.", file=sys.stderr)


if __name__ == "__main__":
    main()
