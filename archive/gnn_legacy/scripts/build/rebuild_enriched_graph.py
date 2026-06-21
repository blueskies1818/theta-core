"""
STEP 3+4: Inject missing lemmas into the dependency graph and rebuild.

Loads the alias map, loads the unresolved lemmas, cleans artifacts,
validates plausible lemma names, injects them as new nodes with domain-only
edges (era-safe), and rebuilds the graph as dependency_graph_full_v2.

Usage:
    python scripts/build/rebuild_enriched_graph.py \
        --graph data/graph/dependency_graph_full \
        --aliases data/lemma_aliases.json \
        --pairs data/raw/proof_step_pairs.jsonl \
        --output data/graph/dependency_graph_full_v2
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Lemma name validation
# ---------------------------------------------------------------------------

def is_plausible_lemma(name: str) -> bool:
    """Check if a cleaned name looks like a plausible Lean lemma name."""
    if len(name) < 3:
        return False
    if len(name) > 120:
        return False
    # Must start with a letter (or Greek letter)
    if not re.match(r'^[a-zA-Zα-ωΑ-Ωλμπφψθ]', name):
        return False
    # Should contain at least one lowercase letter (not just uppercase identifiers)
    if not re.search(r'[a-z]', name):
        return False
    # Reject names that are just single tokens without meaning
    # (like 'A', 'B', 'hf' etc)
    if re.match(r'^[A-Z][a-z]?$', name):
        return False
    # Reject names that are all punctuation-free single words like 'API', 'Adding'
    # These are likely false positives from proof text
    if not re.search(r'[_\.]', name) and len(name) < 8:
        pass  # Could still be real (like 'mem_Icc'), keep it
    return True


def clean_name(name: str) -> str:
    """Aggressively clean a lemma name."""
    # Strip leading artifacts
    name = re.sub(r'^[\(\)@\[\]\{\}\-]+', '', name)
    # Strip trailing artifacts (parens, brackets, dots, colons, digits, primes)
    name = re.sub(r'[\(\)@\[\]\{\}\,\:\;\d\.\']+$', '', name)
    return name


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Rebuild enriched dependency graph")
    parser.add_argument("--graph", type=Path, default=Path("data/graph/dependency_graph_full"))
    parser.add_argument("--aliases", type=Path, default=Path("data/lemma_aliases.json"))
    parser.add_argument("--pairs", type=Path, default=Path("data/raw/proof_step_pairs.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/graph/dependency_graph_full_v2"))
    parser.add_argument("--max-inject", type=int, default=50000)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent

    def resolve(p: Path) -> Path:
        return project_root / p if not p.is_absolute() else p

    graph_path = resolve(args.graph)
    aliases_path = resolve(args.aliases)
    pairs_path = resolve(args.pairs)
    output_path = resolve(args.output)

    print(f"=== Graph Enrichment ===", file=sys.stderr)
    print(f"Source graph: {graph_path}", file=sys.stderr)
    print(f"Aliases: {aliases_path}", file=sys.stderr)
    print(f"Pairs: {pairs_path}", file=sys.stderr)
    print(f"Output: {output_path}", file=sys.stderr)

    # Load existing graph using project module
    print("\nLoading graph...", file=sys.stderr)
    sys.path.insert(0, str(project_root))
    
    try:
        from src.explorer.dependency_graph import DependencyGraph
        dg = DependencyGraph.load(graph_path)
        G = dg._graph
    except Exception:
        # Fallback: load pickle directly with project in path
        with open(graph_path.with_name(graph_path.name + ".nx.pkl"), "rb") as f:
            G = pickle.load(f)
    with open(graph_path.with_name(graph_path.name + ".index.json")) as f:
        index = json.load(f)
    with open(graph_path.with_name(graph_path.name + ".stats.json")) as f:
        stats = json.load(f)

    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges", file=sys.stderr)

    # Load aliases
    print("Loading aliases...", file=sys.stderr)
    with open(aliases_path) as f:
        aliases = json.load(f)
    print(f"  {len(aliases)} aliases loaded", file=sys.stderr)

    # Load pairs to find unresolved and domain info
    print("Loading pairs...", file=sys.stderr)
    pairs = []
    with open(pairs_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    print(f"  {len(pairs)} pairs loaded", file=sys.stderr)

    # Find unresolved lemmas and their domains
    resolved_set = set(aliases.keys())
    lemma_domains: dict[str, str] = {}
    for p in pairs:
        lemma = p["lemma"]
        if lemma not in resolved_set:
            lemma_domains[lemma] = p.get("domain", "Unknown")

    # Clean and validate unresolved
    print("Processing unresolved lemmas...", file=sys.stderr)
    to_inject: dict[str, dict] = {}  # cleaned_name → {domain, original}
    rejected: Counter = Counter()

    for lemma, domain in lemma_domains.items():
        cleaned = clean_name(lemma)
        if not cleaned or cleaned in index:
            rejected["already_indexed_or_empty"] += 1
            continue
        if not is_plausible_lemma(cleaned):
            rejected["not_plausible"] += 1
            continue
        if cleaned in to_inject:
            # Merge — keep first domain
            rejected["duplicate"] += 1
            continue
        if len(to_inject) >= args.max_inject:
            rejected["max_inject"] += 1
            continue
        
        to_inject[cleaned] = {"domain": domain, "original": lemma}

    print(f"  To inject: {len(to_inject)}", file=sys.stderr)
    for reason, count in rejected.most_common():
        print(f"    rejected ({reason}): {count}", file=sys.stderr)

    # Find/Create domain anchor nodes in existing graph
    # The graph uses `node_type` (enum) and `domain` (string) attributes
    domain_nodes: dict[str, str] = {}
    
    # Collect all unique domains from existing nodes
    existing_domains = set()
    for nid, attrs in G.nodes(data=True):
        dom = attrs.get("domain", "")
        if dom:
            existing_domains.add(dom)
    
    # Create domain anchor nodes (era-safe: these are organizational, not from future math)
    for dom in existing_domains:
        anchor_id = f"__domain__{dom}"
        if anchor_id not in G:
            G.add_node(anchor_id, node_type="domain", name=f"Domain:{dom}", domain=dom, injected=False)
            index[anchor_id] = anchor_id
        domain_nodes[dom] = anchor_id
    
    # Also create domain nodes for new domains seen in pairs but not in graph
    all_pair_domains = set(p.get("domain", "Unknown") for p in pairs)
    for dom in all_pair_domains:
        if dom not in domain_nodes:
            anchor_id = f"__domain__{dom}"
            G.add_node(anchor_id, node_type="domain", name=f"Domain:{dom}", domain=dom, injected=False)
            index[anchor_id] = anchor_id
            domain_nodes[dom] = anchor_id

    # Inject new nodes
    print("\nInjecting new nodes...", file=sys.stderr)
    injected_count = 0
    injection_log: list[dict] = []

    for cleaned_name, info in to_inject.items():
        domain = info["domain"]
        
        # Generate a unique node ID
        node_id = cleaned_name
        
        # Skip if somehow already exists (shouldn't happen but be safe)
        if node_id in G:
            continue
        
        # Add node
        G.add_node(
            node_id,
            node_type="lemma",
            name=cleaned_name,
            domain=domain,
            injected=True,  # marker for era tracking
            original_lemma=info["original"],
        )
        index[node_id] = node_id
        injected_count += 1

        # Connect to domain node if it exists
        if domain in domain_nodes:
            G.add_edge(node_id, domain_nodes[domain], relation="in_domain")

        injection_log.append({
            "node_id": node_id,
            "domain": domain,
            "original": info["original"],
        })

    print(f"  Injected {injected_count} new nodes", file=sys.stderr)
    print(f"  New graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges", file=sys.stderr)

    # Save new graph
    print(f"\nSaving to {output_path}...", file=sys.stderr)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save networkx pickle
    nx_path = output_path.with_name(output_path.name + ".nx.pkl")
    with open(nx_path, "wb") as f:
        pickle.dump(G, f)

    # Save index
    index_path_out = output_path.with_name(output_path.name + ".index.json")
    with open(index_path_out, "w") as f:
        json.dump(index, f, indent=2)

    # Compute and save stats
    from src.explorer.dependency_graph import DependencyGraph

    # Use the existing stats computation
    new_stats = {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "density": 2 * G.number_of_edges() / (G.number_of_nodes() * (G.number_of_nodes() - 1)) if G.number_of_nodes() > 1 else 0,
        "injected_nodes": injected_count,
        "source_graph": str(graph_path),
    }
    
    # Count node types
    type_counts = Counter()
    domain_counts = Counter()
    for nid, attrs in G.nodes(data=True):
        nt = attrs.get("node_type", "unknown")
        # Handle enum types
        type_counts[str(nt)] += 1
        domain_counts[attrs.get("domain", "Unknown")] += 1
    
    new_stats["nodes_by_type"] = dict(type_counts.most_common())
    new_stats["nodes_by_domain"] = dict(domain_counts.most_common(80))

    stats_path = output_path.with_name(output_path.name + ".stats.json")
    with open(stats_path, "w") as f:
        json.dump(new_stats, f, indent=2)

    # Save injection log
    log_path = output_path.with_name(output_path.name + ".injection_log.json")
    with open(log_path, "w") as f:
        json.dump(injection_log, f, indent=2)

    # Build adjacency JSON for compatibility
    adj_path = output_path.with_name(output_path.name + ".adjacency.json")
    adj = {}
    for src, dst, edge_data in G.edges(data=True):
        if src not in adj:
            adj[src] = []
        adj[src].append({"target": dst, "relation": edge_data.get("relation", "depends_on")})
    for node in G.nodes():
        if node not in adj:
            adj[node] = []
    with open(adj_path, "w") as f:
        json.dump(adj, f, indent=2)

    print(f"  Saved: {nx_path}", file=sys.stderr)
    print(f"  Saved: {index_path_out}", file=sys.stderr)
    print(f"  Saved: {stats_path}", file=sys.stderr)
    print(f"  Saved: {log_path}", file=sys.stderr)
    print(f"  Saved: {adj_path}", file=sys.stderr)

    # Compute new coverage
    # Add aliases + injected to get new resolved count
    new_resolved = len(aliases) + injected_count
    total_unique = len(set(p["lemma"] for p in pairs))
    new_recall = new_resolved / total_unique if total_unique else 0
    
    # Pair-level
    injected_set = set(to_inject.keys())
    resolved_set_after = resolved_set | injected_set
    pair_resolved_after = sum(1 for p in pairs if p["lemma"] in resolved_set_after)
    pair_recall_after = pair_resolved_after / len(pairs) if pairs else 0

    print(f"\n=== New Coverage ===", file=sys.stderr)
    print(f"  Lemma recall (before): {len(aliases)/total_unique:.1%}", file=sys.stderr)
    print(f"  Lemma recall (after):  {new_recall:.1%}", file=sys.stderr)
    print(f"  Pair recall (before):  {sum(1 for p in pairs if p['lemma'] in resolved_set)/len(pairs):.1%}", file=sys.stderr)
    print(f"  Pair recall (after):   {pair_recall_after:.1%}", file=sys.stderr)

    # Save summary
    summary = {
        "before": {
            "nodes": stats["num_nodes"],
            "edges": stats["num_edges"],
            "lemma_recall": len(aliases) / total_unique,
            "pair_recall": sum(1 for p in pairs if p['lemma'] in resolved_set) / len(pairs),
        },
        "after": {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "injected": injected_count,
            "lemma_recall": new_recall,
            "pair_recall": pair_recall_after,
        },
    }
    summary_path = output_path.with_name(output_path.name + ".enrichment_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone.", file=sys.stderr)


if __name__ == "__main__":
    main()
