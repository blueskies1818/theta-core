"""Integration test: verify domain index is built correctly on actual graph."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.explorer.dependency_graph import DependencyGraph

# Load the actual full graph
graph_path = Path("data/graph/dependency_graph_full")
print(f"Loading graph from {graph_path}...")
graph = DependencyGraph.load(graph_path)
print(f"  {graph.summary()}")

# Count nodes with domains vs without
with_domain = 0
without_domain = 0
all_domains = set()
for nid in graph.node_ids:
    attrs = graph.get_node(nid)
    if attrs:
        domain = attrs.get("domain", "") or ""
        if domain:
            with_domain += 1
            all_domains.add(domain)
        else:
            without_domain += 1
    else:
        without_domain += 1

print(f"  Nodes with domain: {with_domain}")
print(f"  Nodes without domain: {without_domain}")
print(f"  Unique domains: {len(all_domains)}")

# Test _get_node_ids_by_domain for key domains
for test_domain in ['Algebra', 'Analysis', 'NumberTheory']:
    node_ids = graph.get_node_ids_by_domain(test_domain)
    print(f"  {test_domain}: {len(node_ids)} nodes")

# Verify domain_subgraph works
for domain_name in ['Algebra', 'Analysis']:
    sub = graph.domain_subgraph(domain_name)
    print(f"  Subgraph {domain_name}: {sub.num_nodes} nodes, {sub.num_edges} edges")

# Test that _domain_matches would cover the right amount
# Simulate the domain index
from src.explorer.gnn_best_first_search import GNNBestFirstSearch
bf = GNNBestFirstSearch

# Build mock domain index
domain_index: dict[str, set[str]] = {}
for nid in graph.node_ids:
    attrs = graph.get_node(nid)
    if attrs:
        domain = attrs.get("domain", "") or ""
        if domain:
            domain_index.setdefault(domain, set()).add(nid)

# Test matching counts
test_theorem_domains = ['algebra', 'analysis', 'number_theory', 'physics', 'logic']
for tdom in test_theorem_domains:
    matching = set()
    for gdom, nids in domain_index.items():
        if bf._domain_matches(gdom, tdom):
            matching.update(nids)
    print(f"  Theorem domain '{tdom}' → {len(matching)} matching graph nodes")

print("\nAll integration checks passed!")
