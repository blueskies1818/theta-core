"""Math dependency graph data structures for Phase 2.

Nodes: theorems, lemmas, definitions, axioms
Edges: directed logical dependencies (A depends on B if B is used in A's proof)

The graph supports:
- Efficient neighbor lookup for GNN message passing
- Subgraph extraction for local proof search
- Serialization to/from disk
- PyTorch Geometric compatible export
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import networkx as nx


class NodeType(Enum):
    """Type of mathematical entity a node represents."""

    THEOREM = "theorem"
    LEMMA = "lemma"
    DEFINITION = "definition"
    EXAMPLE = "example"
    AXIOM = "axiom"
    STRUCTURE = "structure"
    CLASS = "class"
    INDUCTIVE = "inductive"
    EXTERNAL = "external"  # Referenced but not in our dataset


class EdgeType(Enum):
    """Type of dependency relationship between nodes."""

    USES_IN_PROOF = "uses_in_proof"  # B appears in A's proof
    USES_IN_STATEMENT = "uses_in_statement"  # B appears in A's statement
    GENERALIZES = "generalizes"  # A is a generalization of B
    INSTANTIATES = "instantiates"  # A instantiates B with specific parameters
    CO_OCCURS_IN_PROOF = "co_occurs_in_proof"  # B co-occurs with A in proof-step pairs
    PROVED_BY = "proved_by"  # A is proved using lemma B (from proof-step extraction)


@dataclass
class DependencyNode:
    """A node in the math dependency graph.

    Attributes:
        id: Unique node identifier (theorem name).
        name: Human-readable name.
        node_type: What kind of mathematical entity.
        statement: The theorem/lemma statement (Lean 4 syntax).
        proof: The proof body (Lean 4 tactics or term).
        source_file: Path to the .lean source file.
        domain: Mathematical domain (e.g., "Analysis", "Algebra").
    """

    id: str
    name: str
    node_type: NodeType = NodeType.LEMMA
    statement: str = ""
    proof: str = ""
    source_file: str = ""
    domain: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "node_type": self.node_type.value,
            "statement": self.statement,
            "proof": self.proof[:500],
            "source_file": self.source_file,
            "domain": self.domain,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DependencyNode":
        return cls(
            id=d["id"],
            name=d["name"],
            node_type=NodeType(d["node_type"]),
            statement=d.get("statement", ""),
            proof=d.get("proof", ""),
            source_file=d.get("source_file", ""),
            domain=d.get("domain", ""),
        )


class DependencyGraph:
    """Directed graph of mathematical dependencies.

    Nodes are theorems/definitions. Edges point from dependents
    to dependencies: **A → B means A depends on B**.

    Wraps NetworkX for graph algorithms and provides PyG-compatible
    adjacency for GNN training in Phase 2.2.
    """

    def __init__(self):
        self._graph: nx.DiGraph = nx.DiGraph()
        self._node_index: dict[str, str] = {}  # name → node_id
        self._id_to_idx: dict[str, int] = {}  # node_id → integer index
        self._idx_to_id: dict[int, str] = {}  # integer index → node_id

    # -- Properties ----------------------------------------------------------

    @property
    def graph(self) -> nx.DiGraph:
        """Underlying NetworkX directed graph."""
        return self._graph

    @property
    def num_nodes(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def num_edges(self) -> int:
        return self._graph.number_of_edges()

    @property
    def node_ids(self) -> list[str]:
        return list(self._graph.nodes())

    # -- Node operations -----------------------------------------------------

    def add_node(self, node: DependencyNode) -> None:
        """Add a node to the graph. Silently no-ops if node already exists."""
        self._graph.add_node(
            node.id,
            name=node.name,
            node_type=node.node_type,
            statement=node.statement,
            proof=node.proof,
            source_file=node.source_file,
            domain=node.domain,
        )
        # Track the name → id mapping. If there's a conflict, prefer the
        # first registered (typically the one with the shortest path).
        if node.name not in self._node_index:
            self._node_index[node.name] = node.id

    def has_node(self, node_id: str) -> bool:
        return node_id in self._graph

    def get_node(self, node_id: str) -> dict | None:
        """Get node attributes dict, or None."""
        if node_id in self._graph:
            return dict(self._graph.nodes[node_id])
        return None

    def get_node_ids_by_domain(self, domain: str) -> list[str]:
        """Return all node IDs belonging to a mathematical domain."""
        return [
            n
            for n, attrs in self._graph.nodes(data=True)
            if attrs.get("domain", "") == domain
        ]

    def get_node_ids_by_type(self, node_type: NodeType) -> list[str]:
        """Return all node IDs of a given type."""
        return [
            n
            for n, attrs in self._graph.nodes(data=True)
            if attrs.get("node_type") == node_type
        ]

    # -- Edge operations -----------------------------------------------------

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType = EdgeType.USES_IN_PROOF,
    ) -> bool:
        """Add a directed dependency edge: source depends on target.

        Returns True if the edge was added, False if either endpoint is
        missing.
        """
        if source_id in self._graph and target_id in self._graph:
            self._graph.add_edge(source_id, target_id, type=edge_type)
            return True
        return False

    def bulk_add_edges(
        self,
        edges: list[tuple[str, str, EdgeType]],
    ) -> int:
        """Add many edges at once. Returns the count actually added."""
        added = 0
        for src, tgt, etype in edges:
            if self.add_edge(src, tgt, etype):
                added += 1
        return added

    # -- Graph traversal -----------------------------------------------------

    def get_dependencies(self, node_id: str) -> list[str]:
        """Nodes that node_id depends on (outgoing edges)."""
        return list(self._graph.successors(node_id)) if node_id in self._graph else []

    def get_dependents(self, node_id: str) -> list[str]:
        """Nodes that depend on node_id (incoming edges)."""
        return (
            list(self._graph.predecessors(node_id)) if node_id in self._graph else []
        )

    def get_neighborhood(
        self,
        node_id: str,
        radius: int = 2,
        direction: str = "both",
    ) -> set[str]:
        """Get all nodes within `radius` hops of node_id.

        Args:
            node_id: Center node.
            radius: Number of hops.
            direction: "out" (dependencies only), "in" (dependents only),
                or "both" (default).
        """
        if node_id not in self._graph:
            return set()

        nodes: set[str] = {node_id}
        frontier: set[str] = {node_id}

        for _ in range(radius):
            new_frontier: set[str] = set()
            for n in frontier:
                if direction in ("out", "both"):
                    new_frontier.update(self._graph.successors(n))
                if direction in ("in", "both"):
                    new_frontier.update(self._graph.predecessors(n))
            frontier = new_frontier - nodes
            nodes.update(frontier)

        return nodes

    def shortest_path(self, source: str, target: str) -> list[str]:
        """Shortest dependency path from source to target."""
        try:
            return nx.shortest_path(self._graph, source, target)
        except nx.NetworkXNoPath:
            return []

    def topological_generations(self) -> list[list[str]]:
        """Group nodes by topological generation.

        Generation 0: nodes with no dependencies (foundational).
        Generation N: nodes whose dependencies are all in earlier generations.
        """
        generations: list[list[str]] = []
        remaining: set[str] = set(self._graph.nodes())
        seen: set[str] = set()

        while remaining:
            current: set[str] = set()
            for node in remaining:
                deps = set(self._graph.successors(node))
                if deps.issubset(seen):
                    current.add(node)

            if not current:
                # Cycle or disconnected — dump the rest
                generations.append(sorted(remaining))
                break

            generations.append(sorted(current))
            seen.update(current)
            remaining -= current

        return generations

    def subgraph(self, node_ids: list[str]) -> "DependencyGraph":
        """Extract induced subgraph containing given nodes."""
        sub = DependencyGraph()
        sub._graph = self._graph.subgraph(node_ids).copy()
        for nid in sub._graph.nodes():
            attrs = sub._graph.nodes[nid]
            name = attrs.get("name", nid)
            sub._node_index[name] = nid
        sub._rebuild_indices()
        return sub

    def domain_subgraph(self, domain: str) -> "DependencyGraph":
        """Extract subgraph for a single mathematical domain."""
        domain_nodes = self.get_node_ids_by_domain(domain)
        return self.subgraph(domain_nodes)

    # -- Indexing ------------------------------------------------------------

    def _rebuild_indices(self) -> None:
        """Rebuild integer-index mappings for PyG export."""
        self._id_to_idx = {nid: i for i, nid in enumerate(sorted(self._graph.nodes()))}
        self._idx_to_id = {i: nid for nid, i in self._id_to_idx.items()}

    def node_id_to_idx(self, node_id: str) -> int | None:
        """Map a string node ID to its integer index."""
        if not self._id_to_idx:
            self._rebuild_indices()
        return self._id_to_idx.get(node_id)

    def idx_to_node_id(self, idx: int) -> str | None:
        """Map an integer index back to a string node ID."""
        if not self._idx_to_id:
            self._rebuild_indices()
        return self._idx_to_id.get(idx)

    def resolve_name(self, name: str) -> str | None:
        """Resolve a theorem name to its canonical node ID."""
        return self._node_index.get(name)

    # -- Export --------------------------------------------------------------

    def to_pyg_data(self) -> dict:
        """Export graph in PyTorch Geometric compatible format.

        Returns a dict with:
            edge_index: [2, num_edges] LongTensor
            num_nodes: int
            node_ids: list[str]
        """
        if not self._id_to_idx:
            self._rebuild_indices()

        sources: list[int] = []
        targets: list[int] = []

        for u, v in self._graph.edges():
            if u in self._id_to_idx and v in self._id_to_idx:
                sources.append(self._id_to_idx[u])
                targets.append(self._id_to_idx[v])

        import torch

        return {
            "edge_index": torch.tensor([sources, targets], dtype=torch.long),
            "num_nodes": self.num_nodes,
            "node_ids": [self._idx_to_id[i] for i in range(self.num_nodes)],
        }

    def to_adjacency_dict(self) -> dict[str, list[str]]:
        """Export as adjacency list dict: {node_id: [dependency_ids]}."""
        return {n: list(self._graph.successors(n)) for n in self._graph.nodes()}

    # -- Statistics ----------------------------------------------------------

    def get_statistics(self) -> dict:
        """Compute and return graph-level statistics."""
        if self.num_nodes == 0:
            return {"num_nodes": 0, "num_edges": 0}

        in_deg = [d for _, d in self._graph.in_degree()]
        out_deg = [d for _, d in self._graph.out_degree()]
        generations = self.topological_generations()

        by_type: dict[str, int] = {}
        by_domain: dict[str, int] = {}
        for __, attrs in self._graph.nodes(data=True):
            nt = attrs.get("node_type")
            if nt:
                by_type[nt.value if isinstance(nt, NodeType) else str(nt)] = (
                    by_type.get(
                        nt.value if isinstance(nt, NodeType) else str(nt), 0
                    )
                    + 1
                )
            dom = attrs.get("domain", "Unknown")
            by_domain[dom] = by_domain.get(dom, 0) + 1

        return {
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "density": nx.density(self._graph),
            "avg_in_degree": sum(in_deg) / len(in_deg) if in_deg else 0,
            "avg_out_degree": sum(out_deg) / len(out_deg) if out_deg else 0,
            "max_in_degree": max(in_deg) if in_deg else 0,
            "max_out_degree": max(out_deg) if out_deg else 0,
            "num_weakly_connected": nx.number_weakly_connected_components(self._graph),
            "num_strongly_connected": nx.number_strongly_connected_components(
                self._graph
            ),
            "max_generation": len(generations),
            "nodes_in_largest_wcc": max(
                (len(cc) for cc in nx.weakly_connected_components(self._graph)),
                default=0,
            ),
            "nodes_by_type": by_type,
            "nodes_by_domain": dict(
                sorted(by_domain.items(), key=lambda x: x[1], reverse=True)
            ),
        }

    # -- Persistence ---------------------------------------------------------

    def save(self, path: Path | str) -> None:
        """Save graph to disk (pickle + JSON index)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path.with_suffix(".nx.pkl"), "wb") as f:
            pickle.dump(self._graph, f)

        with open(path.with_suffix(".index.json"), "w") as f:
            json.dump(self._node_index, f, indent=2)

        stats = self.get_statistics()
        with open(path.with_suffix(".stats.json"), "w") as f:
            json.dump(stats, f, indent=2)

        print(f"Graph saved to {path}.nx.pkl ({self.num_nodes} nodes, {self.num_edges} edges)")

    @classmethod
    def load(cls, path: Path | str) -> "DependencyGraph":
        """Load graph from disk."""
        path = Path(path)

        g = cls()
        with open(path.with_suffix(".nx.pkl"), "rb") as f:
            g._graph = pickle.load(f)

        index_path = path.with_suffix(".index.json")
        if index_path.exists():
            with open(index_path) as f:
                g._node_index = json.load(f)

        g._rebuild_indices()
        return g

    # -- Display -------------------------------------------------------------

    def summary(self) -> str:
        """One-line summary string."""
        stats = self.get_statistics()
        return (
            f"DependencyGraph({stats['num_nodes']} nodes, "
            f"{stats['num_edges']} edges, "
            f"density={stats['density']:.6f}, "
            f"{stats['max_generation']} topological gens)"
        )

    def __repr__(self) -> str:
        return self.summary()
