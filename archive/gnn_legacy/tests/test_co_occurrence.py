"""Tests for co-occurrence edge enrichment (PATH 1).

Validates:
- EdgeType.CO_OCCURS_IN_PROOF exists and maps to index 4
- add_co_occurrence_edges() correctly adds edges from proof_step_pairs
- Name aliases enable short-name resolution
- GNNConfig.num_edge_types == 5
- GNN forward pass with edge_type index 4
"""

import json
import sys
import tempfile
from pathlib import Path

import pytest
import torch

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from src.explorer.dependency_graph import (
    DependencyGraph,
    DependencyNode,
    EdgeType,
    NodeType,
)
from src.explorer.gnn_config import GNNConfig
from src.explorer.gnn_encoder import GNNEncoder, prepare_graph_tensors
from src.explorer.graph_builder import add_co_occurrence_edges


class TestEdgeType:
    """CO_OCCURS_IN_PROOF edge type is defined and mapped correctly."""

    def test_edge_type_exists(self):
        assert hasattr(EdgeType, "CO_OCCURS_IN_PROOF")
        assert EdgeType.CO_OCCURS_IN_PROOF.value == "co_occurs_in_proof"

    def test_edge_type_mapped_to_index_4(self):
        """verify prepare_graph_tensors maps CO_OCCURS_IN_PROOF → 4."""
        g = _build_mini_graph()
        g.add_edge("Math.Data.Nat.basic_add", "Math.Data.Nat.basic_mul",
                    EdgeType.CO_OCCURS_IN_PROOF)

        sources, targets, edge_types, num_nodes = prepare_graph_tensors(g)

        # find the CO_OCCURS_IN_PROOF edge
        co_edge_types = edge_types[edge_types == 4]
        assert co_edge_types.numel() > 0, (
            f"Expected at least one edge with type 4, got edges: {edge_types.tolist()}"
        )


class TestGNNConfigEdgeTypes:
    """num_edge_types defaults to 5."""

    def test_default_num_edge_types(self):
        cfg = GNNConfig()
        assert cfg.num_edge_types == 5

    def test_config_override(self):
        cfg = GNNConfig(num_edge_types=3)
        assert cfg.num_edge_types == 3


class TestGNNForwardWithEdgeType5:
    """GNN encoder supports edge type index 4 (CO_OCCURS_IN_PROOF)."""

    def test_forward_with_edge_type_4(self):
        cfg = GNNConfig(hidden_dim=128, num_layers=2, num_heads=4,
                         input_dim=128, num_edge_types=5)
        gnn = GNNEncoder(cfg)
        N = 30
        x = torch.randn(N, cfg.input_dim)
        sources = torch.tensor([0, 1, 2, 3, 4, 5])
        targets = torch.tensor([1, 2, 3, 4, 5, 0])
        edge_types = torch.tensor([0, 1, 4, 0, 4, 2], dtype=torch.long)
        out = gnn(x, sources, targets, edge_types, N)
        assert out.shape == (N, cfg.hidden_dim)
        assert not torch.isnan(out).any()

    def test_all_edge_types_work(self):
        """Forward pass with all 5 edge types simultaneously."""
        cfg = GNNConfig(hidden_dim=128, num_layers=2, num_heads=4,
                         input_dim=128, num_edge_types=5)
        gnn = GNNEncoder(cfg)
        N = 50
        x = torch.randn(N, cfg.input_dim)
        # Build edges using all 5 types
        sources = []
        targets = []
        etypes = []
        for i in range(5):
            for j in range(5):
                sources.append(i)
                targets.append((i + j + 1) % N)
                etypes.append(j % 5)

        sources = torch.tensor(sources)
        targets = torch.tensor(targets)
        edge_types = torch.tensor(etypes, dtype=torch.long)
        out = gnn(x, sources, targets, edge_types, N)
        assert out.shape == (N, cfg.hidden_dim)
        assert not torch.isnan(out).any()


class TestAddCoOccurrenceEdges:
    """add_co_occurrence_edges() from proof_step_pairs."""

    def test_basic_enrichment(self):
        """edges are added from proof_step_pairs."""
        g = _build_mini_graph()

        # Register short name aliases
        for nid in g.node_ids:
            parts = nid.split(".")
            if len(parts) > 1:
                short = parts[-1]
                if short not in g._node_index:
                    g._node_index[short] = nid

        # Write mini proof_step_pairs
        pairs_lines = [
            {"goal": "...", "lemma": "basic_add", "name": "Math.Data.Nat.basic_add", "domain": "Data/Nat"},
            {"goal": "...", "lemma": "basic_mul", "name": "Math.Data.Nat.basic_mul", "domain": "Data/Nat"},
            # short name for lemma
            {"goal": "...", "lemma": "basic_zero", "name": "Math.Algebra.Group.zero_add", "domain": "Algebra/Group"},
        ]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as tf:
            for p in pairs_lines:
                tf.write(json.dumps(p) + "\n")
            tmp_path = tf.name

        try:
            stats = add_co_occurrence_edges(g, tmp_path)
            assert stats["added"] >= 1, f"Expected at least 1 edge added, got {stats}"
            assert stats["total_pairs"] == 3

            # Verify edges exist
            assert g._graph.has_edge(
                "Math.Data.Nat.basic_add", "Math.Data.Nat.basic_add"
            ) or g._graph.has_edge(
                "Math.Data.Nat.basic_add", "Math.Data.Nat.basic_mul"
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_missing_goal_skipped(self):
        """nodes not in graph are gracefully skipped."""
        g = _build_mini_graph()
        lines = [
            {"goal": "...", "lemma": "basic_add", "name": "NonExistent.Theorem.name", "domain": "Unknown"},
        ]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as tf:
            for p in lines:
                tf.write(json.dumps(p) + "\n")
            tmp_path = tf.name

        try:
            stats = add_co_occurrence_edges(g, tmp_path)
            assert stats["added"] == 0
            assert stats["skipped_missing_goal"] == 1
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_missing_lemma_skipped(self):
        """lemma not in graph is skipped."""
        g = _build_mini_graph()
        # Register alias so goal resolves
        g._node_index["zero_add_alias"] = "Math.Algebra.Group.zero_add"
        lines = [
            {"goal": "...", "lemma": "nonexistent_lemma_xyz", "name": "zero_add_alias", "domain": "Algebra"},
        ]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as tf:
            for p in lines:
                tf.write(json.dumps(p) + "\n")
            tmp_path = tf.name

        try:
            stats = add_co_occurrence_edges(g, tmp_path)
            assert stats["added"] == 0
            assert stats["skipped_missing_lemma"] == 1
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_existing_edge_skipped(self):
        """edge already exists (any type) is skipped, not double-counted."""
        g = _build_mini_graph()
        # Register aliases
        g._node_index["ba"] = "Math.Data.Nat.basic_add"
        g._node_index["bm"] = "Math.Data.Nat.basic_mul"
        # Pre-add an edge
        g.add_edge("Math.Data.Nat.basic_add", "Math.Data.Nat.basic_mul",
                    EdgeType.USES_IN_PROOF)

        lines = [
            {"goal": "...", "lemma": "bm", "name": "ba", "domain": "Data/Nat"},
        ]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as tf:
            for p in lines:
                tf.write(json.dumps(p) + "\n")
            tmp_path = tf.name

        try:
            stats = add_co_occurrence_edges(g, tmp_path)
            assert stats["added"] == 0
            assert stats["skipped_existing"] == 1
        finally:
            Path(tmp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_mini_graph() -> DependencyGraph:
    """Build a small dependency graph for testing."""
    g = DependencyGraph()
    nodes = [
        DependencyNode(
            id="Math.Data.Nat.basic_add",
            name="Math.Data.Nat.basic_add",
            node_type=NodeType.THEOREM,
            domain="Data/Nat",
        ),
        DependencyNode(
            id="Math.Data.Nat.basic_mul",
            name="Math.Data.Nat.basic_mul",
            node_type=NodeType.THEOREM,
            domain="Data/Nat",
        ),
        DependencyNode(
            id="Math.Data.Nat.basic_zero",
            name="Math.Data.Nat.basic_zero",
            node_type=NodeType.LEMMA,
            domain="Data/Nat",
        ),
        DependencyNode(
            id="Math.Algebra.Group.zero_add",
            name="Math.Algebra.Group.zero_add",
            node_type=NodeType.LEMMA,
            domain="Algebra/Group",
        ),
    ]
    for n in nodes:
        g.add_node(n)
    return g
