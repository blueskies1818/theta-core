"""Tests for GNNConfig and GNNEncoder with scaled 10M-param architecture.

Validates the architecture change: hidden_dim 256→768, num_layers 3→5, num_heads 8→12.
"""

import sys
from pathlib import Path
import pytest
import torch

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from src.explorer.gnn_config import GNNConfig
from src.explorer.gnn_encoder import GNNEncoder, GATLayer


class TestScaledGNNConfigDefaults:
    """Default config values must match the scaled 10M-param architecture."""

    def test_default_hidden_dim_is_768(self):
        cfg = GNNConfig()
        assert cfg.hidden_dim == 768, (
            f"Expected hidden_dim=768 (scaled from 256), got {cfg.hidden_dim}"
        )

    def test_default_num_layers_is_5(self):
        cfg = GNNConfig()
        assert cfg.num_layers == 5, (
            f"Expected num_layers=5 (scaled from 3), got {cfg.num_layers}"
        )

    def test_default_num_heads_is_12(self):
        cfg = GNNConfig()
        assert cfg.num_heads == 12, (
            f"Expected num_heads=12 (scaled from 8), got {cfg.num_heads}"
        )

    def test_default_input_dim_is_768(self):
        cfg = GNNConfig()
        assert cfg.input_dim == 768, (
            f"Expected input_dim=768, got {cfg.input_dim}"
        )

    def test_default_device_is_cpu(self):
        cfg = GNNConfig()
        assert cfg.device == "cpu", (
            f"Expected device='cpu' (GPU compute fused off), got {cfg.device}"
        )

    def test_hidden_dim_divisible_by_num_heads(self):
        """768 / 12 = 64, must be exact for GATLayer assertion."""
        cfg = GNNConfig()
        assert cfg.hidden_dim % cfg.num_heads == 0, (
            f"hidden_dim {cfg.hidden_dim} must be divisible by num_heads {cfg.num_heads}"
        )

    def test_other_defaults_unchanged(self):
        """Verify that only the scaled parameters changed, others stay the same."""
        cfg = GNNConfig()
        assert cfg.dropout == 0.1
        assert cfg.activation == "gelu"
        assert cfg.use_edge_types is True
        assert cfg.num_edge_types == 5  # was 4, now 5 with CO_OCCURS_IN_PROOF
        assert cfg.bidirectional is True
        assert cfg.use_goal_encoder is True
        assert cfg.goal_encoder_expansion == 2
        assert cfg.learning_rate == 1e-3
        assert cfg.objective == "link_prediction"


class TestGNNEncoderScaledConstruction:
    """GNNEncoder must construct successfully with the scaled config."""

    def test_encoder_constructs_with_scaled_config(self):
        cfg = GNNConfig()
        gnn = GNNEncoder(cfg)
        assert gnn is not None
        assert gnn.config.hidden_dim == 768
        assert gnn.config.num_layers == 5
        assert gnn.config.num_heads == 12

    def test_encoder_parameter_count(self):
        """Verify the scaled model has roughly ~10M params (> 5M, < 20M)."""
        cfg = GNNConfig()
        gnn = GNNEncoder(cfg)
        n_params = sum(p.numel() for p in gnn.parameters())
        # Should be substantially larger than the 1.1M baseline
        assert n_params > 5_000_000, (
            f"Expected >5M params for scaled model, got {n_params:,}"
        )
        assert n_params < 30_000_000, (
            f"Expected <30M params, got {n_params:,}"
        )

    def test_encoder_forward_smoke(self):
        """Smoke test: forward pass with random graph of 100 nodes."""
        cfg = GNNConfig()
        gnn = GNNEncoder(cfg)
        N = 100
        x = torch.randn(N, cfg.input_dim)
        # Create a simple chain graph: 0→1→2→...→N-1
        sources = torch.arange(0, N - 1)
        targets = torch.arange(1, N)
        edge_types = torch.zeros(N - 1, dtype=torch.long)
        out = gnn(x, sources, targets, edge_types, N)
        assert out.shape == (N, cfg.hidden_dim)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_encoder_forward_with_bidirectional(self):
        """Forward pass with bidirectional=True enabled."""
        cfg = GNNConfig(bidirectional=True)
        gnn = GNNEncoder(cfg)
        N = 50
        x = torch.randn(N, cfg.input_dim)
        sources = torch.tensor([0, 1, 2, 3, 4])
        targets = torch.tensor([1, 2, 3, 4, 0])
        edge_types = torch.tensor([0, 1, 2, 0, 1], dtype=torch.long)
        out = gnn(x, sources, targets, edge_types, N)
        assert out.shape == (N, cfg.hidden_dim)

    def test_gradient_flows(self):
        """Verify loss.backward() succeeds and gradients are nonzero."""
        cfg = GNNConfig()
        gnn = GNNEncoder(cfg)
        N = 40
        x = torch.randn(N, cfg.input_dim)
        sources = torch.arange(0, N - 1)
        targets = torch.arange(1, N)
        edge_types = torch.zeros(N - 1, dtype=torch.long)

        out = gnn(x, sources, targets, edge_types, N)
        loss = out.sum()
        loss.backward()

        # Check at least one parameter has gradients
        has_grad = False
        for p in gnn.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "No nonzero gradients after backward pass"

    def test_goal_encoder_works_with_scaled_dim(self):
        """GoalEncoder (2x expansion MLP) works at 768-dim."""
        cfg = GNNConfig()
        gnn = GNNEncoder(cfg)
        assert gnn.goal_encoder is not None
        # Keyword-averaged context embedding
        ctx = torch.randn(768)
        goal = gnn.encode_goal(ctx)
        assert goal.shape == (768,)
        # Batch
        ctx_batch = torch.randn(8, 768)
        goal_batch = gnn.encode_goal(ctx_batch)
        assert goal_batch.shape == (8, 768)


class TestGATLayerDivisibility:
    """Verify the divisibility constraint for head_dim."""

    def test_head_dim_exact_division(self):
        """With hidden_dim=768, num_heads=12, head_dim=64."""
        layer = GATLayer(
            in_dim=768, out_dim=768, num_heads=12,
            num_edge_types=5, dropout=0.1, activation="gelu",
        )
        assert layer.head_dim == 64
        assert layer.out_dim == 768

    def test_disallowed_division_raises(self):
        """out_dim=768 with num_heads=13 should raise AssertionError."""
        with pytest.raises(AssertionError):
            GATLayer(
                in_dim=768, out_dim=768, num_heads=13,
                num_edge_types=5, dropout=0.1, activation="gelu",
            )

    def test_small_forward(self):
        """Small forward pass through a single GATLayer at 768-dim."""
        layer = GATLayer(
            in_dim=768, out_dim=768, num_heads=12,
            num_edge_types=5, dropout=0.1, activation="gelu",
        )
        N = 20
        x = torch.randn(N, 768)
        sources = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
        targets = torch.tensor([1, 2, 3, 4, 5, 6, 7, 0])
        edge_types = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1], dtype=torch.long)
        out = layer(x, sources, targets, edge_types, N)
        assert out.shape == (N, 768)
        assert not torch.isnan(out).any()


class TestBackwardCompatibleConfig:
    """Old config values still work when explicitly passed."""

    def test_old_config_override(self):
        """Explicit overrides should work regardless of new defaults."""
        cfg = GNNConfig(hidden_dim=256, num_layers=3, num_heads=8)
        assert cfg.hidden_dim == 256
        assert cfg.num_layers == 3
        assert cfg.num_heads == 8

    def test_old_config_encoder_constructs(self):
        """GNNEncoder with old-style config should still work."""
        cfg = GNNConfig(hidden_dim=256, num_layers=3, num_heads=8)
        gnn = GNNEncoder(cfg)
        n_params = sum(p.numel() for p in gnn.parameters())
        # Old config: ~1.1M params
        assert n_params < 2_000_000

    def test_config_save_load_roundtrip(self):
        """Config dict roundtrips correctly through save/load."""
        import tempfile
        cfg = GNNConfig(hidden_dim=768, num_layers=5, num_heads=12)
        gnn = GNNEncoder(cfg)

        with tempfile.NamedTemporaryFile(suffix=".pt") as tmp:
            gnn.save(tmp.name)
            loaded = GNNEncoder.load(tmp.name)

        assert loaded.config.hidden_dim == 768
        assert loaded.config.num_layers == 5
        assert loaded.config.num_heads == 12
        assert sum(p.numel() for p in loaded.parameters()) == sum(
            p.numel() for p in gnn.parameters()
        )
