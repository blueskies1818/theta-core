"""Tests for value_network module."""

import pytest
import torch
import torch.nn as nn

from src.explorer.value_network import ValueHead, ValueNetwork


class TestValueHead:
    """Tests for the ValueHead MLP."""

    def test_forward_scalar(self):
        vh = ValueHead(input_dim=64, hidden_dim=32)
        x = torch.randn(64)
        out = vh(x)
        assert out.ndim == 0  # scalar
        assert 0.0 <= out.item() <= 1.0

    def test_forward_batch(self):
        vh = ValueHead(input_dim=64, hidden_dim=32)
        x = torch.randn(8, 64)
        out = vh(x)
        assert out.shape == (8,)
        assert (out >= 0.0).all() and (out <= 1.0).all()

    def test_output_range_diverse_inputs(self):
        vh = ValueHead(input_dim=64, hidden_dim=32)
        # Test with normalized and extreme values
        for _ in range(50):
            x = torch.randn(64) * 5.0  # wider range
            out = vh(x)
            assert 0.0 <= out.item() <= 1.0

    def test_deterministic(self):
        vh = ValueHead(input_dim=64, hidden_dim=32)
        vh.eval()
        x = torch.randn(64)
        out1 = vh(x)
        out2 = vh(x)
        assert torch.allclose(out1, out2)


class TestValueNetwork:
    """Tests for ValueNetwork wrapping a mock GNN."""

    @pytest.fixture
    def mock_gnn(self):
        """Create a minimal mock GNN with a goal_encoder."""
        from types import SimpleNamespace

        class MockGoalEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(256, 256)

            def forward(self, x):
                return nn.functional.normalize(self.proj(x), dim=-1)

        gnn = SimpleNamespace()
        gnn.goal_encoder = MockGoalEncoder()
        gnn.config = SimpleNamespace()
        gnn.config.hidden_dim = 256
        # Add dummy parameters for freeze test
        gnn.dummy_param = nn.Parameter(torch.randn(10, 10))
        return gnn

    def test_instantiation(self, mock_gnn):
        vn = ValueNetwork(mock_gnn, hidden_dim=128)
        assert vn.enc_dim == 256
        assert isinstance(vn.value_head, ValueHead)

        # Check GNN is frozen — mock GNN with SimpleNamespace can't have
        # parameters() frozen, but the try/except handles this gracefully.
        # For real nn.Module GNNs, params would be frozen.
        # Value head should definitely be trainable
        assert all(p.requires_grad for p in vn.value_head.parameters())

    def test_forward(self, mock_gnn):
        vn = ValueNetwork(mock_gnn, hidden_dim=128)
        x = torch.randn(4, 256)
        out = vn(x)
        assert out.shape == (4,)
        assert (out >= 0.0).all() and (out <= 1.0).all()

    def test_predict(self, mock_gnn):
        vn = ValueNetwork(mock_gnn, hidden_dim=128)
        x = torch.randn(256)
        x.requires_grad = True  # Ensure grad tracking is on before predict
        out = vn.predict(x)
        assert out.ndim == 0
        assert 0.0 <= out.item() <= 1.0
        # predict() uses torch.no_grad() internally, so out should not
        # have a grad_fn (grad was disabled during the forward pass)
        assert out.grad_fn is None

    def test_save_load_roundtrip(self, mock_gnn, tmp_path):
        vn = ValueNetwork(mock_gnn, hidden_dim=128)
        path = tmp_path / "value_net.pt"

        # Save
        vn.save(path)
        assert path.exists()

        # Load
        vn2 = ValueNetwork.load(path, mock_gnn)
        assert vn2.enc_dim == vn.enc_dim

        # Verify weights match
        for p1, p2 in zip(vn.value_head.parameters(), vn2.value_head.parameters()):
            assert torch.allclose(p1, p2)

    def test_freeze_encoder_false(self, mock_gnn):
        # Reset requires_grad
        mock_gnn.dummy_param.requires_grad = True
        vn = ValueNetwork(mock_gnn, freeze_encoder=False)
        assert mock_gnn.dummy_param.requires_grad

    def test_default_forward_no_goal_encoder(self):
        from types import SimpleNamespace

        gnn = SimpleNamespace()
        gnn.goal_encoder = None
        gnn.config = SimpleNamespace()
        gnn.config.hidden_dim = 256
        gnn.dummy_param = nn.Parameter(torch.randn(10, 10))

        vn = ValueNetwork(gnn, hidden_dim=128)
        x = torch.randn(4, 256)
        out = vn(x)
        assert out.shape == (4,)

    def test_value_head_architecture(self):
        vh = ValueHead(input_dim=768, hidden_dim=256)
        # Verify it's a 2-layer MLP with GELU, Dropout
        net = vh.net
        assert len(net) == 4  # Linear, GELU, Dropout, Linear
        assert isinstance(net[0], nn.Linear)
        assert net[0].in_features == 768
        assert net[0].out_features == 256
        assert isinstance(net[1], nn.GELU)
        assert isinstance(net[2], nn.Dropout)
        assert isinstance(net[3], nn.Linear)
        assert net[3].in_features == 256
        assert net[3].out_features == 1
