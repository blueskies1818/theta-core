"""Tests for value network integration in GNNBestFirstSearch."""

import pytest
import torch
from unittest.mock import MagicMock, patch

from src.explorer.gnn_best_first_search import GNNBestFirstConfig


class TestValueConfig:
    """Tests for value network configuration fields."""

    def test_defaults(self):
        cfg = GNNBestFirstConfig()
        assert cfg.value_weight == 0.3
        assert cfg.value_prune_threshold == 0.1

    def test_blind_mode(self):
        cfg = GNNBestFirstConfig(value_weight=0.0, value_prune_threshold=None)
        assert cfg.value_weight == 0.0
        assert cfg.value_prune_threshold is None

    def test_value_only_mode(self):
        cfg = GNNBestFirstConfig(value_weight=1.0, value_prune_threshold=0.5)
        assert cfg.value_weight == 1.0
        assert cfg.value_prune_threshold == 0.5


class TestValueIntegration:
    """Tests for _estimate_value method integration."""

    def test_estimate_value_no_network(self):
        """With no value network, _estimate_value returns neutral 0.5."""
        from src.explorer.gnn_best_first_search import GNNBestFirstSearch

        # Create a mock search instance
        search = MagicMock(spec=GNNBestFirstSearch)
        search.value_network = None

        # We test the method directly via the class
        from src.explorer.proof_state import ProofState
        from src.explorer.gnn_best_first_search import GNNBestFirstSearch as BFS

        # Access unbound method
        estimate = BFS._estimate_value

        state = ProofState.initial("example (x : ℝ) : x = x")
        result = estimate(search, state)
        assert result == 0.5

    def test_estimate_value_with_network(self):
        """With a value network, calls predict on goal embedding."""
        from src.explorer.proof_state import ProofState
        from src.explorer.gnn_best_first_search import GNNBestFirstSearch as BFS

        # Create mock
        search = MagicMock(spec=BFS)
        search.value_network = MagicMock()
        search.value_network.predict.return_value = torch.tensor(0.75)
        search._embed_goal = MagicMock(return_value=torch.randn(256))

        estimate = BFS._estimate_value
        state = ProofState.initial("example (x y : ℝ) : x + y = y + x")
        result = estimate(search, state)

        assert 0.0 <= result <= 1.0
        search._embed_goal.assert_called_once()

    def test_estimate_value_encoding_fails(self):
        """When goal encoding returns None, fall back to 0.5."""
        from src.explorer.proof_state import ProofState
        from src.explorer.gnn_best_first_search import GNNBestFirstSearch as BFS

        search = MagicMock(spec=BFS)
        search.value_network = MagicMock()
        search._embed_goal = MagicMock(return_value=None)

        estimate = BFS._estimate_value
        state = ProofState.initial("example : True")
        result = estimate(search, state)
        assert result == 0.5

    def test_estimate_value_exception(self):
        """When value network throws, fall back to 0.5."""
        from src.explorer.proof_state import ProofState
        from src.explorer.gnn_best_first_search import GNNBestFirstSearch as BFS

        search = MagicMock(spec=BFS)
        search.value_network = MagicMock()
        search.value_network.predict.side_effect = RuntimeError("fail")
        search._embed_goal = MagicMock(return_value=torch.randn(256))

        estimate = BFS._estimate_value
        state = ProofState.initial("example : True")
        result = estimate(search, state)
        assert result == 0.5
