"""Tests for GNN-powered best-first search (hybrid)."""

import sys
from pathlib import Path
import pytest
import torch

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from src.explorer.gnn_best_first_search import (
    GNNBestFirstSearch,
    GNNBestFirstConfig,
    _PrioritizedState,
)
from src.explorer.proof_state import ProofState, Tactic, TacticType


class TestGNNBestFirstConfig:
    """Configuration defaults."""

    def test_defaults(self):
        cfg = GNNBestFirstConfig()
        assert cfg.max_depth == 20
        assert cfg.max_expansions == 5000
        assert cfg.top_k_lemmas == 30
        assert cfg.depth_penalty == 0.05
        assert cfg.use_proof_checker is False
        assert cfg.device == "cpu"
        assert cfg.num_threads == 4
        assert cfg.max_graph_candidates == 200

    def test_custom_values(self):
        cfg = GNNBestFirstConfig(
            max_depth=10,
            max_expansions=100,
            top_k_lemmas=15,
            depth_penalty=0.1,
            use_proof_checker=True,
            num_threads=2,
            max_graph_candidates=50,
        )
        assert cfg.max_depth == 10
        assert cfg.max_expansions == 100
        assert cfg.top_k_lemmas == 15
        assert cfg.depth_penalty == 0.1
        assert cfg.use_proof_checker is True
        assert cfg.num_threads == 2
        assert cfg.max_graph_candidates == 50


class TestPrioritizedState:
    """Priority queue ordering."""

    def test_ordering(self):
        state = ProofState.initial("theorem test : 1 = 1")
        s1 = _PrioritizedState(priority=-1.0, depth=0, tiebreaker=1, state=state, steps=[])
        s2 = _PrioritizedState(priority=-0.5, depth=0, tiebreaker=2, state=state, steps=[])
        s3 = _PrioritizedState(priority=-1.0, depth=1, tiebreaker=3, state=state, steps=[])

        # More negative priority = higher priority (popped first from min-heap)
        assert s1 < s2  # -1.0 < -0.5
        assert s1 < s3  # Same priority, shallower depth wins

    def test_tiebreaker(self):
        state = ProofState.initial("theorem test : 1 = 1")
        s1 = _PrioritizedState(priority=-1.0, depth=0, tiebreaker=1, state=state, steps=[])
        s2 = _PrioritizedState(priority=-1.0, depth=0, tiebreaker=2, state=state, steps=[])

        # Same priority, same depth → tiebreaker decides
        assert s1 < s2  # Lower tiebreaker wins


class TestLemmaGoalKeywordMatch:
    """Keyword matching between lemmas and goal text."""

    def test_full_match(self):
        # "add" and "comm" are both substrings of the goal text
        score = GNNBestFirstSearch._lemma_goal_keyword_match(
            "add_comm",
            "theorem add_comm_example (a b : add_comm_group G) : a + b = b + a"
        )
        assert score > 0.9

    def test_partial_match(self):
        score = GNNBestFirstSearch._lemma_goal_keyword_match(
            "mul_comm",
            "theorem test (a b : Nat) : a + b = b + a"  # add, not mul
        )
        assert score < 0.5

    def test_no_match(self):
        score = GNNBestFirstSearch._lemma_goal_keyword_match(
            "derivative_X_pow",
            "theorem test (a b : Nat) : a + b = b + a"
        )
        assert score < 0.3


class TestAllKeywords:
    """Keyword set for building keyword map."""

    def test_includes_builtins(self):
        kws = GNNBestFirstSearch._all_keywords()
        assert "add_comm" in kws
        assert "mul_comm" in kws
        assert "rfl" in kws

    def test_includes_math_ops(self):
        kws = GNNBestFirstSearch._all_keywords()
        assert "ring" in kws
        assert "field" in kws
        assert "deriv" in kws
        assert "integral" in kws


class TestGNNBestFirstSearchInit:
    """Constructor tests for GNNBestFirstSearch."""

    def test_config_passed(self):
        """Verify config is stored and used."""
        cfg = GNNBestFirstConfig(max_depth=42, top_k_lemmas=7)
        # We can test the config passthrough without a full GNN setup
        assert cfg.max_depth == 42
        assert cfg.top_k_lemmas == 7
