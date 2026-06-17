"""Tests for H2 scoring architectures (src/explorer/scoring_architectures.py)."""

import pytest
import torch
import torch.nn.functional as F

from src.explorer.scoring_architectures import (
    BaselineCosineScorer,
    TwoTowerBilinearScorer,
    CrossAttentionScorer,
    GraphFilteredCosineScorer,
    build_goal_embedding,
)


class TestBaselineCosineScorer:
    def test_output_shape(self):
        scorer = BaselineCosineScorer(hidden_dim=256)
        q = torch.randn(4, 256)
        c = torch.randn(100, 256)
        scores = scorer(q, c)
        assert scores.shape == (4, 100)

    def test_single_query(self):
        scorer = BaselineCosineScorer(hidden_dim=256)
        q = torch.randn(256)
        c = torch.randn(100, 256)
        scores = scorer(q, c)
        assert scores.shape == (1, 100)

    def test_identical_gives_max_score(self):
        scorer = BaselineCosineScorer(hidden_dim=64)
        # Normalized identical vectors should give ~1.0
        v = F.normalize(torch.randn(64), dim=-1)
        scores = scorer(v.unsqueeze(0), v.unsqueeze(0))
        assert torch.allclose(scores, torch.tensor([[1.0]]), atol=1e-5)

    def test_range_bounded(self):
        scorer = BaselineCosineScorer(hidden_dim=64)
        q = torch.randn(5, 64)
        c = torch.randn(50, 64)
        scores = scorer(q, c)
        assert scores.min() >= -1.0
        assert scores.max() <= 1.0


class TestTwoTowerBilinearScorer:
    def test_output_shape(self):
        scorer = TwoTowerBilinearScorer(hidden_dim=256, bottleneck_dim=128)
        q = torch.randn(4, 256)
        c = torch.randn(100, 256)
        scores = scorer(q, c)
        assert scores.shape == (4, 100)

    def test_single_query(self):
        scorer = TwoTowerBilinearScorer(hidden_dim=256)
        q = torch.randn(256)
        c = torch.randn(100, 256)
        scores = scorer(q, c)
        assert scores.shape == (1, 100)

    def test_param_count(self):
        scorer = TwoTowerBilinearScorer(hidden_dim=256, bottleneck_dim=128)
        n_params = sum(p.numel() for p in scorer.parameters())
        # Towers: 2 * (256*256 + 256*128 + 128*128 + 128) + 128*128
        assert n_params > 100_000

    def test_calibration_reduces_loss(self):
        scorer = TwoTowerBilinearScorer(hidden_dim=64, bottleneck_dim=32)
        baseline = BaselineCosineScorer(hidden_dim=64)
        q = torch.randn(4, 64)
        c = torch.randn(50, 64)

        # Initial loss should be high
        with torch.no_grad():
            initial_pred = scorer(q, c)
            target = baseline(q, c)
        initial_loss = F.mse_loss(initial_pred, target).item()

        # After calibration, loss should decrease
        final_loss = scorer.calibrate_from_baseline(q, c, baseline, epochs=100, lr=1e-2)
        assert final_loss < initial_loss * 0.8, (
            f"Calibration should reduce loss: {initial_loss:.4f} → {final_loss:.4f}"
        )


class TestCrossAttentionScorer:
    def test_output_shape(self):
        scorer = CrossAttentionScorer(hidden_dim=256, num_heads=8)
        q = torch.randn(4, 256)
        c = torch.randn(100, 256)
        scores = scorer(q, c)
        assert scores.shape == (4, 100)

    def test_single_query(self):
        scorer = CrossAttentionScorer(hidden_dim=256, num_heads=8)
        q = torch.randn(256)
        c = torch.randn(100, 256)
        scores = scorer(q, c)
        assert scores.shape == (1, 100)

    def test_scores_sum_to_one(self):
        """Attention weights should sum to ~1.0 across candidates."""
        scorer = CrossAttentionScorer(hidden_dim=64, num_heads=4)
        scorer.eval()
        q = torch.randn(2, 64)
        c = torch.randn(50, 64)
        with torch.no_grad():
            scores = scorer(q, c)
        # Scores from attention mean should sum to ~1.0 per query
        for i in range(2):
            assert abs(scores[i].sum().item() - 1.0) < 1e-4, (
                f"Attention scores should sum to 1.0, got {scores[i].sum().item()}"
            )

    def test_num_heads_divisible(self):
        with pytest.raises(AssertionError, match="divisible"):
            CrossAttentionScorer(hidden_dim=256, num_heads=7)

    def test_calibration_reduces_loss(self):
        scorer = CrossAttentionScorer(hidden_dim=64, num_heads=4)
        baseline = BaselineCosineScorer(hidden_dim=64)
        q = torch.randn(4, 64)
        c = torch.randn(50, 64)

        with torch.no_grad():
            initial_pred = scorer(q, c)
            target = baseline(q, c)
        initial_loss = F.mse_loss(initial_pred, target).item()

        final_loss = scorer.calibrate_from_baseline(q, c, baseline, epochs=100, lr=1e-2)
        assert final_loss < initial_loss * 0.8, (
            f"Calibration should reduce loss: {initial_loss:.4f} → {final_loss:.4f}"
        )


class TestGraphFilteredCosineScorer:
    def test_output_shape(self):
        scorer = GraphFilteredCosineScorer(hidden_dim=256, k_hops=3)
        q = torch.randn(4, 256)
        c = torch.randn(100, 256)
        mask = torch.ones(4, 100, dtype=torch.bool)
        scores = scorer(q, c, mask)
        assert scores.shape == (4, 100)

    def test_masked_entries_are_neg_inf(self):
        scorer = GraphFilteredCosineScorer(hidden_dim=64, k_hops=2)
        q = torch.randn(2, 64)
        c = torch.randn(10, 64)

        # Mask: only first 3 candidates allowed for query 0
        mask = torch.zeros(2, 10, dtype=torch.bool)
        mask[0, :3] = True
        mask[1, :5] = True

        scores = scorer(q, c, mask)
        assert torch.all(torch.isfinite(scores[0, :3]))
        assert torch.all(scores[0, 3:] == float("-inf"))
        assert torch.all(torch.isfinite(scores[1, :5]))
        assert torch.all(scores[1, 5:] == float("-inf"))

    def test_full_mask_equals_cosine(self):
        """With all True mask, scores should equal cosine baseline."""
        dim = 64
        gf = GraphFilteredCosineScorer(hidden_dim=dim)
        baseline = BaselineCosineScorer(hidden_dim=dim)
        q = torch.randn(3, dim)
        c = torch.randn(20, dim)
        mask = torch.ones(3, 20, dtype=torch.bool)

        gf_scores = gf(q, c, mask)
        bl_scores = baseline(q, c)
        assert torch.allclose(gf_scores, bl_scores, atol=1e-5)

    def test_ranking_preserved_in_masked_region(self):
        """Ranking of unmasked candidates should match cosine ranking."""
        dim = 64
        gf = GraphFilteredCosineScorer(hidden_dim=dim)
        baseline = BaselineCosineScorer(hidden_dim=dim)
        q = torch.randn(2, dim)
        c = torch.randn(10, dim)
        mask = torch.zeros(2, 10, dtype=torch.bool)
        mask[:, :6] = True  # First 6 allowed

        gf_scores = gf(q, c, mask)
        bl_scores = baseline(q, c)

        # Rankings of first 6 candidates should match
        for i in range(2):
            gf_ranks = torch.argsort(gf_scores[i, :6], descending=True)
            bl_ranks = torch.argsort(bl_scores[i, :6], descending=True)
            assert torch.equal(gf_ranks, bl_ranks), (
                f"Ranking should be preserved for unmasked entries"
            )


class TestBuildGoalEmbedding:
    def test_returns_normalized_tensor(self):
        theorem = {
            "statement": "theorem test (a : ℝ) : a + 0 = a",
            "proof": "simp",
        }
        lemma_embs = {
            "add_zero": torch.randn(256),
            "eval_add": torch.randn(256),
        }
        emb = build_goal_embedding(theorem, lemma_embs)
        assert emb.shape == (256,)
        assert abs(emb.norm().item() - 1.0) < 1e-5

    def test_fallback_random(self):
        theorem = {
            "statement": "theorem obscure_test : 1 = 1",
            "proof": "rfl",
        }
        lemma_embs = {}  # No matching lemmas
        emb = build_goal_embedding(theorem, lemma_embs, fallback_dim=128)
        assert emb.shape == (128,)
        assert abs(emb.norm().item() - 1.0) < 1e-5

    def test_matches_eval_add_from_simp(self):
        theorem = {
            "statement": "theorem test : (C a + X).eval x = a + x",
            "proof": "simp",
        }
        lemma_embs = {
            "eval_add": torch.ones(256),
            "eval_X": torch.zeros(256),
            "unrelated": torch.randn(256),
        }
        emb = build_goal_embedding(theorem, lemma_embs)
        # Should match eval_add (since statement has eval and +)
        expected_dir = F.normalize(torch.ones(256), dim=-1)
        similarity = torch.dot(emb, expected_dir).item()
        assert similarity > 0.9, f"Goal embedding should align with eval_add, got sim={similarity:.3f}"

    def test_matches_lemma_from_proof_text(self):
        theorem = {
            "statement": "theorem test : p.derivative = q",
            "proof": "simpa using Polynomial.derivative_X_pow n",
        }
        lemma_embs = {
            "derivative_X_pow": torch.ones(256),
            "unrelated": torch.zeros(256),
        }
        emb = build_goal_embedding(theorem, lemma_embs)
        expected_dir = F.normalize(torch.ones(256), dim=-1)
        similarity = torch.dot(emb, expected_dir).item()
        assert similarity > 0.9, f"Goal embedding should match derivative_X_pow, got sim={similarity:.3f}"
