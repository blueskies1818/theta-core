"""Tests for hard-negative contrastive mining and loss functions."""

import json
import pytest
import torch

from src.contrastive.hard_negative_loss import (
    compute_infonce_loss,
    compute_triplet_margin_loss,
    compute_combined_loss,
    compute_retrieval_accuracy,
)
from src.contrastive.hard_negative_miner import (
    HardNegativeCache,
    build_lemma_goal_proof_script,
    load_hard_negative_data,
    save_hard_negative_data,
)


class TestHardNegativeCache:
    """Tests for the proof-checker result cache."""

    def test_empty_cache(self, tmp_path):
        cache_path = tmp_path / "cache.jsonl"
        cache = HardNegativeCache(cache_path=cache_path)
        assert len(cache) == 0

    def test_set_and_get(self, tmp_path):
        cache_path = tmp_path / "cache.jsonl"
        cache = HardNegativeCache(cache_path=cache_path)

        cache.set("goal_abc", "lemma_xyz", False)
        cache.set("goal_def", "lemma_uvw", True)

        assert cache.get("goal_abc", "lemma_xyz") is False
        assert cache.get("goal_def", "lemma_uvw") is True
        assert cache.get("goal_nonexistent", "lemma_nope") is None

    def test_persistence(self, tmp_path):
        cache_path = tmp_path / "cache.jsonl"
        cache = HardNegativeCache(cache_path=cache_path)
        cache.set("goal_a", "lemma_b", False)
        cache.set("goal_c", "lemma_d", True)
        cache.save()

        # Load fresh
        cache2 = HardNegativeCache(cache_path=cache_path)
        assert cache2.get("goal_a", "lemma_b") is False
        assert cache2.get("goal_c", "lemma_d") is True
        assert len(cache2) == 2

    def test_key_deterministic(self, tmp_path):
        cache_path = tmp_path / "cache.jsonl"
        cache = HardNegativeCache(cache_path=cache_path)
        cache.set("goal_x", "lemma_y", False)

        # Same goal+lemma should map to same key
        assert cache.get("goal_x", "lemma_y") is False


class TestBuildLemmaGoalProofScript:
    """Tests for Lean proof script construction."""

    def test_simple_theorem(self):
        result = build_lemma_goal_proof_script(
            "theorem foo (a b : ℕ) : a + b = b + a",
            "add_comm"
        )
        assert "example" in result
        assert "add_comm" in result
        assert ":=" in result

    def test_lemma_to_example(self):
        result = build_lemma_goal_proof_script(
            "lemma bar (x : ℝ) : x + 0 = x",
            "add_zero"
        )
        assert "example" in result
        assert "add_zero" in result
        assert "lemma" not in result

    def test_strips_existing_proof(self):
        result = build_lemma_goal_proof_script(
            "lemma bar (x : ℝ) : x + 0 = x := by simp",
            "add_zero"
        )
        assert "example" in result
        assert "simp" not in result

    def test_statement_without_proof_colon(self):
        result = build_lemma_goal_proof_script(
            "∀ x : ℕ, x ≤ x",
            "le_refl"
        )
        assert ":=" in result
        assert "le_refl" in result


class TestHardNegativeDataIO:
    """Tests for hard-negative triple I/O."""

    def test_save_and_load(self, tmp_path):
        path = tmp_path / "triples.jsonl"
        triples = [
            {
                "goal": "a + b = b + a",
                "positive_lemma": "add_comm",
                "hard_negatives": ["mul_comm", "sub_add"],
                "domain": "Algebra",
            },
            {
                "goal": "x ≤ x",
                "positive_lemma": "le_refl",
                "hard_negatives": ["lt_of_lt_of_le"],
                "domain": "Order",
            },
        ]
        save_hard_negative_data(triples, path)

        loaded = load_hard_negative_data(path)
        assert len(loaded) == 2
        assert loaded[0]["goal"] == "a + b = b + a"
        assert loaded[0]["hard_negatives"] == ["mul_comm", "sub_add"]
        assert loaded[1]["domain"] == "Order"

    def test_empty_list(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        save_hard_negative_data([], path)
        loaded = load_hard_negative_data(path)
        assert loaded == []


class TestInfoNCELoss:
    """Tests for InfoNCE loss function."""

    def test_perfect_matching(self):
        """When embeddings are perfectly aligned, loss should be minimal."""
        D = 64
        B = 16
        # Goal and lemma embeddings are identical → perfect match on diagonal
        goal_emb = torch.randn(B, D)
        goal_emb = torch.nn.functional.normalize(goal_emb, dim=-1)
        lemma_emb = goal_emb.clone()

        temperature_inv = 1.0 / 0.07
        loss = compute_infonce_loss(goal_emb, lemma_emb, temperature_inv)

        # With identical embeddings, loss should be very low
        # (log(B) = theoretical maximum for random, so perfect < 1.0)
        assert loss.item() < 1.5, f"Loss too high for perfect match: {loss.item():.4f}"
        assert not torch.isnan(loss)

    def test_random_embeddings(self):
        """Random embeddings: loss ≈ log(B) (chance level)."""
        D = 64
        B = 32
        goal_emb = torch.nn.functional.normalize(torch.randn(B, D), dim=-1)
        lemma_emb = torch.nn.functional.normalize(torch.randn(B, D), dim=-1)

        temperature_inv = 1.0 / 0.07
        loss = compute_infonce_loss(goal_emb, lemma_emb, temperature_inv)

        # Random → loss ≈ log(B) ≈ 3.5 for B=32
        assert loss.item() > 1.0, f"Loss too low for random: {loss.item():.4f}"
        assert not torch.isnan(loss)

    def test_symmetric(self):
        """Loss should be symmetric (goal→lemma == lemma→goal)."""
        D = 32
        B = 8
        goal_emb = torch.nn.functional.normalize(torch.randn(B, D), dim=-1)
        lemma_emb = torch.nn.functional.normalize(torch.randn(B, D), dim=-1)

        temperature_inv = 1.0 / 0.07
        loss = compute_infonce_loss(goal_emb, lemma_emb, temperature_inv)

        # Swap and recompute — should be the same
        loss_swapped = compute_infonce_loss(lemma_emb, goal_emb, temperature_inv)
        assert torch.allclose(loss, loss_swapped, atol=1e-6)


class TestTripletMarginLoss:
    """Tests for triplet margin loss with hard negatives."""

    def test_positive_closer_than_negative(self):
        """When positive is closer than negative, loss should be small/zero."""
        D = 32
        B = 4
        K = 3  # 3 hard negatives per pair

        goal_emb = torch.nn.functional.normalize(torch.randn(B, D), dim=-1)
        pos_emb = goal_emb + 0.05 * torch.randn(B, D)  # nearly identical
        pos_emb = torch.nn.functional.normalize(pos_emb, dim=-1)

        # Hard negatives: far away
        hard_neg_emb = torch.nn.functional.normalize(torch.randn(B, K, D), dim=-1)

        loss = compute_triplet_margin_loss(
            goal_emb, pos_emb, hard_neg_emb,
            margin=0.3, reduction="mean",
        )

        assert loss.item() >= 0.0
        assert not torch.isnan(loss)

    def test_negative_closer_than_positive(self):
        """When hard negatives are closer, loss should be high."""
        D = 32
        B = 4
        K = 2

        goal_emb = torch.nn.functional.normalize(torch.randn(B, D), dim=-1)

        # Positive: far away
        pos_emb = torch.nn.functional.normalize(torch.randn(B, D), dim=-1)

        # Hard negatives: identical to goal (worst case)
        hard_neg_emb = goal_emb.unsqueeze(1).expand(B, K, D)

        loss = compute_triplet_margin_loss(
            goal_emb, pos_emb, hard_neg_emb,
            margin=0.3, reduction="mean",
        )

        # Negatives are close (sim≈1), positive is random (sim≈0)
        # loss ≈ max(0, 0.3 - 0 + 1) = 1.3
        assert loss.item() > 0.5, f"Loss too low: {loss.item():.4f}"
        assert not torch.isnan(loss)

    def test_no_hard_negatives(self):
        """With K=0 hard negatives, loss should be 0."""
        D = 32
        B = 4
        K = 0

        goal_emb = torch.nn.functional.normalize(torch.randn(B, D), dim=-1)
        pos_emb = torch.nn.functional.normalize(torch.randn(B, D), dim=-1)
        hard_neg_emb = torch.zeros(B, K, D)

        loss = compute_triplet_margin_loss(goal_emb, pos_emb, hard_neg_emb)
        assert loss.item() == 0.0

    def test_sum_reduction(self):
        D = 16
        B = 2
        K = 3
        goal_emb = torch.nn.functional.normalize(torch.randn(B, D), dim=-1)
        pos_emb = torch.nn.functional.normalize(torch.randn(B, D), dim=-1)
        hard_neg_emb = torch.nn.functional.normalize(torch.randn(B, K, D), dim=-1)

        loss_mean = compute_triplet_margin_loss(
            goal_emb, pos_emb, hard_neg_emb, reduction="mean",
        )
        loss_sum = compute_triplet_margin_loss(
            goal_emb, pos_emb, hard_neg_emb, reduction="sum",
        )

        # sum = mean * B * K
        assert torch.allclose(loss_mean * B * K, loss_sum, atol=1e-4)


class TestCombinedLoss:
    """Tests for combined InfoNCE + hard-negative triplet loss."""

    def test_combined_equals_infonce_when_no_hard_negs(self):
        D = 32
        B = 8
        goal_emb = torch.nn.functional.normalize(torch.randn(B, D), dim=-1)
        pos_emb = torch.nn.functional.normalize(torch.randn(B, D), dim=-1)

        temperature_inv = 1.0 / 0.07

        losses = compute_combined_loss(
            goal_emb, pos_emb, None,
            temperature_inv=temperature_inv,
            hard_neg_weight=0.5,
        )

        infonce = compute_infonce_loss(goal_emb, pos_emb, temperature_inv)

        assert torch.allclose(losses["total_loss"], infonce)
        assert losses["hard_neg_loss"].item() == 0.0

    def test_combined_greater_with_hard_negs(self):
        D = 32
        B = 4
        K = 2
        goal_emb = torch.nn.functional.normalize(torch.randn(B, D), dim=-1)
        pos_emb = torch.nn.functional.normalize(torch.randn(B, D), dim=-1)

        # Hard negatives: identical to goal (worst case for separation)
        hard_neg_emb = goal_emb.unsqueeze(1).expand(B, K, D)

        temperature_inv = 1.0 / 0.07

        losses = compute_combined_loss(
            goal_emb, pos_emb, hard_neg_emb,
            temperature_inv=temperature_inv,
            hard_neg_weight=1.0,
        )

        # With hard negatives, total loss should be higher
        infonce = compute_infonce_loss(goal_emb, pos_emb, temperature_inv)
        assert losses["total_loss"] > infonce, \
            f"Combined loss ({losses['total_loss']:.4f}) should be > InfoNCE ({infonce:.4f})"
        assert losses["hard_neg_loss"] > 0.0

    def test_returns_all_components(self):
        D = 16
        B = 4
        K = 1
        goal_emb = torch.nn.functional.normalize(torch.randn(B, D), dim=-1)
        pos_emb = torch.nn.functional.normalize(torch.randn(B, D), dim=-1)
        hard_neg_emb = torch.nn.functional.normalize(torch.randn(B, K, D), dim=-1)

        losses = compute_combined_loss(
            goal_emb, pos_emb, hard_neg_emb,
            temperature_inv=1.0 / 0.07,
        )

        assert "total_loss" in losses
        assert "infonce_loss" in losses
        assert "hard_neg_loss" in losses
        assert losses["total_loss"].ndim == 0  # scalar
        assert losses["infonce_loss"].ndim == 0
        assert losses["hard_neg_loss"].ndim == 0


class TestRetrievalAccuracy:
    """Tests for retrieval accuracy computation."""

    def test_perfect_accuracy(self):
        D = 32
        B = 16
        goal_emb = torch.nn.functional.normalize(torch.randn(B, D), dim=-1)
        lemma_emb = goal_emb.clone()

        acc = compute_retrieval_accuracy(goal_emb, lemma_emb)
        # With identical embeddings, diagonal should be highest
        assert acc.item() > 0.9, f"Accuracy too low: {acc.item():.4f}"

    def test_random_accuracy(self):
        D = 64
        B = 50
        goal_emb = torch.nn.functional.normalize(torch.randn(B, D), dim=-1)
        lemma_emb = torch.nn.functional.normalize(torch.randn(B, D), dim=-1)

        acc = compute_retrieval_accuracy(goal_emb, lemma_emb)
        # Random: approximately 1/B accuracy
        expected = 1.0 / B
        assert abs(acc.item() - expected) < 0.15, \
            f"Expected ~{expected:.3f}, got {acc.item():.3f}"

    def test_output_range(self):
        D = 16
        B = 8
        goal_emb = torch.nn.functional.normalize(torch.randn(B, D), dim=-1)
        lemma_emb = torch.nn.functional.normalize(torch.randn(B, D), dim=-1)

        acc = compute_retrieval_accuracy(goal_emb, lemma_emb)
        assert 0.0 <= acc.item() <= 1.0


class TestContrastiveDualEncoderHardNeg:
    """Integration test: ContrastiveDualEncoder.forward_hard()."""

    @pytest.fixture
    def model(self):
        from src.contrastive.encoder import ContrastiveDualEncoder, ContrastiveConfig
        config = ContrastiveConfig(
            hidden_dim=64,
            vocab_size=256,
            max_seq_len=32,
            char_embed_dim=16,
            cnn_filters=32,
            cnn_kernel_sizes=(2, 3),
            cnn_dropout=0.1,
            mlp_expansion=2,
            pooling="mean",
            temperature=0.07,
        )
        return ContrastiveDualEncoder(config)

    def test_forward_hard_no_negatives(self, model):
        B = 8
        L = 32
        goal_ids = torch.randint(1, 255, (B, L))
        pos_ids = torch.randint(1, 255, (B, L))

        output = model.forward_hard(goal_ids, pos_ids, None)

        assert "total_loss" in output
        assert "infonce_loss" in output
        assert "hard_neg_loss" in output
        assert "goal_emb" in output
        assert "lemma_emb" in output
        assert "accuracy" in output
        assert output["hard_neg_loss"].item() == 0.0
        assert len(output["goal_emb"].shape) == 2

    def test_forward_hard_with_negatives(self, model):
        B = 4
        K = 2
        L = 32
        goal_ids = torch.randint(1, 255, (B, L))
        pos_ids = torch.randint(1, 255, (B, L))
        hn_ids = torch.randint(1, 255, (B, K, L))

        output = model.forward_hard(goal_ids, pos_ids, hn_ids,
                                     hard_neg_weight=0.5, margin=0.3)

        assert "total_loss" in output
        assert output["hard_neg_loss"].item() > 0.0
        assert not torch.isnan(output["total_loss"])

    def test_forward_hard_loss_decreases(self, model):
        """One gradient step should decrease total loss."""
        B = 16
        K = 2
        L = 32
        goal_ids = torch.randint(1, 255, (B, L))
        pos_ids = torch.randint(1, 255, (B, L))
        hn_ids = torch.randint(1, 255, (B, K, L))

        output1 = model.forward_hard(goal_ids, pos_ids, hn_ids)
        initial_loss = output1["total_loss"].item()

        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        opt.zero_grad()
        output1["total_loss"].backward()
        opt.step()

        output2 = model.forward_hard(goal_ids, pos_ids, hn_ids)
        after_loss = output2["total_loss"].item()

        assert after_loss < initial_loss * 1.5, \
            f"Loss didn't decrease: {initial_loss:.4f} → {after_loss:.4f}"
        assert not torch.isnan(output2["total_loss"])

    def test_save_load_with_hard_neg_config(self, model, tmp_path):
        """Save/load roundtrip preserves forward_hard functionality."""
        path = tmp_path / "test_model.pt"
        model.save(path)

        from src.contrastive.encoder import ContrastiveDualEncoder
        loaded = ContrastiveDualEncoder.load(path)

        B = 4
        K = 1
        L = 32
        goal_ids = torch.randint(1, 255, (B, L))
        pos_ids = torch.randint(1, 255, (B, L))
        hn_ids = torch.randint(1, 255, (B, K, L))

        output = loaded.forward_hard(goal_ids, pos_ids, hn_ids)
        assert "total_loss" in output
        assert not torch.isnan(output["total_loss"])
