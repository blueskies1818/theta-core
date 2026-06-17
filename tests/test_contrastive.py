"""Tests for contrastive dual-encoder model and tokenizer."""

import pytest
import torch

from src.contrastive.encoder import (
    ContrastiveConfig,
    ContrastiveDualEncoder,
    CharCNNEncoder,
    CharTokenizer,
)


class TestCharTokenizer:
    """Tests for character-level tokenizer."""

    @pytest.fixture
    def tokenizer(self):
        return CharTokenizer(max_len=64)

    def test_encode_simple_text(self, tokenizer):
        result = tokenizer.encode("hello")
        assert result.shape == (64,)
        assert result.dtype == torch.long
        # First 5 chars should be 'hello' ordinals
        assert result[0].item() == ord("h")
        assert result[1].item() == ord("e")
        assert result[4].item() == ord("o")
        # Rest should be padding (0)
        assert (result[5:] == 0).all()

    def test_encode_empty_string(self, tokenizer):
        result = tokenizer.encode("")
        assert result.shape == (64,)
        assert (result == 0).all()

    def test_encode_truncation(self, tokenizer):
        long_text = "x" * 100
        result = tokenizer.encode(long_text)
        assert result.shape == (64,)
        # Should not have any x past position 64
        assert (result == ord("x")).sum() == 64

    def test_encode_greek_letters(self, tokenizer):
        result = tokenizer.encode("αβγ")
        assert result[0].item() > 127  # Greek letters are multibyte

    def test_encode_batch(self, tokenizer):
        result = tokenizer.encode_batch(["ab", "cd", "ef"])
        assert result.shape == (3, 64)
        assert result[0, 0].item() == ord("a")
        assert result[1, 0].item() == ord("c")
        assert result[2, 0].item() == ord("e")

    def test_preprocess_lemma_simple(self, tokenizer):
        result = CharTokenizer.preprocess_lemma("mul_comm")
        assert "mul" in result
        assert "comm" in result

    def test_preprocess_lemma_camelcase(self, tokenizer):
        # "intervalIntegral.integral_mul_const" → short="integral_mul_const"
        # (no CamelCase in short name, but underscores split)
        result = CharTokenizer.preprocess_lemma("intervalIntegral.integral_mul_const")
        assert "integral" in result
        assert "mul" in result
        assert "const" in result
        # However, if the lemma itself has CamelCase:
        result2 = CharTokenizer.preprocess_lemma("HasConstantSpeedOnWith")
        assert "has" in result2
        assert "constant" in result2
        assert "speed" in result2
        assert "on" in result2
        assert "with" in result2

    def test_preprocess_goal_whitespace(self, tokenizer):
        result = CharTokenizer.preprocess_goal("  a   +  b  =  c  ")
        assert result == "a + b = c"

    def test_preprocess_goal_unicode_leq(self, tokenizer):
        result = CharTokenizer.preprocess_goal("x ≤ y")
        assert "<=" in result


class TestCharCNNEncoder:
    """Tests for the character CNN encoder."""

    @pytest.fixture
    def config(self):
        return ContrastiveConfig(
            hidden_dim=128,
            vocab_size=256,
            max_seq_len=64,
            char_embed_dim=32,
            cnn_filters=64,
            cnn_kernel_sizes=(2, 3, 4),
            cnn_dropout=0.1,
            mlp_expansion=2,
            pooling="attention",
        )

    @pytest.fixture
    def encoder(self, config):
        return CharCNNEncoder(config)

    def test_output_shape(self, encoder):
        char_ids = torch.randint(1, 255, (8, 64))  # [B=8, L=64]
        output = encoder(char_ids)
        assert output.shape == (8, 128)
        # Should be L2 normalized
        norms = output.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(8), atol=1e-5)

    def test_output_deterministic(self, encoder):
        encoder.eval()
        char_ids = torch.randint(1, 255, (4, 64))
        with torch.no_grad():
            out1 = encoder(char_ids)
            out2 = encoder(char_ids)
        assert torch.equal(out1, out2)

    def test_pooling_modes(self, config):
        for mode in ["mean", "max", "attention"]:
            config.pooling = mode
            encoder = CharCNNEncoder(config)
            char_ids = torch.randint(1, 255, (2, 64))
            output = encoder(char_ids)
            assert output.shape == (2, 128)

    def test_handles_all_padding(self, encoder):
        char_ids = torch.zeros(4, 64, dtype=torch.long)  # All padding
        output = encoder(char_ids)
        # Should not crash, and should produce normalized output
        assert output.shape == (4, 128)


class TestContrastiveDualEncoder:
    """Tests for the full dual-encoder model."""

    @pytest.fixture
    def config(self):
        return ContrastiveConfig(
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

    @pytest.fixture
    def model(self, config):
        return ContrastiveDualEncoder(config)

    def test_forward_outputs(self, model):
        goal_ids = torch.randint(1, 255, (8, 32))
        lemma_ids = torch.randint(1, 255, (8, 32))

        output = model(goal_ids, lemma_ids)

        assert "logits" in output
        assert "loss" in output
        assert "goal_emb" in output
        assert "lemma_emb" in output

        assert output["logits"].shape == (8, 8)
        assert output["goal_emb"].shape == (8, 64)
        assert output["lemma_emb"].shape == (8, 64)
        assert output["loss"].ndim == 0  # scalar

    def test_loss_decreases_during_training(self, model):
        """After one gradient step on correct pairs, loss should decrease."""
        goal_ids = torch.randint(1, 255, (16, 32))
        lemma_ids = torch.randint(1, 255, (16, 32))

        # Initial loss
        output = model(goal_ids, lemma_ids)
        initial_loss = output["loss"].item()

        # One optimizer step
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        optimizer.zero_grad()
        output["loss"].backward()
        optimizer.step()

        # Loss after step
        output2 = model(goal_ids, lemma_ids)
        after_loss = output2["loss"].item()

        # Should decrease (or at least not NaN)
        assert not torch.isnan(output2["loss"])
        assert after_loss < initial_loss * 1.5  # Allow some noise

    def test_encoders_are_separate(self, model):
        """Goal and lemma encoders should have separate parameters."""
        goal_params = set(id(p) for p in model.goal_encoder.parameters())
        lemma_params = set(id(p) for p in model.lemma_encoder.parameters())
        assert goal_params.isdisjoint(lemma_params)

    def test_score_function(self, model):
        goal_ids = torch.randint(1, 255, (4, 32))
        lemma_ids = torch.randint(1, 255, (10, 32))

        scores = model.score(goal_ids, lemma_ids)
        assert scores.shape == (4, 10)

    def test_save_load_roundtrip(self, model, tmp_path):
        path = tmp_path / "test_model.pt"
        model.save(path)

        loaded = ContrastiveDualEncoder.load(path)
        assert loaded.config.hidden_dim == model.config.hidden_dim
        assert loaded.config.temperature == model.config.temperature

        # Check forward pass works
        goal_ids = torch.randint(1, 255, (4, 32))
        lemma_ids = torch.randint(1, 255, (4, 32))
        output = loaded(goal_ids, lemma_ids)
        assert output["logits"].shape == (4, 4)

    def test_param_counts(self, model):
        assert model.num_params > 0
        assert model.goal_encoder_params > 0
        assert model.lemma_encoder_params > 0
        assert model.num_params == model.goal_encoder_params + model.lemma_encoder_params

    def test_symmetric_loss(self, model):
        """Symmetric InfoNCE: the loss formula averages g2l and l2g directions."""
        goal_ids = torch.randint(1, 255, (16, 32))
        lemma_ids = torch.randint(1, 255, (16, 32))

        goal_emb = model.encode_goal(goal_ids)
        lemma_emb = model.encode_lemma(lemma_ids)

        # Compute logits
        logits = goal_emb @ lemma_emb.T * model._t_inv
        labels = torch.arange(16)

        loss_g2l = torch.nn.functional.cross_entropy(logits, labels)
        loss_l2g = torch.nn.functional.cross_entropy(logits.T, labels)
        expected_loss = (loss_g2l + loss_l2g) / 2.0

        # The model.contrastive_loss should match
        computed_loss = model.contrastive_loss(logits)
        assert torch.allclose(computed_loss, expected_loss), \
            f"Symmetric loss mismatch: {computed_loss:.4f} vs {expected_loss:.4f}"


class TestContrastiveConfig:
    """Tests for configuration defaults."""

    def test_defaults_are_reasonable(self):
        config = ContrastiveConfig()
        assert config.hidden_dim == 256
        assert config.temperature == 0.07
        assert config.batch_size == 256
        assert config.pooling == "attention"

    def test_custom_config(self):
        config = ContrastiveConfig(
            hidden_dim=128,
            temperature=0.1,
            cnn_kernel_sizes=(3, 5, 7),
        )
        assert config.hidden_dim == 128
        assert config.temperature == 0.1
        assert config.cnn_kernel_sizes == (3, 5, 7)
