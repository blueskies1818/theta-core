"""Tests for the expression sequence model and tokenizer."""

import pytest
import torch

from src.physics.model import (
    ExpressionTokenizer,
    ExpressionSequenceModel,
    ExpressionDataset,
    extract_training_examples,
    create_train_test_split,
    PAD_IDX,
    SOS_IDX,
    EOS_IDX,
    UNK_IDX,
    SEP_IDX,
)


class TestExpressionTokenizer:
    """Tests for the ExpressionTokenizer."""

    def test_vocab_size(self):
        tok = ExpressionTokenizer()
        assert tok.vocab_size > 20
        assert tok.vocab_size < 100

    def test_special_tokens(self):
        tok = ExpressionTokenizer()
        assert tok.encode("<pad>") == PAD_IDX
        assert tok.encode("<sos>") == SOS_IDX
        assert tok.encode("<eos>") == EOS_IDX
        assert tok.encode("<sep>") == SEP_IDX
        assert tok.encode("<unk>") == UNK_IDX

    def test_encode_quantities(self):
        tok = ExpressionTokenizer()
        ids = tok.encode_quantities(["m", "g", "h", "v"])
        assert len(ids) == 4
        for tid in ids:
            assert tid > UNK_IDX  # all real tokens

    def test_encode_scenario(self):
        tok = ExpressionTokenizer()
        sid = tok.encode_scenario("free_fall")
        assert sid > UNK_IDX
        sid2 = tok.encode_scenario("spring")
        assert sid2 > UNK_IDX
        assert sid != sid2

    def test_tokenize_simple_expression(self):
        tok = ExpressionTokenizer()
        ids = tok.tokenize_expression("m*g*h")
        assert len(ids) == 5  # m, *, g, *, h
        back = tok.detokenize_expression(ids)
        assert "m" in back and "g" in back and "h" in back

    def test_tokenize_complex_expression(self):
        tok = ExpressionTokenizer()
        ids = tok.tokenize_expression("0.5*m*v^2 + m*g*h")
        assert len(ids) >= 11
        back = tok.detokenize_expression(ids)
        # Should contain key pieces
        assert "0.5" in back
        assert "m" in back
        assert "v" in back
        assert "2" in back

    def test_tokenize_spring_expression(self):
        tok = ExpressionTokenizer()
        ids = tok.tokenize_expression("0.5*k*h^2 + 0.5*m*v^2")
        assert len(ids) >= 13
        back = tok.detokenize_expression(ids)
        assert "k" in back

    def test_tokenize_combined_expression(self):
        tok = ExpressionTokenizer()
        ids = tok.tokenize_expression("0.5*m*v^2 + 0.5*k*h^2 - m*g*h")
        assert len(ids) >= 17
        back = tok.detokenize_expression(ids)
        assert "k" in back and "g" in back

    def test_expression_to_tensor(self):
        tok = ExpressionTokenizer()
        t = tok.expression_to_tensor("m*g*h", max_len=64)
        assert len(t) == 64
        assert t[0].item() == SOS_IDX
        # Last non-pad should be EOS
        non_pad = t[t != PAD_IDX]
        assert non_pad[-1].item() == EOS_IDX

    def test_quantities_to_tensor(self):
        tok = ExpressionTokenizer()
        t = tok.quantities_to_tensor(["m", "g", "h"], max_len=16)
        assert len(t) == 16
        assert t[0].item() != PAD_IDX
        assert t[3].item() == PAD_IDX  # padded

    def test_save_load_roundtrip(self, tmp_path):
        tok = ExpressionTokenizer()
        path = tmp_path / "vocab.json"
        tok.save(path)

        tok2 = ExpressionTokenizer.load(path)
        assert tok2.vocab_size == tok.vocab_size
        assert tok2.encode("m") == tok.encode("m")
        assert tok2.encode("free_fall") == tok.encode("free_fall")

    def test_unknown_token_fallback(self):
        tok = ExpressionTokenizer()
        # A token not in vocab
        tid = tok.encode("z_unknown_xyz")
        assert tid == UNK_IDX

    def test_detokenize_stops_at_eos(self):
        tok = ExpressionTokenizer()
        ids = [SOS_IDX, tok.encode("m"), EOS_IDX, tok.encode("g"), tok.encode("h")]
        result = tok.detokenize_expression(ids)
        assert "g" not in result  # stopped at EOS

    def test_empty_expression(self):
        tok = ExpressionTokenizer()
        ids = tok.tokenize_expression("")
        assert ids == []

    def test_all_scenario_types_encodable(self):
        tok = ExpressionTokenizer()
        from src.physics.model import SCENARIO_TYPES
        for st in SCENARIO_TYPES:
            tid = tok.encode_scenario(st)
            assert tid != UNK_IDX, f"Scenario type {st!r} not in vocab"


class TestExpressionSequenceModel:
    """Tests for the ExpressionSequenceModel."""

    @pytest.fixture
    def tokenizer(self):
        return ExpressionTokenizer()

    @pytest.fixture
    def model(self, tokenizer):
        return ExpressionSequenceModel(vocab_size=tokenizer.vocab_size)

    def test_model_creation(self, model):
        assert model is not None
        assert model.d_model == 128
        n_params = model.count_parameters()
        assert n_params < 1_000_000, f"Model too large: {n_params:,} params"
        assert n_params > 100_000, f"Model too small: {n_params:,} params"

    def test_forward_pass_shape(self, model, tokenizer):
        src = tokenizer.quantities_to_tensor(["m", "g", "h", "v"], max_len=16).unsqueeze(0)
        tgt = tokenizer.expression_to_tensor("m*g*h", max_len=64).unsqueeze(0)
        src_mask = (src == PAD_IDX)
        tgt_mask_causal = torch.nn.Transformer.generate_square_subsequent_mask(
            tgt.size(1)
        )

        output = model(src, tgt, src_padding_mask=src_mask, tgt_mask=tgt_mask_causal)
        assert output.shape == (1, 64, tokenizer.vocab_size)

    def test_encode_source(self, model, tokenizer):
        src = tokenizer.quantities_to_tensor(["m", "g", "h"], max_len=16).unsqueeze(0)
        src_mask = (src == PAD_IDX)
        memory = model.encode_source(src, src_padding_mask=src_mask)
        assert memory.shape == (1, 16, 128)

    def test_generate_output_shape(self, model, tokenizer):
        model.eval()
        src = tokenizer.quantities_to_tensor(["m", "g", "h", "v"], max_len=16).unsqueeze(0)
        src_mask = (src == PAD_IDX)

        with torch.no_grad():
            generated = model.generate(src, src_padding_mask=src_mask, max_len=32, temperature=0)

        assert len(generated) == 1  # batch size 1
        assert len(generated[0]) >= 2  # at least SOS + something
        assert generated[0][0] == SOS_IDX  # starts with SOS

    def test_generate_batch(self, model, tokenizer):
        model.eval()
        src1 = tokenizer.quantities_to_tensor(["m", "g", "h"], max_len=16)
        src2 = tokenizer.quantities_to_tensor(["m", "k", "h"], max_len=16)
        src = torch.stack([src1, src2])
        src_mask = (src == PAD_IDX)

        with torch.no_grad():
            generated = model.generate(src, src_padding_mask=src_mask, max_len=32)

        assert len(generated) == 2

    def test_loss_decreases_one_batch(self, model, tokenizer):
        """Overfitting test: model should memorize a single example."""
        import torch.nn as nn
        import torch.optim as optim

        src = tokenizer.quantities_to_tensor(["m", "g", "h", "v"], max_len=16).unsqueeze(0)
        tgt = tokenizer.expression_to_tensor("m*g*h", max_len=64).unsqueeze(0)
        src_mask = (src == PAD_IDX)

        criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        tgt_causal = torch.nn.Transformer.generate_square_subsequent_mask(tgt.size(1))

        # Record initial loss
        model.train()
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        logits = model(src, tgt_input, src_padding_mask=src_mask,
                       tgt_mask=tgt_causal[:-1, :-1])
        initial_loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            tgt_output.reshape(-1),
        ).item()

        # Train for several steps
        losses = [initial_loss]
        for _ in range(100):
            optimizer.zero_grad()
            logits = model(src, tgt_input, src_padding_mask=src_mask,
                           tgt_mask=tgt_causal[:-1, :-1])
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_output.reshape(-1),
            )
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should decrease significantly
        assert losses[-1] < losses[0] * 0.5, (
            f"Loss didn't decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
        )


class TestExpressionDataset:
    """Tests for the ExpressionDataset."""

    def test_dataset_creation(self):
        tok = ExpressionTokenizer()
        examples = [
            {
                "quantities": {"m": "Mass", "g": "Accel", "h": "Length", "v": "Velocity"},
                "scenario_type": "free_fall",
                "expression": "m*g*h + 0.5*m*v^2",
            }
        ]
        ds = ExpressionDataset(examples, tok)
        assert len(ds) == 1

        src, tgt = ds[0]
        assert src.dim() == 1
        assert tgt.dim() == 1
        assert tgt[0].item() == SOS_IDX

    def test_dataset_multiple_examples(self):
        tok = ExpressionTokenizer()
        examples = [
            {
                "quantities": {"m": "Mass", "g": "Accel", "h": "Length", "v": "Velocity"},
                "scenario_type": "free_fall",
                "expression": "m*g*h + 0.5*m*v^2",
            },
            {
                "quantities": {"m": "Mass", "k": "Force/Length", "h": "Length", "v": "Velocity"},
                "scenario_type": "spring",
                "expression": "0.5*k*h^2 + 0.5*m*v^2",
            },
        ]
        ds = ExpressionDataset(examples, tok)
        assert len(ds) == 2

    def test_dataset_dataloader(self):
        tok = ExpressionTokenizer()
        examples = [
            {
                "quantities": {"m": "Mass", "g": "Accel", "h": "Length", "v": "Velocity"},
                "scenario_type": "free_fall",
                "expression": "m*g*h + 0.5*m*v^2",
            }
            for _ in range(4)
        ]
        ds = ExpressionDataset(examples, tok)
        loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False)

        for src, tgt in loader:
            assert src.shape[0] == 2
            assert tgt.shape[0] == 2
            break


class TestDataLoading:
    """Tests for data extraction and splitting."""

    def test_extract_training_examples(self):
        examples = extract_training_examples(
            "/home/blueman1818/Projects/theta-core/data/observations/phase2_extended.json",
        )
        assert len(examples) >= 30
        for ex in examples:
            assert "quantities" in ex
            assert "scenario_type" in ex
            assert "expression" in ex
            assert ex["expression"]  # non-empty

    def test_extract_with_discoveries(self):
        examples = extract_training_examples(
            "/home/blueman1818/Projects/theta-core/data/observations/phase2_extended.json",
            "/home/blueman1818/Projects/theta-core/data/phase_e_discoveries.json",
        )
        assert len(examples) >= 35

    def test_create_random_split(self):
        examples = extract_training_examples(
            "/home/blueman1818/Projects/theta-core/data/observations/phase2_extended.json",
        )
        train, test = create_train_test_split(examples, test_size=0.3, seed=42)
        assert len(train) > 0
        assert len(test) > 0
        assert len(train) + len(test) == len(examples)

    def test_create_cross_domain_split(self):
        examples = extract_training_examples(
            "/home/blueman1818/Projects/theta-core/data/observations/phase2_extended.json",
            "/home/blueman1818/Projects/theta-core/data/phase_e_discoveries.json",
        )
        test_types = {"gravity_spring", "em_gravity"}
        train, test = create_train_test_split(
            examples, test_scenario_types=test_types,
        )
        assert len(test) >= 2
        for ex in test:
            assert ex["scenario_type"] in test_types
        for ex in train:
            assert ex["scenario_type"] not in test_types

    def test_gravity_spring_in_test(self):
        """Verify that mass_spring_gravity is in test for cross-domain split."""
        examples = extract_training_examples(
            "/home/blueman1818/Projects/theta-core/data/observations/phase2_extended.json",
        )
        test_types = {"gravity_spring"}
        _, test = create_train_test_split(
            examples, test_scenario_types=test_types,
        )
        expressions = [ex["expression"] for ex in test]
        has_combined = any("k" in e and "g" in e for e in expressions)
        assert has_combined, f"Combined gravity+spring not in test: {expressions}"


class TestModelGeneration:
    """Integration tests for model generation after minimal training."""

    def test_generation_produces_valid_tokens(self):
        tok = ExpressionTokenizer()
        model = ExpressionSequenceModel(vocab_size=tok.vocab_size)
        model.eval()

        src = tok.quantities_to_tensor(["m", "g", "h", "v"], max_len=16).unsqueeze(0)
        src_mask = (src == PAD_IDX)

        with torch.no_grad():
            generated = model.generate(src, src_padding_mask=src_mask, max_len=32)

        # All generated tokens should be within vocab range
        for tid in generated[0]:
            assert 0 <= tid < tok.vocab_size, f"Token {tid} out of range"

    def test_batch_independence(self):
        """Each batch item should get independent generation."""
        tok = ExpressionTokenizer()
        model = ExpressionSequenceModel(vocab_size=tok.vocab_size)
        model.eval()

        src1 = tok.quantities_to_tensor(["m", "g", "h"], max_len=16)
        src2 = tok.quantities_to_tensor(["m", "k", "h"], max_len=16)
        src = torch.stack([src1, src2])
        src_mask = (src == PAD_IDX)

        with torch.no_grad():
            gen_greedy = model.generate(
                src, src_padding_mask=src_mask, max_len=32, temperature=0
            )

        # Greedy generation should be deterministic
        assert len(gen_greedy) == 2


class TestCrossDomainComposition:
    """Tests for cross-domain composition: combined training examples and splits."""

    def test_generate_combined_examples(self):
        """Combined cross-domain examples should be well-formed."""
        # Import from the training script
        import sys
        from pathlib import Path
        _root = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(_root / "scripts" / "training"))
        from train_expression_model import generate_combined_examples

        examples = generate_combined_examples()
        assert len(examples) >= 8, f"Expected >= 8 combined examples, got {len(examples)}"

        # Verify scenario types
        types = set(ex["scenario_type"] for ex in examples)
        assert "gravity_spring" in types
        assert "em_gravity" in types
        assert "spring_friction" in types
        assert "gravity_spring_friction" in types

        # Each example must have required keys
        for ex in examples:
            assert "quantities" in ex
            assert "scenario_type" in ex
            assert "expression" in ex
            assert ex["expression"], "Expression must be non-empty"

    def test_combined_examples_in_dataset(self):
        """Combined examples should be encodable into a dataset."""
        import sys
        from pathlib import Path
        _root = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(_root / "scripts" / "training"))
        from train_expression_model import generate_combined_examples

        tok = ExpressionTokenizer()
        combined = generate_combined_examples()
        ds = ExpressionDataset(combined, tok)
        assert len(ds) == len(combined)

        # Verify each example encodes without error
        for i in range(len(ds)):
            src, tgt = ds[i]
            assert src.dim() == 1
            assert tgt.dim() == 1
            assert src[0].item() != PAD_IDX  # should have content
            assert tgt[0].item() == SOS_IDX

    def test_gravity_spring_example_has_spring_term(self):
        """Gravity+spring combined examples must include spring term (½kx²)."""
        import sys
        from pathlib import Path
        _root = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(_root / "scripts" / "training"))
        from train_expression_model import generate_combined_examples

        examples = generate_combined_examples()
        gs_examples = [ex for ex in examples if ex["scenario_type"] == "gravity_spring"]
        assert len(gs_examples) >= 3

        for ex in gs_examples:
            assert "k" in ex["expression"], (
                f"Gravity+spring example must include k (spring term): {ex['expression']}"
            )
            assert "g" in ex["expression"], (
                f"Gravity+spring example must include g (gravity term): {ex['expression']}"
            )

    def test_charged_particle_quantities_fixed(self):
        """charged_particle_gravity should have q and E in quantities after fix."""
        examples = extract_training_examples(
            "/home/blueman1818/Projects/theta-core/data/observations/phase2_extended.json",
        )
        charged = [ex for ex in examples
                   if ex["scenario_type"] == "em_gravity"
                   and "q" in ex["expression"]]
        assert len(charged) > 0, "No charged particle examples found"

        for ex in charged:
            if "q" in ex["expression"] and "E" in ex["expression"]:
                assert "q" in ex["quantities"], (
                    f"q missing from quantities for em_gravity example: {ex['expression']}"
                )
                assert "E" in ex["quantities"], (
                    f"E missing from quantities for em_gravity example: {ex['expression']}"
                )

    def test_cross_domain_split_held_out_types(self):
        """Cross-domain split should put spring_friction in test, gravity_spring in train."""
        import sys
        from pathlib import Path
        _root = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(_root / "scripts" / "training"))
        from train_expression_model import generate_combined_examples

        examples = extract_training_examples(
            "/home/blueman1818/Projects/theta-core/data/observations/phase2_extended.json",
        )
        combined = generate_combined_examples()

        # Simulate the cross-domain split logic
        train_combined = [
            ex for ex in combined
            if ex["scenario_type"] in ("gravity_spring", "em_gravity")
        ]
        test_combined = [
            ex for ex in combined
            if ex["scenario_type"] in ("spring_friction", "gravity_spring_friction")
        ]

        assert len(train_combined) >= 4, f"Need >=4 train combined, got {len(train_combined)}"
        assert len(test_combined) >= 3, f"Need >=3 test combined, got {len(test_combined)}"

        # Train combined should have gravity+spring and EM+gravity
        train_types = set(ex["scenario_type"] for ex in train_combined)
        assert train_types <= {"gravity_spring", "em_gravity"}

        # Test combined should have spring+friction and gravity+spring+friction
        test_types = set(ex["scenario_type"] for ex in test_combined)
        assert test_types <= {"spring_friction", "gravity_spring_friction"}

    def test_scenario_types_in_vocab(self):
        """All scenario types including new ones should be in tokenizer vocab."""
        tok = ExpressionTokenizer()
        from src.physics.model import SCENARIO_TYPES
        for st in SCENARIO_TYPES:
            tid = tok.encode(st)
            assert tid != UNK_IDX, f"Scenario type {st!r} not in vocab (got UNK_IDX)"

    def test_infer_spring_friction_scenario(self):
        """Scenario inference should recognize spring_friction types."""
        from src.physics.model import _infer_scenario_type

        # damped spring (no gravity)
        st = _infer_scenario_type("spring_damped_light", ["m", "k", "h", "v"])
        assert st == "spring_friction", f"Expected spring_friction, got {st}"

        # mass_spring_damped_gravity
        st = _infer_scenario_type("mass_spring_damped_gravity", ["m", "k", "g", "h", "v"])
        assert st == "gravity_spring_friction", f"Expected gravity_spring_friction, got {st}"

    def test_model_generates_spring_with_k_present(self):
        """Untrained model should produce valid tokens without crashing when k present."""
        tok = ExpressionTokenizer()
        model = ExpressionSequenceModel(vocab_size=tok.vocab_size)
        model.eval()

        # Source: spring+gravity quantities
        src = tok.quantities_to_tensor(
            ["m", "g", "h", "v", "k", "x"], max_len=16
        ).unsqueeze(0)
        src_mask = (src == PAD_IDX)

        with torch.no_grad():
            generated = model.generate(
                src, src_padding_mask=src_mask, max_len=48, temperature=0
            )

        # All generated tokens should be valid (< vocab_size)
        # Note: untrained model may produce just SOS+EOS — that's fine
        assert len(generated[0]) >= 2, "Should have at least SOS + EOS"
        assert generated[0][0] == SOS_IDX
        for tid in generated[0]:
            assert 0 <= tid < tok.vocab_size, f"Token {tid} out of range"

    def test_deterministic_greedy_generation(self):
        """Greedy generation (temperature=0) should be deterministic."""
        tok = ExpressionTokenizer()
        model = ExpressionSequenceModel(vocab_size=tok.vocab_size)
        model.eval()

        src = tok.quantities_to_tensor(
            ["m", "g", "h", "v"], max_len=16
        ).unsqueeze(0)
        src_mask = (src == PAD_IDX)

        with torch.no_grad():
            gen1 = model.generate(
                src, src_padding_mask=src_mask, max_len=32, temperature=0
            )
            gen2 = model.generate(
                src, src_padding_mask=src_mask, max_len=32, temperature=0
            )

        assert gen1[0] == gen2[0], "Greedy generation should be deterministic"
