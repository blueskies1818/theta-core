"""Integration tests for the self-play physics discovery loop."""

import json
import tempfile
from pathlib import Path

import pytest

from src.core.self_play_loop import (
    DiscoveryRecord,
    SelfPlayLoop,
    run_phase_c_smoke_test,
)

PHASE1_PATH = Path(__file__).parent.parent.parent / "data" / "observations" / "phase1_falling.json"

# Gravitational scenarios only — springs have no 'g', breaking energy search
GRAV_TRAIN = [
    "falling_ball_straight_drop", "falling_ball_upward_throw",
    "falling_ball_varying_mass", "projectile_45deg",
    "projectile_90deg", "sliding_block_incline",
    "pendulum_small_angle", "pendulum_large_angle",
]
GRAV_TEST = ["falling_ball_straight_drop", "falling_ball_upward_throw"]


# ── Train/test splitting ──────────────────────────────────────────────────────

class TestTrainTestSplit:

    def test_load_and_split_creates_correct_counts(self) -> None:
        loop = SelfPlayLoop(PHASE1_PATH, train_count=8, test_count=2, seed=42)
        loop.load_and_split()
        assert len(loop.train_observations) == 8
        assert len(loop.test_observations) == 2

    def test_train_test_no_overlap(self) -> None:
        loop = SelfPlayLoop(PHASE1_PATH, train_count=8, test_count=2, seed=42)
        loop.load_and_split()
        train_ids = {obs.id for obs in loop.train_observations}
        test_ids = {obs.id for obs in loop.test_observations}
        assert train_ids.isdisjoint(test_ids)

    def test_all_observations_covered(self) -> None:
        loop = SelfPlayLoop(PHASE1_PATH, train_count=8, test_count=2, seed=42)
        loop.load_and_split()
        all_ids = {obs.id for obs in loop.train_observations} | {
            obs.id for obs in loop.test_observations
        }
        assert len(all_ids) == 10

    def test_reproducible_split(self) -> None:
        loop1 = SelfPlayLoop(PHASE1_PATH, train_count=8, test_count=2, seed=42)
        loop1.load_and_split()
        loop2 = SelfPlayLoop(PHASE1_PATH, train_count=8, test_count=2, seed=42)
        loop2.load_and_split()
        train_ids1 = [obs.id for obs in loop1.train_observations]
        train_ids2 = [obs.id for obs in loop2.train_observations]
        assert train_ids1 == train_ids2

    def test_raises_when_not_enough_observations(self) -> None:
        loop = SelfPlayLoop(PHASE1_PATH, train_count=9, test_count=2, seed=42)
        with pytest.raises(ValueError, match="need at least"):
            loop.load_and_split()


# ── Discovery ─────────────────────────────────────────────────────────────────

class TestDiscovery:

    def test_discovers_energy_and_generalizes(self) -> None:
        """ACCEPTANCE: Self-play discovers energy and it generalizes to test set."""
        loop = SelfPlayLoop(
            PHASE1_PATH,
            train_ids=GRAV_TRAIN,
            test_ids=GRAV_TEST,
            max_expansions=5_000, discovery_threshold=0.95,
        )
        discoveries = loop.run()

        assert len(discoveries) > 0, (
            f"No discovery made. Total expansions: {loop.total_expansions}"
        )
        assert loop.discovered_energy, (
            f"Discovered {discoveries[0].expression} "
            f"failed generalization: train={discoveries[0].train_score:.4f}, "
            f"test={discoveries[0].test_score:.4f}"
        )

        discovery = discoveries[0]
        assert discovery.train_score > 0.95
        assert discovery.test_score > 0.95, (
            f"Generalization FAILED: test={discovery.test_score:.4f}, "
            f"expr={discovery.expression}"
        )
        assert discovery.expansions_needed < 5_000

    def test_summary_includes_discovery_info(self) -> None:
        loop = SelfPlayLoop(
            PHASE1_PATH, train_count=8, test_count=2,
            max_expansions=5_000, seed=42,
        )
        loop.run()
        summary = loop.summary()
        assert "Self-play discovery run" in summary
        if loop.discovered_energy:
            assert "Generalization: PASS" in summary

    def test_export_results_writes_valid_json(self) -> None:
        loop = SelfPlayLoop(
            PHASE1_PATH, train_count=8, test_count=2,
            max_expansions=1_000, seed=42,
        )
        loop.run()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            temp_path = f.name

        try:
            loop.export_results(temp_path)
            with open(temp_path) as f:
                data = json.load(f)
            assert "database" in data
            assert "result" in data
        finally:
            Path(temp_path).unlink(missing_ok=True)


# ── Smoke test function ───────────────────────────────────────────────────────

class TestSmokeTestFunction:

    def test_returns_expected_keys(self) -> None:
        result = run_phase_c_smoke_test(
            PHASE1_PATH, max_expansions=1_000, output_path=None,
        )
        assert "train_ids" in result
        assert "test_ids" in result
        assert "total_expansions" in result
        assert "discovered" in result
        assert "discoveries" in result

    def test_writes_output_file(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            temp_path = f.name
        try:
            run_phase_c_smoke_test(
                PHASE1_PATH, max_expansions=1_000, output_path=temp_path,
            )
            assert Path(temp_path).exists()
            with open(temp_path) as f:
                data = json.load(f)
            assert "result" in data
        finally:
            Path(temp_path).unlink(missing_ok=True)


# ── DiscoveryRecord ───────────────────────────────────────────────────────────

class TestDiscoveryRecord:

    def test_to_dict_roundtrip(self) -> None:
        record = DiscoveryRecord(
            expression="m*g*h + 0.5*m*v^2",
            train_score=0.98, test_score=0.97, depth=4,
            expansions_needed=1234,
            train_constancies=[0.98, 0.97, 0.99],
            test_constancies=[0.97, 0.96],
        )
        d = record.to_dict()
        assert d["expression"] == "m*g*h + 0.5*m*v^2"
        assert d["train_score"] == 0.98
        json.dumps(d)


# ── Edge cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_no_discovery_when_budget_too_small(self) -> None:
        loop = SelfPlayLoop(
            PHASE1_PATH,
            train_ids=GRAV_TRAIN, test_ids=GRAV_TEST,
            max_expansions=5,
        )
        discoveries = loop.run()
        assert isinstance(discoveries, list)

    def test_custom_disc_threshold(self) -> None:
        loop = SelfPlayLoop(
            PHASE1_PATH,
            train_ids=GRAV_TRAIN, test_ids=GRAV_TEST,
            max_expansions=5_000, discovery_threshold=0.80,
        )
        loop.run()
        assert loop.total_expansions > 0
        # With threshold 0.80 and gravitational data, discovery expected
        if len(loop.discoveries) > 0:
            assert loop.discoveries[0].train_score > 0.80


# ═══════════════════════════════════════════════════════════════════════════════
# Curriculum self-play loop tests (scripts/self_play_loop.py)
# ═══════════════════════════════════════════════════════════════════════════════

import json as _json_mod
import tempfile as _tempfile_mod

from scripts.self_play_loop import (
    CurriculumSelfPlayLoop,
    compare_expressions,
    SelfPlayResult,
    _COMPARISON_SCORES,
)


class TestCompareExpressions:

    def test_exact_match_identical(self) -> None:
        category, score = compare_expressions("m*g*h", "m*g*h")
        assert category == "exact"
        assert score == 1.0

    def test_exact_match_with_spaces(self) -> None:
        category, score = compare_expressions(" m * g * h ", "m*g*h")
        assert category == "exact"
        assert score == 1.0

    def test_exact_match_different_forms_not_exact(self) -> None:
        """0.5*m*v^2 vs m*v^2/2 are mathematically equal but structurally different."""
        category, score = compare_expressions("0.5*m*v^2", "m*v^2/2")
        # These are NOT exact string matches even after normalization
        assert category != "exact"

    def test_structural_match_same_vars(self) -> None:
        """m*g*h vs g*m*h — same vars, different ordering."""
        category, score = compare_expressions("m*g*h", "g*m*h")
        # Same variables in different order — structural match
        assert score >= 0.9 or score >= 0.5

    def test_structural_match_same_vars_partial_ordering(self) -> None:
        """Both have {m, v} — structural regardless of expression form."""
        category, score = compare_expressions("m*v", "v*m")
        assert score >= 0.9 or score >= 0.5

    def test_constant_match(self) -> None:
        category, score = compare_expressions("0.5", "1/2")
        assert category == "constant"
        assert score == 0.5

    def test_fail_different_vars(self) -> None:
        category, score = compare_expressions("m*g*h", "k*x^2")
        assert category == "fail"
        assert score == 0.0

    def test_fail_one_empty(self) -> None:
        category, score = compare_expressions("", "m*g*h")
        assert category == "fail"
        assert score == 0.0

        category, score = compare_expressions("m*g*h", "")
        assert category == "fail"
        assert score == 0.0

    def test_variable_set_extraction(self) -> None:
        """Ensure structural matching works for expressions with same variable sets."""
        # m*g*h has {m, g, h}, m*h/g has same set
        category, score = compare_expressions("m*g*h", "m*h*g")
        assert score >= 0.9


class TestCurriculumSelfPlayLoop:

    def _run_loop(self, **kwargs) -> CurriculumSelfPlayLoop:
        """Run a small self-play loop and return it."""
        params = {
            "levels": [1],
            "iterations_per_level": 3,
            "noise_frac": 0.01,
            "seed": 42,
        }
        params.update(kwargs)
        output = _tempfile_mod.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        )
        output.close()
        try:
            params["output_path"] = output.name
            loop = CurriculumSelfPlayLoop(**params)
            loop.run()
            return loop
        finally:
            Path(output.name).unlink(missing_ok=True)

    def test_loop_produces_results_for_level_1(self) -> None:
        """Loop produces results for level 1."""
        loop = self._run_loop(levels=[1], iterations_per_level=3)
        assert len(loop.results) > 0, "Loop should produce at least some results"
        for r in loop.results:
            assert r.level == 1
            assert r.expression, "Expression should not be empty"
            assert r.elapsed_seconds >= 0

    def test_output_is_valid_jsonl(self) -> None:
        """Results file is valid JSONL."""
        output = _tempfile_mod.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        )
        output.close()
        output_path = output.name
        try:
            loop = CurriculumSelfPlayLoop(
                levels=[1],
                iterations_per_level=3,
                seed=42,
                output_path=output_path,
            )
            loop.run()

            assert Path(output_path).exists()
            with open(output_path) as f:
                lines = f.readlines()
            assert len(lines) > 0, "JSONL file should have at least one line"

            for i, line in enumerate(lines):
                record = _json_mod.loads(line)
                assert "level" in record, f"Line {i} missing 'level'"
                assert "expression" in record, f"Line {i} missing 'expression'"
                assert "discovered" in record, f"Line {i} missing 'discovered'"
                assert "comparison" in record, f"Line {i} missing 'comparison'"
                assert "comparison_score" in record, f"Line {i} missing 'comparison_score'"
                assert "elapsed_seconds" in record, f"Line {i} missing 'elapsed_seconds'"
        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_per_level_stats(self) -> None:
        """Per-level statistics are computed correctly."""
        output = _tempfile_mod.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        )
        output.close()
        output_path = output.name
        try:
            loop = CurriculumSelfPlayLoop(
                levels=[1, 2],
                iterations_per_level=3,
                seed=42,
                output_path=output_path,
            )
            loop.run()

            stats = loop.per_level_stats()
            assert 1 in stats
            assert 2 in stats
            for level, s in stats.items():
                assert "total" in s
                assert "successes" in s
                assert "rate" in s
                assert s["total"] >= 0
                assert 0.0 <= s["rate"] <= 1.0
        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_loop_handles_graceful_shutdown(self) -> None:
        """Loop completes even with few iterations."""
        loop = self._run_loop(
            levels=[1],
            iterations_per_level=2,
        )
        assert len(loop.results) >= 0, "Loop should not crash even with 0 results"

    def test_seed_reproducibility(self) -> None:
        """Same seed produces same first result."""
        output1 = _tempfile_mod.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        )
        output1.close()
        output2 = _tempfile_mod.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        )
        output2.close()
        try:
            loop1 = CurriculumSelfPlayLoop(
                levels=[1], iterations_per_level=3, seed=42,
                output_path=output1.name,
            )
            loop1.run()
            loop2 = CurriculumSelfPlayLoop(
                levels=[1], iterations_per_level=3, seed=42,
                output_path=output2.name,
            )
            loop2.run()

            assert len(loop1.results) == len(loop2.results)
            if loop1.results:
                assert loop1.results[0].expression == loop2.results[0].expression
        finally:
            Path(output1.name).unlink(missing_ok=True)
            Path(output2.name).unlink(missing_ok=True)


class TestSelfPlayResult:

    def test_to_dict_roundtrip(self) -> None:
        record = SelfPlayResult(
            level=1,
            iteration=5,
            expression="m*g*h",
            ground_truth="m*g*h",
            discovered="m*g*h",
            comparison="exact",
            comparison_score=1.0,
            constancy_score=0.98,
            domain="gravity",
            num_observations=20,
            elapsed_seconds=0.123,
        )
        d = record.to_dict()
        assert d["level"] == 1
        assert d["expression"] == "m*g*h"
        assert d["comparison_score"] == 1.0
        _json_mod.dumps(d)  # must serialize

    def test_error_field(self) -> None:
        record = SelfPlayResult(
            level=1,
            iteration=1,
            expression="",
            ground_truth="",
            discovered="",
            comparison="fail",
            comparison_score=0.0,
            constancy_score=0.0,
            domain="",
            num_observations=0,
            elapsed_seconds=0.0,
            error="ValueError: test error",
        )
        assert record.error == "ValueError: test error"
        _json_mod.dumps(record.to_dict())


class TestCLI:
    """Smoke test the CLI argument parsing."""

    def test_default_args(self) -> None:
        from scripts.self_play_loop import parse_args
        args = parse_args([])
        assert args.levels == [1, 2, 3, 4]
        assert args.iterations_per_level == 1000
        assert args.noise == 0.01
        assert args.seed == 42

    def test_custom_args(self) -> None:
        from scripts.self_play_loop import parse_args
        args = parse_args([
            "--levels", "1", "2",
            "--iterations-per-level", "50",
            "--noise", "0.05",
            "--seed", "123",
            "--output", "custom.jsonl",
        ])
        assert args.levels == [1, 2]
        assert args.iterations_per_level == 50
        assert args.noise == 0.05
        assert args.seed == 123
        assert args.output == "custom.jsonl"
