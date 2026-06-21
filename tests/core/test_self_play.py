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
            PHASE1_PATH, train_count=8, test_count=2,
            max_expansions=5_000, discovery_threshold=0.95, seed=42,
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
            PHASE1_PATH, train_count=8, test_count=2,
            max_expansions=5, seed=42,
        )
        discoveries = loop.run()
        assert isinstance(discoveries, list)

    def test_custom_disc_threshold(self) -> None:
        loop = SelfPlayLoop(
            PHASE1_PATH, train_count=8, test_count=2,
            max_expansions=5_000, discovery_threshold=0.80, seed=42,
        )
        discoveries = loop.run()
        assert len(discoveries) > 0
        assert discoveries[0].train_score > 0.80
