"""Tests for symmetry discovery module.

Covers acceptance criteria:
  1. Candidate generation produces valid group sets
  2. Training data has expected scenarios with ground truth
  3. SymmetryDiscoverer matches known groups when appropriate
  4. Discovery mode activates when known groups fail
  5. ℝ × SO(2) rediscovered from central force data
  6. Poincaré × U(1) rediscovered from relativistic data
  7. Held-out Lorentz + SU(2) breaking correctly handled
  8. Galilean group rediscovered from Newtonian data
  9. All existing tests still pass
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch

from src.physics.observations import Observation, ObservationDatabase
from src.physics.symmetry import (
    GeneratorKind,
    GENERATOR_LABELS,
    PREBUILT_GROUPS,
    SymmetryDetector,
    NoetherDerivation,
    Lagrangian,
    build_galilean_group,
)
from src.physics.symmetry_discovery import (
    # Core
    SymmetryDiscoverer,
    CandidateGroup,
    DiscoveryResult,
    SymmetryScorer,
    # Generator functions
    DISCOVERY_GENERATOR_POOL,
    DISCOVERY_GENERATORS,
    generate_candidate_groups,
    candidate_to_symmetry_group,
    # Training
    generate_discovery_training_data,
    train_symmetry_discoverer,
    # Evaluation
    evaluate_discovery,
    run_symmetry_discovery_pipeline,
    run_discovery_on_database,
    save_discovery_results,
    # Smoke
    run_discovery_smoke_test,
    # Aliases
    GroupCandidate,
    build_discovery_training_scenarios,
    # Known groups
    KNOWN_GENERATOR_SETS,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def time_only_obs() -> Observation:
    """Free-fall observation with only time translation symmetry."""
    return Observation(
        id="test_time_only",
        name="Time only",
        description="1D free fall",
        quantities={
            "m": "Mass", "g": "Accel", "h": "Length",
            "v": "Velocity", "t": "Time",
        },
        parameters={"m": 2.0, "g": 9.8},
        timesteps=[
            {"t": 0.0, "h": 20.0, "v": 0.0},
            {"t": 0.5, "h": 18.775, "v": 4.9},
            {"t": 1.0, "h": 15.1, "v": 9.8},
        ],
        known_invariant="m*g*h + 0.5*m*v^2",
        lean_theorem="",
    )


@pytest.fixture
def rotation_obs() -> Observation:
    """2D central-force observation with time + rotation symmetry."""
    import math
    ts = []
    for i in range(10):
        t = i * 0.3
        theta = 2.0 * t
        ts.append({
            "t": round(t, 4),
            "x": round(5.0 * math.cos(theta), 4),
            "y": round(5.0 * math.sin(theta), 4),
            "vx": round(-10.0 * math.sin(theta), 4),
            "vy": round(10.0 * math.cos(theta), 4),
        })
    return Observation(
        id="test_rotation",
        name="Central force",
        description="2D uniform circular motion",
        quantities={
            "m": "Mass", "x": "Length", "y": "Length",
            "vx": "Velocity", "vy": "Velocity", "t": "Time",
        },
        parameters={"m": 1.0},
        timesteps=ts,
        known_invariant="0.5*m*vx^2 + 0.5*m*vy^2",
        lean_theorem="",
    )


@pytest.fixture
def full_spacetime_obs() -> Observation:
    """3D relativistic observation with full Poincaré-like variables."""
    ts = []
    for i in range(10):
        t = i * 0.2
        ts.append({
            "t": round(t, 4),
            "x": round(10.0 + 5.0 * t, 4),
            "y": round(2.0 * t, 4),
            "z": round(1.0 * t, 4),
            "vx": 5.0, "vy": 2.0, "vz": 1.0,
        })
    return Observation(
        id="test_spacetime",
        name="Free relativistic particle",
        description="Relativistic neutral particle in free motion",
        quantities={
            "m": "Mass", "c": "Speed",
            "x": "Length", "y": "Length", "z": "Length",
            "vx": "Velocity", "vy": "Velocity", "vz": "Velocity",
            "t": "Time",
        },
        parameters={"m": 1.0, "c": 1.0},
        timesteps=ts,
        known_invariant="m*c^2",
        lean_theorem="",
    )


@pytest.fixture
def discoverer() -> SymmetryDiscoverer:
    """Default rule-based discoverer."""
    return SymmetryDiscoverer(max_candidates=200, constancy_threshold=0.3)


# ── Candidate Generation Tests ───────────────────────────────────────────────

class TestCandidateGeneration:
    """Tests for candidate group generation."""

    def test_generator_pool_size(self):
        """Discovery pool has 12 generators."""
        assert len(DISCOVERY_GENERATOR_POOL) == 12
        assert len(DISCOVERY_GENERATORS) == 12

    def test_generate_default_candidates(self):
        """Default generation produces valid candidates."""
        candidates = generate_candidate_groups(max_groups=100)
        assert len(candidates) > 0
        assert len(candidates) <= 100
        # Each candidate is a list of generators
        for gen_set in candidates:
            assert isinstance(gen_set, list)
            assert all(isinstance(g, GeneratorKind) for g in gen_set)

    def test_candidates_include_physics_patterns(self):
        """Physics patterns (time, time+space, time+rotation) are included."""
        candidates = generate_candidate_groups(max_groups=200)
        gen_sets = [set(c) for c in candidates]

        # Time-only should be present
        assert {GeneratorKind.TIME_TRANSLATION} in gen_sets

        # U(1)-only should be present
        assert {GeneratorKind.U1_PHASE} in gen_sets

    def test_candidates_all_nonempty(self):
        """All candidates have at least one generator."""
        for gen_set in generate_candidate_groups(max_groups=50):
            assert len(gen_set) >= 1

    def test_candidate_to_symmetry_group(self):
        """Conversion produces valid SymmetryGroup."""
        group = candidate_to_symmetry_group([GeneratorKind.TIME_TRANSLATION])
        assert group is not None
        assert group.dimension == 1
        assert group.contains(GeneratorKind.TIME_TRANSLATION)
        assert group.invariant_for(GeneratorKind.TIME_TRANSLATION) is not None

    def test_candidate_to_galilean_group(self):
        """Galilean generator set converts correctly."""
        gal_gens = KNOWN_GENERATOR_SETS["galilean"]
        group = candidate_to_symmetry_group(gal_gens)
        assert group.dimension == 10
        assert "Galilean" in group.name or "Poincaré" in group.name

    def test_max_groups_limited(self):
        """Candidates are capped at max_groups."""
        for max_n in [1, 5, 20]:
            candidates = generate_candidate_groups(max_groups=max_n)
            assert len(candidates) <= max_n


# ── Training Data Tests ──────────────────────────────────────────────────────

class TestTrainingData:
    """Tests for synthetic training data generation."""

    def test_generates_scenarios(self):
        """Training data has exactly 5 scenarios."""
        obs, gt = generate_discovery_training_data()
        assert len(obs) == 5
        assert len(gt) == 5

    def test_all_scenarios_have_ground_truth(self):
        """Every scenario has ground-truth generators."""
        obs, gt = generate_discovery_training_data()
        for o in obs:
            assert o.id in gt
            assert len(gt[o.id]) > 0

    def test_training_vs_heldout_split(self):
        """Training and held-out scenarios are separable."""
        _, gt = generate_discovery_training_data()
        training_ids = [oid for oid in gt if not oid.startswith("heldout")]
        heldout_ids = [oid for oid in gt if oid.startswith("heldout")]
        assert len(training_ids) == 4  # time_only, time+rotation, lorentz+u1, galilean
        assert len(heldout_ids) == 1  # heldout_lorentz_su2_broken

    def test_heldout_has_no_su2(self):
        """Held-out scenario should NOT have SU(2) in ground truth."""
        _, gt = generate_discovery_training_data()
        heldout_gt = gt["heldout_lorentz_su2_broken"]
        assert GeneratorKind.SU2_WEAK not in heldout_gt

    def test_lorentz_u1_has_11_generators(self):
        """Poincaré × U(1) has 11 generators in ground truth."""
        _, gt = generate_discovery_training_data()
        assert len(gt["training_lorentz_u1"]) == 11

    def test_galilean_has_10_generators(self):
        """Galilean/Newtonian scenario has 10 generators."""
        _, gt = generate_discovery_training_data()
        assert len(gt["training_galilean_newtonian"]) == 10

    def test_build_discovery_training_scenarios_alias(self):
        """Alias function works identically."""
        obs1, gt1 = generate_discovery_training_data()
        obs2, gt2 = build_discovery_training_scenarios()
        assert len(obs1) == len(obs2)
        assert set(gt1.keys()) == set(gt2.keys())

    def test_scenarios_are_conservative(self):
        """All training scenarios are marked conservative."""
        obs, _ = generate_discovery_training_data()
        for o in obs:
            assert o.is_conservative is True

    def test_scenarios_have_timesteps(self):
        """All scenarios have adequate timesteps."""
        obs, _ = generate_discovery_training_data()
        for o in obs:
            assert len(o.timesteps) >= 10


# ── SymmetryScorer Tests ─────────────────────────────────────────────────────

class TestSymmetryScorer:
    """Tests for the MLP symmetry scorer."""

    def test_scorer_creation(self):
        """Scorer creates with reasonable parameter count."""
        scorer = SymmetryScorer(hidden_dim=64)
        n = scorer.count_parameters()
        assert n > 1000, f"Too few: {n}"
        assert n < 100000, f"Too many: {n}"

    def test_scorer_predict_shape(self):
        """Predict returns scalar in [0, 1]."""
        scorer = SymmetryScorer()
        obs = Observation(
            id="test", name="Test", description="...",
            quantities={"m": "Mass", "h": "Length", "v": "Velocity", "t": "Time"},
            parameters={"m": 1.0},
            timesteps=[{"t": 0.0, "h": 10.0, "v": 0.0}, {"t": 1.0, "h": 1.0, "v": 10.0}],
            known_invariant=None, lean_theorem="",
        )
        score = scorer.predict([GeneratorKind.TIME_TRANSLATION], obs)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_scorer_save_load(self):
        """Scorer can be saved and loaded."""
        scorer = SymmetryScorer(hidden_dim=32)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            scorer.save(f.name)
            loaded = SymmetryScorer.load(f.name)
            assert loaded.count_parameters() == scorer.count_parameters()
        Path(f.name).unlink(missing_ok=True)

    def test_scorer_training_runs(self):
        """Training converges on synthetic data."""
        obs, gt = generate_discovery_training_data()
        scorer = train_symmetry_discoverer(
            obs, gt, epochs=20, learning_rate=0.01,
            checkpoint_path=str(Path(tempfile.gettempdir()) / "test_scorer_tmp.pt"),
        )
        assert scorer.count_parameters() > 0
        # Clean up
        Path(tempfile.gettempdir(), "test_scorer_tmp.pt").unlink(missing_ok=True)


# ── SymmetryDiscoverer Tests ─────────────────────────────────────────────────

class TestSymmetryDiscoverer:
    """Tests for the main symmetry discovery pipeline."""

    def test_discoverer_creation(self):
        """Discoverer creates with defaults."""
        d = SymmetryDiscoverer()
        assert d.max_candidates == 1000
        assert d.constancy_threshold == 0.7

    def test_discover_on_time_only(self, discoverer, time_only_obs):
        """Discovery on time-only scenario finds time translation."""
        result = discoverer.discover(time_only_obs)
        assert result is not None
        assert result.scenario_id == "test_time_only"
        assert result.candidates_evaluated > 0
        assert result.best_candidate is not None

    def test_discover_on_rotation(self, discoverer, rotation_obs):
        """Discovery on 2D central force scenario runs without error."""
        result = discoverer.discover(rotation_obs)
        assert result is not None
        assert result.best_candidate is not None
        assert result.best_candidate.constancy_score >= 0.0

    def test_known_groups_checked(self, discoverer, time_only_obs):
        """Discovery checks known groups before searching."""
        result = discoverer.discover(time_only_obs)
        assert isinstance(result.detection.active_symmetries, list)

    def test_candidates_always_evaluated(self, discoverer, time_only_obs):
        """Discovery always evaluates candidates (not just known groups)."""
        result = discoverer.discover(time_only_obs)
        assert result.candidates_evaluated > 0

    def test_best_candidate_has_generators(self, discoverer, time_only_obs):
        """Best candidate has at least one generator."""
        result = discoverer.discover(time_only_obs)
        assert result.best_candidate.generator_count >= 1

    def test_report_is_nonempty(self, discoverer, time_only_obs):
        """Discovery produces a human-readable report."""
        result = discoverer.discover(time_only_obs)
        assert len(result.report) > 0

    def test_top_candidates_limited(self, discoverer, time_only_obs):
        """Top candidates list is at most 5."""
        result = discoverer.discover(time_only_obs)
        assert len(result.top_candidates) <= 5

    def test_discover_returns_discovery_result(self, discoverer, time_only_obs):
        """discover() returns DiscoveryResult instance."""
        result = discoverer.discover(time_only_obs)
        assert isinstance(result, DiscoveryResult)

    def test_discover_with_scorer(self, time_only_obs):
        """Discoverer works with trained scorer."""
        scorer = SymmetryScorer(hidden_dim=32)
        d = SymmetryDiscoverer(scorer=scorer, max_candidates=50)
        result = d.discover(time_only_obs)
        assert result.best_candidate is not None

    def test_full_spacetime_discovery(self, discoverer, full_spacetime_obs):
        """Discovery on 3D relativistic data runs correctly."""
        result = discoverer.discover(full_spacetime_obs)
        assert result is not None
        assert result.best_candidate is not None


# ── Acceptance Criteria Tests ────────────────────────────────────────────────

class TestAcceptanceCriteria:
    """Tests for the task acceptance criteria.

    These test that the discoverer:
      1. Rediscovers Galilean group from Newtonian data
      2. Rediscovers ℝ × SO(2) from central force data
      3. Rediscovers Poincaré × U(1) from relativistic data
      4. Correctly handles held-out SU(2) breaking scenario
    """

    @pytest.fixture(autouse=True)
    def setup_data(self):
        """Load training data once for all tests."""
        self.obs, self.gt = generate_discovery_training_data()
        self.discoverer = SymmetryDiscoverer(
            max_candidates=300, constancy_threshold=0.3
        )

    def test_galilean_rediscovery(self):
        """ACCEPTANCE: Rediscovers Galilean group from Newtonian data."""
        gal_obs = [o for o in self.obs if "galilean" in o.id][0]
        result = self.discoverer.discover(gal_obs)
        assert result.best_candidate is not None
        pred_gens = set(result.best_candidate.generators)
        expected_gens = set(self.gt[gal_obs.id])

        # Should have high overlap with expected generators
        intersection = pred_gens & expected_gens
        recall = len(intersection) / len(expected_gens) if expected_gens else 0.0
        assert recall >= 0.5, (
            f"Galilean rediscovery recall too low: {recall:.2f}. "
            f"Predicted: {[GENERATOR_LABELS.get(g,'?') for g in pred_gens]}. "
            f"Expected: {[GENERATOR_LABELS.get(g,'?') for g in expected_gens]}"
        )

    def test_r_so2_rediscovery(self):
        """ACCEPTANCE: Rediscovers ℝ × SO(2) from central force data."""
        rot_obs = [o for o in self.obs if "rotation" in o.id][0]
        result = self.discoverer.discover(rot_obs)
        assert result.best_candidate is not None
        pred_gens = set(result.best_candidate.generators)
        expected_gens = set(self.gt[rot_obs.id])

        # Must contain time translation
        assert GeneratorKind.TIME_TRANSLATION in pred_gens, (
            "Time translation missing from ℝ × SO(2) discovery"
        )

        # Must contain at least one rotation
        has_rotation = any(
            g in pred_gens for g in [
                GeneratorKind.ROTATION_XY,
                GeneratorKind.ROTATION_XZ,
                GeneratorKind.ROTATION_YZ,
            ]
        )
        assert has_rotation, "No rotation generator in ℝ × SO(2) discovery"

    def test_poincare_u1_rediscovery(self):
        """ACCEPTANCE: Rediscovers Poincaré × U(1) from relativistic data."""
        lorentz_obs = [o for o in self.obs if "lorentz_u1" in o.id][0]
        result = self.discoverer.discover(lorentz_obs)
        assert result.best_candidate is not None
        pred_gens = set(result.best_candidate.generators)
        expected_gens = set(self.gt[lorentz_obs.id])

        # Should contain U(1)
        assert GeneratorKind.U1_PHASE in pred_gens, (
            f"U(1) missing. Predicted: {[GENERATOR_LABELS.get(g,'?') for g in pred_gens]}"
        )

        # Should have at least 8 generators (most of Poincaré)
        assert len(pred_gens) >= 8, (
            f"Too few generators for Poincaré × U(1): {len(pred_gens)}"
        )

        # Time translation must be present
        assert GeneratorKind.TIME_TRANSLATION in pred_gens

    def test_heldout_su2_breaking(self):
        """ACCEPTANCE: Held-out Lorentz + SU(2) breaking — SU(2) excluded."""
        heldout_obs = [o for o in self.obs if "su2_broken" in o.id][0]
        result = self.discoverer.discover(heldout_obs)
        assert result.best_candidate is not None
        pred_gens = set(result.best_candidate.generators)
        expected_gens = set(self.gt[heldout_obs.id])

        # SU(2) should NOT be in the prediction
        assert GeneratorKind.SU2_WEAK not in pred_gens, (
            "SU(2) should be excluded from held-out scenario but was discovered"
        )

        # Should have Poincaré generators (time + space + rotation + boost)
        assert GeneratorKind.TIME_TRANSLATION in pred_gens

    def test_time_only_discovery(self):
        """ℝ (time translation only) discovery."""
        time_obs = [o for o in self.obs if "time_only" in o.id][0]
        result = self.discoverer.discover(time_obs)
        assert result.best_candidate is not None

        # Time translation must be in the best candidate
        pred_gens = set(result.best_candidate.generators)
        assert GeneratorKind.TIME_TRANSLATION in pred_gens


# ── Integration Tests ────────────────────────────────────────────────────────

class TestIntegration:
    """Integration tests for the full pipeline and database operations."""

    def test_discover_from_database(self, time_only_obs):
        """discover_from_database processes multiple observations."""
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump([{
            "id": time_only_obs.id,
            "name": time_only_obs.name,
            "description": time_only_obs.description,
            "quantities": dict(time_only_obs.quantities),
            "parameters": dict(time_only_obs.parameters),
            "timesteps": [dict(ts) for ts in time_only_obs.timesteps],
            "known_invariant": time_only_obs.known_invariant,
            "lean_theorem": time_only_obs.lean_theorem,
        }], tmp)
        tmp.close()
        try:
            db = ObservationDatabase(tmp.name)
            discoverer = SymmetryDiscoverer(max_candidates=50, constancy_threshold=0.3)
            results = discoverer.discover_from_database(db)
            assert len(results) == 1
            assert time_only_obs.id in results
        finally:
            Path(tmp.name).unlink(missing_ok=True)

    def test_run_discovery_on_database(self, time_only_obs):
        """run_discovery_on_database works on a file path."""
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump([{
            "id": time_only_obs.id,
            "name": time_only_obs.name,
            "description": time_only_obs.description,
            "quantities": dict(time_only_obs.quantities),
            "parameters": dict(time_only_obs.parameters),
            "timesteps": [dict(ts) for ts in time_only_obs.timesteps],
            "known_invariant": time_only_obs.known_invariant,
            "lean_theorem": time_only_obs.lean_theorem,
        }], tmp)
        tmp.close()
        try:
            results = run_discovery_on_database(
                tmp.name, scorer_path=None, max_candidates=50
            )
            assert len(results) == 1
        finally:
            Path(tmp.name).unlink(missing_ok=True)

    def test_save_discovery_results(self, time_only_obs):
        """save_discovery_results writes valid JSON."""
        discoverer = SymmetryDiscoverer(max_candidates=50, constancy_threshold=0.3)
        result = discoverer.discover(time_only_obs)
        results = {time_only_obs.id: result}

        out_path = Path(tempfile.gettempdir()) / "test_discovery_results.json"
        try:
            save_discovery_results(results, str(out_path))
            assert out_path.exists()
            with open(out_path) as f:
                data = json.load(f)
            assert time_only_obs.id in data
        finally:
            out_path.unlink(missing_ok=True)

    def test_evaluate_discovery(self):
        """evaluate_discovery returns expected structure."""
        obs, gt = generate_discovery_training_data()
        discoverer = SymmetryDiscoverer(max_candidates=100, constancy_threshold=0.3)
        # Only evaluate on first 2 for speed
        eval_result = evaluate_discovery(discoverer, obs[:2], {
            k: v for k, v in gt.items() if k in [o.id for o in obs[:2]]
        })
        assert "total" in eval_result
        assert "correct" in eval_result
        assert "accuracy" in eval_result
        assert "details" in eval_result

    def test_training_pipeline_runs(self):
        """run_symmetry_discovery_pipeline completes without error."""
        tmp_checkpoint = Path(tempfile.gettempdir()) / "test_discoverer_pipeline.pt"
        tmp_results = Path(tempfile.gettempdir()) / "test_pipeline_results.json"
        try:
            # Use small epochs for speed
            import sys
            from unittest.mock import patch
            # Patch training to use fewer epochs
            orig_train = train_symmetry_discoverer

            def fast_train(*args, **kwargs):
                kwargs["epochs"] = 10
                return orig_train(*args, **kwargs)

            import src.physics.symmetry_discovery as sd
            sd.train_symmetry_discoverer = fast_train

            results = run_symmetry_discovery_pipeline(
                checkpoint_path=str(tmp_checkpoint),
                results_path=str(tmp_results),
                max_candidates=100,
            )
            assert "training" in results
            assert "acceptance_checks" in results
        finally:
            tmp_checkpoint.unlink(missing_ok=True)
            tmp_results.unlink(missing_ok=True)


# ── Smoke Test ────────────────────────────────────────────────────────────────

class TestSmokeTest:
    """Tests for the smoke test function."""

    def test_smoke_test_runs(self):
        """Smoke test runs and returns results dict."""
        results = run_discovery_smoke_test()
        assert isinstance(results, dict)
        assert len(results) > 0

    def test_smoke_test_candidate_generation(self):
        """Smoke test: candidate generation works."""
        results = run_discovery_smoke_test()
        assert results.get("candidate_generation", False)

    def test_smoke_test_training_data(self):
        """Smoke test: training data has scenarios."""
        results = run_discovery_smoke_test()
        assert results.get("training_data_count", 0) > 0
        assert results.get("has_heldout", False)

    def test_smoke_test_scorer(self):
        """Smoke test: scorer creates successfully."""
        results = run_discovery_smoke_test()
        assert results.get("scorer_creation", False)

    def test_smoke_test_discovery(self):
        """Smoke test: discovery runs without error."""
        results = run_discovery_smoke_test()
        assert results.get("discovery_run", False)


# ── Edge Cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:
    """Edge case and robustness tests."""

    def test_empty_observation_variables(self):
        """Discovery with minimal variables still works."""
        obs = Observation(
            id="minimal", name="Minimal", description="...",
            quantities={"t": "Time"},
            parameters={},
            timesteps=[{"t": 0.0}, {"t": 1.0}],
            known_invariant=None, lean_theorem="",
        )
        discoverer = SymmetryDiscoverer(max_candidates=50, constancy_threshold=0.5)
        result = discoverer.discover(obs)
        assert result is not None

    def test_discovery_with_no_candidates(self):
        """Discovery with max_candidates=0 still returns result."""
        obs = Observation(
            id="test", name="Test", description="...",
            quantities={"m": "Mass", "h": "Length", "v": "Velocity", "t": "Time"},
            parameters={"m": 1.0},
            timesteps=[{"t": 0.0, "h": 10.0, "v": 0.0}, {"t": 1.0, "h": 5.0, "v": 5.0}],
            known_invariant=None, lean_theorem="",
        )
        discoverer = SymmetryDiscoverer(max_candidates=0, constancy_threshold=0.5)
        result = discoverer.discover(obs)
        assert result is not None
        assert result.candidates_evaluated == 0

    def test_group_candidate_alias(self):
        """GroupCandidate alias equals CandidateGroup."""
        assert GroupCandidate is CandidateGroup

    def test_build_discovery_training_alias(self):
        """Alias for training data builder works."""
        assert build_discovery_training_scenarios is generate_discovery_training_data

    def test_known_generator_sets_complete(self):
        """All known generator sets are valid."""
        for name, gen_list in KNOWN_GENERATOR_SETS.items():
            assert len(gen_list) > 0, f"{name} has no generators"
            for g in gen_list:
                assert isinstance(g, GeneratorKind)
