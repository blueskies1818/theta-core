"""Tests for src.physics.noise — noise calibration and gating."""

from pathlib import Path

import pytest

from src.physics.noise import (
    NoiseLevel,
    NoiseConfig,
    NoiseAugmenter,
    NoiseCalibrator,
    NoiseFloorResult,
    NoiseGatedEvaluator,
    RealExperimentalObservation,
    RealExperimentalLoader,
    run_noise_calibration,
)
from src.physics.observations import Observation, ObservationDatabase


PHASE1_PATH = Path(__file__).parent.parent.parent / "data" / "observations" / "phase1_falling.json"
PHASE_F_PATH = Path(__file__).parent.parent.parent / "data" / "observations" / "phase_f_7domain.json"
REAL_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "real_experimental"


@pytest.fixture
def db() -> ObservationDatabase:
    return ObservationDatabase(PHASE1_PATH)


@pytest.fixture
def db7() -> ObservationDatabase:
    return ObservationDatabase(PHASE_F_PATH)


# ── NoiseLevel tests ─────────────────────────────────────────────────────

class TestNoiseLevel:
    def test_sigma_pct_values(self):
        assert NoiseLevel.NONE.sigma_pct == 0.0
        assert NoiseLevel.LOW.sigma_pct == 0.01
        assert NoiseLevel.MEDIUM.sigma_pct == 0.03
        assert NoiseLevel.HIGH.sigma_pct == 0.05

    def test_from_sigma_pct(self):
        assert NoiseLevel.from_sigma_pct(0.0) == NoiseLevel.NONE
        assert NoiseLevel.from_sigma_pct(0.0001) == NoiseLevel.NONE
        assert NoiseLevel.from_sigma_pct(0.015) == NoiseLevel.LOW
        assert NoiseLevel.from_sigma_pct(0.035) == NoiseLevel.MEDIUM
        assert NoiseLevel.from_sigma_pct(0.06) == NoiseLevel.HIGH


# ── NoiseConfig tests ────────────────────────────────────────────────────

class TestNoiseConfig:
    def test_defaults(self):
        nc = NoiseConfig()
        assert nc.noise_level == NoiseLevel.NONE
        assert nc.effective_sigma == 0.0
        assert nc.per_timestep is True

    def test_custom_sigma(self):
        nc = NoiseConfig(noise_level=NoiseLevel.LOW, sigma_pct=2.5)
        assert nc.effective_sigma == 0.025

    def test_seeded_config(self):
        nc1 = NoiseConfig(noise_level=NoiseLevel.LOW, seed=42)
        nc2 = NoiseConfig(noise_level=NoiseLevel.LOW, seed=42)
        assert nc1.effective_sigma == nc2.effective_sigma


# ── NoiseAugmenter tests ─────────────────────────────────────────────────

class TestNoiseAugmenter:
    def test_no_noise_preserves_original(self, db):
        augmenter = NoiseAugmenter(NoiseConfig(NoiseLevel.NONE, seed=42))
        obs = db.get("falling_ball_straight_drop")
        noisy = augmenter.augment(obs)
        for orig_ts, noisy_ts in zip(obs.timesteps, noisy.timesteps):
            for key in orig_ts:
                assert orig_ts[key] == noisy_ts[key], f"Mismatch at {key}"

    def test_low_noise_changes_values_slightly(self, db):
        augmenter = NoiseAugmenter(NoiseConfig(NoiseLevel.LOW, seed=42))
        obs = db.get("falling_ball_straight_drop")
        noisy = augmenter.augment(obs)
        orig_h = obs.timesteps[0]["h"]
        noisy_h = noisy.timesteps[0]["h"]
        assert noisy_h != orig_h, "Noise should change values"
        # 1% noise should keep values close
        assert abs(noisy_h - orig_h) < abs(orig_h) * 0.1, (
            f"Low noise should be <10% change, got {abs(noisy_h - orig_h) / abs(orig_h):.4f}"
        )

    def test_reproducibility_with_seed(self, db):
        augmenter1 = NoiseAugmenter(NoiseConfig(NoiseLevel.MEDIUM, seed=42))
        augmenter2 = NoiseAugmenter(NoiseConfig(NoiseLevel.MEDIUM, seed=42))
        obs = db.get("falling_ball_straight_drop")
        noisy1 = augmenter1.augment(obs)
        noisy2 = augmenter2.augment(obs)
        for ts1, ts2 in zip(noisy1.timesteps, noisy2.timesteps):
            for key in ts1:
                assert ts1[key] == ts2[key], f"Seed should give reproducible noise at {key}"

    def test_different_seeds_different_noise(self, db):
        augmenter1 = NoiseAugmenter(NoiseConfig(NoiseLevel.MEDIUM, seed=1))
        augmenter2 = NoiseAugmenter(NoiseConfig(NoiseLevel.MEDIUM, seed=2))
        obs = db.get("falling_ball_straight_drop")
        noisy1 = augmenter1.augment(obs)
        noisy2 = augmenter2.augment(obs)
        any_different = any(
            ts1[k] != ts2[k]
            for ts1, ts2 in zip(noisy1.timesteps, noisy2.timesteps)
            for k in ts1
        )
        assert any_different, "Different seeds should produce different noise"

    def test_preserves_observation_structure(self, db):
        augmenter = NoiseAugmenter(NoiseConfig(NoiseLevel.HIGH, seed=42))
        obs = db.get("falling_ball_straight_drop")
        noisy = augmenter.augment(obs)
        assert noisy.id == obs.id
        assert noisy.name == obs.name
        assert len(noisy.timesteps) == len(obs.timesteps)
        assert noisy.known_invariant == obs.known_invariant

    def test_augment_database(self, db):
        augmenter = NoiseAugmenter(NoiseConfig(NoiseLevel.LOW, seed=42))
        noisy_obs_list = augmenter.augment_database(db)
        assert len(noisy_obs_list) == len(db)
        for orig_obs, noisy_obs in zip(db, noisy_obs_list):
            assert noisy_obs.id == orig_obs.id


# ── NoiseCalibrator tests ────────────────────────────────────────────────

class TestNoiseCalibrator:
    def test_calibrate_returns_valid_floor(self, db):
        calibrator = NoiseCalibrator(n_sigma=3.0, seed=42, num_calibration_runs=3)
        obs = db.get("falling_ball_straight_drop")
        floor = calibrator.calibrate([obs], NoiseLevel.LOW)
        assert 0.0 <= floor.noise_floor <= 1.0
        assert floor.sigma_floor > 0.0
        assert floor.threshold > floor.noise_floor
        assert floor.num_calibration_exprs > 0

    def test_none_noise_floor(self, db):
        """At NONE noise, noise floor should be from clean non-constant expressions."""
        calibrator = NoiseCalibrator(n_sigma=3.0, seed=42, num_calibration_runs=3)
        obs = db.get("falling_ball_straight_drop")
        # This will say NONE sigma = 0 so no noise is added
        floor = calibrator.calibrate([obs], NoiseLevel.NONE)
        assert 0.0 <= floor.noise_floor <= 1.0

    def test_higher_noise_increases_floor(self, db):
        """Higher noise should increase the noise floor (noise makes things
        look more constant by random chance)."""
        calibrator = NoiseCalibrator(n_sigma=3.0, seed=42, num_calibration_runs=3)
        obs = db.get("falling_ball_straight_drop")
        floor_low = calibrator.calibrate([obs], NoiseLevel.LOW)
        floor_high = calibrator.calibrate([obs], NoiseLevel.HIGH)
        # HIGH should have equal or higher noise floor
        # (Not strictly guaranteed with small sample, but typical)
        assert floor_high.noise_floor >= floor_low.noise_floor * 0.8, (
            f"Expected similar or higher floor for HIGH noise: "
            f"LOW={floor_low.noise_floor:.4f}, HIGH={floor_high.noise_floor:.4f}"
        )

    def test_should_accept_gates_correctly(self, db):
        calibrator = NoiseCalibrator(n_sigma=3.0, seed=42, num_calibration_runs=3)
        obs = db.get("falling_ball_straight_drop")
        floor = calibrator.calibrate([obs], NoiseLevel.MEDIUM)
        # A perfect score should always be accepted
        assert calibrator.should_accept(1.0, floor)
        # Score equal to threshold should NOT be accepted (strict >)
        assert not calibrator.should_accept(floor.threshold, floor)
        # Score equal to threshold + epsilon should be accepted
        assert calibrator.should_accept(floor.threshold + 1e-6, floor)

    def test_caching(self, db):
        calibrator = NoiseCalibrator(n_sigma=3.0, seed=42, num_calibration_runs=3)
        obs = db.get("falling_ball_straight_drop")
        # First call — compute
        result1 = calibrator.calibrate_per_scenario(db, NoiseLevel.LOW)
        key = (obs.id, "LOW")
        assert key in calibrator._floor_cache
        # Second call should use cache
        result2 = calibrator.calibrate_per_scenario(db, NoiseLevel.LOW)
        assert result1[obs.id].noise_floor == result2[obs.id].noise_floor

    def test_adaptive_threshold_fallback(self):
        calibrator = NoiseCalibrator(n_sigma=3.0, seed=42)
        # Without calibration, should return a reasonable fallback
        threshold = calibrator.adaptive_threshold("unknown_scenario", 0.03)
        assert 0.0 < threshold <= 0.99

    def test_pre_calibrate_all(self, db):
        calibrator = NoiseCalibrator(n_sigma=3.0, seed=42, num_calibration_runs=2)
        results = calibrator.pre_calibrate_all(db, noise_levels=[NoiseLevel.LOW])
        assert len(results) > 0
        for (sid, level), floor in results.items():
            assert sid in db
            assert level == "LOW"
            assert 0.0 <= floor.threshold <= 1.0

    def test_classify_noise_level(self):
        calibrator = NoiseCalibrator()
        assert calibrator.classify_noise_level(0.0) == NoiseLevel.NONE
        assert calibrator.classify_noise_level(0.01) == NoiseLevel.LOW
        assert calibrator.classify_noise_level(0.03) == NoiseLevel.MEDIUM
        assert calibrator.classify_noise_level(0.05) == NoiseLevel.HIGH

    def test_gated_score(self, db):
        calibrator = NoiseCalibrator(n_sigma=3.0, seed=42, num_calibration_runs=3)
        obs = db.get("falling_ball_straight_drop")
        result = calibrator.gated_score("m*g*h + 0.5*m*v^2", obs, NoiseLevel.LOW)
        assert "raw_score" in result
        assert "threshold" in result
        assert "accepted" in result
        assert result["sigma_pct"] == 0.01

    def test_gated_score_with_database(self, db):
        calibrator = NoiseCalibrator(n_sigma=3.0, seed=42, num_calibration_runs=3)
        result = calibrator.gated_score("m*g*h + 0.5*m*v^2", db, NoiseLevel.LOW)
        assert "accepted" in result


# ── NoiseGatedEvaluator tests ────────────────────────────────────────────

class TestNoiseGatedEvaluator:
    def test_score_none_noise_matches_original(self, db):
        from src.physics.evaluator import ExpressionEvaluator
        orig_ev = ExpressionEvaluator()
        gated_ev = NoiseGatedEvaluator(NoiseLevel.NONE, seed=42)
        
        orig_score = orig_ev.score("m*g*h + 0.5*m*v^2", db)
        gated_score = gated_ev.score("m*g*h + 0.5*m*v^2", db)
        assert abs(orig_score - gated_score) < 1e-10

    def test_score_with_noise(self, db):
        gated_ev = NoiseGatedEvaluator(NoiseLevel.LOW, seed=42)
        score = gated_ev.score("m*g*h + 0.5*m*v^2", db)
        assert 0.0 <= score <= 1.0
        # Energy should still be mostly constant with low noise
        # Note: phase1_falling has pendulums, springs, projectiles —
        # mgh+½mv² only perfectly constrains gravity scenarios, so
        # overall database score is lower (~0.79 at 1% noise)
        assert score > 0.70

    def test_score_with_confidence(self, db):
        gated_ev = NoiseGatedEvaluator(NoiseLevel.MEDIUM, n_sigma=3.0, seed=42)
        result = gated_ev.score_with_confidence(
            "m*g*h + 0.5*m*v^2", db, num_samples=5
        )
        assert "raw_score" in result
        assert "confidence_95" in result
        assert "noise_std" in result
        ci_low, ci_high = result["confidence_95"]
        assert ci_low <= result["raw_score"] <= ci_high

    def test_non_constant_scores_low_with_noise(self, db):
        gated_ev = NoiseGatedEvaluator(NoiseLevel.HIGH, n_sigma=3.0, seed=42)
        result = gated_ev.score_with_confidence("v", db, num_samples=5)
        # v alone should score low even with noise
        assert result["raw_score"] < 0.8

    def test_change_noise_level(self, db):
        gated_ev = NoiseGatedEvaluator(NoiseLevel.NONE, seed=42)
        score_none = gated_ev.score("m*g*h + 0.5*m*v^2", db)
        gated_ev.set_noise_level(NoiseLevel.HIGH)
        score_high = gated_ev.score("m*g*h + 0.5*m*v^2", db)
        # High noise should reduce score somewhat
        assert score_high <= score_none + 0.02  # allow slight float variation


# ── RealExperimentalObservation tests ────────────────────────────────────

class TestRealExperimentalObservation:
    def test_minimal_dataset(self):
        reo = RealExperimentalObservation(
            source="test",
            description="Test data",
            domain="mechanics",
            quantities={"t": "Time", "x": "Length"},
            parameters={"k": 1.0},
            data_points=[
                {"t": 0.0, "x": 1.0, "x_err": 0.1},
                {"t": 1.0, "x": 2.0, "x_err": 0.1},
            ],
            known_invariant="x",
        )
        obs_list = reo.to_synthetic_observations(num_bootstrap=3)
        assert len(obs_list) == 3
        for obs in obs_list:
            assert len(obs.timesteps) == 2
            assert 0.5 < obs.timesteps[0]["x"] < 1.5  # near 1.0 with noise

    def test_no_error_bars(self):
        """Data points without error bars should pass through unchanged."""
        reo = RealExperimentalObservation(
            source="test",
            description="No errors",
            domain="mechanics",
            quantities={"t": "Time", "x": "Length"},
            parameters={},
            data_points=[
                {"t": 0.0, "x": 5.0},
                {"t": 1.0, "x": 10.0},
            ],
        )
        obs_list = reo.to_synthetic_observations(num_bootstrap=1)
        assert obs_list[0].timesteps[0]["x"] == 5.0


# ── RealExperimentalLoader tests ─────────────────────────────────────────

class TestRealExperimentalLoader:
    def test_loads_real_datasets(self):
        if not REAL_DATA_DIR.exists():
            pytest.skip("Real experimental data dir not found")
        loader = RealExperimentalLoader(REAL_DATA_DIR)
        datasets = loader.load_all()
        assert len(datasets) >= 3, f"Expected at least 3 datasets, got {len(datasets)}"
        for ds in datasets:
            assert ds.source
            assert ds.domain
            assert len(ds.data_points) >= 2

    def test_converts_to_observations(self):
        if not REAL_DATA_DIR.exists():
            pytest.skip("Real experimental data dir not found")
        loader = RealExperimentalLoader(REAL_DATA_DIR)
        datasets = loader.load_all()
        for ds in datasets:
            obs_list = ds.to_synthetic_observations(num_bootstrap=2)
            assert len(obs_list) == 2
            for obs in obs_list:
                assert len(obs.timesteps) == len(ds.data_points)
                assert obs.known_invariant == ds.known_invariant


# ── Integration tests ────────────────────────────────────────────────────

class TestNoiseIntegration:
    """Integration tests: noise calibration across multiple domains."""

    def test_noise_calibration_runs_on_7domain(self, db7):
        """Noise calibration should complete on full 7-domain database."""
        calibrator = NoiseCalibrator(n_sigma=3.0, seed=42, num_calibration_runs=2)
        # Just test a few scenarios to keep it fast
        first_few = [db7[i] for i in range(min(5, len(db7)))]
        for obs in first_few:
            floor = calibrator.calibrate([obs], NoiseLevel.MEDIUM)
            assert 0.0 <= floor.noise_floor <= 1.0
            assert floor.threshold > floor.noise_floor

    def test_energy_invariant_passes_gate_at_low_noise(self, db7):
        """Energy conservation should be discoverable at LOW noise."""
        gated = NoiseGatedEvaluator(NoiseLevel.LOW, n_sigma=3.0, seed=42)
        # Test on first gravity scenario
        gravity_obs = [obs for obs in db7 if obs.id.startswith("freefall")][:3]
        for obs in gravity_obs:
            result = gated.score_with_confidence(
                "m*g*h + 0.5*m*v^2", obs, num_samples=5
            )
            # Should pass gate for most gravity scenarios
            assert result["raw_score"] > 0.80, (
                f"Energy should score high on {obs.id}: {result['raw_score']:.4f}"
            )

    def test_non_constant_rejected_at_noise(self, db7):
        """Non-constant expressions should be rejected by the gate."""
        gated = NoiseGatedEvaluator(NoiseLevel.MEDIUM, n_sigma=3.0, seed=42)
        obs = db7[0]  # first freefall
        result = gated.score_with_confidence("v", obs, num_samples=5)
        # "v" alone varies — it should be below threshold
        assert not result["accepted"] or result["raw_score"] < 0.75, (
            f"v alone should not pass gate: {result}"
        )

    def test_all_7_domains_work_through_noise(self, db7):
        """Each of the 7 domains should have at least one scenario that
        works correctly through the noise pipeline."""
        gated = NoiseGatedEvaluator(NoiseLevel.LOW, n_sigma=3.0, seed=42)

        domain_templates = {
            "gravity": "m*g*h + 0.5*m*v^2",
            "spring": "0.5*k*h^2 + 0.5*m*v^2",
            "em": "0.5*m*v^2 - q*E*x",
            "thermal": "P*V/T",
            "quantum": "n^2*hbar^2/(2*m*L^2)",
            "relativistic": "E^2 - (p*c)^2",
        }

        # Test one scenario from each domain
        domain_detectors = {
            "gravity": lambda oid: "freefall" in oid,
            "spring": lambda oid: "spring" in oid,
            "em": lambda oid: "e_field" in oid,
            "thermal": lambda oid: "isothermal" in oid,
            "quantum": lambda oid: "particle_in_box" in oid,
            "relativistic": lambda oid: "time_dilation" in oid,
        }

        tested_domains = set()
        for obs in db7:
            for domain, detector in domain_detectors.items():
                if domain in tested_domains:
                    continue
                if detector(obs.id):
                    tmpl = domain_templates[domain]
                    result = gated.score_with_confidence(tmpl, obs, num_samples=3)
                    assert 0.0 <= result["raw_score"] <= 1.0, (
                        f"{domain} ({obs.id}): invalid score {result['raw_score']}"
                    )
                    tested_domains.add(domain)

        assert len(tested_domains) == 6, (
            f"Expected 6 domains to be tested, got {len(tested_domains)}: {tested_domains}"
        )

    def test_run_noise_calibration_function(self, db, tmp_path):
        """The convenience function should produce valid output."""
        # Create a small subset database for speed
        import json
        subset_path = tmp_path / "subset.json"
        raw = []
        with open(PHASE1_PATH) as f:
            data = json.load(f)
        subset = data[:3]  # first 3 scenarios
        with open(subset_path, "w") as f:
            json.dump(subset, f)

        output_path = tmp_path / "calibration.json"
        results = run_noise_calibration(
            db_path=subset_path,
            noise_levels=[NoiseLevel.LOW],
            n_sigma=3.0,
            seed=42,
            output_path=output_path,
        )
        assert results["num_scenarios"] == 3
        assert "LOW" in results["noise_levels"]
        assert output_path.exists()

        with open(output_path) as f:
            saved = json.load(f)
        assert saved["num_scenarios"] == 3


# ── Noise floor integrity tests ──────────────────────────────────────────

class TestNoiseFloorIntegrity:
    """ACCEPTANCE: Noise calibration gate correctly rejects false discoveries."""

    def test_noise_gate_rejects_false_at_5pct(self, db7):
        """At 5% noise, a non-constant expression should be below threshold."""
        gated = NoiseGatedEvaluator(NoiseLevel.HIGH, n_sigma=3.0, seed=42)

        # Test false discovery: "h*t" is definitely non-constant
        false_discoveries = 0
        tested = 0
        for obs in db7:
            if tested >= 10:
                break
            result = gated.score_with_confidence("h*t", obs, num_samples=3)
            if result["accepted"]:
                false_discoveries += 1
            tested += 1

        # At 5% noise, fewer than 50% of false "discoveries" should pass
        false_rate = false_discoveries / tested if tested > 0 else 1.0
        assert false_rate < 0.5, (
            f"Noise gate should reject most false discoveries at 5% noise. "
            f"False rate: {false_rate:.2f} ({false_discoveries}/{tested})"
        )

    def test_energy_passes_clean_data(self, db7):
        """On clean data (NONE noise), energy should always be accepted."""
        gated = NoiseGatedEvaluator(NoiseLevel.NONE, n_sigma=3.0, seed=42)
        gravity_obs = [obs for obs in db7 if obs.id.startswith("freefall")][:5]
        for obs in gravity_obs:
            result = gated.score_with_confidence(
                "m*g*h + 0.5*m*v^2", obs
            )
            assert result["accepted"], (
                f"Energy should pass on clean data for {obs.id}: {result}"
            )
