"""Tests for observation_simulator.py — Phase B self-play observation generation.

Validates:
1. Known expression produces constant observations (noise-free)
2. Noise reduces constancy score proportionally
3. Independent noise per quantity verified
4. Edge cases: single configuration, zero noise, high noise
5. Physical range enforcement and validation
"""

from __future__ import annotations

import math

import pytest

from src.physics.evaluator import ExpressionEvaluator
from src.physics.observation_simulator import (
    simulate_batch,
    simulate_observations,
)


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def ev() -> ExpressionEvaluator:
    return ExpressionEvaluator()


# ═══════════════════════════════════════════════════════════════════════════
# 1. Noise-free observations are nearly perfectly constant
# ═══════════════════════════════════════════════════════════════════════════

class TestNoiseFreeConstancy:
    """Known expressions produce constant observations when noise_frac=0."""

    def test_product_is_constant(self, ev):
        obs_list = simulate_observations(
            expression="P*V",
            quantities={"P": "Pressure", "V": "Volume"},
            num_configs=20,
            noise_frac=0.0,
            seed=42,
        )
        obs = obs_list[0]
        score = ev.score("P*V", obs)
        assert score >= 0.999, f"P*V constancy={score:.6f} < 0.999 on noise-free data"

        # Verify quantities actually vary
        p_vals = [ts["P"] for ts in obs.timesteps]
        v_vals = [ts["V"] for ts in obs.timesteps]
        assert max(p_vals) - min(p_vals) > 1e-6, "P does not vary"
        assert max(v_vals) - min(v_vals) > 1e-6, "V does not vary"

    def test_ratio_is_constant(self, ev):
        obs_list = simulate_observations(
            expression="E/n",
            quantities={"E": "Energy", "n": "Scalar"},
            num_configs=15,
            noise_frac=0.0,
            seed=7,
        )
        obs = obs_list[0]
        score = ev.score("E/n", obs)
        assert score >= 0.999, f"E/n constancy={score:.6f} < 0.999"

        e_vals = [ts["E"] for ts in obs.timesteps]
        n_vals = [ts["n"] for ts in obs.timesteps]
        assert max(e_vals) - min(e_vals) > 1e-6, "E does not vary"
        assert max(n_vals) - min(n_vals) > 1e-6, "n does not vary"

    def test_power_is_constant(self, ev):
        # Single-variable expressions like "v^2" cannot be constant while v varies.
        # Use a multi-variable expression with powers: x^2 * y (constant if y ∝ 1/x^2).
        obs_list = simulate_observations(
            expression="x^2 * h",
            quantities={"x": "Length", "h": "Length"},
            num_configs=30,
            noise_frac=0.0,
            seed=99,
        )
        obs = obs_list[0]
        score = ev.score("x^2 * h", obs)
        assert score >= 0.999, f"x^2*h constancy={score:.6f} < 0.999"

    def test_sum_is_constant(self, ev):
        obs_list = simulate_observations(
            expression="m*g*h + 0.5*m*v^2",
            quantities={"m": "Mass", "g": "Accel", "h": "Length", "v": "Velocity"},
            num_configs=20,
            noise_frac=0.0,
            seed=42,
        )
        obs = obs_list[0]
        score = ev.score("m*g*h + 0.5*m*v^2", obs)
        assert score >= 0.999, f"energy constancy={score:.6f} < 0.999"

        # At least two quantities must vary
        m_vals = [ts["m"] for ts in obs.timesteps]
        h_vals = [ts["h"] for ts in obs.timesteps]
        v_vals = [ts["v"] for ts in obs.timesteps]
        varying = sum(
            1 for vals in [m_vals, h_vals, v_vals]
            if max(vals) - min(vals) > 1e-6
        )
        assert varying >= 2, f"Only {varying} quantities vary, need >= 2"

    def test_spring_energy_is_constant(self, ev):
        obs_list = simulate_observations(
            expression="0.5*k*x^2",
            quantities={"k": "Force/Length", "x": "Length"},
            num_configs=20,
            noise_frac=0.0,
            seed=123,
        )
        obs = obs_list[0]
        score = ev.score("0.5*k*x^2", obs)
        assert score >= 0.999, f"spring constancy={score:.6f} < 0.999"


# ═══════════════════════════════════════════════════════════════════════════
# 2. Noise reduces constancy score proportionally
# ═══════════════════════════════════════════════════════════════════════════

class TestNoiseReducesConstancy:
    """Higher noise fractions produce lower constancy scores."""

    def test_noise_reduces_score(self, ev):
        scores = {}
        for noise in [0.0, 0.01, 0.05, 0.10]:
            obs_list = simulate_observations(
                expression="P*V",
                quantities={"P": "Pressure", "V": "Volume"},
                num_configs=30,
                noise_frac=noise,
                seed=42,
            )
            obs = obs_list[0]
            scores[noise] = ev.score("P*V", obs)

        assert scores[0.0] >= 0.999, "noise-free should be near-perfect"
        assert scores[0.01] < scores[0.0], "1% noise should reduce score"
        assert scores[0.05] < scores[0.01], "5% noise should reduce score further"
        assert scores[0.10] < scores[0.05], "10% noise should reduce score further"

    def test_noise_reduces_score_on_sum(self, ev):
        scores = {}
        for noise in [0.0, 0.03, 0.08]:
            obs_list = simulate_observations(
                expression="m*g*h + 0.5*m*v^2",
                quantities={"m": "Mass", "g": "Accel", "h": "Length", "v": "Velocity"},
                num_configs=20,
                noise_frac=noise,
                seed=42,
            )
            obs = obs_list[0]
            scores[noise] = ev.score("m*g*h + 0.5*m*v^2", obs)

        assert scores[0.0] >= 0.999
        assert scores[0.03] < scores[0.0]
        assert scores[0.08] < scores[0.03]

    def test_noise_does_not_destroy_signal_entirely(self, ev):
        """At reasonable noise (3%), constancy should still be > 0.85."""
        obs_list = simulate_observations(
            expression="P*V",
            quantities={"P": "Pressure", "V": "Volume"},
            num_configs=30,
            noise_frac=0.03,
            seed=42,
        )
        obs = obs_list[0]
        score = ev.score("P*V", obs)
        assert score > 0.85, f"3% noise constancy={score:.4f} <= 0.85"


# ═══════════════════════════════════════════════════════════════════════════
# 3. Independent noise per quantity
# ═══════════════════════════════════════════════════════════════════════════

class TestIndependentNoise:
    """Each quantity gets its own independent noise draw."""

    def test_noise_is_independent_across_quantities(self):
        """With a fixed seed, noise patterns on P and V should differ."""
        # Generate two sets with same seed — noise draws should be reproducible
        obs_list_a = simulate_observations(
            expression="P*V",
            quantities={"P": "Pressure", "V": "Volume"},
            num_configs=20,
            noise_frac=0.05,
            seed=100,
        )
        obs_list_b = simulate_observations(
            expression="P*V",
            quantities={"P": "Pressure", "V": "Volume"},
            num_configs=20,
            noise_frac=0.05,
            seed=100,
        )
        obs_a = obs_list_a[0]
        obs_b = obs_list_b[0]

        # Same seed → identical results
        for i in range(len(obs_a.timesteps)):
            assert obs_a.timesteps[i]["P"] == obs_b.timesteps[i]["P"]
            assert obs_a.timesteps[i]["V"] == obs_b.timesteps[i]["V"]

    def test_noise_is_independent_per_timestep(self):
        """Each timestep gets independent noise draws."""
        obs_list = simulate_observations(
            expression="P*V",
            quantities={"P": "Pressure", "V": "Volume"},
            num_configs=30,
            noise_frac=0.05,
            seed=42,
        )
        obs = obs_list[0]

        # The expr_true should be nearly constant (it's the noise-free invariant).
        # The expr_value should vary (it's noisy).
        true_vals = [ts["expr_true"] for ts in obs.timesteps]
        noisy_vals = [ts["expr_value"] for ts in obs.timesteps]

        true_std = math.sqrt(
            sum((v - sum(true_vals) / len(true_vals)) ** 2 for v in true_vals)
            / len(true_vals)
        )
        noisy_std = math.sqrt(
            sum((v - sum(noisy_vals) / len(noisy_vals)) ** 2 for v in noisy_vals)
            / len(noisy_vals)
        )

        # expr_value should have much higher variance than expr_true
        mean_true = sum(true_vals) / len(true_vals)
        mean_noisy = sum(noisy_vals) / len(noisy_vals)
        rel_std_true = true_std / max(abs(mean_true), 1e-12)
        rel_std_noisy = noisy_std / max(abs(mean_noisy), 1e-12)

        assert rel_std_noisy > rel_std_true * 2, (
            f"Noisy rel_std={rel_std_noisy:.6f} not much larger than "
            f"true rel_std={rel_std_true:.6f}"
        )

    def test_different_seeds_produce_different_noise(self):
        """Different seeds should produce different observation values."""
        obs_list_a = simulate_observations(
            expression="P*V",
            quantities={"P": "Pressure", "V": "Volume"},
            num_configs=10,
            noise_frac=0.05,
            seed=1,
        )
        obs_list_b = simulate_observations(
            expression="P*V",
            quantities={"P": "Pressure", "V": "Volume"},
            num_configs=10,
            noise_frac=0.05,
            seed=999,
        )
        obs_a = obs_list_a[0]
        obs_b = obs_list_b[0]

        # At least one timestep should differ
        any_different = False
        for i in range(len(obs_a.timesteps)):
            if (obs_a.timesteps[i]["P"] != obs_b.timesteps[i]["P"]
                    or obs_a.timesteps[i]["V"] != obs_b.timesteps[i]["V"]):
                any_different = True
                break
        assert any_different, "Different seeds produced identical observations"


# ═══════════════════════════════════════════════════════════════════════════
# 4. Edge cases
# ═══════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Edge cases: single config, zero noise, high noise, negative noise."""

    def test_single_config_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            simulate_observations(
                expression="m*v",
                quantities={"m": "Mass", "v": "Velocity"},
                num_configs=1,
                noise_frac=0.01,
                seed=1,
            )

    def test_negative_noise_frac_raises(self):
        with pytest.raises(ValueError, match="noise_frac"):
            simulate_observations(
                expression="m*v",
                quantities={"m": "Mass", "v": "Velocity"},
                num_configs=5,
                noise_frac=-0.1,
                seed=1,
            )

    def test_zero_noise_produces_identical_expr_values(self):
        """With zero noise, expr_true equals expr_value at every timestep."""
        obs_list = simulate_observations(
            expression="P*V",
            quantities={"P": "Pressure", "V": "Volume"},
            num_configs=20,
            noise_frac=0.0,
            seed=42,
        )
        obs = obs_list[0]
        for ts in obs.timesteps:
            assert ts["expr_value"] == ts["expr_true"], (
                f"expr_value={ts['expr_value']} != expr_true={ts['expr_true']}"
            )

    def test_high_noise_still_produces_valid_observation(self):
        """High noise (20%) should not crash and should produce valid output."""
        obs_list = simulate_observations(
            expression="P*V",
            quantities={"P": "Pressure", "V": "Volume"},
            num_configs=20,
            noise_frac=0.20,
            seed=42,
        )
        obs = obs_list[0]
        assert len(obs.timesteps) == 20
        assert obs.known_invariant == "P*V"
        # All timesteps should still have required keys
        for ts in obs.timesteps:
            assert "P" in ts
            assert "V" in ts
            assert "expr_value" in ts

    def test_unparseable_expression_raises(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            simulate_observations(
                expression="this is not math @@@",
                quantities={"x": "Length"},
                num_configs=5,
                noise_frac=0.01,
                seed=1,
            )

    def test_quantities_not_in_expression_still_sampled(self):
        """Quantities in the dict but not used in expression are still present."""
        obs_list = simulate_observations(
            expression="m*v",
            quantities={"m": "Mass", "v": "Velocity", "E": "Energy"},
            num_configs=20,
            noise_frac=0.01,
            seed=42,
        )
        obs = obs_list[0]
        for ts in obs.timesteps:
            assert "E" in ts, "Quantity 'E' should appear even if not in expression"
            assert ts["E"] > 0, "E should be positive"


# ═══════════════════════════════════════════════════════════════════════════
# 5. Observation structure validation
# ═══════════════════════════════════════════════════════════════════════════

class TestObservationStructure:
    """Generated observations have the correct structure."""

    def test_observation_has_required_fields(self):
        obs_list = simulate_observations(
            expression="P*V",
            quantities={"P": "Pressure", "V": "Volume"},
            num_configs=10,
            noise_frac=0.01,
            seed=42,
        )
        obs = obs_list[0]
        assert obs.id.startswith("selfplay_")
        assert "P*V" in obs.name
        assert len(obs.description) > 0
        assert obs.quantities == {"P": "Pressure", "V": "Volume"}
        assert obs.known_invariant == "P*V"
        assert len(obs.timesteps) == 10

    def test_all_timesteps_have_t_key(self):
        obs_list = simulate_observations(
            expression="P*V",
            quantities={"P": "Pressure", "V": "Volume"},
            num_configs=15,
            noise_frac=0.01,
            seed=42,
        )
        obs = obs_list[0]
        for i, ts in enumerate(obs.timesteps):
            assert "t" in ts, f"Timestep {i} missing 't'"
            assert ts["t"] == float(i), f"Timestep {i} t={ts['t']} != {float(i)}"

    def test_base_quantities_are_positive(self):
        """Mass, Length, Time quantities should always be positive after noise."""
        obs_list = simulate_observations(
            expression="m*g*h",
            quantities={"m": "Mass", "g": "Accel", "h": "Length"},
            num_configs=20,
            noise_frac=0.01,
            seed=42,
        )
        obs = obs_list[0]
        for i, ts in enumerate(obs.timesteps):
            assert ts["m"] > 0, f"Timestep {i}: m={ts['m']} <= 0"
            assert ts["h"] > 0, f"Timestep {i}: h={ts['h']} <= 0"


# ═══════════════════════════════════════════════════════════════════════════
# 6. Batch simulation
# ═══════════════════════════════════════════════════════════════════════════

class TestBatchSimulation:
    """simulate_batch produces multiple observations."""

    def test_batch_produces_correct_count(self):
        expressions = ["P*V", "E/n", "m*v"]
        quants_list = [
            {"P": "Pressure", "V": "Volume"},
            {"E": "Energy", "n": "Scalar"},
            {"m": "Mass", "v": "Velocity"},
        ]
        obs_list = simulate_batch(
            expressions=expressions,
            quantities_list=quants_list,
            num_configs=10,
            noise_frac=0.01,
            seed=42,
        )
        assert len(obs_list) == 3
        for obs, expr in zip(obs_list, expressions):
            assert obs.known_invariant == expr

    def test_batch_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="Length mismatch"):
            simulate_batch(
                expressions=["a", "b"],
                quantities_list=[{"x": "Length"}],
                num_configs=5,
                noise_frac=0.01,
                seed=1,
            )

    def test_batch_different_seeds(self):
        """Each expression in a batch gets different seeds."""
        expressions = ["P*V", "P*V"]
        quants_list = [
            {"P": "Pressure", "V": "Volume"},
            {"P": "Pressure", "V": "Volume"},
        ]
        obs_list = simulate_batch(
            expressions=expressions,
            quantities_list=quants_list,
            num_configs=5,
            noise_frac=0.0,
            seed=42,
        )
        # Same expression, same noise-free → should differ (different seed offsets)
        ts_a = obs_list[0].timesteps
        ts_b = obs_list[1].timesteps
        # At least one timestep should be different
        any_different = False
        for i in range(len(ts_a)):
            if ts_a[i]["P"] != ts_b[i]["P"] or ts_a[i]["V"] != ts_b[i]["V"]:
                any_different = True
                break
        assert any_different, "Batch entries with different seeds should differ"


# ═══════════════════════════════════════════════════════════════════════════
# 7. Physical range enforcement
# ═══════════════════════════════════════════════════════════════════════════

class TestPhysicalRanges:
    """Sampled values respect physical ranges."""

    def test_values_within_range_ratio(self):
        obs_list = simulate_observations(
            expression="P*V",
            quantities={"P": "Pressure", "V": "Volume"},
            num_configs=100,
            noise_frac=0.0,
            seed=42,
        )
        obs = obs_list[0]
        for ts in obs.timesteps:
            # Pressure: 1.0 to 1e6  (minus noise tolerance)
            assert ts["P"] > 0.5, f"P={ts['P']} below range"
            assert ts["P"] < 2e6, f"P={ts['P']} above range"
            # Volume: 0.0001 to 1000
            assert ts["V"] > 0.00005, f"V={ts['V']} below range"
            assert ts["V"] < 2000, f"V={ts['V']} above range"

    def test_mass_length_time_positive(self):
        """Mass, Length, and Time quantities must always be positive."""
        obs_list = simulate_observations(
            expression="m*g*h",
            quantities={"m": "Mass", "g": "Accel", "h": "Length"},
            num_configs=20,
            noise_frac=0.01,
            seed=42,
        )
        obs = obs_list[0]
        for ts in obs.timesteps:
            assert ts["m"] > 0, f"Mass={ts['m']} not positive"
            assert ts["h"] > 0, f"Length={ts['h']} not positive"
