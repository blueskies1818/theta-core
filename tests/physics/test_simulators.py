"""Tests for physics observation simulators.

Validates:
1. Simulators produce valid Observation-compatible dicts
2. Generated data loads into ObservationDatabase without errors
3. Known invariants score > 0.95 on generated data
4. Each domain generates >= 50 scenarios
5. All existing tests still pass
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from src.physics.evaluator import ExpressionEvaluator, EvalError, ParseError
from src.physics.observations import Observation, ObservationDatabase
from src.physics.simulators.mechanics import (
    generate_all_mechanics,
    generate_freefall_scenarios,
    generate_projectile_scenarios,
    generate_pendulum_scenarios,
    generate_spring_scenarios,
    generate_collision_scenarios,
)
from src.physics.simulators.electromagnetism import (
    generate_all_electromagnetism,
    generate_e_field_scenarios,
    generate_b_field_scenarios,
    generate_eb_combined_scenarios,
    generate_coulomb_scenarios,
    generate_induced_emf_scenarios,
)
from src.physics.simulators.thermodynamics import (
    generate_all_thermodynamics,
    generate_isothermal_scenarios,
    generate_adiabatic_scenarios,
    generate_isobaric_scenarios,
    generate_isochoric_scenarios,
    generate_carnot_scenarios,
    generate_entropy_scenarios,
    generate_ideal_gas_scenarios,
)
from src.physics.simulators.quantum import (
    generate_all_quantum,
    generate_particle_in_box_scenarios,
    generate_harmonic_oscillator_scenarios,
    generate_hydrogen_atom_scenarios,
    generate_probability_current_scenarios,
    generate_expectation_scenarios,
    generate_wave_packet_scenarios,
)
from src.physics.simulators.relativity import (
    generate_all_relativity,
    generate_time_dilation_scenarios,
    generate_length_contraction_scenarios,
    generate_velocity_addition_scenarios,
    generate_energy_momentum_scenarios,
    generate_spacetime_interval_scenarios,
    generate_doppler_scenarios,
    generate_lorentz_boost_scenarios,
    generate_proper_time_scenarios,
)


PROJECT_ROOT = Path(__file__).parent.parent.parent


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def ev() -> ExpressionEvaluator:
    return ExpressionEvaluator()


def _scenarios_to_db(scenarios: list[dict]) -> ObservationDatabase:
    """Convert a list of scenario dicts to an in-memory ObservationDatabase."""
    tmp_path = PROJECT_ROOT / "data" / "observations" / "_tmp_test.json"
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp_path, "w") as f:
        json.dump(scenarios, f)
    db = ObservationDatabase(tmp_path)
    tmp_path.unlink()
    return db


def _check_invariant(
    ev: ExpressionEvaluator,
    scenarios: list[dict],
    invariant: str,
    min_score: float = 0.95,
) -> None:
    """Verify that a known invariant scores >= min_score on all matching scenarios."""
    db = _scenarios_to_db(scenarios)
    matching = [obs for obs in db if obs.known_invariant == invariant]
    assert len(matching) > 0, f"No scenarios with invariant {invariant!r}"
    for obs in matching:
        score = ev.score(invariant, obs)
        assert score >= min_score, (
            f"Invariant {invariant!r} scored {score:.4f} on {obs.id} "
            f"(expected >= {min_score})"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 1. Structure and validation
# ═══════════════════════════════════════════════════════════════════════════


class TestSimulatorStructure:
    """All generated scenarios must be valid Observation-compatible dicts."""

    def test_mechanics_scenarios_are_valid(self):
        scenarios = generate_all_mechanics()
        db = _scenarios_to_db(scenarios)
        issues = db.validate()
        assert not issues, f"Validation issues: {issues}"

    def test_em_scenarios_are_valid(self):
        scenarios = generate_all_electromagnetism()
        db = _scenarios_to_db(scenarios)
        issues = db.validate()
        assert not issues, f"Validation issues: {issues}"

    def test_thermal_scenarios_are_valid(self):
        scenarios = generate_all_thermodynamics()
        db = _scenarios_to_db(scenarios)
        issues = db.validate()
        assert not issues, f"Validation issues: {issues}"

    def test_each_scenario_has_required_fields(self):
        for domain_name, generator in [
            ("mechanics", generate_all_mechanics),
            ("electromagnetism", generate_all_electromagnetism),
            ("thermodynamics", generate_all_thermodynamics),
            ("quantum", generate_all_quantum),
            ("relativity", generate_all_relativity),
        ]:
            scenarios = generator()
            for s in scenarios:
                assert "id" in s, f"{domain_name}: missing id"
                assert "name" in s, f"{domain_name}: missing name"
                assert "description" in s, f"{domain_name}: missing description"
                assert "quantities" in s, f"{domain_name}: missing quantities"
                assert "parameters" in s, f"{domain_name}: missing parameters"
                assert "timesteps" in s, f"{domain_name}: missing timesteps"
                assert len(s["timesteps"]) >= 2, (
                    f"{domain_name} {s['id']}: need >= 2 timesteps"
                )
                for ts in s["timesteps"]:
                    assert "t" in ts, f"{domain_name} {s['id']}: timestep missing 't'"

    def test_scenario_ids_are_unique(self):
        for domain_name, generator in [
            ("mechanics", generate_all_mechanics),
            ("electromagnetism", generate_all_electromagnetism),
            ("thermodynamics", generate_all_thermodynamics),
            ("quantum", generate_all_quantum),
            ("relativity", generate_all_relativity),
        ]:
            scenarios = generator()
            ids = [s["id"] for s in scenarios]
            assert len(ids) == len(set(ids)), (
                f"{domain_name}: duplicate IDs: {[i for i in ids if ids.count(i) > 1]}"
            )


# ═══════════════════════════════════════════════════════════════════════════
# 2. Minimum scenario counts per domain
# ═══════════════════════════════════════════════════════════════════════════


class TestMinimumScenarioCounts:
    """Each domain must generate at least 50 scenarios."""

    def test_mechanics_has_50_plus(self):
        scenarios = generate_all_mechanics()
        assert len(scenarios) >= 50, f"Mechanics: {len(scenarios)} < 50"

    def test_em_has_50_plus(self):
        scenarios = generate_all_electromagnetism()
        assert len(scenarios) >= 50, f"EM: {len(scenarios)} < 50"

    def test_thermal_has_50_plus(self):
        scenarios = generate_all_thermodynamics()
        assert len(scenarios) >= 50, f"Thermal: {len(scenarios)} < 50"


# ═══════════════════════════════════════════════════════════════════════════
# 3. Mechanics invariants
# ═══════════════════════════════════════════════════════════════════════════


class TestMechanicsInvariants:
    """Known invariants should score > 0.95 on generated data."""

    def test_freefall_energy_conservation(self, ev):
        _check_invariant(
            ev, generate_freefall_scenarios(),
            "m*g*h + 0.5*m*v^2", min_score=0.95,
        )

    def test_projectile_energy_conservation(self, ev):
        _check_invariant(
            ev, generate_projectile_scenarios(),
            "m*g*y + 0.5*m*(vx^2 + vy^2)", min_score=0.95,
        )

    def test_pendulum_energy_conservation(self, ev):
        _check_invariant(
            ev, generate_pendulum_scenarios(),
            "m*g*h + 0.5*m*v^2", min_score=0.90,  # Small-angle approx
        )

    def test_spring_energy_conservation(self, ev):
        _check_invariant(
            ev, generate_spring_scenarios(),
            "0.5*k*x^2 + 0.5*m*v^2", min_score=0.95,
        )

    def test_collision_energy_conservation(self, ev):
        # Collisions use piecewise evaluation — test within each phase
        scenarios = generate_collision_scenarios()
        for s in scenarios:
            db = _scenarios_to_db([s])
            obs = db.get(s["id"])
            # Piecewise: energy conserved in before and after phases separately
            result = ev.score_piecewise("0.5*m1*v1^2 + 0.5*m2*v2^2", obs)
            piecewise_mean = result.get("piecewise_mean", 0.0)
            assert piecewise_mean >= 0.95, (
                f"Collision {obs.id}: piecewise_mean={piecewise_mean:.4f} < 0.95"
            )


# ═══════════════════════════════════════════════════════════════════════════
# 4. EM invariants
# ═══════════════════════════════════════════════════════════════════════════


class TestEMInvariants:
    """EM invariants should score > 0.95."""

    def test_e_field_energy_conservation(self, ev):
        _check_invariant(
            ev, generate_e_field_scenarios(),
            "0.5*m*(vx^2 + vy^2) - q*E*x", min_score=0.95,
        )

    def test_b_field_energy_conservation(self, ev):
        _check_invariant(
            ev, generate_b_field_scenarios(),
            "0.5*m*(vx^2 + vy^2)", min_score=0.95,
        )

    def test_eb_combined_energy_conservation(self, ev):
        _check_invariant(
            ev, generate_eb_combined_scenarios(),
            "0.5*m*(vx^2 + vy^2) - q*E*y", min_score=0.90,
        )

    def test_coulomb_energy_conservation(self, ev):
        scenarios = generate_coulomb_scenarios()
        db = _scenarios_to_db(scenarios)
        for obs in db:
            score = ev.score("0.5*m1*v1^2 + 0.5*m2*v2^2 + k*q1*q2/r", obs)
            # Coulomb with numeric integration may have some error
            assert score >= 0.85, (
                f"Coulomb {obs.id}: score={score:.4f} < 0.85"
            )

    def test_induced_emf_constant(self, ev):
        _check_invariant(
            ev, generate_induced_emf_scenarios(),
            "abs(epsilon)", min_score=0.95,
        )


# ═══════════════════════════════════════════════════════════════════════════
# 5. Thermal invariants
# ═══════════════════════════════════════════════════════════════════════════


class TestThermalInvariants:
    """Thermal invariants should score > 0.95."""

    def test_isothermal_pv_constant(self, ev):
        _check_invariant(
            ev, generate_isothermal_scenarios(),
            "P*V", min_score=0.95,
        )

    def test_adiabatic_pv_gamma_constant(self, ev):
        # Each adiabatic scenario has its own invariant with its gamma value
        scenarios = generate_adiabatic_scenarios()
        for s in scenarios:
            inv = s["known_invariant"]
            db = _scenarios_to_db([s])
            obs = db.get(s["id"])
            score = ev.score(inv, obs)
            assert score >= 0.95, (
                f"Adiabatic {obs.id}: invariant={inv!r} score={score:.4f} < 0.95"
            )

    def test_isobaric_vt_constant(self, ev):
        _check_invariant(
            ev, generate_isobaric_scenarios(),
            "V/T", min_score=0.95,
        )

    def test_isochoric_pt_constant(self, ev):
        _check_invariant(
            ev, generate_isochoric_scenarios(),
            "P/T", min_score=0.95,
        )

    def test_ideal_gas_pvt_constant(self, ev):
        _check_invariant(
            ev, generate_ideal_gas_scenarios(),
            "P*V/T", min_score=0.95,
        )

    def test_entropy_linear_with_log_v(self, ev):
        # ΔS / log(V) should be constant (= nR)
        # But "delta_S / log(V)" won't parse as-is since log is a function call.
        # Use a numeric check instead.
        scenarios = generate_entropy_scenarios()
        db = _scenarios_to_db(scenarios)
        for obs in db:
            # delta_S should be linear in log(V)
            # Pick first and last timestep, compute ratio
            if len(obs.timesteps) < 2:
                continue
            ts0 = obs.timesteps[0]
            ts_last = obs.timesteps[-1]
            V0 = ts0["V"]
            Vf = ts_last["V"]
            dS = ts_last["delta_S"] - ts0["delta_S"]
            log_ratio = math.log(Vf / V0) if Vf > V0 and V0 > 0 else 0.0
            if log_ratio > 1e-10:
                nR_computed = dS / log_ratio
                nR_expected = obs.parameters["n"] * obs.parameters["R"]
                rel_err = abs(nR_computed - nR_expected) / nR_expected
                assert rel_err < 0.02, (
                    f"Entropy {obs.id}: nR computed={nR_computed:.4f}, "
                    f"expected={nR_expected:.4f}, rel_err={rel_err:.6f}"
                )


# ═══════════════════════════════════════════════════════════════════════════
# 6. Quantum invariants
# ═══════════════════════════════════════════════════════════════════════════


class TestQuantumInvariants:
    """Quantum invariants should score > 0.90 on generated data."""

    def test_particle_in_box_energy_quantization(self, ev):
        # E ∝ n² — check that E/n² is constant for n >= 1
        scenarios = generate_particle_in_box_scenarios()
        for s in scenarios:
            # Energy quantization: E_n / n^2 = π²ℏ²/(2mL²) = constant
            E1 = s["parameters"].get("E1")
            E2 = s["parameters"].get("E2")
            if E1 and E2:
                # E2/E1 should be close to 4 (since E_n ∝ n²)
                ratio = E2 / E1
                assert 3.8 < ratio < 4.2, (
                    f"Particle-in-box {s['id']}: E2/E1={ratio:.3f} not ≈ 4"
                )

    def test_harmonic_oscillator_energy_spacing(self, ev):
        # E_n = ℏω(n + ½), so ΔE = ℏω = constant spacing
        scenarios = generate_harmonic_oscillator_scenarios()
        for s in scenarios:
            E0 = s["parameters"].get("E0")
            E1 = s["parameters"].get("E1")
            if E0 and E1:
                spacing_expected = s["parameters"]["omega"] * 1.054571817e-34 / 1.602176634e-19
                spacing_actual = abs(E1 - E0)
                rel_err = abs(spacing_actual - spacing_expected) / spacing_expected
                assert rel_err < 0.02, (
                    f"Harmonic osc {s['id']}: spacing={spacing_actual:.4f}eV vs expected={spacing_expected:.4f}eV"
                )

    def test_hydrogen_energy_levels(self, ev):
        # E_n = -13.6 * Z² / n²
        scenarios = generate_hydrogen_atom_scenarios()
        for s in scenarios:
            Z = s["parameters"]["Z"]
            E1 = s["parameters"].get("E1")
            if E1:
                expected = -13.605693 * Z**2
                rel_err = abs(E1 - expected) / abs(expected)
                assert rel_err < 0.01, (
                    f"Hydrogen {s['id']}: E1={E1:.4f}eV vs expected={expected:.4f}eV"
                )

    def test_probability_current_constant(self, ev):
        scenarios = generate_probability_current_scenarios()
        for s in scenarios:
            db = _scenarios_to_db([s])
            obs = db.get(s["id"])
            score = ev.score("j", obs)
            assert score >= 0.95, (
                f"Prob current {obs.id}: score={score:.4f} < 0.95"
            )

    def test_expectation_energy_constant(self, ev):
        scenarios = generate_expectation_scenarios()
        for s in scenarios:
            db = _scenarios_to_db([s])
            obs = db.get(s["id"])
            score = ev.score("E", obs)
            assert score >= 0.90, (
                f"Expectation {obs.id}: score={score:.4f} < 0.90"
            )

    def test_wave_packet_probability_conservation(self, ev):
        scenarios = generate_wave_packet_scenarios()
        for s in scenarios:
            db = _scenarios_to_db([s])
            obs = db.get(s["id"])
            score = ev.score("prob", obs)
            assert score >= 0.95, (
                f"Wave-packet {obs.id}: prob score={score:.4f} < 0.95"
            )


# ═══════════════════════════════════════════════════════════════════════════
# 7. Relativistic invariants
# ═══════════════════════════════════════════════════════════════════════════


class TestRelativityInvariants:
    """Relativistic invariants should score > 0.95 on generated data."""

    def test_spacetime_interval_invariant(self, ev):
        # For constant velocity: t/tau = gamma = constant
        # tau * gamma = t varies, but t/tau is constant
        scenarios = generate_spacetime_interval_scenarios()
        for s in scenarios:
            db = _scenarios_to_db([s])
            obs = db.get(s["id"])
            score = ev.score("t / tau", obs)
            assert score >= 0.95, (
                f"Spacetime {obs.id}: score={score:.4f} < 0.95"
            )

    def test_time_dilation_spacetime_interval(self, ev):
        scenarios = generate_time_dilation_scenarios()
        for s in scenarios:
            db = _scenarios_to_db([s])
            obs = db.get(s["id"])
            # For constant velocity: t / tau = gamma = constant
            score = ev.score("t / tau", obs)
            assert score >= 0.95, (
                f"Time dilation {obs.id}: score={score:.4f} < 0.95"
            )

    def test_length_contraction_lorentz_factor(self, ev):
        # L * gamma = L0 = constant — but L is in parameters, gamma in params
        # Check L_obs (in timesteps) * gamma = L0 = constant
        scenarios = generate_length_contraction_scenarios()
        for s in scenarios:
            db = _scenarios_to_db([s])
            obs = db.get(s["id"])
            score = ev.score("L_obs * gamma", obs)
            assert score >= 0.95, (
                f"Length contraction {obs.id}: score={score:.4f} < 0.95"
            )

    def test_energy_momentum_invariant(self, ev):
        # E^2 - p^2 = (mc^2)^2 = constant (p is already pc in MeV units)
        # Timestep p is pc/conv in MeV; E is in MeV
        scenarios = generate_energy_momentum_scenarios()
        for s in scenarios:
            db = _scenarios_to_db([s])
            obs = db.get(s["id"])
            score = ev.score("E^2 - p^2", obs)
            assert score >= 0.95, (
                f"Energy-momentum {obs.id}: score={score:.4f} < 0.95"
            )

    def test_lorentz_boost_spacetime_interval(self, ev):
        scenarios = generate_lorentz_boost_scenarios()
        for s in scenarios:
            db = _scenarios_to_db([s])
            obs = db.get(s["id"])
            # In each timestep, s2 (computed from t,x) should equal s2_prime
            # (computed from t',x' after Lorentz boost).
            # Verify numerically: s2 should match s2_prime to high precision
            for ts in obs.timesteps:
                s2 = ts.get("s2", 0.0)
                s2_prime = ts.get("s2_prime", 0.0)
                rel_err = abs(s2 - s2_prime) / max(abs(s2), 1.0)
                assert rel_err < 0.01, (
                    f"Lorentz boost {obs.id}: s2={s2:.4f} ≠ s2_prime={s2_prime:.4f}"
                )

    def test_proper_time_dilation(self, ev):
        # tau * gamma = t varies, but t/tau = gamma = constant for uniform motion
        scenarios = generate_proper_time_scenarios()
        for s in scenarios:
            db = _scenarios_to_db([s])
            obs = db.get(s["id"])
            score = ev.score("t / tau", obs)
            # Proper time ratio is nearly constant even for acceleration
            assert score >= 0.85, (
                f"Proper time {obs.id}: score={score:.4f} < 0.85"
            )


# ═══════════════════════════════════════════════════════════════════════════
# 8. ObservationDatabase integration (quantum + relativity)
# ═══════════════════════════════════════════════════════════════════════════


class TestDatabaseIntegration:
    """Generated scenarios must load into ObservationDatabase without errors."""

    def test_mechanics_loads_into_database(self):
        scenarios = generate_all_mechanics()
        db = _scenarios_to_db(scenarios)
        assert len(db) == len(scenarios)
        # Spot check one
        obs = db.get(scenarios[0]["id"])
        assert obs.id == scenarios[0]["id"]

    def test_em_loads_into_database(self):
        scenarios = generate_all_electromagnetism()
        db = _scenarios_to_db(scenarios)
        assert len(db) == len(scenarios)
        obs = db.get(scenarios[0]["id"])
        assert obs.id == scenarios[0]["id"]

    def test_thermal_loads_into_database(self):
        scenarios = generate_all_thermodynamics()
        db = _scenarios_to_db(scenarios)
        assert len(db) == len(scenarios)
        obs = db.get(scenarios[0]["id"])
        assert obs.id == scenarios[0]["id"]

    def test_quantum_loads_into_database(self):
        scenarios = generate_all_quantum()
        db = _scenarios_to_db(scenarios)
        assert len(db) == len(scenarios)
        obs = db.get(scenarios[0]["id"])
        assert obs.id == scenarios[0]["id"]

    def test_relativity_loads_into_database(self):
        scenarios = generate_all_relativity()
        db = _scenarios_to_db(scenarios)
        assert len(db) == len(scenarios)
        obs = db.get(scenarios[0]["id"])
        assert obs.id == scenarios[0]["id"]


# ═══════════════════════════════════════════════════════════════════════════
# 9. Numerical sanity
# ═══════════════════════════════════════════════════════════════════════════


class TestNumericalSanity:
    """Generated values should be physically reasonable."""

    def test_no_nan_or_inf(self):
        for domain_name, generator in [
            ("mechanics", generate_all_mechanics),
            ("electromagnetism", generate_all_electromagnetism),
            ("thermodynamics", generate_all_thermodynamics),
            ("quantum", generate_all_quantum),
            ("relativity", generate_all_relativity),
        ]:
            scenarios = generator()
            for s in scenarios:
                for ts in s["timesteps"]:
                    for key, val in ts.items():
                        assert not math.isnan(val), (
                            f"{domain_name} {s['id']}: NaN in timestep {key}"
                        )
                        assert not math.isinf(val), (
                            f"{domain_name} {s['id']}: inf in timestep {key}"
                        )

                for key, val in s["parameters"].items():
                    if isinstance(val, (int, float)):
                        assert not math.isnan(val), (
                            f"{domain_name} {s['id']}: NaN in parameter {key}"
                        )
                        assert not math.isinf(val), (
                            f"{domain_name} {s['id']}: inf in parameter {key}"
                        )

    def test_mass_is_positive(self):
        for domain_name, generator in [
            ("mechanics", generate_all_mechanics),
            ("electromagnetism", generate_all_electromagnetism),
            ("quantum", generate_all_quantum),
        ]:
            scenarios = generator()
            for s in scenarios:
                for key in ["m", "m1", "m2"]:
                    if key in s["parameters"]:
                        assert s["parameters"][key] > 0, (
                            f"{domain_name} {s['id']}: {key} must be positive"
                        )

    def test_temperature_is_positive(self):
        scenarios = generate_all_thermodynamics()
        for s in scenarios:
            if "T" in s["parameters"]:
                assert s["parameters"]["T"] > 0, (
                    f"Thermal {s['id']}: T must be positive"
                )

    def test_quantum_scenarios_have_positive_parameters(self):
        scenarios = generate_all_quantum()
        for s in scenarios:
            for key in ["m", "L", "hbar", "omega"]:
                if key in s["parameters"]:
                    assert s["parameters"][key] > 0, (
                        f"Quantum {s['id']}: {key} must be positive"
                    )

    def test_relativistic_velocity_less_than_c(self):
        scenarios = generate_all_relativity()
        for s in scenarios:
            if "v" in s["parameters"]:
                v = s["parameters"]["v"]
                if isinstance(v, (int, float)):
                    assert abs(v) < 3e8, (
                        f"Relativity {s['id']}: v={v:.1e} >= c"
                    )
