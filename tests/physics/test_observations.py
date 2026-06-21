"""Tests for the observation database loader."""

from pathlib import Path

import pytest

from src.physics.observations import (
    Observation,
    ObservationDatabase,
)


PHASE1_PATH = Path(__file__).parent.parent.parent / "data" / "observations" / "phase1_falling.json"


@pytest.fixture
def db() -> ObservationDatabase:
    """Load the Phase 1 observation database."""
    return ObservationDatabase(PHASE1_PATH)


class TestObservationDatabase:
    """Smoke tests for loading and validating the Phase 1 database."""

    def test_loads_all_10_scenarios(self, db: ObservationDatabase) -> None:
        """Should load exactly 10 scenarios."""
        assert len(db) == 10, f"Expected 10 scenarios, got {len(db)}"

    def test_all_ids_unique(self, db: ObservationDatabase) -> None:
        """All observation IDs must be unique."""
        ids = db.scenario_ids
        assert len(ids) == len(set(ids))

    def test_expected_ids_present(self, db: ObservationDatabase) -> None:
        """The 10 scenarios from the plan must be present."""
        expected = {
            "falling_ball_straight_drop",
            "falling_ball_upward_throw",
            "falling_ball_varying_mass",
            "pendulum_small_angle",
            "pendulum_large_angle",
            "spring_undamped",
            "spring_damped_light",
            "projectile_45deg",
            "projectile_90deg",
            "sliding_block_incline",
        }
        actual = set(db.scenario_ids)
        assert actual == expected, f"Missing: {expected - actual}, extra: {actual - expected}"

    def test_each_scenario_has_timesteps(self, db: ObservationDatabase) -> None:
        """Every scenario must have at least 2 timesteps."""
        for obs in db:
            assert len(obs.timesteps) >= 2, (
                f"{obs.id} has only {len(obs.timesteps)} timestep(s)"
            )

    def test_each_scenario_has_quantities(self, db: ObservationDatabase) -> None:
        """Every scenario must declare quantities."""
        for obs in db:
            assert obs.quantities, f"{obs.id} has no quantities"
            assert isinstance(obs.quantities, dict)

    def test_each_scenario_has_parameters(self, db: ObservationDatabase) -> None:
        """Every scenario must have at least one parameter."""
        for obs in db:
            assert obs.parameters, f"{obs.id} has no parameters"

    def test_timestep_keys_match_quantities_or_parameters(
        self, db: ObservationDatabase
    ) -> None:
        """All timestep keys must be declared in quantities or parameters."""
        for obs in db:
            valid_keys = set(obs.quantities.keys()) | set(obs.parameters.keys())
            for j, ts in enumerate(obs.timesteps):
                for key in ts:
                    assert key in valid_keys, (
                        f"{obs.id} timestep {j}: unknown key {key!r}. "
                        f"Valid: {sorted(valid_keys)}"
                    )

    def test_known_invariant_or_none(self, db: ObservationDatabase) -> None:
        """known_invariant must be a string or None."""
        for obs in db:
            assert obs.known_invariant is None or isinstance(
                obs.known_invariant, str
            ), f"{obs.id}: known_invariant should be str or None"

    def test_all_quantities(self, db: ObservationDatabase) -> None:
        """all_quantities returns union of all quantity names."""
        quantities = db.all_quantities()
        assert "m" in quantities, "Mass quantity 'm' must be available"
        assert "v" in quantities, "Velocity quantity 'v' must be available"
        core = {"m", "v", "g", "h", "t"}
        found = quantities & core
        assert len(found) >= 3, f"Expected at least 3 core quantities, got {found}"

    def test_get_by_id(self, db: ObservationDatabase) -> None:
        """Should retrieve observations by ID."""
        obs = db.get("falling_ball_straight_drop")
        assert obs.id == "falling_ball_straight_drop"
        assert obs.name == "Ball dropped from rest"

    def test_get_missing_raises_keyerror(self, db: ObservationDatabase) -> None:
        """Missing ID should raise KeyError."""
        with pytest.raises(KeyError):
            db.get("nonexistent_scenario")

    def test_contains_check(self, db: ObservationDatabase) -> None:
        """__contains__ should work for known and unknown IDs."""
        assert "falling_ball_straight_drop" in db
        assert "nonexistent" not in db

    def test_validate_no_issues(self, db: ObservationDatabase) -> None:
        """validate() should return empty list for valid database."""
        issues = db.validate()
        assert issues == [], f"Validation found issues: {issues}"


class TestObservationDatabaseErrors:
    """Tests for error handling when loading invalid data."""

    def test_file_not_found(self) -> None:
        """Should raise FileNotFoundError for nonexistent files."""
        with pytest.raises(FileNotFoundError):
            ObservationDatabase("/nonexistent/path/observations.json")

    def test_missing_required_field(self, tmp_path: Path) -> None:
        """Should raise KeyError if a scenario is missing required fields."""
        import json
        bad = tmp_path / "bad.json"
        json.dump([{"id": "test"}], bad.open("w"))
        with pytest.raises(KeyError):
            ObservationDatabase(bad)

    def test_duplicate_id_detected(self, tmp_path: Path) -> None:
        """Duplicate IDs are silently overwritten (last wins). Not an error."""
        import json
        bad = tmp_path / "bad.json"
        json.dump([
            {"id": "dup", "name": "a", "description": "a",
             "quantities": {"x": "Length"}, "parameters": {"k": 1.0},
             "timesteps": [{"x": 0.0, "t": 0.0}, {"x": 1.0, "t": 1.0}]},
            {"id": "dup", "name": "b", "description": "b",
             "quantities": {"x": "Length"}, "parameters": {"k": 2.0},
             "timesteps": [{"x": 0.0, "t": 0.0}, {"x": 2.0, "t": 1.0}]},
        ], bad.open("w"))
        db = ObservationDatabase(bad)
        assert len(db) == 2
