"""Observation database loader for physics discovery.

Loads the observation database JSON, extracts available quantities per scenario,
and provides iteration/query access.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Observation:
    """A single physical scenario with measurements."""

    id: str
    name: str
    description: str
    quantities: dict[str, str]  # name -> dimension type
    parameters: dict[str, float]  # constant parameters
    timesteps: list[dict[str, float]]  # [{t, h, v, ...}, ...]
    known_invariant: str | None  # expression that should be conserved
    lean_theorem: str  # Lean proof (may be empty)


class ObservationDatabase:
    """Load and query the observation database.

    Parameters
    ----------
    path : str or Path
        Path to the JSON observation database file.
    """

    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        with open(path) as f:
            raw = json.load(f)

        self._observations: list[Observation] = []
        self._by_id: dict[str, Observation] = {}

        for entry in raw:
            obs = Observation(
                id=entry["id"],
                name=entry["name"],
                description=entry["description"],
                quantities=dict(entry["quantities"]),
                parameters=dict(entry["parameters"]),
                timesteps=[dict(ts) for ts in entry["timesteps"]],
                known_invariant=entry.get("known_invariant"),
                lean_theorem=entry.get("lean_theorem", ""),
            )
            self._observations.append(obs)
            self._by_id[obs.id] = obs

    # ── queries ──────────────────────────────────────────────────────────

    def get(self, obs_id: str) -> Observation:
        """Return an observation by its unique id.

        Raises KeyError if not found.
        """
        if obs_id not in self._by_id:
            raise KeyError(f"Observation {obs_id!r} not found")
        return self._by_id[obs_id]

    def get_quantities(self, obs_id: str) -> dict[str, str]:
        """Return available quantities and their dimension types."""
        return dict(self.get(obs_id).quantities)

    def all_quantities(self) -> set[str]:
        """Return the union of all quantity names across all scenarios."""
        names: set[str] = set()
        for obs in self._observations:
            names.update(obs.quantities.keys())
        return names

    # ── iteration ────────────────────────────────────────────────────────

    def __iter__(self):
        return iter(self._observations)

    def __len__(self) -> int:
        return len(self._observations)

    def __getitem__(self, index: int) -> Observation:
        return self._observations[index]

    def __contains__(self, obs_id: str) -> bool:
        return obs_id in self._by_id

    # ── validation ───────────────────────────────────────────────────────

    def validate(self) -> list[str]:
        """Validate all observations and return list of issues (empty = valid).

        Checks:
        - Every observation has id, name, description
        - quantities is non-empty
        - timesteps has at least 2 entries
        - Each timestep contains all quantity keys (except constants in parameters)
        - Parameters are non-negative where physical (mass, length)
        """
        issues: list[str] = []

        for obs in self._observations:
            prefix = f"[{obs.id}]"
            if not obs.id:
                issues.append(f"{prefix} empty id")
            if not obs.name:
                issues.append(f"{prefix} empty name")
            if not obs.quantities:
                issues.append(f"{prefix} no quantities defined")
            if len(obs.timesteps) < 2:
                issues.append(f"{prefix} needs at least 2 timesteps, got {len(obs.timesteps)}")

            for i, ts in enumerate(obs.timesteps):
                if "t" not in ts:
                    issues.append(f"{prefix} timestep {i} missing 't'")

        return issues

    @property
    def scenario_ids(self) -> list[str]:
        """Return all scenario IDs."""
        return list(self._by_id.keys())
