"""Self-play loop for autonomous physics invariant discovery.

Orchestrates ExpressionSearch over a train/test split of physical
observations. Loads data, splits, runs search, tests generalization,
logs results.

Architecture (from plan Sections 1, 4, 6):
    Load observation DB → split 8 train / 2 test
    Initialize search with depth-1 pool
    Run search → evaluate on test → log discoveries
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.physics.dimensions import Dimension
from src.physics.evaluator import ExpressionEvaluator
from src.physics.observations import Observation, ObservationDatabase
from src.physics.search import ExpressionSearch, SearchResult


@dataclass
class DiscoveryRecord:
    """A single discovered invariant with all metadata."""
    expression: str
    train_score: float
    test_score: float
    depth: int
    expansions_needed: int
    train_constancies: list[float]
    test_constancies: list[float]
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "expression": self.expression,
            "train_score": self.train_score,
            "test_score": self.test_score,
            "depth": self.depth,
            "expansions_needed": self.expansions_needed,
            "train_constancies": self.train_constancies,
            "test_constancies": self.test_constancies,
            "timestamp": self.timestamp,
        }


class SelfPlayLoop:
    """Orchestrate the self-play physics discovery process.

    Parameters
    ----------
    db_path : str or Path
        Path to the observation database JSON file.
    train_count : int
        Number of observations for training.
    test_count : int
        Number of observations for generalization testing.
    max_expansions : int
        Maximum search expansions.
    max_depth : int
        Maximum expression tree depth.
    discovery_threshold : float
        Constancy threshold for discovery.
    top_k : int
        Top expressions to keep as expansion seeds.
    seed : int or None
        Random seed for reproducible splits.
    """

    def __init__(
        self,
        db_path: str | Path,
        *,
        train_count: int = 8,
        test_count: int = 2,
        max_expansions: int = 10_000,
        max_depth: int = 6,
        discovery_threshold: float = 0.95,
        top_k: int = 50,
        seed: int | None = 42,
    ) -> None:
        self.db_path = Path(db_path)
        self.train_count = train_count
        self.test_count = test_count
        self.max_expansions = max_expansions
        self.max_depth = max_depth
        self.discovery_threshold = discovery_threshold
        self.top_k = top_k
        self.seed = seed

        self._db: ObservationDatabase | None = None
        self._train: list[Observation] = []
        self._test: list[Observation] = []
        self._quantities: dict[str, Dimension] = {}
        self._discoveries: list[DiscoveryRecord] = []
        self._total_expansions: int = 0
        self._evaluator = ExpressionEvaluator()

    # ── Setup ─────────────────────────────────────────────────────────────

    def load_and_split(self) -> None:
        """Load database and split into train/test."""
        self._db = ObservationDatabase(self.db_path)
        all_obs = list(self._db)
        if len(all_obs) < self.train_count + self.test_count:
            raise ValueError(
                f"Database has {len(all_obs)} observations, "
                f"need at least {self.train_count + self.test_count}"
            )

        rng = random.Random(self.seed)
        indices = list(range(len(all_obs)))
        rng.shuffle(indices)

        self._train = [all_obs[i] for i in indices[:self.train_count]]
        self._test = [all_obs[i] for i in indices[self.train_count:self.train_count + self.test_count]]
        self._quantities = self._extract_quantities(self._train)

    def _extract_quantities(self, obs_list: list[Observation]) -> dict[str, Dimension]:
        quantities: dict[str, Dimension] = {}
        for obs in obs_list:
            for name, dim_name in obs.quantities.items():
                if name not in quantities:
                    quantities[name] = Dimension.named(dim_name)
        return quantities

    # ── Main loop ─────────────────────────────────────────────────────────

    def run(self) -> list[DiscoveryRecord]:
        """Run the self-play discovery loop."""
        if self._db is None:
            self.load_and_split()

        search = ExpressionSearch(
            quantities=self._quantities,
            train_observations=self._train,
            max_depth=self.max_depth,
            max_expansions=self.max_expansions,
            discovery_threshold=self.discovery_threshold,
            top_k=self.top_k,
        )

        result = search.run()
        self._total_expansions = result.expansions

        if not result.is_discovery:
            return []

        test_score = self._evaluate_on_test(result.expression)
        test_constancies = search.per_observation_scores(result.expression, self._test)

        discovery = DiscoveryRecord(
            expression=result.expression,
            train_score=result.score,
            test_score=test_score,
            depth=result.depth,
            expansions_needed=result.expansions,
            train_constancies=result.train_constancies,
            test_constancies=test_constancies,
        )
        self._discoveries.append(discovery)
        return self._discoveries

    def run_with_progress(self) -> list[DiscoveryRecord]:
        """Run with progress logging."""
        if self._db is None:
            self.load_and_split()

        search = ExpressionSearch(
            quantities=self._quantities,
            train_observations=self._train,
            max_depth=self.max_depth,
            max_expansions=self.max_expansions,
            discovery_threshold=self.discovery_threshold,
            top_k=self.top_k,
        )

        for expansion_count, snapshot in search.run_with_snapshots():
            if expansion_count % 500 == 0 or snapshot.is_discovery:
                print(
                    f"  [{expansion_count:5d}] best={snapshot.expression:<30s} "
                    f"score={snapshot.score:.4f} depth={snapshot.depth}"
                )
            if snapshot.is_discovery:
                test_score = self._evaluate_on_test(snapshot.expression)
                test_constancies = search.per_observation_scores(
                    snapshot.expression, self._test
                )
                discovery = DiscoveryRecord(
                    expression=snapshot.expression,
                    train_score=snapshot.score,
                    test_score=test_score,
                    depth=snapshot.depth,
                    expansions_needed=expansion_count,
                    train_constancies=snapshot.train_constancies,
                    test_constancies=test_constancies,
                )
                self._discoveries.append(discovery)
                break

        self._total_expansions = search.expansion_count
        return self._discoveries

    def _evaluate_on_test(self, expression: str) -> float:
        if not self._test:
            return 0.0
        scores = [self._evaluator.score(expression, obs) for obs in self._test]
        return sum(scores) / len(scores)

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def train_observations(self) -> list[Observation]:
        return list(self._train)

    @property
    def test_observations(self) -> list[Observation]:
        return list(self._test)

    @property
    def discoveries(self) -> list[DiscoveryRecord]:
        return list(self._discoveries)

    @property
    def total_expansions(self) -> int:
        return self._total_expansions

    @property
    def discovered_energy(self) -> bool:
        return any(d.test_score >= self.discovery_threshold for d in self._discoveries)

    # ── Logging / export ──────────────────────────────────────────────────

    def summary(self) -> str:
        lines = [
            f"Self-play discovery run",
            f"  Database: {self.db_path}",
            f"  Train: {len(self._train)} observations",
            f"  Test:  {len(self._test)} observations",
            f"  Budget: {self.max_expansions} expansions, depth <= {self.max_depth}",
            f"  Expansions used: {self.total_expansions}",
            "",
        ]
        if not self._discoveries:
            lines.append("  Result: NO invariant discovered within budget")
        else:
            lines.append(f"  Discoveries: {len(self._discoveries)}")
            for i, d in enumerate(self._discoveries):
                lines.append(f"  [{i}] {d.expression}")
                lines.append(f"      Train score: {d.train_score:.6f}")
                lines.append(f"      Test score:  {d.test_score:.6f}")
                lines.append(f"      Depth: {d.depth}, Expansions: {d.expansions_needed}")
                test_pass = "PASS" if d.test_score >= self.discovery_threshold else "FAIL"
                lines.append(f"      Generalization: {test_pass}")
        return "\n".join(lines)

    def export_results(self, output_path: str | Path) -> None:
        output = {
            "database": str(self.db_path),
            "train_observation_ids": [obs.id for obs in self._train],
            "test_observation_ids": [obs.id for obs in self._test],
            "parameters": {
                "train_count": self.train_count,
                "test_count": self.test_count,
                "max_expansions": self.max_expansions,
                "max_depth": self.max_depth,
                "discovery_threshold": self.discovery_threshold,
                "top_k": self.top_k,
                "seed": self.seed,
            },
            "result": {
                "discovered": self.discovered_energy,
                "total_expansions": self.total_expansions,
                "discoveries": [d.to_dict() for d in self._discoveries],
            },
        }
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(output, f, indent=2)


# ── Smoke test ────────────────────────────────────────────────────────────────

def run_phase_c_smoke_test(
    db_path: str | Path = "data/observations/phase1_falling.json",
    *,
    train_count: int = 8,
    test_count: int = 2,
    max_expansions: int = 5_000,
    max_depth: int = 6,
    discovery_threshold: float = 0.95,
    seed: int = 42,
    output_path: str | Path | None = "data/phase_c_discovery.json",
) -> dict[str, Any]:
    """Run the Phase C smoke test."""
    loop = SelfPlayLoop(
        db_path=db_path,
        train_count=train_count,
        test_count=test_count,
        max_expansions=max_expansions,
        max_depth=max_depth,
        discovery_threshold=discovery_threshold,
        seed=seed,
    )
    discoveries = loop.run()
    result: dict[str, Any] = {
        "train_ids": [obs.id for obs in loop.train_observations],
        "test_ids": [obs.id for obs in loop.test_observations],
        "total_expansions": loop.total_expansions,
        "discovered": loop.discovered_energy,
        "discoveries": [d.to_dict() for d in discoveries],
    }
    if output_path is not None:
        loop.export_results(output_path)
    return result
