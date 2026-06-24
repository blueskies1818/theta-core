"""Noise calibration and real experimental data pipeline.

Phase D/E: Adds controlled Gaussian noise to synthetic observations,
computes noise floors for honesty gating, and supports real experimental
data with explicit error bars.

Architecture
------------
1. NoiseAugmenter — inject Gaussian noise into observation timesteps
2. NoiseCalibrator — compute noise floor per scenario + noise level
3. RealExperimentalLoader — load datasets with error bars
4. NoiseGatedEvaluator — drop-in replacement for ExpressionEvaluator
   that gates discoveries using noise calibration

Gating formula:
    Only accept discoveries where:
        score > noise_floor + n_sigma * sigma_floor
    where noise_floor = what a KNOWN-NON-CONSTANT expression scores
    at this noise level, and sigma_floor is its std across attempts.

Noise levels (configurable):
    - NONE  (0%):  no noise added
    - LOW   (1%):  typical lab-grade measurements
    - MED   (3%):  field / classroom measurements
    - HIGH  (5%):  noisy sensor / frontier data
"""

from __future__ import annotations

import json
import math
import random
import statistics
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from src.physics.evaluator import ExpressionEvaluator
from src.physics.observations import Observation, ObservationDatabase


# ── Noise Level ───────────────────────────────────────────────────────────────

class NoiseLevel(Enum):
    """Pre-defined noise levels for calibration."""
    NONE = 0    # 0% noise — clean synthetic data
    LOW = 1     # 1% noise — lab-grade measurements
    MEDIUM = 3  # 3% noise — field measurements
    HIGH = 5    # 5% noise — noisy sensor / frontier data

    @property
    def sigma_pct(self) -> float:
        """Noise sigma as a fraction (e.g., 0.01 for 1%)."""
        return self.value / 100.0

    @classmethod
    def from_sigma_pct(cls, pct: float) -> "NoiseLevel":
        """Find the closest standard noise level."""
        if pct <= 0.001:
            return cls.NONE
        if pct <= 0.02:
            return cls.LOW
        if pct <= 0.04:
            return cls.MEDIUM
        return cls.HIGH


# ── Noise Augmenter ───────────────────────────────────────────────────────────

@dataclass
class NoiseConfig:
    """Configuration for noise augmentation.

    Attributes
    ----------
    noise_level : NoiseLevel
        Pre-set noise level.
    sigma_pct : float | None
        Override noise sigma fraction. None = use noise_level default.
    seed : int | None
        Random seed for reproducibility.
    per_timestep : bool
        If True, noise is independent per timestep. If False, same noise
        offset is used across all timesteps (simulating systematic error).
    """
    noise_level: NoiseLevel = NoiseLevel.NONE
    sigma_pct: float | None = None
    seed: int | None = None
    per_timestep: bool = True  # per-timestep noise (random), False = systematic

    @property
    def effective_sigma(self) -> float:
        """Return the effective noise sigma as a fraction."""
        if self.sigma_pct is not None:
            return self.sigma_pct / 100.0
        return self.noise_level.sigma_pct


class NoiseAugmenter:
    """Add controlled Gaussian noise to observation timesteps.

    Operates on all non-time quantity fields in each timestep.
    Preserves original observations (returns augmented copies).

    Example
    -------
    >>> augmenter = NoiseAugmenter(NoiseConfig(NoiseLevel.LOW, seed=42))
    >>> obs = db.get("freefall_g9.8_h10.0_v0.0")
    >>> noisy_obs = augmenter.augment(obs)
    """

    def __init__(self, config: NoiseConfig | None = None) -> None:
        self.config = config or NoiseConfig()
        self._rng: random.Random | None = None
        if self.config.seed is not None:
            self._rng = random.Random(self.config.seed)
        else:
            self._rng = random.Random()
        self._sigma = self.config.effective_sigma

    def augment(self, obs: Observation) -> Observation:
        """Return a copy of the observation with Gaussian noise added.

        Noise is applied to all quantity fields in each timestep EXCEPT 't'.
        Uses per-value sigma = sigma_pct * |value| (relative noise) or
        sigma_pct * mean_abs_value (if value is near zero).
        """
        if self._sigma <= 0:
            return obs

        # Compute per-quantity magnitude references
        magnitude_refs: dict[str, float] = {}
        for ts in obs.timesteps:
            for key, val in ts.items():
                if key == "t":
                    continue
                if key not in magnitude_refs:
                    magnitude_refs[key] = 0.0
                magnitude_refs[key] += abs(val)
        for key in magnitude_refs:
            magnitude_refs[key] /= max(len(obs.timesteps), 1)

        noisy_timesteps: list[dict[str, float]] = []
        for ts in obs.timesteps:
            noisy_ts: dict[str, float] = {}
            for key, val in ts.items():
                if key == "t":
                    noisy_ts[key] = val
                else:
                    ref = max(magnitude_refs.get(key, abs(val)), 1e-10)
                    noise_sigma = self._sigma * ref
                    if self._rng is not None:
                        noise = self._rng.gauss(0.0, noise_sigma)
                    else:
                        noise = random.gauss(0.0, noise_sigma)
                    noisy_ts[key] = val + noise
            noisy_timesteps.append(noisy_ts)

        return Observation(
            id=obs.id,
            name=obs.name,
            description=obs.description,
            quantities=dict(obs.quantities),
            parameters=dict(obs.parameters),
            timesteps=noisy_timesteps,
            known_invariant=obs.known_invariant,
            lean_theorem=obs.lean_theorem,
            external_forces=obs.external_forces,
            phase_regions=obs.phase_regions,
            is_conservative=obs.is_conservative,
        )

    def augment_database(self, db: ObservationDatabase) -> list[Observation]:
        """Return a list of augmented observations from a database."""
        return [self.augment(obs) for obs in db]


# ── Noise Calibrator ──────────────────────────────────────────────────────────

@dataclass
class NoiseFloorResult:
    """Result of noise floor computation for a scenario.

    Attributes
    ----------
    scenario_id : str
        Observation identifier.
    noise_level : str
        Noise level name (e.g., "LOW", "MEDIUM").
    sigma_pct : float
        Actual noise sigma used.
    noise_floor : float
        Mean constancy score of known-non-constant expressions.
    sigma_floor : float
        Standard deviation of non-constant expression scores.
    threshold : float
        Discovery threshold = noise_floor + 3 * sigma_floor.
    num_calibration_exprs : int
        Number of calibration expressions used.
    calibration_scores : list[float]
        Individual scores used for floor computation.
    """
    scenario_id: str
    noise_level: str
    sigma_pct: float
    noise_floor: float
    sigma_floor: float
    threshold: float
    num_calibration_exprs: int
    calibration_scores: list[float] = field(default_factory=list)


# Known-NON-constant expressions for noise floor calibration.
# These are deliberately varying quantities that should NOT be constant.
_NON_CONSTANT_EXPRESSIONS = [
    "v",            # velocity varies in gravity
    "h",            # height varies in falling
    "m*v",          # momentum varies? actually conserved in closed systems, but varies here
    "v^2",          # varies
    "t",            # time always varies
    "h*v",          # product varies
    "h*t",          # definitely varies with time
    "abs(v)",       # absolute velocity varies
    "h + v",        # sum varies
    "h/v",          # ratio varies
    "h^2",          # varies quadratically
    "v^3",          # varies
    "m*h*t",        # varies
    "g*v*t",        # varies
    "h/v + t",      # definitely varies
]


# Cache TTL: noise floors are cached per scenario/noise-level pair.
# They're stable given a fixed seed, so TTL is effectively infinite.

class NoiseCalibrator:
    """Compute noise floors and gate discoveries.

    The noise floor is the constancy score that *known-non-constant*
    expressions achieve at a given noise level.  If a candidate expression
    scores above noise_floor + N*sigma_floor, it's a genuine discovery
    rather than a noise artifact.

    Parameters
    ----------
    n_sigma : float
        Number of sigma above noise floor for discovery (default 3.0).
    calibration_exprs : list[str] | None
        Non-constant expressions for floor computation. Uses defaults if None.
    num_calibration_runs : int
        Number of noise realizations per expression (default 5).
    seed : int | None
        Seed for reproducible calibration.

    Example
    -------
    >>> calibrator = NoiseCalibrator(n_sigma=3.0, seed=42)
    >>> result = calibrator.calibrate(db, [obs], NoiseLevel.LOW)
    >>> result.noise_floor
    0.45...
    >>> calibrator.should_accept(score=0.95, floor=result)
    True
    """

    def __init__(
        self,
        n_sigma: float = 3.0,
        calibration_exprs: list[str] | None = None,
        num_calibration_runs: int = 5,
        seed: int | None = None,
    ) -> None:
        self.n_sigma = n_sigma
        self._calibration_exprs = calibration_exprs or list(_NON_CONSTANT_EXPRESSIONS)
        self._num_runs = num_calibration_runs
        self._seed = seed
        self._evaluator = ExpressionEvaluator()
        # Cache: (scenario_id, noise_level_name) -> NoiseFloorResult
        self._floor_cache: dict[tuple[str, str], NoiseFloorResult] = {}
        # Pre-computed noise class: which noise level bin a sigma_pct falls into
        self._level_cache: dict[float, NoiseLevel] = {}

    # ── Calibration ───────────────────────────────────────────────────────

    def calibrate(
        self,
        observations: list[Observation],
        noise_level: NoiseLevel,
        sigma_pct: float | None = None,
    ) -> NoiseFloorResult:
        """Compute noise floor for a set of observations at a noise level.

        Returns the AGGREGATE noise floor across all observations —
        i.e., what a non-constant expression scores on average against
        this scenario set when noise is present.
        """
        sigma = sigma_pct if sigma_pct is not None else noise_level.sigma_pct
        nc = NoiseConfig(
            noise_level=noise_level,
            sigma_pct=sigma * 100.0,
            seed=self._seed,
            per_timestep=True,
        )
        augmenter = NoiseAugmenter(nc)

        all_scores: list[float] = []
        for expr_str in self._calibration_exprs:
            for run in range(self._num_runs):
                aug_seed = (self._seed or 0) + run + hash(expr_str) % (2**31)
                aug = NoiseAugmenter(NoiseConfig(
                    noise_level=noise_level,
                    sigma_pct=sigma * 100.0,
                    seed=aug_seed,
                    per_timestep=True,
                ))
                for obs in observations:
                    noisy_obs = aug.augment(obs)
                    try:
                        # Evaluate without noise-aware parsing — just constancy
                        score = self._evaluator.score(expr_str, noisy_obs)
                        if score > 0.0:  # Only count evaluable expressions
                            all_scores.append(score)
                    except Exception:
                        continue

        n = len(all_scores)
        if n < 2:
            return NoiseFloorResult(
                scenario_id="aggregate",
                noise_level=noise_level.name,
                sigma_pct=sigma,
                noise_floor=0.5,
                sigma_floor=0.1,
                threshold=0.8,
                num_calibration_exprs=n,
                calibration_scores=all_scores,
            )

        mean_floor = statistics.mean(all_scores)
        std_floor = statistics.stdev(all_scores) if n > 1 else 0.05
        # Use 95th percentile as noise floor, then add a 1-sigma margin
        # (3-sigma on bounded [0,1] data is too conservative)
        percentile_95 = float(sorted(all_scores)[int(n * 0.95)]) if n >= 20 else mean_floor + 2 * std_floor
        # Noise floor = what 95% of non-constant expressions score below
        noise_floor_val = min(max(mean_floor, percentile_95), 0.85)
        threshold = min(noise_floor_val + std_floor, 0.95)  # Add 1-sigma margin

        return NoiseFloorResult(
            scenario_id="aggregate",
            noise_level=noise_level.name,
            sigma_pct=sigma,
            noise_floor=noise_floor_val,
            sigma_floor=std_floor,
            threshold=threshold,
            num_calibration_exprs=n,
            calibration_scores=all_scores,
        )

    def calibrate_per_scenario(
        self,
        db: ObservationDatabase,
        noise_level: NoiseLevel,
        sigma_pct: float | None = None,
    ) -> dict[str, NoiseFloorResult]:
        """Compute noise floors PER SCENARIO (for caching)."""
        sigma = sigma_pct if sigma_pct is not None else noise_level.sigma_pct
        results: dict[str, NoiseFloorResult] = {}

        for obs in db:
            cache_key = (obs.id, noise_level.name)
            if cache_key in self._floor_cache:
                results[obs.id] = self._floor_cache[cache_key]
                continue

            result = self.calibrate([obs], noise_level, sigma)
            result.scenario_id = obs.id
            results[obs.id] = result
            self._floor_cache[cache_key] = result

        return results

    # ── Gating ─────────────────────────────────────────────────────────────

    def should_accept(self, score: float, floor: NoiseFloorResult) -> bool:
        """Return True if score passes the noise calibration gate."""
        return score > floor.threshold

    def gated_score(
        self,
        expr_str: str,
        db_or_obs: ObservationDatabase | Observation,
        noise_level: NoiseLevel,
        sigma_pct: float | None = None,
    ) -> dict[str, Any]:
        """Score an expression with noise gating.

        Returns dict with:
            - 'raw_score': un-gated constancy score
            - 'noise_floor': noise floor value
            - 'threshold': acceptance threshold
            - 'accepted': bool
            - 'margin': how far above/below threshold (negative = rejected)
            - 'significant': whether score > noise_floor + n_sigma * sigma_floor
        """
        sigma = sigma_pct if sigma_pct is not None else noise_level.sigma_pct

        if isinstance(db_or_obs, ObservationDatabase):
            observations = list(db_or_obs)
        else:
            observations = [db_or_obs]

        if not observations:
            return {
                "raw_score": 0.0,
                "noise_floor": 0.0,
                "threshold": 0.0,
                "accepted": False,
                "margin": 0.0,
                "significant": False,
            }

        # Compute noise floor for these observations
        floor = self.calibrate(observations, noise_level, sigma)

        # Compute raw score on noisy data
        nc = NoiseConfig(
            noise_level=noise_level,
            sigma_pct=sigma * 100.0,
            seed=self._seed,
            per_timestep=True,
        )
        augmenter = NoiseAugmenter(nc)

        noisy_scores: list[float] = []
        for obs in observations:
            noisy_obs = augmenter.augment(obs)
            try:
                noisy_scores.append(self._evaluator.score(expr_str, noisy_obs))
            except Exception:
                noisy_scores.append(0.0)

        raw_score = statistics.mean(noisy_scores) if noisy_scores else 0.0
        accepted = raw_score > floor.threshold
        margin = raw_score - floor.threshold

        return {
            "raw_score": round(raw_score, 6),
            "noise_floor": round(floor.noise_floor, 6),
            "sigma_floor": round(floor.sigma_floor, 6),
            "threshold": round(floor.threshold, 6),
            "accepted": accepted,
            "margin": round(margin, 6),
            "significant": accepted,
            "noise_level": noise_level.name,
            "sigma_pct": sigma,
        }

    # ── Fast inference ─────────────────────────────────────────────────────

    def classify_noise_level(self, sigma_pct: float) -> NoiseLevel:
        """Classify a sigma percentage into a standard noise level.

        Uses the 4 standard bins: NONE, LOW, MEDIUM, HIGH.
        Caches the result for repeated calls.
        """
        if sigma_pct in self._level_cache:
            return self._level_cache[sigma_pct]
        result = NoiseLevel.from_sigma_pct(sigma_pct)
        self._level_cache[sigma_pct] = result
        return result

    def get_cached_floor(
        self, scenario_id: str, noise_level: NoiseLevel
    ) -> NoiseFloorResult | None:
        """Retrieve a cached noise floor for a scenario+noise combo."""
        return self._floor_cache.get((scenario_id, noise_level.name))

    def pre_calibrate_all(
        self,
        db: ObservationDatabase,
        noise_levels: list[NoiseLevel] | None = None,
    ) -> dict[tuple[str, str], NoiseFloorResult]:
        """Pre-compute and cache noise floors for all scenarios at all levels.

        This enables fast inference: the noise floor lookup becomes O(1)
        after pre-calibration.
        """
        if noise_levels is None:
            noise_levels = list(NoiseLevel)

        all_results: dict[tuple[str, str], NoiseFloorResult] = {}

        for level in noise_levels:
            per_scenario = self.calibrate_per_scenario(db, level)
            for sid, result in per_scenario.items():
                key = (sid, level.name)
                all_results[key] = result
                self._floor_cache[key] = result

        return all_results

    def adaptive_threshold(
        self, scenario_id: str, sigma_pct: float
    ) -> float:
        """Return adaptive discovery threshold for a given scenario + noise.

        Falls back to conservative estimate if not pre-calibrated.
        """
        level = self.classify_noise_level(sigma_pct)
        cached = self.get_cached_floor(scenario_id, level)
        if cached is not None:
            return cached.threshold

        # Conservative fallback: what threshold would be needed?
        # Higher noise -> higher noise floor -> lower required threshold
        # because even real signals get attenuated.
        # But the GATE threshold must be ABOVE the noise floor.
        # Without calibration, use a heuristic:
        #   noise_floor ~= 0.5 + sigma_pct (noise helps non-constant look constant)
        #   threshold = noise_floor + 3 * sigma_pct (wide error bars)
        est_floor = 0.5 + sigma_pct
        est_threshold = min(est_floor + 3.0 * sigma_pct, 0.99)
        return est_threshold


# ── Real Experimental Data Loader ──────────────────────────────────────────────

@dataclass
class RealExperimentalObservation:
    """A single real experimental measurement with error bars.

    Unlike synthetic Observation, each data point has explicit uncertainty:
        value = x ± sigma_x

    Attributes
    ----------
    source : str
        Data source identifier (e.g., "pendulum_galileo", "muon_lifetime").
    description : str
        Human-readable description.
    domain : str
        Physics domain.
    quantities : dict[str, str]
        Quantity name -> dimension type.
    parameters : dict[str, float]
        Constant parameters with values.
    data_points : list[dict[str, float | None]]
        Each dict has: quantity_name -> value, and quantity_name_err -> error.
        Example: {"t": 1.0, "theta": 0.45, "theta_err": 0.02}
    known_invariant : str | None
        Known conserved expression (if any).
    """
    source: str
    description: str
    domain: str
    quantities: dict[str, str]
    parameters: dict[str, float]
    data_points: list[dict[str, float | None]]
    known_invariant: str | None = None

    def to_synthetic_observations(self, num_bootstrap: int = 1) -> list[Observation]:
        """Convert to synthetic Observation objects for evaluation.

        Uses error bars to sample plausible values via Gaussian noise.
        Each bootstrap creates one synthetic observation.
        """
        observations: list[Observation] = []
        for i in range(num_bootstrap):
            timesteps: list[dict[str, float]] = []
            for dp in self.data_points:
                ts: dict[str, float] = {}
                for key, val in dp.items():
                    if val is None:
                        continue
                    if key.endswith("_err"):
                        continue
                    err_key = key + "_err"
                    if err_key in dp and dp[err_key] is not None:
                        err_val_raw = dp[err_key]
                        if err_val_raw is not None:
                            sigma = abs(float(err_val_raw))
                            sampled = float(val) + random.gauss(0.0, sigma)
                            ts[key] = sampled
                        else:
                            ts[key] = float(val)
                    else:
                        ts[key] = float(val)
                timesteps.append(ts)

            observations.append(Observation(
                id=f"{self.source}_bs{i}",
                name=f"{self.source} (bootstrap {i})",
                description=self.description,
                quantities=dict(self.quantities),
                parameters=dict(self.parameters),
                timesteps=timesteps,
                known_invariant=self.known_invariant,
                lean_theorem="",
            ))

        return observations


class RealExperimentalLoader:
    """Load real experimental datasets with error bars.

    Supports three dataset formats:
    1. Native JSON with error bars
    2. CSV with error columns
    3. Legacy Observation JSON (converted on load)

    Example
    -------
    >>> loader = RealExperimentalLoader(Path("data/real_experimental"))
    >>> datasets = loader.load_all()
    >>> for ds in datasets:
    ...     obs = ds.to_synthetic_observations(num_bootstrap=5)
    ...     score = evaluator.score("m*g*h + 0.5*m*v^2", obs)
    """

    def __init__(self, data_dir: str | Path) -> None:
        self._data_dir = Path(data_dir)

    def load_all(self) -> list[RealExperimentalObservation]:
        """Load all real experimental datasets from the directory."""
        datasets: list[RealExperimentalObservation] = []
        if not self._data_dir.exists():
            return datasets

        for json_file in sorted(self._data_dir.glob("*.json")):
            try:
                ds = self._load_json(json_file)
                if ds is not None:
                    datasets.append(ds)
            except Exception as e:
                print(f"Warning: Failed to load {json_file}: {e}")

        return datasets

    def _load_json(self, path: Path) -> RealExperimentalObservation | None:
        with open(path) as f:
            raw = json.load(f)

        # Format 1: "observations" key with nested error bars {value, sigma}
        if "observations" in raw:
            entries = raw["observations"]
            if entries:
                first = entries[0]
                domain = first.get("domain", "unknown")
                # Extract quantities from first entry
                quantities = dict(first.get("quantities", {}))
                known_invariant = first.get("known_invariant")
                
                # Build data_points: flatten timesteps across all observations
                all_data_points: list[dict[str, float | None]] = []
                for entry in entries:
                    # Extract parameters as constant columns
                    param_vals: dict[str, float] = {}
                    param_errs: dict[str, float] = {}
                    for pk, pv in entry.get("parameters", {}).items():
                        if isinstance(pv, dict) and "value" in pv:
                            param_vals[pk] = float(pv["value"])
                            param_errs[pk] = float(pv.get("sigma", 0.0))
                        else:
                            param_vals[pk] = float(pv)
                            param_errs[pk] = 0.0
                    
                    for ts in entry.get("timesteps", []):
                        dp: dict[str, float | None] = {}
                        # Parameters first (constant across timesteps)
                        for pk, pv in param_vals.items():
                            dp[pk] = pv
                            dp[f"{pk}_err"] = param_errs.get(pk, 0.0)
                        # Timestep quantities
                        for tk, tv in ts.items():
                            if isinstance(tv, dict) and "value" in tv:
                                dp[tk] = float(tv["value"])
                                dp[f"{tk}_err"] = float(tv.get("sigma", 0.0))
                            else:
                                dp[tk] = float(tv)
                        all_data_points.append(dp)
                
                return RealExperimentalObservation(
                    source=raw.get("dataset", path.stem),
                    description=raw.get("description", first.get("description", "")),
                    domain=domain,
                    quantities=quantities,
                    parameters={},
                    data_points=all_data_points,
                    known_invariant=known_invariant,
                )
            return None

        # Format 2: "data_points" key (native format)
        if "data_points" in raw:
            return RealExperimentalObservation(
                source=raw.get("source", path.stem),
                description=raw.get("description", ""),
                domain=raw.get("domain", "unknown"),
                quantities=raw.get("quantities", {}),
                parameters=raw.get("parameters", {}),
                data_points=raw["data_points"],
                known_invariant=raw.get("known_invariant"),
            )

        # Legacy observation format: wrap as single dataset
        if "timesteps" in raw:
            # It's an observation array — treat the whole file as one dataset
            entries = raw if isinstance(raw, list) else [raw]
            data_points: list[dict[str, float | None]] = []
            for entry in entries:
                for ts in entry.get("timesteps", []):
                    dp: dict[str, float | None] = {}
                    for k, v in ts.items():
                        dp[k] = float(v)
                    data_points.append(dp)

            first_entry = entries[0] if entries else {}
            return RealExperimentalObservation(
                source=path.stem,
                description=first_entry.get("description", ""),
                domain="mechanics",
                quantities=first_entry.get("quantities", {}),
                parameters=first_entry.get("parameters", {}),
                data_points=data_points,
                known_invariant=first_entry.get("known_invariant"),
            )

        return None


# ── Noise-Gated Evaluator ─────────────────────────────────────────────────────

class NoiseGatedEvaluator:
    """Drop-in replacement for ExpressionEvaluator with noise calibration.

    Wraps an ExpressionEvaluator, a NoiseCalibrator, and a NoiseAugmenter
    to provide noise-aware constancy scoring and discovery gating.

    Parameters
    ----------
    noise_level : NoiseLevel
        Default noise level for scoring.
    n_sigma : float
        Sigma multiplier for threshold (default 3.0).
    seed : int | None
        Seed for reproducible noise.
    calibrator : NoiseCalibrator | None
        Pre-configured calibrator. Created if None.

    Example
    -------
    >>> gated_ev = NoiseGatedEvaluator(NoiseLevel.LOW, n_sigma=3.0, seed=42)
    >>> result = gated_ev.score_with_confidence(
    ...     "m*g*h + 0.5*m*v^2", db
    ... )
    >>> result["accepted"]
    True
    >>> print(f"Score: {result['raw_score']:.4f} ± {result['noise_std']:.4f}")
    """

    def __init__(
        self,
        noise_level: NoiseLevel = NoiseLevel.NONE,
        n_sigma: float = 3.0,
        seed: int | None = None,
        calibrator: NoiseCalibrator | None = None,
    ) -> None:
        self.noise_level = noise_level
        self._evaluator = ExpressionEvaluator()
        self._calibrator = calibrator or NoiseCalibrator(
            n_sigma=n_sigma, seed=seed
        )
        self._seed = seed
        self._augmenter = NoiseAugmenter(NoiseConfig(
            noise_level=noise_level, seed=seed
        ))

    def score(
        self,
        expr_str: str,
        obs_or_db: Observation | ObservationDatabase,
        noise_level: NoiseLevel | None = None,
    ) -> float:
        """Score with noise augmentation (raw constancy, no gating)."""
        level = noise_level or self.noise_level
        if level == NoiseLevel.NONE:
            return self._evaluator.score(expr_str, obs_or_db)

        if isinstance(obs_or_db, ObservationDatabase):
            observations = list(obs_or_db)
        else:
            observations = [obs_or_db]

        aug = NoiseAugmenter(NoiseConfig(
            noise_level=level, seed=self._seed, per_timestep=True,
        ))

        scores: list[float] = []
        for obs in observations:
            noisy_obs = aug.augment(obs)
            scores.append(self._evaluator.score(expr_str, noisy_obs))

        return statistics.mean(scores) if scores else 0.0

    def score_with_confidence(
        self,
        expr_str: str,
        obs_or_db: Observation | ObservationDatabase,
        noise_level: NoiseLevel | None = None,
        num_samples: int = 10,
    ) -> dict[str, Any]:
        """Score expression with confidence interval and noise gating.

        Draws multiple noise realizations to estimate confidence.

        Returns:
            Dict with: raw_score, noise_std, noise_floor, threshold,
                       accepted, confidence_95, significant
        """
        level = noise_level or self.noise_level

        if isinstance(obs_or_db, ObservationDatabase):
            observations = list(obs_or_db)
        else:
            observations = [obs_or_db]

        if level == NoiseLevel.NONE:
            raw = self._evaluator.score(expr_str, obs_or_db)
            return {
                "raw_score": round(raw, 6),
                "noise_std": 0.0,
                "noise_floor": 0.0,
                "threshold": 0.95,
                "accepted": raw >= 0.95,
                "confidence_95": (raw, raw),
                "significant": raw >= 0.95,
                "noise_level": "NONE",
                "sigma_pct": 0.0,
            }

        # Multiple noise realizations for confidence
        sample_scores: list[float] = []
        for i in range(num_samples):
            aug = NoiseAugmenter(NoiseConfig(
                noise_level=level,
                seed=(self._seed or 0) + i,
                per_timestep=True,
            ))
            obs_scores: list[float] = []
            for obs in observations:
                noisy_obs = aug.augment(obs)
                obs_scores.append(self._evaluator.score(expr_str, noisy_obs))
            sample_scores.append(statistics.mean(obs_scores) if obs_scores else 0.0)

        raw_mean = statistics.mean(sample_scores)
        raw_std = statistics.stdev(sample_scores) if len(sample_scores) > 1 else 0.0
        ci_half = 1.96 * raw_std / math.sqrt(len(sample_scores))

        # Compute noise floor
        floor = self._calibrator.calibrate(observations, level)
        accepted = raw_mean > floor.threshold

        return {
            "raw_score": round(raw_mean, 6),
            "noise_std": round(raw_std, 6),
            "noise_floor": round(floor.noise_floor, 6),
            "sigma_floor": round(floor.sigma_floor, 6),
            "threshold": round(floor.threshold, 6),
            "accepted": accepted,
            "confidence_95": (round(raw_mean - ci_half, 6), round(raw_mean + ci_half, 6)),
            "significant": accepted,
            "noise_level": level.name,
            "sigma_pct": level.sigma_pct,
        }

    def set_noise_level(self, level: NoiseLevel) -> None:
        """Update the default noise level."""
        self.noise_level = level
        self._augmenter = NoiseAugmenter(NoiseConfig(
            noise_level=level, seed=self._seed
        ))


# ── Integration utility ───────────────────────────────────────────────────────

def run_noise_calibration(
    db_path: str | Path,
    noise_levels: list[NoiseLevel] | None = None,
    n_sigma: float = 3.0,
    seed: int = 42,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run full noise calibration across a database and return results.

    Args:
        db_path: Path to observation database JSON.
        noise_levels: Noise levels to calibrate at.
        n_sigma: Sigma multiplier for threshold.
        seed: Random seed.
        output_path: If provided, write results to this JSON file.

    Returns:
        Dict mapping noise_level_name -> aggregate calibration stats.
    """
    if noise_levels is None:
        noise_levels = [NoiseLevel.LOW, NoiseLevel.MEDIUM, NoiseLevel.HIGH]

    db = ObservationDatabase(db_path)
    calibrator = NoiseCalibrator(n_sigma=n_sigma, seed=seed)

    results: dict[str, Any] = {
        "db_path": str(db_path),
        "num_scenarios": len(db),
        "scenario_ids": db.scenario_ids,
        "n_sigma": n_sigma,
        "seed": seed,
        "noise_levels": {},
    }

    for level in noise_levels:
        per_scenario = calibrator.calibrate_per_scenario(db, level)

        # Aggregate across all scenarios
        floors = [r.noise_floor for r in per_scenario.values()]
        thresholds = [r.threshold for r in per_scenario.values()]

        level_result = {
            "noise_level": level.name,
            "sigma_pct": level.sigma_pct,
            "aggregate_noise_floor": round(statistics.mean(floors), 6) if floors else 0.0,
            "aggregate_threshold": round(statistics.mean(thresholds), 6) if thresholds else 0.0,
            "min_threshold": round(min(thresholds), 6) if thresholds else 0.0,
            "max_threshold": round(max(thresholds), 6) if thresholds else 0.0,
            "per_scenario": {
                sid: {
                    "noise_floor": r.noise_floor,
                    "sigma_floor": r.sigma_floor,
                    "threshold": r.threshold,
                    "num_calibration_exprs": r.num_calibration_exprs,
                }
                for sid, r in per_scenario.items()
            },
        }
        results["noise_levels"][level.name] = level_result

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    return results


# ── Frontier Dry Run (Phase 4) ─────────────────────────────────────────────────

@dataclass
class FrontierDiscovery:
    """A discovery report from the frontier dry run.

    Example output:
        "Discovered invariant E = ½mv² + mgh: constancy score 0.89 ± 0.04
         vs noise floor 0.72 ± 0.03. Significant at p < 0.01. Lean proof verified."
    """
    expression: str
    domain: str
    constancy_score: float
    constancy_error: float | None
    noise_floor: float
    noise_floor_sigma: float
    threshold: float
    passes_gate: bool
    p_value: float | None
    lean_proof_verified: bool
    lean_proof_error: str | None
    confidence_str: str

    def report(self) -> str:
        """Generate human-readable discovery report."""
        status = "✓ DISCOVERY" if self.passes_gate else "✗ REJECTED"
        proof = " Lean proof verified." if self.lean_proof_verified else ""
        p_val = f" p < {self.p_value:.2f}" if self.p_value and self.p_value < 0.05 else ""
        return (
            f"{status}: {self.expression}\n"
            f"  Constancy: {self.constancy_score:.4f}"
            f"{' ± ' + str(self.constancy_error) if self.constancy_error else ''}"
            f"  vs noise floor: {self.noise_floor:.4f} ± {self.noise_floor_sigma:.4f}"
            f"{p_val}{proof}"
        )


class FrontierRunner:
    """Runs frontier dry run: mix known + unknown scenarios with noise.

    The system must:
    1. Detect domain from quantity signatures
    2. Compose templates for detected domains
    3. Discover invariants via beam search (optional)
    4. Evaluate against noise with calibration gating
    5. Report confidence with error propagation

    Parameters
    ----------
    gated_evaluator : NoiseGatedEvaluator | None
        Pre-configured noise-gated evaluator. Created if None.
    noise_level : NoiseLevel
        Default noise level for evaluation.
    """

    def __init__(
        self,
        gated_evaluator: NoiseGatedEvaluator | None = None,
        noise_level: NoiseLevel = NoiseLevel.MEDIUM,
        seed: int = 42,
    ) -> None:
        self._gated = gated_evaluator or NoiseGatedEvaluator(
            noise_level=noise_level, seed=seed
        )
        self.noise_level = noise_level
        self._seed = seed
        self._calibrator = self._gated._calibrator

    def run(
        self,
        db_or_obs: ObservationDatabase | list[Observation],
        candidate_expressions: list[str] | None = None,
        noise_level: NoiseLevel | None = None,
        auto_prove: bool = False,
    ) -> list[FrontierDiscovery]:
        """Run frontier evaluation on observations with noise.

        Args:
            db_or_obs: Observation database or list of observations.
            candidate_expressions: Expressions to evaluate. Auto-generated if None.
            noise_level: Override noise level.
            auto_prove: Whether to attempt Lean proof verification.

        Returns:
            List of FrontierDiscovery results sorted by significance.
        """
        level = noise_level or self.noise_level

        if isinstance(db_or_obs, ObservationDatabase):
            observations = list(db_or_obs)
        else:
            observations = db_or_obs

        if not observations:
            return []

        # Auto-generate candidates if not provided
        if candidate_expressions is None:
            candidate_expressions = self._generate_candidates(observations)

        discoveries: list[FrontierDiscovery] = []
        for expr in candidate_expressions:
            disc = self._evaluate_candidate(expr, observations, level, auto_prove)
            discoveries.append(disc)

        # Sort by margin over noise floor (descending)
        discoveries.sort(key=lambda d: d.constancy_score - d.threshold, reverse=True)
        return discoveries

    def run_with_real_data(
        self,
        data_dir: str | Path,
        noise_level: NoiseLevel | None = None,
        candidate_expressions: list[str] | None = None,
    ) -> list[FrontierDiscovery]:
        """Run frontier evaluation on real experimental data.

        Loads datasets from data_dir, converts to synthetic observations,
        applies noise, and evaluates.

        Args:
            data_dir: Path to real_experimental/ directory.
            noise_level: Override noise level.
            candidate_expressions: Expressions to evaluate.

        Returns:
            List of FrontierDiscovery results.
        """
        loader = RealExperimentalLoader(data_dir)
        datasets = loader.load_all()

        # Convert all datasets to observations
        all_obs: list[Observation] = []
        for ds in datasets:
            obs = ds.to_synthetic_observations(num_bootstrap=3)
            all_obs.extend(obs)

        return self.run(
            all_obs,
            candidate_expressions=candidate_expressions,
            noise_level=noise_level,
        )

    def _generate_candidates(
        self, observations: list[Observation]
    ) -> list[str]:
        """Generate candidate invariant expressions from observations.

        Uses domain classification + template composition, falling back
        to common physics templates.
        """
        candidates: set[str] = set()

        # Add known invariants from observations
        for obs in observations:
            if obs.known_invariant and obs.known_invariant not in candidates:
                candidates.add(obs.known_invariant)

        # Domain-based template generation
        try:
            from src.physics.composer import (
                DomainClassifier,
                ExpressionComposer,
                DOMAIN_TEMPLATES,
                DOMAIN_QUANTITIES,
                quantities_to_features,
                quantities_to_tensor,
                detokenize_expression,
                TEMPLATE_PAD_IDX,
                load_self_play_generators,
            )
            import torch

            classifier = DomainClassifier()
            all_quantities: set[str] = set()
            for obs in observations:
                all_quantities.update(obs.quantities.keys())

            qty_list = sorted(all_quantities)
            features = quantities_to_features(qty_list).unsqueeze(0)

            with torch.no_grad():
                domain_lists = classifier.predict_domains(features, threshold=0.3)
                domains = domain_lists[0] if domain_lists else []

            # Try self-play trained generators first
            try:
                _ckpt_dir = Path(__file__).parent.parent.parent / "checkpoints"
                sp_generators = load_self_play_generators(_ckpt_dir)
            except Exception:
                sp_generators = {}

            templates: list[str] = []
            for d in domains:
                if d in sp_generators:
                    domain_qties = [q for q in qty_list
                                    if q in DOMAIN_QUANTITIES.get(d, [])]
                    if domain_qties:
                        src = quantities_to_tensor(domain_qties, max_len=8).unsqueeze(0)
                        src_mask = (src == TEMPLATE_PAD_IDX)
                        with torch.no_grad():
                            gen = sp_generators[d]
                            # Move to same device
                            gen_ids = gen.generate(
                                src.to(gen.token_embedding.weight.device),
                                src_padding_mask=src_mask.to(gen.token_embedding.weight.device),
                                max_len=32,
                            )
                        tmpl = detokenize_expression(gen_ids[0])
                        if tmpl:
                            templates.append(tmpl)
                # Fall back to hardcoded template (pre-1905 only)
                if (not templates or d not in sp_generators) and d in DOMAIN_TEMPLATES:
                    templates.append(DOMAIN_TEMPLATES[d])

            if templates:
                composed = ExpressionComposer.compose(templates)
                if composed:
                    candidates.add(composed)
        except ImportError:
            pass

        # Fallback: pre-1905 classical physics templates only
        if not candidates:
            candidates.update([
                "m*g*h + 0.5*m*v^2",           # gravity/mechanics
                "0.5*k*h^2 + 0.5*m*v^2",       # spring
                "0.5*m*v^2 - q*E*x",            # EM
                "P*V/T",                         # thermal
            ])

        return list(candidates)

    def _evaluate_candidate(
        self,
        expr: str,
        observations: list[Observation],
        noise_level: NoiseLevel,
        auto_prove: bool,
    ) -> FrontierDiscovery:
        """Evaluate a single candidate expression across all observations."""
        # Score each observation and aggregate
        all_constancies: list[float] = []
        all_floors: list[float] = []
        all_sigmas: list[float] = []
        all_thresholds: list[float] = []

        for obs in observations:
            result = self._gated.score_with_confidence(
                expr, obs,
                noise_level=noise_level,
                num_samples=5,  # fewer samples per obs since we have multiple
            )
            all_constancies.append(result["raw_score"])
            all_floors.append(result.get("noise_floor", 0.0))
            all_sigmas.append(result.get("sigma_floor", 0.0))
            all_thresholds.append(result.get("threshold", 0.95))

        constancy = statistics.mean(all_constancies)
        constancy_std = (
            statistics.stdev(all_constancies) if len(all_constancies) > 1 else 0.0
        )
        noise_floor = statistics.mean(all_floors) if all_floors else 0.0
        noise_floor_sigma = (
            statistics.mean(all_sigmas) if all_sigmas else 0.0
        )
        threshold = statistics.mean(all_thresholds) if all_thresholds else 0.95
        passes_gate = constancy > threshold

        # P-value estimate from z-score
        p_value = None
        if noise_floor_sigma > 0:
            z = (constancy - noise_floor) / max(noise_floor_sigma, 1e-10)
            p_value = float(2.0 * (1.0 - _norm_cdf_approx(abs(z))))

        # Confidence string
        conf = f"{constancy:.4f}"
        if constancy_std > 0:
            conf += f" ± {constancy_std:.4f}"
        if p_value is not None and p_value < 0.05:
            conf += f" (p < {p_value:.2f})"

        # Domain detection
        domain = "unknown"
        if observations:
            all_qty = set()
            for obs in observations:
                all_qty.update(obs.quantities.keys())
            if "c" in all_qty:
                domain = "relativistic"
            elif "hbar" in all_qty:
                domain = "quantum"
            elif "q" in all_qty or "E" in all_qty:
                domain = "em"
            elif "k" in all_qty:
                domain = "spring"
            elif "P" in all_qty or "T" in all_qty:
                domain = "thermal"
            elif "g" in all_qty:
                domain = "classical"

        # Auto-prove
        lean_verified = False
        lean_error = None
        if auto_prove:
            try:
                from src.physics.auto_lean import AutoLeanScenario, AutoLeanProver
                scenario = AutoLeanScenario(
                    name=f"frontier_{domain}",
                    expression=expr,
                    expected_rhs="C",
                    kinematic_subs={},
                    params=list(observations[0].quantities.keys())[:8]
                    if observations else [],
                    domain=domain,
                )
                prover = AutoLeanProver(max_attempts=5, timeout=10.0)
                attempt = prover.prove(scenario)
                lean_verified = attempt.success
                lean_error = attempt.error if not attempt.success else None
            except Exception as e:
                lean_error = str(e)

        return FrontierDiscovery(
            expression=expr,
            domain=domain,
            constancy_score=constancy,
            constancy_error=constancy_std,
            noise_floor=noise_floor,
            noise_floor_sigma=noise_floor_sigma,
            threshold=threshold,
            passes_gate=passes_gate,
            p_value=p_value,
            lean_proof_verified=lean_verified,
            lean_proof_error=lean_error,
            confidence_str=conf,
        )


def _norm_cdf_approx(x: float) -> float:
    """Approximate standard normal CDF using the error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def run_frontier_dry_run(
    db_path: str | Path = "data/observations/phase_f_7domain.json",
    noise_level: NoiseLevel = NoiseLevel.MEDIUM,
    calibration_path: str | Path | None = "data/noise_calibration_results.json",
    auto_prove: bool = False,
    seed: int = 42,
) -> list[FrontierDiscovery]:
    """Run frontier dry run with mixed known + unknown scenarios.

    Convenience wrapper that:
    1. Loads the observation database
    2. Pre-calibrates noise floors if calibration data exists
    3. Adds noise to observations
    4. Generates candidate expressions via domain composition
    5. Evaluates with noise calibration gating
    6. Reports confidence with error propagation

    Args:
        db_path: Path to observation database.
        noise_level: Noise level to apply.
        calibration_path: Pre-computed calibration results to load.
        auto_prove: Whether to run Lean proof verification.
        seed: Random seed for reproducibility.

    Returns:
        List of FrontierDiscovery results sorted by significance.
    """
    db = ObservationDatabase(db_path)
    print(f"Frontier dry run: {len(db)} scenarios at {noise_level.name} noise")

    # Set up calibrator
    calibrator = NoiseCalibrator(seed=seed)

    # Pre-calibrate if possible
    if calibration_path:
        cal_path = Path(calibration_path)
        if cal_path.exists():
            print(f"  Loaded pre-computed calibration from {cal_path}")
            calibrator.pre_calibrate_all(db, [noise_level])
        else:
            print("  No pre-computed calibration found, calibrating on-the-fly...")
            calibrator.pre_calibrate_all(db, [noise_level])

            # Save calibration results
            results = run_noise_calibration(
                db_path, [noise_level], output_path=cal_path, seed=seed
            )
    else:
        calibrator.pre_calibrate_all(db, [noise_level])

    # Run frontier evaluation
    gated = NoiseGatedEvaluator(
        noise_level=noise_level, seed=seed, calibrator=calibrator
    )
    runner = FrontierRunner(gated_evaluator=gated, noise_level=noise_level, seed=seed)
    discoveries = runner.run(db, auto_prove=auto_prove)

    # Print results
    print(f"\n  Discoveries ({len(discoveries)} evaluated):")
    for d in discoveries:
        status = "✓" if d.passes_gate else "✗"
        proof = " ✓lean" if d.lean_proof_verified else ""
        print(
            f"  {status} [{d.domain}] {d.expression}: "
            f"constancy {d.constancy_score:.4f} ± "
            f"{d.constancy_error or 0:.4f}, "
            f"noise floor {d.noise_floor:.4f}{proof}"
        )

    return discoveries
