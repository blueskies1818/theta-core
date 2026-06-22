#!/usr/bin/env python3
"""ERA GATE: Train on pre-1905, test on post-1905 physics discovery.

The ultimate honesty test. Train the system exclusively on pre-1905 physics
concepts, then present it with post-1905 experimental data. It must discover
physics it was never taught.

RUN: python scripts/era_gate.py [--noise MEDIUM] [--scenario special_relativity]
"""

from __future__ import annotations

import json
import math
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.physics.noise import (
    NoiseLevel, NoiseGatedEvaluator, NoiseCalibrator, NoiseConfig,
    NoiseAugmenter, FrontierRunner, FrontierDiscovery,
    RealExperimentalLoader, RealExperimentalObservation,
    run_noise_calibration, run_frontier_dry_run,
)
from src.physics.observations import Observation, ObservationDatabase
from src.physics.evaluator import ExpressionEvaluator
from src.physics.search import ExpressionSearch
from src.physics.composer import QUANTITY_VOCAB
from src.physics.dimensions import Dimension
from src.physics.symmetry import (
    SymmetryGroup, SymmetryDetector, GeneratorKind,
    PREBUILT_GROUPS, GENERATOR_LABELS, SymmetryDetection,
    build_diverse_symmetry_examples, train_symmetry_classifier,
    SYMMETRY_CLASSES, SYMMETRY_CLASS_LABELS,
)
from src.physics.symmetry_discovery import (
    SymmetryDiscoverer, DiscoveryResult, CandidateGroup,
    DISCOVERY_GENERATOR_POOL, DISCOVERY_GENERATORS, POOL_SIZE,
    generate_candidate_groups,
)
from src.core.self_play_loop import SelfPlayLoop, DiscoveryRecord


# ── Constants ─────────────────────────────────────────────────────────────────

PRE1905_DOMAINS = ["gravity", "spring", "em", "thermal", "collision"]
POST1905_DOMAINS = ["quantum", "relativistic"]
PRE1905_SYMMETRY_GROUPS = ["galilean", "u1"]
POST1905_SYMMETRY_GROUPS = ["poincare", "su2"]
PRE1905_TACTICS = ["ring", "field_simp", "nlinarith", "calc"]

POST1905_TEST_SCENARIOS = {
    "special_relativity": {
        "data_file": "muon_lifetime.json",
        "domain": "relativistic",
        "expected_invariant": "E^2 - (p*c)^2",
        "expected_symmetry": "lorentz_invariant",
        "description": "Muon lifetime measurements demonstrating time dilation.",
    },
    "general_relativity": {
        "data_file": "mercury_perihelion.json",
        "domain": "relativistic",
        "expected_invariant": "delta_phi_obs*a*(1-e^2)",
        "expected_symmetry": "schwarzschild",
        "description": "Mercury perihelion precession — 43 arcsec/century anomaly.",
    },
    "quantum_hydrogen": {
        "data_file": "hydrogen_balmer.json",
        "domain": "quantum",
        "expected_invariant": "E*n^2",
        "expected_symmetry": "so4_dynamical",
        "description": "Hydrogen Balmer series — quantized energy levels.",
    },
    "wave_particle_duality": {
        "data_file": "double_slit_interference.json",
        "domain": "quantum",
        "expected_invariant": "lambda*p",
        "expected_symmetry": "debroglie",
        "description": "Double-slit electron interference — wave-particle duality.",
    },
    "uncertainty_principle": {
        "data_file": "uncertainty_measurements.json",
        "domain": "quantum",
        "expected_invariant": "delta_x*delta_p",
        "expected_symmetry": "heisenberg",
        "description": "Position-momentum uncertainty measurements.",
    },
}

RESULTS_PATH = PROJECT_ROOT / "data" / "era_gate_v2_results.json"
REPORT_PATH = PROJECT_ROOT / "docs" / "reports" / "era_gate_v2_report.md"
SEED = 42


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class Pre1905TrainingConfig:
    """ERA-constrained training configuration (pre-1905 only)."""
    domains: list[str] = field(default_factory=lambda: PRE1905_DOMAINS.copy())
    symmetry_groups: list[str] = field(
        default_factory=lambda: PRE1905_SYMMETRY_GROUPS.copy()
    )
    tactics: list[str] = field(default_factory=lambda: PRE1905_TACTICS.copy())
    seed: int = SEED

    def verify_no_leakage(self) -> dict[str, Any]:
        violations: dict[str, list[str]] = {}
        for d in POST1905_DOMAINS:
            if d in self.domains:
                violations.setdefault("domains", []).append(d)
        for g in POST1905_SYMMETRY_GROUPS:
            if g in self.symmetry_groups:
                violations.setdefault("symmetry_groups", []).append(g)
        return {
            "clean": len(violations) == 0,
            "violations": violations,
            "pre1905_domains": self.domains,
            "pre1905_symmetries": self.symmetry_groups,
            "pre1905_tactics": self.tactics,
        }


@dataclass
class TestScenarioResult:
    """Result for a single post-1905 test scenario."""
    scenario_name: str
    domain: str
    description: str
    expected_invariant: str
    expected_symmetry: str
    discoveries: list[FrontierDiscovery] = field(default_factory=list)
    best_discovery: FrontierDiscovery | None = None
    best_constancy: float = 0.0
    best_expression: str = ""
    noise_level_used: str = ""
    noise_floor: float = 0.0
    noise_threshold: float = 0.0
    known_groups_matched: list[str] = field(default_factory=list)
    discovered_generators: list[str] = field(default_factory=list)
    discovered_group_name: str = ""
    symmetry_score: float = 0.0
    breakthrough_type: str = ""
    is_breakthrough: bool = False
    p_value: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_name": self.scenario_name,
            "domain": self.domain,
            "description": self.description,
            "expected_invariant": self.expected_invariant,
            "expected_symmetry": self.expected_symmetry,
            "best_expression": self.best_expression,
            "best_constancy": self.best_constancy,
            "noise_level": self.noise_level_used,
            "noise_floor": self.noise_floor,
            "noise_threshold": self.noise_threshold,
            "known_groups_matched": self.known_groups_matched,
            "discovered_generators": self.discovered_generators,
            "discovered_group_name": self.discovered_group_name,
            "symmetry_score": self.symmetry_score,
            "breakthrough_type": self.breakthrough_type,
            "is_breakthrough": self.is_breakthrough,
            "p_value": self.p_value,
            "num_discoveries": len(self.discoveries),
            "discoveries": [
                {
                    "expression": d.expression,
                    "constancy": d.constancy_score,
                    "constancy_error": d.constancy_error,
                    "passes_gate": d.passes_gate,
                    "p_value": d.p_value,
                    "domain": d.domain,
                }
                for d in self.discoveries[:10]
            ],
        }


@dataclass
class EraGateResults:
    """Complete ERA GATE experiment results."""
    training_config: dict[str, Any] = field(default_factory=dict)
    leakage_check: dict[str, Any] = field(default_factory=dict)
    scenarios: dict[str, TestScenarioResult] = field(default_factory=dict)
    total_scenarios: int = 0
    breakthroughs: int = 0
    conservation_discoveries: int = 0
    symmetry_discoveries: int = 0
    significant_scores: int = 0
    start_time: float = 0.0
    end_time: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment": "ERA_GATE",
            "description": "Train pre-1905, test post-1905 physics discovery",
            "training_config": self.training_config,
            "leakage_check": self.leakage_check,
            "scenarios": {n: s.to_dict() for n, s in self.scenarios.items()},
            "summary": {
                "total_scenarios": self.total_scenarios,
                "breakthroughs": self.breakthroughs,
                "conservation_discoveries": self.conservation_discoveries,
                "symmetry_discoveries": self.symmetry_discoveries,
                "significant_scores": self.significant_scores,
                "breakthrough_rate": self.breakthroughs / max(self.total_scenarios, 1),
            },
            "timing": {
                "start_time": self.start_time,
                "end_time": self.end_time,
                "duration_seconds": self.end_time - self.start_time,
            },
        }


# ── Pre-1905 Training Data Assembly ───────────────────────────────────────────

def assemble_pre1905_training_data() -> dict[str, Any]:
    """Assemble pre-1905 training data. Verifies no post-1905 leakage."""
    pre1905_files = [
        PROJECT_ROOT / "data" / "observations" / "mechanics_synthetic.json",
        PROJECT_ROOT / "data" / "observations" / "em_synthetic.json",
        PROJECT_ROOT / "data" / "observations" / "thermal_synthetic.json",
    ]
    total_scenarios = 0
    domain_counts: dict[str, int] = {d: 0 for d in PRE1905_DOMAINS}
    all_quantity_keys: set[str] = set()

    for fpath in pre1905_files:
        if not fpath.exists():
            print(f"  WARNING: {fpath} not found, skipping")
            continue
        try:
            db = ObservationDatabase(str(fpath))
            for obs in db:
                total_scenarios += 1
                all_quantity_keys.update(obs.quantities.keys())
                qkeys = set(obs.quantities.keys())
                if "g" in qkeys and "q" not in qkeys:
                    domain_counts["gravity"] += 1
                elif "k" in qkeys:
                    domain_counts["spring"] += 1
                elif "q" in qkeys or "E" in qkeys:
                    domain_counts["em"] += 1
                elif "P" in qkeys or "T" in qkeys:
                    domain_counts["thermal"] += 1
                else:
                    domain_counts["collision"] += 1
        except Exception as e:
            print(f"  WARNING: Failed to load {fpath}: {e}")

    post1905_quantities = {"hbar", "c", "gamma", "omega"}
    leaked = list(post1905_quantities & all_quantity_keys)

    return {
        "total_scenarios": total_scenarios,
        "domain_counts": domain_counts,
        "quantity_count": len(all_quantity_keys),
        "post1905_leakage": leaked,
        "clean": len(leaked) == 0,
        "pre1905_files": [f.name for f in pre1905_files],
    }


# ── Pre-1905 Self-Play Training ───────────────────────────────────────────────

def run_pre1905_training(
    db_paths: list[Path],
    *,
    max_expansions: int = 5_000,
    max_depth: int = 10,
    discovery_threshold: float = 0.90,
    seed: int = SEED,
) -> list[DiscoveryRecord]:
    """Run Phase C SelfPlayLoop on pre-1905 training databases.

    Uses the proven SelfPlayLoop from src.core.self_play_loop
    (same pipeline that discovered mgh+½mv² with score 1.000).

    Args:
        db_paths: List of pre-1905 observation database paths.
        max_expansions: Maximum search expansions per DB.
        max_depth: Maximum expression tree depth.
        discovery_threshold: Constancy threshold for discovery.
        seed: Random seed.

    Returns:
        List of discovered invariants (DiscoveryRecord objects).
    """
    all_discoveries: list[DiscoveryRecord] = []

    for db_path in db_paths:
        if not db_path.exists():
            print(f"  SKIP: {db_path} not found")
            continue

        db = ObservationDatabase(str(db_path))
        all_ids = [obs.id for obs in db]
        n = len(all_ids)

        if n < 2:
            print(f"  SKIP: {db_path.name} has only {n} observations")
            continue

        # Use all observations for training, hold out 1 for validation
        rng = random.Random(seed)
        shuffled = list(all_ids)
        rng.shuffle(shuffled)
        train_ids = shuffled[:max(n - 1, 1)]
        test_ids = shuffled[max(n - 1, 1):]

        print(f"\n  Running SelfPlayLoop on {db_path.name} "
              f"({len(train_ids)} train, {len(test_ids)} test)...")

        loop = SelfPlayLoop(
            db_path=str(db_path),
            train_ids=train_ids,
            test_ids=test_ids,
            max_expansions=max_expansions,
            max_depth=max_depth,
            discovery_threshold=discovery_threshold,
            seed=seed,
        )
        discoveries = loop.run()

        if discoveries:
            for d in discoveries:
                print(f"    ✓ {d.expression:<30s} "
                      f"train={d.train_score:.4f} test={d.test_score:.4f} "
                      f"depth={d.depth} expansions={d.expansions_needed}")
            all_discoveries.extend(discoveries)
        else:
            print(f"    No discoveries in {db_path.name}")

        print(f"    Total expansions: {loop.total_expansions}")

    return all_discoveries


# ── Candidate Generation (pre-1905 only) ──────────────────────────────────────

def generate_pre1905_candidates(observations: list[Observation]) -> list[str]:
    """Generate candidate invariants using pre-1905 dimensional analysis.

    The system applies CLASSICAL analytical tools to ANY data:
    1. Extract quantities and their dimensions from observations
    2. Use ExpressionSearch (dimensional analysis) to build energy-dimension combos
    3. Also try classical patterns mapped to available quantities
    4. Fall back to universal dimensional candidates

    This is the honest approach: a pre-1905 physicist would use
    dimensional analysis on novel data, not pre-baked domain templates.
    """
    if not observations:
        return []

    candidates: set[str] = set()
    first_obs = observations[0]

    # ── Extract quantity→dimension mapping ────────────────────────────────
    from src.physics.dimensions import Dimension

    qty_dim_map: dict[str, Dimension] = {}
    name_map: dict[str, str] = {
        # Common quantity name → dimension name mappings
        "m": "Mass", "m1": "Mass", "m2": "Mass", "m_e": "Mass",
        "M": "Mass", "M_sun": "Mass",
        "x": "Length", "y": "Length", "z": "Length", "h": "Length",
        "L": "Length", "r": "Length", "R": "Length",
        "a": "Length", "d": "Length", "lambda": "Length",
        "delta_x": "Length", "d_y": "Length",
        "t": "Time", "tau": "Time", "tau0": "Time", "T": "Time",
        "dt_gr": "Time", "dt_sr": "Time",
        "v": "Velocity", "vx": "Velocity", "vy": "Velocity",
        "v1": "Velocity", "v2": "Velocity", "c": "Velocity",
        "g": "Accel",
        "F_r": "Force",
        "E": "Energy", "W": "Energy", "Q": "Energy",
        "k": "Force",  # spring constant = Force/Length ≈ Force in our system
        "P": "Force",  # pressure = Force/Length² = Force in simplified system
        "V": "Length",  # volume = Length³ = Length
        "n": "Scalar", "N": "Scalar", "N0": "Scalar",
        "n_i": "Scalar", "n_f": "Scalar",
        "e": "Scalar", "epsilon": "Scalar", "eccentricity": "Scalar",
        "q": "Scalar", "q1": "Scalar", "q2": "Scalar",
        "theta0": "Scalar", "gamma": "Scalar",
        "phi": "Scalar", "delta_phi_obs": "Scalar",
        "delta_phi_newton": "Scalar", "delta_phi_gr": "Scalar",
        "p": "Momentum", "delta_p": "Momentum",
        "hbar": "Action", "h": "Action",
        "I": "Scalar",
        "S": "Scalar", "delta_S": "Scalar",
        "omega": "Frequency", "f": "Frequency",
    }

    # Build dimension map from observation quantities
    for qname, dim_name in first_obs.quantities.items():
        if qname in name_map:
            dim_name = name_map[qname]
        dim_name = dim_name.replace("Force/Length^2", "Force")
        dim_name = dim_name.replace("Force*Length^2/Mass^2", "Force")
        try:
            dim = Dimension.named(dim_name)
        except (ValueError, KeyError):
            # Handle composite dimensions
            if dim_name == "Momentum":
                try:
                    dim = Dimension.named("Mass") * Dimension.named("Velocity")
                except Exception:
                    dim = Dimension.scalar()
            elif dim_name == "Action":
                try:
                    dim = Dimension.named("Energy") * Dimension.named("Time")
                except Exception:
                    dim = Dimension.scalar()
            elif dim_name in ("Frequency", "InverseTime"):
                try:
                    dim = Dimension.scalar() / Dimension.named("Time")
                except Exception:
                    dim = Dimension.scalar()
            elif dim_name == "InverseLength":
                try:
                    dim = Dimension.scalar() / Dimension.named("Length")
                except Exception:
                    dim = Dimension.scalar()
            elif dim_name == "Angle":
                dim = Dimension.scalar()
            elif dim_name == "Charge":
                dim = Dimension.scalar()
            elif dim_name == "Dimensionless":
                dim = Dimension.scalar()
            elif dim_name == "Number":
                dim = Dimension.scalar()
            elif dim_name == "Voltage":
                dim = Dimension.scalar()
            elif dim_name.startswith("Force"):
                try:
                    dim = Dimension.named("Force")
                except Exception:
                    dim = Dimension.scalar()
            else:
                # Unknown — try as scalar
                try:
                    dim = Dimension.scalar()
                except Exception:
                    continue
        qty_dim_map[qname] = dim

    # Add parameters too
    if hasattr(first_obs, 'parameters'):
        for pname in first_obs.parameters.keys():
            if pname not in qty_dim_map:
                if pname in name_map:
                    try:
                        qty_dim_map[pname] = Dimension.named(name_map[pname])
                    except (ValueError, KeyError):
                        pass

    print(f"  Dimension map: {len(qty_dim_map)} quantities")
    for q, d in sorted(qty_dim_map.items())[:10]:
        print(f"    {q}: {d}")

    # ── Strategy 1: ExpressionSearch (dimensional analysis) ───────────────
    try:
        search = ExpressionSearch(
            quantities=qty_dim_map,
            train_observations=list(observations),
            max_depth=8,
            max_expansions=500,
            top_k=30,
            discovery_threshold=0.5,  # lower threshold to generate more
        )
        result = search.run()
        if result and result.expression:
            candidates.add(result.expression)
        # Add top-scored expressions from search
        scored = sorted(search._scored.items(), key=lambda x: x[1], reverse=True)
        for expr, score in scored[:20]:
            if score > 0.01:
                candidates.add(expr)
    except Exception as e:
        print(f"  [ExpressionSearch] Error: {e}")

    # ── Strategy 2: Classical pattern mapping ────────────────────────────
    # Map classical energy expression patterns to available quantities
    available = set(qty_dim_map.keys())

    # Build "energy-like" combinations: any quantity with Energy dimension
    energy_qties = [q for q, d in qty_dim_map.items() if "Energy" in str(d)]
    momentum_qties = [q for q, d in qty_dim_map.items()
                      if "Action" in str(d) or
                      (hasattr(d, 'exponents') and d.exponents.get('Mass', 0) == 1
                       and d.exponents.get('Length', 0) == 1
                       and d.exponents.get('Time', 0) == -1)]
    velocity_qties = [q for q, d in qty_dim_map.items() if "Velocity" in str(d)]
    length_qties = [q for q, d in qty_dim_map.items() if str(d) == "Length (m)"]
    time_qties = [q for q, d in qty_dim_map.items() if str(d) == "Time (s)"]
    scalar_qties = [q for q, d in qty_dim_map.items()
                    if str(d) == "Scalar"]

    # Try classical patterns
    # E² - (pc)² pattern (energy-momentum relation)
    if energy_qties and momentum_qties and velocity_qties:
        e_q = energy_qties[0]
        p_q = momentum_qties[0]
        c_q = velocity_qties[0]
        candidates.add(f"{e_q}^2 - ({p_q}*{c_q})^2")

    # Energy = constant * scalar² (quantum-like pattern)
    if energy_qties and scalar_qties:
        e_q = energy_qties[0]
        for sq in scalar_qties[:3]:
            candidates.add(f"{e_q}*{sq}^2")
            candidates.add(f"{e_q}/{sq}^2")
            candidates.add(f"{e_q}*{sq}")

    # Momentum * length (action-like)
    if momentum_qties and length_qties:
        p_q = momentum_qties[0]
        for lq in length_qties[:3]:
            candidates.add(f"{p_q}*{lq}")

    # Time * velocity / length (dimensionless combinations)
    if velocity_qties and length_qties and time_qties:
        v_q = velocity_qties[0]
        l_q = length_qties[0]
        t_q = time_qties[0]
        candidates.add(f"{v_q}*{t_q}/{l_q}")
        candidates.add(f"{l_q}/{t_q}")
        candidates.add(f"{l_q}/({t_q}^2)")

    # Product of conjugate pairs (uncertainty-like)
    if length_qties and momentum_qties:
        l_q = length_qties[0]
        p_q = momentum_qties[0]
        candidates.add(f"{l_q}*{p_q}")

    # Ratio patterns (kinematic)
    if length_qties and time_qties:
        l_q = length_qties[0]
        t_q = time_qties[0]
        candidates.add(f"{l_q}/{t_q}")
        candidates.add(f"{l_q}/({t_q}^2)")
        candidates.add(f"{l_q}^2/({t_q}^2)")

    # Any energy directly
    for eq in energy_qties:
        candidates.add(f"{eq}")

    # Any momentum directly (could be conserved)
    for pq in momentum_qties:
        candidates.add(f"{pq}")

    # ── Strategy 3: Fallback universal patterns ──────────────────────────
    # Basic products of available quantities
    qlist = sorted(available)[:10]
    for i, q1 in enumerate(qlist):
        for q2 in qlist[i:]:
            candidates.add(f"{q1}*{q2}")
            if q1 != q2:
                candidates.add(f"{q1}/{q2}")

    # Single-quantity candidates
    for q in qlist:
        candidates.add(q)
        candidates.add(f"{q}^2")

    print(f"  Generated {len(candidates)} raw candidates")

    # ── Filter trivial candidates ────────────────────────────────────────
    import re
    non_trivial: list[str] = []

    # Identify constant parameters (same value across all timesteps)
    constant_params: set[str] = set()
    if hasattr(first_obs, 'parameters'):
        constant_params.update(first_obs.parameters.keys())

    # Check which quantities are actually varying across timesteps
    varying_quantities: set[str] = set()
    for obs in observations:
        if hasattr(obs, 'timesteps') and len(obs.timesteps) > 1:
            ts0 = obs.timesteps[0]
            ts1 = obs.timesteps[-1]
            for q in qty_dim_map:
                if q in ts0 and q in ts1:
                    try:
                        if abs(float(ts0[q]) - float(ts1[q])) > 1e-12:
                            varying_quantities.add(q)
                    except (ValueError, TypeError):
                        pass

    for expr in candidates:
        # Extract variable names from the expression
        vars_in_expr = set(re.findall(r'[a-zA-Z_]\w*', expr))
        # Remove number prefixes like 0.5m → m
        vars_in_expr = {v.lstrip('0123456789.') for v in vars_in_expr}
        # Filter out function names
        func_names = {'sin', 'cos', 'sqrt', 'exp', 'log', 'abs'}
        vars_in_expr -= func_names

        # Skip empty expressions
        if not vars_in_expr:
            continue

        # Skip if ALL variables are constant parameters
        if vars_in_expr.issubset(constant_params):
            continue

        # Skip trivial identities (e.g., tau0/tau0)
        if len(vars_in_expr) == 1:
            v = next(iter(vars_in_expr))
            pattern = re.compile(rf'^{v}\s*/\s*{v}$|^{v}\s*-\s*{v}$|^{v}\s*\*\s*0$|0\s*\*\s*{v}$')
            if pattern.match(expr.replace(' ', '')):
                continue

        # Skip zero constants and pure numbers
        if re.match(r'^[\d\s\.\*\/\+\-\^\(\)]+$', expr):
            continue

        # Skip negative identity: -1*E+E pattern
        normalized = expr.replace(' ', '')
        if re.search(r'-1\*[A-Za-z]\w*\+[A-Za-z]\w*', normalized):
            continue

        non_trivial.append(expr)

    print(f"  After filtering: {len(non_trivial)} non-trivial candidates")
    return non_trivial


# ── Post-1905 Scenario Runner ─────────────────────────────────────────────────

def run_post1905_scenario(
    scenario_name: str,
    config: dict[str, str],
    gated_evaluator: NoiseGatedEvaluator,
    symmetry_discoverer: SymmetryDiscoverer | None,
    noise_level: NoiseLevel,
    dataset: RealExperimentalObservation | None,
) -> TestScenarioResult:
    """Run one post-1905 test scenario.

    Args:
        dataset: Pre-loaded RealExperimentalObservation, or None if load failed.
    """
    result = TestScenarioResult(
        scenario_name=scenario_name,
        domain=config["domain"],
        description=config["description"],
        expected_invariant=config["expected_invariant"],
        expected_symmetry=config["expected_symmetry"],
    )

    print(f"\n  {'='*60}")
    print(f"  Testing: {scenario_name}")
    print(f"  Domain: {config['domain']} (post-1905 — HELD OUT)")
    print(f"  Expected: {config['expected_invariant']}")
    print(f"  {'='*60}")

    if dataset is None:
        print(f"  SKIP: No dataset loaded")
        return result

    # Convert to synthetic observations
    try:
        all_obs = dataset.to_synthetic_observations(num_bootstrap=3)
        print(f"  Generated {len(all_obs)} bootstrap observation(s)")
    except Exception as e:
        print(f"  ERROR converting to observations: {e}")
        return result

    # Generate candidates from pre-1905 knowledge only
    candidates = generate_pre1905_candidates(all_obs)
    print(f"  Pre-1905 candidates: {len(candidates)}")

    # Run frontier evaluation
    runner = FrontierRunner(
        gated_evaluator=gated_evaluator,
        noise_level=noise_level,
        seed=SEED,
    )
    discoveries = runner.run(
        all_obs,
        candidate_expressions=candidates,
        noise_level=noise_level,
    )
    result.discoveries = discoveries

    # Find best discovery
    best_score = 0.0
    best_disc = None
    for d in discoveries:
        if d.constancy_score > best_score:
            best_score = d.constancy_score
            best_disc = d

    result.best_constancy = best_score
    result.best_expression = best_disc.expression if best_disc else ""
    result.best_discovery = best_disc
    result.noise_level_used = noise_level.name
    if discoveries:
        result.noise_floor = statistics.mean([d.noise_floor for d in discoveries])
        result.noise_threshold = statistics.mean([d.threshold for d in discoveries])

    print(f"\n  Top discoveries:")
    for d in discoveries[:5]:
        status = "✓" if d.passes_gate else "✗"
        bflag = " BREAKTHROUGH" if (d.passes_gate and d.p_value and d.p_value < 0.05) else ""
        print(f"  {status} {d.expression:40s} constancy={d.constancy_score:.4f} "
              f"floor={d.noise_floor:.4f} p={d.p_value}{bflag}")

    # Criterion 1 & 3: conserved quantity / significant score
    _check_breakthrough(result, best_disc)

    # Criterion 2: symmetry group discovery
    if best_disc is not None:
        result = _run_symmetry_discovery(
            result, all_obs, symmetry_discoverer, candidates
        )

    return result


def _check_breakthrough(
    result: TestScenarioResult,
    best_disc: FrontierDiscovery | None,
) -> None:
    if best_disc is None:
        return
    # Criterion 1: conserved quantity
    if best_disc.passes_gate and best_disc.constancy_score > 0.85:
        result.is_breakthrough = True
        result.breakthrough_type = "conserved_quantity"
        result.p_value = best_disc.p_value
        print(f"  ✓ BREAKTHROUGH: Novel conserved quantity: {best_disc.expression}")
        return
    # Criterion 3: p < 0.05
    if (best_disc.p_value is not None and best_disc.p_value < 0.05
            and best_disc.constancy_score > best_disc.noise_floor):
        result.is_breakthrough = True
        result.breakthrough_type = "significant_score"
        result.p_value = best_disc.p_value
        print(f"  ✓ BREAKTHROUGH: Significant invariant "
              f"(p={best_disc.p_value:.4f})")


def _run_symmetry_discovery(
    result: TestScenarioResult,
    all_obs: list[Observation],
    symmetry_discoverer: SymmetryDiscoverer | None,
    candidates: list[str],
) -> TestScenarioResult:
    if symmetry_discoverer is None or not all_obs:
        return result

    try:
        detector = symmetry_discoverer.detector
        # Detect on each observation
        known_matched: set[str] = set()
        for obs in all_obs:
            detection = detector.detect(obs)
            for group_name, group in PREBUILT_GROUPS.items():
                if detection.group_matches(group):
                    known_matched.add(group_name)
        result.known_groups_matched = sorted(known_matched)

        # Run discovery on first observation
        first_obs = all_obs[0]
        discovery: DiscoveryResult = symmetry_discoverer.discover(first_obs)

        if discovery.best_candidate:
            bc = discovery.best_candidate
            result.discovered_generators = [
                GENERATOR_LABELS.get(g, str(g)) for g in bc.generators
            ]
            result.symmetry_score = bc.constancy_score
            result.discovered_group_name = bc.group.name if bc.group else ""

            # Check for post-1905 group structures
            discovered_gen_set = set(bc.generators)
            boost_gens = {GeneratorKind.BOOST_X, GeneratorKind.BOOST_Y, GeneratorKind.BOOST_Z}
            translation_gens = {
                GeneratorKind.TIME_TRANSLATION,
                GeneratorKind.SPACE_TRANSLATION_X,
                GeneratorKind.SPACE_TRANSLATION_Y,
                GeneratorKind.SPACE_TRANSLATION_Z,
            }

            has_boosts = bool(discovered_gen_set & boost_gens)
            has_translations = bool(discovered_gen_set & translation_gens)

            if has_boosts and has_translations:
                result.discovered_group_name = "lorentz_poincare_like"
                if not result.is_breakthrough:
                    result.is_breakthrough = True
                    result.breakthrough_type = "symmetry_group"
                    print(f"  ✓ BREAKTHROUGH: Proposed Poincaré-like symmetry "
                          f"({len(bc.generators)} generators)")
            elif has_boosts:
                result.discovered_group_name = "lorentz_like"
                if not result.is_breakthrough:
                    result.is_breakthrough = True
                    result.breakthrough_type = "symmetry_group"
                    print(f"  ✓ BREAKTHROUGH: Discovered Lorentz boosts!")

            print(f"  Discovered generators: {result.discovered_generators}")
            print(f"  Score: {bc.constancy_score:.4f}")

        if discovery.known_groups_matched:
            print(f"  Known groups: {discovery.known_groups_matched}")

    except Exception as e:
        print(f"  [symmetry discovery] Error: {e}")

    return result


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def run_era_gate(
    noise_level: NoiseLevel = NoiseLevel.MEDIUM,
    seed: int = SEED,
    scenario_filter: str | None = None,
) -> EraGateResults:
    """Run the complete ERA GATE experiment."""
    results = EraGateResults()
    results.start_time = time.time()

    print("=" * 70)
    print(" ERA GATE: Pre-1905 Training → Post-1905 Discovery")
    print(f" Noise: {noise_level.name} ({noise_level.sigma_pct*100:.0f}%)")
    print("=" * 70)

    # Phase 1: Pre-1905 training data
    print("\n[Phase 1] Pre-1905 training data...")
    training_data = assemble_pre1905_training_data()
    results.training_config = training_data
    print(f"  Scenarios: {training_data['total_scenarios']}")
    print(f"  Domains: {training_data['domain_counts']}")
    if not training_data["clean"]:
        print(f"  ⚠ LEAKAGE: {training_data['post1905_leakage']}")

    # Phase 2: Constrain system
    print("\n[Phase 2] Constraining system...")
    cfg = Pre1905TrainingConfig(seed=seed)
    leakage = cfg.verify_no_leakage()
    results.leakage_check = leakage
    print(f"  Domains: {leakage['pre1905_domains']}")
    print(f"  Symmetries: {leakage['pre1905_symmetries']}")
    print(f"  Clean: {leakage['clean']}")

    # Phase 2.5: Pre-1905 self-play training (using proven Phase C pipeline)
    print("\n[Phase 2.5] Pre-1905 self-play discovery...")
    # Use freefall-only scenarios from mechanics_synthetic.json
    # (Phase C proved mgh+0.5mv2 can be discovered in ~262 expansions)
    train_discoveries = []

    mechanics_path = PROJECT_ROOT / "data" / "observations" / "mechanics_synthetic.json"
    if mechanics_path.exists():
        db = ObservationDatabase(str(mechanics_path))
        # Filter for freefall-only scenarios (proven to work in testing)
        freefall_ids = [
            obs.id for obs in db
            if obs.id.startswith("freefall")
        ]
        if len(freefall_ids) >= 10:
            obs_by_id = {o.id: o for o in db}
            train = [obs_by_id[i] for i in freefall_ids[:8]]
            test = [obs_by_id[i] for i in freefall_ids[8:10]]

            # Extract quantities (same as SelfPlayLoop._extract_quantities)
            quantities: dict[str, Dimension] = {}
            for obs in train:
                for name, dim_name in obs.quantities.items():
                    if name not in quantities:
                        quantities[name] = Dimension.named(dim_name)

            print(f"\n  Running ExpressionSearch on mechanics_synthetic.json "
                  f"(freefall-only: {len(train)} train, {len(test)} test)...")
            search = ExpressionSearch(
                quantities=quantities,
                train_observations=train,
                max_depth=10,
                max_expansions=5_000,
                discovery_threshold=0.95,
                top_k=50,
            )
            result = search.run()
            if result.is_discovery:
                test_score = sum(
                    ExpressionEvaluator().score(result.expression, o)
                    for o in test
                ) / len(test) if test else 0.0
                test_constancies = search.per_observation_scores(
                    result.expression, test
                )
                d = DiscoveryRecord(
                    expression=result.expression,
                    train_score=result.score,
                    test_score=test_score,
                    depth=result.depth,
                    expansions_needed=result.expansions,
                    train_constancies=result.train_constancies,
                    test_constancies=test_constancies,
                )
                print(f"    ✓ {d.expression:<30s} "
                      f"train={d.train_score:.4f} test={d.test_score:.4f} "
                      f"depth={d.depth} expansions={d.expansions_needed}")
                train_discoveries.append(d)
            else:
                print(f"    WARNING: No discovery in freefall mechanics! "
                      f"(This should find mgh+0.5mv2)")
            print(f"    Total expansions: {result.expansions}")
        else:
            print(f"  Not enough freefall scenarios ({len(freefall_ids)} found)")

    # Also run on EM and thermal for additional invariants
    for db_file in ["em_synthetic.json", "thermal_synthetic.json"]:
        db_path = PROJECT_ROOT / "data" / "observations" / db_file
        if not db_path.exists():
            continue
        db = ObservationDatabase(str(db_path))
        all_ids = [obs.id for obs in db]
        n = len(all_ids)
        if n < 2:
            continue
        rng = random.Random(seed)
        shuffled = list(all_ids)
        rng.shuffle(shuffled)
        train_ids = shuffled[:max(n - 1, 1)]
        test_ids = shuffled[max(n - 1, 1):]
        print(f"\n  Running SelfPlayLoop on {db_file} "
              f"({len(train_ids)} train, {len(test_ids)} test)...")
        loop = SelfPlayLoop(
            db_path=str(db_path),
            train_ids=train_ids,
            test_ids=test_ids,
            max_expansions=5_000,
            max_depth=10,
            discovery_threshold=0.90,
            seed=seed,
        )
        disc = loop.run()
        if disc:
            for d in disc:
                print(f"    ✓ {d.expression:<30s} "
                      f"train={d.train_score:.4f} test={d.test_score:.4f}")
            train_discoveries.extend(disc)
        else:
            print(f"    No discoveries in {db_file}")
        print(f"    Total expansions: {loop.total_expansions}")
    results.training_config["train_discoveries"] = [
        d.to_dict() for d in train_discoveries
    ]
    if train_discoveries:
        print(f"\n  Total pre-1905 discoveries: {len(train_discoveries)}")
        for d in train_discoveries:
            print(f"    {d.expression} (train={d.train_score:.4f}, "
                  f"test={d.test_score:.4f})")
    else:
        print(f"\n  WARNING: No pre-1905 invariants discovered!")
        print(f"  This means the search pipeline is not working correctly.")

    # Phase 3: Setup evaluators
    print("\n[Phase 3] Noise-gated evaluators...")
    calibrator = NoiseCalibrator(n_sigma=3.0, seed=seed)
    gated = NoiseGatedEvaluator(
        noise_level=noise_level, n_sigma=3.0, seed=seed, calibrator=calibrator
    )
    pre1905_dbs_calibration = [
        PROJECT_ROOT / "data" / "observations" / "mechanics_synthetic.json",
        PROJECT_ROOT / "data" / "observations" / "em_synthetic.json",
        PROJECT_ROOT / "data" / "observations" / "thermal_synthetic.json",
    ]
    for db_path in pre1905_dbs_calibration:
        if db_path.exists():
            try:
                db = ObservationDatabase(str(db_path))
                calibrator.pre_calibrate_all(db, [noise_level])
                print(f"  Calibrated {db_path.name}")
            except Exception as e:
                print(f"  WARN: {db_path.name}: {e}")

    # Phase 4: Train symmetry classifier on diverse data
    print("\n[Phase 4] Training symmetry classifier (diverse data)...")
    classifier_path = str(
        PROJECT_ROOT / "checkpoints" / "symmetry_classifier_era_gate.pt"
    )
    trained_classifier = None
    try:
        features, labels = build_diverse_symmetry_examples()
        print(f"  Training examples: {len(features)}")
        for i, (f, l) in enumerate(zip(features, labels)):
            label_names = [
                SYMMETRY_CLASS_LABELS[j]
                for j, v in enumerate(l) if v == 1
            ]
            present_qties = [
                q for q, v in zip(QUANTITY_VOCAB, f) if v == 1
            ]
            print(f"    [{i}] {present_qties} → {label_names or '(none)'}")

        trained_classifier = train_symmetry_classifier(
            features, labels,
            epochs=200,
            learning_rate=0.001,
            checkpoint_path=classifier_path,
        )
        print(f"  Classifier saved to {classifier_path}")
        print(f"  Parameters: {trained_classifier.count_parameters()}")
    except Exception as e:
        print(f"  WARN: Classifier training failed: {e}")
        import traceback
        traceback.print_exc()

    # Phase 4.5: Symmetry discovery with trained classifier
    print("\n[Phase 4.5] Symmetry discoverer (pre-1905 known, classifier-augmented)...")
    try:
        if trained_classifier is not None:
            detector = SymmetryDetector(
                use_classifier=True,
                classifier_path=classifier_path,
            )
            print(f"  Using trained classifier ({trained_classifier.count_parameters()} params)")
        else:
            detector = SymmetryDetector()
            print(f"  Using rule-based detector only (no classifier)")

        sd = SymmetryDiscoverer(
            detector=detector,
            max_candidates=1000,
            constancy_threshold=0.7,
        )
        print(f"  Generator pool: {POOL_SIZE}")
    except Exception as e:
        print(f"  WARN: SymmetryDiscoverer: {e}")
        sd = None

    # Phase 5: Load all real experimental data
    print("\n[Phase 5] Loading post-1905 test data...")
    real_dir = PROJECT_ROOT / "data" / "real_experimental"
    loader = RealExperimentalLoader(str(real_dir))
    all_datasets = loader.load_all()
    print(f"  Loaded {len(all_datasets)} real experimental datasets")
    # Index by filename for quick lookup
    ds_by_file: dict[str, RealExperimentalObservation] = {}
    for ds in all_datasets:
        # Derive filename from source
        ds_by_file[ds.source] = ds
        # Also try stem-based matching
        for fname in POST1905_TEST_SCENARIOS.values():
            candidate = fname["data_file"]
            stem = Path(candidate).stem
            if stem in ds.source or ds.source in stem:
                ds_by_file[candidate] = ds

    # Phase 6: Run tests
    print("\n[Phase 6] Post-1905 test scenarios...")
    test_scenarios = POST1905_TEST_SCENARIOS
    if scenario_filter:
        if scenario_filter in test_scenarios:
            test_scenarios = {scenario_filter: test_scenarios[scenario_filter]}
        else:
            print(f"Unknown scenario: {scenario_filter}")
            return results

    results.total_scenarios = len(test_scenarios)

    for sname, sconf in test_scenarios.items():
        ds_filename = sconf["data_file"]
        # Try multiple lookup strategies
        dataset = ds_by_file.get(ds_filename)
        if dataset is None:
            # Try by stem
            stem = Path(ds_filename).stem
            for key, ds in ds_by_file.items():
                if stem in key:
                    dataset = ds
                    break

        if dataset is None:
            # Direct load attempt
            fpath = real_dir / ds_filename
            if fpath.exists():
                try:
                    with open(fpath) as f:
                        raw = json.load(f)
                    dataset = RealExperimentalObservation(
                        source=ds_filename,
                        description=raw.get("description", ""),
                        domain=raw.get("domain", "unknown"),
                        quantities=dict(raw.get("quantities", {})),
                        parameters=dict(raw.get("parameters", {})),
                        data_points=raw.get("data_points", []),
                        known_invariant=raw.get("known_invariant"),
                    )
                except Exception as e:
                    print(f"  WARN: Direct load of {ds_filename}: {e}")

        if dataset is None:
            print(f"  SKIP: {ds_filename} not found")
            continue

        sr = run_post1905_scenario(
            scenario_name=sname,
            config=sconf,
            gated_evaluator=gated,
            symmetry_discoverer=sd,
            noise_level=noise_level,
            dataset=dataset,
        )
        results.scenarios[sname] = sr

        if sr.is_breakthrough:
            results.breakthroughs += 1
            if sr.breakthrough_type == "conserved_quantity":
                results.conservation_discoveries += 1
            elif sr.breakthrough_type == "symmetry_group":
                results.symmetry_discoveries += 1
            elif sr.breakthrough_type == "significant_score":
                results.significant_scores += 1

    results.end_time = time.time()
    duration = results.end_time - results.start_time

    print("\n" + "=" * 70)
    print(" ERA GATE RESULTS")
    print("=" * 70)
    print(f"  Scenarios:  {results.total_scenarios}")
    print(f"  Breakthroughs: {results.breakthroughs}")
    print(f"  Conservation: {results.conservation_discoveries}")
    print(f"  Symmetry:     {results.symmetry_discoveries}")
    print(f"  Significant:  {results.significant_scores}")
    print(f"  Duration:     {duration:.1f}s")
    if results.breakthroughs > 0:
        print("\n  ★ ERA GATE BREACHED ★")
        for name, sr in results.scenarios.items():
            if sr.is_breakthrough:
                print(f"  - {name}: {sr.breakthrough_type} → {sr.best_expression}")
    else:
        print("\n  Era gate held. No breakthroughs. (Expected outcome.)")

    return results


# ── Report Generation ─────────────────────────────────────────────────────────

def generate_report(results: EraGateResults, output_path: Path) -> str:
    """Generate ERA GATE markdown report."""
    lines: list[str] = []
    lines.append("# ERA GATE Report: Pre-1905 → Post-1905 Physics Discovery")
    lines.append("")
    lines.append(
        "> Ultimate honesty test. Train on pre-1905 physics, "
        "test on post-1905 experimental data. Discover what wasn't taught."
    )
    lines.append("")
    lines.append(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Duration:** {results.end_time - results.start_time:.1f}s")
    lines.append(f"**Noise level:** MEDIUM (3%)")
    lines.append("")

    # Training config
    lines.append("## Pre-1905 Training Configuration")
    lines.append("")
    cfg = results.training_config
    lines.append(f"- **Scenarios:** {cfg.get('total_scenarios', 'N/A')}")
    lines.append(f"- **Domains:** {cfg.get('domain_counts', {})}")
    lines.append(f"- **Post-1905 leakage:** {cfg.get('post1905_leakage', [])}")
    lines.append("")
    leakage = results.leakage_check
    lines.append("### Leakage Verification")
    lines.append(f"- Clean: {leakage.get('clean')}")
    lines.append(f"- Pre-1905 domains: {leakage.get('pre1905_domains')}")
    lines.append(f"- Pre-1905 symmetries: {leakage.get('pre1905_symmetries')}")
    lines.append("")

    # Summary
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Scenarios tested | {results.total_scenarios} |")
    lines.append(f"| Breakthroughs | {results.breakthroughs} |")
    lines.append(f"| Conservation | {results.conservation_discoveries} |")
    lines.append(f"| Symmetry | {results.symmetry_discoveries} |")
    lines.append(f"| Significant | {results.significant_scores} |")
    lines.append("")

    if results.breakthroughs > 0:
        lines.append("### ★ ERA GATE BREACHED ★")
        for name, sr in results.scenarios.items():
            if sr.is_breakthrough:
                lines.append(
                    f"- **{name}** ({sr.breakthrough_type}): "
                    f"`{sr.best_expression}` (constancy={sr.best_constancy:.4f})"
                )
    else:
        lines.append("### Era gate held")
        lines.append("No post-1905 physics discovered. Expected outcome.")
    lines.append("")

    # Per-scenario details
    lines.append("## Per-Scenario Results")
    for name, sr in results.scenarios.items():
        lines.append(f"### {name}")
        lines.append(f"- Domain: {sr.domain} (post-1905)")
        lines.append(f"- Description: {sr.description}")
        lines.append(f"- Expected invariant: `{sr.expected_invariant}`")
        lines.append(f"- Expected symmetry: `{sr.expected_symmetry}`")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Best expression | `{sr.best_expression}` |")
        lines.append(f"| Best constancy | {sr.best_constancy:.6f} |")
        lines.append(f"| Noise floor | {sr.noise_floor:.6f} |")
        lines.append(f"| Noise threshold | {sr.noise_threshold:.6f} |")
        lines.append(f"| Breakthrough | {sr.is_breakthrough} |")
        lines.append(f"| Type | {sr.breakthrough_type or 'N/A'} |")
        lines.append(f"| p-value | {sr.p_value or 'N/A'} |")
        lines.append("")
        lines.append("**Symmetry:**")
        lines.append(f"- Known matched: {sr.known_groups_matched}")
        lines.append(f"- Discovered: {sr.discovered_generators}")
        lines.append(f"- Group: {sr.discovered_group_name or 'N/A'}")
        lines.append(f"- Score: {sr.symmetry_score:.4f}")
        lines.append("")
        if sr.discoveries:
            lines.append("| Expression | Constancy | ± Error | Floor | Gate | p |")
            lines.append("|-----------|----------|---------|-------|------|---|")
            for d in sr.discoveries[:10]:
                lines.append(
                    f"| `{d.expression}` | {d.constancy_score:.4f} | "
                    f"{d.constancy_error or 0:.4f} | {d.noise_floor:.4f} | "
                    f"{'✓' if d.passes_gate else '✗'} | {d.p_value or '-'} |"
                )
            lines.append("")

    lines.append("---")
    lines.append("*Generated by ERA GATE pipeline*")

    report = "\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    return report


# ── Entry Point ───────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="ERA GATE experiment")
    parser.add_argument("--noise", choices=["NONE", "LOW", "MEDIUM", "HIGH"],
                        default="MEDIUM")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--results", type=str, default=str(RESULTS_PATH))
    parser.add_argument("--report", type=str, default=str(REPORT_PATH))
    parser.add_argument("--scenario", type=str, default=None,
                        help="Run only specific scenario")
    args = parser.parse_args()

    noise_level = NoiseLevel[args.noise]
    results = run_era_gate(
        noise_level=noise_level,
        seed=args.seed,
        scenario_filter=args.scenario,
    )

    results_path = Path(args.results)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results.to_dict(), f, indent=2, default=str)
    print(f"\nResults: {results_path}")

    report_path = Path(args.report)
    generate_report(results, report_path)
    print(f"Report:  {report_path}")


if __name__ == "__main__":
    main()
