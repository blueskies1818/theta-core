#!/usr/bin/env python3
"""Extended ERA GATE v3 — evaluation with HiddenVariableProposer v3.

Tests all 10 scenarios (8 post-1905 + 2 classical) using the v3 model
which emits expression fragments (e.g. "gamma = 1/sqrt(1 - beta^2)")
instead of just variable names.

Key improvements over v2:
  - Proposer emits defining expressions, not just (type, transform) pairs
  - Time dilation: model learns "gamma = 1/sqrt(1 - beta^2)" directly
  - All 7 previous discoveries should still pass (no regression)

RUN: python scripts/extended_era_gate_v3.py
OUTPUTS:
  data/era_gate_template_results.json
  docs/reports/era_gate_template.md
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.physics.dimensions import Dimension
from src.physics.evaluator import ExpressionEvaluator
from src.physics.hidden_variables import (
    HiddenVariableProposer, HiddenVariableDiscovery, DiscoveryResult,
    HiddenVariableProposal, ErrorShapeDetector, ErrorShapeAnalysis,
    load_hidden_var_proposer,
)
from src.physics.noise import (
    NoiseLevel, NoiseConfig, NoiseCalibrator, NoiseFloorResult,
)
from src.physics.observations import Observation, ObservationDatabase
from src.physics.search import ExpressionSearch, SearchResult

# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════

CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "hidden_var_proposer_v3.pt"
RESULTS_PATH = PROJECT_ROOT / "data" / "era_gate_template_results.json"
REPORT_PATH = PROJECT_ROOT / "docs" / "reports" / "era_gate_template.md"
DISCOVERY_THRESHOLD = 0.90
SEED = 42
N_SIGMA = 3.0


# ═══════════════════════════════════════════════════════════════════════════
# Dimension helpers
# ═══════════════════════════════════════════════════════════════════════════

def _dim_from_str(dim_str: str) -> Dimension:
    try:
        return Dimension.named(dim_str)
    except (ValueError, KeyError):
        pass

    handlers: dict[str, Callable[[], Dimension]] = {
        "Action": lambda: Dimension.named("Energy") * Dimension.named("Time"),
        "Momentum": lambda: Dimension.named("Mass") * Dimension.named("Velocity"),
        "Frequency": lambda: Dimension.scalar() / Dimension.named("Time"),
        "InverseLength": lambda: Dimension.scalar() / Dimension.named("Length"),
        "Energy*Time": lambda: Dimension.named("Energy") * Dimension.named("Time"),
    }
    if dim_str in handlers:
        return handlers[dim_str]()

    scalar_names = {"Scalar", "Angle", "Charge", "Dimensionless", "Number",
                    "Voltage", "Dimensionless"}
    if dim_str in scalar_names or dim_str.startswith("Force"):
        return Dimension.scalar()

    try:
        return Dimension.named(dim_str)
    except (ValueError, KeyError):
        return Dimension.scalar()


# ═══════════════════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ExtendedResult:
    scenario_id: str
    scenario_name: str
    domain: str
    known_invariant: str | None
    hidden_variable_hidden: str
    category: str
    baseline_discovered: bool = False
    baseline_expression: str = ""
    baseline_score: float = 0.0
    error_shape: str = ""
    shape_confidence: float = 0.0
    mean_cv: float = 0.0
    proposals: list[dict[str, Any]] = field(default_factory=list)
    num_proposals_tried: int = 0
    top_proposal_type: str = ""
    top_proposal_transform: str = ""
    top_proposal_expression: str = ""  # v3
    top_proposal_confidence: float = 0.0
    augmented_discovered: bool = False
    best_expression: str = ""
    best_score: float = 0.0
    noise_floor: float = 0.0
    noise_threshold: float = 0.0
    passes_noise_gate: bool = False
    closed_loop_success: bool = False
    errors: list[str] = field(default_factory=list)
    timing_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "scenario_name": self.scenario_name,
            "domain": self.domain,
            "category": self.category,
            "known_invariant": self.known_invariant,
            "hidden_variable_hidden": self.hidden_variable_hidden,
            "baseline": {"discovered": self.baseline_discovered, "expression": self.baseline_expression, "score": self.baseline_score},
            "error_analysis": {"shape": self.error_shape, "confidence": self.shape_confidence, "mean_cv": self.mean_cv},
            "proposal": {"type": self.top_proposal_type, "transform": self.top_proposal_transform, "expression": self.top_proposal_expression, "confidence": self.top_proposal_confidence, "num_tried": self.num_proposals_tried, "all": self.proposals},
            "augmented": {"discovered": self.augmented_discovered, "expression": self.best_expression, "score": self.best_score},
            "noise_gate": {"floor": self.noise_floor, "threshold": self.noise_threshold, "passes": self.passes_noise_gate},
            "closed_loop_success": self.closed_loop_success,
            "errors": self.errors,
            "timing_seconds": self.timing_seconds,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Synthetic Data Generators
# ═══════════════════════════════════════════════════════════════════════════

def make_angular_momentum_obs() -> list[Observation]:
    E0 = 13.6
    timesteps = []
    for n_val in [1, 2, 3, 4, 5]:
        E = E0 * n_val * n_val
        for rep in range(5):
            timesteps.append({"t": float(n_val + rep * 0.01), "E": E, "n": float(n_val)})
    return [Observation(id="ang_momentum_all", name="Angular momentum E ∝ n²",
        description=f"Energy E = {E0}*n² for n=1..5", quantities={"E": "Energy"},
        parameters={"E0": E0}, timesteps=timesteps, known_invariant="E / n^2", lean_theorem="")]


def make_spin_measurement_obs() -> list[Observation]:
    E0 = 2.0
    timesteps = []
    n_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    for n_val in n_values:
        E = E0 * n_val + 0.1
        for rep in range(3):
            timesteps.append({"t": float(n_val + rep * 0.01), "E": E, "n": n_val})
    return [Observation(id="spin_measurement_all", name="Spin measurement E ∝ n",
        description="Energy E ∝ n across spin states", quantities={"E": "Energy"},
        parameters={"E0": E0}, timesteps=timesteps, known_invariant="E / n", lean_theorem="")]


def make_blackbody_obs() -> list[Observation]:
    kb = 8.617333262e-5
    timesteps = []
    temperatures = [3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    for i, T in enumerate(temperatures):
        E_photon = kb * T
        for rep in range(3):
            timesteps.append({"t": float(i), "E_photon": E_photon, "T": float(T)})
    return [Observation(id="blackbody_all", name="Blackbody E_photon ∝ T",
        description="E_photon = k*T. Invariant: E_photon/T", quantities={"E_photon": "Energy", "T": "Scalar"},
        parameters={"kb": kb}, timesteps=timesteps, known_invariant="E_photon / T", lean_theorem="")]


def make_photoelectric_obs() -> list[Observation]:
    """v3: h is a quantity, phi is hidden. Tests expression fragment discovery."""
    h_val = 4.135667662e-15
    phi_val = 2.3
    timesteps = []
    frequencies = [6e14, 8e14, 1e15, 1.2e15, 1.4e15, 1.6e15]
    for i, f in enumerate(frequencies):
        K_max = max(0.01, h_val * f - phi_val)
        for rep in range(3):
            timesteps.append({"t": float(i), "K_max": K_max, "f": f, "h": h_val, "phi": phi_val})
    return [Observation(id="photoelectric_all",
        name="Photoelectric K_max = h*f - φ (h known, φ hidden)",
        description=f"K_max = {h_val}*f - {phi_val}. Invariant: h*f - K_max = φ",
        quantities={"K_max": "Scalar", "f": "Scalar", "h": "Scalar"},
        parameters={"phi": phi_val}, timesteps=timesteps,
        known_invariant="h*f - K_max", lean_theorem="")]


def make_simple_relativistic_obs() -> list[Observation]:
    """Simple relativistic energy: E = gamma * E0. Hide gamma."""
    E0 = 938.0
    timesteps = []
    gammas = [1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
    for i, gamma_val in enumerate(gammas):
        E = E0 * gamma_val
        for rep in range(3):
            timesteps.append({"t": float(i), "E": E, "gamma": gamma_val})
    return [Observation(id="rel_energy_all",
        name="Relativistic energy E = gamma*E0",
        description="E = gamma * 938MeV. Invariant: E/gamma = E0",
        quantities={"E": "Energy"}, parameters={"E0": E0},
        timesteps=timesteps, known_invariant="E / gamma", lean_theorem="")]


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _strip_key_from_observation(obs: Observation, key: str) -> Observation:
    new_quantities = {k: v for k, v in obs.quantities.items() if k != key}
    return Observation(id=obs.id, name=obs.name, description=obs.description,
        quantities=new_quantities, parameters=obs.parameters, timesteps=obs.timesteps,
        known_invariant=obs.known_invariant, lean_theorem=obs.lean_theorem,
        external_forces=obs.external_forces, phase_regions=obs.phase_regions,
        is_conservative=obs.is_conservative)


def load_db_scenario(db_path, obs_index, domain, hide_key=None):
    db = ObservationDatabase(str(db_path))
    all_obs = list(db)
    obs = all_obs[obs_index]
    quantities = {}
    for qname, dim_str in obs.quantities.items():
        if hide_key and qname == hide_key:
            continue
        quantities[qname] = _dim_from_str(str(dim_str))
    if hide_key:
        stripped = _strip_key_from_observation(obs, hide_key)
        return quantities, [stripped], obs
    return quantities, [obs], obs


def make_beam_search_fn(max_depth=10, max_expansions=20000, discovery_threshold=DISCOVERY_THRESHOLD, target_dim="Energy"):
    class BeamSearchAdapter:
        def __init__(self, search, result):
            self._search = search
            self._result = result
            self.best_expression = result.expression
            self.best_score = result.score
            self.discovered = result.score >= discovery_threshold
            self._scored = search._scored

    def beam_search(quantities, observations):
        search = ExpressionSearch(quantities=quantities, train_observations=observations,
            max_depth=max_depth, max_expansions=max_expansions,
            discovery_threshold=discovery_threshold, target_dim=target_dim,
            min_discovery_depth=1)  # v3: allow single-variable discoveries (hidden vars)
        result = search.run()
        return BeamSearchAdapter(search, result)

    return beam_search


def run_noise_gate(expression, observations):
    calibrator = NoiseCalibrator(n_sigma=N_SIGMA, seed=SEED)
    noise_level = NoiseLevel.LOW
    floor_result = calibrator.calibrate(observations, noise_level)
    primary_obs = observations[0]
    result = calibrator.gated_score(expression, primary_obs, noise_level)
    return (floor_result.noise_floor, floor_result.threshold, result.get("accepted", False))


def _is_trivial_discovery(expr, score, observations=None):
    if score < 0.95 or not expr:
        return False
    import re
    e = expr.replace(" ", "").replace("+-", "-")
    if re.search(r'\d+\*0\.5\*[A-Za-z_]+-[A-Za-z_]+', e):
        return True
    if re.search(r'([A-Za-z_]\w*)-(\1)(?!\w)', e):
        return True
    return False


def test_scenario(scenario_id, scenario_name, domain, category, known_invariant,
                  hidden_var, quantities, observations, proposer, target_dim="Energy",
                  max_depth=10):
    result = ExtendedResult(scenario_id=scenario_id, scenario_name=scenario_name,
        domain=domain, category=category, known_invariant=known_invariant,
        hidden_variable_hidden=hidden_var)
    t0 = time.time()

    beam_fn = make_beam_search_fn(max_depth=max_depth, target_dim=target_dim)

    # Baseline
    scored_exprs = {}
    try:
        base = beam_fn(quantities, observations)
        result.baseline_expression = base.best_expression
        result.baseline_score = base.best_score
        result.baseline_discovered = base.discovered
        scored_exprs = dict(base._scored) if hasattr(base, '_scored') and base._scored else {}
    except Exception as e:
        result.errors.append(f"baseline_search: {e}")
        result.timing_seconds = time.time() - t0
        return result

    is_trivial = _is_trivial_discovery(result.baseline_expression, result.baseline_score, observations)

    if result.baseline_discovered and not is_trivial:
        try:
            nf, nt, passes = run_noise_gate(result.baseline_expression, observations)
            result.noise_floor, result.noise_threshold, result.passes_noise_gate = nf, nt, passes
            result.closed_loop_success = passes
        except Exception as e:
            result.errors.append(f"noise_gate: {e}")
        result.augmented_discovered = result.baseline_discovered
        result.best_expression, result.best_score = result.baseline_expression, result.baseline_score
        result.timing_seconds = time.time() - t0
        return result

    if is_trivial:
        result.baseline_discovered = False
        result.errors.append(f"trivial discovery: {result.baseline_expression}")

    # Error analysis
    detector = ErrorShapeDetector()
    analysis = detector.analyze(scored_exprs, observations)
    result.error_shape = analysis.shape
    result.shape_confidence = analysis.shape_confidence
    result.mean_cv = analysis.mean_cv

    # Propose and re-search (v3: uses expression fragments)
    discovery = HiddenVariableDiscovery(proposer=proposer, max_proposals=5,
        discovery_threshold=DISCOVERY_THRESHOLD)

    try:
        disc_result = discovery.discover(quantities=quantities, observations=observations,
            beam_search_fn=beam_fn, domain=domain, quantity_names=list(quantities.keys()))

        result.num_proposals_tried = disc_result.num_proposals_tried
        result.proposals = [{"type": p.variable_type, "name": p.variable_name,
            "transform": p.transform, "confidence": p.confidence, "rationale": p.rationale,
            "expression_fragment": p.expression_fragment}  # v3
            for p in disc_result.proposals]

        if disc_result.proposals:
            top_p = disc_result.proposals[0]
            result.top_proposal_type = top_p.variable_type
            result.top_proposal_transform = top_p.transform
            result.top_proposal_expression = top_p.expression_fragment
            result.top_proposal_confidence = top_p.confidence

        result.augmented_discovered = disc_result.discovered
        result.best_expression = disc_result.best_expression
        result.best_score = disc_result.best_score

        if disc_result.discovered and _is_trivial_discovery(disc_result.best_expression, disc_result.best_score, observations):
            result.augmented_discovered = False

        if disc_result.discovered and disc_result.best_expression and not _is_trivial_discovery(disc_result.best_expression, disc_result.best_score, observations):
            noise_floor, noise_thr, passes = run_noise_gate(disc_result.best_expression, observations)
            result.noise_floor, result.noise_threshold, result.passes_noise_gate = noise_floor, noise_thr, passes

    except Exception as e:
        import traceback
        result.errors.append(f"discovery: {e}\n{traceback.format_exc()}")

    result.closed_loop_success = (result.augmented_discovered and result.passes_noise_gate
                                   and result.best_score >= DISCOVERY_THRESHOLD)
    result.timing_seconds = time.time() - t0
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    print("=" * 70)
    print("Extended ERA GATE v3 — 8 Post-1905 + 2 Classical Tests")
    print("Expression Fragment Proposer")
    print("=" * 70)

    print("\n[1] Loading HiddenVariableProposer v3...")
    if CHECKPOINT_PATH.exists():
        proposer = load_hidden_var_proposer(str(CHECKPOINT_PATH))
        print(f"    Loaded v3 checkpoint ({proposer.count_parameters()} params)")
    else:
        proposer = HiddenVariableProposer()
        print(f"    Fresh v3 model ({proposer.count_parameters()} params)")

    DB_R = PROJECT_ROOT / "data" / "observations" / "relativity_synthetic.json"
    DB_M = PROJECT_ROOT / "data" / "observations" / "mechanics_synthetic.json"

    scenario_specs = [
        # ── Quantum (4) ──
        {"scenario_id": "angular_momentum", "scenario_name": "Angular momentum E = E0*n^2",
         "domain": "quantum", "category": "quantum", "known_invariant": "E / n^2",
         "hidden_var": "n", "source": "synthetic", "obs_data": make_angular_momentum_obs(),
         "target_dim": "Energy"},
        {"scenario_id": "spin_measurement", "scenario_name": "Spin measurement E = E0*n",
         "domain": "quantum", "category": "quantum", "known_invariant": "E / n",
         "hidden_var": "n", "source": "synthetic", "obs_data": make_spin_measurement_obs(),
         "target_dim": "Energy"},
        {"scenario_id": "blackbody_peak", "scenario_name": "Blackbody E_photon/T = const",
         "domain": "quantum", "category": "quantum", "known_invariant": "E_photon / T",
         "hidden_var": "none", "source": "synthetic", "obs_data": make_blackbody_obs(),
         "target_dim": "Energy"},
        {"scenario_id": "photoelectric", "scenario_name": "Photoelectric K_max = h*f - φ",
         "domain": "quantum", "category": "quantum", "known_invariant": "h*f - K_max",
         "hidden_var": "phi", "source": "synthetic", "obs_data": make_photoelectric_obs(),
         "target_dim": "Scalar"},

        # ── Relativistic (4) ──
        {"scenario_id": "simple_relativistic", "scenario_name": "Relativistic energy E = gamma*E0",
         "domain": "relativistic", "category": "relativistic", "known_invariant": "E / gamma",
         "hidden_var": "gamma", "source": "synthetic", "obs_data": make_simple_relativistic_obs(),
         "target_dim": "Energy"},
        {"scenario_id": "velocity_addition", "scenario_name": "Velocity addition (u'+v)/(1+u'v/c²)",
         "domain": "relativistic", "category": "relativistic",
         "known_invariant": "(u+v) / (1+u*v/c^2)", "hidden_var": "c",
         "source": "database", "db_path": DB_R, "db_index": 20, "hide_key": "c",
         "target_dim": "Velocity", "max_depth": 10},
        {"scenario_id": "relativistic_momentum", "scenario_name": "Relativistic momentum p = gamma*m*v",
         "domain": "relativistic", "category": "relativistic",
         "known_invariant": "E^2 - (p*c)^2", "hidden_var": "gamma",
         "source": "database", "db_path": DB_R, "db_index": 30, "hide_key": "gamma",
         "target_dim": "Energy"},
        {"scenario_id": "time_dilation", "scenario_name": "Time dilation delta_t = gamma*delta_tau",
         "domain": "relativistic", "category": "relativistic",
         "known_invariant": "(c*t)^2 - x^2", "hidden_var": "gamma",
         "source": "database", "db_path": DB_R, "db_index": 0, "hide_key": "gamma",
         "target_dim": "Length", "max_depth": 10},

        # ── Classical (2) ──
        {"scenario_id": "simple_pendulum", "scenario_name": "Simple pendulum",
         "domain": "gravity", "category": "classical",
         "known_invariant": "m*g*h + 0.5*m*v^2", "hidden_var": "none",
         "source": "database", "db_path": DB_M, "db_index": 28, "hide_key": None,
         "target_dim": "Energy"},
        {"scenario_id": "mass_spring", "scenario_name": "Mass-spring",
         "domain": "spring", "category": "classical",
         "known_invariant": "0.5*k*x^2 + 0.5*m*v^2", "hidden_var": "none",
         "source": "database", "db_path": DB_M, "db_index": 36, "hide_key": None,
         "target_dim": "Energy"},
    ]

    results: list[ExtendedResult] = []

    for i, spec in enumerate(scenario_specs):
        sid = spec["scenario_id"]
        print(f"\n[2.{i+1}] Testing: {sid} ({spec['category']})")
        print(f"    Name: {spec['scenario_name']}")
        print(f"    Target dim: {spec.get('target_dim', 'Energy')}")

        if spec["source"] == "synthetic":
            observations = spec["obs_data"]
            first_obs = observations[0]
            quantities = {}
            for qname, dim_str in first_obs.quantities.items():
                quantities[qname] = _dim_from_str(str(dim_str))
        else:
            if spec.get("hide_key"):
                quantities, observations, _ = load_db_scenario(
                    spec["db_path"], spec["db_index"], spec["domain"],
                    hide_key=spec["hide_key"])
            else:
                quantities, observations, _ = load_db_scenario(
                    spec["db_path"], spec["db_index"], spec["domain"], hide_key=None)

        print(f"    Quantities: {list(quantities.keys())}")

        result = test_scenario(
            scenario_id=spec["scenario_id"], scenario_name=spec["scenario_name"],
            domain=spec["domain"], category=spec["category"],
            known_invariant=spec["known_invariant"], hidden_var=spec["hidden_var"],
            quantities=quantities, observations=observations, proposer=proposer,
            target_dim=spec.get("target_dim", "Energy"),
            max_depth=spec.get("max_depth", 10))

        results.append(result)
        status = "✅ VERIFIED" if result.closed_loop_success else "❌ FAILED"
        print(f"    Result: {status}")
        print(f"    Baseline: {result.baseline_expression[:60]} (score={result.baseline_score:.4f})")
        if result.top_proposal_expression:
            print(f"    Proposal: {result.top_proposal_expression[:60]} (conf={result.top_proposal_confidence:.4f})")
        else:
            print(f"    Proposal: {result.top_proposal_type} → {result.top_proposal_transform} (conf={result.top_proposal_confidence:.4f})")
        print(f"    Augmented: {result.best_expression[:60]} (score={result.best_score:.4f})")

    # ── Write results ──
    print(f"\n[3] Writing results...")
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    output = {"experiment": "extended_era_gate_v3_template", "timestamp": time.time(),
        "config": {"discovery_threshold": DISCOVERY_THRESHOLD, "seed": SEED, "n_sigma": N_SIGMA},
        "results": [r.to_dict() for r in results],
        "summary": {
            "total_scenarios": len(results),
            "verified_discoveries": len([r for r in results if r.closed_loop_success]),
            "quantum_verified": len([r for r in results if r.category == "quantum" and r.closed_loop_success]),
            "relativistic_verified": len([r for r in results if r.category == "relativistic" and r.closed_loop_success]),
            "classical_pass": len([r for r in results if r.category == "classical" and r.closed_loop_success]),
        }}
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"    Done: {len(results)} results → {RESULTS_PATH}")

    # ── Summary ──
    q_r_s = len([r for r in results if r.category in ("quantum", "relativistic") and r.closed_loop_success])
    c_s = len([r for r in results if r.category == "classical" and r.closed_loop_success])
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Verified discoveries (quantum+relativistic): {q_r_s}/{len([r for r in results if r.category != 'classical'])}")
    print(f"  Classical false positives: {len([r for r in results if r.category == 'classical' and not r.closed_loop_success])}")
    print(f"  Threshold: {DISCOVERY_THRESHOLD}")
    for r in results:
        status = "✅" if r.closed_loop_success else "❌"
        expr_str = r.top_proposal_expression[:30] if r.top_proposal_expression else "no expression"
        print(f"  {status} {r.scenario_id:<28s} {r.best_expression[:35]:<35s} score={r.best_score:.4f} [{expr_str}]")


if __name__ == "__main__":
    main()
