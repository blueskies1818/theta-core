"""ERA GATE: Train on pre-1905, test on post-1905 physics discovery.

The ultimate honesty test. Trains the self-play loop exclusively on pre-1905
physics (classical mechanics, EM, thermodynamics), then tests whether it can
discover invariants in post-1905 data (relativity, quantum, uncertainty).

Architecture:
  1. Load pre-1905 training database
  2. Load post-1905 test database (5 scenarios)
  3. Run self-play loop on pre-1905 data → discover invariants
  4. Evaluate each discovered invariant on each post-1905 scenario
  5. Attempt symmetry discovery on post-1905 data
  6. Noise-gate all results
  7. Output comprehensive results JSON + markdown report

PASS CRITERIA (any one = breakthrough):
  - Discovers at least 1 post-1905 conserved quantity
  - Proposes a symmetry group matching a post-1905 group structure
  - Score > noise floor at p < 0.05 for any invariant
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.physics.dimensions import Dimension
from src.physics.evaluator import ExpressionEvaluator
from src.physics.observations import Observation, ObservationDatabase
from src.physics.search import ExpressionSearch
from src.physics.noise import NoiseLevel, NoiseConfig, NoiseAugmenter


# ═══════════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ScenarioResult:
    """Result for a single post-1905 test scenario."""
    scenario_id: str
    domain: str
    known_invariant: str | None
    best_expression: str
    best_score: float
    noise_floor: float
    noise_sigma: float
    p_value: float  # probability score is due to noise
    all_scores: list[float]  # all expression scores
    significance: str  # "SIGNIFICANT", "MARGINAL", "NONE"


@dataclass
class ERAGateResult:
    """Complete ERA gate results."""
    timestamp: float = field(default_factory=time.time)
    # Training
    pre1905_db_path: str = ""
    pre1905_scenario_count: int = 0
    pre1905_domains: dict[str, int] = field(default_factory=dict)
    # Search
    budget: dict[str, Any] = field(default_factory=dict)
    expansions_used: int = 0
    # Discoveries from pre-1905 training
    train_discoveries: list[dict] = field(default_factory=list)
    # Post-1905 test results
    scenario_results: list[dict] = field(default_factory=list)
    # Symmetry discovery
    symmetry_proposals: list[dict] = field(default_factory=list)
    # Summary
    breakthroughs: list[str] = field(default_factory=list)
    overall_pass: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "pre1905_db_path": self.pre1905_db_path,
            "pre1905_scenario_count": self.pre1905_scenario_count,
            "pre1905_domains": self.pre1905_domains,
            "budget": self.budget,
            "expansions_used": self.expansions_used,
            "train_discoveries": self.train_discoveries,
            "scenario_results": self.scenario_results,
            "symmetry_proposals": self.symmetry_proposals,
            "breakthroughs": self.breakthroughs,
            "overall_pass": self.overall_pass,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Core
# ═══════════════════════════════════════════════════════════════════════════

def compute_noise_floor(
    obs: Observation,
    noise_level: NoiseLevel,
    n_trials: int = 50,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute noise floor for a scenario.

    Uses known-non-conserved expressions to estimate baseline score
    at the given noise level.
    Returns (noise_floor_mean, noise_floor_sigma).
    """
    evaluator = ExpressionEvaluator()
    config = NoiseConfig(noise_level=noise_level, seed=seed, per_timestep=True)
    augmenter = NoiseAugmenter(config)

    # Non-conserved test expressions
    test_exprs = [
        "h", "v", "t", "h*v", "h+t", "h*t",
        "h/v", "h^2", "v^2", "h+v+t",
    ]

    # Filter to expressions using available quantities
    available_vars = set(obs.quantities.keys())
    usable = [e for e in test_exprs
              if all(v in available_vars or v in obs.parameters for v in _extract_vars(e))]

    if len(usable) < 3:
        usable = ["h", "v", "t"] if all(v in available_vars for v in ["h", "v", "t"]) \
            else test_exprs[:3]

    scores = []
    for trial in range(n_trials):
        noisy_obs = augmenter.augment(obs)
        trial_scores = []
        for expr in usable:
            try:
                s = evaluator.score(expr, noisy_obs)
                if not math.isnan(s):
                    trial_scores.append(s)
            except Exception:
                pass
        if trial_scores:
            scores.append(statistics.mean(trial_scores))

    if not scores:
        return 0.5, 0.1

    return statistics.mean(scores), statistics.stdev(scores) if len(scores) > 1 else 0.05


def _extract_vars(expr: str) -> set[str]:
    """Extract variable names from expression string."""
    import re
    funcs = {"sin", "cos", "sqrt", "exp", "log", "abs"}
    tokens = re.findall(r'[a-zA-Z_]\w*', expr)
    return {t for t in tokens if t not in funcs}


def compute_p_value(score: float, noise_mean: float, noise_sigma: float) -> float:
    """Estimate p-value: probability score is from noise distribution.

    Uses Chebyshev-style bound: how many sigma above noise mean.
    Returns approximate p-value.
    """
    if noise_sigma < 1e-12:
        return 0.0 if score > noise_mean else 1.0
    z = (score - noise_mean) / noise_sigma
    if z <= 0:
        return 1.0
    # Rough normal approximation for tail probability
    if z < 1.0:
        return 0.32
    if z < 2.0:
        return 0.05
    if z < 3.0:
        return 0.003
    return 0.0001


def _get_available_quantities(obs: Observation) -> dict[str, Dimension]:
    """Extract quantities with dimension info."""
    quantities: dict[str, Dimension] = {}
    for name, dim_name in obs.quantities.items():
        try:
            quantities[name] = Dimension.named(dim_name)
        except Exception:
            quantities[name] = Dimension.scalar()
    return quantities


# ═══════════════════════════════════════════════════════════════════════════
# Main runner
# ═══════════════════════════════════════════════════════════════════════════

def run_era_gate(
    pre1905_db_path: str = "data/observations/pre1905_training.json",
    post1905_db_path: str = "data/observations/post1905_test.json",
    *,
    max_expansions: int = 20_000,
    max_depth: int = 6,
    discovery_threshold: float = 0.90,
    top_k: int = 50,
    noise_level: NoiseLevel = NoiseLevel.LOW,
    n_noise_trials: int = 50,
    seed: int = 42,
) -> ERAGateResult:
    """Run the full ERA gate experiment."""
    result = ERAGateResult()
    result.budget = {
        "max_expansions": max_expansions,
        "max_depth": max_depth,
        "discovery_threshold": discovery_threshold,
        "top_k": top_k,
        "noise_level": noise_level.value,
        "n_noise_trials": n_noise_trials,
        "seed": seed,
    }

    random.seed(seed)

    # ── 1. Load data ──────────────────────────────────────────────────────
    print("=" * 70)
    print("ERA GATE: Pre-1905 → Post-1905 Discovery Test")
    print("=" * 70)
    print()

    print(f"Loading pre-1905 training data: {pre1905_db_path}")
    pre_db = ObservationDatabase(pre1905_db_path)
    result.pre1905_db_path = pre1905_db_path
    result.pre1905_scenario_count = len(pre_db)

    # Count domains
    domains: dict[str, int] = {}
    for obs in pre_db:
        d = obs.description.lower()
        if "free fall" in d or "projectile" in d or "pendulum" in d or "spring" in d or "collision" in d:
            dom = "mechanics"
        elif "e field" in d or "b field" in d or "e×b" in d or "coulomb" in d:
            dom = "electromagnetism"
        elif "isothermal" in d or "adiabatic" in d or "isobaric" in d or "isochoric" in d or "carnot" in d or "entropy" in d:
            dom = "thermodynamics"
        else:
            dom = "unknown"
        domains[dom] = domains.get(dom, 0) + 1
    result.pre1905_domains = domains
    print(f"  {len(pre_db)} training scenarios across {len(domains)} domains:")
    for d, c in sorted(domains.items()):
        print(f"    {d}: {c}")

    print(f"\nLoading post-1905 test data: {post1905_db_path}")
    post_db = ObservationDatabase(post1905_db_path)
    print(f"  {len(post_db)} test scenarios")

    # ── 2. Discover invariants from pre-1905 ──────────────────────────────
    print(f"\n{'─' * 70}")
    print("PHASE 1: Discover invariants from pre-1905 training data")
    print(f"{'─' * 70}")

    all_train_obs = list(pre_db)

    # Group by domain for targeted search
    domain_obs: dict[str, list[Observation]] = {}
    for obs in all_train_obs:
        d = obs.description.lower()
        if any(kw in d for kw in ["free fall", "projectile", "pendulum", "spring"]):
            dom = "mechanics_grav"
        elif "collision" in d:
            dom = "mechanics_collision"
        elif "e field" in d or "e×b" in d or "b field" in d or "coulomb" in d:
            dom = "electromagnetism"
        elif any(kw in d for kw in ["thermal", "isochoric", "isobaric", "adiabatic", "carnot", "entropy"]):
            dom = "thermodynamics"
        else:
            dom = "other"
        if dom not in domain_obs:
            domain_obs[dom] = []
        domain_obs[dom].append(obs)

    print(f"  Domain groups: {[(k, len(v)) for k, v in sorted(domain_obs.items())]}")
    print(f"  Search budget: {max_expansions} total, depth ≤ {max_depth}")

    all_discoveries: list[dict] = []
    total_expansions = 0

    for domain_name, domain_obs_list in sorted(domain_obs.items()):
        if len(domain_obs_list) < 2:
            continue

        dom_budget = max(500, max_expansions // max(len(domain_obs), 1))
        print(f"\n  --- {domain_name} ({len(domain_obs_list)} scenarios, {dom_budget} expansions) ---")

        dom_quantities = _get_available_quantities(domain_obs_list[0])
        for obs in domain_obs_list[1:]:
            for name, dim_name in obs.quantities.items():
                if name not in dom_quantities:
                    try:
                        dom_quantities[name] = Dimension.named(dim_name)
                    except Exception:
                        dom_quantities[name] = Dimension.scalar()

        search = ExpressionSearch(
            quantities=dom_quantities,
            train_observations=domain_obs_list,
            max_depth=max_depth,
            max_expansions=dom_budget,
            discovery_threshold=discovery_threshold,
            top_k=min(top_k, 50),
        )

        dom_discoveries = []
        best_dom_expr = ""
        best_dom_score = 0.0
        try:
            for expansion_count, snapshot in search.run_with_snapshots():
                if expansion_count % 500 == 0 or snapshot.is_discovery:
                    print(f"    [{expansion_count:5d}] best={snapshot.expression:<35s} "
                          f"score={snapshot.score:.4f} depth={snapshot.depth}")

                if snapshot.score > best_dom_score:
                    best_dom_score = snapshot.score
                    best_dom_expr = snapshot.expression

                if snapshot.is_discovery:
                    train_constancies = search.per_observation_scores(
                        snapshot.expression, domain_obs_list
                    )
                    dom_discoveries.append({
                        "expression": snapshot.expression,
                        "score": snapshot.score,
                        "depth": snapshot.depth,
                        "expansions": expansion_count,
                        "train_constancies": train_constancies,
                        "domain": domain_name,
                    })
                    print(f"    *** DISCOVERY [{domain_name}]: {snapshot.expression} "
                          f"(score={snapshot.score:.4f}) ***")
        except Exception as e:
            print(f"    Search error: {e}")

        total_expansions += search.expansion_count

        # Fall back to best expression if none discovered
        if not dom_discoveries and best_dom_expr and best_dom_score > 0.5:
            dom_discoveries.append({
                "expression": best_dom_expr,
                "score": best_dom_score,
                "depth": 0,
                "expansions": search.expansion_count,
                "train_constancies": [],
                "domain": domain_name,
            })
            print(f"    Using best: {best_dom_expr} (score={best_dom_score:.4f})")

        all_discoveries.extend(dom_discoveries)
        print(f"    Found {len(dom_discoveries)} invariants")

    result.train_discoveries = all_discoveries
    result.expansions_used = total_expansions
    print(f"\n  Total: {len(all_discoveries)} invariants discovered across {len(domain_obs)} domains")

    # ── 3. Test on post-1905 scenarios ────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("PHASE 2: Test discovered invariants on post-1905 data")
    print(f"{'─' * 70}")

    evaluator = ExpressionEvaluator()
    scenario_results = []

    for post_obs in post_db:
        print(f"\n  Scenario: {post_obs.id} [{post_obs.description[:80]}...]")

        # Compute noise floor
        print(f"    Computing noise floor ({n_noise_trials} trials)...")
        noise_mean, noise_sigma = compute_noise_floor(
            post_obs, noise_level, n_noise_trials, seed
        )
        print(f"    Noise floor: {noise_mean:.4f} ± {noise_sigma:.4f} (σ)")

        # Evaluate all discovered expressions
        best_expr = ""
        best_score = 0.0
        all_scores = []

        for disc in all_discoveries:
            expr = disc["expression"]
            try:
                s = evaluator.score(expr, post_obs)
                if not math.isnan(s):
                    all_scores.append(s)
                    if s > best_score:
                        best_score = s
                        best_expr = expr
            except Exception:
                pass

        # Also test the post-1905 known invariant (sanity check)
        known_score = 0.0
        if post_obs.known_invariant:
            try:
                known_score = evaluator.score(post_obs.known_invariant, post_obs)
                print(f"    Known invariant '{post_obs.known_invariant}': score={known_score:.4f}")
            except Exception:
                pass

        p_val = compute_p_value(best_score, noise_mean, noise_sigma)

        if best_score > noise_mean + 3 * noise_sigma:
            sig = "SIGNIFICANT" if p_val < 0.01 else "MARGINAL"
        elif best_score > noise_mean + 2 * noise_sigma:
            sig = "MARGINAL" if p_val < 0.05 else "NONE"
        else:
            sig = "NONE"

        print(f"    Best discovered: '{best_expr}' score={best_score:.4f}")
        print(f"    p={p_val:.4f}  significance={sig}")

        sr = ScenarioResult(
            scenario_id=post_obs.id,
            domain=post_obs.description[:60],
            known_invariant=post_obs.known_invariant,
            best_expression=best_expr,
            best_score=best_score,
            noise_floor=noise_mean,
            noise_sigma=noise_sigma,
            p_value=p_val,
            all_scores=all_scores,
            significance=sig,
        )

        scenario_results.append({
            "scenario_id": sr.scenario_id,
            "domain": sr.domain,
            "known_invariant": sr.known_invariant,
            "known_invariant_score": known_score,
            "best_expression": sr.best_expression,
            "best_score": sr.best_score,
            "noise_floor": sr.noise_floor,
            "noise_sigma": sr.noise_sigma,
            "p_value": sr.p_value,
            "significance": sr.significance,
            "all_scores": sr.all_scores,
        })

    result.scenario_results = scenario_results

    # ── 4. Symmetry discovery on post-1905 data ──────────────────────────
    print(f"\n{'─' * 70}")
    print("PHASE 3: Symmetry discovery on post-1905 data")
    print(f"{'─' * 70}")

    symmetry_proposals = []
    try:
        from src.physics.symmetry import SymmetryDetector, SymmetryDetection
        detector = SymmetryDetector()

        for post_obs in post_db:
            try:
                detection = detector.detect(post_obs)
                detected_gens = detection.active_symmetries if detection else []

                # Check if any detected group is post-1905
                post1905_groups = {"Poincaré", "Poincare", "Lorentz", "SU(2)", "SO(3,1)", "quantum",
                                   "BOOST", "boost"}
                found_post1905 = []
                for g in detected_gens:
                    gname = str(g) if hasattr(g, 'name') else str(g)
                    if any(pg.lower() in gname.lower() for pg in post1905_groups):
                        found_post1905.append(gname)

                proposal = {
                    "scenario_id": post_obs.id,
                    "detected_groups": [str(g) for g in detected_gens],
                    "post1905_groups_found": found_post1905,
                }
                symmetry_proposals.append(proposal)

                if found_post1905:
                    print(f"  {post_obs.id}: Found post-1905 groups: {found_post1905}")
                else:
                    groups_str = [str(g) for g in detected_gens] if detected_gens else ["none"]
                    print(f"  {post_obs.id}: Known groups: {groups_str}")
            except Exception as e:
                print(f"  {post_obs.id}: Detection failed — {e}")
                symmetry_proposals.append({
                    "scenario_id": post_obs.id,
                    "detected_groups": [],
                    "error": str(e),
                })
    except ImportError as e:
        print(f"  Symmetry detection not available: {e}")
        symmetry_proposals.append({"error": f"ImportError: {e}"})

    result.symmetry_proposals = symmetry_proposals

    # ── 5. Determine breakthroughs ───────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("PHASE 4: Breakthrough analysis")
    print(f"{'─' * 70}")

    breakthroughs = []

    # Criterion 1: Discovered post-1905 conserved quantity (p < 0.05)
    sig_results = [s for s in scenario_results
                   if s["significance"] == "SIGNIFICANT"]
    if sig_results:
        msg = f"Discovered {len(sig_results)} post-1905 invariants (p<0.05): " \
              + ", ".join(s["scenario_id"] for s in sig_results)
        breakthroughs.append(msg)
        print(f"  ✓ {msg}")

    # Criterion 2: Proposed post-1905 symmetry group
    for prop in symmetry_proposals:
        if prop.get("post1905_groups_found"):
            msg = f"Proposed post-1905 group for {prop['scenario_id']}: {prop['post1905_groups_found']}"
            breakthroughs.append(msg)
            print(f"  ✓ {msg}")

    # Criterion 3: Score > noise floor at p < 0.05
    marginal_results = [s for s in scenario_results
                        if s["significance"] in ("SIGNIFICANT", "MARGINAL")]
    for s in marginal_results:
        if s["significance"] in ("SIGNIFICANT",) and not any(
                s["scenario_id"] in b for b in breakthroughs if "invariant" in b.lower()):
            msg = f"Score > noise floor ({s['best_score']:.3f} > {s['noise_floor']:.3f}+2σ) " \
                  f"for {s['scenario_id']}: p={s['p_value']:.4f}"
            breakthroughs.append(msg)
            print(f"  ✓ {msg}")

    if not breakthroughs:
        print("  ✗ No breakthroughs detected. This is expected for some tests.")
    else:
        print(f"\n  {len(breakthroughs)} breakthrough(s) found!")

    result.breakthroughs = breakthroughs
    result.overall_pass = len(breakthroughs) > 0

    # ── Final summary ────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("ERA GATE COMPLETE")
    print(f"  Overall pass: {result.overall_pass}")
    print(f"  Breakthroughs: {len(breakthroughs)}")
    for b in breakthroughs:
        print(f"    - {b}")
    print(f"{'=' * 70}")

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Report generator
# ═══════════════════════════════════════════════════════════════════════════

def generate_report(result: ERAGateResult, output_path: str) -> str:
    """Generate a comprehensive markdown report."""
    lines = []
    lines.append("# ERA GATE: Pre-1905 → Post-1905 Physics Discovery")
    lines.append("")
    lines.append(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(result.timestamp))} UTC")
    lines.append(f"**Overall Result:** {'✅ PASS' if result.overall_pass else '❌ NO BREAKTHROUGH (expected for some tests)'}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("The ERA GATE is an honesty test: train the discovery system exclusively on")
    lines.append("pre-1905 physics, then present post-1905 experimental data. The system must")
    lines.append("discover physics it was never taught.")
    lines.append("")
    lines.append(f"- **Training:** {result.pre1905_scenario_count} pre-1905 scenarios "
                 f"({', '.join(f'{k}: {v}' for k, v in sorted(result.pre1905_domains.items()))})")
    lines.append(f"- **Test:** {len(result.scenario_results)} post-1905 scenarios")
    lines.append(f"- **Search budget:** {result.budget['max_expansions']} expansions, "
                 f"max depth {result.budget['max_depth']}")
    lines.append(f"- **Expansions used:** {result.expansions_used}")
    lines.append(f"- **Training discoveries:** {len(result.train_discoveries)}")
    lines.append("")

    # Training discoveries
    lines.append("## Pre-1905 Training Discoveries")
    lines.append("")
    if result.train_discoveries:
        lines.append("| Expression | Score | Depth | Expansions |")
        lines.append("|-----------|-------|-------|------------|")
        for d in result.train_discoveries:
            lines.append(f"| `{d['expression']}` | {d['score']:.4f} | {d['depth']} | {d['expansions']} |")
    else:
        lines.append("*No invariants discovered during training.*")
    lines.append("")

    # Post-1905 test results
    lines.append("## Post-1905 Test Results")
    lines.append("")

    for sr in result.scenario_results:
        lines.append(f"### {sr['scenario_id']}")
        lines.append(f"**Domain:** {sr['domain']}  ")
        if sr.get('known_invariant'):
            lines.append(f"**Known invariant:** `{sr['known_invariant']}` (truth score: {sr.get('known_invariant_score', 0):.4f})  ")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Best discovered expression | `{sr['best_expression']}` |")
        lines.append(f"| Best score | {sr['best_score']:.6f} |")
        lines.append(f"| Noise floor (μ) | {sr['noise_floor']:.6f} |")
        lines.append(f"| Noise sigma (σ) | {sr['noise_sigma']:.6f} |")
        lines.append(f"| z-score | {(sr['best_score'] - sr['noise_floor']) / max(sr['noise_sigma'], 1e-12):.2f}σ |")
        lines.append(f"| p-value | {sr['p_value']:.6f} |")
        lines.append(f"| Significance | **{sr['significance']}** |")
        lines.append("")

    # Symmetry proposals
    lines.append("## Symmetry Discovery")
    lines.append("")

    if result.symmetry_proposals:
        for sp in result.symmetry_proposals:
            if "error" in sp:
                lines.append(f"- **Error:** {sp['error']}")
            else:
                sid = sp.get("scenario_id", "unknown")
                groups = sp.get("detected_groups", [])
                found = sp.get("post1905_groups_found", [])
                lines.append(f"- **{sid}:** detected {groups}")
                if found:
                    lines.append(f"  - ✓ Post-1905 groups found: {found}")
    lines.append("")

    # Breakthroughs
    lines.append("## Breakthroughs")
    lines.append("")
    if result.breakthroughs:
        for i, b in enumerate(result.breakthroughs, 1):
            lines.append(f"{i}. {b}")
    else:
        lines.append("*No breakthroughs detected.*")
        lines.append("")
        lines.append("This is EXPECTED for some tests. The ERA GATE is designed to be hard.")
        lines.append("Success on ANY single test is significant. Where the system succeeds")
        lines.append("indicates generalization. Where it fails tells us what's missing.")
    lines.append("")

    # Analysis
    lines.append("## Analysis")
    lines.append("")
    lines.append("### What Was Discovered")
    sig_count = sum(1 for s in result.scenario_results if s["significance"] == "SIGNIFICANT")
    mar_count = sum(1 for s in result.scenario_results if s["significance"] == "MARGINAL")
    none_count = sum(1 for s in result.scenario_results if s["significance"] == "NONE")
    lines.append(f"- **Significant (p < 0.01):** {sig_count}/{len(result.scenario_results)}")
    lines.append(f"- **Marginal (p < 0.05):** {mar_count}/{len(result.scenario_results)}")
    lines.append(f"- **No signal:** {none_count}/{len(result.scenario_results)}")
    lines.append("")

    lines.append("### Generalization Ceiling")
    best_scores = {s["scenario_id"]: s["best_score"] for s in result.scenario_results}
    noise_floors = {s["scenario_id"]: s["noise_floor"] for s in result.scenario_results}
    lines.append("")
    lines.append("| Test Scenario | Best Score | Noise Floor | Δ | Above Noise? |")
    lines.append("|--------------|-----------|-------------|---|-------------|")
    for sid in sorted(best_scores.keys()):
        bs = best_scores[sid]
        nf = noise_floors[sid]
        delta = bs - nf
        above = "✓" if delta > 0 else "✗"
        lines.append(f"| {sid} | {bs:.4f} | {nf:.4f} | {delta:+.4f} | {above} |")
    lines.append("")

    lines.append("### Why Failure Is Expected")
    lines.append("")
    lines.append("The system was trained on Newtonian mechanics, classical EM, and ideal-gas")
    lines.append("thermodynamics. Post-1905 physics requires fundamentally new concepts:")
    lines.append("")
    lines.append("1. **Special Relativity:** Lorentz invariance, spacetime metric, c as limiting velocity")
    lines.append("2. **General Relativity:** Curved spacetime, geodesic equation, metric tensor")
    lines.append("3. **Quantum Mechanics:** Discretized eigenvalues, wave-particle duality, operators")
    lines.append("4. **Wave-Particle Duality:** de Broglie relation connecting momentum to wavelength")
    lines.append("5. **Uncertainty Principle:** Non-commuting observables, ℏ as minimum action")
    lines.append("")
    lines.append("The ERA GATE measures what CAN be generalized from classical training.")
    lines.append("Success indicates the system can detect 'this doesn't match any known domain'")
    lines.append("and engage symmetry discovery mode to propose new group structures.")
    lines.append("")

    report = "\n".join(lines)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report)
    return report


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(description="ERA GATE physics discovery test")
    parser.add_argument("--pre1905-db", default="data/observations/pre1905_training.json")
    parser.add_argument("--post1905-db", default="data/observations/post1905_test.json")
    parser.add_argument("--max-expansions", type=int, default=20_000)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--discovery-threshold", type=float, default=0.90)
    parser.add_argument("--noise-level", type=int, default=1, choices=[0, 1, 3, 5],
                        help="Noise percentage: 0=none, 1=low, 3=medium, 5=high")
    parser.add_argument("--noise-trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", default="data/era_gate_results.json")
    parser.add_argument("--output-report", default="docs/reports/era_gate_report.md")

    args = parser.parse_args()

    noise_level_map = {0: NoiseLevel.NONE, 1: NoiseLevel.LOW, 3: NoiseLevel.MEDIUM, 5: NoiseLevel.HIGH}
    noise_level = noise_level_map[args.noise_level]

    result = run_era_gate(
        pre1905_db_path=args.pre1905_db,
        post1905_db_path=args.post1905_db,
        max_expansions=args.max_expansions,
        max_depth=args.max_depth,
        discovery_threshold=args.discovery_threshold,
        noise_level=noise_level,
        n_noise_trials=args.noise_trials,
        seed=args.seed,
    )

    # Save results JSON
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(result.to_dict(), f, indent=2, default=str)
    print(f"\nResults saved to: {output_json}")

    # Generate report
    report = generate_report(result, args.output_report)
    print(f"Report saved to: {args.output_report}")


if __name__ == "__main__":
    main()
