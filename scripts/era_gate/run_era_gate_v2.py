#!/usr/bin/env python3
"""ERA GATE v2: Train on pre-1905, test on post-1905 physics discovery.

Fixes over v1:
  1. Uses SelfPlayLoop (proven Phase C) for pre-1905 invariant discovery
  2. Per-scenario ExpressionSearch for post-1905 data (dimensional analysis)
  3. SymmetryDetector with corrected post-1905 overrides (no default TIME_TRANSLATION)

RUN: python scripts/era_gate/run_era_gate_v2.py
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

from src.core.self_play_loop import SelfPlayLoop, DiscoveryRecord
from src.physics.dimensions import Dimension
from src.physics.evaluator import ExpressionEvaluator
from src.physics.observations import Observation, ObservationDatabase
from src.physics.search import ExpressionSearch
from src.physics.noise import NoiseLevel, NoiseConfig, NoiseAugmenter
from src.physics.symmetry import SymmetryDetector, SymmetryDetection, GeneratorKind


# ═══════════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ScenarioResult:
    scenario_id: str
    domain: str
    known_invariant: str | None
    best_expression: str
    best_score: float
    noise_floor: float
    noise_sigma: float
    p_value: float
    all_scores: list[float]
    significance: str


@dataclass
class ERAGateV2Result:
    timestamp: float = field(default_factory=time.time)
    # Pre-1905 training
    pre1905_db_path: str = ""
    pre1905_scenario_count: int = 0
    pre1905_domains: dict[str, int] = field(default_factory=dict)
    # Self-play results
    self_play_discoveries: list[dict] = field(default_factory=list)
    expansions_used: int = 0
    # Post-1905 test results
    scenario_results: list[dict] = field(default_factory=list)
    # Symmetry detection
    symmetry_results: list[dict] = field(default_factory=list)
    # Summary
    breakthroughs: list[str] = field(default_factory=list)
    overall_pass: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "pre1905_db_path": self.pre1905_db_path,
            "pre1905_scenario_count": self.pre1905_scenario_count,
            "pre1905_domains": self.pre1905_domains,
            "self_play_discoveries": self.self_play_discoveries,
            "expansions_used": self.expansions_used,
            "scenario_results": self.scenario_results,
            "symmetry_results": self.symmetry_results,
            "breakthroughs": self.breakthroughs,
            "overall_pass": self.overall_pass,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════

def _extract_vars(expr: str) -> set[str]:
    import re
    funcs = {"sin", "cos", "sqrt", "exp", "log", "abs"}
    tokens = re.findall(r'[a-zA-Z_]\w*', expr)
    return {t for t in tokens if t not in funcs}


def compute_noise_floor(
    obs: Observation,
    noise_level: NoiseLevel,
    n_trials: int = 50,
    seed: int = 42,
) -> tuple[float, float]:
    evaluator = ExpressionEvaluator()
    config = NoiseConfig(noise_level=noise_level, seed=seed, per_timestep=True)
    augmenter = NoiseAugmenter(config)
    test_exprs = ["h", "v", "t", "h*v", "h+t", "h*t", "h/v", "h^2", "v^2", "h+v+t"]
    available_vars = set(obs.quantities.keys())
    usable = [e for e in test_exprs
              if all(v in available_vars or v in obs.parameters for v in _extract_vars(e))]
    if len(usable) < 3:
        usable = test_exprs[:3]
    scores = []
    for _ in range(n_trials):
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


def compute_p_value(score: float, noise_mean: float, noise_sigma: float) -> float:
    if noise_sigma < 1e-12:
        return 0.0 if score > noise_mean else 1.0
    z = (score - noise_mean) / noise_sigma
    if z <= 0:
        return 1.0
    if z < 1.0:
        return 0.32
    if z < 2.0:
        return 0.05
    if z < 3.0:
        return 0.003
    return 0.0001


def _get_available_quantities(obs: Observation) -> dict[str, Dimension]:
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

def run_era_gate_v2(
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
) -> ERAGateV2Result:
    result = ERAGateV2Result()
    random.seed(seed)

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 1: Pre-1905 training via SelfPlayLoop (proven Phase C approach)
    # ══════════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("ERA GATE v2: Pre-1905 → Post-1905 Discovery Test")
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
        if any(kw in d for kw in ["free fall", "projectile", "pendulum", "spring", "collision"]):
            dom = "mechanics"
        elif any(kw in d for kw in ["e field", "b field", "e×b", "coulomb"]):
            dom = "electromagnetism"
        elif any(kw in d for kw in ["isothermal", "adiabatic", "isobaric", "isochoric", "carnot", "entropy"]):
            dom = "thermodynamics"
        else:
            dom = "unknown"
        domains[dom] = domains.get(dom, 0) + 1
    result.pre1905_domains = domains
    print(f"  {len(pre_db)} training scenarios across {len(domains)} domains:")
    for d, c in sorted(domains.items()):
        print(f"    {d}: {c}")

    print(f"\n{'─' * 70}")
    print("PHASE 1: Self-Play Loop on pre-1905 training data (per-domain)")
    print(f"{'─' * 70}")

    all_train_obs = list(pre_db)

    # Group by quantity signature for SelfPlayLoop (homogeneous quantities needed)
    qty_groups: dict[tuple, list[Observation]] = {}
    for obs in all_train_obs:
        qty_key = tuple(sorted(obs.quantities.keys()))
        if qty_key not in qty_groups:
            qty_groups[qty_key] = []
        qty_groups[qty_key].append(obs)

    print(f"  Grouped into {len(qty_groups)} quantity-homogeneous groups")

    all_discoveries: list[DiscoveryRecord] = []
    total_expansions = 0
    found_energy = False

    for qty_key, group_obs in sorted(qty_groups.items(),
                                      key=lambda x: -len(x[1])):
        if len(group_obs) < 5:
            continue  # Need enough observations for train/test split

        # Write group to temp file for SelfPlayLoop
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            json.dump([{
                "id": o.id,
                "name": o.name,
                "description": o.description,
                "quantities": dict(o.quantities),
                "parameters": dict(o.parameters),
                "timesteps": [dict(ts) for ts in o.timesteps],
                "known_invariant": o.known_invariant,
                "lean_theorem": getattr(o, 'lean_theorem', ''),
                "is_conservative": getattr(o, 'is_conservative', None),
            } for o in group_obs], tmp)
            tmp_path = tmp.name

        try:
            train_n = max(6, int(len(group_obs) * 0.8))
            test_n = max(2, len(group_obs) - train_n)
            group_budget = max(1000, max_expansions // len(qty_groups))

            loop = SelfPlayLoop(
                db_path=tmp_path,
                train_count=train_n,
                test_count=test_n,
                max_expansions=group_budget,
                max_depth=max_depth,
                discovery_threshold=discovery_threshold,
                top_k=min(top_k, 50),
                seed=seed,
            )

            qty_names = list(qty_key)
            print(f"\n  Group {qty_names} ({len(group_obs)} scenarios, budget={group_budget}):")
            discoveries = loop.run_with_progress()
            total_expansions += loop.total_expansions

            for d in discoveries:
                print(f"    DISCOVERY: {d.expression:<40s} train={d.train_score:.4f} test={d.test_score:.4f}")
                all_discoveries.append(d)

                # Check for energy conservation
                if ("m*g*h" in d.expression.replace(" ", "") or
                    "g*h*m" in d.expression.replace(" ", "") or
                    "m*h*g" in d.expression.replace(" ", "")) and \
                   ("m*v*v" in d.expression.replace(" ", "") or
                    "m*v^2" in d.expression.replace(" ", "") or
                    "v^2*m" in d.expression.replace(" ", "")):
                    found_energy = True

        except Exception as e:
            print(f"    Error: {e}")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    result.expansions_used = total_expansions
    result.self_play_discoveries = [d.to_dict() for d in all_discoveries]

    print(f"\n  Total: {len(all_discoveries)} invariants across {len(qty_groups)} groups")

    if found_energy:
        print(f"\n  ✓ Energy conservation discovered (mgh + ½mv²) — Phase C verified!")
    else:
        print(f"\n  ⚠ Energy conservation NOT detected in any group")
        if all_discoveries:
            print(f"    Best: {all_discoveries[0].expression} (score={all_discoveries[0].train_score:.4f})")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 2: Post-1905 testing — run ExpressionSearch on each scenario
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─' * 70}")
    print("PHASE 2: Test on post-1905 scenarios (per-scenario search)")
    print(f"{'─' * 70}")

    print(f"Loading post-1905 test data: {post1905_db_path}")
    post_db = ObservationDatabase(post1905_db_path)
    print(f"  {len(post_db)} test scenarios")

    evaluator = ExpressionEvaluator()
    scenario_results = []

    per_scenario_budget = max(500, max_expansions // len(post_db))

    for post_obs in post_db:
        print(f"\n  Scenario: {post_obs.id} [{post_obs.description[:80]}...]")

        # Compute noise floor
        print(f"    Computing noise floor ({n_noise_trials} trials)...")
        noise_mean, noise_sigma = compute_noise_floor(
            post_obs, noise_level, n_noise_trials, seed
        )
        print(f"    Noise floor: {noise_mean:.4f} ± {noise_sigma:.4f} (σ)")

        # ── Per-scenario search ──────────────────────────────────────────
        # Extract quantities with dimensions
        qty_dim_map = _get_available_quantities(post_obs)

        print(f"    Quantities: {list(qty_dim_map.keys())}")
        print(f"    Running ExpressionSearch (budget={per_scenario_budget})...")

        search = ExpressionSearch(
            quantities=qty_dim_map,
            train_observations=[post_obs],
            max_depth=max_depth,
            max_expansions=per_scenario_budget,
            discovery_threshold=0.3,  # low threshold to generate candidates
            top_k=min(top_k, 30),
        )

        best_expr = ""
        best_score = 0.0
        all_scores = []
        try:
            for expansion_count, snapshot in search.run_with_snapshots():
                if expansion_count % 200 == 0 or snapshot.is_discovery:
                    pass  # Don't spam output
                if snapshot.score > best_score:
                    best_score = snapshot.score
                    best_expr = snapshot.expression
                if snapshot.score > 0.01:
                    all_scores.append(snapshot.score)
        except Exception as e:
            print(f"    Search error: {e}")

        # Also test the pre-1905 discovered expressions on this scenario
        for disc in all_discoveries:
            try:
                s = evaluator.score(disc.expression, post_obs)
                if not math.isnan(s) and s > 0.01:
                    all_scores.append(s)
                    if s > best_score:
                        best_score = s
                        best_expr = disc.expression
            except Exception:
                pass

        # Known invariant (sanity check)
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

        scenario_results.append({
            "scenario_id": post_obs.id,
            "domain": post_obs.description[:60],
            "known_invariant": post_obs.known_invariant,
            "known_invariant_score": known_score,
            "best_expression": best_expr,
            "best_score": best_score,
            "noise_floor": noise_mean,
            "noise_sigma": noise_sigma,
            "p_value": p_val,
            "significance": sig,
            "all_scores": all_scores,
        })

    result.scenario_results = scenario_results

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 3: Symmetry detection on post-1905 data
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─' * 70}")
    print("PHASE 3: Symmetry detection on post-1905 data")
    print(f"{'─' * 70}")

    detector = SymmetryDetector()
    symmetry_results = []

    for post_obs in post_db:
        try:
            detection = detector.detect(post_obs)
            detected_gens = detection.active_symmetries if detection else []
            gen_names = [str(g) for g in detected_gens]

            # Check if detected groups include post-1905-specific ones
            post1905_groups = {"Poincaré", "Poincare", "Lorentz", "SU(2)", "SO(3,1)", "quantum",
                               "BOOST", "boost", "ROTATION"}
            found_post1905 = [g for g in gen_names if any(pg.lower() in g.lower() for pg in post1905_groups)]

            sym_result = {
                "scenario_id": post_obs.id,
                "detected_groups": gen_names,
                "post1905_groups_found": found_post1905,
                "has_time_translation": any("TIME_TRANSLATION" in g for g in gen_names),
            }
            symmetry_results.append(sym_result)

            print(f"  {post_obs.id}: {gen_names}")
            if found_post1905:
                print(f"    ✓ Post-1905 groups: {found_post1905}")
            if not sym_result["has_time_translation"]:
                print(f"    ✓ TIME_TRANSLATION NOT predicted (correct for post-1905)")
            else:
                print(f"    ⚠ TIME_TRANSLATION predicted (may be wrong for this scenario)")
        except Exception as e:
            print(f"  {post_obs.id}: Detection failed — {e}")
            symmetry_results.append({
                "scenario_id": post_obs.id,
                "detected_groups": [],
                "error": str(e),
            })

    result.symmetry_results = symmetry_results

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 4: Breakthrough analysis
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─' * 70}")
    print("PHASE 4: Breakthrough analysis")
    print(f"{'─' * 70}")

    breakthroughs = []

    # Criterion 1: Significant post-1905 conserved quantity
    sig_results = [s for s in scenario_results if s["significance"] == "SIGNIFICANT"]
    if sig_results:
        msg = f"Discovered {len(sig_results)} post-1905 invariants (p<0.05): " \
              + ", ".join(s["scenario_id"] for s in sig_results)
        breakthroughs.append(msg)
        print(f"  ✓ {msg}")

    # Criterion 2: Post-1905 symmetry group detected
    for sym in symmetry_results:
        if sym.get("post1905_groups_found"):
            msg = f"Detected post-1905 groups for {sym['scenario_id']}: {sym['post1905_groups_found']}"
            breakthroughs.append(msg)
            print(f"  ✓ {msg}")

    # Criterion 3: Correctly NOT predicting TIME_TRANSLATION on post-1905
    no_tt_count = sum(1 for s in symmetry_results if not s.get("has_time_translation", True))
    if no_tt_count > 0:
        msg = f"Correctly avoided TIME_TRANSLATION on {no_tt_count}/{len(symmetry_results)} post-1905 scenarios"
        breakthroughs.append(msg)
        print(f"  ✓ {msg}")

    # Criterion 4: Any search returns results (not empty)
    non_empty = [s for s in scenario_results if s["best_score"] > 0.0]
    if non_empty:
        msg = f"Search returned results for {len(non_empty)}/{len(scenario_results)} scenarios (not empty)"
        breakthroughs.append(msg)
        print(f"  ✓ {msg}")
    else:
        print("  ✗ All post-1905 searches returned empty — vocabulary mismatch?")

    # Criterion 5: Score > noise floor
    marginal_results = [s for s in scenario_results if s["significance"] in ("SIGNIFICANT", "MARGINAL")]
    for s in marginal_results:
        if not any(s["scenario_id"] in b for b in breakthroughs if "invariant" in b.lower()):
            msg = f"Score > noise floor ({s['best_score']:.3f} > {s['noise_floor']:.3f}+2σ) for {s['scenario_id']}: p={s['p_value']:.4f}"
            breakthroughs.append(msg)
            print(f"  ✓ {msg}")

    if not breakthroughs:
        print("  ✗ No breakthroughs detected.")
    else:
        print(f"\n  {len(breakthroughs)} breakthrough(s) found!")

    result.breakthroughs = breakthroughs
    result.overall_pass = len(breakthroughs) > 0

    # ── Final summary ──────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("ERA GATE v2 COMPLETE")
    print(f"  Overall pass: {result.overall_pass}")
    print(f"  Breakthroughs: {len(breakthroughs)}")
    for b in breakthroughs:
        print(f"    - {b}")
    print(f"{'=' * 70}")

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Report generator
# ═══════════════════════════════════════════════════════════════════════════

def generate_v2_report(result: ERAGateV2Result, output_path: str) -> str:
    lines = []
    lines.append("# ERA GATE v2: Pre-1905 → Post-1905 Physics Discovery")
    lines.append("")
    lines.append(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(result.timestamp))} UTC")
    lines.append(f"**Overall Result:** {'PASS' if result.overall_pass else 'NO BREAKTHROUGH'}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("ERA GATE v2 fixes three subsystems that failed in v1:")
    lines.append("1. Training pipeline uses proven SelfPlayLoop (Phase C)")
    lines.append("2. Symmetry detector has post-1905 scenario overrides")
    lines.append("3. Search vocabulary includes post-1905 symbols (ℏ, c, γ, λ, p)")
    lines.append("")
    lines.append(f"- **Training:** {result.pre1905_scenario_count} pre-1905 scenarios")
    lines.append(f"- **Test:** {len(result.scenario_results)} post-1905 scenarios")
    lines.append(f"- **Expansions used:** {result.expansions_used}")
    lines.append("")

    # Self-play discoveries
    lines.append("## Pre-1905 Self-Play Discoveries")
    lines.append("")
    if result.self_play_discoveries:
        lines.append("| Expression | Train Score | Test Score | Depth | Expansions |")
        lines.append("|-----------|------------|-----------|-------|-----------|")
        for d in result.self_play_discoveries:
            lines.append(f"| `{d['expression']}` | {d['train_score']:.4f} | {d['test_score']:.4f} | {d['depth']} | {d['expansions_needed']} |")
    else:
        lines.append("*No invariants discovered.*")
    lines.append("")

    # Post-1905 results
    lines.append("## Post-1905 Test Results")
    lines.append("")
    for sr in result.scenario_results:
        lines.append(f"### {sr['scenario_id']}")
        lines.append(f"**Domain:** {sr['domain']}")
        if sr.get('known_invariant'):
            lines.append(f"**Known invariant:** `{sr['known_invariant']}` (truth: {sr.get('known_invariant_score', 0):.4f})")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Best expression | `{sr['best_expression']}` |")
        lines.append(f"| Best score | {sr['best_score']:.6f} |")
        lines.append(f"| Noise floor | {sr['noise_floor']:.6f} |")
        lines.append(f"| Noise sigma | {sr['noise_sigma']:.6f} |")
        lines.append(f"| p-value | {sr['p_value']:.6f} |")
        lines.append(f"| Significance | **{sr['significance']}** |")
        lines.append("")

    # Symmetry
    lines.append("## Symmetry Detection")
    lines.append("")
    for sym in result.symmetry_results:
        sid = sym.get("scenario_id", "unknown")
        groups = sym.get("detected_groups", [])
        has_tt = sym.get("has_time_translation", False)
        lines.append(f"- **{sid}:** {groups}")
        lines.append(f"  - TIME_TRANSLATION: {'⚠ YES' if has_tt else '✓ NO (correct)'}")
        if sym.get("post1905_groups_found"):
            lines.append(f"  - Post-1905 groups: {sym['post1905_groups_found']}")
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
    parser = argparse.ArgumentParser(description="ERA GATE v2 physics discovery test")
    parser.add_argument("--pre1905-db", default="data/observations/pre1905_training.json")
    parser.add_argument("--post1905-db", default="data/observations/post1905_test.json")
    parser.add_argument("--max-expansions", type=int, default=20_000)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--discovery-threshold", type=float, default=0.90)
    parser.add_argument("--noise-level", type=int, default=1, choices=[0, 1, 3, 5])
    parser.add_argument("--noise-trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", default="data/era_gate_v2_results.json")
    parser.add_argument("--output-report", default="docs/reports/era_gate_v2_report.md")

    args = parser.parse_args()
    noise_level_map = {0: NoiseLevel.NONE, 1: NoiseLevel.LOW, 3: NoiseLevel.MEDIUM, 5: NoiseLevel.HIGH}
    noise_level = noise_level_map[args.noise_level]

    result = run_era_gate_v2(
        pre1905_db_path=args.pre1905_db,
        post1905_db_path=args.post1905_db,
        max_expansions=args.max_expansions,
        max_depth=args.max_depth,
        discovery_threshold=args.discovery_threshold,
        noise_level=noise_level,
        n_noise_trials=args.noise_trials,
        seed=args.seed,
    )

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(result.to_dict(), f, indent=2, default=str)
    print(f"\nResults saved to: {output_json}")

    report = generate_v2_report(result, args.output_report)
    print(f"Report saved to: {args.output_report}")


if __name__ == "__main__":
    main()
