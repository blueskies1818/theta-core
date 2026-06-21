"""Phase E: Conditional conservation discovery on extended observation database.

Runs self-play search with conditional evaluation to discover:
1. m*g*h + 0.5*m*v^2 is conserved in conservative scenarios
2. It is NOT conserved when friction/drag are present  
3. 0.5*k*h^2 + 0.5*m*v^2 for spring systems
4. 0.5*k*h^2 + m*g*h + 0.5*m*v^2 for spring+gravity combined

Output: data/phase_e_discoveries.json — training data for Phase F
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from src.physics.dimensions import Dimension
from src.physics.evaluator import ExpressionEvaluator
from src.physics.observations import Observation, ObservationDatabase
from src.physics.search import ExpressionSearch, SearchResult


def run_phase_e(
    db_path: str = "data/observations/phase2_extended.json",
    *,
    train_count: int = 45,
    test_count: int = 10,
    max_expansions: int = 20_000,
    max_depth: int = 7,
    discovery_threshold: float = 0.95,
    top_k: int = 50,
    seed: int = 42,
    output_path: str = "data/phase_e_discoveries.json",
) -> dict:
    """Run Phase E self-play discovery.

    Searches for invariants in three domains:
    - Freefall/projectile/incline/pendulum (conservative: mgh + ½mv²)
    - Spring (undamped: ½kx² + ½mv²)
    - Spring+gravity combined (½kx² + mgh + ½mv²)

    Returns detailed results including conditional conservation evidence.
    """
    import random
    import time

    db = ObservationDatabase(db_path)
    all_obs = list(db)
    print(f"Loaded {len(all_obs)} observations from {db_path}")

    # Conservative vs non-conservative stats
    cons_obs = [o for o in all_obs if _is_cons(o)]
    noncons_obs = [o for o in all_obs if not _is_cons(o)]
    print(f"  Conservative: {len(cons_obs)}, Non-conservative: {len(noncons_obs)}")

    # Train/test split
    rng = random.Random(seed)
    indices = list(range(len(all_obs)))
    rng.shuffle(indices)
    train_obs = [all_obs[i] for i in indices[:train_count]]
    test_obs = [all_obs[i] for i in indices[train_count:train_count + test_count]]

    train_cons = [o for o in train_obs if _is_cons(o)]
    train_noncons = [o for o in train_obs if not _is_cons(o)]
    print(f"Train: {len(train_obs)} ({len(train_cons)} conservative, {len(train_noncons)} non-conservative)")
    print(f"Test:  {len(test_obs)}")

    evaluator = ExpressionEvaluator()
    discoveries: list[dict] = []

    # ── Domain 1: Freefall/projectile/incline/pendulum ──
    # These have quantities: m, g, h, v (sometimes L)
    print("\n── Domain 1: Gravity-driven systems ──")
    domain1_quantities = {"m": Dimension.named("Mass"), "g": Dimension.named("Accel"),
                          "h": Dimension.named("Length"), "v": Dimension.named("Velocity")}
    domain1_obs = [o for o in train_obs
                   if set(o.quantities.keys()) >= {"m", "g", "h", "v"}
                   and "k" not in o.quantities  # exclude spring scenarios
                   and "cross" not in o.id  # exclude cross-domain
                   and _is_cons(o)]
    print(f"  Training on {len(domain1_obs)} conservative gravity scenarios")

    search1 = ExpressionSearch(
        quantities=domain1_quantities,
        train_observations=domain1_obs,
        max_depth=max_depth,
        max_expansions=max_expansions,
        discovery_threshold=discovery_threshold,
        top_k=top_k,
        scalar_constants=["0", "0.5", "1", "2", "-1"],
    )
    result1 = search1.run()
    print(f"  Best: {result1.expression} (score={result1.score:.4f}, depth={result1.depth}, "
          f"expansions={result1.expansions})")

    if result1.is_discovery:
        # Test generalization
        test_score1 = _test_score(result1.expression, test_obs, evaluator)
        # Conditional check
        cond1 = evaluator.score_conditional(result1.expression, db)
        discoveries.append({
            "domain": "gravity",
            "expression": result1.expression,
            "train_score": result1.score,
            "test_score": test_score1,
            "depth": result1.depth,
            "expansions": result1.expansions,
            "train_constancies": result1.train_constancies,
            "conditional": cond1,
        })
        print(f"  DISCOVERED: {result1.expression}")
        print(f"    Test score: {test_score1:.4f}")
        print(f"    Conditional: conservative={cond1['conservative_score']:.4f}, "
              f"nonconservative={cond1['nonconservative_score']:.4f} "
              f"({cond1['conditional_pattern']})")
    else:
        print(f"  No discovery in gravity domain (best: {result1.expression} = {result1.score:.4f})")

    # ── Domain 2: Spring systems ──
    print("\n── Domain 2: Spring systems ──")
    domain2_quantities = {"m": Dimension.named("Mass"), "k": Dimension.named("Force/Length"),
                          "h": Dimension.named("Length"), "v": Dimension.named("Velocity")}
    domain2_obs = [o for o in train_obs
                   if set(o.quantities.keys()) >= {"m", "k", "h", "v"}
                   and "g" not in o.quantities  # exclude spring+gravity combined
                   and _is_cons(o)]
    print(f"  Training on {len(domain2_obs)} conservative spring scenarios")

    search2 = ExpressionSearch(
        quantities=domain2_quantities,
        train_observations=domain2_obs,
        max_depth=max_depth,
        max_expansions=max_expansions,
        discovery_threshold=discovery_threshold,
        top_k=top_k,
        scalar_constants=["0", "0.5", "1", "2", "-1"],
    )
    result2 = search2.run()
    print(f"  Best: {result2.expression} (score={result2.score:.4f}, depth={result2.depth}, "
          f"expansions={result2.expansions})")

    if result2.is_discovery:
        test_score2 = _test_score(result2.expression, test_obs, evaluator)
        cond2 = evaluator.score_conditional(result2.expression, db)
        discoveries.append({
            "domain": "spring",
            "expression": result2.expression,
            "train_score": result2.score,
            "test_score": test_score2,
            "depth": result2.depth,
            "expansions": result2.expansions,
            "train_constancies": result2.train_constancies,
            "conditional": cond2,
        })
        print(f"  DISCOVERED: {result2.expression}")
        print(f"    Test score: {test_score2:.4f}")
        print(f"    Conditional: conservative={cond2['conservative_score']:.4f}, "
              f"nonconservative={cond2['nonconservative_score']:.4f} "
              f"({cond2['conditional_pattern']})")
    else:
        print(f"  No discovery in spring domain (best: {result2.expression} = {result2.score:.4f})")

    # ── Domain 3: Spring + Gravity combined ──
    print("\n── Domain 3: Cross-domain (spring + gravity) ──")
    domain3_quantities = {"m": Dimension.named("Mass"), "k": Dimension.named("Force/Length"),
                          "g": Dimension.named("Accel"), "h": Dimension.named("Length"),
                          "v": Dimension.named("Velocity")}
    # Find scenarios with all 5 quantities
    domain3_obs = [o for o in train_obs
                   if set(o.quantities.keys()) >= {"m", "k", "g", "h", "v"}
                   and _is_cons(o)]
    if not domain3_obs:
        # Fall back: ALL conservative spring + gravity observations
        domain3_obs = [o for o in train_obs
                       if "k" in o.quantities and "g" in o.parameters
                       and _is_cons(o)]
    print(f"  Training on {len(domain3_obs)} spring+gravity scenarios")

    if domain3_obs:
        search3 = ExpressionSearch(
            quantities=domain3_quantities,
            train_observations=domain3_obs,
            max_depth=max_depth,
            max_expansions=max_expansions,
            discovery_threshold=discovery_threshold,
            top_k=top_k,
            scalar_constants=["0", "0.5", "1", "2", "-1"],
        )
        result3 = search3.run()
        print(f"  Best: {result3.expression} (score={result3.score:.4f}, depth={result3.depth}, "
              f"expansions={result3.expansions})")

        if result3.is_discovery:
            test_score3 = _test_score(result3.expression, test_obs, evaluator)
            cond3 = evaluator.score_conditional(result3.expression, db)
            discoveries.append({
                "domain": "spring_gravity",
                "expression": result3.expression,
                "train_score": result3.score,
                "test_score": test_score3,
                "depth": result3.depth,
                "expansions": result3.expansions,
                "train_constancies": result3.train_constancies,
                "conditional": cond3,
            })
            print(f"  DISCOVERED: {result3.expression}")
            print(f"    Test score: {test_score3:.4f}")
            print(f"    Conditional: conservative={cond3['conservative_score']:.4f}, "
                  f"nonconservative={cond3['nonconservative_score']:.4f} "
                  f"({cond3['conditional_pattern']})")
        else:
            print(f"  No discovery in spring+gravity (best: {result3.expression} = {result3.score:.4f})")
    else:
        print("  No spring+gravity scenarios available.")

    # ── Non-conservative verification ──
    print("\n── Non-conservative verification ──")
    if discoveries:
        for disc in discoveries:
            expr = disc["expression"]
            # Score on non-conservative scenarios only
            if noncons_obs:
                nc_scores = [evaluator.score(expr, obs) for obs in noncons_obs]
                nc_mean = sum(nc_scores) / len(nc_scores)
                disc["nonconservative_verification"] = {
                    "mean_score": nc_mean,
                    "per_scenario": nc_scores,
                    "verdict": "NOT conserved" if nc_mean < 0.5 else "conserved (unexpected)"
                }
                print(f"  {expr}: non-conservative score = {nc_mean:.4f} "
                      f"→ {'NOT conserved ✓' if nc_mean < 0.5 else 'unexpectedly conserved'}")
    else:
        print("  No discoveries to verify")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"Phase E: {len(discoveries)} invariant(s) discovered")
    for i, d in enumerate(discoveries):
        print(f"  [{i}] {d['domain']}: {d['expression']}")
        print(f"      Train: {d['train_score']:.4f}, Test: {d['test_score']:.4f}, "
              f"Depth: {d['depth']}, Expansions: {d['expansions']}")
        cond = d.get("conditional", {})
        if cond:
            print(f"      Conservative: {cond.get('conservative_score', 0):.4f}, "
                  f"Non-conservative: {cond.get('nonconservative_score', 0):.4f} "
                  f"({cond.get('conditional_pattern', '?')})")

    # ── Export ──
    output = {
        "phase": "E",
        "database": str(db_path),
        "total_scenarios": len(db),
        "parameters": {
            "train_count": train_count,
            "test_count": test_count,
            "max_expansions": max_expansions,
            "max_depth": max_depth,
            "discovery_threshold": discovery_threshold,
            "seed": seed,
        },
        "discoveries": discoveries,
        "timestamp": time.time(),
    }

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults written to {output_path}")

    return output


def _is_cons(obs: Observation) -> bool:
    """Determine if an observation is conservative."""
    if obs.is_conservative is not None:
        return obs.is_conservative
    if obs.external_forces:
        return False
    if obs.known_invariant is not None:
        return True
    return True


def _test_score(expr: str, test_obs: list[Observation], evaluator: ExpressionEvaluator) -> float:
    """Score an expression on test observations."""
    if not test_obs:
        return 0.0
    scores = [evaluator.score(expr, obs) for obs in test_obs]
    return sum(scores) / len(scores)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Phase E: Conditional conservation discovery")
    p.add_argument("--db", default="data/observations/phase2_extended.json")
    p.add_argument("--train", type=int, default=45)
    p.add_argument("--test", type=int, default=10)
    p.add_argument("--expansions", type=int, default=20_000)
    p.add_argument("--output", default="data/phase_e_discoveries.json")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    run_phase_e(
        db_path=args.db,
        train_count=args.train,
        test_count=args.test,
        max_expansions=args.expansions,
        output_path=args.output,
        seed=args.seed,
    )
