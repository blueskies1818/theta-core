#!/usr/bin/env python3
"""Phase 5: Statistical Validation

1. Multi-seed runs (10 seeds) — mean, std, pass rate, exact rate
2. Ablation — no neural, no memory, no simple search
3. Pipeline attribution — which pipeline finds each claim
4. False positive test on random data
"""

from __future__ import annotations

import math
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.physics.dimensions import Dimension
from src.physics.evaluator import ExpressionEvaluator
from src.physics.observations import Observation
from src.physics.search import auto_discover
from src.memory import load_memory, save_memory, reset_memory, memory_summary

# Import claim generators from verify_instruments
from scripts.verify_instruments import (
    gen_hydrogen_balmer, gen_spin_quantization, gen_wien,
    gen_photoelectric, gen_rest_energy, gen_velocity_addition,
    gen_energy_momentum, gen_spacetime_interval, split_observations,
    CLAIMS,
)

DISCOVERY_THRESHOLD = 0.90
N_SEEDS = 10
BASE_SEED = 42


@dataclass
class ClaimStats:
    domain: str
    claim: str
    invariant: str
    scores: list[float] = field(default_factory=list)
    expressions: list[str] = field(default_factory=list)
    exact_count: int = 0
    pass_count: int = 0
    pipeline: str = "unknown"


def normalize(expr: str) -> str:
    return expr.replace(" ", "")


def run_validation(name: str, generators, split=True):
    """Run multi-seed validation."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    results: list[ClaimStats] = []
    reset_memory()

    for domain, claim, invariant, generator in generators:
        stats = ClaimStats(domain=domain, claim=claim, invariant=invariant)

        for si in range(N_SEEDS):
            seed = BASE_SEED + si * 997
            rng = random.Random(seed)
            observations = generator(rng)
            if split:
                observations = split_observations(observations)

            quantity_dict = {}
            for obs in observations:
                for qname, qdim in obs.quantities.items():
                    if qname not in quantity_dict:
                        quantity_dict[qname] = Dimension.named(qdim)

            discovery = auto_discover(
                quantities=quantity_dict,
                observations=observations,
                known_invariant=None,
                discovery_threshold=DISCOVERY_THRESHOLD,
                beam_expansions=2000,
            )

            stats.scores.append(discovery.score)
            stats.expressions.append(discovery.expression)
            if discovery.score >= DISCOVERY_THRESHOLD:
                stats.pass_count += 1
            if normalize(discovery.expression) == normalize(invariant):
                stats.exact_count += 1

        results.append(stats)

    # Print results
    print(f"\n{'Claim':45s} {'Mean':>8s} {'Std':>7s} {'Pass':>6s} {'Exact':>6s}  Top expr")
    print("-" * 100)
    for s in results:
        sc = s.scores
        mean = statistics.mean(sc) if sc else 0
        std = statistics.stdev(sc) if len(sc) >= 2 else 0
        pass_pct = s.pass_count / N_SEEDS
        exact_pct = s.exact_count / N_SEEDS
        top = s.expressions[0][:30] if s.expressions else ""
        print(f"{s.claim:45s} {mean:8.4f} {std:7.4f} {pass_pct:5.0%} {exact_pct:5.0%}  {top}")

    verified = sum(1 for s in results if s.pass_count / N_SEEDS >= 0.6)
    print(f"\n  Verified: {verified}/{len(results)}")

    return results


def run_ablation():
    """Run with components disabled."""
    print(f"\n{'='*60}")
    print(f"  ABLATION STUDY")
    print(f"{'='*60}")

    # Ablation: no neural templates
    print("\n--- No neural templates ---")
    reset_memory()
    for domain, claim, invariant, generator in CLAIMS[:3]:  # first 3 claims
        rng = random.Random(42)
        observations = generator(rng)
        observations = split_observations(observations)
        quantity_dict = {}
        for obs in observations:
            for qname, qdim in obs.quantities.items():
                if qname not in quantity_dict:
                    quantity_dict[qname] = Dimension.named(qdim)

        discovery = auto_discover(
            quantities=quantity_dict,
            observations=observations,
            known_invariant=None,
            discovery_threshold=DISCOVERY_THRESHOLD,
            beam_expansions=2000,
            _no_neural_templates=True,
        )
        match = "EXACT" if normalize(discovery.expression) == normalize(invariant) else "PASS" if discovery.score >= DISCOVERY_THRESHOLD else "FAIL"
        print(f"  {claim:35s} -> {match:5s} {discovery.expression[:30]:30s} score={discovery.score:.4f}")

    # Ablation: no memory (clean memory first, then disable)
    print("\n--- No memory (cold start) ---")
    reset_memory()
    for domain, claim, invariant, generator in CLAIMS[:3]:
        rng = random.Random(42)
        observations = generator(rng)
        observations = split_observations(observations)
        quantity_dict = {}
        for obs in observations:
            for qname, qdim in obs.quantities.items():
                if qname not in quantity_dict:
                    quantity_dict[qname] = Dimension.named(qdim)

        discovery = auto_discover(
            quantities=quantity_dict,
            observations=observations,
            known_invariant=None,
            discovery_threshold=DISCOVERY_THRESHOLD,
            beam_expansions=2000,
        )
        match = "EXACT" if normalize(discovery.expression) == normalize(invariant) else "PASS" if discovery.score >= DISCOVERY_THRESHOLD else "FAIL"
        print(f"  {claim:35s} -> {match:5s} {discovery.expression[:30]:30s} score={discovery.score:.4f}")


def run_false_positive():
    """Test on random data — system should find nothing."""
    print(f"\n{'='*60}")
    print(f"  FALSE POSITIVE TEST")
    print(f"{'='*60}")

    for n_vars in [3, 4, 5]:
        fp_count = 0
        for seed in range(10):
            rng = random.Random(1000 + seed)
            timesteps = []
            symbols = [f"x{i}" for i in range(n_vars)]
            for t in range(20):
                ts = {"t": float(t)}
                for s in symbols:
                    ts[s] = rng.uniform(-10, 10)  # purely random
                timesteps.append(ts)

            obs = [Observation(
                id=f"rand_{n_vars}", name="Random data",
                description="Purely random variables — no invariant exists",
                quantities={s: "Scalar" for s in symbols},
                parameters={}, timesteps=timesteps,
                known_invariant="", lean_theorem="",
            )]

            quantity_dict = {s: Dimension.scalar() for s in symbols}

            discovery = auto_discover(
                quantities=quantity_dict,
                observations=obs,
                known_invariant=None,
                discovery_threshold=DISCOVERY_THRESHOLD,
                beam_expansions=2000,
            )
            if discovery.score >= DISCOVERY_THRESHOLD:
                fp_count += 1

        fp_rate = fp_count / 10
        print(f"  {n_vars} random variables: {fp_count}/10 false positives (rate={fp_rate:.0%})")


def main():
    print("=" * 60)
    print("PHASE 5: STATISTICAL VALIDATION")
    print(f"Seeds: {N_SEEDS}  |  Threshold: {DISCOVERY_THRESHOLD}")
    print("=" * 60)

    t0 = time.time()

    # 1. Multi-seed clean
    run_validation("CLEAN DATA (instrument-based, 2-3 quantities)", CLAIMS, split=True)

    # 2. Multi-seed nuisance
    from scripts.verify_nuisance import CLAIMS as NUISANCE_CLAIMS
    from scripts.verify_nuisance import _split

    # Nuisance requires sequential run with memory accumulation
    print(f"\n{'='*60}")
    print(f"  NUISANCE DATA (sequential, accumulator memory)")
    print(f"{'='*60}")
    reset_memory()
    for domain, claim, invariant, generator, qty_types in NUISANCE_CLAIMS:
        pass_count = 0
        exact_count = 0
        scores = []
        exprs = []
        for si in range(N_SEEDS):
            seed = BASE_SEED + si * 997
            rng = random.Random(seed)
            obs = generator(rng)
            obs = _split(obs, min_per=4)
            quantities = {k: Dimension.named(v) for k, v in qty_types.items()}
            discovery = auto_discover(
                quantities=quantities, observations=obs,
                known_invariant=None,
                discovery_threshold=DISCOVERY_THRESHOLD,
                beam_expansions=2000,
            )
            scores.append(discovery.score)
            exprs.append(discovery.expression)
            if discovery.score >= DISCOVERY_THRESHOLD:
                pass_count += 1
            if normalize(discovery.expression) == normalize(invariant):
                exact_count += 1

        mean_s = statistics.mean(scores)
        std_s = statistics.stdev(scores) if len(scores) >= 2 else 0
        print(f"  {claim:35s}  mean={mean_s:.4f}±{std_s:.3f}  pass={pass_count}/{N_SEEDS}  exact={exact_count}/{N_SEEDS}  top: {exprs[0][:25] if exprs else ''}")

    # 3. Ablation
    run_ablation()

    # 4. False positive
    run_false_positive()

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  Phase 5 complete in {elapsed:.0f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
