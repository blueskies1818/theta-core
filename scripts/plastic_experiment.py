#!/usr/bin/env python3
"""Plastic learning experiment — sequential claim processing.

Starts with empty plastic memory. Processes claims across domains.
Measures whether plastic adaptation improves discovery scores.
"""

from __future__ import annotations

import random, sys, time, math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.physics.dimensions import Dimension
from src.physics.observations import Observation
from src.physics.search import auto_discover
from src.math.plastic_seed_scorer import (
    get_plastic_state, reset_plastic,
)


def gen_product_data(a_name, b_name, dims, n_obs=3, n_ts=10, noise=0.0003):
    """Generate data where a*b = constant (different K per observation)."""
    def gen(rng):
        obs = []
        for _ in range(n_obs):
            K = 1 + rng.random() * 100
            ts = []
            for _ in range(n_ts):
                a = 0.5 + rng.random() * 10
                b = K / a
                row = {a_name: a, b_name: b}
                for extra_name, extra_dim in dims.items():
                    if extra_name not in (a_name, b_name):
                        row[extra_name] = rng.random() * 5
                # Add tiny noise
                for k in row:
                    row[k] += rng.gauss(0, noise * abs(row[k]) + noise)
                ts.append(row)
            obs.append(Observation(
                id=f"o{len(obs)}", name="", description="",
                quantities={k: d for k, d in dims.items()},
                parameters={}, timesteps=ts,
                known_invariant=None, lean_theorem=""))
        return obs
    return gen


def run():
    print("=" * 70)
    print("PLASTIC LEARNING — sequential discovery")
    print("=" * 70)

    reset_plastic()

    # Define experiments: 15 claims across domains
    # Format: (name, symbols_with_dims, invariant_form, generator)
    experiments = []

    # Quantum domain — products and ratios
    for i in range(3):
        experiments.append((
            f"Quantum_{i}", "Quantum",
            {"E": "Energy", "lambda": "Scalar", "c": "Velocity"},
            "E*lambda",
            gen_product_data("E", "lambda", {"E": "Energy", "lambda": "Scalar", "c": "Velocity"})
        ))

    # Thermo domain — products
    for i in range(3):
        experiments.append((
            f"Thermo_{i}", "Thermo",
            {"P": "Pressure", "V": "Volume", "T": "Scalar"},
            "P*V",
            gen_product_data("P", "V", {"P": "Pressure", "V": "Volume", "T": "Scalar"})
        ))

    # Mechanics domain — ratios and powers
    for i in range(3):
        experiments.append((
            f"Mech_{i}", "Mechanics",
            {"F": "Force", "a": "Accel", "m": "Mass"},
            "F/a",
            gen_product_data("F", "a", {"F": "Force", "a": "Accel", "m": "Mass"})
        ))

    # EM domain — product with square
    for i in range(3):
        experiments.append((
            f"EM_{i}", "EM",
            {"F": "Force", "r": "Length", "q1": "Scalar"},
            "F*(r^2)",
            gen_product_data("F", "r", {"F": "Force", "r": "Length", "q1": "Scalar"})
        ))

    # Mixed domain — unfamiliar pairs
    for i in range(3):
        experiments.append((
            f"Mixed_{i}", "Mixed",
            {"v": "Velocity", "L": "Length", "g": "Accel"},
            "v^2*L",
            gen_product_data("v", "L", {"v": "Velocity", "L": "Length", "g": "Accel"})
        ))

    results = []

    for idx, (name, domain, qdim_dict, expected_form, gen_fn) in enumerate(experiments):
        rng = random.Random(42 + idx)
        observations = gen_fn(rng)

        qd = {}
        for qname, qdim_str in qdim_dict.items():
            try:
                qd[qname] = Dimension.named(qdim_str)
            except (ValueError, KeyError):
                qd[qname] = Dimension.scalar()

        t0 = time.time()
        discovery = auto_discover(
            quantities=qd, observations=observations,
            known_invariant=None, discovery_threshold=0.90,
            beam_expansions=2000,
        )
        elapsed = time.time() - t0

        passed = discovery.score >= 0.90
        state = get_plastic_state()

        status = "PASS" if passed else "FAIL"
        print(f"[{idx+1:2d}/15] [{domain:10s}] {name:15s}  "
              f"{status:4s}  score={discovery.score:.4f}  "
              f"found={discovery.expression[:30]:30s}  "
              f"plastic_entries={state['entries']:3d}  "
              f"norm={state['norm']:.4f}  "
              f"{elapsed:.1f}s")

        results.append({
            "name": name, "domain": domain,
            "passed": passed, "score": discovery.score,
            "expression": discovery.expression,
            "plastic_entries": state["entries"],
            "plastic_norm": state["norm"],
        })

    # Scorecard
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    print(f"\n{'=' * 70}")
    print(f"SCORECARD: {passed}/{total} ({passed/total*100:.0f}%)")

    # By domain
    from collections import defaultdict
    by_domain = defaultdict(lambda: {"total": 0, "passed": 0})
    for r in results:
        d = r["domain"]
        by_domain[d]["total"] += 1
        if r["passed"]:
            by_domain[d]["passed"] += 1

    for domain, counts in sorted(by_domain.items()):
        print(f"  {domain:15s}: {counts['passed']}/{counts['total']}")

    # Plastic growth
    first_half = results[:len(results)//2]
    second_half = results[len(results)//2:]
    fp = sum(1 for r in first_half if r["passed"])
    sp = sum(1 for r in second_half if r["passed"])
    print(f"\n  First half:  {fp}/{len(first_half)} ({fp/len(first_half)*100:.0f}%)")
    print(f"  Second half: {sp}/{len(second_half)} ({sp/len(second_half)*100:.0f}%)")

    final = get_plastic_state()
    print(f"\n  Final plastic: {final['entries']} entries, norm={final['norm']:.4f}")

    return results


if __name__ == "__main__":
    run()
