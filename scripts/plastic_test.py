#!/usr/bin/env python3
"""Provable plastic test — plastic breaks ties between equally-valid forms.

Design: Generate data where BOTH E*lambda and E/lambda score near 1.0.
Without plastic, the system can't distinguish. Plastic trained on
quantum claims that consistently use PRODUCTS biases toward E*lambda.

Honest: Both forms are mathematically valid from the data alone.
Plastic provides learned experience, not physics knowledge.
"""

from __future__ import annotations

import random, sys, time, math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.physics.dimensions import Dimension
from src.physics.observations import Observation
from src.physics.evaluator import ExpressionEvaluator
from src.math.plastic_seed_scorer import (
    score_seed, update_plastic, get_plastic_state, reset_plastic, _plastic_model,
)


def gen_ambiguous_data(rng, n_obs=2, n_ts=8):
    """Generate data where both E*lambda and E/lambda are near-constant.

    Trick: keep E and lambda anti-correlated within narrow ranges so
    both product and ratio show low variance.
    """
    obs = []
    for _ in range(n_obs):
        ts = []
        base_E = 2 + rng.random() * 8
        base_lambda = 2 + rng.random() * 8

        for _ in range(n_ts):
            # Vary E and lambda in small correlated steps
            delta = (rng.random() - 0.5) * 0.3
            E = base_E * (1 + delta)
            lam = base_lambda * (1 - delta * 0.9)  # anti-correlated
            # Add tiny noise
            E += rng.gauss(0, 0.01)
            lam += rng.gauss(0, 0.01)
            ts.append({'E': E, 'lambda': lam})

        obs.append(Observation(
            id=f"o{len(obs)}", name="", description="",
            quantities={'E': 'Energy', 'lambda': 'Scalar'},
            parameters={}, timesteps=ts,
            known_invariant=None, lean_theorem=""))
    return obs


def gen_product_data(rng, a, b, n_obs=3, n_ts=8):
    """Generate data where a*b = constant."""
    obs = []
    for _ in range(n_obs):
        K = 1 + rng.random() * 100
        ts = []
        for _ in range(n_ts):
            va = 0.5 + rng.random() * 10
            vb = K / va
            row = {a: va, b: vb}
            for extra in ['phi', 'm']:
                if extra not in (a, b):
                    row[extra] = rng.random() * 5
            ts.append(row)
        obs.append(Observation(
            id=f"o{len(obs)}", name="", description="",
            quantities={k: 'Energy' if k in ('E','K') else 'Scalar' for k in row},
            parameters={}, timesteps=ts,
            known_invariant=None, lean_theorem=""))
    return obs


def run():
    print("=" * 70)
    print("PROVABLE PLASTIC TEST — Tie-Breaking")
    print("=" * 70)

    # ── Ambiguous claim: E*lambda vs E/lambda ──
    symbols = ['E', 'lambda']
    product_form = "E*lambda"
    ratio_form = "E/lambda"

    # ── Phase 1: Baseline (no plastic) ──
    print("\nPhase 1: BASELINE (plastic OFF) — ambiguous data")
    reset_plastic()

    rng = random.Random(777)
    obs = gen_ambiguous_data(rng)
    ev = ExpressionEvaluator()

    def score_expr(expr):
        return sum(ev.score(expr, o) for o in obs) / len(obs)

    prod_score = score_expr(product_form)
    ratio_score = score_expr(ratio_form)
    diff = abs(prod_score - ratio_score)

    print(f"  {product_form:15s} constancy = {prod_score:.4f}")
    print(f"  {ratio_form:15s} constancy = {ratio_score:.4f}")
    print(f"  Difference: {diff:.4f} {'(TIE — indistinguishable)' if diff < 0.05 else '(distinct)'}")

    # Also check with plastic scoring (both start same)
    prod_plastic_before = score_seed(symbols, product_form)
    ratio_plastic_before = score_seed(symbols, ratio_form)
    print(f"  Plastic {product_form}: {prod_plastic_before:.4f}")
    print(f"  Plastic {ratio_form}: {ratio_plastic_before:.4f}")

    # ── Phase 2: Train plastic on PRODUCT claims ──
    print("\nPhase 2: Train plastic on 6 PRODUCT-only claims")

    from src.physics.search import auto_discover

    product_claims = [
        ("K*nu", gen_product_data(random.Random(100+i), 'K', 'nu'))
        for i in range(3)
    ] + [
        ("F*r", gen_product_data(random.Random(200+i), 'F', 'r'))
        for i in range(3)
    ]

    for name, obs in product_claims:
        qd = {}
        for o in obs:
            for qn, qs in o.quantities.items():
                if qn not in qd:
                    try: qd[qn] = Dimension.named(qs)
                    except: qd[qn] = Dimension.scalar()

        result = auto_discover(
            quantities=qd, observations=obs,
            known_invariant=None, discovery_threshold=0.90,
        )
        print(f"  {name}: {result.expression:10s} score={result.score:.4f}  "
              f"plastic={get_plastic_state()['entries']}")

    state = get_plastic_state()
    print(f"\n  Plastic: {state['entries']} entries, norm={state['norm']:.4f}")

    # ── Phase 3: Re-test with plastic ──
    print("\nPhase 3: RE-TEST (plastic ON) — same ambiguous data")

    prod_plastic_after = score_seed(symbols, product_form)
    ratio_plastic_after = score_seed(symbols, ratio_form)

    print(f"  {product_form:15s} plastic score: {prod_plastic_before:.4f} → {prod_plastic_after:.4f} "
          f"(Δ{prod_plastic_after - prod_plastic_before:+.4f})")
    print(f"  {ratio_form:15s} plastic score: {ratio_plastic_before:.4f} → {ratio_plastic_after:.4f} "
          f"(Δ{ratio_plastic_after - ratio_plastic_before:+.4f})")

    # Combined: constancy + plastic boost
    combined_prod = prod_score + 0.1 * prod_plastic_after
    combined_ratio = ratio_score + 0.1 * ratio_plastic_after

    print(f"\n  Combined: {product_form}={combined_prod:.4f}  {ratio_form}={combined_ratio:.4f}")

    # ── Verdict ──
    print(f"\n{'=' * 70}")
    prod_advantage = prod_plastic_after - ratio_plastic_after

    if prod_advantage > 0.01:
        print(f"✓ PLASTIC BREAKS TIE: Prefers {product_form} over {ratio_form}")
        print(f"  Product advantage: +{prod_advantage:.4f} plastic score")
        print(f"  All 6 training claims used product form → system learned product bias")
    elif prod_advantage > 0:
        print(f"~ MILD PREFERENCE: {product_form} +{prod_advantage:.4f} — direction correct")
    else:
        print(f"✗ NO EFFECT: Plastic didn't learn product preference")

    # Show learned entries
    if _plastic_model:
        print(f"\n  Plastic memory:")
        for (sym_key, expr_hash), bias in sorted(_plastic_model.memory.items(),
                                                   key=lambda x: -abs(x[1])):
            syms = ','.join(sorted(sym_key))
            print(f"    {{{syms}}} + {expr_hash} → bias={bias:+.4f}")

    print(f"{'=' * 70}")


if __name__ == "__main__":
    run()
