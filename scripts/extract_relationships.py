#!/usr/bin/env python3
"""Extract learned relationships from plastic memory.

Processes all 28 claims through the plastic system, then reads back
the structural patterns the system learned are important — without
any human interpretation. These are genuine extracted relationships,
not hand-coded rules.
"""

from __future__ import annotations

import random, sys, time
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.physics.dimensions import Dimension
from src.physics.search import auto_discover
from src.math.plastic_seed_scorer import (
    get_plastic_state, reset_plastic, _plastic_model,
)


def run_all_claims():
    """Process all claims and extract plastic relationships."""

    print("=" * 70)
    print("PLASTIC RELATIONSHIP EXTRACTION")
    print("Process all claims → read learned structural preferences")
    print("=" * 70)

    reset_plastic()

    # Load all claim generators
    from scripts.verify_instruments import CLAIMS as CLEAN_CLAIMS
    from scripts.verify_generalized import CLAIMS as GEN_CLAIMS

    all_claims = []
    for domain, name, invariant, gen_fn in CLEAN_CLAIMS:
        all_claims.append((f"clean/{domain}", name, invariant, gen_fn))
    for claim in GEN_CLAIMS:
        all_claims.append((f"gen/{claim.domain}", claim.name,
                           claim.invariant_form, claim.generator))

    print(f"\nProcessing {len(all_claims)} claims...\n")

    discoveries = []
    failures = []

    for i, (source, name, invariant_form, gen_fn) in enumerate(all_claims):
        rng = random.Random(42 + i)
        observations = gen_fn(rng)

        qd = {}
        for obs in observations:
            for qname, qdim_str in obs.quantities.items():
                if qname not in qd:
                    try:
                        qd[qname] = Dimension.named(qdim_str)
                    except (ValueError, KeyError):
                        qd[qname] = Dimension.scalar()

        result = auto_discover(
            quantities=qd, observations=observations,
            known_invariant=None, discovery_threshold=0.90,
            beam_expansions=2000,
        )

        status = "✓" if result.score >= 0.90 else "✗"
        short_name = name[:30]

        if result.score >= 0.90:
            discoveries.append((source, name, invariant_form, result.expression))
        else:
            failures.append((source, name, invariant_form, result.expression))

        if (i + 1) % 7 == 0:
            state = get_plastic_state()
            print(f"  [{i+1:2d}/{len(all_claims)}] plastic: {state['entries']} entries, "
                  f"norm={state['norm']:.4f}")

    # ── Extract plastic relationships ──
    state = get_plastic_state()
    print(f"\n{'=' * 70}")
    print(f"RESULTS: {len(discoveries)}/{len(all_claims)} discovered"
          f" ({len(discoveries)/len(all_claims)*100:.0f}%)")
    print(f"Plastic memory: {state['entries']} structural patterns learned")
    print(f"{'=' * 70}")

    # Re-import to ensure we have the current module state
    import src.math.plastic_seed_scorer as pss
    model = pss._plastic_model
    
    if model is None or not model.memory:
        print("\n  No plastic relationships extracted.")
        return

    # ── Relationships by structural form ──
    print("\n─── Learned Structural Preferences ───")
    print("  (positive bias = form was useful, negative = form was unhelpful)")
    print()

    sorted_entries = sorted(model.memory.items(),
                            key=lambda x: -x[1])

    # Group by number of variables
    by_nvars = defaultdict(list)
    for (n_vars, pattern), bias in sorted_entries:
        by_nvars[n_vars].append((pattern, bias))

    for n_vars in sorted(by_nvars):
        entries = by_nvars[n_vars]
        print(f"  {n_vars}-variable forms:")
        for pattern, bias in sorted(entries, key=lambda x: -x[1]):
            bar = "█" * max(1, int(abs(bias) * 50))
            sign = "+" if bias > 0 else " "
            print(f"    {sign}{pattern:12s} {bias:+.4f}  {bar}")

    # ── Summary of learned knowledge ──
    print(f"\n─── Extracted Knowledge ───")
    print("  (These are genuine discoveries from experience, not hand-coded)")

    interpretations = []

    # Find dominant form
    if sorted_entries:
        top_form, top_bias = sorted_entries[0]
        n_vars, pattern = top_form
        form_name = {
            'a*b': 'product (multiplication)',
            'a/b': 'ratio (division)',
            'a^2*b': 'power-law with product',
            'a*b/c': 'triple product/ratio',
            'a+b': 'sum (addition)',
            'a-b': 'difference (subtraction)',
            'a^2': 'square (power law)',
        }.get(pattern, pattern)

        if top_bias > 0.05:
            interpretations.append(
                f"  • The system learned that {form_name} is the most\n"
                f"    reliable structural form for invariants (bias={top_bias:+.3f})"
            )

    # Count forms by sign
    positive = sum(1 for _, b in sorted_entries if b > 0)
    negative = sum(1 for _, b in sorted_entries if b < 0)
    if positive > negative:
        interpretations.append(
            f"  • {positive}/{positive+negative} learned forms are positive —\n"
            f"    most structural patterns the system encountered were useful"
        )

    # Products vs ratios
    prod_entries = [b for (_, p), b in sorted_entries if '*' in p and '/' not in p]
    ratio_entries = [b for (_, p), b in sorted_entries if '/' in p and '*' not in p]
    mixed_entries = [b for (_, p), b in sorted_entries if '*' in p and '/' in p]

    if prod_entries and ratio_entries:
        avg_prod = sum(prod_entries) / len(prod_entries)
        avg_ratio = sum(ratio_entries) / len(ratio_entries)
        if abs(avg_prod - avg_ratio) > 0.01:
            winner = "products" if avg_prod > avg_ratio else "ratios"
            interpretations.append(
                f"  • {winner} are preferred over {'ratios' if winner == 'products' else 'products'}\n"
                f"    (avg product bias={avg_prod:+.3f}, avg ratio bias={avg_ratio:+.3f})"
            )

    if mixed_entries:
        avg_mixed = sum(mixed_entries) / len(mixed_entries)
        interpretations.append(
            f"  • Mixed product/ratio forms (ternary) have bias={avg_mixed:+.3f} —\n"
            f"    {'useful when needed' if avg_mixed > 0 else 'less reliable than binary forms'}"
        )

    for interp in interpretations:
        print(interp)

    # ── Raw memory dump ──
    print(f"\n─── Raw Plastic Memory ───")
    for (n_vars, pattern), bias in sorted_entries:
        hit_count = int(abs(bias) / 0.02)  # approximate
        print(f"  ({n_vars}, {pattern:15s}) → bias={bias:+.4f}  "
              f"(~{hit_count} positive reinforcements)")


if __name__ == "__main__":
    run_all_claims()
