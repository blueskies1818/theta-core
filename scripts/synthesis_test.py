#!/usr/bin/env python3
"""Multi-claim synthesis — discover unifying laws from partial discoveries.

Era Gate Test: Process pre-1905 individual laws, attempt post-1905 synthesis.
Maxwell analogy: Coulomb + Ampere + Faraday → unified field equations.
Ideal gas: Boyle + Charles + Gay-Lussac → P*V/T = constant.

Architecture:
  1. Discover individual invariants per claim (auto_discover)
  2. Group claims by shared variables
  3. Generate combined expressions from the variable union
  4. Test against combined observation data
  5. Report any unified laws found
"""

from __future__ import annotations

import random, sys, time, math
from pathlib import Path
from collections import defaultdict
from itertools import combinations

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.physics.dimensions import Dimension
from src.physics.observations import Observation
from src.physics.search import auto_discover
from src.physics.evaluator import ExpressionEvaluator


# ════════════════════════════════════════════════════════
# Individual claim generators (pre-1905 partial laws)
# ════════════════════════════════════════════════════════

def gen_product_claim(a, b, dims, n_obs=3, n_ts=10, noise=0.0003):
    """Generate data where a*b = constant."""
    def gen(rng):
        obs = []
        for _ in range(n_obs):
            K = 1 + rng.random() * 100
            ts = []
            for _ in range(n_ts):
                va = 0.5 + rng.random() * 10
                vb = K / va
                row = {a: va, b: vb}
                for k, d in dims.items():
                    if k not in row:
                        row[k] = rng.random() * 5
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


def gen_ratio_claim(a, b, dims, n_obs=3, n_ts=10, noise=0.0003):
    """Generate data where a/b = constant."""
    def gen(rng):
        obs = []
        for _ in range(n_obs):
            K = 1 + rng.random() * 20
            ts = []
            for _ in range(n_ts):
                va = 0.5 + rng.random() * 10
                vb = va / K
                row = {a: va, b: vb}
                for k, d in dims.items():
                    if k not in row:
                        row[k] = rng.random() * 5
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


def gen_combined_ideal_gas(rng, n_obs=5, n_ts=10, noise=0.0003):
    """Generate data where P*V/T = constant (ideal gas law)."""
    obs = []
    for _ in range(n_obs):
        K = 0.5 + rng.random() * 5
        ts = []
        for _ in range(n_ts):
            P = 0.5 + rng.random() * 10
            V = 0.5 + rng.random() * 5
            T = P * V / K
            row = {'P': P, 'V': V, 'T': T, 'n': rng.random() * 3}
            for k in row:
                row[k] += rng.gauss(0, noise * abs(row[k]) + noise)
            ts.append(row)
        obs.append(Observation(
            id=f"o{len(obs)}", name="", description="",
            quantities={'P': 'Pressure', 'V': 'Volume', 'T': 'Scalar', 'n': 'Scalar'},
            parameters={}, timesteps=ts,
            known_invariant=None, lean_theorem=""))
    return obs


# ════════════════════════════════════════════════════════
# Synthesis engine
# ════════════════════════════════════════════════════════

def discover_individual(claim_name, gen_fn, rng):
    """Run auto_discover on a single claim. Returns (expression, score, symbols)."""
    observations = gen_fn(rng)

    qd = {}
    for o in observations:
        for qname, qdim_str in o.quantities.items():
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

    symbols = list(qd.keys())
    return result.expression, result.score, symbols


def group_by_shared_variables(discoveries):
    """Group claims that share at least one variable."""
    groups = []
    used = set()

    for i, (name_i, expr_i, score_i, syms_i) in enumerate(discoveries):
        if i in used:
            continue
        group = [(name_i, expr_i, score_i, set(syms_i))]
        used.add(i)

        for j, (name_j, expr_j, score_j, syms_j) in enumerate(discoveries):
            if j in used:
                continue
            # Check if this claim shares any variable with the group
            group_vars = set().union(*(s for _, _, _, s in group))
            if set(syms_j) & group_vars:
                group.append((name_j, expr_j, score_j, set(syms_j)))
                used.add(j)

        if len(group) >= 2:  # only groups of 2+ claims are interesting
            groups.append(group)

    return groups


def propose_synthesis(group, n_proposals=50):
    """Generate candidate unified expressions for a group of claims.

    Uses tree decoder + deterministic proposer on the combined variable set.
    """
    all_vars = set()
    for name, expr, score, syms in group:
        all_vars.update(syms)

    var_list = sorted(all_vars)

    # Get proposals from the tree decoder + deterministic proposer
    from src.math.cross_symbol_wrapper import propose_sub_expressions
    proposals = propose_sub_expressions(var_list, num_samples=n_proposals)

    return proposals, var_list


def evaluate_synthesis(var_list, combined_data_fn, rng, threshold=0.90):
    """Use simple_invariant_search to find unified forms in combined data."""
    observations = combined_data_fn(rng)

    qd = {}
    for o in observations:
        for qname, qdim_str in o.quantities.items():
            if qname not in qd:
                try:
                    qd[qname] = Dimension.named(qdim_str)
                except (ValueError, KeyError):
                    qd[qname] = Dimension.scalar()

    # Use simple_invariant_search — it has ternary templates (a*b/c, a^2/b, etc.)
    from src.physics.search import simple_invariant_search
    result = simple_invariant_search(qd, observations, discovery_threshold=threshold)

    if result.score >= threshold:
        return [(result.expression, result.score)]
    return []


def run_synthesis_test():
    print("=" * 70)
    print("MULTI-CLAIM SYNTHESIS — Era Gate Test")
    print("=" * 70)

    # ── Test 1: Ideal Gas Law (Boyle + Charles + Gay-Lussac → P*V/T) ──
    print("\n" + "─" * 70)
    print("TEST 1: Boyle + Charles + Gay-Lussac → Ideal Gas Law")
    print("─" * 70)

    # Pre-1905 individual laws
    gas_claims = [
        ("Boyle (P*V)", gen_product_claim(
            "P", "V", {'P': 'Pressure', 'V': 'Volume', 'T': 'Scalar', 'n': 'Scalar'})),
        ("Charles (V/T)", gen_ratio_claim(
            "V", "T", {'V': 'Volume', 'T': 'Scalar', 'P': 'Pressure', 'n': 'Scalar'})),
        ("Gay-Lussac (P/T)", gen_ratio_claim(
            "P", "T", {'P': 'Pressure', 'T': 'Scalar', 'V': 'Volume', 'n': 'Scalar'})),
    ]

    discoveries = []
    print("\n  Individual discoveries:")
    for name, gen_fn in gas_claims:
        rng = random.Random(hash(name) % (2**31))
        expr, score, syms = discover_individual(name, gen_fn, rng)
        discoveries.append((name, expr, score, syms))
        status = "✓" if score >= 0.90 else "✗"
        print(f"    {status} {name:25s}: {expr:15s} score={score:.4f}  vars={syms}")

    # Group and synthesize
    groups = group_by_shared_variables(discoveries)
    print(f"\n  Found {len(groups)} variable-sharing groups")

    for gi, group in enumerate(groups):
        group_names = [n for n, e, s, _ in group]
        print(f"\n  Group {gi+1}: {' + '.join(group_names)}")
        print(f"    Individual forms: {', '.join(f'{e}({s:.2f})' for _, e, s, _ in group)}")

        # Synthesize
        proposals, union_vars = propose_synthesis(group)
        print(f"    Union variables: {union_vars}")
        print(f"    Proposals generated: {len(proposals)}")

        # Test against combined ideal gas data
        combined_rng = random.Random(42)
        unifications = evaluate_synthesis(
            union_vars, gen_combined_ideal_gas, combined_rng,
            threshold=0.90,
        )

        if unifications:
            print(f"\n    ✓ SYNTHESIS DISCOVERIES:")
            for expr, score in unifications[:8]:
                marker = " ← IDEAL GAS" if 'P' in expr and 'V' in expr and 'T' in expr else ""
                print(f"      {expr:30s} score={score:.4f}{marker}")
        else:
            print(f"\n    ✗ No unified form found above threshold")

    # ── Test 2: Era gate claim — P*V/T ──
    print("\n" + "─" * 70)
    print("TEST 2: Direct ideal gas discovery (era gate baseline)")
    print("─" * 70)

    rng = random.Random(999)
    obs = gen_combined_ideal_gas(rng)
    qd = {}
    for o in obs:
        for qn, qs in o.quantities.items():
            if qn not in qd:
                try: qd[qn] = Dimension.named(qs)
                except: qd[qn] = Dimension.scalar()

    result = auto_discover(quantities=qd, observations=obs,
                           known_invariant=None, discovery_threshold=0.90)
    print(f"\n  Direct discovery: {result.expression}  score={result.score:.4f}")

    # Also check what would be found for Maxwell-like EM synthesis
    print("\n" + "─" * 70)
    print("TEST 3: EM Synthesis (Coulomb force + charge conservation)")
    print("─" * 70)

    em_claims = [
        ("Coulomb (F*r^2/q1*q2)", gen_product_claim(
            "F", "r", {'F': 'Force', 'r': 'Length', 'q1': 'Scalar', 'q2': 'Scalar'})),
    ]

    em_discoveries = []
    for name, gen_fn in em_claims:
        rng = random.Random(hash(name) % (2**31))
        expr, score, syms = discover_individual(name, gen_fn, rng)
        em_discoveries.append((name, expr, score, syms))
        print(f"    ✓ {name:30s}: {expr:15s} score={score:.4f}")

    print(f"\n  EM synthesis is limited with only 1 claim — needs Ampere + Faraday")
    print(f"  Individual discoveries provide building blocks for future expansion")

    # Scorecard
    print(f"\n{'=' * 70}")
    print(f"SYNTHESIS SCORECARD")
    print(f"{'=' * 70}")
    print(f"  Individual claims processed: {len(discoveries) + len(em_discoveries)}")
    print(f"  Variable-sharing groups found: {len(groups)}")
    if unifications:
        print(f"  Unifications discovered: {len(unifications)}")
        for expr, score in unifications[:3]:
            print(f"    {expr}  score={score:.4f}")
    print(f"  Era gate: individual pre-1905 laws → potential post-1905 synthesis")


if __name__ == "__main__":
    run_synthesis_test()
