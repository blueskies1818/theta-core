#!/usr/bin/env python3
"""Synthesize Phase D template generator training data.

Generates diverse (quantities, domain, expression) triples using
domain-specific physical symbols with dimensionally-valid patterns.
Uses compound expressions to enable diverse sums and squared differences.

Output: data/self_play_phase_d.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

_project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_project_root))

from src.physics.dimensions import Dimension

# ── DOMAIN SYMBOL POOLS (pre-1905 only) ────────────────────────────────────────
# Symbol names must match composer.py TEMPLATE_VOCAB (QUANTITY_VOCAB entries)

DOMAIN_SYMBOL_DIMS: dict[str, dict[str, Dimension]] = {
    "gravity": {
        "m": Dimension.named("Mass"),
        "g": Dimension.named("Accel"),
        "h": Dimension.named("Length"),
        "v": Dimension.named("Velocity"),
        "t": Dimension.named("Time"),
        "x": Dimension.named("Length"),
        "E": Dimension.named("Energy"),
    },
    "spring": {
        "m": Dimension.named("Mass"),
        "k": Dimension.named("Force") / Dimension.named("Length"),
        "h": Dimension.named("Length"),
        "v": Dimension.named("Velocity"),
        "t": Dimension.named("Time"),
        "x": Dimension.named("Length"),
        "E": Dimension.named("Energy"),
    },
    "em": {
        "m": Dimension.named("Mass"),
        "v": Dimension.named("Velocity"),
        "q": Dimension.scalar(),
        "E": Dimension.named("Energy") / Dimension.named("Length"),
        "B": Dimension.named("Force") / Dimension.named("Velocity"),
        "x": Dimension.named("Length"),
        "t": Dimension.named("Time"),
    },
    "thermal": {
        "P": Dimension.named("Pressure"),
        "V": Dimension.named("Volume"),
        "T": Dimension.scalar(),
        "n": Dimension.scalar(),
        "R": Dimension.named("Energy"),
        "S": Dimension.scalar(),
        "W": Dimension.named("Energy"),
        "Q": Dimension.named("Energy"),
        "t": Dimension.named("Time"),
    },
}

import random as _random

# Regex to find multi-char tokens in expressions
_TOKEN_RE = re.compile(r'[a-zA-Z_]\w*|\d+\.?\d*|\S')


def _extract_vars(expr: str) -> list[str]:
    """Extract variable names from expression string."""
    tokens = _TOKEN_RE.findall(expr)
    # Filter: non-numeric, non-operator tokens
    ops = {"+", "-", "*", "/", "^", "(", ")"}
    vars_ = set()
    for tok in tokens:
        if tok in ops:
            continue
        try:
            float(tok)
            continue
        except ValueError:
            pass
        # Alphanumeric tokens are variables
        if any(c.isalpha() for c in tok):
            vars_.add(tok)
    return sorted(vars_)


def _dim_of(expr: str, dims: dict[str, Dimension]) -> Dimension | None:
    """Compute dimension of an expression string."""
    # Handle single symbol
    if expr in dims:
        return dims[expr]
    # Handle coefficient*var: "0.5*m", "2*E"
    parts = re.split(r'\*', expr)
    result_dim = Dimension.scalar()
    for part in parts:
        part = part.strip()
        if part in ("0.5", "2", "-2", "-1", "3", "4", "0", "1"):
            continue
        if part in dims:
            try:
                result_dim = result_dim * dims[part]
            except Exception:
                return None
        elif "^" in part:
            s = part.split("^", 1)
            if s[0] in dims:
                try:
                    result_dim = result_dim * (dims[s[0]] ** float(s[1]))
                except Exception:
                    return None
            else:
                return None
        else:
            return None
    return result_dim


def _pick_quantities(dims: dict[str, Dimension],
                     rng: _random.Random, n: int | None = None) -> list[str]:
    symbols = sorted(dims.keys())
    if n is None:
        n = rng.randint(2, min(4, len(symbols)))
    return sorted(rng.sample(symbols, min(n, len(symbols))))


def _gen_product(qties: list[str], _rng: _random.Random) -> str:
    return "*".join(qties)


def _gen_ratio(qties: list[str], _rng: _random.Random) -> str:
    if len(qties) == 2:
        return f"{qties[0]}/{qties[1]}"
    num = "*".join(qties[:-1])
    return f"({num})/{qties[-1]}"


def _gen_power(qties: list[str], rng: _random.Random) -> str:
    exp = rng.choice(["2", "-1"])
    base = qties[0]
    if len(qties) == 1:
        return f"{base}^{exp}"
    rest = "*".join(qties[1:])
    return f"{base}^{exp}*{rest}"


def _gen_mixed(qties: list[str], rng: _random.Random) -> str:
    if len(qties) < 3:
        return _gen_power(qties, rng)
    exp = rng.choice(["2", "-1"])
    return f"{qties[0]}^{exp}*{qties[1]}/{qties[2]}"


def _build_compound_terms(dims: dict[str, Dimension],
                           rng: _random.Random) -> list[tuple[str, Dimension]]:
    """Build diverse compound terms with their dimensions."""
    symbols = sorted(dims.keys())
    terms: list[tuple[str, Dimension]] = []

    # Single variable + coefficient variants
    for sym in symbols:
        d = dims[sym]
        terms.append((sym, d))
        for coeff in ["0.5", "2"]:
            terms.append((f"{coeff}*{sym}", d))

    # Products of 2-3 variables
    for _ in range(60):
        n_sym = rng.randint(2, min(3, len(symbols)))
        chosen = rng.sample(symbols, n_sym)
        expr = "*".join(chosen)
        d = _dim_of(expr, dims)
        if d is not None:
            terms.append((expr, d))

    # Powers
    for _ in range(30):
        sym = rng.choice(symbols)
        exp = rng.choice(["2", "-1"])
        expr = f"{sym}^{exp}"
        d = _dim_of(expr, dims)
        if d is not None:
            terms.append((expr, d))

    # Ratios
    for _ in range(30):
        if len(symbols) < 2:
            break
        q1, q2 = rng.sample(symbols, 2)
        expr = f"{q1}/{q2}"
        d = _dim_of(expr, dims)
        if d is not None:
            terms.append((expr, d))

    # Product with power: a^2*b
    for _ in range(30):
        if len(symbols) < 2:
            break
        q1, q2 = rng.sample(symbols, 2)
        exp = rng.choice(["2", "-1"])
        expr = f"{q1}^{exp}*{q2}"
        d = _dim_of(expr, dims)
        if d is not None:
            terms.append((expr, d))

    return terms


def _gen_sum(dims: dict[str, Dimension],
             rng: _random.Random) -> tuple[str, list[str]] | None:
    """Generate a 2-term sum of same-dimension expressions."""
    terms = _build_compound_terms(dims, rng)
    by_dim: dict[str, list[str]] = defaultdict(list)
    for expr, dim in terms:
        dim_key = str(dim)
        by_dim[dim_key].append(expr)

    # Find dimension classes with 2+ distinct terms
    usable = [(d, ts) for d, ts in by_dim.items() if len(set(ts)) >= 2]
    if not usable:
        return None

    dim_key, group = rng.choice(usable)
    unique_terms = list(set(group))
    if len(unique_terms) < 2:
        return None

    t1, t2 = rng.sample(unique_terms, 2)

    # Avoid trivial t1+t1
    if t1 == t2:
        return None

    # Avoid self-cancelling: 2*x + x where one's a coefficient of the other
    t1_base = re.sub(r'^[\d.]+[\*]?', '', t1)
    t2_base = re.sub(r'^[\d.]+[\*]?', '', t2)
    if t1_base == t2_base:
        return None

    expr = f"{t1}+{t2}"
    vars_used = _extract_vars(expr)
    return expr, vars_used


def _gen_squared_diff(dims: dict[str, Dimension],
                       rng: _random.Random) -> tuple[str, list[str]] | None:
    """Generate squared difference: q1^2 - q2^2.

    Uses two strategies:
    1. Same-dimension pairs (e.g., h^2 - x^2) — dimensionally valid
    2. Any two distinct symbols (e.g., m^2 - v^2) — teaches structural pattern
       even across different dimensions (template generator learns text patterns)
    """
    symbols = sorted(dims.keys())
    if len(symbols) < 2:
        return None

    # Strategy 1: Same-dimension pairs (40% chance)
    if rng.random() < 0.4:
        terms = [(sym, str(d)) for sym, d in dims.items()]
        for sym, d in dims.items():
            for coeff in ["0.5", "2"]:
                terms.append((f"{coeff}*{sym}", str(d)))
        by_dim: dict[str, list[str]] = defaultdict(list)
        for expr, dim_key in terms:
            by_dim[dim_key].append(expr)
        usable = [(d, ts) for d, ts in by_dim.items() if len(set(ts)) >= 2]
        if usable:
            dim_key, group = rng.choice(usable)
            unique_terms = list(set(group))
            if len(unique_terms) >= 2:
                t1, t2 = rng.sample(unique_terms, 2)
                if t1 != t2:
                    t1_base = re.sub(r'^[\d.]+[\*]?', '', t1)
                    t2_base = re.sub(r'^[\d.]+[\*]?', '', t2)
                    if t1_base != t2_base:
                        if rng.random() < 0.4:
                            c1 = rng.choice(["0.5", "2"])
                            c2 = rng.choice(["0.5", "2"])
                            expr = f"{c1}*{t1}^2-{c2}*{t2}^2"
                        else:
                            expr = f"{t1}^2-{t2}^2"
                        vars_used = _extract_vars(expr)
                        return expr, vars_used

    # Strategy 2: Any two symbols (teaches structural q1^2-q2^2 pattern)
    s1, s2 = rng.sample(symbols, 2)
    if rng.random() < 0.5:
        # Plain: s1^2 - s2^2
        expr = f"{s1}^2-{s2}^2"
    elif rng.random() < 0.6:
        # With coefficients: 0.5*s1^2 - 2*s2^2
        c1 = rng.choice(["0.5", "2"])
        c2 = rng.choice(["0.5", "2"])
        expr = f"{c1}*{s1}^2-{c2}*{s2}^2"
    else:
        # With single coefficient: 0.5*s1^2 - s2^2
        c1 = rng.choice(["0.5", "2"])
        expr = f"{c1}*{s1}^2-{s2}^2"

    vars_used = _extract_vars(expr)
    return expr, vars_used


def _validate_expression(expr: str) -> bool:
    """Check if expression is parseable."""
    try:
        from src.physics.evaluator import ExpressionEvaluator
        ev = ExpressionEvaluator()
        ev.parse(expr)
        return True
    except Exception:
        return False


def synthesize_training_data(
    n_per_domain: int = 2500,
    seed: int = 42,
    output_path: str | Path = "data/self_play_phase_d.jsonl",
) -> list[dict]:
    rng = _random.Random(seed)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    level_weights = {1: 0.40, 2: 0.35, 3: 0.25}
    examples: list[dict] = []
    domain_counts: dict[str, int] = defaultdict(int)

    print(f"Target: {n_per_domain} per domain ({n_per_domain * 4} total)")

    for dom, dims in DOMAIN_SYMBOL_DIMS.items():
        symbols = sorted(dims.keys())
        print(f"\n  {dom}: {len(symbols)} symbols")
        seen_exprs: set[str] = set()
        attempts = 0

        while domain_counts[dom] < n_per_domain and attempts < n_per_domain * 25:
            attempts += 1

            level = rng.choices(
                list(level_weights.keys()),
                weights=list(level_weights.values()),
                k=1,
            )[0]

            if level == 1:
                qties = _pick_quantities(dims, rng)
                pattern = rng.choice([_gen_product, _gen_ratio, _gen_power, _gen_mixed])
                expr = pattern(qties, rng)
                vars_used = _extract_vars(expr)
            elif level == 2:
                result = _gen_sum(dims, rng)
                if result is None:
                    continue
                expr, vars_used = result
            else:
                result = _gen_squared_diff(dims, rng)
                if result is None:
                    continue
                expr, vars_used = result

            if not expr or len(expr) < 3:
                continue
            if expr in seen_exprs:
                continue
            if not _validate_expression(expr):
                continue
            if len(vars_used) < 2:
                continue

            seen_exprs.add(expr)
            record = {
                "quantities": sorted(vars_used),
                "domain": dom,
                "expression": expr,
                "match": "exact_match",
                "complexity_level": level,
            }
            examples.append(record)
            domain_counts[dom] += 1

    with open(output_path, "w") as f:
        for ex in examples:
            json.dump(ex, f)
            f.write("\n")

    print(f"\nCollected {len(examples)}")
    print(f"Domain distribution: {dict(sorted(domain_counts.items()))}")
    print(f"Saved to: {output_path}")
    return examples


def main() -> int:
    parser = argparse.ArgumentParser(description="Synthesize Phase D training data")
    parser.add_argument("--n-per-domain", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="data/self_play_phase_d.jsonl")
    args = parser.parse_args()
    output_path = _project_root / args.output
    synthesize_training_data(
        n_per_domain=args.n_per_domain,
        seed=args.seed,
        output_path=output_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
