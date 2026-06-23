#!/usr/bin/env python3
"""General hidden variable discovery pipeline.

Runs HiddenVariableDiscovery on observation data. Two modes:

1. GENERAL MODE (default): Takes a data file, auto-detects hidden variables,
   runs discovery. No physics knowledge.

2. SCENARIO MODE (--scenario): Runs specific named scenarios for regression
   testing. Uses scenario-specific loaders for known data formats.

Usage:
    # General: discover invariants in any observation database
    python scripts/hidden_var_closed_loop.py data/observations/quantum_synthetic.json --index 0

    # General: with explicit hidden variable
    python scripts/hidden_var_closed_loop.py data/observations/quantum_synthetic.json --hidden n

    # Scenario regression: test known scenarios
    python scripts/hidden_var_closed_loop.py --scenario particle_in_box
    python scripts/hidden_var_closed_loop.py --scenario all
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.physics.dimensions import Dimension
from src.physics.hidden_variables import (
    HiddenVariableProposer, HiddenVariableDiscovery,
    load_hidden_var_proposer,
)
from src.physics.noise import RealExperimentalLoader
from src.physics.observations import Observation, ObservationDatabase
from src.physics.search import ExpressionSearch, SearchResult

SEED = 42
DISCOVERY_THRESHOLD = 0.90


# ═══════════════════════════════════════════════════════════════════════════
# Dimension helpers — general, no physics-specific names
# ═══════════════════════════════════════════════════════════════════════════

def _dim_from_str(dim_str: str) -> Dimension:
    """Convert a dimension string to a Dimension object."""
    try:
        return Dimension.named(str(dim_str))
    except (ValueError, KeyError):
        pass
    # Compound dimensions via algebraic composition
    aliases: dict[str, Dimension] = {}
    for alias, (a, op, b) in [
        ("Action", ("Energy", "*", "Time")),
        ("Momentum", ("Mass", "*", "Velocity")),
        ("Frequency", ("1", "/", "Time")),
        ("InverseLength", ("1", "/", "Length")),
        ("Energy*Time", ("Energy", "*", "Time")),
        ("1/Time", ("1", "/", "Time")),
    ]:
        if op == "*":
            aliases[alias] = (Dimension.scalar() if a == "1" else Dimension.named(a)) * Dimension.named(b)
        elif op == "/":
            aliases[alias] = (Dimension.scalar() if a == "1" else Dimension.named(a)) / Dimension.named(b)
    if str(dim_str) in aliases:
        return aliases[str(dim_str)]
    return Dimension.scalar()


# ═══════════════════════════════════════════════════════════════════════════
# General data preprocessing — no physics knowledge
# ═══════════════════════════════════════════════════════════════════════════

def _expand_indexed_params(obs: Observation) -> Observation:
    """Expand an observation with indexed params into config-spanning timesteps.

    General structural rule: if a quantity Q has no value but Q1..Qk exist
    in params, create one observation with k timesteps, each with Q = Q_i.
    """
    available = set(obs.parameters.keys())
    for ts in obs.timesteps:
        available.update(ts.keys())
    unresolved = set(obs.quantities.keys()) - available

    indexed: dict[str, dict[int, float]] = {}
    for qname in unresolved:
        vals: dict[int, float] = {}
        for pkey, pval in obs.parameters.items():
            if pkey.startswith(qname) and pkey != qname:
                suffix = pkey[len(qname):]
                if suffix.isdigit():
                    vals[int(suffix)] = float(pval)
        if len(vals) >= 2:
            indexed[qname] = vals

    if not indexed:
        return obs

    first_q = next(iter(indexed.values()))
    indices = sorted(first_q.keys())
    new_ts = []
    for idx in indices:
        base = dict(obs.timesteps[0]) if obs.timesteps else {}
        for qname, idx_vals in indexed.items():
            if idx in idx_vals:
                base[qname] = idx_vals[idx]
        new_ts.append(base)

    # Inject configuration indices for any scalar-dimension quantities
    # (these are natural values for hidden integer variables like n)
    for qname, dim_str in obs.quantities.items():
        if str(dim_str) == "Scalar":
            for i, ts in enumerate(new_ts):
                ts[qname] = float(i + 1)

    new_params = dict(obs.parameters)
    first_idx = indices[0]
    for qname, idx_vals in indexed.items():
        if first_idx in idx_vals:
            new_params[qname] = idx_vals[first_idx]

    return Observation(
        id=obs.id, name=obs.name, description=obs.description,
        quantities=obs.quantities, parameters=new_params, timesteps=new_ts,
        known_invariant=obs.known_invariant, lean_theorem=obs.lean_theorem,
        external_forces=obs.external_forces, phase_regions=obs.phase_regions,
        is_conservative=obs.is_conservative,
    )


def _find_hidden_variable(obs: Observation) -> str | None:
    """Auto-detect hidden variable: first dimensionless quantity.

    Treats 'Scalar', 'Dimensionless', 'Number', or any dimension
    that resolves to scalar as candidate hidden variables.
    """
    scalar_aliases = {"Scalar", "Dimensionless", "Number", "Angle", "Charge"}
    for qname, dim_str in obs.quantities.items():
        if str(dim_str) in scalar_aliases:
            return qname
        # Check if the dimension resolves to scalar
        d = _dim_from_str(str(dim_str))
        if d.is_scalar():
            return qname
    return None


def build_quantities(obs: Observation, *, strip: str | None = None) -> dict[str, Dimension]:
    """Build quantities dict, stripping the hidden variable if specified."""
    scalar_aliases = {"Scalar", "Dimensionless", "Number", "Angle", "Charge"}
    quantities: dict[str, Dimension] = {}
    for qname, dim_str in obs.quantities.items():
        if strip and qname == strip:
            continue
        if strip is None and (str(dim_str) in scalar_aliases or _dim_from_str(str(dim_str)).is_scalar()):
            continue
        quantities[qname] = _dim_from_str(str(dim_str))
    return quantities


# ═══════════════════════════════════════════════════════════════════════════
# Beam search adapter
# ═══════════════════════════════════════════════════════════════════════════

class _Adapter:
    def __init__(self, search, result):
        self._search = search; self._result = result
        self.best_expression = result.expression
        self.best_score = result.score
        self.discovered = result.score >= DISCOVERY_THRESHOLD
        self._scored = search._scored


def make_beam_fn(max_expansions=5000):
    def beam_fn(q, obs_list):
        s = ExpressionSearch(quantities=q, train_observations=obs_list,
                             max_depth=8, max_expansions=max_expansions,
                             discovery_threshold=DISCOVERY_THRESHOLD)
        return _Adapter(s, s.run())
    return beam_fn


# ═══════════════════════════════════════════════════════════════════════════
# General discovery runner
# ═══════════════════════════════════════════════════════════════════════════

def discover(
    observations: list[Observation],
    *,
    hidden: str | None = None,
    max_expansions: int = 5000,
    proposer_path: str | None = None,
) -> dict[str, Any]:
    """Run discovery on observations. Returns result dict."""
    if not observations:
        return {"error": "no observations"}

    t0 = time.time()
    obs = _expand_indexed_params(observations[0])
    hidden_var = hidden or _find_hidden_variable(obs)
    quantities = build_quantities(obs, strip=hidden_var)

    ckpt = Path(proposer_path) if proposer_path else (
        PROJECT_ROOT / "checkpoints" / "hidden_var_proposer.pt")
    proposer = load_hidden_var_proposer(str(ckpt)) if ckpt.exists() else HiddenVariableProposer()

    beam_fn = make_beam_fn(max_expansions=max_expansions)
    discovery = HiddenVariableDiscovery(
        proposer=proposer, max_proposals=5, discovery_threshold=DISCOVERY_THRESHOLD)

    result = discovery.discover(
        quantities=quantities, observations=[obs], beam_search_fn=beam_fn,
        domain="unknown", quantity_names=list(quantities.keys()))

    return {
        "hidden_stripped": hidden_var,
        "quantities": list(quantities.keys()),
        "num_observations": len(observations),
        "discovered": result.discovered,
        "expression": result.best_expression,
        "score": result.best_score,
        "baseline_score": result.baseline_score,
        "hidden_found": result.hidden_variable,
        "transform": result.transform,
        "proposals": [{"type": p.variable_type, "name": p.variable_name,
                       "transform": p.transform, "conf": p.confidence}
                      for p in result.proposals],
        "error_shape": result.error_analysis.shape if result.error_analysis else "?",
        "error_conf": result.error_analysis.shape_confidence if result.error_analysis else 0,
        "time": time.time() - t0,
        "name": obs.name,
        "known_invariant": obs.known_invariant,
    }


# ═══════════════════════════════════════════════════════════════════════════
# General data loading (ObservationDatabase + RealExperimental formats)
# ═══════════════════════════════════════════════════════════════════════════

def load_observations(data_path: str | Path) -> list[Observation]:
    """Load observations from a data file, auto-detecting format.

    Supports:
    - ObservationDatabase JSON (list of Observation objects)
    - Real Experimental JSON (single dataset with data_points)
    - Directory of real experimental JSON files
    """
    path = Path(data_path)

    # Format 1: Directory → use RealExperimentalLoader
    if path.is_dir():
        loader = RealExperimentalLoader(path)
        all_obs = []
        for ds in loader.load_all():
            all_obs.extend(ds.to_synthetic_observations(num_bootstrap=1))
        return all_obs

    # Format 2: Try ObservationDatabase first
    try:
        db = ObservationDatabase(str(path))
        obs_list = list(db)
        if obs_list:
            return obs_list
    except Exception:
        pass

    # Format 3: Single JSON file in RealExperimental format
    try:
        with open(path) as f:
            raw = json.load(f)
        # Check if it has data_points (RealExperimental format)
        if "data_points" in raw:
            loader = RealExperimentalLoader(path.parent)
            for ds in loader.load_all():
                if ds.source == raw.get("source", ""):
                    return ds.to_synthetic_observations(num_bootstrap=1)
            # If source doesn't match, try loading all from parent dir
            all_from_dir = []
            for ds in loader.load_all():
                all_from_dir.extend(ds.to_synthetic_observations(num_bootstrap=1))
            if all_from_dir:
                return all_from_dir
    except Exception:
        pass

    raise ValueError(f"Cannot load observations from {path}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="General hidden variable discovery")
    p.add_argument("data_file", nargs="?", help="Observation database JSON")
    p.add_argument("--hidden", help="Name of hidden variable (auto if omitted)")
    p.add_argument("--index", type=int, help="Observation index (all if omitted)")
    p.add_argument("--max-expansions", type=int, default=5000)
    p.add_argument("--output", help="Output JSON file")
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not args.data_file:
        p.print_help()
        sys.exit(1)

    db_path = Path(args.data_file)
    if not db_path.exists():
        print(f"Error: {db_path} not found", file=sys.stderr)
        sys.exit(1)

    all_obs = load_observations(db_path)

    if args.index is not None:
        if args.index >= len(all_obs):
            print(f"Error: index {args.index} out of range ({len(all_obs)})", file=sys.stderr)
            sys.exit(1)
        test_obs = [all_obs[args.index]]
    else:
        test_obs = all_obs

    print(f"{'='*60}")
    print(f"Hidden Variable Discovery — {db_path}")
    print(f"Observations: {len(test_obs)}  |  Hidden: {args.hidden or 'auto'}")
    print(f"{'='*60}")

    results = []
    for i, obs in enumerate(test_obs):
        idx_label = f"[{i}]" if args.index is None else f"[{args.index}]"
        print(f"\n{idx_label} {obs.name}")
        r = discover([obs], hidden=args.hidden, max_expansions=args.max_expansions)

        if "error" in r:
            print(f"  ERROR: {r['error']}")
            continue

        s = "✅" if r["discovered"] else "❌"
        print(f"  {s} hidden={r['hidden_stripped']}  baseline={r['baseline_score']:.4f}  "
              f"best={r['score']:.4f}  expr={r['expression'][:50]}")
        print(f"  found: {r['hidden_found']}/{r['transform']}  "
              f"shape={r['error_shape']}({r['error_conf']:.2f})  "
              f"known={r['known_invariant']}  {r['time']:.1f}s")
        results.append(r)

    discoveries = [r for r in results if r.get("discovered")]
    print(f"\n{'='*60}")
    print(f"Summary: {len(discoveries)}/{len(results)} discovered")
    print(f"{'='*60}")
    for r in results:
        s = "✅" if r.get("discovered") else "❌"
        print(f"  {s} {r.get('name','?')[:45]:<45s} {r.get('score',0):.4f}  {r.get('expression','?')[:40]}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(results, out.open("w"), indent=2)
        print(f"\n→ {out}")


if __name__ == "__main__":
    main()
