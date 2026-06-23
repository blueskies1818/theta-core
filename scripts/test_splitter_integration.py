#!/usr/bin/env python3
"""Minimal integration test of the splitter + closed-loop pipeline."""
import sys
sys.path.insert(0, '/home/blueman1818/Projects/theta-core')

from src.physics.observations import ObservationDatabase, Observation
from src.physics.dimensions import Dimension
from src.physics.search import ExpressionSearch
from src.physics.hidden_variables import (
    HiddenVariableProposer, HiddenVariableDiscovery, load_hidden_var_proposer,
)
from src.physics.evaluator import ExpressionEvaluator


def _dim_from_str(dim_str):
    try: return Dimension.named(str(dim_str))
    except: pass
    handlers = {
        "Action": lambda: Dimension.named("Energy") * Dimension.named("Time"),
        "Momentum": lambda: Dimension.named("Mass") * Dimension.named("Velocity"),
        "Frequency": lambda: Dimension.scalar() / Dimension.named("Time"),
        "InverseLength": lambda: Dimension.scalar() / Dimension.named("Length"),
        "1/Time": lambda: Dimension.scalar() / Dimension.named("Time"),
    }
    if str(dim_str) in handlers: return handlers[str(dim_str)]()
    try: return Dimension.named(str(dim_str))
    except: return Dimension.scalar()


def expand_indexed_params(obs: Observation, quantities: dict[str, Dimension]) -> list[Observation]:
    """If obs has indexed params (E1..E5), create config-spanning observation."""
    # Find unresolvable quantities
    available = set(obs.parameters.keys())
    for ts in obs.timesteps:
        available.update(ts.keys())
    unresolved = set(obs.quantities.keys()) - available

    indexed = {}
    for qname in unresolved:
        vals = {}
        for pkey, pval in obs.parameters.items():
            if pkey.startswith(qname) and pkey != qname:
                suffix = pkey[len(qname):]
                if suffix.isdigit():
                    vals[int(suffix)] = float(pval)
        if len(vals) >= 2:
            indexed[qname] = vals

    if not indexed:
        return [obs]

    first_q = next(iter(indexed.values()))
    indices = sorted(first_q.keys())

    new_ts = []
    for idx in indices:
        base = dict(obs.timesteps[0]) if obs.timesteps else {}
        for qname, idx_vals in indexed.items():
            if idx in idx_vals:
                base[qname] = idx_vals[idx]
        new_ts.append(base)

    new_params = dict(obs.parameters)
    first_idx = indices[0]
    for qname, idx_vals in indexed.items():
        if first_idx in idx_vals:
            new_params[qname] = idx_vals[first_idx]

    return [Observation(
        id=obs.id, name=obs.name, description=obs.description,
        quantities={k: str(v) for k, v in quantities.items()},
        parameters=new_params, timesteps=new_ts,
        known_invariant=obs.known_invariant, lean_theorem=obs.lean_theorem,
    )]


class BeamWrap:
    def __init__(self, search, result):
        self._search = search
        self._result = result
        self.best_expression = result.expression
        self.best_score = result.score
        self.discovered = result.score >= 0.90
        self._scored = search._scored


def make_beam_fn():
    def beam_fn(q, obs_list):
        s = ExpressionSearch(
            quantities=q, train_observations=obs_list,
            max_depth=8, max_expansions=5000,
            discovery_threshold=0.90, target_dim="Energy",
        )
        r = s.run()
        return BeamWrap(s, r)
    return beam_fn


def test_scenario(obs_index: int, name: str, domain: str):
    print(f"\n{'='*60}")
    print(f"Testing: {name} (index {obs_index}, domain={domain})")
    print(f"{'='*60}")

    db = ObservationDatabase('data/observations/quantum_synthetic.json')
    obs = list(db)[obs_index]

    # Build quantities without n
    quantities = {}
    for qname, dim_str in obs.quantities.items():
        if qname == "n":
            continue
        quantities[qname] = _dim_from_str(str(dim_str))

    # Strip n from quantities, apply splitter
    new_qty = {k: v for k, v in obs.quantities.items() if k != "n"}
    stripped = Observation(
        id=obs.id, name=obs.name, description=obs.description,
        quantities=new_qty, parameters=obs.parameters,
        timesteps=obs.timesteps,
        known_invariant=obs.known_invariant, lean_theorem=obs.lean_theorem,
    )
    observations = expand_indexed_params(stripped, quantities)

    # Inject n values (configuration indices) in timesteps
    for observation in observations:
        for i, ts in enumerate(observation.timesteps):
            ts["n"] = float(i + 1)

    # Normalize to eV scale if hbar is in SI
    first_obs = observations[0]
    if "hbar" in quantities and "hbar" in first_obs.parameters:
        hbar_val = first_obs.parameters["hbar"]
        if abs(hbar_val - 1e-34) < 1e-33:
            eV_to_J = 1.602176634e-19
            for observation in observations:
                if "hbar" in observation.parameters:
                    observation.parameters["hbar"] = float(observation.parameters["hbar"]) / eV_to_J
                for ts in observation.timesteps:
                    if "hbar" in ts:
                        ts["hbar"] = float(ts["hbar"]) / eV_to_J

    print(f"Observations after expansion: {len(observations)}")
    for o in observations:
        print(f"  {o.name}: {len(o.timesteps)} timesteps, E in ts: {'E' in o.timesteps[0] if o.timesteps else 'N/A'}")

    beam_fn = make_beam_fn()
    proposer = load_hidden_var_proposer('checkpoints/hidden_var_proposer.pt')

    discovery = HiddenVariableDiscovery(
        proposer=proposer, max_proposals=5, discovery_threshold=0.90,
    )

    result = discovery.discover(
        quantities=quantities, observations=observations,
        beam_search_fn=beam_fn, domain=domain,
        quantity_names=list(quantities.keys()),
    )

    print(f"\n--- Results ---")
    print(f"Discovered: {result.discovered}")
    print(f"Best expr: {result.best_expression}")
    print(f"Best score: {result.best_score:.4f}")
    print(f"Baseline score: {result.baseline_score:.4f}")
    print(f"Hidden var: {result.hidden_variable}")
    print(f"Transform: {result.transform}")

    if result.error_analysis:
        ea = result.error_analysis
        print(f"Error shape: {ea.shape} (conf={ea.shape_confidence:.4f}, cv={ea.mean_cv:.4f})")

    if result.proposals:
        print(f"Proposals ({len(result.proposals)}):")
        for i, p in enumerate(result.proposals[:5]):
            print(f"  {i+1}. {p.variable_type}/{p.variable_name} "
                  f"transform={p.transform} conf={p.confidence:.4f}")

    return result


if __name__ == "__main__":
    # Test particle_in_box (index 0)
    test_scenario(0, "particle_in_box", "quantum")
    # Test harmonic_oscillator (index 10)
    test_scenario(10, "harmonic_oscillator", "quantum")
