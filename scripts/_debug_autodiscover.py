#!/usr/bin/env python3
"""Debug auto_discover for failing scenarios."""
import sys, math
sys.path.insert(0, '.')

from src.physics.dimensions import Dimension
from src.physics.observations import Observation
from src.physics.search import auto_discover, ExpressionSearch, simple_invariant_search

# Test 1: relativistic_momentum
c = 3e8; m = 1.0
timesteps = []
for v in [0, 0.3e8, 0.6e8, 0.9e8, 0.99e8]:
    gamma = 1.0 / math.sqrt(1.0 - v**2 / c**2) if v < c else 10.0
    p = gamma * m * v
    E = gamma * m * c**2
    t_equiv = E / (m * c**2)
    x_equiv = p * t_equiv / E if E > 0 else 0
    for _ in range(3):
        timesteps.append({'t': t_equiv, 'x': x_equiv, 'v': v, 'c': c, 'p': p, 'E': E, 'm': m, 'gamma': gamma})

obs = Observation(
    id='relativistic_momentum', name='Relativistic momentum', description='...',
    quantities={'t': 'Time', 'x': 'Length', 'c': 'Velocity', 'v': 'Velocity', 'm': 'Mass', 'E': 'Energy', 'p': 'Momentum', 'gamma': 'Scalar'},
    parameters={'c': c, 'm': m}, timesteps=timesteps,
    known_invariant='E^2 - (p*c)^2', lean_theorem='',
)

quantities = {qname: Dimension.named(qdim) for qname, qdim in obs.quantities.items()}

print("=== relativistic_momentum ===")
print("quantities:", list(quantities.keys()))

# Step 1: What does auto_discover do?
result = auto_discover(quantities, [obs], known_invariant='E^2 - (p*c)^2', discovery_threshold=0.90, beam_expansions=2000)
print(f"auto_discover: expr={result.expression!r} score={result.score:.4f} depth={result.depth} is_discovery={result.is_discovery}")

# Step 2: Try simple_invariant_search directly
result_simple = simple_invariant_search(quantities, [obs], discovery_threshold=0.90)
print(f"simple_search: expr={result_simple.expression!r} score={result_simple.score:.4f} depth={result_simple.depth}")

# Step 3: Try ExpressionSearch with target_dim=None
search = ExpressionSearch(
    quantities=quantities, train_observations=[obs],
    max_depth=8, max_expansions=20000, discovery_threshold=0.90,
    top_k=20, target_dim=None,
)
result_es = search.run()
print(f"ExpressionSearch(None): expr={result_es.expression!r} score={result_es.score:.4f} depth={result_es.depth} expansions={result_es.expansions}")

# Step 4: Try ExpressionSearch with target_dim="Energy"
search2 = ExpressionSearch(
    quantities=quantities, train_observations=[obs],
    max_depth=8, max_expansions=20000, discovery_threshold=0.90,
    top_k=20, target_dim="Energy",
)
result_es2 = search2.run()
print(f"ExpressionSearch(Energy): expr={result_es2.expression!r} score={result_es2.score:.4f} depth={result_es2.depth}")

print()

# Test 2: mass_energy
print("=== mass_energy_equivalence ===")
timesteps2 = []
for m0 in [1.0, 2.0, 5.0, 10.0]:
    E0 = m0 * c**2
    for v in [0, 0.3e8, 0.6e8]:
        gamma = 1.0 / math.sqrt(1.0 - v**2 / c**2) if v < c else 10.0
        E = gamma * E0
        p = gamma * m0 * v
        t_equiv = E / E0
        x_equiv = p * t_equiv / E if E > 0 else 0
        for _ in range(2):
            timesteps2.append({'t': t_equiv, 'x': x_equiv, 'v': v, 'c': c, 'gamma': gamma, 'm': m0, 'E': E, 'p': p})

obs2 = Observation(
    id='mass_energy', name='E=mc^2', description='...',
    quantities={'t': 'Time', 'x': 'Length', 'c': 'Velocity', 'v': 'Velocity', 'm': 'Mass', 'E': 'Energy', 'p': 'Momentum', 'gamma': 'Scalar'},
    parameters={'c': c}, timesteps=timesteps2,
    known_invariant='E^2 - (p*c)^2', lean_theorem='',
)
quantities2 = {qname: Dimension.named(qdim) for qname, qdim in obs2.quantities.items()}

result2 = auto_discover(quantities2, [obs2], known_invariant='E^2 - (p*c)^2', discovery_threshold=0.90, beam_expansions=2000)
print(f"auto_discover: expr={result2.expression!r} score={result2.score:.4f} is_discovery={result2.is_discovery}")

# Try ExpressionSearch with None
search3 = ExpressionSearch(
    quantities=quantities2, train_observations=[obs2],
    max_depth=8, max_expansions=20000, discovery_threshold=0.90,
    top_k=20, target_dim=None,
)
result_es3 = search3.run()
print(f"ExpressionSearch(None): expr={result_es3.expression!r} score={result_es3.score:.4f} expansions={result_es3.expansions}")

print()

# Test 3: higgs_mechanism
print("=== higgs_mechanism ===")
v_vev = 246.0; lam = 0.13
timesteps3 = []
for phi in [0, 50, 100, 150, 200, 246, 300, 400]:
    V = lam * (phi**2 - v_vev**2 / 2)**2
    for _ in range(2):
        timesteps3.append({'t': float(len(timesteps3)*0.01), 'phi': phi, 'V': V, 'phi_sq': phi**2, 'v_sq_half': v_vev**2/2})

obs3 = Observation(
    id='higgs_mechanism', name='Higgs', description='...',
    quantities={'phi': 'Scalar', 'V': 'Scalar', 'phi_sq': 'Scalar', 'v_sq_half': 'Scalar'},
    parameters={'v': v_vev, 'lam': lam}, timesteps=timesteps3,
    known_invariant='phi^2 - v^2/2', lean_theorem='',
)
quantities3 = {qname: Dimension.named(qdim) for qname, qdim in obs3.quantities.items()}

result3 = auto_discover(quantities3, [obs3], known_invariant='phi^2 - v^2/2', discovery_threshold=0.90, beam_expansions=2000)
print(f"auto_discover: expr={result3.expression!r} score={result3.score:.4f} is_discovery={result3.is_discovery}")

search4 = ExpressionSearch(
    quantities=quantities3, train_observations=[obs3],
    max_depth=8, max_expansions=20000, discovery_threshold=0.90,
    top_k=20, target_dim="Scalar",
)
result_es4 = search4.run()
print(f"ExpressionSearch(Scalar): expr={result_es4.expression!r} score={result_es4.score:.4f} expansions={result_es4.expansions}")

search5 = ExpressionSearch(
    quantities=quantities3, train_observations=[obs3],
    max_depth=8, max_expansions=20000, discovery_threshold=0.90,
    top_k=20, target_dim=None,
)
result_es5 = search5.run()
print(f"ExpressionSearch(None): expr={result_es5.expression!r} score={result_es5.score:.4f} expansions={result_es5.expansions}")
