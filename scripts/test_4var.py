"""Test: can the system discover 4-variable invariants?

Tests forms that simple search cannot find (because they exceed 3-var templates).
Beam search should handle these via bottom-up composition.
"""
import sys, random
sys.path.insert(0, '.')

from src.physics.search import auto_discover
from src.physics.dimensions import Dimension
from src.physics.observations import Observation

random.seed(42)

def make_data(invariant_fn, quantities, n_obs=8, n_ts=6):
    """Generate synthetic data where invariant_fn(qs) is constant."""
    obs_list = []
    base = random.uniform(1, 100)
    for i in range(n_obs):
        rng = random.Random(i * 137 + 42)
        ts = []
        for j in range(n_ts):
            vals = {q: 1.0 + rng.uniform(0, 3) for q in quantities}
            # Compute the invariant, then jitter one variable to preserve it
            target = invariant_fn(vals)
            if target == 0:
                target = 1.0
            # Adjust a random variable to hit the target
            adj_q = random.choice(list(quantities.keys()))
            vals[adj_q] *= base / target
            # Add tiny noise
            for q in quantities:
                vals[q] += rng.gauss(0, 0.001 * vals[q])
            ts.append(vals)
        obs_list.append(Observation(
            id=f'obs_{i}', name=f'obs_{i}', description='',
            quantities={q: 'Scalar' for q in quantities},
            parameters={}, timesteps=ts,
            known_invariant=None, lean_theorem=''
        ))
    return obs_list


# Test 1: a*b + c*d (sum of products)
print("=" * 60)
print("TEST 1: a*b + c*d (sum of products)")
print("=" * 60)
qs1 = {
    'a': Dimension.named('Scalar'), 'b': Dimension.named('Scalar'),
    'c': Dimension.named('Scalar'), 'd': Dimension.named('Scalar'),
}
def inv1(vals):
    return vals['a'] * vals['b'] + vals['c'] * vals['d']
obs1 = make_data(inv1, qs1)
r1 = auto_discover(qs1, obs1, discovery_threshold=0.90, _enable_beam_search=True)
print(f"  Result: {r1.expression!r}  score={r1.score:.4f}")
print(f"  PASS: {r1.is_discovery}")
print()

# Test 2: (a*b)/(c*d) (ratio of products)
print("=" * 60)
print("TEST 2: (a*b)/(c*d) (ratio of products)")
print("=" * 60)
qs2 = dict(qs1)
def inv2(vals):
    return (vals['a'] * vals['b']) / (vals['c'] * vals['d'])
obs2 = make_data(inv2, qs2)
r2 = auto_discover(qs2, obs2, discovery_threshold=0.90, _enable_beam_search=True)
print(f"  Result: {r2.expression!r}  score={r2.score:.4f}")
print(f"  PASS: {r2.is_discovery}")
print()

# Test 3: a*b*c*d (product of all four)
print("=" * 60)
print("TEST 3: a*b*c*d (product of all four)")
print("=" * 60)
qs3 = dict(qs1)
def inv3(vals):
    return vals['a'] * vals['b'] * vals['c'] * vals['d']
obs3 = make_data(inv3, qs3)
r3 = auto_discover(qs3, obs3, discovery_threshold=0.90, _enable_beam_search=True)
print(f"  Result: {r3.expression!r}  score={r3.score:.4f}")
print(f"  PASS: {r3.is_discovery}")
print()

# Test 4: (a+b)/(c+d) (ratio of sums)
print("=" * 60)
print("TEST 4: (a+b)/(c+d) (ratio of sums)")
print("=" * 60)
qs4 = dict(qs1)
def inv4(vals):
    return (vals['a'] + vals['b']) / (vals['c'] + vals['d'])
obs4 = make_data(inv4, qs4)
r4 = auto_discover(qs4, obs4, discovery_threshold=0.90, _enable_beam_search=True)
print(f"  Result: {r4.expression!r}  score={r4.score:.4f}")
print(f"  PASS: {r4.is_discovery}")

print()
print("=" * 60)
passed = sum([r1.is_discovery, r2.is_discovery, r3.is_discovery, r4.is_discovery])
print(f"SCORE: {passed}/4")
