"""FRONTIER TEST v2 — held-out claims WITH nuisance variables.

Each claim includes 2-3 nuisance variables measured alongside the
relevant quantities.  This prevents degenerate single-variable
expressions from scoring high via data correlation.

Run: python scripts/frontier_test.py
"""
import math
import random
import sys
sys.path.insert(0, '.')

from src.physics.search import auto_discover
from src.physics.dimensions import Dimension
from src.physics.observations import Observation

rng = random.Random(42)

NUISANCE_POOL = ['n1', 'n2', 'n3', 'n4', 'n5']


def make_nuisance_data(dims, dep_var, fn, n_obs=8, n_ts=6, n_nuisance=2,
                       noise=0.005):
    """Generate data with nuisance variables.

    Independent vars (including nuisance) are random.  The dependent
    variable is computed to satisfy the invariant.  Nuisance variables
    are unrelated to the invariant and prevent spurious correlations
    from making degenerate expressions score high.
    """
    key_vars = [q for q in dims if q != dep_var]
    nuisance_vars = [f'n{i+1}' for i in range(n_nuisance)]
    all_vars = key_vars + nuisance_vars + [dep_var]
    n_total = len(key_vars) + 1  # key vars + dependent

    str_dims = {}
    for q in key_vars + [dep_var]:
        str_dims[q] = str(dims[q])
    for n in nuisance_vars:
        str_dims[n] = 'Scalar'

    obs_list = []
    for i in range(n_obs):
        ts = []
        for j in range(n_ts):
            vals = {}
            # Independent key vars
            for q in key_vars:
                vals[q] = 1.0 + rng.uniform(0, 3)
            # Uncorrelated nuisance vars
            for n in nuisance_vars:
                vals[n] = rng.uniform(0, 10) + rng.gauss(0, 0.5)
            # Compute dependent variable
            vals[dep_var] = fn(vals)
            # Add noise
            for q in all_vars:
                if q in vals:
                    vals[q] += rng.gauss(0, noise * abs(vals[q]) + 0.001)
            ts.append(vals)
        obs_list.append(Observation(
            id=f'obs_{i}', name=f'obs_{i}', description='',
            quantities=str_dims,
            parameters={}, timesteps=ts,
            known_invariant=None, lean_theorem=''
        ))
    return dims, obs_list


def check(name, expected, result):
    """Score one claim."""
    status = "PASS" if result.is_discovery else "FAIL"
    # Flag suspicious forms (single-variable or constant-heavy)
    suspicious = False
    expr = result.expression
    import re
    funcs = {'sin', 'cos', 'sqrt', 'exp', 'log', 'abs', 'deriv'}
    tokens = re.findall(r'\b[a-zA-Z_]\w*\b', expr)
    vars_used = [t for t in tokens if t not in funcs
                 and not t.replace('.','').replace('-','').replace('e','').isdigit()]
    if len(set(vars_used)) <= 1 and result.is_discovery:
        suspicious = True
        status = "COINC"  # coincidence — degenerate form
    print(f"  {status:5s} {name:20s}  {expr!r:38s}  score={result.score:.4f}"
          + ("  ← suspicious" if suspicious else ""))
    return result.is_discovery and not suspicious


passed = 0


# ════════════════════════════════════════════════════════════
# 1. Hooke's Law: F/x = const (+ nuisance vars)
# ════════════════════════════════════════════════════════════
print("=" * 60)
print("1/8 Hooke: F/x = const")
print("=" * 60)
d1 = {'F': Dimension.named('Force'), 'x': Dimension.named('Length')}
k = rng.uniform(5, 20)
dim1, obs1 = make_nuisance_data(d1, 'F', lambda v: k * v['x'])
r1 = auto_discover(dim1, obs1, discovery_threshold=0.90)
if check("Hooke", "F/x", r1): passed += 1
print()


# ════════════════════════════════════════════════════════════
# 2. Torricelli: v^2/h = const
# ════════════════════════════════════════════════════════════
print("=" * 60)
print("2/8 Torricelli: v^2/h = const")
print("=" * 60)
d2 = {'v': Dimension.named('Velocity'), 'h': Dimension.named('Length')}
g = rng.uniform(5, 15)
dim2, obs2 = make_nuisance_data(d2, 'v',
    lambda v: math.sqrt(2 * g * v['h']))
r2 = auto_discover(dim2, obs2, discovery_threshold=0.90)
if check("Torricelli", "v^2/h", r2): passed += 1
print()


# ════════════════════════════════════════════════════════════
# 3. Capacitor: U/(C*V^2) = const
# ════════════════════════════════════════════════════════════
print("=" * 60)
print("3/8 Capacitor: U/(C*V^2) = const")
print("=" * 60)
d3 = {'U': Dimension.named('Energy'),
      'C': Dimension.named('Scalar'),
      'V': Dimension.named('Scalar')}
C0 = rng.uniform(1, 5)
dim3, obs3 = make_nuisance_data(d3, 'U',
    lambda v: 0.5 * C0 * v['C'] * v['V']**2)
r3 = auto_discover(dim3, obs3, discovery_threshold=0.90)
if check("Capacitor", "U/C/V^2", r3): passed += 1
print()


# ════════════════════════════════════════════════════════════
# 4. Stefan-Boltzmann: P/(A*T^4) = const
# ════════════════════════════════════════════════════════════
print("=" * 60)
print("4/8 Stefan-Boltzmann: P/(A*T^4) = const")
print("=" * 60)
d4 = {'P': Dimension.named('Energy'),
      'A': Dimension.named('Scalar'),
      'T': Dimension.named('Scalar')}
sigma = rng.uniform(1, 5)
dim4, obs4 = make_nuisance_data(d4, 'P',
    lambda v: sigma * v['A'] * v['T']**4)
r4 = auto_discover(dim4, obs4, discovery_threshold=0.90)
if check("Stefan-Boltzmann", "P/A/T^4", r4): passed += 1
print()


# ════════════════════════════════════════════════════════════
# 5. Spring period: T^2*k/m = const
# ════════════════════════════════════════════════════════════
print("=" * 60)
print("5/8 Spring: T^2*k/m = const")
print("=" * 60)
d5 = {'T': Dimension.named('Scalar'),
      'k': Dimension.named('Force'),
      'm': Dimension.named('Mass')}
k0 = rng.uniform(10, 50)
dim5, obs5 = make_nuisance_data(d5, 'T',
    lambda v: 2 * math.pi * math.sqrt(v['m'] / k0))
r5 = auto_discover(dim5, obs5, discovery_threshold=0.90)
if check("Spring", "T^2*k/m", r5): passed += 1
print()


# ════════════════════════════════════════════════════════════
# 6. Snell's Law: sin(t1)/sin(t2) = n2/n1 (+ nuisance)
# ════════════════════════════════════════════════════════════
print("=" * 60)
print("6/8 Snell: sin(t1)/sin(t2) = const")
print("=" * 60)
d6 = {'t1': Dimension.named('Scalar'),
      't2': Dimension.named('Scalar')}
n_ratio = rng.uniform(1.2, 2.0)
dim6, obs6 = make_nuisance_data(d6, 't2',
    lambda v: math.asin(min(1.0, max(-1.0,
        math.sin(v['t1']) / n_ratio))))
r6 = auto_discover(dim6, obs6, discovery_threshold=0.90)
if check("Snell", "sin(t1)/sin(t2)", r6): passed += 1
print()


# ════════════════════════════════════════════════════════════
# 7. Maxwell: deriv(P,T)-deriv(S,V) = 0 (+ nuisance)
# ════════════════════════════════════════════════════════════
print("=" * 60)
print("7/8 Maxwell: deriv(P,T) = deriv(S,V)")
print("=" * 60)
d7 = {'P': Dimension.named('Pressure'),
      'T': Dimension.named('Scalar'),
      'S': Dimension.named('Scalar'),
      'V': Dimension.named('Volume')}
base = rng.uniform(10, 50)
obs7_list = []
for i in range(8):
    ts = []
    for j in range(8):
        t_val = rng.uniform(1, 5)
        v_val = rng.uniform(1, 3)
        n1 = rng.uniform(0, 10)  # nuisance
        n2 = rng.uniform(0, 10)  # nuisance
        p = base * t_val / v_val + rng.gauss(0, 0.01)
        s = base * v_val / t_val + rng.gauss(0, 0.01)
        ts.append({'P': p, 'T': t_val, 'S': s, 'V': v_val,
                   'n1': n1, 'n2': n2})
    obs7_list.append(Observation(
        id=f'obs_{i}', name=f'obs_{i}', description='',
        quantities={'P': 'Pressure', 'T': 'Scalar', 'S': 'Scalar',
                    'V': 'Volume', 'n1': 'Scalar', 'n2': 'Scalar'},
        parameters={}, timesteps=ts,
        known_invariant=None, lean_theorem=''
    ))
r7 = auto_discover(d7, obs7_list, discovery_threshold=0.90)
if check("Maxwell", "deriv(P,T) = deriv(S,V)", r7): passed += 1
print()


# ════════════════════════════════════════════════════════════
# 8. Doppler: fo/fs = sqrt((1+b)/(1-b)) (+ nuisance)
# ════════════════════════════════════════════════════════════
print("=" * 60)
print("8/8 Doppler: fo/fs = sqrt((1+b)/(1-b))")
print("=" * 60)
d8 = {'fo': Dimension.named('Scalar'),
      'fs': Dimension.named('Scalar'),
      'b': Dimension.named('Scalar')}
obs8_list = []
for i in range(8):
    ts = []
    for j in range(8):
        b_val = rng.uniform(0.1, 0.9)
        fs_val = rng.uniform(1, 5)
        n1 = rng.uniform(0, 10)
        n2 = rng.uniform(0, 10)
        fo_val = fs_val * math.sqrt((1 + b_val) / (1 - b_val))
        fo_val += rng.gauss(0, 0.005)
        ts.append({'fo': fo_val, 'fs': fs_val, 'b': b_val,
                   'n1': n1, 'n2': n2})
    obs8_list.append(Observation(
        id=f'obs_{i}', name=f'obs_{i}', description='',
        quantities={'fo': 'Scalar', 'fs': 'Scalar', 'b': 'Scalar',
                    'n1': 'Scalar', 'n2': 'Scalar'},
        parameters={}, timesteps=ts,
        known_invariant=None, lean_theorem=''
    ))
r8 = auto_discover(d8, obs8_list, discovery_threshold=0.90)
if check("Doppler", "fo/(fs*sqrt(...))", r8): passed += 1

# ════════════════════════════════════════════════════════════
print()
print("=" * 60)
print(f"FRONTIER SCORECARD: {passed}/8 genuine")
print("=" * 60)
