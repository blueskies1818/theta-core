# How to Create Test Scenarios for theta-core

A step-by-step guide to creating new physics test scenarios that the system
can evaluate — whether to test generalization, benchmark discovery capability,
or probe the frontier of unknown physics.

## Quick Reference

```
Scenario = one Observation with:
  - id, name, description (metadata)
  - quantities dict (name → dimension type)
  - parameters dict (constants that don't change)
  - timesteps list (measurements across time/conditions)
  - known_invariant (expression that should be constant — your answer key)
```

Minimal valid scenario:
```python
from src.physics.observations import Observation

Observation(
    id="my_test",
    name="My Test Scenario",
    description="What this tests and what the invariant should be.",
    quantities={"v": "Velocity", "d": "Length"},
    parameters={},
    timesteps=[
        {"t": 0.0, "v": 10.0, "d": 100.0},
        {"t": 1.0, "v": 20.0, "d": 200.0},
        {"t": 2.0, "v": 30.0, "d": 300.0},
    ],
    known_invariant="v/d",
    lean_theorem="",
)
```

---

## 1. Available Dimension Types

The system uses physical dimension checking to avoid nonsensical expressions
(e.g., can't add Mass + Velocity). You must assign a dimension to every
quantity in your scenario.

### Built-in named dimensions

| Name       | Base units | What it's used for                    |
|------------|-----------|---------------------------------------|
| `Scalar`   | (none)    | Dimensionless numbers: n, γ, z, ratio |
| `Mass`     | kg        | m, M, m1, m2                          |
| `Length`   | m         | h, x, r, λ, d, a                     |
| `Time`     | s         | t, T, τ, period                       |
| `Velocity` | m/s       | v, c, σ, u                            |
| `Accel`    | m/s²      | g, a, acceleration                    |
| `Force`    | kg·m/s²   | F, tension, weight                    |
| `Energy`   | kg·m²/s²  | E, K, U, W, work                      |

### Compound dimensions

You can use compound dimension names directly:

| Quantity example | Dimension string          |
|------------------|---------------------------|
| momentum p       | `"Mass*Velocity"`         |
| frequency        | `"1/Time"` (not built-in — use Scalar) |
| spring constant  | `"Force/Length"`          |
| angular momentum | `"Mass*Length*Velocity"`  |
| pressure         | `"Force/Length^2"`        |
| power            | `"Energy/Time"`           |

**Important**: Compound dimensions must be parseable by the system.
Test them first:
```python
from src.physics.dimensions import Dimension
d = Dimension.named("Force/Length")  # works
d = Dimension.named("Momentum")       # FAILS — not a named dimension
```

> **Pitfall**: "Momentum" is NOT a built-in dimension. Use `"Mass*Velocity"` instead.
> This has caused test failures in the past (see era gate results for
> `relativistic_momentum` and `mass_energy`).

---

## 2. Structuring Timesteps

Timesteps are the core of your scenario. Each timestep is one "measurement"
in your experiment — a set of values for all quantities at one moment or
under one condition.

### Rules

1. **Every timestep must have `"t"`** — even if time isn't meaningful for
   your scenario. Use a dummy progression like `0.0, 1.0, 2.0...` or an
   index-based counter.

2. **Every quantity in `quantities` must appear in every timestep** — the
   evaluator looks up values by name per timestep.

3. **At least 3 timesteps** — fewer than 3 makes constancy scoring unreliable
   (standard deviation needs at least 2-3 points). 5-20 is a good range.

4. **Values must vary across timesteps** — a scenario where all values are
   constant across timesteps is trivially discoverable (everything is
   "constant").

### Good timestep structure

```python
# Varying one parameter across timesteps — each row is a measurement
timesteps = [
    {"t": 0.0, "v": 100.0, "d": 1000.0},
    {"t": 1.0, "v": 200.0, "d": 2000.0},
    {"t": 2.0, "v": 300.0, "d": 3000.0},
    {"t": 3.0, "v": 400.0, "d": 4000.0},
    {"t": 4.0, "v": 500.0, "d": 5000.0},
]
# Here v/d = 0.1 = constant across all timesteps
```

### Adding repeated measurements (noise simulation)

For more realistic data, repeat each measurement 2-3 times with slight
variation. This gives the beam search more data points and makes the
constancy score more robust:

```python
for d in [10, 50, 100, 200, 400]:
    v = H0 * d
    for _ in range(3):  # 3 repeated measurements
        timesteps.append({"t": float(len(timesteps)), "v": v, "d": d})
```

### When "time" isn't the varying parameter

For scenarios where the varying parameter is something other than time
(e.g., n in quantum numbers, redshift z, temperature T), use `t` as an
index and vary the other quantity:

```python
for n in range(0, 10):
    E = (n + 0.5) * hbar_omega
    for _ in range(3):
        timesteps.append({"t": float(n), "E": E, "n": float(n)})
```

---

## 3. Choosing an Invariant

The invariant is the expression that should evaluate to a constant across
all timesteps. This is your "answer key" — what you expect the system to
discover.

### What makes a good invariant

- **Dimensional consistency**: The expression must have valid dimensions.
  `m*g*h + 0.5*m*v^2` works because both terms are Energy.
  `v + d` fails because Velocity + Length is dimensionally invalid.

- **Physically meaningful**: Pick an invariant that represents a real
  conservation law or constant relationship.

- **Not trivially constant**: `t/t`, `0*t + 1`, `x^0` are trivially
  constant but physically meaningless. The system filters these out.

- **Not purely a parameter**: If your invariant is just `c` (speed of light)
  and `c` is a parameter with the same value in every timestep, the system
  will find `c` trivially. Make the invariant a COMBINATION of quantities.

### Good examples

| Scenario | Invariant | Why it works |
|----------|-----------|-------------|
| Free fall | `m*g*h + 0.5*m*v^2` | Sum of two Energy terms |
| Hubble law | `v/d` | Two quantities co-vary |
| Kepler | `T^2/a^3` | Ratio of powers — dimensional match |
| de Broglie | `lambda*v` | Product of length × velocity |
| Harmonic osc. | `E/(n+0.5)` | Energy divided by scalar offset |

---

## 4. Naming Convention

Use consistent prefixes to organize scenarios by domain:

| Prefix  | Domain                          |
|---------|---------------------------------|
| `qm_`   | Quantum mechanics               |
| `gr_`   | General relativity / gravity    |
| `cosmo_`| Cosmology                       |
| `classical_` | Classical mechanics          |
| `thermal_`   | Thermodynamics / stat mech  |
| `em_`   | Electromagnetism                |
| `dm_`   | Dark matter / astrophysics      |
| `he_`   | High-energy / particle physics  |

---

## 5. Adding Scenarios to the Test Registry

### Option A: Add to frontier_test_scenarios.py

Edit `scripts/frontier_test_scenarios.py`:

1. Write a `make_<scenario_id>()` function that returns `list[Observation]`
2. Add to `FRONTIER_SCENARIO_REGISTRY`:
   ```python
   ("my_scenario_id", make_my_scenario, "Human Name", "description"),
   ```

Then run:
```bash
python scripts/frontier_test_scenarios.py --validate
python scripts/frontier_test_scenarios.py --output data/my_tests.json
```

### Option B: Write a standalone JSON file

Create a JSON file loadable by `ObservationDatabase`:

```json
[
  {
    "id": "my_test_id",
    "name": "My Test",
    "description": "What it tests.",
    "quantities": {"v": "Velocity", "d": "Length"},
    "parameters": {},
    "timesteps": [
      {"t": 0.0, "v": 10.0, "d": 100.0},
      {"t": 1.0, "v": 20.0, "d": 200.0}
    ],
    "known_invariant": "v/d",
    "lean_theorem": ""
  }
]
```

Load and test:
```python
from src.physics.observations import ObservationDatabase
from src.physics.search import ExpressionSearch
from src.physics.dimensions import Dimension

db = ObservationDatabase("data/my_tests.json")
obs = list(db)[0]

quantities = {name: Dimension.named(dim) for name, dim in obs.quantities.items()}
search = ExpressionSearch(
    quantities=quantities,
    train_observations=[obs],
    max_depth=8,
    max_expansions=5000,
    discovery_threshold=0.90,
)
result = search.run()
print(f"Best: {result.expression}  Score: {result.score:.4f}  Discovered: {result.is_discovery}")
```

---

## 6. Running Tests

### Quick constancy check (fast)

```bash
python -c "
from src.physics.evaluator import ExpressionEvaluator
from src.physics.observations import ObservationDatabase
db = ObservationDatabase('data/my_tests.json')
ev = ExpressionEvaluator()
for obs in db:
    s = ev.score(obs.known_invariant, obs)
    print(f'{obs.id}: constancy={s:.6f}')
"
```

### Beam search discovery test (slow, thorough)

```bash
python scripts/frontier_test_scenarios.py --validate
```

Or use the era gate harness (tests pre-1905 training → post-1905 discovery):
```bash
python scripts/spacetime_era_gate.py --era-cutoff 1905
```

### Full test suite

```bash
python -m pytest tests/physics/ tests/core/ -q
```

---

## 7. Common Pitfalls

### Dimension errors

```
Unknown dimension name 'Momentum'. Available: ['Scalar', 'Mass', 'Length', ...]
```
**Fix**: Use compound form `"Mass*Velocity"` instead of `"Momentum"`.

### Missing 't' in timesteps

```
[scenario_id] timestep 0 missing 't'
```
**Fix**: Every timestep dict must include `"t": <float>`.

### Expression evaluates to zero

The system filters expressions that evaluate to zero at all timesteps
(trivially "constant" but meaningless). Make sure your invariant has
non-zero values.

### Very large or very small numbers

Values like `T² = 10^15` and `a³ = 10^33` produce ratios with extreme
magnitudes that can cause numerical issues. **Normalize your quantities**
to reasonable ranges (0.1 to 1000). For Kepler's law, use `T` in years
and `a` in AU rather than seconds and meters.

### Constancy score = 1.000000 for `t/t`

The system rejects `x^0` and some trivial forms, but `t/t` and similar
self-canceling expressions can still score 1.0. The `_is_nontrivial`
check filters these, but the heuristic isn't perfect. Always provide a
meaningful `known_invariant`.

---

## 8. Reference: Existing Test Coverage

### Already tested (8/8 era gate)

| Domain | Law | Invariant |
|--------|-----|-----------|
| Quantum | Hydrogen spectrum | `E*n^2` |
| Quantum | Spin quantization | `E/n` |
| Quantum | Wien's displacement | `E/T` |
| Quantum | Photoelectric effect | `h*f - K_max` |
| Relativistic | Rest energy | `E/gamma` |
| Relativistic | Velocity addition | `(u+v)/(1+u*v/c^2)` |
| Relativistic | Energy-momentum | `E^2 - (p*c)^2` |
| Relativistic | Spacetime interval | `(c*t)^2 - x^2` |

### Already tested (era gate extended)

| Domain | Scenario | Invariant |
|--------|----------|-----------|
| QED | Fine structure | `e^2/hbar_c` |
| QED | Compton scattering | `dlambda/(1-cos(theta))` |
| QCD | Asymptotic freedom | `alpha_s * log(Q/Lambda)` |
| Electroweak | Mixing angle | `gp/g` |
| Higgs | Mechanism potential | `phi^2 - v^2/2` |
| Neutrino | Oscillation phase | `phase * E / L` |

### New in this suite (not yet tested)

| Domain | Scenario | Invariant |
|--------|----------|-----------|
| Classical | Kepler's 3rd law | `T^2/a^3` |
| Thermal | Stefan-Boltzmann | `(P/A)/T^4` |
| Quantum | Harmonic oscillator | `E/(n+0.5)` |
| Quantum | de Broglie wavelength | `lambda*v` |
| GR | Gravitational redshift | `E*(1+g*h/c^2)` |
| Cosmology | Hubble expansion | `v/d` |
| Cosmology | CMB temperature | `T/(1+z)` |

---

## 9. Quick Start Template

Copy this to start a new scenario:

```python
from src.physics.observations import Observation

def make_my_new_scenario() -> list[Observation]:
    """One-line description of the physics."""
    timesteps = []

    # Generate data — vary the key quantity across timesteps
    for x in [values...]:
        # compute dependent quantities from physical law
        for _ in range(3):  # repeated measurements
            timesteps.append({"t": float(len(timesteps)), ...})

    return [Observation(
        id="prefix_my_scenario",
        name="Human-readable name",
        description="What physics this tests and what the invariant should be.",
        quantities={"q1": "DimensionName", "q2": "DimensionName", ...},
        parameters={"const1": value, ...},
        timesteps=timesteps,
        known_invariant="expression that should be constant",
        lean_theorem="",
    )]
```

For questions, see the existing scenarios in `scripts/spacetime_era_gate.py`
(starting at line 127) or `scripts/frontier_test_scenarios.py`.
