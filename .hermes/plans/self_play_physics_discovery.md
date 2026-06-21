# Self-Play Physics Discovery — Architecture Plan

> **Goal:** A system that explores mathematical structures, verifies them against
> known physical observations, and generates testable predictions — discovering
> physics from scratch without being told what physics IS.

---

## 1. The Core Loop — Detailed

```
INITIALIZATION:
  Load observation database O = {obs_1, obs_2, ..., obs_N}
     Each obs = (scenario_description, measurements, expected_invariant)
  Load primitives P = {quantities} ∪ {operations} ∪ {constants}
  Seed candidate pool with depth-1 expressions (all single quantities)

LOOP (until stopped or frontier reached):
  
  1. GENERATE:
     For each candidate structure S in pool:
       Expand S to depth+1 by applying one operation:
         - Binary:   S' = S op quantity      (e.g., m → m*g, m → m+v)
         - Unary:    S' = op(S)              (e.g., v → dv/dt)
         - Scalar:   S' = S * constant       (e.g., v → 2*v)
         - Power:    S' = S^k                (e.g., v → v²)
       Add all well-typed children to candidate pool
  
  2. VERIFY (dimensions):
     For each candidate S:
       Compute physical dimensions of S
       If dimensions are inconsistent (e.g., kg + m/s) → DISCARD
       Must match the target dimension of the invariant we seek
       (For energy: kg·m²/s²)
  
  3. VERIFY (observations):
     For each candidate S that passes dimension check:
       For each observation obs in O:
         Instantiate S with obs.scenario.quantities
         Compute S(obs.scenario) at each measurement timestep
         Measure: std_dev of S across all timesteps
       Aggregate: average std_dev, max deviation
  
  4. SCORE:
     For each candidate S:
       constancy_score = 1.0 / (1.0 + avg_std_dev)
         → 1.0 if S is perfectly constant across ALL observations
         → 0.0 if S varies randomly
       bonus_simplicity = 1.0 / (1.0 + depth(S))
         → shallower expressions preferred
       bonus_novelty = 1.0 if S passes dimension check in novel way
       score = constancy_score × (1.0 + 0.5 × bonus_simplicity) × (1.0 + 0.1 × bonus_novelty)
  
  5. SELECT:
     Sort candidates by score descending
     Keep top-K candidates (K = 50)
     If best_score > success_threshold (0.95):
       → DISCOVERY: system has found a conserved quantity
       → Record the expression, Lean-prove the conservation
       → Optionally continue to find MORE conserved quantities
  
  6. EXPLORE:
     For each top-K candidate:
       Generate children at depth+1 (return to step 1)
     Prune branches that can't beat the current best by depth+3

TERMINATION:
  When no new expressions improve score for N consecutive depths
  → The system has exhausted the search space at current complexity
  → Report all discovered conserved quantities
```

---

## 2. Expression Grammar — Complete Specification

### 2.1 Type System

```
Base types:
  Scalar    — dimensionless number (1, 2, π, ...)
  Mass      — kg
  Length    — m
  Time      — s
  Velocity  — m/s
  Accel     — m/s²
  Force     — kg·m/s² = N
  Energy    — kg·m²/s² = J

Type derivation rules:
  Scalar   op Scalar   → Scalar      (+, -, *, /, ^)
  Length   / Time      → Velocity
  Velocity / Time      → Accel
  Mass     * Accel     → Force
  Force    * Length    → Energy
  Mass     * Velocity² → Energy
  Energy   op Energy   → Energy      (+, -)
  Scalar   * Energy    → Energy
  Mixing incompatible dimensions → TYPE ERROR (discard)
```

### 2.2 Operation Table

```
BINARY (always produce same type as inputs, both inputs must match):
  + (add)       : T × T → T           requires: same type
  - (subtract)  : T × T → T           requires: same type
  * (multiply)  : T × U → T·U         always allowed, produces compound type
  / (divide)    : T × U → T/U         always allowed, produces compound type
  ^ (power)     : T × Scalar → T^k    exponent must be scalar

UNARY:
  d/dt          : T → T/s            time derivative
  d/dx          : T → T/m            spatial derivative
  ∇             : Scalar → Vector/m  gradient
  sin, cos      : Scalar → Scalar    dimensionless argument
  exp, log      : Scalar → Scalar
  sqrt          : Scalar → Scalar    (or T^k → T^(k/2) for powers)

CONSTANTS (Scalar unless annotated):
  0, 1, 2, 3, ...           integers
  ½, ⅓, ...                 rationals
  π                         dimensionless
  e                         dimensionless
  g (m/s²)                  gravitational acceleration (constant in scenario)
  c (m/s)                   speed of light
  ℏ (J·s)                   reduced Planck constant
  G (m³/(kg·s²))            gravitational constant
```

### 2.3 Grammar (EBNF)

```ebnf
expression  := term ("+" term | "-" term)*

term        := factor ("*" factor | "/" factor)*

factor      := atom ("^" scalar_constant)?

atom        := quantity
            |  constant
            |  unary_op "(" expression ")"
            |  "(" expression ")"

quantity    := "m" | "v" | "x" | "t" | "h" | "F" | "p" | "E" | "B" | ...
              (from scenario's available quantities)

constant    := integer | rational | "π" | "e" | "g" | "c" | "ℏ" | "G"

unary_op    := "d/dt" | "d/dx" | "sin" | "cos" | "exp" | "log" | "sqrt"

scalar_const := integer | rational | "π" | "e"
```

### 2.4 Example Generation Trace

```
DEPTH 1 (atoms):
  m, v, x, t, h, F, g, 0, 1, 2, ½, π
  → 12 candidates. All vary across observations → no discoveries.

DEPTH 2 (one operation):
  m+v  → TYPE ERROR (kg + m/s)
  m*v  → kg·m/s (momentum-type, not energy)
  m*g  → kg·m/s² = Force ✓
  v/t  → m/s² = Accel ✓
  v²   → m²/s² (kinematic)
  m*v² → kg·m²/s² = Energy ✓✓
  m*g*h → kg·m²/s² = Energy ✓✓
  → 3 candidates reach correct dimension, test against observations
  → None constant → score < 0.3

DEPTH 3 (two operations):
  ½*m*v²       → kg·m²/s² ✓✓
  m*g*(h-h₀)   → kg·m²/s² ✓✓
  → Both dimensionally correct
  → Test: each varies, but they vary in OPPOSITE directions
  → Neither constant → score < 0.5

DEPTH 4 (three operations):
  m*g*h + ½*m*v² → kg·m²/s² ✓✓
  → Test: CONSTANT across all 10 observations
  → std_dev < ε → score > 0.95
  → DISCOVERY
```

---

## 3. Observation Database Format

### 3.1 Observation Structure

```python
@dataclass
class PhysicalObservation:
    """A single physical scenario with measurements."""
    
    id: str                         # unique identifier
    name: str                       # human-readable description
    description: str                # "A ball dropped from height h₀ in gravity g"
    
    # Quantities involved and their types
    quantities: dict[str, str]      # {"m": "Mass", "g": "Accel", "h": "Length",
                                    #  "v": "Velocity", "t": "Time"}
    
    # Constant parameters for this scenario
    parameters: dict[str, float]    # {"m": 1.0, "g": 9.8, "h0": 10.0}
    
    # Measurements at discrete timesteps
    timesteps: list[dict]           # [{"t": 0.0, "h": 10.0, "v": 0.0},
                                    #  {"t": 0.5, "h": 8.775, "v": 4.9},
                                    #  ...]
    
    # What should be conserved (for verification, not training)
    known_invariant: str | None     # "m*g*h + 0.5*m*v^2" or None
    
    # Lean theorem proving the conservation
    lean_theorem: str                   # Full Lean code
```

### 3.2 Example Observation (Falling Ball)

```json
{
  "id": "falling_ball_straight_drop",
  "name": "Ball dropped from rest",
  "description": "A 1kg ball dropped from 10m in Earth gravity. No air resistance.",
  "quantities": {
    "m": "Mass",
    "g": "Accel",
    "h": "Length",
    "v": "Velocity",
    "t": "Time"
  },
  "parameters": {"m": 1.0, "g": 9.8, "h0": 10.0},
  "timesteps": [
    {"t": 0.0, "h": 10.000, "v": 0.000},
    {"t": 0.5, "h": 8.775, "v": 4.900},
    {"t": 1.0, "h": 5.100, "v": 9.800},
    {"t": 1.5, "h": -1.025, "v": 14.700},
    {"t": 2.0, "h": -9.600, "v": 19.600}
  ],
  "known_invariant": "m*g*h + 0.5*m*v^2",
  "lean_theorem": "theorem energy_conservation_falling ..."
}
```

### 3.3 Observation Categories for Phase Progression

```
Phase 1 — Single conserved quantity (10 obs):
  falling_ball_straight_drop
  falling_ball_upward_throw
  falling_ball_varying_mass
  pendulum_small_angle
  pendulum_large_angle
  spring_undamped
  spring_damped_light
  projectile_45deg
  projectile_90deg
  sliding_block_incline

Phase 2 — Multiple conserved quantities (50 obs):
  Add momentum, angular momentum, charge conservation scenarios
  Elastic and inelastic collisions
  Orbital mechanics (circular, elliptical)
  Coupled oscillators

Phase 3 — Varying conditions (100+ obs):
  Vary g (Moon, Mars scenarios)
  Add friction, drag, external forcing
  Add thermal measurements (temperature, heat)

Phase 4 — Electromagnetic (200+ obs):
  Static charges, current-carrying wires
  Induction, radiation
  Field measurements

Phase 5 — Frontier (no known invariant):
  Quantum measurement outcomes
  High-energy scattering data
  Cosmological observations
  Anything where we HAVE data but NO complete theory
```

---

## 4. Scoring Function — Detailed

### 4.1 Constancy Score

```
For expression S and observation obs with timesteps T₁...Tₙ:

values = [eval(S, Tᵢ) for i in 1..n]
mean_val = mean(values)
std_val = std(values)

If any value is undefined (division by zero, etc.): score = 0.0
Else:
  constancy = 1.0 / (1.0 + std_val / |mean_val|)
  → Perfectly constant: std=0, constancy=1.0
  → Random variation: std ≈ |mean|, constancy ≈ 0.5
  → Anti-correlated: still low constancy (std is large)

Aggregate across observations:
  avg_constancy = mean(constancy over all observations)
  min_constancy = min(constancy over all observations)
  → Both must be high for discovery
```

### 4.2 Simplicity Bonus

```
bonus = 1.0 / (1.0 + depth(S)/depth_max)
  depth=1: bonus = 0.91
  depth=2: bonus = 0.83
  depth=4: bonus = 0.71
  depth=8: bonus = 0.56
```

### 4.3 Frontier Scoring (Phase 5+)

```
score = (N_obs_explained / N_total_obs)          # comprehensiveness
      × (N_predictions_generated > 0 ? 2.0 : 0.5) # predictive power
      × (1.0 / (1.0 + N_free_parameters))          # parsimony
      × (has_limit_reduction ? 1.2 : 0.8)          # consistency with known limits
      × symmetry_bonus                              # 1.0-2.0 based on Lie group symmetries

A structure that explains 100/100 observations with 3 free parameters,
generates 1 testable prediction, and has U(1) × SU(2) symmetry:
  score = 1.0 × 2.0 × 0.25 × 1.2 × 1.5 = 0.90

A structure that explains 50/100 observations with 20 parameters:
  score = 0.5 × 2.0 × 0.047 × 0.8 × 1.0 = 0.038
```

---

## 5. Lean Verification

### 5.1 What Lean Verifies

For each discovered candidate expression S that passes the numerical constancy check:

```lean
-- The system generates this theorem automatically
theorem discovered_invariant_conservation {m g h v : ℝ} (hpos : m > 0) : 
  m * g * h + (1/2) * m * v^2 = m * g * h₀ :=
by
  -- The system attempts to prove this
  -- Using the kinematic equations: v = g*t, h = h₀ - ½*g*t²
  -- Proof: algebraic substitution + ring normalization
  -- If provable: the invariant is mathematically guaranteed
  -- If not provable: numerical constancy was coincidental
```

### 5.2 Verification Flow

```
1. Numerical check:  S is constant across all observations within ε
2. Symbolic check:   Can Lean prove S is constant given the kinematic equations?
3. Generalize:       Does the proof hold for ALL initial conditions (h₀, v₀)?
4. Record:           Save the Lean theorem as a verified conservation law
5. Feed back:        The theorem becomes a NEW axiom for deeper exploration
```

---

## 6. Exploration Strategies

### 6.1 Breadth-First (Phase 1)

```
DEPTH 1: Try all quantities
DEPTH 2: Try all pairwise combos
DEPTH 3: Try all 3-element combos
...

Prune: any branch that fails dimension check
Bound:  max_depth = 10 (½mv² + mgh needs depth 4)
```

### 6.2 Best-First (Phases 2-3)

```
Maintain priority queue sorted by score.
At each step, expand the best candidate.

Priority = score × depth_discount
  → shallow high-scorers expanded first
  → deep candidates only explored when shallow space exhausted
```

### 6.3 Evolutionary (Phase 4+)

```
Initialize population: 50 random depth-3 expressions that pass dimensions.
For N generations:
  1. Evaluate all against observation database
  2. Select top-10 by score
  3. Mutate:
     - Add/subtract a term
     - Multiply by a scalar
     - Replace a quantity with its derivative/integral
     - Apply a gauge transformation
  4. Crossover:
     - Take two high-scoring expressions
     - Swap sub-trees (e.g., kinetic term from A + potential term from B)
  5. Add mutated children to population
  6. Keep top-50 overall
```

### 6.4 Frontier Hopping (Phase 5+)

```
When the observation database runs out:
  1. Identify all structures that explain 100% of known observations
  2. For each, derive the simplest prediction NOT in the database
  3. Score predictions by testability:
     - Requires existing equipment (LHC, JWST, LIGO)? → high testability
     - Requires new technology? → medium
     - Requires fundamentally impossible measurement? → low
  4. Output top-K testable predictions for human experimentalists
  5. When new data arrives → feed into observation database → loop continues
```

---

## 7. Implementation Phases

### Phase A: Expression Infrastructure (1-2 days)

**Files to create:**
- `src/physics/__init__.py`
- `src/physics/grammar.py` — Expression class, type system, dimension tracking
- `src/physics/generator.py` — Breadth-first combinatorial expression builder
- `src/physics/dimensions.py` — Dimension arithmetic (kg·m²/s² × s = kg·m²/s)
- `tests/physics/test_grammar.py` — Unit tests for expression building
- `tests/physics/test_dimensions.py` — Unit tests for dimension checking

**Smoke test:** Generate all valid depth-3 expressions from {m, v, g, h, t} and verify
`m*g*h` and `½*m*v²` are in the output.

### Phase B: First Observation Set (1 day)

**Files to create:**
- `data/observations/phase1_falling.json` — 10 falling/collision scenarios
- `src/physics/observations.py` — Load, query, evaluate against observations
- `src/physics/evaluator.py` — Score single expression against all observations
- `tests/physics/test_evaluator.py` — Test that known invariant scores ~1.0

**Smoke test:** Evaluate `m*g*h + ½*m*v²` against 10 falling ball observations.
Verify constancy score > 0.95.

### Phase C: Self-Play Loop v1 (2-3 days)

**Files to create:**
- `src/core/self_play_loop.py` — The main loop orchestrator
- `src/physics/search.py` — Best-first search over expression space
- `src/physics/theory.py` — Theory class (expression + axioms + predictions)

**Modify:**
- Wire existing `BestFirstSearch` and `dense_rewards` into the new loop

**Smoke test:** Run self-play for 1000 iterations on falling ball observations.
System should discover mgh + ½mv² within 500 iterations.

### Phase D: Energy Discovery (1 day)

**Goal:** System autonomously discovers conservation of mechanical energy.

**Acceptance criteria:**
- Expression `m*g*h + ½*m*v²` found within search budget
- Lean proof generated and verified
- System reports "discovered invariant: energy"
- Score > 0.95 across all Phase 1 observations

### Phase E: Generalize (1 week)

**Add:** 100 mechanical scenarios (varying mass, gravity, initial conditions)
**Expect:** System discovers the WORK-ENERGY THEOREM: ΔE = W
**New capability:** The system learns that energy is conserved WHEN no external forces act

### Phase F: Scale to Known Physics (unknown duration)

**Add:** Electromagnetic, quantum, relativistic observation databases
**Expect:** System discovers Lagrangian mechanics as the unifying framework
**New capability:** The system generates Maxwell's equations from observed field behavior

### Phase G: Frontier Predictions (unknown duration)

**Add:** Observations from LHC, cosmology, quantum optics where theories disagree
**Expect:** System generates testable predictions that distinguish between competing theories
**New capability:** The system produces novel physics predictions for human experimentalists

---

## 8. What Exists vs What Needs Building

### Already Built (theta-core v1.0)

| Component | File | Reusable? |
|-----------|------|-----------|
| Best-first search | `src/explorer/best_first_search.py` | ✅ Core search engine |
| Dense reward system | `src/reward/dense_rewards.py` | ✅ Scoring adapter |
| Lean proof interface | `src/proof_checker/batch_checker.py` | ✅ Verification gate |
| Proof state tracking | `src/explorer/proof_state.py` | ✅ State representation |
| GNN encoder | `src/explorer/gnn_encoder.py` | ❌ Not needed (no graph retrieval) |
| Gate framework | `scripts/gates/*.py` | ✅ Can adapt for physics gates |
| Kanban pipeline | `.hermes/kanban/` | ✅ Task management |

### New to Build

Everything under `src/physics/` is new. The self-play orchestrator is new.
The observation database format and loader are new. The total new code estimate
is ~2000 lines, roughly the size of `src/explorer/`.

---

## 9. Honesty Contract

1. **No physics injected.** The system knows quantities and operations, never interpretations.
2. **Only verification feedback.** Each observation is binary — pass or fail.
3. **Era-safe by construction.** Training observations are pre-1905. Benchmark observations are post-1905.
4. **No theory labels.** The system generates math, not "Newtonian mechanics."
5. **Discovery IS prediction.** A structure succeeds when it implies unmeasured outcomes.
6. **Human verification loop.** Frontier predictions require experimental confirmation.
7. **All discoveries are Lean-proven.** No numerical coincidence passes as discovery.
