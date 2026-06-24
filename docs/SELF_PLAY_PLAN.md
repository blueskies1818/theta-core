# Self-Play Physics Discovery — Architecture Plan

## Principle

Instead of testing against 8 known formulas (circular, limited, hand-written),
generate infinite synthetic mathematical expressions, compute noisy
observations from them, and let the system practice discovering invariants.

Ground truth is always known because we generated it. The system improves
through volume of practice — not through human-provided formulas.

This is AlphaZero for physics discovery. The game: given noisy measurements,
find the invariant expression. The opponent: a generator that keeps making
harder problems.

## Why This Solves Everything

### Critical Problem 1: Circular Test Data
**Current:** Generate data BY evaluating the formula being tested.
           E = h*c/lambda, put E and lambda in data, test whether
           system finds E*lambda = h*c. Circular.

**Self-play fix:** Generate formula FIRST. Then simulate independent
                   measurements with noise. E from calorimeter (noisy),
                   lambda from spectrometer (different noise). The formula
                   isn't in the data. The system must discover it.

### Critical Problem 2: Hand-Written Neural Templates
**Current:** `composer.py` has hardcoded `DOMAIN_TEMPLATES["relativistic"] =
           "E^2 - (p*c)^2"`. A human who knows relativity wrote it.

**Self-play fix:** Train template generators on millions of randomly-
                   generated expressions. The generator learns "when two
                   same-dimension quantities appear, try q1^2-q2^2"
                   because it's seen that pattern thousands of times in
                   the training data. It has never seen post-1905 symbols,
                   but the STRUCTURAL pattern transfers.

### Critical Problem 3: Hidden Variable Proposer Needs Training Data
**Current:** Trained on ~50 hand-picked examples (standing waves, Fourier
           harmonics). Someone who knows quantum mechanics chose them.

**Self-play fix:** Generate expressions containing hidden variables.
                   Remove those variables from observations. Ask proposer
                   to detect the missing variable from residual patterns.
                   Ground truth: we know exactly what we removed.
                   Training signal: perfect. Training data: infinite.

## Self-Play Loop

```
LOOP:
  1. GENERATE: Random valid expression E = f(a, b, c, ...)
     - Start simple (2-variable ratios), increase complexity over time
     - Dimension-checked: all terms must be physically valid
     - Record the expression as ground truth

  2. SIMULATE: Create observation data from the expression
     - Sample random input values for each variable
     - Compute E at each configuration
     - Add independent measurement noise to each quantity
     - Output: timesteps where inputs vary, E is constant

  3. CHALLENGE: Feed to discovery pipeline
     - Pipeline attempts to find invariant
     - No access to ground truth during search

  4. SCORE: Compare discovered expression to ground truth
     - Exact match: 1.0
     - Structurally equivalent (commuted, factored): 0.9
     - Found a different invariant that's also constant: 0.5
     - Failed to find anything: 0.0

  5. TRAIN: Use results to improve components
     - Template generator: (quantities, domain) → correct expression
     - Hidden variable proposer: (residuals, quantities) → missing variable
     - Search heuristics: which beam paths succeeded?

  6. CURRICULUM: Adapt difficulty based on success rate
     - >80% success → increase complexity
     - <30% success → decrease complexity
     - Track which expression types the system struggles with
```

## Expression Generator

### Phase 1: Simple Invariants (what the system already handles)
```
Ratios:     E/n, T^2/L, P*V, v^2*r, E*lambda
Products:   m*v*r, lambda*p, P*V
Powers:     T^2, v^2, a^3
```
Generate by: pick 2-3 quantities with compatible dimensions.
Choose operations (*, /, ^) that produce a valid dimension.

### Phase 2: Sum Invariants (classical mechanics)
```
Conservation: m*g*h + 0.5*m*v^2
              k*x^2 + 0.5*m*v^2
              P + 0.5*rho*v^2
```
Generate by: random terms of same dimension, combine with +.

### Phase 3: Nested Expressions (relativistic patterns)
```
Differences:  (c*t)^2 - x^2, E^2 - (p*c)^2
Fractions:    (u+v)/(1+u*v/c^2)
```
Generate by: structural templates.
Parenthesize: (A*B)^2 - C^2, A/(B+C*D/E)

### Phase 4: Hidden Variable Expressions (quantum patterns)
```
Quantized:    E_n = n^2 * pi^2 * hbar^2 / (2*m*L^2)
              E = (n+1/2) * hbar * omega
Ratios:       lambda = h/p (p varies, lambda varies, h constant)
```
Train hidden variable proposer by: generate expression WITH hidden
variable → remove variable from observations → proposer must detect
its absence from residual structure.

## Observation Simulator

For each generated expression E = f(v1, v2, ..., vn):

1. Sample N configurations: random input values from reasonable ranges
2. Compute E_true for each configuration
3. Add measurement noise:
   - Each quantity gets INDEPENDENT noise (different instrument)
   - Noise model: multiplicative Gaussian, sigma = 1-5%
   - v_measured = v_true * (1 + noise_v)
   - E_measured = E_true * (1 + noise_E)
4. Ensure at least one quantity genuinely varies across configurations
5. Output: Observation with timesteps containing noisy measurements

Key: The data never contains the formula. It contains noisy values that
the formula EXPLAINS.

## Training Pipeline

### Train Template Generator (composer.py)

Input: (quantity set, domain label)
Output: expression string

Training data: (generated_quantities, generated_expression) pairs
from millions of self-play generations.

Current model: Transformer decoder, ~50K params, 32-token output.
Does not need to scale. The output vocabulary is small (~50 tokens).
More data → better generalization within same capacity.

### Train Hidden Variable Proposer (hidden_variables.py)

For each generated expression containing a hidden variable:

1. Compute observations WITHOUT the hidden variable
2. Run auto_discover — it will fail (expression varies without hidden var)
3. Collect residuals: per-observation constancy scores of best candidate
4. Error shape: classify residual pattern (linear, quadratic, 1/n^2, etc.)
5. Training pair: (error_shape, domain, quantities) → (variable_type, transform)

Current model: MLP, ~3K params.
Does not need to scale. The classification space is tiny:
7 variable types × 5 error shapes = 35 decision boundaries.

### Train Search Heuristics (optional)

Track which beam search paths lead to discovery for which expression types.
Use to prioritize beam expansions for future similar expressions.
This is a lookup table + simple scoring, not a neural model.

## Curriculum

```
Level 1: 2-variable ratios, powers [-2, -1, 2]
         Target: 95% success → advance
         
Level 2: 3-variable products and ratios
         Target: 90% success → advance

Level 3: 2-term sums (energy conservation forms)
         Target: 80% success → advance

Level 4: Squared differences (relativistic forms)
         Target: 70% success → advance

Level 5: Nested fractions (velocity addition forms)
         Target: 60% success → advance

Level 6: Hidden variable expressions (quantum forms)
         Target: 50% success → advance
```

The system doesn't advance to frontier physics until it masters structural
patterns at each level. Success rate gates prevent wasted computation.

## Model Scaling Analysis

| Component | Current params | Needs scaling? | Why not |
|-----------|---------------|----------------|---------|
| Domain classifier | ~50K | No | 6 domains, ~30 quantities. A hash table would work. |
| Template generator | ~50K | No | Output vocab ~50 tokens. Learning ~20 structural patterns. Capacity is already overkill. |
| Hidden var proposer | ~3K | No | 7 types × 5 shapes = 35 classes. Linear classifier would work. |
| Beam search | 0 (deterministic) | No | Not ML. Search budget controls quality. |
| Expression evaluator | 0 (deterministic) | No | Not ML. Parsing and arithmetic. |

**When would scaling be needed?** Only if the expression complexity
exceeds current model capacity:
- Expressions longer than 32 tokens
- More than 5 independent terms in a sum
- Deeper nesting than 3 levels

These are CURRICULUM limits, not architecture limits. The system masters
what it can express, then we decide whether to expand capacity or accept
the ceiling.

## Implementation Phases

### Phase A: Expression Generator (1-2 days)
- Random expression generation with dimension validation
- 4 complexity levels (ratios, sums, squared-diffs, nested)
- Output: (expression_string, quantities_dict, domain_label)

### Phase B: Observation Simulator (1 day)
- Convert expression → noisy observations
- Independent noise per quantity
- Validation: verify expression IS constant on noise-free version

### Phase C: Self-Play Loop (1-2 days)
- Generate → simulate → discover → score → train loop
- Curriculum tracking
- Results logging: expression complexity vs discovery rate

### Phase D: Template Generator Training (1-2 days)
- Replace hardcoded DOMAIN_TEMPLATES with self-play-trained generator
- Train on pre-1905 structural patterns only
- Test generalization to post-1905 forms
- Goal: generator outputs correct form without ever seeing the formula

### Phase E: Hidden Variable Proposer Training (1-2 days)
- Train on self-play data with hidden variables removed
- Test on known post-1905 hidden variable scenarios
- Goal: proposer detects missing variables from residual patterns
        without ever seeing quantum or relativistic data

### Phase F: Full Integration (1 day)
- Wire self-play-trained components into discover pipeline
- Run era gate with self-play-trained models
- Compare to hand-written template baseline
- Goal: match or exceed current verification results
        with genuinely learned (not hand-written) components

## Success Criteria

1. Template generator, trained only on pre-1905 structural patterns,
   generates (c*t)^2-x^2 when given {c, t, x} — a form it has never
   seen applied to these specific quantities.

2. Hidden variable proposer, trained on self-play data with variables
   removed, correctly proposes "integer_n" when shown residuals from
   hydrogen spectrum data — without ever seeing quantum mechanics.

3. Era gate results with self-play-trained models match or exceed
   current results with hand-written templates.

4. The system continues to improve as self-play generates more
   training data — no saturation at current model capacity.

## Relationship to Frontier

The self-play system practices on KNOWN physics (generated expressions)
to learn HOW to discover invariants. It never practices on post-1905
physics. The structural discovery skills transfer.

At the frontier:
- Feed real experimental data (dark matter rotation curves, etc.)
- The hidden variable proposer has practiced on millions of cases
  where variables are missing — it knows the residual signatures
- The template generator has learned structural patterns (squared
  differences, nested fractions, conservation sums) from pre-1905
  practice — it applies them to frontier quantities
- The beam search has been tuned by millions of successful discoveries

The system hasn't seen the ANSWER. But it has PRACTICED THE PROCESS
millions of times. That's the difference between self-play and
memorization.
