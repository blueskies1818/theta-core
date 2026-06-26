# theta-core — Autonomous Mathematical Physics Discovery

An AI system that discovers physical invariants from measurement data.
Given only quantity names, their physical dimensions, and instrument
measurements, it finds conserved expressions — no physics textbook,
no equations injected.

## Claims Verified (V2.5)

28 claims across 7 physics domains, held out from development:

```
CLEAN BENCHMARK (8/8):
QUANTUM           E*lambda = h*c (photon energy-wavelength)
QUANTUM           E/nu = constant (energy quantization)
QUANTUM           E_peak/T = constant (Wien's displacement)
QUANTUM           h*nu - K_max = phi (photoelectric effect)
RELATIVISTIC      E/gamma = m*c^2 (energy-mass equivalence)
RELATIVISTIC      (u+v)/(1+u*v/c^2) (velocity addition)
RELATIVISTIC      E^2 - p^2 = (m*c^2)^2 (energy-momentum invariant)
RELATIVISTIC      (c*t)^2 - x^2 (spacetime interval)

NUISANCE BENCHMARK (8/8):
All 8 claims above with 2-3 nuisance variables each. 100% pass rate.

GENERALIZED BENCHMARK (12/12):
THERMO            P*V = constant (Boyle's law, isothermal)
THERMO            V/T = constant (Charles's law, isobaric)
THERMO            P/T = constant (Gay-Lussac's law, isochoric)
MECHANICS         T^2/a^3 = constant (Kepler's third law)
MECHANICS         T^2/L = constant (pendulum period)
MECHANICS         F/a = constant (Newton's second law)
MECHANICS         K/v^2 = constant (kinetic energy)
EM                F*r^2 = constant (Coulomb's law, fixed charges)
EM                w^2*L*C = constant (LC resonance)
GRAVITATION       v^2*r = constant (escape velocity)
WAVES             f*lambda = constant (wave equation)
CIRCUITS          V/I = constant (Ohm's law)
```

## Architecture (V2.5)

```
Instrument Data → Neural Template Generators
                      │
                      ▼
                 Tree Decoder (RPN-based AST, Phase C)
                 Deterministic Proposer (union with neural)
                      │
                      ▼
                 Seed Scorer (Phase A, 95.5% accuracy)
                 Beam Guider (Phase B, 94.2% accuracy)
                      │
                      ▼
                 Tree Beam Search (bottom-up composition)
                 Simple Invariant Search (47 templates + mutation)
                      │
                      ▼
                 Expression Evaluator (6 honesty gates)
                 All-Symbols Filter (min 1 var)
                      │
                      ▼
                 Plastic Update (structural Hebbian learning)
                 ←── feeds back into Seed Scorer ──┐
                      │                            │
                      ▼                            │
                 DISCOVERY REPORT                  │
                      │                            │
                      ▼                            │
                 Relationship Extraction ◄─────────┘
                 (read learned structural preferences)
```

### Neural Components (Phases A-E)

| Phase | Component | Status | Accuracy |
|-------|-----------|--------|----------|
| A | Seed Scorer | Trained | 95.5% val |
| B | Beam Guider | Trained | 94.2% val |
| C | Tree Decoder | Trained | 80.7% acc |
| E | Plastic Adaptation | Active | Structural, 14 patterns |

All three models trained on synthetic structural data (zero physics).
GPU: Intel Arc Pro B70, 31.9GB VRAM, PyTorch XPU.

### Evaluator Gates (6 computational sanity checks)

| Gate | What it catches |
|------|----------------|
| No-variables | Literal constants |
| Trivial-constancy | Expressions that never vary |
| Self-cancellation | x-x, x/x degenerate forms |
| Term-dominance | One term dominates (>10K×) |
| Near-identity-power | Base ≈ 1 (<1%) |
| Numerical-collapse | All values ≈ 0 |

All gates are mathematical/computational. Zero physics knowledge.

### Search Pipeline (`auto_discover`)

1. **Neural template search** — cross-symbol wrapper with tree decoder + beam search
2. **Simple invariant search** — 47 structural templates (polynomial, transcendental) + mutation-based exploration for novel forms
3. **Beam search** — bottom-up tree composition (regime sub-discovery only)
4. **Grouped-quantity discovery** — fallback for multi-observation patterns
5. **Regime discovery** — piecewise-constant phenomena
6. **Squared-difference fallback** — same-dimension pairs (post-processing)

## Honesty Contract

1. **No physics injected.** System knows quantity names and dimensions, never interpretations.
2. **No answer hints.** `known_invariant` parameter accepted but IGNORED — used only for backward compatibility.
3. **Era-safe training.** Pre-1905 classical physics only for training data.
4. **Neural models trained on structure, not physics.** Seed scorer, beam guider, and tree decoder learn expression grammar from synthetic patterns — zero physics content.
5. **Deterministic fallback.** If neural models unavailable, deterministic enumeration handles all claims.
6. **Discoveries verifiable.** Every claim has instrument-simulated data; constancy is measured, not assumed.

## Bias Audit

Documented biases and their resolution status (see `NEURAL_MODEL_PLAN.md`):

| # | Bias | Severity | Status |
|---|------|----------|--------|
| 1 | Structural form templates | HIGH | Mutation engine deployed; heuristic tuning remains |
| 2 | Dimension inference (human taxonomy) | MEDIUM | Removed — dimension-agnostic search |
| 3 | Operator set (functions) | MEDIUM | 25 transcendental templates added |
| 4 | All-symbols filter (min 2 vars) | LOW | Relaxed to min 1 var |
| 5 | Training data distribution | LOW | Deferred to Phase E (plasticity) |

## Verification

```bash
# Clean benchmark (8 claims, instrument-simulated data)
python scripts/verify_instruments.py

# Nuisance benchmark (8 claims, 2-3 nuisance variables each)
python scripts/verify_nuisance.py

# Generalized benchmark (12 claims, 7 domains, held-out)
python scripts/verify_generalized.py

# All tests
python -m pytest tests/ -q
```

## Files

```
src/
  physics/
    search.py           — auto_discover pipeline, simple_invariant_search
    evaluator.py        — ExpressionEvaluator with 6 honesty gates
    dimensions.py       — Dimension type system
    observations.py     — Observation data model
  math/
    tree_decoder.py     — RPN-based AST expression decoder (Phase C)
    seed_scorer.py      — Neural seed scorer (Phase A)
    beam_guider.py      — Neural beam search guider (Phase B)
    cross_symbol_wrapper.py — Neural template generator + deterministic proposer
    tree_beam_search.py — Bottom-up tree composition search
    mutate.py           — Expression mutation engine (bias #1)

scripts/
  training/
    train_seed_scorer.py
    train_beam_guider.py
    train_grammar_decoder.py  (flat decoder — deprecated)
    train_tree_decoder.py     (tree decoder — active)
  verify_instruments.py   — Clean benchmark (8 claims)
  verify_nuisance.py      — Nuisance benchmark (8 claims)
  verify_generalized.py   — Generalized benchmark (12 claims)
  instruments.py          — Instrument simulator base classes

checkpoints/math_self_play/
  seed_scorer.pt
  beam_guider.pt
  tree_decoder.pt
```

## Quick Start

```bash
# Run verification suite
python scripts/verify_instruments.py
python scripts/verify_nuisance.py
python scripts/verify_generalized.py

# Train tree decoder from scratch (GPU recommended)
python scripts/training/train_tree_decoder.py

# Run core tests
python -m pytest tests/physics/ tests/core/ -q
```

## Next: Relationship Extraction & Refinement

The system now learns structural preferences from discovery outcomes
via differentiable plasticity. Key findings from 20 claims:

- **Ratios (a/b) are the most reliable invariant form** (bias +0.219)
- **Products (a*b) follow** (bias +0.100)  
- **14/14 learned structural patterns are positive**
- **Ratios preferred 2:1 over products** (avg +0.084 vs +0.040)

These relationships were NOT hand-coded — they emerged from experience.

Extract learned relationships:
```bash
python scripts/extract_relationships.py
```

Run provable plastic test (shows generalization across domains):
```bash
python scripts/plastic_test.py
```

See `REVIEW_READY_ROADMAP.md` for completed phases and remaining work.
See `NEURAL_MODEL_PLAN.md` for bias audit and Phase E design.
