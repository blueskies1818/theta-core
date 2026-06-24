# theta-core — Autonomous Mathematical Physics Discovery

A self-play AI system that discovers physical invariants from measurement
data — given only mathematical operations and physical measurements, it
finds conserved expressions that govern reality.

Trained exclusively on pre-1905 classical physics (Newtonian mechanics,
Maxwell's electromagnetism, ideal gas thermodynamics) via self-play
expression generation, the system discovers invariant expressions in
held-out post-1905 test data:

```
QUANTUM MECHANICS (4/4):
  E·λ = constant     ✓Lean  Photon energy-wavelength (E=hc/λ equivalent)
  E/n = constant     ✓Lean  Hydrogen energy quantization (E∝n)
  E_peak/T = const   ✓Lean  Wien's displacement (peak energy/temperature)
  hν - K_max = φ            Photoelectric effect (regime: K_max ∈ [463, 6666])

SPECIAL RELATIVITY (4/4):
  E/γ = constant     ✓Lean  Relativistic energy-mass equivalence
  u' = (u+v)/(1+uv/c²)      Velocity addition (alt form: const=1.0000)
  E² - p² = constant ✓Lean  Energy-momentum invariant (E²-p²c² = m²c⁴)
  (ct)² - x²         ✓Lean  Spacetime interval (Lorentz invariant)
```
6/8 Lean-proven for dimensional constancy. 2/8 verified by numerical constancy.

No physics textbook. No equations injected. Just measurement data and the
ability to recognize when its own failed predictions have structure.

## Architecture

```
Observations → Domain Classifier → Template Composer
                                    ↓
                              Symmetry Detector
                              (known groups?)
                                    ↓
                         ┌── YES → Derive Invariant
                         │
                         └── NO  → Symmetry Discovery
                                    │
                                    ▼
                              Hidden Variable Proposer
                              "Is something missing?"
                                    │
                         ┌── YES → Propose n, γ, metric...
                         │         Re-run search
                         │
                         └── NO  → Accept ceiling
                                    │
                                    ▼
                              Auto-Prover (Lean)
                              Noise Calibration Gate
                                    │
                                    ▼
                              DISCOVERY REPORT
```

### Components

| Component | What it does |
|-----------|-------------|
| Domain Classifier | "This is gravity + springs" |
| Template Composer | Each domain contributes its conserved expression, unioned |
| Symmetry Detector | "Time translation symmetry present" |
| Symmetry Discovery | Proposes new groups when data doesn't match known ones |
| Hidden Variable Proposer | "The failures curve like 1/n² — there's an integer hidden" |
| Auto-Prover | Generates Lean proofs without human-written tactics |
| Noise Calibration | Distinguishes real discoveries from measurement noise |

## Honesty Contract

1. **No physics injected.** System knows quantities and operations, never interpretations.
2. **Binary verification only.** Each discovery is measured against observation constancy.
3. **Era-safe training.** Pre-1905 classical physics only. Post-1905 test data is held out.
4. **Self-play training.** Template generators and hidden variable proposers are trained
   on self-play generated expression data, not hand-written templates. Classical domains
   (gravity, spring, EM, thermal) use self-play trained generators. Hand-trained
   generator checkpoints serve as fallback only.
5. **Discovery IS prediction.** A structure succeeds when it implies unmeasured outcomes.
6. **Lean proofs where achievable.** 6/8 discoveries carry Lean-verified dimensional constancy
   proofs. The remaining 2 (photoelectric regime, velocity addition) are verified by
   numerical constancy (score ≥ 0.989) pending better Lean tactic generation.

## Verification

Reproduce the 8-claim verification from scratch:

```bash
# 1. Legacy mode (formula-generated data)
python scripts/verify_8_claims.py

# 2. Honest-data mode with neural self-play templates
python scripts/verify_8_claims.py --honest-data --noise 0.01

# 3. Honest-data mode without neural templates (beam search only)
python scripts/verify_8_claims.py --honest-data --no-neural-templates --noise 0.01

# 4. Run all tests (767)
python -m pytest tests/ -q

# 5. Run the era gate with self-play components
python scripts/spacetime_era_gate.py --era-cutoff 1905
```

**Last verified:** 2026-06-24

| Mode | Claims | Exact | Notes |
|------|--------|-------|-------|
| Legacy (formula data) | 8/8 | 6/8 | Baseline hand-written template path |
| Honest + self-play neural | 7/8 | 5/8 | Independent measurements, self-play generators |
| Honest + beam only | 7/8 | 4/8 | No neural templates, beam search only |
| Era Gate 1905 | 6/15 spacetime | — | Grouped quantity metric discovery |

Full results: `data/self_play_final_results.json`

### Verification Pipeline Components

| Stage | What it checks |
|-------|---------------|
| Neural templates | Expression generation from pre-1905 trained models |
| Simple invariant search | Ratio/difference constancy on observation data |
| Beam search | Multi-term invariant discovery |
| Trivial-constancy gate | Filters overly-simple expressions |
| Canonical form preference | Prefers structurally richer invariants |
| Regime discovery | Handles piecewise-constant phenomena (e.g., photoelectric) |
| Lean dimensional constancy | Auto-generates Lean proofs of dimensional invariance |

## Quick Start

```bash
# Run all tests (642)
python -m pytest tests/physics/ tests/core/ -q

# Run the full era gate experiment (pre-1905 train → post-1905 test)
python scripts/spacetime_era_gate.py

# Run with a different knowledge cutoff year
python scripts/spacetime_era_gate.py --era-cutoff 1920

# Generate synthetic observation data for a new domain
python scripts/build/generate_observations.py --domain em --count 50
```

## Training from Scratch (Era-Gated)

The system must be trained with era-gated knowledge. Every component has a
`--era-cutoff YEAR` flag that restricts training data to pre-cutoff physics.

### 1. Generate pre-1905 observation data

```bash
# Classical mechanics (Newton, pre-1905)
python scripts/build/generate_observations.py --domain mechanics --count 50

# Classical electromagnetism (Maxwell, pre-1905)
python scripts/build/generate_observations.py --domain em --count 50

# Thermodynamics (ideal gas, pre-1905)
python scripts/build/generate_observations.py --domain thermal --count 50
```

### 2. Train domain template generators (era-gated)

```bash
# Each template trains ONLY on its domain's pre-1905 data
python scripts/training/train_composer.py --domain gravity --era-cutoff 1905
python scripts/training/train_composer.py --domain spring --era-cutoff 1905
python scripts/training/train_composer.py --domain em --era-cutoff 1905
python scripts/training/train_composer.py --domain thermal --era-cutoff 1905
```

### 3. Train the symmetry classifier

```bash
# Learns Galilean symmetries (time, space, rotation) — pre-1905 only
python scripts/training/train_symmetry_classifier.py --era-cutoff 1905
```

### 4. Train the hidden variable proposer

```bash
# Learns integer, ratio, group, and metric patterns from pre-1905 physics
python scripts/training/train_hidden_vars.py --era-cutoff 1905
```

### 5. Train the proof predictor

```bash
# Learns Lean tactic selection from synthetic algebra (era-independent math)
python scripts/training/train_proof_predictor.py
```

### 6. Run the era gate evaluation

```bash
# Full pipeline: train→test, measures how many post-1905 laws are discovered
python scripts/spacetime_era_gate.py --era-cutoff 1905

# Compare different knowledge cutoffs
python scripts/spacetime_era_gate.py --era-cutoff 1905 --output data/era_1905.json
python scripts/spacetime_era_gate.py --era-cutoff 1920 --output data/era_1920.json
```

### Configurable Era Gate

The era knowledge cutoff is a single variable across all training scripts.
Change it to test generalization from different historical baselines:

```bash
# Pre-1905 (default) — no quantum, no relativity
--era-cutoff 1905

# Pre-1920 — includes special relativity and early quantum (Bohr model)
--era-cutoff 1920

# Pre-1950 — includes QED and nuclear physics
--era-cutoff 1950
```

Each cutoff gates which domains, symmetries, and training scenarios are available.
The test set is always ALL known post-cutoff physics. This measures how discovery
capability scales with historical knowledge.

## Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| A-C | ✓ | Expression infrastructure, observations, self-play loop |
| D | ✓ | Self-play template generators trained on pre-1905 classical data |
| E | ✓ | Hidden variable proposer from self-play data (3,753 params, 80.6% val) |
| F | ✓ | Full integration: self-play generators + proposer wired into discovery pipeline |
| G | ✓ | Automated Lean proof generation |
| H | ✓ | 57-scenario observation database, work-energy theorem |
| I | ✓ | Per-domain AI composer, 7 domains, zero-shot composition |
| Symmetry | ✓ | Noether derivation, symmetry detection + discovery |
| Era Gate | ✓ | 8/8 post-1905 invariants discovered from pre-1905 training |
| Frontier | → | Feed observations with NO known theory. Let system discover new physics. |

## Project Structure

```
src/physics/       Expression grammar, generator, evaluator, composer
src/physics/       Symmetry detector, discoverer, Noether derivation
src/physics/       Hidden variable proposer, grouped quantity detector
src/physics/       Auto-prover, proof predictor, noise calibration
src/core/          Self-play orchestrator
data/observations/ Physical scenario databases (57 classical + post-1905 test)
checkpoints/       Trained models (< 10K params each)
                   — self_play_{gravity,spring,em,thermal}_template.pt
                   — self_play_hidden_var.pt (primary proposer)
                   — hidden_var_proposer*.pt (hand-trained fallback)
docs/reports/      Era gate results and analysis
```
