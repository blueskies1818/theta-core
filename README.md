# theta-core — Autonomous Mathematical Physics Discovery

A self-play AI system that discovers physics from scratch — given only
mathematical operations and physical measurements, it finds the laws
that govern reality.

## What It Does

Trained exclusively on pre-1905 classical physics (Newtonian mechanics,
Maxwell's electromagnetism, ideal gas thermodynamics), the system has
independently reconstructed 8 post-1905 physical laws:

```
QUANTUM MECHANICS (4/4):
  E = E₀/n²           Hydrogen energy levels
  E ∝ n               Spin quantization
  E/T = constant      Wien's displacement law
  hν - K_max = φ      Photoelectric effect

SPECIAL RELATIVITY (4/4):
  E/γ = constant      Relativistic energy
  u' = (u+v)/(1+uv/c²) Velocity addition
  E² = p²c² + m²c⁴    Energy-momentum relation
  (ct)² - x²          Spacetime interval (time dilation)
```

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
4. **Discovery IS prediction.** A structure succeeds when it implies unmeasured outcomes.
5. **All discoveries are Lean-proven.** No numerical coincidence passes as discovery.

## Quick Start

```bash
# Run all tests (626+)
python -m pytest tests/physics/ tests/core/ -q

# Run the full era gate experiment
# Trains on pre-1905 physics, tests on post-1905 discoveries
python scripts/spacetime_era_gate.py

# Run with a different knowledge cutoff year
python scripts/spacetime_era_gate.py --era-cutoff 1920

# Run self-play physics discovery on a single domain
python src/core/self_play_loop.py --domain gravity

# Train a domain template generator
python scripts/training/train_composer.py --domain em

# Run hidden variable discovery on a specific scenario
python -c "
from src.physics.hidden_variables import HiddenVariableDiscovery
discovery = HiddenVariableDiscovery('data/observations/hydrogen_balmer.json')
result = discovery.run()
print(f'Discovered: {result.expression} (score={result.score:.4f})')
"

# Generate synthetic observation data for a new domain
python scripts/build/generate_observations.py --domain quantum --count 50
```

### Configurable Era Gate

The era knowledge cutoff is a single configurable variable. Change it to test
how well the system generalizes from different historical baselines:

```bash
# Pre-1905 (default) — no quantum, no relativity
python scripts/spacetime_era_gate.py --era-cutoff 1905

# Pre-1920 — includes special relativity and early quantum
python scripts/spacetime_era_gate.py --era-cutoff 1920

# Pre-1950 — includes QED and nuclear physics
python scripts/spacetime_era_gate.py --era-cutoff 1950
```

Each cutoff gates which domains, symmetries, and training scenarios are available.
The test set is always ALL known post-cutoff physics. This measures how discovery
capability scales with historical knowledge.

## Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| A-C | ✓ | Expression infrastructure, observations, self-play loop |
| D | ✓ | Automated Lean proof generation |
| E | ✓ | 57-scenario observation database, work-energy theorem |
| F | ✓ | Per-domain AI composer, 7 domains, zero-shot composition |
| Symmetry | ✓ | Noether derivation, symmetry detection + discovery |
| Era Gate | ✓ | 7→8/8 post-1905 laws reconstructed from pre-1905 training |
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
docs/reports/      Era gate results and analysis
```
