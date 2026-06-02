# theta-core

Autonomous Mathematical Physics AI — a self-play system for exploring formal mathematics beyond the boundaries of current human knowledge.

## Overview

theta-core is inspired by AlphaGo Zero's self-play mechanism — a model that generates its own training signal through interaction with a verifiable environment — applied to formal mathematics rather than a board game. The system learns to prove theorems and propose mathematical structures by generating candidates, checking them against a deterministic proof verifier, and using the verifier's accept/reject output as its sole training signal. No human labels, no human judgment at any step of the training loop.

Where AlphaGo Zero had the game of Go (perfectly verifiable state, unambiguous rules, clear terminal reward), theta-core has formal proof systems (Lean 4) as the verifiable environment, correspondence with established GR and QFT as the rules, and predictive compression of physical observation data as the terminal reward signal. Together these create a self-sustaining training loop.

The primary scientific target is the breakdown zone where general relativity and quantum mechanics are simultaneously necessary and currently incompatible — Planck scale, black hole singularities, Big Bang initial conditions. Current theories produce mathematical infinities at these coordinates; candidate structures are rewarded for remaining finite and consistent precisely where current theories break.

### Why Not Train on Human Data?

Current large language models inherit human cognitive biases, conceptual categories that may carve nature at the wrong joints, and the limits of what humans have thought to write down. AlphaGo Zero demonstrated that removing human priors can produce qualitatively superior results. The challenge is that mathematics is not a game with a clean terminal reward — so the mechanism for removing human bias must be more carefully constructed. This system replaces human judgment with formal verification and physical measurement data, providing automated feedback without human conceptual bottlenecks.

---

## System Architecture

The full system is a heterogeneous ensemble of three components with distinct architectures. No single monolithic model handles all tasks.

```
┌─────────────────────────────────────────────────────────┐
│                    TRAINING LOOP                        │
│                                                         │
│  ┌──────────────┐     ┌──────────────┐                  │
│  │  Mathematical│     │    Lean 4    │                  │
│  │   Explorer   │────▶│    Proof     │                  │
│  │  (GNN + MCTS)│     │   Checker    │                  │
│  └──────┬───────┘     └──────┬───────┘                  │
│         │                    │ verified structures      │
│         │ reward signal      ▼                          │
│         │            ┌──────────────┐                   │
│         └────────────│   Physical   │                   │
│                      │  Prediction  │◀── experimental   │
│                      │   Scorer     │    data corpus    │
│                      └──────┬───────┘                   │
│                             │ flagged anomalies         │
└─────────────────────────────┼───────────────────────────┘
                              ▼
                    ┌──────────────────┐
                    │  Translation     │     human
                    │  Layer (LLM)     │────▶physicists
                    └──────────────────┘
```

### Component 1 — Mathematical Explorer
**Architecture:** Graph Neural Network + Monte Carlo Tree Search (1–7B params)
Navigates formal mathematical space, proposes candidate structures and proof steps. The GNN operates on the dependency graph of mathematical objects; MCTS handles the exploration problem of deciding which proof tactic to try next. The proof checker validates every proposed addition.

### Component 2 — Physical Prediction Scorer
**Architecture:** Multimodal Transformer with domain-specific encoders (10–30B params)
Evaluates candidate structures against raw physical observation data across all measurement modalities. Domain-specific encoders (time series, spatial field, spectroscopic, discrete event, thermodynamic/chemical) convert heterogeneous data into a common representation space before a shared transformer compares predictions to observations.

### Component 3 — Translation Layer
**Architecture:** Fine-tuned LLM (7–70B params)
Converts formal mathematical outputs into natural language for human physicists. Generates experimental proposals for flagged anomalous solutions. Translation outputs are verifiable — claims about formal properties can be checked against the proof checker.

---

## Current Status — Phase 1

Phase 1 validates the self-play loop. A small-scale system proves that the model can learn theorem proving from proof-checker feedback alone.

### What's Implemented

| Component | Status |
|-----------|--------|
| Transformer proof generator (Qwen2.5-1.5B, placeholder for GNN+MCTS) | Done |
| Lean 4 proof checker interface with subprocess invocation | Done |
| Parallel batch proof checking across CPU cores | Done |
| Proof result caching (LRU, SHA-256 keyed) | Done |
| SFT pretraining on Mathlib4 theorem-proof pairs | Done |
| GRPO self-play training with group-relative advantages | Done |
| KL penalty against frozen reference model | Done |
| Experience replay buffer | Done |
| Binary reward + length bonus | Done |
| Checkpoint save/load | Done |
| Intel XPU / CUDA / CPU device support | Done |
| Lake-managed Lean 4 project with Mathlib4 imports | Done |
| Configurable via YAML (model, SFT, GRPO, reward) | Done |

### What's Not Yet Implemented (Phase 2–5)

- GNN + MCTS architecture for the Mathematical Explorer
- Physical Prediction Scorer (Component 2)
- Translation Layer (Component 3)
- Curiosity/exploration reward to prevent mode collapse
- Formal frontier map with zone-weighted rewards (established/uncertain/breakdown)
- Correspondence checks against GR and QFT limits
- Solution enumeration and anomaly flagging (Dirac mechanism)
- Layer 1 / Layer 2 data architecture for physical measurements
- Domain-level holdout strategy
- Simplicity penalty (Occam's razor formalized)

See [IMPROVEMENT_IDEAS.md](IMPROVEMENT_IDEAS.md) for the full running list.

---

## Training Methodology

### Phase 1 Training Pipeline

1. **Data extraction:** Theorem-proof pairs extracted from Mathlib4 `.lean` source files via regex-based parsing. Domains relevant to differential geometry and GR are prioritized.

2. **SFT pretraining:** The base model (Qwen2.5-1.5B-Instruct) is fine-tuned on theorem-proof pairs to learn the syntax and structure of Lean 4 proofs. This gives the model a starting point before self-play begins.

3. **GRPO self-play:** The core AlphaGo Zero analog:
   - Model generates K proofs per theorem statement
   - Lean 4 proof checker verifies each proof (CPU, parallel)
   - Binary reward computed (valid = 1.0, invalid = 0.0) with optional length bonus
   - Group-relative advantages computed (proofs compared within their theorem group)
   - Policy updated via GRPO loss + KL divergence penalty against frozen reference model

### The Three Pressures (Full System)

The full system imposes a nested hierarchy of pressures that together create a gradient toward discovery:

1. **Internal consistency** — proof checker output. Any structure either holds together formally or it doesn't.
2. **Correspondence at known limits** — must reduce to GR when quantum effects are negligible, must reduce to QFT when gravity is negligible.
3. **Predictive compression** — finding structures that describe known physical observations in fewer, more fundamental terms. Occam's razor formalized.

---

## Project Structure

```
theta-core/
├── README.md                          # This file
├── IMPROVEMENT_IDEAS.md               # Running list of planned improvements
├── mathematical_ai_system.md          # Full system design document
├── model_structure_and_data.md        # Detailed technical specification
├── pyproject.toml                     # Python project config
│
├── configs/                           # YAML configuration files
│   ├── model_config.yaml              # Base model, LoRA, precision
│   ├── sft_config.yaml                # Supervised fine-tuning
│   ├── grpo_config.yaml               # GRPO self-play training
│   └── reward_config.yaml             # Reward computation
│
├── src/                               # Source code
│   ├── __init__.py                    # Package docstring, Phase 1 scope
│   ├── model/                         # Model loading & generation
│   │   ├── loader.py                  # Model/tokenizer loading, LoRA
│   │   └── generation.py             # Proof generation with controlled decoding
│   ├── environment/                   # Self-play environment (AlphaGo Zero analog)
│   │   ├── env.py                     # ProofEnvironment (observation/action/reward)
│   │   └── action_space.py           # Token-level action space definition
│   ├── proof_checker/                 # Lean 4 proof verification
│   │   ├── lean_interface.py         # Subprocess-based proof checking
│   │   ├── formats.py                # Code wrapping, error parsing
│   │   ├── batch_checker.py          # Parallel checking via ProcessPoolExecutor
│   │   └── cache.py                  # LRU result cache
│   ├── reward/                        # Reward computation
│   │   ├── base.py                    # compute_reward, group advantages
│   │   └── config.py                 # RewardConfig dataclass
│   ├── training/                      # Training loops
│   │   ├── sft_trainer.py            # Supervised fine-tuning
│   │   ├── grpo_trainer.py           # GRPO self-play training
│   │   └── losses.py                 # GRPO loss, KL penalty, sequence logprob
│   ├── data/                          # Data pipeline
│   │   ├── mathlib_extractor.py      # Theorem extraction from .lean files
│   │   ├── dataset.py                # PyTorch Dataset classes
│   │   └── replay_buffer.py          # Experience replay for GRPO
│   └── utils/                         # Utilities
│       ├── config.py                  # Pydantic config models, YAML loading
│       ├── xpu_utils.py              # XPU/CUDA/CPU device selection
│       ├── logging.py                 # Rich-based metrics logging
│       └── checkpoint.py             # Checkpoint save/load
│
├── scripts/                           # CLI entry points
│   ├── prepare_data.py               # Extract theorems from Mathlib4
│   ├── train_sft.py                  # Run SFT pretraining
│   ├── train_grpo.py                 # Run GRPO self-play training
│   ├── evaluate.py                   # Evaluate model on held-out theorems
│   └── generate_sample.py            # Interactive proof generation
│
├── tests/                             # Tests
│   ├── test_lean_interface.py        # Proof checker tests
│   ├── test_reward.py                # Reward computation tests
│   └── test_dataset.py              # Data pipeline tests
│
├── proof_checker_env/                 # Lake-managed Lean 4 project
│   ├── lakefile.lean                 # Lake project config
│   ├── lean-toolchain                # Lean version (v4.29.1)
│   └── ProofChecker/
│       ├── Imports.lean              # Mathlib4 imports
│       └── Templates.lean           # Template code for generated proofs
│
├── checkpoints/                       # Model checkpoints (gitignored)
├── data/                              # Training data (gitignored)
│   └── raw/mathlib4_theorems.jsonl   # Extracted theorem-proof pairs
└── logs/                              # Training logs (gitignored)
```

---

## Getting Started

### Prerequisites

- Python ≥ 3.11
- Lean 4 (v4.29.1) — install via [elan](https://github.com/leanprover/elan)
- Mathlib4 — `git clone https://github.com/leanprover-community/mathlib4.git`
- PyTorch with XPU or CUDA support (CPU fallback available)

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd theta-core

# Install Python dependencies
pip install -e .

# Verify Lean 4 is available
lean --version
```

### Prepare Training Data

```bash
# Extract theorem-proof pairs from Mathlib4
python scripts/prepare_data.py \
  --mathlib-dir ../mathlib4 \
  --output-dir data \
  --max-theorems 50000
```

### Run Supervised Fine-Tuning

```bash
# Pretrain the model on theorem-proof pairs
python scripts/train_sft.py \
  --data-dir data \
  --output-dir checkpoints/sft \
  --use-lora  # optional, for memory-constrained setups
```

### Run GRPO Self-Play Training

```bash
# Self-play training against the Lean 4 proof checker
python scripts/train_grpo.py \
  --sft-checkpoint checkpoints/sft/final \
  --data-dir data \
  --output-dir checkpoints/grpo \
  --max-theorems 1000
```

### Evaluate

```bash
# Test on held-out theorems
python scripts/evaluate.py \
  --checkpoint checkpoints/grpo/checkpoint-1000 \
  --num-theorems 100
```

### Interactive Testing

```bash
# Generate and check proofs interactively
python scripts/generate_sample.py \
  --checkpoint checkpoints/sft/final \
  --theorem "theorem add_comm (a b : Nat) : a + b = b + a"
```

---

## Development Roadmap

| Phase | Goal | Key Deliverable |
|-------|------|-----------------|
| **Phase 1** (current) | Validate the self-play loop | Model demonstrably finds proofs it didn't know before training |
| **Phase 2** | Scale the explorer | GNN+MCTS architecture, full GR+QFT Mathlib coverage, known failure condition encoding |
| **Phase 3** | Integrate physical grounding | Physical Prediction Scorer, Layer 1/2 data architecture, domain-level holdout |
| **Phase 4** | Translation layer | Fine-tuned formal-to-natural translator with automated verification |
| **Phase 5** | Open-ended operation | Continuous integrated operation, holdout commitment against future experiments |

---

## Key References

- **AlphaGo Zero** — Silver et al. 2017. Self-play without human priors. The foundational precedent.
- **AlphaProof** — DeepMind 2024. 3B parameter model solving IMO problems via Lean 4 + RL.
- **FunSearch** — DeepMind 2023. LLM generating programs evaluated against mathematical objectives.
- **Chinchilla scaling laws** — Hoffmann et al. 2022. Optimal parameter count vs. training tokens.
- **Mathlib** — Community formal mathematics library for Lean 4.
- **Penrose-Hawking singularity theorems** — Formally proven GR breakdown conditions.
- **Dirac equation** — Historical model for the anomalous solution flagging mechanism.

---

## Design Documents

- [mathematical_ai_system.md](mathematical_ai_system.md) — Full system design: architecture, training methodology, theoretical foundations, hardware, roadmap.
- [model_structure_and_data.md](model_structure_and_data.md) — Detailed technical specification: component internals, data pipeline, metadata schema, preprocessing pipelines, open problems.
- [IMPROVEMENT_IDEAS.md](IMPROVEMENT_IDEAS.md) — Running list of planned improvements and deviations from the plan.

---

## License

Proprietary research code. All rights reserved.
