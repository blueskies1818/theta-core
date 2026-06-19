# theta-core

Autonomous Mathematical Physics AI — a self-play system for exploring formal mathematics beyond the boundaries of current human knowledge.

## Overview

theta-core is inspired by AlphaGo Zero's self-play mechanism — a model that generates its own training signal through interaction with a verifiable environment — applied to formal mathematics rather than a board game. The system learns to prove theorems and propose mathematical structures by generating candidates, checking them against a deterministic proof verifier, and using the verifier's accept/reject output as its sole training signal. No human labels, no human judgment at any step of the training loop.

Where AlphaGo Zero had the game of Go (perfectly verifiable state, unambiguous rules, clear terminal reward), theta-core has formal proof systems (Lean 4) as the verifiable environment, reproduction of experimentally verified results as the rules, and predictive compression of physical observation data as the terminal reward signal. Crucially, the "rules" are not specific theories like GR or QFT — those are models, not gospel. The rules are the empirical facts: conservation laws, spectral lines, particle masses, symmetry constraints confirmed across independent measurements. Together these create a self-sustaining training loop.

The primary scientific target is the breakdown zone where general relativity and quantum mechanics are simultaneously necessary and currently incompatible — Planck scale, black hole singularities, Big Bang initial conditions. Current theories produce mathematical infinities at these coordinates; candidate structures are rewarded for remaining finite and consistent precisely where current theories break.

### Why Not Train on Human Data?

Current large language models inherit human cognitive biases, conceptual categories that may carve nature at the wrong joints, and the limits of what humans have thought to write down. AlphaGo Zero demonstrated that removing human priors can produce qualitatively superior results. The challenge is that mathematics is not a game with a clean terminal reward — so the mechanism for removing human bias must be more carefully constructed. This system replaces human judgment with formal verification and physical measurement data, providing automated feedback without human conceptual bottlenecks.

---

## System Architecture

The full system is a heterogeneous ensemble of three components with distinct architectures. No single monolithic model handles all tasks. Below is the complete data-flow diagram showing every processing stage from training data through to the final GRPO gradient update.

```
┌───────────────────────────────────────────────────────────────────────────┐
│                          TRAINING DATA                                    │
│  Mathlib4 .lean files → regex extractor → JSONL theorem-proof pairs       │
│  (4,886+ theorems from Algebra, GroupTheory, LinearAlgebra, Data domains) │
└────────────────────────────────────┬──────────────────────────────────────┘
                                     │
                                     ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                      PROMPT FORMATTING                                    │
│  "Theorem: lemma add_comm (a b : ℕ) : a + b = b + a\nProof:"              │
│  (plain text, no few-shot, no instructions)                               │
└────────────────────────────────────┬──────────────────────────────────────┘
                                     │
                                     ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                                                                           │
│  ┌──────────────────────────┐     ┌──────────────────────────┐            │
│  │    DEPENDENCY GRAPH      │     │      GNN ENCODER         │            │
│  │  (69K nodes, 280K edges) │────▶│  GAT layers + GoalEncoder│            │
│  │                          │     │                          │            │
│  │  Nodes: theorems, lemmas │     │  Input: initial features │            │
│  │  Edges: "uses in proof"  │     │  (random/onehot/transf.) │            │
│  │                          │     │                          │            │
│  │  Stored as NetworkX      │     │  Output: [N, hidden_dim] │            │
│  │  DiGraph + pickle        │     │  node embeddings         │            │
│  └──────────────────────────┘     └──────────────┬───────────┘            │
│                                                  │                        │
│                                                  │ embeddings             │
│                                                  ▼                        │
│  ┌──────────────────────────────────────────────────────────────────┐     │
│  │                    MCTS SEARCH ENGINE                            │     │
│  │                                                                  │     │
│  │  For each theorem, K independent searches (K = group_size):      │     │
│  │                                                                  │     │
│  │  ┌──────────┐   ┌───────────┐   ┌──────────┐   ┌────────────┐    │     │
│  │  │ SELECT   │──▶│  EXPAND   │──▶│ EVALUATE │──▶│ BACKPROP   │    │     │
│  │  │ PUCT     │   │ GNN scores│   │ value    │   │ update     │    │     │
│  │  │ treewalk │   │ candidates│   │ estimate │   │ visit cnts │    │     │
│  │  └──────────┘   └───────────┘   └──────────┘   └────────────┘    │     │
│  │       │              │                                           │     │
│  │       │     ┌────────┴────────┐                                  │     │
│  │       │     │  Action scoring │                                  │     │
│  │       │     │  ────────────── │                                  │     │
│  │       │     │  goal_emb ·     │                                  │     │
│  │       │     │  lemma_emb      │  ← differentiable! gradient      │     │
│  │       │     │  (cosine sim)   │    flows back to GNN here        │     │
│  │       │     │                 │                                  │     │
│  │       │     │  + heuristic    │                                  │     │
│  │       │     │    (arithmetic  │  ← annealed 1.0 → 0.25           │     │
│  │       │     │     patterns)   │    during training               │     │
│  │       │     │                 │                                  │     │
│  │       │     │  + keyword      │                                  │     │
│  │       │     │    relevance    │  ← penalty for irrelevant        │     │
│  │       │     │    matching     │    lemmas (e.g., zero_pow        │     │
│  │       │     │                 │    for goals without 0)          │     │
│  │       │     │  + centrality   │                                  │     │
│  │       │     │    (in-degree)  │  ← fundamental lemmas boosted    │     │
│  │       │     │                 │                                  │     │
│  │       │     │  + trivial      │                                  │     │
│  │       │     │    lemma penalty│  ← id, rfl, Function.id etc      │     │
│  │       │     │    (-1.5 score) │    heavily penalized             │     │
│  │       │     └────────┬────────┘                                  │     │
│  │       │              │                                           │     │
│  │       │              ▼                                           │     │
│  │       │     ┌──────────────────┐                                 │     │
│  │       │     │  1000 sims       │                                 │     │
│  │       │     │  per proof       │──▶ best_proof_steps, root_node  │     │
│  │       │     └──────────────────┘                                 │     │
│  │       │                                                          │     │
│  │       │  ┌──────────────────────────────────────────┐            │     │
│  │       │  │  Verification gate (optional, root only) │            │     │
│  │       │  │  Batch-check candidate tactics through   │            │     │
│  │       │  │  Lean before creating child nodes.       │            │     │
│  │       │  │  Only valid steps become MCTS children.  │            │     │
│  │       │  └──────────────────────────────────────────┘            │     │
│  │       │                                                          │     │
│  │       └────▶  proof_steps = [Tactic(APPLY, "add_comm"), ...]     │     │
│  │                          ↓                                       │     │
│  │              _render_proof(steps) → "  apply add_comm\n  rfl"    │     │
│  └──────────────────────────────────────────────────────────────────┘     │
│                                                                           │
└────────────────────────────────────┬──────────────────────────────────────┘
                                     │ proof text string
                                     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                      PROOF WRAPPING                                        │
│  wrap_theorem_with_proof(statement, proof)                                 │
│                                                                            │
│  Input:  "theorem add_comm (a b:ℕ): a+b = b+a" + "  apply add_comm"        │
│  Output: "example (a b:ℕ): a+b = b+a := by\n  apply add_comm"              │
│  (lemma→example to avoid Mathlib name collisions, auto-detect tactic/term) │
└────────────────────────────────────┬───────────────────────────────────────┘
                                     │ wrapped Lean code
                                     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                      PROOF CHECKER (Lean 4)                                │
│                                                                            │
│  ┌──────────────────┐    ┌───────────────────┐    ┌────────────────────┐   │
│  │ SHA-256 Cache    │    │ lean --stdin      │    │ BatchChecker       │   │
│  │ LRU, 50K entries │───▶│ subprocess.run()  │◀───│ ProcessPoolExecutor│   │
│  │                  │    │                   │    │ spawn, 3-12 workers│   │
│  │ Avoids re-check- │    │ lake env lean     │    │                    │   │
│  │ ing identical    │    │ (Mathlib4 imports)│    │ Each check:        │   │
│  │ code strings     │    │                   │    │ independent, CPU   │   │
│  └──────────────────┘    └─────────┬─────────┘    └────────────────────┘   │
│                                    │                                       │
│                ProofResult(success=True/False, errors=[...], num_tokens)   │
└────────────────────────────────────┬───────────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                      REWARD COMPUTATION                                    │
│                                                                            │
│  base_reward = 1.0 (valid) or 0.0 (invalid)                                │
│                                                                            │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────────┐  │
│  │ Length bonus     │    │ Curiosity bonus  │    │  Anti-hack threshold │  │
│  │ shorter=higher   │    │ novelty /        │    │  proofs < 10 tokens  │  │
│  │ decay_rate=0.002 │    │ sqrt(count + 1)  │    │  → reward = 0.0      │  │
│  └──────────────────┘    └──────────────────┘    └──────────────────────┘  │
└────────────────────────────────────┬───────────────────────────────────────┘
                                     │ raw reward
                                     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                    CORRESPONDENCE LAYER (reward shaper)                    │
│                                                                            │
│  ┌──────────────────────┐   ┌─────────────────────┐    ┌──────────────┐    │
│  │ Frontier Map         │   │ Failure Coordinates │    │  Era Tracker │    │
│  │ ─────────────        │   │ ─────────────────── │    │  ─────────── │    │
│  │ ESTABLISHED: 0.1-0.3×│   │ Singularity points  │    │  Passive     │    │
│  │ UNCERTAIN:   1.2-2.0×│   │ Planck breakdown    │    │  discovery   │    │
│  │ BREAKDOWN:   2.5-3.0×│   │ QFT divergences     │    │  monitor     │    │
│  │                      │   │ Big Bang t=0        │    │  (no reward  │    │
│  │ Keyword-classifies   │   │                     │    │   signal)    │    │
│  │ theorem text into    │   │ Resolve=bonus       │    │              │    │
│  │ zone → multiplier    │   │ Reproduce=penalty   │    │  "Did it     │    │
│  └─────────┬────────────┘   └─────────┬───────────┘    │   discover   │    │
│            │                          │                │   Lorentz    │    │
│            └───────────┬──────────────┘                │   invar.?"   │    │
│                        │                               └──────────────┘    │
│                        ▼                                                   │
│  modified_reward = base_reward × zone_multiplier + failure_modifier        │
└────────────────────────────────────┬───────────────────────────────────────┘
                                     │ final reward
                                     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                      GRPO TRAINING (loss.backward())                       │
│                                                                            │
│  1. Group-relative advantages: (rᵢ - mean(r_group)) / (std(r_group) + ε)   │
│                                                                            │
│  2. Policy loss (through MCTS logits → GNN):                               │
│     CrossEntropy(softmax(child_logits), MCTS_visit_distribution)           │
│     Weighted by advantage. Gradients flow:                                 │
│       loss → log_softmax → cosine_sim(goal_emb, lemma_emb)                 │
│            → goal_emb = embeddings[matched].mean()                         │
│            → embeddings = GNN(features, edges) ✓                           │
│                                                                            │
│  3. Value loss: MSE(predicted_value, actual_success)                       │
│                                                                            │
│  4. Entropy bonus: -weight × H(probs)  (maximize, prevents collapse)       │
│                                                                            │
│  5. KL penalty vs frozen reference model (LLM Phase 1 only)                │
│                                                                            │
│  6. Heuristic annealing: scale 1.0→0.25 over 2000 epochs                   │
│     (GNN gradually takes over from hand-coded arithmetic patterns)         │
└────────────────────────────────────────────────────────────────────────────┘
```

### How Data Flows Through the System

| Stage | What happens | Key files |
|-------|-------------|-----------|
| **Training data** | Theorem-proof pairs extracted from Mathlib4 `.lean` files | `src/data/mathlib_extractor.py` |
| **Prompt formatting** | Theorem statement formatted as `"Theorem: <stmt>\nProof:"` | `src/data/dataset.py` |
| **Dependency graph** | 69K nodes, 280K edges of mathematical dependencies | `src/explorer/dependency_graph.py` |
| **GNN encoder** | GAT with edge-type conditioning, learns embeddings over the graph | `src/explorer/gnn_encoder.py` |
| **MCTS search** | 1000 simulations per proof, PUCT tree search guided by GNN scores | `src/explorer/mcts.py` |
| **Proof wrapping** | Converts theorem+proof into checkable Lean code with Mathlib4 imports | `src/proof_checker/formats.py` |
| **Proof checker** | Subprocess `lean --stdin`, parallel batch verification, SHA-256 LRU cache | `src/proof_checker/` |
| **Reward computation** | Binary (valid/invalid) + length bonus + curiosity/exploration bonus | `src/reward/base.py` |
| **Correspondence layer** | Frontier zone multiplier + failure point bonus/penalty → modified reward | `src/correspondence/reward_integration.py` |
| **GRPO update** | Group-relative advantages → policy loss + value loss + entropy bonus → GNN update | `src/explorer/explorer_trainer.py` |

### Component 1 — Mathematical Explorer
**Architecture:** Graph Neural Network + Monte Carlo Tree Search (1.1M params current, targeting 5–10M for Phase 2, 1–7B for Phase 3+)
Navigates formal mathematical space, proposes candidate structures and proof steps. The GNN operates on the dependency graph of mathematical objects; MCTS handles the exploration problem of deciding which proof tactic to try next. The proof checker validates every proposed addition. **Implemented and validated — self-play loop produces genuine learning from proof-checker feedback alone.**

### Component 2 — Physical Prediction Scorer
**Architecture:** Multimodal Transformer with domain-specific encoders (10–30B params)
Evaluates candidate structures against raw physical observation data across all measurement modalities. Domain-specific encoders (time series, spatial field, spectroscopic, discrete event, thermodynamic/chemical) convert heterogeneous data into a common representation space before a shared transformer compares predictions to observations. **Not yet implemented — raw experimental data currently enters only through the correspondence layer's reward shaping.**

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

- Physical Prediction Scorer (Component 2) — design-only
- Translation Layer (Component 3) — not started
- Multi-step proof chaining (GNN currently learns single-tactic proofs only)
- Full 58K graph training (currently Algebra subgraph: 16,842 nodes)
- Lemma-level discrimination beyond top-30 candidate filter
- Genuine era-gated discovery test (current theorems are algebraic identities with physics labels)
- Solution enumeration and anomaly flagging (Dirac mechanism)
- Layer 1 / Layer 2 data architecture for physical measurements
- Domain-level holdout strategy
- Simplicity penalty (Occam's razor formalized)

### Phase 2 Progress (June 2026)

The GNN+MCTS explorer is operational and validated:
- **Self-play loop works:** 1.1M-param GNN improved from 40%→56% on held-out post-1905 theorems through GRPO training from proof-checker feedback alone (Run 8)
- **GNN exceeds heuristics:** Trained GNN at H=0.0 (14/25) beat hand-coded heuristics at H=1.0 (13/25)
- **Entropy bonus prevents collapse:** Policy entropy stable at 3.4/3.5 max even at H=0.25
- **Architecture validated:** GNN+GoalEncoder → MCTS with Dirichlet noise → Proof checker verification → GRPO → Heuristic annealing
- **Wave 1 (June 2026):** Dirichlet ε=0.45, 500 sims, H-min=0.25, entropy bonus, trivial baselines (linarith ceiling: 64%), multi-run eval (±2.3pp)
- **Wave 2 (June 2026):** Multi-step theorems (14 new Level 4), curriculum learning, gradient diversity fix

See [docs/training/README.md](docs/training/README.md) for full training history and [docs/reviews/roadmap_review_june2026.md](docs/reviews/roadmap_review_june2026.md) for the roadmap assessment.

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
2. **Reproduction of experimentally verified results** — must predict the outcomes that have been confirmed across independent measurements. GR and QFT are the best current fits to these results, not the results themselves.
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
├── scripts/                           # CLI entry points (organized by function)
│   ├── gates/                         # Audit and verification scripts
│   │   ├── audit_structural.py        # Structural audit
│   │   ├── gate4_evaluate.py          # Gate 4 evaluation
│   │   └── hybrid_gates.py            # Hybrid gate runner
│   ├── training/                      # Training scripts
│   │   ├── train_explorer.py          # Main training entry point
│   │   ├── train_sft.py               # Run SFT pretraining
│   │   └── train_grpo.py              # Run GRPO self-play training
│   ├── eval/                          # Evaluation and benchmarking
│   │   ├── infer_explorer.py          # Inference / evaluation entry point
│   │   ├── evaluate.py                # Evaluate model on held-out theorems
│   │   └── run_full_gate3_v2.py       # Full gate3 benchmark
│   ├── build/                         # Data builders and generators
│   │   ├── prepare_data.py            # Extract theorems from Mathlib4
│   │   ├── build_dependency_graph.py  # Build math dependency graph
│   │   └── generate_sample.py         # Interactive proof generation
│   └── tools/                         # Debug, smoke tests, studies
│       ├── debug_mcts_proof.py        # MCTS proof debugger
│       └── smoke_test_contrastive.py  # Contrastive retrieval smoke test
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
python scripts/build/prepare_data.py \
  --mathlib-dir ../mathlib4 \
  --output-dir data \
  --max-theorems 50000
```

### Run Supervised Fine-Tuning

```bash
# Pretrain the model on theorem-proof pairs
python scripts/training/train_sft.py \
  --data-dir data \
  --output-dir checkpoints/sft \
  --use-lora  # optional, for memory-constrained setups
```

### Run GRPO Self-Play Training

```bash
# Self-play training against the Lean 4 proof checker
python scripts/training/train_grpo.py \
  --sft-checkpoint checkpoints/sft/final \
  --data-dir data \
  --output-dir checkpoints/grpo \
  --max-theorems 1000
```

### Evaluate

```bash
# Test on held-out theorems
python scripts/eval/evaluate.py \
  --checkpoint checkpoints/grpo/checkpoint-1000 \
  --num-theorems 100
```

### Interactive Testing

```bash
# Generate and check proofs interactively
python scripts/build/generate_sample.py \
  --checkpoint checkpoints/sft/final \
  --theorem "theorem add_comm (a b : Nat) : a + b = b + a"
```

---

## Development Roadmap

| Phase | Goal | Key Deliverable | Status |
|-------|------|-----------------|--------|
| **Phase 1** | Validate the self-play loop | GNN improved 40%→56% from proof-checker feedback alone | ✅ Complete |
| **Phase 2** | Scale the explorer | Multi-step proofs, lemma discrimination, full 58K graph, failure condition encoding | 🔄 60% — training |
| **Phase 3** | Integrate physical grounding | Physical Prediction Scorer, Layer 1/2 data architecture, domain-level holdout | ⬜ Design only |
| **Phase 4** | Translation layer | Fine-tuned formal-to-natural translator with automated verification | ⬜ Not started |
| **Phase 5** | Open-ended operation | Continuous integrated operation, holdout commitment against future experiments | ⬜ Not started |

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
