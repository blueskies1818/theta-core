# Roadmap — Step-by-Step Research Build Plan

*Concrete goals, specific changes, and measurable success criteria for each phase.
Updated 2026-06-01. Current status: Phase 1, pre-training.*

---

## Phase 0 — Foundation (Complete ✓)

**Goal:** Working code skeleton. All interfaces defined. Tests pass. Nothing trained yet.

| # | Task | Status | Files |
|---|---|---|---|
| 0.1 | Lean 4 proof checker interface (subprocess, caching, parallel batch) | Done | `src/proof_checker/` |
| 0.2 | SFT trainer with Mathlib4 data pipeline | Done | `src/training/sft_trainer.py`, `src/data/` |
| 0.3 | GRPO self-play trainer with group-relative advantages | Done | `src/training/grpo_trainer.py` |
| 0.4 | Reward system (binary + length bonus + anti-hack threshold) | Done | `src/reward/` |
| 0.5 | Gym-like environment abstraction (env, action space) | Done | `src/environment/` |
| 0.6 | XPU/CUDA/CPU device support | Done | `src/utils/xpu_utils.py` |
| 0.7 | YAML config system (model, SFT, GRPO, reward) | Done | `configs/`, `src/utils/config.py` |
| 0.8 | Checkpoint save/load, metrics logging | Done | `src/utils/checkpoint.py`, `src/utils/logging.py` |
| 0.9 | Lake-managed Lean project with Mathlib4 imports | Done | `proof_checker_env/` |
| 0.10 | Basic theorem extraction from Mathlib4 (4,886 theorems, 4 domains) | Done | `src/data/mathlib_extractor.py`, `data/raw/` |
| 0.11 | Test suite (Lean interface, reward, data pipeline — 15 tests) | Done | `tests/` |
| 0.12 | Documentation (README, EXPLAINER, MODEL_INTERNALS, IMPROVEMENT_IDEAS) | Done | Root `.md` files |

**Phase 0 deliverable:** Codebase compiles, imports resolve, tests pass. Ready for training.

---

## Phase 1 — Validate the Self-Play Loop

**Goal:** Prove that the model can learn theorem proving from proof-checker feedback alone, with no human-labeled proofs during RL training.

**Success criterion:** Proof success rate on held-out theorems increases measurably over GRPO training steps. The model demonstrably finds valid proofs for theorems it was never shown during SFT.

### 1.1 Build the Lake project (Mathlib4 dependency)

```
What: lake update && lake build in proof_checker_env/
Why:  Every non-trivial proof check needs Mathlib4. Currently
      bare lean --stdin works for omega/native_decide/rfl only.
      Without this, lean_interface.py auto-detects the project
      but lake env lean fails because Mathlib4 isn't built.
Files: proof_checker_env/lakefile.lean (exists, needs lake build)
Risk:  Mathlib4 is ~2 GB download, build takes 30-60 min on 8 cores
```

### 1.2 Re-extract data from physics-relevant Mathlib4 domains

```
What: Run prepare_data.py with expanded domain list
Why:  Current 4,886 theorems are from Algebra/GroupTheory/
      LinearAlgebra/Data only. Missing Analysis (limits, DEs),
      Geometry/Manifold (the language of GR), Topology.
      These are the domains where the GR-QFT interface lives.
Files: scripts/prepare_data.py (exists, needs re-run)
       src/data/mathlib_extractor.py (exists, no changes needed)
Command:
  python scripts/prepare_data.py \
    --mathlib-dir ../mathlib4 \
    --domains Analysis Geometry/Manifold Topology Data/Complex \
              Algebra GroupTheory LinearAlgebra Data/Real Data/Nat \
    --max-theorems 100000
Target: 50,000–100,000 theorems across 8+ domains
```

### 1.3 Run SFT pretraining

```
What: Supervised fine-tuning of Qwen2.5-1.5B on theorem-proof pairs
Why:  Model needs to learn Lean 4 syntax before self-play can work.
      Without SFT, GRPO has zero signal — model generates gibberish
      that never type-checks, no rewards, no learning.
Files: scripts/train_sft.py (exists)
       configs/sft_config.yaml (exists)
Command:
  python scripts/train_sft.py --data-dir data --output-dir checkpoints/sft
Expected: Loss decreases from ~3.0 to ~0.5 over 2 epochs.
          Model learns to output syntactically valid Lean 4.
Time:    2-4 hours on 1 GPU
```

### 1.4 Run GRPO self-play training

```
What: Self-play training against Lean proof checker
Why:  The core experiment. Does the model improve at theorem proving
      when the ONLY signal is proof checker accept/reject?
Files: scripts/train_grpo.py (exists)
       configs/grpo_config.yaml (exists)
Command:
  python scripts/train_grpo.py \
    --sft-checkpoint checkpoints/sft/final \
    --data-dir data --output-dir checkpoints/grpo \
    --max-theorems 1000
Expected: Success rate starts near 0-5% and climbs to 15-30% over
          5,000 steps. If no improvement, debug reward/KL/temperature.
Time:    4-12 hours on 1 GPU + 12 CPU cores
```

### 1.5 Add curiosity/exploration reward

```
What: Count-based exploration bonus in src/reward/base.py
Why:  Plan calls this CRITICAL — without it, model mode-collapses
      onto one proof pattern. "Without an explicit incentive toward
      novelty, the model risks converging on one region."
      This is Phase 1 work, not Phase 3.
How:  Track generated proof hashes. Give small bonus for novel proofs.
      Penalize proofs that exactly match previously generated ones.
      bonus = α / sqrt(count(proof_signature) + 1)
Files: src/reward/base.py (modify compute_reward)
       src/reward/config.py (add curiosity_weight, curiosity_decay)
       configs/reward_config.yaml (add curiosity params)
```

### 1.6 Measure scaling curve

```
What: Train at multiple model sizes (300M, 1.5B, 3B) and measure
      final success rate vs parameter count.
Why:  Plan says "measure the empirical scaling curve before committing
      to larger runs." Chinchilla laws may not apply to self-play.
      We need our own data to plan Phase 2 parameter scale.
Files: configs/model_config.yaml (swap base_model.name)
       New: scripts/benchmark_scaling.py (run eval across checkpoints)
```

### 1.7 Phase 1 write-up

```
What: Document: does self-play work for theorem proving?
      - SFT baseline success rate
      - GRPO final success rate
      - Scaling curve data
      - Failure modes encountered
      - Reward hacking attempts observed
      - Recommendation for Phase 2 parameter scale
Files: New: docs/phase1_report.md
```

**Phase 1 deliverable:** Empirical evidence that the self-play loop functions. A trained model, a scaling curve, and a go/no-go decision for Phase 2.

---

## Phase 2 — Scale the Mathematical Explorer

**Goal:** Build the real architecture. The model generates novel mathematical structures — not just proofs of existing theorems — and is guided toward the GR-QFT breakdown zone.

**Success criterion:** Model proposes a mathematical structure that (a) is internally consistent, (b) reduces to GR at large scales, (c) reduces to QFT at small scales, and (d) was not present in Mathlib4 before training.

### 2.1 Build the math dependency graph

```
What: Parse Mathlib4 into a directed graph where nodes are
      theorems/definitions/structures and edges are logical
      dependencies ("uses," "depends on," "generalizes").
Why:  The GNN operates on this graph. It's the model's "board."
Files: New: src/explorer/dependency_graph.py
       New: src/explorer/graph_builder.py
Deps:  Needs full Mathlib4 built (Phase 1.1)
```

### 2.2 Implement GNN encoder

```
What: Graph neural network that learns node embeddings over the
      math dependency graph. Each node gets a vector representing
      its position in the logical web of mathematics.
Why:  Replaces the transformer as the core architecture.
      The GNN sees structure (what depends on what), not surface
      syntax (token sequences).
Files: New: src/explorer/gnn_encoder.py
       New: src/explorer/gnn_config.py
Arch:  Message-passing GNN (GAT or GIN, 3-5 layers, 512-1024 dim)
       with edge-type-specific transformations.
Train: Initialized from transformer embeddings, then fine-tuned
       on proof-success prediction task.
```

### 2.3 Implement MCTS proof search

```
What: Monte Carlo Tree Search over proof states. At each state,
      evaluate possible tactics with GNN, allocate search budget
      to promising branches, backtrack from dead ends.
Why:  The plan is explicit: "A 3B model searching for ten thousand
      steps explores more of mathematical space than a 100B model
      proposing one candidate immediately."
      Test-time compute is a first-class design axis.
Files: New: src/explorer/mcts.py
       New: src/explorer/proof_state.py
       New: src/explorer/tactic_space.py
Deps:  GNN encoder (Phase 2.2), dependency graph (Phase 2.1)
Params: num_simulations=1000-10000 per proof, UCB exploration constant
```

### 2.4 Implement structure proposal (beyond proofs)

```
What: Model generates NEW mathematical objects — metrics, Lagrangians,
      symmetry groups, connection forms — not just proofs of existing
      theorems. These are candidate structures for physical reality.
Why:  This is where the system starts doing what humans haven't.
      The explorer becomes a generative model over mathematical
      structures, not just a proof completer.
Files: New: src/explorer/structure_generator.py
       New: src/explorer/structure_validator.py
How:  GNN proposes modifications to known structures (add terms,
      change symmetry groups, modify action functionals).
      Validator checks internal consistency before passing to
      correspondence and physical scoring.
```

### 2.5 Build the formal frontier map

```
What: Machine-readable map of mathematical space with three zones:
      Established (proven + experimentally confirmed)
      Uncertain (competing theories, limited data)
      Breakdown (known infinities, singularities)
Why:  Guides exploration. Reward weighting pulls model toward
      breakdown zone. Without this, model has no compass.
Files: New: src/correspondence/frontier.py
       New: configs/frontier_map.yaml
Data:   Encode Penrose-Hawking singularity conditions,
        Standard Model gauge group, Einstein field equations,
        known QFT divergences as formal zone boundaries.
```

### 2.6 Implement correspondence checks

```
What: Formal verification that candidate structures reduce to GR
      at large scales and QFT at small scales.
Why:  Pressure 2 of the three-pressure training hierarchy.
      Ensures proposals don't contradict known physics.
      "They act like the banks of a river — they do not tell the
      water where to go but massively constrain the paths."
Files: src/correspondence/limits.py (stub exists, expand)
       New: src/correspondence/gr_limit.py
       New: src/correspondence/qft_limit.py
How:  Encoded as formal Lean 4 theorems. Candidate structures
      must formally satisfy these theorems. Automated proof
      checking verifies compliance.
```

### 2.7 Encode known failure coordinates

```
What: Formal encoding of exact conditions where current theories
      produce infinities. Every candidate evaluated at these points.
Why:  "Remaining consistent and finite where current theories
      diverge contributes positively to reward. This directly
      incentivizes the system to solve the problems rather than
      reproduce existing failures."
Files: New: src/correspondence/failure_points.py
       New: configs/failure_coordinates.yaml
Points: Planck scale (E ~ 10^19 GeV), black hole singularities,
        Big Bang t=0, non-renormalizable QFT divergences
```

### 2.8 Scale to 3B parameters

```
What: Train the GNN+MCTS explorer at 3B parameter scale
Why:  AlphaProof precedent. 1.5B Qwen was Phase 1 placeholder.
      Phase 2 uses custom architecture, not off-the-shelf LLM.
Hardware: 1× A100 80GB for 3B params, multi-GPU for data parallel
Files: configs/model_config.yaml (base_model.name → custom GNN)
       src/explorer/gnn_encoder.py (scale up hidden dims, layers)
```

**Phase 2 deliverable:** GNN+MCTS explorer that generates novel mathematical structures guided by the frontier map, verified against correspondence requirements, and evaluated at known failure coordinates.

---

## Phase 3 — Physical Grounding

**Goal:** Connect the explorer to real experimental data. Candidate structures are scored on their ability to predict physical measurements, not just mathematical consistency.

**Success criterion:** System assigns higher scores to known physical theories (GR, QFT, Standard Model) than to mathematically consistent but physically wrong structures.

### 3.1 Implement Layer 1 — physical encoding pipelines

```
What: Domain-specific preprocessing for each measurement modality.
      Each pipeline converts raw instrument data → physically
      meaningful common format with uncertainty quantification.
Why:  "Naive approaches fail here. You cannot concatenate these
      into a single array and feed them to a model."
Files: New: src/data/physical/
       New: src/data/physical/metadata.py (Pydantic schema)
       New: src/data/physical/gravitational_wave.py
       New: src/data/physical/spectroscopic.py
       New: src/data/physical/particle_collision.py
       New: src/data/physical/cosmological.py
       New: src/data/physical/thermodynamic.py
Deps:  Need domain physicist for each pipeline validation
```

### 3.2 Implement Layer 2 — mathematical object extraction

```
What: From Layer 1 outputs, extract formal mathematical objects:
      symmetry groups, conservation laws, scaling relations,
      anomaly residuals.
Why:  "The model sees only Layer 2 outputs — never raw experimental
      data directly."
Files: New: src/data/math_objects/
       New: src/data/math_objects/symmetry_extractor.py
       New: src/data/math_objects/conservation.py
       New: src/data/math_objects/scaling.py
       New: src/data/math_objects/residuals.py
```

### 3.3 Source physical measurement data

```
What: Acquire and organize raw experimental datasets
Sources:
  - LIGO/Virgo O3 strain data (HDF5, public, ~10 TB)
  - LHC CMS/ATLAS Open Data (ROOT, public, ~100 TB)
  - Planck CMB maps (FITS, public, ~500 GB)
  - SDSS spectroscopic survey (FITS/CSV, public, ~50 TB)
  - NANOGrav pulsar timing (public, ~10 GB)
Why:  "These are numerical arrays — time series, frequency spectra,
      spatial tensors. Not descriptions of experiments. Not papers.
      The numbers themselves."
Initial: Start with LIGO (smallest, cleanest) and one cosmological
         dataset (CMB). Add domains incrementally.
```

### 3.4 Build the Physical Prediction Scorer

```
What: Multimodal transformer (10-30B params) with domain-specific
      encoders. Initialized from existing scientific model.
Why:  Largest component. Must learn mapping from mathematical
      structure to physical prediction across all modalities.
Files: New: src/scorer/
       New: src/scorer/encoders.py (time series, spatial, spectral,
            discrete event, thermo/chemical)
       New: src/scorer/transformer.py (shared representation space)
       New: src/scorer/training.py (fine-tuning on prediction task)
Init:  Start from a model pretrained on physics literature
       (e.g., Galactica, or fine-tuned scientific LLaMA variant)
```

### 3.5 Implement domain-level holdout

```
What: Replace random 90/10 split with whole-domain holdout.
      Train on GW + particle + spectroscopic. Hold out cosmology.
Why:  "A structure capturing deep physics should predict
      cosmological observations even without training on them."
Files: src/data/dataset.py (modify split logic)
       New: configs/holdout.yaml
Metric: Prediction accuracy on held-out domain vs trained domains
```

### 3.6 Implement simplicity penalty

```
What: Occam's razor as a regularization term. Structures with
      more free parameters, more Lagrangian terms, or larger
      symmetry groups are penalized.
Why:  "A structure fitting all training observations with ten
      thousand free parameters is penalized relative to a
      structure fitting them with five."
Files: src/reward/base.py (modify compute_reward, add complexity term)
       New: src/reward/complexity.py
```

### 3.7 Integrate scorer into training loop

```
What: Wire the physical scorer into the reward pipeline.
      Full three-pressure reward now active:
      consistency + correspondence + compression.
Files: src/training/grpo_trainer.py (modify reward computation)
       src/environment/env.py (add scorer to environment)
       New: scripts/train_full.py (end-to-end training)
```

**Phase 3 deliverable:** Integrated system where mathematical structures are scored against real physical data with proper holdout. Known theories score high; physically wrong structures score low.

---

## Phase 4 — Translation and Human Interface

**Goal:** Human physicists can read, verify, and act on the system's outputs.

**Success criterion:** Translation layer generates experimental proposals for flagged anomalous solutions that a domain physicist rates as coherent and actionable.

### 4.1 Fine-tune Translation Layer

```
What: Fine-tune a 7-70B LLM on formal-to-natural mathematical
      physics translation. Input: Lean 4 formal objects. Output:
      natural language explanation for domain physicists.
Files: New: src/translation/
       New: src/translation/model.py
       New: src/translation/training.py
       New: src/translation/verification.py
Init:  Start from frontier LLM (e.g., fine-tuned LLaMA or Qwen)
```

### 4.2 Automated translation verification

```
What: When translator claims "this structure predicts a spin-1
      boson," verify the formal structure actually has that property.
      Every verifiable claim gets checked. Incorrect claims
      produce negative training signal for the translator.
Files: src/translation/verification.py
```

### 4.3 Implement Dirac mechanism — anomaly flagging

```
What: Solution enumeration + entity matching + anomaly flagging.
      When a structure produces unmatched solution families,
      generate formal characterization and experimental proposals.
Files: New: src/explorer/solution_enumerator.py
       New: src/explorer/entity_matcher.py
       New: src/explorer/anomaly_flagger.py
```

### 4.4 Build physicist interface

```
What: Interface through which human physicists:
      - Review flagged anomalous solutions
      - Read translated structure descriptions
      - Generate experimental proposals
      - Direct system toward domains of interest
Files: New: src/interface/ (API or CLI for physicist interaction)
```

**Phase 4 deliverable:** Physicist-usable system. Anomalous solutions are flagged, translated, and accompanied by experimental proposals.

---

## Phase 5 — Continuous Operation

**Goal:** System runs autonomously, continuously exploring mathematical physics with holdout commitments against future experiments.

**Success criterion:** System makes a prediction committed to before an experiment reports results, and the prediction is testable.

### 5.1 Deploy continuous training infrastructure

```
What: Training loop runs indefinitely. New experimental data
      incorporated as it's released. Models versioned with
      data provenance tracked.
Files: New: infra/ (deployment configs, monitoring, data pipeline automation)
```

### 5.2 Register future-experiment holdout commitments

```
What: Formal, pre-registered commitments to treat specific planned
      experimental results as holdout data. System predicts before
      experiment reports. Prediction compared after.
Why:  "The strongest possible validation. The prediction is made
      and committed to before the result is known, eliminating
      any possibility of post-hoc fitting."
```

### 5.3 Curiosity mechanism refinement

```
What: Based on observed behavior, refine the exploration/exploitation
      balance. Tune curiosity reward schedule.
Why:  "The right calibration is probably dynamic — favoring
      exploration early and exploitation as promising structures
      are identified — but the specific schedule is empirical."
```

### 5.4 Expand preprocessing pipeline coverage

```
What: Add Layer 1 pipelines for additional experimental domains:
      condensed matter, quantum information, biological systems.
Why:  "Each new domain requires domain experts to design and
      validate its pipeline, which is a continuing investment."
```

**Phase 5 deliverable:** Continuously operating system with pre-registered predictions, expanding data coverage, and autonomous exploration.

---

## Dependency Graph

```
Phase 0 ──── DONE
  │
  ▼
Phase 1.1 (build Lake) ──┬──▶ 1.2 (extract data) ──▶ 1.3 (run SFT) ──▶ 1.4 (run GRPO)
                         │                                                    │
                         │                                                    ▼
                         │                                         1.5 (curiosity reward)
                         │                                                    │
                         │                                                    ▼
                         │                                         1.6 (scaling curve)
                         │                                                    │
                         │                                                    ▼
                         │                                         1.7 (write-up)
                         │
                         ▼
Phase 2.1 (dep graph) ──▶ 2.2 (GNN) ──▶ 2.3 (MCTS) ──▶ 2.4 (structure gen)
                                                    │
                    2.5 (frontier map) ──────────────┤
                                                     │
                    2.6 (correspondence) ────────────┤
                                                     │
                    2.7 (failure coords) ────────────┤
                                                     │
                                                     ▼
                                              2.8 (scale 3B)
                                                     │
                                                     ▼
Phase 3.1 (Layer 1) ──▶ 3.2 (Layer 2) ──▶ 3.3 (acquire data)
                                                     │
                    3.4 (scorer) ────────────────────┤
                                                     │
                    3.5 (holdout) ───────────────────┤
                                                     │
                    3.6 (simplicity penalty) ────────┤
                                                     │
                                                     ▼
                                              3.7 (integrate)
                                                     │
                                                     ▼
Phase 4.1 (translator) ──▶ 4.2 (trans verification) ──▶ 4.3 (Dirac) ──▶ 4.4 (interface)
                                                                              │
                                                                              ▼
                                                                      Phase 5 (continuous)
```

---

## Current Blockers (June 2026)

| Blocker | Blocks | Resolution |
|---|---|---|
| Lake project not built (`lake build` never run) | Phase 1.3, 1.4, everything downstream | Run `lake update && lake build` in `proof_checker_env/` |
| Only 4,886 theorems from 4 basic domains | Phase 1.3, 1.4 (need physics domains) | Re-run `prepare_data.py` with expanded domains |
| No training runs executed | All of Phase 1 | Run SFT then GRPO |
| No curiosity reward | Phase 1.4 quality (mode collapse risk) | Implement count-based exploration bonus |
| No scaling curve data | Phase 2 parameter decisions | Train at multiple scales, measure |

---

*Next action: Phase 1.1 — build the Lake project.*
