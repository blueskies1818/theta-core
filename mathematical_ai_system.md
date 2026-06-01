# Autonomous Mathematical Physics AI — System Design

## Overview

This document outlines the architecture, training methodology, and theoretical foundations for an autonomous AI system capable of exploring mathematical physics beyond the boundaries of current human knowledge. The system is inspired by AlphaGo Zero's self-play mechanism — a model that generates its own training signal through interaction with a verifiable environment — applied to formal mathematics and physical observation data rather than a board game.

The core insight is that Go gave AlphaGo Zero three things simultaneously: a perfectly verifiable state, unambiguous rules, and a clear terminal reward. The challenge of building an equivalent system for physics is finding analogs to each of these in mathematical and physical space.

The analogs are: formal proof systems (Lean 4) as the verifiable environment, correspondence with established GR and QFT as the rules, and predictive compression of physical observation data as the terminal reward signal. Together these create a self-sustaining training loop that requires no human judgment at any step — the environment provides all feedback automatically.

The system is a heterogeneous ensemble of three components: a Mathematical Explorer built on a graph neural network with Monte Carlo Tree Search that navigates formal mathematical space; a Physical Prediction Scorer that evaluates candidate structures against raw experimental data across all measurement modalities; and a Translation Layer that converts formal outputs into natural language for human physicists. Each component uses the architecture suited to its specific task rather than one monolithic model attempting everything.

Experimental data enters the system through a two-layer data architecture. Layer 1 converts heterogeneous raw measurements — from gravitational wave detectors, particle colliders, spectrometers, titration experiments, and cosmological surveys — into a physically meaningful common format via domain-specific preprocessing pipelines. Layer 2 extracts formal mathematical objects from this common format: symmetry groups, conservation law status, scaling relations, and anomaly residuals. The model operates entirely on these mathematical objects and never on raw numbers directly. This allows cross-domain reasoning — recognizing that the same mathematical structure underlies phenomena in completely different experimental domains — that would be impossible working at the raw data level.

The primary scientific target is the breakdown zone where general relativity and quantum mechanics are simultaneously necessary and currently incompatible. Current theories produce mathematical infinities at Planck scale, inside black hole singularities, and at Big Bang initial conditions. These failure coordinates are encoded explicitly as exploration targets, with candidate structures rewarded for remaining finite and consistent precisely where current theories break.

Novel discoveries emerge through a mechanism modeled on Dirac's derivation of antimatter: a formally verified structure may produce solution families that do not correspond to any known physical entity. These are flagged rather than discarded, formally characterized, and translated into experimental proposals — asking what conditions would make the predicted entity detectable if it is real.

**Companion document:** *Model Structure and Data Use* covers the internal architecture of each component, the complete data pipeline from raw experimental measurement to reward signal, domain-specific preprocessing pipelines, the metadata schema, and a detailed account of open engineering problems. Both documents should be read together for complete system understanding.

---

## Theoretical Foundations

### Why AlphaGo Zero Is The Right Model

AlphaGo Zero beat the human-trained AlphaGo because it was unconstrained by human strategic intuitions. It found moves that looked wrong by human standards but were deeply correct. The system didn't need humans to tell it what good looked like — the game structure itself provided that signal through outcomes.

The goal of the system described here is to replicate this mechanism in a domain where the game is formal mathematics, the board is the space of possible mathematical structures, and the win condition is a combination of internal consistency and predictive accuracy against physical observation data.

### The Problem With Training On Human Data

Current large language models are trained on human-generated text and inherit:

- Human cognitive biases baked into the structure of knowledge
- Human conceptual categories which may carve nature at the wrong joints
- Human errors, consensus beliefs that are wrong, cultural assumptions
- The limits of what humans have thought to write down

A system trained this way is bounded by human imagination. The AlphaGo Zero result suggests that removing human priors — at the cost of starting from nothing — can produce qualitatively superior results. The challenge is that mathematics is not a game with a clean terminal reward, so the mechanism for removing human bias must be more carefully constructed.

### The Divide Between Relativity and Quantum Mechanics

General relativity and quantum mechanics use incompatible mathematical languages to describe reality.

General relativity is written in differential geometry — smooth, continuous, deterministic. It describes how mass curves spacetime and how that curvature directs motion. Its equations are exact.

Quantum mechanics is written in Hilbert spaces and operator algebras — granular, discrete, fundamentally probabilistic. It describes probability amplitudes rather than definite states.

When the two theories are forced to interact — at Planck scale, inside black hole singularities, at the Big Bang — the combined equations produce infinities that cannot be cancelled. This is not a computational problem. It is a problem of mathematical incompatibility. The two theories do not share a common foundation.

This breakdown is the primary target for the system. The breakdown zones are precisely where current mathematical structures flag their own failure — the equations themselves point to where they stop working. These coordinates serve as the compass for exploration.

### The Dirac Mechanism

Dirac's derivation of antimatter is the model for how this system produces novel discoveries.

He imposed two constraints: the equation must describe an electron, and it must be consistent with special relativity. The simplest equation satisfying both constraints produced four solution families where two were expected. Two corresponded to electrons with spin up and spin down — known and expected. Two had negative energy — seemingly nonsensical.

Rather than discarding the anomalous solutions as mathematical artifacts, Dirac asked: what would have to be true for these solutions to be physical? The answer was a particle identical to the electron but with opposite charge. Antimatter was discovered experimentally four years later.

The system described here replicates this mechanism systematically. When a verified candidate structure produces solution families that do not correspond to any known physical entity, those solutions are flagged rather than discarded, and the system generates formal descriptions of their properties for experimental investigation.

### Predictive Compression As The Win Condition

The entire history of physics is a history of compression. Newton compressed Kepler's laws and Galileo's observations into one framework. Maxwell compressed electricity, magnetism, and optics into four equations. Einstein compressed Newton's gravity and special relativity into spacetime curvature.

Each compression revealed something hidden. Predictive compression — finding mathematical structures that describe known physical observations in fewer and more fundamental terms while retaining full predictive accuracy — is measurable without human judgment and has historically been the mechanism by which hidden physical entities become visible.

This is the system's primary reward signal at the physical grounding stage. A structure that compresses more observations into simpler terms scores higher than one that fits the same data with more complexity. Occam's razor formalized as a regularization term.

---

## System Architecture

The system is a heterogeneous ensemble of three components with distinct architectures, connected by well-defined interfaces. No single monolithic model handles all tasks. Each component uses the architecture suited to its specific job.

### Component 1 — The Mathematical Explorer

**Job:** Navigate formal mathematical space, propose candidate structures, generate proof attempts.

**Architecture:** Graph neural network combined with a tree search algorithm. Not a standard transformer. Formal mathematics has sequential surface structure but its deep structure is a graph — theorems depend on other theorems, structures relate to other structures in a web of logical dependency that is not sequential. The model navigates this graph, proposing new nodes and edges. The proof checker validates each proposed addition.

This is structurally more like AlphaGo Zero navigating a tree of game states than like a language model processing token sequences.

**Parameter scale:** 1–7 billion parameters. AlphaProof — DeepMind's formal mathematics system that solved four of six IMO problems — operates at 3 billion parameters. Formal mathematics has a more compact and regular pattern space than natural language. The explorer's job is learned heuristic search, not knowledge storage, which is substantially less parameter-hungry. The external proof checker handles all verification, further reducing the burden on the model itself.

**Interface out:** Formal mathematical structures and proof attempts in Lean 4 syntax, passed to the proof checker for verification and to the physical scorer for prediction evaluation.

### Component 2 — The Physical Prediction Scorer

**Job:** Evaluate candidate structures against raw physical observation data. Produce a numerical score reflecting how accurately the structure predicts measurements across multiple experimental domains.

**Architecture:** Large transformer with modality-specific input encoders. Different physical data types — time series gravitational wave signals, frequency domain spectroscopic data, spatial cosmological maps — require different encoding strategies before entering a common representation space. The scorer learns to map formal mathematical structures to predictions across all modalities simultaneously.

**Parameter scale:** 10–30 billion parameters. This is the largest component because it is doing genuine multimodal representation learning with no external checker to offload verification onto. It must learn the mapping from mathematical structure to physical prediction entirely in its weights. In practice this component is initialized from an existing pretrained scientific model and fine-tuned rather than trained from scratch, reducing cost substantially.

**Interface in:** Verified mathematical structures from the proof checker output.
**Interface out:** Numerical prediction scores per domain, aggregate reward contribution.

### Component 3 — The Translation Layer

**Job:** Convert formal mathematical structures and proof outputs into natural language descriptions that human physicists can read and act on. Generate experimental proposals for flagged anomalous solutions.

**Architecture:** Standard transformer, fine-tuned from an existing large language model. Translation from formal mathematics to natural language is a well-defined task that existing models handle reasonably well. Training from scratch is unnecessary.

**Parameter scale:** 7–70 billion parameters, initialized from an existing model.

**Verification of translation:** The translation layer's outputs are verifiable. When the translator claims a structure implies a certain property, that claim can be checked formally — the proof checker can verify whether the formal structure actually has that property. This enables automated detection of translation errors and provides a training signal for improving the translation layer independently.

**Limitation:** For solutions radically unlike anything in existing physics — entities that do not interact via known forces, structures requiring genuinely new conceptual vocabulary — the translation layer will fail to produce meaningful experimental proposals. This failure is itself informative: it signals that the solution is either deeply novel or unphysical, and that human physicist engagement is required.

---

## Formal Proof System

### What Formal Proof Looks Like

The system uses Lean 4 as its formal language. Lean 4 is a proof assistant — a programming language in which mathematical statements and their proofs can be written in a form that a deterministic program can verify completely. There is no ambiguity, no interpretation, no probability. A proof either type-checks or it does not.

A simple example of what a physical relation looks like in Lean 4:

```lean
theorem energy_momentum_relation 
  (E p m c : ℝ) (hc : c > 0) : 
  E ^ 2 = (p * c) ^ 2 + (m * c ^ 2) ^ 2 := by
  ring
```

This is Einstein's energy-momentum relation encoded as a formal object. The proof checker reads this and either accepts or rejects it. The checker is deterministic software — a sophisticated type checker — and its accept/reject output is the primary training signal for the mathematical explorer.

### Mathlib As Foundation

Mathlib is the main formally verified mathematical library for Lean 4. It contains thousands of theorems covering differential geometry, topology, linear algebra, and measure theory. Physicists and mathematicians are actively formalizing GR and QFT results into it.

Mathlib serves two roles in the system:

First as pretraining data. Every theorem in the library is a correct input-output pair. The model trains on these before attempting to generate anything new, learning what valid mathematical reasoning looks like across a broad range of structures.

Second as hard constraints baked into the verification environment. When the model generates a candidate structure, part of the verification step checks that the structure is formally consistent with the established library. The model cannot contradict a proven theorem. Established physics acts like the axioms of the game — the model learns very quickly that proposals violating them are rejected immediately, and learns to build from them rather than against them.

### Internal Representation vs Output Language

The model does not need to understand Lean the way a human programmer does. It learns to produce strings that the checker accepts, and the checker's accept/reject signal shapes its internal representations toward whatever works. The internal geometry of mathematical space that the model develops in its weights need not resemble Lean at all — just as a chess engine's internal board representation looks nothing like a visual chessboard. The formal language is the interface between the model and the verifier, not the model's internal language of thought.

---

## Training Methodology

### The Three Pressures

Rather than directing the system toward specific targets, training imposes a nested hierarchy of pressures that together create a gradient toward discovery without constraining where that discovery occurs.

**Pressure 1 — Internal consistency.** Provided automatically by the proof checker. Any structure the model proposes either holds together formally or it does not. This filter alone eliminates the vast majority of possible mathematical objects as candidates. It is not a direction but a constraint, and it is extraordinarily powerful.

**Pressure 2 — Correspondence at known limits.** Whatever structure the model builds, it must reduce to general relativity when quantum effects are negligible, and must reduce to quantum field theory when gravity is negligible. These are hard mathematical requirements encoded as formal theorems in the verification environment. They act like the banks of a river — they do not tell the water where to go but they massively constrain the space of possible paths. The overlap space of structures consistent with GR at large scales AND consistent with QFT at small scales is extremely narrow.

**Pressure 3 — Predictive compression.** The reward contribution from the physical scorer rewards finding mathematical structures that describe known physical observations in fewer and more fundamental terms while retaining full predictive accuracy. This is the primary compass — measurable without human judgment, historically reliable as a guide toward deeper structure.

### The Self-Play Loop

The core training loop runs continuously and requires no human intervention at any step:

1. The mathematical explorer proposes a candidate structure or proof step in Lean 4
2. The proof checker evaluates internal consistency — deterministic, immediate, binary output
3. If consistent, the structure is checked against correspondence requirements — also formal verification
4. If correspondence is satisfied, the physical scorer evaluates predictive accuracy against observation data — numerical calculation
5. The combined reward signal updates the explorer's parameters
6. Return to step 1

This is the genuine AlphaGo Zero analog. The environment — formal mathematics plus physical observation data — provides all feedback automatically. No human labels, no human judgment at any step of the loop.

### Encoding Known Failures

The singularities and divergences in current physics are not vague — they occur at precisely, mathematically definable conditions.

In quantum field theory, specific integrals that diverge to infinity are known exactly and can be written formally. In general relativity, the Penrose-Hawking singularity theorems formally establish conditions under which the equations must break down. These are mathematical objects, not descriptions.

These failures are encoded as explicit test conditions in the verification environment — Planck scale interactions, black hole interior conditions, Big Bang initial conditions. Every candidate structure is evaluated at these conditions as part of its scoring. Remaining consistent and finite where current theories diverge contributes positively to reward. This directly incentivizes the system to solve the problems rather than reproduce the existing failures, and the failure coordinates serve as the primary compass for where to explore.

### Candidate Structures

A candidate structure is a formal mathematical object proposed by the explorer as a potential description of physical reality. Concretely this might be:

- A new metric tensor with additional terms that modify GR at small scales
- A new symmetry group containing the Standard Model symmetry groups as subgroups
- A modified action functional that remains finite at Planck scale
- A new type of connection on a fiber bundle that behaves like spacetime curvature at large scales and like a quantum field at small scales

The model generates these by combining and modifying known mathematical objects from the Mathlib library in ways that haven't been tried — analogous to AlphaGo Zero generating novel board positions by trying moves human players hadn't considered. The space of possible modifications is enormous, which is why the reward signal is necessary to navigate it. Most candidates fail immediately at the consistency check. A fraction survive consistency. A smaller fraction survive correspondence. A tiny fraction score well on prediction. Those form the basis for the next generation of proposals.

### Solution Enumeration and Anomaly Flagging

When a candidate structure passes all verification steps, an automated solution enumeration step finds all mathematical objects that satisfy the structure's equations. For each solution family, a matching step checks whether there is a known physical entity in the formal catalog with the same formal properties — same mass dimension, charge, spin, interaction terms, symmetry properties.

Unmatched solution families are flagged automatically. The flag includes the complete formal mathematical description of the unmatched solution, its predicted behavior under known physical conditions, and what formally distinguishes it from all known entities. This output goes to the translation layer for conversion into natural language experimental proposals, and to human physicists for experimental imagination work — asking what experimental conditions would make this entity detectable if it is real.

---

## Training Data

### Formal Mathematics Corpus

The formal mathematics training corpus is effectively unbounded because the model generates its own training examples through self-play. Every accepted proof is a valid positive training example. Every rejected proof attempt is a valid negative example. This is not limited by how much data humans have produced — it is limited only by compute. Overfitting in the traditional sense is substantially mitigated because the training distribution is dynamic and generated by the model's own current behavior.

The initial corpus — used for pretraining before self-play begins — consists of the complete Mathlib library and any additional formally verified physics results encoded in Lean 4. Every example is guaranteed correct by construction.

### Physical Observation Data

Raw physical measurement data with no human interpretation layer:

- Gravitational wave detector time series from LIGO and Virgo
- Spectroscopic measurement records across astronomical surveys
- Cosmic microwave background maps from Planck and WMAP
- Particle collision event records from LHC experiments
- Pulsar timing arrays
- Neutrino detection records

These are numerical arrays — time series, frequency spectra, spatial tensors. Not descriptions of experiments. Not papers about what the experiments showed. The numbers themselves.

This data enters training through the physical prediction scorer as a scoring function, not as direct training signal. The model generates structures; the structures make predictions; the predictions are compared to measurements; the comparison produces a score. The model never trains directly to memorize observations — it trains to find structures that predict observations, which is a fundamentally different optimization target.

Cycling through different datasets during training — gravitational wave data in some iterations, spectroscopic in others, cosmological in others — prevents overfitting to any single domain and pushes toward structures that describe deep physics rather than domain-specific patterns.

### Holdout Strategy

Standard random sample holdout is insufficient for physical data because different measurements of the same domain are not independent samples. A model can learn GR from two thirds of gravitational wave measurements and trivially interpolate to the remaining third without having learned anything new.

The correct holdout strategy is domain-level holdout — holding out entire experimental domains rather than random samples within domains. Train on gravitational waves, spectroscopy, and particle collisions. Hold out cosmological survey data entirely. A structure capturing deep physics should predict cosmological observations even without training on them.

The most rigorous validation is against experiments that do not yet exist — future runs at planned facilities, next-generation detector results. Formal commitments to use specific planned experiments as validation sets, made before training begins, provide the strongest possible test that the system is not merely compressing existing knowledge.

---

## Hardware Architecture

### Memory Requirements

Training memory requirement is approximately 16 bytes per parameter in standard bf16 mixed precision training:

- Model weights: 2 bytes per parameter
- Gradients: 2 bytes per parameter
- Optimizer states (Adam): 8 bytes per parameter
- Activations and working memory: approximately 4 bytes per parameter

On A100 80GB GPUs — current standard research hardware:

| Model Size | Memory Required | Minimum GPUs |
|---|---|---|
| 300M | ~5GB | 1 GPU |
| 1B | ~16GB | 1 GPU |
| 3B | ~48GB | 1 GPU |
| 7B | ~112GB | 2 GPUs |
| 30B | ~480GB | 6–8 GPUs |

The mathematical explorer at 1–3B parameters fits on a single A100.

### The Real Bottleneck — Proof Checker Parallelism

The dominant computational bottleneck is not GPU throughput but proof checker throughput. Lean 4 proof checking runs on CPU. A complex proof check takes between 0.01 and several seconds. Millions of checks per training run require massive CPU parallelism regardless of GPU count.

The practical hardware configuration is therefore a mixed CPU-GPU system:

- GPUs for model inference and parameter updates
- Large CPU clusters for parallel proof checker instances running simultaneously
- High-speed interconnect between CPU and GPU nodes

This changes the infrastructure design significantly from a standard GPU cluster. The GPU count is modest; the CPU core count is large.

### Staged Hardware Requirements

**Stage 1 — Proof of concept**
Validate that the self-play loop functions. Demonstrate the model finds proofs it did not know before. Measure the empirical scaling curve.
- 1–2 A100 80GB GPUs
- 64–128 CPU cores for parallel proof checking
- 500GB–1TB RAM

**Stage 2 — First serious training run**
Train the mathematical explorer to meaningful capability on physical mathematics.
- 8 A100 80GB GPUs (one DGX A100 server)
- 512 CPU cores across several nodes
- High-speed interconnect

**Stage 3 — Full system integration**
All three components trained in coordination.
- 32–64 GPUs
- Thousands of CPU cores
- Distributed multi-node cluster

H100 GPUs — current generation — are approximately 3x faster than A100s for this workload. The same training run requires roughly one third the GPU count on H100s, with similar total cost due to higher per-unit pricing.

### Test-Time Compute

Parameter count is less important than compute allocated to search at inference time. AlphaProof's 3B parameters run many forward passes — searching, backtracking, trying different proof paths — before committing to an output. The same 3B parameters doing thousands of inference steps is functionally more powerful than a larger model doing one pass.

This is the test-time compute scaling axis. A 3B model searching for ten thousand steps before proposing a candidate structure explores more of mathematical space than a 100B model proposing one candidate immediately. The system should be designed to keep parameter counts moderate while allocating substantial compute to the search process — cheaper to train, easier to iterate, structurally better suited to exploration.

---

## Overfitting Mitigation

### Simplicity Penalty

A structure fitting all training observations with ten thousand free parameters is penalized relative to a structure fitting them with five. Occam's razor is formalized as a regularization term in the reward function. This is not an arbitrary design choice — it reflects the actual historical structure of physics, where the most powerful theories have been extraordinarily compact relative to what they describe.

### Dynamic Training Distribution

Self-play generates a training distribution that evolves with the model's current capabilities. The model cannot overfit to a fixed dataset because the dataset is not fixed. This is a structural property of the self-play approach and one of its primary advantages over supervised learning.

### Curiosity Reward

Without an explicit incentive toward novelty, the model risks converging on one region of mathematical space and optimizing deeply within it rather than exploring genuinely new territory. AlphaGo Zero avoided this naturally because the game forced diversity — different opponents, different board states. Mathematical exploration has no equivalent natural forcing mechanism.

A curiosity reward — explicitly incentivizing exploration of regions of mathematical space far from previously visited regions, even at some cost to immediate reward — is necessary to maintain exploration breadth. This is an active research area in reinforcement learning. Current best approaches include count-based exploration bonuses and learned novelty estimators that measure distance from the frontier of known space.

---

## Interpretability and Human Role

### What Humans Do Not Need To Do

Human physicists do not need to interpret the system's internal reasoning process. The internal steps by which the explorer arrives at a candidate structure do not need to be human-readable. AlphaGo Zero's internal representations of board positions are not meaningful to human Go players. This is acceptable — you do not need to understand how the system thinks to use what it produces.

Human physicists do not need to verify formal proofs. The proof checker has already done this. A valid proof in Lean 4 is a valid proof. The checker's deterministic output is the verification.

### What Humans Do Need To Do

Human physicists must perform experimental imagination for flagged anomalous solutions. When the system produces a formally verified structure with unmatched solution families, a physicist must ask: what experiment would produce something with these properties if it exists? What detector, what energy scale, what signature, what would distinguish it from background?

This is different in character from current theoretical physics work. The physicist is not solving equations — the system has done that. They are designing experiments to test whether a formally real mathematical entity is physically real. This requires deep domain knowledge but is a qualitatively different and in some ways more creative task.

Human physicists must also direct the system toward domains of interest. The system cannot decide what matters — which deep truths to pursue is a human values question, not a mathematical one. The system is an exploration partner, not an autonomous agent. Humans point it at the frontier; the system follows the mathematical thread further than any human could.

### The Interpretability Limit

A system like this, at sufficient capability, will produce mathematical structures that are formally verified and physically predictive but potentially completely uninterpretable in terms of existing conceptual frameworks. Physics has encountered this before — quantum mechanics works perfectly but nobody agrees on what it means. The difference is that humans built quantum mechanics and retain some conceptual grip on it. A system building mathematics autonomously could produce results where that grip is entirely absent.

Whether results that are verifiably correct and experimentally testable but conceptually opaque constitute genuine scientific understanding is a philosophical question the physics community would have to work through. The pressure to accept such results, if their predictions are experimentally confirmed, would be substantial.

---

## Development Roadmap

### Phase 1 — Validate the loop

Build the smallest possible version of the self-play system. 100–300M parameter explorer. Simplified formal domain — a subset of differential geometry relevant to general relativity. Simple reward signal. Run until the model demonstrably finds proofs it did not know before the training run. Measure the empirical scaling curve. This phase validates the architecture and provides data for planning subsequent phases.

### Phase 2 — Scale the explorer

Using the empirical scaling curve from Phase 1, scale the mathematical explorer to the parameter count where the loss curve bends. Expand the formal domain to cover the full GR and QFT Mathlib libraries. Introduce the known failure conditions as explicit test points in the verification environment. Run until the explorer is finding genuinely novel structures in the vicinity of the GR-QFT interface.

### Phase 3 — Integrate physical grounding

Build and validate the physical prediction scorer in isolation. Establish that it correctly scores known physical theories highly and produces meaningful differentiation between candidate structures. Connect the scorer to the explorer's verification pipeline. Introduce the full physical observation corpus with domain-level holdout.

### Phase 4 — Translation layer and human interface

Fine-tune the translation layer on formal-to-natural-language mathematical physics translation. Establish the formal verification pipeline for translation correctness. Build the interface through which human physicists receive flagged solutions and generate experimental proposals.

### Phase 5 — Open-ended operation

Run the full integrated system continuously. Maintain the holdout commitment against planned future experiments. Refine reward signal design based on observed behavior. Expand physical observation data as new experiments report. Iterate on the curiosity reward mechanism to maintain exploration breadth.

---

## Key References and Anchors

- **AlphaGo Zero** — Silver et al. 2017. Self-play without human priors exceeds human-trained performance. The foundational precedent.
- **AlphaProof** — DeepMind 2024. 3B parameter model solving IMO problems via Lean 4 integration and reinforcement learning. Direct precedent for the mathematical explorer component.
- **FunSearch** — DeepMind 2023. LLM generating programs evaluated against mathematical objectives, discovering new mathematical results. Precedent for open-ended mathematical exploration with automated reward.
- **Chinchilla scaling laws** — Hoffmann et al. 2022. Optimal parameter count scales with training tokens. Basis for parameter estimation methodology.
- **Mathlib** — Community formal mathematics library for Lean 4. Foundation of the formal training corpus.
- **Penrose-Hawking singularity theorems** — Formally proven results establishing where GR breaks down. Encoded as exploration targets in the verification environment.
- **Dirac equation** — Historical model for the anomalous solution flagging mechanism. The template for how formally real mathematical entities become physically real discoveries.
