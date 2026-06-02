# Running List — Better Implementation Ideas

*Issues and ideas identified during plan-to-code alignment review. To be addressed after the README
rewrite and immediate code fixes are complete.*

---

## Architecture

### 1. GNN + MCTS for the Mathematical Explorer
The plan calls for a Graph Neural Network with Monte Carlo Tree Search as the core architecture
of Component 1 (the Mathematical Explorer). The current implementation uses a standard causal
transformer (Qwen2.5-1.5B-Instruct). This is a reasonable Phase 1 placeholder — the plan
explicitly says "100–300M parameter explorer" and "simplified formal domain" for Phase 1
validation — but when scaling to Phase 2 the architecture needs to switch.

**Plan reference:** `mathematical_ai_system.md` § Component 1, `model_structure_and_data.md` § 1.2

### 2. Curiosity / Exploration Reward
The plan calls out a curiosity reward as **critical** to prevent mode collapse:
> "Without an explicit incentive toward novelty, the model risks converging on one region of
> mathematical space and optimizing deeply within it rather than exploring genuinely new territory."

The current reward system has only binary valid/invalid + optional length bonus. No exploration
incentive of any kind. This should be added even in Phase 1 — it's called out in the plan as
necessary and is not Phase 3+ work.

**Plan reference:** `mathematical_ai_system.md` § Curiosity Reward, § Overfitting Mitigation

### 3. Formal Frontier Map (Three-Zone Exploration)
The plan describes the explorer maintaining a formal frontier map with three zones:
established, uncertain, and breakdown — each with differently weighted rewards. This is
entirely absent. For Phase 2+.

**Plan reference:** `model_structure_and_data.md` § 1.2 — The Exploration Frontier

### 4. Solution Enumeration and Anomaly Flagging (The Dirac Mechanism)
When a candidate structure produces solution families with no known physical counterpart,
the plan describes an automated flagging and characterization pipeline. Not present.
For Phase 3+.

**Plan reference:** `mathematical_ai_system.md` § Solution Enumeration and Anomaly Flagging,
`model_structure_and_data.md` Step 11

---

## Data Pipeline

### 5. Layer 1 / Layer 2 Data Architecture
The plan describes a two-layer architecture: Layer 1 converts raw measurements to physically
meaningful format, Layer 2 extracts formal mathematical objects. The current code has only
a simple theorem extractor. This is appropriate for Phase 1 (formal mathematics only), but
the data pipeline design should be kept aligned with the plan as physical data is added.

**Plan reference:** `model_structure_and_data.md` § 2.1–2.5

### 6. Metadata Schema
The plan's comprehensive JSON metadata schema for experiments is not implemented at all.
Should be added as a data model (Pydantic) even as a stub, so the data pipeline has a clear
target shape.

**Plan reference:** `model_structure_and_data.md` § 2.2 — The Metadata Schema

### 7. Domain-Level Holdout Strategy
The plan explicitly rejects random-split holdout for physical data and describes domain-level
and future-experiment holdout. The current code uses simple 90/10 random split. Appropriate
for Phase 1 but needs redesign before physical data is introduced.

**Plan reference:** `mathematical_ai_system.md` § Holdout Strategy,
`model_structure_and_data.md` § 2.6

---

## Reward Design

### 8. Multi-Pressure Reward (Consistency + Correspondence + Compression)
The plan describes three nested pressures that together form the reward gradient:
internal consistency, correspondence at known limits, and predictive compression.
Phase 1 implements only the first (binary proof checker output). The correspondence
check (reduction to GR and QFT at appropriate limits) and compression scoring should
be planned for Phase 2+.

**Plan reference:** `mathematical_ai_system.md` § The Three Pressures

### 9. Known Failure Coordinates as Test Conditions
The plan describes encoding specific mathematically precise failure points
(Planck scale, black hole interiors) as explicit evaluation conditions. Not present.
Valuable even as evaluation anchors in Phase 2.

**Plan reference:** `mathematical_ai_system.md` § Encoding Known Failures

### 10. Simplicity Penalty (Occam's Razor Formalized)
The plan describes a regularization term penalizing structures with more free parameters.
The current length bonus is a weak proxy. A more structural complexity measure (counting
free parameters, Lagrangian terms, symmetry group size) should be designed.

**Plan reference:** `mathematical_ai_system.md` § Simplicity Penalty

---

## Code Quality

### 11. Config Duplication — Two RewardConfig Classes
`src/reward/config.py` defines a `dataclass`-based `RewardConfig`.
`src/utils/config.py` defines a `Pydantic`-based `RewardConfig` with slightly different
fields. The code imports from `src/reward.config` everywhere, but the YAML loading in
`src/utils/config.py` loads into the Pydantic version. These should be consolidated.
**See fix applied — remaining work:** ensure YAML config loading uses the same class.

### 12. Lean Project Integration
`lean_interface.py` defaults to `lean --stdin` mode (no Mathlib4). The `proof_checker_env/`
directory contains a proper Lake project with Mathlib4 imports. The interface should
default to using this project when available rather than requiring explicit `project_dir`.

### 13. GRPO Trainer Data Interface
`train_grpo.py` passes a raw `list[dict]` to `GRPOTrainer.train()` instead of the
`ProofGenerationDataset` class. The trainer calls `_sample_theorems` which does manual
index-based sampling from the list. This works but bypasses the dataset abstraction.
The trainer should accept the dataset object and use its interface.

### 14. Pyproject.toml Package Discovery
`pyproject.toml` uses `packages = ["src"]` which may not correctly discover subpackages.
Should use `setuptools` automatic discovery or explicitly list all packages.

---

## Phase 2–5 Planning

### 15. Physical Prediction Scorer (Component 2)
Not implemented. The plan calls for a 10–30B parameter multimodal transformer with
domain-specific encoders, initialized from a pretrained scientific model. This is
the largest component. Requires Phase 3 planning.

### 16. Translation Layer (Component 3)
Not implemented. The plan calls for a 7–70B parameter LLM fine-tuned for formal-to-natural
mathematical physics translation with automated verification. Phase 4 work.

### 17. Regime Map
The two-dimensional energy/gravity regime map described in the plan is not implemented.
Needed when physical data is introduced.

### 18. Test-Time Compute Scaling
The plan emphasizes allocating substantial compute to search at inference time:
> "A 3B model searching for ten thousand steps before proposing a candidate structure
> explores more of mathematical space than a 100B model proposing one candidate immediately."

The current generation uses standard beam-free sampling with fixed `num_return_sequences`.
No MCTS-style iterative search at inference. This is the whole point of the MCTS
architecture — the search budget should be a first-class hyperparameter.

---

*Last updated: 2026-06-01*
*Review with: mathematical_ai_system.md, model_structure_and_data.md*
