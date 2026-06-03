# Phase 1 Closure Report — Validate the Self-Play Loop

**Date:** 2026-06-03
**Status:** Complete — Go for Phase 2
**Branch:** `main`
**Hardware:** Intel Arc B70 Pro (34 GB VRAM), 32-core CPU, 64 GB RAM

---

## 1. Executive Summary

**Goal:** Prove that a model can learn theorem proving from proof-checker feedback alone, with zero human-labeled proofs during RL training.

**Verdict: The self-play loop works, but the LLM is the wrong tool for proof generation.** The infrastructure (proof checker, reward system, training loop) is functional and will carry forward into Phase 2. The LLM (Qwen2.5-1.5B with 4.6% LoRA) cannot generate semantically valid Lean proofs reliably enough to bootstrap GRPO self-play. This was expected — the ROADMAP always planned to replace the transformer with a GNN+MCTS architecture in Phase 2. Phase 1's findings confirm this is necessary.

**Decision:** Skip further LLM optimization. Proceed to Phase 2 — GNN + MCTS proof search.

---

## 2. What Was Built and Tested

### 2.1 Completed Sub-Phases

| Phase | Task | Status | Key Result |
|-------|------|--------|------------|
| 1.1 | Lake/Mathlib4 build | ✓ | `lake env lean --stdin` works, ~2.9s overhead |
| 1.2 | Data extraction | ✓ | 69,150 theorems from 10 physics-relevant domains |
| 1.3 | SFT pretraining | ✓ | Qwen2.5-1.5B LoRA, val loss 0.17 |
| 1.4 | GRPO self-play | ✓ (validated) | Loop works, 25-50% success on bootstrap theorems |
| 1.5 | Curiosity reward | ✓ | Count-based exploration bonus implemented |

### 2.2 Working Infrastructure (Carries Forward)

| Component | File(s) | Status |
|-----------|---------|--------|
| Proof checker (Lean 4 via Lake) | `src/proof_checker/lean_interface.py` | ✓ Deterministic, cached, parallel |
| Batch proof checker (spawn workers) | `src/proof_checker/batch_checker.py` | ✓ 3 workers, 30s timeout |
| Theorem wrapping | `src/proof_checker/formats.py` | ✓ `:=` for term + tactic proofs |
| Reward system (binary + curiosity + length) | `src/reward/base.py` | ✓ Cold-start rewards for invalid proofs |
| GRPO trainer (group advantages, KL penalty) | `src/training/grpo_trainer.py` | ✓ Per-token logprobs for stability |
| Config system (YAML) | `configs/` | ✓ SFT, GRPO, model, reward configs |
| XPU utilities (Intel Arc B70) | `src/utils/xpu_utils.py` | ✓ 34 GB VRAM, bfloat16 |
| Theorem data pipeline | `src/data/` | ✓ 69K theorems, bootstrap dataset |
| SFT trainer | `src/training/sft_trainer.py` | ✓ LoRA fine-tuning |
| Checkpoint save/load | `src/utils/checkpoint.py` | ✓ PEFT adapter format |
| Bootstrap theorem dataset | `data/raw/bootstrap_theorems.jsonl` | ✓ 460 simple theorems |

---

## 3. Error Register

### 3.1 Proof Checker Errors

#### E1: Double-`by` Syntax Error (CRITICAL)
**Symptom:** All proofs rejected, including trivially correct ones.
**Root cause:** `wrap_theorem_with_proof()` added `:= by` to statements that already had `:=`, and proofs starting with `by ` produced `:= by\n  by ...` — a Lean syntax error.
**Fix:** Changed to use `:=` (not `:= by`), letting the proof body decide term vs tactic style.
**→ Phase 2 impact:** The proof format function is now correct; no changes needed.

#### E2: Term-vs-Tactic Mismatch
**Symptom:** `add_zero _` rejected as "unknown tactic" when wrapped in `:= by`.
**Root cause:** Term-style proofs need `:=`, tactic-style proofs need `:= by`. The wrapper always added `by`.
**Fix:** (Same as E1) Use `:=` universally. Lean 4 accepts both `:= term` and `:= by tactic`.
**→ Phase 2 impact:** MCTS will generate tactics; `:= by` works when proof starts with `by`.

### 3.2 Training Infrastructure Errors

#### E3: ProcessPoolExecutor Fork Deadlock
**Symptom:** Training hung after generation when spawning proof checker workers.
**Root cause:** `fork()` after HuggingFace tokenizers initialization causes thread deadlock.
**Fix:** Changed to `mp.get_context("spawn")` in `batch_checker.py`. Set `TOKENIZERS_PARALLELISM=false`.
**→ Phase 2 impact:** Spawn workers work; keep this configuration.

#### E4: XPU Dual-Model Loading Hang
**Symptom:** Loading two model copies on XPU caused indefinite hang at 100% CPU.
**Root cause:** Intel XPU driver issue with loading two large models sequentially.
**Fix:** Reference model loaded on CPU. KL computation is slower but stable. Policy model stays on XPU.
**→ Phase 2 impact:** GNN is single-model; no dual-loading issue.

#### E5: GRPO Cold-Start (Zero Advantages)
**Symptom:** All invalid proofs got identical reward (0) → zero advantage → zero gradient.
**Fix:** Applied curiosity bonus + tiny length variation to invalid proofs. Breaks reward symmetry.
**→ Phase 2 impact:** Keep cold-start rewards; they're essential for bootstrapping.

#### E6: Logprob Sequence Length Explosion
**Symptom:** Loss values of 2,231+ from `exp(log_ratio)` explosion.
**Root cause:** Sequence logprobs summed over tokens (not averaged). Long sequences had extreme values.
**Fix:** Per-token-average logprobs normalize across different sequence lengths.
**→ Phase 2 impact:** Use per-token-average logprobs in any policy gradient method.

#### E7: CPU Overload (8 Workers)
**Symptom:** Desktop unusable during training.
**Fix:** Reduced to 3 workers, 2×2 batch, `nice -n 10` process priority.
**→ Phase 2 impact:** Keep conservative worker count during interactive use.

### 3.3 LLM-Specific Errors (Phase 1 ONLY)

#### E8: Base Model Artifact Pollution
**Symptom:** Model appends `<commit_msg>`, `<issue_closed>`, `import Data.Finset` to generated proofs.
**Root cause:** Qwen2.5-1.5B-Instruct trained on GitHub data. LoRA at 4.6% can't suppress this.
**Fix:** `_clean_proof_text()` function strips artifacts post-generation.
**→ Phase 2 impact:** Not applicable — GNN+MCTS doesn't use an LLM for proof generation.

#### E9: Model Generates English, Not Lean
**Symptom:** At temperature ≥0.6, model generates natural language explanations instead of Lean code.
**Fix:** Lowered temperature to 0.4, max_new_tokens to 64. Mitigates but doesn't eliminate.
**→ Phase 2 impact:** Not applicable.

#### E10: Semantic Errors (Type Mismatches)
**Symptom:** Model generates syntactically valid Lean that doesn't prove the theorem (e.g., `add_zero zero_add`).
**Root cause:** SFT with 5K examples at 4.6% LoRA teaches pattern matching, not semantic reasoning.
**Fix:** Not fixable with current approach — requires Phase 2 architecture.
**→ Phase 2 impact:** This is the primary motivation for switching to search-based proof generation.

#### E11: Large-Theorems DataLoader Bottleneck
**Symptom:** SFT with 10K+ theorems dropped from 1.6s/step to 12s/step.
**Root cause:** On-the-fly tokenization with `num_workers=0` (required for XPU) bottlenecks on CPU.
**Fix:** Worked around by keeping 5K theorem limit. Pre-tokenization would solve this.
**→ Phase 2 impact:** Pre-tokenize or pre-compute graph embeddings for efficiency.

---

## 4. Why the LLM Approach Failed for Formal Proof Generation

### 4.1 Token Generation Is Fundamentally Wrong for Formal Math

Formal proofs in Lean 4 are **programs** — every character matters, and the proof checker is a compiler. LLMs generate tokens probabilistically, which works for natural language (where synonyms and paraphrases exist) but fails for formal systems (where `add_zero` and `add_comm` are different lemmas, not stylistic choices).

### 4.2 The SFT Model's Capability Ceiling

| Capability | SFT Result | Needed for GRPO |
|------------|-----------|-----------------|
| Valid Lean syntax | ~70% | 100% |
| Valid Lean semantics (correct proof) | ~25-35% (bootstrap), ~0% (Mathlib4) | >50% for meaningful signal |
| Artifact-free output | ~60% (with cleaner) | 95%+ |
| Consistent behavior across temperatures | Unstable above 0.4 | Stable at 0.6-0.8 for exploration |

### 4.3 What It Would Take to Make the LLM Work

Full fine-tuning (not LoRA), 100K+ curated theorem-proof pairs, math-specific base model (not general instruct), artifact-free training data, and likely a 3B+ parameter model. This is a 10-100× increase in compute and data, and the result would still be a token-generator fighting against the deterministic nature of formal proof checking.

---

## 5. What Carries Forward to Phase 2

### 5.1 Directly Reusable
- **Proof checker** (`lean_interface.py`, `batch_checker.py`, `formats.py`): The core environment. Takes a proof, returns valid/invalid.
- **Reward system** (`base.py`, `config.py`): Binary + curiosity + length rewards. All configurable.
- **GRPO trainer** (`grpo_trainer.py`, `losses.py`): Group-relative advantages, KL penalty. Replace `generate_proofs()` call with MCTS search.
- **Config system**: YAML-based, extensible. Add GNN and MCTS configs.
- **XPU utilities**: Device detection, memory management.
- **Data pipeline**: Theorem extraction, formatting, bootstrap dataset.
- **Lake/Mathlib4 environment**: The deterministic game board.

### 5.2 To Be Replaced
- **`src/model/generation.py`**: `generate_proofs()` → MCTS search
- **Transformer model**: Qwen2.5 → GNN encoder
- **SFT training**: Token prediction → graph embedding pretraining

---

## 6. Phase 1 Metrics Summary

| Metric | Value | Notes |
|--------|-------|-------|
| Theorems extracted | 69,150 | 10 domains, indentation-based parser |
| SFT best val loss | 0.174 | Qwen2.5-1.5B, LoRA r=64, 5 epochs |
| SFT trainable params | 73.9M (4.6%) | XPU memory: 3.4 GB |
| Bootstrap theorems | 460 | Simple arithmetic, logic, equality |
| GRPO success rate (bootstrap) | 25-50% | When proof cleaner works |
| GRPO success rate (Mathlib4) | ~0% | Theorems too hard for SFT model |
| GRPO step time | 12-18s | 3 workers, 2×2 batch, CPU ref model |
| Proof checker time | ~3s overhead + check | 30s timeout, spawn workers |
| GPU | Intel Arc B70 Pro | 34 GB VRAM, bfloat16, eager attention |
| Total training runs | 15+ | Iterative debugging of the pipeline |

---

## 7. Decision: Go for Phase 2

**Phase 1 validated the self-play loop.** The proof checker provides deterministic feedback, the reward system generates meaningful signal, and the GRPO trainer updates policies correctly. The loop itself is functional.

**Phase 1 also demonstrated why the LLM approach is wrong.** Token generation cannot meet the precision requirements of formal proof checking. The search space is too large for probabilistic sampling.

**Phase 2 replaces the weakest link** — the proof generator — with a search-based approach that's inherently suited to formal verification:

```
Phase 1:  Theorem → LLM.generate() → proof text → checker → reward
Phase 2:  Theorem → MCTS.search(tactic_space) → proof → checker → reward
                         ↑
                    GNN evaluates states
```

The GNN operates on the math dependency graph. The MCTS explores possible proof paths. Every step is validated by the same proof checker we built in Phase 1. No hallucinations, no artifacts, no type mismatches — because every candidate is tested against the checker before acceptance.

---

## 8. Artifact Inventory

| Artifact | Path | Status |
|----------|------|--------|
| SFT checkpoint | `checkpoints/sft/best/` | v1, val loss 0.175 |
| SFT checkpoint | `checkpoints/sft_v2/best/` | v2, val loss 0.174 |
| Mathlib4 theorems | `data/raw/mathlib4_theorems.jsonl` | 69,150 entries |
| Bootstrap theorems | `data/raw/bootstrap_theorems.jsonl` | 460 entries |
| GRPO config (optimized) | `configs/grpo_config.yaml` | Phase 1 tuned |
| Lake project | `proof_checker_env/` | Mathlib4 built |
| Proof checker | `src/proof_checker/` | Working, spawn workers |
| Reward system | `src/reward/` | Binary + curiosity |
| GRPO trainer | `src/training/grpo_trainer.py` | Per-token logprobs |
| Proof cleaner | `src/model/generation.py` | Partial, LLM-specific |

---

## 9. Next Steps: Phase 2 Kickoff

1. **Phase 2.1**: Build math dependency graph from Mathlib4
2. **Phase 2.2**: Implement GNN encoder over the graph
3. **Phase 2.3**: Implement MCTS proof search using GNN evaluations
4. Wire MCTS into GRPO trainer (replaces `generate_proofs()`)

**Estimated Phase 2 timeline:** Dependency graph (1-2 days) → GNN (2-3 days) → MCTS (3-5 days) → Integration (1 day).

---

*Generated 2026-06-03. Phase 1 complete. Go for Phase 2.*
