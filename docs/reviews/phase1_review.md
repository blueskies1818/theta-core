# Phase 1 Review — Validate the Self-Play Loop

**Date:** 2026-06-02
**Status:** In Progress (1.1 ✓, 1.2 ✓, 1.3 ✓, 1.5 ✓, 1.4 pending, 1.6 pending, 1.7 pending)
**Branch:** `main`
**Hardware:** Intel Arc B70 Pro (34 GB VRAM), 32-core CPU, 64 GB RAM

---

## 1. Goal of Phase 1

Prove the self-play theorem proving loop works. The model must learn to prove theorems from proof-checker feedback alone, with zero human-labeled proofs during RL training.

This is the **AlphaGo Zero analog** ([EXPLAINER.md](../../EXPLAINER.md)):
- AlphaGo Zero learned Go from self-play + binary win/lose signal
- Our system learns theorem proving from self-play + binary proof-valid/invalid signal
- Phase 1 is "9×9 board" scale — small model, narrow domain, validate the mechanism before scaling

**Success criterion:** Proof success rate on held-out theorems increases measurably over GRPO training steps.

---

## 2. What Was Done

### 2.1 Phase 1.1 — Lake/Mathlib4 Build ✓

**Files:** `proof_checker_env/`

Built the Lake-managed Lean 4 project with Mathlib4 dependency. Every non-trivial proof check requires Mathlib4 imports — without this, only trivial proofs (`rfl`, `omega`, `native_decide`) would verify.

- `lake update` → downloads Mathlib4 (~2 GB)
- `lake build` → compiles Mathlib4 (~30-60 min on 32 cores)
- Result: `lake env lean --stdin` works, ~2.9s overhead per check (Mathlib4 loading)

**→ Larger goal link:** The proof checker is the "rules of the game." Just as AlphaGo Zero needed a Go rules engine, GRPO needs a deterministic proof verifier that says yes/no with 100% certainty. This is the environment that provides all training signal during self-play.

---

### 2.2 Phase 1.2 — Data Extraction ✓

**Files:** `src/data/mathlib_extractor.py` (rewritten), `data/raw/mathlib4_theorems.jsonl`

Rewrote extraction from regex-based to indentation-based parsing to capture both `:= by` tactic proofs and `:=` term-style proofs. Key changes:
- Added `_DECL_START` to match `theorem|lemma|example`
- Added `_ASSIGN` regex: `r':=\s*(by)?\s*(.*)'`
- Added `_line_indent()` helper for indentation-based block detection
- Filters trivial proofs (`.rfl`, `rfl`, `trivial`, `sorry`)

**Result:** 69,150 theorems — 14× increase from original 4,886.

| Domain | Count | Physics Relevance |
|--------|-------|-------------------|
| Algebra | 21,624 | Symmetry groups, operator algebras |
| Analysis | 19,220 | Functional analysis, spectral theory, PDEs |
| Topology | 13,379 | Spacetime structure, manifold topology |
| LinearAlgebra | 7,300 | Hilbert spaces, tensor products |
| GroupTheory | 3,302 | Gauge groups, Lorentz group |
| Data | 2,380 | Real/complex/nat numbers, sets |
| Geometry/Manifold | 1,945 | Differential geometry, GR language |

**→ Larger goal link:** These are the mathematical domains where the GR-QFT interface lives. Analysis provides the language of QFT (operator algebras, spectral theory). Geometry/Manifold and Topology provide the language of GR (differential geometry, spacetime structure). Phase 2's structure generator will operate in this combined mathematical space.

---

### 2.3 Phase 1.3 — SFT Pretraining ✓

**Files:** `scripts/train_sft.py`, `src/training/sft_trainer.py`, `configs/sft_config.yaml`

Supervised fine-tuning of Qwen2.5-1.5B-Instruct on 5,000 theorem-proof pairs.

**Configuration:**
- Model: Qwen2.5-1.5B-Instruct with LoRA (r=64, alpha=128)
- Target modules: q_proj, k_proj, v_proj, gate_proj, up_proj, down_proj
- Trainable params: 73.9M / 1,617.6M (4.6%)
- Precision: bfloat16, attention: eager (XPU SDPA compatibility)
- Dataset: 4,500 train / 500 val
- Training: 2 epochs, 2,250 steps, learning rate 2e-5 (cosine decay)

**Results vs ROADMAP targets:**

| Metric | ROADMAP Target | Actual | 
|--------|---------------|--------|
| SFT final loss | ~0.50 | **0.14** (3.6× better) |
| Validation loss | — | **0.17** (best) |
| Epoch 1 avg loss | — | 0.45 |
| Epoch 2 avg loss | — | 0.14 |
| Training time | 2-4 hours | ~60 min |
| Steps/sec | — | ~0.63 (1.6 sec/step) |
| Checkpoints | — | `checkpoints/sft/best`, `checkpoints/sft/final` |
| GPU | any | Intel Arc B70 Pro (XPU) |

**→ Larger goal link:** The model must know Lean 4 syntax before self-play can work. Without SFT, GRPO generates gibberish that never type-checks → zero rewards → no learning signal. SFT teaches the "grammar" so at least some generated proofs pass the checker, providing the initial positive reward needed to bootstrap the self-play loop.

---

### 2.4 Phase 1.5 — Curiosity/Exploration Reward ✓

**Files:** `src/reward/base.py`, `src/reward/config.py`, `configs/reward_config.yaml`, `src/training/grpo_trainer.py`

Implemented count-based exploration bonus to prevent mode collapse during self-play. The ROADMAP flags this as **CRITICAL** — "without an explicit incentive toward novelty, the model risks converging on one region."

**Mechanism:**
```
bonus = curiosity_weight / sqrt(count(proof_signature) + 1)
```
- SHA-256 proof signatures (16 hex chars) with whitespace/comment normalization
- Novel proofs get full bonus (0.05); repeated patterns decay toward zero
- Signature counter prunes to top 50,000 when exceeding 100,000 tracked
- Wired into GRPO training loop: computed before reward, recorded after

**→ Larger goal link:** Mode collapse is a known failure mode of self-play RL. The model finds one proof pattern that works (e.g., `apply rfl`) and generates it for every theorem. The curiosity bonus provides a counter-pressure: exploring novel proof strategies is rewarded, keeping the search diverse enough to discover genuinely new mathematics in Phase 2+.

---

## 3. Error Analysis

### 3.1 Errors Encountered and Resolved

#### E1: PyTorch 2.12.0+xpu Segfault on Intel Arc B70

**Severity:** Blocker
**Symptom:** Immediate segfault on any torch operation when using `xpu` device.
**Root cause:** SYCL library version mismatch. The pip-installed `torch==2.12.0+xpu` linked against Intel oneAPI 2026.0 SYCL libraries, but the system had conflicting library paths from prior Intel driver installations.
**Fix:** Downgraded to PyTorch 2.8.0+xpu. Key: do NOT source `setvars.sh` from Intel oneAPI — the pip-installed SYCL runtime is self-contained.
**Prevention:** Pin PyTorch version in a `requirements.txt` or `environment.yml`. Document the XPU setup explicitly.
**→ Future risk:** Any system update to Intel drivers or oneAPI packages could reintroduce this. Monitor after `apt upgrade`.

---

#### E2: DataLoader Deadlock on XPU with num_workers > 0

**Severity:** Blocker
**Symptom:** Training hung after first batch when `num_workers=2` with `pin_memory=True`.
**Root cause:** PyTorch DataLoader uses `fork()` for multiprocessing. After XPU context initialization in the parent process, forked children inherit an invalid device context, causing deadlock. This is a known limitation of Intel's XPU runtime — CUDA has the same issue but handles it more gracefully.
**Fix:** Changed `num_workers=0` and `pin_memory=False` for XPU device in `src/training/sft_trainer.py` (both `train()` and `evaluate()` methods).
**Impact:** No data loading parallelism. For Phase 1's small dataset (5,000 theorems), this is negligible. For Phase 2+ larger datasets, this will become a bottleneck.
**→ Future risk:** Phase 2+ with larger datasets will need async data loading. Possible solutions: (a) pre-tokenize and cache datasets, (b) use `spawn` start method with careful XPU context management, (c) stream data from a separate process that never touches XPU.

---

#### E3: `bf16` Invalid PyTorch dtype

**Severity:** Minor (quick fix)
**Symptom:** `AttributeError: module 'torch' has no attribute 'bf16'` on model load.
**Root cause:** Config file `configs/model_config.yaml` used `bf16` — PyTorch requires `bfloat16`.
**Fix:** Changed `mixed_precision: "bf16"` → `mixed_precision: "bfloat16"`.
**Prevention:** Add dtype validation in `src/utils/config.py` load function.

---

#### E4: `total_mem` AttributeError on XPU Device Properties

**Severity:** Minor (quick fix)
**Symptom:** Crash in `xpu_utils.py` when printing GPU VRAM info.
**Root cause:** PyTorch 2.8 uses `total_memory` property; our code used the older `total_mem`.
**Fix:** Added fallback: `getattr(props, 'total_memory', getattr(props, 'total_mem', 0))`.
**→ Future risk:** PyTorch XPU API is less stable than CUDA. Property names may change again in future versions. Test `xpu_utils.py` after any PyTorch upgrade.

---

#### E5: Double-LoRA Bug in GRPO Trainer

**Severity:** Major (would crash or produce garbage)
**Symptom:** GRPO `train_grpo.py` called `apply_lora()` on a model already loaded with LoRA adapters from SFT checkpoint.
**Root cause:** Loading from SFT checkpoint with `from_pretrained()` loads LoRA weights as regular weights. Calling `apply_lora()` again wraps them in LoRA layers → double parameterization → 2× parameter count and incorrect forward pass.
**Fix:** Added guard in `scripts/train_grpo.py`: only calls `apply_lora()` when `not args.sft_checkpoint`.
**→ Future risk:** Any future training script that chains checkpoints must be aware of this. Consider adding a `is_lora_model()` check utility.

---

#### E6: SDPA Attention Not Supported on XPU

**Severity:** Minor (config fix)
**Symptom:** Warning about unsupported attention implementation, falling back to eager.
**Root cause:** `configs/model_config.yaml` had `attn_implementation: "sdpa"`. Intel XPU has limited SDPA support.
**Fix:** Changed to `attn_implementation: "eager"`.
**Impact:** Slightly higher memory usage and slower attention computation. Negligible for 1.5B model.
**→ Future risk:** As Intel improves XPU SDPA support, we may want to re-enable it. Test after major oneAPI/driver updates.

---

#### E7: Proof Checker Timeout Too Short

**Severity:** Medium (silently killed valid proofs)
**Symptom:** Some valid proofs returned as failures (timeout) because `lake env lean --stdin` needs ~2.9s just to load Mathlib4.
**Root cause:** Default 10s timeout didn't account for the 3s overhead + actual proof checking time.
**Fix:** Increased to `timeout_seconds: 30.0` in `configs/grpo_config.yaml`.
**Impact:** 30s timeout × 8 workers × 16 proofs/batch = worst case 60s per step. Acceptable for Phase 1.
**→ Future risk:** Longer/more complex proofs in Phase 2+ may need longer timeouts. MCTS proof search will compound this (many proof attempts per theorem). Consider streaming proof checking with early termination.

---

#### E8: SFT Training Output Buffering

**Severity:** Cosmetic
**Symptom:** Training progress output appeared delayed/hung even though training was proceeding.
**Root cause:** Bash task wrapper buffered stdout despite `PYTHONUNBUFFERED=1` and `python -u` flags.
**Fix:** None needed — output arrived in batches, training completed correctly. Verified by checking step timing (consistent 1.6 sec/step throughout).
**→ Future risk:** For long-running Phase 2+ training, consider file-based logging with `tee` or structured log files rather than relying on real-time stdout.

---

### 3.2 Future Error Risks (Not Yet Encountered)

These are risks identified during Phase 1 work that could manifest in remaining Phase 1 or later phases.

#### FR1: GRPO Mode Collapse Despite Curiosity Bonus

**Risk:** Curiosity weight of 0.05 may be insufficient to prevent mode collapse if the proof checker accepts only a small set of proof patterns.
**Mitigation:** Monitor `unique_signatures` count in GRPO logs. If it plateaus early, increase `curiosity_weight` or decrease `temperature`. Consider decaying curiosity over time (high early, low late).
**Affects:** Phase 1.4

#### FR2: KL Divergence Too Restrictive

**Risk:** KL penalty (beta=0.01) may prevent the policy from diverging enough from the SFT checkpoint to discover novel proof strategies. The SFT model may have learned only one way to prove each theorem.
**Mitigation:** Monitor KL divergence in GRPO logs. If it stays near zero and success rate doesn't improve, reduce `kl_beta`. Consider KL annealing (high early to prevent collapse, low later to allow exploration).
**Affects:** Phase 1.4

#### FR3: Proof Checker Is the Bottleneck (CPU-bound)

**Risk:** 8 parallel Lake processes may saturate CPU/memory, slowing training below useful throughput.
**Mitigation:** Monitor step times. If proof checking dominates (>80% of step time), reduce `batch_theorems` or `group_size`, or add more CPU cores.
**Affects:** Phase 1.4, Phase 2+

#### FR4: XPU Memory Exhaustion with Reference Model

**Risk:** GRPO loads two copies of the model (policy + reference). With LoRA, the base weights are shared, but both models + optimizer states + activations could exceed 34 GB VRAM.
**Mitigation:** Monitor VRAM usage during first GRPO steps. If near limit, move reference model to CPU (slower KL computation but safe) or use a single model with careful gradient management.
**Affects:** Phase 1.4

#### FR5: Scaling Curve Requires Multiple Model Sizes

**Risk:** Phase 1.6 needs models at 300M, 1.5B, and 3B parameter scales. 3B may not fit in 34 GB VRAM even with LoRA (3B × 2 bytes bfloat16 = 6 GB for weights alone, but full model + optimizer + activations may exceed).
**Mitigation:** Test 3B memory requirements early. Consider gradient checkpointing, activation offloading, or renting cloud GPU for the 3B run.
**Affects:** Phase 1.6

#### FR6: Random 90/10 Split May Be Too Easy

**Risk:** Random train/val split may put near-duplicate theorems on both sides, inflating eval success rate.
**Mitigation:** Consider domain-level or statement-similarity-based splits for more honest evaluation.
**Affects:** Phase 1.4 eval, Phase 1.7 conclusions

---

## 4. Current Artifact Inventory

| Artifact | Path | Status |
|----------|------|--------|
| Mathlib4 theorems (69,150) | `data/raw/mathlib4_theorems.jsonl` | Ready |
| Lake project (Mathlib4 built) | `proof_checker_env/.lake/` | Ready |
| SFT checkpoint (best val loss) | `checkpoints/sft/best/` | Ready |
| SFT checkpoint (final) | `checkpoints/sft/final/` | Ready |
| SFT training log | `/tmp/claude-1000/...tasks/bi3qjgxfv.output` | Archived |
| GRPO trainer (with curiosity) | `src/training/grpo_trainer.py` | Ready |
| Reward system (binary + curiosity) | `src/reward/base.py` | Ready |
| GRPO config (Phase 1 optimized) | `configs/grpo_config.yaml` | Ready |
| GRPO launch script | `scripts/train_grpo.py` | Ready |
| Proof checker (8 workers, 30s timeout) | `src/proof_checker/batch_checker.py` | Ready |

---

## 5. Next Steps

### Phase 1.4 — GRPO Self-Play Training (NEXT)

```
python scripts/train_grpo.py \
  --sft-checkpoint checkpoints/sft/best \
  --data-dir data \
  --output-dir checkpoints/grpo \
  --max-theorems 500 \
  --use-lora
```

**What to watch:**
- Success rate trajectory (should climb from 0-5% toward 15-30%)
- Unique proof signatures (curiosity working?)
- KL divergence (policy drifting enough to learn?)
- Step time (proof checking bottleneck?)

### Phase 1.6 — Scaling Curve

Train at 300M, 1.5B, 3B parameter scales. Measure final success rate vs parameter count.

### Phase 1.7 — Phase 1 Write-up

Synthesize all Phase 1 data into go/no-go recommendation for Phase 2.

---

*Generated 2026-06-02. Update after each sub-phase completion.*
