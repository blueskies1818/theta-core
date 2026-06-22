# Phase 2 Training Review — What We Learned

**Date:** 2026-06-04
**Runs covered:** 1–7 (bootstrap → 5000-epoch self-play)
**Status:** Architecture validated. GNN learning bottleneck identified. Path forward clear.

---

## 1. All Training Runs

| Run | Epochs | Theorems | Heuristics | H-scale | Best Success | Key Result |
|---|---|---|---|---|---|---|
| 1 | 50 | 500 bootstrap (`0=0`) | None (broken) | — | 25% | First valid proof found. Cold-start confirmed. |
| 2 | 20 | 9 reflexive physics | rfl only | 1.0 | 50% | Correspondence + era tracking active. |
| 4 | 200 | 29 physics (all zones) | rfl, add_comm, mul_comm | 1.0 | 100% | Full pipeline validated. 6 bugs fixed. |
| 5 | 1000 | 29 physics | rfl, add_comm, mul_comm | 1.0→0.0 | 100%→0% | Annealing killed GNN at H=0.0. Too fast. |
| 6 | 2000 | 29 physics | rfl, add_comm, mul_comm | 1.0 | 100% | GNN learned 1 pattern (rfl) with heuristics off. |
| 7 | 5000 | 29 physics | All (9 heuristics) | 1.0 | 100% | Same result as Run 6. More epochs ≠ more learning. |

### Inference Results (held-out post-1905 theorems, H-scale=0.0)

| Model | Proved | Notes |
|---|---|---|
| Untrained baseline | 0/25 (0%) | GNN has no proof knowledge |
| Run 6 (2000 epochs) | 1/25 (4%) | Learned `v²=v²` → rfl |
| Run 7 (5000 epochs) | 1/25 (4%) | Identical to Run 6 |
| **With heuristics H=1.0** | **8–10/25 (32–40%)** | Heuristics carry all weight |

## 2. What Worked

- **Training pipeline:** GNN → MCTS → Proof Checker → GRPO update. Fully functional, reliable, checkpoints save/load correctly.
- **Gradient flow:** `loss.backward()` propagates through differentiable MCTS logits to all GNN parameters. Verified in Run 1, maintained through Run 7.
- **Correspondence layer:** Frontier zone classification, era-gated discovery monitoring, reward shaping all wire correctly.
- **Proof finding:** MCTS consistently finds valid Lean 4 proofs for reflexive and commutative goals (12/29 training theorems).
- **Physics theorem dataset:** 54 theorems spanning 7 eras and 8 frontier zones. All verified as valid Lean 4 (29/29 pass checker).

## 3. What Didn't Work

### 3.1 Heuristics dominate everything

The hand-coded heuristics contribute +1.5 to the right lemma's logit. The GNN contributes ~0.4 at best. The GNN has zero incentive to learn because it's always outvoted. This is the root cause of all training failures.

**Evidence:** H-scale=1.0 → 40% inference success. H-scale=0.0 → 4%. The GNN contributes almost nothing.

### 3.2 Only 12/29 theorems provide training signal

17 of 29 training theorems have proof patterns that no heuristic covers (hypothesis usage, apply, multi-step chains). The GNN sees only failure from these theorems — zero positive signal.

**Evidence:** All successful proofs in 7000+ epochs were `exact rfl`, `rw [add_comm]`, or `rw [mul_comm]`. Not a single hypothesis-using proof or `apply`-based proof was ever found.

### 3.3 More epochs didn't help

Run 6 (2000 epochs) and Run 7 (5000 epochs) produced identical inference results. The GNN saturated after ~1000 epochs because it had seen all the signal there was to see — two proof patterns, repeated thousands of times.

**Evidence:** Loss curves flattened by epoch 1000. Novel proofs plateaued. No new proof types emerged after epoch 500.

### 3.4 GNN pretrained on wrong task

The pretrained GNN learned link prediction ("does lemma A depend on lemma B?"). That's structurally relevant but doesn't answer the question MCTS needs: "given this goal, which lemma proves it?" The GNN starts with embeddings that reflect dependency structure, not proof strategy.

**Evidence:** Cosine similarity between goal embeddings and correct lemma embeddings is only weakly correlated with proof success.

### 3.5 Model is tiny

230K parameters. A 2-layer GAT with 128-dim hidden. For comparison, even small proof assistants use models in the 100M+ range. At this scale, the GNN may simply lack capacity to represent proof strategies.

## 4. Root Cause: The GNN Never Had a Chance

The architecture is correct. The pipeline works. But the GNN was set up to fail from the start:

```
Training loop:  Heuristics (+1.5) >> GNN (+0.4) → GNN outvoted every time
Pretraining:    Link prediction ≠ proof-step prediction
Scale:          230K params, 2 layers — minimal capacity
Data diversity: 2 proof patterns, 12 theorems — no richness
```

The GNN is a passenger in its own training. It watches heuristics pick answers and never gets to make a real decision. When the heuristics turn off, the GNN has learned nothing because it was never asked to contribute.

## 5. The Path Forward

### Principle: The GNN must be the driver

Every design decision must answer: "does this let the GNN make real decisions and learn from the consequences?"

### Specific Changes

| Change | Why |
|---|---|
| **Drop heuristics** except reflexive (X=X → rfl) | GNN must learn lemma selection, not parrot hand-coded rules |
| **Pretrain on proof-step prediction** | Extract (goal, lemma) pairs from 69K mathlib4 proofs — supervised signal |
| **Boost GNN contribution weight** | Increase GNN scoring from 0.4 to 0.8 in `_score_actions` |
| **Keep MCTS sims at 200** | Test first if GNN improvements suffice; increase only if needed |
| **Train with correspondence + era gating** | Full pipeline from the start — measure what matters |

### Expected Timeline

1. Build proof-step pretraining dataset → pretrain GNN (~1 hour)
2. Strip heuristics, boost GNN weight (~30 min)
3. Run 2000-epoch training with honest GNN-first approach (~3 hours)
4. Evaluate with H-scale=0.0 on held-out post-1905 theorems

---

*Generated 2026-06-04. Phase 2 training review complete. The architecture works. The training signal was wrong. Fixing it now.*
