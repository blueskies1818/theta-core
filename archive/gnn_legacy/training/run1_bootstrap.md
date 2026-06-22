# Training Run 1 — Bootstrap Theorems with Pretrained GNN

**Date:** 2026-06-03
**Status:** Completed — first successful proof found
**Branch:** `main`
**Checkpoint:** `checkpoints/explorer_run1/`

---

## 1. Purpose

First end-to-end training run of the GNN+MCTS explorer on real mathlib4 data. The goals:

1. **Verify the training loop works** — GNN → MCTS → Proof Checker → GRPO Update → backward
2. **Test whether MCTS can find valid Lean proofs** with a pretrained GNN guiding search
3. **Measure cold-start characteristics** — how long until the first success?
4. **Validate the gradient path fix** — does `loss.backward()` produce non-zero GNN gradients throughout training?

## 2. Configuration

### Architecture

```
GNN Encoder (2-layer GAT, 128-dim hidden)
  ↓ embeddings
MCTS (PUCT, 100 sims per proof, 30 top-k lemmas)
  ↓ best proof steps
Lean 4 Proof Checker (with Mathlib4)
  ↓ valid/invalid
Correspondence Reward Modifier (13 zones, 12 failures)
  ↓ modified rewards
GRPO Advantages (group-relative, K=2)
  ↓
Policy Loss + Value Loss → backward → GNN update
```

### Hyperparameters

| Parameter | Value |
|---|---|
| Graph | Algebra subgraph — 16,800 nodes, 22,684 edges |
| GNN | 2-layer GAT, hidden=128, heads=4, 230,784 params |
| GNN init | Pretrained (link prediction on full 58K graph) |
| Training theorems | 500 bootstrap (`0 = 0`, `1 + 0 = 1`, …) |
| Train/val split | 480 / 20 |
| Epochs | 50 |
| Batch size | 4 theorems per epoch |
| MCTS simulations | 100 per proof search |
| GRPO group size | 2 proofs per theorem |
| Learning rate | 1e-3 |
| Policy weight | 1.0 |
| Value weight | 0.5 |
| Optimizer | AdamW (weight_decay=1e-5) |

### Era Gating

| Setting | Value |
|---|---|
| Era | `pre_relativity` (≤1904) |
| Cutoff year | 1904 |
| Known concepts | 0 (nothing past classical physics) |
| Discoverable concepts | 19 (special relativity through quantum gravity) |
| Discovery bonus | **None** — passive monitoring only, no reward shaping |

The era tracker monitored for 19 post-1904 physics concepts (special relativity, QM, GR, Standard Model, etc.) but found **zero discoveries** — expected since bootstrap theorems are pure arithmetic with no physics keywords.

### Device

- **Intel Arc Pro B70** (34 GB VRAM, XPU backend)
- PyTorch 2.7.1+xpu

## 3. Results

### Training Metrics

| Epoch | Success Rate | Reward | Loss | Novel Proofs |
|---|---|---|---|---|
| 0 | 0.00% | 0.051 | 1.9023 | 4 |
| 5 | 0.00% | 0.051 | 1.9063 | 24 |
| 10 | 0.00% | 0.051 | 1.8774 | 44 |
| **15** | **25.00%** | **0.613** | **2.0294** | **64** |
| 20 | 0.00% | 0.047 | 1.9047 | 82 |
| 25 | 0.00% | 0.051 | 1.8946 | 102 |
| 30 | 0.00% | 0.051 | 1.8785 | 122 |
| 35 | 0.00% | 0.047 | 1.8968 | 140 |
| 40 | 0.00% | 0.051 | 1.8372 | 160 |
| 45 | 0.00% | 0.047 | 1.8568 | 179 |
| 49 | 0.00% | 0.051 | 1.8900 | 194 |

### Summary

| Metric | Value |
|---|---|
| Total time | 326 seconds (5.4 min) |
| Total proof attempts | 200 (50 epochs × 4 theorems) |
| Successful proofs | 1 (epoch 15) |
| Overall success rate | 0.5% |
| Best epoch success | 25.0% (epoch 15 — 1 of 4 theorems) |
| Final loss | 1.8900 (started at 1.9023) |
| Loss range | 1.84 – 2.03 |
| Era discoveries | 0 (pure math, no physics keywords) |
| Correspondence mods | 2 per epoch (zone classification only) |

### Checkpoints Saved

| Epoch | File | Size |
|---|---|---|
| 25 | `gnn_epoch_25.pt` | 931 KB |
| 50 (final) | `gnn_final.pt` | 931 KB |
| — | `dependency_graph.*` | 8.0 MB |

## 4. Analysis

### 4.1 The first successful proof (Epoch 15)

At epoch 15, one of four bootstrap theorems was successfully proved. This is the first evidence that the full training pipeline works end-to-end:

- MCTS, guided by the pretrained GNN, explored the proof space
- It selected `exact rfl` or a similar simple tactic
- The proof checker validated it as correct Lean 4
- The reward jumped from ~0.05 (curiosity-only) to 0.613 (valid proof + curiosity)
- The policy loss increased (1.91 → 2.03) because the GNN's logits were compared against MCTS visit distributions that now contained a meaningful success signal
- Gradients flowed back through the differentiable logits to update GNN parameters

### 4.2 Cold-start characteristics

The pretrained GNN (trained on link prediction over the full 58K graph) provides a meaningful starting point — it knows that `rfl` is related to equality goals, that `add_comm` is related to addition, etc. Without this pretraining, MCTS would pick lemmas at random and success would be essentially impossible.

However, the cold-start problem is severe:
- **200 proof attempts, 1 success = 0.5% hit rate**
- The GNN learned slowly (loss decreased from 1.90 → 1.84–1.89)
- With only one positive example, the policy gradient signal is very sparse
- Most epochs show identical reward (0.051) — the curiosity bonus on failed proofs

### 4.3 Loss dynamics

The loss is dominated by the value loss component (MSE between GNN value estimate and actual outcome). Since 99.5% of outcomes are failure (value=0.0), the GNN learns to predict ~0 for most states. The policy loss only activates when MCTS finds a successful branch, which is rare.

The loss spike at epoch 15 (2.03) and gradual decline (to 1.84 by epoch 40) suggest:
1. The first success creates a gradient signal that temporarily increases loss (the GNN is "surprised")
2. Over subsequent epochs, the GNN adjusts its weights to better predict MCTS behavior
3. But without more successes, the signal fades

### 4.4 What the GNN learned

With only one positive example in 200 attempts, the GNN's learning is minimal. The loss decrease (1.90 → 1.84) is mostly from the value head learning to consistently predict failure — which is correct for bootstrap theorems where MCTS rarely succeeds.

For meaningful learning, we need:
- **More successes** — either simpler theorems or more MCTS simulations
- **More training steps** — 50 epochs is not enough for this sparse reward setting
- **Curriculum** — start with theorems where MCTS can succeed >10% of the time

## 5. What Worked

- [x] Full training pipeline executes without errors
- [x] `loss.backward()` propagates gradients to all GNN parameters (gradient path fix verified)
- [x] Pretrained GNN loads and provides meaningful lemma scoring
- [x] MCTS finds at least one valid proof (epoch 15)
- [x] Proof checker validates real Lean 4 proofs
- [x] Correspondence modifier classifies proofs into frontier zones
- [x] Era tracker passively monitors for physics discoveries
- [x] Checkpoints saved successfully (new config_dict format)
- [x] Training runs on Intel Arc GPU (XPU backend)

## 6. What Needs Improvement

| Issue | Severity | Proposed Fix |
|---|---|---|
| 0.5% proof success rate | Critical | Use simpler theorems, increase MCTS sims to 200–400 |
| Only 1 success in 50 epochs | Critical | Curriculum: start with `rfl`-only proofs |
| No physics discoveries | Expected | Bootstrap theorems have no physics keywords — need physics-themed theorems for era validation |
| Loss dominated by value head | Moderate | Increase policy weight, decrease value weight |
| GNN learning barely visible | Moderate | More epochs (500+), more successes needed |
| Eval not reporting | Minor | Fix eval logging in ExplorerTrainer |

## 7. Next Steps

### Immediate
1. **Run with simpler theorems** — filter bootstrap to only `rfl`/`simp`-provable theorems
2. **Increase MCTS sims** — 200–400 sims per proof for better search
3. **Longer training** — 200–500 epochs for meaningful GNN learning
4. **Physics-themed theorems** — create a small set of theorems with physics keywords to test era-gated discovery monitoring

### Medium-term
5. **Curriculum learning** — start with 1-step proofs, gradually increase difficulty
6. **Better cold-start** — pretrain GNN on proof-step prediction, not just link prediction
7. **Full physics validation** — train on pre-1905 theorems, measure spontaneous discovery rate of special relativity, QM, GR concepts

## 8. Artifacts

```
checkpoints/explorer_run1/
├── gnn_epoch_25.pt              # Mid-training checkpoint (931 KB)
├── gnn_final.pt                 # Final GNN weights (931 KB)
├── dependency_graph.nx.pkl      # Algebra subgraph (7.1 MB)
├── dependency_graph.index.json  # Node index (864 KB)
└── dependency_graph.stats.json  # Graph statistics (467 B)

docs/training/
└── run1_bootstrap.md            # This report
```

---

*Generated 2026-06-03. First training run complete. The architecture works — now it needs more data and more time.*
