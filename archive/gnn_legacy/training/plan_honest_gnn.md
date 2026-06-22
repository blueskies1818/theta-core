# Implementation Plan — Honest GNN Training

**Date:** 2026-06-04
**Updated:** 2026-06-04 (scaled architecture, goal encoder, longer annealing)
**Goal:** Make the GNN the primary decision-maker in MCTS, not a passenger to heuristics.

---

## Root Cause Recap

The GNN (230K params, 128-dim, 2 layers) was set up to fail:

1. **Heuristics outvote the GNN** — hand-coded patterns contribute +1.5 to logits; GNN similarity ~0.4 at best. GNN is never the deciding vote.
2. **Pretraining on the wrong task** — link prediction ("does lemma A depend on lemma B?") instead of proof-step prediction ("given this goal, which lemma proves it?").
3. **Goal embeddings are lossy** — keyword-matched averaging of lemma embeddings is a blurry approximation. No learned goal encoder.
4. **MCTS sims too low** — 200 simulations without heuristics = nearly random walk. The GNN needs to be extremely accurate for 200 sims to find proofs.
5. **Capacity too small** — 230K params is an order of magnitude below even small proof assistants.

All five are addressed in this plan.

---

## Changes Summary

| # | Change | File(s) | Why |
|---|--------|---------|-----|
| 1 | Scale GNN: 256-dim, 3 layers, 8 heads (~1M params) | `gnn_config.py` | 4× capacity for proof-step representation |
| 2 | Add learned GoalEncoder (2-layer MLP + LayerNorm) | `gnn_encoder.py` | Projects blurry keyword-average into sharp lemma space |
| 3 | Increase MCTS sims: 200 → 1000 | `mcts.py` | 5× search budget compensates for imperfect GNN |
| 4 | Drop all heuristics except rfl for X=X | `mcts.py` | GNN must learn lemma selection — rfl is definitional |
| 5 | Reduce rfl boost: +1.5 → +0.5 | `mcts.py` | rfl is a hint, not a crutch |
| 6 | GNN weight already at 0.8 (stays) | `mcts.py` | Already boosted from 0.4 in previous edit |
| 7 | Pretrain on proof-step prediction with multi-class CE loss | `pretrain_proof_step.py` | Proper supervised signal from 69K proofs |
| 8 | Train goal encoder during pretraining | `pretrain_proof_step.py` | Goal encoder learns to map goals → correct lemmas |
| 9 | Slow annealing: 2000 epochs, linear 1.0→0.0 | `explorer_trainer.py` | GNN gradually takes over from rfl heuristic |
| 10 | Default MCTS sims to 1000 in train script | `train_explorer.py` | Safe default for honest training |

---

## Step 1: Scale GNN Architecture

**What:** Increase model capacity from 230K → ~1M params.

**Changes to `src/explorer/gnn_config.py`:**
```python
hidden_dim: int = 256       # was 128
num_layers: int = 3         # was 4 (reducing for speed, 3 is enough)
num_heads: int = 8          # stays 8
```

**Estimated params:**
- 3-layer GAT, 256-dim hidden, 8 heads: ~900K params
- Plus GoalEncoder (~400K): ~1.3M total
- Fits easily in 34GB VRAM (Intel Arc B70)

---

## Step 2: Add Learned Goal Encoder

**What:** A 2-layer MLP that takes the keyword-averaged lemma embedding and projects it into a better goal embedding space. Trained jointly with the GNN.

**New code in `src/explorer/gnn_encoder.py`:**
```python
class GoalEncoder(nn.Module):
    """Learned projection from keyword-averaged context to goal embedding."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
    
    def forward(self, context_embedding: torch.Tensor) -> torch.Tensor:
        return self.proj(context_embedding)
```

**Added to `GNNEncoder`:**
- `self.goal_encoder = GoalEncoder(config.hidden_dim)`
- `encode_goal(keyword_embedding)` — passes through goal encoder
- Save/load includes goal encoder state

**Why this works:** During pretraining, the goal encoder learns to map "keyword-matched average of add_comm, add_zero, add_assoc" to the specific embedding of the correct lemma. During GRPO, the goal encoder gets gradients through the MCTS policy loss, refining goal → lemma mapping.

---

## Step 3: Proof-Step Pretraining Dataset

**What:** Extract (goal_text, lemma_used) pairs from mathlib4 proofs.

**How:**
- Parse `data/raw/mathlib4_theorems.jsonl` (69K theorems with proof bodies)
- For each proof, extract the first lemma referenced
- Create goal text from the theorem statement
- Build a classification dataset: given a goal, predict which lemma was used

**Output:** `data/raw/proof_step_pairs.jsonl` — goal → lemma mapping, ~50K training pairs

**Code:** Existing script `scripts/build/build_proof_step_data.py` (no changes needed)

## Step 4: Pretrain GNN on Proof-Step Prediction

**What:** Train the GNN + GoalEncoder to predict which lemma closes a given goal.

**How:**
- Load the proof-step dataset, filter to lemmas present in the graph
- For each (goal, lemma) pair:
  1. Extract goal keywords, find matching lemma nodes, average their GNN embeddings → keyword context
  2. Pass through GoalEncoder → learned goal embedding
  3. Compute cosine similarity of goal embedding vs ALL candidate lemma embeddings
  4. Multi-class cross-entropy: correct lemma should have highest similarity
- Train for 200 epochs on the domain-specific graph (e.g., Algebra = 16.8K nodes)

**Loss:** Cross-entropy over all lemma candidates:
```python
logits = goal_emb @ all_lemma_embs.T  # [1, num_nodes]
loss = F.cross_entropy(logits / temperature, target_index)
```

**Output:** `checkpoints/gnn/proof_step_pretrained.pt`

**Code:** Rewrite `scripts/training/pretrain_proof_step.py`

## Step 5: Strip Heuristics, Keep Only rfl

**What:** Remove commutative, associative, distributive, and hypothesis heuristics. Keep only rfl for X=X goals at reduced weight.

**Changes to `src/explorer/mcts.py` (`_score_actions`):**
```python
# REMOVED: _detect_commutative_goal boost for add_comm/mul_comm
# REMOVED: _detect_associative_goal boost
# REMOVED: _detect_distributive_goal boost
# REMOVED: _boost_hypothesis_match
# REMOVED: _boost_hypothesis_rewrite

# KEPT (reduced): rfl boost for reflexive goals
if _is_reflexive_goal(goal_only):
    for i, action in enumerate(actions):
        if action.lemma in ("rfl", "Eq.refl", "Eq.refl'") and action.tactic_type.value == "exact":
            logits[i] = logits[i] + 0.5  # reduced from 1.5
```

**GNN similarity weight:** Already at 0.8 (changed in previous edit). Stays.

**Centrality weight:** Already at 0.1 (changed in previous edit). Stays.

**MCTS sims:** 200 → 1000 default.

**Why the rfl heuristic stays:** "X = X → rfl" is definitional, not a learned pattern. Even a mathematician doesn't "learn" this — it's the definition of equality. The reduced weight (0.5) makes it a nudge, not a decision-maker.

## Step 6: Train with Honest Setup

**What:** Full training run with pretrained GNN+GoalEncoder, minimal heuristics, correspondence + era gating, slow annealing.

**Config:**
```
--pretrained checkpoints/gnn/proof_step_pretrained.pt
--hidden-dim 256
--num-layers 3
--num-heads 8
--mcts-sims 1000
--heuristic-anneal-epochs 2000
--heuristic-scale-min 0.0
--steps 2000
--era pre_relativity
# correspondence enabled, eval enabled
```

**Expected:** Lower initial success rate but GNN+GoalEncoder learns genuinely. After annealing to H=0.0, GNN maintains proof-finding ability without heuristics.

## Step 7: Evaluate

Inference on 25 held-out post-1905 theorems with H-scale=0.0.

**Success criterion:** Trained GNN with H=0.0 achieves >2/25 proofs (beating current best of 1/25).

---

## File Changes Summary

| File | Action | Purpose |
|---|---|---|
| `src/explorer/gnn_config.py` | Edit | Scale defaults: 256-dim, 3 layers |
| `src/explorer/gnn_encoder.py` | Edit | Add GoalEncoder class, encode_goal method, save/load |
| `src/explorer/mcts.py` | Edit | Use goal encoder, 1000 sims default, strip heuristics, reduce rfl boost |
| `scripts/training/pretrain_proof_step.py` | Rewrite | Multi-class CE loss, goal encoder, 200 epochs |
| `scripts/training/train_explorer.py` | Edit | Defaults: 1000 sims, 256-dim, 3 layers, 2000 anneal |
| `src/explorer/explorer_trainer.py` | Edit | Default anneal epochs 2000 |
| `scripts/build/build_proof_step_data.py` | No change | Already correct |
| `scripts/eval/infer_explorer.py` | No change | Already supports H-scale |

---

## Risk: Cold Start

Without commutative/associative/distributive/hypothesis heuristics, MCTS may find zero proofs initially for all 17 theorems with those patterns.

**Mitigations:**
1. **Proof-step pretrained GNN + GoalEncoder** provides better-than-random lemma scoring from step 1
2. **1000 MCTS sims** gives 5× more search budget to find proofs with imperfect GNN
3. **rfl heuristic retained** — 9 training theorems are reflexive (X=X), guaranteed to be found
4. **Slow 2000-epoch annealing** — GNN has many epochs at H=1.0 to learn before heuristics are removed
5. If 0% for >300 epochs without heuristics: add back commutative heuristic at reduced weight (+0.5, not +1.5) and continue

---
*Implementation starts now.*
