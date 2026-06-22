# Architecture Decision: Gate 3 Lemma-Novelty Synthesis

**Date**: 2026-06-16
**Decision**: RETHINK — fundamental architecture redesign required
**Author**: implementer (synthesis of H1, H2, H3 studies)

## Executive Summary

Three hypothesis-driven studies (H1: capacity scaling, H2: scoring architectures,
H3: traversal reward) were conducted to improve Gate 3 lemma-novelty
generalization above the baseline of 21.4%. **No approach exceeded the 45%
minimum effective threshold.** The best result (28.6% from H1's 50M-parameter
GNN) is only +7.2 percentage points above baseline and 16.4pp below the
threshold. Non-monotonic scaling, persistent overfitting, and the failure of all
three scoring architectures to beat simple cosine similarity collectively
indicate the current GNN+GRPO+MCTS architecture has hit a fundamental ceiling
that cannot be breached through parameter scaling or reward shaping alone.

## Evidence Table

### GNN Gate 3 Scores (target: ≥45%)

| Study | Approach | Gate 3 Score | Δ vs Baseline (21.4%) | ≥45%? | Key Weakness |
|-------|----------|-------------|-----------------------|-------|--------------|
| H1    | 50M-param GNN (hidden=1536, 4 layers) | 28.6% (4/14) | +7.2pp | **No** | Diminishing returns: 45× params for +7pp |
| H1    | 5M-param GNN (hidden=544, 3 layers) | 21.4% (3/14) | 0pp | **No** | Zero improvement over 1M baseline |
| H1    | 10M-param GNN (hidden=768, 3 layers) | 14.3% (2/14) | −7.1pp | **No** | Non-monotonic: worse than 5M |
| H3    | GNN + traversal reward (H=0.0 eval) | 7.1% (1/14) | −14.3pp | **No** | Only 2 training lemmas in graph; reward never triggered |
| H1    | 1M-param GNN (hidden=256, 3 layers) | 7.1% (1/14) | −14.3pp | **No** | Baseline architecture, insufficient capacity |

### H2 Lemma Scoring Architectures (retrieval quality)

| Scorer | MRR | Top-1 % | Top-5 % | Improves over cosine? | Verdict |
|--------|-----|---------|---------|----------------------|---------|
| Cosine baseline | 0.7858 | 78.6% | 78.6% | N/A (baseline) | **Already used at 21.4% Gate 3** |
| Two-tower bilinear | 0.0133 | 0% | 0% | No (learned W distorts embeddings) | FAIL |
| Cross-attention | 0.0002 | 0% | 0% | No (softmax collapses to uniform) | FAIL |
| Graph-filtered cosine (k=5) | 0.0003 | 0% | 0% | No (ground-truth lemmas outside k-hop) | FAIL |

**Critical insight**: Cosine similarity already achieves 78.6% Top-1 lemma retrieval
accuracy, yet end-to-end proof success is only 21.4%. The bottleneck is NOT
lemma retrieval quality — it lies elsewhere in the system.

### H3 Shape-Matcher Constraint (must stay ≤5%)

| Approach | Shape-Matcher Score | Constraint (≤5%) | Pass? |
|----------|-------------------|-------------------|-------|
| H3 shape-matcher (H=1.0) | 35.7% (5/14) | ≤5% | **FAIL** |

## Theorems Proved by Each Approach

| Theorem | Baseline GNN | H1 50M | H3 GNN | Shape-Matcher |
|---------|-------------|--------|--------|---------------|
| poly_eval_C_add_X | ✓ | ✓ | — | ✓ |
| poly_mul_X_add_C | ✓ | ✓ | ✓ | — |
| poly_derivative_add | ✓ | ✓ | — | — |
| poly_eval_mul_X_sub_C | — | — | — | ✓ |
| poly_eval_derivative_C_mul | — | — | — | ✓ |
| poly_degree_X_pow | — | — | — | ✓ |
| poly_map_id | — | — | — | ✓ |
| All others (7 theorems) | — | — | — | — |

Only 3 theorems are ever provable by GNN across all studies. 7 of 14 gates are
never proved by any approach. The shape-matcher (heuristic baseline) succeeds on
5 theorems where GNN fails, suggesting the GNN is actively worse than simple
pattern matching for certain problem types.

## Root Cause Analysis

### 1. Lemma discrimination is not the bottleneck
Cosine similarity retrieves correct lemmas at 78.6% Top-1, but end-to-end proof
success is only 21.4%. The GNN cannot convert accurate lemma retrieval into
successful proof trajectories. The 50M-parameter GNN (45× larger) only adds 1
more theorem (poly_eval_C_add_X was already provable at baseline).

### 2. Training instability is systemic
Every study exhibits the same overfitting pattern: high best-success during
training (93.8%–100%) collapsing to low final success (37.5%–43.8%) with poor
generalization. This is not a capacity problem — it is a credit assignment
problem in GRPO where the binary reward signal is too sparse for the GNN to
learn which lemma choices matter.

### 3. Non-monotonic scaling reveals structural ceiling
5M (21.4%) > 10M (14.3%) < 50M (28.6%). If the bottleneck were pure capacity,
scores would rise monotonically. The non-monotonic pattern suggests the
architecture has a representational ceiling that more parameters cannot break
through — likely the GNN's inability to learn multi-step proof composition.

### 4. All theorems are single-tactic successes
Every proof across all studies is a single-tactic application (e.g., `simp`,
`exact lemma_name`). No approach has ever demonstrated multi-step proof
chaining. The 7 theorems that are never proved likely require 2+ tactics, which
the current MCTS with binary reward cannot discover.

### 5. Shape-matcher consistently beats GNN
The shape-matcher baseline (keyword matching + heuristic lemma selection) scores
28.6%–35.7% while the GNN scores 7.1%–28.6%. On 5 theorems, shape-matcher
succeeds where GNN fails. This means the GNN has not learned to exploit patterns
that simple heuristics can find — a strong signal that the GNN representation is
not capturing the right features.

## Recommendation: Fundamental Architecture Rethink

The current architecture (GAT-GNN encoder → cosine lemma scoring → MCTS search →
binary GRPO reward) has reached a ceiling at ~28% that no incremental
improvement can breach. The evidence from H1, H2, and H3 collectively rules out
capacity scaling, scoring architecture changes, and reward shaping as viable
paths forward.

### What must change

1. **Proof strategy representation**: The GNN must learn to compose multi-step
   proofs, not just select a single lemma. This requires a sequence-to-sequence
   or autoregressive proof generation approach rather than single-action MCTS.

2. **Dense reward signal**: Binary proof success per trajectory is too sparse.
   Consider step-level rewards (e.g., tactic validity, lemma relevance,
   partial goal decomposition) or learned reward models.

3. **Lemma embedding quality**: Cosine similarity works well for retrieval
   (78.6% Top-1) but the embeddings themselves may not encode proof-relevant
   features. Consider contrastive learning on proof trajectories or
   proof-step co-occurrence in Mathlib.

4. **Alternative search strategies**: MCTS with PUCT was designed for board
   games with dense state evaluation. Proof search may benefit from different
   strategies: best-first search with learned value functions, diffusion-based
   proof generation, or language-model-guided search.

5. **Training data scale**: Gate 3 tests on 14 theorems. Training on gate2's 55
   theorems with only 2 lemmas resolved in the dependency graph leaves the GNN
   with almost no signal about lemma relationships. The dependency graph must be
   enriched before any architecture can learn from it.

### What to preserve

- The dependency graph infrastructure (58K nodes, 160K edges) is valuable and
  should be retained in any redesign.
- The proof checker interface (Lean 4 subprocess, batch checking, SHA-256 cache)
  is robust and architecture-agnostic.
- The correspondence layer (frontier map, era tracker) is orthogonal to
  architecture and remains valuable.
- Cosine similarity lemma retrieval (78.6% Top-1) should remain the retrieval
  baseline in any new architecture.

### Next steps

1. Design a proof-step-level training pipeline with dense rewards
2. Explore sequence-to-sequence proof generation (transformer decoder on lemma
   sequences)
3. Enrich the dependency graph with proof-step co-occurrence data from Mathlib
4. Build a learned value function for proof states (predict proof completability)
5. Run a controlled experiment: language-model-guided proof search vs. current
   GNN+MCTS to establish a new upper bound

## Appendix: Study Configurations

| Parameter | H1 (Capacity) | H2 (Scoring) | H3 (Traversal) |
|-----------|---------------|--------------|-----------------|
| Pretrained base | explorer_wave2 | explorer_gate3_v4 | explorer_wave2 |
| Training data | gate2 (55 theorems) | N/A (eval only) | gate2 (55 theorems) |
| Test data | gate3 (14 theorems) | gate3 (14 theorems) | gate3 (14 theorems) |
| Training budget | 7 steps × 2 batch × 50 sims | 500 calib epochs | 25 epochs × 500 sims |
| Heuristic annealing | N/A | N/A | H=1.0 constant (no anneal) |
| Special config | 4 GNN scales | 3 scoring archs + cosine | traversal weight=0.5, threshold=3 hops |

## Appendix: Data Files

| File | Contents |
|------|----------|
| `data/h1_capacity_results.json` | H1: GNN scores at 1M/5M/10M/50M params |
| `data/h2_scoring_results.json` | H2: Full retrieval benchmark (746 lines) |
| `data/h3_traversal_results.json` | H3: Traversal reward experiment results |
| `data/gate3_result.json` | Gate 3 baseline: 21.4% GNN, 28.6% shape-matcher |
