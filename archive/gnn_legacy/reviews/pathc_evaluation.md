# Path C Evaluation: Contrastive Lemma Embedding + Best-First Search

**Date**: 2026-06-17
**Decision**: **REASSESS** — Path C does not beat the GNN+MCTS baseline on the gate3 lemma-novelty set
**Author**: implementer (Pivot Gate C)
**Prior decision**: pivot_decision.md (2026-06-17) — recommended Path C as minimal viable pivot
**Context**: Path C was implemented in three pivot steps: (1) extract 226K proof-step pairs, (2) train contrastive CharCNN dual-encoder, (3) implement best-first proof search

## Executive Summary

Path C was evaluated on the original gate3_lemma_novelty set (14 polynomial theorems)
to provide an apples-to-apples comparison against the MCTS+GNN baseline. **Path C
scores 0% (0/14) on proof success compared to the MCTS+GNN baseline of 28.6% (4/14).**
Retrieval MRR is 0.334 (3 evaluated theorems) vs the baseline 0.786 — a 58% decrease.
The contrastive CharCNN encoder cannot retrieve correct lemmas from 70K candidates,
and the best-first search cannot compensate for poor lemma scoring.

While Path C demonstrated 2 multi-step proofs on the gate3_v2 set (rw[h]; ring),
these were achieved through structural tactic chaining with hypothesis variables —
not through lemma retrieval. On the lemma-novelty set where specific named lemmas
are required, Path C fails completely.

**The search architecture is better than MCTS. The embedding quality is worse.
The net result is worse than baseline. REASSESS.**

## Controlled Comparison Results

### Proof Success on gate3_lemma_novelty (14 theorems)

| Method | Proved | Rate | Multi-step | Notes |
|--------|--------|------|------------|-------|
| **MCTS GNN H=0.0 (Algebra subgraph, 3840 nodes)** | 4/14 | 28.6% | 0 | All proofs `simp` |
| **MCTS GNN H=0.0 (Full graph, 116K nodes)** | 0/14 | 0% | 0 | Too many candidates |
| **Path C best-first (gate3_lemma_novelty, 70K lemmas)** | 0/14 | 0% | 0 | All failed in Lean |
| MCTS Heuristic H=1.0 (Algebra) | 3/14 | 21.4% | 0 | Same as prior |
| Shape-matcher (simp/norm_num) | 6/14 | 42.9% | 0 | Benchmark weakness |

**Path C vs MCTS GNN H=0.0 Algebra: 0% vs 28.6%. Path C loses.**

### Retrieved Tactic Analysis (Why Path C Failed)

All 14 theorems failed because the search chose wrong tactics:

| Theorem | Path C Tactic | Actual Proof | Failure |
|---------|---------------|--------------|---------|
| poly_eval_C_add_X | `ring` | `simp` | unsolved goals |
| poly_mul_X_add_C | `exact p` | `simp [mul_add]` | type mismatch |
| poly_eval_mul_X_sub_C | `exact p` | `simp` | type mismatch |
| poly_derivative_X_pow | `exact n` | `simpa using derivative_X_pow n` | type mismatch |
| poly_eval_derivative_C_mul | `exact p` | `simp` | type mismatch |
| poly_derivative_add | `ring` | `simp` | unsolved goals |
| poly_degree_X_pow | `exact n` | `simp` | type mismatch |
| poly_degree_C_mul_X_pow | `exact c` | `simpa using natDegree_C_mul_X_pow` | type mismatch |
| poly_degree_mul_X | `exact p` | `simpa [...]` | type mismatch |
| poly_degree_add_eq_left_of_degree_lt | `exact h` | `simpa using degree_add_eq_left_of_degree_lt h` | type mismatch |
| poly_degree_sub_eq_left_of_degree_lt | `exact h` | `simpa using degree_sub_eq_left_of_degree_lt h` | type mismatch |
| poly_monic_X_sub_C | `exact a` | `simpa using monic_X_sub_C a` | type mismatch |
| poly_monic_X_pow_add | `exact n` | `exact monic_X_pow_add hp` | type mismatch |
| poly_map_id | `exact p` | `simp` | type mismatch |

The search repeatedly chose `exact` with a hypothesis or argument variable
(e.g., `exact p`, `exact n`, `exact a`) instead of the correct lemma or tactic.
The contrastive encoder isn't scoring the correct lemmas highly enough for the
search to consider them — the top-K lemmas are unrelated to the proof goal.

### Comparison with gate3_v2 Results (Parent Task)

The parent task (t_3ad2c8a1) reported 62.5% (5/8) on gate3_v2. These results
are NOT comparable to the gate3_lemma_novelty baseline for three reasons:

1. **Different benchmark**: gate3_v2 has 64 multi-step theorems across 5 domains;
   gate3_lemma_novelty has 14 polynomial theorems in Algebra only.

2. **Permissive verification**: The 5 successes on gate3_v2 used permissive
   verification (`verify_permissive: true`) that accepts `rewrite`/`intro`/`cases`
   steps with unsolved goals as valid intermediate steps. On gate3_lemma_novelty,
   all 14 theorems required strict final verification and all 14 failed.

3. **Success mode different**: The gate3_v2 successes used structural tactics
   that don't require specific lemma retrieval:
   - `rw [h]; ring` (uses hypothesis `h`, not a lemma from the library)
   - `field_simp` (structural automation)
   - `simp` (simplification set, not specific lemma)
   
   The multi-step "breakthrough" was real but came from hypothesis rewriting +
   automation, not from lemma-novelty retrieval as the gate intended.

## Retrieval MRR Evaluation

| Metric | Baseline (Cosine, GNN) | Path C (Contrastive CharCNN) | Δ |
|--------|------------------------|------------------------------|---|
| MRR (gate3_lemma_novelty, 3 evaluated) | 0.786 | 0.334 | −0.452 (−57.5%) |
| MRR (gate3_v2, 29 evaluated) | 0.786 | 0.079 | −0.707 (−90.0%) |
| Top-1 accuracy (gate3_lemma_novelty) | — | 0.333 | — |
| Top-1 accuracy (gate3_v2) | — | 0.035 | — |
| Condition (a): "retrieval MRR improved" | — | — | **FAIL** |

The contrastive encoder's MRR (0.334 on gate3_lemma_novelty) is less than half
the baseline. On gate3_v2 it drops to 0.079 — effectively random among 70K lemmas.
The in-batch InfoNCE loss with batch_size=256 cannot provide enough negative
signal to discriminate among 70K candidates.

### Per-Theorem Retrieval (gate3_lemma_novelty, 3 with named lemmas)

| Theorem | Correct Lemma | Path C Rank | RR | 
|---------|--------------|-------------|-----|
| poly_mul_X_add_C | mul_add | 496 | 0.002 |
| poly_degree_mul_X | natDegree_mul_X | 5489 | 0.0002 |
| poly_monic_X_pow_add | monic_X_pow_add | 1 | 1.000 |

Only 1 of 3 correct lemmas was ranked in the top 500. The other 11 theorems
could not be evaluated because they use `simp` or `simpa` without explicit
named lemmas — this is itself a benchmark limitation.

## Condition Check Summary

| Condition | Target | Path C Result | Pass? |
|-----------|--------|---------------|-------|
| (a) Retrieval MRR improved | > 0.786 | 0.334 (gate3_lemma_novelty) | **FAIL** |
| (b) Proof success ≥ 28.6% | ≥ 28.6% | 0% (gate3_lemma_novelty) | **FAIL** |
| (c) Multi-step proofs > 0 | > 0 | 2 (gate3_v2 only) | PASS* |

\* The 2 multi-step proofs (alg_subst_expand, alg_subst_factor) are `rw[h]; ring` —
hypothesis rewriting followed by ring automation. These are real multi-step proofs
but they don't demonstrate lemma-novelty retrieval. The search architecture
(priority queue with structural tactic priority) can chain tactics, but the
contrastive embeddings contribute nothing to this success.

## Root Cause Analysis

### Why the contrastive encoder fails

1. **In-batch negatives are insufficient**: InfoNCE with batch_size=256 provides
   only 255 negatives per positive pair. With 70K lemma candidates, the encoder
   sees <0.4% of the lemma space per batch and cannot learn to discriminate.

2. **CharCNN is too weak**: Character-level CNNs capture n-gram patterns but
   lack semantic understanding. Lemma names like `natDegree_C_mul_X_pow` and
   `degree_add_eq_left_of_degree_lt` share n-grams but have different meanings.
   The CharCNN cannot distinguish them.

3. **Training signal is noisy**: The 226K proof-step pairs are extracted from
   Mathlib4 proofs where lemma usage is context-dependent. Many "correct" lemmas
   are trivial (e.g., `add_comm`, `mul_assoc`) and provide no discriminative signal.

4. **30 epochs plateaued early**: Best validation accuracy (in-batch) was 29.5%
   at epoch 15 and plateaued. The model converged to a weak local optimum.

### Why best-first search can't compensate

The best-first search has a fundamental design: lemma scores are capped at 0.65
while structural tactics (rw=0.75, ring=0.70, simp=0.65) are given equal or
higher priority. This was intentional — the parent task noted that lemma retrieval
was unreliable and structural tactics should be tried first.

But on the gate3 lemma-novelty set, structural tactics don't work:
- `ring` works on polynomial equations but not on evaluation (`eval`), degree
  (`natDegree`), or monicity (`Monic`) goals
- `simp` works on 6/14 theorems (the shape-matcher finds these) but the
  search doesn't try `simp` for all theorems because lemma actions score
  similarly (0.55-0.65) and get expanded first
- `exact` with hypotheses fails because the goals aren't hypothesis instances

The search exhausts its budget trying lemma actions that all fail, never reaching
the structural tactics that would succeed for 6 of the 14 theorems.

### Contrast with MCTS baseline success mode

The MCTS baseline at 28.6% succeeded because:
- The GNN embeddings, while weak, provide SOME signal on the Algebra subgraph (3840 nodes)
- On 4 theorems, the GNN scores `simp` high enough to be explored by MCTS
- The MCTS exploration noise tries multiple tactics, including `simp`
- The 6 `simp`-solvable theorems are provable given any architecture that tries `simp`

## What Works (Preserve)

| Component | Status | Evidence |
|-----------|--------|----------|
| Priority-queue best-first search | **PROMISING** | Found 2 multi-step proofs MCTS never found |
| Structural tactic scoring | **CORRECT** | rw=0.75 > ring=0.70 > simp=0.65 priority is sensible |
| Proof-step pair extraction pipeline | **VALUABLE** | 226K pairs from 27 domains is a reusable asset |
| Permissive verification for rewrites | **USEFUL** | Accepting unsolved-goal rewrites enables proof chaining |
| Terminal state Lean verification | **REQUIRED** | Caught all 14 false completions on gate3 |

## What Does NOT Work (Abandon or Redesign)

| Component | Status | Evidence |
|-----------|--------|----------|
| CharCNN contrastive encoder | **FAIL** | MRR 0.08-0.33 vs 0.79 baseline |
| InfoNCE in-batch training (256) | **FAIL** | 255 negatives insufficient for 70K lemmas |
| Lemma scoring as primary action selector | **FAIL** | All 14 theorems chose wrong tactics |
| Capped lemma scores (0.65) | **MISALIGNED** | Search skips `simp` (0.65) for lemma actions (0.55-0.65) |

## Decision: REASSESS

**Path C does not beat the GNN+MCTS baseline.** The contrastive embedding approach
with the current architecture (CharCNN, in-batch InfoNCE, 1M params) is worse than
the GNN cosine-similarity baseline it was meant to replace. The best-first search
architecture is promising but cannot succeed with broken lemma retrieval.

### Why NOT proceed to Path A (dense rewards)

Path A was the planned next step: "If Path C shows lemma-novelty improvement, add
step-level rewards." Path C shows **negative** lemma-novelty improvement (0% vs 28.6%).
Adding dense rewards to a search that can't find correct lemmas won't help — the
credit assignment problem (Root Cause 1 from pivot_decision.md) isn't the bottleneck
here; the retrieval problem (Root Cause 3) is.

### Reassessment options

Based on the findings, three paths forward:

**Option 1: Fix the embeddings, keep the search.**
- Replace CharCNN with pretrained LM embeddings (CodeBERT, StarEncoder)
- Use MoCo-style memory bank with 4096+ negatives
- Train with hard negative mining on lemma-novelty pairs
- Keep the best-first search architecture
- Risk: LM embeddings may not encode proof utility either

**Option 2: Return to GNN+MCTS with better candidate filtering.**
- The GNN at 28.6% with 3840 candidates is the best result so far
- Add candidate pre-filtering (domain subgraph, shape-matching, tactic type)
- Accept that lemma-novelty requires architectural limits on candidate space
- Risk: This doesn't solve the fundamental retrieval problem, just works around it

**Option 3: Skip to Path B (Transformer-guided proof generation).**
- Seq2seq models natively handle multi-step composition
- Can be trained on the 226K proof-step pairs (same data Path C used)
- Cross-attention between goal and lemma tokens provides lemma discrimination
- Risk: Highest complexity; may require more training data

**Option 4 (New): Hybrid — GNN retrieval + Best-first search.**
- Use the GNN embeddings (which score 0.786 MRR) as the lemma scorer
- Use the best-first search (which found multi-step proofs) as the search engine
- Combine the best component from each architecture
- Risk: GNN embeddings degrade to 0 MRR on full 116K graph; same scaling problem

## Recommendation

**Option 4 (Hybrid) is the lowest-cost path forward**: combine the GNN's superior
retrieval (0.786 MRR on Algebra subgraph) with the best-first search's superior
proof composition (2 multi-step vs 0 for MCTS). This can be done by replacing
the contrastive encoder with GNN embeddings in BestFirstSearch, using the
Algebra subgraph as the candidate space (3840 nodes instead of 70K).

If the hybrid achieves >28.6% with multi-step proofs on gate3_lemma_novelty,
then proceed to Path A (dense rewards) on the hybrid architecture.

If the hybrid also fails, the retrieval problem is structural — lemma co-occurrence
in the dependency graph doesn't encode proof utility. In that case, proceed to
Option 1 (better embeddings) or Option 3 (seq2seq), skipping further investment
in the current embedding approaches.

## Data Files

| File | Description |
|------|-------------|
| `data/pathc_search_gate3_original.json` | Path C best-first search on gate3_lemma_novelty (0/14) |
| `data/pathc_search_comparison.json` | Path C best-first search on gate3_v2 (5/8 with permissive) |
| `data/pathc_retrieval_results.json` | Contrastive encoder MRR evaluation |
| `data/gate3_fullgraph_result.json` | MCTS+GNN baseline on gate3 (28.6%) |
| `checkpoints/contrastive/lemma_encoder.pt` | Path C contrastive encoder (1.05M params) |
| `data/raw/gate3_lemma_novelty.jsonl` | Original gate3 test set (14 theorems) |
| `data/raw/gate3_v2.jsonl` | Revised gate3 test set (64 theorems, multi-step) |

## Appendix: Detailed Run Configuration

**Path C run on gate3_lemma_novelty**:
- Encoder: CharCNN dual-encoder, 1,053,184 params, hidden_dim=256
- Lemmas: 70,093 unique from 226K proof-step pairs
- Search: max 200 expansions, top_k=3 lemmas, depth_penalty=0.05
- Verification: strict (no permissive accept, final Lean check only)
- Runtime: 37 seconds for 14 theorems

**MCTS baseline** (from gate3_fullgraph_result.json):
- GNN: GAT, 1,118,848 params, 256-dim, 3 layers, 8 heads
- Graph: 116K nodes, 436K edges (full) / 3840 nodes (Algebra subgraph)
- Search: MCTS with PUCT, 500 simulations, Dirichlet noise
- Verification: Lean proof checker, batch mode
