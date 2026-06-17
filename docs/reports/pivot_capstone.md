# Pivot Capstone: Gates 1-5 with New Architecture

**Date:** 2026-06-17  
**Author:** implementer (capstone synthesis)  
**Architecture:** Contrastive CharCNN embeddings + Best-first search + Dense rewards  
**Benchmark:** gate3_v2 (64 multi-step theorems, 5 domains, 21 tactic categories)  
**Parent tasks:** t_8bf182a4 (gate3_v2 benchmark), t_ec8b3351 (dense rewards)  
**Prior pivot decisions:** pivot_decision.md, pathc_evaluation.md  

## Overall Verdict: FAIL

**v1.0 cannot be tagged.** The new architecture (contrastive embeddings + best-first
search + dense rewards) underperforms the old GNN+MCTS baseline on every gate where
comparison is possible. The root cause is broken lemma retrieval: MRR 0.334
(vs 0.786 baseline) — a 58% degradation. The best-first search architecture is
superior to MCTS for proof composition, but it cannot compensate for the embedding
quality gap. Dense rewards cannot contribute meaningfully when the search cannot
find correct lemmas.

**Gates passed: 1 of 5 (Gate 1 only).**

---

## Gate 1: Infrastructure Validation — PASS

**Definition:** Core system components are functional — tests pass, dependency graph
builds, proof checker operates, training data pipeline works, Lean toolchain available.

### Results

| Component | Status | Evidence |
|-----------|--------|----------|
| Unit tests | 80/83 pass (96.4%) | 3 Lean failures are environmental (elan toolchain not default-configured) |
| Dependency graph | 116K nodes, 436K edges | Builds from 84 Mathlib4 domains |
| Proof checker | Functional | Batch checker, SHA-256 cache, 30s timeout |
| Proof-step pairs | 226K pairs extracted | 27 domains, 4.5x target of 50K |
| Contrastive encoder | 1.05M params, trains | CharCNN dual-encoder, InfoNCE loss, 30 epochs |
| Best-first search | Functional | Priority queue, depth penalty, structural tactic priority |
| Dense reward tracker | Functional | Step validity +0.1, goal proximity, completion bonus +1.0 |
| gate3_v2 benchmark | 64 theorems | 5 domains, 21 tactic categories, 0 simp-solvable |

**Verdict: PASS.** All infrastructure components are operational. The Lean
failures in 3 of 83 tests are a pre-existing environment configuration issue
(`elan default stable` not set), not a code defect.

---

## Gate 2: Structural Independence — FAIL (new architecture degraded)

**Definition:** The system must not succeed by matching surface-level theorem
shapes to training data. The shape-matcher (closest training theorem → copy its
proof) must have match rate ≤ allowed threshold (random baseline + margin).

### Old Architecture Result (GNN+MCTS)

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Shape-matcher match rate | 4.55% (1/22) | ≤ 61.66% | PASS |
| Random baseline | 56.66% | — | — |
| Lean verification | 82/82 pass | — | — |

**Source:** `data/gate2_audit_result.json` (2026-06-16, commit 06e33a7)  
**Verdict (old): PASS**

### New Architecture Result (Contrastive + Best-first + Dense)

The structural independence test was NOT directly re-run with the new
architecture. The gate2_audit measures whether the proof system overfits to
surface patterns in training theorems — it compares test theorem proofs against
training theorem proofs by string similarity.

However, the new architecture's **end-to-end proof success rate on gate2 theorems**
provides a relevant signal:

| Metric | Old GNN+MCTS | New Arch (Contrastive+Best-first+Dense) |
|--------|-------------|------------------------------------------|
| Gate2 proof rate | 56% (best at H=0.0) | 23.3% (14/60) |
| Multi-step proofs | 0 | 0 |
| Mean steps | 1.0 | 1.43 |
| Mean total reward | N/A (binary) | 0.408 |

**Source:** `data/patha_dense_results.json` (t_ec8b3351, 2026-06-17)

**Verdict (new): FAIL (indirect).** The structural independence gate was designed
for the GNN+MCTS architecture. With the new architecture, the test needs to be
redefined. The proof rate degradation (56% → 23%) indicates the new architecture
is less effective on in-distribution theorems, which undermines confidence in
its generalization. The structural independence question is secondary to the
fundamental retrieval failure documented in Gate 3.

---

## Gate 3: Lemma Novelty — FAIL

**Definition:** The system must prove theorems requiring unseen lemma combinations.
For the new architecture, this means: (a) retrieval MRR exceeds baseline,
(b) proof success ≥ threshold on lemma-novelty theorems,
(c) proofs use lemmas (not structural tactics).

### Old Architecture Baseline (GNN+MCTS)

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| GNN H=0.0 (Algebra subgraph, 3840 nodes) | 28.6% (4/14) | ≥ 45% | FAIL |
| GNN H=0.0 (Full graph, 116K nodes) | 0% (0/14) | ≥ 45% | FAIL |
| Shape-matcher (simp/norm_num) | 42.9% (6/14) | ≤ 5% | FAIL |
| MRR (cosine, GNN embeddings) | 0.786 | — | — |
| Multi-step proofs | 0 | > 0 | FAIL |

**Source:** `data/gate3_fullgraph_result.json`, `data/gate3_result.json`

### New Architecture (Contrastive + Best-first + Dense)

#### On gate3_lemma_novelty (14 theorems, Algebra only)

| Metric | Old (GNN+MCTS) | New (Contrastive+Best-first) | Δ |
|--------|---------------|------------------------------|---|
| Proof success | 28.6% (4/14) | 0% (0/14) | −28.6pp |
| MRR | 0.786 | 0.334 | −0.452 (−57.5%) |
| Top-1 accuracy | — | 33.3% | — |
| Multi-step proofs | 0 | 0 | — |

**Source:** `data/pathc_search_gate3_original.json` (t_db871788, 2026-06-17)

All 14 theorems failed because the search chose wrong tactics:
- 9/14 chose `exact` with a hypothesis variable (`exact p`, `exact n`)
- 2/14 chose `ring` (wrong for evaluation/degree goals)
- 3/14 chose lemmas that failed type-checking

#### On gate3_v2 (64 theorems, 5 domains)

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Proof success | 0% (0/64) | ≥ 45% | FAIL |
| Multi-step proofs | 0 | > 0 | FAIL |
| Mean total reward | 0.194 | — | — |
| Mean steps attempted | 1.67 | — | — |
| MRR (retrieval) | 0.079 | > 0.786 | FAIL |
| Top-1 accuracy | 3.5% | — | — |

**Source:** `data/patha_dense_results.json` (t_ec8b3351, 2026-06-17)

**Verdict: FAIL.** The contrastive CharCNN encoder with InfoNCE in-batch training
(batch_size=256, 255 negatives) cannot discriminate among 70K lemma candidates.
MRR drops from 0.786 to 0.334 on the small lemma-novelty set and to 0.079 on
gate3_v2 — effectively random among 70K lemmas. Dense rewards cannot contribute
because the search never reaches correct lemmas to reward.

### Root Cause: Broken Embedding Quality

The contrastive encoder fails for three reasons:

1. **In-batch negatives insufficient:** InfoNCE with 255 negatives per batch
   cannot cover the 70K lemma space (<0.4% coverage).
2. **CharCNN too weak:** Character-level n-gram CNNs lack semantic understanding.
   Lemma names like `natDegree_C_mul_X_pow` and `degree_add_eq_left_of_degree_lt`
   share n-grams but have different meanings. The CharCNN cannot distinguish them.
3. **Training signal noisy:** Many extracted proof-step pairs use trivial lemmas
   (add_comm, mul_assoc) that provide no discriminative signal. Validation accuracy
   plateaued at 29.5% by epoch 15.

**This is a structural failure, not a parametric one.** The embedding approach
(CharCNN + InfoNCE) is fundamentally incapable of the required discrimination.
Fixing it requires either: (a) a memory bank with 4096+ negatives, (b) pretrained
LM embeddings (CodeBERT), or (c) a different architecture (seq2seq, Path B).

---

## Gate 4: Negative Control (Era-Gated Discovery) — FAIL

**Definition:** Two models trained on disjoint era-separated data must show
era-specific learning — GNN-A (trained on pre-1905 continuous-assumption
theorems) should outperform GNN-B on continuous-era test theorems, and GNN-B
(trained on post-1925 quantized-assumption theorems) should outperform GNN-A
on quantized-era test theorems. Statistical significance required: p ≤ 0.05
with correct interaction direction.

### Old Architecture Result (GNN+MCTS)

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| GNN-A overall | 65.0% | — | — |
| GNN-B overall | 65.0% | — | — |
| Interaction direction | Wrong (both models same) | GNN-A ↑ continuous, GNN-B ↑ quantized | FAIL |
| Interaction magnitude | 0.0pp | > 0 | FAIL |
| p-value | 0.9255 | ≤ 0.05 | FAIL |

**Source:** `docs/reports/gate4_analysis.md`, `data/gate4_result.json`

Both GNNs scored identically: 70.0% on continuous, 60.0% on quantized. No
era-specific learning detected.

### New Architecture Result

**Not evaluated.** The new architecture uses a single contrastive encoder trained
on all proof-step pairs — it does not have the two-model era-separated training
design that Gate 4 requires. Running Gate 4 with the new architecture would
require:

1. Splitting the 226K proof-step pairs into era-separated subsets
2. Training two separate contrastive encoders
3. Running best-first search with each encoder on the mixed test set
4. Computing the interaction effect

This evaluation was NOT performed because:
- The contrastive encoder already fails at basic lemma retrieval (Gate 3: MRR 0.334)
- Era separation would further reduce training data per encoder
- The underlying retrieval failure makes era-specific learning impossible
- The gate4 theorems are mostly single-tactic — era differences are subtle

**Verdict: FAIL (by proxy).** The new architecture cannot pass Gate 4 because
it cannot retrieve correct lemmas for ANY theorem (Gate 3: 0/64). Era-specific
learning requires basic lemma retrieval, which is absent. Running the full Gate 4
experiment would consume hours of CPU time to confirm what Gate 3 already
demonstrates.

### Note on Gate 4 Theorem Quality

The audit_structural script found that 51 of 60 physics theorems had
identical proof patterns (shape similarity = 1.0) to their training set
counterparts. The theorems use different names but the same proof tactics
(`rfl`, `ring`, `linarith`). For genuine negative control, the theorem pairs
need genuinely different proof strategies — not the same math tactics with
different labels.

---

## Gate 5: Multi-Step Proof Capstone — FAIL

**Definition:** The system must demonstrate multi-step proof chaining (≥2
distinct tactics in a single proof) on at least one lemma-novelty theorem.
This is the capstone: combining retrieval + search + reward to compose proofs
that no single heuristic can solve.

### Results

| Evaluation | Multi-step proofs | Context |
|------------|------------------|---------|
| Best-first on gate3_v2 (10 theorems, permissive) | 2 (alg_subst_expand, alg_subst_factor) | `rw[h]; ring` — hypothesis rewriting + ring automation |
| Best-first on gate3_lemma_novelty (14, strict) | 0 | All 14 failed |
| Dense rewards on gate2 (60 theorems) | 0 | 14/60 proved, all single-step |
| Dense rewards on gate3_v2 (64 theorems) | 0 | 0/64 proved |
| MCTS+GNN on gate3_lemma_novelty (14) | 0 | 4/14 proved, all `simp` |

**Source:** `data/patha_dense_results.json`, `data/pathc_search_comparison.json`

### The Two Multi-Step Proofs: Structural, Not Lemma-Novelty

The best-first search found 2 multi-step proofs on gate3_v2:

```
alg_subst_expand: rw [h]; ring
alg_subst_factor: rw [h]; ring
```

These are real multi-step proofs — two distinct tactics chained together.
However:
- They use hypothesis variables (`h`), not lemmas from the library
- `ring` is structural automation, not lemma-novelty retrieval
- The contrastive embeddings contributed nothing — the search architecture
  (structural tactic priority: rw=0.75, ring=0.70) found these independently

These multi-step proofs demonstrate that the best-first search architecture
can compose proof steps — something MCTS never achieved. But they do not
demonstrate lemma-novelty because the contrastive encoder's lemma scores
were irrelevant to the success.

### Dense Rewards: No Impact

The step-level dense reward system works correctly in isolation:
- Step validity: +0.1 per valid tactic (verified post-hoc)
- Goal proximity: embedding distance from target goal
- Completion bonus: +1.0 for successful proofs

But with 0/64 theorems proved on gate3_v2, the completion bonus never triggers.
Step validity is always +0.1 because step-level verification is permissive
(doesn't require goal closure). The dense reward signal is flat — it cannot
guide search improvement when no proof ever completes.

**Verdict: FAIL.** The new architecture has demonstrated multi-step composition
capability in the best-first search (2 proofs, `rw[h]; ring`), but cannot
replicate this on lemma-novelty theorems because the retrieval is broken.
The capstone requires lemma-novelty multi-step proofs, which are impossible
when the system cannot retrieve correct lemmas (MRR 0.079).

---

## Architecture Comparison: Old vs New

| Component | Old (GNN+MCTS+GRPO) | New (Contrastive+Best-first+Dense) | Winner |
|-----------|---------------------|-------------------------------------|--------|
| Lemma retrieval MRR | 0.786 (GNN cosine) | 0.079-0.334 (CharCNN contrastive) | **OLD** |
| Proof success (gate3) | 28.6% (4/14) | 0% (0/14) | **OLD** |
| Proof success (gate2) | 56% | 23% | **OLD** |
| Multi-step proofs | 0 | 2 (structural only) | **NEW** |
| Search architecture | MCTS PUCT | Priority-queue best-first | **NEW** |
| Reward signal | Binary (sparse) | Step-level dense | **NEW** |
| Training data | 55 theorems | 226K proof-step pairs | **NEW** |

**The best-first search is better than MCTS. The contrastive embeddings are
worse than GNN cosine similarity. The net result is worse than the baseline.**

---

## What Works (Preserve)

| Component | Evidence |
|-----------|----------|
| Best-first search | 2 multi-step proofs MCTS never found; priority queue with structural tactic priority is correct design |
| Proof-step pair extraction | 226K pairs from 27 Mathlib4 domains — reusable training asset |
| Dense reward mechanism | Correctly tracks step validity, goal proximity, completion bonus |
| gate3_v2 benchmark | 64 multi-step theorems, 0 simp-solvable, 21 tactic categories — much stronger test than original gate3 |
| Infrastructure | Tests pass, graph builds, checker works |

## What Does NOT Work (Abandon or Redesign)

| Component | Evidence |
|-----------|----------|
| CharCNN contrastive encoder | MRR 0.08-0.33 vs 0.79 baseline — cannot discriminate 70K lemmas |
| InfoNCE in-batch training (batch=256) | 255 negatives insufficient for 70K candidate space |
| Lemma scoring as primary action | All 14 gate3 theorems chose wrong tactics |
| Dense rewards without retrieval | Cannot contribute when 0/64 theorems prove successfully |

---

## Next Steps

### Immediate (fix retrieval, preserve search)

1. **Replace CharCNN with stronger embeddings:**
   - Option A: Pretrained LM (CodeBERT, StarEncoder) — semantic understanding
   - Option B: GNN embeddings as lemma scorer in best-first search — hybrid approach
     (GNN MRR 0.786 on Algebra subgraph + best-first search for multi-step)
   - Option C: MoCo-style memory bank with 4096+ negatives for contrastive training

2. **Hybrid approach (Option 4 from pathc_evaluation.md):**
   - Use GNN cosine similarity for lemma scoring (0.786 MRR on Algebra subgraph)
   - Use best-first search for proof composition (found multi-step proofs MCTS missed)
   - Combine the best component from each architecture
   - Risk: GNN degrades to 0 MRR on full 116K graph; candidate filtering needed

### Medium-term (if retrieval fixed)

3. **Re-run dense rewards** after retrieval is operational
4. **Redesign Gate 4 theorems** — currently all share identical proof shapes
5. **Scale gate3_v2** — 64 theorems is good but 200+ needed for statistical reliability

### Long-term (new architecture)

6. **Path B: Transformer-guided proof generation** — seq2seq with cross-attention
   between goals and lemmas, trained on 226K proof-step pairs
7. **Path D: Full hybrid** — contrastive retrieval + transformer generation +
   step-level rewards + proof-checker verification

---

## Data Files Referenced

| File | Content |
|------|---------|
| `data/gate2_audit_result.json` | Gate 2 structural independence (old arch) |
| `data/gate3_result.json` | Gate 3 baseline (old arch) |
| `data/gate3_fullgraph_result.json` | Gate 3 full-graph evaluation |
| `data/gate4_result.json` | Gate 4 negative control (old arch) |
| `data/patha_dense_results.json` | Dense rewards on gate2+gate3 (new arch) |
| `data/pathc_retrieval_results.json` | Contrastive encoder MRR evaluation |
| `data/pathc_search_gate3_original.json` | Best-first on gate3_lemma_novelty (0/14) |
| `data/pathc_search_comparison.json` | Best-first on gate3_v2 (5/8 permissive) |
| `data/raw/gate3_v2.jsonl` | Revised gate3 benchmark (64 theorems) |
| `data/raw/gate3_lemma_novelty.jsonl` | Original gate3 benchmark (14 theorems) |
| `checkpoints/contrastive/lemma_encoder.pt` | CharCNN contrastive encoder (1.05M params) |
| `docs/reviews/pivot_decision.md` | Original pivot decision |
| `docs/reviews/pathc_evaluation.md` | Path C evaluation |
| `docs/reports/gate4_analysis.md` | Gate 4 analysis (old arch) |
| `docs/reports/gate2_pass_20260616_0229.md` | Gate 2 pass report (old arch) |

---

## Decision: NO v1.0 Release

**The new architecture does not pass Gates 2-5.** The core failure is lemma
retrieval quality — the contrastive CharCNN encoder (MRR 0.334) is significantly
worse than the GNN cosine-similarity baseline (MRR 0.786) it was meant to replace.
All downstream components (best-first search, dense rewards) are sound but cannot
compensate for the embedding failure.

**v1.0 requires at minimum:**
- Gate 3 lemma-novelty proof success ≥ 45% (currently 0%)
- Gate 5 multi-step lemma-novelty proofs > 0 (currently 0)
- Gate 4 era-specific learning with significant interaction (not tested, but
  impossible without Gate 3)

The best-first search architecture and step-level dense reward mechanism are
preserved as reusable components. The contrastive embedding approach needs
fundamental redesign — stronger encoders, larger negative sets, or a different
training paradigm entirely.

*End of capstone report.*
