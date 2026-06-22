# Pivot Decision: GNN+MCTS Architecture Ceiling Confirmed

**Date**: 2026-06-17
**Decision**: **PIVOT** — abandon current GNN+MCTS architecture for lemma-novelty generalization
**Author**: implementer (Gate 3 full-graph evaluation)
**Trigger**: Gate 3 full-graph GNN score 28.6% < 45% MARGINAL threshold
**Prior synthesis**: architecture_decision.md (2026-06-16)

## Executive Summary

The Gate 3 full-graph re-run was the definitive test: rebuild the dependency graph
with all 84 Mathlib4 domains (116K nodes, 436K edges), pretrain GNN embeddings on
the complete graph, fine-tune GoalEncoder on gate2 theorem data, and test on 14
gate3 lemma-novelty theorems. **The GNN scored 28.6% on the Algebra subgraph
(3840 nodes) and 0% on the full graph (116K nodes).**

This is 16.4 percentage points below the 45% MARGINAL threshold and only +7.2pp
above the prior Gate 3 baseline (21.4%) achieved with a partial graph. The full
graph — with 99% more nodes, 181% more edges, all gate3-required lemmas confirmed
present, and GoalEncoder fine-tuning — provided minimal improvement. On the full
116K graph, the GNN degrades to 0% because it cannot discriminate specific lemmas
from 116K candidates.

**The synthesis was correct. The ceiling is real. We pivot.**

## The Final Evidence

### Gate 3 Full-Graph Result (2026-06-17)

| Metric | Value | Notes |
|--------|-------|-------|
| GNN H=0.0 (Algebra subgraph, 3840 nodes) | 28.6% (4/14) | Beats prior 21.4%, beats heuristics 21.4% |
| GNN H=0.0 (Full graph, 116K nodes) | 0% (0/14) | 116K candidates overwhelm embeddings |
| Heuristic H=1.0 | 21.4% (3/14) | Same as prior |
| Shape-matcher (simp/norm_num) | 42.9% (6/14) | Exceeds 5% bound — gate3 theorems too simple |
| All GNN proofs | `simp` only | Zero multi-step proofs recovered |
| Theorems never proved (either) | 8 of 14 | poly_derivative_X_pow, poly_degree_C_mul_X_pow, etc. |

### Cumulative Evidence Across All Gate 3 Studies

| Study | Date | Approach | Gate 3 Score | Δ vs Baseline | ≥45%? |
|-------|------|----------|-------------|---------------|-------|
| Baseline (partial graph) | 2026-06-10 | 1.1M GNN, partial graph | 21.4% | — | No |
| H1 50M capacity | 2026-06-16 | 50M-param GNN | 28.6% | +7.2pp | No |
| H2 scoring archs | 2026-06-16 | Two-tower, cross-attn, graph-filtered | 0% MRR | −78.6pp | No |
| H3 traversal reward | 2026-06-16 | Graph-traversal bonus in GRPO | 7.1% | −14.3pp | No |
| **Full graph (this study)** | **2026-06-17** | **Complete 116K graph + SFT** | **28.6% (subgraph) / 0% (full)** | **+7.2pp** | **No** |

**Five independent studies. Zero exceed 45%. Maximum achieved: 28.6%.**

### Key Observations from Full-Graph Study

1. **Complete data doesn't help**: Even with all gate3-required lemmas
   (derivative_X_pow, natDegree_C_mul_X_pow, degree_add_eq_left_of_degree_lt)
   confirmed present in the graph, the GNN cannot retrieve them from 3840+
   candidates.

2. **GoalEncoder SFT is marginal**: Fine-tuning the GoalEncoder on gate2
   training data improved validation top-1 retrieval from ~4.4% to 8.3%,
   but this translated to only +7.2pp end-to-end — less than one additional
   theorem proven.

3. **Full graph degrades to zero**: When all 116K lemma candidates are
   available, the GNN cannot discriminate at all. The embedding space does
   not encode proof-relevant lemma features at the required resolution.

4. **Shape-matcher exposes theorem simplicity**: 43% of gate3 theorems are
   provable with `simp` alone. The gate3 benchmark is not effectively testing
   lemma-novelty — it's testing whether the proof can be closed by Mathlib4's
   simp set. This weakens the "lemma-novelty" framing of the test.

5. **Single-tactic ceiling persists**: Across all studies, zero multi-step
   proofs have been discovered. The architecture cannot chain tactics.

## Why This Triggers a Pivot

The Gate 3 full-graph study was the best-case scenario for the current
architecture:

- ✅ **Best possible data**: All 116K lemmas from all 84 Mathlib4 domains
- ✅ **Best possible training**: GoalEncoder SFT on gate2 theorem data
- ✅ **Best possible test**: Lemma-novelty theorems where required lemmas exist in graph
- ✅ **45× capacity scaling attempted** (H1): 1M → 50M GNN params
- ✅ **Alternative scoring architectures attempted** (H2)
- ✅ **Alternative reward signals attempted** (H3)

**Result: 28.6% maximum. Zero multi-step proofs. Zero full-graph capability.**

The architecture's failure is structural, not parametric. Three root causes
identified across all studies:

### Root Cause 1: Binary reward is too sparse for proof search
GRPO provides one reward per full proof trajectory (success=1.0, failure=0.0).
The GNN cannot learn which lemma choices within a trajectory contributed to
success or failure. This is a credit assignment problem that no amount of
parameter scaling or reward bonus can fix — the signal simply doesn't exist.

### Root Cause 2: Single-action MCTS cannot compose proofs
MCTS with PUCT evaluates states greedily — it scores candidate lemmas for the
immediate goal but has no mechanism to plan multi-step trajectories. The value
network (if it existed) would need to predict proof completability, not lemma
relevance. Without step-level planning, the system is a tactic selector, not
a theorem prover.

### Root Cause 3: GNN embeddings don't encode proof utility
Cosine similarity retrieves correct lemmas at 78.6% Top-1 for known theorems,
but this doesn't translate to retrieval accuracy on novel theorems. The GNN
embeddings encode lemma co-occurrence in the dependency graph — not lemma
utility for a given proof state. When the proof goal requires a specific lemma
from a different Mathlib domain (e.g., polynomial lemmas for a ring theory
goal), the GNN has no signal.

## The Decision: PIVOT

**We abandon the GNN+MCTS+GRPO architecture for lemma-novelty generalization.**
The architecture is not fundamentally broken — it works for tactic selection
on in-distribution problems — but it cannot generalize to lemma-novelty proofs
and shows no path to doing so.

### What stays (preserved for future use)

| Component | Status | Rationale |
|-----------|--------|-----------|
| Dependency graph (116K nodes, 436K edges) | **PRESERVE** | Valuable knowledge representation; architecture-agnostic |
| Proof checker interface (Lean 4 subprocess) | **PRESERVE** | Robust, well-tested, architecture-agnostic |
| GNN embeddings (full_graph_pretrained.pt) | **ARCHIVE** | Useful for retrieval baselines but not as primary architecture |
| Correspondence layer | **PRESERVE** | Orthogonal to proof architecture |
| Theorem sets (gate2, gate3) | **PRESERVE** | Reusable benchmarks for any architecture |
| Training pipeline (scripts/training/train_explorer.py) | **ARCHIVE** | Specific to current architecture; reference only |

### What we pivot away from

| Component | Reason |
|-----------|--------|
| GAT-GNN encoder as sole lemma scorer | Cannot discriminate 116K candidates |
| Single-action MCTS with PUCT | Cannot compose multi-step proofs |
| Binary GRPO reward on full trajectories | Credit assignment impossible |
| Heuristic annealing (MCTS-specific) | Architecture-specific; no reuse |

## Alternative Architecture Paths

The task requires identifying what to pivot TO (no implementation — design only).
Based on the root cause analysis, three architectural directions are candidates:

### Path A: Dense Reward + Learned Value Function
Keep MCTS but replace binary GRPO with step-level rewards:
- **Step validity reward**: Each tactic application verified by Lean = +0.1
- **Goal proximity reward**: Embedding distance between current and target goal
- **Learned value network**: Predict proof completability from proof state
- **Requires**: State-value training data from proof trajectories

**Rationale**: Addresses Root Cause 1 (sparse reward). Step-level rewards provide
gradient for lemma selection within trajectories. Value network enables deeper
search without requiring end-to-end proof success as the only signal.

**Risk**: Still uses GNN lemma scoring; may inherit the same embedding limitations.

### Path B: Transformer-Guided Proof Generation (Seq2Seq)
Replace GNN+MCTS with autoregressive proof generation:
- **Encoder**: Transformer that encodes goal + available lemmas
- **Decoder**: Autoregressively generates tactic sequence ([lemma₁, lemma₂, ..., qed])
- **Training**: Behavioral cloning on Mathlib proof trajectories, then RL fine-tuning
- **Inference**: Beam search over tactic sequences with proof-checker verification

**Rationale**: Addresses Root Cause 2 (multi-step composition). Seq2seq models
natively handle sequence generation and can learn proof patterns from Mathlib.
Transformer attention can cross-attend between goals and lemmas, providing
lemma-level discrimination (partial fix for Root Cause 3).

**Risk**: Requires substantial Mathlib proof trajectory extraction. Seq2seq
models may overfit to training proof patterns. Beam search may be expensive.

### Path C: Contrastive Lemma Embedding + Best-First Search
Replace cosine-similarity lemma retrieval with learned relevance:
- **Contrastive pretraining**: (goal, correct_lemma) as positive pairs,
  (goal, random_lemma) as negatives — learn embeddings that encode proof utility
- **Best-first search**: Expand most promising proof state (scored by value
  network), select lemma by contrastive embedding similarity
- **Reward**: Step-level proof progress + final proof success

**Rationale**: Addresses Root Cause 3 (embeddings don't encode proof utility).
Contrastive learning on proof-step pairs from Mathlib would teach the model
which lemmas are useful for which goals, rather than which lemmas co-occur
in the dependency graph. Best-first search is simpler than MCTS and may
perform better when lemma scoring is accurate.

**Risk**: Requires extracting (goal, lemma_used) pairs from Mathlib proofs.
Best-first search may still fail on long proof chains.

### Path D: Hybrid — Dense Reward Transformer + Contrastive Retrieval
Combine Paths A, B, and C:
- Contrastive lemma embeddings for retrieval (Path C)
- Transformer-guided proof generation with step-level rewards (Path B + Path A)
- Learned value function for beam search pruning (Path A)
- Proof-checker verification as ground-truth signal

**Rationale**: Addresses all three root causes simultaneously. Most ambitious
but also most likely to demonstrate multi-step lemma-novelty proofs.

**Risk**: Highest implementation complexity. Multiple components must work
together. Training data requirements are largest.

## Recommendation

**Preferred path: Path C (Contrastive Lemma Embedding + Best-First Search)**
as the minimal viable pivot — it directly addresses the embedding quality
root cause (the most fundamental limitation) and has the lowest implementation
complexity.

**If Path C shows lemma-novelty improvement**: Layer on Path A (dense rewards)
to enable multi-step proof chaining.

**If multi-step proofs are demonstrated**: Evaluate Path B (seq2seq) as a
more scalable architecture, but only after the embedding problem is solved.

**If resources permit**: Path D is the most promising long-term architecture
but should not be attempted until Paths A and C are independently validated.

## Gate 3 Benchmark Limitations

The Gate 3 evaluation itself has revealed weaknesses that should inform
future benchmarking:

1. **Shape-matcher at 43% invalidates lemma-novelty framing**: When 6 of 14
   "lemma-novelty" theorems are provable with `simp` alone, the benchmark is
   not effectively testing whether the model can retrieve novel lemmas.

2. **Gate 3 theorems are too simple**: Polynomial identities with
   physics-sounding names (poly_eval_C_add_X, poly_derivative_add) are closer
   to algebra exercises than physics theorem proving.

3. **Future benchmarks need**: Multi-step proof requirements (≥2 tactics),
   genuine lemma novelty (lemmas not derivable from simp/norm_num/ring),
   and larger eval sets (50+ theorems) for statistical reliability.

## Next Steps (Implementation NOT in this task)

1. Extract (goal, lemma_used) pairs from Mathlib4 proof trajectories for
   contrastive pretraining
2. Build contrastive embedding model (Path C) with evaluation on gate3
3. Design step-level reward function (Path A) with proof-state value targets
4. Create new benchmark set (gate4) with multi-step, genuine lemma-novelty
   requirements
5. Run controlled comparison: contrastive + best-first vs. current GNN+MCTS

## Appendix: Full Study Data

| File | Contents |
|------|----------|
| `data/gate3_fullgraph_result.json` | Gate 3 full-graph evaluation (this study) |
| `data/h1_capacity_results.json` | H1: GNN capacity scaling (1M/5M/10M/50M) |
| `data/h2_scoring_results.json` | H2: Lemma scoring architecture comparison |
| `data/h3_traversal_results.json` | H3: Traversal reward experiment |
| `data/gate3_result.json` | Gate 3 baseline (partial graph) |
| `docs/reviews/architecture_decision.md` | H1/H2/H3 synthesis (prior decision doc) |
| `docs/reviews/roadmap_review_june2026.md` | Phase 2 roadmap review |
| `checkpoints/gnn/full_graph_pretrained.pt` | Full-graph pretrained GNN embeddings |
| `checkpoints/gnn/gate2_fullgraph_finetuned.pt` | GoalEncoder SFT on gate2 |
