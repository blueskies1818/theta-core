# Neural Model Roadmap — Phases A-D

## Goal
Make the neural model contribute meaningfully to invariant discovery,
moving from "decorative" to "functional."

Current state: `propose_sub_expressions` is deterministic. The model
checkpoint exists (`sub_expr_proposer.pt`) but generates garbage.
All discovery work is done by enumeration + beam search.

---

## Phase A: Train a Seed Scorer

### What
A small neural model that scores sub-expression candidates for relevance.
Given {symbols} + candidate_expression → score (0-1) indicating whether
this candidate is likely to be a useful building block.

### Why
Currently, all combinatoric pairs (products, ratios, squares) are included
as seeds. With 4-6 quantities, this floods the beam search. A scorer can
rank seeds, keeping top-K and dropping noise. This directly reduces
degenerate results (h*I, (R+T)/E_peak) and speeds up beam search.

### Architecture
```
Input:  [symbol_embeddings, candidate_token_ids]
Model:  Small transformer encoder (d_model=64, nhead=4, layers=2)
Output: Single scalar score (sigmoid)
Size:   ~50K parameters, trainable on CPU
```

### Training Data
- **Positive examples**: Sub-expressions from known invariants.
  E.g., {E, lambda} → E*lambda (score 1.0), E/lambda (0.8), lambda^2 (0.5)
- **Negative examples**: Random symbol pairs that don't form invariants.
  E.g., {B, T} → B*T (score 0.0), B/T (0.0)
- **Source**: Generate from clean benchmark runs + random nuisance data.
  5K-10K examples, no physics knowledge needed.

### Integration
In `cross_symbol_wrapper.py`, after `propose_sub_expressions` returns
candidates, score each with the model. Keep top-K seeds (K=8-12).
Replace current score-based filtering (threshold 0.5) with model scoring.

### Success Criteria
- Model scores E*lambda > 0.8 for {E, lambda}, B*T < 0.2 for {E, lambda, B, T}
- Beam search uses 8-12 seeds instead of 27 for 3-symbol sets
- Degenerate results reduced: no more h*I or (R+T)/E_peak
- Clean verify: still 8/8
- Nuisance verify: still 8/8, faster

### Files
- `scripts/training/train_seed_scorer.py` — training script
- `src/math/seed_scorer.py` — inference wrapper
- `src/math/cross_symbol_wrapper.py` — integration (existing, modify)

---

## Phase B: Train a Beam Search Guider

### What
A model that guides tree beam search expansion. Given (left_expr, right_expr,
operator) → score indicating whether this composition is worth exploring.

### Why
Tree beam search currently tries ALL operator+operand combinations
(O(n² × ops) per depth). Most are pruned by var-set gates and quality checks,
but they're still evaluated against data (expensive). A guider predicts
which compositions are productive BEFORE data evaluation.

### Architecture
```
Input:  [left_embedding, right_embedding, operator_embedding]
Model:  Small MLP or transformer (d_model=64, layers=2)
Output: Score (sigmoid) — higher = more likely to produce a high-constancy
        composed expression
Size:   ~30K parameters
```

### Training Data
- **Positive**: Compositions that produced high-constancy results in
  previous runs. E.g., `(c*t) + (-(x^2))` → score 1.0 (produced spacetime)
- **Negative**: Compositions that were pruned or produced low constancy.
  E.g., `(B) + (T)` → score 0.0
- **Source**: Log beam search expansions from clean benchmark runs.
  Cache which compositions led to discoveries.

### Integration
In `tree_beam_search.py`, before calling `evaluator.score()` on a
candidate, check the guider score. If below a threshold (e.g., 0.3),
skip without evaluating against data. This gives ~10x speedup on
beam search with nuisance variables.

### Success Criteria
- Beam search explores 50-70% fewer candidate expressions
- Same discovery results (8/8 clean, 8/8 nuisance)
- Velocity addition (currently 3.2s) drops to <1s
- No regression on held-out P*V/T discovery

### Files
- `scripts/training/train_beam_guider.py` — training script
- `src/math/beam_guider.py` — inference wrapper
- `src/math/tree_beam_search.py` — integration (existing, modify)

---

## Phase C: Tree-Based Expression Decoder

### What
Replace the token-by-token flat transformer decoder with a model that
generates AST nodes directly. Instead of predicting characters (E, *, l,
a, m, b, d, a), predict tree operations (BinaryOp(*, VarNode(E),
VarNode(lambda))).

### Why
The token-by-token approach failed because the model must learn both
expression grammar AND symbol relationships simultaneously. With 171
tokens in the vocabulary, the decoder can't learn valid expression
structure from 50K examples. A tree decoder constrains output to valid
ASTs by construction — it CANNOT generate garbage like "u-u^2/".

### Architecture
```
Encoder: Same transformer encoder as current (maps symbols → embeddings)
Decoder: Tree-based autoregressive decoder
  - At each step, predict: (operation, left_child_id, right_child_id)
  - Operations: MAKE_VAR, MAKE_CONST, BINARY_OP(+/−/*/ / /^)
  - Tree is built bottom-up from leaf nodes
  - Grammar constraints enforced at each step (can't divide by zero, etc.)
Size: ~100K parameters, requires GPU for training (batch tree operations)
```

### Training Data
- Same structural training data as sub_expr_proposer (50K examples)
- Each example: (symbols, AST tree)
- Model learns to map {E, lambda} → BinaryOp(*, VarNode(E), VarNode(lambda))

### Integration
Replace `propose_sub_expressions` entirely. Model generates valid
sub-expressions as ASTs, decoder enforces grammar, output is always
a parseable expression.

### Challenges
- Tree batching is harder than sequence batching (variable-sized trees)
- GPU training required (Intel Arc B70 available but display-only)
- May need to train on a cloud GPU or accept slower CPU training
- This is the highest-risk, highest-reward approach

### Success Criteria
- Model NEVER generates invalid expressions (grammar-enforced)
- Generates E*lambda, c*t, h*nu, K_max/nu reliably
- Generates (c*t)^2, not just c*t (multi-depth composition)
- Can propose novel sub-expressions not in training data

### Files
- `scripts/training/train_tree_decoder.py` — training script
- `src/math/tree_decoder.py` — model + inference
- `src/math/cross_symbol_wrapper.py` — integration

---

## Phase D: Invariant Type Classifier

### What
A small classifier that predicts which structural pattern an invariant
likely follows. Given {E, p} → predict "squared-difference." Given
{E, lambda} → predict "product." The search then focuses on that one
pattern.

### Why
The beam search tries all patterns (products, ratios, sums, differences,
powers) for every candidate. Knowing the pattern upfront narrows the
search to O(1) pattern × O(n²) symbols instead of O(patterns × n²).

### Architecture
```
Input:  [symbol_embeddings]
Model:  Small MLP (d_model=64, 2 layers)
Output: Softmax over pattern types:
        [product, ratio, sum, difference, squared-diff, linear-combo, power]
Size:   ~10K parameters, trivial on CPU
```

### Training Data
- Labels derived from the structural form of known invariants
- {E, lambda} → product (E*lambda = constant)
- {E, p} → squared-diff (E²-p² = constant)
- {c, t, x} → squared-diff ((c*t)²-x² = constant)
- Negative: random symbol sets → uniform/unknown

### Integration
After classification, the beam search only composes with the predicted
pattern's operators. E.g., "product" → only try * and / operators.
"squared-diff" → only try ^2, -, and + combinations.

### Limitations
- Only works for known pattern types (7 classes)
- Fails on genuinely novel invariant structures
- Narrow — doesn't help with within-pattern symbol selection
- Best combined with Phase A (scorer) for within-pattern ranking

### Success Criteria
- Classifies E*lambda as product, E²-p² as squared-diff with >90% accuracy
- Beam search 3-5x faster for classified patterns
- No regression: still finds all 8 claims

### Files
- `scripts/training/train_pattern_classifier.py` — training script
- `src/math/pattern_classifier.py` — inference wrapper
- `src/math/cross_symbol_wrapper.py` — integration

---

## Execution Order

1. **Phase A** — train scorer, integrate, verify. Target: 1 session.
2. **Phase B** — depends on A's scorer for seed quality.
   Train guider, integrate, verify. Target: 1-2 sessions.
3. **Phase D** — quick win, can run in parallel with B.
   Target: 1 session.
4. **Phase C** — long-term, GPU-dependent, highest risk.
   Target: 2-4 sessions. Only after A+B+D are stable.

---

## Concerns (June 2026)

After completing Phases A+B, remaining issues for independent review:

1. **Models trained on synthetic patterns, not learned from data.** The seed
   scorer and beam guider were told "a*b is good" — they didn't discover it.
2. **System only finds pre-enumerated structural types.** Product, ratio, sum,
   squared-difference. Nothing novel emerges from the search.
3. **Models are frozen at training time.** They don't improve as the system
   processes more experiments. No online learning.

These are architectural limitations, not implementation gaps. Phase C (tree
decoder) addresses #1 and #2 by enabling genuinely novel structure proposal.

---

## Current State (June 2026)

| Component | Status |
|-----------|--------|
| Proposer | Deterministic, generates 11-27 seeds |
| Beam search | O(n²) expansion, pruned by var-set gates |
| Neural model | Checkpoint exists, generates garbage |
| Scoring | Constancy-based (evaluator against data) |
| Seed filtering | Score threshold 0.5 or memory > 0 |

| Phase | Status |
|-------|--------|
| A: Seed Scorer | 🟢 Complete | 2026-06-25 |
| B: Beam Guider | 🟢 Complete | 2026-06-25 |
| C: Tree Decoder | 🟢 Complete | 2026-06-26 |
| D: Pattern Classifier | 🔴 Not started |
