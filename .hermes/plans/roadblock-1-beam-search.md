# Roadblock #1: Beam Search Anti-Degenerate

## Goal
Make beam search safe for top-level use so the system can discover
multi-term invariants (a*b+c*d, (a+b)/(c-d)) that simple search misses.

Currently beam search is disabled at top level because it produces
high-scoring coincidences — expressions that pass the 6 honesty gates
but don't generalize as genuine invariants.

## Why
Simple search only finds single-term forms (a*b, a/b, a^2, a/b^2, a^2-b^2).
Forms like a*b+c*d, a^2+b^2, (a+b)/c are invisible without beam search.
This is the single biggest gap between "finds things" and "finds everything
in the search space."

## Approach

### Step 1: Cross-validation gate (anti-coincidence)
The core problem: beam search overfits to the train observations. A random
combinatorial expression can score 0.95 on 8 train points by chance.

Fix: Add a train/test split validation inside `ExpressionSearch` or at the
`auto_discover` level. Before accepting a beam search result:
- Split observations 70/30 train/test
- Run beam search on train only
- Re-score the best expression on held-out test observations
- Reject if test score < 0.85 or test/train gap > 0.05

This catches coincidences: a genuine invariant scores consistently on both
splits; a coincidence drops on unseen data.

### Step 2: Integrate beam guider into ExpressionSearch
Phase B's beam guider (94.2% accuracy) predicts which operator compositions
are productive. Currently only used in tree_beam_search.py, not in
ExpressionSearch.run().

Fix: Add optional guider to ExpressionSearch.__init__. Before evaluating
a new candidate against data (expensive), check guider score. Skip if
below threshold (0.2). This prunes ~50% of candidates before the evaluator
sees them, reducing the surface area for coincidences.

### Step 3: Enable beam search at top level with safeguards
After steps 1+2:
- Set `_enable_beam_search=True` as default in auto_discover
- Run beam search AFTER simple search (as fallback for forms simple search misses)
- Apply cross-validation gate on beam search results only (simple search
  already has implicit cross-validation via the template structure)
- Apply all-symbols filter

### Step 4: Verify
- Current 28/28 claims must still pass
- Generate 3 test claims that simple search CANNOT find (a*b+c*d, a^2+b^2, (a+b)/c)
  and verify beam search discovers them
- Confirm no coincidences on random null data (system should return empty)

## Files
- src/physics/search.py — ExpressionSearch cross-validation, guider integration
- src/math/beam_guider.py — existing, load into ExpressionSearch
- REVIEW_READY_ROADMAP.md — mark #1 in progress

## Pitfalls
- Cross-validation split must be random but deterministic (seed per claim)
- Beam guider checkpoint may not exist — graceful fallback to unguded
- Don't break simple search behavior (it works perfectly as-is)
- The 28 existing claims must pass WITHOUT beam search — simple search should
  still handle them
