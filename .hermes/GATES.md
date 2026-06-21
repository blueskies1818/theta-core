# theta-core — Honesty Gates for Physics Discovery

Adapted from the original lemma-retrieval gates. Same principles,
applied to physical theory discovery.

## Gate 1: Dimensional Purity
Every generated expression must respect physical dimensions.
m + v = TYPE ERROR (kg vs m/s).
Score = fraction of expressions that pass dimension check.
Threshold: 100% of candidates must be dimensionally valid.

## Gate 2: Constancy Significance
A discovery must be constant within tolerance, not coincidental.
For discovered invariant E:
  Training: std(E) / mean(E) < 0.05 across ALL training scenarios
  Test: std(E) / mean(E) < 0.10 across ALL held-out scenarios

## Gate 3: Generalization
Discovered laws must generalize to unseen scenarios.
At least 1 held-out scenario must validate the invariant.
Score must be > 0.90 on test set.

## Gate 4: Nontriviality
The expression must not be a constant itself.
depth ≥ 2, contains at least 1 dynamic quantity.
Reject: "1" (constant), "m/m" (always 1).

## Gate 5: Mathematical Proof
Numerical constancy must be provable in Lean.
The expression's invariance must hold for ALL initial conditions,
not just the observation set.
