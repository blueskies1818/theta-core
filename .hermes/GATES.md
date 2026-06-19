# theta-core — Honesty Gates

The system must prove it learned physical concepts, not pattern-matched theorem shapes.
Five sequential gates. All must pass.

Tolerance: ±5% on every threshold. Below tolerance = FAIL, within tolerance = MARGINAL (retest
with larger sample), above tolerance + margin = PASS.

---

## Gate 1: Training Data Purity

**Threshold:** 0 post-1904 data points in ANY training pipeline component.

**Tolerance (5%):** Zero. Purity is binary — any leak = fail, no wiggle room.

| Component | Audit Method |
|-----------|-------------|
| Training theorems | Scan all theorem names, statements, ground-truth proofs for post-1904 keywords (quantum, relativity, Planck, Heisenberg, Schrodinger, Dirac, photon, gauge, quark, etc.) |
| Proof-step pretraining | If used: list all source theorem names. Any post-1904 name = fail. If source can't be filtered, pretraining is turned OFF for this run. |
| Dependency graph | `grep -i` the graph node list for post-1904 keywords. Any hit = fail. |
| Heuristics | Code audit: does any heuristic pattern target a specific test theorem structure? If yes, the pattern must be provably derivable from pre-1904 math only. |
| Reward system | Binary proof-checker output only. Correspondence layer disabled. Verify reward log: all values ∈ {0.0, 1.0}. |

**Verdict:**
- CLEAN: 0 leaks across all components
- FAIL: any leak detected → fix and re-audit

---

## Gate 2: Structural Independence

**Threshold:** Shape-matcher baseline must score ≤ random on test theorems.
GNN must beat shape-matcher by statistically significant margin.

**Tolerance (5%):** Shape-matcher may score up to random_baseline + 5pp without triggering fail.
GNN must beat shape-matcher + 5pp minimum.

| Metric | FAIL | MARGINAL | PASS |
|--------|------|----------|------|
| Shape-matcher on test set | > random + 5pp | random + 0-5pp | ≤ random |
| GNN vs shape-matcher | GNN ≤ shape-matcher + 5pp | GNN > shape-matcher, < +5pp | GNN > shape-matcher + 5pp |

MARGINAL on shape-matcher → redesign weakest theorems and retest.
MARGINAL on GNN margin → increase eval set size and retest.

**Run BEFORE training.** If it fails, redesign theorem set.

---

## Gate 3: Lemma Novelty

**Threshold:** GNN proves ≥50% of lemma-novelty theorems where the required lemma
was never used in training and is only reachable through graph traversal.

**Tolerance (5%):** ≥45% = MARGINAL (retest with more theorems). <45% = FAIL.

| Metric | FAIL | MARGINAL | PASS |
|--------|------|----------|------|
| GNN on lemma-novelty set | <45% | 45-50% | ≥50% |
| Shape-matcher on same set | >5% | — | ≤5% |

Shape-matcher on lemma-novelty is a separate check: if the shape-matcher can
solve ANY lemma-novelty theorems, those theorems aren't truly novel — remove them.

Minimum 10 lemma-novelty theorems required for statistical validity.

---

## Gate 4: Negative Control

**Threshold:** Significant interaction effect between training era and test era.
GNN trained on pre-1905 must score higher on continuous-assumption theorems.
GNN trained on post-1925 must score higher on quantized-assumption theorems.

**Tolerance (5%):** Interaction p-value ≤ 0.05 (strict, standard). Effect size:
GNN-A on continuous must exceed GNN-A on quantized by ≥5pp, and vice versa.

| Metric | FAIL | MARGINAL | PASS |
|--------|------|----------|------|
| Interaction p-value | >0.05 | — | ≤0.05 |
| Era effect size (each direction) | <5pp difference | 0-5pp | >5pp |
| Shape-matcher era difference | >5pp | 0-5pp | ≤0pp (no era effect) |

Minimum 10 theorem pairs per era (40 theorems total: 10 continuous-train, 10 continuous-test,
10 quantized-train, 10 quantized-test).

MARGINAL on effect size: increase theorem pairs and retest.
If the shape-matcher shows ANY era effect, the theorem pairs aren't genuinely era-differentiated.

---

## Gate 5: Statistical Validity

**Threshold:** Results replicable across ≥3 independent training runs.
Reported as mean ± std, not single-run best.

**Tolerance (5%):** Std must be ≤5pp across runs. >5pp variance = measurement noise,
not genuine uncertainty — increase eval set size or MCTS sims.

| Metric | FAIL | MARGINAL | PASS |
|--------|------|----------|------|
| Std across runs | >5pp | 3-5pp | <3pp |
| Min eval theorems | <80 | 80-100 | ≥100 |
| Runs per condition | 1-2 | 2 | ≥3 |

MARGINAL on std: increase sims or eval set, retest.
MARGINAL on sample size: add theorems or runs, retest.

---

## Gate Sequence (Non-Negotiable Order)

```
Gate 1 ──→ Gate 2 ──→ [TRAINING] ──→ Gate 3 ──→ Gate 4 ──→ Gate 5
 (audit)   (verify                              (novelty)  (control)  (stats)
            theorem
            design)
  │          │                                    │          │          │
  ▼          ▼                                    ▼          ▼          ▼
PASS:      PASS:                                PASS:      PASS:      PASS:
  continue   continue                             continue   continue   SYSTEM
                                                                        PROVEN
FAIL:      FAIL:
  fix leak   redesign theorems
```

---

## Reporting Trail

After every gate result, a reporter task fires. The reporter:
1. Writes: `docs/reports/gateN_YYYY-MM-DD_HHMM.md`
2. Captures: git commit, checkpoint path, config snapshot, raw data source
3. Commits and pushes to `https://github.com/Matthew-Goley/theta-core.git`

Each report is a permanent, git-tracked checkpoint. Anyone can checkout the
commit and reproduce the gate measurement.

## Backtracking

If a later gate reveals a problem that traces back to an earlier gate:
1. Checkout the commit from the earlier gate's report
2. Fix the issue
3. Re-run the gate sequence from that point
4. New reports document the fork
