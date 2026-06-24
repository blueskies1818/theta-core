# Error Analysis — theta-core Shortcomings

Documenting every known limitation, weakness, and failure mode. When a
shortcoming is fixed, mark it resolved with date and commit. This document
is the honest companion to the README claims — it's what an independent
reviewer would find if they dug deeper than the verification script.

**Last updated:** June 23, 2026

---

## 1. Circular Test Data (CRITICAL)

**Severity:** Undermines the core discovery claim.

7 of 8 verification claims generate test data BY EVALUATING THE FORMULA
being tested. The photoelectric generator literally computes
`K_max = h*nu - phi` and feeds both sides to the system. The system then
"discovers" `h*nu - K_max = phi` — it factored out a relationship that was
baked into the data.

This isn't era-gate leakage. It's data-generation leakage. The system
discovers whatever algebraic relationship the test author encoded.

**Fixed partially for 1/8 claims (June 23):** Photoelectric now includes
below-threshold data points where K_max=0, and the regime loop correctly
identifies that `h*nu - K_max` only holds above threshold. The formula
is still baked in, but the system must now discover the CONDITION.

**Remaining:** The other 7 claims still use formula-inverted data.

**What a real test looks like:** Feed the system raw measurements from a
physical experiment (or a simulator whose internal equations differ from
the invariant being tested). The system should find the invariant without
access to the formula used to generate the data.

---

## 2. Hand-Written Neural Templates (CRITICAL)

**Severity:** The neural generators don't discover — they deliver
pre-written answers.

`composer.py` contains hardcoded template strings:
```python
DOMAIN_TEMPLATES = {
    "relativistic": "E^2 - (p*c)^2",
    "quantum":      "n^2*hbar^2/(2*m*L^2)",
    ...
}
```

The neural template generators were trained to output these exact strings.
When the relativistic domain fires, the generator outputs `(c*t)^2-x^2`
or `E^2-(p*c)^2` — formulas a human who knew special relativity wrote
into the template table.

The system discovered nothing. A human encoded known physics into templates,
then the system regurgitates them. The era gate prevents the SYSTEM from
seeing post-1905 formulas, but it doesn't prevent the DEVELOPER from
knowing them and writing them into the code.

**Fix:** Remove the hardcoded templates. Train template generators from
scratch on pre-1905 expression patterns only (ratios, products, sums,
conservation forms). The generators should learn STRUCTURAL patterns
(how to compose multi-term invariants) without learning specific formulas.
Then test whether those structural patterns generalize to post-1905 forms.

---

## 3. Lean Proofs Prove the Wrong Thing (HIGH)

**Severity:** The honesty contract says "all discoveries are Lean-proven"
but the proofs don't prove discovery.

The README states: "No numerical coincidence passes as discovery."

The Lean proofs in `auto_lean.py` prove **dimensional well-formedness**:
"The expression E/n has dimension Energy/Scalar and its evaluation is
well-typed." This is a type-checking statement, not a constancy statement.

A real proof would state: "For all timesteps t1, t2 in observation O,
evaluate(expr, t1) = evaluate(expr, t2)." This requires encoding the
numerical values in Lean and running the arithmetic — fundamentally
different from a type check.

**Current state:** 6/8 claims have dimensional proofs. Zero claims
have numerical constancy proofs.

**Fix:** Implement numerical constancy proofs for at least one claim
(e.g., E/n — simple ratio of two scalars). Even one real proof would
satisfy the honesty contract's spirit.

---

## 4. No Statistical Variance (HIGH)

**Severity:** Results could be lucky single runs.

Every verification result is from one run. No error bars, no seed
variation, no p-values. `verify_8_claims.py` is deterministic — same
output every time, but that's because it's formula-inverted data.
If the data had noise (see #7), results would vary.

**Fix:** Run `verify_8_claims.py` with 5 different random seeds (varying
beam search tie-breaking, not data). Report mean ± std for each claim's
discovery score and whether the exact formula was found.

---

## 5. Era Gate Tests Different Scenarios at Different Cutoffs (MEDIUM)

**Severity:** Makes progressive discovery unverifiable.

The multi-cutoff era gate (`spacetime_era_gate.py`) tests:
- 1905 cutoff: 15 scenarios (muon, velocity addition, relativity, QED...)
- 1920 cutoff: 7 scenarios (QED, Compton, Dirac, QCD...)

These are DIFFERENT test sets. The 1920 cutoff scored 0/7 not because
the system got worse, but because the test scenarios changed to topics
the system hasn't been trained for.

A proper era gate: test the SAME set of post-1970 scenarios at every
cutoff. Show discovery rate increases as training era expands. If the
system discovers the same things regardless of cutoff, the era gate
is decorative.

**Fix:** Create a fixed test set of 10 post-1970 scenarios. Run the
era gate at cutoffs 1905, 1920, 1950, 1970 against this SAME set.
The discovery count should monotonically increase.

---

## 6. Formula-Inverted Data (applies to 7/8 claims) (MEDIUM)

**Severity:** Testing discovery with answer-key data.

Each verification scenario constructs data where the known invariant
is mathematically constant by construction:

```python
# Hydrogen Balmer
E = H * C / lambda_val  # compute E from lambda
timesteps.append({"E": E, "lambda": lambda_val})
# Tests: "discover E*lambda = h*c"
# But we literally computed E = h*c/lambda and put both in the data!
```

The system isn't discovering physics — it's discovering that the test
author used a formula to generate the data. This is equivalent to
testing a student by giving them the answer key and asking them to
copy it.

**Fixed for photoelectric (June 23).** Remaining 7 claims still use
formula-inverted data.

**Fix:** For each claim, generate data from a DIFFERENT formula than
the invariant being tested. Example: generate orbital data from
Newton's law of gravitation + kinematics, then test whether the system
discovers conservation of angular momentum. The data generator and
the invariant should use different mathematical paths.

---

## 7. No Noise Testing (MEDIUM)

**Severity:** System is only tested on perfect data.

Every test scenario uses exact values with constancy scores of 1.000000.
Real experimental data has measurement error (1-20%). The noise
calibration module (`src/physics/noise.py`) exists but is never
exercised in the verification pipeline.

Would the system survive 1% Gaussian noise? Would coincidence
expressions (which are perfectly constant on exact data) degrade faster
than genuine invariants, or slower?

**Fix:** Add a `--noise 0.01` flag to `verify_8_claims.py`. Run with
1%, 5%, and 10% noise. Report how many claims survive at each level.
A real invariant should maintain high constancy; a coincidence should
collapse.

---

## 8. Dimension Bug: Spring Constant (LOW)

**Severity:** Training data error, not architecture flaw.

The spring constant `k` is assigned dimension "Force" (Mass·L/T²) in
the training data. But a spring constant is Force/Length (Mass/T²).
This means `k*x^2` has dimension Energy·Length, not Energy — beam
search can't compose spring energy terms because the dimensions don't
match.

The hardcoded domain template `0.5*k*h^2 + 0.5*m*v^2` in composer.py
works because templates skip dimension checking, but the beam search
can't independently discover this form.

**Fix:** Change k's dimension to "Force/Length" (compound dimension)
or add a new "SpringConstant" dimension (Mass·T⁻²). Update training
data, retrain spring domain template.

---

## 9. Search Finds Coincidences Before Truth (MEDIUM)

**Severity:** Architecture favors coincidences on clean data.

The beam search is rewarded for finding ANY expression that scores 1.0.
On clean (noise-free, formula-generated) data, there are often MULTIPLE
expressions that score perfectly — the genuine invariant AND algebraic
coincidences. The search returns whichever it encounters first.

Fixes applied (June 23):
- Trivial-constancy gate: blocks expressions with only fixed inputs
- Self-cancellation gate: blocks `1/v*1*v` style algebraic constants
- Multi-term tiebreaker: prefers structurally richer invariants when scores tie

These eliminate the most common coincidence patterns, but the
fundamental issue remains: the system can't distinguish "this expression
is constant because it encodes a physical law" from "this expression is
constant because the test data happens to make it so." On the frontier,
where there's no answer key, a coincidence IS indistinguishable from a
discovery.

**Fix partially applied.** No complete fix possible without statistical
testing (noise, cross-validation, held-out scenarios). This is inherent
to any system that searches over expression space.

---

## 10. Unbounded Search Without Complexity Penalty (LOW)

**Severity:** Longer expressions can always fit data better.

The beam search has a depth discount (0.95) but no explicit complexity
penalty. `0.5*k*x^2 + 0.5*m*v^2` and `0.5*k*x^2 + 0.5*m*v^2 + 0*m*g*h`
score identically if the data makes the third term zero. The system may
prefer longer expressions because they have more terms (multi-term
tiebreaker) even when extra terms are decorative.

This is partially addressed by the multi-term tiebreaker preferring more
terms only when scores tie exactly. But the fundamental issue — that
adding irrelevant terms never hurts — remains.

**Fix:** Add an AIC/BIC-style complexity penalty: score = constancy -
lambda * (number of terms). Calibrate lambda so that adding a term with
zero contribution reduces the score enough to lose to a simpler
expression.

---

## 11. Dimension Weights May Encode Post-1905 Knowledge (LOW)

**Severity:** Suspiciously perfect — warrants investigation, not
necessarily a flaw.

The canonicalizer dimension weights were "learned from pre-1905
invariants" (m*g*h → Mass before Length, P*V/T → Pressure·Volume before
Temperature). These weights then produce EXACTLY the right ordering for
every post-1905 formula:

- E*lambda (Energy 0.95 > Length 0.8)
- E/n (Energy 0.95 > Scalar 0.3)
- E/gamma (Energy 0.95 > Scalar 0.3)
- (c*t)^2-x^2 (Time·Velocity is complex but ordering is correct)

Zero conflicts between pre-1905 training patterns and post-1905
canonical forms. This is either a genuine discovery (physical conventions
are dimensionally universal) or the set of "pre-1905 invariants" was
curated with post-1905 outcomes in mind.

**Fix:** Document the EXACT set of pre-1905 invariants used to derive
the weights. Show the weight derivation step by step. If the weights
generalize, it's a genuine finding. If they were tuned, document that.

---

## 12. No Cross-Validation or Held-Out Scenario Testing (MEDIUM)

**Severity:** System is tested on data it was configured for.

Every test scenario was designed by someone who knows the answer. There's
no held-out set of scenarios the system has never been configured for.
The novel tests (`scripts/novel_tests.py`, June 23) are the first
attempt, and they immediately exposed bugs (self-cancellation garbage,
dimension mismatch).

**Fix:** Maintain a held-out test suite of 20+ scenarios that are NEVER
used during development. Only run them at release time. Track discovery
rate over releases. This is standard ML practice (test set hygiene)
applied to physics discovery.

---

## Summary Table

| # | Shortcoming | Severity | Status |
|---|---|---|---|
| 1 | Circular test data (7/8 claims) | CRITICAL | Partial (1/8 fixed Jun 23) |
| 2 | Hand-written neural templates | CRITICAL | Open |
| 3 | Lean proofs prove dimensions, not constancy | HIGH | Open |
| 4 | No statistical variance | HIGH | Open |
| 5 | Different test sets at different cutoffs | MEDIUM | Open |
| 6 | Formula-inverted data (7/8) | MEDIUM | Partial (1/8 fixed Jun 23) |
| 7 | No noise testing | MEDIUM | Open |
| 8 | Spring constant dimension bug | LOW | Open |
| 9 | Coincidence preference over truth | MEDIUM | Mitigated (3 gates added Jun 23) |
| 10 | No complexity penalty | LOW | Open |
| 11 | Dimension weights potentially curated | LOW | Investigate |
| 12 | No held-out test suite | MEDIUM | Started (novel tests Jun 23) |

---

## What an Independent Reviewer Would Conclude

The architecture is sound and the era gate is a genuine innovation.
5/8 claims produce exact textbook formulas from measurement data —
this is impressive and worth publishing.

However, the system currently tests for discovery using data that
contains the answer. The neural templates deliver pre-written formulas.
The Lean proofs don't prove what the README claims. And the system
has never been tested on noisy data, varied seeds, or genuinely
held-out scenarios.

After fixing items 1, 2, and 3, the system would withstand scrutiny.
After fixing items 1-6, it would be publication-ready. After fixing
all 12, it would be frontier-ready.

This document should be updated whenever a shortcoming is addressed.
Each fix should include the date, commit hash, and what changed.
