# Roadmap — Self-Play Physics Discovery

## ✓ Phase A: Expression Infrastructure
- Grammar, type system, dimensional analysis
- Combinatorial expression generator
- 301K expressions at depth 4

## ✓ Phase B: Observation Database
- 10 falling/spring/pendulum/projectile scenarios
- Expression evaluator with constancy scoring
- mgh + ½mv² scores 1.000

## ✓ Phase C: Self-Play Loop v1
- 8 train / 2 test split with hold-out generalization
- Discovers energy conservation in 314 expansions
- Generalizes to unseen pendulum (0.984)

## → Phase D: Lean Verification
- Generated Lean theorems for discovered laws
- Proof holds for ALL trajectories, not just observations

## Phase E: Scale Observations
- 50-100 scenarios across multiple domains
- Work-energy theorem discovery
- Distinguish conservative vs non-conservative forces

## Phase F: AI Generalizer
- Train model on Phase A-E discoveries
- Learn expression patterns without brute force
- Cross-domain composition: gravity + electromagnetism

## Phase G: Frontier Predictions
- Feed all known physics observations
- Generate testable predictions for unmeasured regimes
- Human experimentalists close the loop
