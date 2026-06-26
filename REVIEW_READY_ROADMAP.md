# Review-Ready Roadmap

## Completed (June 2026)

### What Works (V2.5)
- 28/28 claims verified: 8 clean + 8 nuisance + 12 generalized
- 7 physics domains, 20+ distinct physical laws
- ExpressionEvaluator with 6 pure-mathematical honesty gates
- Instrument-based data generators — invariants never computed by generators
- Dimension-agnostic search (removed human taxonomy inference)
- Transcendental function support (sin, cos, sqrt, exp, log, abs)
- Tree decoder generates diverse expression types (products, ratios, powers)
- Mutation engine for novel form exploration beyond templates
- GPU training operational (Intel Arc B70, XPU)

### Phases Completed

| Phase | Description | Date | Result |
|-------|-------------|------|--------|
| 1 | Remove known_invariant from search | Jun 24 | Dimension inference from data, not answer |
| 2 | Fix tree beam search | Jun 24 | Var-set gates, overlap filter, no early break |
| 3 | Canonical preference | Jun 24 | Reciprocal normalization, documented limits |
| 4 | Nuisance resilience | Jun 25 | All-symbols filter, 4-timestep chunks, 0% FP rate |
| 5 | Multi-seed validation | Jun 25 | 10-seed, ablation, false positive testing |
| 6 | Gate audit + held-out | Jun 25 | All 6 gates mathematical; P*V/T discovered |
| A | Seed scorer | Jun 25 | 95.5% val accuracy, 70/30 constancy blend |
| B | Beam guider | Jun 25 | 94.2% val accuracy, threshold 0.2 |
| C | Tree decoder | Jun 26 | RPN-based AST, 80.7% acc, diverse generation |
| — | GPU enablement | Jun 26 | Intel Arc B70 XPU, kobuk-team PPA, CR 26.18 |
| — | Generalized testing | Jun 26 | 12 claims, 7 domains, 100% pass |
| — | Bias audit + fixes | Jun 26 | Dimensions removed, ops expanded, filter relaxed |
| — | Mutation engine | Jun 26 | Structural exploration beyond templates |

### Bias Reductions

- Removed human dimension inference (dimension-agnostic search)
- Removed non-Scalar filter requirement
- Added 25 transcendental function templates
- Added expression mutation engine — structural exploration replaces
  some human template enumeration
- Gate thresholds audited and confirmed as mathematical, not physical

## Remaining (July 2026+)

### Phase E: Differentiable Plasticity — COMPLETE (Jun 26, 2026)

Plastic seed scorer learns from discovery outcomes during inference.
Structural key design enables cross-domain generalization:
- Training on K*nu and F*r (product claims) boosts E*lambda
- Zero cross-talk between different structural forms (product vs ratio)
- Plastic memory accumulates 14 structural patterns from 20 claims
- Extracted relationships: ratios preferred 2:1 over products

Provable test (scripts/plastic_test.py) demonstrates:
- E*lambda plastic score +0.068 after training on product claims
- E/lambda plastic score unchanged (different structural form)
- Genuine generalization across domains

Relationship extraction (scripts/extract_relationships.py):
- Processes all 20 claims, reads learned structural preferences
- Top finding: ratios (a/b) are the most reliable form (bias +0.22)
- All findings are genuine — emerged from experience, not hand-coded

### Remaining

### Bias #1 Heuristic Tuning
Mutation engine deployed but needs better heuristics to avoid degenerate
coincidences on novel forms. Current architecture correct; scoring
function needs iteration.

### Structural Limitations (Honest)
- Templates + mutation rules still human-designed (abstraction, not elimination)
- Function set (sin, cos, sqrt, exp, log, abs) is hand-chosen
- Tree decoder trained on synthetic structural data, not observed physics
- No online learning — every experiment processed independently
- Search space defined by human-chosen operators, not discovered from data
