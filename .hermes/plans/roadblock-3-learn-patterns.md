# Roadblock #3: Learn Structural Patterns from Data

## Goal
Replace hand-enumerated template catalog with patterns the system
discovers organically. Start with minimal base templates (a*b, a/b,
a^2, a+b) and let the search vocabulary grow from experience.

## Why
The 72 templates are the single biggest human bias. If a frontier claim
uses a structural pattern we didn't enumerate, the system can't find it.
The tree decoder, mutation engine, and plastic memory already exist —
they just need to feed back into the template catalog.

## Approach

### Cumulative template learning
1. Start with a minimal base set: a*b, a/b, a^2, a+b (5 templates)
2. After each discovery via tree decoder or mutation engine, extract
   the structural pattern from the discovered expression
3. Add it to the learned template catalog (capped at 200)
4. Future `simple_invariant_search` calls use BOTH base + learned templates

### Pattern extraction
Reuse the `_structural_pattern` logic from plastic_seed_scorer.py which
already extracts abstract forms (a*b, a/b, a^2*b, a*b/c, etc.)

### Integration
- Module-level `_learned_templates: list[str]` in search.py
- After auto_discover succeeds, extract pattern and add to catalog
- In simple_invariant_search, append learned templates to _THREE_QTY_TEMPLATES
- Cap at 200 to prevent exponential slowdown

### Bootstrap test
Start with only 5 base templates, run all 28 claims. Which ones still pass?
The tree decoder and beam search should handle some; the learned catalog
grows as each claim is processed. The goal: after processing all 28,
the learned catalog should contain the patterns needed for all claims.

## Files
- src/physics/search.py — _learned_templates, pattern extraction, integration
- src/math/plastic_seed_scorer.py — reuse _structural_pattern
