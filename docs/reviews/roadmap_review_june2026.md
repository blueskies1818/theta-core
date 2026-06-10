# Roadmap Review — June 2026

**Date:** 2026-06-10
**Scope:** Full project — where we are against the Phase 1–5 roadmap, what's proven, what's missing
**Runs covered:** 1–8 (bootstrap through Honest GNN self-play)

---

## 1. Executive Summary

The self-play loop works. A 1.1M-parameter GNN, trained exclusively from Lean 4 proof-checker feedback on pre-1905 physics theorems, learned to prove 56% of held-out post-1905 theorems without heuristic assistance — exceeding the hand-coded heuristic baseline (52%). This validates the core AlphaGo Zero analog.

The system is at a capability ceiling, however. The remaining 44% of failures require capabilities the current architecture cannot provide: lemma-level discrimination, multi-step proof chaining, and genuine physical reasoning. The era-gated discovery test — the project's primary scientific validation — has not yet been demonstrated because the theorem set tests proof-shape matching, not physical concept discovery.

**Phase 1 is complete. Phase 2 is 60% done — the explorer trains and learns, but doesn't yet scale or cover failure conditions. Phases 3–5 are design-only.**

---

## 2. Roadmap Status

| Phase | Goal | Status | Evidence |
|---|---|---|---|
| **Phase 1** | Validate the self-play loop | **✓ COMPLETE** | GNN improved 40%→56% from proof-checker feedback alone. Learned policy exceeds hand-coded heuristics. |
| **Phase 2** | Scale the explorer | **~60%** | GNN+MCTS architecture works. Training pipeline validated. Missing: full 58K graph, failure condition encoding, multi-step proofs, lemma discrimination. |
| **Phase 3** | Physical grounding | **Design only** | Physical Prediction Scorer, Layer 1/2 data architecture are specified but not implemented. |
| **Phase 4** | Translation layer | **Not started** | Formal-to-natural translator specified. No code. |
| **Phase 5** | Open-ended operation | **Not started** | Continuous operation, holdout commitment. Dependent on Phase 3+4. |

---

## 3. What's Been Proven

### 3.1 The self-play loop generates genuine learning (Phase 1 ✓)

The GNN improved from a 40% pretrained baseline to 56% on held-out theorems through GRPO training against the Lean 4 proof checker. This is the central claim of the system — and it holds.

**Evidence:**
- Pretrained GNN (no GRPO): 10/25 post-era theorems at H=0.0
- Trained GNN (1000 epochs GRPO): 14/25 at H=0.0
- The trained GNN beats the heuristic baseline (13/25)
- The GNN discovered hypothesis usage independently (`simp [h]` on electroweak_unification)

### 3.2 The architecture functions end-to-end (Phase 2 partial ✓)

All components are implemented and validated:
- GNN encoder with GoalEncoder → produces differentiable lemma scores
- MCTS with Dirichlet noise → generates diverse proof candidates
- Proof checker verification → validates candidates through Lean during search
- GRPO with group advantages → provides training signal from binary rewards
- Heuristic annealing → enables gradual GNN takeover
- Correspondence layer + era tracker → monitors for physics concepts

### 3.3 Heuristic annealing enables GNN takeover (Phase 2 partial ✓)

Previous runs (5–7) proved that without annealing, the GNN never learns because heuristics always outvote it. Run 8 proved that with slow linear annealing (2000 epochs), the GNN gradually assumes control and maintains proof-finding at H=0.00.

**Sweet spot:** H≈0.25–0.50. Below that, policy collapse risk increases.

---

## 4. What's Not Yet Proven

### 4.1 Lemma-level discrimination

The GNN learned to select general-purpose tactics (`simp`, `linarith`, `ring`, `field_simp`) but not specific lemmas. All H=0.0 proofs are single-tactic. The 11 failed theorems all require specific lemma knowledge:

- `born_probability_identity` → needs `apply pow_two_nonneg`
- `gravitational_wave_strain_identity` → needs `apply add_nonneg; exact ⟨pow_two_nonneg _, pow_two_nonneg _⟩`
- `planck_scale_completion_identity` → needs `apply Real.sqrt_pos.mpr; positivity`

The GNN's cosine-similarity-based lemma scoring is too weak to discriminate between 16,842 candidate lemmas. The top-30 candidate filter may exclude the needed lemma entirely.

### 4.2 Multi-step proof chaining

No H=0.0 proof exceeds one tactic. The GNN has no mechanism for sequential reasoning — it scores lemmas for the current goal but cannot plan multi-step trajectories. Failures like `schrodinger_equation_identity` (`rw [h]; ring`) and `hubble_tension_identity` (`intro hzero; apply h; linarith`) require chaining that the architecture doesn't support.

### 4.3 Era-gated discovery

This is the project's headline scientific test: train on pre-1905 data, measure spontaneous discovery of post-1905 physics concepts. It has not been demonstrated because:

1. **The theorem set doesn't test physical reasoning** — all 54 theorems are single-tactic algebraic identities with physics-sounding names. The "era" label is cosmetic.
2. **Proofs contain no physics concepts** — the era tracker scans proof text for keywords like "quantum," "relativity," etc. But proofs are `simp`, `linarith`, `ring` — pure math tactics with zero physics content.
3. **Both eras use the same math** — `newton_cooling_identity` (pre-1905) and `heisenberg_uncertainty_identity` (post-1905) are both proved with `simp` or `linarith`. The era makes no difference to the proof strategy.

For genuine era-gated testing, we need theorems where the proof strategy *depends on physical assumptions* that differ between eras.

### 4.4 Failure condition encoding

The correspondence layer maps theorems to frontier zones (thermodynamics, QED, GR, etc.) and tracks failure coordinates (Planck scale, singularities). But the encoding is keyword-based on theorem *names*, not proof *structure*. The GNN receives no gradient signal from correspondence classification — it's purely passive monitoring.

### 4.5 Scale

The GNN is 1.1M parameters. AlphaProof — DeepMind's IMO-solving system — uses 3B parameters. Even small proof assistants use 100M+. The current model may simply lack capacity for lemma discrimination. Scaling to 10–50M params would test whether the architecture is fundamentally limited or just parameter-starved.

---

## 5. Sources of Error in Current Evaluations

### 5.1 The 25-theorem eval set has low statistical power

A single theorem flipping represents 4 percentage points. The 56% vs 44% difference between Run 8a and 8b is only 3 theorems. Run-to-run variance from MCTS stochasticity could explain some of the difference. Larger eval sets (100+) would provide more reliable measurements.

### 5.2 MCTS sims are a confound

At 400 simulations, MCTS search depth is limited. A theorem might be provable with 800 sims but fail at 400. The GNN's ranking could be correct but MCTS doesn't search deep enough. H=0.0 performance is a *lower bound* on GNN capability — the true capability could be higher with more sims.

### 5.3 The pretrained baseline advantages the GNN

The proof-step pretrained GNN already achieves 40% at H=0.0 before any GRPO training. The 40%→56% improvement is genuine learning, but the high baseline means the "cold start" problem (Zero's random play) hasn't been fully tested. The GNN started with substantial proof knowledge from Mathlib pretraining.

### 5.4 The training set is hand-crafted

The 55 training theorems were designed to be provable by the heuristics, which means they're biased toward the patterns the heuristics encode. The GNN may be learning a subset of what the heuristics already know, rather than discovering genuinely new proof strategies. The one counterexample — hypothesis usage — is encouraging but limited.

### 5.5 Single tactics inflate apparent success

All successful proofs use one tactic. The system isn't "proving theorems" in the traditional sense — it's selecting which automation tactic to throw at a goal. The proof checker then validates whether that tactic happens to close the goal. This is closer to tactic selection than theorem proving.

### 5.6 No comparison to formal theorem-proving baselines

The system hasn't been benchmarked against standard proof automation (aesop, simp alone, ring alone). A baseline of "try simp on everything" or "try linarith on everything" would contextualize the 56% result. If a trivial loop achieves 40%, the GNN's contribution is smaller than it appears.

### 5.7 The GPU issue forced CPU-only training

Training completed on CPU (16 cores) rather than the intended Intel Arc GPU. This slowed training ~3–5× but didn't affect correctness — the GNN forward pass is deterministic regardless of device. The GPU's compute engines are physically fused off on this specific Battlemage SKU (8086:e223, subsystem 1701). This is a hardware limitation, not a software bug.

---

## 6. What's Needed to Complete Each Phase

### Phase 2 Completion (estimate: 2–3 months)

| Item | Priority | Effort | Dependencies |
|---|---|---|---|
| Multi-step training theorems | Critical | 1 week | None |
| Curriculum learning (1-step → 2-step → 3-step) | Critical | 2 weeks | Multi-step theorems |
| Scale GNN to 5–10M params | High | 1 week | None (config change + retrain) |
| Full 58K graph for lemma access | High | 1 week | GNN scale-up (more params needed for more nodes) |
| Proof trajectory pretraining | High | 2 weeks | Mathlib proof extraction |
| Lemma retrieval beyond top-30 | Medium | 1 week | GNN improvements |
| Refined era-gated theorem pairs | Critical | 2 weeks | Domain expertise in physics |
| Stop annealing at H=0.25 | Medium | 1 day | None (config change) |

### Phase 3 Start Conditions

Before Phase 3 can begin, Phase 2 must demonstrate:
1. Multi-step proofs (3+ tactics) found without heuristics
2. Lemma-level discrimination (GNN selects correct lemma from 16K candidates)
3. GNN at H=0.0 proves >60% of post-era theorems on a genuinely era-differentiated test set

### Phase 3 Implementation (estimate: 4–6 months)

- Physical Prediction Scorer: multimodal transformer, 10–30B params
- Layer 1 data architecture: domain-specific encoders for raw measurement data
- Layer 2 extraction: formal mathematical objects from common format
- Domain-level holdout strategy
- Requires: substantial compute (multi-GPU cluster), physics datasets, domain expert collaboration

---

## 7. Architecture Assessment

### What's solid

| Component | Verdict |
|---|---|
| GNN+MCTS architecture | Works. Gradient flows through MCTS logits to GNN params. |
| Proof checker integration | Works. Lean 4 subprocess, batch checking, result caching, inline MCTS verification. |
| GRPO self-play | Works. Group-relative advantages provide nonzero gradient. Group size 4 sufficient. |
| Heuristic annealing | Works. Slow linear decay from 1.0→0.0 enables GNN takeover. Optimal stop at 0.25. |
| Proof-step pretraining | Works. 40% baseline vs 0% for old link-prediction pretraining. |
| Correspondence + era tracking | Functional but passive. No gradient signal from correspondence classification. |

### What needs redesign

| Component | Issue | Proposed Fix |
|---|---|---|
| Lemma scoring | Cosine similarity too weak for 16K candidates | Learned attention over lemma embeddings, or two-tower retrieval architecture |
| Multi-step reasoning | GNN scores only the current goal | Add lookahead: score (goal, lemma) pairs with predicted resulting goal |
| Era-gated test | Proofs contain no physics content | Theorems must encode physical assumptions in their statements |
| Correspondence reward | Passive monitoring, no gradient | Add correspondence-based reward shaping with differentiable zone classification |

---

## 8. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| GNN architecture fundamentally cannot learn lemma discrimination | Medium | High — blocks Phase 2 completion | Scale GNN first; if still fails, switch to small transformer |
| Multi-step proof chaining requires architectural change | Medium | High — blocks Phase 2 | Add rollout/value network like AlphaGo Zero |
| Era-gated test is impossible with algebraic identities | High | Medium — delays headline result | Redesign theorem set with physics content in proofs |
| Physical Prediction Scorer requires compute beyond budget | High | High — blocks Phase 3 | Start with toy physical model; seek compute grants |
| GPU permanently unavailable | High (confirmed) | Low — CPU training works for current scale | Accept CPU for Phase 2; acquire compute for Phase 3 |
| Policy collapse at low H-scale | High | Medium — degrades final performance | Stop annealing at H=0.25; add entropy bonus to loss |

---

## 9. Timeline Projection

```
2026 June (now):      Phase 2 — multi-step proofs, lemma discrimination, scale GNN
2026 July:            Phase 2 — full 58K graph, proof trajectory pretraining, refined era-gated test
2026 August:          Phase 2 completion — GNN proves multi-step theorems without heuristics
2026 Sept–Oct:        Phase 3 prep — Physical Prediction Scorer prototype, Layer 1 data pipeline
2026 Nov–Dec:         Phase 3 — Physical grounding integration, domain-level holdout
2027 Q1:              Phase 4 — Translation layer
2027 Q2+:             Phase 5 — Open-ended operation
```

This timeline assumes one developer working part-time. With additional resources (compute, collaborators), Phase 2 could complete in 4–6 weeks and Phase 3 could begin by August 2026.

---

## 10. Honest Assessment

The system works. The GNN learned to prove theorems from proof-checker feedback without human labels. The learned policy beats the hand-coded heuristics. This is a genuine research result.

**But.** The theorems are trivial algebraic identities. The era-gated test is cosmetic. The GNN can't chain tactics or discriminate lemmas. The model is tiny. The correspondence layer is passive. The GPU doesn't work.

The gap between "56% on single-tactic identities" and "spontaneously discovering quantum mechanics from classical training data" is enormous. The current system is a validated microcosm — it proves the self-play mechanism works at toy scale. Scaling it to genuine physics discovery requires solving lemma discrimination, multi-step reasoning, and physical content encoding.

The path is clear, the architecture is correct, and the next steps are well-defined. The work ahead is execution, not invention.

---

*Generated 2026-06-10. Phase 1 complete. Phase 2 at 60%. The self-play loop is real. Now scale it.*
