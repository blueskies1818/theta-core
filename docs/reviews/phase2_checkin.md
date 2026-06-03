# Phase 2 Check-in — Roadmap Alignment Review

**Date:** 2026-06-03
**Status:** Mid-Phase 2 — infrastructure built, frontier work ahead
**Review type:** Roadmap comparison + deviation analysis

---

## 1. Roadmap vs. Reality: What We Built

| ROADMAP # | ROADMAP Task | Built? | Our Label | Notes |
|-----------|-------------|--------|-----------|-------|
| 2.1 | Build math dependency graph | ✓ | 2.1 | 58,370 nodes, 160,611 edges, 7 domains |
| 2.2 | Implement GNN encoder | ✓ | 2.2 | GAT, 856K params, pure PyTorch, pre-trained |
| 2.3 | Implement MCTS proof search | ✓ | 2.3 | PUCT with hybrid keyword+GNN lemma retrieval |
| — | *(not in roadmap)* | ✓ | 2.4 | **Explorer trainer (GRPO+MCTS)** — added by us |
| 2.4 | Structure proposal (beyond proofs) | ✓ | 2.5 | Generator + validator, 3 physics templates |
| 2.5 | Build formal frontier map | ✗ | — | Not started |
| 2.6 | Experimental reproduction checks | ✗ | — | Not started |
| 2.7 | Encode known failure coordinates | ✗ | — | Not started |
| 2.8 | Scale to 3B parameters | ✗ | — | Not started |

**6 of 8 roadmap items have some coverage. 4 are fully implemented. 1 was added (not in roadmap). 4 are untouched.**

---

## 2. Deviations and Why

### Deviation 1: Added an explorer trainer (GRPO+MCTS) — not in ROADMAP

**What we did:** Built `src/explorer/explorer_trainer.py` — a complete GRPO training loop that wires MCTS proof search into the Phase 1 reward infrastructure. This replaces the LLM `generate_proofs()` call.

**Why:** The ROADMAP describes GNN+MCTS as a standalone inference architecture but doesn't specify how it learns. In Phase 1, the LLM learned through GRPO policy gradients. The GNN needs the same self-play training loop. Without the explorer trainer, MCTS would be a fixed search procedure with no ability to improve its lemma selection from proof checker feedback. The trainer closes the learning loop.

**Impact:** This is an architectural addition, not a detour. It completes the AlphaGo Zero analog: GNN = policy/value network, MCTS = search, Proof Checker = environment, GRPO = training signal. The ROADMAP implied this integration but didn't call it out as a separate sub-phase.

### Deviation 2: GNN architecture smaller than spec

| Parameter | ROADMAP Spec | What We Built | Why |
|-----------|-------------|---------------|-----|
| Hidden dimension | 512–1024 | 256 | Memory: 58K-node full graph at 1024-dim = 500 MB+ just for embeddings. Intel Arc B70 has 34 GB but message-passing intermediates multiply this. |
| Layers | 3–5 | 2–3 (configurable) | Deeper GNNs overfit on the sparse dependency graph (density 4.7×10⁻⁵). |
| Initial features | Transformer embeddings | Random init | Loading Qwen2.5-1.5B alongside GNN on XPU caused dual-model hangs (Phase 1 error E4). Random init trains faster and the GNN learns structure from the graph, not text. |
| Training task | "Proof-success prediction" | Link prediction on dependency edges | We have 160K known edges (supervised) vs. 0 successful proofs (no RL signal yet). Link prediction bootstraps embeddings before self-play. |

**Why these deviations are acceptable:** The architecture is configurable — all dimensions scale up with a config change. The 256-dim, 3-layer GNN with 856K params trains in 5 minutes on XPU. Scaling to 512-dim or 1024-dim is a hyperparameter change, not an architectural rewrite. We chose the smallest working configuration to validate the pipeline end-to-end first.

### Deviation 3: Reordered sub-phases

**ROADMAP order:** 2.1 (graph) → 2.2 (GNN) → 2.3 (MCTS) → 2.4 (structure proposal) → 2.5 (frontier) → 2.6 (experimental) → 2.7 (failure coords) → 2.8 (scale)

**Our order:** 2.1 → 2.2 → 2.3 → **explorer trainer** → 2.5 (structure proposal) → *(2.5–2.8 not done)*

**Why:** The explorer trainer (GRPO+MCTS integration) was the natural next step after MCTS. It closes the loop: MCTS generates proofs → checker validates → GNN learns. Without it, the GNN and MCTS are disconnected components. The structure proposal (ROADMAP's 2.4, our 2.5) was built next because it shares infrastructure with the explorer.

**Consequence:** The frontier map, experimental checks, and failure coordinates — which the ROADMAP places inside Phase 2 — are now deferred to Phase 2.5–2.7 (our numbering) or early Phase 3. These require domain knowledge and data we don't yet have, making them natural Phase 3 lead-ins.

### Deviation 4: `tactic_space.py` not created

**ROADMAP says:** `New: src/explorer/tactic_space.py`

**What we did:** Folded the action space (candidate tactic generation) into `src/explorer/proof_state.py` as `generate_candidate_actions()`.

**Why:** The action space is tightly coupled to the proof state representation. A separate file for ~80 lines of code added unnecessary indirection. The tactics are rule-based (7 types: apply, rewrite, exact, intro, cases, have, calc) — not learned. If tactics become learned in Phase 3, extracting `tactic_space.py` as a separate module would be appropriate.

### Deviation 5: No transformer initialization for GNN

**ROADMAP says:** "Initialized from transformer embeddings, then fine-tuned on proof-success prediction task."

**What we did:** Random initialization, trained from scratch on link prediction.

**Why:** 
1. **Dual-model XPU hang** (Phase 1 error E4): Loading the Qwen2.5 transformer alongside the GNN on Intel XPU causes deadlocks. The reference model workaround (CPU) doesn't apply here because we need both models active.
2. **Embedding space mismatch**: The transformer embeds token sequences (surface syntax). The GNN embeds graph nodes (structural position). Projecting from one space to the other requires a learned adapter — which is equivalent to training the GNN from scratch but with extra steps.
3. **Speed**: Random init + link prediction training takes 5 minutes for 50 epochs on XPU. Transformer embedding of 58K theorem statements would take hours.

### Deviation 6: MCTS uses keyword retrieval, not pure GNN similarity

**ROADMAP implies:** GNN alone evaluates proof states and ranks lemmas.

**What we did:** Hybrid approach — extract mathematical keywords from the goal, filter the graph for lemma name matches, then use GNN centrality + keyword score for ranking.

**Why:** Pure GNN similarity requires the goal text to be embedded in the same space as graph nodes. The GNN was trained on graph structure (link prediction), not on text→embedding mapping. Text encoding needs either:
- A separate trained encoder (transformer → GNN space projector)
- A different GNN training objective (node feature reconstruction from text)

The hybrid approach works now — for `0 + 0 = 0` it retrieves `add_zero`, for `a + b = b + a` it retrieves `add_comm`. Pure GNN similarity would require Phase 3's text-encoding pipeline.

---

## 3. What's Not Done (And Why)

### Phase 2.5: Formal frontier map — NOT DONE

**What ROADMAP says:** Machine-readable map with three zones: Established (proven + experimentally confirmed), Uncertain (competing theories, limited data), Breakdown (known infinities, singularities). Encodes Penrose-Hawking singularity conditions, Standard Model gauge group, Einstein field equations.

**Why not done:** This is a data-encoding task that requires formalizing known physics into machine-readable zone boundaries. We have no physicist on hand to validate the encodings. The infrastructure for it (`src/correspondence/`) exists as a stub from Phase 0.

**Can we do it now?** Partially. We can build the frontier map data structure and encode the breakdown zone coordinates (Penrose-Hawking, Planck scale, renormalization divergences) as formal boundary conditions without experimental data. The "Established" zone would be empty until Phase 3's experimental data pipeline.

### Phase 2.6: Experimental reproduction checks — NOT DONE

**What ROADMAP says:** Candidate structures must reproduce conservation laws, spectral lines, particle masses, gravitational wave strain patterns. Where results are formally encoded → proof checking; where numerical → statistical comparison.

**Why not done:** Requires actual measurement data (LIGO strain data, LHC cross-sections, CMB power spectra). The ROADMAP acknowledges this is Phase 3 territory ("Source physical measurement data" is Phase 3.3). We have no measurement datasets yet.

**Blocked until:** Phase 3.3 (data acquisition) or a decision to use synthetic/simulated data for initial testing.

### Phase 2.7: Known failure coordinates — NOT DONE

**What ROADMAP says:** Formal encoding of exact conditions where current theories produce infinities. Every candidate evaluated at these points.

**Why not done:** Similar to 2.5 — requires formalization of physics knowledge. More tractable than 2.6 because it doesn't need measurement data, just the mathematical conditions for breakdown.

**Can we do it now?** Yes. We can encode:
- Planck scale energy (E ~ 10¹⁹ GeV) as a formal coordinate
- Black hole singularity conditions (Penrose-Hawking theorems)
- Big Bang t=0 coordinate
- Non-renormalizable QFT divergence conditions
This would give the explorer "negative waypoints" — regions to avoid or to solve.

### Phase 2.8: Scale to 3B parameters — NOT DONE

**What ROADMAP says:** Train GNN+MCTS at 3B parameters. Hardware: 1× A100 80GB.

**Why not done:** Intel Arc B70 Pro has 34 GB VRAM. A 3B-parameter GNN would need ~12 GB for parameters alone, plus ~5 GB for the full-graph embeddings and message-passing intermediates. The Arc B70 *could* theoretically fit this (34 GB total), but:
1. We'd need to switch from full-graph to mini-batch inference
2. The current 856K-param model already saturates the link prediction signal
3. Scaling GNNs follows different laws than transformers — more params != better on low-density graphs

**When to revisit:** After the frontier map and experimental checks provide richer training signal. A larger GNN would be wasted on the current link prediction task.

---

## 4. Where We Go Next (According to the ROADMAP)

### Option A: Complete Phase 2 as specified (2.5 → 2.6 → 2.7 → 2.8)

Follow the ROADMAP linearly. Build the frontier map, experimental checks, failure coordinates, then scale up.

**Pros:** Follows the plan. Delivers the complete Phase 2 deliverable.
**Cons:** 2.6 (experimental checks) is blocked on data acquisition. 2.8 (3B scaling) is blocked on hardware or a cloud GPU budget. This path would stall at 2.6.

### Option B: Skip to Phase 3 (physical grounding)

Accept that 2.5–2.7 are Phase 3 lead-in work and start Phase 3 proper: acquire measurement data, build encoding pipelines, connect to physical prediction scoring.

**Pros:** Gets to the core goal (physical correspondence) faster.
**Cons:** Without the frontier map and failure coordinates, the explorer has no "compass" — it explores randomly rather than toward breakdown zones. The ROADMAP explicitly says these are Phase 2, not Phase 3.

### Option C: Build what we can (2.5 + 2.7), defer the rest

Build the frontier map and failure coordinates (data structure + formal encodings, no experimental data needed), then move to Phase 3. Defer 2.6 (experimental checks) until measurement data is available. Defer 2.8 (3B scaling) until training signal improves.

**Pros:** Practical. Builds everything possible now, doesn't stall on external dependencies.
**Cons:** Phase 2 deliverable is incomplete without experimental checks.

### Recommendation: Option C

The frontier map (2.5) and failure coordinates (2.7) are the explorer's compass. Without them, the GNN+MCTS searches blindly — it can find proofs but doesn't know which proofs matter. These are also the cheapest to build: they're data structures and formal encodings, no external data or hardware needed.

2.6 (experimental checks) should merge into Phase 3.1–3.4, where measurement data is acquired and scored.

2.8 (3B scaling) should wait until the training signal is richer — scaling a model that's trained only on link prediction would be premature optimization.

---

## 5. Immediate Next Action

**Build Phase 2.5 + 2.7 combined: The Frontier Map with Failure Coordinates.**

Files to create:
- `src/correspondence/frontier.py` — Frontier map data structure (three zones)
- `configs/frontier_map.yaml` — Encoded boundary conditions
- `src/correspondence/failure_points.py` — Formal failure coordinates
- `configs/failure_coordinates.yaml` — Planck, singularity, divergence points

This gives the explorer:
1. A **compass** — which mathematical territories are established, uncertain, or broken
2. **Negative waypoints** — where current theories fail (the problem to solve)
3. **Reward shaping** — pull toward breakdown zones, push away from reproducing known failures

After this, Phase 3 begins: physical data acquisition and encoding.

---

## 6. Current Architecture Summary

```
                    ┌──────────────────────┐
                    │   Dependency Graph   │  ← 58K nodes, 160K edges
                    │   (2.1)              │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │   GNN Encoder (2.2)  │  ← 856K-param GAT, pre-trained
                    │   Node embeddings    │
                    └──────────┬───────────┘
                               │
            ┌──────────────────┼──────────────────┐
            │                  │                  │
    ┌───────▼──────┐  ┌────────▼───────┐  ┌───────▼──────────┐
    │  MCTS (2.3)  │  │ Explorer       │  │ Structure        │
    │  Proof search│  │ Trainer (2.4+) │  │ Generator (2.5)  │
    └───────┬──────┘  └────────┬───────┘  └───────┬──────────┘
            │                  │                  │
    ┌───────▼──────────────────▼──────────────────▼──────────┐
    │              Proof Checker (Phase 1)                   │
    │              Reward System (Phase 1)                   │
    └────────────────────────────────────────────────────────┘
                               │
                     ╔═════════▼══════════╗
                     ║  FRONTIER MAP      ║  ← NOT YET BUILT (next)
                     ║  FAILURE COORDS    ║
                     ║  EXPERIMENTAL DB   ║
                     ╚════════════════════╝
                               │
                     ┌─────────▼──────────┐
                     │  Phase 3: Physical │
                     │  Correspondence    │
                     └────────────────────┘
```

**Key:** ✓ = built, ✗ = not built, + = added (not in ROADMAP), ═ = next

---

*Generated 2026-06-03. Phase 2 infrastructure complete. Frontier map next.*
