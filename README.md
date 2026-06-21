# theta-core — Autonomous Mathematical Physics Discovery

A self-play AI system that discovers physics from scratch — given only mathematical
operations and physical observations, it finds the laws that govern reality.

## What It Does

```
Input:   "Here are 10 recorded moments of a falling ball.
          Mass=m, gravity=g, height=h, velocity=v."
         
System:  Generates math expressions from {m, g, h, v, +, -, *, /, ^}
         Scores each by: "is it constant across all 10 observations?"
         Discovers: m*g*h + ½*m*v² never changes.
         
Output:  "CONSERVATION LAW DISCOVERED: mgh + ½mv² = constant"
         + Lean proof that this holds for ALL trajectories.
```

The system does not know what "energy" is. It discovers it because
reality constrains which math structures work.

## The Final Vision

```
┌─────────────────────────────────────────────────────────────────┐
│                    SELF-PLAY PHYSICS ENGINE                      │
│                                                                  │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────────────┐    │
│  │ Generate │    │   Evaluate   │    │     Lean-Prove       │    │
│  │  math    │───▶│  against all │───▶│  conservation for    │    │
│  │structure │    │ observations │    │  all conditions      │    │
│  └──────────┘    └──────────────┘    └──────────┬──────────┘    │
│                                                  │               │
│                     ┌────────────────────────────▼──────────┐   │
│                     │       AI GENERALIZER                   │   │
│                     │  Learns patterns: "½·m·v² appears      │   │
│                     │   wherever velocity exists"             │   │
│                     │  Generates: ½·k·x² for springs         │   │
│                     │   WITHOUT enumerating all combos        │   │
│                     └────────────────────┬───────────────────┘   │
│                                          │                       │
│     ┌────────────────────────────────────▼───────────────────┐   │
│     │            THEORETICAL FRONTIER                         │   │
│     │  Unobserved regime → generates testable predictions    │   │
│     │  "At 10¹⁶ GeV, this structure predicts X% deviation    │   │
│     │   from Standard Model — measurable at future collider"  │   │
│     └────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Phase 1-4: Known Physics (training)

The system rediscovers: conservation laws, Lagrangian mechanics,
Maxwell's equations, relativity, quantum constraints — purely from
observational data and mathematical search.

Generalization test: discover energy conservation from falling balls
and springs separately, then compose both for a gravity+spring
scenario it's never seen.

### Phase 5: Cross-Domain Composition

Train on gravity-only data and electromagnetism-only data.
Test on a charged particle in a gravitational field — the system
must compose gravitational + electromagnetic invariants from
its separate understanding of each domain.

### Phase 6: The Frontier

Given all confirmed physical observations but NO complete theory
for high-energy regimes, the system:
1. Generates mathematical structures that match all known data
2. Derives predictions for unmeasured regimes
3. Ranks predictions by testability
4. Outputs: "Build this experiment to distinguish Theory A from Theory B"

## Architecture

- **Expression Generator:** Combinatorial math builder with physical dimensions
- **Observation Database:** Physical scenarios as JSON — "here's what happened"
- **Constancy Evaluator:** Scores expressions by how invariant they are
- **Self-Play Loop:** Generate → score → select → expand → repeat
- **Lean Prover:** Turns numerical constancy into mathematical theorems
- **AI Generalizer:** Learns expression patterns to explore intelligently

## Current Status

Phase A-C complete (June 2026). System discovers conservation of
mechanical energy from falling-ball observations and generalizes
to unseen pendulum scenarios. Score: 1.000 training, 0.984 test.

Next: Lean-prove the discoveries. Then scale to electromagnetism,
quantum, and relativity observation databases.

## Honesty Contract

1. No physics injected — system knows quantities and operations, not interpretations
2. Only binary verification — each observation passes or fails
3. Era-safe — training on pre-1905 observations, testing on post-1905
4. Discovery IS prediction — a structure succeeds when it implies unmeasured outcomes
5. Human verification loop — frontier predictions confirmed by experiment
