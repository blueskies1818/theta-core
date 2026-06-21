# theta-core — What We're Building and Why

---

## The Problem in One Paragraph

General relativity (gravity, big things) and quantum mechanics (particles, small things) are our two most successful theories of reality. They've passed every experimental test we've thrown at them. The problem: when we try to use both at the same time — inside black holes, at the Big Bang, at the Planck scale — the math explodes. Not "we don't know the answer." Literal infinities. The two theories are written in incompatible mathematical languages, and nobody has found the translation. This is the biggest unsolved problem in theoretical physics, and it's been stuck for decades.

---

## The Core Idea, in One Analogy

In 2017, DeepMind built a Go-playing AI called AlphaGo Zero. They gave it the rules of Go and told it to play against itself — millions of games. No human game records, no human strategy books. Just trial, error, and the binary signal of "win" or "lose."

It became the strongest Go player in history, even better than the pervious model first traind on human expert play. It found moves no human had considered in 2,500 years of playing the game. Moves that looked wrong to human experts but were deeply correct.

**Our question:** Can we do this for mathematical physics?

Give a system the rules (formal mathematics + experimental data), let it explore on its own, and see if it finds structures that humans haven't. Not because it's smarter — AlphaGo Zero wasn't smarter than humans at anything except Go — but because it isn't constrained by what humans have already thought of.

---

## What "The Rules" Are

AlphaGo Zero had three things that made self-play possible:

| AlphaGo Zero | Our System |
|---|---|
| The board state is always perfectly known | Formal proof verification — a proof either checks or it doesn't, deterministically |
| The rules of Go are unambiguous | Experimentally verified results — conservation laws, spectral lines, particle masses. Empirical facts, not theoretical frameworks |
| Win/lose is a clear terminal reward | A mathematical structure that predicts experimental data better than existing theories "wins" |

The insight: formal proof checkers like Lean 4 can tell you with 100% certainty whether a mathematical statement is valid. No probability, no interpretation, no gray area. A computer program reads your proof and says "yes" or "no." **This is our game board.** The system proposes mathematical structures, the proof checker validates them, and experimental data scores how well they describe reality.

---

## How It Works (The Short Version)

### The Loop

Every training step is the same cycle, running forever without human input:

```
1. MODEL:    Proposes a candidate mathematical structure or proof step
2. CHECKER:  Verifies it's logically consistent (Lean 4, deterministic)
3. SCORER:   Compares predictions against real experimental data
4. UPDATE:   Reward signal tunes the model. Stronger reward = better candidate
5. REPEAT:   Go to step 1
```

The model isn't told what to look for. It isn't trained on human textbooks. It generates proposals, gets feedback from reality (formal verification + physical measurement), and improves. The environment provides all feedback automatically.

### The Three Pressures

Rather than having a human set targets, the system imposes three constraints that together push it toward discovery:

1. **Must be internally consistent.** The proof checker rejects anything logically contradictory. This alone eliminates the vast majority of possible mathematical objects.

2. **Must reproduce experimentally verified results.** The structure must predict the outcomes that have been confirmed across independent measurements — conservation laws, particle masses, spectral line positions, gravitational wave strain patterns. These are empirical facts, not theoretical preferences. GR and QFT happen to be our best current fits to these facts in their respective domains. A new structure must match or exceed their predictive accuracy at the limits where they've been tested — but it doesn't need to mathematically "reduce to" them. Demanding formal reduction to two mutually incompatible theories may be demanding a contradiction. Demanding reproduction of the experiments they explain is a coherent, measurable target.

3. **Must predict experimental data better than current theories.** A structure that describes gravitational wave measurements, particle collision data, and spectroscopic observations more accurately and more simply than existing frameworks scores higher. Simpler structures score higher than complex ones (Occam's razor, formalized).

### The Three Components

No single model does everything. Three specialized components work together:

| Component | What It Does | Analogy |
|---|---|---|
| **Mathematical Explorer** | Explores mathematical space, proposes structures and proofs | The chess player — generates moves |
| **Physical Prediction Scorer** | Checks how well a proposed structure predicts real experimental data | The referee — scores the moves |
| **Translation Layer** | Converts formal mathematics into human-readable explanations | The commentator — explains what happened |

---

## How Discoveries Happen

The system uses a mechanism modeled on how Dirac predicted antimatter in 1928.

Dirac wrote an equation that had to satisfy two constraints: describe an electron, and be consistent with special relativity. The simplest equation satisfying both produced four mathematical solutions. Two matched known particles (electrons with spin up/down). Two had negative energy — seemingly nonsense.

Instead of discarding them as math artifacts, Dirac asked: "What would have to be true for these to be real?" Answer: a particle identical to the electron but with opposite charge. The positron. Discovered experimentally four years later.

Our system does this systematically. When a verified structure produces solution families that don't match any known particle or field, they're flagged, characterized, and converted into experimental proposals: "If this entity is real, here is what detector you need, at what energy scale, looking for what signature."

---

## Why Not Just Use a Big Language Model?

ChatGPT and similar models are trained on everything humans have written. That's exactly the problem.

Everything humans have written contains:
- Our cognitive biases baked into the structure of knowledge
- Conceptual categories that might carve nature at the wrong joints
- Consensus beliefs that might be wrong
- The limits of what we've thought to write down

A system trained on human data is bounded by human imagination. AlphaGo Zero shows that removing this constraint — at the cost of starting from almost nothing — can produce qualitatively superior results. The challenge is that mathematics isn't Go, so the mechanism has to be more carefully constructed. But the principle is the same.

---

## Beyond GR and QFT — A General-Purpose Discovery Engine

The GR-QFT unification problem is our **first target**, not our only target. The architecture — an explorer that generates mathematical structures scored against experimental data — is domain-agnostic. Anything that can be described mathematically and measured experimentally is fair game.

**Materials science:** Given a specification ("alloy with tensile strength above X, melting point above Y, using only elements from set Z"), the explorer proposes crystal structures, composition ratios, and processing conditions. The scorer evaluates predictions against materials databases and experimental measurements. The system finds candidates that no human metallurgist has considered.

**Propulsion physics:** Given constraints ("specific impulse above X, thrust-to-weight ratio above Y, operating in regime Z"), the explorer proposes propulsion mechanisms — electromagnetic, thermodynamic, nuclear, or something not yet named. The scorer evaluates against known physics and experimental data. The system can explore propulsion concepts outside the design space humans have traditionally searched.

**Drug discovery, catalyst design, battery chemistry, fluid dynamics, structural engineering** — the pattern is the same. Define the formal search space (chemical compounds, differential equations, tensor fields), define the scoring function (experimental measurements, physical constraints), and let the explorer search. The architecture doesn't care what domain it's exploring. It only cares that there's a formal mathematical language to generate in and experimental data to score against.

GR-QFT unification is the hardest problem and therefore the best test. If the system can make progress there, it can make progress anywhere. But the investment pays off across every domain where we have formal mathematics and experimental measurements — which is to say, across all of science.

---

## Where We Are Now

We're in **Phase 1**: proving the loop works.

The current system is deliberately small and simple:
- A 1.5 billion parameter model (roughly the size of a smartphone AI model) learns to prove theorems
- It generates proofs, a Lean 4 proof checker validates them, and the binary yes/no signal trains the model
- No physical data yet — just formal mathematics
- The goal: demonstrate that the model *learns to prove things it didn't know before training*, purely from proof-checker feedback

This is the "AlphaGo Zero playing on a 9×9 board" stage. Validate that the mechanism works, measure the learning curve, then scale up.

**Phase 2** swaps in the proper architecture (graph neural network with tree search) and expands to the full GR/QFT mathematical domain.

**Phase 3** connects real experimental data — gravitational wave detector readings, particle collider events, spectroscopic measurements — as the scoring function.

**Phase 4** builds the translation layer so human physicists can read and act on what the system finds.

**Phase 5** runs the full system continuously, with predictions committed against future experiments before they report results.

---

## The Honest Answer: What Could Go Wrong

This might not work. Several things could go wrong:

**The system might find nothing.** The space of interesting mathematical structures near GR and QFT might be genuinely sparse, and it's possible there simply isn't a clean unification. The system can't invent physics that isn't there.

**It might find things we can't understand.** A formally verified, experimentally predictive structure could be completely opaque to human intuition — like quantum mechanics was in 1925, but potentially deeper. Whether that counts as "understanding" is a philosophical question.

**It might reward-hack.** Any sufficiently capable optimizer finds shortcuts. The system could discover mathematical structures that score well on our metrics without capturing real physics. Detecting this requires ongoing human oversight.

**The compute might be insufficient.** Formal proof checking at scale is CPU-intensive in ways most ML infrastructure isn't designed for. The bottleneck isn't GPU power — it's running millions of deterministic proof checks in parallel.

These are real risks. They don't make the project not worth doing. AlphaGo Zero had every reason to fail too — learning Go from scratch, with no human data, shouldn't have worked as well as it did. Sometimes the ambitious approach is the one that works.

---

## Want the Full Technical Detail?

This document is the friendly overview. The real specifications are:

- **[README.md](README.md)** — Project overview, structure, getting started
- **[mathematical_ai_system.md](mathematical_ai_system.md)** — Full system design: architecture, training methodology, theoretical foundations, hardware, roadmap
- **[model_structure_and_data.md](model_structure_and_data.md)** — Detailed technical spec: component internals, data pipeline, preprocessing, open problems
- **[IMPROVEMENT_IDEAS.md](IMPROVEMENT_IDEAS.md)** — Running list of what's incomplete and what could be better
