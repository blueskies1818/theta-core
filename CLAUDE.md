# theta-core — Project Context for AI Workers

Autonomous mathematical physics discovery via self-play against physical
observation and Lean 4 proof verification.

## Architecture (pivot — June 21, 2026)

```
                    ┌──────────────────────────────┐
                    │    OBSERVATION DATABASE        │
                    │  Physical scenarios as JSON    │
                    │  "Here's what's true"          │
                    └──────────────┬─────────────────┘
                                   │
    ┌──────────────┐    ┌──────────▼──────────┐
    │  Expression   │    │                     │
    │  Generator    │───▶│    SELF-PLAY LOOP    │
    │  (math combos)│    │  Generate → Score    │
    └──────────────┘    │  → Select → Expand    │
                        └──────────┬───────────┘
                                   │
                        ┌──────────▼───────────┐
                        │   LEAN VERIFICATION    │
                        │  Numerical constancy   │
                        │  → Mathematical proof  │
                        └────────────────────────┘
```

## Current Status (June 21, 2026)

- Phase A ✓ — Expression grammar, type system, combinatorial generator
- Phase B ✓ — 10 physics observation scenarios, constancy evaluator
- Phase C ✓ — Self-play loop discovers mgh + ½mv² (score 1.000) from 6 training
                scenarios, generalizes to 2 held-out (score 0.984)
- Phase D → — Lean-prove the discovered conservation laws
- Phase E/F — Scale observations, train AI generalizer
- Phase G — Frontier predictions

Legacy GNN/lemma work archived at `archive/attempts_1-12/` and `archive/docs_legacy/`.

## Key Paths

| Path | Purpose |
|------|---------|
| `src/physics/` | Expression grammar, generator, evaluator, observations, search |
| `src/core/` | Self-play orchestrator |
| `src/proof_checker/` | Lean 4 subprocess, batch checker |
| `data/observations/` | Physical scenario databases |
| `.hermes/plans/self_play_physics_discovery.md` | Full architecture plan |

## Hardware

- Intel Arc B70 drives display — NEVER touch GPU drivers/firmware. No GPU compute.
- CPU: i5-12600KF, 16 cores. Training: max 6 threads. Eval: max 4 threads.
- Python 3.12, Lean 4.29.1, PyTorch.

## Security

- Never commit: API keys, absolute paths, `.env`, personal data, system config.
- `git diff --cached` before every commit. Scan for `sk-`, `DEEPSEEK_API_KEY`.

## Commands

```bash
# Test everything
python -m pytest tests/ -q

# Run self-play loop
python src/core/self_play_loop.py

# Generate expressions
python -c "from src.physics.generator import ExpressionGenerator; ..."
```
