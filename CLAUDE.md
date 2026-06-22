# theta-core — Project Context for AI Workers

Autonomous mathematical physics discovery via self-play against physical
observation. Era-gated: trains on pre-1905 classical physics, discovers
post-1905 quantum and relativistic laws. Currently 8/8 passed.

## Architecture

```
Observations → Domain Classifier → Template Composer
              → Symmetry Detector/Discovery
              → Hidden Variable Proposer
              → Auto-Prover (Lean) → Noise Gate → Discovery
```

## Current Status (June 22, 2026)

- All phases A-F complete
- Era gate: 8/8 post-1905 laws reconstructed from pre-1905 training
- Quantum: hydrogen spectrum, spin, blackbody, photoelectric
- Relativistic: energy, momentum, velocity addition, time dilation
- 626+ tests pass. Zero physics knowledge injected.

## Key Directories

| Path | Purpose |
|------|---------|
| `src/physics/` | Expression grammar, evaluator, composer, symmetry, hidden vars, auto-prover |
| `src/core/` | Self-play orchestrator |
| `data/observations/` | Physical scenario databases |
| `checkpoints/` | Trained models |
| `docs/reports/` | Era gate results |

## Hardware

- Intel Arc B70 drives display — NEVER touch GPU drivers/firmware
- CPU: i5-12600KF, 16 cores. Training: max 6 threads. Eval: max 4 threads.
- Python 3.12, Lean 4.29.1, PyTorch

## Security

- Never commit: API keys, absolute paths, `.env`, personal data, system config
- `git diff --cached` before every commit

## Quick Commands

```bash
python -m pytest tests/physics/ tests/core/ -q
python scripts/spacetime_era_gate.py
python src/core/self_play_loop.py
```
