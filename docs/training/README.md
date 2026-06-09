# Explorer Training Runs

Chronological training runs of the GNN+MCTS explorer with correspondence-layer reward shaping and era-gated discovery monitoring.

## Run Summary

| Run | Date | Theorems | Epochs | Best Success | Key Result |
|---|---|---|---|---|---|
| [Run 1](run1_bootstrap.md) | 2026-06-03 | 500 bootstrap (`0=0`, `1+0=1`) | 50 | 25% | First valid proof found (epoch 15). Cold-start confirmed. |
| [Run 2](run2_physics.md) | 2026-06-03 | 9 reflexive physics | 20 | 50% | Correspondence + era tracker active. rfl heuristic proven. |
| [Run 4](run4_full_physics.md) | 2026-06-03 | 29 physics (all zones) | 200 | 100% | Full pipeline validated. 6 bugs fixed. 13/29 theorems provable. |
| [Inference](run4_inference.md) | 2026-06-03 | 25 held-out post-1905 | — | 32–40% | Trained GNN: 8/25, Untrained: 10/25. Heuristics dominate. GNN needs more signal. |

## Architecture Maturity

| Component | Run 1 | Run 2 | Run 4 |
|---|---|---|---|
| GNN gradient flow | ✓ | ✓ | ✓ |
| MCTS proof search | ✓ | ✓ | ✓ |
| Proof checker validation | ✓ | ✓ | ✓ |
| Correspondence zone classification | — | ✓ (UNCERTAIN only) | ✓ (UNCERTAIN only) |
| Era-gated discovery monitoring | — | ✓ (inflated) | ✓ (correct) |
| rfl heuristic | — | ✓ | ✓ |
| add_comm/mul_comm heuristic | — | — | ✓ |
| Proof truncation | — | — | ✓ |
| Multi-zone rewards | — | — | — |

## Key Files

```
docs/training/
├── README.md                  # This index
├── run1_bootstrap.md          # First training run — bootstrap theorems
├── run2_physics.md            # Reflexive physics theorems
└── run4_full_physics.md       # Full 29-theorem physics run
```

## Quick Start

```bash
# Generate physics theorems
python scripts/build_physics_theorems.py --era pre_relativity

# Verify proofs are valid Lean 4
python scripts/verify_physics_proofs.py

# Run training
python scripts/train_explorer.py \
  --domain Algebra \
  --theorems data/raw/physics_theorems.jsonl \
  --max-theorems 29 \
  --steps 200 \
  --mcts-sims 300 \
  --pretrained checkpoints/gnn/gnn_best.pt \
  --era pre_relativity \
  --output checkpoints/explorer_run5
```

---

*Training infrastructure built 2026-06-03.*
