# Explorer Training Runs

Chronological training runs of the GNN+MCTS explorer with correspondence-layer reward shaping and era-gated discovery monitoring.

## Run Summary

| Run | Date | Theorems | Epochs | Best Success | Key Result |
|---|---|---|---|---|---|
| [Run 1](run1_bootstrap.md) | 2026-06-03 | 500 bootstrap (`0=0`, `1+0=1`) | 50 | 25% | First valid proof found (epoch 15). Cold-start confirmed. |
| [Run 2](run2_physics.md) | 2026-06-03 | 9 reflexive physics | 20 | 50% | Correspondence + era tracker active. rfl heuristic proven. |
| [Run 4](run4_full_physics.md) | 2026-06-03 | 29 physics (all zones) | 200 | 100% | Full pipeline validated. 6 bugs fixed. 13/29 theorems provable. |
| [Inference](run4_inference.md) | 2026-06-03 | 25 held-out post-1905 | — | 32-40% | Trained GNN: 8/25, Untrained: 10/25. Heuristics dominate. GNN needs more signal. |
| **[Run 8](run8_honest_gnn_results.md)** | **2026-06-09** | **55 (29 physics + 26 richer)** | **2000** | **56% at H=0.0** | **GNN beats heuristics. Self-play loop proven. First genuine GNN learning.** |

## Architecture Maturity

| Component | Run 1 | Run 2 | Run 4 | Run 8 |
|---|---|---|---|---|
| GNN gradient flow | Y | Y | Y | Y |
| MCTS proof search | Y | Y | Y | Y |
| Proof checker validation | Y | Y | Y | Y (inline verify) |
| Correspondence zone classification | — | Y (UNCERTAIN only) | Y (UNCERTAIN only) | Y |
| Era-gated discovery monitoring | — | Y (inflated) | Y (correct) | Y |
| rfl heuristic | — | Y | Y | Y |
| add_comm/mul_comm heuristic | — | — | Y | Y |
| Proof truncation | — | — | Y | Y |
| ring/field_simp/linarith/simp heuristics | — | — | — | Y |
| Hypothesis usage heuristics | — | — | — | Y |
| Heuristic annealing (1.0->0.0) | — | — | — | Y |
| Dirichlet noise (MCTS exploration) | — | — | — | Y |
| Proof-step pretraining (GNN+GoalEncoder) | — | — | — | Y |
| Inline proof checker verification | — | — | — | Y |
| GRPO group_size=4 | — | — | — | Y |
| Multi-zone rewards | — | — | — | Y (passive) |
| **GNN exceeds heuristic baseline** | — | — | — | **Y** |

## Key Files

```
docs/training/
├── README.md                      # This index
├── plan_honest_gnn.md              # Honest GNN implementation plan
├── review_phase2_training.md       # Phase 2 training review (Runs 1-7)
├── run1_bootstrap.md               # First training run (bootstrap theorems)
├── run2_physics.md                 # Reflexive physics theorems
├── run4_full_physics.md            # Full 29-theorem physics run
├── run4_inference.md              # First held-out inference test
└── run8_honest_gnn_results.md     # Honest GNN training (GNN beats heuristics)
```

## Quick Start

```bash
# Generate richer theorems
python scripts/build_richer_theorems.py

# Train with all improvements
python scripts/train_explorer.py \
  --domain Algebra \
  --pretrained checkpoints/gnn/proof_step_pretrained.pt \
  --theorems data/raw/training_combined.jsonl \
  --max-theorems 55 \
  --steps 1000 \
  --mcts-sims 400 \
  --batch-size 2 \
  --group-size 4 \
  --heuristic-anneal-epochs 2000 \
  --heuristic-scale-min 0.0 \
  --era pre_relativity \
  --output checkpoints/explorer_run9

# Evaluate
python scripts/infer_explorer.py \
  --checkpoint checkpoints/explorer_run9/gnn_final.pt \
  --theorems data/raw/physics_theorems_post1905.jsonl \
  --no-era-filter --compare
```

---
*Training infrastructure built 2026-06-03. Updated 2026-06-10 with Run 8 results.*
