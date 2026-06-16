# theta-core — Project Context for AI Workers

Autonomous Mathematical Physics AI — a self-play system for exploring formal
mathematics beyond the boundaries of current human knowledge.

## Architecture (what's built)

```
Dependency Graph (58K nodes, 160K edges)
       │
       ▼
GNN Encoder (GAT, 1.1M params, GoalEncoder)
       │
       ├──→ MCTS Search (PUCT, 400-1000 sims, Dirichlet noise)
       │         │
       │         ▼
       │    Proof Checker (Lean 4.29.1, subprocess, batch, SHA-256 cache)
       │         │
       │         ▼
       │    Reward + GRPO Training (group_size=4, heuristic annealing)
       │
       └──→ Structure Generator + Validator
                 │
                 ▼
            Correspondence Layer (frontier map, era tracker, failure coords)
```

## Current Status (June 10, 2026)

- Phase 1: COMPLETE — self-play loop validated
- Phase 2: 60% — GNN beats heuristics (56% vs 52%) but at capability ceiling
- Phase 3-5: Design only

## Key Files

| File | Purpose |
|------|---------|
| `src/explorer/gnn_encoder.py` | GNN with GAT layers, GoalEncoder, edge-type embeddings |
| `src/explorer/gnn_config.py` | GNN hyperparameters (hidden=256, layers=3, heads=8) |
| `src/explorer/mcts.py` | MCTS with PUCT, Dirichlet noise, inline proof verification |
| `src/explorer/proof_state.py` | Proof state representation, candidate action generation |
| `src/explorer/explorer_trainer.py` | GRPO training loop with heuristic annealing |
| `src/explorer/dependency_graph.py` | 58K-node math dependency graph (NetworkX + PyTorch) |
| `src/proof_checker/` | Lean 4 subprocess interface, batch checker, cache |
| `src/correspondence/` | Frontier map, era tracker, failure points, reward integration |
| `src/reward/base.py` | Binary reward + length bonus + curiosity bonus |
| `scripts/train_explorer.py` | Training entry point |
| `scripts/infer_explorer.py` | Evaluation entry point (comparison mode, baselines) |
| `scripts/build_richer_theorems.py` | Theorem set generation |
| `configs/` | YAML configs (model, GRPO, reward, frontier) |

## Key Numbers

- GNN: 1,118,848 params, 3-layer GAT, 256-dim, 8 heads
- Training: 55 theorems, 2000 epochs, group_size=4, 400 MCTS sims
- Best result: 56% at H=0.0 on 25-theorem held-out set (beats heuristics at 52%)
- Ceiling: 11/25 theorems fail due to lemma discrimination + multi-step gaps
- Hardware: Intel i5-12600KF (16 cores CPU), Intel Arc B70 (GPU compute fused off)
- Python 3.12, Lean 4.29.1, PyTorch

## Hardware Constraints (READ BEFORE ANY TRAINING OR DRIVER CHANGE)

**CRITICAL: Intel Arc B70 GPU drives the display.** This GPU is connected to the
monitor and renders the desktop. Do NOT:
- Unload/reload GPU kernel drivers (will kill the display)
- Flash GPU firmware or update GPU BIOS
- Run GPU compute workloads that saturate VRAM (can cause display freeze)
- Install Intel GPU drivers that might break display output

GPU compute is confirmed fused off at hardware level (Battlemage e223 SKU,
8086:e223, subsystem 1701). This is permanent — do not attempt to enable it.
Training runs on CPU only.

**CPU headroom required.** Training uses CPU (16 cores, Intel i5-12600KF).
Always leave at least 2 cores free for system responsiveness:
- Proof checker batch size: max 12 workers (leaves 4 cores for OS + desktop)
- Training batch size: keep memory usage under 80% of system RAM
- If the system becomes unresponsive during training, reduce worker count

**No Python multiprocessing with `spawn` that consumes all cores.**
Use `torch.set_num_threads()` to cap PyTorch CPU threads.
Default safe max: 12 threads for training, 8 for concurrent training + desktop use.

## Security Rules (NEVER VIOLATE)

- **Never commit API keys, tokens, or credentials.** Check diffs before committing.
  `git diff --cached` before every commit. If you see `sk-`, `DEEPSEEK_API_KEY`,
  or any key-like string, unstage it immediately.
- **Never commit absolute paths containing `/home/blueman1818/`.** Use relative
  paths or `~/.hermes/` notation in code. In configs, use environment variables.
- **Never commit `.env` files or any file containing secrets.**
- **Never commit system config** (`/etc/`, kernel params, bootloader config).
- **Never commit personal data** (names, emails, IPs, hardware serials).
- The `.gitignore` should already cover `.env`, `*.key`, `*.pem`. Verify before
  first commit in a session.

## Capability Gaps (what needs building)

1. Lemma-level discrimination — GNN can't pick specific lemma from 16K candidates
2. Multi-step proof chaining — all successful proofs are single-tactic
3. Era-gated test is cosmetic — theorems need physics content in proofs
4. GNN is tiny (1.1M params) — need 5-10M for lemma discrimination

## Win Conditions

See `.hermes/WIN_CONDITIONS.md` for task-type-specific win/pause conditions.
Always verify your task against the appropriate category before completing.

## Commands

```bash
# Train
python scripts/train_explorer.py --domain Algebra --pretrained <path> ...

# Evaluate
python scripts/infer_explorer.py --checkpoint <path> --compare --repeat 3

# Run tests
python -m pytest tests/ -q
```
