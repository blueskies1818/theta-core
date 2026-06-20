# theta-core — Project Context for AI Workers

Autonomous mathematical physics discovery via self-play against Lean 4 proof checker.
AlphaGo Zero analog: formal mathematics as the verifiable environment.

## Architecture (current)

```
Dependency Graph (116K nodes, 436K edges)
       │
       ▼
GNN Encoder (GAT, 1.1M params, GoalEncoder)
       │
       ├──→ Best-First Search (replaced MCTS)
       │    + domain-filtered lemma retrieval
       │    + value network pruning (2-3x speedup)
       │    + dense step-level rewards
       │         │
       │         ▼
       │    Proof Checker (Lean 4.29.1, SHA-256 cache)
       │
       └──→ Correspondence Layer (passive monitoring)
```

## Current Status (June 19, 2026)

- Phase 1: COMPLETE — self-play validated (40%→56% from proof-checker alone)
- Phase 2: ADVANCED — GNN+best-first hybrid, 4/5 honesty gates, v1.0 tagged
- Gates: G1(purity) G2(structural) G3(lemma-novelty) G5(stats) PASS. G4(era-discrimination) FAIL.
- Multi-step proofs: routine. Multi-domain: 5 domains. Ceiling: 15.6% on 64-theorem benchmark.
- Phase 3: design only. Blocked on lemma discrimination bottleneck.

## Key Paths

| Path | Purpose |
|------|---------|
| `src/explorer/` | GNN encoder, best-first search, value network, proof state, dependency graph |
| `src/proof_checker/` | Lean 4 subprocess, batch checker, cache |
| `src/reward/` | Binary + dense step-level rewards |
| `src/correspondence/` | Frontier map, era tracker |
| `scripts/gates/` | Audit scripts (purity, structural, gate4, gate5) |
| `scripts/training/` | Train scripts |
| `scripts/eval/` | Eval, inference, benchmarks |
| `scripts/build/` | Data builders, theorem generators, graph builders |
| `configs/` | YAML configs |

## Hardware

- **Intel Arc B70 drives display — NEVER touch GPU drivers/firmware. No GPU compute.**
- CPU: i5-12600KF, 16 cores. Training: max 6 threads. Eval: max 4 threads.
- Python 3.12, Lean 4.29.1, PyTorch.

## Security

- Never commit: API keys, absolute paths, `.env`, personal data, system config.
- `git diff --cached` before every commit. Scan for `sk-`, `DEEPSEEK_API_KEY`.

## Key Bottleneck

GNN cosine similarity can't discriminate 70K+ lemma candidates. Scaling params doesn't help (14.8M = 1.1M). Current research: enriching graph with proof co-occurrence edges (Path 1), iterative Lean-feedback search (Path 3).

## Commands

```bash
# Train
python scripts/training/train_explorer.py --domain Algebra --pretrained <path> ...
# Evaluate
python scripts/eval/infer_explorer.py --checkpoint <path> --compare --repeat 3
# Audit
python scripts/gates/audit_structural.py
# Tests
python -m pytest tests/ -q
```
