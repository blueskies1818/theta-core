# Training Run 8 — Honest GNN (Proof-Step Pretrained + Heuristic Annealing)

**Date:** 2026-06-09
**Status:** Launching
**Branch:** `main` (uncommitted Honest GNN code)
**Pretrained model:** `checkpoints/gnn/proof_step_pretrained.pt`

---

## 1. Purpose

First training run where the GNN has a real chance to learn. Key changes from Runs 1–7:

1. **Proof-step pretrained GNN** — 1.1M params, 256-dim, 3-layer GAT + GoalEncoder, trained on (goal, lemma) pairs from mathlib4
2. **Heuristic annealing** — full arithmetic heuristics (rfl, comm, assoc, identity, ring, field_simp, linarith, simp) start at scale=1.0 and linearly decay to 0.0 over 2000 epochs
3. **Proof checker verification in MCTS** — candidates verified through Lean during node expansion, invalid steps filtered before child creation
4. **Scaled architecture** — 4× larger than old 230K-param GNN

## 2. Baseline (Pretrained Only, No GRPO Training)

| Metric | Value |
|---|---|
| Proof-step ranking Top-1 | 53.7% (54 physics theorems) |
| Proof-step ranking Top-5 | 59.3% |
| Inference H=1.0 (heuristics) | 60% (6/10 post-era theorems) |
| Inference H=0.0 (pure GNN) | 40% (4/10 post-era theorems) |
| Old system H=0.0 baseline | 0–4% (0–1/25) |

The pretrained GNN at H=0.0 is 10× better than the old system. But all H=0.0 proofs are tactic-based (`simp`, `linarith`), not lemma-based. The GNN hasn't learned lemma discrimination yet.

## 3. Configuration

| Parameter | Value | Why |
|---|---|---|
| Graph | Algebra subgraph — 16,842 nodes, 22,684 edges | |
| GNN | 1,118,848 params, 256-dim, 3-layer GAT, 8 heads, GoalEncoder | Honest GNN scaled arch |
| Training theorems | 29 pre-relativity (≤1904) | Era-gated: classical + classical_crisis + pre_relativity |
| Eval theorems | 25 post-1905 (held-out) | Separate file, evaluated with infer_explorer.py |
| MCTS simulations | 400 per proof | Balance speed vs exploration |
| Eval during training | Disabled | Manual eval with infer_explorer.py after training |
| Batch size | 2 theorems | |
| GRPO group size | 2 proofs per theorem | |
| Learning rate | 1e-3 | |
| Policy weight | 1.0 | |
| Value weight | 0.5 | |
| Heuristic anneal epochs | 2000 | Linear decay 1.0 → 0.0 |
| Heuristic scale min | 0.0 | Pure GNN at end |
| Proof checker | Enabled in MCTS | verify_timeout=5s |
| Device | CPU (XPU Level Zero not installed) | |

## 4. Success Criteria

| Tier | Criterion | Status |
|---|---|---|
| Minimum | Training completes without crashes | |
| Baseline | GNN at H=0.0 proves ≥40% of post-era theorems (matching pretrained) | |
| Target | GNN at H=0.0 proves ≥50% of post-era theorems (beating pretrained) | |
| Stretch | GNN at H=0.0 proves ≥60% of post-era theorems (matching heuristics) | |
| Breakthrough | GNN finds a lemma-based multi-step proof without heuristics | |

## 5. Monitoring Plan

Track per-epoch:
- Success rate, reward, loss, PG loss, KL div
- Proof pattern distribution (rfl, comm, ring, field_simp, linarith, simp, other)
- Heuristic scale, gradient norm
- Correspondence mods

Eval every 100 epochs:
- H=1.0 and H=0.0 inference on 25 held-out theorems
- Proof pattern diversity comparison

## 6. Run Command

```bash
python scripts/train_explorer.py \
  --domain Algebra \
  --pretrained checkpoints/gnn/proof_step_pretrained.pt \
  --theorems data/raw/physics_theorems_full.jsonl \
  --max-theorems 29 \
  --steps 500 \
  --mcts-sims 400 \
  --batch-size 2 \
  --group-size 2 \
  --lr 1e-3 \
  --heuristic-anneal-epochs 2000 \
  --heuristic-scale-min 0.0 \
  --era pre_relativity \
  --eval-theorems 25 \
  --eval-every 100 \
  --save-every 50 \
  --log-every 1 \
  --output checkpoints/explorer/verified_run3
```

## 7. Expected Timeline

| Phase | Epochs | Est. Time | Key Event |
|---|---|---|---|
| Phase 1 | 0–100 | ~3 hours | Cold start. Heuristics at >0.95. High success rate expected. |
| Phase 2 | 100–250 | ~5 hours | Heuristics decay to ~0.875. GNN should start contributing. |
| Phase 3 | 250–500 | ~5 hours | Heuristics decay to ~0.75. Critical phase — does GNN maintain success? |

Total: ~13 hours at 400 sims on CPU.

---

*Run starting 2026-06-09. This is the first training run where the GNN has a real chance.*
