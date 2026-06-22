# GNN + Hard-Negative Adapter — Results Report

**Date:** June 20, 2026
**Architecture:** Frozen GNN (1.1M) + GNNAdapterHead (130K) + Best-First Search

## Training Configuration

| Parameter | Value |
|-----------|-------|
| GNN Checkpoint | `checkpoints/gnn/full_graph_pretrained.pt` |
| GNN Parameters (frozen) | 1,118,848 |
| Adapter Parameters (trainable) | 132,096 |
| Training Pairs | 226,003 proof-step pairs (pre-1905 era) |
| Train/Val Split | 80/20 (180,802 / 45,201) |
| Batch Size | 256 |
| Epochs | 20 |
| Optimizer | AdamW (lr=1e-3, wd=1e-5) |
| Scheduler | CosineAnnealingLR |
| Multi-task Loss | InfoNCE (τ=0.07) + TripletMargin (0.5×) + LinkPredAnchor (0.1×) |
| Triplet Margin | 0.3 |
| Hardware | CPU, 4 threads |

## Safety Gates

| Gate | Description | Status | Detail |
|------|-------------|--------|--------|
| A | GNN frozen, ≤150K trainable | TBD | |
| B | Link-prediction preservation ≤20% above baseline | TBD | |
| C | Validation retrieval MRR > 0.786 | TBD | |
| D | Embedding diversity (std > 0.1, rank = 256) | TBD | |
| E | Gate 3 proof success > 15.6% | TBD | |

## Training Results

TBD — full training in progress.

### Loss Curves

TBD

### Validation MRR

TBD

### Gate 3 Evaluation (64 theorems)

TBD

## Comparison to Baseline

| Metric | GNN Baseline | GNN + Adapter | Change |
|--------|-------------|---------------|--------|
| Gate 3 Rate | 15.6% | TBD | TBD |
| Multi-step proofs | TBD | TBD | TBD |
| Lemma-novelty proofs | TBD | TBD | TBD |
| Analysis domain | TBD | TBD | TBD |
| Physics domain | TBD | TBD | TBD |

## Honesty Gate Re-run

| Gate | Pre-training | Post-training |
|------|-------------|---------------|
| G1 (Purity) | PASS | TBD |
| G2 (Structural Independence) | PASS | TBD |
| G3 (Lemma Novelty) | 15.6% | TBD |
| G4 (Era Discrimination) | FAIL (pre-existing) | TBD |
| G5 (Statistical Validation) | TBD | TBD |

## Analysis

TBD

## Next Steps

TBD
