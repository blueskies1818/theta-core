#!/usr/bin/env python3
"""Standalone training script for binary scorer."""
import sys, time
sys.path.insert(0, '.')
from src.scoring.binary_scorer import (
    FrozenGNNEncoder, load_training_pairs, BinaryScorer, train_binary_scorer
)

print("=" * 60, flush=True)
print("BINARY SCORER TRAINING", flush=True)
print("=" * 60, flush=True)

N_PAIRS = 20000
print(f"Training on {N_PAIRS} pairs, 10 epochs", flush=True)

print("Initializing encoder...", flush=True)
enc = FrozenGNNEncoder(
    'checkpoints/gnn/gate2_fullgraph_finetuned.pt',
    'data/graph/dependency_graph_full'
)

print("Loading pairs...", flush=True)
pairs = load_training_pairs('data/raw/proof_step_pairs.jsonl', limit=N_PAIRS)

print(f"Creating scorer (hidden_dim={enc.hidden_dim})...", flush=True)
scorer = BinaryScorer(hidden_dim=enc.hidden_dim)

print("Starting training...", flush=True)
t_start = time.time()
history = train_binary_scorer(
    enc, pairs, scorer,
    'checkpoints/scorer/binary_scorer.pt',
    num_epochs=10,
    batch_size=512,
    lr=1e-3,
    num_threads=6,
    num_negatives=5,
    seed=42,
)
elapsed = time.time() - t_start
print(f"Training complete in {elapsed:.1f}s ({elapsed/60:.1f}m)", flush=True)
print("Checkpoint saved to checkpoints/scorer/binary_scorer.pt", flush=True)
