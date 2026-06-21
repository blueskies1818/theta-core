#!/usr/bin/env python3
"""Debug: forward pass only, no training."""
import json, random, sys, time
from pathlib import Path
import os as _os
for _env in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    _os.environ[_env] = "4"
import torch
torch.set_num_threads(4)
import torch.nn.functional as F

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))
from src.explorer.dependency_graph import DependencyGraph
from src.explorer.gnn_config import GNNConfig
from src.explorer.gnn_encoder import GNNEncoder, prepare_graph_tensors, extract_initial_features
from scripts.eval.eval_gnn_prover import build_lemma_norm_index, normalize_expression
from collections import defaultdict

print("Loading graph...", flush=True)
graph = DependencyGraph.load("data/graph/dependency_graph_full_v3")
print(f"Graph: {graph.summary()}", flush=True)

# Build lemma index
lemma_to_idx = {}
for node_id in graph.node_ids:
    short_name = node_id.split(".")[-1] if "." in node_id else node_id
    idx = graph.node_id_to_idx(node_id)
    lemma_to_idx[node_id] = idx
    if short_name not in lemma_to_idx:
        lemma_to_idx[short_name] = idx
print(f"Lemma index: {len(lemma_to_idx)} entries", flush=True)

# Create config
config = GNNConfig(
    hidden_dim=64, num_layers=2, num_heads=2,
    input_dim=64, dropout=0.1, activation="gelu",
    use_goal_encoder=True, goal_encoder_expansion=2,
    use_edge_types=True, num_edge_types=6,
)
gnn = GNNEncoder(config)
print(f"GNN: {sum(p.numel() for p in gnn.parameters()):,} params", flush=True)

# Graph tensors
print("Computing graph tensors...", flush=True)
features = extract_initial_features(graph, config)
sources, targets, edge_types, num_nodes = prepare_graph_tensors(graph)
print(f"  features: {features.shape}", flush=True)
print(f"  sources: {sources.shape}, edge_types: {edge_types.unique(return_counts=True)}", flush=True)

# Forward pass
print("Forward pass...", flush=True)
t0 = time.time()
with torch.no_grad():
    embeddings = gnn(features, sources, targets, edge_types, num_nodes)
print(f"  embeddings: {embeddings.shape}, time: {time.time()-t0:.1f}s", flush=True)

embeddings_norm = F.normalize(embeddings.detach(), dim=-1)

# Test goal_encoder forward
print("Testing goal_encoder...", flush=True)
test_input = torch.randn(4, 64)
out = gnn.goal_encoder(test_input)
print(f"  goal_encoder output: {out.shape}", flush=True)

# Test full training forward (one batch)
print("Testing batch logits...", flush=True)
batch_goal_embs = gnn.goal_encoder(torch.randn(16, 64))
logits = torch.matmul(batch_goal_embs, embeddings_norm.T) / 1.0
print(f"  logits: {logits.shape}", flush=True)

loss = F.cross_entropy(logits, torch.randint(0, num_nodes, (16,)))
print(f"  loss: {loss.item():.4f}", flush=True)

# Test backward
print("Testing backward...", flush=True)
gnn.goal_encoder.train()
batch_goal_embs = gnn.goal_encoder(torch.randn(16, 64))
logits = torch.matmul(batch_goal_embs, embeddings_norm.T) / 1.0
loss = F.cross_entropy(logits, torch.randint(0, num_nodes, (16,)))
loss.backward()
print(f"  backward OK, loss: {loss.item():.4f}", flush=True)

print("All tests passed!", flush=True)
