#!/usr/bin/env python3
"""Minimal training loop test."""
import json, random, sys, time, os
from pathlib import Path

for _env in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_env] = "2"
import torch
torch.set_num_threads(2)
import torch.nn.functional as F
from collections import defaultdict

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))
from src.explorer.dependency_graph import DependencyGraph
from src.explorer.gnn_config import GNNConfig
from src.explorer.gnn_encoder import GNNEncoder, prepare_graph_tensors, extract_initial_features
from scripts.eval.eval_gnn_prover import build_lemma_norm_index, normalize_expression
from src.explorer.mcts import _extract_math_keywords

print("Loading graph...", flush=True)
graph = DependencyGraph.load("data/graph/dependency_graph_full_v3")

# Build lemma index
lemma_to_idx = {}
for node_id in graph.node_ids:
    short_name = node_id.split(".")[-1] if "." in node_id else node_id
    idx = graph.node_id_to_idx(node_id)
    lemma_to_idx[node_id] = idx
    if short_name not in lemma_to_idx:
        lemma_to_idx[short_name] = idx

# Load enriched index
with open("data/graph/dependency_graph_full_v3.lemma_index.json") as f:
    enriched_li = json.load(f)

# Build idx_to_node
idx_to_node = {}
for nid in graph.node_ids:
    idx = graph.node_id_to_idx(nid)
    if idx is not None:
        idx_to_node[idx] = nid

# Load pairs
pairs = []
with open("data/raw/proof_step_pairs.jsonl") as f:
    for line in f:
        d = json.loads(line)
        lemma = d["lemma"]
        resolved = None
        if lemma in enriched_li:
            ni = enriched_li[lemma]
            nd = idx_to_node.get(ni)
            if nd is not None:
                resolved = graph.node_id_to_idx(nd)
        if resolved is None and lemma in lemma_to_idx:
            resolved = lemma_to_idx[lemma]
        if resolved is not None:
            d["_lemma_idx"] = resolved
            pairs.append(d)

print(f"Pairs: {len(pairs)}", flush=True)
random.seed(42)
random.shuffle(pairs)
train_pairs = pairs[:int(len(pairs)*0.9)]
print(f"Train: {len(train_pairs)}", flush=True)

# Build norm index
idx_to_norm = build_lemma_norm_index(graph, lemma_to_idx)

# Config
config = GNNConfig(
    hidden_dim=64, num_layers=2, num_heads=2, input_dim=64,
    use_edge_types=True, num_edge_types=6,
)
gnn = GNNEncoder(config)
print(f"Params: {sum(p.numel() for p in gnn.parameters()):,}", flush=True)

# Graph tensors
features = extract_initial_features(graph, config)
sources, targets, edge_types, num_nodes = prepare_graph_tensors(graph)
print(f"Edges: {sources.shape[0]}", flush=True)

# Forward pass
print("GNN forward...", flush=True)
with torch.no_grad():
    embeddings = gnn(features, sources, targets, edge_types, num_nodes)
embeddings = embeddings.detach()
embeddings_norm = F.normalize(embeddings, dim=-1)
print(f"Embeddings: {embeddings.shape}", flush=True)

# Precompute goal contexts (simplified)
print("Precomputing contexts...", flush=True)
norm_to_indices = defaultdict(list)
for idx, norm in idx_to_norm.items():
    norm_to_indices[norm].append(idx)

train_contexts = []
for p in train_pairs:
    kw = _extract_math_keywords(p["goal"])
    matched = []
    for k in kw:
        kn = normalize_expression(k)
        if kn in norm_to_indices:
            matched.extend(norm_to_indices[kn][:5])
    train_contexts.append(list(set(matched)))
print(f"Contexts done, sample lens: {[len(c) for c in train_contexts[:5]]}", flush=True)

# Raw context embeddings
raw_train_ctxs = torch.zeros(len(train_pairs), config.hidden_dim)
for i, ctx in enumerate(train_contexts):
    if ctx:
        idx_t = torch.tensor(ctx)
        raw_train_ctxs[i] = embeddings_norm[idx_t].mean(dim=0)
print(f"Raw ctx: {raw_train_ctxs.shape}", flush=True)

# Training targets
train_targets = torch.tensor([p["_lemma_idx"] for p in train_pairs])

# Optimizer
gnn.goal_encoder.train()
optimizer = torch.optim.AdamW(gnn.goal_encoder.parameters(), lr=1e-3)

print("Starting training loop...", flush=True)
indices = list(range(len(train_pairs)))
random.shuffle(indices)

for bi, batch_start in enumerate(range(0, len(indices), 64)):
    batch_idx = indices[batch_start:batch_start + 64]
    if len(batch_idx) < 2:
        continue
    
    batch_idx_t = torch.tensor(batch_idx)
    batch_goal = gnn.goal_encoder(raw_train_ctxs[batch_idx_t])
    batch_tgt = train_targets[batch_idx]
    
    logits = torch.matmul(batch_goal, embeddings_norm.T) / 1.0
    loss = F.cross_entropy(logits, batch_tgt)
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(gnn.parameters(), 1.0)
    optimizer.step()
    
    if bi < 3:
        print(f"  Batch {bi}: loss={loss.item():.4f}", flush=True)

print("Training loop complete!", flush=True)
