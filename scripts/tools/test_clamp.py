#!/usr/bin/env python3
"""Verify the clamp(min=0.0) gradient-killing hypothesis."""
import sys
from pathlib import Path
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

import torch
import torch.nn.functional as F
from src.explorer.dependency_graph import DependencyGraph
from src.explorer.gnn_encoder import GNNEncoder, extract_initial_features, prepare_graph_tensors

graph = DependencyGraph.load(_project_root / "data/graph/dependency_graph")
graph = graph.domain_subgraph("Algebra")
gnn = GNNEncoder.load(str(_project_root / "checkpoints/explorer_wave2/gnn_final.pt"))
gnn.train()

features = extract_initial_features(graph, gnn.config)
sources, targets, edge_types, num_nodes = prepare_graph_tensors(graph)
embeddings = gnn(features, sources, targets, edge_types, num_nodes)

# Check cosine similarity values for random lemma pairs
print("=== Cosine similarity between random lemma pairs ===")
neg_count = 0
pos_count = 0
for i in range(100):
    a = embeddings[i]
    b = embeddings[i + 100]
    sim = torch.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=1).item()
    if sim < 0:
        neg_count += 1
    else:
        pos_count += 1
print(f"  Negative: {neg_count}, Positive: {pos_count}")

# Test: what happens when we clamp a negative cosine sim?
print("\n=== Gradient through clamp(min=0.0) on negative sim ===")
gnn2 = GNNEncoder.load(str(_project_root / "checkpoints/explorer_wave2/gnn_final.pt"))
gnn2.train()
features2 = extract_initial_features(graph, gnn2.config)
sources2, targets2, edge_types2, num_nodes2 = prepare_graph_tensors(graph)
emb2 = gnn2(features2, sources2, targets2, edge_types2, num_nodes2)

# Find a pair with negative cosine similarity
for i in range(20):
    for j in range(i+1, min(i+20, emb2.size(0))):
        a = emb2[i]
        b = emb2[j]
        sim_raw = torch.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=1)
        sim_val = sim_raw.item()
        
        # Test WITHOUT clamp
        gnn2.zero_grad()
        score_no_clamp = torch.tensor(0.05) + sim_raw.squeeze() * 0.8
        score_no_clamp.backward(retain_graph=True)
        grad_no = sum(p.grad.data.norm(2).item() ** 2 for p in gnn2.parameters() if p.grad is not None) ** 0.5
        
        # Test WITH clamp
        gnn2.zero_grad()
        clamped = sim_raw.clamp(min=0.0)
        score_clamped = torch.tensor(0.05) + clamped.squeeze() * 0.8
        score_clamped.backward()
        grad_clamped = sum(p.grad.data.norm(2).item() ** 2 for p in gnn2.parameters() if p.grad is not None) ** 0.5
        
        print(f"  sim={sim_val:+.4f} → no_clamp grad={grad_no:.6f}, clamp grad={grad_clamped:.6f}")
        break
    else:
        continue
    break

# Test softplus alternative
print("\n=== Test softplus alternative ===")
gnn3 = GNNEncoder.load(str(_project_root / "checkpoints/explorer_wave2/gnn_final.pt"))
gnn3.train()
features3 = extract_initial_features(graph, gnn3.config)
sources3, targets3, edge_types3, num_nodes3 = prepare_graph_tensors(graph)
emb3 = gnn3(features3, sources3, targets3, edge_types3, num_nodes3)

for i in range(20):
    for j in range(i+1, min(i+20, emb3.size(0))):
        a = emb3[i]
        b = emb3[j]
        sim = torch.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=1)
        sim_val = sim.item()
        
        # softplus: smooth approximation of ReLU
        gnn3.zero_grad()
        sp = F.softplus(sim.squeeze())  # log(1+exp(x)), always positive, always differentiable
        score = torch.tensor(0.05) + sp * 0.8
        score.backward()
        grad_sp = sum(p.grad.data.norm(2).item() ** 2 for p in gnn3.parameters() if p.grad is not None) ** 0.5
        
        print(f"  sim={sim_val:+.4f} → softplus grad={grad_sp:.6f}")
        break
    else:
        continue
    break
