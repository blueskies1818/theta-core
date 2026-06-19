#!/usr/bin/env python3
"""Pinpoint: why does clamp produce zero gradient for specific indices?"""
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

# Test with indices 100, 200 (same as original test 3)
print("=== Test with indices [100], [200] ===")
a = embeddings[100]
b = embeddings[200]
sim = torch.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=1)
print(f"  sim value: {sim.item():.6f}")

# No clamp
gnn.zero_grad()
s = torch.tensor(0.05) + sim.squeeze() * 0.8
s.backward()
g1 = sum(p.grad.data.norm(2).item() ** 2 for p in gnn.parameters() if p.grad is not None) ** 0.5
print(f"  no_clamp grad: {g1:.10f}")

# Need fresh GNN+embeddings since backward consumed the graph
gnn2 = GNNEncoder.load(str(_project_root / "checkpoints/explorer_wave2/gnn_final.pt"))
gnn2.train()
features2 = extract_initial_features(graph, gnn2.config)
sources2, targets2, edge_types2, num_nodes2 = prepare_graph_tensors(graph)
emb2 = gnn2(features2, sources2, targets2, edge_types2, num_nodes2)

a2 = emb2[100]
b2 = emb2[200]
sim2 = torch.cosine_similarity(a2.unsqueeze(0), b2.unsqueeze(0), dim=1)
print(f"  sim2 value: {sim2.item():.6f}")

# With clamp
gnn2.zero_grad()
s2 = torch.tensor(0.05) + sim2.clamp(min=0.0).squeeze() * 0.8
s2.backward()
g2 = sum(p.grad.data.norm(2).item() ** 2 for p in gnn2.parameters() if p.grad is not None) ** 0.5
print(f"  clamp grad: {g2:.10f}")

# Now test with the _score_actions flow for a negative sim
print("\n=== Find and test negative sim ===")
gnn3 = GNNEncoder.load(str(_project_root / "checkpoints/explorer_wave2/gnn_final.pt"))
gnn3.train()
features3 = extract_initial_features(graph, gnn3.config)
sources3, targets3, edge_types3, num_nodes3 = prepare_graph_tensors(graph)
emb3 = gnn3(features3, sources3, targets3, edge_types3, num_nodes3)

found_neg = False
for i in range(50):
    for j in range(i+1, min(i+50, emb3.size(0))):
        a = emb3[i]
        b = emb3[j]
        sim_raw = torch.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=1)
        if sim_raw.item() < 0:
            print(f"  Negative pair found: [{i}], [{j}], sim={sim_raw.item():.6f}")
            found_neg = True
            
            # No clamp
            gnn3_copy = GNNEncoder.load(str(_project_root / "checkpoints/explorer_wave2/gnn_final.pt"))
            gnn3_copy.train()
            f3c = extract_initial_features(graph, gnn3_copy.config)
            s3, t3, e3, n3 = prepare_graph_tensors(graph)
            e3c = gnn3_copy(f3c, s3, t3, e3, n3)
            a3 = e3c[i]
            b3 = e3c[j]
            sim3 = torch.cosine_similarity(a3.unsqueeze(0), b3.unsqueeze(0), dim=1)
            gnn3_copy.zero_grad()
            s_c = torch.tensor(0.05) + sim3.squeeze() * 0.8
            s_c.backward()
            g_no = sum(p.grad.data.norm(2).item() ** 2 for p in gnn3_copy.parameters() if p.grad is not None) ** 0.5
            print(f"    no_clamp: grad={g_no:.6f}")
            
            # With clamp
            gnn3_copy2 = GNNEncoder.load(str(_project_root / "checkpoints/explorer_wave2/gnn_final.pt"))
            gnn3_copy2.train()
            f3c2 = extract_initial_features(graph, gnn3_copy2.config)
            s32, t32, e32, n32 = prepare_graph_tensors(graph)
            e3c2 = gnn3_copy2(f3c2, s32, t32, e32, n32)
            a32 = e3c2[i]
            b32 = e3c2[j]
            sim32 = torch.cosine_similarity(a32.unsqueeze(0), b32.unsqueeze(0), dim=1)
            gnn3_copy2.zero_grad()
            s_c2 = torch.tensor(0.05) + sim32.clamp(min=0.0).squeeze() * 0.8
            s_c2.backward()
            g_cl = sum(p.grad.data.norm(2).item() ** 2 for p in gnn3_copy2.parameters() if p.grad is not None) ** 0.5
            print(f"    clamp:    grad={g_cl:.6f}")
            break
    if found_neg:
        break

if not found_neg:
    print("  No negative pair found in first 50×50")
