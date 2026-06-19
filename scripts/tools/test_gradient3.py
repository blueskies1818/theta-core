#!/usr/bin/env python3
"""Trace gradient flow from child_logits back to GNN parameters."""
import sys
from pathlib import Path
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

import torch
import torch.nn.functional as F

from src.explorer.dependency_graph import DependencyGraph
from src.explorer.gnn_encoder import GNNEncoder, extract_initial_features, prepare_graph_tensors
from src.explorer.mcts import MCTS, MCTSConfig

if __name__ == "__main__":
    graph = DependencyGraph.load(_project_root / "data/graph/dependency_graph")
    graph = graph.domain_subgraph("Algebra")

    def make_gnn():
        gnn = GNNEncoder.load(str(_project_root / "checkpoints/explorer_wave2/gnn_final.pt"))
        gnn.train()
        return gnn

    def make_embeddings(gnn, graph):
        features = extract_initial_features(graph, gnn.config)
        sources, targets, edge_types, num_nodes = prepare_graph_tensors(graph)
        return gnn(features, sources, targets, edge_types, num_nodes)

    # Test 1: Direct gradient from embeddings
    print("=== Test 1: Direct gradient from embeddings ===")
    gnn1 = make_gnn()
    emb1 = make_embeddings(gnn1, graph)
    lemma_emb = emb1[100]
    loss1 = lemma_emb.sum()
    gnn1.zero_grad()
    loss1.backward()
    grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in gnn1.parameters() if p.grad is not None) ** 0.5
    print(f"  loss1.sum() → grad_norm: {grad_norm:.6f}")

    # Test 2: cosine_similarity
    print("\n=== Test 2: cosine_similarity gradient ===")
    gnn2 = make_gnn()
    emb2 = make_embeddings(gnn2, graph)
    lemma_emb_a = emb2[100]
    lemma_emb_b = emb2[200]
    sim = torch.cosine_similarity(lemma_emb_a.unsqueeze(0), lemma_emb_b.unsqueeze(0), dim=1)
    loss2 = sim.squeeze()
    gnn2.zero_grad()
    loss2.backward()
    grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in gnn2.parameters() if p.grad is not None) ** 0.5
    print(f"  cosine_sim.backward() → grad_norm: {grad_norm:.6f}")

    # Test 3: clamp + scale (full _score_actions path)
    print("\n=== Test 3: clamp + scale ===")
    gnn3 = make_gnn()
    emb3 = make_embeddings(gnn3, graph)
    lemma_emb_a = emb3[100]
    lemma_emb_b = emb3[200]
    sim = torch.cosine_similarity(lemma_emb_a.unsqueeze(0), lemma_emb_b.unsqueeze(0), dim=1)
    clamped = sim.clamp(min=0.0)
    scaled = clamped.squeeze() * 0.8
    score = torch.tensor(0.05) + scaled
    gnn3.zero_grad()
    score.backward()
    grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in gnn3.parameters() if p.grad is not None) ** 0.5
    print(f"  score.backward() → grad_norm: {grad_norm:.6f}")

    # Test 4: Full MCTS path
    print("\n=== Test 4: Full MCTS path (logits grad_fn check) ===")
    gnn4 = make_gnn()
    emb4 = make_embeddings(gnn4, graph)
    
    mcts_config = MCTSConfig(
        num_simulations=3, max_depth=10, top_k_lemmas=30,
        c_puct=1.4, heuristic_scale=0.0, use_proof_checker=False,
    )
    mcts = MCTS(gnn_encoder=gnn4, dependency_graph=graph, config=mcts_config, proof_checker=None)
    mcts.set_embeddings(emb4, sorted(graph.node_ids))

    test_stmt = "theorem square_expansion (a b : ℝ) : (a + b)^2 = a^2 + 2*a*b + b^2 := by"
    best_steps, root = mcts.search(test_stmt, verbose=False)

    if root.child_logits is not None:
        grad_connected = 0
        no_grad = 0
        for i in range(len(root.child_logits)):
            if root.child_logits[i].grad_fn is not None:
                grad_connected += 1
            else:
                no_grad += 1
        print(f"  logits: {grad_connected} with grad_fn, {no_grad} without")

        total_visits = sum(c.visit_count for c in root.children.values())
        if total_visits > 0:
            target_probs = []
            for action in root._child_action_order:
                child = root.children.get(action)
                if child is not None:
                    target_probs.append(child.visit_count / total_visits)
                else:
                    target_probs.append(0.0)
            
            target_sum = sum(target_probs)
            target = torch.tensor([p / target_sum for p in target_probs])
            logits = root.child_logits
            log_probs = torch.log_softmax(logits, dim=0)
            policy_loss = -(target * log_probs).sum()
            
            print(f"  policy_loss = {policy_loss.item():.6f}")
            print(f"  policy_loss.requires_grad = {policy_loss.requires_grad}")
            print(f"  policy_loss.grad_fn = {policy_loss.grad_fn}")
            
            gnn4.zero_grad()
            policy_loss.backward()
            grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in gnn4.parameters() if p.grad is not None) ** 0.5
            print(f"  policy_loss → grad_norm: {grad_norm:.10f}")
            
            non_zero = 0
            for p in gnn4.parameters():
                if p.grad is not None and p.grad.data.norm(2).item() > 1e-10:
                    non_zero += 1
            total_p = sum(1 for _ in gnn4.parameters())
            print(f"  Non-zero grad params: {non_zero}/{total_p}")
