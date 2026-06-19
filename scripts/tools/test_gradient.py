#!/usr/bin/env python3
"""Quick test to verify gradient flow in the ExplorerTrainer."""
import sys
from pathlib import Path
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

import torch
import torch.nn.functional as F

from src.explorer.dependency_graph import DependencyGraph
from src.explorer.gnn_encoder import GNNEncoder, extract_initial_features, prepare_graph_tensors
from src.explorer.mcts import MCTS, MCTSConfig
from src.explorer.explorer_trainer import ExplorerTrainer, ExplorerConfig
from src.proof_checker.batch_checker import BatchChecker

# Load graph
graph = DependencyGraph.load(_project_root / "data/graph/dependency_graph")
graph = graph.domain_subgraph("Algebra")
print(f"Graph: {graph.num_nodes} nodes")

# Load GNN
gnn = GNNEncoder.load(str(_project_root / "checkpoints/explorer_wave2/gnn_final.pt"))
gnn.train()
print(f"GNN: {sum(p.numel() for p in gnn.parameters()):,} params")

# Compute embeddings with grad
features = extract_initial_features(graph, gnn.config)
sources, targets, edge_types, num_nodes = prepare_graph_tensors(graph)

embeddings = gnn(features, sources, targets, edge_types, num_nodes)
print(f"Embeddings shape: {embeddings.shape}")
print(f"Embeddings requires_grad: {embeddings.requires_grad}")

# Test gradient: pick a lemma embedding and compute a simple loss
lemma_idx = 100
if lemma_idx < embeddings.size(0):
    lemma_emb = embeddings[lemma_idx]
    loss = lemma_emb.norm()
    print(f"Loss requires_grad: {loss.requires_grad}")

    # Check if backward works
    gnn.zero_grad()
    loss.backward()
    
    total_grad = 0.0
    for p in gnn.parameters():
        if p.grad is not None:
            total_grad += p.grad.data.norm(2).item() ** 2
    total_grad = total_grad ** 0.5
    print(f"Gradient norm (embedding→loss test): {total_grad:.6f}")

# Now test the full MCTS + child_logits path
print("\n--- Testing MCTS child_logits gradient ---")
checker = BatchChecker(timeout=30, max_workers=4, cache_size=128)

mcts_config = MCTSConfig(
    num_simulations=5,
    max_depth=10,
    top_k_lemmas=30,
    c_puct=1.4,
    heuristic_scale=0.0,
    use_proof_checker=True,
    verify_timeout=5.0,
)

mcts = MCTS(gnn_encoder=gnn, dependency_graph=graph, config=mcts_config, proof_checker=checker)
mcts.set_embeddings(embeddings, sorted(graph.node_ids))

# Run a quick search
test_stmt = "theorem square_expansion (a b : ℝ) : (a + b)^2 = a^2 + 2*a*b + b^2 := by"
best_steps, root = mcts.search(test_stmt, verbose=False)

print(f"Root children: {len(root.children) if root.children else 0}")
print(f"Root child_logits is None: {root.child_logits is None}")
if root.child_logits is not None:
    print(f"child_logits shape: {root.child_logits.shape}")
    print(f"child_logits requires_grad: {root.child_logits.requires_grad}")
    print(f"child_logits grad_fn: {root.child_logits.grad_fn}")
    
    # Check each child_logits element for grad_fn
    for i in range(min(3, len(root.child_logits))):
        print(f"  logits[{i}] grad_fn: {root.child_logits[i].grad_fn}")

# Try computing policy loss and backprop
if root.children and root.child_logits is not None and root._child_action_order:
    total_visits = sum(c.visit_count for c in root.children.values())
    if total_visits > 0:
        target_probs = []
        for action in root._child_action_order:
            child = root.children.get(action)
            if child is not None:
                target_probs.append(child.visit_count / total_visits)
            else:
                target_probs.append(0.0)
        
        if target_probs and sum(target_probs) > 0:
            target_sum = sum(target_probs)
            target = torch.tensor([p / target_sum for p in target_probs])
            logits = root.child_logits
            log_probs = torch.log_softmax(logits, dim=0)
            policy_loss = -(target * log_probs).sum()
            
            print(f"\nPolicy loss: {policy_loss.item():.6f}")
            print(f"Policy loss requires_grad: {policy_loss.requires_grad}")
            print(f"Policy loss grad_fn: {policy_loss.grad_fn}")
            
            gnn.zero_grad()
            policy_loss.backward()
            
            total_grad = 0.0
            for p in gnn.parameters():
                if p.grad is not None:
                    total_grad += p.grad.data.norm(2).item() ** 2
            total_grad = total_grad ** 0.5
            print(f"Gradient norm (policy_loss→GNN): {total_grad:.6f}")

checker.shutdown()
