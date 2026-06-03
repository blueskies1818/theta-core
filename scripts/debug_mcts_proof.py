#!/usr/bin/env python3
"""Debug MCTS proof generation — trace why proofs are failing."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.explorer.dependency_graph import DependencyGraph
from src.explorer.gnn_config import GNNConfig
from src.explorer.gnn_encoder import GNNEncoder, prepare_graph_tensors, extract_initial_features
from src.explorer.mcts import MCTS, MCTSConfig, _is_reflexive_goal
from src.explorer.proof_state import ProofState
from src.proof_checker.batch_checker import BatchChecker
from src.proof_checker.formats import wrap_theorem_with_proof

# Load graph and GNN — use the same approach as train_explorer.py
graph = DependencyGraph.load('data/graph/dependency_graph')
graph = graph.domain_subgraph('Algebra')
print(f"Graph: {graph.num_nodes} nodes")

gnn = GNNEncoder.load(str(Path('checkpoints/gnn/gnn_best.pt').resolve()))
gnn.eval()
total_params = sum(p.numel() for p in gnn.parameters())
print(f"GNN: {total_params:,} params")

features = extract_initial_features(graph, gnn.config)
sources, targets, edge_types, num_nodes = prepare_graph_tensors(graph)
with torch.no_grad():
    embeddings = gnn(features, sources, targets, edge_types, num_nodes)

mcts = MCTS(gnn_encoder=gnn, dependency_graph=graph,
            config=MCTSConfig(num_simulations=200, max_depth=5, top_k_lemmas=30))
mcts.set_embeddings(embeddings, sorted(graph.node_ids))

# Test each theorem
checker = BatchChecker(timeout=30, max_workers=1, cache_size=64)

import json
with open('data/raw/physics_theorems.jsonl') as f:
    theorems = [json.loads(line) for line in f]

for t in theorems[:5]:  # First 5
    stmt = t['statement']
    name = t['name']
    ground_truth = t['proof']
    print(f"\n{'='*60}")
    print(f"Theorem: {name}")
    print(f"Statement: {stmt[:100]}...")
    is_reflexive = _is_reflexive_goal(stmt.split(':')[-1].strip() if ':' in stmt else stmt)
    print(f"Reflexive: {is_reflexive}")
    print(f"Ground truth: {ground_truth.strip()}")

    best_steps, root = mcts.search(stmt, verbose=False)
    proof_text = ProofState._render_proof(best_steps)
    full_code = wrap_theorem_with_proof(stmt, proof_text or 'sorry')

    print(f"MCTS steps ({len(best_steps)}): {[s.to_lean() for s in best_steps]}")
    print(f"Rendered proof: {proof_text}")

    # Check the MCTS proof
    results = checker.check_batch([full_code])
    if results[0].success:
        print(f"✓ PROOF VALID!")
    else:
        err = results[0].errors[0][:200] if results[0].errors else "unknown"
        print(f"✗ FAILED: {err}")

    # Also check the ground truth for comparison
    gt_code = wrap_theorem_with_proof(stmt, ground_truth)
    gt_results = checker.check_batch([gt_code])
    print(f"Ground truth: {'✓' if gt_results[0].success else '✗ ' + str(gt_results[0].errors[0][:100] if gt_results[0].errors else '')}")

    # Show top actions by MCTS visits
    if root.children:
        ranked = sorted(root.children.items(), key=lambda x: x[1].visit_count, reverse=True)
        print(f"\nTop actions (of {len(root.children)}):")
        for action, child in ranked[:5]:
            print(f"  visits={child.visit_count:3d} prior={child.prior:.4f} | {action.to_lean()}")

    print(f"\nFull code being checked:")
    print(full_code[:300])

try:
    checker.shutdown()
except:
    pass
