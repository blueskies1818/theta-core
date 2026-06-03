#!/usr/bin/env python3
"""Test MCTS proof generation on all physics theorems with new heuristics."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.explorer.dependency_graph import DependencyGraph
from src.explorer.gnn_encoder import GNNEncoder, prepare_graph_tensors, extract_initial_features
from src.explorer.mcts import MCTS, MCTSConfig
from src.explorer.proof_state import ProofState
from src.proof_checker.batch_checker import BatchChecker
from src.proof_checker.formats import wrap_theorem_with_proof


def main():
    graph = DependencyGraph.load('data/graph/dependency_graph')
    graph = graph.domain_subgraph('Algebra')
    print(f"Graph: {graph.num_nodes} nodes")

    gnn = GNNEncoder.load(str(Path('checkpoints/gnn/gnn_best.pt').resolve()))
    gnn.eval()

    features = extract_initial_features(graph, gnn.config)
    sources, targets, edge_types, num_nodes = prepare_graph_tensors(graph)
    with torch.no_grad():
        embeddings = gnn(features, sources, targets, edge_types, num_nodes)

    mcts = MCTS(gnn_encoder=gnn, dependency_graph=graph,
                config=MCTSConfig(num_simulations=200, max_depth=5, top_k_lemmas=30))
    mcts.set_embeddings(embeddings, sorted(graph.node_ids))

    with open('data/raw/physics_theorems.jsonl') as f:
        theorems = [json.loads(line) for line in f]

    checker = BatchChecker(timeout=30, max_workers=4, cache_size=64)

    print(f"\nTesting {len(theorems)} theorems...\n")
    results = []
    for t in theorems:
        stmt = t['statement']
        name = t['name']
        best_steps, root = mcts.search(stmt, verbose=False)
        proof_text = ProofState._render_proof(best_steps)
        full_code = wrap_theorem_with_proof(stmt, proof_text or 'sorry')
        check_results = checker.check_batch([full_code])
        ok = check_results[0].success
        err = check_results[0].errors[0][:100] if check_results[0].errors else ""
        top_action = best_steps[0].to_lean() if best_steps else "none"
        results.append((ok, name, top_action, err, t['frontier_zone']))
        status = "✓" if ok else "✗"
        print(f"  {status} [{t['frontier_zone']:25s}] {name:40s} | {top_action:30s} {'| ' + err if err else ''}")

    passed = sum(1 for r in results if r[0])
    print(f"\n{passed}/{len(theorems)} passed ({passed/len(theorems)*100:.0f}%)")

    from collections import Counter
    zone_pass = Counter()
    zone_total = Counter()
    for ok, name, action, err, zone in results:
        zone_total[zone] += 1
        if ok:
            zone_pass[zone] += 1

    print("\nZone coverage:")
    for zone in sorted(zone_total):
        p = zone_pass.get(zone, 0)
        t = zone_total[zone]
        print(f"  {zone:30s}: {p}/{t} ({p/t*100:.0f}%)")

    try:
        checker.shutdown()
    except:
        pass


if __name__ == '__main__':
    main()
