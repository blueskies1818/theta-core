#!/usr/bin/env python3
"""Debug: Run theorem 3 (alg_cross_multiply) with verbose output and timing."""
import sys, time, json
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import torch
torch.set_num_threads(4)

from src.explorer.dependency_graph import DependencyGraph
from src.explorer.gnn_encoder import GNNEncoder, extract_initial_features, prepare_graph_tensors
from src.explorer.gnn_best_first_search import GNNBestFirstSearch, GNNBestFirstConfig
from src.explorer.proof_state import ProofState
from src.proof_checker.batch_checker import BatchChecker
from src.proof_checker.formats import wrap_theorem_with_proof
from scripts.eval.eval_gnn_prover import build_lemma_index, extract_conclusion, normalize_expression

def build_norm_index(graph, lemma_to_idx):
    idx_to_norm = {}
    for node_id in graph.node_ids:
        idx = lemma_to_idx.get(node_id)
        if idx is None:
            continue
        node = graph.get_node(node_id)
        if node:
            statement = node.get("statement", "")
            if statement:
                conclusion = extract_conclusion(statement)
                if conclusion:
                    idx_to_norm[idx] = normalize_expression(conclusion)
    return idx_to_norm

def main():
    theorem = {
        "name": "alg_cross_multiply",
        "statement": "theorem alg_cross_multiply (a b c d : ℝ) (hb : b ≠ 0) (hd : d ≠ 0) (h : a / b = c / d) : a * d = b * c",
        "domain": "algebra",
        "proof": "have h' := congrArg (fun t => t * (b * d)) h; field_simp [hb, hd] at h'; exact h'",
    }

    t0 = time.time()
    print(f"Loading GNN... ({time.time()-t0:.1f}s)")
    gnn = GNNEncoder.load(str(_PROJECT_ROOT / "checkpoints/gnn/gate2_fullgraph_finetuned.pt"))
    gnn.eval()
    print(f"  Params: {sum(p.numel() for p in gnn.parameters()):,}")

    print(f"Loading graph... ({time.time()-t0:.1f}s)")
    graph = DependencyGraph.load(_PROJECT_ROOT / "data/graph/dependency_graph_full")
    print(f"  {graph.summary()}")

    print(f"Computing node embeddings... ({time.time()-t0:.1f}s)")
    features = extract_initial_features(graph, gnn.config)
    sources, targets, edge_types, num_nodes = prepare_graph_tensors(graph)
    with torch.no_grad():
        node_embeddings = gnn(features, sources, targets, edge_types, num_nodes)

    lemma_to_idx = build_lemma_index(graph)
    idx_to_norm = build_norm_index(graph, lemma_to_idx)

    config = GNNBestFirstConfig(
        max_depth=20, max_expansions=50, top_k_lemmas=10,
        use_proof_checker=True, num_threads=4,
        verify_timeout=5.0, max_graph_candidates=200,
    )
    checker = BatchChecker(timeout=8, max_workers=4, cache_size=128)

    print(f"Creating GNNBestFirstSearch... ({time.time()-t0:.1f}s)")
    bf = GNNBestFirstSearch(
        gnn=gnn, graph=graph, node_embeddings=node_embeddings,
        lemma_index=lemma_to_idx, idx_to_norm=idx_to_norm,
        config=config, proof_checker=checker,
    )

    print(f"\nDomain index: {len(bf._domain_node_ids)} unique graph domains")
    print(f"All domain names: {sorted(bf._all_domain_names)[:20]}...")

    # Test domain filtering
    for use_domain in [True, False]:
        d = theorem['domain'] if use_domain else None
        t1 = time.time()
        lemmas = bf._get_relevant_lemmas(theorem['statement'], domain=d)
        dt = time.time() - t1
        print(f"\nDomain filter={'ON' if use_domain else 'OFF'}: {len(lemmas)} candidates ({dt:.2f}s)")
        for lem in lemmas[:10]:
            print(f"  - {lem}")

    print(f"\n{'='*60}")
    print(f"Running search on theorem: {theorem['name']}")
    print(f"  Statement: {theorem['statement'][:80]}...")
    print(f"  Max expansions: {config.max_expansions}, Top-K: {config.top_k_lemmas}")
    print(f"{'='*60}")

    t1 = time.time()
    proof_steps, final_state = bf.search(theorem['statement'], domain=theorem['domain'], verbose=True)
    elapsed = time.time() - t1

    if proof_steps:
        proof_text = ProofState._render_proof(proof_steps)
        full_code = wrap_theorem_with_proof(theorem['statement'], proof_text)
        print(f"\nVerifying final proof...")
        check_results = checker.check_batch([full_code])
        ok = check_results[0].success
        steps_str = [s.to_lean() for s in proof_steps[:10]]
        print(f"  Result: {'✓ VERIFIED' if ok else '✗ Lean rejected'}")
        print(f"  Steps: {steps_str}")
        print(f"  Search time: {elapsed:.1f}s")
        if not ok:
            print(f"  Error: {check_results[0].errors[0][:200] if check_results[0].errors else '?'}")
    else:
        print(f"\n  Result: ✗ No proof found")
        print(f"  Search time: {elapsed:.1f}s")

    print(f"\nTotal time: {time.time()-t0:.1f}s")

if __name__ == '__main__':
    main()
