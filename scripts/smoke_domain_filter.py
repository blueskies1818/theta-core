#!/usr/bin/env python3
"""Smoke test: verify domain-filtered lemma retrieval works on 3 theorems."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import time

from src.explorer.dependency_graph import DependencyGraph
from src.explorer.gnn_encoder import GNNEncoder, extract_initial_features, prepare_graph_tensors
from src.explorer.gnn_best_first_search import GNNBestFirstSearch, GNNBestFirstConfig
from src.explorer.proof_state import ProofState
from src.proof_checker.batch_checker import BatchChecker
from src.proof_checker.formats import wrap_theorem_with_proof
from scripts.eval_gnn_prover import build_lemma_index, extract_conclusion, normalize_expression

_PROJECT_ROOT = Path(__file__).parent.parent

# Test theorems from each domain
TEST_THEOREMS = [
    {
        "name": "alg_subst_expand",
        "statement": "theorem alg_subst_expand (x y : ℝ) (h : x = y + 1) : x^2 - 2*x + 1 = y^2",
        "domain": "algebra",
        "proof": "rw [h]; ring",
    },
    {
        "name": "ana_triangle_rev",
        "statement": "theorem ana_triangle_rev (x y : ℝ) : |x| - |y| ≤ |x - y|",
        "domain": "analysis",
        "proof": "have h := abs_sub_abs_le_abs_sub x y; linarith",
    },
    {
        "name": "logic_contrapositive",
        "statement": "theorem logic_contrapositive (P Q : Prop) : (P → Q) → (¬Q → ¬P)",
        "domain": "logic",
        "proof": "intro h; intro hnq; intro hp; apply hnq; apply h; exact hp",
    },
]

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
    torch.set_num_threads(4)
    
    print("Loading GNN...")
    gnn = GNNEncoder.load(str(_PROJECT_ROOT / "checkpoints/gnn/gate2_fullgraph_finetuned.pt"))
    gnn.eval()
    
    print("Loading graph...")
    graph = DependencyGraph.load(_PROJECT_ROOT / "data/graph/dependency_graph_full")
    print(f"  {graph.summary()}")
    
    print("Computing node embeddings...")
    features = extract_initial_features(graph, gnn.config)
    sources, targets, edge_types, num_nodes = prepare_graph_tensors(graph)
    with torch.no_grad():
        node_embeddings = gnn(features, sources, targets, edge_types, num_nodes)
    
    lemma_to_idx = build_lemma_index(graph)
    idx_to_norm = build_norm_index(graph, lemma_to_idx)
    
    config = GNNBestFirstConfig(
        max_depth=20, max_expansions=200, top_k_lemmas=30,
        use_proof_checker=True, num_threads=4,
    )
    
    checker = BatchChecker(timeout=15, max_workers=4, cache_size=64)
    
    bf = GNNBestFirstSearch(
        gnn=gnn, graph=graph, node_embeddings=node_embeddings,
        lemma_index=lemma_to_idx, idx_to_norm=idx_to_norm,
        config=config, proof_checker=checker,
    )
    
    print(f"\nDomain index: {len(bf._domain_node_ids)} unique graph domains")
    
    for t in TEST_THEOREMS:
        print(f"\n{'='*60}")
        print(f"Theorem: {t['name']} (domain={t['domain']})")
        
        # Show candidate counts with and without domain filtering
        for use_domain in [True, False]:
            d = t['domain'] if use_domain else None
            lemmas = bf._get_relevant_lemmas(t['statement'], domain=d)
            print(f"  Domain filter={'ON' if use_domain else 'OFF'}: {len(lemmas)} candidates")
            if len(lemmas) <= 5:
                for lem in lemmas:
                    print(f"    - {lem}")
        
        # Run search with domain filtering
        t0 = time.time()
        proof_steps, final_state = bf.search(t['statement'], domain=t['domain'], verbose=True)
        elapsed = time.time() - t0
        
        if proof_steps:
            proof_text = ProofState._render_proof(proof_steps)
            full_code = wrap_theorem_with_proof(t['statement'], proof_text)
            check_results = checker.check_batch([full_code])
            ok = check_results[0].success
            status = "✓ VERIFIED" if ok else "✗ Lean rejected"
            steps_str = [s.to_lean() for s in proof_steps[:5]]
            print(f"  Result: {status}")
            print(f"  Steps: {steps_str}")
            print(f"  Time: {elapsed:.1f}s")
        else:
            print(f"  Result: ✗ No proof found ({elapsed:.1f}s)")

if __name__ == '__main__':
    main()
