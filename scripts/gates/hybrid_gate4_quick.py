#!/usr/bin/env python3
"""Gate 4 Hybrid: Era-gated negative control with GNN+best-first.

Tests whether the hybrid architecture shows era-specific learning by
running on era-tagged (continuous vs quantized) theorem subsets.
Proper Gate 4 requires two separately-trained GNNs; this simplified
version uses the single finetuned GNN and tests for era effects.

Usage: python scripts/gates/hybrid_gate4_quick.py
"""
import sys, json, time
from pathlib import Path
from collections import Counter

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

import torch
from src.explorer.dependency_graph import DependencyGraph
from src.explorer.gnn_encoder import GNNEncoder, extract_initial_features, prepare_graph_tensors
from src.explorer.gnn_best_first_search import GNNBestFirstSearch, GNNBestFirstConfig
from src.explorer.proof_state import ProofState
from src.proof_checker.batch_checker import BatchChecker
from src.proof_checker.formats import wrap_theorem_with_proof
from scripts.eval.eval_gnn_prover import build_lemma_index, normalize_expression, extract_conclusion
from scripts.eval.eval_hybrid_retrieval import build_norm_index

def main():
    torch.set_num_threads(4)

    test_path = _project_root / "data/raw/gate4_test_mixed.jsonl"
    if not test_path.exists():
        print("Gate 4 test data not found")
        return 1

    with open(test_path) as f:
        theorems = [json.loads(line) for line in f]

    continuous = [t for t in theorems if t.get("era", "") == "continuous"]
    quantized = [t for t in theorems if t.get("era", "") == "quantized"]
    print(f"Gate 4 theorems: {len(theorems)} total "
          f"({len(continuous)} continuous, {len(quantized)} quantized)")

    if not continuous or not quantized:
        print("Missing era data — cannot evaluate")
        return 1

    ckpt = _project_root / "checkpoints/gnn/gate2_fullgraph_finetuned.pt"
    graph_path = _project_root / "data/graph/dependency_graph_full"

    print("Loading GNN...")
    gnn = GNNEncoder.load(str(ckpt))
    gnn.eval()

    print("Loading graph (algebra subgraph)...")
    graph = DependencyGraph.load(graph_path)
    graph = graph.domain_subgraph("Algebra")
    print(f"  {graph.summary()}")

    print("Computing embeddings...")
    features = extract_initial_features(graph, gnn.config)
    sources, targets, edge_types, num_nodes = prepare_graph_tensors(graph)
    with torch.no_grad():
        node_embeddings = gnn(features, sources, targets, edge_types, num_nodes)

    lemma_to_idx = build_lemma_index(graph)
    idx_to_norm = build_norm_index(graph, lemma_to_idx)

    config = GNNBestFirstConfig(
        max_depth=20, max_expansions=500, top_k_lemmas=30,
        depth_penalty=0.05, use_proof_checker=True,
        verify_timeout=5.0, num_threads=4, max_graph_candidates=200,
    )
    checker = BatchChecker(timeout=30, max_workers=1, cache_size=128)

    bf = GNNBestFirstSearch(
        gnn=gnn, graph=graph, node_embeddings=node_embeddings,
        lemma_index=lemma_to_idx, idx_to_norm=idx_to_norm,
        config=config, proof_checker=checker,
    )

    # Cap at 10 per era for speed
    max_per_era = 10
    era_results = {}

    for era_label, era_theorems in [("continuous", continuous), ("quantized", quantized)]:
        subset = era_theorems[:max_per_era]
        print(f"\n--- Evaluating {era_label} era ({len(subset)} theorems) ---")

        passed = 0
        t0 = time.time()
        for i, t in enumerate(subset):
            name = t["name"]
            stmt = t["statement"]
            proof_steps, _ = bf.search(stmt, verbose=False)
            ok = False
            if proof_steps:
                proof_text = ProofState._render_proof(proof_steps)
                code = wrap_theorem_with_proof(stmt, proof_text)
                check_results = checker.check_batch([code])
                ok = check_results[0].success
            if ok:
                passed += 1
                print(f"  ✓ {name}: {' '.join(s.to_lean() for s in proof_steps[:3])}")
            else:
                print(f"  ✗ {name}")

        elapsed = time.time() - t0
        era_results[era_label] = {
            "tested": len(subset), "passed": passed,
            "rate": passed / len(subset), "time_s": elapsed,
        }
        print(f"  {era_label}: {passed}/{len(subset)} ({passed/len(subset):.0%}) in {elapsed:.0f}s")

    c_rate = era_results["continuous"]["rate"]
    q_rate = era_results["quantized"]["rate"]
    interaction = abs(c_rate - q_rate)
    has_effect = interaction > 0.05

    print(f"\n{'='*60}")
    print(f"Gate 4 Results:")
    print(f"  Continuous era: {c_rate:.0%} ({era_results['continuous']['passed']}/{era_results['continuous']['tested']})")
    print(f"  Quantized era:  {q_rate:.0%} ({era_results['quantized']['passed']}/{era_results['quantized']['tested']})")
    print(f"  Interaction:    {interaction:.1%}")
    print(f"  Verdict:        {'PASS (era effect detected)' if has_effect else 'FAIL (no era effect)'}")

    result = {
        "continuous": era_results["continuous"],
        "quantized": era_results["quantized"],
        "interaction": interaction,
        "has_era_effect": has_effect,
        "verdict": "PASS" if has_effect else "FAIL",
        "note": "Simplified Gate 4: single GNN, not two era-separated models.",
    }

    output_path = _project_root / "data/hybrid_gate4_quick.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {output_path}")

if __name__ == "__main__":
    main()
