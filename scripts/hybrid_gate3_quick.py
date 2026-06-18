#!/usr/bin/env python3
"""Quick hybrid gate3 evaluation on a small algebra subset for immediate results."""
import sys, json, time
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

import torch
import torch.nn.functional as F

from src.explorer.dependency_graph import DependencyGraph
from src.explorer.gnn_encoder import GNNEncoder, extract_initial_features, prepare_graph_tensors
from src.explorer.gnn_best_first_search import GNNBestFirstSearch, GNNBestFirstConfig
from src.explorer.proof_state import ProofState
from src.proof_checker.batch_checker import BatchChecker
from src.proof_checker.formats import wrap_theorem_with_proof
from scripts.eval_gnn_prover import build_lemma_index, extract_conclusion, normalize_expression
from scripts.eval_hybrid_retrieval import build_norm_index, classify_proof_pattern

def main():
    torch.set_num_threads(4)

    # Known-working algebra theorems
    theorems = [
        {"name": "alg_subst_expand", "statement": "example (a b : ℕ) (h : a = b) : a + 0 = b + 0 := by", "domain": "algebra", "proof": "rw [h]; ring"},
        {"name": "alg_subst_factor", "statement": "example (a b : ℕ) (h : a = b) : a * 1 = b * 1 := by", "domain": "algebra", "proof": "rw [h]; ring"},
        {"name": "alg_cross_multiply", "statement": "example (a b c d : ℝ) (hb : b ≠ 0) (hd : d ≠ 0) (h : a / b = c / d) : a * d = b * c := by", "domain": "algebra", "proof": "field_simp [hb, hd] at h; exact h"},
        {"name": "alg_add_comm", "statement": "example (a b : ℕ) : a + b = b + a := by", "domain": "algebra", "proof": "apply add_comm"},
        {"name": "alg_mul_assoc", "statement": "example (a b c : ℕ) : (a * b) * c = a * (b * c) := by", "domain": "algebra", "proof": "apply mul_assoc"},
    ]

    ckpt = _project_root / "checkpoints/gnn/gate2_fullgraph_finetuned.pt"
    graph_path = _project_root / "data/graph/dependency_graph_full"

    print("Loading GNN...")
    gnn = GNNEncoder.load(str(ckpt))
    gnn.eval()
    print(f"  {sum(p.numel() for p in gnn.parameters()):,} params")

    print("Loading graph (algebra subgraph)...")
    graph = DependencyGraph.load(graph_path)
    graph = graph.domain_subgraph("Algebra")
    print(f"  {graph.summary()}")

    print("Computing embeddings...")
    features = extract_initial_features(graph, gnn.config)
    sources, targets, edge_types, num_nodes = prepare_graph_tensors(graph)
    with torch.no_grad():
        node_embeddings = gnn(features, sources, targets, edge_types, num_nodes)
    print(f"  {node_embeddings.shape}")

    lemma_to_idx = build_lemma_index(graph)
    idx_to_norm = build_norm_index(graph, lemma_to_idx)
    print(f"  {len(lemma_to_idx)} lemmas indexed")

    config = GNNBestFirstConfig(
        max_depth=20, max_expansions=2000, top_k_lemmas=30,
        depth_penalty=0.05, use_proof_checker=True,
        verify_timeout=5.0, num_threads=4, max_graph_candidates=200,
    )
    checker = BatchChecker(timeout=30, max_workers=1, cache_size=128)

    bf = GNNBestFirstSearch(
        gnn=gnn, graph=graph, node_embeddings=node_embeddings,
        lemma_index=lemma_to_idx, idx_to_norm=idx_to_norm,
        config=config, proof_checker=checker,
    )

    results = []
    for i, t in enumerate(theorems):
        name = t["name"]
        stmt = t["statement"]
        gt = t["proof"]
        print(f"\n[{i+1}/{len(theorems)}] {name}")
        print(f"  Statement: {stmt}")
        print(f"  Ground truth: {gt}")

        t0 = time.time()
        proof_steps, final_state = bf.search(stmt, verbose=False)
        elapsed = time.time() - t0

        if proof_steps:
            proof_text = ProofState._render_proof(proof_steps)
            code = wrap_theorem_with_proof(stmt, proof_text)
            check_results = checker.check_batch([code])
            ok = check_results[0].success
            err = check_results[0].errors[0][:120] if check_results[0].errors else ""
        else:
            ok = False
            err = "no proof found"

        steps_str = [s.to_lean() for s in proof_steps[:10]]
        pattern = classify_proof_pattern(steps_str) if ok else "FAILED"
        match = " ".join(steps_str) == gt

        print(f"  Result: {'✓ PASS' if ok else '✗ FAIL'} [{pattern}] in {elapsed:.1f}s")
        print(f"  Proof: {steps_str}")
        if ok:
            print(f"  Match ground truth: {match}")
        else:
            print(f"  Error: {err[:120]}")

        results.append({
            "name": name, "success": ok, "proof": steps_str,
            "pattern": pattern, "match_gt": match,
            "time_s": round(elapsed, 1), "error": err,
        })

    passed = sum(1 for r in results if r["success"])
    matched = sum(1 for r in results if r.get("match_gt"))
    multi = sum(1 for r in results if r.get("pattern") == "multi")

    print(f"\n{'='*60}")
    print(f"RESULTS: {passed}/{len(results)} passed, {matched} exact match, {multi} multi-step")
    for r in results:
        s = "✓" if r["success"] else "✗"
        print(f"  {s} {r['name']:<30s} [{r['pattern']:<10s}] {r['time_s']:>6.1f}s  → {r['proof']}")

    output_path = _project_root / "data/hybrid_gate3_quick.json"
    with open(output_path, "w") as f:
        json.dump({"results": results, "passed": passed, "total": len(results)}, f, indent=2)
    print(f"\nSaved: {output_path}")

if __name__ == "__main__":
    main()
