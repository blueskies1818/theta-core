#!/usr/bin/env python3
"""Gate 3 v2 evaluation with enriched lemma index.

Uses the original 116K graph + existing pretrained GNN model, but with
the enriched lemma_index (96% recall) instead of the default (40%).

This isolates the effect of name resolution on proof search.
Output: data/gnn_enriched_baseline_gate3.json
"""

import json, sys, time, re
from pathlib import Path
from collections import Counter

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import torch
import torch.nn.functional as F

from src.explorer.dependency_graph import DependencyGraph
from src.explorer.gnn_encoder import GNNEncoder, extract_initial_features, prepare_graph_tensors
from src.explorer.gnn_best_first_search import GNNBestFirstSearch, GNNBestFirstConfig
from src.explorer.proof_state import ProofState, Tactic
from src.proof_checker.batch_checker import BatchChecker
from src.proof_checker.formats import wrap_theorem_with_proof
from scripts.eval.eval_gnn_prover import (
    build_lemma_index, extract_conclusion, normalize_expression, build_lemma_norm_index
)


def main():
    print("=" * 70)
    print("GATE 3: Enriched Lemma Index on gate3_v2")
    print("=" * 70)

    # Load original graph
    print("\nLoading graph...")
    graph = DependencyGraph.load(_PROJECT_ROOT / "data/graph/dependency_graph_full")
    print(f"  {graph.summary()}")

    # Load enriched lemma index
    print("Loading enriched lemma index...")
    with open(_PROJECT_ROOT / "data/graph/dependency_graph_full_v2.lemma_index.json") as f:
        enriched_idx = json.load(f)

    # Filter to only include indices valid for the ORIGINAL graph
    num_nodes = graph.num_nodes
    lemma_to_idx = {k: v for k, v in enriched_idx.items()
                    if isinstance(v, int) and 0 <= v < num_nodes}
    print(f"  Enriched lemma index: {len(lemma_to_idx)} entries (filtered to {num_nodes} nodes)")

    # Also build the default index for comparison
    default_idx = build_lemma_index(graph)
    print(f"  Default lemma index: {len(default_idx)} entries")

    # Build norm index
    idx_to_norm = build_lemma_norm_index(graph, lemma_to_idx)
    print(f"  Norm index: {len(idx_to_norm)} entries")

    # Load model
    print("Loading pretrained model...")
    model_path = _PROJECT_ROOT / "checkpoints/gnn/full_graph_pretrained.pt"
    gnn = GNNEncoder.load(model_path)
    gnn.eval()
    print(f"  Model: {sum(p.numel() for p in gnn.parameters()):,} params")

    # Compute embeddings
    print("Computing node embeddings...")
    features = extract_initial_features(graph, gnn.config)
    sources, targets, edge_types, n = prepare_graph_tensors(graph)
    print(f"  Graph: {n} nodes, {sources.size(0)} edges")

    with torch.no_grad():
        node_embeddings = gnn(features, sources, targets, edge_types, n)
    print(f"  Embeddings: {node_embeddings.shape}")

    # Setup proof checker
    try:
        checker = BatchChecker(timeout=30, max_workers=4)
        use_checker = True
        print("  Proof checker: ready")
    except Exception as e:
        checker = None
        use_checker = False
        print(f"  Proof checker: unavailable ({e})")

    # Load theorems
    theorems_path = _PROJECT_ROOT / "data/raw/gate3_v2.jsonl"
    if not theorems_path.exists():
        theorems_path = _PROJECT_ROOT / "data/gate3_v2_test.jsonl"
    with open(theorems_path) as f:
        theorems = [json.loads(line) for line in f]
    print(f"\nTheorems: {len(theorems)}")

    # Config
    config = GNNBestFirstConfig(
        max_depth=20,
        max_expansions=2000,
        top_k_lemmas=30,
        depth_penalty=0.05,
        use_proof_checker=use_checker,
        verify_timeout=5.0,
        max_proof_length=500,
        device="cpu",
        num_threads=4,
    )

    # Setup search
    bf_search = GNNBestFirstSearch(
        gnn=gnn,
        graph=graph,
        node_embeddings=node_embeddings,
        lemma_index=lemma_to_idx,
        idx_to_norm=idx_to_norm,
        config=config,
        proof_checker=checker if use_checker else None,
    )

    # Run evaluation
    print(f"\n--- Running on {len(theorems)} theorems ---")
    print(f"Max expansions: {config.max_expansions}, Top-K: {config.top_k_lemmas}")
    print()

    results = []
    t_start = time.time()
    verified = 0
    found_unverified = 0

    for i, t in enumerate(theorems):
        stmt = t["statement"]
        name = t.get("name", f"thm_{i}")
        domain = t.get("domain", "unknown")

        t0 = time.time()
        proof_steps, final_state = bf_search.search(stmt, domain=None, verbose=False)
        search_time = time.time() - t0

        proof_text = ProofState._render_proof(proof_steps)

        ok = False
        err = ""
        if not proof_steps:
            err = "no proof found"
        elif checker is None:
            ok = True
        else:
            full_code = wrap_theorem_with_proof(stmt, proof_text)
            try:
                check_results = checker.check_batch([full_code])
                if check_results[0].success:
                    ok = True
                    verified += 1
                else:
                    err = check_results[0].errors[0][:120] if check_results[0].errors else "verification failed"
            except Exception as e:
                err = str(e)[:120]

        if ok:
            print(f"  [{i+1:2d}/{len(theorems)}] ✓ {name[:60]} ({domain}) [{search_time:.1f}s]")
        elif proof_steps:
            found_unverified += 1
            print(f"  [{i+1:2d}/{len(theorems)}] ✗ {name[:60]} ({domain}) — {err[:80]} [{search_time:.1f}s]")
        else:
            print(f"  [{i+1:2d}/{len(theorems)}] — {name[:60]} ({domain}) — no proof [{search_time:.1f}s]")

        results.append({
            "name": name,
            "domain": domain,
            "verified": ok,
            "proof_steps": [str(s) for s in proof_steps] if proof_steps else [],
            "proof_text": proof_text,
            "error": err,
            "search_time": search_time,
            "num_steps": len(proof_steps),
        })

    total_time = time.time() - t_start

    # Summary
    print(f"\n{'=' * 70}")
    print(f"RESULTS")
    print(f"{'=' * 70}")
    print(f"Total theorems: {len(theorems)}")
    print(f"Verified: {verified}/{len(theorems)} ({verified/len(theorems)*100:.1f}%)")
    print(f"Found (unverified): {found_unverified}")
    print(f"Total time: {total_time:.1f}s ({total_time/len(theorems):.1f}s/theorem)")
    print(f"\nBaseline: 10/64 = 15.6%")
    print(f"Change: {verified/len(theorems)*100 - 15.6:+.1f}%")

    # Save results
    output = {
        "model": "full_graph_pretrained.pt",
        "lemma_index": "enriched (v2, 96.1% recall)",
        "total_theorems": len(theorems),
        "verified": verified,
        "verified_pct": round(verified / len(theorems) * 100, 1),
        "found_unverified": found_unverified,
        "total_time_s": round(total_time, 1),
        "baseline_pct": 15.6,
        "change_pct": round(verified / len(theorems) * 100 - 15.6, 1),
        "theorems": results,
    }
    out_path = _PROJECT_ROOT / "data" / "gnn_enriched_baseline_gate3.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
