#!/usr/bin/env python3
"""Run inference with a trained GNN+MCTS explorer on held-out physics theorems.

This is the moment-of-truth test:
  - Train on pre-1905 physics theorems (classical through pre-relativity)
  - Run inference on post-1905 theorems (old quantum through modern)
  - Measure: can the explorer find valid proofs for physics it wasn't trained on?

Usage:
    # Inference on all post-era theorems
    python scripts/infer_explorer.py --checkpoint checkpoints/explorer_run4/gnn_final.pt

    # Inference on specific theorems
    python scripts/infer_explorer.py --checkpoint checkpoints/explorer_run4/gnn_final.pt --max-theorems 10

    # With era discovery monitoring
    python scripts/infer_explorer.py --checkpoint checkpoints/explorer_run4/gnn_final.pt --era pre_relativity
"""

import sys, json, argparse, time
from pathlib import Path
from collections import Counter

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

import torch
from src.explorer.dependency_graph import DependencyGraph
from src.explorer.gnn_encoder import GNNEncoder, prepare_graph_tensors, extract_initial_features
from src.explorer.mcts import MCTS, MCTSConfig
from src.explorer.proof_state import ProofState
from src.proof_checker.batch_checker import BatchChecker
from src.proof_checker.formats import wrap_theorem_with_proof
from src.correspondence.era_tracker import create_era_tracker


def main():
    parser = argparse.ArgumentParser(description="Run inference with trained GNN+MCTS explorer")
    parser.add_argument("--checkpoint", required=True, help="Path to trained GNN checkpoint")
    parser.add_argument("--graph", default="data/graph/dependency_graph", help="Graph path prefix")
    parser.add_argument("--domain", default="Algebra", help="Graph domain filter")
    parser.add_argument("--theorems", default="data/raw/physics_theorems_full.jsonl", help="Theorem file")
    parser.add_argument("--max-theorems", type=int, default=None, help="Max theorems to test")
    parser.add_argument("--mcts-sims", type=int, default=400, help="MCTS simulations per proof")
    parser.add_argument("--era", default="pre_relativity", help="Era cutoff for discovery tracking")
    parser.add_argument("--device", default="cpu", help="Device for GNN inference")
    parser.add_argument("--heuristic-scale", type=float, default=1.0,
                        help="Heuristic scale (0.0 = pure GNN, 1.0 = full heuristics)")
    parser.add_argument("--verbose", action="store_true", help="Show detailed proof output")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    # ---- Load graph ----
    graph_path = _project_root / args.graph
    if not graph_path.with_suffix(".nx.pkl").exists():
        print(f"Error: graph not found at {graph_path}.nx.pkl")
        sys.exit(1)

    graph = DependencyGraph.load(graph_path)
    if args.domain:
        available = graph.get_statistics().get("nodes_by_domain", {})
        if args.domain in available:
            graph = graph.domain_subgraph(args.domain)
            print(f"Graph: {graph.num_nodes} nodes ({args.domain} subgraph)")
        else:
            print(f"Warning: domain '{args.domain}' not found, using full graph")

    # ---- Load GNN ----
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = _project_root / ckpt_path
    if not ckpt_path.exists():
        print(f"Error: checkpoint not found: {ckpt_path}")
        sys.exit(1)

    gnn = GNNEncoder.load(str(ckpt_path))
    gnn.eval()
    gnn = gnn.to(device)
    n_params = sum(param.numel() for param in gnn.parameters())
    print(f"GNN: {n_params:,} params, hidden={gnn.config.hidden_dim}")

    # ---- Compute embeddings ----
    features = extract_initial_features(graph, gnn.config)
    sources, targets, edge_types, num_nodes = prepare_graph_tensors(graph)
    features = features.to(device)
    sources = sources.to(device)
    targets = targets.to(device)
    edge_types = edge_types.to(device)

    with torch.no_grad():
        embeddings = gnn(features, sources, targets, edge_types, num_nodes)

    # ---- Load theorems ----
    theorems_path = _project_root / args.theorems
    if not theorems_path.exists():
        print(f"Error: theorem file not found: {theorems_path}")
        sys.exit(1)

    with open(theorems_path) as f:
        all_theorems = [json.loads(line) for line in f]

    # Filter to post-era theorems (those after the cutoff)
    era_tracker = create_era_tracker(args.era) if args.era else None
    cutoff_year = era_tracker.cutoff_year if era_tracker else 1904

    # Define era order for filtering
    era_order = ["classical", "classical_crisis", "pre_relativity",
                 "pre_gr", "old_quantum", "pre_qed", "pre_sm",
                 "sm_construction", "sm_confirmed", "precision_era", "modern"]
    cutoff_idx = era_order.index(args.era) if args.era in era_order else 2  # pre_relativity

    post_theorems = []
    pre_theorems = []
    for t in all_theorems:
        t_era = t.get("era", "modern")
        t_idx = era_order.index(t_era) if t_era in era_order else len(era_order)
        if t_idx > cutoff_idx:
            post_theorems.append(t)
        else:
            pre_theorems.append(t)

    print(f"\nTheorem split:")
    print(f"  Pre-era (≤{cutoff_year}, training): {len(pre_theorems)} theorems")
    print(f"  Post-era (>{cutoff_year}, inference): {len(post_theorems)} theorems")

    if args.max_theorems:
        post_theorems = post_theorems[:args.max_theorems]

    # ---- Proof checker ----
    checker = BatchChecker(timeout=30, max_workers=4, cache_size=128)

    # ---- MCTS ----
    mcts_config = MCTSConfig(
        num_simulations=args.mcts_sims,
        max_depth=10,
        top_k_lemmas=30,
        c_puct=1.4,
        heuristic_scale=args.heuristic_scale,
        use_proof_checker=True,
        verify_timeout=5.0,
    )
    mcts = MCTS(gnn_encoder=gnn, dependency_graph=graph, config=mcts_config,
                proof_checker=checker)
    mcts.set_embeddings(embeddings, sorted(graph.node_ids))
    print(f"\nRunning inference on {len(post_theorems)} post-era theorems...")
    print(f"MCTS simulations: {args.mcts_sims}")
    print()

    # ---- Run inference ----
    results = []
    t_start = time.time()

    for i, t in enumerate(post_theorems):
        stmt = t['statement']
        name = t['name']
        zone = t.get('frontier_zone', 'unknown')
        era = t.get('era', 'unknown')
        ground_truth = t.get('proof', '?')

        t0 = time.time()
        best_steps, root = mcts.search(stmt, verbose=False)
        search_time = time.time() - t0

        proof_text = ProofState._render_proof(best_steps)

        # Apply truncation heuristic (same as training)
        full_code = wrap_theorem_with_proof(stmt, proof_text or 'sorry')
        if len(best_steps) > 1:
            first_action = best_steps[0]
            if (first_action.tactic_type.value in ("rewrite", "apply")
                    and first_action.lemma in ("add_comm", "mul_comm", "rfl", "Eq.refl")):
                single_text = ProofState._render_proof(best_steps[:1])
                full_code = wrap_theorem_with_proof(stmt, single_text or 'sorry')

        check_results = checker.check_batch([full_code])
        ok = check_results[0].success
        err = check_results[0].errors[0][:120] if check_results[0].errors else ""
        mcts_steps = [s.to_lean() for s in best_steps[:5]]  # First 5 steps

        # Era discovery monitoring
        discoveries = []
        if era_tracker:
            discoveries = era_tracker.scan_proof(proof_text)

        result = {
            "name": name, "era": era, "zone": zone,
            "success": ok, "error": err,
            "mcts_steps": mcts_steps, "num_steps": len(best_steps),
            "ground_truth": ground_truth,
            "search_time_s": search_time,
            "discoveries": discoveries,
        }
        results.append(result)

        status = "✓ PROVED" if ok else "✗ failed"
        eta_name = era.replace("_", " ")
        print(f"  [{i+1:2d}/{len(post_theorems)}] {status} | {name:40s} | {eta_name:20s} | {zone:25s} | {search_time:.1f}s")
        if ok:
            print(f"         Proof: {mcts_steps}")
            if discoveries:
                print(f"         Discoveries: {discoveries}")
        elif args.verbose:
            print(f"         Steps: {mcts_steps}")
            print(f"         Error: {err}")

    elapsed = time.time() - t_start

    # ---- Summary ----
    passed = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    print(f"\n{'='*70}")
    print(f"INFERENCE RESULTS")
    print(f"{'='*70}")
    print(f"  Total time:       {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Theorems tested:  {len(results)}")
    print(f"  Proved:           {len(passed)} ({len(passed)/max(1,len(results))*100:.0f}%)")
    print(f"  Failed:           {len(failed)}")
    print()

    # By era
    print("By era:")
    for era_name in ["old_quantum", "pre_qed", "pre_sm", "sm_construction",
                      "sm_confirmed", "precision_era", "modern"]:
        era_results = [r for r in results if r["era"] == era_name]
        if era_results:
            era_passed = [r for r in era_results if r["success"]]
            print(f"  {era_name:25s}: {len(era_passed)}/{len(era_results)} ({len(era_passed)/max(1,len(era_results))*100:.0f}%)")
            if args.verbose and era_passed:
                for r in era_passed:
                    print(f"    ✓ {r['name']}")

    # By zone
    print("\nBy frontier zone:")
    zone_counts = Counter(r["zone"] for r in results)
    zone_passed = Counter(r["zone"] for r in passed)
    for zone in sorted(zone_counts):
        p = zone_passed.get(zone, 0)
        t = zone_counts[zone]
        bar = "█" * int(p/t*20) + "░" * (20 - int(p/t*20))
        print(f"  {zone:30s}: {p}/{t} ({p/t*100:3.0f}%) {bar}")

    # Era discoveries
    if era_tracker:
        print(f"\nEra discovery monitoring (passive):")
        print(f"  Era: {era_tracker.era_name} (≤{era_tracker.cutoff_year})")
        top = era_tracker.get_top_discoveries(10)
        if top:
            for name, count, sig in top:
                print(f"  {name:40s}: {count}× [sig={sig:.1f}]")
        else:
            print(f"  No post-era physics concepts detected in proof text")

    # Prove list
    if passed:
        print(f"\nProved theorems:")
        for r in passed:
            print(f"  ✓ [{r['zone']:25s}] [{r['era']:20s}] {r['name']}")
            print(f"    Proof: {r['mcts_steps']}")
            print(f"    Ground truth: {r['ground_truth']}")

    if failed and args.verbose:
        print(f"\nFailed theorems (sample):")
        for r in failed[:10]:
            print(f"  ✗ [{r['zone']:25s}] [{r['era']:20s}] {r['name']}")
            print(f"    Steps: {r['mcts_steps'][:3]}")

    try:
        checker.shutdown()
    except:
        pass

    # Return exit code
    return 0 if len(passed) > 0 else 1


if __name__ == '__main__':
    sys.exit(main())
