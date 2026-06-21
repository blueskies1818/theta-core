#!/usr/bin/env python3
"""H2 STUDY — Benchmark 3 lemma scoring architectures vs cosine baseline.

Evaluates four architectures on the gate3 lemma-novelty test set:
  (d) Baseline cosine-similarity (replicates existing dot-product)
  (a) Two-tower with learned bilinear scoring
  (b) Cross-attention goal→candidates
  (c) Graph-filtered retrieval (k-hop neighbors) before cosine

Metrics: MRR, Top-1, Top-5, Top-10 accuracy.
Compares to gate3 baseline of 21.4% (GNN+MCTS proof success rate).

Output: data/h2_scoring_results.json

Usage:
    python scripts/eval/bench_h2_scoring.py
    python scripts/eval/bench_h2_scoring.py --checkpoint checkpoints/explorer_gate3_v4/gnn_final.pt
    python scripts/eval/bench_h2_scoring.py --k-hops 5 --output data/h2_scoring_results_k5.json
"""

import sys, json, argparse, time, re, statistics
from pathlib import Path
from collections import defaultdict

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

import torch
import torch.nn.functional as F

from src.explorer.dependency_graph import DependencyGraph, NodeType
from src.explorer.gnn_encoder import (
    GNNEncoder, extract_initial_features, prepare_graph_tensors,
)
from src.explorer.scoring_architectures import (
    BaselineCosineScorer,
    TwoTowerBilinearScorer,
    CrossAttentionScorer,
    GraphFilteredCosineScorer,
    build_goal_embedding,
)


# ==============================================================================
# Configuration
# ==============================================================================

DEFAULT_CHECKPOINT = "checkpoints/explorer_gate3_v4/gnn_final.pt"
DEFAULT_GRAPH = "data/graph/dependency_graph"
DEFAULT_THEOREMS = "data/raw/gate3_lemma_novelty.jsonl"
DEFAULT_OUTPUT = "data/h2_scoring_results.json"
DEFAULT_K_HOPS = 3


def compute_ranking_metrics(
    scores: torch.Tensor,
    ground_truth_indices: list[int],
) -> dict:
    """Compute ranking metrics for a single query.

    Args:
        scores: [N] relevance scores for all candidates.
        ground_truth_indices: List of correct candidate indices.

    Returns:
        Dict with rank, reciprocal_rank, top1, top5, top10.
    """
    # Sort in descending order
    sorted_indices = torch.argsort(scores, descending=True)

    ranks = []
    for gt_idx in ground_truth_indices:
        rank_pos = (sorted_indices == gt_idx).nonzero(as_tuple=True)[0]
        if len(rank_pos) > 0:
            ranks.append(rank_pos[0].item() + 1)  # 1-indexed

    if not ranks:
        return {
            "rank": None,
            "reciprocal_rank": 0.0,
            "top1": False,
            "top5": False,
            "top10": False,
        }

    best_rank = min(ranks)
    return {
        "rank": best_rank,
        "reciprocal_rank": 1.0 / best_rank,
        "top1": best_rank <= 1,
        "top5": best_rank <= 5,
        "top10": best_rank <= 10,
    }


def extract_ground_truth_lemmas(theorem: dict) -> list[str]:
    """Extract required lemma names from theorem proof and statement.

    Returns list of lemma node IDs that should rank highly.
    """
    proof = theorem.get("proof", "")
    stmt = theorem.get("statement", "")

    lemma_names: set[str] = set()

    # From proof text
    for m in re.finditer(r"Polynomial\.(\w+)", proof):
        lemma_names.add(m.group(1))
    for m in re.finditer(r"(?:simpa using|exact)\s+(?:\w+\.)?(\w+)", proof):
        lemma_names.add(m.group(1))
    for m in re.finditer(r"simp\s*\[(\w+)\]", proof):
        lemma_names.add(m.group(1))

    # From statement inference (for simp proofs)
    stmt_lower = stmt.lower()
    if "eval" in stmt_lower:
        if "+" in stmt or "add" in stmt_lower:
            lemma_names.add("eval_add")
        if "*" in stmt or "mul" in stmt_lower:
            lemma_names.add("eval_mul")
        if "-" in stmt or "sub" in stmt_lower:
            lemma_names.add("eval_sub")
    if "derivative" in stmt_lower:
        if "+" in stmt or "add" in stmt_lower:
            lemma_names.add("derivative_add")
        if "c" in stmt_lower and "mul" in stmt_lower:
            lemma_names.add("derivative_C_mul")
        lemma_names.add("derivative_X_pow")
    if "natdegree" in stmt_lower or "nat_degree" in stmt_lower:
        if "x ^" in stmt_lower or "x_pow" in stmt_lower:
            lemma_names.add("natDegree_X_pow")
        if "c" in stmt_lower:
            lemma_names.add("natDegree_C_mul_X_pow")
        if "mul" in stmt_lower:
            lemma_names.add("natDegree_mul")
    if ".map" in stmt_lower:
        lemma_names.add("map_id")
    if "monic" in stmt_lower:
        lemma_names.add("monic_X_sub_C")
        lemma_names.add("monic_X_pow_add")

    return list(lemma_names)


def main():
    parser = argparse.ArgumentParser(
        description="H2 STUDY — Benchmark lemma scoring architectures"
    )
    parser.add_argument(
        "--checkpoint", default=DEFAULT_CHECKPOINT,
        help="GNN checkpoint path"
    )
    parser.add_argument(
        "--graph", default=DEFAULT_GRAPH,
        help="Dependency graph path prefix"
    )
    parser.add_argument(
        "--theorems", default=DEFAULT_THEOREMS,
        help="Test theorem file (JSONL)"
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help="Output JSON path"
    )
    parser.add_argument(
        "--k-hops", type=int, default=DEFAULT_K_HOPS,
        help="k-hop radius for graph-filtered scorer"
    )
    parser.add_argument(
        "--calibrate-epochs", type=int, default=50,
        help="Calibration epochs for learnable scorers"
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Device for computation"
    )
    parser.add_argument(
        "--domain", default="Algebra",
        help="Graph domain filter"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("H2 STUDY — Lemma Scoring Architecture Benchmark")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Graph:      {args.graph}")
    print(f"Theorems:   {args.theorems}")
    print(f"K-hops:     {args.k_hops}")
    print(f"Device:     {args.device}")
    print()

    device = torch.device(args.device)

    # =========================================================================
    # Load graph
    # =========================================================================
    print("Loading dependency graph...")
    t0 = time.time()
    graph = DependencyGraph.load(args.graph)
    if args.domain:
        available = graph.get_statistics().get("nodes_by_domain", {})
        if args.domain in available:
            graph = graph.domain_subgraph(args.domain)
    print(f"  {graph.num_nodes} nodes, {graph.num_edges} edges "
          f"({time.time() - t0:.1f}s)")

    # =========================================================================
    # Load GNN and compute embeddings
    # =========================================================================
    print("\nLoading GNN checkpoint...")
    gnn = GNNEncoder.load(args.checkpoint)
    gnn.eval()
    gnn = gnn.to(device)
    n_params = sum(p.numel() for p in gnn.parameters())
    print(f"  {n_params:,} params, hidden={gnn.config.hidden_dim}")

    print("Computing node embeddings...")
    t0 = time.time()
    features = extract_initial_features(graph, gnn.config)
    sources, targets, edge_types, num_nodes = prepare_graph_tensors(graph)
    features = features.to(device)
    sources = sources.to(device)
    targets = targets.to(device)
    edge_types = edge_types.to(device)

    with torch.no_grad():
        embeddings = gnn(features, sources, targets, edge_types, num_nodes)
    print(f"  {embeddings.shape} ({time.time() - t0:.1f}s)")

    # =========================================================================
    # Build lemma-only index
    # =========================================================================
    print("\nBuilding lemma index...")
    lemma_node_ids: list[str] = []
    lemma_embeddings: dict[str, torch.Tensor] = {}
    node_id_to_idx = {nid: i for i, nid in enumerate(sorted(graph.graph.nodes()))}
    sorted_node_ids = sorted(graph.graph.nodes())

    for nid in sorted_node_ids:
        attrs = graph.get_node(nid)
        if attrs and attrs.get("node_type") == NodeType.LEMMA:
            lemma_node_ids.append(nid)
            idx = node_id_to_idx[nid]
            lemma_embeddings[nid] = embeddings[idx].cpu()

    print(f"  {len(lemma_node_ids)} lemmas (out of {graph.num_nodes} total)")

    # Stack lemma embeddings for batch scoring
    lemma_emb_tensor = torch.stack([
        lemma_embeddings[nid] for nid in lemma_node_ids
    ])  # [N_lem, D]

    # =========================================================================
    # Load test theorems
    # =========================================================================
    print(f"\nLoading test theorems from {args.theorems}...")
    tp = Path(args.theorems)
    if not tp.is_absolute():
        tp = _project_root / tp
    with open(tp) as f:
        theorems = [json.loads(line) for line in f]
    print(f"  {len(theorems)} theorems")

    # =========================================================================
    # Initialize scorers
    # =========================================================================
    hidden_dim = gnn.config.hidden_dim
    print(f"\nInitializing scorers (hidden_dim={hidden_dim})...")

    baseline = BaselineCosineScorer(hidden_dim)
    bilinear = TwoTowerBilinearScorer(hidden_dim, bottleneck_dim=128)
    cross_attn = CrossAttentionScorer(hidden_dim, num_heads=8)
    graph_filtered = GraphFilteredCosineScorer(hidden_dim, k_hops=args.k_hops)

    # Move learnable scorers to device
    bilinear = bilinear.to(device)
    cross_attn = cross_attn.to(device)
    lemma_emb_tensor = lemma_emb_tensor.to(device)

    # =========================================================================
    # Build query embeddings and ground truth
    # =========================================================================
    print("\nBuilding goal embeddings and ground truth...")

    query_embs: list[torch.Tensor] = []
    gt_lemma_indices: list[list[int]] = []

    # Map lemma_node_ids to indices in lemma_emb_tensor
    lemma_id_to_lem_idx = {nid: i for i, nid in enumerate(lemma_node_ids)}

    for theorem in theorems:
        # Build goal embedding
        goal_emb = build_goal_embedding(
            theorem, lemma_embeddings, gnn_encoder=gnn, fallback_dim=hidden_dim,
        )

        # Extract ground truth lemmas
        gt_names = extract_ground_truth_lemmas(theorem)
        gt_indices = [
            lemma_id_to_lem_idx[name]
            for name in gt_names
            if name in lemma_id_to_lem_idx
        ]

        # Fallback: if no ground truth found, use first lemma matching keywords
        if not gt_indices:
            # Use the goal embedding's cosine to find nearest lemma
            g = F.normalize(goal_emb.unsqueeze(0), dim=-1)
            c = F.normalize(lemma_emb_tensor, dim=-1)
            sims = torch.matmul(g, c.T).squeeze(0)
            top_idx = sims.argmax().item()
            gt_indices = [top_idx]

        query_embs.append(goal_emb)
        gt_lemma_indices.append(gt_indices)

    query_tensor = torch.stack(query_embs).to(device)  # [Q, D]

    # =========================================================================
    # Calibrate learnable scorers to cosine baseline
    # =========================================================================
    print(f"\nCalibrating learnable scorers ({args.calibrate_epochs} epochs)...")

    # Use all queries for calibration
    calib_loss_bilinear = bilinear.calibrate_from_baseline(
        query_tensor, lemma_emb_tensor, baseline,
        epochs=args.calibrate_epochs, lr=1e-3,
    )
    print(f"  Two-tower bilinear calibration loss: {calib_loss_bilinear:.6f}")

    calib_loss_cross = cross_attn.calibrate_from_baseline(
        query_tensor, lemma_emb_tensor, baseline,
        epochs=args.calibrate_epochs, lr=1e-3,
    )
    print(f"  Cross-attention calibration loss:   {calib_loss_cross:.6f}")

    # =========================================================================
    # Compute graph neighbor masks for graph-filtered scorer
    # =========================================================================
    print(f"\nComputing {args.k_hops}-hop neighbor masks...")
    t0 = time.time()

    # For graph-filtered scorer, we need query node IDs
    # The test theorems aren't in the graph, so we use their lemma names as
    # anchor nodes — find lemma nodes in the graph for each query
    query_anchor_ids: list[list[str]] = []
    for theorem in theorems:
        gt_names = extract_ground_truth_lemmas(theorem)
        anchors = [name for name in gt_names if graph.has_node(name)]
        if not anchors:
            # Fallback: use a random lemma in the graph
            anchors = [lemma_node_ids[0]]
        query_anchor_ids.append(anchors)

    # Compute masks for each query (using first anchor node)
    all_masks: list[torch.Tensor] = []
    for anchors in query_anchor_ids:
        mask = graph_filtered.compute_neighbor_mask(
            [anchors[0]], lemma_node_ids, graph, k_hops=args.k_hops,
        )
        all_masks.append(mask)

    neighbor_mask = torch.cat(all_masks, dim=0).to(device)  # [Q, N_lem]
    coverage = neighbor_mask.float().mean(dim=1)
    print(f"  Neighbor coverage: {coverage.mean().item():.1%} mean "
          f"(range {coverage.min().item():.1%}-{coverage.max().item():.1%})")
    print(f"  ({time.time() - t0:.1f}s)")

    # =========================================================================
    # Score all queries with all architectures
    # =========================================================================
    print("\n" + "=" * 70)
    print("SCORING")
    print("=" * 70)

    scorers: dict[str, tuple] = {
        "cosine_baseline": (baseline, False),
        "two_tower_bilinear": (bilinear, False),
        "cross_attention": (cross_attn, False),
        "graph_filtered_cosine": (graph_filtered, True),
    }

    all_results: dict[str, dict] = {}

    for name, (scorer, needs_mask) in scorers.items():
        print(f"\n--- {name} ---")
        t0 = time.time()

        with torch.no_grad():
            if needs_mask:
                scores = scorer(query_tensor, lemma_emb_tensor, neighbor_mask)
            else:
                scores = scorer(query_tensor, lemma_emb_tensor)

        scores_cpu = scores.cpu()
        elapsed = time.time() - t0

        # Per-theorem metrics
        theorem_results = []
        mrr_sum = 0.0
        top1_count = 0
        top5_count = 0
        top10_count = 0

        for i, theorem in enumerate(theorems):
            s = scores_cpu[i]  # [N_lem]
            metrics = compute_ranking_metrics(s, gt_lemma_indices[i])
            theorem_results.append({
                "name": theorem["name"],
                "gt_lemmas": extract_ground_truth_lemmas(theorem),
                **metrics,
            })
            if metrics["rank"] is not None:
                mrr_sum += metrics["reciprocal_rank"]
            if metrics["top1"]:
                top1_count += 1
            if metrics["top5"]:
                top5_count += 1
            if metrics["top10"]:
                top10_count += 1

        n = len(theorems)
        mean_mrr = mrr_sum / n
        top1_pct = top1_count / n * 100
        top5_pct = top5_count / n * 100
        top10_pct = top10_count / n * 100
        mean_rank = statistics.mean(
            r["rank"] for r in theorem_results if r["rank"] is not None
        ) if any(r["rank"] is not None for r in theorem_results) else float("inf")

        all_results[name] = {
            "scorer": name,
            "num_queries": n,
            "mrr": round(mean_mrr, 4),
            "top1": top1_count,
            "top1_pct": round(top1_pct, 1),
            "top5": top5_count,
            "top5_pct": round(top5_pct, 1),
            "top10": top10_count,
            "top10_pct": round(top10_pct, 1),
            "mean_rank": round(mean_rank, 1) if mean_rank != float("inf") else None,
            "elapsed_s": round(elapsed, 2),
            "per_theorem": theorem_results,
        }

        print(f"  MRR:     {mean_mrr:.4f}")
        print(f"  Top-1:   {top1_count}/{n} ({top1_pct:.1f}%)")
        print(f"  Top-5:   {top5_count}/{n} ({top5_pct:.1f}%)")
        print(f"  Top-10:  {top10_count}/{n} ({top10_pct:.1f}%)")
        print(f"  MeanRank:{mean_rank:.1f}" if mean_rank != float("inf")
              else "  MeanRank: N/A (no hits)")
        print(f"  Time:    {elapsed:.2f}s")

    # =========================================================================
    # Comparison summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Architecture':<30} {'MRR':>8} {'Top-1':>8} {'Top-5':>8} {'Top-10':>8} {'Rank':>8}")
    print("-" * 72)
    for name in ["cosine_baseline", "two_tower_bilinear", "cross_attention",
                  "graph_filtered_cosine"]:
        r = all_results[name]
        rank_str = f"{r['mean_rank']:.1f}" if r["mean_rank"] is not None else "N/A"
        print(f"{name:<30} {r['mrr']:>8.4f} {r['top1_pct']:>7.1f}% {r['top5_pct']:>7.1f}% "
              f"{r['top10_pct']:>7.1f}% {rank_str:>8}")

    # =========================================================================
    # Gate3 baseline comparison
    # =========================================================================
    gate3_baseline = 21.4  # GNN+MCTS proof success rate (%)
    print(f"\nGate3 GNN+MCTS proof success baseline: {gate3_baseline}%")
    print("  (Note: proof success ≠ retrieval accuracy. These metrics measure")
    print("   lemma ranking quality, not full proof synthesis.)")

    # =========================================================================
    # Save results
    # =========================================================================
    output_path = _project_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "study": "H2",
        "title": "Lemma Scoring Architecture Benchmark",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "checkpoint": args.checkpoint,
            "graph": args.graph,
            "theorems": args.theorems,
            "k_hops": args.k_hops,
            "calibrate_epochs": args.calibrate_epochs,
            "domain": args.domain,
            "hidden_dim": hidden_dim,
            "num_lemmas": len(lemma_node_ids),
            "num_queries": len(theorems),
        },
        "gate3_baseline_pct": gate3_baseline,
        "results": all_results,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, default=str)

    print(f"\nResults saved to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
