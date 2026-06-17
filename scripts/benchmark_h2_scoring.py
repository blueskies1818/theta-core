#!/usr/bin/env python3
"""
H2 Scoring Architecture Benchmark.

Benchmarks three lemma scoring architectures against the baseline
cosine-similarity approach on the Gate 3 lemma-novelty dataset.

Scoring architectures tested:
  (a) Two-Tower with learned bilinear scoring
  (b) Cross-attention goal→candidates
  (c) Graph-filtered retrieval (k-hop neighbors + cosine)

Metrics:
  - Precision@k: Fraction of top-k lemmas that prove the theorem
  - MRR: Mean Reciprocal Rank of the first proving lemma
  - Proof coverage: % of theorems where any lemma in top-k proves it
  - Compare to baseline cosine-similarity (21.4% MCTS proof success)

Usage:
    python scripts/benchmark_h2_scoring.py \
        --checkpoint checkpoints/explorer_gate3_v4/gnn_final.pt \
        --graph data/graph/dependency_graph \
        --domain Algebra \
        --theorems data/raw/gate3_lemma_novelty.jsonl \
        --output data/h2_scoring_results.json
"""

import sys, json, argparse, time, statistics
from pathlib import Path
from collections import defaultdict

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

import torch
import torch.nn.functional as F

from src.explorer.dependency_graph import DependencyGraph
from src.explorer.gnn_encoder import GNNEncoder, prepare_graph_tensors, extract_initial_features
from src.explorer.gnn_config import GNNConfig
from src.explorer.h2_scoring import (
    TwoTowerBilinear,
    CrossAttentionScorer,
    graph_filtered_retrieval,
    cosine_similarity_scoring,
)
from src.explorer.mcts import _extract_math_keywords
from src.proof_checker.batch_checker import BatchChecker
from src.proof_checker.formats import wrap_theorem_with_proof


# =============================================================================
# Goal Embedding Helper
# =============================================================================


def compute_goal_embedding(
    goal_text: str,
    gnn: GNNEncoder,
    node_embeddings: torch.Tensor,
    lemma_names: list[str],
    lemma_to_idx: dict[str, int],
    graph: DependencyGraph,
) -> torch.Tensor:
    """Compute a goal embedding from theorem text.

    Strategy (mirrors MCTS._embed_goal):
    1. Extract keywords from goal
    2. Find keyword-matching lemmas in graph
    3. Average their GNN embeddings as context
    4. Project through GoalEncoder
    """
    keywords = _extract_math_keywords(goal_text)
    device = node_embeddings.device
    node_emb_norm = F.normalize(node_embeddings, dim=-1)

    # Build keyword → lemma index map
    kw_map: dict[str, list[int]] = defaultdict(list)
    for name, idx in lemma_to_idx.items():
        short = name.split(".")[-1] if "." in name else name
        tokens = short.lower().replace("_", " ").split()
        for token in tokens:
            if len(token) >= 2:
                kw_map[token].append(idx)
        kw_map[short.lower()].append(idx)

    # Score candidate lemmas by keyword match
    candidate_scores: dict[int, float] = {}
    for kw in keywords:
        matches = kw_map.get(kw.lower(), [])
        for rank, idx in enumerate(matches):
            if idx >= node_embeddings.size(0):
                continue
            score = 1.0 / (1.0 + rank * 0.1)
            candidate_scores[idx] = candidate_scores.get(idx, 0.0) + score

    sorted_candidates = sorted(candidate_scores.items(), key=lambda x: -x[1])[:100]
    matching_indices = [idx for idx, _ in sorted_candidates]

    if matching_indices:
        indices_t = torch.tensor(matching_indices, device=device)
        context_emb = node_emb_norm[indices_t].mean(dim=0)
    else:
        return torch.zeros(node_emb_norm.size(1), device=device)

    if gnn.goal_encoder is not None:
        return gnn.encode_goal(context_emb)
    return F.normalize(context_emb, dim=-1)


# =============================================================================
# Candidate Lemma Collection
# =============================================================================


_BUILTIN_LEMMAS = {
    "eq": ["add_comm", "add_assoc", "mul_comm", "mul_assoc",
           "add_zero", "zero_add", "mul_one", "one_mul",
           "sub_self", "add_sub_cancel", "sub_add_cancel",
           "Eq.refl", "rfl", "eq_self_iff_true"],
    "ne": ["ne_of_gt", "ne_of_lt", "ne_self_iff_false"],
    "le": ["le_refl", "le_trans", "le_antisymm", "lt_of_lt_of_le"],
    "lt": ["lt_irrefl", "lt_trans", "lt_of_le_of_lt"],
    "not": ["not_false_iff", "not_true_iff", "not_not"],
    "div": ["div_self", "div_one", "one_div"],
    "ring": ["add_comm", "add_assoc", "mul_comm", "mul_assoc", "distrib"],
}


def _get_builtin_lemmas(keywords: list[str]) -> list[str]:
    """Get built-in lemma names matching keywords."""
    result = []
    for kw in keywords:
        for lemma_kw, lemmas in _BUILTIN_LEMMAS.items():
            if kw.lower() in lemma_kw.lower() or lemma_kw.lower() in kw.lower():
                result.extend(lemmas)
    # Deduplicate preserving order
    seen = set()
    deduped = []
    for l in result:
        if l not in seen:
            seen.add(l)
            deduped.append(l)
    return deduped


def get_candidate_lemmas(
    goal_text: str,
    graph: DependencyGraph,
    lemma_to_idx: dict[str, int],
    top_k: int = 50,
) -> list[str]:
    """Get candidate lemmas for a goal using keyword matching + graph centrality.

    Mirrors MCTS._get_relevant_lemmas but exposes the candidate list directly.
    """
    keywords = _extract_math_keywords(goal_text)
    builtins = _get_builtin_lemmas(keywords)

    # Filter graph lemmas by keyword match
    candidates: list[tuple[str, float]] = []
    seen: set[str] = set()

    for kw in keywords[:8]:
        for nid in graph.node_ids:
            if nid in seen:
                continue
            attrs = graph.get_node(nid)
            name = attrs.get("name", nid) if attrs else nid
            if kw.lower() in name.lower():
                kw_score = sum(
                    1.0 for k in keywords if k.lower() in name.lower()
                )
                candidates.append((nid, kw_score))
                seen.add(nid)
                if len(candidates) >= 300:
                    break
        if len(candidates) >= 300:
            break

    # Rank by centrality + keyword score
    in_degrees = dict(graph.graph.in_degree())
    max_in = max(in_degrees.values()) if in_degrees else 1

    ranked = []
    for name, kw_score in candidates:
        centrality = in_degrees.get(name, 0) / max_in
        combined = 0.3 * kw_score + 0.7 * centrality
        ranked.append((name, combined))

    ranked.sort(key=lambda x: x[1], reverse=True)
    lemmas = [name for name, _ in ranked[:top_k]]

    # Prepend builtins
    result = []
    seen_result = set()
    for bl in builtins:
        if bl not in seen_result:
            result.append(bl)
            seen_result.add(bl)
    for gl in lemmas:
        if gl not in seen_result:
            result.append(gl)
            seen_result.add(gl)

    return result[:top_k]


# =============================================================================
# Proof Validation
# =============================================================================


def check_lemma_proves_theorem(
    theorem_statement: str,
    lemma_name: str,
    checker: BatchChecker,
) -> bool:
    """Check if using a single lemma as a tactic proves the theorem.

    Tries apply and exact tactics with the lemma.
    """
    # Try 'apply'
    proof_code = wrap_theorem_with_proof(theorem_statement, f"  apply {lemma_name}")
    results = checker.check_batch([proof_code])
    if results[0].success:
        return True

    # Try 'exact'
    proof_code = wrap_theorem_with_proof(theorem_statement, f"  exact {lemma_name}")
    results = checker.check_batch([proof_code])
    if results[0].success:
        return True

    # Try 'simp [lemma]'
    proof_code = wrap_theorem_with_proof(theorem_statement, f"  simp [{lemma_name}]")
    results = checker.check_batch([proof_code])
    if results[0].success:
        return True

    return False


# =============================================================================
# Ranking Metrics
# =============================================================================


def compute_ranking_metrics(
    scores: torch.Tensor,
    proving_indices: set[int],
    ks: list[int] = [1, 3, 5, 10],
) -> dict:
    """Compute ranking metrics given scores and ground-truth proving indices.

    Args:
        scores: [C] score tensor.
        proving_indices: Set of candidate indices that prove the theorem.
        ks: k values for precision@k.

    Returns:
        Dict with metrics.
    """
    # Sort candidates by score (descending)
    ranked_indices = torch.argsort(scores, descending=True).tolist()

    metrics = {}

    # Precision@k
    for k in ks:
        top_k = set(ranked_indices[:k])
        hits = len(top_k & proving_indices)
        metrics[f"precision@{k}"] = hits / min(k, len(ranked_indices)) if ranked_indices else 0.0

    # MRR
    mrr = 0.0
    for i, idx in enumerate(ranked_indices):
        if idx in proving_indices:
            mrr = 1.0 / (i + 1)
            break
    metrics["mrr"] = mrr

    # Hit@k (any proving lemma in top-k)
    for k in ks:
        top_k = set(ranked_indices[:k])
        metrics[f"hit@{k}"] = 1.0 if (top_k & proving_indices) else 0.0

    # Top rank of first proving lemma (-1 if none)
    first_rank = -1
    for i, idx in enumerate(ranked_indices):
        if idx in proving_indices:
            first_rank = i + 1
            break
    metrics["first_proving_rank"] = first_rank

    return metrics


# =============================================================================
# Main Benchmark
# =============================================================================


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="H2 Scoring Architecture Benchmark"
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to trained GNN checkpoint (e.g., checkpoints/explorer_gate3_v4/gnn_final.pt)"
    )
    parser.add_argument(
        "--graph", default="data/graph/dependency_graph",
        help="Graph path prefix"
    )
    parser.add_argument(
        "--domain", default="Algebra",
        help="Graph domain filter"
    )
    parser.add_argument(
        "--theorems", default="data/raw/gate3_lemma_novelty.jsonl",
        help="Test theorem file"
    )
    parser.add_argument(
        "--output", default="data/h2_scoring_results.json",
        help="Output JSON path"
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Device for GNN inference"
    )
    parser.add_argument(
        "--top-k-lemmas", type=int, default=50,
        help="Number of candidate lemmas to score"
    )
    parser.add_argument(
        "--max-check-lemmas", type=int, default=30,
        help="Max lemmas to proof-check per theorem (top from each method)"
    )
    parser.add_argument(
        "--k-hops-graph", type=int, default=3,
        help="Hops for graph-filtered retrieval"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print detailed per-theorem results"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("H2 Scoring Architecture Benchmark")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Graph:      {args.graph} ({args.domain})")
    print(f"Theorems:   {args.theorems}")
    print(f"Device:     {args.device}")
    print(f"Top-K:      {args.top_k_lemmas}")
    print()

    device_t = torch.device(args.device)
    torch.set_num_threads(8)  # CPU headroom constraint

    # ---- Load Graph ----
    gp = Path(args.graph)
    if not gp.is_absolute():
        gp = _project_root / gp
    if not gp.with_suffix(".nx.pkl").exists():
        print(f"Error: graph not found at {gp}.nx.pkl")
        return 1

    graph = DependencyGraph.load(gp)
    print(f"Full graph: {graph.num_nodes} nodes, {graph.num_edges} edges")

    if args.domain:
        available = graph.get_statistics().get("nodes_by_domain", {})
        if args.domain in available:
            graph = graph.domain_subgraph(args.domain)
            print(f"Filtered:   {graph.num_nodes} nodes ({args.domain})")

    # ---- Load GNN ----
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = _project_root / ckpt_path
    if not ckpt_path.exists():
        print(f"Error: checkpoint not found: {ckpt_path}")
        return 1

    gnn = GNNEncoder.load(str(ckpt_path))
    gnn.eval()
    gnn = gnn.to(device_t)
    n_params = sum(p.numel() for p in gnn.parameters())
    print(f"GNN:        {n_params:,} params, hidden={gnn.config.hidden_dim}")

    # ---- Compute GNN Embeddings ----
    print("\nComputing GNN embeddings...")
    t0 = time.time()
    features = extract_initial_features(graph, gnn.config, device=device_t)
    sources, targets, edge_types, num_nodes = prepare_graph_tensors(graph, device_t)

    with torch.no_grad():
        node_embeddings = gnn(features, sources, targets, edge_types, num_nodes)
    print(f"Embeddings: {node_embeddings.shape} ({time.time()-t0:.1f}s)")

    # Build lemma name → index mapping
    node_ids = sorted(graph.node_ids)
    lemma_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    # ---- Load Theorems ----
    tp = Path(args.theorems)
    if not tp.is_absolute():
        tp = _project_root / tp
    if not tp.exists():
        print(f"Error: theorem file not found: {tp}")
        return 1

    with open(tp) as f:
        theorems = [json.loads(line) for line in f]
    print(f"Theorems:   {len(theorems)}")
    print()

    # ---- Initialize Scoring Architectures ----
    embed_dim = gnn.config.hidden_dim
    two_tower = TwoTowerBilinear(embed_dim=embed_dim).to(device_t)
    two_tower.eval()
    cross_attn = CrossAttentionScorer(embed_dim=embed_dim).to(device_t)
    cross_attn.eval()

    # ---- Proof Checker ----
    checker = BatchChecker(timeout=30, max_workers=4, cache_size=256)

    # ---- Benchmark ----
    architectures = {
        "baseline_cosine": {
            "name": "Cosine Similarity (Baseline)",
            "type": "scoring",
            "scorer": None,  # Uses cosine_similarity_scoring function
        },
        "two_tower_bilinear": {
            "name": "Two-Tower Bilinear",
            "type": "scoring",
            "scorer": two_tower,
        },
        "cross_attention": {
            "name": "Cross-Attention Goal→Candidates",
            "type": "scoring",
            "scorer": cross_attn,
        },
        "graph_filtered_k2": {
            "name": "Graph-Filtered (k=2) + Cosine",
            "type": "graph_filtered",
            "k_hops": 2,
        },
        "graph_filtered_k3": {
            "name": "Graph-Filtered (k=3) + Cosine",
            "type": "graph_filtered",
            "k_hops": 3,
        },
    }

    # Accumulate metrics across all theorems
    all_results: dict[str, dict] = {
        arch_key: {
            "name": arch_info["name"],
            "type": arch_info["type"],
            "per_theorem": [],
            "aggregate": {},
        }
        for arch_key, arch_info in architectures.items()
    }
    theorem_order: list[str] = []

    # For each theorem, collect proving lemmas
    print("Running benchmark...")
    print("-" * 70)

    for i, theorem in enumerate(theorems):
        name = theorem["name"]
        statement = theorem["statement"]
        description = theorem.get("description", name)

        t_start = time.time()

        if args.verbose:
            print(f"\n[{i+1}/{len(theorems)}] {name}")
            print(f"  Statement: {statement[:100]}...")

        # Compute goal embedding
        goal_emb = compute_goal_embedding(
            statement, gnn, node_embeddings,
            node_ids, lemma_to_idx, graph,
        )

        # Get candidate lemmas
        candidates = get_candidate_lemmas(
            statement, graph, lemma_to_idx, top_k=args.top_k_lemmas
        )

        # Get candidate embeddings
        candidate_embs = []
        valid_candidates = []
        for lemma in candidates:
            if lemma in lemma_to_idx:
                candidate_embs.append(node_embeddings[lemma_to_idx[lemma]])
                valid_candidates.append(lemma)
            else:
                # Built-in lemma not in graph — use random small vector
                candidate_embs.append(torch.randn(embed_dim, device=device_t) * 0.01)
                valid_candidates.append(lemma)

        if not valid_candidates:
            continue

        cand_embs_t = torch.stack(candidate_embs)  # [C, D]

        # ---- Find "proving" lemmas: check which top lemmas prove the theorem ----
        # Pool all unique scores from all methods, check top candidates
        all_scored: dict[str, list[tuple[int, float, str]]] = {}

        # Score using each architecture
        for arch_key, arch_info in architectures.items():
            if arch_info["type"] == "scoring":
                scorer = arch_info["scorer"]
                if scorer is None:  # Baseline cosine
                    scores = cosine_similarity_scoring(goal_emb, cand_embs_t)
                else:
                    with torch.no_grad():
                        scores = scorer(goal_emb, cand_embs_t)
            elif arch_info["type"] == "graph_filtered":
                # Find goal-relevant nodes for graph filtering
                keywords = _extract_math_keywords(statement)
                goal_node_ids = []
                for nid in graph.node_ids:
                    for kw in keywords[:5]:
                        if kw.lower() in nid.lower():
                            goal_node_ids.append(nid)
                            break
                    if len(goal_node_ids) >= 10:
                        break

                scores = graph_filtered_retrieval(
                    goal_emb, cand_embs_t, valid_candidates,
                    goal_node_ids, graph,
                    k_hops=arch_info["k_hops"],
                )
            else:
                continue  # Unknown architecture type, skip

            all_scored[arch_key] = [
                (idx, scores[idx].item(), valid_candidates[idx])
                for idx in range(len(valid_candidates))
            ]
            all_scored[arch_key].sort(key=lambda x: -x[1])

        # Collect unique lemmas to proof-check (top from each method)
        lemmas_to_check: set[str] = set()
        for arch_key in all_scored:
            for idx, score, lemma_name in all_scored[arch_key][:args.max_check_lemmas]:
                lemmas_to_check.add(lemma_name)

        # Proof-check each unique lemma
        proving_lemmas: dict[str, bool] = {}
        for lemma_name in lemmas_to_check:
            proving_lemmas[lemma_name] = check_lemma_proves_theorem(
                statement, lemma_name, checker
            )

        # Compute proving indices for each architecture's ranked list
        proving_indices_all: dict[str, set[int]] = {}
        for arch_key in all_scored:
            proving = set()
            for idx, score, lemma_name in all_scored[arch_key]:
                if proving_lemmas.get(lemma_name, False):
                    proving.add(idx)
            proving_indices_all[arch_key] = proving

        # Compute metrics for each architecture on this theorem
        theorem_result = {
            "theorem": name,
            "statement": statement[:200],
            "num_candidates": len(valid_candidates),
            "candidates": valid_candidates[:20],
            "proving_lemmas": [l for l, ok in proving_lemmas.items() if ok],
            "duration_s": time.time() - t_start,
            "architectures": {},
        }

        for arch_key in architectures:
            # Build score tensor
            scores_list = [s[1] for s in all_scored[arch_key]]
            scores_t = torch.tensor(scores_list, device=device_t)
            proving_idxs = proving_indices_all[arch_key]

            metrics = compute_ranking_metrics(scores_t, proving_idxs)
            theorem_result["architectures"][arch_key] = {
                "name": architectures[arch_key]["name"],
                "top5_lemmas": [
                    all_scored[arch_key][j][2] for j in range(min(5, len(all_scored[arch_key])))
                ],
                "metrics": metrics,
            }

        theorem_order.append(name)
        for arch_key in architectures:
            all_results[arch_key]["per_theorem"].append(
                theorem_result["architectures"][arch_key]["metrics"]
            )

        if args.verbose:
            n_proving = len([l for l, ok in proving_lemmas.items() if ok])
            print(f"  Proving lemmas found: {n_proving}")
            for arch_key in architectures:
                m = theorem_result["architectures"][arch_key]["metrics"]
                print(f"    {architectures[arch_key]['name']:40s} "
                      f"P@1={m['precision@1']:.2f} P@3={m['precision@3']:.3f} "
                      f"P@5={m['precision@5']:.3f} MRR={m['mrr']:.3f}")
        else:
            # Terse progress
            status = ""
            for arch_key in architectures:
                mrr = theorem_result["architectures"][arch_key]["metrics"]["mrr"]
                p1 = theorem_result["architectures"][arch_key]["metrics"]["precision@1"]
                status += f"  {arch_key[:6]:6s} P@1={p1:.1f} MRR={mrr:.2f} |"
            print(f"  [{i+1:2d}/{len(theorems)}] {name:35s}{status}")

    # ---- Aggregate Metrics ----
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)

    agg_results = {}
    for arch_key, arch_info in architectures.items():
        per_theorem = all_results[arch_key]["per_theorem"]
        n = len(per_theorem)

        agg = {}
        for metric in ["precision@1", "precision@3", "precision@5", "precision@10",
                       "mrr", "hit@1", "hit@3", "hit@5", "hit@10"]:
            values = [t[metric] for t in per_theorem if metric in t]
            if values:
                agg[f"mean_{metric}"] = statistics.mean(values)
                agg[f"stdev_{metric}"] = statistics.stdev(values) if n > 1 else 0.0

        # % of theorems with at least one proving lemma in test set
        agg["n_theorems"] = n
        agg["theorems_with_proving_lemma"] = sum(
            1 for t in per_theorem if t.get("first_proving_rank", -1) > 0
        )

        all_results[arch_key]["aggregate"] = agg
        agg_results[arch_key] = agg

        print(f"\n{arch_info['name']}:")
        print(f"  Mean P@1:  {agg.get('mean_precision@1', 0):.3f} ± {agg.get('stdev_precision@1', 0):.3f}")
        print(f"  Mean P@3:  {agg.get('mean_precision@3', 0):.3f} ± {agg.get('stdev_precision@3', 0):.3f}")
        print(f"  Mean P@5:  {agg.get('mean_precision@5', 0):.3f} ± {agg.get('stdev_precision@5', 0):.3f}")
        print(f"  Mean MRR:  {agg.get('mean_mrr', 0):.3f} ± {agg.get('stdev_mrr', 0):.3f}")
        print(f"  Theorems w/ proving lemma: {agg.get('theorems_with_proving_lemma', 0)}/{n}")

    # ---- Write Results ----
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = _project_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build clean output
    output = {
        "study": "H2 Scoring Architecture Benchmark",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "checkpoint": str(ckpt_path),
            "graph_domain": args.domain,
            "theorems_file": str(tp),
            "n_theorems": len(theorems),
            "top_k_lemmas": args.top_k_lemmas,
            "max_check_lemmas": args.max_check_lemmas,
            "device": args.device,
            "baseline_gnn_success": 0.214,  # from gate3 study
        },
        "theorem_order": theorem_order,
        "architectures": {},
    }

    for arch_key, arch_data in all_results.items():
        output["architectures"][arch_key] = {
            "name": arch_data["name"],
            "type": arch_data["type"],
            "aggregate": arch_data["aggregate"],
            "per_theorem": dict(zip(
                theorem_order,
                arch_data["per_theorem"]
            )),
        }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults written to: {output_path}")

    # Clean shutdown
    try:
        checker.check_batch([])  # Force any pending cleanup
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
