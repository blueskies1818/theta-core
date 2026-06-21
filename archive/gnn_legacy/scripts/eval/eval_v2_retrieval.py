"""
Evaluate pretrained GNN+GoalEncoder on gate3_v2 lemma retrieval (MRR).

Loads the v2 enriched graph + pretrained model, then for each theorem in
gate3_v2.jsonl, computes the goal embedding, ranks all lemma embeddings by
cosine similarity, and reports MRR and Top-k accuracy.

Usage:
    python scripts/eval/eval_v2_retrieval.py \
        --checkpoint checkpoints/gnn/v2_enriched_pretrained.pt \
        --graph data/graph/dependency_graph_full_v2 \
        --gate3 data/raw/gate3_v2.jsonl \
        --output data/gnn_enriched_baseline_gate3.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_os = __import__("os")
for _env in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    _os.environ.setdefault(_env, "4")

import torch
import torch.nn.functional as F

try:
    torch.set_num_threads(4)
except RuntimeError:
    pass

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))

from src.explorer.dependency_graph import DependencyGraph
from src.explorer.gnn_config import GNNConfig
from src.explorer.gnn_encoder import GNNEncoder, prepare_graph_tensors, extract_initial_features


def load_checkpoint(checkpoint_path: Path, device: torch.device):
    """Load trained model from checkpoint."""
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = GNNConfig(**state.get("config", {}))
    gnn = GNNEncoder(config).to(device)
    gnn.load_state_dict(state["model"])
    gnn.eval()
    return gnn, config


def compute_retrieval_metrics(
    goal_embs: torch.Tensor,    # [N, D]
    lemma_embs: torch.Tensor,   # [M, D]
    target_indices: list[int],  # [N] — correct lemma index for each goal
) -> dict:
    """Compute MRR, Top-1, Top-5, Top-10 retrieval metrics."""
    goal_embs = F.normalize(goal_embs, dim=-1)
    lemma_embs = F.normalize(lemma_embs, dim=-1)

    similarities = torch.matmul(goal_embs, lemma_embs.T)  # [N, M]
    
    # Rank: higher similarity = better
    _, sorted_indices = torch.sort(similarities, dim=1, descending=True)

    total = len(target_indices)
    reciprocal_ranks = []
    top1 = 0
    top5 = 0
    top10 = 0
    top50 = 0

    for i, tgt in enumerate(target_indices):
        rank_positions = (sorted_indices[i] == tgt).nonzero(as_tuple=True)[0]
        if len(rank_positions) == 0:
            continue
        rank = rank_positions[0].item() + 1  # 1-indexed
        reciprocal_ranks.append(1.0 / rank)
        if rank <= 1:
            top1 += 1
        if rank <= 5:
            top5 += 1
        if rank <= 10:
            top10 += 1
        if rank <= 50:
            top50 += 1

    n = max(1, total)
    return {
        "mrr": sum(reciprocal_ranks) / n if reciprocal_ranks else 0.0,
        "top1_accuracy": top1 / n,
        "top5_accuracy": top5 / n,
        "top10_accuracy": top10 / n,
        "top50_accuracy": top50 / n,
        "num_evaluated": total,
        "num_candidates": lemma_embs.shape[0],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/gnn/v2_enriched_pretrained.pt")
    parser.add_argument("--graph", default="data/graph/dependency_graph_full_v2")
    parser.add_argument("--gate3", default="data/raw/gate3_v2.jsonl")
    parser.add_argument("--lemma-index", default="data/graph/dependency_graph_full_v2.lemma_index.json")
    parser.add_argument("--output", default="data/gnn_enriched_baseline_gate3.json")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device} | Threads: {torch.get_num_threads()}")

    # Load graph
    graph_path = _project_root / args.graph
    print(f"\nLoading graph: {graph_path}...", flush=True)
    t0 = time.time()
    dg = DependencyGraph.load(graph_path)
    print(f"  {dg.summary()} ({time.time() - t0:.1f}s)", flush=True)

    # Load lemma index
    lemma_index_path = _project_root / args.lemma_index
    with open(lemma_index_path) as f:
        lemma_to_idx = json.load(f)
    print(f"Lemma index: {len(lemma_to_idx)} entries", flush=True)

    # Load checkpoint
    ckpt_path = _project_root / args.checkpoint
    print(f"\nLoading model: {ckpt_path}...", flush=True)
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        print("The model hasn't finished training yet. Run pretrain_full_graph_v2.py first.")
        sys.exit(1)

    gnn, config = load_checkpoint(ckpt_path, device)
    total_params = sum(p.numel() for p in gnn.parameters())
    print(f"  Model: {total_params:,} params", flush=True)

    # Compute graph tensors
    print("Computing graph tensors...", end=" ", flush=True)
    t0 = time.time()
    features = extract_initial_features(dg, config).to(device)
    sources, targets, edge_types, num_nodes = prepare_graph_tensors(dg)
    sources = sources.to(device)
    targets = targets.to(device)
    edge_types = edge_types.to(device)
    print(f"done ({time.time() - t0:.1f}s)", flush=True)

    # Compute lemma embeddings (GNN forward)
    print("Computing lemma embeddings...", end=" ", flush=True)
    t0 = time.time()
    with torch.no_grad():
        lemma_embs = gnn(features, sources, targets, edge_types, num_nodes)
    lemma_embs = F.normalize(lemma_embs, dim=-1)
    print(f"done ({time.time() - t0:.1f}s). Shape: {list(lemma_embs.shape)}", flush=True)

    # Load gate3_v2 theorems
    gate3_path = _project_root / args.gate3
    print(f"\nLoading gate3_v2 from {gate3_path}...", flush=True)
    theorems = []
    with open(gate3_path) as f:
        for line in f:
            line = line.strip()
            if line:
                theorems.append(json.loads(line))
    print(f"  {len(theorems)} theorems loaded", flush=True)

    # Prepare goal embeddings
    print("Computing goal embeddings...", flush=True)
    goals = [t.get("goal", t.get("statement", "")) for t in theorems]
    ground_truth_lemmas = [t.get("lemma", t.get("ground_truth_lemma", "")) for t in theorems]

    # Build goal context using keyword matching (same as training)
    from scripts.eval.eval_gnn_prover import build_lemma_norm_index
    from scripts.training.pretrain_full_graph_v2 import precompute_goal_contexts

    idx_to_norm = build_lemma_norm_index(dg, lemma_to_idx)
    print(f"  Norm index: {len(idx_to_norm)} entries", flush=True)

    print("  Precomputing goal contexts...", end=" ", flush=True)
    t0 = time.time()
    contexts = precompute_goal_contexts(goals, lemma_to_idx, idx_to_norm)
    print(f"done ({time.time() - t0:.1f}s)", flush=True)

    # Compute raw context embeddings
    raw_ctxs = torch.zeros(len(goals), config.hidden_dim, device=device)
    for i, ctx_indices in enumerate(contexts):
        if ctx_indices:
            idx_t = torch.tensor(ctx_indices, device=device)
            raw_ctxs[i] = lemma_embs[idx_t].mean(dim=0)

    # Compute goal embeddings via GoalEncoder
    print("Computing goal embeddings via GoalEncoder...", end=" ", flush=True)
    t0 = time.time()
    with torch.no_grad():
        goal_embs = gnn.goal_encoder(raw_ctxs)
    print(f"done ({time.time() - t0:.1f}s)", flush=True)

    # Map ground-truth lemmas to indices
    target_indices = []
    missing = 0
    for lemma in ground_truth_lemmas:
        idx = lemma_to_idx.get(lemma)
        if idx is not None:
            target_indices.append(idx)
        else:
            target_indices.append(-1)
            missing += 1
    if missing:
        print(f"  WARNING: {missing} ground-truth lemmas not in lemma_index", flush=True)

    # Filter to theorems with valid targets
    valid_mask = [i >= 0 for i in target_indices]
    valid_goal_embs = goal_embs[torch.tensor(valid_mask)]
    valid_targets = [t for t, ok in zip(target_indices, valid_mask) if ok]
    valid_theorems = [t for t, ok in zip(theorems, valid_mask) if ok]

    print(f"  Valid theorems: {len(valid_targets)}/{len(theorems)}", flush=True)

    # Compute retrieval metrics
    print(f"\n{'='*60}")
    print(f"MRR Evaluation on gate3_v2 ({len(valid_targets)} theorems)")
    print(f"{'='*60}")
    
    results = compute_retrieval_metrics(valid_goal_embs, lemma_embs, valid_targets)

    print(f"\nMRR Results:")
    print(f"  MRR:               {results['mrr']:.4f}")
    print(f"  Top-1 accuracy:    {results['top1_accuracy']:.1%}")
    print(f"  Top-5 accuracy:    {results['top5_accuracy']:.1%}")
    print(f"  Top-10 accuracy:   {results['top10_accuracy']:.1%}")
    print(f"  Top-50 accuracy:   {results['top50_accuracy']:.1%}")
    print(f"  Theorems evaluated: {results['num_evaluated']}")
    print(f"  Candidate lemmas:   {results['num_candidates']}")

    # Per-domain breakdown
    from collections import defaultdict
    domain_results = defaultdict(lambda: {"total": 0, "hits": [], "targets": []})
    for theorem, tgt in zip(valid_theorems, valid_targets):
        domain = theorem.get("domain", "Unknown")
        domain_results[domain]["total"] += 1
        domain_results[domain]["targets"].append(tgt)

    if len(valid_targets) > 0:
        print(f"\nPer-Domain Top-1:")
        similarities = torch.matmul(
            F.normalize(valid_goal_embs, dim=-1), 
            F.normalize(lemma_embs, dim=-1)
        )
        _, sorted_indices = torch.sort(similarities, dim=1, descending=True)
        
        domain_top1 = defaultdict(lambda: {"correct": 0, "total": 0})
        for i, tgt in enumerate(valid_targets):
            domain = valid_theorems[i].get("domain", "Unknown")
            domain_top1[domain]["total"] += 1
            if sorted_indices[i, 0].item() == tgt:
                domain_top1[domain]["correct"] += 1
        
        for domain in sorted(domain_top1.keys()):
            d = domain_top1[domain]
            print(f"  {domain:25}: {d['correct']:2d}/{d['total']:2d} ({d['correct']/d['total']:.0%})")

    # Save results
    output_path = _project_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "checkpoint": str(ckpt_path),
        "graph": str(graph_path),
        "gate3_theorems": len(theorems),
        "valid_theorems": len(valid_targets),
        **results,
        "domain_breakdown": dict(domain_top1) if len(valid_targets) > 0 else {},
        "baseline_15_6_pct": 15.6,
        "beats_baseline": results.get("top1_accuracy", 0) > 0.156,
    }
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")

    if results.get("top1_accuracy", 0) > 0.156:
        print(f"\n✓ EXCEEDS 15.6% baseline — indexing was the bottleneck!")
    else:
        print(f"\n✗ Below 15.6% baseline — embedding quality may also need improvement")


if __name__ == "__main__":
    main()
