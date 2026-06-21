#!/usr/bin/env python3
"""
Train GNN from scratch on dependency_graph_full_v3 with proof edge types.

Uses the enriched v3 graph (139K nodes, 650K edges, 6 edge types including
PROVED_BY and CO_OCCURS_IN_PROOF). Trains link prediction on 214K resolved
proof-step pairs.

Key changes from pretrain_full_graph.py:
  - v3 graph with 6 edge types (PROVED_BY + CO_OCCURS_IN_PROOF added)
  - All 214K resolved pairs for training (was ~15K)
  - Fresh training from scratch (GoalEncoder only, batched)
  - Batched validation to avoid OOM

Output: checkpoints/gnn/proof_edge_scratch.pt
"""
import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

# ---- CPU thread limit (2 threads to avoid OOM on 139K-node graph) ----
import os as _os
for _env in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    _os.environ.setdefault(_env, "2")

import torch
import torch.nn.functional as F
try:
    torch.set_num_threads(2)
except RuntimeError:
    pass

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))

from src.explorer.dependency_graph import DependencyGraph
from src.explorer.gnn_config import GNNConfig
from src.explorer.gnn_encoder import GNNEncoder, prepare_graph_tensors, extract_initial_features
from scripts.eval.eval_gnn_prover import (
    normalize_expression, build_lemma_norm_index,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_lemma_index(graph: DependencyGraph) -> dict[str, int]:
    """Map lemma names to integer graph indices."""
    index = {}
    for node_id in graph.node_ids:
        short_name = node_id.split(".")[-1] if "." in node_id else node_id
        idx = graph.node_id_to_idx(node_id)
        index[node_id] = idx
        if short_name not in index:
            index[short_name] = idx
    return index


def precompute_goal_contexts(
    goals: list[str],
    lemma_to_idx: dict[str, int],
    idx_to_norm: dict[int, str],
) -> list[list[int]]:
    """Match goal keywords to lemma nodes."""
    from src.explorer.mcts import _extract_math_keywords
    norm_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, norm in idx_to_norm.items():
        norm_to_indices[norm].append(idx)

    contexts = []
    for goal in goals:
        keywords = _extract_math_keywords(goal)
        matched = []
        for kw in keywords:
            kw_norm = normalize_expression(kw)
            if kw_norm in norm_to_indices:
                matched.extend(norm_to_indices[kw_norm][:5])
        contexts.append(list(set(matched)))
    return contexts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train GNN from scratch on v3 graph")
    parser.add_argument("--graph", default="data/graph/dependency_graph_full_v3")
    parser.add_argument("--data", default="data/raw/proof_step_pairs.jsonl")
    parser.add_argument("--output", default="checkpoints/gnn/proof_edge_scratch.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}  |  Threads: {torch.get_num_threads()}")

    # ---- Load graph ----
    graph_path = _project_root / args.graph
    graph = DependencyGraph.load(graph_path)
    print(f"Graph: {graph.summary()}")

    # ---- Build indices ----
    lemma_to_idx = build_lemma_index(graph)
    print(f"Lemma index: {len(lemma_to_idx)} entries")
    idx_to_norm = build_lemma_norm_index(graph, lemma_to_idx)
    print(f"Norm index: {len(idx_to_norm)} normalized conclusions")

    # ---- Load proof-step pairs (using enriched lemma_index) ----
    data_path = _project_root / args.data
    lemma_index_path = graph_path.parent / (graph_path.name + ".lemma_index.json")

    # Load enriched lemma index: lemma_name → int index
    enriched_lemma_idx = {}
    if lemma_index_path.exists():
        with open(lemma_index_path) as f:
            enriched_lemma_idx = json.load(f)
        print(f"Enriched lemma index: {len(enriched_lemma_idx)} entries")

    # Build reverse map: int index → node_id
    idx_to_node = {}
    for node_id in graph.node_ids:
        idx = graph.node_id_to_idx(node_id)
        if idx is not None:
            idx_to_node[idx] = node_id

    pairs = []
    with open(data_path) as f:
        for line in f:
            d = json.loads(line)
            lemma = d["lemma"]
            resolved_idx = None

            # Try enriched index first (96% recall)
            if lemma in enriched_lemma_idx:
                lemma_int = enriched_lemma_idx[lemma]
                node_id = idx_to_node.get(lemma_int)
                if node_id is not None:
                    resolved_idx = graph.node_id_to_idx(node_id)

            # Fallback: direct node_id → idx lookup
            if resolved_idx is None and lemma in lemma_to_idx:
                resolved_idx = lemma_to_idx[lemma]

            if resolved_idx is not None:
                d["_lemma_idx"] = resolved_idx
                pairs.append(d)

    print(f"Proof-step pairs: {len(pairs)} (matching graph lemmas)")
    if len(pairs) < 500:
        print("Error: too few matching pairs.")
        sys.exit(1)

    # ---- Train/val split ----
    random.seed(42)
    random.shuffle(pairs)
    split_idx = int(len(pairs) * 0.9)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]
    print(f"Train pairs: {len(train_pairs)}  |  Val pairs: {len(val_pairs)}")

    # ---- Create GNN ----
    config = GNNConfig(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        input_dim=args.hidden_dim,
        dropout=0.1,
        activation="gelu",
        use_goal_encoder=True,
        goal_encoder_expansion=2,
        goal_encoder_dropout=0.1,
        use_edge_types=True,
        num_edge_types=6,
    )
    gnn = GNNEncoder(config).to(device)
    total_params = sum(p.numel() for p in gnn.parameters())
    ge_params = sum(p.numel() for p in gnn.goal_encoder.parameters())
    print(f"GNN: {total_params:,} params ({ge_params:,} goal encoder)")

    # ---- Pre-compute graph tensors ----
    print("Computing graph tensors...", end=" ", flush=True)
    t0 = time.time()
    features = extract_initial_features(graph, config).to(device)
    sources, targets, edge_types, num_nodes = prepare_graph_tensors(graph)
    sources = sources.to(device)
    targets = targets.to(device)
    edge_types = edge_types.to(device)
    print(f"done ({time.time() - t0:.1f}s)")
    print(f"  Virtual edge types: {edge_types.unique(return_counts=True)}")

    # ---- Training: GoalEncoder only, GNN frozen ----
    print("Training GoalEncoder only (GNN frozen)")
    optimizer = torch.optim.AdamW(
        gnn.goal_encoder.parameters(), lr=args.lr, weight_decay=1e-5
    )
    gnn.eval()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ---- Prepare data ----
    train_goals = [p["goal"] for p in train_pairs]
    train_targets = [p["_lemma_idx"] for p in train_pairs]
    val_goals = [p["goal"] for p in val_pairs]
    val_targets = [p["_lemma_idx"] for p in val_pairs]

    # ---- Precompute goal contexts ----
    print("Precomputing goal contexts...", end=" ", flush=True)
    t_pre = time.time()
    train_contexts = precompute_goal_contexts(train_goals, lemma_to_idx, idx_to_norm)
    val_contexts = precompute_goal_contexts(val_goals, lemma_to_idx, idx_to_norm)
    hits = sum(1 for c in train_contexts if c)
    print(f"done ({time.time() - t_pre:.1f}s). {hits}/{len(train_goals)} train goals have context matches.")

    # ---- Initial GNN embeddings (frozen) ----
    print("Computing initial GNN embeddings...", end=" ", flush=True)
    t_gnn = time.time()
    with torch.no_grad():
        embeddings = gnn(features, sources, targets, edge_types, num_nodes)
    embeddings = embeddings.detach()
    embeddings_norm = F.normalize(embeddings, dim=-1)
    print(f"done ({time.time() - t_gnn:.1f}s)")

    # ---- Precompute context embeddings ----
    print("Precomputing context embeddings...", end=" ", flush=True)
    t_ctx = time.time()
    raw_train_ctxs = torch.zeros(len(train_pairs), config.hidden_dim, device=device)
    for i, ctx_indices in enumerate(train_contexts):
        if ctx_indices:
            idx_t = torch.tensor(ctx_indices, device=device)
            raw_train_ctxs[i] = embeddings_norm[idx_t].mean(dim=0)
    raw_val_ctxs = torch.zeros(len(val_pairs), config.hidden_dim, device=device)
    for i, ctx_indices in enumerate(val_contexts):
        if ctx_indices:
            idx_t = torch.tensor(ctx_indices, device=device)
            raw_val_ctxs[i] = embeddings_norm[idx_t].mean(dim=0)
    print(f"done ({time.time() - t_ctx:.1f}s)")

    print(f"\nTraining: {args.epochs} epochs, batch_size={args.batch_size}, "
          f"temperature={args.temperature}")

    best_val_acc = 0.0
    best_val_loss = float("inf")
    train_targets_t = torch.tensor(train_targets, device=device)
    val_targets_t = torch.tensor(val_targets, device=device)

    for epoch in range(args.epochs):
        t_epoch = time.time()

        gnn.goal_encoder.train()

        # ---- Batched training ----
        epoch_loss = 0.0
        num_batches = 0
        correct_top1 = 0
        correct_top5 = 0
        total_eval = 0

        indices = list(range(len(train_pairs)))
        random.shuffle(indices)

        for batch_start in range(0, len(indices), args.batch_size):
            batch_indices = indices[batch_start:batch_start + args.batch_size]
            if len(batch_indices) < 2:
                continue

            batch_idx_t = torch.tensor(batch_indices, device=device)
            batch_goal_embs = gnn.goal_encoder(raw_train_ctxs[batch_idx_t])
            batch_targets = train_targets_t[batch_indices]

            logits = torch.matmul(batch_goal_embs, embeddings_norm.T) / args.temperature
            loss = F.cross_entropy(logits, batch_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            with torch.no_grad():
                _, top5 = torch.topk(logits.detach(), k=min(5, num_nodes), dim=1)
                for j, tgt in enumerate(batch_targets.tolist()):
                    total_eval += 1
                    if top5[j, 0].item() == tgt:
                        correct_top1 += 1
                    if tgt in top5[j].tolist():
                        correct_top5 += 1

        scheduler.step()
        avg_loss = epoch_loss / max(1, num_batches)
        train_top1 = correct_top1 / max(1, total_eval)
        train_top5 = correct_top5 / max(1, total_eval)

        # ---- Batched validation (every 5 epochs) ----
        val_top1 = 0.0
        val_top5 = 0.0
        val_loss = 0.0
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            gnn.eval()
            with torch.no_grad():
                # Batched to avoid [21727, 139183] = 12GB OOM
                val_all_logits = []
                for vb in range(0, len(val_pairs), 256):
                    vb_end = min(vb + 256, len(val_pairs))
                    vb_t = torch.arange(vb, vb_end, device=device)
                    vb_goal = gnn.goal_encoder(raw_val_ctxs[vb_t])
                    vb_logits = torch.matmul(vb_goal, embeddings_norm.T) / args.temperature
                    val_all_logits.append(vb_logits)

                val_logits = torch.cat(val_all_logits, dim=0)
                val_loss = F.cross_entropy(
                    val_logits[:len(val_targets)], val_targets_t[:len(val_logits)]
                ).item()

                _, top5_val = torch.topk(val_logits, k=min(5, num_nodes), dim=1)
                val_correct1 = sum(
                    1 for j, tgt in enumerate(val_targets[:top5_val.size(0)])
                    if top5_val[j, 0].item() == tgt
                )
                val_correct5 = sum(
                    1 for j, tgt in enumerate(val_targets[:top5_val.size(0)])
                    if tgt in top5_val[j].tolist()
                )
                n_val = min(len(val_targets), top5_val.size(0))
                val_top1 = val_correct1 / max(1, n_val)
                val_top5 = val_correct5 / max(1, n_val)

        elapsed = time.time() - t_epoch

        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"Epoch {epoch:3d}/{args.epochs} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Train Top-1: {train_top1:.1%} | "
                  f"Train Top-5: {train_top5:.1%} | "
                  f"Val Top-1: {val_top1:.1%} | "
                  f"Val Top-5: {val_top5:.1%} | "
                  f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                  f"Time: {elapsed:.1f}s", flush=True)

            if val_top1 > best_val_acc:
                best_val_acc = val_top1
                gnn.save(_project_root / args.output)
                print(f"  -> Saved best model (val top-1={best_val_acc:.1%})")

            if val_loss < best_val_loss and val_loss > 0:
                best_val_loss = val_loss

    # ---- Final ----
    print(f"\nTraining complete: {args.epochs} epochs")
    print(f"  Best val top-1: {best_val_acc:.1%}")
    print(f"  Best val loss:  {best_val_loss:.4f}")
    print(f"  Model saved to {args.output}")


if __name__ == "__main__":
    main()
