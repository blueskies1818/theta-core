#!/usr/bin/env python3
"""Fine-tune GNN on gate2 theorems using full-graph embeddings.

Loads the pretrained full-graph GNN (GoalEncoder trained on 15K proof-step pairs)
and fine-tunes it on gate2_training.jsonl theorems. Matches theorem statements
to graph nodes via normalized text similarity, then trains the GNN to retrieve
the correct lemma.

This bridges the gap between the generic proof-step pretraining and the
specific theorem-proving task tested at Gate 3.

Usage:
    python scripts/finetune_gate2_fullgraph.py

Output: checkpoints/gnn/gate2_fullgraph_finetuned.pt
"""

import argparse
import json
import os as _os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

# ---- CPU thread limit (before any torch computation) ----
for _env in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    _os.environ.setdefault(_env, "4")

import torch
import torch.nn.functional as F

try:
    torch.set_num_threads(4)
except RuntimeError:
    pass

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from src.explorer.dependency_graph import DependencyGraph
from src.explorer.gnn_config import GNNConfig
from src.explorer.gnn_encoder import (
    GNNEncoder, prepare_graph_tensors, extract_initial_features,
)
from scripts.eval_gnn_prover import (
    normalize_expression, build_lemma_norm_index,
)


# ---------------------------------------------------------------------------
# Matching: theorem statement → graph node
# ---------------------------------------------------------------------------

def build_lemma_index(graph: DependencyGraph) -> dict[str, int]:
    """Map lemma names to their integer indices in the graph."""
    index = {}
    for node_id in graph.node_ids:
        short_name = node_id.split(".")[-1] if "." in node_id else node_id
        idx = graph.node_id_to_idx(node_id)
        index[node_id] = idx
        if short_name not in index:
            index[short_name] = idx
    return index


def match_theorems_to_graph(
    theorems: list[dict],
    graph: DependencyGraph,
    lemma_to_idx: dict[str, int],
    idx_to_norm: dict[int, str],
) -> tuple[list[tuple[str, int]], list[str]]:
    """Match theorem statements to graph node indices.

    Returns (matched_pairs, unmatched_names).
    """
    matched = []
    unmatched = []

    # Build inverted index for fast lookup
    norm_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, norm in idx_to_norm.items():
        norm_to_indices[norm].append(idx)

    # Build sub-word token index for partial matching
    token_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, norm in idx_to_norm.items():
        for tok in norm.split():
            if len(tok) >= 2:
                token_to_indices[tok].append(idx)

    for t in theorems:
        stmt = t["statement"]
        name = t["name"]
        norm_stmt = normalize_expression(stmt)

        # Strategy 1: exact normalized match
        candidates = list(norm_to_indices.get(norm_stmt, []))
        if candidates:
            matched.append((stmt, candidates[0]))
            continue

        # Strategy 2: equality decomposition matching
        if " = " in norm_stmt:
            lhs, rhs = norm_stmt.rsplit(" = ", 1)
            for idx, norm in idx_to_norm.items():
                if " = " in norm:
                    nlhs, nrhs = norm.rsplit(" = ", 1)
                    if nlhs.strip() == lhs.strip() and nrhs.strip() == rhs.strip():
                        candidates.append(idx)
                        break
        if candidates:
            matched.append((stmt, candidates[0]))
            continue

        # Strategy 3: token overlap (best match)
        stmt_tokens = set(norm_stmt.split())
        best_idx = None
        best_score = 0
        candidate_set = set()
        for tok in stmt_tokens:
            candidate_set.update(token_to_indices.get(tok, [])[:50])

        for idx in candidate_set:
            norm = idx_to_norm.get(idx, "")
            norm_tokens = set(norm.split())
            overlap = len(stmt_tokens & norm_tokens)
            score = overlap / max(len(stmt_tokens), len(norm_tokens))
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_score >= 0.3:
            matched.append((stmt, best_idx))
        else:
            unmatched.append(name)

    return matched, unmatched


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune full-graph GNN on gate2 theorem data"
    )
    parser.add_argument(
        "--theorems", default="data/raw/gate2_training.jsonl",
        help="Path to gate2 training theorems"
    )
    parser.add_argument(
        "--graph", default="data/graph/dependency_graph_full",
        help="Path to the full dependency graph"
    )
    parser.add_argument(
        "--pretrained", default="checkpoints/gnn/full_graph_pretrained.pt",
        help="Path to pretrained GNN checkpoint"
    )
    parser.add_argument(
        "--output", default="checkpoints/gnn/gate2_fullgraph_finetuned.pt",
        help="Output checkpoint path"
    )
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of fine-tuning epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--gnn-lr", type=float, default=1e-5,
                        help="GNN backbone learning rate (when --finetune-gnn)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--temperature", type=float, default=0.5,
                        help="Softmax temperature")
    parser.add_argument("--finetune-gnn", action="store_true",
                        help="Fine-tune full GNN backbone (not just GoalEncoder)")
    parser.add_argument("--device", default="cpu",
                        help="Device (default: cpu)")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}  |  Threads: {torch.get_num_threads()}")

    # ---- Load full graph ----
    graph_path = _project_root / args.graph
    graph = DependencyGraph.load(graph_path)
    print(f"Graph: {graph.summary()}")

    # ---- Build indices ----
    lemma_to_idx = build_lemma_index(graph)
    print(f"Lemma index: {len(lemma_to_idx)} entries")
    idx_to_norm = build_lemma_norm_index(graph, lemma_to_idx)
    print(f"Norm index: {len(idx_to_norm)} normalized conclusions")

    # ---- Load theorems ----
    tp = _project_root / args.theorems
    with open(tp) as f:
        theorems = [json.loads(line) for line in f]
    print(f"Theorems: {len(theorems)}")

    # ---- Match theorems to graph nodes ----
    matched, unmatched = match_theorems_to_graph(
        theorems, graph, lemma_to_idx, idx_to_norm
    )
    print(f"Matched: {len(matched)}/{len(theorems)} theorems to graph nodes")
    if unmatched:
        print(f"  Unmatched: {unmatched}")
    if len(matched) < 5:
        print("Error: too few matches for training")
        sys.exit(1)

    # Train/val split (80/20)
    random.seed(42)
    random.shuffle(matched)
    split_idx = int(len(matched) * 0.8)
    train_pairs = matched[:split_idx]
    val_pairs = matched[split_idx:]
    print(f"Train pairs: {len(train_pairs)}  |  Val pairs: {len(val_pairs)}")

    # ---- Load pretrained GNN ----
    ckpt_path = _project_root / args.pretrained
    if not ckpt_path.exists():
        print(f"Error: pretrained checkpoint not found: {ckpt_path}")
        sys.exit(1)

    gnn = GNNEncoder.load(str(ckpt_path))
    gnn = gnn.to(device)
    total_params = sum(p.numel() for p in gnn.parameters())
    ge_params = sum(p.numel() for p in gnn.goal_encoder.parameters())
    print(f"GNN: {total_params:,} params ({ge_params:,} goal encoder)")

    # ---- Pre-compute graph tensors ----
    print("Computing graph tensors...", end=" ", flush=True)
    t0 = time.time()
    features = extract_initial_features(graph, gnn.config).to(device)
    sources, targets, edge_types, num_nodes = prepare_graph_tensors(graph)
    sources = sources.to(device)
    targets_edges = targets.to(device)
    edge_types = edge_types.to(device)
    print(f"done ({time.time() - t0:.1f}s)")

    # ---- Compute initial GNN embeddings ----
    print("Computing GNN embeddings...", end=" ", flush=True)
    t_gnn = time.time()
    gnn.eval()
    with torch.no_grad():
        raw_embeddings = gnn(features, sources, targets_edges, edge_types, num_nodes)
    embeddings_norm = F.normalize(raw_embeddings, dim=-1)
    print(f"done ({time.time() - t_gnn:.1f}s)")

    # ---- Build context embeddings for each theorem ----
    # For each theorem statement, find related graph nodes via keyword overlap
    # and average their embeddings as context input to GoalEncoder.
    from src.explorer.mcts import _extract_math_keywords

    def build_context_embedding(stmt: str) -> torch.Tensor:
        keywords = _extract_math_keywords(stmt)
        if not keywords:
            keywords = ["unknown"]

        kw_map = defaultdict(list)
        for lemma_name, idx in lemma_to_idx.items():
            short = lemma_name.lower().split(".")[-1]
            tokens = short.replace("_", " ").split()
            for token in tokens:
                if len(token) >= 2:
                    kw_map[token].append(idx)
            kw_map[short].append(idx)

        candidates = set()
        for kw in keywords:
            for idx in kw_map.get(kw.lower(), [])[:40]:
                candidates.add(idx)

        matching_list = list(candidates)[:100]
        if matching_list:
            indices_t = torch.tensor(matching_list, device=device)
            return embeddings_norm[indices_t].mean(dim=0)
        return torch.zeros(gnn.config.hidden_dim, device=device)

    print("Building context embeddings...", end=" ", flush=True)
    t_ctx = time.time()
    train_contexts = []
    train_target_indices = []
    for stmt, target_idx in train_pairs:
        ctx_emb = build_context_embedding(stmt)
        train_contexts.append(ctx_emb)
        train_target_indices.append(target_idx)

    val_contexts = []
    val_target_indices = []
    for stmt, target_idx in val_pairs:
        ctx_emb = build_context_embedding(stmt)
        val_contexts.append(ctx_emb)
        val_target_indices.append(target_idx)

    train_ctx_t = torch.stack(train_contexts)
    train_targets_t = torch.tensor(train_target_indices, device=device)
    val_ctx_t = torch.stack(val_contexts)
    val_targets_t = torch.tensor(val_target_indices, device=device)
    print(f"done ({time.time() - t_ctx:.1f}s)")

    # ---- Setup optimizer ----
    finetune_gnn = args.finetune_gnn
    if finetune_gnn:
        optimizer = torch.optim.AdamW([
            {"params": gnn.goal_encoder.parameters(), "lr": args.lr},
            {"params": [p for n, p in gnn.named_parameters()
                        if "goal_encoder" not in n], "lr": args.gnn_lr},
        ], lr=args.lr, weight_decay=1e-5)
        print(f"Fine-tuning: GNN (lr={args.gnn_lr}) + GoalEncoder (lr={args.lr})")
    else:
        optimizer = torch.optim.AdamW(
            gnn.goal_encoder.parameters(), lr=args.lr, weight_decay=1e-5
        )
        print(f"Training GoalEncoder only (lr={args.lr})")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # ---- Training loop ----
    best_val_acc = 0.0
    print(f"\nTraining: {args.epochs} epochs, batch_size={args.batch_size}, "
          f"temperature={args.temperature}, finetune_gnn={finetune_gnn}")
    print(f"{'Epoch':>5} | {'Loss':>8} | {'Train Top-1':>10} | "
          f"{'Val Top-1':>9} | {'Val Top-5':>9} | {'LR':>10} | {'Time':>6}")

    for epoch in range(args.epochs):
        t0 = time.time()

        # Recompute embeddings if fine-tuning GNN
        if finetune_gnn:
            gnn.train()
            raw_embeddings = gnn(
                features, sources, targets_edges, edge_types, num_nodes
            )
            embeddings_norm = F.normalize(raw_embeddings, dim=-1)
            # Recompute context embeddings
            for i, (stmt, _) in enumerate(train_pairs):
                train_ctx_t[i] = build_context_embedding(stmt)
            for i, (stmt, _) in enumerate(val_pairs):
                val_ctx_t[i] = build_context_embedding(stmt)
        else:
            gnn.goal_encoder.train()

        # Training
        epoch_loss = 0.0
        correct_top1 = 0
        correct_top5 = 0
        n_train = len(train_pairs)

        indices = list(range(n_train))
        random.shuffle(indices)

        for batch_start in range(0, n_train, args.batch_size):
            batch_indices = indices[batch_start:batch_start + args.batch_size]
            if len(batch_indices) < 2:
                continue

            batch_idx_t = torch.tensor(batch_indices, device=device)
            batch_goal_embs = gnn.goal_encoder(train_ctx_t[batch_idx_t])
            batch_targets = train_targets_t[batch_indices]

            logits = torch.matmul(
                batch_goal_embs, embeddings_norm.T
            ) / args.temperature
            loss = F.cross_entropy(logits, batch_targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gnn.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

            with torch.no_grad():
                _, top5 = torch.topk(logits.detach(), k=min(5, num_nodes), dim=1)
                for j, tgt in enumerate(batch_targets.tolist()):
                    if top5[j, 0].item() == tgt:
                        correct_top1 += 1
                    if tgt in top5[j].tolist():
                        correct_top5 += 1

        scheduler.step()
        avg_loss = epoch_loss / max(1, n_train)
        train_top1 = correct_top1 / max(1, n_train)
        train_top5 = correct_top5 / max(1, n_train)

        # Validation
        gnn.eval()
        val_top1 = 0.0
        val_top5 = 0.0
        with torch.no_grad():
            val_goal_embs = gnn.goal_encoder(val_ctx_t)
            val_logits = torch.matmul(
                val_goal_embs, embeddings_norm.T
            ) / args.temperature

            _, top5_val = torch.topk(
                val_logits.detach(), k=min(5, num_nodes), dim=1
            )
            val_correct1 = 0
            val_correct5 = 0
            n_val = min(len(val_pairs), top5_val.size(0))
            for j, tgt in enumerate(val_targets_t[:n_val].tolist()):
                if top5_val[j, 0].item() == tgt:
                    val_correct1 += 1
                if tgt in top5_val[j].tolist():
                    val_correct5 += 1
            val_top1 = val_correct1 / max(1, n_val)
            val_top5 = val_correct5 / max(1, n_val)

        elapsed = time.time() - t0
        print(f"{epoch+1:5d} | {avg_loss:8.4f} | {train_top1:10.1%} | "
              f"{val_top1:9.1%} | {val_top5:9.1%} | "
              f"{scheduler.get_last_lr()[0]:10.2e} | {elapsed:5.1f}s")

        if val_top1 > best_val_acc:
            best_val_acc = val_top1
            output_path = _project_root / args.output
            output_path.parent.mkdir(parents=True, exist_ok=True)
            gnn.save(output_path)
            print(f"  → Saved best model (val top-1={best_val_acc:.1%})")

    # Save final
    output_path = _project_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gnn.save(output_path)
    print(f"\nFine-tuning complete. Best val top-1: {best_val_acc:.1%}")
    print(f"Model saved to {output_path}")

    # ---- Per-theorem verification ----
    print("\nPer-theorem accuracy (all training + validation):")
    gnn.eval()
    with torch.no_grad():
        eval_emb = gnn(features, sources, targets_edges, edge_types, num_nodes)
        eval_emb_norm = F.normalize(eval_emb, dim=-1)
        for stmt, target_idx in matched:
            ctx_emb = build_context_embedding(stmt)
            goal_emb = gnn.goal_encoder(ctx_emb.unsqueeze(0)).squeeze(0)
            logits = goal_emb @ eval_emb_norm.T / args.temperature
            pred = logits.argmax().item()
            correct = "✓" if pred == target_idx else "✗"
            top5 = torch.topk(logits, k=min(5, num_nodes)).indices.tolist()
            in_top5 = "top5" if target_idx in top5 else ""
            print(f"  {correct} {stmt[:60]:60s} → pred={pred} target={target_idx} {in_top5}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
