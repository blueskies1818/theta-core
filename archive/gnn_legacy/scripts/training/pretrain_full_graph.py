#!/usr/bin/env python3
"""Pretrain GNN on full dependency graph using proof-step prediction.

Trains GNN + GoalEncoder jointly on (goal, lemma) pairs from all Mathlib4
domains. Uses the FULL 116K-node dependency graph — not just Algebra.

Differences from the Algebra-only pretrain_proof_step.py:
- Uses the complete graph (all 84 domains)
- No domain subgraph filtering
- Processes 15K+ matching proof-step pairs

Output: checkpoints/gnn/full_graph_pretrained.pt
"""

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

# ---- CPU thread limit (before any torch computation) ----
import os as _os
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
from src.explorer.gnn_encoder import GNNEncoder, prepare_graph_tensors, extract_initial_features
from scripts.eval.eval_gnn_prover import (
    normalize_expression, build_lemma_norm_index,
)


# ---------------------------------------------------------------------------
# Helpers
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


def embed_goal(
    goal_text: str,
    node_emb_norm: torch.Tensor,
    lemma_to_idx: dict[str, int],
    gnn: GNNEncoder | None = None,
    precomputed_context: list[int] | None = None,
) -> torch.Tensor:
    """Create a goal embedding using precomputed context or keyword fallback."""
    device = node_emb_norm.device

    if precomputed_context:
        indices_t = torch.tensor(precomputed_context, device=device)
        context_emb = node_emb_norm[indices_t].mean(dim=0)
    else:
        # Slow keyword fallback
        from src.explorer.mcts import _extract_math_keywords
        keywords = _extract_math_keywords(goal_text)
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
            context_emb = node_emb_norm[indices_t].mean(dim=0)
        else:
            context_emb = torch.zeros(node_emb_norm.size(1), device=device)

    if gnn is not None and gnn.goal_encoder is not None:
        return gnn.encode_goal(context_emb)
    if context_emb.norm() > 1e-8:
        return F.normalize(context_emb, dim=-1)
    return context_emb


def precompute_goal_contexts(
    goals: list[str],
    lemma_to_idx: dict[str, int],
    idx_to_norm: dict[int, str],
) -> list[list[int]]:
    """Precompute matching lemma indices for each training goal.

    Uses inverted indices for O(1) lookups instead of O(N) scans.
    """
    import re

    # Build inverted indices
    norm_to_indices: dict[str, list[int]] = defaultdict(list)
    stripped_to_indices: dict[str, list[int]] = defaultdict(list)
    iff_rhs_to_indices: dict[str, list[int]] = defaultdict(list)
    iff_lhs_to_indices: dict[str, list[int]] = defaultdict(list)
    imp_rhs_to_indices: dict[str, list[int]] = defaultdict(list)

    rfl_norm = normalize_expression("a = a")

    for idx, lemma_norm in idx_to_norm.items():
        norm_to_indices[lemma_norm].append(idx)

        stripped = re.sub(r'\s*\^\s*\d+', '', lemma_norm)
        stripped_to_indices[stripped].append(idx)

        if " ↔ " in lemma_norm:
            left, right = lemma_norm.split(" ↔ ", 1)
            iff_lhs_to_indices[left.strip()].append(idx)
            iff_rhs_to_indices[right.strip()].append(idx)
        elif " → " in lemma_norm:
            parts = lemma_norm.rsplit(" → ", 1)
            imp_rhs_to_indices[parts[-1].strip()].append(idx)

    contexts = []
    for goal_text in goals:
        goal_norm = normalize_expression(goal_text)
        exact_matches = list(norm_to_indices.get(goal_norm, []))

        # Reflexivity
        if not exact_matches:
            if "=" in goal_norm and "↔" not in goal_norm and "→" not in goal_norm and "≠" not in goal_norm:
                sides = goal_norm.split("=", 1)
                if len(sides) == 2 and sides[0].strip() == sides[1].strip():
                    exact_matches.extend(norm_to_indices.get(rfl_norm, []))

        # Iff/implication matching
        exact_matches.extend(iff_lhs_to_indices.get(goal_norm, []))
        exact_matches.extend(iff_rhs_to_indices.get(goal_norm, []))
        exact_matches.extend(imp_rhs_to_indices.get(goal_norm, []))

        # Power-stripping fallback
        if not exact_matches:
            goal_stripped = re.sub(r'\s*\^\s*\d+', '', goal_norm)
            exact_matches.extend(stripped_to_indices.get(goal_stripped, []))

        # Deduplicate
        seen = set()
        exact_dedup = []
        for idx in exact_matches:
            if idx not in seen:
                seen.add(idx)
                exact_dedup.append(idx)

        contexts.append(exact_dedup[:100])

    return contexts


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Pretrain GNN on full graph via proof-step prediction"
    )
    parser.add_argument(
        "--data", default="data/raw/proof_step_pairs.jsonl",
        help="Path to proof-step pairs JSONL"
    )
    parser.add_argument(
        "--graph", default="data/graph/dependency_graph_full",
        help="Path to the full dependency graph"
    )
    parser.add_argument("--epochs", type=int, default=150,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--output", default="checkpoints/gnn/full_graph_pretrained.pt",
                        help="Output checkpoint path")
    parser.add_argument("--hidden-dim", type=int, default=256,
                        help="Hidden dimension")
    parser.add_argument("--num-layers", type=int, default=3,
                        help="Number of GAT layers")
    parser.add_argument("--num-heads", type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Softmax temperature")
    parser.add_argument("--finetune-gnn", action="store_true",
                        help="Fine-tune full GNN (not just GoalEncoder)")
    parser.add_argument("--gnn-lr", type=float, default=1e-4,
                        help="GNN learning rate when fine-tuning")
    parser.add_argument("--device", default=None,
                        help="Device (default: cpu)")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device("cpu")
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

    # ---- Load proof-step pairs ----
    data_path = _project_root / args.data
    pairs = []
    with open(data_path) as f:
        for line in f:
            d = json.loads(line)
            lemma = d["lemma"]
            if lemma in lemma_to_idx:
                pairs.append(d)

    print(f"Proof-step pairs: {len(pairs)} (matching graph lemmas)")
    if len(pairs) < 500:
        print("Error: too few matching pairs. Check lemma name resolution.")
        sys.exit(1)

    # ---- Train/val split (90/10) ----
    random.seed(42)
    random.shuffle(pairs)
    split_idx = int(len(pairs) * 0.9)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]
    print(f"Train pairs: {len(train_pairs)}  |  Val pairs: {len(val_pairs)}")

    # ---- Create GNN + GoalEncoder ----
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

    # ---- Setup optimizer ----
    finetune_gnn = args.finetune_gnn
    if finetune_gnn:
        optimizer = torch.optim.AdamW([
            {"params": gnn.goal_encoder.parameters(), "lr": args.lr},
            {"params": [p for n, p in gnn.named_parameters()
                        if "goal_encoder" not in n], "lr": args.gnn_lr},
        ], lr=args.lr, weight_decay=1e-5)
        gnn.train()
        print(f"Fine-tuning: GNN (lr={args.gnn_lr}) + GoalEncoder (lr={args.lr})")
    else:
        optimizer = torch.optim.AdamW(
            gnn.goal_encoder.parameters(), lr=args.lr, weight_decay=1e-5
        )
        gnn.eval()
        print("Training GoalEncoder only (GNN frozen)")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ---- Prepare training/validation data ----
    train_goals = [p["goal"] for p in train_pairs]
    train_targets = [lemma_to_idx[p["lemma"]] for p in train_pairs]

    val_goals = [p["goal"] for p in val_pairs]
    val_targets = [lemma_to_idx[p["lemma"]] for p in val_pairs]

    # ---- Precompute goal contexts ----
    print("Precomputing goal contexts...", end=" ", flush=True)
    t_pre = time.time()
    train_contexts = precompute_goal_contexts(train_goals, lemma_to_idx, idx_to_norm)
    val_contexts = precompute_goal_contexts(val_goals, lemma_to_idx, idx_to_norm)
    hits = sum(1 for c in train_contexts if c)
    print(f"done ({time.time() - t_pre:.1f}s). {hits}/{len(train_goals)} train goals have context matches.")

    # ---- Compute initial GNN embeddings ----
    print("Computing initial GNN embeddings...", end=" ", flush=True)
    t_gnn = time.time()
    with torch.no_grad():
        embeddings = gnn(features, sources, targets, edge_types, num_nodes)
    if not finetune_gnn:
        embeddings = embeddings.detach()
    embeddings_norm = F.normalize(embeddings, dim=-1)
    print(f"done ({time.time() - t_gnn:.1f}s)")

    # ---- Precompute raw context embeddings ----
    print("Precomputing raw context embeddings...", end=" ", flush=True)
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
          f"temperature={args.temperature}, finetune_gnn={finetune_gnn}")

    best_val_acc = 0.0
    best_val_loss = float("inf")
    train_targets_t = torch.tensor(train_targets, device=device)
    val_targets_t = torch.tensor(val_targets, device=device)

    for epoch in range(args.epochs):
        t0 = time.time()

        # ---- Recompute embeddings if fine-tuning ----
        if finetune_gnn:
            gnn.train()
            embeddings = gnn(features, sources, targets, edge_types, num_nodes)
            embeddings_norm = F.normalize(embeddings, dim=-1)
            for i, ctx_indices in enumerate(train_contexts):
                if ctx_indices:
                    idx_t = torch.tensor(ctx_indices, device=device)
                    raw_train_ctxs[i] = embeddings_norm[idx_t].mean(dim=0)
            for i, ctx_indices in enumerate(val_contexts):
                if ctx_indices:
                    idx_t = torch.tensor(ctx_indices, device=device)
                    raw_val_ctxs[i] = embeddings_norm[idx_t].mean(dim=0)
        else:
            gnn.goal_encoder.train()

        # ---- Train ----
        epoch_loss = 0.0
        num_batches = 0
        correct_top1 = 0
        correct_top5 = 0
        total_eval = 0

        if finetune_gnn:
            # Full forward/backward (avoids retain_graph issues)
            all_goal_embs = gnn.goal_encoder(raw_train_ctxs)
            logits = torch.matmul(all_goal_embs, embeddings_norm.T) / args.temperature
            loss = F.cross_entropy(logits, train_targets_t[:len(all_goal_embs)])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gnn.parameters(), 1.0)
            optimizer.step()

            epoch_loss = loss.item()
            num_batches = 1

            with torch.no_grad():
                _, top5 = torch.topk(logits.detach(), k=min(5, num_nodes), dim=1)
                for j, tgt in enumerate(train_targets):
                    if j < top5.size(0):
                        correct_top1 += 1 if top5[j, 0].item() == tgt else 0
                        correct_top5 += 1 if tgt in top5[j].tolist() else 0
                total_eval = min(len(train_targets), top5.size(0))
        else:
            # Frozen GNN: batched training
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
                torch.nn.utils.clip_grad_norm_(gnn.parameters(), 1.0)
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

        # ---- Validation ----
        val_top1 = 0.0
        val_top5 = 0.0
        val_loss = 0.0
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            gnn.eval()
            with torch.no_grad():
                val_goal_embs = gnn.goal_encoder(raw_val_ctxs)
                val_logits = torch.matmul(val_goal_embs, embeddings_norm.T) / args.temperature
                val_loss = F.cross_entropy(
                    val_logits[:len(val_targets)], val_targets_t[:len(val_logits)]
                ).item()

                _, top5_val = torch.topk(val_logits.detach(), k=min(5, num_nodes), dim=1)
                val_correct1 = 0
                val_correct5 = 0
                for j, tgt in enumerate(val_targets[:top5_val.size(0)]):
                    if top5_val[j, 0].item() == tgt:
                        val_correct1 += 1
                    if tgt in top5_val[j].tolist():
                        val_correct5 += 1
                n_val = min(len(val_targets), top5_val.size(0))
                val_top1 = val_correct1 / max(1, n_val)
                val_top5 = val_correct5 / max(1, n_val)

        elapsed = time.time() - t0

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
                print(f"  → Saved best model (val top-1={best_val_acc:.1%})")

            if val_loss < best_val_loss and val_loss > 0:
                best_val_loss = val_loss

    # ---- Final ----
    print(f"\nTraining complete: {args.epochs} epochs")
    print(f"  Best val top-1: {best_val_acc:.1%}")
    print(f"  Best val loss:  {best_val_loss:.4f}")
    print(f"  Model saved to {args.output}")

    # ---- Per-lemma accuracy for top lemmas ----
    print("\nPer-lemma accuracy (top-20 most frequent):")
    lemma_freq = defaultdict(int)
    for p in pairs:
        lemma_freq[p["lemma"]] += 1
    top_lemmas = sorted(lemma_freq.items(), key=lambda x: x[1], reverse=True)[:20]

    gnn.eval()
    with torch.no_grad():
        eval_emb = gnn(features, sources, targets, edge_types, num_nodes)
        eval_emb_norm = F.normalize(eval_emb, dim=-1)
        for lemma_name, freq in top_lemmas:
            pair_indices = [i for i, p in enumerate(pairs) if p["lemma"] == lemma_name]
            if not pair_indices:
                continue
            correct = 0
            for idx in pair_indices[:100]:
                goal_emb = embed_goal(
                    pairs[idx]["goal"], eval_emb_norm, lemma_to_idx, gnn,
                    precomputed_context=(
                        train_contexts[idx] if idx < len(train_contexts)
                        else val_contexts[idx - len(train_contexts)]
                    ),
                )
                logits = goal_emb @ eval_emb_norm.T / args.temperature
                pred = logits.argmax().item()
                target_idx = lemma_to_idx[pairs[idx]["lemma"]]
                if pred == target_idx:
                    correct += 1
            acc = correct / min(100, len(pair_indices))
            print(f"  {lemma_name:40s} freq={freq:5d} acc={acc:.1%}")


if __name__ == "__main__":
    main()
