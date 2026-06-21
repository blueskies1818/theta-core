#!/usr/bin/env python3
"""Pretrain GNN on v2 enriched dependency graph (with 98.3% lemma recall).

Uses the enriched lemma_index from dependency_graph_full_v2.lemma_index.json.
Trains GNN + GoalEncoder jointly on (goal, lemma) proof-step pairs.

Output: checkpoints/gnn/full_graph_v2_pretrained.pt
"""

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

# ---- CPU thread limit ----
import os as _os
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
from scripts.eval.eval_gnn_prover import build_lemma_norm_index


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_enriched_lemma_index(graph: DependencyGraph, index_path: Path) -> dict[str, int]:
    """Load enriched lemma_index from JSON with graph index validation."""
    if index_path.exists():
        with open(index_path) as f:
            enriched = json.load(f)
        # Validate indices are within bounds
        num_nodes = graph.num_nodes
        valid = {k: v for k, v in enriched.items()
                 if isinstance(v, int) and 0 <= v < num_nodes}
        print(f"Loaded enriched lemma_index: {len(valid)} entries (from {len(enriched)} JSON entries)")
        return valid
    else:
        print(f"Warning: {index_path} not found, building from graph")
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
    """Precompute matching lemma indices for each training goal.

    Uses inverted indices on normalized lemma conclusions for O(1) lookups
    instead of O(N) scans. (Matches original pretrain_full_graph.py pattern.)
    """
    import re
    from scripts.eval.eval_gnn_prover import normalize_expression

    # Build inverted indices on idx_to_norm (maps node_idx → normalized conclusion)
    norm_to_indices: dict[str, list[int]] = defaultdict(list)
    stripped_to_indices: dict[str, list[int]] = defaultdict(list)
    iff_lhs_to_indices: dict[str, list[int]] = defaultdict(list)
    iff_rhs_to_indices: dict[str, list[int]] = defaultdict(list)
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
        matched: list[int] = list(norm_to_indices.get(goal_norm, []))

        # Reflexivity: try matching rfl if goal is self-equality
        if not matched:
            if "=" in goal_norm and "↔" not in goal_norm and "→" not in goal_norm and "≠" not in goal_norm:
                sides = goal_norm.split("=", 1)
                if len(sides) == 2 and sides[0].strip() == sides[1].strip():
                    matched.extend(norm_to_indices.get(rfl_norm, []))

        # Iff/implication matching
        matched.extend(iff_lhs_to_indices.get(goal_norm, []))
        matched.extend(iff_rhs_to_indices.get(goal_norm, []))
        matched.extend(imp_rhs_to_indices.get(goal_norm, []))

        # Power-stripping fallback
        if not matched:
            goal_stripped = re.sub(r'\s*\^\s*\d+', '', goal_norm)
            matched = list(stripped_to_indices.get(goal_stripped, []))

        contexts.append(list(dict.fromkeys(matched))[:200])  # deduplicate, cap
    return contexts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Pretrain GNN on v2 enriched graph")
    parser.add_argument("--graph", default="data/graph/dependency_graph_full_v2",
                        help="Path to dependency graph (base, without extension)")
    parser.add_argument("--data", default="data/raw/proof_step_pairs.jsonl",
                        help="Path to proof_step_pairs.jsonl")
    parser.add_argument("--output", default="checkpoints/gnn/full_graph_v2_pretrained.pt",
                        help="Output checkpoint path")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4096,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=128,
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

    # ---- Load enriched lemma_index ----
    index_path = graph_path.parent / (graph_path.name + ".lemma_index.json")
    lemma_to_idx = load_enriched_lemma_index(graph, index_path)
    
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

    total_in_file = sum(1 for _ in open(data_path))
    print(f"Proof-step pairs: {len(pairs)} matching / {total_in_file} total "
          f"({len(pairs)/total_in_file*100:.1f}%)")
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
            # Full forward/backward
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

        # ---- Validation (batched to avoid OOM on large graphs) ----
        val_top1 = 0.0
        val_top5 = 0.0
        val_loss = 0.0
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            gnn.eval()
            with torch.no_grad():
                val_batch_size = 1024
                val_loss_total = 0.0
                val_correct1 = 0
                val_correct5 = 0
                n_val = min(len(val_targets), 5000)  # cap val set for memory
                val_indices = list(range(n_val))

                for vb in range(0, n_val, val_batch_size):
                    vb_indices = val_indices[vb:vb + val_batch_size]
                    vb_t = torch.tensor(vb_indices, device=device)
                    vb_goal_embs = gnn.goal_encoder(raw_val_ctxs[vb_t])
                    vb_logits = torch.matmul(vb_goal_embs, embeddings_norm.T) / args.temperature
                    vb_targets = val_targets_t[vb_indices]
                    val_loss_total += F.cross_entropy(vb_logits, vb_targets).item() * len(vb_indices)

                    _, vb_top5 = torch.topk(vb_logits, k=min(5, num_nodes), dim=1)
                    for j, tgt in enumerate(vb_targets.tolist()):
                        if vb_top5[j, 0].item() == tgt:
                            val_correct1 += 1
                        if tgt in vb_top5[j].tolist():
                            val_correct5 += 1

                val_loss = val_loss_total / n_val
                val_top1 = val_correct1 / n_val
                val_top5 = val_correct5 / n_val

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
                output_path = _project_root / args.output
                output_path.parent.mkdir(parents=True, exist_ok=True)
                gnn.save(output_path)
                print(f"  → Saved best model (val top-1={best_val_acc:.1%})")

            if val_loss < best_val_loss and val_loss > 0:
                best_val_loss = val_loss

    # ---- Final ----
    print(f"\nTraining complete: {args.epochs} epochs")
    print(f"  Best val top-1: {best_val_acc:.1%}")
    print(f"  Best val loss:  {best_val_loss:.4f}")
    print(f"  Model saved to {args.output}")

    print("\nDone!")


if __name__ == "__main__":
    main()
