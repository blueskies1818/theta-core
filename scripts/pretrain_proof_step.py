#!/usr/bin/env python3
"""Pretrain GNN + GoalEncoder on proof-step prediction.

Trains the GNN and GoalEncoder jointly to predict which lemma proves a given
goal, using (goal, lemma) pairs extracted from real mathlib4 proofs.

This replaces link-prediction pretraining with proof-relevant knowledge:
the GNN learns that add_comm closes a+b=b+a, mul_comm closes a*b=b*a, etc.

The GoalEncoder learns to project structurally-matched lemma embeddings into a
sharper representation that identifies the correct lemma. Normalized text
matching replaces the old keyword-averaging approach.

Loss: multi-class cross-entropy over ALL lemma candidates in the graph.
The correct lemma should have the highest cosine similarity to the goal
embedding.

Usage:
    python scripts/pretrain_proof_step.py --epochs 200
"""

import argparse, json, random, sys, time
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from src.explorer.dependency_graph import DependencyGraph
from src.explorer.gnn_config import GNNConfig
from src.explorer.gnn_encoder import GNNEncoder, prepare_graph_tensors, extract_initial_features
from src.explorer.mcts import _extract_math_keywords
from scripts.eval_gnn_prover import (
    normalize_expression, extract_conclusion, build_lemma_norm_index,
    tokenize_expression,
)


def build_lemma_index(graph: DependencyGraph) -> dict[str, int]:
    """Map lemma names to their integer indices in the graph."""
    index = {}
    for node_id in graph.node_ids:
        short_name = node_id.split(".")[-1] if "." in node_id else node_id
        index[node_id] = graph.node_id_to_idx(node_id)
        if short_name not in index:
            index[short_name] = graph.node_id_to_idx(node_id)
    return index


def precompute_goal_contexts(
    goals: list[str],
    lemma_to_idx: dict[str, int],
    idx_to_norm: dict[int, str],
) -> list[list[int]]:
    """Precompute matching lemma indices for each training goal.

    Uses inverted indices for O(1) lookups instead of O(N) scans.
    """
    import re
    from collections import defaultdict

    # ---- Build inverted indices (one-time) ----
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

    # ---- Per-goal matching (fast O(1) lookups) ----
    contexts = []
    for goal_text in goals:
        goal_norm = normalize_expression(goal_text)

        # Exact match via dict lookup
        exact_matches = list(norm_to_indices.get(goal_norm, []))

        # Reflexivity: add rfl lemmas
        if not exact_matches:
            if "=" in goal_norm and "↔" not in goal_norm and "→" not in goal_norm and "≠" not in goal_norm:
                sides = goal_norm.split("=", 1)
                if len(sides) == 2 and sides[0].strip() == sides[1].strip():
                    exact_matches.extend(norm_to_indices.get(rfl_norm, []))

        # Iff matching: goal matches either direction
        exact_matches.extend(iff_lhs_to_indices.get(goal_norm, []))
        exact_matches.extend(iff_rhs_to_indices.get(goal_norm, []))

        # Implication matching: goal matches conclusion
        exact_matches.extend(imp_rhs_to_indices.get(goal_norm, []))

        # Power-stripping fallback
        if not exact_matches:
            goal_stripped = re.sub(r'\s*\^\s*\d+', '', goal_norm)
            exact_matches.extend(stripped_to_indices.get(goal_stripped, []))

        # Deduplicate while preserving order
        seen = set()
        exact_dedup = []
        for idx in exact_matches:
            if idx not in seen:
                seen.add(idx)
                exact_dedup.append(idx)

        contexts.append(exact_dedup[:100])

    return contexts


def embed_goal(
    goal_text: str,
    node_emb_norm: torch.Tensor,
    lemma_to_idx: dict[str, int],
    gnn: GNNEncoder | None = None,
    precomputed_context: list[int] | None = None,
) -> torch.Tensor:
    """Create a goal embedding using precomputed context or keyword fallback.

    node_emb_norm should already be L2-normalized.
    """
    device = node_emb_norm.device

    if precomputed_context:
        indices_t = torch.tensor(precomputed_context, device=device)
        context_emb = node_emb_norm[indices_t].mean(dim=0)
    else:
        # Slow keyword fallback (only used when precomputed context unavailable)
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


def main():
    parser = argparse.ArgumentParser(description="Pretrain GNN on proof-step prediction")
    parser.add_argument("--data", default="data/raw/proof_step_pairs.jsonl")
    parser.add_argument("--graph", default="data/graph/dependency_graph")
    parser.add_argument("--domain", default="Algebra")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", default="checkpoints/gnn/proof_step_pretrained.pt")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--finetune-gnn", action="store_true",
                        help="Fine-tune the full GNN (not just GoalEncoder)")
    parser.add_argument("--gnn-lr", type=float, default=1e-4,
                        help="Learning rate for GNN parameters when fine-tuning")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else (
        torch.device("xpu:0") if torch.xpu.is_available() else torch.device("cpu")
    )
    print(f"Device: {device}")

    # ---- Load graph ----
    graph_path = _project_root / args.graph
    graph = DependencyGraph.load(graph_path)
    if args.domain:
        graph = graph.domain_subgraph(args.domain)
    print(f"Graph: {graph.num_nodes} nodes ({args.domain})")

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
    if len(pairs) < 100:
        print("Error: too few matching pairs. Check lemma name resolution.")
        sys.exit(1)

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
    goal_encoder_params = sum(p.numel() for p in gnn.goal_encoder.parameters())
    print(f"GNN: {total_params:,} total params ({goal_encoder_params:,} in goal encoder)")

    # ---- Pre-compute graph tensors (on GPU for speed) ----
    features = extract_initial_features(graph, config).to(device)
    sources, targets, edge_types, num_nodes = prepare_graph_tensors(graph)
    sources = sources.to(device)
    targets = targets.to(device)
    edge_types = edge_types.to(device)

    # ---- Setup optimizer ----
    finetune_gnn = args.finetune_gnn
    if finetune_gnn:
        # Fine-tune GNN + GoalEncoder jointly with separate learning rates
        optimizer = torch.optim.AdamW([
            {"params": gnn.goal_encoder.parameters(), "lr": args.lr},
            {"params": [p for n, p in gnn.named_parameters()
                        if "goal_encoder" not in n], "lr": args.gnn_lr},
        ], lr=args.lr, weight_decay=1e-5)
        gnn.train()
        print(f"Fine-tuning: GNN (lr={args.gnn_lr}) + GoalEncoder (lr={args.lr})")
    else:
        optimizer = torch.optim.AdamW(gnn.goal_encoder.parameters(), lr=args.lr, weight_decay=1e-5)
        gnn.eval()
        print("Training GoalEncoder only (GNN frozen)")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ---- Prepare training data ----
    goals = [p["goal"] for p in pairs]
    target_indices = [lemma_to_idx[p["lemma"]] for p in pairs]

    # ---- Precompute goal contexts (one-time, avoids 105M comparisons/epoch) ----
    print("Precomputing goal contexts...", end=" ", flush=True)
    t_pre = time.time()
    precomputed_contexts = precompute_goal_contexts(goals, lemma_to_idx, idx_to_norm)
    context_hits = sum(1 for c in precomputed_contexts if c)
    print(f"done ({time.time() - t_pre:.1f}s). {context_hits}/{len(goals)} goals have context matches.", flush=True)

    print(f"\nTraining: {args.epochs} epochs, batch_size={args.batch_size}, "
          f"temperature={args.temperature}, finetune_gnn={finetune_gnn}")

    # ---- Compute initial GNN embeddings (frozen for non-finetune mode) ----
    print("Computing initial GNN embeddings...", end=" ", flush=True)
    t_gnn = time.time()
    with torch.no_grad():
        embeddings = gnn(features, sources, targets, edge_types, num_nodes)
    if not finetune_gnn:
        embeddings = embeddings.detach()
    embeddings_norm = F.normalize(embeddings, dim=-1)
    print(f"done ({time.time() - t_gnn:.1f}s).", flush=True)

    # ---- Precompute raw context embeddings (one-time, since GNN is frozen) ----
    # These are the mean embeddings of matching lemmas WITHOUT the GoalEncoder.
    # They stay constant across epochs; only the GoalEncoder weights change.
    print("Precomputing raw context embeddings...", end=" ", flush=True)
    t_ctx = time.time()
    raw_contexts = torch.zeros(len(pairs), config.hidden_dim, device=device)
    for i, ctx_indices in enumerate(precomputed_contexts):
        if ctx_indices:
            idx_t = torch.tensor(ctx_indices, device=device)
            raw_contexts[i] = embeddings_norm[idx_t].mean(dim=0)
    print(f"done ({time.time() - t_ctx:.1f}s).", flush=True)

    best_acc = 0.0
    best_loss = float("inf")

    for epoch in range(args.epochs):
        t0 = time.time()
        if finetune_gnn:
            gnn.train()
        else:
            gnn.goal_encoder.train()

        # ---- Recompute GNN embeddings if fine-tuning ----
        if finetune_gnn:
            embeddings = gnn(features, sources, targets, edge_types, num_nodes)
            embeddings_norm = F.normalize(embeddings, dim=-1)
            # Build fresh raw contexts from new embeddings
            raw_contexts = torch.zeros(len(pairs), config.hidden_dim, device=device)
            for i, ctx_indices in enumerate(precomputed_contexts):
                if ctx_indices:
                    idx_t = torch.tensor(ctx_indices, device=device)
                    raw_contexts[i] = embeddings_norm[idx_t].mean(dim=0)

        target_tensor_all = torch.tensor(target_indices, device=device)

        epoch_loss = 0.0
        num_batches = 0
        correct_top1 = 0
        correct_top5 = 0
        total_eval = 0

        # ---- Full forward/backward (avoids retain_graph issues with GNN) ----
        if finetune_gnn:
            all_goal_embs = gnn.goal_encoder(raw_contexts)  # [6229, D]
            logits = torch.matmul(all_goal_embs, embeddings_norm.T) / args.temperature
            loss = F.cross_entropy(logits, target_tensor_all)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gnn.parameters(), 1.0)
            optimizer.step()

            epoch_loss = loss.item()
            num_batches = 1

            with torch.no_grad():
                _, top5 = torch.topk(logits.detach(), k=min(5, num_nodes), dim=1)
                for j, tgt in enumerate(target_indices):
                    correct_top1 += 1 if top5[j, 0].item() == tgt else 0
                    correct_top5 += 1 if tgt in top5[j].tolist() else 0
                total_eval = len(target_indices)

        else:
            # ---- Frozen GNN: batched training (GNN graph not in backward) ----
            indices = list(range(len(pairs)))
            random.shuffle(indices)

            for batch_start in range(0, len(indices), args.batch_size):
                batch_indices = indices[batch_start:batch_start + args.batch_size]
                if len(batch_indices) < 2:
                    continue

                batch_idx_t = torch.tensor(batch_indices, device=device)
                batch_context = raw_contexts[batch_idx_t]  # [B, D]
                batch_goal_embs = gnn.goal_encoder(batch_context)  # [B, D]
                batch_targets = target_tensor_all[batch_idx_t]  # [B]

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
        top1_acc = correct_top1 / max(1, total_eval)
        top5_acc = correct_top5 / max(1, total_eval)
        elapsed = time.time() - t0

        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"Epoch {epoch:3d}/{args.epochs} | Loss: {avg_loss:.4f} | "
                  f"Top-1: {top1_acc:.1%} | Top-5: {top5_acc:.1%} | "
                  f"LR: {scheduler.get_last_lr()[0]:.2e} | Time: {elapsed:.1f}s", flush=True)

            if top1_acc > best_acc:
                best_acc = top1_acc
                gnn.save(_project_root / args.output)
                print(f"  → Saved best model (top-1={best_acc:.1%})")

            if avg_loss < best_loss:
                best_loss = avg_loss

    # ---- Final evaluation ----
    print(f"\nTraining complete: {args.epochs} epochs")
    print(f"  Best top-1 accuracy: {best_acc:.1%}")
    print(f"  Best loss: {best_loss:.4f}")
    print(f"  Model saved to {args.output}")

    # Report per-lemma accuracy for top-20 most common lemmas
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
                    goals[idx], eval_emb_norm, lemma_to_idx, gnn,
                    precomputed_context=precomputed_contexts[idx],
                )
                logits = goal_emb @ eval_emb_norm.T / args.temperature
                pred = logits.argmax().item()
                if pred == target_indices[idx]:
                    correct += 1
            acc = correct / min(100, len(pair_indices))
            print(f"  {lemma_name:40s} freq={freq:5d} acc={acc:.1%}")


if __name__ == "__main__":
    main()
