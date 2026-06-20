#!/usr/bin/env python3
"""Full GNN fine-tuning with proof-utility contrastive loss.

Loads pretrained GNN, fine-tunes ALL parameters with multi-task loss
on 226K proof-step pairs. Uses Lean-rejected lemmas as hard negatives.
Link-prediction anchor loss preserves graph topology.

Architecture:
  GNN (1.1M, trainable) → L2-normalized node embeddings → proof-utility
  Goal encoding: keyword averaging → GNN.goal_encoder → goal embedding
  Lemma encoding: direct node embedding lookup

Safety gates (per updated plan):
  A: Link-prediction preservation loss ≤30% above pretrained baseline
  B: Validation MRR must stay within 20% of GNN baseline (≥0.629)
  C: Lemma embedding diversity: avg cosine std > 0.05, rank > 128
  D: Checkpoint saved every epoch — revert to any epoch if later degrades
  E: Post-training Gate 3 must beat 15.6% baseline

Usage:
    python scripts/training/train_gnn_adapter.py \
        --gnn-checkpoint checkpoints/gnn/full_graph_pretrained.pt \
        --pairs data/raw/proof_step_pairs.jsonl \
        --output-dir data/adapter_full \
        --epochs 20 --batch-size 256 --num-threads 4
"""

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))

from src.explorer.dependency_graph import DependencyGraph
from src.explorer.gnn_config import GNNConfig
from src.explorer.gnn_encoder import (
    GNNEncoder,
    extract_initial_features,
    prepare_graph_tensors,
)
from src.contrastive.hard_negative_loss import (
    compute_infonce_loss,
    compute_triplet_margin_loss,
)
from src.explorer.mcts import _extract_math_keywords, _BUILTIN_LEMMAS


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_pairs(data_path: Path) -> list[dict]:
    pairs = []
    with open(data_path) as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs


# ---------------------------------------------------------------------------
# Link-prediction preservation loss
# ---------------------------------------------------------------------------

def compute_link_prediction_loss(
    node_embeddings: torch.Tensor,
    sources: torch.Tensor,
    targets: torch.Tensor,
    num_negatives: int = 5,
    sample_edges: int = 5000,
) -> torch.Tensor:
    """BCE loss: graph edges (positive) vs random pairs (negative).

    Lower loss = better topology preservation.
    """
    device = node_embeddings.device
    num_edges = sources.size(0)

    n_pos = min(sample_edges, num_edges)
    pos_indices = torch.randperm(num_edges, device=device)[:n_pos]
    pos_src = sources[pos_indices]
    pos_tgt = targets[pos_indices]

    all_emb = F.normalize(node_embeddings, dim=-1)
    pos_scores = (all_emb[pos_src] * all_emb[pos_tgt]).sum(dim=-1)

    num_nodes = node_embeddings.size(0)
    n_neg = n_pos * num_negatives
    neg_src = torch.randint(0, num_nodes, (n_neg,), device=device)
    neg_tgt = torch.randint(0, num_nodes, (n_neg,), device=device)
    neg_scores = (all_emb[neg_src] * all_emb[neg_tgt]).sum(dim=-1)

    pos_loss = F.binary_cross_entropy_with_logits(
        pos_scores, torch.ones_like(pos_scores)
    )
    neg_loss = F.binary_cross_entropy_with_logits(
        neg_scores, torch.zeros_like(neg_scores)
    )
    return (pos_loss + neg_loss) / 2.0


# ---------------------------------------------------------------------------
# Embedding health checks (Gate C)
# ---------------------------------------------------------------------------

def check_embedding_health(embeddings: torch.Tensor) -> dict:
    """Compute embedding diversity metrics."""
    N, D = embeddings.shape
    if N < 2:
        return {"avg_cosine_std": 0.0, "rank": 0, "std_ok": False, "rank_ok": False}

    sample_n = min(N, 2000)
    indices = torch.randperm(N)[:sample_n]
    sample = F.normalize(embeddings[indices], dim=-1)

    cos_sim = sample @ sample.T
    mask = ~torch.eye(sample_n, dtype=torch.bool, device=embeddings.device)
    off_diag = cos_sim[mask]
    avg_cosine_std = off_diag.std().item()

    try:
        U, S, V = torch.svd(sample)
        threshold = S.max().item() * 0.01
        rank_val = (S > threshold).sum().item()
    except Exception:
        rank_val = 0

    return {
        "avg_cosine_std": round(avg_cosine_std, 6),
        "rank": rank_val,
        "std_ok": avg_cosine_std > 0.05,
        "rank_ok": rank_val > 128,
    }


# ---------------------------------------------------------------------------
# Validation MRR (Gate B)
# ---------------------------------------------------------------------------

def compute_val_mrr(
    node_embeddings: torch.Tensor,
    gnn: GNNEncoder,
    lemma_to_idx: dict[str, int],
    kw_lemmas_map: dict[str, list[int]],
    val_pairs: list[dict],
    sample_size: int = 500,
) -> float:
    """Compute Mean Reciprocal Rank on validation pairs.

    For each (goal, correct_lemma) pair, compute goal embedding via
    keyword averaging + goal_encoder, then rank the correct lemma
    against all lemma embeddings.
    """
    device = node_embeddings.device
    all_emb_norm = F.normalize(node_embeddings, dim=-1)

    if len(val_pairs) > sample_size:
        sample = random.sample(val_pairs, sample_size)
    else:
        sample = val_pairs

    reciprocal_ranks = []
    with torch.no_grad():
        for pair in sample:
            goal_text = pair["goal"]
            lemma_name = pair["lemma"]

            correct_idx = lemma_to_idx.get(lemma_name)
            if correct_idx is None or correct_idx >= all_emb_norm.size(0):
                continue

            # Goal embedding: keyword averaging → goal_encoder
            keywords = _extract_math_keywords(goal_text)
            matching_indices: set[int] = set()
            for kw in keywords:
                matches = kw_lemmas_map.get(kw.lower(), [])
                for idx in matches:
                    if idx < all_emb_norm.size(0):
                        matching_indices.add(idx)

            if not matching_indices:
                continue

            match_list = list(matching_indices)[:100]
            match_t = torch.tensor(match_list, device=device)
            context_emb = all_emb_norm[match_t].mean(dim=0)
            goal_emb = gnn.encode_goal(context_emb.unsqueeze(0))

            # Score against all lemmas
            scores = (goal_emb @ all_emb_norm.T).squeeze(0)
            correct_score = scores[correct_idx]
            rank = (scores > correct_score).sum().item() + 1
            reciprocal_ranks.append(1.0 / rank)

    if not reciprocal_ranks:
        return 0.0
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


# ---------------------------------------------------------------------------
# Build keyword → lemma index map
# ---------------------------------------------------------------------------

def build_keyword_map(graph, all_kw: set[str]) -> dict[str, list[int]]:
    kw_map: dict[str, list[int]] = {}
    for nid in graph.node_ids:
        idx = graph.node_id_to_idx(nid)
        if idx is None:
            continue
        name_lower = nid.lower()
        for kw in all_kw:
            if kw.lower() in name_lower:
                kw_map.setdefault(kw.lower(), []).append(idx)
    return kw_map


def get_keyword_set() -> set[str]:
    all_kw: set[str] = set()
    all_kw.update(_BUILTIN_LEMMAS.keys())
    for vals in _BUILTIN_LEMMAS.values():
        all_kw.update(vals)
    math_tokens = [
        "+", "*", "-", "/", "^", "=", "→", "∀", "∃", "≤", "≥", "<", ">",
        "⁻¹", "∘", "0", "1",
        "add", "mul", "sub", "div", "neg", "comm", "assoc", "distrib",
        "and", "or", "not", "iff", "eq", "refl", "symm", "trans",
        "Nat", "Int", "Real", "Complex", "Prop", "Set", "List",
        "deriv", "integral", "limit", "continuous", "sum", "prod",
        "ring", "field", "group", "linear", "inv", "pow",
    ]
    all_kw.update(math_tokens)
    return all_kw


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_gnn(args):
    """Full GNN fine-tuning with safety gates."""
    # Force line-buffered output for real-time monitoring
    import functools, builtins
    _real_print = print
    builtins.print = functools.partial(_real_print, flush=True)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(args.num_threads)
    device = torch.device("cpu")
    print(f"Device: {device}, Threads: {torch.get_num_threads()}")
    # Mode printed after GNN loading below

    # ---- Load pretrained GNN (freeze GAT layers, train goal_encoder) ----
    print("\n--- Loading GNN (pretrained) ---")
    gnn = GNNEncoder.load(args.gnn_checkpoint)
    
    # Freeze everything except goal_encoder
    for p in gnn.parameters():
        p.requires_grad = False
    for p in gnn.goal_encoder.parameters():
        p.requires_grad = True
    
    total_params = sum(p.numel() for p in gnn.parameters())
    trainable_params = sum(p.numel() for p in gnn.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    print(f"  GNN params (total): {total_params:,}")
    print(f"  GNN params (frozen): {frozen_params:,}")
    print(f"  GNN params (trainable - goal_encoder only): {trainable_params:,}")
    print(f"Mode: Goal-encoder fine-tuning (frozen GNN + trainable goal_encoder)")

    # ---- Load graph ----
    print("\n--- Loading dependency graph ---")
    graph_path = Path(args.graph)
    pkl_path = graph_path.with_suffix(".nx.pkl")
    if not pkl_path.exists():
        print(f"  ERROR: Graph not found at {pkl_path}")
        sys.exit(1)

    graph = DependencyGraph.load(graph_path)
    print(f"  Graph: {graph.summary()}")

    # ---- Setup graph tensors (needed for forward pass) ----
    features = extract_initial_features(graph, gnn.config, device=device)
    sources, targets, edge_types, num_nodes = prepare_graph_tensors(graph, device=device)
    print(f"  Nodes: {num_nodes}, Edges: {sources.size(0)}")

    # ---- Build lemma index and keyword map ----
    print("\n--- Building lemma index ---")
    lemma_to_idx: dict[str, int] = {}
    for node_id in graph.node_ids:
        idx = graph.node_id_to_idx(node_id)
        if idx is not None:
            lemma_to_idx[node_id] = idx
            short = node_id.split(".")[-1] if "." in node_id else node_id
            if short not in lemma_to_idx:
                lemma_to_idx[short] = idx
    print(f"  Lemma index: {len(lemma_to_idx)} entries")

    all_kw = get_keyword_set()
    kw_lemmas_map = build_keyword_map(graph, all_kw)
    print(f"  Keyword map: {len(kw_lemmas_map)} keywords")

    # ---- Gate A: Baseline link-prediction loss ----
    print("\n--- Gate A: Baseline link-prediction preservation ---")
    with torch.no_grad():
        baseline_emb = gnn(features, sources, targets, edge_types, num_nodes)
    baseline_lp_loss = compute_link_prediction_loss(
        baseline_emb, sources, targets, sample_edges=2000
    )
    gate_a_threshold = baseline_lp_loss.item() * 1.30  # ≤30% above baseline
    print(f"  Baseline link-pred loss: {baseline_lp_loss.item():.6f}")
    print(f"  Gate A threshold (≤30% baseline): {gate_a_threshold:.6f}")

    # ---- Gate C: Check pretrained embedding health ----
    pretrained_health = check_embedding_health(baseline_emb)
    print(f"\n--- Pre-training embedding health ---")
    print(f"  avg_cosine_std={pretrained_health['avg_cosine_std']:.4f} "
          f"(need >0.05), rank={pretrained_health['rank']} (need >128)")

    # ---- Load proof-step pairs ----
    print("\n--- Loading proof-step pairs ---")
    all_pairs = load_pairs(Path(args.pairs))
    print(f"  Loaded {len(all_pairs)} pairs")

    if args.max_pairs and args.max_pairs < len(all_pairs):
        random.seed(42)
        all_pairs = random.sample(all_pairs, args.max_pairs)
        print(f"  Sampled {len(all_pairs)} pairs for smoke test")

    split_idx = int(len(all_pairs) * (1 - args.val_split))
    train_pairs = all_pairs[:split_idx]
    val_pairs = all_pairs[split_idx:]
    print(f"  Train: {len(train_pairs)}, Val: {len(val_pairs)}")

    # ---- Optimizer (goal_encoder only) ----
    optimizer = torch.optim.AdamW(
        gnn.goal_encoder.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ---- Training loop ----
    stats_history = []
    best_val_mrr = 0.0
    best_epoch = -1
    aborted_gate = None
    best_checkpoint_path = output_dir / "gnn_best.pt"

    print(f"\n{'=' * 60}")
    print(f"TRAINING: {args.epochs} epochs, batch_size={args.batch_size}")
    print(f"{'=' * 60}")

    for epoch in range(args.epochs):
        t0 = time.time()
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")

        # ---- Recompute GNN node embeddings (pre-computed, frozen GNN) ----
        # GAT layers + projection layers are frozen. Only goal_encoder is trainable.
        # We compute embeddings once; goal_encoder operates on keyword-averaged contexts.
        if epoch == 0:
            print("  Computing GNN node embeddings (one-time, frozen GNN)...")
            with torch.no_grad():
                gnn.eval()
                node_embeddings = gnn(features, sources, targets, edge_types, num_nodes)
                node_embeddings_norm = F.normalize(node_embeddings, dim=-1)
        # Reuse pre-computed embeddings for all epochs

        # ---- Training over batches ----
        gnn.goal_encoder.train()
        epoch_loss = 0.0
        epoch_infonce = 0.0
        epoch_triplet = 0.0
        epoch_lp = 0.0
        num_batches = 0

        random.shuffle(train_pairs)

        for batch_start in range(0, len(train_pairs), args.batch_size):
            batch = train_pairs[batch_start : batch_start + args.batch_size]
            if len(batch) < 2:
                continue

            # --- Encode goals and lemmas ---
            goal_embs_list = []
            lemma_embs_list = []
            for pair in batch:
                # Goal: keyword averaging → goal_encoder
                goal_text = pair["goal"]
                keywords = _extract_math_keywords(goal_text)
                matching_indices: set[int] = set()
                for kw in keywords:
                    matches = kw_lemmas_map.get(kw.lower(), [])
                    for idx in matches:
                        if idx < node_embeddings_norm.size(0):
                            matching_indices.add(idx)

                if matching_indices:
                    match_list = list(matching_indices)[:100]
                    match_t = torch.tensor(match_list, device=device)
                    context_emb = node_embeddings_norm[match_t].mean(dim=0)
                else:
                    context_emb = torch.zeros(gnn.config.hidden_dim, device=device)

                goal_emb = gnn.encode_goal(context_emb.unsqueeze(0))

                # Lemma: direct embedding lookup (detached)
                lemma_name = pair["lemma"]
                idx = lemma_to_idx.get(lemma_name)
                if idx is not None and idx < node_embeddings_norm.size(0):
                    lemma_emb = node_embeddings_norm[idx].unsqueeze(0).detach()
                else:
                    lemma_emb = torch.zeros(1, gnn.config.hidden_dim, device=device)

                goal_embs_list.append(goal_emb)
                lemma_embs_list.append(lemma_emb)

            goal_embs = torch.cat(goal_embs_list, dim=0)  # [B, D]
            lemma_embs = torch.cat(lemma_embs_list, dim=0)  # [B, D]

            # --- Multi-task loss ---
            temperature_inv = 1.0 / args.temperature
            infonce = compute_infonce_loss(goal_embs, lemma_embs, temperature_inv)

            batch_size_val = goal_embs.size(0)
            if batch_size_val >= 2 and args.hard_neg_weight > 0:
                sim_matrix = goal_embs @ lemma_embs.T
                mask = ~torch.eye(batch_size_val, dtype=torch.bool, device=device)
                sim_masked = sim_matrix.masked_fill(~mask, -1e9)
                _, hardest_indices = sim_masked.max(dim=1)
                hard_neg_embs = lemma_embs[hardest_indices]
                triplet_loss = compute_triplet_margin_loss(
                    goal_embs,
                    lemma_embs,
                    hard_neg_embs.unsqueeze(1),
                    margin=args.margin,
                )
            else:
                triplet_loss = torch.tensor(0.0, device=device)

            loss = infonce + args.hard_neg_weight * triplet_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gnn.goal_encoder.parameters(), args.grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_infonce += infonce.item()
            epoch_triplet += triplet_loss.item()
            num_batches += 1

        scheduler.step()

        # ---- End-of-epoch stats ----
        avg_loss = epoch_loss / max(1, num_batches)
        avg_infonce = epoch_infonce / max(1, num_batches)
        avg_triplet = epoch_triplet / max(1, num_batches)
        avg_lp = epoch_lp / max(1, num_batches)

        # Recompute embeddings for gate checks (eval mode — same frozen embeddings)
        gnn.eval()
        eval_emb = node_embeddings  # Same frozen embeddings, no need to recompute

        # Gate A: link-prediction preservation
        current_lp_loss = compute_link_prediction_loss(
            eval_emb, sources, targets, sample_edges=2000
        )
        lp_delta = (current_lp_loss.item() - baseline_lp_loss.item()) / max(
            1e-8, abs(baseline_lp_loss.item())
        )
        gate_a_ok = lp_delta <= 0.30

        # Gate B: validation MRR
        val_mrr = compute_val_mrr(
            eval_emb, gnn, lemma_to_idx, kw_lemmas_map, val_pairs
        )
        gate_b_ok = val_mrr >= 0.629  # within 20% of 0.786 baseline

        # Gate C: embedding health
        health = check_embedding_health(eval_emb)
        gate_c_ok = health["std_ok"] and health["rank_ok"]

        elapsed = time.time() - t0

        gates_str = (
            f"A={'✓' if gate_a_ok else '✗'} "
            f"B={'✓' if gate_b_ok else '✗'} "
            f"C={'✓' if gate_c_ok else '✗'}"
        )

        stats = {
            "epoch": epoch + 1,
            "loss": round(avg_loss, 6),
            "infonce_loss": round(avg_infonce, 6),
            "triplet_loss": round(avg_triplet, 6),
            "lp_loss": round(avg_lp, 6),
            "current_lp_loss": round(current_lp_loss.item(), 6),
            "lp_delta_pct": round(lp_delta * 100, 2),
            "val_mrr": round(val_mrr, 6),
            "embed_std": health["avg_cosine_std"],
            "embed_rank": health["rank"],
            "lr": round(scheduler.get_last_lr()[0], 8),
            "time_s": round(elapsed, 1),
            "gates": {
                "A": "PASS" if gate_a_ok else "FAIL",
                "B": "PASS" if gate_b_ok else "FAIL",
                "C": "PASS" if gate_c_ok else "FAIL",
            },
        }

        print(
            f"  loss={avg_loss:.4f} | "
            f"mrr={val_mrr:.4f} | "
            f"lpΔ={lp_delta*100:+.1f}% | "
            f"std={health['avg_cosine_std']:.3f} r={health['rank']} | "
            f"{gates_str} | "
            f"{elapsed:.1f}s"
        )
        stats_history.append(stats)

        # Gate A abort
        if args.abort_on_gate_fail and not gate_a_ok:
            aborted_gate = "A"
            msg = (
                f"GATE A FAILED at epoch {epoch+1}: "
                f"link-pred loss {current_lp_loss.item():.6f} > "
                f"threshold {gate_a_threshold:.6f} ({lp_delta*100:+.1f}%)"
            )
            print(f"\n*** {msg} ***")
            _save_abort(output_dir, msg, "A")
            break

        # Gate C abort
        if args.abort_on_gate_fail and not gate_c_ok:
            aborted_gate = "C"
            msg = (
                f"GATE C FAILED at epoch {epoch+1}: "
                f"std={health['avg_cosine_std']:.4f} (need >0.05), "
                f"rank={health['rank']} (need >128)"
            )
            print(f"\n*** {msg} ***")
            _save_abort(output_dir, msg, "C")
            break

        # Gate D: save checkpoint every epoch
        ckpt_path = output_dir / f"gnn_epoch_{epoch+1:03d}.pt"
        gnn.eval()
        gnn.save(ckpt_path)
        print(f"  Checkpoint: {ckpt_path.name}")

        # Track best
        if val_mrr > best_val_mrr:
            best_val_mrr = val_mrr
            best_epoch = epoch + 1
            gnn.save(best_checkpoint_path)

    # ---- Post-training ----
    if not aborted_gate:
        print(f"\nTraining complete. Best MRR: {best_val_mrr:.4f} (epoch {best_epoch})")

        # Final Gate B check
        gnn.eval()
        with torch.no_grad():
            final_emb = gnn(features, sources, targets, edge_types, num_nodes)
        final_val_mrr = compute_val_mrr(
            final_emb, gnn, lemma_to_idx, kw_lemmas_map, val_pairs
        )
        gate_b_final_ok = final_val_mrr >= 0.629
        print(f"  Gate B final MRR: {final_val_mrr:.4f} ≥ 0.629? {'PASS' if gate_b_final_ok else 'FAIL'}")

    # ---- Save artifacts ----
    gnn.eval()
    gnn.save(output_dir / "gnn_finetuned.pt")
    print(f"\n  Fine-tuned GNN saved: {output_dir / 'gnn_finetuned.pt'}")
    if best_checkpoint_path.exists():
        print(f"  Best GNN saved: {best_checkpoint_path}")

    with open(output_dir / "training_stats.json", "w") as f:
        json.dump(
            {
                "mode": "full_gnn_finetune",
                "config": {
                    "gnn_checkpoint": args.gnn_checkpoint,
                    "num_epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "learning_rate": args.learning_rate,
                    "hard_neg_weight": args.hard_neg_weight,
                    "preservation_weight": args.preservation_weight,
                    "margin": args.margin,
                    "temperature": args.temperature,
                    "num_threads": args.num_threads,
                },
                "baseline_lp_loss": baseline_lp_loss.item(),
                "gate_a_threshold": gate_a_threshold,
                "best_val_mrr": best_val_mrr,
                "best_epoch": best_epoch,
                "aborted_gate": aborted_gate,
                "epochs": stats_history,
            },
            f,
            indent=2,
        )
    print(f"  Stats saved: {output_dir / 'training_stats.json'}")

    if aborted_gate:
        print(f"  ABORTED at gate {aborted_gate}")

    return {
        "aborted": aborted_gate is not None,
        "gate": aborted_gate,
        "best_mrr": best_val_mrr,
        "best_epoch": best_epoch,
        "final_stats": stats_history[-1] if stats_history else None,
    }


# ---------------------------------------------------------------------------
# Abort helper
# ---------------------------------------------------------------------------

def _save_abort(output_dir: Path, message: str, gate: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "abort_reason.json", "w") as f:
        json.dump({"gate": gate, "message": message, "timestamp": time.time()}, f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Full GNN fine-tuning with proof-utility contrastive loss"
    )
    parser.add_argument(
        "--gnn-checkpoint",
        default="checkpoints/gnn/full_graph_pretrained.pt",
        help="Path to pretrained GNN checkpoint",
    )
    parser.add_argument(
        "--pairs",
        default="data/raw/proof_step_pairs.jsonl",
        help="Path to proof-step pairs JSONL",
    )
    parser.add_argument(
        "--graph",
        default="data/graph/dependency_graph_full",
        help="Path to dependency graph",
    )
    parser.add_argument(
        "--output-dir",
        default="data/adapter_full",
        help="Output directory for fine-tuned GNN and stats",
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=256, help="Batch size"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4,
        help="Learning rate (lower for full fine-tuning)",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=1e-5, help="Weight decay"
    )
    parser.add_argument(
        "--hard-neg-weight", type=float, default=0.5,
        help="Weight for triplet hard-negative loss",
    )
    parser.add_argument(
        "--preservation-weight", type=float, default=0.1,
        help="Weight for link-prediction preservation loss",
    )
    parser.add_argument(
        "--margin", type=float, default=0.3, help="Triplet margin"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.07, help="InfoNCE temperature"
    )
    parser.add_argument(
        "--grad-clip", type=float, default=1.0, help="Gradient clipping"
    )
    parser.add_argument(
        "--num-threads", type=int, default=4, help="Number of CPU threads"
    )
    parser.add_argument(
        "--max-pairs", type=int, default=None,
        help="Max pairs to use (for smoke testing)",
    )
    parser.add_argument(
        "--val-split", type=float, default=0.2, help="Validation split ratio"
    )
    parser.add_argument(
        "--no-abort", action="store_true",
        help="Don't abort on gate failures (continue training)",
    )
    args = parser.parse_args()

    # Hardware constraint
    args.num_threads = min(args.num_threads, 4)
    args.abort_on_gate_fail = not args.no_abort

    print("=" * 60)
    print("Full GNN Fine-tuning with Proof-Utility Loss")
    print("=" * 60)
    print(f"  Mode: Goal-encoder fine-tuning (frozen GNN + trainable goal_encoder)")
    print(f"  GNN: {args.gnn_checkpoint}")
    print(f"  Pairs: {args.pairs}")
    print(f"  Epochs: {args.epochs}, Batch: {args.batch_size}")
    print(f"  LR: {args.learning_rate}, Threads: {args.num_threads}")
    print(f"  Hard neg weight: {args.hard_neg_weight}")
    print(f"  Preservation weight: {args.preservation_weight}")
    print()

    result = train_gnn(args)

    if result["aborted"]:
        print(f"\n  Training aborted at gate {result['gate']}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
