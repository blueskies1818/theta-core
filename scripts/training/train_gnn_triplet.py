#!/usr/bin/env python3
"""Full GNN fine-tuning with InfoNCE + triplet margin loss (V2).

Loads pretrained GNN (1.1M params), fine-tunes ALL parameters using
InfoNCE + triplet margin loss on proof-step pairs + link-prediction anchor.

Combines InfoNCE (in-batch softmax ranking — global pull) with triplet
margin loss (hardest in-batch negative — targeted push). Link-prediction
preservation anchors the embedding space.

Architecture:
  GNN (1.1M, ALL trainable) → L2-normalized node embeddings
  Goal encoding: keyword averaging → goal_encoder → goal embedding
  Lemma encoding: direct node embedding lookup
  Loss = InfoNCE + α * triplet_margin + λ * link_prediction

Key differences from frozen-backbone: ALL 1.1M GNN params trainable.
Node embeddings recomputed every epoch because GNN weights change.

Safety gates:
  A: Link-prediction preservation loss ≤30% above pretrained baseline
  B: Validation MRR ≥ 0.60
  C: Lemma embedding diversity: avg cosine std > 0.05, rank > 128
  D: Checkpoint saved every epoch — revert to any epoch
  E: Post-training Gate 3 must beat 15.6% baseline (eval separately)

Usage:
    python scripts/training/train_gnn_triplet.py \
        --gnn-checkpoint checkpoints/gnn/full_graph_pretrained.pt \
        --pairs data/raw/proof_step_pairs.jsonl \
        --output-dir data/gnn_triplet_full \
        --epochs 20 --batch-size 256 --num-threads 4 --no-abort
"""

import argparse
import json
import math
import random
import sys
import time
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
from src.explorer.mcts import _extract_math_keywords, _BUILTIN_LEMMAS
from src.contrastive.hard_negative_loss import compute_infonce_loss


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_pairs(data_path: Path, pre_extract_keywords: bool = True) -> list[dict]:
    """Load proof-step pairs from JSONL with pre-extracted keywords."""
    pairs = []
    with open(data_path) as f:
        for line in f:
            pair = json.loads(line)
            if pre_extract_keywords:
                pair['_keywords'] = _extract_math_keywords(pair['goal'])
            pairs.append(pair)
    return pairs


# ---------------------------------------------------------------------------
# Triplet margin loss (hardest in-batch negative)
# ---------------------------------------------------------------------------

def compute_triplet_loss(
    goal_embs: torch.Tensor,
    lemma_embs: torch.Tensor,
    margin: float = 0.3,
) -> torch.Tensor:
    """Triplet margin loss using hardest in-batch negative.

    For each goal i, positive = lemma i, negative = lemma j≠i with highest
    cosine similarity to goal i.
    """
    B = goal_embs.size(0)
    if B < 2:
        return torch.tensor(0.0, device=goal_embs.device)

    sim = goal_embs @ lemma_embs.T  # [B, B]
    pos_scores = sim.diag()  # [B]
    mask = ~torch.eye(B, dtype=torch.bool, device=sim.device)
    sim_masked = sim.masked_fill(~mask, -1e9)
    neg_scores, _ = sim_masked.max(dim=1)  # [B]
    losses = F.relu(margin - pos_scores + neg_scores)
    return losses.mean()


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
    """BCE loss: graph edges (positive) vs random pairs (negative)."""
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
    """Compute Mean Reciprocal Rank on validation pairs."""
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

            keywords = pair.get('_keywords') or _extract_math_keywords(goal_text)
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
# Abort helper
# ---------------------------------------------------------------------------

def _save_abort(output_dir: Path, message: str, gate: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "abort_reason.json", "w") as f:
        json.dump({"gate": gate, "message": message, "timestamp": time.time()}, f, indent=2)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_gnn_triplet(args):
    """Full GNN fine-tuning with triplet-only loss.

    V2: ALL 1.1M GNN params trainable. Embeddings recomputed every epoch.
    """
    import functools, builtins
    _real_print = print
    builtins.print = functools.partial(_real_print, flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(args.num_threads)
    device = torch.device("cpu")
    print(f"Device: {device}, Threads: {torch.get_num_threads()}")

    # ---- Load pretrained GNN (ALL params trainable) ----
    print("\n--- Loading GNN (pretrained) ---")
    gnn = GNNEncoder.load(args.gnn_checkpoint)
    trainable_params = sum(p.numel() for p in gnn.parameters())
    print(f"  GNN params (ALL trainable): {trainable_params:,}")
    print(f"  Mode: FULL GNN FINE-TUNING, triplet-only loss (V2)")
    hidden_dim = gnn.config.hidden_dim
    print(f"  hidden_dim={hidden_dim}, layers={gnn.config.num_layers}, "
          f"heads={gnn.config.num_heads}")

    # ---- Load graph ----
    print("\n--- Loading dependency graph ---")
    graph_path = Path(args.graph)
    pkl_path = graph_path.with_suffix(".nx.pkl")
    if not pkl_path.exists():
        print(f"  ERROR: Graph not found at {pkl_path}")
        sys.exit(1)
    graph = DependencyGraph.load(graph_path)
    print(f"  Graph: {graph.summary()}")

    # ---- Setup graph tensors ----
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
    gnn.eval()
    with torch.no_grad():
        baseline_emb = gnn(features, sources, targets, edge_types, num_nodes)
    baseline_lp_loss = compute_link_prediction_loss(baseline_emb, sources, targets, sample_edges=2000)
    gate_a_threshold = baseline_lp_loss.item() * 1.30
    print(f"  Baseline link-pred loss: {baseline_lp_loss.item():.6f}")
    print(f"  Gate A threshold (≤30% above baseline): {gate_a_threshold:.6f}")

    # ---- Pre-training embedding health ----
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

    # ---- Optimizer (ALL params) ----
    optimizer = torch.optim.AdamW(gnn.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ---- Training loop ----
    stats_history = []
    best_val_mrr = 0.0
    best_epoch = -1
    aborted_gate = None
    best_checkpoint_path = output_dir / "gnn_best.pt"

    print(f"\n{'=' * 60}")
    print(f"TRAINING (V2 FULL FINE-TUNE): {args.epochs} epochs, batch_size={args.batch_size}")
    print(f"  Loss: TRIPLET ({args.margin}) + InfoNCE (T={args.temperature}) + λ={args.preservation_weight} * link-prediction")
    print(f"  Hard neg weight: {args.hard_neg_weight}")
    print(f"  ALL 1.1M params trainable — embeddings recomputed every epoch")
    print(f"  InfoNCE + triplet margin — best of both worlds")
    print(f"{'=' * 60}")

    for epoch in range(args.epochs):
        t0 = time.time()
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")

        # ---- Recompute GNN node embeddings (weights changed!) ----
        gnn.eval()
        with torch.no_grad():
            node_embeddings = gnn(features, sources, targets, edge_types, num_nodes)
            node_embeddings_norm = F.normalize(node_embeddings, dim=-1)

        # ---- Training over batches ----
        gnn.train()
        epoch_loss = 0.0
        epoch_triplet = 0.0
        epoch_infonce = 0.0
        epoch_lp = 0.0
        num_batches = 0
        random.shuffle(train_pairs)

        for batch_start in range(0, len(train_pairs), args.batch_size):
            batch = train_pairs[batch_start : batch_start + args.batch_size]
            if len(batch) < 2:
                continue

            goal_embs_list = []
            lemma_embs_list = []
            for pair in batch:
                keywords = pair.get('_keywords') or _extract_math_keywords(pair['goal'])
                matching_indices: list[int] = []
                seen: set[int] = set()
                for kw in keywords:
                    matches = kw_lemmas_map.get(kw.lower(), [])
                    for idx in matches:
                        if idx < node_embeddings_norm.size(0) and idx not in seen:
                            matching_indices.append(idx)
                            seen.add(idx)
                            if len(matching_indices) >= 100:
                                break
                    if len(matching_indices) >= 100:
                        break

                if matching_indices:
                    match_t = torch.tensor(matching_indices, device=device)
                    context_emb = node_embeddings_norm[match_t].mean(dim=0)
                else:
                    context_emb = torch.zeros(hidden_dim, device=device)

                goal_emb = gnn.encode_goal(context_emb.unsqueeze(0))

                lemma_name = pair["lemma"]
                idx = lemma_to_idx.get(lemma_name)
                if idx is not None and idx < node_embeddings_norm.size(0):
                    lemma_emb = node_embeddings_norm[idx].unsqueeze(0).detach()
                else:
                    lemma_emb = torch.zeros(1, hidden_dim, device=device)

                goal_embs_list.append(goal_emb)
                lemma_embs_list.append(lemma_emb)

            goal_embs = torch.cat(goal_embs_list, dim=0)
            lemma_embs = torch.cat(lemma_embs_list, dim=0)

            # --- Triplet margin loss (hardest in-batch negative) ---
            goal_norm = F.normalize(goal_embs, dim=-1)
            lemma_norm = F.normalize(lemma_embs, dim=-1)

            triplet_loss = compute_triplet_loss(
                goal_norm, lemma_norm, margin=args.margin,
            )

            # --- InfoNCE loss (in-batch softmax ranking) ---
            temperature_inv = 1.0 / args.temperature
            infonce_loss = compute_infonce_loss(goal_norm, lemma_norm, temperature_inv)

            # --- Link-prediction preservation loss ---
            lp_loss = (compute_link_prediction_loss(node_embeddings, sources, targets, sample_edges=500)
                       if args.preservation_weight > 0 else torch.tensor(0.0, device=device))

            loss = infonce_loss + args.hard_neg_weight * triplet_loss + args.preservation_weight * lp_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gnn.parameters(), args.grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_triplet += triplet_loss.item()
            epoch_infonce += infonce_loss.item()
            epoch_lp += lp_loss.item()
            num_batches += 1

        scheduler.step()

        # ---- End-of-epoch stats ----
        avg_loss = epoch_loss / max(1, num_batches)
        avg_triplet = epoch_triplet / max(1, num_batches)
        avg_infonce = epoch_infonce / max(1, num_batches)
        avg_lp = epoch_lp / max(1, num_batches)

        gnn.eval()
        with torch.no_grad():
            eval_emb = gnn(features, sources, targets, edge_types, num_nodes)

        current_lp_loss = compute_link_prediction_loss(eval_emb, sources, targets, sample_edges=2000)
        lp_delta = (current_lp_loss.item() - baseline_lp_loss.item()) / max(1e-8, abs(baseline_lp_loss.item()))
        gate_a_ok = lp_delta <= 0.30

        val_mrr = compute_val_mrr(eval_emb, gnn, lemma_to_idx, kw_lemmas_map, val_pairs)
        gate_b_ok = val_mrr >= args.gate_b_mrr_threshold

        health = check_embedding_health(eval_emb)
        gate_c_ok = health["std_ok"] and health["rank_ok"]

        elapsed = time.time() - t0

        gates_str = (f"A={'✓' if gate_a_ok else '✗'} "
                      f"B={'✓' if gate_b_ok else '✗'} "
                      f"C={'✓' if gate_c_ok else '✗'}")

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
            "gates": {"A": "PASS" if gate_a_ok else "FAIL",
                       "B": "PASS" if gate_b_ok else "FAIL",
                       "C": "PASS" if gate_c_ok else "FAIL"},
        }

        print(f"  loss={avg_loss:.4f} | infoNCE={avg_infonce:.4f} | triplet={avg_triplet:.4f} | "
              f"mrr={val_mrr:.4f} | lpΔ={lp_delta*100:+.1f}% | "
              f"std={health['avg_cosine_std']:.3f} r={health['rank']} | "
              f"{gates_str} | {elapsed:.1f}s")
        stats_history.append(stats)

        if args.abort_on_gate_fail and not gate_a_ok:
            aborted_gate = "A"
            msg = (f"GATE A FAILED at epoch {epoch+1}: "
                   f"link-pred loss {current_lp_loss.item():.6f} > "
                   f"threshold {gate_a_threshold:.6f} ({lp_delta*100:+.1f}%)")
            print(f"\n*** {msg} ***")
            _save_abort(output_dir, msg, "A")
            break

        if args.abort_on_gate_fail and not gate_b_ok:
            aborted_gate = "B"
            msg = (f"GATE B FAILED at epoch {epoch+1}: "
                   f"val MRR {val_mrr:.4f} < {args.gate_b_mrr_threshold}")
            print(f"\n*** {msg} ***")
            _save_abort(output_dir, msg, "B")
            break

        if args.abort_on_gate_fail and not gate_c_ok:
            aborted_gate = "C"
            msg = (f"GATE C FAILED at epoch {epoch+1}: "
                   f"std={health['avg_cosine_std']:.4f} (need >0.05), "
                   f"rank={health['rank']} (need >128)")
            print(f"\n*** {msg} ***")
            _save_abort(output_dir, msg, "C")
            break

        ckpt_path = output_dir / f"gnn_epoch_{epoch+1:03d}.pt"
        gnn.save(ckpt_path)
        print(f"  Checkpoint: {ckpt_path.name}")

        if val_mrr > best_val_mrr:
            best_val_mrr = val_mrr
            best_epoch = epoch + 1
            gnn.save(best_checkpoint_path)

    if not aborted_gate:
        print(f"\nTraining complete. Best MRR: {best_val_mrr:.4f} (epoch {best_epoch})")
        gnn.eval()
        with torch.no_grad():
            final_emb = gnn(features, sources, targets, edge_types, num_nodes)
        final_val_mrr = compute_val_mrr(final_emb, gnn, lemma_to_idx, kw_lemmas_map, val_pairs)
        gate_b_final_ok = final_val_mrr >= args.gate_b_mrr_threshold
        print(f"  Gate B final MRR: {final_val_mrr:.4f} ≥ {args.gate_b_mrr_threshold}? "
              f"{'PASS' if gate_b_final_ok else 'FAIL'}")

    gnn.eval()
    gnn.save(output_dir / "gnn_finetuned.pt")
    print(f"\n  Fine-tuned GNN saved: {output_dir / 'gnn_finetuned.pt'}")
    if best_checkpoint_path.exists():
        print(f"  Best GNN saved: {best_checkpoint_path}")

    with open(output_dir / "training_stats.json", "w") as f:
        json.dump({
            "mode": "full_gnn_finetune_infonce_triplet_v2",
            "config": {
                "gnn_checkpoint": args.gnn_checkpoint,
                "num_epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "preservation_weight": args.preservation_weight,
                "margin": args.margin,
                "num_threads": args.num_threads,
            },
            "baseline_lp_loss": baseline_lp_loss.item(),
            "gate_a_threshold": gate_a_threshold,
            "gate_b_threshold": args.gate_b_mrr_threshold,
            "best_val_mrr": best_val_mrr,
            "best_epoch": best_epoch,
            "aborted_gate": aborted_gate,
            "epochs": stats_history,
        }, f, indent=2)
    print(f"  Stats saved: {output_dir / 'training_stats.json'}")

    if aborted_gate:
        print(f"  ABORTED at gate {aborted_gate}")

    return {"aborted": aborted_gate is not None, "gate": aborted_gate,
            "best_mrr": best_val_mrr, "best_epoch": best_epoch,
            "final_stats": stats_history[-1] if stats_history else None}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Full GNN fine-tuning — TRIPLET-ONLY margin loss (V2)")
    parser.add_argument("--gnn-checkpoint", default="checkpoints/gnn/full_graph_pretrained.pt")
    parser.add_argument("--pairs", default="data/raw/proof_step_pairs.jsonl")
    parser.add_argument("--graph", default="data/graph/dependency_graph_full")
    parser.add_argument("--output-dir", default="data/gnn_triplet_full")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--preservation-weight", type=float, default=0.1)
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--no-abort", action="store_true")
    parser.add_argument("--gate-b-mrr-threshold", type=float, default=0.60)
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="InfoNCE temperature (higher = softer distribution)")
    parser.add_argument("--hard-neg-weight", type=float, default=0.5,
                        help="Weight for triplet hard-negative loss")
    args = parser.parse_args()

    args.num_threads = min(args.num_threads, 4)
    args.abort_on_gate_fail = not args.no_abort

    print("=" * 60)
    print("Full GNN Fine-tuning — InfoNCE + Triplet Loss (V2)")
    print("=" * 60)
    print(f"  Mode: FULL GNN FINE-TUNING (ALL 1.1M params trainable)")
    print(f"  GNN: {args.gnn_checkpoint}")
    print(f"  Epochs: {args.epochs}, Batch: {args.batch_size}")
    print(f"  LR: {args.learning_rate}, Threads: {args.num_threads}")
    print(f"  Margin: {args.margin}, Temperature: {args.temperature}")
    print(f"  Hard neg weight: {args.hard_neg_weight}, LP weight: {args.preservation_weight}")
    print(f"  Gate B MRR threshold: {args.gate_b_mrr_threshold}")
    print(f"  Abort on gate fail: {args.abort_on_gate_fail}")
    print(f"  Loss: InfoNCE + triplet margin — global pull + targeted push")
    print(f"  V2: Embeddings recomputed every epoch (full fine-tune)")
    print()

    result = train_gnn_triplet(args)
    if result["aborted"]:
        print(f"\n  Training aborted at gate {result['gate']}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
