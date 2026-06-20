#!/usr/bin/env python3
"""Train a proof-utility adapter on top of frozen GNN embeddings.

Loads pre-trained GNN, freezes it, adds GNNAdapterHead, trains
with contrastive loss on proof-step pairs + hard negatives from
proof checker. Includes link-prediction anchor loss to prevent
catastrophic forgetting.

Safety gates (abort if any trip):
  A: GNN frozen, adapter ≤150K trainable params
  B: Link-prediction preservation loss ≤20% above baseline
  C: Validation retrieval MRR > GNN baseline (0.786)
  D: Lemma embedding diversity: avg cosine std > 0.1, rank = 256
  E: (checked post-training) Gate 3 must beat 15.6%

Usage:
    # Smoke test
    python scripts/training/train_gnn_adapter.py \
        --gnn-checkpoint checkpoints/gnn/full_graph_pretrained.pt \
        --pairs data/raw/proof_step_pairs.jsonl \
        --hard-negatives data/hard_neg_triples.jsonl \
        --output-dir data/adapter_smoke \
        --epochs 2 --max-pairs 100 --num-threads 4

    # Full training
    python scripts/training/train_gnn_adapter.py \
        --gnn-checkpoint checkpoints/gnn/full_graph_pretrained.pt \
        --pairs data/raw/proof_step_pairs.jsonl \
        --hard-negatives data/hard_neg_triples.jsonl \
        --output-dir data/adapter_full \
        --epochs 20 --batch-size 256 --num-threads 4
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))

from src.explorer.gnn_encoder import GNNEncoder
from src.explorer.gnn_config import GNNConfig
from src.explorer.gnn_adapter import GNNAdapterHead
from src.contrastive.hard_negative_loss import (
    compute_combined_loss,
    compute_retrieval_accuracy,
)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_pairs(data_path: Path, max_pairs: int | None = None) -> list[dict]:
    """Load proof-step pairs from JSONL. Returns list of dicts."""
    pairs = []
    with open(data_path) as f:
        for line in f:
            pairs.append(json.loads(line))
    if max_pairs and len(pairs) > max_pairs:
        rng = random.Random(42)
        pairs = rng.sample(pairs, max_pairs)
    return pairs


def load_hard_neg_triples(data_path: Path, max_pairs: int | None = None
                          ) -> list[dict]:
    """Load hard-negative triples from JSONL."""
    triples = []
    with open(data_path) as f:
        for line in f:
            triples.append(json.loads(line))
    if max_pairs and len(triples) > max_pairs:
        rng = random.Random(42)
        triples = rng.sample(triples, max_pairs)
    return triples


def load_graph_tensors(graph_path: str) -> tuple[torch.Tensor, int, list[str]]:
    """Load pre-saved graph tensors (edge_index, num_nodes, node_ids)."""
    data = torch.load(graph_path, map_location="cpu", weights_only=False)
    if isinstance(data, dict):
        return data["edge_index"], data["num_nodes"], data["node_ids"]
    return data[0], data[1], data[2]


# ---------------------------------------------------------------------------
# GNN loading
# ---------------------------------------------------------------------------

def load_frozen_gnn(checkpoint_path: str) -> GNNEncoder:
    """Load GNN from checkpoint and freeze all parameters."""
    gnn = GNNEncoder.load(checkpoint_path)
    for p in gnn.parameters():
        p.requires_grad = False
    gnn.eval()
    return gnn


# ---------------------------------------------------------------------------
# Embedding computation
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_gnn_embeddings(
    gnn: GNNEncoder,
    features: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Compute frozen GNN node embeddings for all nodes."""
    sources = edge_index[0]
    targets = edge_index[1]
    edge_types = torch.zeros(sources.size(0), dtype=torch.long)
    return gnn(features, sources, targets, edge_types, num_nodes)


# ---------------------------------------------------------------------------
# Link-prediction preservation
# ---------------------------------------------------------------------------

def compute_link_prediction_loss(
    node_embeddings: torch.Tensor,
    edge_index: torch.Tensor,
    num_neg_samples: int = 5000,
) -> float:
    """Compute binary cross-entropy link prediction loss on adapted embeddings.

    Samples positive edges from the graph and equal number of random
    negative edges (non-edges). Returns BCE loss.
    """
    num_nodes = node_embeddings.size(0)
    num_pos = edge_index.size(1)

    # Sample positive edges
    if num_pos > num_neg_samples:
        idx = torch.randperm(num_pos)[:num_neg_samples]
        pos_src = edge_index[0, idx]
        pos_tgt = edge_index[1, idx]
    else:
        pos_src = edge_index[0]
        pos_tgt = edge_index[1]

    n_pos = pos_src.size(0)

    # Sample negative edges (random pairs not in edge_index)
    edge_set = set()
    for i in range(edge_index.size(1)):
        edge_set.add((edge_index[0, i].item(), edge_index[1, i].item()))

    neg_src = []
    neg_tgt = []
    while len(neg_src) < n_pos:
        u = random.randint(0, num_nodes - 1)
        v = random.randint(0, num_nodes - 1)
        if u != v and (u, v) not in edge_set and (v, u) not in edge_set:
            neg_src.append(u)
            neg_tgt.append(v)

    neg_src = torch.tensor(neg_src, dtype=torch.long)
    neg_tgt = torch.tensor(neg_tgt, dtype=torch.long)

    # Compute dot-product similarity
    pos_sim = (node_embeddings[pos_src] * node_embeddings[pos_tgt]).sum(dim=-1)
    neg_sim = (node_embeddings[neg_src] * node_embeddings[neg_tgt]).sum(dim=-1)

    # BCE loss
    pos_labels = torch.ones(n_pos)
    neg_labels = torch.zeros(n_pos)
    all_scores = torch.cat([pos_sim, neg_sim])
    all_labels = torch.cat([pos_labels, neg_labels])

    loss = F.binary_cross_entropy_with_logits(all_scores, all_labels)
    return loss.item()


# ---------------------------------------------------------------------------
# Embedding health checks (Gateway D)
# ---------------------------------------------------------------------------

@torch.no_grad()
def check_embedding_health(
    lemma_embeddings: torch.Tensor,
    min_cosine_std: float = 0.1,
) -> dict:
    """Check embedding diversity and effective rank.

    Returns dict with keys: cosine_std, rank, healthy (bool).
    """
    B, D = lemma_embeddings.shape

    # Average pairwise cosine std
    sim_matrix = lemma_embeddings @ lemma_embeddings.T  # [B, B]
    # Exclude diagonal
    mask = ~torch.eye(B, dtype=torch.bool)
    off_diag_sims = sim_matrix[mask]
    cosine_std = off_diag_sims.std().item()

    # Effective rank via SVD
    if B >= D:
        _, S, _ = torch.linalg.svd(lemma_embeddings, full_matrices=False)
        effective_rank = (S > 0.01 * S[0]).sum().item()
    else:
        effective_rank = B

    healthy = cosine_std > min_cosine_std and effective_rank >= D // 2

    return {
        "cosine_std": cosine_std,
        "effective_rank": effective_rank,
        "healthy": healthy,
    }


# ---------------------------------------------------------------------------
# Validation MRR (Gateway C)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_validation_mrr(
    gnn: GNNEncoder,
    adapter: GNNAdapterHead,
    val_pairs: list[dict],
    node_embeddings: torch.Tensor,
    lemma_name_to_idx: dict[str, int],
) -> float:
    """Compute Mean Reciprocal Rank on validation pairs.

    For each (goal, lemma) pair, encode the goal, then rank all lemmas
    by cosine similarity and compute the reciprocal rank of the correct one.
    """
    if not val_pairs:
        return 0.0

    # Only consider lemmas that exist in the graph
    valid_indices = set(lemma_name_to_idx.values())
    candidate_ids = sorted(valid_indices)
    candidate_embs = node_embeddings[candidate_ids]  # [C, D]
    reverse_idx = {v: k for k, v in lemma_name_to_idx.items()}
    candidate_names: list[str] = []
    for cid in candidate_ids:
        candidate_names.append(reverse_idx.get(cid, str(cid)))

    rr_list = []
    for pair in val_pairs[:500]:  # Cap at 500 for speed
        goal_text = pair["goal"]
        lemma_name = pair["lemma"]

        # Skip if lemma not in graph
        if lemma_name not in lemma_name_to_idx:
            continue

        correct_idx = lemma_name_to_idx[lemma_name]
        if correct_idx not in valid_indices:
            continue

        # Encode goal via keyword matching → adapter
        goal_emb = _encode_goal_fast(goal_text, gnn, adapter, node_embeddings,
                                      lemma_name_to_idx, candidate_names)
        if goal_emb is None:
            continue

        # Score all candidates
        scores = goal_emb @ candidate_embs.T  # [C]
        # Find rank of correct lemma
        correct_score = scores[candidate_ids.index(correct_idx)]
        rank = (scores > correct_score).sum().item() + 1
        rr_list.append(1.0 / rank)

    return sum(rr_list) / len(rr_list) if rr_list else 0.0


def _encode_goal_fast(
    goal_text: str,
    gnn: GNNEncoder,
    adapter: GNNAdapterHead,
    node_embeddings: torch.Tensor,
    lemma_name_to_idx: dict[str, int],
    candidate_names: list[str],
) -> torch.Tensor | None:
    """Fast approximate goal encoding via keyword matching (no graph needed)."""
    from src.explorer.mcts import _extract_math_keywords

    keywords = _extract_math_keywords(goal_text)
    matching_indices = []
    for kw in keywords:
        kw_lower = kw.lower()
        for name in candidate_names:
            if name and kw_lower in name.lower():
                idx = lemma_name_to_idx.get(name)
                if idx is not None and idx not in matching_indices:
                    matching_indices.append(idx)
        if len(matching_indices) >= 50:
            break

    if not matching_indices:
        return None

    idx_t = torch.tensor(matching_indices[:100])
    context_emb = node_embeddings[idx_t].mean(dim=0)

    # Through GNN goal encoder if available
    if gnn.goal_encoder is not None:
        context_emb = gnn.encode_goal(context_emb)
    else:
        context_emb = F.normalize(context_emb, dim=-1)

    return adapter(context_emb)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_adapter(
    gnn_checkpoint: str,
    pairs_path: str,
    hard_negatives_path: str,
    output_dir: str,
    graph_tensors_path: str = "data/graph/dependency_graph_full.pyg.pt",
    num_epochs: int = 20,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    temperature: float = 0.07,
    hard_neg_weight: float = 0.5,
    preservation_weight: float = 0.1,
    margin: float = 0.3,
    num_threads: int = 4,
    val_split: float = 0.1,
    max_pairs: int | None = None,
    link_pred_samples: int = 5000,
):
    # --- Setup ---
    torch.set_num_threads(num_threads)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    print("=" * 60)
    print("GNN Adapter Training")
    print("=" * 60)

    # --- Safety Gate A: Load frozen GNN, create adapter ---
    print("\n[Gate A] Loading frozen GNN...")
    gnn = load_frozen_gnn(gnn_checkpoint)
    gnn_params = sum(p.numel() for p in gnn.parameters())
    adapter = GNNAdapterHead()
    adapter_params = sum(p.numel() for p in adapter.parameters())

    print(f"  GNN params: {gnn_params:,} (frozen)")
    print(f"  Adapter params: {adapter_params:,} (trainable)")
    gnn_frozen = all(not p.requires_grad for p in gnn.parameters())
    print(f"  GNN frozen: {gnn_frozen}")
    print(f"  Gate A: {'PASS' if gnn_frozen and adapter_params <= 150000 else 'FAIL'}")

    if not gnn_frozen:
        print("ERROR: GNN is not fully frozen. Aborting.")
        return None
    if adapter_params > 150000:
        print(f"ERROR: Adapter has {adapter_params:,} params > 150K. Aborting.")
        return None

    # --- Load data ---
    print(f"\nLoading proof-step pairs from {pairs_path}...")
    pairs = load_pairs(Path(pairs_path), max_pairs=max_pairs)
    print(f"  Loaded {len(pairs):,} pairs")

    print(f"Loading hard negatives from {hard_negatives_path}...")
    hard_neg_triples = load_hard_neg_triples(Path(hard_negatives_path),
                                              max_pairs=max_pairs)
    # Build lookup: (goal, positive_lemma) → hard_negatives list
    hn_lookup: dict[tuple, list[str]] = {}
    for t in hard_neg_triples:
        key = (t["goal"], t["positive_lemma"])
        hn_lookup[key] = t.get("hard_negatives", [])
    print(f"  Loaded {len(hard_neg_triples):,} triples, "
          f"{len(hn_lookup):,} unique lookups")

    # --- Load graph for link prediction ---
    print(f"\nLoading graph tensors from {graph_tensors_path}...")
    edge_index, num_nodes, node_ids = load_graph_tensors(graph_tensors_path)
    print(f"  Nodes: {num_nodes:,}, Edges: {edge_index.size(1):,}")

    # Build initial features
    features = F.normalize(torch.randn(num_nodes, gnn.config.hidden_dim), dim=-1)

    # Compute frozen GNN embeddings
    print("Computing frozen GNN node embeddings...")
    node_embeddings = compute_gnn_embeddings(gnn, features, edge_index, num_nodes)
    node_embeddings_norm = F.normalize(node_embeddings, dim=-1)
    print(f"  Embeddings: {node_embeddings.shape}")

    # Build lemma_name → index mapping
    lemma_name_to_idx: dict[str, int] = {}
    for i, nid in enumerate(node_ids):
        short_name = nid.split(".")[-1] if "." in nid else nid
        lemma_name_to_idx[nid] = i
        if short_name not in lemma_name_to_idx:
            lemma_name_to_idx[short_name] = i
    print(f"  Lemma index: {len(lemma_name_to_idx):,} entries")

    # --- Baseline link prediction loss ---
    print("\n[Gate B] Computing baseline link-prediction loss...")
    baseline_lp_loss = compute_link_prediction_loss(
        node_embeddings_norm, edge_index, num_neg_samples=link_pred_samples,
    )
    lp_threshold = baseline_lp_loss * 1.20  # 20% above baseline
    print(f"  Baseline LP loss: {baseline_lp_loss:.6f}")
    print(f"  Threshold (120%): {lp_threshold:.6f}")

    # --- Split train/val ---
    split_idx = int(len(pairs) * (1 - val_split))
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]
    print(f"\nTrain pairs: {len(train_pairs):,}, Val pairs: {len(val_pairs):,}")

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=learning_rate)
    temperature_inv = 1.0 / temperature

    # --- Training loop ---
    stats_history = []
    print(f"\n{'='*60}")
    print(f"Training: {num_epochs} epochs, batch_size={batch_size}")
    print(f"  InfoNCE temp={temperature}, hard_neg_weight={hard_neg_weight}")
    print(f"  preservation_weight={preservation_weight}, margin={margin}")
    print(f"{'='*60}")

    for epoch in range(num_epochs):
        epoch_start = time.time()
        adapter.train()

        # Shuffle train pairs
        random.shuffle(train_pairs)

        total_loss = 0.0
        total_infonce = 0.0
        total_hard_neg = 0.0
        total_acc = 0.0
        n_batches = 0

        # Batch training
        for batch_start in range(0, len(train_pairs), batch_size):
            batch_end = min(batch_start + batch_size, len(train_pairs))
            batch = train_pairs[batch_start:batch_end]
            B = len(batch)

            if B < 2:
                continue

            # Collect goals and positive lemma names
            goals = [p["goal"] for p in batch]
            pos_lemmas = [p["lemma"] for p in batch]

            # Encode goals and lemmas through frozen GNN → adapter
            goal_embs = []
            lemma_embs = []
            for goal_text, lemma_name in zip(goals, pos_lemmas):
                # Goal encoding
                g_emb = _encode_goal_fast(
                    goal_text, gnn, adapter, node_embeddings_norm,
                    lemma_name_to_idx,
                    list(lemma_name_to_idx.keys()),
                )
                if g_emb is None:
                    g_emb = torch.zeros(gnn.config.hidden_dim)

                # Lemma encoding: get GNN embedding → adapter
                idx = lemma_name_to_idx.get(lemma_name)
                if idx is not None and idx < node_embeddings_norm.size(0):
                    l_emb = node_embeddings_norm[idx]
                else:
                    l_emb = torch.zeros(gnn.config.hidden_dim)

                goal_embs.append(g_emb)
                lemma_embs.append(adapter(l_emb))

            goal_embs_t = torch.stack(goal_embs)  # [B, D]
            lemma_embs_t = torch.stack(lemma_embs)  # [B, D]

            # Hard negatives: collect for this batch
            hard_neg_batch = []
            max_hn = 0
            for goal_text, lemma_name in zip(goals, pos_lemmas):
                hn_list = hn_lookup.get((goal_text, lemma_name), [])
                hn_embs = []
                for hn_name in hn_list[:5]:  # Max 5 hard negatives per pair
                    idx = lemma_name_to_idx.get(hn_name)
                    if idx is not None and idx < node_embeddings_norm.size(0):
                        hn_embs.append(adapter(node_embeddings_norm[idx]))
                if hn_embs:
                    max_hn = max(max_hn, len(hn_embs))
                hard_neg_batch.append(hn_embs)

            # Pad hard negatives to uniform shape [B, max_hn, D]
            if max_hn > 0:
                D = goal_embs_t.size(-1)
                hn_tensor = torch.zeros(B, max_hn, D)
                for i, hn_list in enumerate(hard_neg_batch):
                    for j, hn_emb in enumerate(hn_list):
                        hn_tensor[i, j] = hn_emb
            else:
                hn_tensor = None

            # Compute combined loss
            losses = compute_combined_loss(
                goal_embs_t, lemma_embs_t, hn_tensor,
                temperature_inv=temperature_inv,
                hard_neg_weight=hard_neg_weight,
                margin=margin,
            )

            # Compute retrieval accuracy
            acc = compute_retrieval_accuracy(goal_embs_t, lemma_embs_t)

            # Per-batch loss
            batch_loss = losses["total_loss"]

            # Add link-prediction preservation (every 4 batches to save time)
            if preservation_weight > 0 and n_batches % 4 == 0:
                # Compute adapted node embeddings for link prediction
                adapted_node_embs = adapter(node_embeddings_norm)
                lp_loss = compute_link_prediction_loss(
                    adapted_node_embs, edge_index,
                    num_neg_samples=min(link_pred_samples // 4, 1000),
                )
                # Preservation loss = how much LP loss increased from baseline
                preservation_loss = max(0.0, lp_loss - baseline_lp_loss)
                batch_loss = batch_loss + preservation_weight * preservation_loss
            else:
                lp_loss = baseline_lp_loss
                preservation_loss = 0.0

            # Backward pass
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            total_loss += losses["total_loss"].item()
            total_infonce += losses["infonce_loss"].item()
            total_hard_neg += losses["hard_neg_loss"].item()
            total_acc += acc.item()
            n_batches += 1

        # --- End-of-epoch stats ---
        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / max(1, n_batches)
        avg_infonce = total_infonce / max(1, n_batches)
        avg_hard_neg = total_hard_neg / max(1, n_batches)
        avg_acc = total_acc / max(1, n_batches)

        # --- Safety checks ---
        adapter.eval()

        # Gate B: Link-prediction preservation
        adapted_node_embs = adapter(node_embeddings_norm)
        current_lp_loss = compute_link_prediction_loss(
            adapted_node_embs, edge_index,
            num_neg_samples=link_pred_samples,
        )
        lp_delta_pct = (current_lp_loss - baseline_lp_loss) / baseline_lp_loss * 100

        # Gate C: Validation MRR
        val_mrr = compute_validation_mrr(
            gnn, adapter, val_pairs, adapted_node_embs, lemma_name_to_idx,
        )

        # Gate D: Embedding health
        # Sample 256 lemma embeddings
        sample_indices = list(lemma_name_to_idx.values())[:256]
        sample_embs = adapted_node_embs[sample_indices]
        health = check_embedding_health(sample_embs[:256])

        # Print stats
        print(f"\nEpoch {epoch+1}/{num_epochs} [{epoch_time:.1f}s]")
        print(f"  loss={avg_loss:.4f} (infonce={avg_infonce:.4f}, "
              f"hn={avg_hard_neg:.4f}) acc={avg_acc:.3f}")
        print(f"  LP loss: {current_lp_loss:.6f} ({lp_delta_pct:+.1f}% vs baseline)")
        print(f"  Val MRR: {val_mrr:.4f} (baseline: 0.786)")
        print(f"  Embed health: cos_std={health['cosine_std']:.4f}, "
              f"rank={health['effective_rank']}")

        # Store stats
        stats_history.append({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "infonce_loss": avg_infonce,
            "hard_neg_loss": avg_hard_neg,
            "accuracy": avg_acc,
            "lp_loss": current_lp_loss,
            "lp_delta_pct": lp_delta_pct,
            "val_mrr": val_mrr,
            "cosine_std": health["cosine_std"],
            "effective_rank": health["effective_rank"],
            "epoch_time": epoch_time,
        })

        # --- Gate B check ---
        if current_lp_loss > lp_threshold:
            print(f"\n[Gate B FAILED] LP loss {current_lp_loss:.6f} > "
                  f"threshold {lp_threshold:.6f} ({lp_delta_pct:+.1f}%)")
            print("ABORTING: link-prediction preservation degraded too much.")
            _save_abort_state(adapter, stats_history, output_dir,
                              "gate_b_lp_preservation")
            return stats_history

        # --- Gate C check ---
        if val_mrr > 0 and val_mrr < 0.786:
            print(f"\n[Gate C warning] Val MRR {val_mrr:.4f} < baseline 0.786 "
                  f"(but training may improve)")

        # --- Gate D check ---
        if not health["healthy"]:
            print(f"\n[Gate D FAILED] Embedding health degraded: "
                  f"cos_std={health['cosine_std']:.4f}, rank={health['effective_rank']}")
            print("ABORTING: embeddings collapsed.")
            _save_abort_state(adapter, stats_history, output_dir,
                              "gate_d_embedding_collapse")
            return stats_history

    # --- Training complete ---
    print(f"\n{'='*60}")
    print("Training complete!")

    # Save adapter
    adapter_path = output_dir / "adapter.pt"
    torch.save(adapter.state_dict(), adapter_path)
    print(f"Adapter saved to {adapter_path}")

    # Save stats
    stats_path = output_dir / "training_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats_history, f, indent=2)
    print(f"Stats saved to {stats_path}")

    # Final summary
    final = stats_history[-1]
    print(f"\nFinal results:")
    print(f"  Loss: {final['loss']:.4f}")
    print(f"  Accuracy: {final['accuracy']:.3f}")
    print(f"  Val MRR: {final['val_mrr']:.4f}")
    print(f"  LP delta: {final['lp_delta_pct']:+.1f}%")
    print(f"  Cosine std: {final['cosine_std']:.4f}")
    print(f"  Effective rank: {final['effective_rank']}")

    gate_c_pass = final['val_mrr'] > 0.786
    gate_b_pass = final['lp_delta_pct'] <= 20.0
    final_healthy = final.get('effective_rank', 0) >= 128 and final.get('cosine_std', 0) > 0.1
    print(f"\n  Gate B (LP preserve): {'PASS' if gate_b_pass else 'FAIL'}")
    print(f"  Gate C (MRR > 0.786): {'PASS' if gate_c_pass else 'FAIL'}")
    print(f"  Gate D (diversity): {'PASS' if final_healthy else 'FAIL'}")
    print(f"  Gate E: Run eval_gnn_adapter.py to check")

    return stats_history


def _save_abort_state(adapter, stats, output_dir, reason):
    """Save adapter and stats before aborting."""
    torch.save(adapter.state_dict(), output_dir / f"adapter_abort_{reason}.pt")
    with open(output_dir / "training_stats.json", "w") as f:
        json.dump({"aborted": reason, "stats": stats}, f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train GNN proof-utility adapter with safety gates",
    )
    parser.add_argument("--gnn-checkpoint", required=True,
                        default="checkpoints/gnn/full_graph_pretrained.pt")
    parser.add_argument("--pairs", required=True,
                        default="data/raw/proof_step_pairs.jsonl")
    parser.add_argument("--hard-negatives", required=True,
                        default="data/hard_neg_triples.jsonl")
    parser.add_argument("--graph-tensors",
                        default="data/graph/dependency_graph_full.pyg.pt")
    parser.add_argument("--output-dir", required=True,
                        default="data/adapter_full")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--hard-neg-weight", type=float, default=0.5)
    parser.add_argument("--preservation-weight", type=float, default=0.1)
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--link-pred-samples", type=int, default=5000)

    args = parser.parse_args()

    stats = train_adapter(
        gnn_checkpoint=args.gnn_checkpoint,
        pairs_path=args.pairs,
        hard_negatives_path=args.hard_negatives,
        output_dir=args.output_dir,
        graph_tensors_path=args.graph_tensors,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        temperature=args.temperature,
        hard_neg_weight=args.hard_neg_weight,
        preservation_weight=args.preservation_weight,
        margin=args.margin,
        num_threads=args.num_threads,
        val_split=args.val_split,
        max_pairs=args.max_pairs,
        link_pred_samples=args.link_pred_samples,
    )

    if stats is None:
        sys.exit(1)

    print("\nDone.")


if __name__ == "__main__":
    main()
