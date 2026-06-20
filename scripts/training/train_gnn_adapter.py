#!/usr/bin/env python3
"""Train a proof-utility adapter on top of frozen GNN embeddings.

Loads pre-trained GNN, freezes it, adds GNNAdapterHead, trains
with contrastive loss on proof-step pairs + hard negatives from
proof checker. Includes link-prediction anchor loss to prevent
catastrophic forgetting.

Safety gates:
  A: GNN frozen, ≤150K trainable params
  B: Link-prediction preservation loss ≤20% above baseline
  C: Validation retrieval MRR > GNN baseline (0.786)
  D: Embedding health (avg cosine std > 0.1, rank = 256)
  E: Gate 3 must beat 15.6% baseline (checked post-training)

Usage:
    # Smoke test
    python scripts/training/train_gnn_adapter.py \
        --gnn-checkpoint checkpoints/gnn/full_graph_pretrained.pt \
        --pairs data/raw/proof_step_pairs.jsonl \
        --output-dir data/adapter_smoke \
        --epochs 2 --max-pairs 100 --num-threads 4

    # Full training
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
from src.explorer.gnn_adapter import GNNAdapterHead
from src.contrastive.hard_negative_loss import (
    compute_infonce_loss,
    compute_triplet_margin_loss,
    compute_retrieval_accuracy,
)
from src.explorer.mcts import _extract_math_keywords


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_pairs(data_path: Path) -> list[dict]:
    """Load proof-step pairs from JSONL."""
    pairs = []
    with open(data_path) as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs


# ---------------------------------------------------------------------------
# Frozen GNN + Adapter wrapper
# ---------------------------------------------------------------------------

class GNNWithAdapter:
    """Frozen GNN backbone + trainable adapter head.

    Provides unified interface for encoding goals and lemmas through
    the frozen GNN → adapter pipeline.
    """

    def __init__(
        self,
        gnn: GNNEncoder,
        node_embeddings: torch.Tensor,
        lemma_to_idx: dict[str, int],
        kw_lemmas_map: dict[str, list[int]],
        adapter: GNNAdapterHead | None = None,
    ):
        self.gnn = gnn
        self.node_embeddings = node_embeddings  # [N, D] pre-computed
        self.node_embeddings_norm = F.normalize(node_embeddings, dim=-1)
        self.lemma_to_idx = lemma_to_idx
        self.kw_lemmas_map = kw_lemmas_map
        self.adapter = adapter or GNNAdapterHead()
        self.hidden_dim = node_embeddings.size(1)

    def encode_goal(self, goal_text: str) -> torch.Tensor:
        """Encode a goal text through GNN keyword averaging + adapter."""
        device = self.node_embeddings.device

        # Extract keywords and find matching lemma embeddings
        keywords = _extract_math_keywords(goal_text)
        candidate_scores: dict[int, float] = {}
        for kw in keywords:
            matches = self.kw_lemmas_map.get(kw.lower(), [])
            for rank, idx in enumerate(matches):
                if idx >= self.node_embeddings_norm.size(0):
                    continue
                score = 1.0 / (1.0 + rank * 0.1)
                candidate_scores[idx] = candidate_scores.get(idx, 0.0) + score

        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: -x[1])[:100]

        if sorted_candidates:
            matching_indices = [idx for idx, _ in sorted_candidates]
            indices_t = torch.tensor(matching_indices, device=device)
            context_emb = self.node_embeddings_norm[indices_t].mean(dim=0)
        else:
            context_emb = torch.zeros(self.hidden_dim, device=device)

        # Pass through adapter
        with torch.no_grad():
            if context_emb.norm() > 1e-8:
                context_emb = F.normalize(context_emb, dim=-1)

        adapted = self.adapter(context_emb.unsqueeze(0)).squeeze(0)
        return adapted

    def encode_lemma(self, lemma_name: str) -> torch.Tensor | None:
        """Get adapted embedding for a lemma by name."""
        idx = self.lemma_to_idx.get(lemma_name)
        if idx is None or idx >= self.node_embeddings.size(0):
            return None
        emb = self.node_embeddings[idx]
        adapted = self.adapter(emb.unsqueeze(0)).squeeze(0)
        return adapted

    def encode_lemma_batch(self, lemma_names: list[str]) -> torch.Tensor:
        """Get adapted embeddings for a batch of lemmas."""
        indices = []
        for name in lemma_names:
            idx = self.lemma_to_idx.get(name)
            if idx is not None and idx < self.node_embeddings.size(0):
                indices.append(idx)
            else:
                indices.append(0)  # fallback
        idx_t = torch.tensor(indices, device=self.node_embeddings.device)
        embs = self.node_embeddings[idx_t]
        adapted = self.adapter(F.normalize(embs, dim=-1))
        return adapted

    def train(self):
        self.adapter.train()

    def eval(self):
        self.adapter.eval()

    def parameters(self):
        return self.adapter.parameters()

    def state_dict(self):
        return self.adapter.state_dict()

    def load_state_dict(self, sd):
        self.adapter.load_state_dict(sd)


# ---------------------------------------------------------------------------
# Link-prediction preservation
# ---------------------------------------------------------------------------

def compute_link_prediction_loss(
    adapter: GNNAdapterHead,
    node_embeddings: torch.Tensor,  # [N, D] raw GNN embeddings
    sources: torch.Tensor,  # [E] source indices
    targets: torch.Tensor,  # [E] target indices
    num_negatives: int = 5,
    sample_edges: int = 5000,
) -> torch.Tensor:
    """Compute link-prediction preservation loss.

    Samples graph edges (positive pairs) and random non-edges (negative pairs).
    Scores them via cosine similarity after adapter transformation.
    Uses binary cross-entropy to preserve graph topology.

    Returns BCE loss — lower = better topology preservation.
    """
    device = node_embeddings.device
    num_edges = sources.size(0)

    # Sample positive edges
    n_pos = min(sample_edges, num_edges)
    pos_indices = torch.randperm(num_edges, device=device)[:n_pos]
    pos_src = sources[pos_indices]
    pos_tgt = targets[pos_indices]

    # Get adapted embeddings
    all_emb = adapter(F.normalize(node_embeddings, dim=-1))
    pos_src_emb = all_emb[pos_src]  # [n_pos, D]
    pos_tgt_emb = all_emb[pos_tgt]  # [n_pos, D]
    pos_scores = (pos_src_emb * pos_tgt_emb).sum(dim=-1)  # [n_pos]

    # Sample negative edges (random pairs not in edges)
    num_nodes = node_embeddings.size(0)
    n_neg = n_pos * num_negatives
    neg_src = torch.randint(0, num_nodes, (n_neg,), device=device)
    neg_tgt = torch.randint(0, num_nodes, (n_neg,), device=device)
    neg_src_emb = all_emb[neg_src]
    neg_tgt_emb = all_emb[neg_tgt]
    neg_scores = (neg_src_emb * neg_tgt_emb).sum(dim=-1)  # [n_neg]

    # BCE: positive edges → score=1, negative edges → score=0
    pos_loss = F.binary_cross_entropy_with_logits(
        pos_scores, torch.ones_like(pos_scores)
    )
    neg_loss = F.binary_cross_entropy_with_logits(
        neg_scores, torch.zeros_like(neg_scores)
    )

    return (pos_loss + neg_loss) / 2.0


# ---------------------------------------------------------------------------
# Embedding health checks (Gate D)
# ---------------------------------------------------------------------------

def check_embedding_health(
    embeddings: torch.Tensor,  # [N, D] adapted embeddings
) -> dict:
    """Check embedding diversity and rank.

    Returns:
        Dict with 'avg_cosine_std', 'rank', 'rank_ok', 'std_ok'.
    """
    N, D = embeddings.shape
    if N < 2:
        return {"avg_cosine_std": 0.0, "rank": 0, "rank_ok": False, "std_ok": False}

    # Sample to avoid O(N²) for large N
    sample_n = min(N, 2000)
    indices = torch.randperm(N)[:sample_n]
    sample = embeddings[indices]

    # Pairwise cosine similarity std
    cos_sim = sample @ sample.T  # [S, S]
    # Exclude diagonal
    mask = ~torch.eye(sample_n, dtype=torch.bool, device=embeddings.device)
    off_diag = cos_sim[mask]
    avg_cosine_std = off_diag.std().item()

    # Rank: SVD on sample
    U, S, V = torch.svd(sample)
    # Rank = number of singular values > 0.01 * max
    threshold = S.max().item() * 0.01
    rank = (S > threshold).sum().item()

    return {
        "avg_cosine_std": round(avg_cosine_std, 6),
        "rank": rank,
        "rank_ok": rank >= D,  # Full rank
        "std_ok": avg_cosine_std > 0.1,
    }


# ---------------------------------------------------------------------------
# MRR computation (Gate C)
# ---------------------------------------------------------------------------

def compute_mrr(
    adapter: GNNAdapterHead,
    node_embeddings: torch.Tensor,
    lemma_to_idx: dict[str, int],
    val_pairs: list[dict],
    device: torch.device,
    sample_size: int = 1000,
) -> float:
    """Compute Mean Reciprocal Rank on validation pairs.

    For each (goal, correct_lemma) pair:
    1. Encode goal through adapter
    2. Score all lemma embeddings
    3. Find rank of correct lemma
    4. MRR = mean(1/rank)
    """
    adapter.eval()
    all_emb = adapter(F.normalize(node_embeddings, dim=-1))

    # Use a sample for efficiency
    sample = val_pairs[:sample_size] if len(val_pairs) > sample_size else val_pairs

    reciprocal_ranks = []
    with torch.no_grad():
        for pair in sample:
            goal_text = pair["goal"]
            lemma_name = pair["lemma"]

            idx = lemma_to_idx.get(lemma_name)
            if idx is None or idx >= all_emb.size(0):
                continue

            # Encode goal through keyword averaging + adapter
            # Simplified: extract keywords, get embedding of matching lemmas
            keywords = _extract_math_keywords(goal_text)
            matching_indices = []
            for kw in keywords:
                # Use the pre-built keyword map from GNNWithAdapter
                pass  # This needs the kw_lemmas_map

    adapter.train()
    # Placeholder; will be properly implemented in GNNWithAdapter context
    return 0.0


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_adapter(args):
    """Main training routine with all safety gates inline."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(args.num_threads)
    device = torch.device("cpu")
    print(f"Device: {device}, Threads: {torch.get_num_threads()}")

    # ---- Gate A: Load frozen GNN, verify ≤150K trainable ----
    print("\n--- Loading frozen GNN ---")
    gnn = GNNEncoder.load(args.gnn_checkpoint)
    for p in gnn.parameters():
        p.requires_grad = False
    gnn.eval()
    gnn_params = sum(p.numel() for p in gnn.parameters())
    print(f"  GNN params (frozen): {gnn_params:,}")
    print(f"  GNN hidden_dim: {gnn.config.hidden_dim}")

    adapter = GNNAdapterHead(input_dim=gnn.config.hidden_dim)
    adapter_params = sum(p.numel() for p in adapter.parameters())
    print(f"  Adapter params (trainable): {adapter_params:,}")
    print(f"  Total trainable: {adapter_params:,}")

    if adapter_params > 150_000:
        gate_a_msg = (
            f"GATE A FAILED: {adapter_params:,} trainable params exceeds 150K limit"
        )
        print(f"\n*** {gate_a_msg} ***")
        _save_abort(output_dir, gate_a_msg, "A")
        return {"aborted": True, "gate": "A", "message": gate_a_msg}
    print(f"  Gate A PASS: {adapter_params:,} ≤ 150,000")

    # ---- Load graph and pre-compute GNN embeddings ----
    print("\n--- Loading dependency graph ---")
    graph_path = Path(args.graph)
    if not graph_path.with_suffix(".nx.pkl").exists():
        print(f"  ERROR: Graph not found at {graph_path}.nx.pkl")
        sys.exit(1)

    graph = DependencyGraph.load(graph_path)
    print(f"  Graph: {graph.summary()}")

    print("\n--- Computing GNN node embeddings ---")
    features = extract_initial_features(graph, gnn.config)
    sources, targets, edge_types, num_nodes = prepare_graph_tensors(graph)
    print(f"  Nodes: {num_nodes}, Edges: {sources.size(0)}")

    with torch.no_grad():
        node_embeddings = gnn(features, sources, targets, edge_types, num_nodes)
    print(f"  Embeddings shape: {node_embeddings.shape}")

    # ---- Build lemma index and keyword map ----
    print("\n--- Building lemma index ---")
    lemma_to_idx: dict[str, int] = {}
    for node_id in graph.node_ids:
        idx = graph.node_id_to_idx(node_id)
        if idx is not None:
            lemma_to_idx[node_id] = idx
            # Also index by short name
            short = node_id.split(".")[-1] if "." in node_id else node_id
            if short not in lemma_to_idx:
                lemma_to_idx[short] = idx
    print(f"  Lemma index: {len(lemma_to_idx)} entries")

    # Build keyword → lemma index map
    kw_lemmas_map: dict[str, list[int]] = {}
    from src.explorer.mcts import _BUILTIN_LEMMAS

    all_kw = set()
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

    for nid in graph.node_ids:
        idx = graph.node_id_to_idx(nid)
        if idx is None:
            continue
        name_lower = nid.lower()
        for kw in all_kw:
            if kw.lower() in name_lower:
                kw_lemmas_map.setdefault(kw.lower(), []).append(idx)
    print(f"  Keyword map: {len(kw_lemmas_map)} keywords")

    # ---- Build GNNWithAdapter wrapper ----
    model = GNNWithAdapter(gnn, node_embeddings, lemma_to_idx, kw_lemmas_map, adapter)

    # ---- Load proof-step pairs ----
    print("\n--- Loading proof-step pairs ---")
    all_pairs = load_pairs(Path(args.pairs))
    print(f"  Loaded {len(all_pairs)} pairs")

    if args.max_pairs and args.max_pairs < len(all_pairs):
        random.seed(42)
        all_pairs = random.sample(all_pairs, args.max_pairs)
        print(f"  Sampled {len(all_pairs)} pairs for smoke test")

    # Split train/val
    split_idx = int(len(all_pairs) * (1 - args.val_split))
    train_pairs = all_pairs[:split_idx]
    val_pairs = all_pairs[split_idx:]
    print(f"  Train: {len(train_pairs)}, Val: {len(val_pairs)}")

    # ---- Gate B: Baseline link-prediction loss ----
    print("\n--- Gate B: Baseline link-prediction preservation ---")
    baseline_adapter = GNNAdapterHead(input_dim=gnn.config.hidden_dim)
    baseline_adapter.eval()
    baseline_lp_loss = compute_link_prediction_loss(
        baseline_adapter, node_embeddings, sources, targets
    )
    print(f"  Baseline link-pred loss: {baseline_lp_loss.item():.6f}")
    gate_b_threshold = baseline_lp_loss.item() * 1.20  # 20% above baseline
    print(f"  Gate B threshold (≤20% baseline): {gate_b_threshold:.6f}")

    # ---- Setup optimizer ----
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # ---- Training loop ----
    stats_history = []
    best_val_mrr = 0.0
    best_epoch = -1
    aborted_gate = None

    print(f"\n{'=' * 60}")
    print(f"TRAINING: {args.epochs} epochs, batch_size={args.batch_size}")
    print(f"{'=' * 60}")

    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        epoch_loss = 0.0
        epoch_infonce = 0.0
        epoch_triplet = 0.0
        epoch_lp = 0.0
        num_batches = 0

        # Shuffle train pairs
        random.shuffle(train_pairs)

        for batch_start in range(0, len(train_pairs), args.batch_size):
            batch = train_pairs[batch_start : batch_start + args.batch_size]
            if len(batch) < 2:
                continue

            # Encode goals and positive lemmas through adapter
            goal_embs = []
            lemma_embs = []
            for pair in batch:
                g_emb = model.encode_goal(pair["goal"])
                l_emb = model.encode_lemma(pair["lemma"])
                if l_emb is None:
                    # Fallback: use zero embedding
                    l_emb = torch.zeros(model.hidden_dim, device=device)
                goal_embs.append(g_emb)
                lemma_embs.append(l_emb)

            goal_embs = torch.stack(goal_embs)  # [B, D]
            lemma_embs = torch.stack(lemma_embs)  # [B, D]

            # ---- Multi-task loss ----
            # 1) InfoNCE contrastive loss (in-batch hard negatives)
            temperature_inv = 1.0 / args.temperature
            infonce = compute_infonce_loss(goal_embs, lemma_embs, temperature_inv)

            # 2) Triplet margin loss using in-batch hardest negatives
            # For each goal, the hardest negative is the non-matching lemma
            # with the highest cosine similarity
            batch_size = goal_embs.size(0)
            if batch_size >= 2 and args.hard_neg_weight > 0:
                # Build hard negatives: for each goal, find the most similar
                # non-matching lemma in the batch
                sim_matrix = goal_embs @ lemma_embs.T  # [B, B]
                # Mask out the diagonal (positive pairs)
                mask = ~torch.eye(batch_size, dtype=torch.bool, device=device)
                sim_masked = sim_matrix.masked_fill(~mask, -1e9)
                _, hardest_indices = sim_masked.max(dim=1)  # [B]

                hard_neg_embs = lemma_embs[hardest_indices].unsqueeze(1)  # [B, 1, D]
                triplet_loss = compute_triplet_margin_loss(
                    goal_embs,
                    lemma_embs,  # positive
                    hard_neg_embs,  # hard negatives [B, 1, D]
                    margin=args.margin,
                )
            else:
                triplet_loss = torch.tensor(0.0, device=device)

            # 3) Link-prediction preservation loss
            if args.preservation_weight > 0:
                lp_loss = compute_link_prediction_loss(
                    adapter, node_embeddings, sources, targets, sample_edges=1000
                )
            else:
                lp_loss = torch.tensor(0.0, device=device)

            # Combined loss
            loss = (
                infonce
                + args.hard_neg_weight * triplet_loss
                + args.preservation_weight * lp_loss
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), args.grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_infonce += infonce.item()
            epoch_triplet += triplet_loss.item()
            epoch_lp += lp_loss.item()
            num_batches += 1

        scheduler.step()

        # Epoch stats
        avg_loss = epoch_loss / max(1, num_batches)
        avg_infonce = epoch_infonce / max(1, num_batches)
        avg_triplet = epoch_triplet / max(1, num_batches)
        avg_lp = epoch_lp / max(1, num_batches)
        elapsed = time.time() - t0

        # ---- Gate B check: link-prediction preservation ----
        model.eval()
        current_lp_loss = compute_link_prediction_loss(
            adapter, node_embeddings, sources, targets, sample_edges=2000
        )
        lp_delta = (current_lp_loss.item() - baseline_lp_loss.item()) / max(
            1e-8, abs(baseline_lp_loss.item())
        )

        # ---- Gate D: Embedding health ----
        adapted_emb = adapter(F.normalize(node_embeddings, dim=-1))
        health = check_embedding_health(adapted_emb)

        # ---- Gate C: Validation MRR ----
        val_mrr = _compute_val_mrr_fast(
            adapter, node_embeddings, lemma_to_idx, kw_lemmas_map, val_pairs, device
        )

        # ---- Logging ----
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
            "gates": {},
        }

        # Gate checks
        gate_b_ok = lp_delta <= 0.20  # Within 20% of baseline
        gate_d_ok = health["std_ok"] and health["rank_ok"]

        gates_str = f"B={'✓' if gate_b_ok else '✗'} C={'✓' if val_mrr > 0.786 else '✗'} D={'✓' if gate_d_ok else '✗'}"
        stats["gates"] = {
            "B": "PASS" if gate_b_ok else "FAIL",
            "C": "PASS" if val_mrr > 0.786 else "FAIL",
            "D": "PASS" if gate_d_ok else "FAIL",
        }

        print(
            f"  Epoch {epoch + 1:3d}/{args.epochs} | "
            f"loss={avg_loss:.4f} | "
            f"mrr={val_mrr:.4f} | "
            f"lpΔ={lp_delta*100:+.1f}% | "
            f"std={health['avg_cosine_std']:.3f} r={health['rank']} | "
            f"{gates_str} | "
            f"{elapsed:.1f}s"
        )

        stats_history.append(stats)

        # ---- Gate B abort ----
        if args.abort_on_gate_fail and not gate_b_ok:
            aborted_gate = "B"
            gate_b_msg = (
                f"GATE B FAILED at epoch {epoch + 1}: "
                f"link-pred loss {current_lp_loss.item():.6f} > "
                f"threshold {gate_b_threshold:.6f} ({lp_delta*100:+.1f}%)"
            )
            print(f"\n*** {gate_b_msg} ***")
            _save_abort(output_dir, gate_b_msg, "B")
            break

        # ---- Gate D abort ----
        if args.abort_on_gate_fail and not gate_d_ok:
            aborted_gate = "D"
            gate_d_msg = (
                f"GATE D FAILED at epoch {epoch + 1}: "
                f"std={health['avg_cosine_std']:.4f} (need >0.1), "
                f"rank={health['rank']} (need ={model.hidden_dim})"
            )
            print(f"\n*** {gate_d_msg} ***")
            _save_abort(output_dir, gate_d_msg, "D")
            break

        # Track best
        if val_mrr > best_val_mrr:
            best_val_mrr = val_mrr
            best_epoch = epoch + 1
            torch.save(adapter.state_dict(), output_dir / "adapter_best.pt")

    # ---- Post-training ----
    if not aborted_gate:
        print(f"\nTraining complete. Best MRR: {best_val_mrr:.4f} (epoch {best_epoch})")

        # Check Gate C on final model
        model.eval()
        final_val_mrr = _compute_val_mrr_fast(
            adapter, node_embeddings, lemma_to_idx, kw_lemmas_map, val_pairs, device
        )
        gate_c_ok = final_val_mrr > 0.786
        print(f"  Gate C final MRR: {final_val_mrr:.4f} > 0.786? {'PASS' if gate_c_ok else 'FAIL'}")

    # ---- Save artifacts ----
    torch.save(adapter.state_dict(), output_dir / "adapter.pt")
    with open(output_dir / "training_stats.json", "w") as f:
        json.dump(
            {
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
                "gate_b_threshold": gate_b_threshold,
                "best_val_mrr": best_val_mrr,
                "best_epoch": best_epoch,
                "aborted_gate": aborted_gate,
                "epochs": stats_history,
            },
            f,
            indent=2,
        )

    print(f"\n  Adapter saved: {output_dir / 'adapter.pt'}")
    print(f"  Stats saved:   {output_dir / 'training_stats.json'}")
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
# Fast validation MRR
# ---------------------------------------------------------------------------

def _compute_val_mrr_fast(
    adapter: GNNAdapterHead,
    node_embeddings: torch.Tensor,
    lemma_to_idx: dict[str, int],
    kw_lemmas_map: dict[str, list[int]],
    val_pairs: list[dict],
    device: torch.device,
    sample_size: int = 500,
) -> float:
    """Compute Mean Reciprocal Rank on a sample of validation pairs.

    For each (goal, correct_lemma) pair:
    1. Get goal embedding via keyword averaging + adapter
    2. Score against lemma embeddings for matched keywords
    3. Compute rank of correct lemma among scored lemmas
    """
    adapter.eval()
    all_emb = adapter(F.normalize(node_embeddings, dim=-1))

    sample = val_pairs[:sample_size] if len(val_pairs) > sample_size else val_pairs

    reciprocal_ranks = []

    with torch.no_grad():
        for pair in sample:
            goal_text = pair["goal"]
            lemma_name = pair["lemma"]

            correct_idx = lemma_to_idx.get(lemma_name)
            if correct_idx is None or correct_idx >= all_emb.size(0):
                continue

            # Encode goal through keyword averaging using pre-built map
            keywords = _extract_math_keywords(goal_text)
            matching_indices: set[int] = set()
            for kw in keywords:
                matches = kw_lemmas_map.get(kw.lower(), [])
                for idx in matches:
                    if idx < all_emb.size(0):
                        matching_indices.add(idx)

            if not matching_indices:
                continue

            match_list = list(matching_indices)[:100]
            match_t = torch.tensor(match_list, device=device)
            context_emb = all_emb[match_t].mean(dim=0)
            goal_emb = F.normalize(context_emb, dim=-1)

            # Score all lemmas
            scores = goal_emb @ all_emb.T  # [N]
            correct_score = scores[correct_idx]
            rank = (scores > correct_score).sum().item() + 1
            reciprocal_ranks.append(1.0 / rank)

    adapter.train()

    if not reciprocal_ranks:
        return 0.0
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


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
        description="Train proof-utility adapter on frozen GNN embeddings"
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
        help="Output directory for adapter and stats",
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=256, help="Batch size"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-3, help="Learning rate"
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
    print("GNN Adapter Training with Safety Gates")
    print("=" * 60)
    print(f"  GNN: {args.gnn_checkpoint}")
    print(f"  Pairs: {args.pairs}")
    print(f"  Epochs: {args.epochs}, Batch: {args.batch_size}")
    print(f"  LR: {args.learning_rate}, Threads: {args.num_threads}")
    print(f"  Hard neg weight: {args.hard_neg_weight}")
    print(f"  Preservation weight: {args.preservation_weight}")
    print()

    result = train_adapter(args)

    if result["aborted"]:
        print(f"\n  Training aborted at gate {result['gate']}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
