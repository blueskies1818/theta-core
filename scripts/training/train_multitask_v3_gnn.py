#!/usr/bin/env python3
"""
MULTI-TASK v3: Joint link-prediction + proof-step training from scratch.

HEAD 1: "does theorem A import lemma B?" — import edges (BCE)
HEAD 2: "does lemma L prove goal G?" — proof-step pairs + hard negatives (BCE)

KEY FIX over v1/v2: Uses the GNN's GoalEncoder for the proof head, which
projects context-averaged goal embeddings through an MLP for sharper
representations. v1 (flat MRR 0.0054) lacked this — simple linear projections
couldn't discriminate.

ARCHITECTURE:
  GNN encoder (GAT, 256-dim) + GoalEncoder (MLP)
  ├── Head 1: Import — BCE on dot(src_emb, tgt_emb)
  └── Head 2: Proof — BCE on dot(goal_encoder(ctx_emb), lemma_emb)
       + hard negatives refreshed every 5 epochs

TRAINING: from scratch, two-step optimization per epoch:
  1. Import loss backward → GNN backbone
  2. Proof loss backward per batch → GoalEncoder + GNN

Usage:
  # Smoke test (10 epochs, 5000 pairs):
  python train_multitask_v3.py --smoke-test

  # Full training (50 epochs, all pairs):
  python train_multitask_v3.py --epochs 50 --full

  # Custom:
  python train_multitask_v3.py --epochs 50 --domain Algebra \
      --max-pairs 50000 --output-dir data/multitask_v3
"""

import argparse
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

# ── Path setup ──────────────────────────────────────────────────────────────
_project_root = Path("/home/blueman1818/Projects/theta-core")
sys.path.insert(0, str(_project_root))

from src.explorer.dependency_graph import DependencyGraph
from src.explorer.gnn_config import GNNConfig
from src.explorer.gnn_encoder import (
    GNNEncoder,
    extract_initial_features,
    prepare_graph_tensors,
)
from src.explorer.mcts import _extract_math_keywords
from scripts.eval.eval_gnn_prover import (
    normalize_expression,
    build_lemma_norm_index,
    tokenize_expression,
)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_pairs(data_path: Path, lemma_to_idx: dict[str, int],
               max_pairs: int | None = None) -> list[dict]:
    """Load proof-step pairs, filtering to lemmas in the graph."""
    pairs = []
    with open(data_path) as f:
        for line in f:
            if max_pairs and len(pairs) >= max_pairs:
                break
            d = json.loads(line)
            if d["lemma"] in lemma_to_idx:
                d["_keywords"] = _extract_math_keywords(d["goal"])
                pairs.append(d)
    return pairs


def build_lemma_index(graph: DependencyGraph) -> dict[str, int]:
    """Map lemma names to integer indices in the graph."""
    index = {}
    for node_id in graph.node_ids:
        idx = graph.node_id_to_idx(node_id)
        if idx is not None:
            index[node_id] = idx
            short = node_id.split(".")[-1] if "." in node_id else node_id
            if short not in index:
                index[short] = idx
    return index


# ═══════════════════════════════════════════════════════════════════════════════
# GOAL CONTEXT CONSTRUCTION (structural matching — from pretrain_proof_step.py)
# ═══════════════════════════════════════════════════════════════════════════════

def precompute_goal_contexts(
    goals: list[str],
    lemma_to_idx: dict[str, int],
    idx_to_norm: dict[int, str],
    max_context: int = 100,
) -> list[list[int]]:
    """Precompute matching lemma indices for each goal via structural matching.

    Uses inverted indices for O(1) lookups: exact norm match, power-stripping,
    iff/implication decomposition, reflexivity.
    """
    norm_to_indices: dict[str, list[int]] = defaultdict(list)
    stripped_to_indices: dict[str, list[int]] = defaultdict(list)
    iff_lhs_to: dict[str, list[int]] = defaultdict(list)
    iff_rhs_to: dict[str, list[int]] = defaultdict(list)
    imp_rhs_to: dict[str, list[int]] = defaultdict(list)

    rfl_norm = normalize_expression("a = a")

    for idx, lemma_norm in idx_to_norm.items():
        norm_to_indices[lemma_norm].append(idx)
        stripped_key = re.sub(r'\s*\^\s*\d+', '', lemma_norm)
        stripped_to_indices[stripped_key].append(idx)

        if " ↔ " in lemma_norm:
            left, right = lemma_norm.split(" ↔ ", 1)
            iff_lhs_to[left.strip()].append(idx)
            iff_rhs_to[right.strip()].append(idx)
        elif " → " in lemma_norm:
            parts = lemma_norm.rsplit(" → ", 1)
            imp_rhs_to[parts[-1].strip()].append(idx)

    contexts = []
    for goal_text in goals:
        goal_norm = normalize_expression(goal_text)
        matches = set(norm_to_indices.get(goal_norm, []))

        # Reflexivity
        if not matches and "=" in goal_norm:
            sides = goal_norm.split("=", 1)
            if len(sides) == 2 and sides[0].strip() == sides[1].strip():
                matches.update(norm_to_indices.get(rfl_norm, []))

        # Iff/implication decomposition
        matches.update(iff_lhs_to.get(goal_norm, []))
        matches.update(iff_rhs_to.get(goal_norm, []))
        matches.update(imp_rhs_to.get(goal_norm, []))

        # Power-stripping
        if not matches:
            goal_stripped = re.sub(r'\s*\^\s*\d+', '', goal_norm)
            matches.update(stripped_to_indices.get(goal_stripped, []))

        contexts.append(list(matches)[:max_context])

    return contexts


# ═══════════════════════════════════════════════════════════════════════════════
# HEAD 1: IMPORT LINK-PREDICTION LOSS (BCE)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_import_loss(
    node_embeddings: torch.Tensor,
    sources: torch.Tensor,
    targets: torch.Tensor,
    num_negatives: int = 5,
    sample_edges: int = 5000,
) -> tuple[torch.Tensor, float]:
    """BCE loss on import edges: dot(src, tgt) → sigmoid → BCE(label)."""
    device = node_embeddings.device
    num_edges = sources.size(0)
    num_nodes = node_embeddings.size(0)

    n_pos = min(sample_edges, num_edges)
    pos_indices = torch.randperm(num_edges, device=device)[:n_pos]
    pos_src = sources[pos_indices]
    pos_tgt = targets[pos_indices]

    emb_norm = F.normalize(node_embeddings, dim=-1)
    pos_scores = (emb_norm[pos_src] * emb_norm[pos_tgt]).sum(dim=-1)

    n_neg = n_pos * num_negatives
    neg_src = torch.randint(0, num_nodes, (n_neg,), device=device)
    neg_tgt = torch.randint(0, num_nodes, (n_neg,), device=device)
    neg_scores = (emb_norm[neg_src] * emb_norm[neg_tgt]).sum(dim=-1)

    pos_loss = F.binary_cross_entropy_with_logits(
        pos_scores, torch.ones_like(pos_scores)
    )
    neg_loss = F.binary_cross_entropy_with_logits(
        neg_scores, torch.zeros_like(neg_scores)
    )
    loss = (pos_loss + neg_loss) / 2.0

    with torch.no_grad():
        pos_acc = (pos_scores > 0).float().mean().item()
        neg_acc = (neg_scores <= 0).float().mean().item()
        acc = (pos_acc + neg_acc) / 2.0

    return loss, acc


# ═══════════════════════════════════════════════════════════════════════════════
# HEAD 2: PROOF-STEP BCE LOSS (with hard negatives)
# ═══════════════════════════════════════════════════════════════════════════════

def sample_hard_negatives_from_embeddings(
    node_emb_norm: torch.Tensor,
    goal_embs: torch.Tensor,       # [P, D] goal embeddings (GoalEncoder outputs)
    lemma_indices: torch.Tensor,   # [P] correct lemma indices
    num_negatives: int = 5,
) -> torch.Tensor:
    """For each goal, sample `num_negatives` lemmas with highest cosine sim
    to the goal from the CURRENT embedding space (hard negatives).

    Returns [P, K] tensor of negative lemma indices.
    """
    device = goal_embs.device
    P = goal_embs.size(0)
    K = min(num_negatives, node_emb_norm.size(0) - 1)

    # Cosine similarity: [P, D] @ [D, N] = [P, N]
    cos_sim = torch.matmul(goal_embs, node_emb_norm.T)

    # Exclude correct lemmas
    cos_sim[torch.arange(P, device=device), lemma_indices] = -float("inf")

    # Get top K*3 candidates, shuffle for diversity
    topk = min(K * 3, node_emb_norm.size(0))
    _, top_indices = torch.topk(cos_sim, k=topk, dim=1)  # [P, topk]

    # Shuffle within each row and take first K
    neg_indices = torch.zeros(P, K, dtype=torch.long, device=device)
    for i in range(P):
        perm = torch.randperm(topk, device=device)[:K]
        neg_indices[i] = top_indices[i, perm]

    return neg_indices


def compute_proof_bce(
    goal_embs: torch.Tensor,
    lemma_embs: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    """BCE loss on proof-step pairs: dot(goal, lemma) → sigmoid → BCE(label).

    goal_embs: [B, D] — organized as [pos, neg1..negK, pos, neg1..negK, ...]
    lemma_embs: [B, D] — corresponding lemma embeddings
    Every (K+1)th item starting at index 0 is positive.
    """
    B = goal_embs.size(0)
    if B == 0:
        return torch.tensor(0.0, device=goal_embs.device, requires_grad=True), 0.0

    scores = (goal_embs * lemma_embs).sum(dim=-1)  # [B]
    num_per = 6  # 1 positive + 5 negatives
    labels = torch.zeros(B, device=scores.device)
    labels[0::num_per] = 1.0

    loss = F.binary_cross_entropy_with_logits(scores, labels)

    with torch.no_grad():
        pos_mask = labels > 0.5
        pos_acc = (scores[pos_mask] > 0).float().mean().item() if pos_mask.any() else 0.0
        neg_acc = (scores[~pos_mask] <= 0).float().mean().item() if (~pos_mask).any() else 0.0
        acc = (pos_acc + neg_acc) / 2.0

    return loss, acc


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION MRR (Gate B)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_val_mrr(
    gnn: GNNEncoder,
    node_embeddings: torch.Tensor,
    lemma_to_idx: dict[str, int],
    precomputed_contexts: list[list[int]],
    val_goal_indices: list[int],
    val_target_indices: list[int],
    sample_size: int = 500,
) -> float:
    """Compute Mean Reciprocal Rank on validation pairs."""
    device = node_embeddings.device
    all_emb_norm = F.normalize(node_embeddings, dim=-1)
    num_nodes = all_emb_norm.size(0)

    if len(val_goal_indices) > sample_size:
        sample = random.sample(
            list(zip(val_goal_indices, val_target_indices)), sample_size
        )
    else:
        sample = list(zip(val_goal_indices, val_target_indices))

    reciprocal_ranks = []
    with torch.no_grad():
        for goal_idx, correct_idx in sample:
            if correct_idx >= num_nodes:
                continue

            ctx = precomputed_contexts[goal_idx]
            if not ctx:
                continue

            ctx_t = torch.tensor(ctx, device=device)
            raw_ctx = all_emb_norm[ctx_t].mean(dim=0)
            goal_emb = gnn.encode_goal(raw_ctx.unsqueeze(0))

            scores = (goal_emb @ all_emb_norm.T).squeeze(0)
            correct_score = scores[correct_idx]
            rank = (scores > correct_score).sum().item() + 1
            reciprocal_ranks.append(1.0 / rank)

    if not reciprocal_ranks:
        return 0.0
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


# ═══════════════════════════════════════════════════════════════════════════════
# EMBEDDING HEALTH (Gate C)
# ═══════════════════════════════════════════════════════════════════════════════

def check_embedding_health(embeddings: torch.Tensor) -> dict:
    """Check embedding rank and cosine std."""
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
        _, S, _ = torch.linalg.svd(sample.float(), full_matrices=False)
        rank_val = (S > S.max().item() * 0.01).sum().item()
    except Exception:
        rank_val = 0

    return {
        "avg_cosine_std": round(avg_cosine_std, 6),
        "rank": rank_val,
        "std_ok": avg_cosine_std > 0.03,
        "rank_ok": rank_val > 128,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Multi-task GNN v3 training")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--proof-weight", type=float, default=0.5,
                        help="Weight of proof loss relative to import loss")
    parser.add_argument("--num-hard-negatives", type=int, default=5)
    parser.add_argument("--neg-refresh-every", type=int, default=5)
    parser.add_argument("--num-threads", type=int, default=6)
    parser.add_argument("--domain", default="Algebra",
                        help="Domain subgraph (empty for full graph)")
    parser.add_argument("--full", action="store_true",
                        help="Use full graph (no domain filter)")
    parser.add_argument("--smoke-test", action="store_true",
                        help="10 epochs on 5000 pairs (after any --max-pairs)")
    parser.add_argument("--max-pairs", type=int, default=None,
                        help="Max proof-step pairs to load")
    parser.add_argument("--output-dir", default="data/multitask_v3")
    parser.add_argument("--checkpoint-name", default="multitask_v3.pt")
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--mrr-every", type=int, default=5)
    parser.add_argument("--skip-smoke", action="store_true")
    args = parser.parse_args()

    # Smoke test overrides
    if args.smoke_test:
        args.epochs = 10
        args.max_pairs = min(args.max_pairs or 5000, 5000)

    # ── Hardware ─────────────────────────────────────────────────────────
    torch.set_num_threads(args.num_threads)
    import os
    os.environ["OMP_NUM_THREADS"] = str(args.num_threads)
    os.environ["MKL_NUM_THREADS"] = str(args.num_threads)
    device = torch.device("cpu")
    print(f"Device: {device}, Threads: {args.num_threads}")

    # ── Load graph ────────────────────────────────────────────────────────
    graph_path = _project_root / "data/graph/dependency_graph"
    print(f"\n--- Loading graph ---")
    graph = DependencyGraph.load(graph_path)
    if args.full:
        pass  # Use full graph
    elif args.domain:
        graph = graph.domain_subgraph(args.domain)
    print(f"  {graph.num_nodes} nodes, {graph.num_edges} edges")

    num_nodes = graph.num_nodes

    # ── Build indices ─────────────────────────────────────────────────────
    lemma_to_idx = build_lemma_index(graph)
    print(f"  Lemma index: {len(lemma_to_idx)} entries")

    idx_to_norm = build_lemma_norm_index(graph, lemma_to_idx)
    print(f"  Norm index: {len(idx_to_norm)} normalized conclusions")

    # ── Load proof-step pairs ─────────────────────────────────────────────
    data_path = _project_root / "data/raw/proof_step_pairs.jsonl"
    print(f"\n--- Loading proof-step pairs ---")
    pairs = load_pairs(data_path, lemma_to_idx, args.max_pairs)
    print(f"  {len(pairs)} pairs (lemmas in graph)")

    if len(pairs) < 100:
        print("ERROR: Too few matching pairs. Check lemma resolution.")
        sys.exit(1)

    # Train/val split
    split_idx = int(len(pairs) * (1 - args.val_split))
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]
    print(f"  Train: {len(train_pairs)}, Val: {len(val_pairs)}")

    # ── Precompute goal contexts ──────────────────────────────────────────
    goals = [p["goal"] for p in pairs]
    train_goals = [p["goal"] for p in train_pairs]
    val_goals = [p["goal"] for p in val_pairs]

    print("\n--- Precomputing goal contexts ---")
    t0 = time.time()
    precomputed_contexts = precompute_goal_contexts(goals, lemma_to_idx, idx_to_norm)
    hits = sum(1 for c in precomputed_contexts if c)
    print(f"  {hits}/{len(goals)} goals have context matches ({time.time()-t0:.1f}s)")

    # Split precomputed contexts
    train_contexts = precomputed_contexts[:split_idx]
    val_contexts = precomputed_contexts[split_idx:]

    # Target lemma indices
    train_target_indices = [lemma_to_idx[p["lemma"]] for p in train_pairs]
    val_target_indices = [lemma_to_idx[p["lemma"]] for p in val_pairs]

    # ── Initialize GNN from scratch ───────────────────────────────────────
    print("\n--- Initializing GNN from scratch ---")
    config = GNNConfig(
        hidden_dim=256,
        num_layers=3,
        num_heads=8,
        input_dim=256,
        dropout=0.1,
        activation="gelu",
        use_edge_types=True,
        num_edge_types=5,
        bidirectional=True,
        use_goal_encoder=True,
        goal_encoder_expansion=2,
        goal_encoder_dropout=0.1,
        init_features="random",  # FROM SCRATCH, no pretrained weights
    )

    gnn = GNNEncoder(config).to(device)
    total_params = sum(p.numel() for p in gnn.parameters())
    ge_params = sum(p.numel() for p in gnn.goal_encoder.parameters())
    print(f"  GNN: {total_params:,} params ({ge_params:,} in GoalEncoder)")
    print(f"  Fresh random init — NO pretrained weights")

    # Pre-compute graph tensors
    features = extract_initial_features(graph, config).to(device)
    sources, targets, edge_types, n_nodes = prepare_graph_tensors(graph)
    sources = sources.to(device)
    targets = targets.to(device)
    edge_types = edge_types.to(device)
    print(f"  Graph tensors: {features.shape}, {sources.size(0)} edges")

    # ── Optimizer ─────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        gnn.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # ── Training setup ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"MULTI-TASK v3 TRAINING: {args.epochs} epochs, batch={args.batch_size}")
    print(f"  Loss = BCE(import) + {args.proof_weight} * BCE(proof)")
    print(f"  Hard negatives: {args.num_hard_negatives}, refresh every {args.neg_refresh_every} epochs")
    print(f"  Smoke test: {args.smoke_test}")
    print(f"  lr={args.lr}, CPU threads={args.num_threads}")
    print(f"{'='*60}")

    # ── Training loop ─────────────────────────────────────────────────────
    stats_history = []
    best_val_mrr = 0.0
    best_epoch = 0
    aborted_gate = None
    init_import_loss = None
    init_proof_loss = None
    init_import_acc = None
    hidden_dim = config.hidden_dim

    for epoch in range(args.epochs):
        t_epoch_start = time.time()
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")

        # ── Single GNN forward pass ────────────────────────────────────
        gnn.train()
        node_emb = gnn(features, sources, targets, edge_types, n_nodes)
        node_emb_norm = F.normalize(node_emb, dim=-1)

        # ── HEAD 1: Import link-prediction loss (BCE) ──────────────────
        import_loss, import_acc = compute_import_loss(
            node_emb, sources, targets, sample_edges=5000
        )

        # ── HEAD 2: Proof-step loss (BCE) ──────────────────────────────
        train_ctx_indices = [i for i, c in enumerate(train_contexts) if c]
        if not train_ctx_indices or len(train_ctx_indices) < 2:
            epoch_proof_loss = 0.0
            epoch_proof_acc = 0.0
            proof_loss = torch.tensor(0.0, device=device, requires_grad=False)
        else:
            B_sample = min(args.batch_size, len(train_ctx_indices))
            random.shuffle(train_ctx_indices)
            batch_ctx_indices = train_ctx_indices[:B_sample]
            K = args.num_hard_negatives

            # Build raw context embeddings (NOT detached — gradients flow to GNN)
            raw_ctx_tensors = []
            correct_lemma_indices = []
            for idx in batch_ctx_indices:
                ctx = train_contexts[idx]
                ctx_t = torch.tensor(ctx, device=device)
                raw_ctx = node_emb_norm[ctx_t].mean(dim=0)  # gradients through GNN!
                raw_ctx_tensors.append(raw_ctx)
                correct_lemma_indices.append(train_target_indices[idx])

            raw_ctx_batch = torch.stack(raw_ctx_tensors)  # [B, D]
            correct_idx_tensor = torch.tensor(
                correct_lemma_indices, dtype=torch.long, device=device
            )

            # Goal embeddings via GoalEncoder (gradients → GoalEncoder → GNN)
            goal_embs = gnn.encode_goal(raw_ctx_batch)  # [B, D]

            # Hard negatives from detached embeddings (no gradient through sampling)
            neg_idx_tensor = sample_hard_negatives_from_embeddings(
                F.normalize(node_emb.detach(), dim=-1),
                goal_embs.detach(),
                correct_idx_tensor, num_negatives=K,
            )  # [B, K]

            # Build (positive + negative) lemma embeddings
            # These index into node_emb_norm which IS in the computation graph
            lemma_embs_list = []
            for i in range(B_sample):
                lemma_embs_list.append(node_emb_norm[correct_idx_tensor[i]].unsqueeze(0))
                for k in range(K):
                    lemma_embs_list.append(
                        node_emb_norm[neg_idx_tensor[i, k]].unsqueeze(0)
                    )
            lemma_embs_all = torch.cat(lemma_embs_list, dim=0)  # [B*(K+1), D]

            goal_embs_repeated = goal_embs.unsqueeze(1).expand(
                -1, K+1, -1
            ).reshape(-1, hidden_dim)

            proof_loss, proof_acc = compute_proof_bce(
                goal_embs_repeated, lemma_embs_all
            )
            epoch_proof_loss = proof_loss.item()
            epoch_proof_acc = proof_acc

        # ── Combined loss + single backward ─────────────────────────────
        loss = import_loss + args.proof_weight * proof_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gnn.parameters(), 1.0)
        optimizer.step()

        # ── Health check (Gate C) ─────────────────────────────────────
        with torch.no_grad():
            health = check_embedding_health(node_emb)

        # ── Validation MRR (Gate B) ───────────────────────────────────
        val_mrr = None
        if (epoch + 1) % args.mrr_every == 0 or epoch == args.epochs - 1:
            gnn.eval()
            with torch.no_grad():
                val_node_emb = gnn(features, sources, targets, edge_types, n_nodes)
            val_mrr = compute_val_mrr(
                gnn, val_node_emb, lemma_to_idx,
                val_contexts, list(range(len(val_target_indices))),
                val_target_indices,
            )
            gnn.train()

        # ── Logging ───────────────────────────────────────────────────
        epoch_time = time.time() - t_epoch_start
        print(f"  Import Loss: {import_loss.item():.6f}  Acc: {import_acc:.4f}")
        print(f"  Proof Loss:  {epoch_proof_loss:.6f}  Acc: {epoch_proof_acc:.4f}")
        print(f"  Health: std={health['avg_cosine_std']:.4f} rank={health['rank']}")
        if val_mrr is not None:
            print(f"  Val MRR: {val_mrr:.6f}")

        stats_history.append({
            "epoch": epoch + 1,
            "import_loss": import_loss.item(),
            "proof_loss": epoch_proof_loss,
            "import_acc": import_acc,
            "proof_acc": epoch_proof_acc,
            "cosine_std": health["avg_cosine_std"],
            "rank": health["rank"],
            "val_mrr": val_mrr,
            "epoch_time_s": round(epoch_time, 1),
        })

        # ── Gate A: Losses must decrease (not diverge) ────────────────
        if epoch == 0:
            init_import_loss = import_loss.item()
            init_proof_loss = epoch_proof_loss
            init_import_acc = import_acc

        if init_import_loss is not None and epoch >= 5:
            if import_loss.item() > init_import_loss * 1.5:
                print(f"\n  GATE A FAILED: Import loss diverged "
                      f"({init_import_loss:.3f} → {import_loss.item():.3f})")
                aborted_gate = "A"
                break
        if init_proof_loss is not None and init_proof_loss > 0 and epoch >= 5:
            if epoch_proof_loss > init_proof_loss * 1.5:
                print(f"\n  GATE A FAILED: Proof loss diverged "
                      f"({init_proof_loss:.3f} → {epoch_proof_loss:.3f})")
                aborted_gate = "A"
                break

        # ── Gate B: MRR signal check ──────────────────────────────────
        if epoch == 9 and val_mrr is not None:
            if val_mrr <= 0.0001:
                print(f"\n  GATE B WARNING: MRR {val_mrr:.6f} very low at epoch 10")
            elif val_mrr <= 0.10:
                print(f"\n  GATE B: MRR {val_mrr:.6f} at epoch 10 — continuing, target >0.10")

        if epoch == 24 and val_mrr is not None:
            if val_mrr <= 0.01:
                print(f"\n  GATE B FAILED: MRR {val_mrr:.6f} <= 0.01 at epoch 25")
                aborted_gate = "B"
                break

        # ── Gate C: Embedding health ──────────────────────────────────
        if epoch >= 3 and not health["std_ok"] and health["avg_cosine_std"] < 0.01:
            print(f"\n  GATE C FAILED: cosine_std {health['avg_cosine_std']:.4f} < 0.01")
            aborted_gate = "C"
            break

        # ── Gate D: Import accuracy must not collapse ─────────────────
        if init_import_acc is not None and epoch >= 5:
            min_acc = max(init_import_acc * 0.8, 0.50)
            if import_acc < min_acc:
                print(f"\n  GATE D FAILED: Import acc {import_acc:.3f} < {min_acc:.3f}")
                aborted_gate = "D"
                break

        # ── Save best model ───────────────────────────────────────────
        gnn.eval()
        if val_mrr is not None and val_mrr > best_val_mrr:
            best_val_mrr = val_mrr
            best_epoch = epoch + 1
            ckpt_path = _project_root / "checkpoints/gnn" / args.checkpoint_name
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            gnn.save(str(ckpt_path))
            print(f"  ✓ Saved best (MRR={best_val_mrr:.6f})")

        scheduler.step()

    # ── Training done ─────────────────────────────────────────────────────
    gnn.eval()
    print(f"\n{'='*60}")
    if aborted_gate:
        print(f"TRAINING ABORTED (Gate {aborted_gate})")
    else:
        print(f"TRAINING COMPLETE: {args.epochs} epochs")
    print(f"  Best Val MRR: {best_val_mrr:.4f} (epoch {best_epoch})")
    print(f"{'='*60}")

    # Save final checkpoint
    final_ckpt = _project_root / "checkpoints/gnn" / args.checkpoint_name
    final_ckpt.parent.mkdir(parents=True, exist_ok=True)
    gnn.save(str(final_ckpt))
    print(f"Encoder saved to: {final_ckpt}")

    # Save training stats
    output_dir = _project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = output_dir / "training_stats.json"
    with open(stats_path, "w") as f:
        json.dump({
            "config": {
                "hidden_dim": config.hidden_dim,
                "num_layers": config.num_layers,
                "num_heads": config.num_heads,
                "total_params": total_params,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "proof_weight": args.proof_weight,
                "num_hard_negatives": args.num_hard_negatives,
                "neg_refresh_every": args.neg_refresh_every,
                "num_train_pairs": len(train_pairs),
                "num_val_pairs": len(val_pairs),
            },
            "history": stats_history,
            "best_val_mrr": best_val_mrr,
            "best_epoch": best_epoch,
            "aborted_gate": aborted_gate,
        }, f, indent=2)
    print(f"Training stats saved to: {stats_path}")

    return 0 if not aborted_gate else 1


if __name__ == "__main__":
    sys.exit(main())
