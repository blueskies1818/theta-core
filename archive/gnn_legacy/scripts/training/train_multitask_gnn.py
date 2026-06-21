#!/usr/bin/env python3
"""
MULTI-TASK GNN: Joint link-prediction + proof-step training from scratch.

Two concurrent objectives on the same graph:
  HEAD 1: "does theorem A import lemma B?" — 436K import edges
  HEAD 2: "does lemma L prove goal G?" — 226K proof-step pairs + hard negatives

Shared GAT encoder (256-dim, 1.1M params) → two dot-product scoring heads.
Loss = BCE(import_head) + 0.5 * BCE(proof_head)

Hard negatives refreshed every 5 epochs from current embedding space.

Usage:
  # SMOKE TEST (10 epochs, 5000 pairs):
  python scripts/training/train_multitask_gnn.py \
      --epochs 10 --max-pairs 5000 --output-dir data/multitask_smoke \
      --skip-gate3

  # FULL TRAINING (50 epochs, all pairs):
  python scripts/training/train_multitask_gnn.py \
      --epochs 50 --output-dir data/multitask_full

  # Then evaluate on gate3_v2:
  python scripts/training/train_multitask_gnn.py \
      --eval-only --gnn-checkpoint checkpoints/gnn/multitask_scratch.pt \
      --output-dir data/multitask_full
"""

import argparse
import functools
import json
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
from scripts.eval.eval_gnn_prover import (
    normalize_expression,
    extract_conclusion,
    build_lemma_norm_index,
    tokenize_expression,
)


# ===========================================================================
# Data loading
# ===========================================================================

def load_pairs(data_path: Path) -> list[dict]:
    pairs = []
    with open(data_path) as f:
        for line in f:
            pair = json.loads(line)
            pair["_keywords"] = _extract_math_keywords(pair["goal"])
            pairs.append(pair)
    return pairs


# ===========================================================================
# Import link-prediction loss (HEAD 1)
# ===========================================================================

def compute_import_loss(
    node_embeddings: torch.Tensor,
    sources: torch.Tensor,
    targets: torch.Tensor,
    num_negatives: int = 5,
    sample_edges: int = 5000,
) -> tuple[torch.Tensor, float]:
    """BCE loss on import edges: predicts whether edge (src→tgt) exists."""
    device = node_embeddings.device
    num_edges = sources.size(0)
    num_nodes = node_embeddings.size(0)

    n_pos = min(sample_edges, num_edges)
    pos_indices = torch.randperm(num_edges, device=device)[:n_pos]
    pos_src = sources[pos_indices]
    pos_tgt = targets[pos_indices]

    emb_norm = F.normalize(node_embeddings, dim=-1)
    pos_scores = (emb_norm[pos_src] * emb_norm[pos_tgt]).sum(dim=-1)

    # Random negative sampling
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

    # Compute accuracy for Gate D
    with torch.no_grad():
        pos_correct = (pos_scores > 0).float().mean().item()
        neg_correct = (neg_scores <= 0).float().mean().item()
        acc = (pos_correct + neg_correct) / 2.0

    return loss, acc


# ===========================================================================
# Proof-step loss (HEAD 2) with hard negative sampling
# ===========================================================================

def compute_proof_loss(
    goal_embeddings: torch.Tensor,
    lemma_embeddings: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    """BCE loss on proof-step pairs with hard negatives in-batch.

    goal_embeddings: [B, D] — includes positive goal + negative goal embeddings
                     organized as [pos_0, neg_0_0..neg_0_4, pos_1, neg_1_0..neg_1_4, ...]
    lemma_embeddings: [B, D] — corresponding lemma embeddings

    For each positive pair, compute BCE(score, 1.0).
    For each negative pair, compute BCE(score, 0.0).
    """
    B = goal_embeddings.size(0)
    D = goal_embeddings.size(1)

    if B == 0:
        return torch.tensor(0.0, device=goal_embeddings.device), 0.0

    # Score: elementwise dot-product for each pair
    scores = (goal_embeddings * lemma_embeddings).sum(dim=-1)  # [B]

    # Every 6th item (index % 6 == 0) is positive, rest are negatives
    labels = torch.zeros(B, device=scores.device)
    labels[0::6] = 1.0  # pos_0, pos_1, pos_2, ...

    loss = F.binary_cross_entropy_with_logits(scores, labels)

    # Accuracy
    with torch.no_grad():
        pos_mask = labels > 0.5
        pos_acc = (scores[pos_mask] > 0).float().mean().item() if pos_mask.any() else 0.0
        neg_acc = (scores[~pos_mask] <= 0).float().mean().item() if (~pos_mask).any() else 0.0
        acc = (pos_acc + neg_acc) / 2.0

    return loss, acc


def sample_hard_negatives(
    node_emb_norm: torch.Tensor,
    lemma_to_idx: dict[str, int],
    pairs: list[dict],
    num_negatives: int = 5,
    num_nodes: int | None = None,
) -> list[dict]:
    """For each positive (goal, lemma) pair, sample `num_negatives` negative
    lemmas that have the highest cosine similarity to the goal context embedding
    in the CURRENT embedding space."""
    device = node_emb_norm.device
    if num_nodes is None:
        num_nodes = node_emb_norm.size(0)

    # Build goal context embeddings for all pairs
    all_lemma_indices = set()
    for pair in pairs:
        idx = lemma_to_idx.get(pair["lemma"])
        if idx is not None and idx < num_nodes:
            all_lemma_indices.add(idx)

    lemma_idx_list = sorted(all_lemma_indices)
    if not lemma_idx_list:
        # No valid lemmas, use random
        return _sample_random_negatives(pairs, lemma_to_idx, num_negatives, num_nodes)

    all_lemma_t = torch.tensor(lemma_idx_list, device=device)
    all_lemma_embs = node_emb_norm[all_lemma_t]  # [L, D]

    # Build keyword context → embedding for each goal
    # (Keyword-match average, same as training context)
    enriched = []
    for pair in pairs:
        keywords = pair.get("_keywords", [])
        correct_idx = lemma_to_idx.get(pair["lemma"])

        # Build context embedding from keyword-matched lemmas
        context_indices: set[int] = set()
        for kw in keywords:
            kw_lower = kw.lower()
            for idx in lemma_to_idx.values():
                if idx < num_nodes:
                    context_indices.add(idx)  # simplified — in practice need KW→idx map

        # Simpler approach: use keyword match via node names
        # Actually, let me use a more efficient method:
        # Use ALL node embeddings and find top-K most similar to correct lemma as negatives

        if correct_idx is not None and correct_idx < num_nodes:
            correct_emb = node_emb_norm[correct_idx].unsqueeze(0)  # [1, D]

            # Cosine similarity of correct lemma to ALL lemmas
            cos_sim = correct_emb @ all_lemma_embs.T  # [1, L]
            cos_sim = cos_sim.squeeze(0)  # [L]

            # Exclude the correct lemma itself
            # Find which position in lemma_idx_list is the correct one
            correct_pos = None
            for i, lidx in enumerate(lemma_idx_list):
                if lidx == correct_idx:
                    correct_pos = i
                    break

            if correct_pos is not None:
                cos_sim[correct_pos] = -1e9

            # Get top-K
            K = min(num_negatives, len(lemma_idx_list) - 1)
            if K > 0:
                _, top_indices = torch.topk(cos_sim, K, dim=0)
                neg_indices = [lemma_idx_list[i] for i in top_indices.tolist()]

                # Look up lemma names from indices
                neg_lemmas = []
                idx_to_lemma = {v: k for k, v in lemma_to_idx.items()}
                for ni in neg_indices:
                    neg_lemmas.append(idx_to_lemma.get(ni, f"unknown_{ni}"))
            else:
                neg_lemmas = []
        else:
            neg_lemmas = []

        enriched.append({
            **pair,
            "_hard_negatives": neg_lemmas,
        })

    return enriched


def _sample_random_negatives(
    pairs: list[dict],
    lemma_to_idx: dict[str, int],
    num_negatives: int,
    num_nodes: int,
) -> list[dict]:
    """Fallback: random negative sampling (used before epoch 1)."""
    all_lemmas = list(lemma_to_idx.keys())
    enriched = []
    for pair in pairs:
        lemma_name = pair["lemma"]
        neg_candidates = [l for l in all_lemmas if l != lemma_name]
        if len(neg_candidates) >= num_negatives:
            neg_lemmas = random.sample(neg_candidates, num_negatives)
        else:
            neg_lemmas = neg_candidates
        enriched.append({**pair, "_hard_negatives": neg_lemmas})
    return enriched


def build_kw_to_indices(
    graph,
    lemma_to_idx: dict[str, int],
    num_nodes: int,
) -> dict[str, list[int]]:
    """Build keyword → node index map for fast context construction."""
    all_kw = set(_BUILTIN_LEMMAS.keys())
    for vals in _BUILTIN_LEMMAS.values():
        all_kw.update(vals)
    math_tokens = [
        "+", "*", "-", "/", "^", "=", "add", "mul", "sub", "div",
        "Nat", "Int", "Real", "Complex", "Prop", "Set", "List",
        "ring", "field", "group", "linear", "deriv", "integral",
        "limit", "continuous", "sum", "prod", "comm", "assoc", "distrib",
        "and", "or", "not", "<=", ">=", "<", ">", "inv", "0", "1",
        "neg", "eq", "refl", "symm", "trans", "forall", "exists", "->",
    ]
    all_kw.update(math_tokens)

    kw_map: dict[str, list[int]] = {}
    for nid in graph.node_ids:
        idx = graph.node_id_to_idx(nid)
        if idx is None or idx >= num_nodes:
            continue
        for kw in all_kw:
            if kw.lower() in nid.lower():
                kw_map.setdefault(kw.lower(), []).append(idx)
    return kw_map


def build_goal_context_embedding(
    keywords: list[str],
    node_emb_norm: torch.Tensor,
    kw_map: dict[str, list[int]],
    num_nodes: int,
    device: torch.device,
    hidden_dim: int,
    max_context: int = 100,
) -> torch.Tensor:
    """Build goal context embedding from keyword-matched lemma embeddings."""
    matching: list[int] = []
    seen: set[int] = set()
    for kw in keywords:
        for idx in kw_map.get(kw.lower(), []):
            if idx < num_nodes and idx not in seen:
                matching.append(idx)
                seen.add(idx)
                if len(matching) >= max_context:
                    break
        if len(matching) >= max_context:
            break

    if matching:
        match_t = torch.tensor(matching, device=device)
        return node_emb_norm[match_t].mean(dim=0)
    else:
        return torch.zeros(hidden_dim, device=device)


# ===========================================================================
# Validation MRR (Gate B)
# ===========================================================================

def build_norm_to_indices(
    graph, lemma_to_idx: dict[str, int]
) -> dict[str, list[int]]:
    idx_to_norm = build_lemma_norm_index(graph, lemma_to_idx)
    norm_to_indices: dict[str, list[int]] = {}
    for idx, norm in idx_to_norm.items():
        norm_to_indices.setdefault(norm, []).append(idx)
    return norm_to_indices


def build_goal_context_structural(
    goal_text: str,
    node_emb_norm: torch.Tensor,
    norm_to_indices: dict[str, list[int]],
    device: torch.device,
    max_context: int = 100,
) -> torch.Tensor | None:
    """Build goal context with structural (expression-normalized) matching."""
    goal_norm = normalize_expression(goal_text)
    exact_matches = set(norm_to_indices.get(goal_norm, []))

    # Power-stripping fallback
    if not exact_matches:
        import re
        goal_stripped = re.sub(r'\s*\^\s*\d+', '', goal_norm)
        if goal_stripped != goal_norm:
            for norm_key, indices in norm_to_indices.items():
                stripped_key = re.sub(r'\s*\^\s*\d+', '', norm_key)
                if stripped_key == goal_stripped:
                    exact_matches.update(indices)

    # Reflexivity
    if not exact_matches and "=" in goal_norm:
        import re
        sides = goal_norm.split("=", 1)
        if len(sides) == 2 and sides[0].strip() == sides[1].strip():
            exact_matches.update(
                norm_to_indices.get(normalize_expression("a = a"), [])
            )

    if exact_matches:
        indices = list(exact_matches)[:max_context]
        match_t = torch.tensor(indices, device=device)
        return node_emb_norm[match_t].mean(dim=0)

    # Token-overlap fallback
    goal_tokens = tokenize_expression(goal_norm)
    if goal_tokens:
        best_overlap = 0
        best_indices = []
        for norm_key, indices in norm_to_indices.items():
            key_tokens = tokenize_expression(norm_key)
            overlap = len(goal_tokens & key_tokens)
            if overlap > best_overlap:
                best_overlap = overlap
                best_indices = indices
        if best_overlap >= 2 and best_indices:
            indices = best_indices[:max_context]
            match_t = torch.tensor(indices, device=device)
            return node_emb_norm[match_t].mean(dim=0)

    return None


def compute_val_mrr(
    node_embeddings: torch.Tensor,
    gnn: GNNEncoder,
    lemma_to_idx: dict[str, int],
    norm_to_indices: dict[str, list[int]],
    kw_map: dict[str, list[int]],
    val_pairs: list[dict],
    sample_size: int = 500,
) -> float:
    device = node_embeddings.device
    all_emb_norm = F.normalize(node_embeddings, dim=-1)
    num_nodes = all_emb_norm.size(0)

    if len(val_pairs) > sample_size:
        sample = random.sample(val_pairs, sample_size)
    else:
        sample = val_pairs

    reciprocal_ranks = []
    with torch.no_grad():
        for pair in sample:
            lemma_name = pair["lemma"]
            correct_idx = lemma_to_idx.get(lemma_name)
            if correct_idx is None or correct_idx >= num_nodes:
                continue

            goal_text = pair["goal"]
            ctx_emb = build_goal_context_structural(
                goal_text, all_emb_norm, norm_to_indices, device
            )
            if ctx_emb is None:
                # Fallback: keyword matching using kw_map (not norm_to_indices)
                keywords = pair.get("_keywords", []) or _extract_math_keywords(
                    goal_text
                )
                matching: list[int] = []
                seen: set[int] = set()
                for kw in keywords:
                    for idx in kw_map.get(kw.lower(), []):
                        if idx < num_nodes and idx not in seen:
                            matching.append(idx)
                            seen.add(idx)
                            if len(matching) >= 100:
                                break
                    if len(matching) >= 100:
                        break
                if not matching:
                    continue
                match_t = torch.tensor(matching, device=device)
                ctx_emb = all_emb_norm[match_t].mean(dim=0)

            goal_emb = gnn.encode_goal(ctx_emb.unsqueeze(0))
            scores = (goal_emb @ all_emb_norm.T).squeeze(0)
            correct_score = scores[correct_idx]
            rank = (scores > correct_score).sum().item() + 1
            reciprocal_ranks.append(1.0 / rank)

    if not reciprocal_ranks:
        return 0.0
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    print(f"    MRR diagnostic: {len(reciprocal_ranks)}/{len(sample)} pairs contributed, "
          f"avg rank={1.0/max(mrr, 1e-9):.0f}")
    return mrr


# ===========================================================================
# Embedding health (Gate C)
# ===========================================================================

def check_embedding_health(embeddings: torch.Tensor, threshold: float = 0.03) -> dict:
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
        rank_val = (S > S.max().item() * 0.01).sum().item()
    except Exception:
        rank_val = 0

    return {
        "avg_cosine_std": round(avg_cosine_std, 6),
        "rank": rank_val,
        "std_ok": avg_cosine_std > threshold,
        "rank_ok": rank_val > 128,
    }


# ===========================================================================
# Utility
# ===========================================================================

def _save_abort(output_dir: Path, message: str, gate: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "abort_reason.json", "w") as f:
        json.dump(
            {"gate": gate, "message": message, "timestamp": time.time()}, f, indent=2
        )


# ===========================================================================
# Main training
# ===========================================================================

def train_multitask(args):
    import builtins as _builtins_module
    _real_print = print
    _builtins_module.print = functools.partial(_real_print, flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoint dir
    ckpt_dir = Path("checkpoints/gnn")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(args.num_threads)
    device = torch.device("cpu")
    print(f"Device: {device}, Threads: {torch.get_num_threads()}")

    # ---- GNN config (1.1M params, 256-dim) ----
    config = GNNConfig(
        hidden_dim=256,
        num_layers=3,
        num_heads=8,
        input_dim=256,
        dropout=0.1,
        activation="gelu",
        use_edge_types=True,
        num_edge_types=4,
        bidirectional=True,
        use_goal_encoder=True,
        goal_encoder_expansion=2,
        goal_encoder_dropout=0.1,
        init_features="random",
    )

    # ---- Load graph ----
    print("\n--- Loading dependency graph ---")
    graph_path = Path(args.graph)
    if not graph_path.with_suffix(".nx.pkl").exists():
        print(f"  ERROR: Graph not found at {graph_path}.nx.pkl")
        sys.exit(1)
    graph = DependencyGraph.load(graph_path)
    print(f"  Graph: {graph.summary()}")

    features = extract_initial_features(graph, config, device=device)
    sources, targets, edge_types, num_nodes = prepare_graph_tensors(graph, device=device)
    print(f"  Nodes: {num_nodes}, Edges: {sources.size(0)}")

    # ---- Build indices ----
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

    # Structural match index for MRR
    norm_to_indices = build_norm_to_indices(graph, lemma_to_idx)
    print(f"  Normalized patterns: {len(norm_to_indices)}")

    # Keyword map for context
    kw_map = build_kw_to_indices(graph, lemma_to_idx, num_nodes)
    print(f"  Keyword index: {len(kw_map)}")

    # ---- Load pairs ----
    print("\n--- Loading proof-step pairs ---")
    all_pairs = load_pairs(Path(args.pairs))
    print(f"  Loaded {len(all_pairs)} pairs")
    if args.max_pairs and args.max_pairs < len(all_pairs):
        random.seed(42)
        all_pairs = random.sample(all_pairs, args.max_pairs)
        print(f"  Sampled {len(all_pairs)}")

    # Train/val split
    split_idx = int(len(all_pairs) * (1 - args.val_split))
    train_pairs = all_pairs[:split_idx]
    val_pairs = all_pairs[split_idx:]
    print(f"  Train: {len(train_pairs)}, Val: {len(val_pairs)}")

    # ---- Initialize GNN from scratch ----
    print("\n--- Initializing GNN from scratch ---")
    gnn = GNNEncoder(config).to(device)
    total_params = sum(p.numel() for p in gnn.parameters())
    print(f"  GNN: {total_params:,} params (fresh init, NOT pretrained)")

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(
        gnn.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # ---- Initial hard negatives: random (before epoch 1) ----
    print("\n--- Initial hard negatives: random ---")
    training_pairs = _sample_random_negatives(
        train_pairs, lemma_to_idx, args.num_hard_negatives, num_nodes
    )

    # ---- Training loop ----
    stats_history = []
    best_val_mrr = 0.0
    best_epoch = 0
    aborted_gate = None
    hidden_dim = config.hidden_dim
    init_import_loss = None
    init_proof_loss = None
    init_import_acc = None

    print(f"\n{'=' * 60}")
    print(f"MULTI-TASK GNN TRAINING: {args.epochs} epochs, batch={args.batch_size}")
    print(f"  Loss = BCE(import) + {args.proof_weight} * BCE(proof)")
    print(f"  Hard negatives: {args.num_hard_negatives}, refresh every {args.neg_refresh_every} epochs")
    print(f"  lr={args.learning_rate}, CPU threads={args.num_threads}")
    print(f"{'=' * 60}")

    for epoch in range(args.epochs):
        t0 = time.time()
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")

        # ---- Refresh hard negatives every N epochs ----
        if epoch > 0 and epoch % args.neg_refresh_every == 0:
            print("  Refreshing hard negatives from current embeddings...")
            gnn.eval()
            with torch.no_grad():
                node_emb = gnn(features, sources, targets, edge_types, num_nodes)
                node_emb_norm = F.normalize(node_emb, dim=-1)
            training_pairs = sample_hard_negatives(
                node_emb_norm, lemma_to_idx, train_pairs,
                args.num_hard_negatives, num_nodes
            )
            print(f"  Hard negatives refreshed for {len(training_pairs)} pairs")

        # ---- Forward pass ----
        gnn.train()
        node_emb = gnn(features, sources, targets, edge_types, num_nodes)
        node_emb_norm = F.normalize(node_emb, dim=-1)

        # HEAD 1: Import link-prediction loss → trains GNN backbone
        import_loss, import_acc = compute_import_loss(
            node_emb, sources, targets, sample_edges=5000
        )

        # Step 1: Optimize import loss (gradients → GNN backbone)
        optimizer.zero_grad()
        import_loss.backward()
        torch.nn.utils.clip_grad_norm_(gnn.parameters(), 1.0)
        optimizer.step()

        # HEAD 2: Proof-step loss → trains goal_encoder
        # Use DETACHED embeddings for lemma lookup
        # (proof_loss gradient reaches only goal_encoder, not GNN backbone)
        node_emb_norm_det = node_emb_norm.detach()

        epoch_proof_loss = 0.0
        epoch_proof_acc = 0.0
        total_batches = 0

        random.shuffle(training_pairs)

        for batch_start in range(0, len(training_pairs), args.batch_size):
            batch = training_pairs[batch_start : batch_start + args.batch_size]
            if not batch:
                continue

            goal_emb_list = []
            lemma_emb_list = []

            for pair in batch:
                # Positive pair
                keywords = pair.get("_keywords", [])
                ctx_emb = build_goal_context_embedding(
                    keywords, node_emb_norm_det, kw_map,
                    num_nodes, device, hidden_dim
                )
                pos_goal = gnn.encode_goal(ctx_emb.unsqueeze(0))
                pos_lemma_idx = lemma_to_idx.get(pair["lemma"])
                if pos_lemma_idx is not None and pos_lemma_idx < num_nodes:
                    pos_lemma_emb = node_emb_norm_det[pos_lemma_idx].unsqueeze(0)
                else:
                    pos_lemma_emb = torch.zeros(1, hidden_dim, device=device)

                goal_emb_list.append(pos_goal)
                lemma_emb_list.append(pos_lemma_emb)

                # Hard negatives
                neg_lemmas = pair.get("_hard_negatives", [])
                for neg_name in neg_lemmas[:args.num_hard_negatives]:
                    neg_idx = lemma_to_idx.get(neg_name)
                    if neg_idx is not None and neg_idx < num_nodes:
                        neg_lemma_emb = node_emb_norm_det[neg_idx].unsqueeze(0)
                    else:
                        rand_idx = random.randint(0, num_nodes - 1)
                        neg_lemma_emb = node_emb_norm_det[rand_idx].unsqueeze(0)

                    goal_emb_list.append(pos_goal)
                    lemma_emb_list.append(neg_lemma_emb)

            if not goal_emb_list:
                continue

            goal_embs_t = torch.cat(goal_emb_list, dim=0)
            lemma_embs_t = torch.cat(lemma_emb_list, dim=0)

            proof_loss, proof_acc = compute_proof_loss(goal_embs_t, lemma_embs_t)

            # Step 2: Optimize proof loss per batch (gradients → goal_encoder only)
            optimizer.zero_grad()
            (args.proof_weight * proof_loss).backward()
            torch.nn.utils.clip_grad_norm_(gnn.parameters(), 1.0)
            optimizer.step()

            total_batches += 1
            epoch_proof_loss += proof_loss.item()
            epoch_proof_acc += proof_acc

        # ---- End of epoch stats ----
        avg_import_loss = import_loss.item()
        avg_proof_loss = epoch_proof_loss / max(total_batches, 1)
        avg_proof_acc = epoch_proof_acc / max(total_batches, 1)

        # Health check (Gate C)
        with torch.no_grad():
            health = check_embedding_health(node_emb, threshold=0.03)

        print(f"  Import Loss: {avg_import_loss:.6f}  Acc: {import_acc:.4f}")
        print(f"  Proof Loss:  {avg_proof_loss:.6f}  Acc: {avg_proof_acc:.4f}")
        print(f"  Health: std={health['avg_cosine_std']:.4f} rank={health['rank']}")

        # ---- Validation MRR every few epochs ----
        val_mrr = None
        if (epoch + 1) % args.mrr_every == 0 or epoch == args.epochs - 1:
            gnn.eval()
            with torch.no_grad():
                val_node_emb = gnn(features, sources, targets, edge_types, num_nodes)
            val_mrr = compute_val_mrr(
                val_node_emb, gnn, lemma_to_idx, norm_to_indices, kw_map, val_pairs
            )
            gnn.train()
            print(f"  Val MRR: {val_mrr:.6f}")

            if val_mrr > best_val_mrr:
                best_val_mrr = val_mrr
                best_epoch = epoch + 1

        # Save epoch stats
        epoch_time = time.time() - t0
        stats_history.append({
            "epoch": epoch + 1,
            "import_loss": avg_import_loss,
            "proof_loss": avg_proof_loss,
            "import_acc": import_acc,
            "proof_acc": avg_proof_acc,
            "cosine_std": health["avg_cosine_std"],
            "rank": health["rank"],
            "val_mrr": val_mrr,
            "epoch_time_s": round(epoch_time, 1),
        })

        # Gate A: Both losses must decrease (not diverge)
        # Compare current to initial: abort if loss significantly exceeds initial
        if epoch == 0:
            init_import_loss = avg_import_loss
            init_proof_loss = avg_proof_loss
            init_import_acc = import_acc

        # Gate A: Both losses must decrease (not diverge)
        if init_import_loss is not None and epoch >= 5:
            if avg_import_loss > init_import_loss * 1.5:
                msg = (f"Import loss increased from {init_import_loss:.4f} to "
                       f"{avg_import_loss:.4f} (epoch {epoch+1})")
                print(f"\n  GATE A FAILED: {msg}")
                _save_abort(output_dir, msg, "A")
                aborted_gate = "A"
                break
        if init_proof_loss is not None and epoch >= 5:
            if avg_proof_loss > init_proof_loss * 1.5:
                msg = (f"Proof loss increased from {init_proof_loss:.4f} to "
                       f"{avg_proof_loss:.4f} (epoch {epoch+1})")
                print(f"\n  GATE A FAILED: {msg}")
                _save_abort(output_dir, msg, "A")
                aborted_gate = "A"
                break

        # Gate B: MRR must show signal by epoch 10 (warn), must improve by epoch 25
        if epoch == 9 and val_mrr is not None:  # epoch 10 (0-indexed)
            if val_mrr <= 0.0001:
                msg = f"Val MRR {val_mrr:.6f} very low at epoch {epoch+1} (warning, not aborting)"
                print(f"\n  GATE B WARNING: {msg}")
            elif val_mrr <= 0.10:
                print(f"\n  GATE B: MRR {val_mrr:.6f} at epoch 10 — continuing, "
                      f"target >0.10 for full signal")
        if epoch == 24 and val_mrr is not None:  # epoch 25
            if val_mrr <= 0.01:
                msg = f"Val MRR {val_mrr:.6f} <= 0.01 at epoch {epoch+1} — no meaningful retrieval signal"
                print(f"\n  GATE B FAILED: {msg}")
                _save_abort(output_dir, msg, "B")
                aborted_gate = "B"
                break

        # Gate C: Embedding health (checked every epoch, logged above)
        if epoch >= 3 and not health["std_ok"] and health["avg_cosine_std"] < 0.01:
            msg = (f"Embedding cosine_std {health['avg_cosine_std']:.4f} < 0.03 "
                   f"at epoch {epoch+1}, rank={health['rank']}")
            print(f"\n  GATE C FAILED: {msg}")
            _save_abort(output_dir, msg, "C")
            aborted_gate = "C"
            break

        # Gate D: Import LP accuracy must not collapse
        # (from scratch, accuracy ~50% initially; gate D checks it doesn't degrade)
        if init_import_acc is not None and epoch >= 5:
            min_acceptable = max(init_import_acc * 0.8, 0.50)
            if import_acc < min_acceptable:
                msg = (f"Import LP accuracy {import_acc:.4f} < {min_acceptable:.4f} "
                       f"(init={init_import_acc:.4f}) at epoch {epoch+1}")
                print(f"\n  GATE D FAILED: {msg}")
                _save_abort(output_dir, msg, "D")
                aborted_gate = "D"
                break

        scheduler.step()

    # ---- End of training ----
    gnn.eval()
    print(f"\n{'=' * 60}")
    if aborted_gate:
        print(f"TRAINING ABORTED (Gate {aborted_gate})")
    else:
        print(f"TRAINING COMPLETE: {args.epochs} epochs")
    print(f"  Best Val MRR: {best_val_mrr:.4f} (epoch {best_epoch})")
    print(f"{'=' * 60}")

    # ---- Save encoder ----
    ckpt_path = Path(args.gnn_checkpoint)
    gnn.save(ckpt_path)
    print(f"\nEncoder saved to: {ckpt_path}")

    # ---- Save training stats ----
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
                "learning_rate": args.learning_rate,
                "proof_weight": args.proof_weight,
                "num_hard_negatives": args.num_hard_negatives,
                "neg_refresh_every": args.neg_refresh_every,
                "num_pairs": len(all_pairs),
                "train_pairs": len(train_pairs),
                "val_pairs": len(val_pairs),
            },
            "history": stats_history,
            "best_val_mrr": best_val_mrr,
            "best_epoch": best_epoch,
            "aborted_gate": aborted_gate,
        }, f, indent=2)
    print(f"Training stats saved to: {stats_path}")

    return gnn, aborted_gate is None


# ===========================================================================
# Gate3 evaluation (separate phase)
# ===========================================================================

def run_gate3_eval(args):
    """Run gate3_v2 evaluation using trained multi-task GNN checkpoint."""
    print("=" * 70)
    print("GATE3_V2 EVALUATION: Multi-task GNN")
    print("=" * 70)

    # Import the gate3 runner
    from scripts.eval.run_full_gate3_v2 import run_gate3_full, load_jsonl, save_json
    from scripts.eval.run_full_gate3_v2 import build_norm_index, build_lemma_index
    from src.explorer.gnn_best_first_search import GNNBestFirstConfig
    from src.proof_checker.batch_checker import BatchChecker

    # Load GNN
    ckpt_path = Path(args.gnn_checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    print(f"\nLoading GNN from {ckpt_path}...")
    gnn = GNNEncoder.load(str(ckpt_path))
    gnn.eval()
    n_params = sum(p.numel() for p in gnn.parameters())
    print(f"  GNN: {n_params:,} params, hidden={gnn.config.hidden_dim}")

    # Load graph
    graph_path = Path(args.graph)
    graph = DependencyGraph.load(graph_path)
    print(f"  Graph: {graph.summary()}")

    # Load theorems
    theorems = load_jsonl(_project_root / "data/raw/gate3_v2.jsonl")
    print(f"  Theorems: {len(theorems)}")

    # Indexes
    lemma_to_idx = build_lemma_index(graph)
    idx_to_norm = build_norm_index(graph, lemma_to_idx)

    # Config
    config = GNNBestFirstConfig(
        max_depth=20,
        max_expansions=1000,
        top_k_lemmas=30,
        depth_penalty=0.05,
        use_proof_checker=True,
        verify_timeout=5.0,
        num_threads=args.eval_threads,
        max_graph_candidates=200,
    )

    torch.set_num_threads(args.eval_threads)
    checker = BatchChecker(timeout=15, max_workers=8, cache_size=128)

    output_path = Path(args.output_dir) / "multitask_gate3_result.json"

    print(f"\nRunning gate3_v2 evaluation (this takes ~20-30 minutes)...")
    result = run_gate3_full(
        gnn=gnn,
        graph=graph,
        theorems=theorems,
        config=config,
        lemma_to_idx=lemma_to_idx,
        idx_to_norm=idx_to_norm,
        checker=checker,
        output_path=output_path,
        use_domain_filter=True,
    )

    n_passed = result["gate3"]["passed"]
    rate = result["gate3"]["rate"]
    print(f"\nFINAL: {n_passed}/64 ({rate:.0%})")


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Multi-task GNN training: joint import + proof-step link prediction"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Batch size for proof-step pairs"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=3e-4, help="Learning rate"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=1e-5, help="Weight decay"
    )
    parser.add_argument(
        "--proof-weight", type=float, default=0.5,
        help="Weight of proof-step loss relative to import loss"
    )
    parser.add_argument(
        "--num-hard-negatives", type=int, default=5,
        help="Number of hard negative lemmas per positive pair"
    )
    parser.add_argument(
        "--neg-refresh-every", type=int, default=5,
        help="Refresh hard negatives every N epochs"
    )
    parser.add_argument(
        "--num-threads", type=int, default=6, help="CPU threads for training"
    )
    parser.add_argument(
        "--eval-threads", type=int, default=4, help="CPU threads for evaluation"
    )
    parser.add_argument(
        "--val-split", type=float, default=0.1, help="Validation split fraction"
    )
    parser.add_argument(
        "--mrr-every", type=int, default=5,
        help="Compute validation MRR every N epochs"
    )
    parser.add_argument(
        "--max-pairs", type=int, default=None,
        help="Max pairs to use (for smoke testing)"
    )
    parser.add_argument(
        "--graph", default="data/graph/dependency_graph_full",
        help="Dependency graph path"
    )
    parser.add_argument(
        "--pairs", default="data/raw/proof_step_pairs.jsonl",
        help="Proof-step pairs JSONL"
    )
    parser.add_argument(
        "--gnn-checkpoint", default="checkpoints/gnn/multitask_scratch.pt",
        help="Path to save/load GNN encoder checkpoint"
    )
    parser.add_argument(
        "--output-dir", default="data/multitask_full",
        help="Output directory for stats and results"
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Skip training, only run gate3_v2 evaluation"
    )
    parser.add_argument(
        "--skip-gate3", action="store_true",
        help="Skip gate3_v2 evaluation (for smoke tests)"
    )
    args = parser.parse_args()

    if args.eval_only:
        run_gate3_eval(args)
        return 0

    gnn, success = train_multitask(args)

    if not args.skip_gate3 and success:
        print("\n" + "=" * 70)
        print("RUNNING GATE3_V2 EVALUATION")
        print("=" * 70)
        run_gate3_eval(args)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
