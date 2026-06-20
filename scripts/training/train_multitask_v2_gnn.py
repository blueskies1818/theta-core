#!/usr/bin/env python3
"""
MULTI-TASK v2: Amplified proof signal with InfoNCE ranking loss.

Same multi-task architecture as v1 (GNN from scratch, two heads) but with
amplified proof signal to fix the MRR 0.0054 problem.

KEY CHANGE: Replace BCE on the proof head with InfoNCE ranking loss.
BCE averages away hard negatives -- ranking loss forces the correct lemma
above EVERY negative in the batch.

ARCHITECTURE:
  GNN encoder (1.1M, GAT, 256-dim) -- trained from scratch
  Head 1: Import link prediction -- BCE loss, weight 1.0
  Head 2: Proof utility scoring -- InfoNCE ranking loss, weight 2.0

PROOF HEAD RANKING LOSS (InfoNCE, temperature=1.0):
  proof_scores = dot(goal_emb, lemma_emb)
  ranking_loss = InfoNCE(proof_scores, labels, t=1.0)
  Total = 1.0 * BCE_import + proof_weight * InfoNCE_proof

ADJUSTABLE KNOBS:
  --proof-weight: start 2.0. Increase to 5.0 if MRR < 0.01.
    Decrease to 1.0 if import accuracy drops below 55%.
  --lr: start 3e-4
  --neg-refresh-every: 3 epochs (faster feedback)
  --batch-size: 256

Usage:
  # SMOKE TEST (10 epochs, 5000 pairs):
  python scripts/training/train_multitask_v2_gnn.py \
      --epochs 10 --max-pairs 5000 --output-dir data/multitask_v2_smoke \
      --skip-gate3

  # FULL TRAINING (50 epochs, all pairs):
  python scripts/training/train_multitask_v2_gnn.py \
      --epochs 50 --output-dir data/multitask_v2_full

  # Evaluate on gate3_v2:
  python scripts/training/train_multitask_v2_gnn.py \
      --eval-only --gnn-checkpoint checkpoints/gnn/multitask_v2.pt \
      --output-dir data/multitask_v2_full
"""

import argparse
import functools
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
from scripts.eval.eval_gnn_prover import (
    normalize_expression,
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
# Head 1: Import link-prediction loss (BCE)
# ===========================================================================

def compute_import_loss(
    node_embeddings: torch.Tensor,
    sources: torch.Tensor,
    targets: torch.Tensor,
    num_negatives: int = 5,
    sample_edges: int = 5000,
) -> tuple[torch.Tensor, float]:
    """BCE loss on import edges."""
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
        pos_correct = (pos_scores > 0).float().mean().item()
        neg_correct = (neg_scores <= 0).float().mean().item()
        acc = (pos_correct + neg_correct) / 2.0

    return loss, acc


# ===========================================================================
# Head 2: Proof InfoNCE loss (ranking loss, temperature=1.0)
# ===========================================================================

def compute_proof_infonce(
    goal_embs: torch.Tensor,
    lemma_embs: torch.Tensor,
) -> torch.Tensor:
    """InfoNCE ranking loss on proof-step batch.

    goal_embs: [B, D] L2-normalized goal embeddings.
    lemma_embs: [B, D] L2-normalized lemma embeddings.

    Returns scalar InfoNCE loss with temperature=1.0.
    Temperature 1.0 prevents sharp softmax that caused MRR collapse before.
    Random-chance loss = ln(batch_size) e.g. ln(256) ~ 5.545.
    """
    temperature_inv = 1.0  # temperature = 1.0
    return compute_infonce_loss(goal_embs, lemma_embs, temperature_inv)


# ===========================================================================
# Keyword and context helpers
# ===========================================================================

def build_kw_to_indices(
    graph,
    lemma_to_idx: dict[str, int],
    num_nodes: int,
) -> dict[str, list[int]]:
    """Build keyword -> node index map."""
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
        name_lower = nid.lower()
        for kw in all_kw:
            if kw.lower() in name_lower:
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

    if not exact_matches:
        import re
        goal_stripped = re.sub(r'\s*\^\s*\d+', '', goal_norm)
        if goal_stripped != goal_norm:
            for norm_key, indices in norm_to_indices.items():
                stripped_key = re.sub(r'\s*\^\s*\d+', '', norm_key)
                if stripped_key == goal_stripped:
                    exact_matches.update(indices)

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
    """Compute Mean Reciprocal Rank on validation pairs."""
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
    print(f"    MRR diagnostic: {len(reciprocal_ranks)}/{len(sample)} pairs, "
          f"avg rank={1.0/max(mrr, 1e-9):.0f}")
    return mrr


# ===========================================================================
# Embedding health (Gate C)
# ===========================================================================

def check_embedding_health(embeddings: torch.Tensor, threshold: float = 0.03) -> dict:
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
        rank_val = (S > S.max().item() * 0.01).sum().item()
    except Exception:
        rank_val = 0

    return {
        "avg_cosine_std": round(avg_cosine_std, 6),
        "rank": rank_val,
        "std_ok": avg_cosine_std > threshold,
        "rank_ok": rank_val > 200,
    }


def _save_abort(output_dir: Path, message: str, gate: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "abort_reason.json", "w") as f:
        json.dump(
            {"gate": gate, "message": message, "timestamp": time.time()}, f, indent=2
        )


# ===========================================================================
# Main training
# ===========================================================================

def train_multitask_v2(args):
    import builtins as _builtins_module
    import gc as _gc
    _real_print = print
    _builtins_module.print = functools.partial(_real_print, flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = Path("checkpoints/gnn")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(args.num_threads)
    device = torch.device("cpu")
    print(f"Device: {device}, Threads: {torch.get_num_threads()}")

    # ---- GNN config (1.1M params, 256-dim) ----
    config = GNNConfig(
        hidden_dim=256,
        num_layers=2,
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
    
    # Apply domain filter if specified
    if args.domain:
        print(f"  Filtering to domain: {args.domain}...")
        graph = graph.domain_subgraph(args.domain)
    
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

    norm_to_indices = build_norm_to_indices(graph, lemma_to_idx)
    print(f"  Normalized patterns: {len(norm_to_indices)}")

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

    split_idx = int(len(all_pairs) * (1 - args.val_split))
    train_pairs = all_pairs[:split_idx]
    val_pairs = all_pairs[split_idx:]
    print(f"  Train: {len(train_pairs)}, Val: {len(val_pairs)}")

    # ---- Initialize GNN from scratch ----
    print("\n--- Initializing GNN from scratch ---")
    gnn = GNNEncoder(config).to(device)
    total_params = sum(p.numel() for p in gnn.parameters())
    print(f"  GNN: {total_params:,} params (fresh init)")

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(
        gnn.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
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

    random_infonce_baseline = math.log(args.batch_size)
    print(f"\n{'=' * 60}")
    print(f"MULTI-TASK v2 GNN TRAINING: {args.epochs} epochs, batch={args.batch_size}")
    print(f"  Loss = BCE(import) + {args.proof_weight} * InfoNCE(proof, t=1.0)")
    print(f"  Hard negatives: in-batch ({args.batch_size - 1} per pair), refresh every {args.neg_refresh_every} epochs")
    print(f"  Random InfoNCE baseline: ln({args.batch_size}) = {random_infonce_baseline:.4f}")
    print(f"  lr={args.learning_rate}, CPU threads={args.num_threads}")
    print(f"{'=' * 60}")

    for epoch in range(args.epochs):
        t0 = time.time()
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")

        # ---- Forward pass (once per epoch) ----
        gnn.train()
        node_emb = gnn(features, sources, targets, edge_types, num_nodes)
        node_emb_norm = F.normalize(node_emb, dim=-1)

        # ---- HEAD 1: Import BCE loss -> GNN backbone ----
        import_loss, import_acc = compute_import_loss(
            node_emb, sources, targets, sample_edges=5000
        )

        optimizer.zero_grad()
        import_loss.backward()
        torch.nn.utils.clip_grad_norm_(gnn.parameters(), 1.0)
        optimizer.step()

        # ---- HEAD 2: Proof InfoNCE loss -> goal_encoder ----
        # Use DETACHED embeddings for lemma lookup (gradient goes to goal_encoder only)
        node_emb_norm_det = node_emb_norm.detach()

        epoch_proof_loss = 0.0
        total_batches = 0

        random.shuffle(train_pairs)

        for batch_start in range(0, len(train_pairs), args.batch_size):
            batch = train_pairs[batch_start : batch_start + args.batch_size]
            if len(batch) < 2:
                continue

            goal_emb_list = []
            lemma_emb_list = []

            for pair in batch:
                # Goal: keyword context -> average -> goal_encoder -> normalized
                keywords = pair.get("_keywords", [])
                ctx_emb = build_goal_context_embedding(
                    keywords, node_emb_norm_det, kw_map,
                    num_nodes, device, hidden_dim
                )
                goal_emb = gnn.encode_goal(ctx_emb.unsqueeze(0))
                goal_emb_list.append(goal_emb)

                # Lemma: direct node embedding lookup (DETACHED)
                lemma_idx = lemma_to_idx.get(pair["lemma"])
                if lemma_idx is not None and lemma_idx < num_nodes:
                    lemma_emb = node_emb_norm_det[lemma_idx].unsqueeze(0)
                else:
                    lemma_emb = torch.zeros(1, hidden_dim, device=device)
                lemma_emb_list.append(lemma_emb)

            goal_embs = torch.cat(goal_emb_list, dim=0)
            lemma_embs = torch.cat(lemma_emb_list, dim=0)

            # Normalize + InfoNCE
            goal_embs_norm = F.normalize(goal_embs, dim=-1)
            lemma_embs_norm = F.normalize(lemma_embs, dim=-1)

            proof_loss = compute_proof_infonce(goal_embs_norm, lemma_embs_norm)

            optimizer.zero_grad()
            (args.proof_weight * proof_loss).backward()
            torch.nn.utils.clip_grad_norm_(gnn.parameters(), 1.0)
            optimizer.step()

            total_batches += 1
            epoch_proof_loss += proof_loss.item()

        scheduler.step()

        # ---- End of epoch stats (must come before cleanup) ----
        avg_import_loss = import_loss.item()
        avg_proof_loss = epoch_proof_loss / max(total_batches, 1)

        with torch.no_grad():
            health = check_embedding_health(node_emb, threshold=0.03)

        # ---- Explicit memory cleanup ----
        del node_emb, node_emb_norm
        _gc.collect()

        print(f"  Import Loss: {avg_import_loss:.6f}  Acc: {import_acc:.4f}")
        print(f"  Proof Loss:  {avg_proof_loss:.6f}  (random baseline ln({args.batch_size})={math.log(args.batch_size):.3f})")
        print(f"  Health: std={health['avg_cosine_std']:.4f} rank={health['rank']}")

        # ---- Validation MRR ----
        val_mrr = None
        if (epoch + 1) % args.mrr_every == 0 or epoch == args.epochs - 1 or epoch == 9:
            gnn.eval()
            with torch.no_grad():
                val_node_emb = gnn(features, sources, targets, edge_types, num_nodes)
            val_mrr = compute_val_mrr(
                val_node_emb, gnn, lemma_to_idx, norm_to_indices, kw_map, val_pairs
            )
            print(f"  Val MRR: {val_mrr:.6f}")

            if val_mrr > best_val_mrr:
                best_val_mrr = val_mrr
                best_epoch = epoch + 1

        epoch_time = time.time() - t0
        stats_history.append({
            "epoch": epoch + 1,
            "import_loss": avg_import_loss,
            "proof_loss": avg_proof_loss,
            "import_acc": import_acc,
            "cosine_std": health["avg_cosine_std"],
            "rank": health["rank"],
            "val_mrr": val_mrr,
            "epoch_time_s": round(epoch_time, 1),
        })

        # ---- GATE A: Import accuracy > 50% ----
        if init_import_loss is None:
            init_import_loss = avg_import_loss
            init_proof_loss = avg_proof_loss
            init_import_acc = import_acc

        if import_acc < 0.50:
            msg = (f"Import LP accuracy {import_acc:.4f} < 0.50 "
                   f"at epoch {epoch+1}")
            print(f"\n  GATE A FAILED: {msg}")
            _save_abort(output_dir, msg, "A")
            aborted_gate = "A"
            break

        # ---- GATE B: MRR staged targets ----
        if epoch == 9 and val_mrr is not None:
            if val_mrr <= 0.01:
                if args.proof_weight < 5.0:
                    old = args.proof_weight
                    args.proof_weight = 5.0
                    print(f"  AUTO-ADJUST: proof_weight {old} -> {args.proof_weight}")
                else:
                    msg = f"Val MRR {val_mrr:.6f} <= 0.01 at epoch 10 with proof_weight=5.0"
                    print(f"\n  GATE B FAILED: {msg}")
                    _save_abort(output_dir, msg, "B")
                    aborted_gate = "B"
                    break
            elif val_mrr <= 0.10:
                print(f"  GATE B: MRR {val_mrr:.6f} at epoch 10 -- below 0.10 target")
            else:
                print(f"  GATE B: MRR {val_mrr:.6f} at epoch 10 -- PASS (>=0.10)")

        if epoch == 29 and val_mrr is not None:
            if val_mrr <= 0.30:
                print(f"  GATE B: MRR {val_mrr:.6f} at epoch 30 -- below 0.30 target")
            else:
                print(f"  GATE B: MRR {val_mrr:.6f} at epoch 30 -- PASS (>=0.30)")

        # ---- GATE C: Embedding rank > 200, cosine_std > 0.03 ----
        if epoch >= 3:
            if health["rank"] <= 200:
                msg = f"Embedding rank {health['rank']} <= 200 at epoch {epoch+1}"
                print(f"\n  GATE C FAILED: {msg}")
                _save_abort(output_dir, msg, "C")
                aborted_gate = "C"
                break
            if health["avg_cosine_std"] <= 0.03:
                msg = (f"Embedding cosine_std {health['avg_cosine_std']:.4f} <= 0.03 "
                       f"at epoch {epoch+1}")
                print(f"\n  GATE C FAILED: {msg}")
                _save_abort(output_dir, msg, "C")
                aborted_gate = "C"
                break

        # ---- GATE D: Proof loss must decrease (not plateau at random) ----
        # During smoke test (first 10 epochs): WARNING only
        # Full training: abort
        if init_proof_loss is not None and epoch >= 5:
            plateau = random_infonce_baseline * 0.98
            if avg_proof_loss >= plateau:
                msg = (f"Proof loss {avg_proof_loss:.4f} plateaued at random "
                       f"({random_infonce_baseline:.3f}) at epoch {epoch+1}")
                if args.max_pairs and args.max_pairs <= 5000:
                    print(f"\n  GATE D WARNING (smoke test): {msg} — continuing")
                else:
                    print(f"\n  GATE D FAILED: {msg}")
                    _save_abort(output_dir, msg, "D")
                    aborted_gate = "D"
                    break

        # ---- Auto-adjust proof weight based on import accuracy ----
        # DISABLED for smoke test: want to test sustained proof_weight=5.0
        if False and import_acc < 0.55 and args.proof_weight > 1.0:
            old = args.proof_weight
            args.proof_weight = max(1.0, args.proof_weight - 0.5)
            print(f"  AUTO-ADJUST: proof_weight {old} -> {args.proof_weight} "
                  f"(import acc {import_acc:.4f} < 0.55)")

        # ---- Save checkpoint every 10 epochs ----
        if (epoch + 1) % 10 == 0:
            ckpt_path = ckpt_dir / f"multitask_v2_epoch_{epoch+1:03d}.pt"
            gnn.save(ckpt_path)
            print(f"  Checkpoint: {ckpt_path.name}")

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
                "version": "v2",
                "hidden_dim": config.hidden_dim,
                "num_layers": config.num_layers,
                "num_heads": config.num_heads,
                "total_params": total_params,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "proof_weight": args.proof_weight,
                "neg_refresh_every": args.neg_refresh_every,
                "num_pairs": len(all_pairs),
                "train_pairs": len(train_pairs),
                "val_pairs": len(val_pairs),
                "proof_loss_type": "InfoNCE",
                "temperature": 1.0,
            },
            "history": stats_history,
            "best_val_mrr": best_val_mrr,
            "best_epoch": best_epoch,
            "aborted_gate": aborted_gate,
        }, f, indent=2)
    print(f"Training stats saved to: {stats_path}")

    return gnn, aborted_gate is None


# ===========================================================================
# Gate3 evaluation
# ===========================================================================

def run_gate3_eval(args):
    """Run gate3_v2 evaluation using trained multi-task GNN checkpoint."""
    print("=" * 70)
    print("GATE3_V2 EVALUATION: Multi-task V2 GNN")
    print("=" * 70)

    from scripts.eval.run_full_gate3_v2 import run_gate3_full, load_jsonl
    from scripts.eval.run_full_gate3_v2 import build_norm_index, build_lemma_index
    from src.explorer.gnn_best_first_search import GNNBestFirstConfig
    from src.proof_checker.batch_checker import BatchChecker

    ckpt_path = Path(args.gnn_checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    print(f"\nLoading GNN from {ckpt_path}...")
    gnn = GNNEncoder.load(str(ckpt_path))
    gnn.eval()
    n_params = sum(p.numel() for p in gnn.parameters())
    print(f"  GNN: {n_params:,} params, hidden={gnn.config.hidden_dim}")

    graph_path = Path(args.graph)
    graph = DependencyGraph.load(graph_path)
    print(f"  Graph: {graph.summary()}")

    theorems = load_jsonl(_project_root / "data/raw/gate3_v2.jsonl")
    print(f"  Theorems: {len(theorems)}")

    lemma_to_idx = build_lemma_index(graph)
    idx_to_norm = build_norm_index(graph, lemma_to_idx)

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

    output_path = Path(args.output_dir) / "multitask_v2_gate3.json"

    print(f"\nRunning gate3_v2 evaluation (~20-30 min)...")
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
        description="Multi-task V2 GNN: import BCE + proof InfoNCE ranking loss"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--proof-weight", type=float, default=2.0,
                        help="Weight of InfoNCE proof loss (start 2.0, increase to 5.0 if needed)")
    parser.add_argument("--neg-refresh-every", type=int, default=3,
                        help="Refresh hard negative candidates every N epochs")
    parser.add_argument("--num-threads", type=int, default=6)
    parser.add_argument("--eval-threads", type=int, default=4)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--mrr-every", type=int, default=5,
                        help="Compute validation MRR every N epochs")
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--graph", default="data/graph/dependency_graph_full")
    parser.add_argument("--pairs", default="data/raw/proof_step_pairs.jsonl")
    parser.add_argument("--gnn-checkpoint", default="checkpoints/gnn/multitask_v2.pt")
    parser.add_argument("--output-dir", default="data/multitask_v2_full")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--domain", default=None,
                        help="Domain filter (e.g., Algebra) — reduces graph size")
    parser.add_argument("--skip-gate3", action="store_true")
    args = parser.parse_args()

    if args.eval_only:
        run_gate3_eval(args)
        return 0

    gnn, success = train_multitask_v2(args)

    if not args.skip_gate3 and success:
        print("\n" + "=" * 70)
        print("RUNNING GATE3_V2 EVALUATION")
        print("=" * 70)
        run_gate3_eval(args)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
