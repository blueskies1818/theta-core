#!/usr/bin/env python3
"""Full GNN fine-tuning with TRIPLET-ONLY margin loss (NO InfoNCE).

Loads pretrained GNN (1.1M params), fine-tunes ALL parameters using
TripletMarginLoss ONLY. Uses normalized-expression structural matching
for goal context and validation MRR (matching eval pipeline).

Loss = triplet_margin_loss + lambda * link_prediction

Usage:
    python scripts/training/train_gnn_triplet_only.py \
        --epochs 2 --max-pairs 1000 --output-dir data/gnn_triplet_only_smoke
"""

import argparse
import json
import random
import re
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


# ---------------------------------------------------------------------------
# Assertion: NO INFONCE
# ---------------------------------------------------------------------------

def _assert_no_infonce():
    import inspect, textwrap
    triplet_src = inspect.getsource(compute_triplet_loss_batch)
    triplet_dedent = textwrap.dedent(triplet_src)
    assert "infonce" not in triplet_dedent.lower(), "FATAL: InfoNCE in triplet loss!"
    assert "cross_entropy" not in triplet_dedent, "FATAL: cross_entropy in triplet loss!"
    train_src = inspect.getsource(train_gnn_triplet_only)
    train_dedent = textwrap.dedent(train_src)
    assert "compute_infonce" not in train_dedent, "FATAL: compute_infonce in training!"
    assert "infonce_loss" not in train_dedent.lower(), "FATAL: infonce_loss in training!"
    assert "F.cross_entropy" not in train_dedent, "FATAL: F.cross_entropy in training!"
    print("CONFIRMED: Triplet-only loss, no InfoNCE")
    print("CONFIRMED: No cross_entropy in main loss")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_pairs(data_path: Path) -> list[dict]:
    pairs = []
    with open(data_path) as f:
        for line in f:
            pair = json.loads(line)
            pair['_keywords'] = _extract_math_keywords(pair['goal'])
            pairs.append(pair)
    return pairs


# ---------------------------------------------------------------------------
# Triplet loss (cosine-distance, manual)
# ---------------------------------------------------------------------------

def compute_triplet_loss_batch(
    goal_embs: torch.Tensor,
    lemma_embs: torch.Tensor,
    margin: float = 0.3,
    max_hard_negatives: int = 5,
) -> tuple[torch.Tensor, float, float]:
    B = goal_embs.size(0)
    if B < 2:
        return torch.tensor(0.0, device=goal_embs.device), 0.0, 0.0
    K = min(max_hard_negatives, B - 1)
    sim = goal_embs @ lemma_embs.T
    pos_scores = sim.diag()
    mask = ~torch.eye(B, dtype=torch.bool, device=sim.device)
    sim_masked = sim.masked_fill(~mask, -1e9)
    neg_scores_topk, neg_indices = torch.topk(sim_masked, K, dim=1)
    neg_embs = lemma_embs[neg_indices]
    pos_cos = pos_scores
    neg_cos = F.cosine_similarity(goal_embs.unsqueeze(1), neg_embs, dim=-1).mean(dim=1)
    triplet_loss = F.relu(margin - pos_cos + neg_cos).mean()
    return triplet_loss, pos_scores.mean().item(), neg_cos.mean().item()


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
    pos_loss = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores))
    neg_loss = F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))
    return (pos_loss + neg_loss) / 2.0


# ---------------------------------------------------------------------------
# Embedding health checks (Gate C)
# ---------------------------------------------------------------------------

def check_embedding_health(embeddings: torch.Tensor) -> dict:
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
    return {"avg_cosine_std": round(avg_cosine_std, 6), "rank": rank_val,
            "std_ok": avg_cosine_std > 0.05, "rank_ok": rank_val > 128}


# ---------------------------------------------------------------------------
# Goal context builder (structural matching, matching eval pipeline)
# ---------------------------------------------------------------------------

def build_goal_context(
    goal_text: str,
    node_emb_norm: torch.Tensor,
    norm_to_indices: dict[str, list[int]],
    device: torch.device,
    max_context: int = 100,
) -> torch.Tensor | None:
    """Build goal context embedding using normalized-expression matching.

    Returns [D] context embedding or None if no matches found.
    """
    goal_norm = normalize_expression(goal_text)

    # Exact matches
    exact_matches = set(norm_to_indices.get(goal_norm, []))

    # Power-stripping fallback
    if not exact_matches:
        goal_stripped = re.sub(r'\s*\^\s*\d+', '', goal_norm)
        if goal_stripped != goal_norm:
            for norm_key, indices in norm_to_indices.items():
                stripped_key = re.sub(r'\s*\^\s*\d+', '', norm_key)
                if stripped_key == goal_stripped:
                    exact_matches.update(indices)

    # Reflexivity
    if not exact_matches and "=" in goal_norm:
        sides = goal_norm.split("=", 1)
        if len(sides) == 2 and sides[0].strip() == sides[1].strip():
            exact_matches.update(norm_to_indices.get(normalize_expression("a = a"), []))

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


# ---------------------------------------------------------------------------
# Validation MRR (Gate B) — uses structural matching
# ---------------------------------------------------------------------------

def compute_val_mrr(
    node_embeddings: torch.Tensor,
    gnn: GNNEncoder,
    lemma_to_idx: dict[str, int],
    norm_to_indices: dict[str, list[int]],
    val_pairs: list[dict],
    sample_size: int = 500,
) -> float:
    device = node_embeddings.device
    all_emb_norm = F.normalize(node_embeddings, dim=-1)
    num_nodes = all_emb_norm.size(0)
    hidden_dim = all_emb_norm.size(1)

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
            ctx_emb = build_goal_context(
                goal_text, all_emb_norm, norm_to_indices, device
            )
            if ctx_emb is None:
                # Fallback: keyword matching
                keywords = pair.get('_keywords', []) or _extract_math_keywords(goal_text)
                matching: set[int] = set()
                for kw in keywords:
                    for idx in norm_to_indices.get(kw.lower(), []):
                        if idx < num_nodes:
                            matching.add(idx)
                if not matching:
                    continue
                match_list = list(matching)[:100]
                match_t = torch.tensor(match_list, device=device)
                ctx_emb = all_emb_norm[match_t].mean(dim=0)

            goal_emb = gnn.encode_goal(ctx_emb.unsqueeze(0))
            scores = (goal_emb @ all_emb_norm.T).squeeze(0)
            correct_score = scores[correct_idx]
            rank = (scores > correct_score).sum().item() + 1
            reciprocal_ranks.append(1.0 / rank)

    if not reciprocal_ranks:
        return 0.0
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


# ---------------------------------------------------------------------------
# Build norm → indices hash map
# ---------------------------------------------------------------------------

def build_norm_to_indices(
    graph, lemma_to_idx: dict[str, int]
) -> dict[str, list[int]]:
    """Build map from normalized conclusion → list of node indices."""
    idx_to_norm = build_lemma_norm_index(graph, lemma_to_idx)
    norm_to_indices: dict[str, list[int]] = {}
    for idx, norm in idx_to_norm.items():
        norm_to_indices.setdefault(norm, []).append(idx)
    return norm_to_indices


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

def train_gnn_triplet_only(args):
    import functools, builtins
    _real_print = print
    builtins.print = functools.partial(_real_print, flush=True)

    _assert_no_infonce()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(args.num_threads)
    device = torch.device("cpu")
    print(f"Device: {device}, Threads: {torch.get_num_threads()}")

    # ---- Load pretrained GNN ----
    print("\n--- Loading GNN (pretrained) ---")
    gnn = GNNEncoder.load(args.gnn_checkpoint)
    trainable = sum(p.numel() for p in gnn.parameters())
    goal_enc = sum(p.numel() for p in gnn.goal_encoder.parameters()) if gnn.goal_encoder else 0
    print(f"  GNN total: {trainable:,}, Goal encoder: {goal_enc:,}")
    hidden_dim = gnn.config.hidden_dim
    print(f"  hidden_dim={hidden_dim}, layers={gnn.config.num_layers}")

    # ---- Load graph ----
    print("\n--- Loading dependency graph ---")
    graph_path = Path(args.graph)
    if not graph_path.with_suffix(".nx.pkl").exists():
        print(f"  ERROR: Graph not found")
        sys.exit(1)
    graph = DependencyGraph.load(graph_path)
    print(f"  Graph: {graph.summary()}")

    features = extract_initial_features(graph, gnn.config, device=device)
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

    # Build structural matching index
    print("--- Building structural match index ---")
    norm_to_indices = build_norm_to_indices(graph, lemma_to_idx)
    print(f"  Normalized patterns: {len(norm_to_indices)}")

    # Build keyword map (fallback)
    all_kw = set(_BUILTIN_LEMMAS.keys())
    for vals in _BUILTIN_LEMMAS.values():
        all_kw.update(vals)
    math_tokens = ["+","*","-","/","^","=","add","mul","sub","div","Nat","Int","Real",
        "Complex","Prop","Set","List","ring","field","group","linear","deriv","integral",
        "limit","continuous","sum","prod","comm","assoc","distrib","and","or","not",
        "<=",">=","<",">","inv","0","1","neg","eq","refl","symm","trans","forall","exists","->"]
    all_kw.update(math_tokens)
    kw_map: dict[str, list[int]] = {}
    for nid in graph.node_ids:
        idx = graph.node_id_to_idx(nid)
        if idx is None: continue
        for kw in all_kw:
            if kw.lower() in nid.lower():
                kw_map.setdefault(kw.lower(), []).append(idx)
    print(f"  Keywords: {len(kw_map)}")

    # ---- Baseline stats ----
    print("\n--- Gate A: Baseline link-prediction ---")
    gnn.eval()
    with torch.no_grad():
        baseline_emb = gnn(features, sources, targets, edge_types, num_nodes)
    baseline_lp = compute_link_prediction_loss(baseline_emb, sources, targets, sample_edges=2000)
    gate_a_threshold = baseline_lp.item() * 1.30
    print(f"  Baseline LP: {baseline_lp.item():.6f}, Threshold: {gate_a_threshold:.6f}")

    health0 = check_embedding_health(baseline_emb)
    print(f"  Embed health: std={health0['avg_cosine_std']:.4f} rank={health0['rank']}")

    # ---- Load pairs ----
    print("\n--- Loading pairs ---")
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

    # ---- Basline MRR ----
    print("\n--- Baseline MRR ---")
    with torch.no_grad():
        base_node_emb = gnn(features, sources, targets, edge_types, num_nodes)
    base_mrr = compute_val_mrr(base_node_emb, gnn, lemma_to_idx, norm_to_indices, val_pairs)
    print(f"  Baseline MRR: {base_mrr:.4f}")

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(gnn.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ---- Training loop ----
    stats_history = []
    best_val_mrr = base_mrr
    best_epoch = 0
    aborted_gate = None

    print(f"\n{'=' * 60}")
    print(f"TRAINING (TRIPLET-ONLY): {args.epochs} epochs, batch={args.batch_size}")
    print(f"  m={args.margin}, K={args.max_hard_negatives_per_pair}, lp_w={args.preservation_weight}")
    print(f"  MRR computed every {args.mrr_every} epochs")
    print(f"  Grad: LP->full_GNN, Triplet->goal_encoder")
    print(f"{'=' * 60}")

    for epoch in range(args.epochs):
        t0 = time.time()
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")

        # Phase 1: Full GNN forward + LP backward (release graph after)
        gnn.train()
        node_emb = gnn(features, sources, targets, edge_types, num_nodes)
        node_emb_norm = F.normalize(node_emb, dim=-1)

        # LP preservation loss -> full GNN backbone, backward immediately
        lp_loss_val = compute_link_prediction_loss(node_emb, sources, targets, sample_edges=500)
        lp_loss = args.preservation_weight * lp_loss_val

        optimizer.zero_grad()
        lp_loss.backward()  # gradients through full GNN, graph released after
        # Detach and free GNN graph
        node_emb_norm_det = node_emb_norm.detach().clone()
        del node_emb, node_emb_norm  # help GC

        # Phase 2: Triplet loss over batches -> goal_encoder only
        total_triplet_val = 0.0
        total_batches = 0
        epoch_triplet_val = 0.0
        epoch_pos_sim = 0.0
        epoch_neg_sim = 0.0
        random.shuffle(train_pairs)

        n_batches_est = max(1, len(train_pairs) // args.batch_size + (1 if len(train_pairs) % args.batch_size else 0))

        for batch_start in range(0, len(train_pairs), args.batch_size):
            batch = train_pairs[batch_start: batch_start + args.batch_size]
            if len(batch) < 2:
                continue

            goal_embs_list = []
            lemma_embs_list = []
            for pair in batch:
                # Keyword-only context for training speed
                keywords = pair.get('_keywords', [])
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
                if matching:
                    match_t = torch.tensor(matching, device=device)
                    ctx_emb = node_emb_norm_det[match_t].mean(dim=0)
                else:
                    ctx_emb = torch.zeros(hidden_dim, device=device)

                goal_emb = gnn.encode_goal(ctx_emb.unsqueeze(0))

                lemma_name = pair["lemma"]
                idx = lemma_to_idx.get(lemma_name)
                if idx is not None and idx < num_nodes:
                    lemma_emb = node_emb_norm_det[idx].unsqueeze(0)
                else:
                    lemma_emb = torch.zeros(1, hidden_dim, device=device)

                goal_embs_list.append(goal_emb)
                lemma_embs_list.append(lemma_emb)

            goal_embs = torch.cat(goal_embs_list, dim=0)
            lemma_embs = torch.cat(lemma_embs_list, dim=0)
            goal_norm = F.normalize(goal_embs, dim=-1)
            lemma_norm = F.normalize(lemma_embs, dim=-1)

            triplet_loss, pos_sim, neg_sim = compute_triplet_loss_batch(
                goal_norm, lemma_norm,
                margin=args.margin,
                max_hard_negatives=args.max_hard_negatives_per_pair,
            )

            # Backward per batch — releases intermediates immediately
            scaled_loss = triplet_loss / n_batches_est
            scaled_loss.backward()

            total_batches += 1
            epoch_triplet_val += triplet_loss.item()
            epoch_pos_sim += pos_sim
            epoch_neg_sim += neg_sim

        # Clip + step after all backward calls
        torch.nn.utils.clip_grad_norm_(gnn.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        # ---- Validation ----
        gnn.eval()
        with torch.no_grad():
            eval_emb = gnn(features, sources, targets, edge_types, num_nodes)

        curr_lp = compute_link_prediction_loss(eval_emb, sources, targets, sample_edges=2000)
        lp_delta = (curr_lp.item() - baseline_lp.item()) / max(1e-8, abs(baseline_lp.item()))
        gate_a_ok = lp_delta <= 0.30

        # MRR only at baseline and final — too expensive per-epoch (100s)
        do_mrr = (epoch == 0 or epoch == args.epochs - 1 or (epoch + 1) % args.mrr_every == 0)
        if do_mrr:
            val_mrr = compute_val_mrr(eval_emb, gnn, lemma_to_idx, norm_to_indices, val_pairs)
            gate_b_ok = val_mrr >= args.gate_b_mrr_threshold
        else:
            val_mrr = -1.0  # skipped
            gate_b_ok = True  # don't abort on skipped MRR

        health = check_embedding_health(eval_emb)
        gate_c_ok = health["std_ok"] and health["rank_ok"]

        elapsed = time.time() - t0
        n_b = total_batches or 1

        gates_str = (f"A={'PASS' if gate_a_ok else 'FAIL'} "
                      f"B={'PASS' if gate_b_ok else 'FAIL'} "
                      f"C={'PASS' if gate_c_ok else 'FAIL'}")

        combined_loss = epoch_triplet_val / max(1, n_b) + lp_loss_val.item()
        stats = {
            "epoch": epoch + 1,
            "loss": round(combined_loss, 6),
            "triplet_loss": round(epoch_triplet_val / n_b, 6),
            "lp_loss": round(lp_loss_val.item(), 6),
            "pos_cos": round(epoch_pos_sim / n_b, 4),
            "neg_cos": round(epoch_neg_sim / n_b, 4),
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

        mrr_str = f"{val_mrr:.4f}" if val_mrr >= 0 else "SKIP"
        print(f"  loss={combined_loss:.4f} | triplet={epoch_triplet_val/n_b:.4f} | "
              f"lp={lp_loss_val.item():.4f} | mrr={mrr_str} | "
              f"pos={epoch_pos_sim/n_b:.3f} neg={epoch_neg_sim/n_b:.3f} | "
              f"lpΔ={lp_delta*100:+.1f}% | {gates_str} | {elapsed:.1f}s")
        stats_history.append(stats)

        # ---- Safety gates ----
        if args.abort_on_gate_fail and not gate_a_ok:
            aborted_gate = "A"
            _save_abort(output_dir, f"LP {curr_lp.item():.6f} > {gate_a_threshold:.6f}", "A")
            print(f"\n*** GATE A FAILED ***")
            break
        if args.abort_on_gate_fail and not gate_b_ok:
            aborted_gate = "B"
            _save_abort(output_dir, f"MRR {val_mrr:.4f} < {args.gate_b_mrr_threshold}", "B")
            print(f"\n*** GATE B FAILED ***")
            break
        if args.abort_on_gate_fail and not gate_c_ok:
            aborted_gate = "C"
            _save_abort(output_dir, f"std={health['avg_cosine_std']} rank={health['rank']}", "C")
            print(f"\n*** GATE C FAILED ***")
            break

        ckpt_path = output_dir / f"gnn_epoch_{epoch+1:03d}.pt"
        gnn.save(ckpt_path)
        if val_mrr > best_val_mrr:
            best_val_mrr = val_mrr
            best_epoch = epoch + 1
            gnn.save(output_dir / "gnn_best.pt")

    if not aborted_gate:
        print(f"\nTraining complete. Best MRR: {best_val_mrr:.4f} (epoch {best_epoch})")

    gnn.eval()
    gnn.save(output_dir / "gnn_finetuned.pt")

    with open(output_dir / "training_stats.json", "w") as f:
        json.dump({
            "mode": "triplet_only_v7",
            "config": {
                "gnn_checkpoint": args.gnn_checkpoint,
                "epochs": args.epochs, "batch_size": args.batch_size,
                "lr": args.learning_rate, "margin": args.margin,
                "preservation_weight": args.preservation_weight,
                "max_hard_negatives": args.max_hard_negatives_per_pair,
                "num_threads": args.num_threads,
            },
            "baseline_lp": baseline_lp.item(), "baseline_mrr": base_mrr,
            "best_mrr": best_val_mrr, "best_epoch": best_epoch,
            "aborted_gate": aborted_gate, "epochs": stats_history,
        }, f, indent=2)

    if aborted_gate:
        print(f"  ABORTED at gate {aborted_gate}")
    return {"aborted": aborted_gate is not None, "gate": aborted_gate,
            "best_mrr": best_val_mrr, "best_epoch": best_epoch}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GNN triplet-only fine-tuning (NO InfoNCE)")
    parser.add_argument("--gnn-checkpoint", default="checkpoints/gnn/full_graph_pretrained.pt")
    parser.add_argument("--pairs", default="data/raw/proof_step_pairs.jsonl")
    parser.add_argument("--graph", default="data/graph/dependency_graph_full")
    parser.add_argument("--output-dir", default="data/gnn_triplet_only_full")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--preservation-weight", type=float, default=0.1)
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-threads", type=int, default=6)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--no-abort", action="store_true")
    parser.add_argument("--gate-b-mrr-threshold", type=float, default=0.60)
    parser.add_argument("--max-hard-negatives-per-pair", type=int, default=5)
    parser.add_argument("--mrr-every", type=int, default=5,
                        help="Compute validation MRR every N epochs (expensive, ~100s)")

    args = parser.parse_args()
    args.num_threads = min(args.num_threads, 6)
    args.abort_on_gate_fail = not args.no_abort

    print("=" * 60)
    print("Full GNN Fine-tuning — TRIPLET-ONLY (NO InfoNCE)")
    print("=" * 60)
    print(f"  Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.learning_rate}")
    print(f"  m={args.margin}, K={args.max_hard_negatives_per_pair}")
    print(f"  LP weight: {args.preservation_weight}")
    print(f"  Loss: triplet_margin + lp_preservation")
    print()

    result = train_gnn_triplet_only(args)
    return 1 if result["aborted"] else 0


if __name__ == "__main__":
    sys.exit(main())
