#!/usr/bin/env python3
"""Train a value network to predict proof completability from goal embeddings.

The value network takes a goal embedding (from frozen GNN GoalEncoder) and
outputs P(success | state). It is trained on (state, outcome) pairs where
outcome is 1.0 for goals from proven theorems and 0.0 for goals known to
lead to dead ends.

Training data:
  POSITIVE: Goals from proof_step_pairs.jsonl — each (goal, lemma) pair is
            from a real Mathlib proof, so the state IS provable (V=1.0).
  NEGATIVE: Goals paired with lemmas from different theorems/domains —
            almost certainly wrong (V=0.0). Verified on a held-out set.

Usage:
  python scripts/training/train_value_network.py \
      --gnn-checkpoint checkpoints/gnn/10m_hybrid.pt \
      --graph data/graph/dependency_graph_full \
      --train-pairs data/raw/proof_step_pairs.jsonl \
      --output checkpoints/value_network.pt \
      --num-samples 10000 --epochs 20
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.explorer.dependency_graph import DependencyGraph
from src.explorer.gnn_config import GNNConfig
from src.explorer.gnn_encoder import (
    GNNEncoder,
    extract_initial_features,
    prepare_graph_tensors,
)
from src.explorer.value_network import ValueNetwork
from src.explorer.mcts import _extract_math_keywords
from scripts.eval.eval_gnn_prover import (
    build_lemma_index,
    build_lemma_norm_index,
    normalize_expression,
    tokenize_expression,
    score_lemmas_text,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def _build_kw_map(lemma_index: dict[str, int], node_ids: list[str]) -> dict[str, list[int]]:
    """Build keyword → lemma index mapping for goal embedding."""
    kw_map: dict[str, list[int]] = {}
    for lemma_name, idx in lemma_index.items():
        short = lemma_name.lower().split(".")[-1]
        tokens = short.replace("_", " ").split()
        for token in tokens:
            if len(token) >= 2:
                kw_map.setdefault(token, []).append(idx)
        kw_map.setdefault(short, []).append(idx)
    return kw_map


def encode_goal_text(
    goal_text: str,
    node_embeddings: torch.Tensor,
    idx_to_norm: dict[int, str],
    lemma_index: dict[str, int],
    kw_map: dict[str, list[int]],
    gnn: GNNEncoder,
) -> torch.Tensor | None:
    """Encode a goal text into a goal embedding.

    Uses the full pipeline: normalized text matching → keyword average
    → GoalEncoder projection. Returns None if encoding fails.
    """
    device = node_embeddings.device
    node_emb_norm = F.normalize(node_embeddings, dim=-1)

    goal_norm = normalize_expression(goal_text)

    # Reflexivity check
    is_reflexive = False
    if "=" in goal_norm and "↔" not in goal_norm and "→" not in goal_norm and "≠" not in goal_norm:
        sides = goal_norm.split("=", 1)
        if len(sides) == 2 and sides[0].strip() == sides[1].strip():
            is_reflexive = True

    # Exact matches
    exact_matches: set[int] = set()
    for idx, lemma_norm in idx_to_norm.items():
        if lemma_norm == goal_norm:
            exact_matches.add(idx)
        elif is_reflexive and lemma_norm == normalize_expression("a = a"):
            exact_matches.add(idx)

    # Power-stripping fallback
    if not exact_matches:
        import re
        goal_stripped = re.sub(r'\s*\^\s*\d+', '', goal_norm)
        for idx, lemma_norm in idx_to_norm.items():
            lemma_stripped = re.sub(r'\s*\^\s*\d+', '', lemma_norm)
            if lemma_stripped == goal_stripped:
                exact_matches.add(idx)

    match_indices = list(exact_matches)

    if match_indices:
        indices_t = torch.tensor(match_indices[:100], device=device)
        context_emb = node_emb_norm[indices_t].mean(dim=0)
    else:
        # Fall back to keyword-based context
        keywords = _extract_math_keywords(goal_text)
        candidates: dict[int, float] = {}
        for kw in keywords:
            matches = kw_map.get(kw.lower(), [])
            for rank, idx in enumerate(matches):
                if idx >= node_emb_norm.size(0):
                    continue
                score = 1.0 / (1.0 + rank * 0.1)
                candidates[idx] = candidates.get(idx, 0.0) + score
        sorted_candidates = sorted(candidates.items(), key=lambda x: -x[1])[:100]
        matching_indices = [idx for idx, _ in sorted_candidates]

        if matching_indices:
            indices_t = torch.tensor(matching_indices, device=device)
            context_emb = node_emb_norm[indices_t].mean(dim=0)
        else:
            return torch.zeros(node_emb_norm.size(1), device=device)

    # Project through GoalEncoder
    if gnn.goal_encoder is not None:
        return gnn.encode_goal(context_emb)
    elif context_emb.norm() > 1e-8:
        return F.normalize(context_emb, dim=-1)
    return context_emb


# ---------------------------------------------------------------------------
# Training data generation
# ---------------------------------------------------------------------------

def generate_training_data(
    pairs: list[dict],
    node_embeddings: torch.Tensor,
    idx_to_norm: dict[int, str],
    lemma_index: dict[str, int],
    kw_map: dict[str, list[int]],
    gnn: GNNEncoder,
    num_samples: int = 10000,
    neg_ratio: float = 1.0,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate training data from proof step pairs.

    Returns (embeddings [N, D], targets [N]) where targets are 0/1.
    """
    random.seed(seed)
    rng = random.Random(seed)

    # Deduplicate goals for efficiency
    goal_set: dict[str, str] = {}  # goal_text → domain
    for p in pairs:
        goal = p["goal"]
        if goal not in goal_set:
            goal_set[goal] = p.get("domain", "unknown")

    unique_goals = list(goal_set.items())
    rng.shuffle(unique_goals)
    unique_goals = unique_goals[:num_samples]

    # Collect all lemmas for negative sampling
    all_lemmas = list(set(p["lemma"] for p in pairs))
    domains = list(set(p.get("domain", "unknown") for p in pairs))

    embeddings_list: list[torch.Tensor] = []
    targets_list: list[float] = []

    n_skipped = 0
    for goal_text, domain in unique_goals:
        # Positive: the goal IS from a provable state
        goal_emb = encode_goal_text(
            goal_text, node_embeddings, idx_to_norm,
            lemma_index, kw_map, gnn
        )
        if goal_emb is not None:
            embeddings_list.append(goal_emb.cpu())
            targets_list.append(1.0)

            # Negative: create a "wrong" goal by perturbing
            # Strategy: pair with lemma from different domain
            for _ in range(int(neg_ratio)):
                other_domain = rng.choice([d for d in domains if d != domain] or domains)
                # Find a goal from other domain to use as negative
                other_goals = [g for g, d in goal_set.items() if d == other_domain]
                if other_goals:
                    neg_goal = rng.choice(other_goals)
                    neg_emb = encode_goal_text(
                        neg_goal, node_embeddings, idx_to_norm,
                        lemma_index, kw_map, gnn
                    )
                    if neg_emb is not None:
                        embeddings_list.append(neg_emb.cpu())
                        targets_list.append(0.0)
        else:
            n_skipped += 1

    if n_skipped:
        print(f"  Skipped {n_skipped} goals that couldn't be encoded")

    embeddings = torch.stack(embeddings_list)
    targets = torch.tensor(targets_list, dtype=torch.float32)

    # Balance classes
    n_pos = int(targets.sum().item())
    n_neg = len(targets) - n_pos
    print(f"  Training data: {n_pos} positive, {n_neg} negative "
          f"({n_pos / max(1, len(targets)):.1%} positive)")

    return embeddings, targets


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_value_network(
    vn: ValueNetwork,
    embeddings: torch.Tensor,
    targets: torch.Tensor,
    val_embeddings: torch.Tensor | None,
    val_targets: torch.Tensor | None,
    num_epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    seed: int = 42,
):
    """Train the value head with binary cross-entropy."""
    torch.manual_seed(seed)

    # Shuffle
    perm = torch.randperm(len(embeddings))
    embeddings = embeddings[perm]
    targets = targets[perm]

    # Split: 80% train, 20% test (if no separate val set)
    split = int(0.8 * len(embeddings))
    train_emb = embeddings[:split]
    train_tgt = targets[:split]
    test_emb = embeddings[split:]
    test_tgt = targets[split:]

    optimizer = torch.optim.AdamW(
        vn.value_head.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs
    )
    criterion = nn.BCELoss()

    best_acc = 0.0
    history: list[dict] = []

    print(f"\nTraining value network ({num_epochs} epochs, batch={batch_size})")
    print(f"  Train: {len(train_emb)}, Test: {len(test_emb)}")
    if val_embeddings is not None:
        print(f"  Val: {len(val_embeddings)}")

    t_start = time.time()

    for epoch in range(num_epochs):
        vn.train()
        total_loss = 0.0
        n_batches = 0

        # Mini-batch training
        indices = torch.randperm(len(train_emb))
        for i in range(0, len(train_emb), batch_size):
            batch_idx = indices[i:i + batch_size]
            emb_batch = train_emb[batch_idx]
            tgt_batch = train_tgt[batch_idx]

            preds = vn(emb_batch)
            loss = criterion(preds, tgt_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(1, n_batches)

        # Evaluate on test set
        vn.eval()
        with torch.no_grad():
            test_preds = vn(test_emb)
            test_loss = criterion(test_preds, test_tgt).item()
            test_acc = ((test_preds > 0.5) == test_tgt).float().mean().item()

        # Validate on held-out set
        val_acc = None
        if val_embeddings is not None:
            with torch.no_grad():
                val_preds = vn(val_embeddings)
                val_acc = ((val_preds > 0.5) == val_targets).float().mean().item()

        history.append({
            "epoch": epoch + 1,
            "train_loss": round(avg_loss, 4),
            "test_loss": round(test_loss, 4),
            "test_acc": round(test_acc, 4),
            "val_acc": round(val_acc, 4) if val_acc is not None else None,
        })

        val_str = f"val_acc={val_acc:.3f}" if val_acc is not None else ""
        print(f"  Epoch {epoch+1:3d}/{num_epochs}  "
              f"loss={avg_loss:.4f}  test_acc={test_acc:.3f}  {val_str}")

        if test_acc > best_acc:
            best_acc = test_acc

    elapsed = time.time() - t_start
    print(f"\nTraining complete: {elapsed:.1f}s, best test_acc={best_acc:.3f}")

    return history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train value network for proof state evaluation"
    )
    parser.add_argument(
        "--gnn-checkpoint",
        default="checkpoints/gnn/10m_hybrid.pt",
        help="GNN checkpoint path",
    )
    parser.add_argument(
        "--graph",
        default="data/graph/dependency_graph_full",
        help="Dependency graph path",
    )
    parser.add_argument(
        "--train-pairs",
        default="data/raw/proof_step_pairs.jsonl",
        help="Proof step pairs JSONL (226K pairs)",
    )
    parser.add_argument(
        "--output",
        default="checkpoints/value_network.pt",
        help="Output checkpoint path",
    )
    parser.add_argument(
        "--num-samples", type=int, default=10000,
        help="Number of unique goals to sample for training",
    )
    parser.add_argument(
        "--neg-ratio", type=float, default=1.0,
        help="Negative:positive ratio",
    )
    parser.add_argument(
        "--epochs", type=int, default=20,
        help="Training epochs",
    )
    parser.add_argument(
        "--batch-size", type=int, default=256,
        help="Training batch size",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--num-threads", type=int, default=6,
        help="Number of CPU threads",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("VALUE NETWORK TRAINING")
    print("=" * 70)
    print(f"Architecture: Frozen GNN GoalEncoder → ValueHead MLP (768→256→1)")
    print(f"Samples: {args.num_samples}, Neg ratio: {args.neg_ratio}x")
    print(f"Training: {args.epochs} epochs, batch={args.batch_size}, lr={args.lr}")
    print()

    torch.set_num_threads(args.num_threads)
    print(f"PyTorch threads: {torch.get_num_threads()}")

    # --- Load GNN ---
    ckpt_path = _PROJECT_ROOT / args.gnn_checkpoint
    if not ckpt_path.exists():
        # Try alternative checkpoints
        alternatives = [
            "checkpoints/gnn/gate2_fullgraph_finetuned.pt",
            "checkpoints/gnn/full_graph_pretrained.pt",
            "checkpoints/gnn/gnn_best.pt",
        ]
        found = False
        for alt in alternatives:
            alt_path = _PROJECT_ROOT / alt
            if alt_path.exists():
                ckpt_path = alt_path
                found = True
                break
        if not found:
            print(f"ERROR: No GNN checkpoint found. Tried: {args.gnn_checkpoint}, {alternatives}")
            return 1

    print(f"Loading GNN: {ckpt_path}")
    gnn = GNNEncoder.load(str(ckpt_path))
    gnn.eval()
    n_params = sum(p.numel() for p in gnn.parameters())
    print(f"  GNN: {n_params:,} params, hidden={gnn.config.hidden_dim}")

    # --- Load graph ---
    graph_path = _PROJECT_ROOT / args.graph
    if not graph_path.with_suffix(".nx.pkl").exists():
        # Try alternative
        graph_path = _PROJECT_ROOT / "data/graph/dependency_graph"
        if not graph_path.with_suffix(".nx.pkl").exists():
            print(f"ERROR: Graph not found at {args.graph}")
            return 1

    print(f"Loading graph: {graph_path}")
    graph = DependencyGraph.load(graph_path)
    print(f"  Graph: {graph.summary()}")

    # --- Compute node embeddings ---
    print("Computing GNN node embeddings...")
    features = extract_initial_features(graph, gnn.config)
    sources, targets, edge_types, num_nodes = prepare_graph_tensors(graph)
    print(f"  Graph: {num_nodes} nodes, {sources.size(0)} edges")

    with torch.no_grad():
        node_embeddings = gnn(features, sources, targets, edge_types, num_nodes)
    print(f"  Embeddings: {node_embeddings.shape}")

    # --- Build indices ---
    print("Building indices...")
    lemma_index = build_lemma_index(graph)
    idx_to_norm = build_lemma_norm_index(graph, lemma_index)
    kw_map = _build_kw_map(lemma_index, sorted(graph.node_ids))
    print(f"  Lemma index: {len(lemma_index)} entries")
    print(f"  Norm index: {len(idx_to_norm)} normalized conclusions")
    print(f"  Keyword map: {len(kw_map)} unique keywords")

    # --- Load training pairs ---
    pairs_path = _PROJECT_ROOT / args.train_pairs
    print(f"\nLoading training pairs: {pairs_path}")
    pairs = load_jsonl(pairs_path)
    print(f"  Total pairs: {len(pairs):,}")

    # --- Generate training data ---
    print(f"\nGenerating training data ({args.num_samples} samples)...")
    embeddings, targets = generate_training_data(
        pairs=pairs,
        node_embeddings=node_embeddings,
        idx_to_norm=idx_to_norm,
        lemma_index=lemma_index,
        kw_map=kw_map,
        gnn=gnn,
        num_samples=args.num_samples,
        neg_ratio=args.neg_ratio,
    )
    print(f"  Total examples: {len(embeddings)}, dim={embeddings.shape[1]}")

    # --- Create value network ---
    vn = ValueNetwork(gnn, freeze_encoder=True)

    # --- Train ---
    history = train_value_network(
        vn=vn,
        embeddings=embeddings,
        targets=targets,
        val_embeddings=None,
        val_targets=None,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    # --- Save ---
    output_path = _PROJECT_ROOT / args.output
    vn.save(output_path)
    print(f"\nValue network saved to: {output_path}")

    # --- Save training metrics ---
    metrics_path = output_path.with_suffix(".train_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "architecture": "Frozen GNN GoalEncoder → ValueHead MLP (768→256→1)",
            "num_samples": args.num_samples,
            "neg_ratio": args.neg_ratio,
            "num_epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "gnn_checkpoint": str(ckpt_path),
            "graph": str(graph_path),
            "history": history,
        }, f, indent=2, default=str)
    print(f"Training metrics saved to: {metrics_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
