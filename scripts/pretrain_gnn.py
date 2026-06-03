#!/usr/bin/env python3
"""Pre-train the GNN encoder on link prediction over the dependency graph.

This bootstraps the GNN before MCTS self-play so that the MCTS can
select relevant lemmas for proof states. Without this, the GNN has
random embeddings and MCTS has no signal.

Training task: Given a theorem (node), predict which other theorems
it depends on (outgoing edges). This is a multi-label link prediction
task over the dependency graph.

Usage:
    python scripts/pretrain_gnn.py                     # Full pre-training
    python scripts/pretrain_gnn.py --epochs 200         # Longer training
    python scripts/pretrain_gnn.py --device cpu          # CPU-only
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from src.explorer.dependency_graph import DependencyGraph
from src.explorer.gnn_config import GNNConfig
from src.explorer.gnn_encoder import (
    GNNEncoder,
    prepare_graph_tensors,
    extract_initial_features,
)


def pretrain_gnn(
    graph: DependencyGraph,
    config: GNNConfig,
    num_epochs: int = 100,
    batch_size: int = 256,
    device: torch.device | None = None,
    output_dir: str = "checkpoints/gnn",
) -> GNNEncoder:
    """Pre-train the GNN on link prediction.

    The GNN learns to embed theorems such that related theorems
    (those with dependency edges) have similar embeddings.
    """
    if device is None:
        device = torch.device("xpu:0" if torch.xpu.is_available() else "cpu")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Pre-training GNN on {graph.num_nodes} nodes, {graph.num_edges} edges")
    print(f"Device: {device}")
    print(f"Config: {config.hidden_dim}d, {config.num_layers} layers, "
          f"{config.num_heads} heads")

    # Prepare graph tensors
    sources, targets, edge_types, num_nodes = prepare_graph_tensors(graph)

    # Move to device in batches to avoid OOM
    print("Moving graph tensors to device...")
    sources = sources.to(device)
    targets = targets.to(device)
    edge_types = edge_types.to(device)

    # Initial features
    features = extract_initial_features(graph, config, device=device)
    print(f"Initial features: {features.shape}")

    # Initialize GNN
    model = GNNEncoder(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Build positive edge index for efficient sampling
    # (source_node → {dependencies})
    edge_set = set()
    node_to_deps: dict[int, set[int]] = {}
    for s, t in zip(sources.tolist(), targets.tolist()):
        edge_set.add((s, t))
        node_to_deps.setdefault(s, set()).add(t)

    nodes_with_deps = list(node_to_deps.keys())
    print(f"Nodes with dependencies: {len(nodes_with_deps)}")

    # Training loop
    best_loss = float("inf")
    all_losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        # Shuffle nodes
        perm = torch.randperm(len(nodes_with_deps), device=device)

        for start in range(0, len(nodes_with_deps), batch_size):
            batch_nodes = [nodes_with_deps[i] for i in perm[start:start + batch_size].tolist()]

            # ---- Positive samples: actual dependencies ----
            pos_src = []
            pos_tgt = []
            for n in batch_nodes:
                deps = node_to_deps.get(n, set())
                for d in deps:
                    pos_src.append(n)
                    pos_tgt.append(d)

            if not pos_src:
                continue

            # ---- Negative samples: random non-edges ----
            neg_src = []
            neg_tgt = []
            for n in batch_nodes:
                deps = node_to_deps.get(n, set())
                for _ in range(min(config.negative_ratio, max(1, len(deps)))):
                    neg_t = torch.randint(0, num_nodes, (1,), device=device).item()
                    while (n, neg_t) in edge_set or neg_t == n:
                        neg_t = torch.randint(0, num_nodes, (1,), device=device).item()
                    neg_src.append(n)
                    neg_tgt.append(neg_t)

            # ---- Forward pass ----
            embeddings = model(features, sources, targets, edge_types, num_nodes)

            # Positive scores
            pos_src_t = torch.tensor(pos_src, device=device)
            pos_tgt_t = torch.tensor(pos_tgt, device=device)
            pos_q = embeddings[pos_src_t]
            pos_c = embeddings[pos_tgt_t]
            pos_scores = (pos_q * pos_c).sum(dim=1)  # [num_pos]

            # Negative scores
            neg_src_t = torch.tensor(neg_src, device=device)
            neg_tgt_t = torch.tensor(neg_tgt, device=device)
            neg_q = embeddings[neg_src_t]
            neg_c = embeddings[neg_tgt_t]
            neg_scores = (neg_q * neg_c).sum(dim=1)  # [num_neg]

            # ---- Loss: margin-based ranking ----
            # Positive pairs should have higher similarity than negative pairs
            margin = 0.5
            pos_loss = F.relu(margin - pos_scores).mean()
            neg_loss = F.relu(margin + neg_scores).mean()

            # Additional: contrastive (InfoNCE-style)
            if len(pos_scores) > 0 and len(neg_scores) > 0:
                # Encourage positive scores >> negative scores
                loss = pos_loss + neg_loss + 0.1 * (
                    neg_scores.mean() - pos_scores.mean() + margin
                ).clamp(min=0)
            else:
                loss = pos_loss + neg_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(1, num_batches)
        all_losses.append(avg_loss)

        if epoch % 10 == 0 or epoch == num_epochs - 1:
            # Compute link prediction accuracy
            with torch.no_grad():
                model.eval()
                embeddings_eval = model(features, sources, targets, edge_types, num_nodes)
                # Sample 100 positive and negative pairs
                pos_sample = torch.tensor(
                    [list(edge_set)[i] for i in torch.randint(0, len(edge_set), (100,))],
                    device=device,
                )
                neg_sample = torch.randint(0, num_nodes, (100, 2), device=device)
                pos_q = embeddings_eval[pos_sample[:, 0]]
                pos_c = embeddings_eval[pos_sample[:, 1]]
                pos_acc = ((pos_q * pos_c).sum(dim=1) > 0).float().mean().item()
                neg_q = embeddings_eval[neg_sample[:, 0]]
                neg_c = embeddings_eval[neg_sample[:, 1]]
                neg_acc = ((neg_q * neg_c).sum(dim=1) < 0).float().mean().item()

            print(
                f"Epoch {epoch:3d}/{num_epochs} | "
                f"Loss: {avg_loss:.4f} | "
                f"Pos acc: {pos_acc:.2%} | "
                f"Neg acc: {neg_acc:.2%}"
            )

            # Save best
            if avg_loss < best_loss:
                best_loss = avg_loss
                model.save(output_dir / "gnn_best.pt")
                print(f"  → Saved best model (loss={best_loss:.4f})")

        # Clear GPU
        if device.type == "xpu":
            torch.xpu.empty_cache()

    # Final save
    model.save(output_dir / "gnn_final.pt")

    # Print summary
    print(f"\nPre-training complete: {num_epochs} epochs")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Model saved to {output_dir}/")

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Pre-train GNN on math dependency graph"
    )
    parser.add_argument(
        "--graph", default="data/graph/dependency_graph",
        help="Path to built dependency graph",
    )
    parser.add_argument(
        "--output", default="checkpoints/gnn",
        help="Output directory for trained model",
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=256,
        help="GNN hidden dimension",
    )
    parser.add_argument(
        "--layers", type=int, default=3,
        help="Number of GNN layers",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--device", default=None,
        help="Device (cpu, xpu:0, cuda:0)",
    )
    args = parser.parse_args()

    graph_path = _project_root / args.graph
    output_dir = _project_root / args.output

    if not graph_path.with_suffix(".nx.pkl").exists():
        print(f"Error: graph not found at {graph_path}.nx.pkl")
        print("Run scripts/build_dependency_graph.py first.")
        sys.exit(1)

    # Load graph
    print(f"Loading graph from {graph_path}...")
    graph = DependencyGraph.load(graph_path)
    print(f"Graph: {graph.summary()}")

    # Config
    config = GNNConfig(
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        num_heads=max(4, args.hidden_dim // 64),
        learning_rate=args.lr,
        num_epochs=args.epochs,
    )

    device = torch.device(args.device) if args.device else None

    pretrain_gnn(
        graph=graph,
        config=config,
        num_epochs=args.epochs,
        device=device,
        output_dir=str(output_dir),
    )


if __name__ == "__main__":
    main()
