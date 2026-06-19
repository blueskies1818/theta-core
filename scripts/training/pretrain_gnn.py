#!/usr/bin/env python3
"""Pre-train the GNN encoder on link prediction over the dependency graph.

Bootstraps GNN embeddings so MCTS can select relevant lemmas for proof states.
Without this, the GNN has random embeddings and MCTS picks irrelevant lemmas.

Training: mini-batch neighborhood sampling for memory-efficient GNN training
on the 58K-node dependency graph.

Usage:
    python scripts/training/pretrain_gnn.py --epochs 100 --device xpu:0
    python scripts/training/pretrain_gnn.py --epochs 50 --device cpu --hidden-dim 128
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from src.explorer.dependency_graph import DependencyGraph
from src.explorer.gnn_config import GNNConfig
from src.explorer.gnn_encoder import GNNEncoder, prepare_graph_tensors


def sample_subgraph(
    seed_nodes: list[int],
    sources: torch.Tensor,
    targets: torch.Tensor,
    edge_types: torch.Tensor,
    num_nodes: int,
    num_neighbors: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[int, int]]:
    """Sample a k-hop subgraph around seed nodes.

    Args:
        seed_nodes: Starting node indices.
        sources, targets, edge_types: Full graph edge tensors (on device).
        num_nodes: Total nodes in full graph.
        num_neighbors: Neighbors to sample per hop, e.g., [25, 15, 10].

    Returns:
        (sub_sources, sub_targets, sub_edge_types, sub_x, node_map)
        where node_map maps original idx → subgraph idx.
    """
    device = sources.device
    current_nodes = set(seed_nodes)
    all_nodes = set(seed_nodes)
    sampled_edges: list[tuple[int, int, int]] = []

    # Convert edges to adjacency for fast lookup: src → [(tgt, edge_type)]
    # Only do this once and cache, but for simplicity we build per call
    src_to_edges: dict[int, list[tuple[int, int]]] = {}
    for i in range(sources.size(0)):
        s = sources[i].item()
        t = targets[i].item()
        e = edge_types[i].item()
        src_to_edges.setdefault(s, []).append((t, e))

    for hop, k in enumerate(num_neighbors):
        next_nodes: set[int] = set()
        for node in current_nodes:
            neighbors = src_to_edges.get(node, [])
            if len(neighbors) > k:
                sampled = random.sample(neighbors, k)
            else:
                sampled = neighbors
            for tgt, etype in sampled:
                sampled_edges.append((node, tgt, etype))
                next_nodes.add(tgt)

        all_nodes.update(next_nodes)
        current_nodes = next_nodes

    # Build node mapping
    node_list = sorted(all_nodes)
    node_map = {orig: i for i, orig in enumerate(node_list)}

    # Build subgraph tensors
    sub_src = torch.tensor(
        [node_map[s] for s, _, _ in sampled_edges], dtype=torch.long, device=device
    )
    sub_tgt = torch.tensor(
        [node_map[t] for _, t, _ in sampled_edges], dtype=torch.long, device=device
    )
    sub_et = torch.tensor(
        [et for _, _, et in sampled_edges], dtype=torch.long, device=device
    )
    sub_x = torch.randn(
        len(node_list), 256, device=device
    )  # placeholder, will be replaced

    return sub_src, sub_tgt, sub_et, sub_x, node_map


def pretrain_gnn(
    graph: DependencyGraph,
    config: GNNConfig,
    num_epochs: int = 100,
    batch_size: int = 512,
    device: torch.device | None = None,
    output_dir: str = "checkpoints/gnn",
) -> GNNEncoder:
    """Pre-train GNN with mini-batch neighborhood sampling."""
    if device is None:
        device = torch.device("xpu:0" if torch.xpu.is_available() else "cpu")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load full graph tensors ----
    sources, targets, edge_types, num_nodes = prepare_graph_tensors(graph, device)

    # Build edge set for negative sampling
    edge_set: set[tuple[int, int]] = set()
    for s, t in zip(sources.tolist(), targets.tolist()):
        edge_set.add((s, t))

    # Build adjacency for neighborhood sampling
    print("Building adjacency index...")
    src_to_edges: dict[int, list[tuple[int, int]]] = {}
    for i in range(sources.size(0)):
        s = sources[i].item()
        t = targets[i].item()
        e = edge_types[i].item()
        src_to_edges.setdefault(s, []).append((t, e))

    # Nodes that have outgoing edges (can be training targets)
    nodes_with_deps = sorted(src_to_edges.keys())
    print(f"Nodes with deps: {len(nodes_with_deps)}, "
          f"Total edges: {len(edge_set)}")

    # ---- Model ----
    model = GNNEncoder(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} params, device={device}")

    # ---- Training ----
    best_loss = float("inf")
    margin = 0.5
    neighbor_depths = [25, 15]  # 2-hop sampling

    for epoch in range(num_epochs):
        t0 = time.time()
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        # Shuffle training nodes
        random.shuffle(nodes_with_deps)

        for batch_start in range(0, len(nodes_with_deps), batch_size):
            batch_seeds = nodes_with_deps[batch_start : batch_start + batch_size]

            # ---- Build subgraph around batch seeds ----
            current_nodes = set(batch_seeds)
            all_nodes = set(batch_seeds)
            sampled_edges: list[tuple[int, int, int]] = []
            sub_seeds_local: set[int] = set(
                range(len(batch_seeds))
            )  # indices in subgraph

            for hop, k in enumerate(neighbor_depths):
                next_nodes: set[int] = set()
                for node in current_nodes:
                    neighbors = src_to_edges.get(node, [])
                    if len(neighbors) > k:
                        sampled = random.sample(neighbors, k)
                    else:
                        sampled = neighbors
                    for tgt, etype in sampled:
                        sampled_edges.append((node, tgt, etype))
                        next_nodes.add(tgt)
                all_nodes.update(next_nodes)
                current_nodes = next_nodes

            # Build node mapping
            node_list = sorted(all_nodes)
            node_map = {orig: i for i, orig in enumerate(node_list)}
            sub_num_nodes = len(node_list)

            # Seed node local indices
            seed_local = torch.tensor(
                [node_map[s] for s in batch_seeds], device=device
            )

            # Subgraph edges
            if not sampled_edges:
                continue
            sub_src = torch.tensor(
                [node_map[s] for s, _, _ in sampled_edges],
                dtype=torch.long, device=device,
            )
            sub_tgt = torch.tensor(
                [node_map[t] for _, t, _ in sampled_edges],
                dtype=torch.long, device=device,
            )
            sub_et = torch.tensor(
                [et for _, _, et in sampled_edges],
                dtype=torch.long, device=device,
            )

            # Initial features for subgraph nodes
            sub_features = torch.randn(sub_num_nodes, config.input_dim, device=device)

            # ---- Forward pass on subgraph ----
            embeddings = model(sub_features, sub_src, sub_tgt, sub_et, sub_num_nodes)

            # ---- Link prediction on seed nodes ----
            seed_emb = embeddings[seed_local]  # [batch_size, hidden_dim]

            # Positive targets: actual dependencies of seed nodes
            pos_src_list = []
            pos_tgt_list = []
            for s in batch_seeds:
                deps = [t for t, _ in src_to_edges.get(s, [])]
                for d in deps:
                    if d in node_map:  # only if in subgraph
                        pos_src_list.append(node_map[s])
                        pos_tgt_list.append(node_map[d])

            if not pos_src_list:
                continue

            pos_src = torch.tensor(pos_src_list, device=device)
            pos_tgt = torch.tensor(pos_tgt_list, device=device)

            # Negative targets: random nodes not in the dependency set
            neg_src_list = []
            neg_tgt_list = []
            for s in batch_seeds:
                deps_set = {t for t, _ in src_to_edges.get(s, [])}
                for _ in range(min(config.negative_ratio, 5)):
                    neg_t = random.randrange(0, sub_num_nodes)
                    attempts = 0
                    # Avoid actual edges (use original IDs for lookup)
                    orig_neg = node_list[neg_t]
                    while ((s, orig_neg) in edge_set or orig_neg == s) and attempts < 50:
                        neg_t = random.randrange(0, sub_num_nodes)
                        orig_neg = node_list[neg_t]
                        attempts += 1
                    neg_src_list.append(node_map[s])
                    neg_tgt_list.append(neg_t)

            neg_src = torch.tensor(neg_src_list, device=device)
            neg_tgt = torch.tensor(neg_tgt_list, device=device)

            # ---- Compute loss ----
            pos_scores = (embeddings[pos_src] * embeddings[pos_tgt]).sum(dim=1)
            neg_scores = (embeddings[neg_src] * embeddings[neg_tgt]).sum(dim=1)

            loss = (
                F.relu(margin - pos_scores).mean()
                + F.relu(margin + neg_scores).mean()
                + 0.05 * (neg_scores.mean() - pos_scores.mean() + margin).clamp(min=0)
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(1, num_batches)
        elapsed = time.time() - t0

        # ---- Logging + evaluation ----
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            with torch.no_grad():
                model.eval()
                # Quick accuracy check on a small evaluation subgraph
                eval_seeds = random.sample(nodes_with_deps, min(100, len(nodes_with_deps)))
                eval_all = set(eval_seeds)
                eval_edges = []
                for s in eval_seeds:
                    deps = src_to_edges.get(s, [])
                    for t, e in deps[:5]:
                        eval_edges.append((s, t, e))
                        eval_all.add(t)

                eval_list = sorted(eval_all)
                eval_map = {o: i for i, o in enumerate(eval_list)}
                eval_src = torch.tensor(
                    [eval_map[s] for s, t, _ in eval_edges], device=device
                )
                eval_tgt = torch.tensor(
                    [eval_map[t] for _, t, _ in eval_edges], device=device
                )
                eval_et = torch.zeros(len(eval_edges), dtype=torch.long, device=device)
                eval_feat = torch.randn(len(eval_list), config.input_dim, device=device)

                eval_emb = model(eval_feat, eval_src, eval_tgt, eval_et, len(eval_list))

                # Positive accuracy
                pos_q = eval_emb[eval_src]
                pos_c = eval_emb[eval_tgt]
                pos_acc = ((pos_q * pos_c).sum(dim=1) > 0).float().mean().item()

                # Negative: random pairs
                neg_a = torch.randint(0, len(eval_list), (500,), device=device)
                neg_b = torch.randint(0, len(eval_list), (500,), device=device)
                neg_sim = (eval_emb[neg_a] * eval_emb[neg_b]).sum(dim=1).mean().item()
                pos_sim = (pos_q * pos_c).sum(dim=1).mean().item()

            print(
                f"Epoch {epoch:3d}/{num_epochs} | "
                f"Loss: {avg_loss:.4f} | "
                f"Pos acc: {pos_acc:.1%} | "
                f"Edge sim: {pos_sim:.4f} | "
                f"Rand sim: {neg_sim:.4f} | "
                f"Time: {elapsed:.1f}s"
            )

            if avg_loss < best_loss:
                best_loss = avg_loss
                model.save(output_dir / "gnn_best.pt")
                print(f"  → Saved best model (loss={best_loss:.4f})")

        # Clear GPU
        if device.type == "xpu":
            torch.xpu.empty_cache()

    model.save(output_dir / "gnn_final.pt")
    print(f"\nDone: {num_epochs} epochs, best loss {best_loss:.4f}")
    print(f"Model saved to {output_dir}/")
    return model


def main():
    parser = argparse.ArgumentParser(description="Pre-train GNN on link prediction")
    parser.add_argument("--graph", default="data/graph/dependency_graph")
    parser.add_argument("--output", default="checkpoints/gnn")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=768)
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    graph_path = _project_root / args.graph
    if not graph_path.with_suffix(".nx.pkl").exists():
        print(f"Error: graph not found at {graph_path}.nx.pkl")
        print("Run scripts/build/build_dependency_graph.py first.")
        sys.exit(1)

    graph = DependencyGraph.load(graph_path)
    print(f"Graph: {graph.summary()}")

    config = GNNConfig(
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        num_heads=max(4, args.hidden_dim // 64),
        input_dim=args.hidden_dim,
        learning_rate=args.lr,
        num_epochs=args.epochs,
    )

    device = torch.device(args.device) if args.device else None
    output_dir = _project_root / args.output

    pretrain_gnn(
        graph=graph,
        config=config,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
        output_dir=str(output_dir),
    )


if __name__ == "__main__":
    main()
