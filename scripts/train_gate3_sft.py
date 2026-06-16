#!/usr/bin/env python3
"""Fast supervised fine-tune of GNN on gate2_training.jsonl theorems.

Loads the pretrained proof_step GNN and trains it on gate2 theorem statements
to predict the correct lemma from the dependency graph. No MCTS, no Lean calls.
Runs in ~2 minutes on CPU.
"""
import sys, json, argparse, time, re, random
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from src.explorer.dependency_graph import DependencyGraph
from src.explorer.gnn_config import GNNConfig
from src.explorer.gnn_encoder import GNNEncoder, prepare_graph_tensors, extract_initial_features
from src.explorer.mcts import _extract_math_keywords


def normalize_expression(expr: str) -> str:
    """Normalize a Lean expression for comparison."""
    expr = re.sub(r'\((\w+)\s*:\s*[^)]+\)', r'\1', expr)
    expr = re.sub(r'[()\u2080-\u2089\u00b2\u00b3\u00b9\u2070\u2074-\u2079\u2090-\u209c]', '', expr)
    expr = re.sub(r'\s+', ' ', expr).strip()
    return expr


def build_lemma_norm_index(graph: DependencyGraph) -> dict[int, str]:
    """Build index: node_idx -> normalized lemma conclusion."""
    idx_to_norm = {}
    for node_id in graph.node_ids:
        idx = graph.node_id_to_idx(node_id)
        info = graph.node_info.get(node_id, {})
        conclusion = info.get("conclusion", node_id)
        idx_to_norm[idx] = normalize_expression(conclusion)
    return idx_to_norm


def find_matching_lemma(
    statement: str,
    graph: DependencyGraph,
    idx_to_norm: dict[int, str],
) -> int | None:
    """Find the graph node whose conclusion matches the theorem statement."""
    norm_stmt = normalize_expression(statement)
    
    # Extract equality sides for matching
    stmt_parts = norm_stmt.split(" = ")
    if len(stmt_parts) == 2:
        lhs, rhs = stmt_parts
    else:
        lhs = norm_stmt
        rhs = norm_stmt
    
    best_idx = None
    best_score = 0
    
    for idx, norm_conclusion in idx_to_norm.items():
        score = 0
        # Exact match
        if norm_conclusion == norm_stmt:
            score = 100
        elif norm_conclusion in norm_stmt or norm_stmt in norm_conclusion:
            score = 50
        else:
            # Token overlap
            stmt_tokens = set(norm_stmt.split())
            conc_tokens = set(norm_conclusion.split())
            overlap = len(stmt_tokens & conc_tokens)
            if overlap > 0:
                score = overlap / max(len(stmt_tokens), len(conc_tokens)) * 30
        
        if score > best_score:
            best_score = score
            best_idx = idx
    
    # Require minimum similarity
    if best_score >= 20:
        return best_idx
    return None


def main():
    parser = argparse.ArgumentParser(description="Fast supervised fine-tune GNN on gate2 theorems")
    parser.add_argument("--theorems", default="data/raw/gate2_training.jsonl")
    parser.add_argument("--graph", default="data/graph/dependency_graph")
    parser.add_argument("--domain", default="Algebra")
    parser.add_argument("--pretrained", default="checkpoints/gnn/proof_step_pretrained.pt")
    parser.add_argument("--output", default="checkpoints/gate3_gnn/gnn_final.pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    # ---- Load graph ----
    graph_path = _project_root / args.graph
    graph = DependencyGraph.load(graph_path)
    print(f"Graph: {graph.num_nodes} nodes")
    if args.domain:
        graph = graph.domain_subgraph(args.domain)
        print(f"  Domain '{args.domain}': {graph.num_nodes} nodes")

    # ---- Build lemma index ----
    idx_to_norm = build_lemma_norm_index(graph)
    print(f"Norm index: {len(idx_to_norm)} entries")

    # ---- Load theorems ----
    tp = _project_root / args.theorems
    with open(tp) as f:
        theorems = [json.loads(line) for line in f]
    print(f"Theorems: {len(theorems)}")

    # ---- Match theorems to graph nodes ----
    matched = []
    unmatched = []
    for t in theorems:
        stmt = t["statement"]
        lemma_idx = find_matching_lemma(stmt, graph, idx_to_norm)
        if lemma_idx is not None:
            matched.append((stmt, lemma_idx))
        else:
            unmatched.append(t["name"])
    
    print(f"Matched: {len(matched)}/{len(theorems)} theorems to graph nodes")
    if len(matched) < 10:
        print(f"  Unmatched: {unmatched}")
        print("Error: too few matches for training")
        sys.exit(1)

    # ---- Load GNN ----
    ckpt_path = _project_root / args.pretrained
    gnn = GNNEncoder.load(str(ckpt_path))
    gnn = gnn.to(device)
    print(f"GNN: {sum(p.numel() for p in gnn.parameters()):,} params")

    # ---- Pre-compute graph tensors ----
    features = extract_initial_features(graph, gnn.config).to(device)
    sources, targets, edge_types, num_nodes = prepare_graph_tensors(graph)
    sources = sources.to(device)
    targets = targets.to(device)
    edge_types = edge_types.to(device)

    # ---- Training ----
    optimizer = torch.optim.AdamW(gnn.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    gnn.train()
    best_acc = 0.0

    for epoch in range(args.epochs):
        random.shuffle(matched)
        epoch_loss = 0.0
        correct = 0
        total = 0

        embeddings = gnn(features, sources, targets, edge_types, num_nodes)
        embeddings_norm = F.normalize(embeddings, dim=-1)

        # Goal encoder if available
        has_goal_encoder = hasattr(gnn, 'goal_encoder') and gnn.goal_encoder is not None

        for stmt, target_idx in matched:
            # Simple keyword embedding for the statement
            keywords = _extract_math_keywords(stmt)
            if not keywords:
                keywords = ["unknown"]
            
            # Build query embedding from keyword nodes
            query_indices = []
            for kw in keywords:
                for node_id in graph.node_ids:
                    if kw.lower() in node_id.lower():
                        idx = graph.node_id_to_idx(node_id)
                        query_indices.append(idx)
                        break
            
            if not query_indices:
                continue
            
            query_idx_t = torch.tensor(query_indices, device=device)
            query_emb = embeddings_norm[query_idx_t].mean(dim=0)

            if has_goal_encoder:
                query_emb = gnn.goal_encoder(query_emb.unsqueeze(0)).squeeze(0)

            # Compute logits against all nodes
            logits = torch.matmul(query_emb, embeddings_norm.T)
            
            # Loss: cross-entropy with target
            target_t = torch.tensor([target_idx], device=device)
            loss = F.cross_entropy(logits.unsqueeze(0), target_t)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gnn.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            total += 1
            pred = logits.argmax().item()
            if pred == target_idx:
                correct += 1

        scheduler.step()
        avg_loss = epoch_loss / max(1, total)
        acc = correct / max(1, total)
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.1%} | LR: {scheduler.get_last_lr()[0]:.2e}")

        if acc > best_acc:
            best_acc = acc
            output_path = _project_root / args.output
            output_path.parent.mkdir(parents=True, exist_ok=True)
            gnn.save(output_path)
            print(f"  → Saved best model (acc={best_acc:.1%}) to {output_path}")

    # Save final
    output_path = _project_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gnn.save(output_path)
    print(f"\nTraining complete. Best acc: {best_acc:.1%}")
    print(f"Model saved to {output_path}")


if __name__ == "__main__":
    main()
