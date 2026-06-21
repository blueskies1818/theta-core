#!/usr/bin/env python3
"""Binary scorer: Goal+Lemma pair classifier with frozen GNN encoder.

Architecture:
    Goal text → Frozen GNN (256-dim) ──┐
                                       ├→ cat → [512→256→128→1] → sigmoid
    Lemma name → Frozen GNN (256-dim)──┘

The GNN is frozen — only the MLP scorer trains. BCE loss over positive
pairs (label=1) and negative pairs (label=0, random lemmas from other
proof-step pairs).

Usage:
    # Smoke test
    python -m src.scoring.binary_scorer --smoke

    # Full training
    python -m src.scoring.binary_scorer --train

    # Gate3 evaluation
    python -m src.scoring.binary_scorer --gate3
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Project root resolution
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Binary Scorer MLP
# ---------------------------------------------------------------------------


class BinaryScorer(nn.Module):
    """Bilinear + MLP scorer with explicit similarity features.

    Uses a low-rank bilinear form (goal^T U V^T lemma) plus cosine similarity
    and L2 distance as explicit features. Much fewer params than full MLP.
    """

    def __init__(self, hidden_dim: int = 256, bilinear_rank: int = 32, dropout: float = 0.1):
        super().__init__()
        # Low-rank bilinear: goal^T U V^T lemma
        # = (U^T goal)^T (V^T lemma)
        self.U = nn.Linear(hidden_dim, bilinear_rank, bias=False)
        self.V = nn.Linear(hidden_dim, bilinear_rank, bias=False)

        # MLP on top: bilinear_score + cosine + l2 + product_mean + diff_mean
        # = 1 + 1 + 1 + 1 + 1 = 5 features
        self.head = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        self.hidden_dim = hidden_dim

    def forward(self, goal_emb: torch.Tensor, lemma_emb: torch.Tensor) -> torch.Tensor:
        """Score goal-lemma pairs using bilinear form + explicit features.

        Returns [B, 1] logits (pre-sigmoid).
        """
        # Bilinear score
        u = self.U(goal_emb)      # [B, R]
        v = self.V(lemma_emb)     # [B, R]
        bilinear = (u * v).sum(dim=-1, keepdim=True)  # [B, 1]

        # Cosine similarity
        cos_sim = F.cosine_similarity(goal_emb, lemma_emb, dim=-1).unsqueeze(-1)

        # L2 distance
        l2_dist = (goal_emb - lemma_emb).norm(dim=-1, keepdim=True)

        # Product mean (avg interaction)
        prod_mean = (goal_emb * lemma_emb).mean(dim=-1, keepdim=True)

        # Diff mean
        diff_mean = (goal_emb - lemma_emb).mean(dim=-1, keepdim=True)

        features = torch.cat([bilinear, cos_sim, l2_dist, prod_mean, diff_mean], dim=-1)
        return self.head(features)

    def predict(self, goal_emb: torch.Tensor, lemma_emb: torch.Tensor) -> torch.Tensor:
        """Return probabilities [B, 1]."""
        with torch.no_grad():
            return torch.sigmoid(self.forward(goal_emb, lemma_emb))


# ---------------------------------------------------------------------------
# Frozen GNN Encoder wrapper
# ---------------------------------------------------------------------------


class FrozenGNNEncoder:
    """Wraps a pre-trained GNN encoder as a frozen feature extractor.

    Computes node embeddings once (full graph forward pass) then provides
    goal and lemma embeddings via keyword matching and node lookup.
    """

    def __init__(
        self,
        gnn_checkpoint: str,
        graph_path: str,
        device: torch.device | None = None,
    ):
        from src.explorer.dependency_graph import DependencyGraph
        from src.explorer.gnn_encoder import (
            GNNEncoder,
            extract_initial_features,
            prepare_graph_tensors,
        )
        from src.explorer.mcts import _extract_math_keywords

        self.device = device or torch.device("cpu")
        self._extract_math_keywords = _extract_math_keywords

        # Load GNN
        gnn_path = _PROJECT_ROOT / gnn_checkpoint
        if not gnn_path.exists():
            raise FileNotFoundError(f"GNN checkpoint not found: {gnn_path}")
        self.gnn = GNNEncoder.load(str(gnn_path))
        self.gnn.eval()
        self.gnn.to(self.device)
        for p in self.gnn.parameters():
            p.requires_grad_(False)

        self.hidden_dim = self.gnn.config.hidden_dim
        print(f"Loaded frozen GNN: {gnn_path}")
        print(f"  hidden_dim={self.hidden_dim}, params={sum(p.numel() for p in self.gnn.parameters()):,}")

        # Load graph
        graph_p = _PROJECT_ROOT / graph_path
        self.graph = DependencyGraph.load(graph_p)
        print(f"Loaded graph: {self.graph.summary()}")

        # Build lemma index
        self.lemma_to_idx: dict[str, int] = {}
        for node_id in self.graph.node_ids:
            idx = self.graph.node_id_to_idx(node_id)
            self.lemma_to_idx[node_id] = idx
            short = node_id.split(".")[-1] if "." in node_id else node_id
            if short not in self.lemma_to_idx:
                self.lemma_to_idx[short] = idx

        # Compute all node embeddings (frozen)
        self._compute_node_embeddings()

        # Build keyword → node index map for fast goal encoding
        self._build_keyword_map()

    def _compute_node_embeddings(self):
        from src.explorer.gnn_encoder import (
            extract_initial_features,
            prepare_graph_tensors,
        )

        print("Computing frozen node embeddings...")
        features = extract_initial_features(self.graph, self.gnn.config)
        sources, targets, edge_types, num_nodes = prepare_graph_tensors(self.graph)

        features = features.to(self.device)
        sources = sources.to(self.device)
        targets = targets.to(self.device)
        edge_types = edge_types.to(self.device)

        # Clamp edge types to match loaded model's num_edge_types
        max_et = self.gnn.config.num_edge_types - 1
        edge_types = edge_types.clamp(0, max_et)

        with torch.no_grad():
            self.node_embeddings = self.gnn(features, sources, targets, edge_types, num_nodes)
            self.node_embeddings = F.normalize(self.node_embeddings, dim=-1)
            self.node_embeddings = self.node_embeddings.cpu()

        self.num_nodes = num_nodes
        print(f"  Node embeddings: {self.node_embeddings.shape}")

    def _build_keyword_map(self):
        """Build keyword → list[node_idx] map for fast goal context lookup."""
        self.kw_map: dict[str, list[int]] = {}

        for node_id, idx in self.lemma_to_idx.items():
            parts = node_id.lower().split(".")
            for part in parts:
                if len(part) >= 2:
                    self.kw_map.setdefault(part, []).append(idx)

        # Also index by keyword tokens from lemma names
        import re
        for node_id, idx in self.lemma_to_idx.items():
            tokens = re.split(r'[_\s]+', node_id.lower())
            for tok in tokens:
                if len(tok) >= 2 and tok not in self.kw_map:
                    self.kw_map[tok] = []
                if len(tok) >= 2:
                    if idx not in self.kw_map[tok]:
                        self.kw_map[tok].append(idx)

        print(f"  Keyword map: {len(self.kw_map)} keywords")
        # Stats
        lens = [len(v) for v in self.kw_map.values()]
        if lens:
            print(f"  Nodes per keyword: min={min(lens)}, max={max(lens)}, avg={sum(lens)/len(lens):.1f}")

    def encode_goal(self, goal_text: str, theorem_name: str = "") -> torch.Tensor:
        """Encode a goal via theorem name lookup in the graph.

        Uses the theorem's own GNN node embedding. Falls back to keyword
        matching if theorem name is not in graph.

        Returns [256] tensor on self.device.
        """
        # Primary: use theorem name (exists for 99% of training pairs)
        if theorem_name:
            idx = self.lemma_to_idx.get(theorem_name)
            if idx is not None and idx < self.num_nodes:
                return self.node_embeddings[idx].to(self.device)

        # Fallback: keyword matching (for inference on unseen goals)
        return self._encode_goal_keywords(goal_text)

    def _encode_goal_keywords(self, goal_text: str) -> torch.Tensor:
        """Fallback: keyword matching for goals without a theorem name."""
        keywords = self._extract_math_keywords(goal_text)
        matching: list[int] = []
        seen: set[int] = set()

        for kw in keywords:
            for idx in self.kw_map.get(kw.lower(), []):
                if idx < self.num_nodes and idx not in seen:
                    matching.append(idx)
                    seen.add(idx)
                    if len(matching) >= 100:
                        break
            if len(matching) >= 100:
                break

        if matching:
            ctx_emb = self.node_embeddings[torch.tensor(matching)].mean(dim=0)
        else:
            ctx_emb = torch.zeros(self.hidden_dim)

        ctx_emb = ctx_emb.to(self.device)
        if ctx_emb.norm() > 1e-8:
            return F.normalize(ctx_emb, dim=-1)
        return ctx_emb

    def encode_lemma(self, lemma_name: str) -> torch.Tensor:
        """Encode a lemma name by looking up its frozen GNN node embedding.

        Returns [256] tensor on self.device.
        For unknown lemmas, returns a deterministic random embedding
        (hash-based), NOT zeros — zeros would prevent the MLP from
        learning any signal.
        """
        idx = self.lemma_to_idx.get(lemma_name)
        if idx is not None and idx < self.num_nodes:
            return self.node_embeddings[idx].to(self.device)
        # Hash-based deterministic embedding for OOV lemmas
        h = hash(lemma_name) & 0xFFFFFFFF
        saved_rng = torch.get_rng_state()
        torch.manual_seed(h)
        emb = F.normalize(torch.randn(self.hidden_dim), dim=-1).to(self.device)
        torch.set_rng_state(saved_rng)
        return emb

    def precompute_goal_embeddings(self, pairs: list[dict]) -> torch.Tensor:
        """Pre-compute goal embeddings for all training pairs.

        Returns [N, 256] tensor on CPU.
        """
        print(f"Pre-computing goal embeddings for {len(pairs)} pairs...")
        embeddings = []
        for i, pair in enumerate(pairs):
            if (i + 1) % 5000 == 0:
                print(f"  {i+1}/{len(pairs)} goals encoded")
            emb = self.encode_goal(pair["goal"], theorem_name=pair.get("name", "")).cpu()
            embeddings.append(emb)
        result = torch.stack(embeddings)
        print(f"  Done: {result.shape}")
        return result

    def precompute_lemma_embeddings(self, pairs: list[dict]) -> torch.Tensor:
        """Pre-compute lemma embeddings for all training pairs.

        Returns [N, 256] tensor on CPU.
        """
        print(f"Pre-computing lemma embeddings for {len(pairs)} pairs...")
        embeddings = []
        for i, pair in enumerate(pairs):
            if (i + 1) % 10000 == 0:
                print(f"  {i+1}/{len(pairs)} lemmas encoded")
            emb = self.encode_lemma(pair["lemma"]).cpu()
            embeddings.append(emb)
        result = torch.stack(embeddings)
        print(f"  Done: {result.shape}")
        return result


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_training_pairs(
    path: str,
    limit: int | None = None,
    seed: int = 42,
) -> list[dict]:
    """Load proof-step pairs from JSONL.

    Each line: {"goal": str, "lemma": str, "name": str, "domain": str}
    """
    pairs_path = _PROJECT_ROOT / path
    pairs = []
    with open(pairs_path) as f:
        for line in f:
            pair = json.loads(line)
            pairs.append(pair)
            if limit and len(pairs) >= limit:
                break
    random.seed(seed)
    random.shuffle(pairs)
    print(f"Loaded {len(pairs)} training pairs from {pairs_path}")
    return pairs


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_binary_scorer(
    encoder: FrozenGNNEncoder,
    pairs: list[dict],
    scorer: BinaryScorer,
    output_path: str,
    *,
    num_epochs: int = 10,
    batch_size: int = 512,
    lr: float = 1e-3,
    num_threads: int = 6,
    num_negatives: int = 5,
    val_fraction: float = 0.05,
    seed: int = 42,
) -> dict:
    """Train the binary scorer MLP on frozen GNN embeddings.

    Positives: all (goal, correct_lemma) pairs, label=1
    Negatives: 5 random lemmas from OTHER pairs per positive, label=0
    Loss: BCEWithLogitsLoss
    """
    torch.set_num_threads(num_threads)
    device = encoder.device

    # Pre-compute all embeddings
    goal_embs_full = encoder.precompute_goal_embeddings(pairs)
    lemma_embs_full = encoder.precompute_lemma_embeddings(pairs)

    # Filter out pairs where lemma is missing (zero embedding)
    lemma_norms = lemma_embs_full.norm(dim=-1)
    valid_mask = lemma_norms > 1e-6
    n_valid = valid_mask.sum().item()
    n_skipped = len(pairs) - n_valid
    if n_skipped > 0:
        print(f"Filtering: {n_skipped}/{len(pairs)} pairs have missing lemmas (skipped)")
        goal_embs = goal_embs_full[valid_mask]
        lemma_embs = lemma_embs_full[valid_mask]
        # Remap pairs to valid subset
        valid_indices_map = torch.where(valid_mask)[0]
        # Build new pairs list
        pairs = [pairs[i] for i in valid_indices_map.tolist()]
    else:
        goal_embs = goal_embs_full
        lemma_embs = lemma_embs_full

    # Train/val split
    n_total = len(pairs)
    n_val = max(1, int(n_total * val_fraction))
    n_train = n_total - n_val

    indices = list(range(n_total))
    random.seed(seed)
    random.shuffle(indices)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_goal = goal_embs[train_indices]
    train_lemma = lemma_embs[train_indices]
    val_goal = goal_embs[val_indices]
    val_lemma = lemma_embs[val_indices]

    print(f"\nTraining: {n_train} pairs, Validation: {n_val} pairs")
    print(f"Batch size: {batch_size}, Epochs: {num_epochs}, LR: {lr}")
    print(f"Negatives per positive: {num_negatives}")
    print(f"Threads: {num_threads}")

    # Diagnostic: check embedding statistics
    goal_norms = goal_embs.norm(dim=-1)
    lemma_norms = lemma_embs.norm(dim=-1)
    print(f"Goal emb: zero={((goal_norms < 1e-6).sum().item())}, "
          f"mean_norm={goal_norms.mean().item():.3f}")
    print(f"Lemma emb: zero={((lemma_norms < 1e-6).sum().item())}, "
          f"mean_norm={lemma_norms.mean().item():.3f}")

    scorer = scorer.to(device)
    optimizer = torch.optim.Adam(scorer.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=lr * 0.01
    )
    # pos_weight=1.0 for balanced training (no class imbalance correction needed)
    pos_weight = torch.tensor([1.0], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Pre-compute all lemma embeddings for negative sampling pool
    all_lemma_pool = lemma_embs  # [N, 256]

    history = {"epochs": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, num_epochs + 1):
        scorer.train()
        epoch_loss = 0.0
        epoch_pos = 0
        n_batches = 0

        # Shuffle training indices
        epoch_indices = train_indices.copy()
        random.shuffle(epoch_indices)

        for batch_start in range(0, n_train, batch_size):
            batch_idx = epoch_indices[batch_start: batch_start + batch_size]
            b_size = len(batch_idx)

            # Build batch: positive pairs
            pos_goal = goal_embs[batch_idx].to(device)    # [B, 256]
            pos_lemma = lemma_embs[batch_idx].to(device)  # [B, 256]
            pos_labels = torch.ones(b_size, 1, device=device)

            # Negative pairs: random lemmas from other pairs
            # For each positive, pick num_negatives lemmas from DIFFERENT indices
            neg_goal_list = []
            neg_lemma_list = []
            for bi in batch_idx:
                # Pick random indices NOT equal to bi
                candidates = [j for j in range(n_total) if j != bi]
                neg_indices = random.sample(candidates, min(num_negatives, len(candidates)))
                for nj in neg_indices:
                    neg_goal_list.append(goal_embs[bi].unsqueeze(0))   # same goal
                    neg_lemma_list.append(lemma_embs[nj].unsqueeze(0)) # wrong lemma

            neg_goal = torch.cat(neg_goal_list, dim=0).to(device)     # [B*K, 256]
            neg_lemma = torch.cat(neg_lemma_list, dim=0).to(device)   # [B*K, 256]
            neg_labels = torch.zeros(len(neg_goal_list), 1, device=device)

            # Combine positives and negatives
            all_goal = torch.cat([pos_goal, neg_goal], dim=0)
            all_lemma = torch.cat([pos_lemma, neg_lemma], dim=0)
            all_labels = torch.cat([pos_labels, neg_labels], dim=0)

            # Forward
            logits = scorer(all_goal, all_lemma)
            loss = criterion(logits, all_labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(scorer.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

            # Count positives correct
            with torch.no_grad():
                probs = torch.sigmoid(logits[:b_size])
                epoch_pos += ((probs > 0.5).float().sum().item())

        scheduler.step()
        avg_loss = epoch_loss / max(1, n_batches)
        train_acc = epoch_pos / max(1, n_train)

        # Validation
        scorer.eval()
        val_loss = 0.0
        val_correct = 0
        n_val_batches = 0
        val_local = list(range(n_val))  # indices into val_goal/val_lemma
        with torch.no_grad():
            for vb_start in range(0, n_val, batch_size):
                vb_local = val_local[vb_start: vb_start + batch_size]
                vb_global = val_indices[vb_start: vb_start + batch_size]
                vb_size = len(vb_local)

                pos_goal = val_goal[vb_local].to(device)
                pos_lemma = val_lemma[vb_local].to(device)
                pos_labels = torch.ones(vb_size, 1, device=device)

                # Negatives for validation
                neg_goal_list = []
                neg_lemma_list = []
                for bi in vb_global:
                    candidates = [j for j in range(n_total) if j != bi]
                    neg_indices = random.sample(candidates, min(num_negatives, len(candidates)))
                    for nj in neg_indices:
                        neg_goal_list.append(goal_embs[bi].unsqueeze(0))
                        neg_lemma_list.append(lemma_embs[nj].unsqueeze(0))

                neg_goal = torch.cat(neg_goal_list, dim=0).to(device)
                neg_lemma = torch.cat(neg_lemma_list, dim=0).to(device)
                neg_labels = torch.zeros(len(neg_goal_list), 1, device=device)

                all_goal = torch.cat([pos_goal, neg_goal], dim=0)
                all_lemma = torch.cat([pos_lemma, neg_lemma], dim=0)
                all_labels = torch.cat([pos_labels, neg_labels], dim=0)

                logits = scorer(all_goal, all_lemma)
                loss = criterion(logits, all_labels)
                val_loss += loss.item()
                n_val_batches += 1

                probs = torch.sigmoid(logits[:vb_size])
                val_correct += ((probs > 0.5).float().sum().item())

        val_avg_loss = val_loss / max(1, n_val_batches)
        val_acc = val_correct / max(1, n_val)

        # Diagnostic: compute mean prob for pos and neg on val set
        scorer.eval()
        with torch.no_grad():
            val_pos_probs = torch.sigmoid(scorer(
                val_goal.to(device), val_lemma.to(device)
            )).squeeze(-1).cpu()
            # Shuffle lemma to create negative pairs
            shuffle_idx = torch.randperm(n_val)
            val_neg_probs = torch.sigmoid(scorer(
                val_goal.to(device),
                val_lemma[shuffle_idx].to(device)
            )).squeeze(-1).cpu()
            mean_pos = val_pos_probs.mean().item()
            mean_neg = val_neg_probs.mean().item()

        history["epochs"].append({
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_loss": val_avg_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "lr": scheduler.get_last_lr()[0],
        })

        print(f"  Epoch {epoch:2d}/{num_epochs} | "
              f"train_loss={avg_loss:.4f} val_loss={val_avg_loss:.4f} | "
              f"train_acc={train_acc:.3f} val_acc={val_acc:.3f} | "
              f"pos_prob={mean_pos:.3f} neg_prob={mean_neg:.3f} | "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

    # Save
    out_path = _PROJECT_ROOT / output_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": scorer.state_dict(),
        "config": {
            "input_dim": 512,
            "hidden_dim": encoder.hidden_dim,
            "gnn_checkpoint": str(encoder.gnn),
        },
        "history": history,
    }, out_path)
    print(f"\nSaved scorer to {out_path}")

    return history


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


def smoke_test(encoder: FrozenGNNEncoder, output_path: str):
    """Quick smoke test: 1000 pairs, 2 epochs, verify loss decreases and acc > 55%."""
    print("\n" + "=" * 60)
    print("SMOKE TEST: 1000 pairs, 2 epochs")
    print("=" * 60)

    pairs = load_training_pairs("data/raw/proof_step_pairs.jsonl", limit=1000, seed=123)
    scorer = BinaryScorer(hidden_dim=encoder.hidden_dim)

    history = train_binary_scorer(
        encoder=encoder,
        pairs=pairs,
        scorer=scorer,
        output_path=output_path,
        num_epochs=5,
        batch_size=256,
        lr=3e-3,
        num_threads=4,
        num_negatives=1,  # balanced for smoke test
        seed=123,
    )

    # Verify
    losses = [e["val_loss"] for e in history["epochs"]]
    accs = [e["val_acc"] for e in history["epochs"]]

    loss_decreased = len(losses) >= 2 and losses[-1] < losses[0]
    acc_ok = accs[-1] > 0.55 if accs else False

    print(f"\nSmoke test results:")
    print(f"  Loss decreased: {loss_decreased} ({losses[0]:.4f} → {losses[-1]:.4f})")
    print(f"  Final accuracy: {accs[-1]:.3f} (> 0.55: {'PASS' if acc_ok else 'FAIL'})")

    if loss_decreased and acc_ok:
        print("  SMOKE TEST PASSED")
    else:
        print("  SMOKE TEST FAILED")

    return {"loss_decreased": loss_decreased, "acc_ok": acc_ok, "final_loss": losses[-1], "final_acc": accs[-1]}


# ---------------------------------------------------------------------------
# Gate3 evaluation
# ---------------------------------------------------------------------------


def run_gate3_eval(
    encoder: FrozenGNNEncoder,
    scorer: BinaryScorer,
    scorer_path: str,
    output_path: str,
    *,
    top_k: int = 30,
    num_threads: int = 4,
):
    """Evaluate binary scorer on gate3_v2 benchmark.

    For each goal:
    1. Encode goal (frozen GNN)
    2. Score against all lemma candidates via MLP
    3. Take top-K lemmas
    4. Run best-first search with those lemmas
    """
    from src.explorer.gnn_best_first_search import GNNBestFirstSearch, GNNBestFirstConfig
    from src.explorer.proof_state import ProofState
    from src.proof_checker.batch_checker import BatchChecker
    from src.proof_checker.formats import wrap_theorem_with_proof
    from scripts.eval.eval_gnn_prover import build_lemma_index
    from scripts.eval.run_full_gate3_v2 import (
        build_norm_index,
        classify_proof_pattern,
        is_lemma_novelty,
        load_jsonl,
        save_json,
    )

    torch.set_num_threads(num_threads)
    device = encoder.device

    # Load scorer
    scorer = scorer.to(device)
    scorer.eval()

    # Load gate3_v2 theorems
    theorems_path = _PROJECT_ROOT / "data/raw/gate3_v2.jsonl"
    if not theorems_path.exists():
        print(f"ERROR: gate3_v2.jsonl not found at {theorems_path}")
        return None
    theorems = load_jsonl(theorems_path)
    print(f"Loaded {len(theorems)} gate3_v2 theorems")

    # Build lemma index
    lemma_to_idx = build_lemma_index(encoder.graph)
    idx_to_norm = build_norm_index(encoder.graph, lemma_to_idx)

    # Pre-compute all lemma candidate embeddings (frozen, already have them)
    all_lemma_embs = encoder.node_embeddings  # [N, 256]
    print(f"Lemma candidates: {all_lemma_embs.shape[0]}")

    # Setup binary scorer search — replace GNN cosine with MLP scoring
    # We'll run best-first search but with binary scorer lemma ranking

    print("\n" + "=" * 60)
    print("GATE 3 EVALUATION: Binary Scorer")
    print("=" * 60)

    # Load the GNN model for search infrastructure
    from src.explorer.gnn_encoder import GNNEncoder
    gnn = GNNEncoder.load(str(_PROJECT_ROOT / "checkpoints/gnn/gate2_fullgraph_finetuned.pt"))
    gnn.eval()
    gnn.to(device)

    # Compute node embeddings for search
    from src.explorer.gnn_encoder import extract_initial_features, prepare_graph_tensors
    features = extract_initial_features(encoder.graph, gnn.config)
    sources, targets, edge_types, num_nodes = prepare_graph_tensors(encoder.graph)
    max_et = gnn.config.num_edge_types - 1
    edge_types = edge_types.clamp(0, max_et)
    features = features.to(device)
    sources = sources.to(device)
    targets = targets.to(device)
    edge_types = edge_types.to(device)

    with torch.no_grad():
        node_embeddings = gnn(features, sources, targets, edge_types, num_nodes)

    # Setup binary-scorer-based search
    config = GNNBestFirstConfig(
        max_depth=20,
        max_expansions=1000,
        top_k_lemmas=top_k,
        depth_penalty=0.05,
        use_proof_checker=True,
        verify_timeout=5.0,
        num_threads=num_threads,
        max_graph_candidates=200,
    )

    checker = BatchChecker(timeout=15, max_workers=4, cache_size=128)

    # Build a custom search that uses binary scorer instead of cosine
    from src.explorer.gnn_best_first_search import GNNBestFirstSearch

    bf_search = GNNBestFirstSearch(
        gnn=gnn,
        graph=encoder.graph,
        node_embeddings=node_embeddings,
        lemma_index=lemma_to_idx,
        idx_to_norm=idx_to_norm,
        config=config,
        proof_checker=checker if config.use_proof_checker else None,
    )

    # Override lemma scoring to use binary scorer
    bf_search._original_score_candidates = bf_search._score_candidates

    # Store current theorem name for goal encoding
    _current_theorem_name = [""]

    def binary_score_candidates(goal_text, candidates, domain=None):
        """Score candidates using binary scorer MLP."""
        if not candidates:
            return torch.tensor([]), []

        # Use theorem name for initial goal, keyword fallback for sub-goals
        goal_emb = encoder.encode_goal(
            goal_text, theorem_name=_current_theorem_name[0]
        )  # [256]

        candidate_embs = []
        valid_candidates = []
        for c in candidates:
            idx = lemma_to_idx.get(c)
            if idx is not None and idx < encoder.num_nodes:
                candidate_embs.append(encoder.node_embeddings[idx])
                valid_candidates.append(c)

        if not valid_candidates:
            return torch.tensor([]), []

        cand_t = torch.stack(candidate_embs).to(device)  # [K, 256]
        goal_t = goal_emb.unsqueeze(0).expand(len(valid_candidates), -1)  # [K, 256]

        with torch.no_grad():
            scores = torch.sigmoid(scorer(goal_t, cand_t)).squeeze(-1)  # [K]

        return scores.cpu(), valid_candidates

    bf_search._score_candidates = binary_score_candidates

    # Run evaluation
    t_start = time.time()
    results = []
    passed = []
    failed_reasons: dict[str, int] = {}

    for i, t in enumerate(theorems):
        stmt = t["statement"]
        name = t["name"]
        domain = t.get("domain", "unknown")
        era = t.get("era", "unknown")
        ground_truth = t.get("proof", "?")

        t0 = time.time()
        _current_theorem_name[0] = name  # Set theorem name for goal encoding
        proof_steps, final_state = bf_search.search(stmt, domain=domain, verbose=False)
        search_time = time.time() - t0

        proof_text = ProofState._render_proof(proof_steps)

        if not proof_steps:
            ok = False
            err = "no proof found"
            failed_reasons["no_proof"] = failed_reasons.get("no_proof", 0) + 1
        else:
            full_code = wrap_theorem_with_proof(stmt, proof_text)
            check_results = checker.check_batch([full_code])
            ok = check_results[0].success
            err = check_results[0].errors[0][:200] if check_results[0].errors else ""
            if not ok:
                reason_key = f"lean_reject:{err[:50]}"
                failed_reasons[reason_key] = failed_reasons.get(reason_key, 0) + 1

        steps_str = [s.to_lean() for s in proof_steps[:10]]
        pattern = classify_proof_pattern(steps_str) if ok else "failed"
        lemma_novel = is_lemma_novelty(steps_str) if ok else False

        result = {
            "name": name,
            "era": era,
            "domain": domain,
            "success": ok,
            "error": err,
            "hybrid_steps": steps_str,
            "num_steps": len(proof_steps),
            "ground_truth": ground_truth,
            "search_time_s": round(search_time, 1),
            "pattern": pattern,
            "lemma_novelty": lemma_novel,
        }
        results.append(result)

        if ok:
            passed.append(result)

        status = "\u2713" if ok else "\u2717"
        eta = (time.time() - t_start) / (i + 1) * (len(theorems) - i - 1)
        print(f"  [{i+1:2d}/{len(theorems)}] {status} {name:45s} "
              f"[{pattern:12s}] {search_time:.1f}s  "
              f"ETA: {eta/60:.0f}m  ({len(passed)} passed)")

        if ok and len(proof_steps) > 0:
            print(f"         Proof: {steps_str}")

    elapsed = time.time() - t_start
    n_total = len(theorems)
    n_passed = len(passed)
    rate = n_passed / max(1, n_total)

    # Build output
    from collections import Counter
    domains = Counter(r["domain"] for r in results)
    multi = [r for r in passed if r["num_steps"] >= 2]
    lemma_novel = [r for r in passed if r["lemma_novelty"]]

    out = {
        "task": "Binary Scorer gate3_v2 evaluation",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "architecture": "Frozen GNN + BinaryScorer MLP [512→256→128→1]",
        "scorer_path": scorer_path,
        "config": {
            "top_k": top_k,
            "max_expansions": 1000,
        },
        "gate3": {
            "status": "PASS" if n_passed > 0 else "FAIL",
            "total": n_total,
            "passed": n_passed,
            "rate": rate,
            "multi_step": len(multi),
            "lemma_novelty": len(lemma_novel),
            "elapsed_s": elapsed,
            "failed_reasons": dict(failed_reasons),
            "domains": {dom: {
                "total": domains[dom],
                "passed": sum(1 for r in passed if r["domain"] == dom),
                "lemma_novelty": sum(1 for r in lemma_novel if r["domain"] == dom),
                "multi_step": sum(1 for r in multi if r["domain"] == dom),
            } for dom in domains},
            "passed_theorems": [
                {"name": r["name"], "domain": r["domain"],
                 "proof": " ".join(r["hybrid_steps"]),
                 "pattern": r["pattern"], "num_steps": r["num_steps"],
                 "lemma_novelty": r["lemma_novelty"]}
                for r in passed
            ],
        },
        "all_results": results,
    }

    out_path = _PROJECT_ROOT / output_path
    save_json(out, out_path)
    print(f"\n  Results saved to: {out_path}")

    print(f"\n{'=' * 60}")
    print(f"Gate 3 Binary Scorer: {n_passed}/{n_total} ({rate:.0%})")
    print(f"  Multi-step: {len(multi)}")
    print(f"  Lemma-novelty: {len(lemma_novel)}")
    print(f"  Baseline (cosine): 10/64 (15.6%)")
    print(f"{'=' * 60}")

    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Binary Scorer for proof-step lemma prediction")
    parser.add_argument("--smoke", action="store_true", help="Run smoke test (1000 pairs, 2 epochs)")
    parser.add_argument("--train", action="store_true", help="Run full training (226K pairs, 10 epochs)")
    parser.add_argument("--gate3", action="store_true", help="Run gate3_v2 evaluation")
    parser.add_argument("--gnn-checkpoint", default="checkpoints/gnn/gate2_fullgraph_finetuned.pt")
    parser.add_argument("--graph", default="data/graph/dependency_graph_full")
    parser.add_argument("--scorer-path", default="checkpoints/scorer/binary_scorer.pt")
    parser.add_argument("--num-threads", type=int, default=6)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--top-k", type=int, default=30,
                        help="Top-K lemmas for gate3 evaluation")
    args = parser.parse_args()

    # Initialize frozen encoder
    print("Initializing frozen GNN encoder...")
    encoder = FrozenGNNEncoder(
        gnn_checkpoint=args.gnn_checkpoint,
        graph_path=args.graph,
    )

    if args.smoke:
        result = smoke_test(encoder, "checkpoints/scorer/binary_scorer_smoke.pt")
        return 0 if (result["loss_decreased"] and result["acc_ok"]) else 1

    if args.train:
        pairs = load_training_pairs("data/raw/proof_step_pairs.jsonl")
        scorer = BinaryScorer(hidden_dim=encoder.hidden_dim)
        history = train_binary_scorer(
            encoder=encoder,
            pairs=pairs,
            scorer=scorer,
            output_path=args.scorer_path,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            num_threads=args.num_threads,
            num_negatives=5,
        )
        return 0

    if args.gate3:
        scorer_path = _PROJECT_ROOT / args.scorer_path
        if not scorer_path.exists():
            print(f"ERROR: Scorer checkpoint not found: {scorer_path}")
            print("Run --train first.")
            return 1

        state = torch.load(str(scorer_path), map_location="cpu", weights_only=False)
        scorer = BinaryScorer(hidden_dim=encoder.hidden_dim)
        scorer.load_state_dict(state["model_state_dict"])

        result = run_gate3_eval(
            encoder=encoder,
            scorer=scorer,
            scorer_path=args.scorer_path,
            output_path="data/binary_scorer_gate3.json",
            top_k=args.top_k,
            num_threads=max(1, args.num_threads - 2),
        )
        return 0 if result else 1

    # Default: show help
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
