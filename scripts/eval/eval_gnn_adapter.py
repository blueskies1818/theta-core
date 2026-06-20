#!/usr/bin/env python3
"""Evaluate GNN+Adapter on gate3_v2 benchmark.

Loads frozen GNN + trained adapter, runs best-first search
on all 64 gate3_v2 theorems. Compares to GNN-only baseline.

Usage:
    # Smoke test
    python scripts/eval/eval_gnn_adapter.py \
        --gnn-checkpoint checkpoints/gnn/full_graph_pretrained.pt \
        --adapter data/adapter_smoke/adapter.pt \
        --output data/adapter_smoke_eval.json \
        --max-theorems 5

    # Full evaluation
    python scripts/eval/eval_gnn_adapter.py \
        --gnn-checkpoint checkpoints/gnn/full_graph_pretrained.pt \
        --adapter data/adapter_full/adapter.pt \
        --output data/adapter_gate3_result.json
"""

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))

from src.explorer.dependency_graph import DependencyGraph
from src.explorer.gnn_encoder import (
    GNNEncoder,
    extract_initial_features,
    prepare_graph_tensors,
)
from src.explorer.gnn_adapter import GNNAdapterHead
from src.explorer.gnn_best_first_search import GNNBestFirstSearch, GNNBestFirstConfig
from src.explorer.proof_state import ProofState
from src.proof_checker.batch_checker import BatchChecker
from src.proof_checker.formats import wrap_theorem_with_proof
from scripts.eval.eval_gnn_prover import (
    build_lemma_index,
    extract_conclusion,
    normalize_expression,
)


# ---------------------------------------------------------------------------
# Adapter-aware GNN wrapper
# ---------------------------------------------------------------------------

class AdapterGNN(nn.Module):
    """Wraps a frozen GNN + adapter to produce adapted embeddings.

    All GNN parameters remain frozen; adapter is applied to node
    embeddings and goal encodings.
    """

    def __init__(self, gnn: GNNEncoder, adapter: GNNAdapterHead):
        super().__init__()
        self._gnn = gnn
        self._adapter = adapter
        self.config = gnn.config
        self.goal_encoder = None  # Disable — we handle this in encode_goal

    def encode_goal(self, context_embedding: torch.Tensor) -> torch.Tensor:
        """Encode goal: GNN goal encoder → adapter."""
        if self._gnn.goal_encoder is not None:
            out = self._gnn.goal_encoder(context_embedding)
        else:
            out = F.normalize(context_embedding, dim=-1) if context_embedding.norm() > 0 else context_embedding
        return self._adapter(out)

    def forward(self, *args, **kwargs):
        """Delegate to frozen GNN (used for computing node embeddings)."""
        return self._gnn(*args, **kwargs)

    def eval(self):
        self._gnn.eval()
        self._adapter.eval()
        return self

    def parameters(self, recurse=True):
        return []  # No trainable params at eval time


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def classify_proof_pattern(proof_steps: list[str]) -> str:
    if not proof_steps:
        return "empty"
    tactic_types = set()
    for step in proof_steps:
        s = step.strip().lower()
        if s.startswith("rw"): tactic_types.add("rw")
        elif s.startswith("exact"): tactic_types.add("exact")
        elif s.startswith("apply"): tactic_types.add("apply")
        elif s.startswith("intro"): tactic_types.add("intro")
        elif s.startswith("have"): tactic_types.add("have")
        elif s in ("ring", "simp", "linarith", "field_simp", "positivity",
                    "norm_num", "nlinarith"):
            tactic_types.add(s)
        elif s.startswith("calc"): tactic_types.add("calc")
        elif s.startswith("constructor"): tactic_types.add("constructor")
        else: tactic_types.add("other")
    if len(tactic_types) >= 2:
        return "multi"
    steps_text = " ".join(proof_steps).lower()
    for p in ["rfl", "add_comm", "mul_comm", "ring", "field_simp",
              "linarith", "simp", "intro", "apply", "nlinarith"]:
        if p in steps_text:
            return p
    return "other"


def build_norm_index(graph: DependencyGraph, lemma_to_idx: dict[str, int]
                     ) -> dict[int, str]:
    idx_to_norm: dict[int, str] = {}
    for node_id in graph.node_ids:
        idx = lemma_to_idx.get(node_id)
        if idx is None:
            continue
        node = graph.get_node(node_id)
        if node:
            statement = node.get("statement", "")
            if statement:
                conclusion = extract_conclusion(statement)
                if conclusion:
                    idx_to_norm[idx] = normalize_expression(conclusion)
    return idx_to_norm


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def eval_adapter(
    gnn_checkpoint: str,
    adapter_checkpoint: str,
    graph_path: str = "data/graph/dependency_graph_full.nx.pkl",
    theorems_path: str = "data/raw/gate3_v2.jsonl",
    output_path: str = "data/adapter_gate3_result.json",
    num_threads: int = 4,
    max_theorems: int | None = None,
    use_proof_checker: bool = False,
    max_expansions: int = 5000,
    top_k_lemmas: int = 30,
    value_weight: float = 0.3,
    use_domain_filter: bool = True,
):
    print("=" * 70)
    print("GNN+ADAPTER EVALUATION: gate3_v2 Benchmark")
    print("=" * 70)

    # --- Load GNN ---
    print(f"\nLoading GNN from {gnn_checkpoint}...")
    gnn = GNNEncoder.load(gnn_checkpoint)
    for p in gnn.parameters():
        p.requires_grad = False
    gnn.eval()
    gnn_params = sum(p.numel() for p in gnn.parameters())
    print(f"  GNN params: {gnn_params:,} (frozen)")

    # --- Load adapter ---
    print(f"Loading adapter from {adapter_checkpoint}...")
    adapter = GNNAdapterHead()
    adapter.load_state_dict(torch.load(adapter_checkpoint, map_location="cpu",
                                        weights_only=True))
    adapter.eval()
    adapter_params = sum(p.numel() for p in adapter.parameters())
    print(f"  Adapter params: {adapter_params:,}")
    print(f"  Gate A: {'PASS' if adapter_params <= 150000 else 'FAIL'}")

    # --- Wrap GNN with adapter ---
    wrapped_gnn = AdapterGNN(gnn, adapter)
    wrapped_gnn.eval()

    # --- Load graph ---
    print(f"Loading graph from {graph_path}...")
    graph = DependencyGraph.load(graph_path)
    print(f"  Nodes: {graph.num_nodes:,}")

    lemma_to_idx = build_lemma_index(graph)
    idx_to_norm = build_norm_index(graph, lemma_to_idx)
    print(f"  Lemma index: {len(lemma_to_idx):,} entries")

    # --- Compute node embeddings ---
    print("\nComputing node embeddings (GNN + adapter)...")
    features = extract_initial_features(graph, gnn.config)
    sources, targets, edge_types, num_nodes = prepare_graph_tensors(graph)
    print(f"  Graph: {num_nodes} nodes, {sources.size(0)} edges")

    with torch.no_grad():
        raw_embeddings = gnn(features, sources, targets, edge_types, num_nodes)
        adapted_embeddings = adapter(raw_embeddings)
    print(f"  Embeddings: {adapted_embeddings.shape}")

    # --- Load theorems ---
    theorems = load_jsonl(Path(theorems_path))
    if max_theorems:
        theorems = theorems[:max_theorems]
    print(f"Theorems: {len(theorems)}")

    # --- Setup search ---
    config = GNNBestFirstConfig(
        max_expansions=max_expansions,
        top_k_lemmas=top_k_lemmas,
        num_threads=num_threads,
        use_proof_checker=use_proof_checker,
        value_weight=value_weight,
    )

    bf_search = GNNBestFirstSearch(
        gnn=wrapped_gnn,
        graph=graph,
        node_embeddings=adapted_embeddings,
        lemma_index=lemma_to_idx,
        idx_to_norm=idx_to_norm,
        config=config,
    )

    # --- Run evaluation ---
    print(f"\n--- Running GNN+Adapter best-first search on {len(theorems)} theorems ---")
    print(f"    Max expansions: {max_expansions}, Top-K: {top_k_lemmas}")
    print(f"    Threads: {num_threads}, Domain filter: {use_domain_filter}")

    results = []
    t_start = time.time()
    domain_stats: dict[str, dict] = {}

    for i, t in enumerate(theorems):
        stmt = t["statement"]
        name = t["name"]
        domain = t.get("domain", "unknown")
        era = t.get("era", "unknown")
        ground_truth = t.get("proof", "?")

        t0 = time.time()
        search_domain = domain if use_domain_filter else None
        proof_steps, final_state = bf_search.search(stmt, domain=search_domain,
                                                      verbose=False)
        search_time = time.time() - t0

        proof_text = ProofState._render_proof(proof_steps)

        if not proof_steps:
            ok = False
            err = "no proof found"
        elif use_proof_checker:
            full_code = wrap_theorem_with_proof(stmt, proof_text)
            checker = BatchChecker(timeout=10.0, max_workers=1)
            check_results = checker.check_batch([full_code])
            ok = check_results[0].success
            err = check_results[0].errors[0][:100] if check_results[0].errors else ""
        else:
            ok = True
            err = ""

        pattern = classify_proof_pattern([str(s) for s in proof_steps])

        result = {
            "name": name,
            "domain": domain,
            "era": era,
            "success": ok,
            "error": err if not ok else "",
            "proof_text": proof_text,
            "proof_steps_raw": [str(s) for s in proof_steps],
            "ground_truth": ground_truth,
            "pattern": pattern,
            "search_time": round(search_time, 2),
            "num_steps": len(proof_steps),
        }
        results.append(result)

        # Domain tracking
        if domain not in domain_stats:
            domain_stats[domain] = {"total": 0, "proved": 0, "time": 0.0}
        domain_stats[domain]["total"] += 1
        domain_stats[domain]["time"] += search_time
        if ok:
            domain_stats[domain]["proved"] += 1

        # Progress
        status = "✓" if ok else "✗"
        print(f"  [{i+1:2d}/{len(theorems)}] {status} {name} "
              f"({domain}, {search_time:.1f}s) {pattern}")

    total_time = time.time() - t_start

    # --- Summary ---
    n_proved = sum(1 for r in results if r["success"])
    n_total = len(results)
    rate = n_proved / n_total * 100 if n_total > 0 else 0

    print(f"\n{'='*70}")
    print(f"RESULTS: {n_proved}/{n_total} proved ({rate:.1f}%)")
    print(f"Total time: {total_time:.1f}s")

    print(f"\nPer domain:")
    for domain in sorted(domain_stats):
        ds = domain_stats[domain]
        dr = ds["proved"] / ds["total"] * 100 if ds["total"] > 0 else 0
        avg_time = ds["time"] / ds["total"] if ds["total"] > 0 else 0
        print(f"  {domain:20s}: {ds['proved']:2d}/{ds['total']:2d} "
              f"({dr:5.1f}%) avg {avg_time:.1f}s")

    print(f"\nPer pattern:")
    pattern_counts = Counter(r["pattern"] for r in results if r["success"])
    for pat, count in pattern_counts.most_common():
        print(f"  {pat:20s}: {count:2d}")

    # --- Compare to baseline ---
    baseline_rate = 15.6  # Known baseline
    print(f"\n--- Gate E: Post-training Gate 3 ---")
    print(f"  Adapter rate: {rate:.1f}%")
    print(f"  Baseline:     {baseline_rate:.1f}%")
    if rate > baseline_rate:
        print(f"  PASS: +{rate - baseline_rate:.1f}pp over baseline")
    else:
        print(f"  FAIL: {baseline_rate - rate:.1f}pp below baseline")

    # --- Save results ---
    output = {
        "model": "GNN + Adapter",
        "gnn_checkpoint": gnn_checkpoint,
        "adapter_checkpoint": adapter_checkpoint,
        "n_proved": n_proved,
        "n_total": n_total,
        "rate": round(rate, 2),
        "baseline_rate": baseline_rate,
        "total_time": round(total_time, 1),
        "domain_stats": {k: {
            "proved": v["proved"],
            "total": v["total"],
            "rate": round(v["proved"] / v["total"] * 100, 1) if v["total"] > 0 else 0,
            "avg_time": round(v["time"] / v["total"], 1) if v["total"] > 0 else 0,
        } for k, v in domain_stats.items()},
        "pattern_counts": dict(pattern_counts),
        "results": results,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate GNN+Adapter on gate3_v2 benchmark",
    )
    parser.add_argument("--gnn-checkpoint", required=True,
                        default="checkpoints/gnn/full_graph_pretrained.pt")
    parser.add_argument("--adapter", required=True,
                        default="data/adapter_full/adapter.pt")
    parser.add_argument("--graph", default="data/graph/dependency_graph_full.nx.pkl")
    parser.add_argument("--theorems", default="data/raw/gate3_v2.jsonl")
    parser.add_argument("--output", default="data/adapter_gate3_result.json")
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--max-theorems", type=int, default=None)
    parser.add_argument("--use-proof-checker", action="store_true")
    parser.add_argument("--max-expansions", type=int, default=5000)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--value-weight", type=float, default=0.3)
    parser.add_argument("--no-domain-filter", action="store_true")

    args = parser.parse_args()

    eval_adapter(
        gnn_checkpoint=args.gnn_checkpoint,
        adapter_checkpoint=args.adapter,
        graph_path=args.graph,
        theorems_path=args.theorems,
        output_path=args.output,
        num_threads=args.num_threads,
        max_theorems=args.max_theorems,
        use_proof_checker=args.use_proof_checker,
        max_expansions=args.max_expansions,
        top_k_lemmas=args.top_k,
        value_weight=args.value_weight,
        use_domain_filter=not args.no_domain_filter,
    )


if __name__ == "__main__":
    main()
