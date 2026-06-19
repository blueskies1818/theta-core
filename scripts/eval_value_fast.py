#!/usr/bin/env python3
"""Fast value network eval: share graph + embeddings between runs.

Loads graph + GNN once, shares embeddings, runs blind + value-guided
in sequence on gate3_v2 theorems. Much faster than loading twice.

Usage:
  python scripts/eval_value_fast.py \
      --gnn-checkpoint checkpoints/gnn/proof_step_pretrained.pt \
      --value-checkpoint checkpoints/value_network.pt \
      --graph data/graph/dependency_graph \
      --theorems data/raw/gate3_v2.jsonl \
      --output data/value_net_result.json \
      --max-expansions 200 --timeout-per-theorem 30
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import threading
from collections import Counter
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.explorer.dependency_graph import DependencyGraph
from src.explorer.gnn_encoder import (
    GNNEncoder,
    extract_initial_features,
    prepare_graph_tensors,
)
from src.explorer.gnn_best_first_search import GNNBestFirstSearch, GNNBestFirstConfig
from src.explorer.value_network import ValueNetwork
from src.explorer.proof_state import ProofState
from src.proof_checker.batch_checker import BatchChecker
from src.proof_checker.formats import wrap_theorem_with_proof
from scripts.eval_gnn_prover import (
    build_lemma_index,
    build_lemma_norm_index,
)


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


class TimeoutError(Exception):
    pass


def search_with_timeout(bf, stmt, domain, timeout_s):
    result = [None]
    exc = [None]
    done = threading.Event()

    def _run():
        try:
            result[0] = bf.search(stmt, domain=domain, verbose=False)
        except Exception as e:
            exc[0] = e
        finally:
            done.set()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    if not done.wait(timeout=timeout_s):
        raise TimeoutError()
    if exc[0]:
        raise exc[0]
    return result[0]


def run_search(bf, theorems, checker, timeout_s, label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    results = []
    passed = []
    timeouts = 0
    t0 = time.time()

    for i, thm in enumerate(theorems):
        stmt = thm["statement"]
        name = thm["name"]
        domain = thm.get("domain", "unknown")
        t1 = time.time()

        try:
            steps, final_state = search_with_timeout(bf, stmt, domain, timeout_s)
        except TimeoutError:
            steps = []
            ok = False
            timeouts += 1
            err = "timeout"
        else:
            if not steps:
                ok = False
                err = "no proof"
            else:
                code = wrap_theorem_with_proof(stmt, ProofState._render_proof(steps))
                cr = checker.check_batch([code])
                ok = cr[0].success
                err = cr[0].errors[0][:100] if cr[0].errors else ""

        dt = time.time() - t1
        steps_str = [s.to_lean() for s in steps[:5]]
        results.append({
            "name": name, "domain": domain, "success": ok,
            "error": err, "steps": steps_str,
            "num_steps": len(steps), "time_s": round(dt, 1),
        })
        if ok:
            passed.append(name)

        eta = (time.time() - t0) / (i + 1) * (len(theorems) - i - 1) if i > 0 else 0
        status = "OK" if ok else "FAIL"
        print(f"  [{i+1:2d}/{len(theorems)}] {status:4s} {name:40s} "
              f"{dt:.1f}s  ETA:{eta/60:.0f}m  ({len(passed)} ok)")

    elapsed = time.time() - t0
    rate = len(passed) / len(theorems)
    ms = sum(1 for r in results if r["success"] and r["num_steps"] >= 2)

    print(f"\n  Result: {len(passed)}/{len(theorems)} ({rate:.0%}), "
          f"{ms} multi-step, {timeouts} timeouts, {elapsed:.0f}s")

    return {
        "label": label, "total": len(theorems), "passed": len(passed),
        "rate": rate, "multi_step": ms, "timeouts": timeouts,
        "elapsed_s": elapsed, "results": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gnn-checkpoint", default="checkpoints/gnn/proof_step_pretrained.pt")
    parser.add_argument("--value-checkpoint", default="checkpoints/value_network.pt")
    parser.add_argument("--graph", default="data/graph/dependency_graph")
    parser.add_argument("--theorems", default="data/raw/gate3_v2.jsonl")
    parser.add_argument("--output", default="data/value_net_result.json")
    parser.add_argument("--max-expansions", type=int, default=200)
    parser.add_argument("--timeout-per-theorem", type=float, default=30.0)
    parser.add_argument("--value-weight", type=float, default=0.3)
    parser.add_argument("--num-threads", type=int, default=6)
    parser.add_argument("--no-blind", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(min(args.num_threads, 6))
    root = _PROJECT_ROOT

    print("=" * 60)
    print("VALUE NETWORK EVAL (fast shared-embeddings)")
    print("=" * 60)

    # Load GNN
    gnn_path = root / args.gnn_checkpoint
    if not gnn_path.exists():
        gnn_path = root / "checkpoints/gnn/gate2_fullgraph_finetuned.pt"
    print(f"GNN: {gnn_path}")
    gnn = GNNEncoder.load(str(gnn_path))
    gnn.eval()
    print(f"  {sum(p.numel() for p in gnn.parameters()):,} params, "
          f"hidden={gnn.config.hidden_dim}")

    # Load value network
    vn = None
    vn_path = root / args.value_checkpoint
    if vn_path.exists():
        print(f"Value: {vn_path}")
        vn = ValueNetwork.load(str(vn_path), gnn, freeze_encoder=True)
        vn.eval()
    else:
        print("WARNING: No value checkpoint, running blind only")
        args.value_weight = 0.0

    # Load graph
    graph_path = root / args.graph
    if not graph_path.with_suffix(".nx.pkl").exists():
        graph_path = root / "data/graph/dependency_graph"
    print(f"Graph: {graph_path}")
    graph = DependencyGraph.load(graph_path)
    print(f"  {graph.summary()}")

    # Indices
    lemma_to_idx = build_lemma_index(graph)
    idx_to_norm = build_lemma_norm_index(graph, lemma_to_idx)

    # Compute node embeddings ONCE
    print("Computing GNN node embeddings (once)...")
    features = extract_initial_features(graph, gnn.config)
    sources, targets, edge_types, num_nodes = prepare_graph_tensors(graph)

    with torch.no_grad():
        node_embeddings = gnn(features, sources, targets, edge_types, num_nodes)
    print(f"  Embeddings: {node_embeddings.shape}")

    # Load theorems
    thm_path = root / args.theorems
    theorems = load_jsonl(thm_path)
    print(f"Theorems: {len(theorems)}")

    # Checker
    checker = BatchChecker(max_workers=4, timeout=args.timeout_per_theorem * 2)

    results = {}
    all_passed: dict[str, set[str]] = {}
    domain_stats = Counter(t["domain"] for t in theorems)

    # --- RUN BLIND ---
    if not args.no_blind:
        blind_cfg = GNNBestFirstConfig(
            max_expansions=args.max_expansions,
            value_weight=0.0, value_prune_threshold=None,
            num_threads=args.num_threads,
        )
        blind_bf = GNNBestFirstSearch(
            gnn=gnn, graph=graph,
            node_embeddings=node_embeddings,
            lemma_index=lemma_to_idx,
            idx_to_norm=idx_to_norm,
            config=blind_cfg,
            proof_checker=checker,
            value_network=None,
        )
        results["blind"] = run_search(
            blind_bf, theorems, checker,
            args.timeout_per_theorem, "BLIND (value_weight=0.0)"
        )
        all_passed["blind"] = {r["name"] for r in results["blind"]["results"] if r["success"]}

    # --- RUN VALUE-GUIDED ---
    value_cfg = GNNBestFirstConfig(
        max_expansions=args.max_expansions,
        value_weight=args.value_weight,
        value_prune_threshold=0.1,
        num_threads=args.num_threads,
    )
    value_bf = GNNBestFirstSearch(
        gnn=gnn, graph=graph,
        node_embeddings=node_embeddings,
        lemma_index=lemma_to_idx,
        idx_to_norm=idx_to_norm,
        config=value_cfg,
        proof_checker=checker,
        value_network=vn,
    )
    results["value_guided"] = run_search(
        value_bf, theorems, checker,
        args.timeout_per_theorem, f"VALUE (weight={args.value_weight})"
    )
    all_passed["value_guided"] = {r["name"] for r in results["value_guided"]["results"] if r["success"]}

    # --- COMPARISON ---
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")

    v = results["value_guided"]
    print(f"  Value-guided: {v['passed']}/{v['total']} ({v['rate']:.0%}) "
          f"multi-step={v['multi_step']} timeouts={v['timeouts']}")

    if not args.no_blind:
        b = results["blind"]
        print(f"  Blind:        {b['passed']}/{b['total']} ({b['rate']:.0%}) "
              f"multi-step={b['multi_step']} timeouts={b['timeouts']}")
        delta = v["passed"] - b["passed"]
        print(f"  Δ:            {delta:+d} proofs")

        new_proofs = all_passed["value_guided"] - all_passed["blind"]
        lost_proofs = all_passed["blind"] - all_passed["value_guided"]
        if new_proofs:
            print(f"  New (value only): {sorted(new_proofs)}")
        if lost_proofs:
            print(f"  Lost (blind only): {sorted(lost_proofs)}")

    # Domain breakdown
    print(f"\n  By domain:")
    for dom in sorted(domain_stats):
        bp = sum(1 for r in results.get("blind", {}).get("results", [])
                 if r["success"] and r["domain"] == dom) if not args.no_blind else 0
        vp = sum(1 for r in results["value_guided"]["results"]
                 if r["success"] and r["domain"] == dom)
        bt = domain_stats[dom]
        print(f"    {dom:<20} blind={bp}/{bt}  value={vp}/{bt}")

    # Save
    out = {
        "task": "Value network eval (fast shared)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "gnn": str(gnn_path),
            "value_checkpoint": str(vn_path) if vn_path.exists() else "none",
            "value_weight": args.value_weight,
            "max_expansions": args.max_expansions,
            "timeout_per_theorem_s": args.timeout_per_theorem,
        },
        "blind": results.get("blind"),
        "value_guided": results["value_guided"],
    }
    out_path = root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
