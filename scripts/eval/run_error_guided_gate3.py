#!/usr/bin/env python3
"""
Gate 3 evaluation: Error-Guided Lemma Search.

Instead of predicting the right lemma blindly, try a lemma,
capture Lean's error message, and use the error to redirect
toward the correct lemma.

Architecture:
  Goal → GoalOnlyEncoder → top-10 lemmas
  → For each lemma: try, capture Lean error
  → Classify error (could_not_unify, made_no_progress, etc.)
  → Redirect to next lemma based on error type
  → Max 5 initial lemmas × 5 retries = 10 total checks/theorem

Usage:
    python scripts/eval/run_error_guided_gate3.py [--smoke] [--verbose]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))

import torch
import torch.nn.functional as F

from src.retrieval.goal_only_encoder import (
    GoalOnlyEncoder,
    build_vocabulary,
    prepare_lemma_groups,
    _tokenize_batch,
)
from src.explorer.error_guided_search import (
    ErrorGuidedSearch,
    ErrorGuideConfig,
    ErrorType,
    classify_result,
)
from src.explorer.proof_state import Tactic, TacticType
from src.proof_checker.batch_checker import BatchChecker
from src.proof_checker.lean_interface import LeanProofChecker
from src.proof_checker.formats import wrap_theorem_with_proof


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def save_json(data: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_dependency_graph() -> tuple:
    """Load the dependency graph and lemma index for neighbor lookups.

    Returns (graph, lemma_index, lemma_index_to_name).
    """
    graph = None
    lemma_index: dict[str, int] = {}
    lemma_index_to_name: dict[int, str] = {}

    # Try loading the NetworkX graph
    graph_path = _project_root / "data" / "graph" / "dependency_graph_full_v3.nx.pkl"
    if graph_path.exists():
        try:
            import pickle
            import networkx as nx
            with open(graph_path, "rb") as f:
                graph = pickle.load(f)
            print(f"  Loaded graph: {graph.number_of_nodes()} nodes, "
                  f"{graph.number_of_edges()} edges")
        except Exception as e:
            print(f"  WARNING: Could not load v3 graph: {e}")
    else:
        # Fall back to v2
        graph_path = _project_root / "data" / "graph" / "dependency_graph_full_v2.nx.pkl"
        if graph_path.exists():
            try:
                import pickle
                import networkx as nx
                with open(graph_path, "rb") as f:
                    graph = pickle.load(f)
                print(f"  Loaded graph v2: {graph.number_of_nodes()} nodes, "
                      f"{graph.number_of_edges()} edges")
            except Exception as e:
                print(f"  WARNING: Could not load v2 graph: {e}")

    # Load lemma index
    lemma_index_path = (
        _project_root / "data" / "graph" / "dependency_graph_full_v3.lemma_index.json"
    )
    if not lemma_index_path.exists():
        lemma_index_path = (
            _project_root / "data" / "graph" / "dependency_graph_full_v2.lemma_index.json"
        )

    if lemma_index_path.exists():
        with open(lemma_index_path) as f:
            lemma_index = json.load(f)
        # Build reverse index
        lemma_index_to_name = {v: k for k, v in lemma_index.items()}
        print(f"  Lemma index: {len(lemma_index)} names")
    else:
        print(f"  WARNING: Lemma index not found")

    # Wrap graph in a simple interface if it's a raw nx graph
    if graph is not None and not hasattr(graph, 'get_neighborhood'):
        class _GraphWrapper:
            def __init__(self, g, idx_to_name):
                self.graph = g
                self.idx_to_name = idx_to_name

            def get_neighborhood(self, node_id, radius=1, direction="both"):
                neighbors = set()
                nid = str(node_id)
                if not self.graph.has_node(nid):
                    return neighbors
                # BFS for radius
                visited = {nid}
                frontier = {nid}
                for _ in range(radius):
                    next_frontier = set()
                    for node in frontier:
                        if direction in ("both", "out"):
                            for succ in self.graph.successors(node):
                                if succ not in visited:
                                    visited.add(succ)
                                    next_frontier.add(succ)
                                    neighbors.add(succ)
                        if direction in ("both", "in"):
                            for pred in self.graph.predecessors(node):
                                if pred not in visited:
                                    visited.add(pred)
                                    next_frontier.add(pred)
                                    neighbors.add(pred)
                    frontier = next_frontier
                return neighbors

        graph = _GraphWrapper(graph, lemma_index_to_name)

    return graph, lemma_index, lemma_index_to_name


def retrieve_lemmas_for_theorem(
    encoder: GoalOnlyEncoder,
    vocab: dict[str, int],
    theorem: str,
    index_goals: list[str],
    index_lemmas: list[str],
    index_embeddings: torch.Tensor,
    k: int = 10,
    device: torch.device | None = None,
) -> list[tuple[str, float]]:
    """Retrieve top-K lemmas using goal-only encoder + cosine similarity.

    Pipeline:
    1. Encode the theorem statement
    2. Find k-NN training goals
    3. Collect their lemmas, rank by frequency × similarity
    """
    if device is None:
        device = torch.device("cpu")

    # Encode the theorem
    with torch.no_grad():
        batch_ids = _tokenize_batch([theorem], vocab, 128).to(device)
        goal_emb = encoder(batch_ids)  # [1, hidden_dim]
        goal_emb = F.normalize(goal_emb, dim=-1)

        # Cosine similarity with all training goals
        index_emb_norm = F.normalize(index_embeddings, dim=-1)
        sims = (goal_emb @ index_emb_norm.T).squeeze(0)  # [N]

        # Get top-K similar goals
        retrieval_k = min(k * 5, len(index_goals))
        top_sims, top_indices = torch.topk(sims, retrieval_k)

        # Collect lemmas from similar goals, weighted by similarity
        from collections import Counter
        lemma_scores: Counter = Counter()
        lemma_count: Counter = Counter()

        for i in range(retrieval_k):
            idx = top_indices[i].item()
            sim = top_sims[i].item()
            lemma = index_lemmas[idx]
            lemma_scores[lemma] += max(0.0, sim)
            lemma_count[lemma] += 1

        # Rank: frequency × avg_similarity
        ranked = []
        for lemma in lemma_scores:
            avg_sim = lemma_scores[lemma] / lemma_count[lemma]
            freq_bonus = min(1.0, lemma_count[lemma] / 5.0)
            score = avg_sim * (0.5 + 0.5 * freq_bonus)
            ranked.append((lemma, score))

        ranked.sort(key=lambda x: -x[1])
        return ranked[:k]


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def run_error_guided_gate3(
    encoder: GoalOnlyEncoder,
    vocab: dict[str, int],
    index_goals: list[str],
    index_lemmas: list[str],
    index_embeddings: torch.Tensor,
    theorems: list[dict],
    checker: LeanProofChecker,
    graph,
    lemma_index: dict[str, int],
    lemma_index_to_name: dict[int, str],
    output_path: Path,
    verbose: bool = False,
) -> dict:
    """Run gate3 evaluation with error-guided search."""
    config = ErrorGuideConfig(
        max_initial_lemmas=5,
        max_retries=5,
        max_total_checks=10,
        retrieval_top_k=10,
        max_neighbors=5,
        check_timeout=15.0,
        max_theorem_time=120.0,
        structural_first=True,
        num_threads=4,
    )

    searcher = ErrorGuidedSearch(
        encoder=encoder,
        vocab=vocab,
        proof_checker=checker,
        graph=graph,
        lemma_index=lemma_index,
        lemma_index_to_name=lemma_index_to_name,
        config=config,
    )

    print("\n" + "=" * 70)
    print("GATE 3: ERROR-GUIDED LEMMA SEARCH")
    print("=" * 70)
    print(f"  Encoder params: {encoder.count_params():,}")
    print(f"  Retrieval top-K: {config.retrieval_top_k}")
    print(f"  Max initial lemmas: {config.max_initial_lemmas}")
    print(f"  Max retries: {config.max_retries}")
    print(f"  Max total checks/theorem: {config.max_total_checks}")
    print(f"  Graph neighbors: {config.max_neighbors}")
    print()

    results = []
    t_start = time.time()
    passed = []
    error_counts: Counter = Counter()

    for i, t in enumerate(theorems):
        stmt = t["statement"]
        name = t["name"]
        domain = t.get("domain", "unknown")
        era = t.get("era", "unknown")
        ground_truth = t.get("proof", "?")

        # Phase 1: Retrieve top lemmas for this theorem
        device = next(encoder.parameters()).device
        top_lemmas = retrieve_lemmas_for_theorem(
            encoder, vocab, stmt,
            index_goals, index_lemmas, index_embeddings,
            k=config.retrieval_top_k,
            device=device,
        )

        # Phase 2: Error-guided search
        t0 = time.time()
        try:
            proof_steps, stats = searcher.search(
                stmt,
                top_lemmas=top_lemmas,
                verbose=verbose,
            )
        except Exception as e:
            search_time = time.time() - t0
            result = {
                "name": name, "era": era, "domain": domain,
                "success": False,
                "error": f"search crash: {type(e).__name__}: {str(e)[:200]}",
                "proof_steps": [], "num_steps": 0,
                "ground_truth": ground_truth,
                "search_time_s": round(search_time, 1),
                "num_checks": 0,
                "error_types_seen": [],
                "method": "error_guided",
            }
            results.append(result)
            error_counts[f"crash:{type(e).__name__}"] += 1
            print(f"  [{i+1:2d}/{len(theorems)}] ✗ {name:45s} "
                  f"[CRASH      ] {search_time:.1f}s  "
                  f"ETA: 0m  ({len(passed)} passed)")
            continue

        search_time = time.time() - t0

        if proof_steps is None:
            result = {
                "name": name, "era": era, "domain": domain,
                "success": False,
                "error": "no proof found",
                "proof_steps": [], "num_steps": 0,
                "ground_truth": ground_truth,
                "search_time_s": round(search_time, 1),
                "num_checks": stats.get("total_checks", 0),
                "error_types_seen": stats.get("error_types_seen", []),
                "attempts": stats.get("attempts", []),
                "method": "error_guided",
                "top_lemmas": [(n, round(s, 3)) for n, s in top_lemmas[:5]],
            }
            results.append(result)
            error_counts["no_proof"] += 1
        else:
            # Convert Tactics to strings
            proof_steps_str = [s.to_lean() if hasattr(s, 'to_lean') else str(s)
                               for s in proof_steps]
            proof_text = "\n  ".join(proof_steps_str)
            full_code = wrap_theorem_with_proof(stmt, proof_text)
            check_results = checker.check_batch([full_code])
            ok = check_results[0].success
            err = check_results[0].errors[0][:200] if check_results[0].errors else ""

            if not ok:
                error_counts[f"lean_reject:{err[:60]}"] += 1

            result = {
                "name": name, "era": era, "domain": domain,
                "success": ok,
                "error": err,
                "proof_steps": proof_steps_str,
                "num_steps": len(proof_steps),
                "ground_truth": ground_truth,
                "search_time_s": round(search_time, 1),
                "num_checks": stats.get("total_checks", 0),
                "error_types_seen": stats.get("error_types_seen", []),
                "attempts": stats.get("attempts", []),
                "method": "error_guided",
                "top_lemmas": [(n, round(s, 3)) for n, s in top_lemmas[:5]],
            }
            results.append(result)

            if ok:
                passed.append(result)

        status = "✓" if result["success"] else "✗"
        eta = (time.time() - t_start) / (i + 1) * (len(theorems) - i - 1)
        checks = result.get("num_checks", 0)
        print(f"  [{i+1:2d}/{len(theorems)}] {status} {name:45s} "
              f"[{checks} checks] {search_time:.1f}s  "
              f"ETA: {eta/60:.0f}m  ({len(passed)} passed)")

        if result["success"] and len(proof_steps_str) > 0:
            print(f"         Proof: {proof_steps_str}")
            if len(proof_steps) >= 2:
                print(f"         ★ MULTI-STEP ({len(proof_steps)} steps)")

    # Stats
    elapsed = time.time() - t_start
    n_total = len(theorems)
    n_passed = len(passed)
    rate = n_passed / max(1, n_total)

    print(f"\n{'=' * 70}")
    print("GATE 3: ERROR-GUIDED LEMMA SEARCH RESULTS")
    print(f"{'=' * 70}")
    print(f"  Total:    {n_passed}/{n_total} ({rate:.1%})")
    print(f"  Elapsed:  {elapsed:.0f}s ({elapsed/60:.1f}m)")

    # Domain breakdown
    domains = Counter(r["domain"] for r in results)
    print(f"\n  By domain:")
    for dom in sorted(domains.keys()):
        dom_total = domains[dom]
        dom_passed = sum(1 for r in passed if r["domain"] == dom)
        print(f"    {dom:<20} {dom_passed}/{dom_total} "
              f"({dom_passed/max(1,dom_total)*100:.0f}%)")

    print(f"\n  Error types encountered:")
    all_errors: Counter = Counter()
    for r in results:
        for et in r.get("error_types_seen", []):
            all_errors[et] += 1
    for et, count in all_errors.most_common(15):
        print(f"    {et:<30} {count}")

    print(f"\n  Failure reasons:")
    for reason, count in error_counts.most_common(10):
        print(f"    {reason:<70} {count}")

    # Build output
    gate3_status = "PASS" if n_passed >= 10 else "FAIL"
    baseline = 0.156  # 10/64 baseline

    out = {
        "task": "Gate 3: Error-guided lemma search",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "architecture": (
            "Goal-only encoder retrieval + error-guided redirection. "
            "Lean error messages steer search toward correct lemma "
            "via tactic rotation, graph-neighbor expansion, and arithmetic escalation."
        ),
        "baseline_comparison": {
            "baseline_proof_rate": baseline,
            "this_proof_rate": rate,
            "improvement": rate - baseline,
        },
        "config": {
            "max_initial_lemmas": config.max_initial_lemmas,
            "max_retries": config.max_retries,
            "max_total_checks": config.max_total_checks,
            "retrieval_top_k": config.retrieval_top_k,
            "max_neighbors": config.max_neighbors,
            "check_timeout": config.check_timeout,
            "max_theorem_time": config.max_theorem_time,
            "structural_first": config.structural_first,
        },
        "gate3": {
            "status": gate3_status,
            "total": n_total,
            "passed": n_passed,
            "rate": rate,
            "elapsed_s": elapsed,
            "error_distribution": dict(all_errors.most_common()),
            "failure_reasons": dict(error_counts.most_common()),
        },
        "passed": [
            {
                "name": r["name"],
                "domain": r["domain"],
                "era": r["era"],
                "proof": r["proof_steps"],
                "num_steps": r["num_steps"],
                "num_checks": r["num_checks"],
            }
            for r in passed
        ],
        "results": results,
    }

    save_json(out, output_path)
    print(f"\nResults saved to {output_path}")

    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Gate 3: Error-guided lemma search evaluation"
    )
    parser.add_argument(
        "--encoder", default="checkpoints/gnn/goal_only_encoder.pt",
        help="Goal-only encoder checkpoint",
    )
    parser.add_argument(
        "--pairs", default="data/raw/proof_step_pairs.jsonl",
        help="Proof-step pairs for index building",
    )
    parser.add_argument(
        "--theorems", default="data/raw/gate3_v2.jsonl",
        help="Gate3 v2 theorems",
    )
    parser.add_argument(
        "--vocab-size", type=int, default=3000,
        help="Vocabulary size",
    )
    parser.add_argument(
        "--max-initial", type=int, default=5,
        help="Max initial lemmas to try",
    )
    parser.add_argument(
        "--max-retries", type=int, default=5,
        help="Max error-guided retries per lemma",
    )
    parser.add_argument(
        "--max-checks", type=int, default=10,
        help="Max total Lean checks per theorem",
    )
    parser.add_argument(
        "--output", default="data/error_guided_gate3.json",
        help="Output JSON file",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Smoke test: only run first 3 theorems",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--num-threads", type=int, default=4,
        help="Number of CPU threads",
    )
    parser.add_argument(
        "--max-pairs", type=int, default=0,
        help="Max training pairs for index (0 = all)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("GATE 3: ERROR-GUIDED LEMMA SEARCH EVALUATION")
    print("=" * 70)
    print(f"  Encoder: {args.encoder}")
    print(f"  Max initial lemmas: {args.max_initial}")
    print(f"  Max retries: {args.max_retries}")
    print(f"  Max total checks/theorem: {args.max_checks}")
    print(f"  Smoke: {'YES' if args.smoke else 'NO'}")
    print()

    torch.set_num_threads(args.num_threads)
    print(f"PyTorch threads: {torch.get_num_threads()}")

    # --- Load encoder ---
    encoder_path = _project_root / args.encoder
    if not encoder_path.exists():
        print(f"ERROR: Encoder not found: {encoder_path}")
        return 1

    encoder = GoalOnlyEncoder.load(str(encoder_path))
    encoder.eval()
    print(f"Encoder: {encoder.count_params():,} params, hidden_dim={encoder.hidden_dim}")

    # --- Load dependency graph ---
    graph, lemma_index, lemma_index_to_name = load_dependency_graph()

    # --- Load training pairs & build index ---
    pairs_path = _project_root / args.pairs
    max_pairs = args.max_pairs if args.max_pairs > 0 else None

    print(f"\nLoading training pairs for index...")
    goals, lemmas, _lemma_to_indices = prepare_lemma_groups(pairs_path, max_pairs)
    print(f"  Loaded {len(goals)} pairs")

    print("Building vocabulary...")
    vocab = build_vocabulary(goals, max_vocab=args.vocab_size)
    print(f"  Vocabulary: {len(vocab)} tokens")

    print("Encoding training goals...")
    device = torch.device("cpu")
    encoder = encoder.to(device)

    index_embs_list = []
    batch_size = 256
    with torch.no_grad():
        for i in range(0, len(goals), batch_size):
            batch = goals[i : i + batch_size]
            batch_ids = _tokenize_batch(batch, vocab, 128).to(device)
            embs = encoder(batch_ids)
            index_embs_list.append(embs.cpu())
    index_embeddings = torch.cat(index_embs_list, dim=0)
    print(f"  Index embeddings: {index_embeddings.shape}")

    # --- Load theorems ---
    theorems_path = _project_root / args.theorems
    if not theorems_path.exists():
        print(f"ERROR: Theorems not found: {theorems_path}")
        return 1

    theorems = load_jsonl(theorems_path)
    if args.smoke:
        theorems = theorems[:3]
    print(f"Theorems: {len(theorems)}")

    # --- Proof checker ---
    checker = LeanProofChecker(timeout=15.0, max_retries=3)
    print(f"Proof checker: initialized")

    # --- Run ---
    output_path = _project_root / args.output
    result = run_error_guided_gate3(
        encoder=encoder,
        vocab=vocab,
        index_goals=goals,
        index_lemmas=lemmas,
        index_embeddings=index_embeddings,
        theorems=theorems,
        checker=checker,
        graph=graph,
        lemma_index=lemma_index,
        lemma_index_to_name=lemma_index_to_name,
        output_path=output_path,
        verbose=args.verbose,
    )

    n_passed = result["gate3"]["passed"]
    print(f"\n{'=' * 70}")
    print("FINAL")
    print(f"{'=' * 70}")
    print(f"  Proofs found: {n_passed}/{len(theorems)} ({result['gate3']['rate']:.0%})")
    print(f"  Baseline:     15.6%")
    print(f"  Output:       {output_path}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
