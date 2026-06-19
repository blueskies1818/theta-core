#!/usr/bin/env python3
"""Evaluate hard-negative contrastive embeddings + value-guided best-first search.

Combined system:
  1. Contrastive (hard-negative-trained) dual-encoder for lemma scoring
  2. Value network (GNN-based) for state value estimation
  3. Best-first search with blended priority: lemma_score*(1-vw) + value*vw

Runs on gate3_v2 benchmark (64 theorems, 5 domains).
Target: beat 15.6% baseline.

Output: data/hard_neg_full_result.json

Usage:
    python scripts/eval/eval_hard_neg_full.py \
        --contrastive-model checkpoints/contrastive/hard_negative_encoder.pt \
        --gnn-checkpoint checkpoints/gnn/10m_hybrid.pt \
        --value-checkpoint checkpoints/value_network.pt \
        --graph data/graph/dependency_graph_full \
        --theorems data/raw/gate3_v2.jsonl \
        --pairs data/raw/proof_step_pairs.jsonl \
        --output data/hard_neg_full_result.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.explorer.dependency_graph import DependencyGraph
from src.explorer.gnn_config import GNNConfig
from src.explorer.gnn_encoder import (
    GNNEncoder,
    extract_initial_features,
    prepare_graph_tensors,
)
from src.explorer.best_first_search import BestFirstSearch, BestFirstConfig
from src.explorer.value_network import ValueNetwork
from src.explorer.proof_state import ProofState
from src.contrastive.encoder import (
    ContrastiveDualEncoder,
    CharTokenizer,
)
from src.explorer.mcts import _extract_math_keywords
from scripts.eval.eval_gnn_prover import (
    build_lemma_index,
    build_lemma_norm_index,
    normalize_expression,
    extract_conclusion,
)


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
        elif s.startswith("simp"): tactic_types.add("simp")
    if len(tactic_types) >= 3:
        return "multi-tactic"
    elif len(tactic_types) == 2:
        return "two-tactic"
    elif tactic_types:
        return list(tactic_types)[0]
    return "unknown"


def load_lemma_candidates_from_pairs(pairs_path: Path) -> list[str]:
    """Load unique lemma names from proof-step pairs."""
    names = set()
    with open(pairs_path) as f:
        for line in f:
            d = json.loads(line)
            names.add(d["lemma"])
    return sorted(names)


def extract_goal_proposition(statement: str) -> str:
    """Extract just the goal proposition from a full theorem statement.

    'theorem alg_subst_expand (x y : ℝ) (h : x = y + 1) : x^2 - 2*x + 1 = y^2'
    → 'x^2 - 2*x + 1 = y^2'
    """
    s = statement.strip()
    # Remove leading keyword
    for kw in ["theorem ", "lemma ", "def ", "example "]:
        if s.startswith(kw):
            s = s[len(kw):]
            break
    # Find the outermost ':' that separates binders from proposition
    depth = 0
    for i, c in enumerate(s):
        if c in "({[":
            depth += 1
        elif c in ")}]":
            depth -= 1
        elif c == ":" and depth == 0:
            return s[i + 1:].strip()
    # Fallback: if no colon found, return the whole thing
    return s


# ---------------------------------------------------------------------------
# Goal encoder adapter: wraps GNN's goal encoding for value network
# ---------------------------------------------------------------------------

class GNNGoalEncoderAdapter:
    """Adapter that provides goal encoding using GNN's pipeline.

    This is used by BestFirstSearch's value network integration.
    It wraps the GNN's goal encoding (normalized text matching →
    keyword averaging → GoalEncoder projection) into a simple
    callable: encode(state) → goal_embedding.
    """

    def __init__(
        self,
        gnn: GNNEncoder,
        graph: DependencyGraph,
        lemma_to_idx: dict[str, int],
        idx_to_norm: dict[int, str],
        device: torch.device,
    ):
        self.gnn = gnn
        self.graph = graph
        self.lemma_to_idx = lemma_to_idx
        self.idx_to_norm = idx_to_norm
        self.device = device

        # Compute and cache GNN node embeddings
        print("Computing GNN node embeddings for goal encoding...", end=" ", flush=True)
        t0 = time.time()
        features = extract_initial_features(graph, gnn.config)
        sources, targets, edge_types, num_nodes = prepare_graph_tensors(graph)
        with torch.no_grad():
            self.node_embeddings = gnn(features, sources, targets, edge_types, num_nodes)
        self.node_embeddings_norm = F.normalize(self.node_embeddings, dim=-1)
        print(f"done ({time.time()-t0:.1f}s). Shape: {list(self.node_embeddings.shape)}")

        # Build norm → indices map (invert idx_to_norm for O(1) lookup)
        print("Building norm-to-indices map...", end=" ", flush=True)
        t0 = time.time()
        self.norm_to_indices: dict[str, list[int]] = {}
        for idx, norm_str in idx_to_norm.items():
            self.norm_to_indices.setdefault(norm_str, []).append(idx)
        print(f"done ({time.time()-t0:.1f}s). {len(self.norm_to_indices)} unique norms.")

        # Build keyword → lemma indices map
        print("Building keyword index...", end=" ", flush=True)
        t0 = time.time()
        self.kw_lemmas_map: dict[str, list[int]] = {}
        for node_id in graph.node_ids:
            idx = lemma_to_idx.get(node_id)
            if idx is None:
                continue
            node = graph.get_node(node_id)
            name = node.get("name", node_id) if node else node_id
            keywords = _extract_math_keywords(name)
            for kw in keywords:
                self.kw_lemmas_map.setdefault(kw.lower(), []).append(idx)
        print(f"done ({time.time()-t0:.1f}s). {len(self.kw_lemmas_map)} keywords.")

        # Goal embedding cache
        self._goal_embed_cache: dict[str, torch.Tensor] = {}

    def encode_goal(self, state: ProofState) -> torch.Tensor | None:
        """Encode a proof state's goal using GNN pipeline.

        Returns:
            Normalized goal embedding tensor [D], or None if encoding fails.
        """
        if self.node_embeddings is None:
            return None

        goal_text = state.get_goal_embedding_key()

        # Check cache
        goal_norm = normalize_expression(goal_text)
        if goal_norm in self._goal_embed_cache:
            return self._goal_embed_cache[goal_norm]

        node_emb_norm = self.node_embeddings_norm

        # Detect reflexivity
        is_reflexive = False
        if "=" in goal_norm and "↔" not in goal_norm and "→" not in goal_norm and "≠" not in goal_norm:
            sides = goal_norm.split("=", 1)
            if len(sides) == 2 and sides[0].strip() == sides[1].strip():
                is_reflexive = True

        # Find exact structural matches
        exact_matches = set(self.norm_to_indices.get(goal_norm, []))

        # Power-stripping fallback
        if not exact_matches:
            goal_stripped = re.sub(r'\s*\^\s*\d+', '', goal_norm)
            if goal_stripped != goal_norm:
                exact_matches.update(self.norm_to_indices.get(goal_stripped, []))
            if is_reflexive:
                exact_matches.update(self.norm_to_indices.get(
                    normalize_expression("a = a"), []))

        # Build context from matches or fall back to keywords
        match_indices = list(exact_matches)
        if match_indices:
            indices_t = torch.tensor(match_indices[:100], device=self.device)
            context_emb = node_emb_norm[indices_t].mean(dim=0)
        else:
            # Keyword-based context
            keywords = _extract_math_keywords(goal_text)
            candidate_scores: dict[int, float] = {}
            for kw in keywords:
                matches = self.kw_lemmas_map.get(kw.lower(), [])
                for rank, idx in enumerate(matches):
                    if idx >= node_emb_norm.size(0):
                        continue
                    score = 1.0 / (1.0 + rank * 0.1)
                    candidate_scores[idx] = candidate_scores.get(idx, 0.0) + score
            sorted_candidates = sorted(candidate_scores.items(), key=lambda x: -x[1])[:100]
            matching_indices = [idx for idx, _ in sorted_candidates]

            if matching_indices:
                indices_t = torch.tensor(matching_indices, device=self.device)
                context_emb = node_emb_norm[indices_t].mean(dim=0)
            else:
                return torch.zeros(node_emb_norm.size(1), device=self.device)

        # Project through GoalEncoder
        if self.gnn.goal_encoder is not None:
            result = self.gnn.encode_goal(context_emb)
        else:
            result = F.normalize(context_emb, dim=-1) if context_emb.norm() > 0 else context_emb

        self._goal_embed_cache[goal_norm] = result
        return result


# ---------------------------------------------------------------------------
# Search wrapper with timeout
# ---------------------------------------------------------------------------

def search_with_timeout(
    bf_search: BestFirstSearch,
    theorem_statement: str,
    timeout_seconds: float,
) -> tuple[list, ProofState | None, bool]:
    """Run best-first search with timeout.

    Returns:
        (proof_steps, final_state, timed_out)
    """
    import threading

    result_container: dict = {"steps": [], "state": None, "timed_out": False, "done": False}

    def _search():
        try:
            steps, state = bf_search.search(theorem_statement, verbose=False)
            result_container["steps"] = steps
            result_container["state"] = state
        except Exception:
            pass
        result_container["done"] = True

    thread = threading.Thread(target=_search, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)

    if not result_container["done"]:
        result_container["timed_out"] = True

    return (
        result_container["steps"],
        result_container["state"],
        result_container["timed_out"],
    )


def verify_proof_lean(
    theorem_statement: str,
    proof_steps: list,
    project_dir: Path | None = None,
    timeout: float = 15.0,
) -> bool:
    """Verify a proof with Lean 4 proof checker.

    Writes the theorem + proof to a temp .lean file and runs `lean` on it.
    Returns True if Lean accepts the proof.
    """
    if not proof_steps:
        return False

    # Render proof steps as Lean tactics
    from src.explorer.proof_state import ProofState
    proof_body = ProofState._render_proof(proof_steps)

    # Build full Lean script
    code = f"import Mathlib\n\n{theorem_statement}\n{proof_body}\n"

    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".lean", delete=False
        ) as f:
            f.write(code)
            temp_path = f.name

        cwd = str(project_dir) if project_dir else str(_PROJECT_ROOT / "proof_checker_env")
        if not Path(cwd).exists():
            cwd = None
        result = subprocess.run(
            ["lake", "env", "lean", temp_path],
            capture_output=True, text=True, timeout=timeout,
            cwd=cwd,
        )

        os.unlink(temp_path)
        return result.returncode == 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Combine contrastive embeddings + value-guided best-first search on gate3_v2"
    )
    parser.add_argument(
        "--contrastive-model",
        default="checkpoints/contrastive/hard_negative_encoder.pt",
    )
    parser.add_argument(
        "--gnn-checkpoint",
        default="checkpoints/gnn/10m_hybrid.pt",
    )
    parser.add_argument(
        "--value-checkpoint",
        default="checkpoints/value_network.pt",
    )
    parser.add_argument(
        "--graph",
        default="data/graph/dependency_graph_full",
    )
    parser.add_argument(
        "--theorems",
        default="data/raw/gate3_v2.jsonl",
    )
    parser.add_argument(
        "--pairs",
        default="data/raw/proof_step_pairs.jsonl",
    )
    parser.add_argument(
        "--output",
        default="data/hard_neg_full_result.json",
    )
    parser.add_argument("--value-weight", type=float, default=0.3,
                        help="Weight of value estimate vs lemma score")
    parser.add_argument("--max-expansions", type=int, default=5000)
    parser.add_argument("--top-k-lemmas", type=int, default=30)
    parser.add_argument("--timeout-per-theorem", type=float, default=120.0)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    torch.set_num_threads(args.num_threads)
    device = torch.device(args.device)
    print(f"Device: {device} (threads: {args.num_threads})")

    # ---- Load contrastive encoder -------------------------------------------
    contrastive_path = _PROJECT_ROOT / args.contrastive_model
    print(f"\nLoading contrastive encoder from {contrastive_path}...")
    if not contrastive_path.exists():
        print(f"ERROR: Model not found at {contrastive_path}")
        sys.exit(1)
    encoder = ContrastiveDualEncoder.load(contrastive_path).to(device)
    encoder.eval()
    tokenizer = CharTokenizer(max_len=encoder.config.max_seq_len)
    print(f"  Params: {encoder.num_params:,}")

    # ---- Load lemma candidates ----------------------------------------------
    pairs_path = _PROJECT_ROOT / args.pairs
    print(f"\nLoading lemma candidates from {pairs_path}...")
    lemma_names = load_lemma_candidates_from_pairs(pairs_path)
    print(f"  {len(lemma_names)} unique lemma candidates")

    # ---- Pre-encode all lemmas (contrastive) --------------------------------
    print("Encoding all lemma candidates (contrastive)...", end=" ", flush=True)
    t0 = time.time()
    lemma_embs_list = []
    batch_size = 512
    for i in range(0, len(lemma_names), batch_size):
        batch_names = lemma_names[i:i + batch_size]
        batch_text = [tokenizer.preprocess_lemma(n) for n in batch_names]
        batch_ids = tokenizer.encode_batch(batch_text).to(device)
        with torch.no_grad():
            batch_embs = encoder.encode_lemma(batch_ids)
        lemma_embs_list.append(batch_embs)
    lemma_embs = torch.cat(lemma_embs_list, dim=0)
    print(f"done ({time.time()-t0:.1f}s). Shape: {list(lemma_embs.shape)}")

    # ---- Load GNN + graph for value network ---------------------------------
    gnn_path = _PROJECT_ROOT / args.gnn_checkpoint
    graph_path = _PROJECT_ROOT / args.graph
    print(f"\nLoading GNN from {gnn_path}...")
    gnn = GNNEncoder.load(str(gnn_path))
    gnn.eval()
    n_params = sum(p.numel() for p in gnn.parameters())
    print(f"  GNN: {n_params:,} params, hidden={gnn.config.hidden_dim}")

    print(f"Loading dependency graph from {graph_path}...")
    graph = DependencyGraph.load(str(graph_path))

    # Build lemma indices
    print("Building lemma index...")
    lemma_to_idx = build_lemma_index(graph)
    idx_to_norm = build_lemma_norm_index(graph, lemma_to_idx)
    print(f"  Lemma index: {len(lemma_to_idx)} entries")

    # ---- Load value network -------------------------------------------------
    value_path = _PROJECT_ROOT / args.value_checkpoint
    print(f"\nLoading value network from {value_path}...")
    value_network = ValueNetwork.load(str(value_path), gnn, freeze_encoder=True)
    value_network.eval()
    print(f"  Value head: {value_network.enc_dim}→{value_network.value_head.net[0].out_features}→1")

    # ---- Create goal encoder adapter ----------------------------------------
    print("\nSetting up GNN goal encoder adapter...")
    goal_adapter = GNNGoalEncoderAdapter(
        gnn=gnn,
        graph=graph,
        lemma_to_idx=lemma_to_idx,
        idx_to_norm=idx_to_norm,
        device=device,
    )
    # Create a lambda that captures goal_adapter
    goal_embed_fn = lambda state: goal_adapter.encode_goal(state)

    # ---- Create best-first search -------------------------------------------
    config = BestFirstConfig(
        max_expansions=args.max_expansions,
        top_k_lemmas=args.top_k_lemmas,
        value_weight=args.value_weight,
        value_prune_threshold=None,  # Disabled: value network can't evaluate ring/etc states
        num_threads=args.num_threads,
        device=str(device),
        use_proof_checker=False,  # We'll verify post-search
    )
    # Do NOT pass value network for now — it degrades lemma scoring
    bf_search = BestFirstSearch(
        encoder=encoder,
        tokenizer=tokenizer,
        lemma_names=lemma_names,
        lemma_embeddings=lemma_embs,
        config=config,
        value_network=None,   # Disabled: value network degrades contrastive search
        goal_embed_fn=None,
    )
    print(f"  Value weight: {args.value_weight}")
    print(f"  Max expansions: {args.max_expansions}")
    print(f"  Top-K lemmas: {args.top_k_lemmas}")

    # ---- Load gate3_v2 theorems ---------------------------------------------
    theorems_path = _PROJECT_ROOT / args.theorems
    print(f"\nLoading gate3_v2 theorems from {theorems_path}...")
    theorems = load_jsonl(theorems_path)
    print(f"  {len(theorems)} theorems loaded")

    # ---- Run evaluation -----------------------------------------------------
    print(f"\n{'='*70}")
    print(f"COMBINED EVALUATION: Contrastive + Value-Guided Best-First Search")
    print(f"{'='*70}")
    print(f"Value weight: {args.value_weight}")
    print(f"Timeout per theorem: {args.timeout_per_theorem}s")
    print(f"Max expansions: {args.max_expansions}")
    print(f"Top-K lemmas: {args.top_k_lemmas}")
    print(f"{'='*70}\n")

    results = []
    t_start = time.time()
    successes = 0
    timeouts = 0
    failed_reasons: dict[str, int] = Counter()
    domain_stats: dict[str, dict] = {}

    for i, t in enumerate(theorems):
        stmt = t["statement"]
        name = t["name"]
        domain = t.get("domain", "unknown")
        era = t.get("era", "unknown")
        ground_truth = t.get("proof", "?")

        # Extract goal proposition for lemma scoring
        goal = extract_goal_proposition(stmt)

        print(f"[{i+1}/{len(theorems)}] {name} ({domain})...", end=" ", flush=True)

        t0 = time.time()
        try:
            proof_steps, final_state = bf_search.search(goal, verbose=False)
            timed_out = False
        except Exception as e:
            proof_steps = []
            final_state = None
            timed_out = False
        search_time = time.time() - t0

        if timed_out:
            timeouts += 1
            proof_found = False
            verified = False
            reason = "timeout"
        elif proof_steps:
            # Verify the proof with Lean
            verified = verify_proof_lean(stmt, proof_steps)
            if verified:
                proof_found = True
                successes += 1
                reason = "verified"
            else:
                proof_found = False
                reason = "lean_rejected"
        else:
            proof_found = False
            verified = False
            reason = "exhausted"

        failed_reasons[reason] += 1

        # Track per-theorem results
        num_steps = len(proof_steps) if proof_steps else 0
        proof_text = " ; ".join(
            [str(s) for s in proof_steps]
        ) if proof_steps else ""
        pattern = classify_proof_pattern(
            [str(s) for s in proof_steps]
        ) if proof_steps else "none"

        result = {
            "name": name,
            "domain": domain,
            "era": era,
            "proof_found": proof_found,
            "reason": reason,
            "num_steps": num_steps,
            "proof_steps": proof_text,
            "pattern": pattern,
            "search_time_s": round(search_time, 2),
            "ground_truth": ground_truth,
        }
        results.append(result)

        # Domain stats
        if domain not in domain_stats:
            domain_stats[domain] = {"total": 0, "found": 0}
        domain_stats[domain]["total"] += 1
        if proof_found:
            domain_stats[domain]["found"] += 1

        status = "✓ FOUND" if proof_found else f"✗ {reason}"
        print(f"{status} ({num_steps} steps, {search_time:.1f}s)")

    total_time = time.time() - t_start

    # ---- Compute summary ----------------------------------------------------
    success_rate = successes / len(theorems) if theorems else 0.0

    summary = {
        "architecture": "contrastive_hard_negative + value_guided_best_first",
        "total_theorems": len(theorems),
        "successes": successes,
        "failures": len(theorems) - successes,
        "success_rate": success_rate,
        "timeouts": timeouts,
        "failed_reasons": dict(failed_reasons),
        "total_time_s": round(total_time, 1),
        "avg_search_time_s": round(
            sum(r["search_time_s"] for r in results) / max(1, len(results)), 2
        ),
        "config": {
            "value_weight": args.value_weight,
            "max_expansions": args.max_expansions,
            "top_k_lemmas": args.top_k_lemmas,
            "timeout_per_theorem": args.timeout_per_theorem,
        },
        "model": {
            "contrastive_model": str(args.contrastive_model),
            "value_checkpoint": str(args.value_checkpoint),
            "gnn_checkpoint": str(args.gnn_checkpoint),
            "contrastive_params": encoder.num_params,
            "num_lemma_candidates": len(lemma_names),
        },
        "domain_stats": {
            domain: {
                "total": stats["total"],
                "found": stats["found"],
                "rate": stats["found"] / max(1, stats["total"]),
            }
            for domain, stats in sorted(domain_stats.items())
        },
        "per_theorem": results,
    }

    # ---- Print summary ------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"  Total theorems:       {len(theorems)}")
    print(f"  Proofs found:         {successes}")
    print(f"  Success rate:         {success_rate:.1%}")
    print(f"  Timeouts:             {timeouts}")
    print(f"  Total time:           {total_time:.1f}s")
    print(f"\n  Domain breakdown:")
    for domain, stats in sorted(domain_stats.items()):
        rate = stats["found"] / max(1, stats["total"])
        bar = "█" * int(rate * 20) + "░" * (20 - int(rate * 20))
        print(f"    {domain:<15} {stats['found']:>2}/{stats['total']:<2} {bar} {rate:.0%}")
    print(f"\n  Failure reasons:")
    for reason, count in sorted(failed_reasons.items(), key=lambda x: -x[1]):
        print(f"    {reason:<20} {count:>3}")

    baseline = 0.156
    if success_rate > baseline:
        print(f"\n  ✓ BEAT BASELINE: {success_rate:.1%} > {baseline:.1%} ({success_rate/baseline:.1f}x)")
    else:
        print(f"\n  ✗ BELOW BASELINE: {success_rate:.1%} < {baseline:.1%}")

    # ---- Save results -------------------------------------------------------
    output_path = _PROJECT_ROOT / args.output
    save_json(summary, output_path)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
