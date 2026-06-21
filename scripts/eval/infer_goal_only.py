#!/usr/bin/env python3
"""
Goal-Only Inference: KNN retrieval + frequency × similarity ranking.

Given a theorem/goal:
  1. Encode goal with GoalOnlyEncoder
  2. Find top-K (50) most similar training goals via cosine similarity
  3. Collect their lemmas, weighted by frequency × similarity
  4. Rank lemmas, return top-N (30)

Integrates with best-first search for Gate 3 evaluation.

Usage:
  # Standalone retrieval test
  python scripts/eval/infer_goal_only.py \
    --model checkpoints/gnn/goal_only/goal_only_encoder.pt \
    --pairs data/raw/proof_step_pairs.jsonl \
    --goal "a + b = b + a"

  # Gate 3 eval (64 theorems)
  python scripts/eval/infer_goal_only.py \
    --model checkpoints/gnn/goal_only/goal_only_encoder.pt \
    --gate3 data/raw/gate3_v2.jsonl \
    --pairs data/raw/proof_step_pairs.jsonl \
    --eval-gate3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict, Counter
from pathlib import Path

import torch
import torch.nn.functional as F

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))

from src.explorer.goal_only_encoder import GoalOnlyEncoder, build_vocab, tokenize_goal


# ---------------------------------------------------------------------------
# Retrieval index
# ---------------------------------------------------------------------------


class GoalRetrievalIndex:
    """Index for fast goal → lemma retrieval via cosine similarity.

    Pre-encodes all training goals and maintains goal→lemma mappings.
    """

    def __init__(
        self,
        model: GoalOnlyEncoder,
        pairs: list[dict],
        device: torch.device,
        chunk_size: int = 2000,
    ):
        self.model = model
        self.device = device

        # Build mappings
        print("Building goal→lemma mapping...")
        self.goal_to_lemmas: dict[str, list[str]] = defaultdict(list)
        self.goal_to_idx: dict[str, int] = {}
        goals_seen: list[str] = []
        for pair in pairs:
            g = pair["goal"]
            if g not in self.goal_to_idx:
                self.goal_to_idx[g] = len(goals_seen)
                goals_seen.append(g)
            self.goal_to_lemmas[g].append(pair["lemma"])
        self.goals = goals_seen
        print(f"  Unique goals: {len(self.goals)}")

        # Pre-encode all goals
        print(f"Pre-encoding {len(self.goals)} goals (chunk_size={chunk_size})...")
        model.eval()
        all_embs = []
        with torch.no_grad():
            for start in range(0, len(self.goals), chunk_size):
                chunk = self.goals[start:start + chunk_size]
                embs = model(chunk, device)
                all_embs.append(embs)
                if (start // chunk_size) % 10 == 0:
                    print(f"  {start}/{len(self.goals)}...")
        self.embeddings = F.normalize(torch.cat(all_embs, dim=0), dim=-1)
        print(f"  Index shape: {self.embeddings.shape}")

        # Pre-compute lemma→goals mapping for score aggregation
        self.lemma_to_goals: dict[str, list[int]] = defaultdict(list)
        for g, lemmas in self.goal_to_lemmas.items():
            gidx = self.goal_to_idx.get(g)
            if gidx is not None:
                for lemma in lemmas:
                    self.lemma_to_goals[lemma].append(gidx)
        print(f"  Unique lemmas in index: {len(self.lemma_to_goals)}")

    def query(
        self,
        goal_text: str,
        top_k: int = 50,
        return_top_n: int = 30,
    ) -> list[tuple[str, float]]:
        """Query the index for lemma suggestions.

        Args:
            goal_text: The proof goal (theorem statement conclusion).
            top_k: Number of nearest training goals to retrieve.
            return_top_n: Number of lemmas to return.

        Returns:
            List of (lemma_name, score) sorted by descending score,
            where score = sum(cosine_similarity) over goals that use this lemma.
        """
        self.model.eval()
        with torch.no_grad():
            goal_emb = self.model.encode_single(goal_text, self.device)
            goal_emb = F.normalize(goal_emb, dim=-1)

            # Cosine similarity to all index goals
            sims = (goal_emb @ self.embeddings.T)  # [N]

            # Top-K goals
            topk_sims, topk_indices = torch.topk(sims, min(top_k, len(self.goals)))

            # Aggregate lemma scores: frequency × similarity
            lemma_scores: dict[str, float] = defaultdict(float)
            for i in range(len(topk_indices)):
                idx = topk_indices[i].item()
                sim = max(0.0, topk_sims[i].item())
                goal = self.goals[idx]
                for lemma in self.goal_to_lemmas.get(goal, []):
                    lemma_scores[lemma] += sim

            # Sort by descending score
            sorted_lemmas = sorted(lemma_scores.items(), key=lambda x: -x[1])
            return sorted_lemmas[:return_top_n]

    def query_mrr(
        self,
        goal_text: str,
        correct_lemma: str,
        top_k: int = 50,
    ) -> float:
        """Compute MRR: 1/rank of correct lemma among retrieved results."""
        results = self.query(goal_text, top_k=top_k, return_top_n=top_k)
        for rank, (lemma, score) in enumerate(results, 1):
            if lemma == correct_lemma:
                return 1.0 / rank
        return 0.0


# ---------------------------------------------------------------------------
# Gate 3 evaluation
# ---------------------------------------------------------------------------


def evaluate_gate3(
    model: GoalOnlyEncoder,
    index: GoalRetrievalIndex,
    theorems: list[dict],
    top_k: int = 50,
    top_n: int = 30,
    use_lean: bool = True,
) -> dict:
    """Run Gate 3 evaluation: retrieve lemmas → try as proofs → Lean check.

    Simple strategy: for each theorem, try top-N lemmas as single-step
    `exact lemma` or `apply lemma` proofs. Multi-step via simple sequential
    application of top lemmas.

    Returns:
        Evaluation results dict.
    """
    print(f"\n{'='*70}")
    print(f"GATE 3 EVAL (Goal-Only): {len(theorems)} theorems")
    print(f"  Top-K retrieval: {top_k}, Top-N lemmas: {top_n}")
    print(f"  Lean checking: {use_lean}")
    print(f"{'='*70}\n")

    # Setup Lean checker
    checker = None
    if use_lean:
        try:
            from src.proof_checker.batch_checker import BatchChecker
            checker = BatchChecker()
            print("Lean 4 proof checker initialized.")
        except Exception as e:
            print(f"WARNING: Could not init Lean checker: {e}")
            print("Falling back to structural check only.")
            use_lean = False

    results = []
    passed = 0
    t_start = time.time()

    for i, theorem in enumerate(theorems):
        name = theorem["name"]
        statement = theorem.get("statement", "")
        domain = theorem.get("domain", "unknown")
        ground_truth = theorem.get("proof", "?")

        # Extract goal from statement
        goal_text = extract_goal_from_statement(statement)

        t0 = time.time()

        # Retrieve lemmas
        lemma_candidates = index.query(goal_text, top_k=top_k, return_top_n=top_n)

        # Try lemmas as proofs
        proof_steps, success, error = try_lemmas_as_proof(
            lemma_candidates, statement, checker, use_lean
        )
        elapsed = time.time() - t0

        result = {
            "name": name,
            "domain": domain,
            "success": success,
            "error": error,
            "proof_steps": proof_steps,
            "ground_truth": ground_truth,
            "top_lemmas": [(l, round(s, 4)) for l, s in lemma_candidates[:10]],
            "num_candidates": len(lemma_candidates),
            "search_time_s": round(elapsed, 2),
        }
        results.append(result)
        if success:
            passed += 1

        status = "\u2713" if success else "\u2717"
        eta = (time.time() - t_start) / (i + 1) * (len(theorems) - i - 1)
        print(f"  [{i+1:2d}/{len(theorems)}] {status} {name:<30s} "
              f"domain={domain:<12s} time={elapsed:.1f}s  "
              f"ETA={eta:.0f}s  passed={passed}")

    elapsed_total = time.time() - t_start
    print(f"\n--- Gate 3 Results ---")
    print(f"  Passed: {passed}/{len(theorems)} ({100*passed/len(theorems):.1f}%)")
    print(f"  Total time: {elapsed_total:.0f}s")

    if checker:
        try:
            checker.shutdown()
        except Exception:
            pass

    return {
        "passed": passed,
        "total": len(theorems),
        "rate": passed / len(theorems) if theorems else 0.0,
        "results": results,
        "elapsed_s": elapsed_total,
    }


def extract_goal_from_statement(statement: str) -> str:
    """Extract the goal proposition from a theorem statement.

    'theorem alg_add_comm (a b : ℝ) : a + b = b + a' → 'a + b = b + a'
    """
    s = statement.strip()
    # Remove leading keyword
    for kw in ["theorem ", "lemma ", "example ", "def "]:
        if s.startswith(kw):
            s = s[len(kw):]
            break
    # Find the type colon (after name and arguments)
    # Simple heuristic: split on ' : ' and take the last part
    parts = s.split(" : ")
    if len(parts) >= 2:
        # Find the first colon that separates binder from type
        # Look for pattern: name (args) : type := proof
        depth = 0
        for i, c in enumerate(s):
            if c in "({[":
                depth += 1
            elif c in ")}]":
                depth -= 1
            elif c == ":" and depth == 0:
                goal = s[i+1:].strip()
                if ":=" in goal:
                    goal = goal.split(":=")[0].strip()
                return goal
        # Fallback: last colon
        goal = parts[-1].strip()
        if ":=" in goal:
            goal = goal.split(":=")[0].strip()
        return goal
    return s


def try_lemmas_as_proof(
    lemma_candidates: list[tuple[str, float]],
    statement: str,
    checker,
    use_lean: bool,
    max_attempts: int = 30,
) -> tuple[list[str], bool, str]:
    """Try lemma candidates as proof steps.

    Strategy: for each lemma, try `exact lemma_name` and `apply lemma_name`.
    First success wins. Falls back to multi-step attempts.

    Returns: (proof_steps, success, error_message)
    """
    proof_attempts = []

    # Single-step attempts
    for lemma_name, score in lemma_candidates[:max_attempts]:
        for tactic in ["exact", "apply"]:
            proof = f"{tactic} {lemma_name}"
            proof_attempts.append(proof)
            if use_lean and checker:
                try:
                    from src.proof_checker.formats import wrap_theorem_with_proof
                    code = wrap_theorem_with_proof(statement, proof)
                    results = checker.check_batch([code])
                    if results[0].success:
                        return [proof], True, ""
                except Exception:
                    pass

    # If no single-step works, try combinations (simple: apply then exact)
    if len(lemma_candidates) >= 2:
        for i, (lem1, s1) in enumerate(lemma_candidates[:5]):
            for j, (lem2, s2) in enumerate(lemma_candidates[:5]):
                if i == j:
                    continue
                for combo in [
                    f"apply {lem1}\nexact {lem2}",
                    f"apply {lem1}\napply {lem2}",
                ]:
                    if use_lean and checker:
                        try:
                            from src.proof_checker.formats import wrap_theorem_with_proof
                            code = wrap_theorem_with_proof(statement, combo)
                            results = checker.check_batch([code])
                            if results[0].success:
                                return combo.split("\n"), True, ""
                        except Exception:
                            pass

    return proof_attempts[:3], False, "No proof found among candidates"


# ---------------------------------------------------------------------------
# Standalone retrieval query
# ---------------------------------------------------------------------------


def query_single(args):
    """Query the index for a single goal and print results."""
    device = torch.device("cpu")

    # Load model
    print(f"Loading model from {args.model}...")
    model = GoalOnlyEncoder.load(args.model)
    model = model.to(device)
    model.eval()
    print(f"Model params: {model.count_parameters():,}")

    # Load pairs for index
    print(f"Loading pairs from {args.pairs}...")
    pairs = []
    with open(args.pairs) as f:
        for line in f:
            pairs.append(json.loads(line))
            if len(pairs) >= (args.max_pairs or 50000):
                break
    print(f"Loaded {len(pairs)} pairs")

    # Build index
    index = GoalRetrievalIndex(model, pairs, device, chunk_size=2000)

    # Query
    results = index.query(args.goal, top_k=args.top_k, return_top_n=args.top_n)

    print(f"\n--- Results for goal: {args.goal[:100]}... ---")
    for rank, (lemma, score) in enumerate(results, 1):
        print(f"  {rank:3d}. {lemma:<40s} score={score:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Goal-Only GNN Inference: KNN retrieval + lemma ranking"
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Path to GoalOnlyEncoder checkpoint")
    parser.add_argument("--pairs", type=str,
                        default="data/raw/proof_step_pairs.jsonl",
                        help="Path to proof_step_pairs.jsonl")
    parser.add_argument("--max-pairs", type=int, default=0,
                        help="Max pairs for index (0=all)")

    # Query mode
    parser.add_argument("--goal", type=str, default="",
                        help="Goal text for standalone query")
    parser.add_argument("--top-k", type=int, default=50,
                        help="Number of nearest goals to retrieve")
    parser.add_argument("--top-n", type=int, default=30,
                        help="Number of lemmas to return")

    # Gate 3 eval mode
    parser.add_argument("--eval-gate3", action="store_true",
                        help="Run Gate 3 (64 theorem) evaluation")
    parser.add_argument("--gate3", type=str,
                        default="data/raw/gate3_v2.jsonl",
                        help="Path to gate3_v2.jsonl")
    parser.add_argument("--no-lean", action="store_true",
                        help="Skip Lean proof checking")

    # Output
    parser.add_argument("--output", type=str, default="",
                        help="Output path for Gate 3 results JSON")

    args = parser.parse_args()

    # Resolve paths
    def resolve(p: str) -> Path:
        path = Path(p)
        if not path.is_absolute():
            path = _project_root / path
        return path

    args.model = str(resolve(args.model))
    args.pairs = str(resolve(args.pairs))
    if args.gate3:
        args.gate3 = str(resolve(args.gate3))

    # Standalone query
    if args.goal:
        query_single(args)
        return 0

    # Gate 3 eval
    if args.eval_gate3:
        device = torch.device("cpu")

        # Load model
        print(f"Loading model from {args.model}...")
        model = GoalOnlyEncoder.load(args.model)
        model = model.to(device)
        model.eval()
        print(f"Model params: {model.count_parameters():,}")

        # Load pairs
        print(f"Loading pairs from {args.pairs}...")
        pairs = []
        with open(args.pairs) as f:
            for line in f:
                pairs.append(json.loads(line))
                if args.max_pairs and len(pairs) >= args.max_pairs:
                    break
        print(f"Loaded {len(pairs)} pairs")

        # Build index
        index = GoalRetrievalIndex(model, pairs, device, chunk_size=2000)

        # Load Gate 3 theorems
        print(f"Loading Gate 3 theorems from {args.gate3}...")
        with open(args.gate3) as f:
            theorems = [json.loads(line) for line in f]
        print(f"Loaded {len(theorems)} theorems")

        # Run eval
        results = evaluate_gate3(
            model, index, theorems,
            top_k=args.top_k, top_n=args.top_n,
            use_lean=not args.no_lean,
        )

        # Save output
        output_path = args.output or "data/goal_only_gate3.json"
        output_path = str(resolve(output_path))
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {output_path}")

        # Summary
        rate = results["rate"]
        baseline = 0.156
        print(f"\nGoal-Only pass rate: {rate:.1%}")
        print(f"Baseline (GNN+best-first): {baseline:.1%}")
        print(f"Delta: {rate - baseline:+.1%}")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
