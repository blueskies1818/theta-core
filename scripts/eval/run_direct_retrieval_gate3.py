#!/usr/bin/env python3
"""Gate 3 evaluation: Direct TF-IDF retrieval + best-first search + Lean checker.

Architecture:
  Goal → TF-IDF → k-NN over 226K training goals → collect lemmas
  → rank by frequency × similarity → top-30 to best-first search
  → Lean proof checker

Bypasses the GNN entirely. No graph, no contrastive loss.
Output: data/direct_retrieval_gate3.json

Usage:
    python scripts/eval/run_direct_retrieval_gate3.py [--max-expansions N] [--top-k N]
"""

from __future__ import annotations

import heapq
import json
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.retrieval.direct_lookup import DirectRetrieval
from src.explorer.proof_state import ProofState, Tactic, TacticType
from src.proof_checker.batch_checker import BatchChecker
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


def classify_proof_pattern(proof_steps: list[str]) -> str:
    if not proof_steps:
        return "empty"
    tactic_types = set()
    for step in proof_steps:
        s = step.strip().lower()
        if s.startswith("rw"):
            tactic_types.add("rw")
        elif s.startswith("exact"):
            tactic_types.add("exact")
        elif s.startswith("apply"):
            tactic_types.add("apply")
        elif s.startswith("intro"):
            tactic_types.add("intro")
        elif s.startswith("have"):
            tactic_types.add("have")
        elif s in ("ring", "simp", "linarith", "field_simp",
                    "norm_num", "nlinarith"):
            tactic_types.add(s)
        elif s.startswith("calc"):
            tactic_types.add("calc")
        elif s.startswith("constructor"):
            tactic_types.add("constructor")
        else:
            tactic_types.add("other")
    if len(tactic_types) >= 2:
        return "multi"
    steps_text = " ".join(proof_steps).lower()
    patterns = ["rfl", "add_comm", "mul_comm", "ring", "field_simp",
                "linarith", "simp", "intro", "apply", "nlinarith"]
    for p in patterns:
        if p in steps_text:
            return p
    return "other"


def is_lemma_novelty(proof_steps: list[str]) -> bool:
    structural = {"simp", "ring", "linarith", "field_simp", "rfl", "norm_num",
                  "nlinarith", "omega", "native_decide"}
    for step in proof_steps:
        s = step.strip().lower()
        tactic = s.split()[0] if s else ""
        if tactic not in structural and not s.startswith("exact"):
            return True
    lemma_refs = re.findall(r'rw\s*\[([^\]]+)\]', " ".join(proof_steps))
    for ref in lemma_refs:
        parts = ref.split(",")
        for p in parts:
            p = p.strip()
            if p not in structural and p not in ("h", "h1", "h2", "h3", "h'"):
                return True
    return False


# ---------------------------------------------------------------------------
# Search configuration
# ---------------------------------------------------------------------------

@dataclass
class DirectSearchConfig:
    max_depth: int = 10
    max_expansions: int = 500
    top_k_lemmas: int = 30
    depth_penalty: float = 0.05
    use_proof_checker: bool = True
    num_threads: int = 4


@dataclass(order=True)
class _PrioritizedState:
    priority: float
    depth: int
    tiebreaker: int
    state: ProofState = field(compare=False)
    steps: list = field(compare=False)


# ---------------------------------------------------------------------------
# Best-first search with pre-scored TF-IDF lemmas
# ---------------------------------------------------------------------------

class DirectBestFirstSearch:
    """Best-first proof search using pre-scored TF-IDF lemma retrieval.

    Unlike GNNBestFirstSearch, no GNN is involved. Lemma scores come from
    TF-IDF k-NN retrieval, done once per theorem before search begins.
    """

    def __init__(
        self,
        retriever: DirectRetrieval,
        config: DirectSearchConfig | None = None,
        proof_checker=None,
    ):
        self.retriever = retriever
        self.config = config or DirectSearchConfig()
        self.proof_checker = proof_checker

        import torch
        torch.set_num_threads(self.config.num_threads)

        self._tiebreaker: int = 0
        self._lemma_cache: dict[str, list[tuple[str, float]]] = {}

    def _next_tiebreaker(self) -> int:
        self._tiebreaker += 1
        return self._tiebreaker

    # ------------------------------------------------------------------
    # Main search
    # ------------------------------------------------------------------

    def search(
        self,
        theorem_statement: str,
        domain: str | None = None,
        verbose: bool = False,
    ) -> tuple[list, ProofState | None]:
        """Search for a proof using TF-IDF-retrieved lemmas."""
        self._tiebreaker = 0

        # Retrieve scored lemmas for this theorem
        cache_key = f"{theorem_statement[:200]}__{domain or ''}"
        if cache_key not in self._lemma_cache:
            scored = self.retriever.retrieve(
                theorem_statement,
                k=50,
                top_n=self.config.top_k_lemmas,
                domain=domain,
            )
            self._lemma_cache[cache_key] = scored
        scored_lemmas = self._lemma_cache[cache_key]

        root_state = ProofState.initial(theorem_statement)
        root = _PrioritizedState(
            priority=-1.0,
            depth=0,
            tiebreaker=self._next_tiebreaker(),
            state=root_state,
            steps=[],
        )

        heap = [root]
        expansions = 0
        t_start = time.time()

        while heap and expansions < self.config.max_expansions:
            current = heapq.heappop(heap)
            state = current.state
            depth = current.depth

            if state.is_complete:
                if self.proof_checker is not None and state.steps:
                    proof_body = ProofState._render_proof(state.steps)
                    code = wrap_theorem_with_proof(state.theorem_statement, proof_body)
                    check_results = self.proof_checker.check_batch([code])
                    if check_results[0].success:
                        if verbose:
                            print(f"  ✓ Verified proof at depth {depth}, "
                                  f"expansions={expansions}, {time.time()-t_start:.1f}s")
                        return (state.steps, state)
                    else:
                        state.is_complete = False
                        state.is_dead = True
                        continue
                else:
                    if verbose:
                        print(f"  ✓ Found proof at depth {depth}")
                    return (state.steps, state)

            if state.is_dead:
                continue
            if depth >= self.config.max_depth:
                continue

            expansions += 1

            # Generate actions: TF-IDF lemmas + structural tactics
            candidates = self._generate_actions(state, scored_lemmas)

            if not candidates:
                continue

            for tactic, score in candidates:
                child_state = state.apply_tactic(tactic)
                if child_state is None:
                    continue
                priority = -(score / (1 + (depth + 1) * self.config.depth_penalty))
                heapq.heappush(
                    heap,
                    _PrioritizedState(
                        priority=priority,
                        depth=depth + 1,
                        tiebreaker=self._next_tiebreaker(),
                        state=child_state,
                        steps=current.steps + [tactic],
                    ),
                )

        return ([], None)

    # ------------------------------------------------------------------
    # Action generation
    # ------------------------------------------------------------------

    def _generate_actions(
        self,
        state: ProofState,
        scored_lemmas: list[tuple[str, float]],
    ) -> list[tuple[Tactic, float]]:
        """Generate candidate tactics with scores."""
        candidates: list[tuple[Tactic, float]] = []

        # 1. Lemma applications from TF-IDF retrieval
        for lemma, score in scored_lemmas:
            candidates.append((Tactic(TacticType.APPLY, lemma=lemma), score))
            candidates.append((Tactic(TacticType.EXACT, lemma=lemma), score * 0.9))
            # Rewrite for algebraic lemmas
            if any(kw in lemma.lower()
                   for kw in ("add", "mul", "eq", "comm", "assoc", "zero", "one",
                              "neg", "sub", "div", "ring", "field", "simp")):
                candidates.append((Tactic(TacticType.REWRITE, lemma=lemma), score * 0.95))

        # 2. Built-in structural tactics similar to original best-first
        hypotheses = state.hypotheses

        for hyp_name in list(hypotheses.keys())[:5]:
            candidates.append((Tactic(TacticType.EXACT, hypothesis=hyp_name), 0.72))

        for hyp_name, hyp_type in hypotheses.items():
            if "=" in hyp_type or "↔" in hyp_type:
                candidates.append((Tactic(TacticType.REWRITE, hypothesis=hyp_name), 0.77))
                break

        if state.goals and ("→" in state.goals[0] or "∀" in state.goals[0]):
            candidates.append((Tactic(TacticType.INTRO, hypothesis="h"), 0.82))

        for hyp_name, hyp_type in list(hypotheses.items())[:3]:
            if any(op in hyp_type for op in ("=", "≠", "↔", "≤", "≥", "<", ">")):
                continue
            candidates.append((Tactic(TacticType.CASES, hypothesis=hyp_name), 0.35))
            break

        # Automation tactics
        if state.goals:
            goal = state.goals[0]
            has_implication = "→" in goal or "∀" in goal
            if not has_implication:
                if any(op in goal for op in ("*", "^", "+", "-", "=")):
                    candidates.append((Tactic(TacticType.RING), 0.72))
            if ("/" in goal or "⁻¹" in goal) and not has_implication:
                candidates.append((Tactic(TacticType.FIELD_SIMP), 0.72))
            if any(op in goal for op in ("≤", "≥", "<", ">", "=")) and not has_implication:
                candidates.append((Tactic(TacticType.LINARITH), 0.72))
            candidates.append((Tactic(TacticType.SIMP), 0.67))

        # Limit candidates
        max_actions = self.config.top_k_lemmas * 2 + 12
        if len(candidates) > max_actions:
            candidates = candidates[:max_actions]

        return candidates


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def run_gate3_direct(
    retriever: DirectRetrieval,
    theorems: list[dict],
    config: DirectSearchConfig,
    checker: BatchChecker | None,
    output_path: Path,
) -> dict:
    print("\n" + "=" * 70)
    print("GATE 3: Direct TF-IDF Retrieval + Best-First Search (64 theorems)")
    print("=" * 70)
    print(f"  Index: {retriever.n_pairs} proof-step pairs")
    print(f"  TF-IDF vocab: {len(retriever.vectorizer.vocabulary_)}")
    print(f"  Top-K lemmas: {config.top_k_lemmas}")
    print(f"  Max expansions: {config.max_expansions}")
    print()

    search = DirectBestFirstSearch(
        retriever=retriever,
        config=config,
        proof_checker=checker if config.use_proof_checker else None,
    )

    results = []
    t_start = time.time()
    passed = []
    failed_reasons: dict[str, int] = {}

    for i, t in enumerate(theorems):
        stmt = t["statement"]
        name = t["name"]
        domain = t.get("domain", "unknown")
        era = t.get("era", "unknown")
        ground_truth = t.get("proof", "?")

        t0 = time.time()
        proof_steps, final_state = search.search(stmt, domain=domain)
        search_time = time.time() - t0

        if not proof_steps:
            ok = False
            err = "no proof found"
            failed_reasons["no_proof"] = failed_reasons.get("no_proof", 0) + 1
        elif checker is None:
            ok = True
            err = ""
        else:
            proof_text = ProofState._render_proof(proof_steps)
            full_code = wrap_theorem_with_proof(stmt, proof_text)
            check_results = checker.check_batch([full_code])
            ok = check_results[0].success
            err = check_results[0].errors[0][:200] if check_results[0].errors else ""
            if not ok:
                reason_key = f"lean_reject:{err[:60]}"
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
            "proof_steps": steps_str,
            "num_steps": len(proof_steps),
            "ground_truth": ground_truth,
            "search_time_s": round(search_time, 1),
            "pattern": pattern,
            "lemma_novelty": lemma_novel,
        }
        results.append(result)

        if ok:
            passed.append(result)

        status = "✓" if ok else "✗"
        eta = (time.time() - t_start) / (i + 1) * (len(theorems) - i - 1)
        print(f"  [{i+1:2d}/{len(theorems)}] {status} {name:45s} "
              f"[{pattern:12s}] {search_time:.1f}s  "
              f"ETA: {eta/60:.0f}m  ({len(passed)} passed)")

        if ok and len(proof_steps) > 0:
            print(f"         Proof: {steps_str}")
            if len(proof_steps) >= 2:
                print(f"         ★ MULTI-STEP ({len(proof_steps)} steps)")

    elapsed = time.time() - t_start
    n_total = len(theorems)
    n_passed = len(passed)
    rate = n_passed / max(1, n_total)

    multi = [r for r in passed if r["num_steps"] >= 2]
    lemma_novel = [r for r in passed if r["lemma_novelty"]]
    structural = [r for r in passed if not r["lemma_novelty"]]

    # Stats
    print(f"\n{'=' * 70}")
    print("GATE 3: DIRECT TF-IDF RETRIEVAL RESULTS")
    print(f"{'=' * 70}")
    print(f"  Total:    {n_passed}/{n_total} ({rate:.1%})")
    print(f"  Multi-step: {len(multi)}")
    print(f"  Lemma-novelty: {len(lemma_novel)}")
    print(f"  Structural-only: {len(structural)}")
    print(f"  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}m)")

    # Domain breakdown
    domains = Counter(r["domain"] for r in results)
    print(f"\n  By domain:")
    for dom in sorted(domains.keys()):
        dom_total = domains[dom]
        dom_passed = sum(1 for r in passed if r["domain"] == dom)
        dom_ln = sum(1 for r in lemma_novel if r["domain"] == dom)
        dom_ms = sum(1 for r in multi if r["domain"] == dom)
        print(f"    {dom:<20} {dom_passed}/{dom_total} "
              f"({dom_passed/max(1,dom_total)*100:.0f}%) "
              f"LN: {dom_ln}  MS: {dom_ms}")

    print(f"\n  Failure reasons:")
    for reason, count in sorted(failed_reasons.items(), key=lambda x: -x[1])[:10]:
        print(f"    {reason:<70} {count}")

    # Build output
    gate3_status = "PASS" if n_passed >= 10 else "FAIL"
    out = {
        "task": "Gate 3: Direct TF-IDF retrieval + Best-first search + Lean checker",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "architecture": "TF-IDF k-NN over 226K proof-step pairs, no GNN",
        "smoke_test": retriever.smoke_test(n_samples=100, k=50, rank_threshold=30),
        "config": {
            "max_depth": config.max_depth,
            "max_expansions": config.max_expansions,
            "top_k_lemmas": config.top_k_lemmas,
            "depth_penalty": config.depth_penalty,
            "use_proof_checker": config.use_proof_checker,
            "num_threads": config.num_threads,
            "tfidf_max_features": retriever.max_features,
        },
        "index": {
            "n_pairs": retriever.n_pairs,
            "vocab_size": len(retriever.vectorizer.vocabulary_),
        },
        "gate3": {
            "status": gate3_status,
            "total": n_total,
            "passed": n_passed,
            "rate": rate,
            "multi_step": len(multi),
            "lemma_novelty": len(lemma_novel),
            "structural_only": len(structural),
            "elapsed_s": elapsed,
            "failed_reasons": dict(sorted(failed_reasons.items(), key=lambda x: -x[1])),
        },
        "passed": [
            {
                "name": r["name"],
                "domain": r["domain"],
                "era": r["era"],
                "proof": r["proof_steps"],
                "pattern": r["pattern"],
                "lemma_novelty": r["lemma_novelty"],
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
    import argparse

    parser = argparse.ArgumentParser(description="Gate 3: Direct TF-IDF retrieval")
    parser.add_argument("--max-expansions", type=int, default=500)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--no-lean", action="store_true",
                        help="Skip Lean proof checking")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--retrieval-k", type=int, default=50,
                        help="Number of TF-IDF neighbors for retrieval")
    parser.add_argument("--tfidf-features", type=int, default=8000)
    parser.add_argument("--output", type=str,
                        default="data/direct_retrieval_gate3.json")
    parser.add_argument("--pairs", type=str,
                        default="data/raw/proof_step_pairs.jsonl")
    parser.add_argument("--theorems", type=str,
                        default="data/raw/gate3_v2.jsonl")
    parser.add_argument("--project-dir", type=str, default=None,
                        help="Lean project directory for proof checking")
    args = parser.parse_args()

    _PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

    # Load retriever
    retriever = DirectRetrieval(
        _PROJECT_ROOT / args.pairs,
        max_features=args.tfidf_features,
        cache_dir=_PROJECT_ROOT / "data/tfidf_cache",
    )

    # Load theorems
    theorems = load_jsonl(_PROJECT_ROOT / args.theorems)
    print(f"Loaded {len(theorems)} theorems from {args.theorems}")

    # Config
    config = DirectSearchConfig(
        max_depth=args.depth,
        max_expansions=args.max_expansions,
        top_k_lemmas=args.top_k,
        use_proof_checker=not args.no_lean,
        num_threads=args.threads,
    )

    # Proof checker
    checker = None
    if not args.no_lean:
        project_dir = args.project_dir or str(_PROJECT_ROOT / "lean_project")
        from pathlib import Path as _Path
        if _Path(project_dir).exists():
            checker = BatchChecker(project_dir=project_dir)
            print(f"Proof checker: {project_dir}")
        else:
            checker = BatchChecker()
            print(f"Proof checker: default (no project dir)")

    # Run
    output_path = _PROJECT_ROOT / args.output
    run_gate3_direct(retriever, theorems, config, checker, output_path)


if __name__ == "__main__":
    main()
