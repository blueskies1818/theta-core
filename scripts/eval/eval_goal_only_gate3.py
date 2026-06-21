#!/usr/bin/env python3
"""Gate 3 evaluation: Goal-only encoder + best-first search + Lean checker.

Architecture:
  Goal → GoalOnlyEncoder → embedding → k-NN over 226K training goals
  → collect lemmas → rank by frequency × similarity → top-30
  → best-first search → Lean proof checker

NO lemma encoder. NO GNN graph. NO import edges.
Output: data/goal_only_gate3.json

Usage:
    # Smoke (no proof checker, just structural):
    python scripts/eval/eval_goal_only_gate3.py --no-proof-checker --max-expansions 100

    # Full eval with Lean:
    python scripts/eval/eval_goal_only_gate3.py
"""

from __future__ import annotations

import argparse
import heapq
import json
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
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
    retrieve_lemmas,
    tokenize_goal,
    _tokenize_batch,
)
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
        if s.startswith("rw"): tactic_types.add("rw")
        elif s.startswith("exact"): tactic_types.add("exact")
        elif s.startswith("apply"): tactic_types.add("apply")
        elif s.startswith("intro"): tactic_types.add("intro")
        elif s.startswith("have"): tactic_types.add("have")
        elif s in ("ring", "simp", "linarith", "field_simp", "norm_num",
                     "nlinarith", "positivity"):
            tactic_types.add(s)
        elif s.startswith("calc"): tactic_types.add("calc")
        elif s.startswith("constructor"): tactic_types.add("constructor")
        else: tactic_types.add("other")
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
class GoalOnlySearchConfig:
    max_depth: int = 10
    max_expansions: int = 500
    top_k_lemmas: int = 30
    retrieval_k: int = 50
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
# Best-first search with goal-only embeddings
# ---------------------------------------------------------------------------

class GoalOnlyBestFirstSearch:
    """Best-first proof search using goal-only encoder + k-NN lemma retrieval."""

    def __init__(
        self,
        encoder: GoalOnlyEncoder,
        vocab: dict[str, int],
        index_goals: list[str],
        index_lemmas: list[str],
        index_embeddings: torch.Tensor,
        config: GoalOnlySearchConfig | None = None,
        proof_checker=None,
    ):
        self.encoder = encoder
        self.vocab = vocab
        self.index_goals = index_goals
        self.index_lemmas = index_lemmas
        self.index_embeddings = index_embeddings
        self.config = config or GoalOnlySearchConfig()
        self.proof_checker = proof_checker

        import torch
        torch.set_num_threads(self.config.num_threads)

        self._tiebreaker: int = 0
        self._lemma_cache: dict[str, list[tuple[str, float]]] = {}

    def _next_tiebreaker(self) -> int:
        self._tiebreaker += 1
        return self._tiebreaker

    def search(
        self,
        theorem_statement: str,
        domain: str | None = None,
        verbose: bool = False,
    ) -> tuple[list, ProofState | None]:
        self._tiebreaker = 0

        # Phase 0: Fast structural pre-check — try structural tactics only
        # before the expensive lemma retrieval search.  Many theorems
        # (algebra, basic analysis) are solvable with ring/field_simp/linarith
        # and hypothesis rewrites alone.
        if self.proof_checker is not None:
            result = self._structural_search(theorem_statement)
            if result is not None:
                return result

        # Retrieve scored lemmas for this theorem
        cache_key = theorem_statement[:200]
        if cache_key not in self._lemma_cache:
            scored = retrieve_lemmas(
                self.encoder, self.vocab,
                theorem_statement,
                self.index_goals, self.index_lemmas,
                self.index_embeddings,
                k=self.config.retrieval_k,
                top_n=self.config.top_k_lemmas,
            )
            self._lemma_cache[cache_key] = scored
        scored_lemmas = self._lemma_cache[cache_key]

        if verbose and scored_lemmas:
            print(f"  Top-5 lemmas: {[(n, f'{s:.3f}') for n, s in scored_lemmas[:5]]}")

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
                            print(f"  ✓ Verified at depth {depth}, expansions={expansions}")
                        return (state.steps, state)
                    else:
                        state.is_complete = False
                        state.is_dead = True
                        continue
                else:
                    return (state.steps, state)

            if state.is_dead:
                continue
            if depth >= self.config.max_depth:
                continue

            expansions += 1

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

    def _generate_actions(
        self,
        state: ProofState,
        scored_lemmas: list[tuple[str, float]],
    ) -> list[tuple[Tactic, float]]:
        candidates: list[tuple[Tactic, float]] = []

        # 1. Built-in structural tactics FIRST — they survive truncation
        # and have scores competitive with lemma retrieval (0.80–0.92).
        hypotheses = state.hypotheses

        for hyp_name in list(hypotheses.keys())[:5]:
            candidates.append((Tactic(TacticType.EXACT, hypothesis=hyp_name), 0.85))

        for hyp_name, hyp_type in hypotheses.items():
            if "=" in hyp_type or "↔" in hyp_type:
                candidates.append((Tactic(TacticType.REWRITE, hypothesis=hyp_name), 0.92))
                break

        if state.goals and ("→" in state.goals[0] or "∀" in state.goals[0]):
            candidates.append((Tactic(TacticType.INTRO, hypothesis="h"), 0.90))

        for hyp_name, hyp_type in list(hypotheses.items())[:3]:
            if any(op in hyp_type for op in ("=", "≠", "↔", "≤", "≥", "<", ">")):
                continue
            candidates.append((Tactic(TacticType.CASES, hypothesis=hyp_name), 0.35))
            break

        if state.goals:
            goal = state.goals[0]
            has_implication = "→" in goal or "∀" in goal
            if not has_implication:
                if any(op in goal for op in ("*", "^", "+", "-", "=")):
                    candidates.append((Tactic(TacticType.RING), 0.88))
            if ("/" in goal or "⁻¹" in goal) and not has_implication:
                candidates.append((Tactic(TacticType.FIELD_SIMP), 0.85))
            if any(op in goal for op in ("≤", "≥", "<", ">", "=")) and not has_implication:
                candidates.append((Tactic(TacticType.LINARITH), 0.87))
                candidates.append((Tactic(TacticType.NLINARITH), 0.90))
            candidates.append((Tactic(TacticType.SIMP), 0.80))

        # 2. Lemma applications from retrieval
        for lemma, score in scored_lemmas:
            candidates.append((Tactic(TacticType.APPLY, lemma=lemma), score))
            candidates.append((Tactic(TacticType.EXACT, lemma=lemma), score * 0.9))
            if any(kw in lemma.lower()
                   for kw in ("add", "mul", "eq", "comm", "assoc", "zero", "one",
                              "neg", "sub", "div", "ring", "field", "simp")):
                candidates.append((Tactic(TacticType.REWRITE, lemma=lemma), score * 0.95))

        max_actions = self.config.top_k_lemmas * 2 + 12
        if len(candidates) > max_actions:
            candidates = candidates[:max_actions]

        return candidates

    def _structural_search(
        self, theorem_statement: str
    ) -> tuple[list, ProofState | None] | None:
        """Fast structural pre-check: try only built-in tactics (no lemma retrieval).

        Many theorems are solvable with ring/field_simp/linarith/hypothesis rewrites
        alone.  This avoids the expensive lemma-chain search for simple cases.
        Uses a small expansion budget and only structural candidates.
        """
        state = ProofState.initial(theorem_statement)
        heap = [_PrioritizedState(
            priority=-1.0, depth=0, tiebreaker=0,
            state=state, steps=[],
        )]
        expansions = 0
        max_exp = min(50, self.config.max_expansions // 4)
        tiebreaker = 1

        while heap and expansions < max_exp:
            current = heapq.heappop(heap)
            state = current.state
            depth = current.depth

            if state.is_complete:
                if self.proof_checker is not None and state.steps:
                    proof_body = ProofState._render_proof(state.steps)
                    code = wrap_theorem_with_proof(state.theorem_statement, proof_body)
                    check_results = self.proof_checker.check_batch([code])
                    if check_results[0].success:
                        return (state.steps, state)
                    else:
                        state.is_complete = False
                        state.is_dead = True
                        continue
                else:
                    return (state.steps, state)

            if state.is_dead or depth >= 10:
                continue

            expansions += 1

            # Generate ONLY structural actions (no lemmas)
            candidates: list[tuple[Tactic, float]] = []
            hypotheses = state.hypotheses

            for hyp_name in list(hypotheses.keys())[:5]:
                candidates.append((Tactic(TacticType.EXACT, hypothesis=hyp_name), 0.88))

            for hyp_name, hyp_type in hypotheses.items():
                if "=" in hyp_type or "↔" in hyp_type:
                    candidates.append((Tactic(TacticType.REWRITE, hypothesis=hyp_name), 0.95))
                    break

            if state.goals and ("→" in state.goals[0] or "∀" in state.goals[0]):
                candidates.append((Tactic(TacticType.INTRO, hypothesis="h"), 0.93))

            if state.goals:
                goal = state.goals[0]
                has_impl = "→" in goal or "∀" in goal
                if not has_impl:
                    if any(op in goal for op in ("*", "^", "+", "-", "=")):
                        candidates.append((Tactic(TacticType.RING), 0.92))
                if ("/" in goal or "⁻¹" in goal) and not has_impl:
                    candidates.append((Tactic(TacticType.FIELD_SIMP), 0.90))
                if any(op in goal for op in ("≤", "≥", "<", ">", "=")) and not has_impl:
                    candidates.append((Tactic(TacticType.LINARITH), 0.90))
                    candidates.append((Tactic(TacticType.NLINARITH), 0.93))
                candidates.append((Tactic(TacticType.SIMP), 0.85))

            for tactic, score in candidates:
                child_state = state.apply_tactic(tactic)
                if child_state is None:
                    continue
                priority = -(score / (1 + (depth + 1) * self.config.depth_penalty))
                heapq.heappush(heap, _PrioritizedState(
                    priority=priority, depth=depth + 1,
                    tiebreaker=tiebreaker,
                    state=child_state,
                    steps=current.steps + [tactic],
                ))
                tiebreaker += 1

        return None


# ---------------------------------------------------------------------------
# Gate 3 evaluation
# ---------------------------------------------------------------------------

def _save_intermediate(
    output_path: Path,
    results: list[dict],
    passed: list[dict],
    n_done: int,
    n_total: int,
) -> None:
    """Save partial results for crash recovery."""
    partial = {
        "partial": True,
        "n_done": n_done,
        "n_total": n_total,
        "n_passed": len(passed),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": results,
        "passed": [
            {"name": r["name"], "domain": r["domain"], "era": r.get("era", ""),
             "proof": r.get("proof_steps", []), "pattern": r.get("pattern", ""),
             "lemma_novelty": r.get("lemma_novelty", False)}
            for r in passed
        ],
    }
    partial_path = Path(str(output_path) + ".partial")
    save_json(partial, partial_path)
    print(f"  [save] Partial results → {partial_path}")


def run_gate3(
    encoder: GoalOnlyEncoder,
    vocab: dict[str, int],
    index_goals: list[str],
    index_lemmas: list[str],
    index_embeddings: torch.Tensor,
    theorems: list[dict],
    config: GoalOnlySearchConfig,
    checker: BatchChecker | None,
    output_path: Path,
) -> dict:
    print("\n" + "=" * 70)
    print("GATE 3: Goal-Only Encoder + Best-First Search")
    print("=" * 70)
    print(f"  Encoder params: {encoder.count_params():,}")
    print(f"  Index: {len(index_goals)} training goals")
    print(f"  Top-K lemmas: {config.top_k_lemmas}")
    print(f"  Max expansions: {config.max_expansions}")
    print(f"  Proof checker: {'ON' if config.use_proof_checker else 'OFF'}")
    print()

    search = GoalOnlyBestFirstSearch(
        encoder=encoder,
        vocab=vocab,
        index_goals=index_goals,
        index_lemmas=index_lemmas,
        index_embeddings=index_embeddings,
        config=config,
        proof_checker=checker if config.use_proof_checker else None,
    )

    results = []
    t_start = time.time()
    passed = []
    failed_reasons: dict[str, int] = {}
    max_theorem_time = 300  # 5 min per theorem maximum

    # Track partial results for crash recovery
    _last_save_time = t_start
    _save_interval = 120  # save every 2 minutes

    for i, t in enumerate(theorems):
        stmt = t["statement"]
        name = t["name"]
        domain = t.get("domain", "unknown")
        era = t.get("era", "unknown")
        ground_truth = t.get("proof", "?")

        t0 = time.time()
        try:
            import threading

            search_result: list = [None, None]  # [proof_steps, final_state]
            search_exception: list = [None]

            def _run_search():
                try:
                    steps, st = search.search(stmt, domain=domain)
                    search_result[0] = steps
                    search_result[1] = st
                except Exception as ex:
                    search_exception[0] = ex

            search_thread = threading.Thread(target=_run_search, daemon=True)
            search_thread.start()
            search_thread.join(timeout=max_theorem_time)

            if search_thread.is_alive():
                # Theorem search taking too long — abandon it
                proof_steps = []
                search_time = time.time() - t0
                ok = False
                err = f"timed out after {max_theorem_time}s"
                failed_reasons["search_timeout"] = failed_reasons.get("search_timeout", 0) + 1
                print(f"  [{i+1:2d}/{len(theorems)}] ✗ {name:45s} "
                      f"[TIMEOUT    ] {search_time:.1f}s  "
                      f"ETA: 0m  ({len(passed)} passed)")
                result = {
                    "name": name, "era": era, "domain": domain,
                    "success": False, "error": err,
                    "proof_steps": [], "num_steps": 0,
                    "ground_truth": ground_truth,
                    "search_time_s": round(search_time, 1),
                    "pattern": "timeout", "lemma_novelty": False,
                }
                results.append(result)
                # Periodic save
                _now = time.time()
                if _now - _last_save_time > _save_interval:
                    _save_intermediate(output_path, results, passed, i + 1, len(theorems))
                    _last_save_time = _now
                continue
            elif search_exception[0] is not None:
                raise search_exception[0]
            else:
                proof_steps, final_state = search_result[0], search_result[1]
        except Exception as e:
            proof_steps = []
            search_time = time.time() - t0
            ok = False
            err = f"search crash: {str(e)[:200]}"
            failed_reasons[f"crash:{type(e).__name__}"] = \
                failed_reasons.get(f"crash:{type(e).__name__}", 0) + 1
            print(f"  [{i+1:2d}/{len(theorems)}] ✗ {name:45s} "
                  f"[CRASH      ] {search_time:.1f}s  "
                  f"ETA: 0m  ({len(passed)} passed)")
            print(f"         Error: {type(e).__name__}: {str(e)[:120]}")
            result = {
                "name": name, "era": era, "domain": domain,
                "success": False, "error": err,
                "proof_steps": [], "num_steps": 0,
                "ground_truth": ground_truth,
                "search_time_s": round(search_time, 1),
                "pattern": "crash", "lemma_novelty": False,
            }
            results.append(result)
            # Periodic save
            _now = time.time()
            if _now - _last_save_time > _save_interval:
                _save_intermediate(output_path, results, passed, i + 1, len(theorems))
                _last_save_time = _now
            continue
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

        # Periodic save for crash recovery
        _now = time.time()
        if _now - _last_save_time > _save_interval:
            _save_intermediate(output_path, results, passed, i + 1, len(theorems))
            _last_save_time = _now

    elapsed = time.time() - t_start
    n_total = len(theorems)
    n_passed = len(passed)
    rate = n_passed / max(1, n_total)

    multi = [r for r in passed if r["num_steps"] >= 2]
    lemma_novel = [r for r in passed if r["lemma_novelty"]]
    structural = [r for r in passed if not r["lemma_novelty"]]

    # Stats
    print(f"\n{'=' * 70}")
    print("GATE 3: GOAL-ONLY ENCODER RESULTS")
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
    baseline = 0.156  # Gate 2 full-graph baseline
    gate3_status = "PASS" if n_passed >= 10 else "FAIL"
    out = {
        "task": "Gate 3: Goal-only encoder + best-first search + Lean checker",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "architecture": "Goal-only encoder (text embed → avg pool → MLP), no GNN graph",
        "baseline_comparison": {
            "baseline_proof_rate": baseline,
            "this_proof_rate": rate,
            "improvement": rate - baseline,
        },
        "encoder": {
            "params": encoder.count_params(),
            "hidden_dim": encoder.hidden_dim,
            "vocab_size": encoder.vocab_size,
        },
        "config": {
            "max_depth": config.max_depth,
            "max_expansions": config.max_expansions,
            "top_k_lemmas": config.top_k_lemmas,
            "retrieval_k": config.retrieval_k,
            "depth_penalty": config.depth_penalty,
            "use_proof_checker": config.use_proof_checker,
            "num_threads": config.num_threads,
            "index_size": len(index_goals),
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
    parser = argparse.ArgumentParser(
        description="Gate 3: Goal-only encoder evaluation"
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
        "--max-expansions", type=int, default=500,
        help="Max expansions per theorem",
    )
    parser.add_argument(
        "--top-k", type=int, default=30,
        help="Top-K lemmas per state",
    )
    parser.add_argument(
        "--retrieval-k", type=int, default=50,
        help="Number of nearest neighbors for retrieval",
    )
    parser.add_argument(
        "--depth-penalty", type=float, default=0.05,
        help="Depth penalty factor",
    )
    parser.add_argument(
        "--num-threads", type=int, default=4,
        help="Number of CPU threads",
    )
    parser.add_argument(
        "--output", default="data/goal_only_gate3.json",
        help="Output JSON file",
    )
    parser.add_argument(
        "--no-proof-checker", action="store_true",
        help="Disable proof checker (faster, structural-only eval)",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Smoke test: only run first 3 theorems",
    )
    parser.add_argument(
        "--max-pairs", type=int, default=0,
        help="Max training pairs for index (0 = all)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("GATE 3: GOAL-ONLY ENCODER EVALUATION")
    print("=" * 70)
    print(f"  Encoder: {args.encoder}")
    print(f"  Proof checker: {'OFF' if args.no_proof_checker else 'ON'}")
    print(f"  Threads: {args.num_threads}")
    print(f"  Smoke: {'YES' if args.smoke else 'NO'}")
    print()

    # Hardware constraint
    if args.num_threads > 12:
        print(f"WARNING: Reducing threads from {args.num_threads} to 12")
        args.num_threads = 12

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

    # --- Load training pairs & build index ---
    pairs_path = _project_root / args.pairs
    max_pairs = args.max_pairs if args.max_pairs > 0 else None

    print(f"\nLoading training pairs for index...")
    goals, lemmas, _lemma_to_indices = prepare_lemma_groups(pairs_path, max_pairs)
    print(f"  Loaded {len(goals)} pairs")

    # Build vocab from training pairs
    print("Building vocabulary...")
    vocab = build_vocabulary(goals, max_vocab=args.vocab_size)
    print(f"  Vocabulary: {len(vocab)} tokens")

    # Encode all training goals
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

    # --- Config ---
    use_pc = not args.no_proof_checker
    config = GoalOnlySearchConfig(
        max_depth=10,
        max_expansions=args.max_expansions,
        top_k_lemmas=args.top_k,
        retrieval_k=args.retrieval_k,
        depth_penalty=args.depth_penalty,
        use_proof_checker=use_pc,
        num_threads=args.num_threads,
    )

    checker = BatchChecker(timeout=15, max_workers=4, cache_size=50000) if use_pc else None

    output_path = _project_root / args.output

    # --- Run ---
    result = run_gate3(
        encoder=encoder,
        vocab=vocab,
        index_goals=goals,
        index_lemmas=lemmas,
        index_embeddings=index_embeddings,
        theorems=theorems,
        config=config,
        checker=checker,
        output_path=output_path,
    )

    n_passed = result["gate3"]["passed"]
    n_multi = result["gate3"]["multi_step"]
    n_ln = result["gate3"]["lemma_novelty"]

    print(f"\n{'=' * 70}")
    print("FINAL")
    print(f"{'=' * 70}")
    print(f"  Proofs found: {n_passed}/{len(theorems)} ({result['gate3']['rate']:.0%})")
    print(f"  Baseline:     15.6%")
    print(f"  Multi-step:   {n_multi}")
    print(f"  Lemma-novelty: {n_ln}")
    print(f"  Output:       {output_path}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
