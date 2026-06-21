#!/usr/bin/env python3
"""Direct retrieval: k-NN proof-step lookup bypassing GNN graph.

ARCHITECTURE:
  Goal -> TF-IDF encoder -> embedding
  -> k-NN over 226K training goals -> collect their correct lemmas
  -> rank by frequency x similarity -> feed top-30 to best-first search
  -> Lean proof checker

No graph. No contrastive loss. No GNN. Proof signal IS the index.

Usage:
    python src/retrieval/direct_lookup.py --smoke-test    # Smoke test
    python src/retrieval/direct_lookup.py --eval           # Full gate3_v2 eval
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.explorer.proof_state import ProofState, Tactic, TacticType
from src.proof_checker.batch_checker import BatchChecker
from src.proof_checker.formats import wrap_theorem_with_proof


# ---------------------------------------------------------------------------
# Tokenization for Lean goal text
# ---------------------------------------------------------------------------

_MATH_SPLIT_RE = re.compile(
    r'[\s+\-*/^=()\[\]{}:.,;→←↔⇒⇔∀∃λ≤≥<>&|!~@#$%\\]+'
)


def tokenize_goal(goal_text: str) -> list[str]:
    """Tokenize a Lean goal expression into meaningful tokens.

    Splits on math operators, whitespace, and punctuation. Keeps
    tokens of length >= 1. Lowercases everything.
    """
    parts = _MATH_SPLIT_RE.split(goal_text)
    tokens = []
    for part in parts:
        part = part.strip().lower()
        if len(part) >= 1:
            tokens.append(part)
    return tokens


# ---------------------------------------------------------------------------
# TF-IDF Index
# ---------------------------------------------------------------------------


class TFIDFIndex:
    """TF-IDF index over proof-step pair goals.

    Builds a vocabulary from the 226K training goals, computes IDF
    weights, and stores sparse TF-IDF vectors for fast k-NN retrieval.
    """

    def __init__(self, vocab_size: int = 8000):
        self.vocab_size = vocab_size
        self.vocab: dict[str, int] = {}  # term -> index
        self.idf: np.ndarray | None = None  # [vocab_size]
        self._doc_terms: list[np.ndarray] = []  # per-doc term indices
        self._doc_vals: list[np.ndarray] = []  # per-doc tf-idf values
        self._doc_lemmas: list[str] = []  # per-doc correct lemma
        self._doc_goals: list[str] = []  # per-doc goal text
        self.N: int = 0

    @staticmethod
    def _compute_df(tokenized_docs: list[list[str]]) -> dict[str, int]:
        df: dict[str, int] = {}
        for tokens in tokenized_docs:
            for term in set(tokens):
                df[term] = df.get(term, 0) + 1
        return df

    def fit(self, goals: list[str], lemmas: list[str]) -> "TFIDFIndex":
        """Build vocabulary and index from goal-lemma pairs."""
        print(f"Fitting TF-IDF index on {len(goals)} goals...")
        t0 = time.time()

        # Tokenize all goals
        print("  Tokenizing goals...")
        tokenized = [tokenize_goal(g) for g in goals]

        # Compute document frequencies
        print("  Computing document frequencies...")
        df = self._compute_df(tokenized)

        # Build vocabulary: top-N by document frequency
        print(f"  Building vocabulary (top {self.vocab_size})...")
        sorted_terms = sorted(df.items(), key=lambda x: -x[1])[:self.vocab_size]
        self.vocab = {term: i for i, (term, _) in enumerate(sorted_terms)}

        # Compute IDF
        N = len(goals)
        self.idf = np.ones(self.vocab_size, dtype=np.float32)
        for term, count in df.items():
            idx = self.vocab.get(term)
            if idx is not None and count > 0:
                self.idf[idx] = math.log((N + 1) / (count + 1)) + 1.0

        # Build sparse TF-IDF vectors
        print(f"  Building TF-IDF vectors for {N} documents...")
        doc_terms: list[np.ndarray] = []
        doc_vals: list[np.ndarray] = []

        for tokens in tokenized:
            tf: dict[int, float] = {}
            for term in tokens:
                idx = self.vocab.get(term)
                if idx is not None:
                    tf[idx] = tf.get(idx, 0.0) + 1.0

            if not tf:
                doc_terms.append(np.array([], dtype=np.int32))
                doc_vals.append(np.array([], dtype=np.float32))
                continue

            indices = np.array(list(tf.keys()), dtype=np.int32)
            values = np.array(
                [(1.0 + math.log(v)) * float(self.idf[i])
                 for i, v in tf.items()],
                dtype=np.float32,
            )

            # L2-normalize
            norm = float(np.linalg.norm(values))
            if norm > 1e-12:
                values = values / norm

            doc_terms.append(indices)
            doc_vals.append(values)

        self._doc_terms = doc_terms
        self._doc_vals = doc_vals
        self._doc_lemmas = list(lemmas)
        self._doc_goals = list(goals)
        self.N = N

        elapsed = time.time() - t0
        n_entries = sum(len(t) for t in doc_terms)
        mem_mb = (n_entries * 8) / (1024 * 1024)
        print(f"  Built index: {N} docs, vocab={len(self.vocab)}, "
              f"entries={n_entries:,} ({mem_mb:.1f} MB) in {elapsed:.1f}s")
        return self

    def search(
        self, goal_text: str, k: int = 50
    ) -> list[tuple[int, float, str]]:
        """Find k nearest training goals and their correct lemmas.

        Returns:
            List of (doc_index, cosine_similarity, correct_lemma)
            sorted by similarity descending.
        """
        if self.idf is None:
            return []

        tokens = tokenize_goal(goal_text)
        tf: dict[int, float] = {}
        for term in tokens:
            idx = self.vocab.get(term)
            if idx is not None:
                tf[idx] = tf.get(idx, 0.0) + 1.0

        if not tf:
            return []

        # Build query vector (normalized)
        q_indices = np.array(list(tf.keys()), dtype=np.int32)
        q_vals = np.array(
            [(1.0 + math.log(v)) * float(self.idf[i])
             for i, v in tf.items()],
            dtype=np.float32,
        )
        norm = float(np.linalg.norm(q_vals))
        if norm > 1e-12:
            q_vals = q_vals / norm
        else:
            return []

        # Compute cosine similarity with all documents
        N = self.N
        scores = np.zeros(N, dtype=np.float32)

        for i in range(N):
            doc_terms = self._doc_terms[i]
            doc_vals = self._doc_vals[i]
            if len(doc_terms) == 0:
                continue

            # Dot product via sparse intersection
            if len(q_indices) < len(doc_terms):
                s = 0.0
                for j in range(len(q_indices)):
                    qi = q_indices[j]
                    qv = q_vals[j]
                    pos = np.searchsorted(doc_terms, qi)
                    if pos < len(doc_terms) and doc_terms[pos] == qi:
                        s += qv * float(doc_vals[pos])
                scores[i] = s
            else:
                q_map = dict(zip(
                    q_indices.astype(int), q_vals.astype(float)
                ))
                s = 0.0
                for j in range(len(doc_terms)):
                    di = int(doc_terms[j])
                    if di in q_map:
                        s += q_map[di] * float(doc_vals[j])
                scores[i] = s

        # Get top-k indices
        if k >= N:
            top_indices = np.argsort(-scores)
        else:
            top_indices = np.argpartition(-scores, k)[:k]
            top_indices = top_indices[np.argsort(-scores[top_indices])]

        results = []
        for idx in top_indices:
            sim = float(scores[idx])
            if sim > 0:
                results.append((int(idx), sim, self._doc_lemmas[idx]))

        return results[:k]

    def retrieve_lemmas(
        self, goal_text: str, k: int = 50, top_n: int = 30
    ) -> list[tuple[str, float]]:
        """Retrieve and rank lemmas for a goal.

        1. Find k nearest training goals.
        2. Collect their correct lemmas.
        3. Score each lemma: mean_similarity x log(1 + count).
        4. Return top-N by score.
        """
        neighbors = self.search(goal_text, k=k)

        if not neighbors:
            return []

        # Collect lemma stats
        lemma_sims: dict[str, list[float]] = defaultdict(list)
        for _, sim, lemma in neighbors:
            lemma_sims[lemma].append(sim)

        # Score: mean_similarity x log(1 + count)
        scored = []
        for lemma, sims in lemma_sims.items():
            mean_sim = sum(sims) / len(sims)
            count_bonus = math.log(1 + len(sims))
            score = mean_sim * count_bonus
            scored.append((lemma, score))

        scored.sort(key=lambda x: -x[1])

        # Normalize scores to [0, 1] range
        if scored:
            max_s = max(s for _, s in scored)
            if max_s > 0:
                scored = [(name, s / max_s) for name, s in scored]

        return scored[:top_n]


# ---------------------------------------------------------------------------
# Best-First Search (lightweight, no GNN dependency)
# ---------------------------------------------------------------------------


@dataclass(order=True)
class _PrioritizedState:
    priority: float
    depth: int
    tiebreaker: int
    state: ProofState = field(compare=False)
    steps: list = field(compare=False)


@dataclass
class SearchConfig:
    max_depth: int = 20
    max_expansions: int = 1000
    top_k_lemmas: int = 30
    depth_penalty: float = 0.05
    max_verifications: int = 5


class DirectRetrievalSearch:
    """Best-first proof search using TF-IDF retrieved lemmas directly.

    No GNN. No graph. Just TF-IDF-scored lemmas + best-first search + Lean.
    """

    def __init__(
        self,
        scorer,   # callable: goal_text -> list[(lemma, score)]
        proof_checker,
        config: SearchConfig | None = None,
    ):
        self.scorer = scorer
        self.proof_checker = proof_checker
        self.config = config or SearchConfig()
        self._verification_cache: dict[tuple, bool] = {}
        self._tiebreaker: int = 0

    def search(
        self,
        theorem_statement: str,
        verbose: bool = False,
    ) -> tuple[list[Tactic], ProofState | None]:
        """Search for a proof using TF-IDF retrieved and scored lemmas."""
        self._tiebreaker = 0
        self._verification_cache.clear()

        # Get top-K lemma candidates with TF-IDF scores
        scored_lemmas = self.scorer(theorem_statement)

        if verbose:
            print(f"  TF-IDF retrieved {len(scored_lemmas)} lemma candidates")
            if scored_lemmas:
                top5 = scored_lemmas[:5]
                print(f"    Top-5: {[(n, f'{s:.3f}') for n, s in top5]}")

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
        verify_count = 0
        last_unverified: list[Tactic] | None = None
        last_unverified_state: ProofState | None = None

        while heap and expansions < self.config.max_expansions:
            current = heapq.heappop(heap)
            state = current.state
            depth = current.depth

            if state.is_complete:
                if self.proof_checker is not None and state.steps:
                    verify_count += 1
                    if verify_count <= self.config.max_verifications:
                        proof_body = ProofState._render_proof(state.steps)
                        code = wrap_theorem_with_proof(
                            state.theorem_statement, proof_body
                        )
                        check_results = self.proof_checker.check_batch([code])
                        if check_results[0].success:
                            if verbose:
                                print(
                                    f"  OK Verified proof at depth {depth}, "
                                    f"expansions={expansions}, "
                                    f"{time.time() - t_start:.1f}s"
                                )
                            return (state.steps, state)
                        else:
                            state.is_complete = False
                            state.is_dead = True
                            if verbose:
                                err = (
                                    check_results[0].errors[0][:80]
                                    if check_results[0].errors
                                    else "?"
                                )
                                print(f"  X Rejected: {err}")
                            continue
                    else:
                        if last_unverified is None:
                            last_unverified = list(state.steps)
                            last_unverified_state = state
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

            valid_mask = self._maybe_verify(state, candidates)
            if not any(valid_mask):
                valid_mask = [True] * len(candidates)

            for i, (action, score) in enumerate(candidates):
                if i < len(valid_mask) and not valid_mask[i]:
                    continue
                child_state = state.apply_tactic(action)
                child_depth = depth + 1
                priority = -(
                    score / (1.0 + child_depth * self.config.depth_penalty)
                )
                child = _PrioritizedState(
                    priority=priority,
                    depth=child_depth,
                    tiebreaker=self._next_tiebreaker(),
                    state=child_state,
                    steps=state.steps + [action],
                )
                heapq.heappush(heap, child)

            if verbose and expansions % 100 == 0:
                top_msg = (
                    f"top_score={-heap[0].priority:.3f}" if heap else ""
                )
                print(
                    f"  [{expansions}/{self.config.max_expansions}] "
                    f"depth={depth}, heap={len(heap)}, {top_msg}"
                )

        elapsed = time.time() - t_start
        if verbose:
            if heap:
                print(
                    f"  X Budget exhausted: {expansions} expansions, "
                    f"{len(heap)} states remaining, {elapsed:.1f}s"
                )
            else:
                print(
                    f"  X Search space exhausted: {expansions} expansions, "
                    f"{elapsed:.1f}s"
                )

        if last_unverified is not None:
            if verbose:
                print("  -> Returning unverified candidate")
            return (last_unverified, last_unverified_state)

        return ([], None)

    def _generate_actions(
        self,
        state: ProofState,
        scored_lemmas: list[tuple[str, float]],
    ) -> list[tuple[Tactic, float]]:
        """Generate candidate tactics from scored lemmas and structural tactics."""
        candidates: list[tuple[Tactic, float]] = []

        # Lemma-based tactics — scale scores down so structural tactics
        # (scores 0.35-0.80) get explored before TF-IDF lemma applications.
        # Without this, the heap floods with lemma candidates and structural
        # proofs like rw[h];ring are never reached within budget.
        for lemma, score in scored_lemmas[:self.config.top_k_lemmas]:
            # Scale TF-IDF scores to [0.15, 0.35] range — below structural tactics
            scaled = 0.15 + score * 0.20
            candidates.append((Tactic(TacticType.APPLY, lemma=lemma), scaled))

            if any(
                kw in lemma.lower()
                for kw in (
                    "add", "mul", "eq", "comm", "assoc",
                    "zero", "one", "neg", "sub", "div",
                    "ring", "field", "simp",
                )
            ):
                candidates.append(
                    (Tactic(TacticType.REWRITE, lemma=lemma), scaled * 0.9)
                )

        hypotheses = state.hypotheses

        # Exact: close with a hypothesis
        for hyp_name in list(hypotheses.keys())[:5]:
            candidates.append(
                (Tactic(TacticType.EXACT, hypothesis=hyp_name), 0.70)
            )

        # Rewrite using equality hypotheses
        for hyp_name, hyp_type in hypotheses.items():
            if "=" in hyp_type or "<->" in hyp_type:
                candidates.append(
                    (Tactic(TacticType.REWRITE, hypothesis=hyp_name), 0.75)
                )
                break

        # Intro for implications
        if state.goals and ("->" in state.goals[0] or "∀" in state.goals[0]):
            candidates.append(
                (Tactic(TacticType.INTRO, hypothesis="h"), 0.80)
            )

        # Cases for non-equality hypotheses
        for hyp_name, hyp_type in list(hypotheses.items())[:3]:
            skip_ops = ("=", "!=", "<->", "<=", ">=", "<", ">")
            if any(op in hyp_type for op in skip_ops):
                continue
            candidates.append(
                (Tactic(TacticType.CASES, hypothesis=hyp_name), 0.35)
            )
            break

        # Automation tactics
        if state.goals:
            goal = state.goals[0]
            has_implication = "->" in goal or "∀" in goal

            if not has_implication:
                if any(op in goal for op in ("*", "^", "+", "-", "=")):
                    candidates.append((Tactic(TacticType.RING), 0.70))

            if ("/" in goal or "inv" in goal.lower()) and not has_implication:
                nonzero_hyps = [
                    name
                    for name, typ in hypotheses.items()
                    if "!=" in typ or "h" in name.lower()
                ]
                candidates.append(
                    (
                        Tactic(TacticType.FIELD_SIMP, args=nonzero_hyps[:3]),
                        0.70,
                    )
                )

            if (
                any(op in goal for op in ("<=", ">=", "<", ">", "="))
                and not has_implication
            ):
                candidates.append((Tactic(TacticType.LINARITH), 0.70))

            candidates.append((Tactic(TacticType.SIMP), 0.65))

        return candidates

    def _maybe_verify(
        self,
        state: ProofState,
        candidates: list[tuple[Tactic, float]],
    ) -> list[bool]:
        """Verify root-level candidates with Lean proof checker.

        DISABLED for performance: batch-checking 60+ lemma candidates at
        the root level takes 10-50s per theorem. The search will instead
        verify completions individually (capped at max_verifications=5).
        """
        return [True] * len(candidates)

    def _verify_candidates(
        self, state: ProofState, tactics: list[Tactic]
    ) -> list[bool]:
        """Verify tactic applications against Lean."""
        _incomplete_ok = {"rewrite", "intro", "cases", "have", "refine"}

        uncached: list[tuple[int, str, Tactic]] = []
        results: dict[int, bool] = {}

        for i, tactic in enumerate(tactics):
            cache_key = (hash(state.proof_so_far), hash(tactic))
            if cache_key in self._verification_cache:
                results[i] = self._verification_cache[cache_key]
            else:
                uncached.append((i, state.proof_so_far, tactic))

        if uncached and self.proof_checker is not None:
            codes = []
            for _, proof_so_far, tactic in uncached:
                if proof_so_far:
                    proof_body = proof_so_far + "\n  " + tactic.to_lean()
                else:
                    proof_body = tactic.to_lean()
                code = wrap_theorem_with_proof(
                    state.theorem_statement, proof_body
                )
                codes.append(code)

            check_results = self.proof_checker.check_batch(codes)

            for j, (orig_idx, _, tactic) in enumerate(uncached):
                success = check_results[j].success
                if (
                    not success
                    and tactic.tactic_type.value in _incomplete_ok
                ):
                    errors_str = " ".join(
                        check_results[j].errors
                    ).lower()
                    if "unsolved goals" in errors_str:
                        success = True
                results[orig_idx] = success
                cache_key = (
                    hash(state.proof_so_far),
                    hash(tactic),
                )
                self._verification_cache[cache_key] = success

        return [results.get(i, True) for i in range(len(tactics))]

    def _next_tiebreaker(self) -> int:
        self._tiebreaker += 1
        return self._tiebreaker


# ---------------------------------------------------------------------------
# Lemma scorer adapter (wraps TFIDFIndex.retrieve_lemmas)
# ---------------------------------------------------------------------------


class LemmaScorer:
    """Callable that scores lemmas for a goal using TF-IDF retrieval."""

    def __init__(self, index: TFIDFIndex, k: int = 50, top_n: int = 30):
        self.index = index
        self.k = k
        self.top_n = top_n

    def __call__(self, goal_text: str) -> list[tuple[str, float]]:
        # Extract the goal part (after the last ' : ')
        if " : " in goal_text:
            parts = goal_text.rsplit(" : ", 1)
            query = parts[1] if len(parts) > 1 else goal_text
        else:
            query = goal_text
        return self.index.retrieve_lemmas(query, k=self.k, top_n=self.top_n)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


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
        elif s in (
            "ring", "simp", "linarith", "field_simp", "positivity",
            "norm_num", "nlinarith",
        ):
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
    patterns = [
        "rfl", "add_comm", "mul_comm", "ring", "field_simp",
        "linarith", "simp", "intro", "apply", "nlinarith",
    ]
    for p in patterns:
        if p in steps_text:
            return p
    return "other"


def is_lemma_novelty(proof_steps: list[str]) -> bool:
    structural = {
        "simp", "ring", "linarith", "field_simp", "rfl",
        "norm_num", "nlinarith", "positivity", "omega",
        "native_decide",
    }
    has_lemma = False
    for step in proof_steps:
        s = step.strip().lower()
        tactic = s.split()[0] if s else ""
        if tactic not in structural and not s.startswith("exact"):
            has_lemma = True
            break
    lemma_refs = re.findall(
        r'rw\s*\[([^\]]+)\]', " ".join(proof_steps)
    )
    for ref in lemma_refs:
        parts = ref.split(",")
        for p in parts:
            p = p.strip()
            if p not in structural and p not in (
                "h", "h1", "h2", "h3", "h'",
            ):
                has_lemma = True
                break
    return has_lemma


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def save_json(data: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


def run_smoke_test(index: TFIDFIndex, n_samples: int = 100) -> dict:
    """Smoke test: verify retrieval recall on random training goals."""
    print(
        f"\n{'='*60}\n"
        f"SMOKE TEST: Retrieval recall on {n_samples} random training goals\n"
        f"{'='*60}"
    )

    np.random.seed(42)
    sample_indices = np.random.choice(
        index.N, size=min(n_samples, index.N), replace=False
    )

    found_at_rank = []
    found_in_topk = {k: 0 for k in [1, 5, 10, 20, 30, 50]}
    total = 0

    for idx in sample_indices:
        goal = index._doc_goals[idx]
        correct_lemma = index._doc_lemmas[idx]
        total += 1

        results = index.retrieve_lemmas(goal, k=50, top_n=50)
        retrieved_lemmas = [lemma for lemma, _ in results]
        try:
            rank = retrieved_lemmas.index(correct_lemma) + 1
            found_at_rank.append(rank)
            for k in found_in_topk:
                if rank <= k:
                    found_in_topk[k] += 1
        except ValueError:
            pass

    print(f"\n  Samples: {total}")
    n_found = len(found_at_rank)
    print(
        f"  Found correct lemma: {n_found}/{total} "
        f"({n_found/max(1,total)*100:.1f}%)"
    )

    if found_at_rank:
        print(f"  Median rank: {np.median(found_at_rank):.0f}")
        print(f"  Mean rank:   {np.mean(found_at_rank):.1f}")
        print(f"  Min rank:    {np.min(found_at_rank)}")
        print(f"\n  Recall @ K:")
        for k in [1, 5, 10, 20, 30, 50]:
            r = found_in_topk[k] / total
            print(f"    @{k:2d}: {found_in_topk[k]}/{total} ({r:.1%})")

    return {
        "samples": total,
        "found": n_found,
        "recall": n_found / max(1, total),
        "median_rank": (
            float(np.median(found_at_rank)) if found_at_rank else None
        ),
        "mean_rank": (
            float(np.mean(found_at_rank)) if found_at_rank else None
        ),
        "recall_at_k": {
            str(k): found_in_topk[k] / total
            for k in [1, 5, 10, 20, 30, 50]
        },
    }


# ---------------------------------------------------------------------------
# Full Gate 3 evaluation
# ---------------------------------------------------------------------------


def run_gate3_eval(
    index: TFIDFIndex,
    theorems: list[dict],
    checker,  # BatchChecker | None
    output_path: Path,
    search_config: SearchConfig | None = None,
) -> dict:
    """Evaluate direct retrieval on full gate3_v2 benchmark."""
    print(
        f"\n{'='*70}\n"
        f"DIRECT RETRIEVAL: Gate 3 evaluation on gate3_v2 "
        f"({len(theorems)} theorems)\n"
        f"{'='*70}"
    )

    scorer = LemmaScorer(index, k=50, top_n=30)
    config = search_config or SearchConfig()

    bf_search = DirectRetrievalSearch(
        scorer=scorer,
        proof_checker=checker,
        config=config,
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

        t0_per = time.time()
        proof_steps, _final_state = bf_search.search(
            stmt, verbose=(i < 3)
        )
        search_time = time.time() - t0_per

        proof_text = ProofState._render_proof(proof_steps)

        if not proof_steps:
            ok = False
            err = "no proof found"
            failed_reasons["no_proof"] = (
                failed_reasons.get("no_proof", 0) + 1
            )
        elif checker is None:
            ok = True
            err = ""
        else:
            full_code = wrap_theorem_with_proof(stmt, proof_text)
            check_results = checker.check_batch([full_code])
            ok = check_results[0].success
            err = (
                check_results[0].errors[0][:200]
                if check_results[0].errors
                else ""
            )
            if not ok:
                reason_key = f"lean_reject:{err[:50]}"
                failed_reasons[reason_key] = (
                    failed_reasons.get(reason_key, 0) + 1
                )

        steps_str = [s.to_lean() for s in proof_steps[:10]]
        pattern = classify_proof_pattern(steps_str) if ok else "failed"
        lemma_novel = is_lemma_novelty(steps_str) if ok else False

        result = {
            "name": name,
            "era": era,
            "domain": domain,
            "success": ok,
            "error": err,
            "retrieval_steps": steps_str,
            "num_steps": len(proof_steps),
            "ground_truth": ground_truth,
            "search_time_s": round(search_time, 1),
            "pattern": pattern,
            "lemma_novelty": lemma_novel,
        }
        results.append(result)
        if ok:
            passed.append(result)

        status = "OK" if ok else "FAIL"
        eta = (
            (time.time() - t_start) / (i + 1) * (len(theorems) - i - 1)
        )
        print(
            f"  [{i+1:2d}/{len(theorems)}] {status} {name:45s} "
            f"[{pattern:12s}] {search_time:.1f}s  "
            f"ETA: {eta/60:.0f}m  ({len(passed)} passed)"
        )

        if ok and len(proof_steps) > 0:
            print(f"         Proof: {steps_str}")
            if len(proof_steps) >= 2:
                print(f"         ** MULTI-STEP ({len(proof_steps)} steps)")

    elapsed = time.time() - t_start
    n_total = len(theorems)
    n_passed = len(passed)
    rate = n_passed / max(1, n_total)

    multi = [r for r in passed if r["num_steps"] >= 2]
    lemma_novel_list = [r for r in passed if r["lemma_novelty"]]
    structural = [r for r in passed if not r["lemma_novelty"]]

    # Stats
    print(f"\n{'='*70}")
    print("DIRECT RETRIEVAL GATE 3 RESULTS")
    print(f"{'='*70}")
    print(f"  Total:    {n_passed}/{n_total} ({rate:.0%})")
    print(f"  Multi-step: {len(multi)}")
    print(f"  Lemma-novelty: {len(lemma_novel_list)}")
    print(f"  Structural-only: {len(structural)}")
    print(f"  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}m)")

    # Domain breakdown
    domains = Counter(r["domain"] for r in results)
    print(f"\n  By domain:")
    for dom in sorted(domains.keys()):
        dom_total = domains[dom]
        dom_passed = sum(1 for r in passed if r["domain"] == dom)
        dom_ln = sum(
            1 for r in lemma_novel_list if r["domain"] == dom
        )
        dom_ms = sum(1 for r in multi if r["domain"] == dom)
        print(
            f"    {dom:<20} {dom_passed}/{dom_total} "
            f"({dom_passed/max(1,dom_total)*100:.0f}%) "
            f"LN: {dom_ln}  MS: {dom_ms}"
        )

    print(f"\n  Failure reasons:")
    for reason, count in sorted(
        failed_reasons.items(), key=lambda x: -x[1]
    )[:10]:
        print(f"    {reason:<60} {count}")

    # Build output
    out = {
        "task": (
            "DIRECT RETRIEVAL: TF-IDF k-NN proof-step lookup "
            "on full gate3_v2"
        ),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "architecture": (
            "TF-IDF goal encoder + k-NN retrieval + "
            "Best-first search + Lean"
        ),
        "index": {
            "num_pairs": index.N,
            "vocab_size": len(index.vocab),
        },
        "config": {
            "retrieval_k": 50,
            "top_n_lemmas": 30,
            "max_expansions": config.max_expansions,
            "max_depth": config.max_depth,
            "depth_penalty": config.depth_penalty,
        },
        "gate3": {
            "status": "PASS" if n_passed > 0 else "FAIL",
            "total": n_total,
            "passed": n_passed,
            "rate": rate,
            "multi_step": len(multi),
            "lemma_novelty": len(lemma_novel_list),
            "structural_only": len(structural),
            "elapsed_s": elapsed,
            "failed_reasons": dict(failed_reasons),
            "domains": {
                dom: {
                    "total": domains[dom],
                    "passed": sum(
                        1 for r in passed if r["domain"] == dom
                    ),
                    "lemma_novelty": sum(
                        1
                        for r in lemma_novel_list
                        if r["domain"] == dom
                    ),
                    "multi_step": sum(
                        1 for r in multi if r["domain"] == dom
                    ),
                }
                for dom in domains
            },
            "passed_theorems": [
                {
                    "name": r["name"],
                    "domain": r["domain"],
                    "proof": " ".join(r["retrieval_steps"]),
                    "pattern": r["pattern"],
                    "num_steps": r["num_steps"],
                    "lemma_novelty": r["lemma_novelty"],
                }
                for r in passed
            ],
            "multi_step_theorems": [
                {
                    "name": r["name"],
                    "domain": r["domain"],
                    "proof": " ".join(r["retrieval_steps"]),
                    "num_steps": r["num_steps"],
                    "lemma_novelty": r["lemma_novelty"],
                }
                for r in multi
            ],
        },
        "all_results": results,
    }

    save_json(out, output_path)
    print(f"\n  Results saved to: {output_path}")

    g3 = out["gate3"]
    print(
        f"\n  Gate 3: {g3['status']} ({n_passed}/{n_total} proofs, "
        f"{len(multi)} multi-step, {len(lemma_novel_list)} lemma-novelty)"
    )
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Direct Retrieval: TF-IDF k-NN proof-step lookup"
    )
    parser.add_argument(
        "--pairs",
        default="data/raw/proof_step_pairs.jsonl",
        help="Proof-step pairs JSONL",
    )
    parser.add_argument(
        "--theorems",
        default="data/raw/gate3_v2.jsonl",
        help="Gate3 theorems JSONL",
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Run smoke test on training goals",
    )
    parser.add_argument(
        "--smoke-samples", type=int, default=100,
        help="Number of samples for smoke test",
    )
    parser.add_argument(
        "--eval", action="store_true",
        help="Run full gate3_v2 evaluation",
    )
    parser.add_argument(
        "--output",
        default="data/direct_retrieval_gate3.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--vocab-size", type=int, default=8000,
        help="TF-IDF vocabulary size",
    )
    parser.add_argument(
        "--retrieval-k", type=int, default=50,
        help="Number of nearest neighbors for retrieval",
    )
    parser.add_argument(
        "--top-n", type=int, default=30,
        help="Number of top lemmas to feed to search",
    )
    parser.add_argument(
        "--max-expansions", type=int, default=1000,
        help="Max best-first search expansions",
    )
    parser.add_argument(
        "--depth-penalty", type=float, default=0.05,
        help="Depth penalty for search priority",
    )
    parser.add_argument(
        "--no-proof-checker", action="store_true",
        help="Disable proof checker (faster, no Lean verification)",
    )
    args = parser.parse_args()

    if not args.smoke_test and not args.eval:
        print("Specify --smoke-test and/or --eval")
        return 1

    pairs_path = _PROJECT_ROOT / args.pairs
    theorems_path = _PROJECT_ROOT / args.theorems
    output_path = _PROJECT_ROOT / args.output

    # --- Load proof-step pairs ---
    print(f"Loading proof-step pairs from {pairs_path}...")
    t0 = time.time()
    pairs = load_jsonl(pairs_path)
    print(f"  Loaded {len(pairs)} pairs in {time.time() - t0:.1f}s")

    goals = [p["goal"] for p in pairs]
    lemmas = [p["lemma"] for p in pairs]

    # --- Build TF-IDF index ---
    index = TFIDFIndex(vocab_size=args.vocab_size)
    index.fit(goals, lemmas)

    # --- Smoke test ---
    smoke_result = None
    if args.smoke_test:
        smoke_result = run_smoke_test(index, n_samples=args.smoke_samples)

    # --- Gate 3 evaluation ---
    eval_result = None
    if args.eval:
        theorems = load_jsonl(theorems_path)
        print(f"\nLoaded {len(theorems)} gate3_v2 theorems")

        use_pc = not args.no_proof_checker
        checker = (
            BatchChecker(timeout=15, max_workers=4, cache_size=128)
            if use_pc
            else None
        )

        if checker:
            print("Proof checker: Lean 4 + Mathlib4 (Lake project)")

        search_config = SearchConfig(
            max_depth=20,
            max_expansions=args.max_expansions,
            top_k_lemmas=args.top_n,
            depth_penalty=args.depth_penalty,
        )

        eval_result = run_gate3_eval(
            index=index,
            theorems=theorems,
            checker=checker,
            output_path=output_path,
            search_config=search_config,
        )

    # --- Summary ---
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    if smoke_result:
        print(
            f"  Smoke test: {smoke_result['found']}/"
            f"{smoke_result['samples']} "
            f"({smoke_result['recall']:.1%}) recall"
        )
        if smoke_result.get("recall_at_k"):
            print(
                f"    Recall @10: "
                f"{smoke_result['recall_at_k']['10']:.1%}"
            )
            print(
                f"    Recall @30: "
                f"{smoke_result['recall_at_k']['30']:.1%}"
            )

    if eval_result:
        g3 = eval_result["gate3"]
        print(
            f"  Gate 3: {g3['passed']}/{g3['total']} "
            f"({g3['rate']:.0%})"
        )
        print(f"    Multi-step: {g3['multi_step']}")
        print(f"    Lemma-novelty: {g3['lemma_novelty']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
