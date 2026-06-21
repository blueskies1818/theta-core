"""Best-first proof search replacing MCTS (Pivot Step 3).

Uses contrastive embeddings from the Path C dual-encoder for lemma
scoring, and a simple max-priority queue (heapq) to always expand
the most promising proof state first.

Architecture:
  1. Pre-compute lemma embeddings for all available lemmas via the
     contrastive lemma_encoder (CharCNN).
  2. For each proof state, encode the goal via contrastive goal_encoder
     and score all lemma candidates via dot-product similarity.
  3. Push child states onto a max-heap with priority = lemma_score /
     (1 + depth * depth_penalty), favoring high-scoring shallow states.
  4. Continue until proof complete or budget exhausted.

Key difference from MCTS:
  - NO Monte Carlo rollouts (no backpropagation)
  - NO UCB exploration (no visit counts)
  - Pure greedy best-first expansion guided by learned relevance scores
  - Lemma scoring replaces graph-neighborhood lookup
"""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

from src.contrastive.encoder import (
    CharTokenizer,
    ContrastiveConfig,
    ContrastiveDualEncoder,
)
from src.explorer.proof_state import (
    ProofState,
    Tactic,
    TacticType,
    generate_candidate_actions,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class BestFirstConfig:
    """Configuration for best-first proof search."""

    # Maximum proof depth (steps)
    max_depth: int = 20

    # Maximum number of node expansions before giving up
    max_expansions: int = 5000

    # Number of top lemmas to consider per state (from encoder scoring)
    top_k_lemmas: int = 30

    # Depth penalty: priority = score / (1 + depth * depth_penalty)
    depth_penalty: float = 0.05

    # Whether to use the proof checker for validation during expansion.
    # When enabled, each candidate tactic is verified through Lean before
    # a child state is pushed — only valid steps become children.
    use_proof_checker: bool = False

    # Timeout in seconds for verifying individual candidate steps
    verify_timeout: float = 5.0

    # Maximum proof text length before pruning
    max_proof_length: int = 500

    # Device for encoder inference (CPU only for this project)
    device: str = "cpu"

    # Number of CPU threads for PyTorch inference
    num_threads: int = 4

    # Value network integration
    # Weight of value estimate vs lemma score in priority computation.
    # 0.0 = pure lemma score (default behavior), 1.0 = pure value estimate.
    value_weight: float = 0.3

    # Prune states with value estimate below this threshold.
    # None = no pruning, float = prune if value < threshold.
    value_prune_threshold: float | None = 0.1


# ---------------------------------------------------------------------------
# Priority queue wrapper
# ---------------------------------------------------------------------------


@dataclass(order=True)
class _PrioritizedState:
    """State wrapper for heapq priority queue.

    heapq is a min-heap, so we negate priority to get a max-heap
    (most promising state first).

    Fields (in order for tuple comparison):
      priority:  negated score  (lower → popped first)
      depth:     for tiebreaking (shallower first)
      tiebreaker: monotonic counter (guaranteed unique)
    """

    priority: float
    depth: int
    tiebreaker: int
    state: ProofState = field(compare=False)
    steps: list = field(compare=False)


# ---------------------------------------------------------------------------
# Best-first search
# ---------------------------------------------------------------------------


class BestFirstSearch:
    """Best-first proof search using contrastive lemma scoring.

    Optionally integrates a value network for state evaluation.
    When provided, lemma score and value estimate are blended according
    to config.value_weight.

    Usage:
        bf = BestFirstSearch(encoder, tokenizer, lemma_names, lemma_embeddings,
                              config=BestFirstConfig())
        # With value network:
        bf = BestFirstSearch(encoder, tokenizer, lemma_names, lemma_embeddings,
                              config=BestFirstConfig(value_weight=0.3),
                              value_network=vn, goal_embed_fn=encode_goal)
        proof_steps, final_state = bf.search(theorem_statement)
    """

    def __init__(
        self,
        encoder: ContrastiveDualEncoder,
        tokenizer: CharTokenizer,
        lemma_names: list[str],
        lemma_embeddings: torch.Tensor,
        config: BestFirstConfig | None = None,
        proof_checker=None,
        value_network=None,
        goal_embed_fn=None,
    ):
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.lemma_names = lemma_names
        self.lemma_embeddings = lemma_embeddings  # [num_lemmas, hidden_dim]
        self.config = config or BestFirstConfig()
        self.proof_checker = proof_checker

        # Value network (optional)
        self.value_network = value_network
        self.goal_embed_fn = goal_embed_fn  # callable(state) → goal_embedding

        # Set PyTorch threads
        torch.set_num_threads(self.config.num_threads)
        self.encoder.eval()

        # L2-normalize lemma embeddings once (they should already be normalized
        # from the encoder, but ensure it for dot-product scoring).
        self.lemma_embeddings = F.normalize(self.lemma_embeddings, dim=-1)

        # Verification cache: (proof_so_far_hash, tactic_hash) → is_valid
        self._verification_cache: dict[tuple[int, int], bool] = {}

        # Tiebreaker counter for priority queue ordering
        self._tiebreaker: int = 0

    # ------------------------------------------------------------------
    # Main search
    # ------------------------------------------------------------------

    def search(
        self,
        theorem_statement: str,
        verbose: bool = False,
    ) -> tuple[list[Tactic], ProofState | None]:
        """Run best-first search to find a proof.

        Args:
            theorem_statement: The theorem to prove (Lean 4 statement).
            verbose: Print progress information.

        Returns:
            (proof_steps, final_state) — proof steps and terminal state.
            Returns ([], None) if no proof found.
        """
        self._tiebreaker = 0
        # Clear verification cache for new theorem
        self._verification_cache.clear()

        # Create root state
        root_state = ProofState.initial(theorem_statement)
        root = _PrioritizedState(
            priority=-1.0,  # Root has highest priority (most negative → popped first)
            depth=0,
            tiebreaker=self._next_tiebreaker(),
            state=root_state,
            steps=[],
        )

        # Priority queue (min-heap → most negative = highest priority = popped first)
        heap: list[_PrioritizedState] = [root]

        expansions = 0
        t_start = time.time()

        while heap and expansions < self.config.max_expansions:
            # Pop most promising state
            current = heapq.heappop(heap)
            state = current.state
            depth = current.depth

            # Terminal checks
            if state.is_complete:
                # Verify the complete proof with Lean before returning
                if self.config.use_proof_checker and self.proof_checker is not None and state.steps:
                    from src.proof_checker.formats import wrap_theorem_with_proof
                    proof_body = ProofState._render_proof(state.steps)
                    code = wrap_theorem_with_proof(state.theorem_statement, proof_body)
                    check_results = self.proof_checker.check_batch([code])
                    if check_results[0].success:
                        if verbose:
                            print(f"  ✓ Found verified proof at depth {depth}, "
                                  f"expansions={expansions}, {time.time()-t_start:.1f}s")
                        return (state.steps, state)
                    else:
                        # State symbolically complete but Lean disagrees — mark dead
                        state.is_complete = False
                        state.is_dead = True
                        if verbose:
                            err = check_results[0].errors[0][:80] if check_results[0].errors else "?"
                            print(f"  ✗ Symbolic-complete but Lean rejects: {err}")
                        continue
                else:
                    if verbose:
                        print(f"  ✓ Found proof at depth {depth}, "
                              f"expansions={expansions}, {time.time()-t_start:.1f}s")
                    return (state.steps, state)

            if state.is_dead:
                continue

            if depth >= self.config.max_depth:
                continue

            # Proof length guard
            if len(state.proof_so_far) > self.config.max_proof_length:
                continue

            # ---- EXPAND ----
            expansions += 1

            # Score lemmas for current goal and create candidate actions
            scored_lemmas = self._score_lemmas(state)
            candidates = self._generate_actions(state, scored_lemmas)

            if not candidates:
                if verbose and expansions % 100 == 0:
                    print(f"  [{expansions}/{self.config.max_expansions}] "
                          f"depth={depth}, heap={len(heap)}, no candidates")
                continue

            # Optional proof checker verification
            valid_mask = self._maybe_verify(state, candidates)

            # If ALL candidates rejected, fall back to all candidates
            if not any(valid_mask):
                valid_mask = [True] * len(candidates)

            # Create child states and push to heap
            for action, lemma_score, valid in zip(
                candidates, self._last_scores, valid_mask
            ):
                if not valid:
                    continue

                child_state = state.apply_tactic(action)
                child_depth = depth + 1

                # Compute value estimate if value network is available
                value_estimate = 0.5  # neutral default
                if self.value_network is not None and self.config.value_weight > 0.0:
                    try:
                        if self.goal_embed_fn is not None:
                            goal_emb = self.goal_embed_fn(child_state)
                            if goal_emb is not None:
                                value_estimate = float(
                                    self.value_network.predict(goal_emb).item()
                                )
                                value_estimate = max(0.0, min(1.0, value_estimate))
                    except Exception:
                        value_estimate = 0.5

                    # Prune if below threshold
                    if (self.config.value_prune_threshold is not None
                            and value_estimate < self.config.value_prune_threshold):
                        continue

                # Blend lemma score and value estimate
                vw = self.config.value_weight
                blended_score = lemma_score * (1.0 - vw) + value_estimate * vw

                # Priority: blended_score / (1 + depth * penalty)
                # Negate for max-heap behavior
                priority = -(blended_score / (1.0 + child_depth * self.config.depth_penalty))

                child = _PrioritizedState(
                    priority=priority,
                    depth=child_depth,
                    tiebreaker=self._next_tiebreaker(),
                    state=child_state,
                    steps=state.steps + [action],
                )
                heapq.heappush(heap, child)

            if verbose and expansions % 100 == 0:
                print(f"  [{expansions}/{self.config.max_expansions}] "
                      f"depth={depth}, heap={len(heap)}, "
                      f"top_score={-heap[0].priority:.3f}" if heap else "")

        elapsed = time.time() - t_start
        if verbose:
            if heap:
                print(f"  ✗ Budget exhausted: {expansions} expansions, "
                      f"{len(heap)} states remaining, {elapsed:.1f}s")
            else:
                print(f"  ✗ Search space exhausted: {expansions} expansions, "
                      f"{elapsed:.1f}s")

        return ([], None)

    # ------------------------------------------------------------------
    # Lemma scoring
    # ------------------------------------------------------------------

    def _score_lemmas(self, state: ProofState) -> list[tuple[str, float]]:
        """Score all available lemmas for the current goal.

        Uses the contrastive encoder: encodes the goal text, computes
        dot-product similarity with all pre-computed lemma embeddings,
        and returns the top-K lemmas with normalized scores in [0, 1].

        Normalization: raw score = dot(goal_emb, lemma_emb) / temperature.
        Then sigmoid(raw/2) maps to [0,1] range comparable to structural
        tactic scores (0.3-0.6).

        Args:
            state: Current proof state.

        Returns:
            List of (lemma_name, score) tuples, sorted by score descending.
        """
        # Get goal text
        goal_text = state.goals[0] if state.goals else state.theorem_statement

        # Encode goal
        goal_preprocessed = CharTokenizer.preprocess_goal(goal_text)
        goal_ids = self.tokenizer.encode(goal_preprocessed).unsqueeze(0)  # [1, L]
        goal_ids = goal_ids.to(self.lemma_embeddings.device)

        with torch.no_grad():
            goal_emb = self.encoder.encode_goal(goal_ids)  # [1, D]
            goal_emb = F.normalize(goal_emb, dim=-1)

            # Raw dot product: [1, D] @ [D, N] → [1, N] (cosine similarity in [-1, 1])
            raw_scores = (goal_emb @ self.lemma_embeddings.T).squeeze(0)  # [N]

            # Normalize to [0, 1] range using sigmoid(x/2)
            # Raw dot products are typically in [-0.3, 0.3] for most pairs,
            # so x/2 is in [-0.15, 0.15], sigmoid gives ~[0.46, 0.54].
            # Cap at 0.65 so structural tactics (0.55-0.80) get priority.
            scores = torch.sigmoid(raw_scores / 2.0) * 1.2
            scores = torch.clamp(scores, max=0.65)

        # Get top-K
        k = min(self.config.top_k_lemmas, len(self.lemma_names))
        top_scores, top_indices = torch.topk(scores, k)

        # Build result list
        result = []
        for i in range(k):
            idx = top_indices[i].item()
            score = top_scores[i].item()
            result.append((self.lemma_names[idx], score))

        return result

    # ------------------------------------------------------------------
    # Action generation
    # ------------------------------------------------------------------

    def _generate_actions(
        self, state: ProofState, scored_lemmas: list[tuple[str, float]]
    ) -> list[Tactic]:
        """Generate candidate tactics from scored lemmas.

        Uses the top-K scored lemmas to create apply/rewrite/exact actions,
        plus structural tactics (intro, cases, automation) that don't
        depend on lemma selection.

        Args:
            state: Current proof state.
            scored_lemmas: List of (lemma_name, score) from _score_lemmas.

        Returns:
            List of candidate Tactic objects.
        """
        candidates: list[Tactic] = []
        self._last_scores = []

        # For each top lemma, generate apply action
        for lemma, score in scored_lemmas:
            candidates_for_this_lemma = []

            # Apply: use the lemma
            candidates_for_this_lemma.append(
                Tactic(TacticType.APPLY, lemma=lemma)
            )

            # Rewrite: only for equality/algebraic lemmas
            if any(kw in lemma.lower()
                   for kw in ("add", "mul", "eq", "comm", "assoc", "zero", "one",
                               "neg", "sub", "div", "ring", "field", "simp")):
                candidates_for_this_lemma.append(
                    Tactic(TacticType.REWRITE, lemma=lemma)
                )

            # Exact: try exact match
            candidates_for_this_lemma.append(
                Tactic(TacticType.EXACT, lemma=lemma)
            )

            # Add first viable action per lemma (avoid flooding the queue)
            for tactic in candidates_for_this_lemma:
                candidates.append(tactic)
                self._last_scores.append(score)
                break  # Only one action per lemma to keep queue manageable

        # Also generate structural tactics (intro, cases, automation)
        # These don't depend on lemma scoring but are useful for branching
        hypotheses = state.hypotheses

        # Exact: close with hypothesis
        for hyp_name in list(hypotheses.keys())[:5]:
            candidates.append(
                Tactic(TacticType.EXACT, hypothesis=hyp_name)
            )
            self._last_scores.append(0.70)

        # Rewrite using local equality hypotheses
        for hyp_name, hyp_type in hypotheses.items():
            if "=" in hyp_type or "↔" in hyp_type:
                candidates.append(
                    Tactic(TacticType.REWRITE, hypothesis=hyp_name)
                )
                self._last_scores.append(0.75)
                break  # Just one

        # Intro if goal is implication/forall
        if state.goals and ("→" in state.goals[0] or "∀" in state.goals[0]):
            candidates.append(Tactic(TacticType.INTRO, hypothesis="h"))
            self._last_scores.append(0.80)

        # Cases for inductive hypotheses
        for hyp_name, hyp_type in list(hypotheses.items())[:3]:
            if any(op in hyp_type
                   for op in ("=", "≠", "↔", "≤", "≥", "<", ">")):
                continue
            candidates.append(Tactic(TacticType.CASES, hypothesis=hyp_name))
            self._last_scores.append(0.35)
            break

        # Automation tactics — these can close arithmetic goals without lemmas
        if state.goals:
            goal = state.goals[0]
            has_implication = "→" in goal or "∀" in goal

            # ring/nlinarith for polynomial/arithmetic goals
            if not has_implication:
                if any(op in goal for op in ("*", "^", "+", "-", "=")):
                    candidates.append(Tactic(TacticType.RING))
                    self._last_scores.append(0.70)

            if ("/" in goal or "⁻¹" in goal) and not has_implication:
                candidates.append(Tactic(TacticType.FIELD_SIMP))
                self._last_scores.append(0.70)

            if any(op in goal for op in ("≤", "≥", "<", ">", "=")) and not has_implication:
                candidates.append(Tactic(TacticType.LINARITH))
                self._last_scores.append(0.70)

            # simp: general simplification
            candidates.append(Tactic(TacticType.SIMP))
            self._last_scores.append(0.65)

        # Limit candidates
        max_actions = self.config.top_k_lemmas * 2 + 10
        if len(candidates) > max_actions:
            # Keep top-scored (they appear first in candidates)
            candidates = candidates[:max_actions]
            self._last_scores = self._last_scores[:max_actions]

        return candidates

    # ------------------------------------------------------------------
    # Proof checker verification (optional)
    # ------------------------------------------------------------------

    def _maybe_verify(
        self, state: ProofState, candidates: list[Tactic]
    ) -> list[bool]:
        """Optionally verify candidates through Lean proof checker.

        Only verifies lemma-based actions at the root (where false positives
        hurt the most). Structural/automation tactics always pass.

        Args:
            state: Current proof state.
            candidates: Candidate tactics.

        Returns:
            Boolean mask: True = valid, False = rejected.
        """
        is_root = len(state.steps) == 0
        _lemma_tactics = {"apply", "rewrite", "exact"}

        if (
            is_root
            and self.config.use_proof_checker
            and self.proof_checker is not None
        ):
            # Identify which candidates to verify
            verify_indices = []
            for i, c in enumerate(candidates):
                if c.tactic_type.value in _lemma_tactics and (
                    c.lemma is not None or c.hypothesis is not None
                ):
                    verify_indices.append(i)

            if verify_indices:
                verify_results = self._verify_candidates(
                    state,
                    [candidates[i] for i in verify_indices],
                )
                # Build full mask
                valid_mask = [True] * len(candidates)
                for j, idx in enumerate(verify_indices):
                    valid_mask[idx] = verify_results[j]
                return valid_mask

        return [True] * len(candidates)

    def _verify_candidates(
        self, state: ProofState, tactics: list[Tactic]
    ) -> list[bool]:
        """Batch-verify tactics through Lean proof checker.

        For rewrite/intro tactics, accepts steps that leave unsolved goals
        (the step is valid but doesn't close the proof). Only rejects steps
        with actual errors (type mismatch, unknown identifier).

        Cached by (proof_so_far_hash, tactic_hash).
        """
        from src.proof_checker.formats import wrap_theorem_with_proof

        # Tactics that are valid even with unsolved goals
        _incomplete_ok = {"rewrite", "intro", "cases", "have", "refine"}

        # Separate cached from uncached
        uncached_indices: list[int] = []
        uncached_codes: list[str] = []
        results: dict[int, bool] = {}

        for i, tactic in enumerate(tactics):
            cache_key = (hash(state.proof_so_far), hash(tactic))
            if cache_key in self._verification_cache:
                results[i] = self._verification_cache[cache_key]
            else:
                uncached_indices.append(i)
                if state.proof_so_far:
                    proof_body = state.proof_so_far + "\n  " + tactic.to_lean()
                else:
                    proof_body = tactic.to_lean()
                code = wrap_theorem_with_proof(state.theorem_statement, proof_body)
                uncached_codes.append(code)

        if uncached_codes:
            # Verify one at a time to avoid BatchChecker process pool crashes
            for j, idx in enumerate(uncached_indices):
                single_code = uncached_codes[j]
                check_results = self.proof_checker.check_batch([single_code])
                cr = check_results[0]
                tactic = tactics[idx]
                if cr.success:
                    is_valid = True
                elif tactic.tactic_type.value in _incomplete_ok:
                    # Accept if error is just "unsolved goals" (step is valid)
                    error_text = " ".join(cr.errors) if cr.errors else ""
                    is_valid = "unsolved goals" in error_text.lower()
                else:
                    is_valid = False
                cache_key = (hash(state.proof_so_far), hash(tactics[idx]))
                self._verification_cache[cache_key] = is_valid
                results[idx] = is_valid

        return [results[i] for i in range(len(tactics))]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _next_tiebreaker(self) -> int:
        self._tiebreaker += 1
        return self._tiebreaker
