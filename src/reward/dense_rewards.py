"""Step-level dense reward system for proof search (Path A).

Replaces binary trajectory reward with per-step reward shaping:

  (a) Step validity from Lean verification: +0.1 per valid tactic step.
      Invalid steps get 0.0 (but the search path continues anyway).
  (b) Goal proximity from embedding distance: reward proportional to
      how different the current goal embedding is from the initial goal
      embedding (proxy for progress toward a simpler goal).
  (c) Proof completion bonus: +1.0 when the full proof chain is
      verified correct by Lean (the trajectory-level success signal).

These dense rewards are computed DURING best-first search and can be
used to (i) guide search priorities, (ii) weight contrastive training
samples, or (iii) compute advantages for policy-gradient updates.

Usage:
    from src.reward.dense_rewards import DenseRewardConfig, DenseRewardTracker

    config = DenseRewardConfig(step_validity=0.1, completion_bonus=1.0)
    tracker = DenseRewardTracker(config, encoder, tokenizer, theorem_statement)
    # ... during search, call tracker.record_step() and tracker.record_completion()
    trajectory = tracker.to_trajectory()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from src.contrastive.encoder import ContrastiveDualEncoder, CharTokenizer


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DenseRewardConfig:
    """Configuration for step-level dense reward computation.

    All weights default to values calibrated for the Path A experiment:
    - step_validity dominates early search (incentivizes valid steps)
    - goal_proximity provides shaping signal between successful steps
    - completion_bonus is the ultimate success signal
    """

    # (a) Step validity — per-step Lean verification reward
    step_validity: float = 0.1
    """Reward for each Lean-verified valid tactic step."""

    # (b) Goal proximity — embedding-distance-based progress reward
    goal_proximity_enabled: bool = True
    """Whether to compute goal proximity reward (requires encoder)."""
    goal_proximity_weight: float = 0.2
    """Multiplier for goal proximity reward in [0, goal_proximity_weight]."""
    goal_proximity_temperature: float = 0.5
    """Temperature for proximity sigmoid: higher → smoother gradients."""

    # (c) Proof completion bonus
    completion_bonus: float = 1.0
    """Bonus added when the full proof is Lean-verified correct."""

    # Search limits
    max_steps: int = 20
    """Maximum number of proof steps before reward saturates."""


# ---------------------------------------------------------------------------
# Step reward record
# ---------------------------------------------------------------------------

@dataclass
class StepReward:
    """Recorded reward components for a single proof step.

    All fields are scalars (Python floats). step_index is 0-based.
    """

    step_index: int
    """0-based index of this step in the proof trajectory."""

    tactic: str
    """Lean tactic string applied at this step (e.g., 'apply mul_comm')."""

    is_valid: bool
    """Whether this step passed Lean verification."""

    step_validity: float = 0.0
    """Reward component (a): +config.step_validity if is_valid, else 0.0."""

    goal_proximity: float = 0.0
    """Reward component (b): embedding-distance-based progress signal."""

    cumulative: float = 0.0
    """Cumulative reward up to and including this step."""


@dataclass
class DenseTrajectory:
    """Complete dense reward trajectory for one proof attempt.

    Includes per-step records, total reward, and metadata about the
    proof attempt outcome.
    """

    theorem_name: str = ""
    theorem_statement: str = ""
    steps: list[StepReward] = field(default_factory=list)
    completion_bonus: float = 0.0
    proof_success: bool = False
    total_reward: float = 0.0
    num_valid_steps: int = 0
    num_invalid_steps: int = 0


# ---------------------------------------------------------------------------
# Dense reward tracker
# ---------------------------------------------------------------------------


class DenseRewardTracker:
    """Tracks step-level dense rewards during best-first proof search.

    The tracker is stateful per proof attempt. It caches the initial goal
    embedding and computes step-by-step rewards as the search progresses.

    Usage inside BestFirstSearch or similar search loop:

        tracker = DenseRewardTracker(config, encoder, tokenizer, theorem_statement)
        # For each expansion:
        tracker.record_step(tactic_str, is_valid=True, current_goal_text=goal)
        # When proof completes:
        tracker.record_completion(success=True, proof_steps=[...])
        # Retrieve result:
        traj = tracker.to_trajectory()
    """

    def __init__(
        self,
        config: DenseRewardConfig,
        encoder: "ContrastiveDualEncoder | None" = None,
        tokenizer: "CharTokenizer | None" = None,
        theorem_statement: str = "",
        theorem_name: str = "",
    ):
        self.config = config
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.theorem_name = theorem_name
        self.theorem_statement = theorem_statement

        # ---- Step records -------------------------------------------------
        self._steps: list[StepReward] = []
        self._current_step_index: int = 0
        self._cumulative: float = 0.0

        # ---- Goal proximity cache -----------------------------------------
        self._initial_goal_emb: torch.Tensor | None = None
        if (
            config.goal_proximity_enabled
            and encoder is not None
            and tokenizer is not None
            and theorem_statement
        ):
            self._initial_goal_emb = self._encode_goal(theorem_statement)

        # ---- Completion state ---------------------------------------------
        self._completed: bool = False
        self._completion_bonus_applied: float = 0.0
        self._proof_success: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_step(
        self,
        tactic: str,
        is_valid: bool,
        current_goal_text: str = "",
    ) -> float:
        """Record a single proof step and return the step's total reward.

        Args:
            tactic: The Lean tactic string applied (e.g., 'apply mul_comm').
            is_valid: Whether Lean verified this step as valid.
            current_goal_text: The goal text AFTER applying this tactic.
                               Empty string if unknown.

        Returns:
            The total reward for this step (step_validity + goal_proximity).
            This is ADDED to the cumulative tracker state.
        """
        if self._completed:
            raise RuntimeError("DenseRewardTracker: record_step called after completion")

        idx = self._current_step_index
        self._current_step_index += 1

        # (a) Step validity
        validity_reward = self.config.step_validity if is_valid else 0.0

        # (b) Goal proximity
        proximity_reward = self._compute_proximity(current_goal_text)

        step_total = validity_reward + proximity_reward
        self._cumulative += step_total

        step = StepReward(
            step_index=idx,
            tactic=tactic,
            is_valid=is_valid,
            step_validity=validity_reward,
            goal_proximity=proximity_reward,
            cumulative=self._cumulative,
        )
        self._steps.append(step)

        return step_total

    def record_completion(
        self,
        success: bool,
        proof_steps: list[str] | None = None,
    ) -> float:
        """Record the proof completion outcome and return bonus.

        Must be called AFTER the final step has been recorded via
        record_step(). Applies the completion bonus if the proof is
        Lean-verified correct.

        Args:
            success: Whether Lean verified the complete proof.
            proof_steps: Optional list of tactic strings for logging.

        Returns:
            The completion bonus (config.completion_bonus or 0.0).
        """
        self._completed = True
        self._proof_success = success

        if success:
            self._completion_bonus_applied = self.config.completion_bonus
            self._cumulative += self._completion_bonus_applied
        else:
            self._completion_bonus_applied = 0.0

        return self._completion_bonus_applied

    def to_trajectory(self) -> DenseTrajectory:
        """Export the full dense reward trajectory.

        Returns:
            DenseTrajectory with per-step records and aggregated stats.
        """
        if not self._completed:
            # Auto-complete as failure if not explicitly completed
            self.record_completion(success=False)

        valid_count = sum(1 for s in self._steps if s.is_valid)
        invalid_count = len(self._steps) - valid_count

        return DenseTrajectory(
            theorem_name=self.theorem_name,
            theorem_statement=self.theorem_statement,
            steps=list(self._steps),
            completion_bonus=self._completion_bonus_applied,
            proof_success=self._proof_success,
            total_reward=self._cumulative,
            num_valid_steps=valid_count,
            num_invalid_steps=invalid_count,
        )

    @property
    def cumulative_reward(self) -> float:
        """Current cumulative reward (excluding unapplied completion bonus)."""
        return self._cumulative

    @property
    def num_steps(self) -> int:
        """Number of steps recorded so far."""
        return len(self._steps)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_goal(self, goal_text: str) -> torch.Tensor | None:
        """Encode a goal text string into a normalized embedding vector.

        Returns [D] float tensor (L2-normalized, on CPU), or None if
        encoder/tokenizer are unavailable.
        """
        if self.tokenizer is None or self.encoder is None:
            return None

        from src.contrastive.encoder import CharTokenizer

        # Preprocess just like the encoder does for search scoring
        preprocessed = CharTokenizer.preprocess_goal(goal_text)
        char_ids = self.tokenizer.encode(preprocessed).unsqueeze(0)  # [1, max_len]

        with torch.no_grad():
            emb = self.encoder.encode_goal(char_ids)  # [1, D]
            emb = F.normalize(emb, dim=-1)
        return emb.squeeze(0)  # [D]

    def _compute_proximity(self, current_goal_text: str) -> float:
        """Compute goal proximity reward from embedding distance.

        Measures how different the current goal embedding is from the
        initial goal embedding. As the proof progresses and the goal
        simplifies, the embedding should become more different →
        cosine distance increases → reward increases.

        proximity = (1 - cos_sim(initial_goal_emb, current_goal_emb))
                    * goal_proximity_weight

        If proximity is disabled or encoder/text unavailable, returns 0.0.

        Returns:
            Scalar float in [0, goal_proximity_weight].
        """
        if not self.config.goal_proximity_enabled:
            return 0.0

        if self._initial_goal_emb is None:
            return 0.0

        if not current_goal_text:
            return 0.0

        try:
            current_emb = self._encode_goal(current_goal_text)
        except Exception:
            return 0.0

        if current_emb is None:
            return 0.0

        # Cosine similarity in [-1, 1]
        cos_sim = torch.dot(self._initial_goal_emb, current_emb).item()

        # Clamp to [-1, 1] for numerical safety
        cos_sim = max(-1.0, min(1.0, cos_sim))

        # Cosine distance = 1 - cos_sim  → range [0, 2]
        # But since both are normalized embeddings from the same encoder,
        # typical cos_sim is in [0.5, 1.0] → distance in [0, 0.5]
        distance = 1.0 - cos_sim

        return self.config.goal_proximity_weight * distance


# ---------------------------------------------------------------------------
# Dense reward best-first search wrapper
# ---------------------------------------------------------------------------


class DenseRewardBestFirstSearch:
    """Best-first proof search with step-level dense reward tracking.

    Wraps a standard BestFirstSearch instance and integrates
    DenseRewardTracker for per-step reward computation during search.

    The search proceeds exactly like the standard best-first algorithm,
    but each expansion records step validity, goal proximity, and
    completion rewards.

    Usage:
        from src.reward.dense_rewards import (
            DenseRewardBestFirstSearch, DenseRewardConfig,
        )
        from src.explorer.best_first_search import BestFirstSearch

        # Standard setup...
        bf_search = BestFirstSearch(encoder, tokenizer, lemma_names,
                                     lemma_embeddings, bf_config,
                                     proof_checker=checker)

        # Wrap with dense rewards
        dense_config = DenseRewardConfig()
        dense_search = DenseRewardBestFirstSearch(
            bf_search, dense_config,
            encoder=encoder, tokenizer=tokenizer,
        )

        proof_steps, final_state, trajectory = dense_search.search(
            theorem_statement, theorem_name="my_theorem"
        )
    """

    def __init__(
        self,
        bf_search,  # type: ignore — BestFirstSearch
        reward_config: DenseRewardConfig,
        encoder: "ContrastiveDualEncoder | None" = None,
        tokenizer: "CharTokenizer | None" = None,
    ):
        self.bf_search = bf_search
        self.reward_config = reward_config
        self.encoder = encoder
        self.tokenizer = tokenizer

    def search(
        self,
        theorem_statement: str,
        theorem_name: str = "",
        verbose: bool = False,
    ) -> tuple[list, any, DenseRewardTracker]:
        """Run best-first search with dense reward tracking.

        Returns the TRACKER (not trajectory) so callers can defer
        completion recording until after post-hoc Lean verification.

        Args:
            theorem_statement: The theorem to prove.
            theorem_name: Name for logging/reporting.
            verbose: Print progress.

        Returns:
            (proof_steps, final_state, tracker)
            - proof_steps: list of Tactic objects (empty if failed)
            - final_state: ProofState or None
            - tracker: DenseRewardTracker (caller MUST call
              tracker.record_completion() after post-hoc verification,
              then tracker.to_trajectory() to get the DenseTrajectory)
        """
        from src.explorer.proof_state import Tactic
        from src.proof_checker.formats import wrap_theorem_with_proof

        # Create tracker for this attempt
        tracker = DenseRewardTracker(
            self.reward_config,
            encoder=self.encoder,
            tokenizer=self.tokenizer,
            theorem_statement=theorem_statement,
            theorem_name=theorem_name,
        )

        # Use whatever proof checker setting is configured (don't auto-enable)
        self.bf_search._tiebreaker = 0
        self.bf_search._verification_cache.clear()

        # Run the search
        proof_steps, final_state = self._search_with_tracking(
            theorem_statement, tracker, verbose
        )

        return proof_steps, final_state, tracker

    def _search_with_tracking(
        self,
        theorem_statement: str,
        tracker: DenseRewardTracker,
        verbose: bool = False,
    ):
        """Internal search that records dense rewards at each step.

        This mirrors BestFirstSearch.search() but injects reward tracking
        at each expansion point.
        """
        import heapq
        import time

        from src.explorer.proof_state import ProofState, Tactic
        from src.explorer.best_first_search import _PrioritizedState

        # Create root state
        root_state = ProofState.initial(theorem_statement)
        root = _PrioritizedState(
            priority=-1.0,
            depth=0,
            tiebreaker=self.bf_search._next_tiebreaker(),
            state=root_state,
            steps=[],
        )

        heap = [root]
        expansions = 0
        t_start = time.time()

        while heap and expansions < self.bf_search.config.max_expansions:
            current = heapq.heappop(heap)
            state = current.state
            depth = current.depth

            # Terminal check
            if state.is_complete:
                if (
                    self.bf_search.config.use_proof_checker
                    and self.bf_search.proof_checker is not None
                    and state.steps
                ):
                    from src.proof_checker.formats import wrap_theorem_with_proof

                    proof_body = ProofState._render_proof(state.steps)
                    code = wrap_theorem_with_proof(
                        state.theorem_statement, proof_body
                    )
                    check_results = self.bf_search.proof_checker.check_batch([code])
                    if check_results[0].success:
                        if verbose:
                            elapsed = time.time() - t_start
                            print(
                                f"  [dense] Proof found at depth {depth}, "
                                f"reward={tracker.cumulative_reward:.3f}, "
                                f"{elapsed:.1f}s"
                            )
                        return (state.steps, state)
                    else:
                        state.is_complete = False
                        state.is_dead = True
                        # Record failed completion attempt as invalid step
                        last_tactic = (
                            state.steps[-1].to_lean() if state.steps else "unknown"
                        )
                        tracker.record_step(
                            last_tactic,
                            is_valid=False,
                            current_goal_text=state.goals[0] if state.goals else "",
                        )
                        continue
                else:
                    return (state.steps, state)

            if state.is_dead:
                continue

            if depth >= self.bf_search.config.max_depth:
                continue

            if len(state.proof_so_far) > self.bf_search.config.max_proof_length:
                continue

            # Expand
            expansions += 1
            scored_lemmas = self.bf_search._score_lemmas(state)
            candidates = self.bf_search._generate_actions(state, scored_lemmas)

            if not candidates:
                continue

            # Verify candidates
            valid_mask: list[bool] = list(self.bf_search._maybe_verify(state, candidates))
            if not any(valid_mask):
                valid_mask = [True] * len(candidates)

            # Push children and record step rewards for the best child
            best_child = None
            best_priority = float("inf")  # min-heap: lower = better

            for action, lemma_score, valid in zip(
                candidates, self.bf_search._last_scores, valid_mask
            ):
                if not valid:
                    continue

                child_state = state.apply_tactic(action)
                child_depth = depth + 1
                priority = -(
                    lemma_score
                    / (1.0 + child_depth * self.bf_search.config.depth_penalty)
                )

                child = _PrioritizedState(
                    priority=priority,
                    depth=child_depth,
                    tiebreaker=self.bf_search._next_tiebreaker(),
                    state=child_state,
                    steps=state.steps + [action],
                )
                heapq.heappush(heap, child)

                if priority < best_priority:
                    best_priority = priority
                    best_child = child

            # Record step reward for the most promising child
            if best_child is not None:
                tactic_str = best_child.steps[-1].to_lean() if best_child.steps else ""
                current_goal = (
                    best_child.state.goals[0] if best_child.state.goals else ""
                )
                # The child was generated from a candidate; if valid_mask has
                # at least one True, the best child was a valid step
                any_valid = any(valid_mask)
                tracker.record_step(
                    tactic_str,
                    is_valid=any_valid,
                    current_goal_text=current_goal,
                )

            if verbose and expansions % 200 == 0:
                elapsed = time.time() - t_start
                top_score = -heap[0].priority if heap else 0.0
                print(
                    f"  [dense {expansions}/{self.bf_search.config.max_expansions}] "
                    f"depth={depth}, heap={len(heap)}, top_score={top_score:.3f}, "
                    f"reward={tracker.cumulative_reward:.3f}, {elapsed:.1f}s"
                )

        elapsed = time.time() - t_start
        if verbose:
            if heap:
                print(
                    f"  [dense] Budget exhausted: {expansions} expansions, "
                    f"{len(heap)} remaining, reward={tracker.cumulative_reward:.3f}, "
                    f"{elapsed:.1f}s"
                )
            else:
                print(
                    f"  [dense] Space exhausted: {expansions} expansions, "
                    f"reward={tracker.cumulative_reward:.3f}, {elapsed:.1f}s"
                )

        return ([], None)


# ---------------------------------------------------------------------------
# Trajectory summary helpers
# ---------------------------------------------------------------------------


def summarize_trajectories(
    trajectories: list[DenseTrajectory],
) -> dict:
    """Compute aggregate statistics over a list of dense reward trajectories.

    Args:
        trajectories: List of DenseTrajectory objects (one per theorem).

    Returns:
        Dict with keys: num_theorems, num_proved, proof_rate,
        mean_total_reward, mean_completion_bonus, mean_validity_per_step,
        mean_proximity_per_step, mean_steps, total_valid_steps,
        total_invalid_steps, multi_step_proofs.
    """
    if not trajectories:
        return {"num_theorems": 0}

    proved = [t for t in trajectories if t.proof_success]
    multi_step = [t for t in proved if t.num_valid_steps >= 2]
    all_steps = [s for t in trajectories for s in t.steps]

    return {
        "num_theorems": len(trajectories),
        "num_proved": len(proved),
        "proof_rate": len(proved) / max(1, len(trajectories)),
        "multi_step_proofs": len(multi_step),
        "multi_step_rate": len(multi_step) / max(1, len(trajectories)),
        "mean_total_reward": (
            sum(t.total_reward for t in trajectories) / len(trajectories)
        ),
        "mean_completion_bonus": (
            sum(t.completion_bonus for t in proved) / max(1, len(proved))
        ),
        "mean_validity_per_step": (
            sum(s.step_validity for s in all_steps) / max(1, len(all_steps))
        ),
        "mean_proximity_per_step": (
            sum(s.goal_proximity for s in all_steps) / max(1, len(all_steps))
        ),
        "mean_steps": (
            sum(t.num_valid_steps + t.num_invalid_steps for t in trajectories)
            / len(trajectories)
        ),
        "total_valid_steps": sum(t.num_valid_steps for t in trajectories),
        "total_invalid_steps": sum(t.num_invalid_steps for t in trajectories),
    }
