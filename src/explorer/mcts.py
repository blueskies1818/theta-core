"""Monte Carlo Tree Search for proof exploration (Phase 2.3).

The MCTS replaces LLM generation from Phase 1. Instead of sampling tokens,
it systematically searches the space of possible proof steps, using the
GNN to evaluate which lemmas are relevant at each state.

Architecture (AlphaGo Zero analog):
- Selection: UCB traversal from root → leaf
- Expansion: GNN scores candidate actions, creates children
- Evaluation: GNN value head estimates proof success probability
- Backpropagation: Update visit counts and values up the tree

At the end of search, returns the best proof path found, which is
validated by the Lean proof checker (Phase 2.4).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from src.explorer.proof_state import (
    ProofState,
    Tactic,
    generate_candidate_actions,
)


@dataclass
class MCTSNode:
    """A node in the Monte Carlo search tree."""

    state: ProofState
    parent: "MCTSNode | None" = None
    incoming_action: Tactic | None = None  # Action that led to this node

    # Children: action → node
    children: dict[Tactic, "MCTSNode"] = field(default_factory=dict)

    # MCTS statistics
    visit_count: int = 0
    total_value: float = 0.0

    # Prior probability from GNN policy (for PUCT)
    # This is a detached float — used for MCTS tree search.
    prior: float = 1.0

    # Differentiable logits from the GNN for each child action.
    # Shape [num_children] — connected to the GNN computation graph.
    # None when actions were scored without the GNN.
    child_logits: "torch.Tensor | None" = None

    # Order of actions matching child_logits indices.
    # child_logits[i] corresponds to _child_action_order[i].
    _child_action_order: list = field(default_factory=list)

    # Value estimate from GNN (cached)
    value_estimate: float = 0.0

    # Whether this node has been evaluated by the proof checker
    verified: bool = False

    @property
    def mean_value(self) -> float:
        """Average value from MCTS rollouts through this node."""
        if self.visit_count == 0:
            return self.value_estimate
        return self.total_value / self.visit_count

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def is_expanded(self) -> bool:
        return len(self.children) > 0

    @property
    def is_terminal(self) -> bool:
        return self.state.is_complete or self.state.is_dead

    def ucb_score(self, parent_visit_count: int, c_puct: float = 1.4) -> float:
        """PUCT score for this node (used during selection).

        UCB = Q(s,a) + c_puct * P(s,a) * √(N_parent) / (1 + N(s,a))

        Where:
        - Q(s,a) = mean value from rollouts
        - P(s,a) = prior probability from GNN policy
        - N_parent = visit count of parent
        - N(s,a) = visit count of this node
        """
        q_value = self.mean_value
        exploration = c_puct * self.prior * math.sqrt(parent_visit_count) / (
            1 + self.visit_count
        )
        return q_value + exploration

    def best_child(self, c_puct: float = 1.4) -> "MCTSNode":
        """Return the child with highest UCB score."""
        if not self.children:
            raise ValueError("Cannot select best_child from leaf node")
        return max(
            self.children.values(),
            key=lambda child: child.ucb_score(self.visit_count, c_puct),
        )

    def most_visited_child(self) -> "MCTSNode":
        """Return the child with most visits (for final action selection)."""
        if not self.children:
            raise ValueError("No children available")
        return max(self.children.values(), key=lambda child: child.visit_count)


@dataclass
class MCTSConfig:
    """Configuration for MCTS proof search."""

    # Number of MCTS simulations per proof
    num_simulations: int = 1000

    # PUCT exploration constant
    c_puct: float = 1.4

    # Maximum proof depth (steps)
    max_depth: int = 20

    # Maximum branching factor (progressive widening)
    max_actions_per_node: int = 50

    # Temperature for action selection after search
    # 0.0 = deterministic (most visited), 1.0 = proportional to visits
    temperature: float = 0.5

    # Dirichlet noise concentration for root (exploration)
    dirichlet_alpha: float = 0.3
    dirichlet_weight: float = 0.45  # ε=0.45 provides enough noise to create gradient diversity at 500+ sims

    # Number of top lemmas to consider from GNN
    top_k_lemmas: int = 30

    # Whether to use the GNN for state evaluation
    use_gnn: bool = True

    # Whether to use the proof checker for validation during node expansion.
    # When enabled, each candidate tactic is verified through Lean before
    # a child node is created — only valid steps become children.
    use_proof_checker: bool = False

    # Timeout in seconds for verifying individual candidate steps
    verify_timeout: float = 5.0

    # Maximum proof text length before pruning
    max_proof_length: int = 500

    # Heuristic scale factor for cold-start proof patterns.
    # 1.0 = full heuristics, 0.0 = pure GNN (no heuristics).
    # Anneal from 1.0 → 0.0 during training to let the GNN take over.
    heuristic_scale: float = 1.0


class MCTS:
    """Monte Carlo Tree Search for proof exploration.

    Given a theorem statement, searches for a valid proof by:
    1. Building a search tree over proof states
    2. Using the GNN to guide exploration (which lemmas to try)
    3. Evaluating states with the GNN value head
    4. Returning the most promising proof path

    Usage:
        mcts = MCTS(gnn_encoder, graph, config)
        best_proof = mcts.search(theorem_statement)
    """

    def __init__(
        self,
        gnn_encoder: "GNNEncoder | None" = None,
        dependency_graph: "DependencyGraph | None" = None,
        config: MCTSConfig | None = None,
        proof_checker: "BatchChecker | None" = None,
    ):
        self.gnn = gnn_encoder
        self.graph = dependency_graph
        self.config = config or MCTSConfig()
        self.proof_checker = proof_checker

        # Cached GNN embeddings (populated before search)
        self._node_embeddings: torch.Tensor | None = None
        self._lemma_names: list[str] = []
        self._lemma_to_idx: dict[str, int] = {}

        # Approximate embeddings for built-in lemmas not in the graph
        self._builtin_embeddings: dict[str, torch.Tensor] = {}

        # Keyword → lemma index map for fast goal embedding
        self._kw_lemmas_map: dict[str, list[int]] = {}

        # Node reuse (transposition table)
        self._state_cache: dict[int, MCTSNode] = {}

        # Verification cache: (proof_so_far_hash, tactic_hash) → is_valid
        self._verification_cache: dict[tuple[int, int], bool] = {}

    # ------------------------------------------------------------------
    # Main search
    # ------------------------------------------------------------------

    def search(
        self,
        theorem_statement: str,
        node_embeddings: torch.Tensor | None = None,
        verbose: bool = False,
    ) -> tuple[list[Tactic], MCTSNode]:
        """Run MCTS to find a proof for the given theorem.

        Args:
            theorem_statement: The theorem to prove (Lean 4 statement).
            node_embeddings: Pre-computed GNN embeddings for all graph nodes.
                If None and GNN is available, uses the graph for scoring.
            verbose: Print progress information.

        Returns:
            (best_proof_steps, root_node) — the best proof and search tree.
        """
        if node_embeddings is not None:
            self._node_embeddings = node_embeddings

        # Create root
        root_state = ProofState.initial(theorem_statement)
        root = MCTSNode(state=root_state)

        # Identify relevant lemmas for this theorem from the graph
        available_lemmas = self._get_relevant_lemmas(theorem_statement)

        # Track whether root has been expanded (for Dirichlet noise)
        _root_expanded = False

        for sim in range(self.config.num_simulations):
            node = root
            depth = 0

            # ---- SELECT ----
            # Traverse to a leaf node using UCB
            path = [node]
            while node.is_expanded and not node.is_terminal and depth < self.config.max_depth:
                node = node.best_child(self.config.c_puct)
                path.append(node)
                depth += 1

            # ---- EXPAND ----
            # If the leaf isn't terminal, expand it
            if not node.is_terminal and depth < self.config.max_depth:
                self._expand(node, available_lemmas)

                # Apply Dirichlet noise to root priors after first expansion
                # (AlphaGo Zero: noisy_prior = (1-eps)*prior + eps*Dirichlet(alpha))
                # This ensures different MCTS searches explore different paths even
                # with identical heuristic/GNN guidance.
                if node is root and not _root_expanded:
                    _root_expanded = True
                    if root.children:
                        n_children = len(root.children)
                        alpha = self.config.dirichlet_alpha
                        eps = self.config.dirichlet_weight
                        # Generate Dirichlet noise: Dir(alpha, alpha, ..., alpha)
                        noise = torch.distributions.Dirichlet(
                            torch.full((n_children,), alpha)
                        ).sample()
                        children_list = list(root.children.values())
                        for i, child in enumerate(children_list):
                            child.prior = (1.0 - eps) * child.prior + eps * noise[i].item()

                # Pick a random child for the rollout
                if node.children:
                    node = random.choice(list(node.children.values()))
                    path.append(node)

            # ---- EVALUATE ----
            # Get value estimate for the leaf
            value = self._evaluate(node)

            # ---- BACKPROPAGATE ----
            # Update all nodes on the path
            for ancestor in path:
                ancestor.visit_count += 1
                ancestor.total_value += value

            if verbose and (sim + 1) % 50 == 0:
                best = root.most_visited_child()
                best_action = best.incoming_action
                print(
                    f"  MCTS sim {sim+1}/{self.config.num_simulations} | "
                    f"Root visits: {root.visit_count} | "
                    f"Best: {best_action.to_lean() if best_action else '?'} "
                    f"(N={best.visit_count}, Q={best.mean_value:.3f})"
                )

        # ---- RETURN BEST PATH ----
        if not root.children:
            return ([], root)

        # Select best proof path by following most-visited children
        proof_steps, best_node = self._extract_best_path(root)

        if verbose:
            print(f"\nMCTS search complete: {root.visit_count} root visits")
            print(f"Best proof: {len(proof_steps)} steps")
            for s in proof_steps:
                print(f"  {s.to_lean()}")

        return proof_steps, root

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _expand(self, node: MCTSNode, available_lemmas: list[str]) -> None:
        """Expand a leaf node by generating candidate actions and scoring them.

        Uses the GNN to score candidate lemmas for the current goal.
        When proof checker is enabled, verifies each candidate through Lean
        before creating child nodes — only valid steps become children.

        Stores differentiable logits in the parent node for ALL candidates
        (including rejected ones), so the training loss can learn to avoid
        invalid actions (target=0.0 for rejected candidates).
        """
        # Generate candidates
        candidates = generate_candidate_actions(
            node.state, available_lemmas, node.state.hypotheses
        )

        if not candidates:
            node.state.is_dead = True
            return

        # Limit branching factor
        if len(candidates) > self.config.max_actions_per_node:
            candidates = random.sample(candidates, self.config.max_actions_per_node)

        # Score candidates using GNN (if available).
        # _score_actions returns (detached_priors, differentiable_logits).
        if self.gnn is not None and self._node_embeddings is not None:
            priors, logits = self._score_actions(node.state, candidates)
            if logits is not None:
                node.child_logits = logits
                node._child_action_order = list(candidates)
        else:
            priors = [1.0 / len(candidates)] * len(candidates)

        # ---- Verification gate (AlphaGo Zero: proof checker = game engine) ----
        # Verify lemma-based and exact-with-hypothesis actions at root.
        # These are where false positives cause damage. Exact with a
        # hypothesis is verified because the hypothesis type must match.
        is_root = node.parent is None
        _lemma_tactics = {"apply", "rewrite", "exact"}
        verify_candidates = [
            c for c in candidates
            if c.tactic_type.value in _lemma_tactics
            and (c.lemma is not None or c.hypothesis is not None)
        ]
        if (
            is_root
            and self.config.use_proof_checker
            and self.proof_checker is not None
            and len(verify_candidates) > 0
        ):
            verify_mask = self._verify_candidates(node.state, verify_candidates)
            # Build full valid_mask: structural tactics always pass
            valid_mask = []
            vi = 0
            for c in candidates:
                if c.tactic_type.value in _lemma_tactics and (
                    c.lemma is not None or c.hypothesis is not None
                ):
                    valid_mask.append(verify_mask[vi])
                    vi += 1
                else:
                    valid_mask.append(True)  # structural/auto tactics always pass
        else:
            valid_mask = [True] * len(candidates)

        # If ALL candidates were rejected by Lean, fall back to creating all
        # children. This prevents degenerate training steps where the MCTS
        # has no branches to explore.
        if not any(valid_mask):
            valid_mask = [True] * len(candidates)
        elif is_root and self.config.use_proof_checker and self.proof_checker is not None:
            rejected = sum(1 for v in valid_mask if not v)
            if rejected > 0:
                import sys
                print(f"  [verify] Root: {rejected}/{len(candidates)} candidates "
                      f"rejected by Lean", file=sys.stderr, flush=True)

        # Create child nodes only for verified actions.
        # Rejected actions get target=0.0 in the training loss because
        # _child_action_order includes them but they have no child node.
        for action, prior, valid in zip(candidates, priors, valid_mask):
            if not valid:
                continue
            child_state = node.state.apply_tactic(action)
            child = MCTSNode(
                state=child_state,
                parent=node,
                incoming_action=action,
                prior=prior,
            )
            node.children[action] = child

    def _verify_candidates(
        self, state: ProofState, candidates: list[Tactic]
    ) -> list[bool]:
        """Batch-verify candidate tactics through the Lean proof checker.

        For each candidate, builds the full proof code (proof-so-far + new step)
        and checks it with Lean. Returns a boolean mask indicating which
        candidates produce valid partial proofs (no type errors).

        Results are cached by (proof_so_far_hash, tactic_hash) to avoid
        redundant verification across MCTS simulations.
        """
        from src.proof_checker.formats import wrap_theorem_with_proof

        # Separate cached from uncached
        uncached_indices: list[int] = []
        uncached_codes: list[str] = []
        results: dict[int, bool] = {}

        for i, tactic in enumerate(candidates):
            cache_key = (hash(state.proof_so_far), hash(tactic))
            if cache_key in self._verification_cache:
                results[i] = self._verification_cache[cache_key]
            else:
                uncached_indices.append(i)
                # Build the proof body: existing steps + new tactic
                if state.proof_so_far:
                    proof_body = state.proof_so_far + "\n  " + tactic.to_lean()
                else:
                    proof_body = tactic.to_lean()
                code = wrap_theorem_with_proof(state.theorem_statement, proof_body)
                uncached_codes.append(code)

        # Batch-check uncached candidates.
        # This is called at most once per theorem (root expansion only), so
        # the ProcessPoolExecutor overhead in BatchChecker is acceptable.
        if uncached_codes:
            check_results = self.proof_checker.check_batch(uncached_codes)
            for j, idx in enumerate(uncached_indices):
                is_valid = check_results[j].success
                cache_key = (hash(state.proof_so_far), hash(candidates[idx]))
                self._verification_cache[cache_key] = is_valid
                results[idx] = is_valid

        return [results[i] for i in range(len(candidates))]

    def _evaluate(self, node: MCTSNode) -> float:
        """Evaluate a leaf node's value.

        Value sources (in order of preference):
        1. Terminal check: completed proof → 1.0, dead end → -1.0
        2. Proof checker (if enabled): validates the proof so far
        3. GNN value head: estimates success probability
        4. Heuristic: based on proof length and goal count
        """
        # Terminal states
        if node.state.is_complete:
            return 1.0
        if node.state.is_dead:
            return -1.0

        # Proof length penalty (favor shorter proofs)
        length_penalty = max(0.0, 1.0 - len(node.state.steps) / self.config.max_depth)

        # GNN-based evaluation
        if self.gnn is not None and self._node_embeddings is not None:
            gnn_value = self._gnn_evaluate(node)
            return 0.7 * gnn_value + 0.3 * length_penalty

        # Heuristic fallback
        # Fewer goals = closer to completion
        num_goals = len(node.state.goals)
        if num_goals == 0:
            return 0.95
        goal_penalty = 1.0 / (1.0 + num_goals)

        return 0.5 * length_penalty + 0.5 * goal_penalty

    def _gnn_evaluate(self, node: MCTSNode) -> float:
        """Use the GNN to estimate the value of a proof state.

        This is a simplified value head: it compares the goal embedding
        to all known theorem embeddings and returns the maximum similarity
        to any theorem that has been successfully proved (in-degree > 0
        in the dependency graph).
        """
        if self._node_embeddings is None:
            return 0.5

        # Get the goal text embedding (simplified: use random projection)
        # In the full system, the goal would be encoded through the same
        # embedding pipeline as the graph nodes
        goal_embedding = self._embed_goal(node.state)

        if goal_embedding is None:
            return 0.5

        # Find the closest lemma embedding (cosine similarity)
        sims = torch.cosine_similarity(
            goal_embedding.unsqueeze(0), self._node_embeddings, dim=1
        )
        max_sim = sims.max().item()

        # Map similarity to value: higher sim → more likely to find a proof
        # (because there exists a lemma close to the goal)
        return min(1.0, max(0.0, max_sim))

    def _score_actions(
        self, state: ProofState, actions: list[Tactic]
    ) -> tuple[list[float], "torch.Tensor | None"]:
        """Score candidate actions using hybrid GNN + structural signals.

        Combines:
        1. GNN embedding similarity (goal vs lemma) — if both in embedding space
        2. Lemma centrality (in-degree in dependency graph) — fundamental = useful
        3. Tactic-type prior — structural tactics get moderate prior

        Returns:
            (priors_list, logits_tensor) where:
            - priors_list: list of detached float priors for MCTS tree search
            - logits_tensor: differentiable tensor of raw scores, connected to
              the GNN computation graph — used in the training loss for gradient flow
        """
        if self._node_embeddings is None:
            uniform = 1.0 / len(actions)
            return [uniform] * len(actions), None

        goal_embedding = self._embed_goal(state)
        device = self._node_embeddings.device
        goal_text = state.get_goal_embedding_key()

        # Pre-compute centrality scores from the dependency graph
        if self.graph is not None:
            in_degrees = dict(self.graph.graph.in_degree())
            max_in = max(in_degrees.values()) if in_degrees else 1
        else:
            in_degrees = {}
            max_in = 1

        # Build scores as differentiable tensors (not Python floats).
        # Each score is a scalar tensor connected to the GNN graph via
        # cosine_similarity(goal_embedding, lemma_embedding).
        score_tensors: list[torch.Tensor] = []
        for action in actions:
            # Start with a small constant base score
            score = torch.tensor(0.05, device=device, dtype=torch.float32)

            lemma = action.lemma
            lemma_emb = None
            if lemma:
                if lemma in self._lemma_to_idx:
                    lemma_emb = self._node_embeddings[self._lemma_to_idx[lemma]]
                elif lemma in self._builtin_embeddings:
                    lemma_emb = self._builtin_embeddings[lemma]

            if lemma_emb is not None:
                # GNN similarity: goal vs lemma (differentiable)
                if goal_embedding is not None and goal_embedding.norm() > 0:
                    sim = torch.cosine_similarity(
                        goal_embedding.unsqueeze(0), lemma_emb.unsqueeze(0), dim=1
                    )
                    score = score + sim.clamp(min=0.0).squeeze() * 0.8

                # Keyword relevance: penalize lemmas whose key terms are
                # absent from the goal. This counters GNN false positives
                # where the GNN matches a lemma to an unrelated goal
                # (e.g., zero_pow for a goal with no 0).
                relevance = _lemma_goal_keyword_match(lemma, goal_text)

                # Additional check for rw tactics: the lemma's rewrite
                # pattern must actually appear in the goal. Without this,
                # MCTS wastes simulations on `rw [mul_one]` for goals
                # that don't contain `*1`.
                if action.tactic_type.value == "rewrite" and not _rw_pattern_in_goal(
                    lemma, goal_text
                ):
                    relevance = min(relevance, 0.3)

                if relevance < 1.0:
                    # Proportional penalty: fully irrelevant → -0.5, half → -0.25
                    score = score - 0.5 * (1.0 - relevance)

                # Centrality: fundamental lemmas are more useful
                centrality = in_degrees.get(lemma, 0) / max_in
                score = score + centrality * 0.1

                # Built-in lemmas get a base boost (they're fundamental)
                if lemma in self._builtin_embeddings:
                    score = score + 0.25

                # Penalize identity/trivial lemmas. These type-check against
                # any goal (so Lean verification won't reject them) but never
                # make actual progress. A strong penalty ensures MCTS doesn't
                # waste simulations on them even when GNN similarity is high.
                _trivial_lemmas = {
                    "id", "Function.id", "Function.comp_id",
                    "rfl", "Eq.refl", "Eq.refl'",
                    "true_and", "and_true", "false_or", "or_false",
                    "true_implies", "implies_true",
                    "eq_self", "eq_true", "true_eq",
                }
                if lemma in _trivial_lemmas:
                    score = score - 1.5

            elif action.tactic_type.value in ("intro", "cases", "have"):
                score = score + 0.25  # Structural tactics get moderate prior
            elif action.tactic_type.value in ("ring", "field_simp", "linarith", "simp"):
                score = score + 0.25  # Automation tactics get moderate prior
            elif action.tactic_type.value == "exact" and action.hypothesis:
                score = score + 0.30  # Using a local hypothesis is promising
            elif action.tactic_type.value == "rewrite" and action.hypothesis:
                score = score + 0.28  # Rewriting with a hypothesis is promising

            score_tensors.append(score)

        # Stack into a single differentiable tensor [num_actions]
        logits = torch.stack(score_tensors)

        # ── Cold-start heuristics for arithmetic patterns ──
        # These are mathematically correct: e.g., add_comm ALWAYS proves
        # a+b=b+a. The GNN should learn this from data, but until it does,
        # these heuristics ensure the MCTS can find correct proofs.
        # They are annealed to zero (via heuristic_scale) during training.
        # Extract the actual goal from "theorem name : goal := by"
        # Format: "theorem name : <goal> := by"
        if ":=" in goal_text:
            before_by = goal_text.split(":=")[0].strip()
            # Remove "theorem name :" prefix
            if ":" in before_by:
                goal_only = before_by.split(":", 1)[-1].strip()
            else:
                goal_only = before_by
        elif ":" in goal_text:
            goal_only = goal_text.rsplit(":", 1)[-1].strip()
        else:
            goal_only = goal_text

        # Detect goal patterns and boost matching lemmas
        _apply_arithmetic_heuristics(logits, actions, goal_only)

        # Apply global heuristic scale (for annealing during training)
        heuristic_scale = getattr(self.config, 'heuristic_scale', 1.0)
        if heuristic_scale != 1.0:
            # Scale heuristic boosts relative to GNN scores
            # GNN scores are in score_tensors; heuristic boosts are in logits
            # We compute the boost amount and scale it
            base_logits = torch.stack(score_tensors)
            boost = logits - base_logits
            logits = base_logits + boost * heuristic_scale

        # Softmax to get prior probabilities (temperature=0.5 sharpens distribution)
        priors = torch.softmax(logits / 0.5, dim=0)

        # Return detached priors for MCTS search + differentiable logits for loss
        return priors.detach().tolist(), logits

    def _embed_goal(self, state: ProofState) -> torch.Tensor | None:
        """Create an embedding for the current proof goal.

        Uses normalized text matching to find structurally similar lemmas,
        then averages their GNN embeddings as context for the GoalEncoder.
        Falls back to keyword matching for complex goals without structural matches.
        """
        if self._node_embeddings is None or self.graph is None:
            return None

        import re
        from scripts.eval_gnn_prover import normalize_expression

        goal_text = state.get_goal_embedding_key()
        device = self._node_embeddings.device
        node_emb_norm = F.normalize(self._node_embeddings, dim=-1)

        goal_norm = normalize_expression(goal_text)

        # Detect reflexivity
        is_reflexive = False
        if "=" in goal_norm and "↔" not in goal_norm and "→" not in goal_norm and "≠" not in goal_norm:
            sides = goal_norm.split("=", 1)
            if len(sides) == 2 and sides[0].strip() == sides[1].strip():
                is_reflexive = True

        # Find exact structural matches
        exact_matches = set()
        for idx, lemma_norm in self._idx_to_norm.items():
            if lemma_norm == goal_norm:
                exact_matches.add(idx)
            elif is_reflexive and lemma_norm == normalize_expression("a = a"):
                exact_matches.add(idx)
            elif " ↔ " in lemma_norm:
                left, right = lemma_norm.split(" ↔ ", 1)
                if left.strip() == goal_norm or right.strip() == goal_norm:
                    exact_matches.add(idx)
            elif " → " in lemma_norm:
                parts = lemma_norm.rsplit(" → ", 1)
                if parts[-1].strip() == goal_norm:
                    exact_matches.add(idx)

        # Power-stripping fallback
        if not exact_matches:
            goal_stripped = re.sub(r'\s*\^\s*\d+', '', goal_norm)
            for idx, lemma_norm in self._idx_to_norm.items():
                lemma_stripped = re.sub(r'\s*\^\s*\d+', '', lemma_norm)
                if lemma_stripped == goal_stripped:
                    exact_matches.add(idx)

        # Subterm matches
        subterm_matches = set()
        if " = " in goal_norm:
            goal_lhs, goal_rhs = goal_norm.split(" = ", 1)
            for idx, lemma_norm in self._idx_to_norm.items():
                if idx in exact_matches or " = " not in lemma_norm:
                    continue
                lemma_lhs, lemma_rhs = lemma_norm.split(" = ", 1)
                if len(lemma_lhs) < 3 or len(lemma_rhs) < 3:
                    continue
                if lemma_lhs in goal_lhs and lemma_rhs in goal_rhs:
                    subterm_matches.add(idx)
                elif lemma_lhs in goal_rhs and lemma_rhs in goal_lhs:
                    subterm_matches.add(idx)

        # Build context from matches
        match_indices = list(exact_matches) + list(subterm_matches)

        if match_indices:
            indices_t = torch.tensor(match_indices[:100], device=device)
            context_emb = node_emb_norm[indices_t].mean(dim=0)
        else:
            # Fall back to keyword-based context
            keywords = _extract_math_keywords(goal_text)
            candidate_scores: dict[int, float] = {}
            for kw in keywords:
                matches = self._kw_lemmas_map.get(kw.lower(), [])
                for rank, idx in enumerate(matches):
                    if idx >= self._node_embeddings.size(0):
                        continue
                    score = 1.0 / (1.0 + rank * 0.1)
                    candidate_scores[idx] = candidate_scores.get(idx, 0.0) + score
            sorted_candidates = sorted(candidate_scores.items(), key=lambda x: -x[1])[:100]
            matching_indices = [idx for idx, _ in sorted_candidates]

            if matching_indices:
                indices_t = torch.tensor(matching_indices, device=device)
                context_emb = node_emb_norm[indices_t].mean(dim=0)
            else:
                return torch.zeros(node_emb_norm.size(1), device=device)

        if self.gnn is not None and self.gnn.goal_encoder is not None:
            return self.gnn.encode_goal(context_emb)
        return F.normalize(context_emb, dim=-1) if context_emb.norm() > 0 else context_emb

    def _get_relevant_lemmas(self, theorem_statement: str) -> list[str]:
        """Get lemmas relevant to this theorem using hybrid retrieval.

        1. Extract mathematical keywords from the goal text
        2. Filter graph lemmas by keyword match on name
        3. Add built-in fundamental lemmas (add_comm, rfl, etc.)
        4. Rank candidates by GNN centrality (in-degree) × keyword score
        5. Return top-K
        """
        if self.graph is None:
            return _get_builtin_lemmas(["eq", "refl"])

        keywords = _extract_math_keywords(theorem_statement)

        # Get built-in lemmas for these keywords
        builtins = _get_builtin_lemmas(keywords)

        # Phase 1: Filter graph lemmas by keyword match
        candidates: list[tuple[str, float]] = []  # (name, combined_score)
        seen: set[str] = set()

        for kw in keywords[:8]:  # Limit keyword expansion
            for nid in self.graph.node_ids:
                if nid in seen:
                    continue
                attrs = self.graph.get_node(nid)
                name = attrs.get("name", nid) if attrs else nid
                if kw.lower() in name.lower():
                    # Score: how many keywords match this lemma name?
                    kw_score = sum(
                        1.0 for k in keywords if k.lower() in name.lower()
                    )
                    candidates.append((nid, kw_score))
                    seen.add(nid)
                    if len(candidates) >= 300:
                        break
            if len(candidates) >= 300:
                break

        # Phase 2: Rank by structural centrality + keyword score
        in_degrees = dict(self.graph.graph.in_degree())
        max_in = max(in_degrees.values()) if in_degrees else 1

        ranked = []
        for name, kw_score in candidates:
            centrality = in_degrees.get(name, 0) / max_in
            combined = 0.3 * kw_score + 0.7 * centrality  # Centrality-weighted
            ranked.append((name, combined))

        ranked.sort(key=lambda x: x[1], reverse=True)
        lemmas = [name for name, _ in ranked[: self.config.top_k_lemmas]]

        # Phase 3: Prepend built-in lemmas (they take priority for bootstrap)
        # Put builtins first, then graph lemmas
        result = []
        seen_result = set()
        for bl in builtins:
            if bl not in seen_result:
                result.append(bl)
                seen_result.add(bl)

        for gl in lemmas:
            if gl not in seen_result:
                result.append(gl)
                seen_result.add(gl)

        # Cache lemma indices for fast lookup in _score_actions
        if self._node_embeddings is not None and self.graph is not None:
            for lemma in result:
                idx = self.graph.node_id_to_idx(lemma)
                if idx is not None:
                    self._lemma_to_idx[lemma] = idx

        return result[: self.config.top_k_lemmas]

    def _extract_best_path(self, root: MCTSNode) -> tuple[list[Tactic], MCTSNode]:
        """Extract the best proof path from the search tree.

        Follows the most-visited child at each node to construct
        the proof. Stops when reaching a completed state or max depth.
        """
        steps = []
        node = root

        for _ in range(self.config.max_depth):
            if not node.children or node.state.is_complete:
                break

            # Select child with most visits
            best = node.most_visited_child()
            if best.incoming_action:
                steps.append(best.incoming_action)
            node = best

        return steps, node

    # ------------------------------------------------------------------
    # Integration with GNN
    # ------------------------------------------------------------------

    def set_embeddings(
        self,
        embeddings: torch.Tensor,
        lemma_names: list[str],
    ) -> None:
        """Set pre-computed GNN embeddings for lemma scoring."""
        self._node_embeddings = embeddings
        self._lemma_names = lemma_names
        self._lemma_to_idx = {name: i for i, name in enumerate(lemma_names)}

        # Build keyword → lemma index map and normalized conclusion index
        self._build_indices()

        # Create approximate embeddings for built-in lemmas not in the graph
        self._build_builtin_embeddings()

    def _build_indices(self) -> None:
        """Build keyword map and normalized conclusion index for goal embedding."""
        from collections import defaultdict
        from scripts.eval_gnn_prover import (
            normalize_expression, extract_conclusion, build_lemma_norm_index,
        )

        # Keyword map (fallback)
        kw_map = defaultdict(list)
        idx_to_short = {}
        for lemma_name, idx in self._lemma_to_idx.items():
            short = lemma_name.split(".")[-1] if "." in lemma_name else lemma_name
            idx_to_short[idx] = short.lower()

        for lemma_name, idx in self._lemma_to_idx.items():
            short = lemma_name.split(".")[-1] if "." in lemma_name else lemma_name
            tokens = short.lower().replace("_", " ").split()
            for token in tokens:
                if len(token) >= 2:
                    kw_map[token].append(idx)
            kw_map[short.lower()].append(idx)

        for kw in kw_map:
            seen = set()
            deduped = []
            for idx in kw_map[kw]:
                if idx not in seen:
                    seen.add(idx)
                    deduped.append(idx)
            deduped.sort(key=lambda i: (
                0 if idx_to_short.get(i, "") == kw else 1,
                len(idx_to_short.get(i, "")),
            ))
            kw_map[kw] = deduped
        self._kw_lemmas_map = dict(kw_map)

        # Normalized conclusion index (primary matching)
        self._idx_to_norm = build_lemma_norm_index(self.graph, self._lemma_to_idx)

    def _build_builtin_embeddings(self) -> None:
        """Create approximate embeddings for built-in lemmas not in the graph.

        For each built-in lemma keyword group, averages the embeddings of
        graph lemmas matching that keyword. This gives built-in lemmas a
        meaningful embedding for GNN similarity scoring.
        """
        if self._node_embeddings is None:
            return

        device = self._node_embeddings.device
        D = self._node_embeddings.size(1)

        for kw, lemma_names in _BUILTIN_LEMMAS.items():
            # Collect graph lemma indices matching this keyword
            matching_indices = set()
            for lemma_name, idx in self._lemma_to_idx.items():
                if kw.lower() in lemma_name.lower():
                    matching_indices.add(idx)

            if matching_indices:
                # Average the embeddings of matching lemmas
                indices_t = torch.tensor(
                    list(matching_indices), device=device
                )
                avg_emb = self._node_embeddings[indices_t].mean(dim=0)
            else:
                # Fallback: small random vector
                avg_emb = torch.randn(D, device=device) * 0.01

            # Store approximate embedding for each lemma in this keyword group
            for lemma_name in lemma_names:
                if lemma_name not in self._lemma_to_idx:
                    self._builtin_embeddings[lemma_name] = avg_emb.clone()

    def clear_cache(self) -> None:
        """Clear the transposition table and cached state."""
        self._state_cache.clear()


def _lemma_goal_keyword_match(lemma_name: str, goal_text: str) -> float:
    """Check what fraction of a lemma's key terms appear in the goal.

    Returns 0.0 to 1.0 — low values mean the lemma is likely irrelevant.
    Used to penalize GNN false positives like zero_pow for goals without 0.
    """
    # Map number words to digits for matching
    _number_map = {
        "zero": "0", "one": "1", "two": "2", "three": "3",
        "add": "+", "mul": "*", "sub": "-", "div": "/",
        "pow": "^", "sq": "^2",
        "le": "≤", "ge": "≥", "lt": "<", "gt": ">",
        "eq": "=", "ne": "≠", "not": "¬",
    }
    generic = {"of", "the", "a", "an", "is", "to", "and", "or", "not",
               "self", "left", "right", "comm", "assoc", "symm", "trans",
               "refl", "iff", "mem", "prop", "type", "coe"}

    short = lemma_name.split(".")[-1] if "." in lemma_name else lemma_name
    terms = short.lower().replace("_", " ").split()
    distinctive = [t for t in terms if t not in generic and len(t) >= 2]

    if not distinctive:
        return 0.5  # Neutral — can't determine

    goal_lower = goal_text.lower()
    matches = 0
    for t in distinctive:
        if t in goal_lower:
            matches += 1
        elif t in _number_map and _number_map[t] in goal_text:
            matches += 1
        elif t.endswith("_eq") or t.endswith("_comm") or t.endswith("_assoc"):
            matches += 0.5  # Structural lemmas are broadly applicable

    return matches / len(distinctive)


def _rw_pattern_in_goal(lemma_name: str, goal_text: str) -> bool:
    """Check if a lemma's rewrite pattern could match the goal.

    For rw to succeed, the lemma's LHS pattern must appear in the goal.
    Examples:
      mul_one: pattern a*1 = a → need *1 or 1* in goal
      add_comm: pattern a+b = b+a → + is enough (any + matches)
      zero_mul: pattern 0*a = 0 → need 0* or *0 in goal
    """
    short = lemma_name.split(".")[-1] if "." in lemma_name else lemma_name
    name = short.lower()

    # Pattern-specific checks
    patterns = {
        "mul_one": ("*1", "1*", "* 1", "1 *"),
        "one_mul": ("1*", "*1", "1 *", "* 1"),
        "zero_mul": ("0*", "*0", "0 *", "* 0"),
        "mul_zero": ("*0", "0*", "* 0", "0 *"),
        "add_zero": ("+0", "0+", "+ 0", "0 +"),
        "zero_add": ("0+", "+0", "0 +", "+ 0"),
        "sub_zero": ("-0", "- 0"),
        "zero_sub": ("0-", "0 -"),
        "sub_self": ("-",),
        "add_self": ("+",),
        "mul_self": ("*",),
        "pow_zero": ("^0", "^ 0"),
        "pow_one": ("^1", "^ 1"),
        "one_pow": ("1^", "1 ^"),
        "div_one": ("/1", "/ 1"),
    }

    if name in patterns:
        return any(p in goal_text for p in patterns[name])

    # Generic: check if any digit or operator from the lemma name
    # appears in the goal
    for term in name.split("_"):
        if term in ("zero", "one", "two", "add", "mul", "sub", "div", "pow",
                      "comm", "assoc", "symm", "trans", "refl", "eq", "ne",
                      "self", "left", "right", "of", "the", "a", "an"):
            continue
        if len(term) >= 2 and term in goal_text.lower():
            return True

    # Default: assume the pattern could match (no penalty)
    return True


def _apply_arithmetic_heuristics(
    logits: torch.Tensor, actions: list, goal_text: str
) -> None:
    """Boost lemma scores based on goal arithmetic patterns.

    These are mathematically correct heuristics that encode basic algebra:
    - add_comm for commutative addition goals (a+b = b+a)
    - mul_comm for commutative multiplication goals (a*b = b*a)
    - add_zero/zero_add for identity addition goals (a+0 = a, 0+a = a)
    - mul_one/one_mul for identity multiplication goals (a*1 = a, 1*a = a)
    - add_assoc/mul_assoc for associative goals
    - rfl for reflexive goals (a = a)

    All boosts are scaled by heuristic_scale during training (annealed to 0).
    """
    import re

    text = goal_text.strip()
    has_add = "+" in text
    has_mul = "*" in text
    has_eq = "=" in text

    # No equality → not an equational goal we can pattern-match
    if not has_eq:
        return

    # Split into LHS and RHS
    parts = text.split("=", 1)
    if len(parts) != 2:
        return
    lhs = parts[0].strip()
    rhs = parts[1].strip()

    # Normalize for comparison
    def _norm(s: str) -> str:
        return re.sub(r"\s+", "", s.replace("(", "").replace(")", ""))

    lhs_norm = _norm(lhs)
    rhs_norm = _norm(rhs)

    # --- Reflexive: a = a ---
    if lhs_norm == rhs_norm:
        for i, action in enumerate(actions):
            if action.lemma in ("rfl", "Eq.refl", "Eq.refl'") and action.tactic_type.value in ("exact", "apply"):
                logits[i] = logits[i] + 3.0

    # --- Commutative: a+b = b+a or a*b = b*a ---
    # Pattern: check if LHS and RHS are reorderings of each other
    if has_add:
        # Extract tokens split by + on both sides
        lhs_terms = [t.strip() for t in lhs.split("+")]
        rhs_terms = [t.strip() for t in rhs.split("+")]
        lhs_terms = [t for t in lhs_terms if t]
        rhs_terms = [t for t in rhs_terms if t]
        lhs_sorted = sorted(lhs_terms)
        rhs_sorted = sorted(rhs_terms)

        # Commutative: a+b = b+a
        if (len(lhs_terms) == 2 and len(rhs_terms) == 2
                and lhs_sorted == rhs_sorted and lhs_terms != rhs_terms):
            for i, action in enumerate(actions):
                if action.lemma == "add_comm":
                    logits[i] = logits[i] + 3.0

        # Identity: a+0 = a, 0+a = a, a = a+0, a = 0+a
        # Only boost when pattern actually present in text
        all_terms = set(lhs_terms + rhs_terms)
        has_add_zero_pattern = (
            "+0" in text or "0+" in text
            or text.strip().endswith("=0") or text.strip().endswith("= 0")
        )
        if "0" in all_terms and has_add_zero_pattern:
            for i, action in enumerate(actions):
                if action.lemma == "add_zero" and action.tactic_type.value != "rewrite":
                    logits[i] = logits[i] + 2.5
                if action.lemma == "zero_add" and action.tactic_type.value != "rewrite":
                    logits[i] = logits[i] + 2.5
            # Direction-specific boost
            lhs_has_0 = any(t.strip() == "0" for t in lhs_terms)
            rhs_has_0 = any(t.strip() == "0" for t in rhs_terms)
            if lhs_has_0 and not rhs_has_0:
                for i, action in enumerate(actions):
                    if action.lemma == "zero_add" and action.tactic_type.value != "rewrite":
                        logits[i] = logits[i] + 1.0
            if rhs_has_0 and not lhs_has_0:
                for i, action in enumerate(actions):
                    if action.lemma == "add_zero" and action.tactic_type.value != "rewrite":
                        logits[i] = logits[i] + 1.0

    if has_mul:
        lhs_terms = [t.strip() for t in lhs.split("*")]
        rhs_terms = [t.strip() for t in rhs.split("*")]
        lhs_terms = [t for t in lhs_terms if t]
        rhs_terms = [t for t in rhs_terms if t]
        lhs_sorted = sorted(lhs_terms)
        rhs_sorted = sorted(rhs_terms)

        # Commutative: a*b = b*a
        if (len(lhs_terms) == 2 and len(rhs_terms) == 2
                and lhs_sorted == rhs_sorted and lhs_terms != rhs_terms):
            for i, action in enumerate(actions):
                if action.lemma == "mul_comm":
                    logits[i] = logits[i] + 3.0

        # Identity: a*1 = a, 1*a = a, a = a*1, a = 1*a
        # Only boost when pattern actually present in text
        has_mul_one_pattern = "*1" in text or "1*" in text or "* 1" in text or "1 *" in text
        all_terms = set(lhs_terms + rhs_terms)
        if "1" in all_terms and has_mul_one_pattern:
            for i, action in enumerate(actions):
                if action.lemma == "mul_one" and action.tactic_type.value != "rewrite":
                    logits[i] = logits[i] + 2.5
                if action.lemma == "one_mul" and action.tactic_type.value != "rewrite":
                    logits[i] = logits[i] + 2.5
            # Direction-specific boost: which side has the 1?
            lhs_has_1 = any("1" in t for t in lhs_terms)
            rhs_has_1 = any("1" in t for t in rhs_terms)
            if lhs_has_1 and not rhs_has_1:
                for i, action in enumerate(actions):
                    if action.lemma == "one_mul" and action.tactic_type.value != "rewrite":
                        logits[i] = logits[i] + 1.0  # Extra boost for 1*a = a
            if rhs_has_1 and not lhs_has_1:
                for i, action in enumerate(actions):
                    if action.lemma == "mul_one" and action.tactic_type.value != "rewrite":
                        logits[i] = logits[i] + 1.0  # Extra boost for a*1 = a

    # --- Associative: (a+b)+c = a+(b+c) or (a*b)*c = a*(b*c) ---
    # Simple heuristic: if one side has nested structure and the other doesn't match
    if ("+" in lhs and "+" in rhs) or ("*" in lhs and "*" in rhs):
        # Check for parentheses indicating nesting
        if ("(" in lhs) != ("(" in rhs):
            for i, action in enumerate(actions):
                if has_add and action.lemma == "add_assoc":
                    logits[i] = logits[i] + 2.5
                if has_mul and action.lemma == "mul_assoc":
                    logits[i] = logits[i] + 2.5

    # --- Automation tactics for complex goals ---
    # Skip goals containing → or ∀ — automation tactics can't handle these
    has_implication = "→" in text or "∀" in text

    # ring: polynomial identities with multiple operations
    op_count = sum(1 for c in text if c in "+*^")
    if op_count >= 3 and has_eq and not has_implication:
        for i, action in enumerate(actions):
            if action.tactic_type.value == "ring":
                logits[i] = logits[i] + 2.0

    # field_simp: goals with division or inverses
    if ("/" in text or "⁻¹" in text) and has_eq and not has_implication:
        for i, action in enumerate(actions):
            if action.tactic_type.value == "field_simp":
                logits[i] = logits[i] + 2.5

    # linarith: linear arithmetic goals (inequalities or simple equations)
    if any(c in text for c in ("≤", "≥", "<", ">")) and not has_implication:
        for i, action in enumerate(actions):
            if action.tactic_type.value == "linarith":
                logits[i] = logits[i] + 2.5

    # simp: available as a general fallback (low boost)
    if not has_implication:
        for i, action in enumerate(actions):
            if action.tactic_type.value == "simp":
                logits[i] = logits[i] + 0.3


def _is_reflexive_goal(goal_text: str) -> bool:
    """Check if a goal is a simple reflexive equality (X = X).

    These goals are trivially provable with `exact rfl`. The GNN doesn't
    know this (it was pretrained for link prediction, not proof search),
    so we add a heuristic boost for rfl on reflexive goals.
    """
    text = goal_text.strip()
    # Pattern: something = same something (reflexive equality)
    if "=" in text:
        parts = text.split("=", 1)
        if len(parts) == 2:
            left = parts[0].strip()
            right = parts[1].strip()
            # Allow whitespace/parenthesis variations
            left_clean = left.replace(" ", "").replace("(", "").replace(")", "")
            right_clean = right.replace(" ", "").replace("(", "").replace(")", "")
            return left_clean == right_clean
    return False


def _extract_math_keywords(text: str) -> list[str]:
    """Extract mathematical keywords from theorem text for lemma matching.

    Uses word-boundary matching to avoid false positives like 'le' in 'example'.
    """
    import re

    # Map common math tokens to lemma name fragments
    # Only match on word boundaries to avoid substring false positives
    token_map = {
        # Arithmetic operators → lemma prefixes
        "+": ["add"],
        "*": ["mul"],
        "-": ["sub"],
        "/": ["div"],
        "^": ["pow"],
        "=": ["eq", "refl"],
        # Relations
        "→": ["imp", "of"],
        "∀": ["forall"],
        "∃": ["exists"],
        "≤": ["le"],
        "≥": ["ge"],
        "<": ["lt"],
        ">": ["gt"],
        "⁻¹": ["inv"],
        "∘": ["comp"],
    }

    # Word-boundary patterns for text matching
    word_patterns = {
        # Numbers
        "0": ["zero"],
        "1": ["one"],
        # Arithmetic lemmas
        "add": ["add"],
        "mul": ["mul"],
        "sub": ["sub"],
        "div": ["div"],
        "neg": ["neg"],
        # Properties
        "comm": ["comm"],
        "assoc": ["assoc"],
        "distrib": ["distrib"],
        # Logic
        "and": ["and"],
        "or": ["or"],
        "not": ["not"],
        "iff": ["iff"],
        "eq": ["eq"],
        "refl": ["refl"],
        "symm": ["symm"],
        "trans": ["trans"],
        # Types (capitalized to avoid false matches)
        "Nat": ["Nat"],
        "Int": ["Int"],
        "Real": ["Real"],
        "Complex": ["Complex"],
        "Prop": ["Prop"],
        "Set": ["Set"],
        "List": ["List"],
        # Calculus
        "deriv": ["deriv"],
        "integral": ["integral"],
        "limit": ["limit"],
        "continuous": ["continuous"],
        "sum": ["sum"],
        "prod": ["prod"],
        # Algebra
        "ring": ["ring"],
        "field": ["field"],
        "group": ["group"],
        "linear": ["linear"],
        "inv": ["inv"],
        "pow": ["pow"],
    }

    found = []

    # Symbol matching
    for sym, terms in token_map.items():
        if sym in text:
            found.extend(terms)

    # Word-boundary text matching
    text_lower = text.lower()
    for pattern, terms in word_patterns.items():
        # Match as whole word/subword with boundaries
        if re.search(r'\b' + re.escape(pattern.lower()) + r'\b', text_lower):
            found.extend(terms)

    # Deduplicate preserving order
    seen = set()
    result = []
    for f in found:
        if f not in seen:
            result.append(f)
            seen.add(f)

    return result if result else ["eq", "refl"]  # default: equality lemmas


# Built-in lemma bank for bootstrap theorems.
# These are fundamental lemmas from core Lean/Std that may not appear
# as explicit theorem nodes in the extracted dependency graph.
_BUILTIN_LEMMAS: dict[str, list[str]] = {
    "add": ["add_comm", "add_zero", "zero_add", "add_assoc", "add_left_cancel"],
    "mul": ["mul_comm", "mul_zero", "zero_mul", "mul_one", "one_mul", "mul_assoc"],
    "sub": ["sub_self", "sub_zero", "zero_sub"],
    "div": ["div_self", "div_one", "one_div"],
    "neg": ["neg_add", "neg_mul", "neg_neg"],
    "eq": ["rfl", "Eq.refl", "Eq.symm", "Eq.trans", "Eq.subst"],
    "refl": ["rfl", "Eq.refl"],
    "symm": ["Eq.symm"],
    "trans": ["Eq.trans"],
    "imp": ["id", "Function.id"],
    "and": ["And.intro", "And.left", "And.right"],
    "or": ["Or.inl", "Or.inr"],
    "not": ["not_not_intro", "by_contra"],
    "forall": ["forall_intro"],
    "exists": ["Exists.intro"],
    "Nat": ["Nat.add_comm", "Nat.add_zero", "Nat.zero_add", "Nat.mul_comm",
             "Nat.mul_zero", "Nat.zero_mul", "Nat.succ_eq_add_one"],
    "Int": ["Int.add_comm", "Int.mul_comm"],
    "Real": ["Real.add_comm", "Real.mul_comm"],
    "zero": ["add_zero", "zero_add", "mul_zero", "zero_mul", "sub_self"],
    "one": ["mul_one", "one_mul", "div_one", "one_div"],
    "pow": ["pow_zero", "pow_one", "zero_pow", "one_pow"],
    "inv": ["inv_mul_cancel", "mul_inv_cancel"],
    "comm": ["add_comm", "mul_comm"],
    "assoc": ["add_assoc", "mul_assoc"],
    "distrib": ["mul_add", "add_mul", "left_distrib", "right_distrib"],
}


def _get_builtin_lemmas(keywords: list[str]) -> list[str]:
    """Return built-in lemmas matching the given keywords."""
    lemmas = []
    seen = set()
    for kw in keywords:
        for builtin in _BUILTIN_LEMMAS.get(kw, []):
            if builtin not in seen:
                lemmas.append(builtin)
                seen.add(builtin)
    return lemmas
