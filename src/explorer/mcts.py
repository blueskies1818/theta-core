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
    prior: float = 1.0

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
    num_simulations: int = 200

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
    dirichlet_weight: float = 0.25

    # Number of top lemmas to consider from GNN
    top_k_lemmas: int = 30

    # Whether to use the GNN for state evaluation
    use_gnn: bool = True

    # Whether to use the proof checker for validation
    use_proof_checker: bool = False

    # Maximum proof text length before pruning
    max_proof_length: int = 500


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
    ):
        self.gnn = gnn_encoder
        self.graph = dependency_graph
        self.config = config or MCTSConfig()

        # Cached GNN embeddings (populated before search)
        self._node_embeddings: torch.Tensor | None = None
        self._lemma_names: list[str] = []
        self._lemma_to_idx: dict[str, int] = {}

        # Node reuse (transposition table)
        self._state_cache: dict[int, MCTSNode] = {}

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

        # Add Dirichlet noise to root for exploration
        root_prior_noise = None

        # Identify relevant lemmas for this theorem from the graph
        available_lemmas = self._get_relevant_lemmas(theorem_statement)

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
        Creates child nodes with prior probabilities proportional to scores.
        """
        # Generate candidates
        candidates = generate_candidate_actions(
            node.state, available_lemmas, node.state.hypotheses
        )

        if not candidates:
            # No actions available — dead end
            node.state.is_dead = True
            return

        # Limit branching factor
        if len(candidates) > self.config.max_actions_per_node:
            candidates = random.sample(candidates, self.config.max_actions_per_node)

        # Score candidates using GNN (if available)
        if self.gnn is not None and self._node_embeddings is not None:
            priors = self._score_actions(node.state, candidates)
        else:
            # Uniform priors if no GNN
            priors = [1.0 / len(candidates)] * len(candidates)

        # Create child nodes
        for action, prior in zip(candidates, priors):
            child_state = node.state.apply_tactic(action)
            child = MCTSNode(
                state=child_state,
                parent=node,
                incoming_action=action,
                prior=prior,
            )
            node.children[action] = child

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
    ) -> list[float]:
        """Score candidate actions using hybrid GNN + structural signals.

        Combines:
        1. GNN embedding similarity (goal vs lemma) — if both in embedding space
        2. Lemma centrality (in-degree in dependency graph) — fundamental = useful
        3. Tactic-type prior — structural tactics get moderate prior
        """
        if self._node_embeddings is None:
            uniform = 1.0 / len(actions)
            return [uniform] * len(actions)

        goal_embedding = self._embed_goal(state)

        # Pre-compute centrality scores from the dependency graph
        if self.graph is not None:
            in_degrees = dict(self.graph.graph.in_degree())
            max_in = max(in_degrees.values()) if in_degrees else 1
        else:
            in_degrees = {}
            max_in = 1

        scores = []
        for action in actions:
            score = 0.05  # Base small prior for any action

            lemma = action.lemma
            if lemma and lemma in self._lemma_to_idx:
                idx = self._lemma_to_idx[lemma]
                lemma_emb = self._node_embeddings[idx]

                # GNN similarity: goal vs lemma
                if goal_embedding is not None and goal_embedding.norm() > 0:
                    sim = torch.cosine_similarity(
                        goal_embedding.unsqueeze(0), lemma_emb.unsqueeze(0), dim=1
                    )
                    score += max(0.0, sim.item()) * 0.4  # 40% weight on GNN

                # Centrality: fundamental lemmas are more useful
                centrality = in_degrees.get(lemma, 0) / max_in
                score += centrality * 0.3  # 30% weight on centrality

            elif action.tactic_type.value in ("intro", "cases", "have"):
                score += 0.25  # Structural tactics get moderate prior
            elif action.tactic_type.value == "exact" and action.hypothesis:
                score += 0.30  # Using a local hypothesis is promising

            scores.append(score)

        # Softmax to get prior probabilities (with temperature)
        scores_t = torch.tensor(scores)
        if scores_t.sum() > 0:
            priors = torch.softmax(scores_t / 0.5, dim=0)
            return priors.tolist()
        else:
            uniform = 1.0 / len(actions)
            return [uniform] * len(actions)

    def _embed_goal(self, state: ProofState) -> torch.Tensor | None:
        """Create an embedding for the current proof goal.

        Uses keyword-based pseudo-embedding: extracts mathematical keywords
        from the goal and averages GNN embeddings of matching lemmas.
        This grounds the goal embedding in the GNN's semantic space.
        """
        if self._node_embeddings is None or self.graph is None:
            return None

        goal_text = state.get_goal_embedding_key()
        keywords = _extract_math_keywords(goal_text)

        # Find lemma nodes whose names/statements match the keywords
        matching_indices = []
        for kw in keywords:
            for lemma_name, idx in self._lemma_to_idx.items():
                if kw.lower() in lemma_name.lower():
                    matching_indices.append(idx)
                    if len(matching_indices) >= 20:
                        break
            if len(matching_indices) >= 20:
                break

        if matching_indices:
            # Average the GNN embeddings of matching lemmas
            indices_t = torch.tensor(matching_indices, device=self._node_embeddings.device)
            return self._node_embeddings[indices_t].mean(dim=0)
        else:
            # Fallback: use a zero vector (will have low similarity to everything)
            return torch.zeros(self._node_embeddings.size(1), device=self._node_embeddings.device)

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

    def clear_cache(self) -> None:
        """Clear the transposition table and cached state."""
        self._state_cache.clear()


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
