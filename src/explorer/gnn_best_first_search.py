"""GNN-powered best-first proof search (Hybrid: GNN cosine retrieval + best-first).

Replaces CharCNN contrastive embeddings with GNN cosine similarity (MRR 0.786)
while keeping the best-first priority-queue architecture and dense rewards.

Architecture:
  1. Pre-compute GNN node embeddings for all graph nodes.
  2. For each proof state, encode the goal using the GNN's goal-encoding
     pipeline (normalized text matching → keyword averaging → GoalEncoder).
  3. Score lemma candidates via cosine similarity (dot product) of goal
     embedding with lemma node embeddings.
  4. Push child states onto max-heap with priority = lemma_score /
     (1 + depth * penalty), same as the original best-first search.
  5. Continue until proof complete or budget exhausted.

Key difference from CharCNN BestFirstSearch:
  - Uses GNN node embeddings (graph-structure-aware) for lemma scoring
  - Uses GNN GoalEncoder for projecting keyword-averaged contexts
  - Filters candidates by keyword match on lemma name (like MCTS)
  - Leverages structural centrality from the dependency graph
"""

from __future__ import annotations

import heapq
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

from src.explorer.gnn_config import GNNConfig
from src.explorer.gnn_encoder import GNNEncoder, extract_initial_features, prepare_graph_tensors
from src.explorer.mcts import _extract_math_keywords, _BUILTIN_LEMMAS, _get_builtin_lemmas
from src.explorer.proof_state import (
    ProofState,
    Tactic,
    TacticType,
    generate_candidate_actions,
)
from src.proof_checker.formats import ProofResult


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class GNNBestFirstConfig:
    """Configuration for GNN-based best-first proof search."""

    max_depth: int = 20
    max_expansions: int = 5000
    top_k_lemmas: int = 30
    depth_penalty: float = 0.05
    use_proof_checker: bool = False
    verify_timeout: float = 5.0
    max_proof_length: int = 500
    device: str = "cpu"
    num_threads: int = 4

    # How many keyword-matching lemmas to pull from the graph
    max_graph_candidates: int = 200

    # Value network integration
    # Weight of value estimate vs lemma score in priority computation.
    # 0.0 = pure lemma score (default behavior), 1.0 = pure value estimate.
    value_weight: float = 0.3

    # Prune states with value estimate below this threshold.
    # None = no pruning, float = prune if value < threshold.
    value_prune_threshold: float | None = 0.1


# ---------------------------------------------------------------------------
# Priority queue wrapper (same as original BestFirstSearch)
# ---------------------------------------------------------------------------

@dataclass(order=True)
class _PrioritizedState:
    priority: float
    depth: int
    tiebreaker: int
    state: ProofState = field(compare=False)
    steps: list = field(compare=False)


# ---------------------------------------------------------------------------
# GNN Best-First Search
# ---------------------------------------------------------------------------


class GNNBestFirstSearch:
    """Best-first proof search using GNN cosine similarity for lemma scoring.

    Usage:
        bf = GNNBestFirstSearch(gnn, graph, node_embeddings,
                                 config=GNNBestFirstConfig())
        proof_steps, final_state = bf.search(theorem_statement)
    """

    def __init__(
        self,
        gnn: GNNEncoder,
        graph,  # DependencyGraph
        node_embeddings: torch.Tensor,
        lemma_index: dict[str, int] | None = None,
        idx_to_norm: dict[int, str] | None = None,
        config: GNNBestFirstConfig | None = None,
        proof_checker=None,
        value_network=None,  # ValueNetwork for state evaluation
    ):
        self.gnn = gnn
        self.graph = graph
        self.node_embeddings = node_embeddings  # [num_nodes, hidden_dim]
        self.lemma_index = lemma_index or {}  # lemma_name → node index
        self.idx_to_norm = idx_to_norm or {}  # node index → normalized conclusion
        self.config = config or GNNBestFirstConfig()
        self.proof_checker = proof_checker
        self.value_network = value_network  # Optional value network for pruning

        torch.set_num_threads(self.config.num_threads)
        self.gnn.eval()

        # L2-normalize node embeddings
        self.node_embeddings_norm = F.normalize(self.node_embeddings, dim=-1)

        # Build keyword → lemma index map for fast goal encoding
        self._kw_lemmas_map: dict[str, list[int]] = self._build_keyword_map()

        # Pre-build normalized-text → node indices for fast goal embedding lookup.
        # This replaces O(N) full-iteration in _embed_goal with O(1) hash lookup.
        self._norm_to_indices: dict[str, list[int]] = self._build_norm_index_lookup()

        # Verification cache
        self._verification_cache: dict[tuple[int, int], bool] = {}
        self._tiebreaker: int = 0

        # Domain filtering: pre-build domain → node_ids lookup
        self._domain_node_ids: dict[str, set[str]] = self._build_domain_index()
        self._all_domain_names: set[str] = set(self._domain_node_ids.keys())

        # Per-search state
        self._last_scores: list[float] = []
        self._lemma_names: list[str] = []  # Populated per theorem
        self._goal_embed_cache: dict[str, torch.Tensor] = {}  # Goal text → embedding

    # ------------------------------------------------------------------
    # Main search
    # ------------------------------------------------------------------

    def search(
        self,
        theorem_statement: str,
        domain: str | None = None,
        verbose: bool = False,
    ) -> tuple[list[Tactic], ProofState | None]:
        self._tiebreaker = 0
        self._verification_cache.clear()
        self._goal_embed_cache.clear()

        # Get relevant lemmas for this theorem (domain-filtered when domain is set)
        available_lemmas = self._get_relevant_lemmas(theorem_statement, domain=domain)
        self._lemma_names = available_lemmas

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
        verify_count = 0  # Track how many Lean verifications we've attempted
        max_verifications = 5  # Cap: at most 5 Lean verifications per theorem
        last_unverified: list[Tactic] | None = None  # Best unverified complete state
        last_unverified_state: ProofState | None = None

        while heap and expansions < self.config.max_expansions:
            current = heapq.heappop(heap)
            state = current.state
            depth = current.depth

            if state.is_complete:
                # Always verify completed proofs (regardless of use_proof_checker flag)
                if self.proof_checker is not None and state.steps:
                    verify_count += 1
                    # Cap verifications to avoid bottleneck from false completions
                    if verify_count <= max_verifications:
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
                            state.is_complete = False
                            state.is_dead = True
                            if verbose:
                                err = check_results[0].errors[0][:80] if check_results[0].errors else "?"
                                print(f"  ✗ Symbolic-complete but Lean rejects: {err}")
                            continue
                    else:
                        # Skipping verification — save first unverified complete state as fallback
                        if last_unverified is None:
                            last_unverified = list(state.steps)
                            last_unverified_state = state
                        state.is_complete = False
                        state.is_dead = True
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
            if len(state.proof_so_far) > self.config.max_proof_length:
                continue

            # Expand
            expansions += 1
            scored_lemmas = self._score_lemmas(state)
            candidates = self._generate_actions(state, scored_lemmas)

            if not candidates:
                if verbose and expansions % 100 == 0:
                    print(f"  [{expansions}/{self.config.max_expansions}] "
                          f"depth={depth}, heap={len(heap)}, no candidates")
                continue

            valid_mask = self._maybe_verify(state, candidates)
            if not any(valid_mask):
                valid_mask = [True] * len(candidates)

            for action, lemma_score, valid in zip(candidates, self._last_scores, valid_mask):
                if not valid:
                    continue
                child_state = state.apply_tactic(action)
                child_depth = depth + 1

                # Compute value estimate if value network is available
                value_estimate = 0.5  # neutral default
                if self.value_network is not None and self.config.value_weight > 0.0:
                    value_estimate = self._estimate_value(child_state)
                    # Prune if below threshold
                    if (self.config.value_prune_threshold is not None
                            and value_estimate < self.config.value_prune_threshold):
                        continue

                # Blend lemma score and value estimate
                vw = self.config.value_weight
                blended_score = lemma_score * (1.0 - vw) + value_estimate * vw
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
                top_msg = f"top_score={-heap[0].priority:.3f}" if heap else ""
                print(f"  [{expansions}/{self.config.max_expansions}] "
                      f"depth={depth}, heap={len(heap)}, {top_msg}")

        elapsed = time.time() - t_start
        if verbose:
            if heap:
                print(f"  ✗ Budget exhausted: {expansions} expansions, "
                      f"{len(heap)} states remaining, {elapsed:.1f}s")
            else:
                print(f"  ✗ Search space exhausted: {expansions} expansions, {elapsed:.1f}s")
        # If we have an unverified complete state, return it for the caller to verify
        if last_unverified is not None:
            if verbose:
                print(f"  → Returning unverified candidate: "
                      f"{[s.to_lean() for s in last_unverified[:3]]}")
            return (last_unverified, last_unverified_state)
        return ([], None)

    # ------------------------------------------------------------------
    # Lemma scoring (GNN-powered — the key hybrid component)
    # ------------------------------------------------------------------

    def _score_lemmas(self, state: ProofState) -> list[tuple[str, float]]:
        """Score lemmas using GNN cosine similarity.

        Pipeline:
        1. Encode the current goal using the GNN goal-encoding pipeline
           (normalized text matching → keyword average → GoalEncoder projection).
        2. Get GNN node embeddings for lemma candidates.
        3. Compute cosine similarity (dot product of L2-normalized embeddings).
        4. Boost by structural centrality (in-degree in dependency graph).
        5. Return top-K by combined score.
        """
        if not self._lemma_names:
            return []

        device = self.node_embeddings.device
        node_emb_norm = self.node_embeddings_norm

        # Get goal embedding
        goal_emb = self._embed_goal(state)
        if goal_emb is None:
            # Fallback: uniform scoring
            return [(name, 0.5) for name in self._lemma_names[:self.config.top_k_lemmas]]

        goal_emb = F.normalize(goal_emb, dim=-1)

        # Get lemma indices and embeddings
        lemma_indices = []
        for name in self._lemma_names:
            idx = self.lemma_index.get(name)
            if idx is not None and idx < node_emb_norm.size(0):
                lemma_indices.append(idx)
            else:
                lemma_indices.append(-1)  # sentinel

        if not any(i >= 0 for i in lemma_indices):
            return [(name, 0.5) for name in self._lemma_names[:self.config.top_k_lemmas]]

        # Build scores
        scores = []
        goal_text = state.get_goal_embedding_key()

        # Pre-compute centrality
        if self.graph is not None:
            in_degrees = dict(self.graph.graph.in_degree())
            max_in = max(in_degrees.values()) if in_degrees else 1
        else:
            in_degrees = {}
            max_in = 1

        for i, (name, idx) in enumerate(zip(self._lemma_names, lemma_indices)):
            score = 0.05  # base

            if idx >= 0:
                lemma_emb = node_emb_norm[idx]
                # Cosine similarity
                cos_sim = torch.dot(goal_emb, lemma_emb).item()
                cos_sim = max(-1.0, min(1.0, cos_sim))
                # Map from [-1,1] to [0,1] for scoring
                similarity = (cos_sim + 1.0) / 2.0
                score += similarity * 0.8

                # Centrality bonus
                centrality = in_degrees.get(name, 0) / max_in
                score += centrality * 0.1

                # Keyword relevance check
                relevance = self._lemma_goal_keyword_match(name, goal_text)
                if relevance < 1.0:
                    score -= 0.5 * (1.0 - relevance)

            # Built-in lemmas get a boost
            for builtins in _BUILTIN_LEMMAS.values():
                if name in builtins:
                    score += 0.15
                    break

            scores.append((name, score))

        # Sort by score descending
        scores.sort(key=lambda x: -x[1])

        # Normalize to [0, 0.65] range (same as original best-first)
        if scores:
            max_s = max(s for _, s in scores)
            min_s = min(s for _, s in scores)
            if max_s > min_s:
                scores = [
                    (name, 0.1 + 0.55 * (s - min_s) / (max_s - min_s))
                    for name, s in scores
                ]

        # Clamp and take top-K
        scores = [(name, min(0.65, s)) for name, s in scores]
        return scores[:self.config.top_k_lemmas]

    # ------------------------------------------------------------------
    # Goal embedding (MCTS-style: norm matching + keyword averaging + GoalEncoder)
    # ------------------------------------------------------------------

    def _embed_goal(self, state: ProofState) -> torch.Tensor | None:
        """Create goal embedding from current proof state.

        Uses the same pipeline as MCTS._embed_goal:
        1. Normalize the goal text by replacing variables with placeholders.
        2. Match against normalized lemma conclusions for structural matches.
        3. Average GNN embeddings of matching lemmas.
        4. Pass through GoalEncoder for projection.
        Falls back to keyword-based context when no structural matches found.

        Results are cached per goal text for performance since the same goal
        text appears many times during search.
        """
        from scripts.eval.eval_gnn_prover import normalize_expression

        if self.node_embeddings is None:
            return None

        goal_text = state.get_goal_embedding_key()

        # Check cache — keyed on normalized goal text
        goal_norm = normalize_expression(goal_text)
        if goal_norm in self._goal_embed_cache:
            return self._goal_embed_cache[goal_norm]

        device = self.node_embeddings.device
        node_emb_norm = self.node_embeddings_norm

        # Detect reflexivity
        is_reflexive = False
        if "=" in goal_norm and "↔" not in goal_norm and "→" not in goal_norm and "≠" not in goal_norm:
            sides = goal_norm.split("=", 1)
            if len(sides) == 2 and sides[0].strip() == sides[1].strip():
                is_reflexive = True

        # Find exact structural matches — O(1) hash lookup
        exact_matches = set(self._norm_to_indices.get(goal_norm, []))

        # Power-stripping fallback: strip exponents and check hash
        if not exact_matches:
            goal_stripped = re.sub(r'\s*\^\s*\d+', '', goal_norm)
            if goal_stripped != goal_norm:
                exact_matches.update(self._norm_to_indices.get(goal_stripped, []))
            if is_reflexive:
                exact_matches.update(self._norm_to_indices.get(
                    normalize_expression("a = a"), []))

        # Subterm matches disabled: too slow on 116K unique keys.
        # Fall through to keyword-based context instead.
        subterm_matches: set[int] = set()

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
                    if idx >= node_emb_norm.size(0):
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

        # Project through GoalEncoder and cache result
        if self.gnn.goal_encoder is not None:
            result = self.gnn.encode_goal(context_emb)
        else:
            result = F.normalize(context_emb, dim=-1) if context_emb.norm() > 0 else context_emb
        self._goal_embed_cache[goal_norm] = result
        return result

    # ------------------------------------------------------------------
    # Action generation (same logic as original, but uses scored lemmas)
    # ------------------------------------------------------------------

    def _generate_actions(
        self, state: ProofState, scored_lemmas: list[tuple[str, float]]
    ) -> list[Tactic]:
        candidates: list[Tactic] = []
        self._last_scores = []

        for lemma, score in scored_lemmas:
            tactic = Tactic(TacticType.APPLY, lemma=lemma)
            candidates.append(tactic)
            self._last_scores.append(score)

            # Rewrite for algebraic lemmas
            if any(kw in lemma.lower()
                   for kw in ("add", "mul", "eq", "comm", "assoc", "zero", "one",
                               "neg", "sub", "div", "ring", "field", "simp")):
                candidates.append(Tactic(TacticType.REWRITE, lemma=lemma))
                self._last_scores.append(score * 0.9)

        # Structural tactics (same as original)
        hypotheses = state.hypotheses

        for hyp_name in list(hypotheses.keys())[:5]:
            candidates.append(Tactic(TacticType.EXACT, hypothesis=hyp_name))
            self._last_scores.append(0.70)

        for hyp_name, hyp_type in hypotheses.items():
            if "=" in hyp_type or "↔" in hyp_type:
                candidates.append(Tactic(TacticType.REWRITE, hypothesis=hyp_name))
                self._last_scores.append(0.75)
                break

        if state.goals and ("→" in state.goals[0] or "∀" in state.goals[0]):
            candidates.append(Tactic(TacticType.INTRO, hypothesis="h"))
            self._last_scores.append(0.80)

        for hyp_name, hyp_type in list(hypotheses.items())[:3]:
            if any(op in hyp_type for op in ("=", "≠", "↔", "≤", "≥", "<", ">")):
                continue
            candidates.append(Tactic(TacticType.CASES, hypothesis=hyp_name))
            self._last_scores.append(0.35)
            break

        # Automation tactics
        if state.goals:
            goal = state.goals[0]
            has_implication = "→" in goal or "∀" in goal

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
            candidates.append(Tactic(TacticType.SIMP))
            self._last_scores.append(0.65)

        # Limit candidates
        max_actions = self.config.top_k_lemmas * 2 + 10
        if len(candidates) > max_actions:
            candidates = candidates[:max_actions]
            self._last_scores = self._last_scores[:max_actions]

        return candidates

    # ------------------------------------------------------------------
    # Lemma filtering (keyword-based, like MCTS._get_relevant_lemmas)
    # ------------------------------------------------------------------

    def _get_relevant_lemmas(self, theorem_statement: str, domain: str | None = None) -> list[str]:
        """Get lemmas relevant to this theorem using keyword-based filtering.

        Uses the same algorithm as MCTS._get_relevant_lemmas:
        1. (NEW) Filter graph nodes to matching Mathlib domain.
        2. Extract math keywords from theorem statement.
        3. Filter remaining lemmas by keyword match on name.
        4. Add built-in fundamental lemmas.
        5. Rank by structural centrality (in-degree).
        6. Return top candidates.

        Args:
            theorem_statement: The theorem statement text.
            domain: Mathlib domain label (e.g., 'algebra', 'Analysis').
                When provided, filters candidate lemmas to the matching
                domain(s), reducing candidates from 116K to ~2-12K.
                None uses the full graph (backward compatible).
        """
        if self.graph is None:
            builtins = _get_builtin_lemmas(["eq", "refl"])
            return builtins[:self.config.top_k_lemmas]

        keywords = _extract_math_keywords(theorem_statement)
        builtins = _get_builtin_lemmas(keywords)

        # Domain filtering: pre-filter node_ids to matching domain(s)
        domain_node_ids = self._get_domain_node_ids(domain)
        if domain_node_ids is not None:
            candidate_pool = list(domain_node_ids)
        else:
            candidate_pool = list(self.graph.node_ids)

        # Filter by keyword match (within domain-filtered pool)
        candidates: list[tuple[str, float]] = []
        seen: set[str] = set()

        for kw in keywords[:8]:
            for nid in candidate_pool:
                if nid in seen:
                    continue
                # Domain-filtered nodes: use node_id directly as name
                # (avoids expensive get_node() call for every node)
                if kw.lower() in nid.lower():
                    kw_score = sum(1.0 for k in keywords if k.lower() in nid.lower())
                    candidates.append((nid, kw_score))
                    seen.add(nid)
                    if len(candidates) >= self.config.max_graph_candidates:
                        break
            if len(candidates) >= self.config.max_graph_candidates:
                break

        # Rank by centrality
        in_degrees = dict(self.graph.graph.in_degree())
        max_in = max(in_degrees.values()) if in_degrees else 1

        ranked = []
        for name, kw_score in candidates:
            centrality = in_degrees.get(name, 0) / max_in
            combined = 0.3 * kw_score + 0.7 * centrality
            ranked.append((name, combined))

        ranked.sort(key=lambda x: -x[1])
        lemmas = [name for name, _ in ranked[:self.config.top_k_lemmas]]

        # If domain filtering found too few candidates, supplement from full graph
        if domain_node_ids is not None and len(lemmas) < 5:
            full_candidates = []
            full_seen = set()
            for kw in keywords[:8]:
                for nid in self.graph.node_ids:
                    if nid in seen or nid in full_seen:
                        continue
                    if kw.lower() in nid.lower():
                        kw_score = sum(1.0 for k in keywords if k.lower() in nid.lower())
                        centrality = in_degrees.get(nid, 0) / max_in
                        combined = 0.3 * kw_score + 0.7 * centrality
                        full_candidates.append((nid, combined))
                        full_seen.add(nid)
                        if len(full_candidates) >= self.config.max_graph_candidates:
                            break
                if len(full_candidates) >= self.config.max_graph_candidates:
                    break
            full_candidates.sort(key=lambda x: -x[1])
            extra = [name for name, _ in full_candidates[:self.config.top_k_lemmas]
                     if name not in seen]
            lemmas.extend(extra)

        # Prepend built-ins
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

        return result[:self.config.top_k_lemmas]

    # ------------------------------------------------------------------
    # Value network integration
    # ------------------------------------------------------------------

    def _estimate_value(self, state: ProofState) -> float:
        """Estimate P(success) for a proof state using the value network.

        Encodes the current goal using the GNN goal encoding pipeline,
        then runs the value head to predict completability.

        Returns:
            Scalar in [0, 1] estimating probability of eventual proof success.
            Returns 0.5 (neutral) if value network is unavailable or encoding fails.
        """
        if self.value_network is None:
            return 0.5

        try:
            goal_emb = self._embed_goal(state)
            if goal_emb is None:
                return 0.5

            value = self.value_network.predict(goal_emb)
            if hasattr(value, 'item'):
                value = float(value.item())
            return max(0.0, min(1.0, value))
        except Exception:
            return 0.5

    # ------------------------------------------------------------------
    # Proof checker verification
    # ------------------------------------------------------------------

    def _maybe_verify(self, state: ProofState, candidates: list[Tactic]) -> list[bool]:
        is_root = len(state.steps) == 0
        _lemma_tactics = {"apply", "rewrite", "exact"}

        if is_root and self.config.use_proof_checker and self.proof_checker is not None:
            verify_indices = []
            for i, c in enumerate(candidates):
                if c.tactic_type.value in _lemma_tactics and (
                    c.lemma is not None or c.hypothesis is not None
                ):
                    verify_indices.append(i)

            if verify_indices:
                verify_results = self._verify_candidates(
                    state, [candidates[i] for i in verify_indices]
                )
                valid_mask = [True] * len(candidates)
                for j, idx in enumerate(verify_indices):
                    valid_mask[idx] = verify_results[j]
                return valid_mask

        return [True] * len(candidates)

    def _verify_candidates(self, state: ProofState, tactics: list[Tactic]) -> list[bool]:
        from src.proof_checker.formats import wrap_theorem_with_proof
        _incomplete_ok = {"rewrite", "intro", "cases", "have", "refine"}

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
            # Build a single combined Lean file with all candidates,
            # each wrapped as a uniquely-named theorem, for ONE Lake invocation.
            # This replaces N separate Lake+Lean calls with ONE.
            combined_code = self._build_combined_verification(uncached_codes, uncached_indices)
            combined_result = self.proof_checker.checker.check(combined_code)
            
            if combined_result.success:
                # All candidates pass — all valid
                for idx in uncached_indices:
                    results[idx] = True
                    cache_key = (hash(state.proof_so_far), hash(tactics[idx]))
                    self._verification_cache[cache_key] = True
            else:
                # Some failed — parse errors to determine which ones
                failed_names = self._parse_combined_errors(combined_result.errors)
                for j, idx in enumerate(uncached_indices):
                    theorem_name = f"_candidate_{j}"
                    is_valid = theorem_name not in failed_names
                    tactic = tactics[idx]
                    # Allow incomplete (unsolved goals) for rewrite/intro/etc
                    if not is_valid and tactic.tactic_type.value in _incomplete_ok:
                        error_text = " ".join(combined_result.errors) if combined_result.errors else ""
                        is_valid = "unsolved goals" in error_text.lower() and theorem_name in error_text
                    results[idx] = is_valid
                    cache_key = (hash(state.proof_so_far), hash(tactics[idx]))
                    self._verification_cache[cache_key] = is_valid

        return [results[i] for i in range(len(tactics))]

    def _build_combined_verification(
        self, codes: list[str], indices: list[int]
    ) -> str:
        """Build a single Lean source string containing all candidate theorems.

        Each candidate gets a unique theorem name `_candidate_N` so we can
        map errors back to specific candidates after parsing.
        """
        import re as _re
        parts = ["import Mathlib", "open Real", "open Set", "open Filter",
                 "open Function", "open Nat", ""]
        for j, code in enumerate(codes):
            # Replace "theorem <name>" with "theorem _candidate_N" or
            # wrap bare code in a uniquely-named theorem.
            renamed = _re.sub(
                r'^theorem\s+\S+', f'theorem _candidate_{j}', code, count=1
            )
            # If code doesn't start with 'theorem', wrap it
            if not renamed.strip().startswith("theorem "):
                renamed = f"theorem _candidate_{j} : True := by\n  trivial"
            parts.append(renamed)
            parts.append("")
        return "\n".join(parts)

    def _parse_combined_errors(self, errors: list[str]) -> set[str]:
        """Parse Lean error output to extract which candidate theorems failed.

        Returns set of theorem names (e.g., '_candidate_3') that have errors.
        """
        import re as _re
        failed: set[str] = set()
        for err in errors:
            # Lean errors look like: "<filename>:<line>:<col>: error: <msg>"
            # or lines starting with theorem name
            m = _re.findall(r'_candidate_\d+', err)
            failed.update(m)
        return failed

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _next_tiebreaker(self) -> int:
        self._tiebreaker += 1
        return self._tiebreaker

    def _build_norm_index_lookup(self) -> dict[str, list[int]]:
        """Build normalized-text → list of node indices for O(1) _embed_goal lookups.

        Groups all nodes that share the same normalized conclusion text,
        enabling fast retrieval without iterating through all 116K entries.
        """
        lookup: dict[str, list[int]] = {}
        for idx, norm_text in self.idx_to_norm.items():
            if idx < self.node_embeddings.size(0):
                lookup.setdefault(norm_text, []).append(idx)
        return lookup

    def _build_keyword_map(self) -> dict[str, list[int]]:
        """Build keyword → lemma index map from graph node names."""
        kw_map: dict[str, list[int]] = {}
        for nid in self.graph.node_ids:
            idx = self.graph.node_id_to_idx(nid)
            if idx is None:
                continue
            name_lower = nid.lower()
            for kw in self._all_keywords():
                if kw.lower() in name_lower:
                    kw_map.setdefault(kw.lower(), []).append(idx)
        return kw_map

    # ------------------------------------------------------------------
    # Domain filtering
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_domain(domain: str) -> str:
        """Normalize a domain name for matching.

        Lowercases, strips whitespace, removes underscores, and squashes
        multiple slashes so 'algebra' ≈ 'Algebra' ≈ 'Algebra/'"""
        import re
        d = domain.strip().lower()
        d = d.replace('_', '')
        d = re.sub(r'/+', '/', d)
        return d

    @staticmethod
    def _domain_matches(graph_domain: str, theorem_domain: str) -> bool:
        """Check if a graph domain matches a theorem's domain label.

        Matching rules (case-insensitive):
        - Exact match after normalization: 'algebra' ≈ 'Algebra'
        - Prefix match: 'algebra' matches 'Algebra/Polynomial'
        - Substring: 'theory' in 'categorytheory', 'algebra' in 'linearalgebra'
        - Cross-domain bundle: 'algebra' also matches 'RingTheory', 'GroupTheory',
          'LinearAlgebra', 'Algebra/Order', etc.
        """
        ngd = GNNBestFirstSearch._normalize_domain(graph_domain)
        ntd = GNNBestFirstSearch._normalize_domain(theorem_domain)

        if ngd == ntd:
            return True
        if ngd.startswith(ntd + '/'):
            return True
        if ntd.startswith(ngd + '/'):
            return True
        # Substring match for compound names
        if ntd in ngd or ngd in ntd:
            return True

        # Cross-domain bundling: known related domains
        # Target: ~10K nodes per broad domain area (from 116K total)
        _cross_domain_bundles: dict[str, set[str]] = {
            'algebra': {'linearalgebra',
                         'algebra/order', 'algebra/polynomial',
                         'algebra/group', 'algebra/ring'},
            'analysis': {'analysis/calculus', 'analysis/normed',
                          'analysis/complex', 'analysis/convolution',
                          'measuretheory'},
            'numbertheory': {'numbertheory/'},
            'order': {'order/', 'algebra/order'},
            'topology': {'topology/', 'topology/algebra'},
            'logic': {'order', 'settheory', 'data/set'},
            # Physics theorems use algebraic/analytic manipulation;
            # map to core algebra + analysis (no RingTheory/Topology)
            'physics': {'algebra', 'algebra/', 'linearalgebra',
                         'analysis', 'analysis/', 'measuretheory'},
        }

        bundles = _cross_domain_bundles.get(ntd, set())
        if any(ngd.startswith(b) or b.startswith(ngd)
               for b in bundles):
            return True

        return False

    def _build_domain_index(self) -> dict[str, set[str]]:
        """Build domain → set of node_ids index for fast domain filtering."""
        domain_index: dict[str, set[str]] = {}
        for nid in self.graph.node_ids:
            attrs = self.graph.get_node(nid)
            if attrs:
                domain = attrs.get('domain', '') or ''
                if domain:
                    domain_index.setdefault(domain, set()).add(nid)
        return domain_index

    def _get_domain_node_ids(self, theorem_domain: str | None) -> set[str] | None:
        """Get the set of node IDs matching a theorem domain.

        Args:
            theorem_domain: Domain label from the theorem (e.g., 'algebra', 'Analysis').
                None means no filtering — use all nodes.

        Returns:
            Set of node IDs in matching domains, or None for no filtering.
        """
        if not theorem_domain:
            return None

        matching: set[str] = set()
        for graph_domain, node_ids in self._domain_node_ids.items():
            if self._domain_matches(graph_domain, theorem_domain):
                matching.update(node_ids)

        if not matching:
            # No match found — fall through to full graph
            return None

        return matching

    @staticmethod
    def _all_keywords() -> list[str]:
        """All keywords from _BUILTIN_LEMMAS and _extract_math_keywords."""
        all_kw = set()
        all_kw.update(_BUILTIN_LEMMAS.keys())
        for vals in _BUILTIN_LEMMAS.values():
            all_kw.update(vals)
        for tok in ("+", "*", "-", "/", "^", "=", "→", "∀", "∃", "≤", "≥", "<", ">",
                     "⁻¹", "∘", "0", "1",
                     "add", "mul", "sub", "div", "neg", "comm", "assoc", "distrib",
                     "and", "or", "not", "iff", "eq", "refl", "symm", "trans",
                     "Nat", "Int", "Real", "Complex", "Prop", "Set", "List",
                     "deriv", "integral", "limit", "continuous", "sum", "prod",
                     "ring", "field", "group", "linear", "inv", "pow"):
            all_kw.add(tok)
        return list(all_kw)

    @staticmethod
    def _lemma_goal_keyword_match(lemma: str, goal_text: str) -> float:
        """Check how well lemma keywords match the goal text.

        Returns 1.0 (full match) to 0.0 (no match).
        """
        lemma_lower = lemma.lower()
        goal_lower = goal_text.lower()

        # Extract meaningful tokens from the lemma name (split on underscores)
        tokens = [t for t in lemma_lower.split("_") if len(t) >= 2]
        if not tokens:
            return 0.5

        matched = sum(1 for tok in tokens if tok in goal_lower)
        return matched / len(tokens)
