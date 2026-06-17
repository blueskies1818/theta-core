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
    ):
        self.gnn = gnn
        self.graph = graph
        self.node_embeddings = node_embeddings  # [num_nodes, hidden_dim]
        self.lemma_index = lemma_index or {}  # lemma_name → node index
        self.idx_to_norm = idx_to_norm or {}  # node index → normalized conclusion
        self.config = config or GNNBestFirstConfig()
        self.proof_checker = proof_checker

        torch.set_num_threads(self.config.num_threads)
        self.gnn.eval()

        # L2-normalize node embeddings
        self.node_embeddings_norm = F.normalize(self.node_embeddings, dim=-1)

        # Build keyword → lemma index map for fast goal encoding
        self._kw_lemmas_map: dict[str, list[int]] = self._build_keyword_map()

        # Verification cache
        self._verification_cache: dict[tuple[int, int], bool] = {}
        self._tiebreaker: int = 0

        # Per-search state
        self._last_scores: list[float] = []
        self._lemma_names: list[str] = []  # Populated per theorem

    # ------------------------------------------------------------------
    # Main search
    # ------------------------------------------------------------------

    def search(
        self,
        theorem_statement: str,
        verbose: bool = False,
    ) -> tuple[list[Tactic], ProofState | None]:
        self._tiebreaker = 0
        self._verification_cache.clear()

        # Get relevant lemmas for this theorem
        available_lemmas = self._get_relevant_lemmas(theorem_statement)
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

        while heap and expansions < self.config.max_expansions:
            current = heapq.heappop(heap)
            state = current.state
            depth = current.depth

            if state.is_complete:
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
                priority = -(lemma_score / (1.0 + child_depth * self.config.depth_penalty))
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
        """
        from scripts.eval_gnn_prover import normalize_expression

        if self.node_embeddings is None:
            return None

        goal_text = state.get_goal_embedding_key()
        device = self.node_embeddings.device
        node_emb_norm = self.node_embeddings_norm

        goal_norm = normalize_expression(goal_text)

        # Detect reflexivity
        is_reflexive = False
        if "=" in goal_norm and "↔" not in goal_norm and "→" not in goal_norm and "≠" not in goal_norm:
            sides = goal_norm.split("=", 1)
            if len(sides) == 2 and sides[0].strip() == sides[1].strip():
                is_reflexive = True

        # Find exact structural matches
        exact_matches = set()
        for idx, lemma_norm in self.idx_to_norm.items():
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
            for idx, lemma_norm in self.idx_to_norm.items():
                lemma_stripped = re.sub(r'\s*\^\s*\d+', '', lemma_norm)
                if lemma_stripped == goal_stripped:
                    exact_matches.add(idx)

        # Subterm matches
        subterm_matches = set()
        if " = " in goal_norm:
            goal_lhs, goal_rhs = goal_norm.split(" = ", 1)
            for idx, lemma_norm in self.idx_to_norm.items():
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

        # Project through GoalEncoder
        if self.gnn.goal_encoder is not None:
            return self.gnn.encode_goal(context_emb)
        return F.normalize(context_emb, dim=-1) if context_emb.norm() > 0 else context_emb

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

    def _get_relevant_lemmas(self, theorem_statement: str) -> list[str]:
        """Get lemmas relevant to this theorem using keyword-based filtering.

        Uses the same algorithm as MCTS._get_relevant_lemmas:
        1. Extract math keywords from theorem statement.
        2. Filter graph lemmas by keyword match on name.
        3. Add built-in fundamental lemmas.
        4. Rank by structural centrality (in-degree).
        5. Return top candidates.
        """
        if self.graph is None:
            builtins = _get_builtin_lemmas(["eq", "refl"])
            return builtins[:self.config.top_k_lemmas]

        keywords = _extract_math_keywords(theorem_statement)
        builtins = _get_builtin_lemmas(keywords)

        # Filter graph lemmas by keyword match
        candidates: list[tuple[str, float]] = []
        seen: set[str] = set()

        for kw in keywords[:8]:
            for nid in self.graph.node_ids:
                if nid in seen:
                    continue
                attrs = self.graph.get_node(nid)
                name = attrs.get("name", nid) if attrs else nid
                if kw.lower() in name.lower():
                    kw_score = sum(1.0 for k in keywords if k.lower() in name.lower())
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
            for j, idx in enumerate(uncached_indices):
                single_code = uncached_codes[j]
                check_results = self.proof_checker.check_batch([single_code])
                cr = check_results[0]
                tactic = tactics[idx]
                if cr.success:
                    is_valid = True
                elif tactic.tactic_type.value in _incomplete_ok:
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
