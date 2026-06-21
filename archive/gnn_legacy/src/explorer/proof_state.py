"""Proof state representation for MCTS proof search (Phase 2.3).

A proof state captures the current goal, local context, and proof history.
The MCTS explores the tree of possible proof states, using the GNN to
evaluate which lemmas and tactics are promising at each step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class TacticType(Enum):
    """Categories of proof steps available to the explorer."""

    APPLY = "apply"  # apply a lemma to the goal
    REWRITE = "rewrite"  # rewrite using an equality lemma
    EXACT = "exact"  # exact proof term or lemma
    INTRO = "intro"  # introduce a hypothesis
    CASES = "cases"  # case analysis on a hypothesis
    HAVE = "have"  # create new hypothesis from lemma
    CALC = "calc"  # calculational proof step
    REFINE = "refine"  # refine the goal with a term (holes allowed)
    RING = "ring"  # ring tactic for polynomial identities
    FIELD_SIMP = "field_simp"  # field_simp for field equations
    LINARITH = "linarith"  # linarith for linear arithmetic
    NLINARITH = "nlinarith"  # nlinarith for nonlinear arithmetic
    SIMP = "simp"  # simplifier with hypotheses


@dataclass
class Tactic:
    """A single tactic application in a proof.

    Example:
        Tactic(TacticType.APPLY, lemma="add_comm")
        → generates: apply add_comm
    """

    tactic_type: TacticType
    lemma: str | None = None  # Lemma name to apply/rewrite/exact
    hypothesis: str | None = None  # Local hypothesis name
    args: list[str] = field(default_factory=list)  # Additional arguments

    def to_lean(self) -> str:
        """Render the tactic as Lean 4 code."""
        if self.tactic_type == TacticType.APPLY:
            return f"apply {self.lemma}"
        elif self.tactic_type == TacticType.REWRITE:
            if self.lemma:
                return f"rw [{self.lemma}]"
            if self.hypothesis:
                return f"rw [{self.hypothesis}]"
            return f"rw [{', '.join(self.args)}]"
        elif self.tactic_type == TacticType.EXACT:
            if self.lemma:
                return f"exact {self.lemma}"
            return f"exact {self.hypothesis or 'sorry'}"
        elif self.tactic_type == TacticType.INTRO:
            name = self.hypothesis or self.args[0] if self.args else "h"
            return f"intro {name}"
        elif self.tactic_type == TacticType.CASES:
            return f"cases {self.hypothesis or 'h'}"
        elif self.tactic_type == TacticType.HAVE:
            name = self.hypothesis or "h"
            return f"have {name} := {self.lemma or 'sorry'}"
        elif self.tactic_type == TacticType.CALC:
            return f"calc\n    ... = _ := {self.lemma or 'rfl'}"
        elif self.tactic_type == TacticType.REFINE:
            return f"refine {self.lemma or '?'}"
        elif self.tactic_type == TacticType.RING:
            return "ring"
        elif self.tactic_type == TacticType.FIELD_SIMP:
            args = [a for a in self.args if a]
            if self.hypothesis:
                args.append(self.hypothesis)
            if args:
                return f"field_simp [{', '.join(args)}]"
            return "field_simp"
        elif self.tactic_type == TacticType.LINARITH:
            args = [a for a in self.args if a]
            if self.hypothesis:
                args.append(self.hypothesis)
            if args:
                return f"linarith [{', '.join(args)}]"
            return "linarith"
        elif self.tactic_type == TacticType.NLINARITH:
            args = [a for a in self.args if a]
            if self.hypothesis:
                args.append(self.hypothesis)
            if args:
                return f"nlinarith [{', '.join(args)}]"
            return "nlinarith"
        elif self.tactic_type == TacticType.SIMP:
            args = [a for a in self.args if a]
            if self.hypothesis:
                args.append(self.hypothesis)
            if args:
                return f"simp [{', '.join(args)}]"
            return "simp"
        else:
            return "sorry"

    def __hash__(self) -> int:
        return hash(
            (self.tactic_type, self.lemma, self.hypothesis, tuple(self.args))
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Tactic):
            return False
        return (
            self.tactic_type == other.tactic_type
            and self.lemma == other.lemma
            and self.hypothesis == other.hypothesis
            and self.args == other.args
        )


@dataclass
class ProofState:
    """A state in the proof search tree.

    Represents a point in a proof where we have:
    - A goal to prove
    - Local hypotheses available
    - A history of steps taken so far

    The MCTS evaluates these states using the GNN and explores
    forward by applying tactics.
    """

    # The theorem being proved (original goal)
    theorem_statement: str

    # Current goal(s) — typically one, but can be multiple from `apply`
    goals: list[str] = field(default_factory=list)

    # Local hypotheses (name → type/statement)
    hypotheses: dict[str, str] = field(default_factory=dict)

    # Proof steps taken so far
    steps: list[Tactic] = field(default_factory=list)

    # Proof text generated so far (Lean 4 code)
    proof_so_far: str = ""

    # Whether this state represents a completed proof
    is_complete: bool = False

    # Whether this state is a dead end (contradiction or impossible goal)
    is_dead: bool = False

    # Error message if this state resulted from a failed step
    error: str = ""

    @classmethod
    def initial(cls, theorem_statement: str) -> "ProofState":
        """Create the initial proof state for a theorem.

        Parses the theorem statement to extract binder arguments
        (e.g., (h : a = b), (hc : c ≠ 0)) as initial hypotheses, so the
        MCTS can use them for rewrites, field_simp, linarith, etc.
        """
        hypotheses = cls._parse_binder_args(theorem_statement)
        return cls(
            theorem_statement=theorem_statement,
            goals=[theorem_statement],
            hypotheses=hypotheses,
        )

    @staticmethod
    def _parse_binder_args(statement: str) -> dict[str, str]:
        """Extract named binder arguments from a theorem statement.

        Parses patterns like:
            theorem name (h : a = b) (hc : c ≠ 0) : goal
        Returns:
            {'h': 'a = b', 'hc': 'c ≠ 0'}
        """
        import re
        hyps: dict[str, str] = {}

        # Strip "theorem name" prefix to get to the first binder
        # Match: "theorem <name>" then optional binder args before ":"
        # Use a regex to find all (name : type) binders
        # Pattern: parenthesized binder with a named argument: (name : type)
        binder_pattern = re.compile(r'\((\w+)\s*:\s*([^()]+?)\)')
        # Find the last ':' that marks the goal (after the binders)
        # Strategy: find all binder matches, then take their types

        for match in binder_pattern.finditer(statement):
            name = match.group(1)
            typ = match.group(2).strip()
            # Skip binders that are just type annotations (e.g., (ν T : ℝ))
            # Type-only binders have multiple names or look like type names
            if " " in name or name[0].isupper():
                # Likely a variable binder like (ν T : ℝ) — skip
                continue
            hyps[name] = typ

        return hyps

    def apply_tactic(self, tactic: Tactic) -> "ProofState":
        """Create a new state by applying a tactic.

        Note: This is a symbolic transition. In Phase 2.4, these are
        validated by the actual Lean proof checker.
        """
        new_steps = self.steps + [tactic]
        new_proof = self._render_proof(new_steps)

        # Heuristic goal update based on tactic type
        new_goals = self.goals.copy()
        new_hyps = self.hypotheses.copy()

        if tactic.tactic_type == TacticType.INTRO:
            # Intro removes the implication arrow and adds a hypothesis
            name = tactic.hypothesis or "h"
            if new_goals:
                goal = new_goals.pop(0)
                # If goal is "A → B", hypothesis is "A", new goal is "B"
                if "→" in goal:
                    parts = goal.split("→", 1)
                    new_hyps[name] = parts[0].strip()
                    new_goals.insert(0, parts[1].strip())
            if not new_goals:
                new_goals = ["..."]

        elif tactic.tactic_type == TacticType.EXACT:
            # Exact closes the current goal
            if new_goals:
                new_goals.pop(0)
            if not new_goals:
                new_goals = []

        elif tactic.tactic_type == TacticType.APPLY:
            # Apply transforms the goal — leaves subgoals
            if new_goals and tactic.lemma:
                # Approximation: apply replaces goal with lemma's premises
                new_goals.pop(0)
                # In reality, apply creates new goals from lemma premises
                new_goals.insert(0, "subgoal (apply " + tactic.lemma + ")")

        elif tactic.tactic_type == TacticType.REWRITE:
            # Rewrite transforms the goal
            if new_goals and (tactic.lemma or tactic.hypothesis or tactic.args):
                goal = new_goals.pop(0)
                # Preserve the original goal operators by keeping full text
                new_goals.insert(0, goal + " (rewritten)")

        elif tactic.tactic_type in (
            TacticType.RING,
            TacticType.FIELD_SIMP,
            TacticType.LINARITH,
            TacticType.NLINARITH,
            TacticType.SIMP,
        ):
            # Automation tactics close the current goal
            if new_goals:
                new_goals.pop(0)

        elif tactic.tactic_type == TacticType.HAVE:
            # Have adds a new hypothesis
            if tactic.hypothesis and tactic.lemma:
                new_hyps[tactic.hypothesis] = tactic.lemma

        return ProofState(
            theorem_statement=self.theorem_statement,
            goals=new_goals,
            hypotheses=new_hyps,
            steps=new_steps,
            proof_so_far=new_proof,
            is_complete=len(new_goals) == 0,
        )

    @staticmethod
    def _render_proof(steps: list[Tactic]) -> str:
        """Render proof steps as Lean 4 code."""
        if not steps:
            return ""
        lines = []
        for step in steps:
            lines.append("  " + step.to_lean())
        return "\n".join(lines)

    def get_goal_embedding_key(self) -> str:
        """Get a key for embedding this state's goal via GNN.

        For now, returns the first goal's text. In the full system,
        this would be the formal statement for GNN embedding.
        """
        return self.goals[0] if self.goals else self.theorem_statement

    def __hash__(self) -> int:
        return hash((self.theorem_statement, tuple(self.goals), tuple(self.steps)))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ProofState):
            return False
        return (
            self.theorem_statement == other.theorem_statement
            and self.goals == other.goals
            and self.steps == other.steps
        )


# ---------------------------------------------------------------------------
# Action space: generates candidate tactics for a proof state
# ---------------------------------------------------------------------------


def generate_candidate_actions(
    state: ProofState,
    available_lemmas: list[str],
    hypotheses: dict[str, str] | None = None,
) -> list[Tactic]:
    """Generate candidate tactic applications for a proof state.

    Uses the available lemmas (from the dependency graph neighborhood)
    and local hypotheses to propose possible next steps.

    Args:
        state: Current proof state.
        available_lemmas: Lemma names from the graph neighborhood.
        hypotheses: Local hypotheses available.

    Returns:
        List of candidate Tactic objects.
    """
    if hypotheses is None:
        hypotheses = state.hypotheses

    candidates: list[Tactic] = []

    # Exact: close the goal with a hypothesis
    for hyp_name in hypotheses:
        candidates.append(
            Tactic(TacticType.EXACT, hypothesis=hyp_name)
        )

    # Apply: use a lemma (preferred over exact for typeclass resolution)
    for lemma in available_lemmas[:20]:
        candidates.append(Tactic(TacticType.APPLY, lemma=lemma))

    # Rewrite: use an equality lemma
    for lemma in available_lemmas[:10]:
        if any(
            kw in lemma.lower() for kw in ("add", "mul", "eq", "comm", "assoc", "zero", "one")
        ):
            candidates.append(Tactic(TacticType.REWRITE, lemma=lemma))

    # Rewrite using local hypotheses (e.g., rw [h] where h: a=b)
    for hyp_name, hyp_type in hypotheses.items():
        if "=" in hyp_type or "↔" in hyp_type:
            candidates.append(Tactic(TacticType.REWRITE, hypothesis=hyp_name))

    # Intro: if the goal is an implication or forall
    if state.goals and ("→" in state.goals[0] or "∀" in state.goals[0]):
        candidates.append(Tactic(TacticType.INTRO, hypothesis="h"))

    # Cases: if there are hypotheses to analyze.
    # Skip non-inductive hypotheses — `cases` only works on inductive types.
    # Equality (=, ≠), iff (↔), and inequalities (≤, ≥, <, >) are not inductive.
    for hyp_name, hyp_type in list(hypotheses.items())[:3]:
        if any(op in hyp_type for op in ("=", "≠", "↔", "≤", "≥", "<", ">")):
            continue
        candidates.append(Tactic(TacticType.CASES, hypothesis=hyp_name))

    # Have: duplicate an existing hypothesis under a fresh name
    for hyp_name in list(hypotheses.keys())[:3]:
        candidates.append(
            Tactic(TacticType.HAVE, lemma=hyp_name, hypothesis="h_" + hyp_name)
        )

    # Automation tactics for goals they match.
    # Skip goals containing → or ∀ — automation tactics (ring, field_simp,
    # linarith) operate on arithmetic goals, not implications or foralls.
    if state.goals:
        goal = state.goals[0]
        has_implication = "→" in goal or "∀" in goal

        # ring: polynomial/ring identities
        if "=" in goal and any(op in goal for op in ("*", "^", "+")) and not has_implication:
            candidates.append(Tactic(TacticType.RING))

        # field_simp: field equations with division or inverses
        if ("/" in goal or "⁻¹" in goal) and not has_implication:
            nonzero_hyps = [
                name for name, typ in hypotheses.items()
                if "≠" in typ or "0 ≠" in typ or "h" in name.lower()
            ]
            candidates.append(Tactic(
                TacticType.FIELD_SIMP,
                args=nonzero_hyps[:3],
            ))

        # linarith: linear arithmetic
        if any(op in goal for op in ("≤", "≥", "<", ">", "=")) and not has_implication:
            candidates.append(Tactic(TacticType.LINARITH))

        # simp: general simplification.
        # Only pass hypotheses that are equalities or simple propositions.
        # Complex binder args (functions, non-propositional variables) cause
        # "Invalid argument: Variable X is not a proposition" errors.
        simp_hyps = [
            name for name, typ in hypotheses.items()
            if "=" in typ or "↔" in typ or "→" in typ
            or typ.strip() in ("True", "False")
        ]
        candidates.append(Tactic(
            TacticType.SIMP,
            args=simp_hyps[:5],
        ))

    return candidates
