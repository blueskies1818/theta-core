"""Expression constancy evaluator for physics discovery.

Scores a candidate expression against one or more physical observations
by evaluating the expression at each timestep and measuring how constant
the result is.

Scoring formula (from plan Section 4.1):
    For each observation:
        values = [eval(expr, timestep) for each timestep]
        constancy = 1.0 / (1.0 + std(values) / |mean(values)|)
    Aggregate: mean constancy across all observations

Perfect constancy -> 1.0, random noise -> ~0.5, anti-correlated -> low.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable

from src.physics.observations import Observation, ObservationDatabase


# ── Tokenizer / Parser ───────────────────────────────────────────────────────

_TOKEN_RE = re.compile(
    r"""
    \s*                             # skip whitespace
    (?:
        (?P<number>\d+\.?\d*(?:[eE][+-]?\d+)?)  # numbers: 0.5, 2, 1e-3
      | (?P<func>sin|cos|sqrt|exp|log|abs)    # function names
      | (?P<ident>[a-zA-Z_]\w*)               # variable names
      | (?P<op>[+\-*/^()])                     # operators and parens
      | (?P<bad>\S)                            # unexpected
    )
    """,
    re.VERBOSE,
)


class ParseError(ValueError):
    """Raised when an expression string cannot be parsed."""
    pass


class EvalError(ValueError):
    """Raised when an expression cannot be evaluated (missing variable, div/0)."""
    pass


# ── AST nodes ────────────────────────────────────────────────────────────────

@dataclass
class NumberNode:
    value: float


@dataclass
class VarNode:
    name: str


@dataclass
class FuncNode:
    name: str
    func: Callable[[float], float]
    arg: "ExprNode"


@dataclass
class BinOpNode:
    op: str
    left: "ExprNode"
    right: "ExprNode"


ExprNode = NumberNode | VarNode | FuncNode | BinOpNode


# ── Tokenizer ────────────────────────────────────────────────────────────────

def _tokenize(expr: str) -> list[tuple[str, str]]:
    """Tokenize a physics expression string.

    Returns list of (type, value) pairs.
    """
    tokens: list[tuple[str, str]] = []
    for m in _TOKEN_RE.finditer(expr):
        if m.lastgroup == "bad":
            raise ParseError(
                f"Unexpected character {m.group('bad')!r} at position {m.start()}"
            )
        kind = m.lastgroup
        if kind is not None:
            tokens.append((kind, m.group(kind)))
    return tokens


# ── Parser ───────────────────────────────────────────────────────────────────

class _Parser:
    """Recursive descent parser.

    Grammar (precedence low to high):
        expr    := term (('+' | '-') term)*
        term    := unary (('*' | '/') unary)*
        unary   := ('+' | '-')? power
        power   := atom ('^' unary)?
        atom    := NUMBER | ident | func '(' expr ')' | '(' expr ')'
    """

    def __init__(self, tokens: list[tuple[str, str]]) -> None:
        self.tokens = tokens
        self.pos = 0

    def _peek(self) -> tuple[str, str] | None:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def _advance(self) -> tuple[str, str]:
        if self.pos >= len(self.tokens):
            raise ParseError("Unexpected end of expression")
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def parse(self) -> ExprNode:
        node = self._expr()
        if self.pos < len(self.tokens):
            remaining = [f"{k}:{v}" for k, v in self.tokens[self.pos:]]
            raise ParseError(f"Unexpected tokens after expression: {remaining}")
        return node

    def _expr(self) -> ExprNode:
        left = self._term()
        while True:
            tok = self._peek()
            if tok is None:
                break
            if tok[0] == "op" and tok[1] in ("+", "-"):
                op = self._advance()[1]
                right = self._term()
                left = BinOpNode(op=op, left=left, right=right)
            else:
                break
        return left

    def _term(self) -> ExprNode:
        left = self._unary()
        while True:
            tok = self._peek()
            if tok is None:
                break
            if tok[0] == "op" and tok[1] in ("*", "/"):
                op = self._advance()[1]
                right = self._unary()
                left = BinOpNode(op=op, left=left, right=right)
            else:
                break
        return left

    def _unary(self) -> ExprNode:
        tok = self._peek()
        if tok is not None and tok[0] == "op" and tok[1] in ("+", "-"):
            op = self._advance()[1]
            if op == "-":
                operand = self._unary()
                return BinOpNode(op="*", left=NumberNode(-1.0), right=operand)
            else:
                return self._unary()
        return self._power()

    def _power(self) -> ExprNode:
        left = self._atom()
        tok = self._peek()
        if tok is not None and tok[0] == "op" and tok[1] == "^":
            self._advance()
            right = self._unary()  # right-associative
            return BinOpNode(op="^", left=left, right=right)
        return left

    def _atom(self) -> ExprNode:
        tok = self._peek()
        if tok is None:
            raise ParseError("Unexpected end of expression")

        if tok[0] == "number":
            self._advance()
            return NumberNode(float(tok[1]))

        if tok[0] in ("ident", "func"):
            name = self._advance()[1]
            nxt = self._peek()
            if nxt is not None and nxt[0] == "op" and nxt[1] == "(":
                from math import sin, cos, sqrt, exp, log, fabs
                _FUNCS: dict[str, Callable[[float], float]] = {
                    "sin": sin, "cos": cos, "sqrt": sqrt,
                    "exp": exp, "log": log, "abs": fabs,
                }
                if name in _FUNCS:
                    self._advance()  # consume '('
                    arg = self._expr()
                    tok2 = self._peek()
                    if tok2 is None or tok2[0] != "op" or tok2[1] != ")":
                        raise ParseError(f"Expected ')' after function args, got {tok2}")
                    self._advance()  # consume ')'
                    return FuncNode(name=name, func=_FUNCS[name], arg=arg)
            return VarNode(name=name)

        if tok[0] == "op" and tok[1] == "(":
            self._advance()
            node = self._expr()
            tok2 = self._peek()
            if tok2 is None or tok2[0] != "op" or tok2[1] != ")":
                raise ParseError(f"Expected ')', got {tok2}")
            self._advance()
            return node

        raise ParseError(f"Unexpected token: {tok[0]}:{tok[1]}")


# ── Public parse / evaluate API ──────────────────────────────────────────────

def parse_expression(expr_str: str) -> ExprNode:
    """Parse a physics expression string into an AST.

    Args:
        expr_str: Expression like "m*g*h + 0.5*m*v^2"

    Returns:
        AST root node.

    Raises:
        ParseError: if the expression is syntactically invalid.
    """
    tokens = _tokenize(expr_str)
    if not tokens:
        raise ParseError("Empty expression")
    return _Parser(tokens).parse()


def evaluate_node(node: ExprNode, context: dict[str, float]) -> float:
    """Evaluate an AST with the given variable bindings.

    Args:
        node: AST from parse_expression()
        context: Dict mapping variable names to numeric values

    Returns:
        Numeric result.

    Raises:
        EvalError: if a variable is undefined or domain error occurs.
    """
    if isinstance(node, NumberNode):
        return node.value

    if isinstance(node, VarNode):
        if node.name not in context:
            raise EvalError(
                f"Undefined variable: {node.name!r}. "
                f"Available: {sorted(context.keys())}"
            )
        return context[node.name]

    if isinstance(node, FuncNode):
        arg_val = evaluate_node(node.arg, context)
        try:
            return node.func(arg_val)
        except (ValueError, OverflowError) as e:
            raise EvalError(f"Error evaluating {node.name}({arg_val}): {e}")

    if isinstance(node, BinOpNode):
        left_val = evaluate_node(node.left, context)
        right_val = evaluate_node(node.right, context)
        if node.op == "+":
            return left_val + right_val
        if node.op == "-":
            return left_val - right_val
        if node.op == "*":
            return left_val * right_val
        if node.op == "/":
            if right_val == 0:
                raise EvalError("Division by zero")
            return left_val / right_val
        if node.op == "^":
            try:
                return left_val ** right_val
            except (ValueError, OverflowError):
                raise EvalError(f"Cannot compute {left_val} ^ {right_val}")
        raise EvalError(f"Unknown binary operator {node.op!r}")

    raise EvalError(f"Unknown AST node type: {type(node)}")


# ── Evaluator ────────────────────────────────────────────────────────────────

class ExpressionEvaluator:
    """Score physics expressions against observations.

    Example
    -------
    >>> db = ObservationDatabase("data/observations/phase1_falling.json")
    >>> ev = ExpressionEvaluator()
    >>> obs = db.get("falling_ball_straight_drop")
    >>> score = ev.score("m*g*h + 0.5*m*v^2", obs)
    >>> score > 0.95
    True
    """

    def __init__(self) -> None:
        self._ast_cache: dict[str, ExprNode] = {}
        # Cache: observation id -> set of quantity names that vary across timesteps
        self._varying_cache: dict[str, set[str]] = {}

    def _get_varying_quantities(self, obs: Observation) -> set[str]:
        """Return the set of quantity names that change across *obs* timesteps.

        Cached per observation id.  Used by the trivial-constancy gate:
        if an expression uses only quantities that never vary, its constancy
        is meaningless.
        """
        if obs.id in self._varying_cache:
            return self._varying_cache[obs.id]

        varying: set[str] = set()
        if len(obs.timesteps) < 2:
            self._varying_cache[obs.id] = varying
            return varying

        # Collect all quantity names that appear in any timestep
        all_keys = set()
        for ts in obs.timesteps:
            all_keys.update(ts.keys())

        for key in all_keys:
            first_val = obs.timesteps[0].get(key)
            for ts in obs.timesteps[1:]:
                if ts.get(key) != first_val:
                    varying.add(key)
                    break

        self._varying_cache[obs.id] = varying
        return varying

    def parse(self, expr_str: str) -> ExprNode:
        """Parse an expression string into an AST.  Results are cached."""
        key = expr_str.replace(" ", "")
        if key not in self._ast_cache:
            self._ast_cache[key] = parse_expression(expr_str)
        return self._ast_cache[key]

    def evaluate(self, expr_str: str, context: dict[str, float]) -> float:
        """Evaluate a physics expression with the given variable bindings.

        Args:
            expr_str: Expression like "m*g*h + 0.5*m*v^2"
            context: Dict mapping variable names to numeric values

        Returns:
            Numeric result of the expression.

        Raises:
            ParseError: if the expression string is invalid.
            EvalError: if a variable is missing or division by zero occurs.
        """
        ast = self.parse(expr_str)
        return evaluate_node(ast, context)

    def score(
        self,
        expr_str: str,
        obs_or_db: Observation | ObservationDatabase,
        epsilon: float = 1e-12,
    ) -> float:
        """Score an expression against one observation or an entire database.

        Args:
            expr_str: Expression string e.g. "m*g*h + 0.5*m*v^2"
            obs_or_db: A single Observation or an ObservationDatabase
            epsilon: Tolerance for zero-mean detection.

        Returns:
            Constancy score in [0.0, 1.0].
            For a database, returns mean constancy across all observations.
            For a single observation, returns its individual constancy.
        """
        if isinstance(obs_or_db, ObservationDatabase):
            observations: list[Observation] = list(obs_or_db)
        else:
            observations = [obs_or_db]

        if not observations:
            return 0.0

        try:
            ast = self.parse(expr_str)
        except ParseError:
            return 0.0

        scores: list[float] = []
        for obs in observations:
            if obs.phase_regions:
                # For scenarios with velocity discontinuities (collisions),
                # use piecewise evaluation — constancy within each phase
                # matters, not constancy across the discontinuity.
                pw = self.score_piecewise(expr_str, obs)
                scores.append(pw.get("piecewise_mean", 0.0))
            else:
                scores.append(self._score_observation(ast, obs, epsilon))
        return sum(scores) / len(scores)

    def score_all(
        self, expr_str: str, db: ObservationDatabase, epsilon: float = 1e-12
    ) -> list[float]:
        """Return per-observation constancy scores."""
        return [
            self.score(expr_str, obs, epsilon)
            for obs in db
        ]

    def score_per_observation(
        self,
        expr_str: str,
        observations: list[Observation],
        epsilon: float = 1e-12,
    ) -> list[float]:
        """Score an expression against each observation individually.

        Returns a list of constancy scores, one per observation, in the
        same order as the input list.  Unlike score_all (which works on
        a database), this accepts a plain list of observations — useful
        for splitting observations into regimes and measuring per-regime
        constancy.
        """
        if not observations:
            return []
        return [self.score(expr_str, obs, epsilon) for obs in observations]

    # ── Phase E: Piecewise and conditional evaluation ───────────────────

    def score_piecewise(
        self,
        expr_str: str,
        obs: Observation,
    ) -> dict[str, float]:
        """Score an expression separately in each phase region.

        For collision scenarios and other piecewise-physics scenarios,
        evaluate constancy independently in each time segment.

        Returns:
            Dict with keys:
            - 'overall': same as score() across all timesteps
            - '<label>': constancy within each phase region
            - 'piecewise_mean': mean of per-phase constancies
        """
        if len(obs.timesteps) < 2:
            return {"overall": 0.0}

        try:
            ast = self.parse(expr_str)
        except ParseError:
            return {"overall": 0.0}

        regions = obs.phase_regions
        if not regions:
            return {"overall": self._score_observation(ast, obs)}

        result: dict[str, float] = {}
        result["overall"] = self._score_observation(ast, obs)

        for region in regions:
            label = region.get("label", "unknown")
            t_range = region.get("t_range", [0.0, float("inf")])
            t_min, t_max = t_range[0], t_range[1]

            # Filter timesteps in this region
            region_ts = [ts for ts in obs.timesteps if t_min <= ts["t"] <= t_max]
            if len(region_ts) < 2:
                result[label] = 0.0
                continue

            values: list[float] = []
            for ts in region_ts:
                context = {**obs.parameters, **ts}
                try:
                    val = evaluate_node(ast, context)
                    if isinstance(val, complex):
                        break
                    values.append(val)
                except (EvalError, ZeroDivisionError, ValueError, OverflowError):
                    break

            if len(values) < 2:
                result[label] = 0.0
                continue

            n = len(values)
            mean_val = sum(values) / n
            if abs(mean_val) < 1e-12:
                scale = max(abs(v) for v in values)
                if scale < 1e-12:
                    result[label] = 1.0
                    continue
                variance = sum((v - mean_val) ** 2 for v in values) / n
                std_val = math.sqrt(max(variance, 0.0))
                result[label] = 1.0 / (1.0 + std_val / scale)
            else:
                variance = sum((v - mean_val) ** 2 for v in values) / n
                std_val = math.sqrt(max(variance, 0.0))
                result[label] = 1.0 / (1.0 + std_val / abs(mean_val))

        # Compute piecewise mean
        region_scores = [v for k, v in result.items() if k != "overall"]
        result["piecewise_mean"] = sum(region_scores) / len(region_scores) if region_scores else 0.0

        return result

    def score_conditional(
        self,
        expr_str: str,
        db: ObservationDatabase,
    ) -> dict:
        """Score an expression conditionally — separately for conservative
        and non-conservative scenarios.

        This enables the system to discover:
          "m*g*h + 0.5*m*v^2 is constant WHEN no friction is present"
          "m*g*h + 0.5*m*v^2 is NOT constant when friction is present"

        Returns:
            Dict with:
            - 'conservative_score': mean constancy in conservative scenarios
            - 'nonconservative_score': mean constancy in non-conservative scenarios
            - 'conservative_count': number of conservative observations
            - 'nonconservative_count': number of non-conservative observations
            - 'conditional_pattern': description of the conditional pattern
            - 'conservative_constancies': per-obs scores for conservative
            - 'nonconservative_constancies': per-obs scores for non-conservative
        """
        cons_obs: list[Observation] = []
        noncons_obs: list[Observation] = []

        for obs in db:
            is_cons = self._is_conservative_observation(obs)
            if is_cons:
                cons_obs.append(obs)
            else:
                noncons_obs.append(obs)

        cons_scores = [self.score(expr_str, obs) for obs in cons_obs]
        noncons_scores = [self.score(expr_str, obs) for obs in noncons_obs]

        cons_mean = sum(cons_scores) / len(cons_scores) if cons_scores else 0.0
        noncons_mean = sum(noncons_scores) / len(noncons_scores) if noncons_scores else 0.0

        # Detect conditional pattern
        if cons_mean >= 0.90 and noncons_mean < 0.50:
            pattern = "conservative_only"
        elif cons_mean >= 0.90 and noncons_mean >= 0.90:
            pattern = "universal"
        elif cons_mean < 0.50 and noncons_mean < 0.50:
            pattern = "no_conservation"
        elif noncons_mean >= 0.90 and cons_mean < 0.50:
            pattern = "nonconservative_only"  # unusual but possible
        else:
            pattern = "partial"

        return {
            "conservative_score": cons_mean,
            "nonconservative_score": noncons_mean,
            "conservative_count": len(cons_scores),
            "nonconservative_count": len(noncons_scores),
            "conditional_pattern": pattern,
            "conservative_constancies": cons_scores,
            "nonconservative_constancies": noncons_scores,
        }

    def _is_conservative_observation(self, obs: Observation) -> bool:
        """Determine if an observation represents a conservative scenario."""
        # Explicit flag takes precedence
        if obs.is_conservative is not None:
            return obs.is_conservative
        # If it has external forces, it's non-conservative
        if obs.external_forces:
            return False
        # If it has a known_invariant, it's likely conservative
        if obs.known_invariant is not None:
            return True
        # Default: conservative
        return True

    def score_with_context(
        self,
        expr_str: str,
        db: ObservationDatabase,
    ) -> dict:
        """Comprehensive evaluation: overall, piecewise, and conditional.

        This is the primary Phase E scoring function. It returns all
        evaluation dimensions needed for conditional discovery.

        Returns:
            Dict with keys: overall, conditional, piecewise_summary
        """
        conditional = self.score_conditional(expr_str, db)

        # Piecewise: evaluate on scenarios with phase_regions
        piecewise_scores: dict[str, float] = {}
        for obs in db:
            if obs.phase_regions:
                pw = self.score_piecewise(expr_str, obs)
                piecewise_scores[obs.id] = pw.get("piecewise_mean", 0.0)

        overall = self.score(expr_str, db)

        return {
            "overall": overall,
            "conditional": conditional,
            "piecewise_scores": piecewise_scores,
            "piecewise_mean": (
                sum(piecewise_scores.values()) / len(piecewise_scores)
                if piecewise_scores else None
            ),
        }

    def score_piecewise_aware(
        self,
        expr_str: str,
        obs_or_db: Observation | ObservationDatabase,
        epsilon: float = 1e-12,
    ) -> float:
        """Score an expression with piecewise-awareness for collision scenarios.

        When an observation has phase_regions (e.g., collision with velocity
        discontinuity at impact), uses the piecewise mean — evaluating
        constancy separately before and after the impact point.

        Without phase_regions, falls back to standard constancy scoring.

        This is the recommended scoring method for evaluating conservation
        laws in scenarios with discontinuities.
        """
        if isinstance(obs_or_db, ObservationDatabase):
            observations: list[Observation] = list(obs_or_db)
        else:
            observations = [obs_or_db]

        if not observations:
            return 0.0

        try:
            ast = self.parse(expr_str)
        except ParseError:
            return 0.0

        scores: list[float] = []
        for obs in observations:
            if obs.phase_regions:
                pw = self.score_piecewise(expr_str, obs)
                scores.append(pw.get("piecewise_mean", 0.0))
            else:
                scores.append(self._score_observation(ast, obs, epsilon))

        return sum(scores) / len(scores)

    def _score_observation(
        self, ast: ExprNode, obs: Observation, epsilon: float = 1e-12
    ) -> float:
        """Score a parsed expression against a single observation.

        Includes the "it must dance" gate: if every variable in the expression
        has the same value across all timesteps, the expression isn't revealing
        a conserved quantity — it's just reflecting that its inputs don't change.
        A quantity must vary before its constancy is meaningful.
        """
        if len(obs.timesteps) < 2:
            return 0.0

        # Gate: expressions with no variables (pure numbers like "2+2")
        # are trivially constant — they're just arithmetic results.
        var_names = _collect_var_names(ast)
        if not var_names:
            return 0.0

        # Gate: at least one variable must actually change across timesteps.
        # Uses precomputed varying-quantities cache — O(1) per expression.
        varying = self._get_varying_quantities(obs)
        if not (var_names & varying):
            return 0.0

        # Gate: expressions that algebraically cancel their own variables
        # (e.g. 1/v*1*v → always 2) are trivially constant — the variable
        # appears in both numerator and denominator with net exponent 0.
        if _has_self_cancellation(ast):
            return 0.0

        # Gate: term dominance — if one additive term's magnitude overwhelms
        # the others, the expression's constancy is degenerate.
        # Example: (t*x)^2 - c^2 where c=3e8 makes c^2 ≈ 9e16 dominate
        # (t*x)^2 ≈ 4e4.  The expression is "constant" only because one
        # term soaks up all the variation.  Not a genuine invariant.
        if _has_term_dominance(ast, obs):
            return 0.0

        # Gate: near-identity power — if a power operation's base is
        # always within 1% of 1.0, the power is degenerate (1^y = 1
        # regardless of exponent).  Catches (c^t)^x where tiny t makes
        # c^t ≈ 1.00002, making the whole expression ≈ constant.
        if _has_near_identity_power(ast, obs):
            return 0.0

        # Gate: numerical underflow — if the expression evaluates to
        # zero (or near-zero) at every timestep due to extreme exponents
        # rather than algebraic cancellation, it's not a genuine invariant.
        # Catches t^(c+x) where t≈1e-6, c+x≈3e8 → underflows to 0.0.
        if _has_numerical_collapse(ast, obs):
            return 0.0

        values: list[float] = []
        for ts in obs.timesteps:
            context = {**obs.parameters, **ts}
            try:
                val = evaluate_node(ast, context)
                if isinstance(val, complex):
                    return 0.0
                values.append(val)
            except (EvalError, ZeroDivisionError, ValueError, OverflowError):
                return 0.0

        n = len(values)
        mean_val = sum(values) / n

        # Guard against overflow: if values are astronomically large,
        # the expression isn't physically meaningful.
        if any(abs(v) > 1e150 for v in values):
            return 0.0

        # Use 1e-300 for zero-mean detection — only truly-zero values
        # (within float64 precision) should trigger the scale-based path.
        # Previously 1e-30 was high enough to catch quantum-scale values
        # (E^2 ~ 1e-37) and return false 1.0 scores for varying expressions.
        _eps = 1e-300
        if abs(mean_val) < _eps:
            scale = max(abs(v) for v in values)
            if scale < _eps:
                return 1.0
            variance = sum((v - mean_val) ** 2 for v in values) / n
            variance = _safe_real(variance)
            std_val = math.sqrt(max(variance, 0.0))
            return 1.0 / (1.0 + std_val / scale)

        variance = sum((v - mean_val) ** 2 for v in values) / n
        variance = _safe_real(variance)
        std_val = math.sqrt(max(variance, 0.0))
        rel_std = std_val / abs(mean_val)

        return 1.0 / (1.0 + rel_std)


# ── Helper ───────────────────────────────────────────────────────────────

def _collect_var_names(node: "ExprNode") -> set[str]:
    """Return the set of all variable names in an AST."""
    names: set[str] = set()
    def walk(n):
        if isinstance(n, VarNode):
            names.add(n.name)
        elif isinstance(n, FuncNode):
            walk(n.arg)
        elif isinstance(n, BinOpNode):
            walk(n.left)
            walk(n.right)
    walk(node)
    return names


def _has_self_cancellation(node: "ExprNode") -> bool:
    """Check if a variable algebraically cancels itself — either within a
    multiplicative group (v/v, 1/v*v) or across additive terms (-a+a, a-a).

    Within a multiplicative group: a variable has net exponent 0 (appears in
    both numerator and denominator with equal total power).

    Across additive terms: the same variable structure appears with opposite
    signs and the net contribution is zero.  This catches patterns like
    -psi+psi that always evaluate to 0 regardless of the variable's value.
    We require that EVERY variable group cancels (net 0) — if any variable
    has a non-zero net contribution the expression varies with that variable
    and is not self-cancelling.  Constants (terms with no variables) do not
    prevent cancellation.

    Additive cancellation is checked once at the top level (the flattened
    tree already includes nested +/-).  Recursive calls only check
    multiplicative cancellation so that -a+a inside a larger expression
    like -a+a+b is NOT falsely flagged — the outer variable b prevents
    the overall expression from being self-cancelling.
    """
    return _has_multiplicative_cancellation(node) or _has_additive_cancellation(node)


def _has_multiplicative_cancellation(node: "ExprNode") -> bool:
    """Check for cancellation within multiplicative groups (v/v, 1/v*v)."""
    if isinstance(node, (NumberNode, VarNode, FuncNode)):
        return False

    if isinstance(node, BinOpNode):
        if node.op in ("+", "-"):
            return (_has_multiplicative_cancellation(node.left)
                    or _has_multiplicative_cancellation(node.right))

        # Multiplicative group (*, /, ^): collect variable exponents.
        exponents: dict[str, float] = {}
        _collect_multiplicative_exponents(node, exponents, sign=1.0)
        for exp in exponents.values():
            if abs(exp) < 1e-9:
                return True

    return False


def _has_additive_cancellation(node: "ExprNode") -> bool:
    """Check for cancellation across additive terms (-a+a, a-a)."""
    if isinstance(node, (NumberNode, VarNode, FuncNode)):
        return False

    if isinstance(node, BinOpNode) and node.op in ("+", "-"):
        # Flatten the entire additive tree.
        terms: list[tuple[float, ExprNode]] = []
        _flatten_additive(node, terms, sign=1.0)

        # Group by variable-exponent fingerprint, summing effective
        # coefficients (not just signs — -2*a+a nets to -a, which varies).
        groups: dict[frozenset[tuple[str, float]], float] = {}
        for flatten_sign, term in terms:
            coeff_val, body = _extract_coeff(term)
            effective_coeff = flatten_sign * coeff_val
            exps: dict[str, float] = {}
            _collect_multiplicative_exponents(body, exps, sign=1.0)
            fp = frozenset((k, round(v, 6)) for k, v in exps.items())
            groups[fp] = groups.get(fp, 0.0) + effective_coeff

        # If every variable group has net coefficient ≈ 0 → self-cancelling.
        nonempty = {
            fp: total for fp, total in groups.items()
            if fp  # skip constant-only terms
        }
        if nonempty and all(abs(total) < 1e-9 for total in nonempty.values()):
            return True

    return False


def _flatten_additive(
    node: "ExprNode", out: list[tuple[float, "ExprNode"]], sign: float,
) -> None:
    """Flatten an additive tree into (sign, subexpression) pairs.

    sign tracks the accumulated sign: +1 for addition, -1 for subtraction.
    Nested +/- are flattened so the output is a flat list where sign
    indicates whether the subexpression is added or subtracted.
    """
    if isinstance(node, (NumberNode, VarNode, FuncNode)):
        out.append((sign, node))
        return

    if isinstance(node, BinOpNode):
        if node.op == "+":
            _flatten_additive(node.left, out, sign)
            _flatten_additive(node.right, out, sign)
        elif node.op == "-":
            _flatten_additive(node.left, out, sign)
            _flatten_additive(node.right, out, -sign)
        else:
            # Non-additive sub-expression (*, /, ^) → treat as atomic term.
            out.append((sign, node))


def _has_term_dominance(ast: "ExprNode", obs: Observation) -> bool:
    """Check if one additive term dominates the expression's magnitude.

    When one term is orders of magnitude larger than the others,
    the expression's constancy is degenerate — the dominant term
    absorbs all variation.  e.g., (t*x)^2 - c^2 where c=3e8
    makes c^2 ≈ 9e16 while (t*x)^2 ≈ 4e4.
    """
    # Only applies to additive expressions
    if not isinstance(ast, BinOpNode) or ast.op not in ("+", "-"):
        return False

    # Flatten into terms
    terms: list[tuple[float, ExprNode]] = []
    _flatten_additive(ast, terms, sign=1.0)
    if len(terms) < 2:
        return False

    # Evaluate each term at a few timesteps and check magnitude ratios
    sample_ts = obs.timesteps[:min(3, len(obs.timesteps))]
    all_dominant = True
    any_dominant = False

    for ts in sample_ts:
        context = {**obs.parameters, **ts}
        magnitudes: list[float] = []
        for sign, term_node in terms:
            try:
                val = evaluate_node(term_node, context)
                if isinstance(val, (int, float)) and not isinstance(val, complex):
                    magnitudes.append(abs(val * sign))
                else:
                    magnitudes.append(0.0)
            except Exception:
                magnitudes.append(0.0)

        if not magnitudes or max(magnitudes) == 0:
            continue

        max_mag = max(magnitudes)
        other_sum = sum(magnitudes) - max_mag

        # Skip timesteps where any term is zero (boundary case like v=0 → p=0)
        if any(m == 0 for m in magnitudes):
            continue

        # Dominant if one term > 10,000x the sum of all others
        if max_mag > 10000 * max(other_sum, 1e-30):
            any_dominant = True
        else:
            all_dominant = False

    # Only flag if dominant in EVERY sampled timestep
    return any_dominant and all_dominant


def _has_near_identity_power(ast: "ExprNode", obs: Observation) -> bool:
    """Check if any power operation has a base always near 1.0.

    When X ≈ 1.0, X^Y ≈ 1.0 regardless of Y — the power operation is
    degenerate.  Catches expressions like (c^t)^x where tiny t (≈1e-6)
    makes c^t ≈ 1.00002, making the whole expression artificially constant.
    """
    return _walk_for_near_identity(ast, obs)


def _walk_for_near_identity(node: "ExprNode", obs: Observation) -> bool:
    """Recursively check for power ops with near-identity bases."""
    if isinstance(node, (NumberNode, VarNode, FuncNode)):
        return False

    if isinstance(node, BinOpNode):
        if node.op == "^":
            # Check if base is always near 1.0
            sample_ts = obs.timesteps[:min(4, len(obs.timesteps))]
            all_near_one = True
            for ts in sample_ts:
                context = {**obs.parameters, **ts}
                try:
                    base_val = evaluate_node(node.left, context)
                    if isinstance(base_val, (int, float)) and not isinstance(base_val, complex):
                        if abs(base_val - 1.0) > 0.01:
                            all_near_one = False
                            break
                    else:
                        return False
                except Exception:
                    return False
            if all_near_one and len(sample_ts) > 0:
                return True

        # Recurse into children
        return (_walk_for_near_identity(node.left, obs)
                or _walk_for_near_identity(node.right, obs))

    return False


def _has_numerical_collapse(ast: "ExprNode", obs: Observation) -> bool:
    """Check if the expression always evaluates to zero (numerical underflow).

    Catches expressions like t^(c+x) where tiny base + huge exponent causes
    float64 underflow to 0.0 at every timestep.  This is not algebraic
    cancellation — it's a numerical artifact.
    """
    values: list[float] = []
    for ts in obs.timesteps:
        context = {**obs.parameters, **ts}
        try:
            val = evaluate_node(ast, context)
            if isinstance(val, (int, float)) and not isinstance(val, complex):
                values.append(float(val))
            else:
                return False
        except Exception:
            return False

    if len(values) < 2:
        return False

    # All values are zero (or within float64 epsilon of zero)
    return all(abs(v) < 1e-300 for v in values)


def _extract_coeff(node: "ExprNode") -> tuple[float, "ExprNode"]:
    """Extract the leading numeric coefficient from a multiplicative term.

    Returns (value, body) where value is the product of all constant factors
    and body is the term with constants stripped.  For example:
      -a    → (-1.0, VarNode(a))
      a     → (1.0,  VarNode(a))
      5     → (5.0,  NumberNode(1))
      -2*a  → (-2.0, VarNode(a))
      2*a   → (2.0,  VarNode(a))
      3*5*a → (15.0, VarNode(a))
    """
    if isinstance(node, NumberNode):
        return (node.value, NumberNode(1.0))
    if isinstance(node, (VarNode, FuncNode)):
        return (1.0, node)
    if isinstance(node, BinOpNode) and node.op == "*":
        # Walk left through nested * to accumulate constant factors.
        left = node.left
        product = 1.0
        while isinstance(left, BinOpNode) and left.op == "*":
            inner = left.left
            if isinstance(inner, NumberNode):
                product *= inner.value
                left = left.right
            else:
                break
        if isinstance(left, NumberNode):
            product *= left.value
            return (product, node.right)
    return (1.0, node)


def _collect_multiplicative_exponents(
    node: "ExprNode", out: dict[str, float], sign: float,
) -> None:
    """Walk a multiplicative sub-tree, accumulating variable exponents.

    sign is +1 for multiplication, -1 for division, and multiplies
    the power for exponentiation.
    """
    if isinstance(node, NumberNode):
        return
    if isinstance(node, VarNode):
        out[node.name] = out.get(node.name, 0.0) + sign
        return
    if isinstance(node, FuncNode):
        _collect_multiplicative_exponents(node.arg, out, sign)
        return

    if isinstance(node, BinOpNode):
        if node.op == "*":
            _collect_multiplicative_exponents(node.left, out, sign)
            _collect_multiplicative_exponents(node.right, out, sign)
        elif node.op == "/":
            _collect_multiplicative_exponents(node.left, out, sign)
            _collect_multiplicative_exponents(node.right, out, -sign)
        elif node.op == "^":
            if isinstance(node.right, NumberNode):
                _collect_multiplicative_exponents(
                    node.left, out, sign * node.right.value,
                )


def _safe_real(x: float | complex) -> float:
    """Return the real part of x, or 0.0 if complex or NaN."""
    if isinstance(x, complex):
        return x.real
    if isinstance(x, float) and math.isnan(x):
        return 0.0
    return float(x)


# ── Alias for backward compatibility ──────────────────────────────────────────

Evaluator = ExpressionEvaluator


# ── Regime discovery helpers ─────────────────────────────────────────────

def find_regime_threshold(
    expr_str: str,
    observations: list[Observation],
    evaluator: ExpressionEvaluator | None = None,
    *,
    min_regime_size: int = 3,
) -> dict | None:
    """Detect the best regime split for a candidate expression.

    When an invariant holds in some regimes but not others, the
    per-observation constancy scores will be bimodal — high in one
    cluster, low in another.  This function finds the quantity that
    best separates the observations into two regimes where the
    expression's constancy differs most.

    Algorithm:
        1. Score the expression per-observation.
        2. For each candidate key quantity, sort observations by that
           quantity's value and find the largest adjacent score gap.
        3. Return the split with the largest gap, provided each regime
           has at least *min_regime_size* observations.

    Parameters
    ----------
    expr_str : str
        The candidate expression to evaluate.
    observations : list[Observation]
        Observations to split into regimes.
    evaluator : ExpressionEvaluator | None
        Reusable evaluator; created if omitted.
    min_regime_size : int
        Minimum observations per regime (anti-hacking guard).

    Returns
    -------
    dict or None
        If a viable split is found:
            key_quantity : str
            split_index : int (into the sorted list)
            gap : float (score difference at the split)
            regime_a_obs : list[Observation]
            regime_b_obs : list[Observation]
            regime_a_scores : list[float]
            regime_b_scores : list[float]
            sorted_per_obs_scores : list[float]
        Returns None if no split meets the minimum-size constraint.
    """
    if evaluator is None:
        evaluator = ExpressionEvaluator()

    if len(observations) < 2 * min_regime_size:
        return None

    # Collect all quantity names that appear across observations.
    candidate_keys: set[str] = set()
    for obs in observations:
        for ts in obs.timesteps:
            candidate_keys.update(ts.keys())
        candidate_keys.update(obs.parameters.keys())

    if not candidate_keys:
        return None

    best_split: dict | None = None
    best_gap = -1.0

    for key_qty in candidate_keys:
        # Get a representative value for each observation under this key.
        pairs: list[tuple[float, Observation]] = []
        for obs in observations:
            vals: list[float] = []
            for ts in obs.timesteps:
                if key_qty in ts:
                    vals.append(ts[key_qty])
            if key_qty in obs.parameters:
                vals.append(obs.parameters[key_qty])
            if not vals:
                continue
            pairs.append((sum(vals) / len(vals), obs))

        if len(pairs) < 2 * min_regime_size:
            continue

        # Sort by the key quantity value.
        pairs.sort(key=lambda x: x[0])
        sorted_obs = [p[1] for p in pairs]

        # Score per-observation in sorted order.
        per_obs_scores = evaluator.score_per_observation(
            expr_str, sorted_obs,
        )

        if len(per_obs_scores) < 2 * min_regime_size:
            continue

        # Find largest adjacent score gap.
        for i in range(min_regime_size, len(per_obs_scores) - min_regime_size):
            left_mean = sum(per_obs_scores[:i]) / i
            right_mean = (
                sum(per_obs_scores[i:]) / (len(per_obs_scores) - i)
            )
            gap = abs(left_mean - right_mean)
            if gap > best_gap:
                best_gap = gap
                best_split = {
                    "key_quantity": key_qty,
                    "split_index": i,
                    "gap": gap,
                    "regime_a_obs": sorted_obs[:i],
                    "regime_b_obs": sorted_obs[i:],
                    "regime_a_scores": per_obs_scores[:i],
                    "regime_b_scores": per_obs_scores[i:],
                    "sorted_per_obs_scores": per_obs_scores,
                }

    return best_split


# ── Convenience ──────────────────────────────────────────────────────────────

def score_expression(
    expr_str: str,
    db_path: str = "data/observations/phase1_falling.json",
) -> float:
    """One-shot convenience: score an expression against the default database.

    >>> score_expression("m*g*h + 0.5*m*v^2")
    0.99...
    """
    db = ObservationDatabase(db_path)
    ev = ExpressionEvaluator()
    return ev.score(expr_str, db)
