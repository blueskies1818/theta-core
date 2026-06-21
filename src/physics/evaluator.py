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

        scores = [self._score_observation(ast, obs) for obs in observations]
        return sum(scores) / len(scores)

    def score_all(
        self, expr_str: str, db: ObservationDatabase, epsilon: float = 1e-12
    ) -> list[float]:
        """Return per-observation constancy scores."""
        return [
            self.score(expr_str, obs, epsilon)
            for obs in db
        ]

    def _score_observation(
        self, ast: ExprNode, obs: Observation
    ) -> float:
        """Score a parsed expression against a single observation."""
        if len(obs.timesteps) < 2:
            return 0.0

        values: list[float] = []
        for ts in obs.timesteps:
            context = {**obs.parameters, **ts}
            try:
                val = evaluate_node(ast, context)
                values.append(val)
            except (EvalError, ZeroDivisionError, ValueError, OverflowError):
                return 0.0

        n = len(values)
        mean_val = sum(values) / n

        if abs(mean_val) < 1e-12:
            scale = max(abs(v) for v in values)
            if scale < 1e-12:
                return 1.0
            variance = sum((v - mean_val) ** 2 for v in values) / n
            std_val = math.sqrt(variance)
            return 1.0 / (1.0 + std_val / scale)

        variance = sum((v - mean_val) ** 2 for v in values) / n
        std_val = math.sqrt(variance)
        rel_std = std_val / abs(mean_val)

        return 1.0 / (1.0 + rel_std)


# ── Alias for backward compatibility ──────────────────────────────────────────

Evaluator = ExpressionEvaluator


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
