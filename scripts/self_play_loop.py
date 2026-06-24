#!/usr/bin/env python3
"""Self-play loop for autonomous physics discovery with curriculum learning.

Orchestrates the full self-play cycle:
    GENERATE → SIMULATE → DISCOVER → COMPARE → LOG

Uses curriculum across complexity levels 1-4, advancing to the next level
when the success rate exceeds 80% on the current level.

Phase C of the self-play architecture.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from src.physics.dimensions import Dimension
from src.physics.expression_generator import (
    SelfPlayExpressionGenerator,
    GeneratedExpression,
    PRE_1905_QUANTITY_DIMS,
)
from src.physics.evaluator import ExpressionEvaluator
from src.physics.observation_simulator import simulate_observations
from src.physics.search import auto_discover, SearchResult


# ═══════════════════════════════════════════════════════════════════════════════
# Expression comparison
# ═══════════════════════════════════════════════════════════════════════════════

_COMPARISON_SCORES = {
    "exact": 1.0,
    "structural": 0.9,
    "constant": 0.5,
    "fail": 0.0,
}

# Minimum discovery constancy score to even consider comparison
_COMPARISON_MIN_CONSTANCY = 0.70


def _parse_safe(evaluator: ExpressionEvaluator, expr: str):
    """Parse an expression safely, returning None on failure."""
    try:
        return evaluator.parse(expr)
    except Exception:
        return None


def _collect_vars_safe(evaluator: ExpressionEvaluator, expr: str) -> set[str]:
    """Collect variable names from an expression string, or empty set on parse failure."""
    ast = _parse_safe(evaluator, expr)
    if ast is None:
        return set()
    from src.physics.evaluator import _collect_var_names
    return _collect_var_names(ast)


def _is_constant_expr(evaluator: ExpressionEvaluator, expr: str) -> bool:
    """Check if an expression is purely constant (no variable references)."""
    vars_ = _collect_vars_safe(evaluator, expr)
    return len(vars_) == 0 and len(expr.strip()) > 0


def _normalize_expr(expr: str) -> str:
    """Normalize an expression string for comparison.

    Removes spaces, normalises exponent notation, strips leading zeros.
    """
    s = expr.replace(" ", "")
    # Normalise ^ to consistent form, remove redundant parens where safe
    # Normalise decimal: 0.5 vs .5
    import re
    # Leading decimal: .5 -> 0.5
    s = re.sub(r'(?<![0-9])\.', '0.', s)
    return s


def compare_expressions(
    discovered: str,
    ground_truth: str,
    *,
    evaluator: ExpressionEvaluator | None = None,
) -> tuple[str, float]:
    """Compare a discovered expression against ground truth.

    Returns (category, score) where category is one of:
        "exact"       — normalized strings identical
        "structural"  — same variables and dimensional relationships
        "constant"    — both are constant-valued expressions
        "fail"        — no meaningful match

    Score: 1.0 exact, 0.9 structural, 0.5 constant, 0.0 fail.
    """
    if evaluator is None:
        evaluator = ExpressionEvaluator()

    if not discovered or not ground_truth:
        return ("fail", 0.0)

    # ── Exact match ───────────────────────────────────────────────────────
    norm_disc = _normalize_expr(discovered)
    norm_gt = _normalize_expr(ground_truth)

    if norm_disc == norm_gt:
        return ("exact", 1.0)

    # ── Parse both for structural comparison ──────────────────────────────
    disc_vars = _collect_vars_safe(evaluator, discovered)
    gt_vars = _collect_vars_safe(evaluator, ground_truth)

    if not disc_vars and not gt_vars:
        # Both are constant expressions (different forms)
        return ("constant", 0.5)

    # ── Structural: same variables ────────────────────────────────────────
    if disc_vars and gt_vars and disc_vars == gt_vars:
        # Check dimensional compatibility
        try:
            disc_dim = _expression_dimension(evaluator, discovered)
            gt_dim = _expression_dimension(evaluator, ground_truth)
            if disc_dim is not None and gt_dim is not None:
                disc_str = str(disc_dim)
                gt_str = str(gt_dim)
                if disc_str == gt_str:
                    return ("structural", 0.9)
        except Exception:
            pass

    # ── Constant forms (one has vars, other doesn't — unlikely match) ─────
    if _is_constant_expr(evaluator, discovered) and _is_constant_expr(evaluator, ground_truth):
        return ("constant", 0.5)

    return ("fail", 0.0)


def _expression_dimension(evaluator: ExpressionEvaluator, expr: str) -> Dimension | None:
    """Compute the dimension of an expression using variable dimension lookup."""
    ast = _parse_safe(evaluator, expr)
    if ast is None:
        return None

    # Build dimension lookup from pre-1905 quantities + scalars
    dim_lookup: dict[str, Dimension] = dict(PRE_1905_QUANTITY_DIMS)
    for c in ["0", "0.5", "1", "2", "-1", "-2", "3", "4"]:
        dim_lookup[c] = Dimension.scalar()

    try:
        from src.physics.evaluator import NumberNode, VarNode, BinOpNode

        def dim_of(node) -> Dimension | None:
            if isinstance(node, NumberNode):
                return Dimension.scalar()
            if isinstance(node, VarNode):
                d = dim_lookup.get(node.name)
                return d if d is not None else Dimension.scalar()
            if isinstance(node, BinOpNode):
                ld = dim_of(node.left)
                rd = dim_of(node.right)
                if ld is None or rd is None:
                    return None
                try:
                    if node.op in ("+", "-"):
                        if isinstance(node.left, NumberNode):
                            return rd
                        if isinstance(node.right, NumberNode):
                            return ld
                        return ld if ld.compatible_with(rd) else None
                    elif node.op == "*":
                        return ld * rd
                    elif node.op == "/":
                        return ld / rd
                    elif node.op == "^":
                        if isinstance(node.right, NumberNode):
                            return ld ** float(node.right.value)
                except Exception:
                    pass
            return None

        return dim_of(ast)
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Result record
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SelfPlayResult:
    """A single self-play trial result."""
    level: int
    iteration: int
    expression: str
    ground_truth: str
    discovered: str
    comparison: str          # "exact", "structural", "constant", "fail"
    comparison_score: float
    constancy_score: float
    domain: str
    num_observations: int
    elapsed_seconds: float
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "iteration": self.iteration,
            "expression": self.expression,
            "ground_truth": self.ground_truth,
            "discovered": self.discovered,
            "comparison": self.comparison,
            "comparison_score": self.comparison_score,
            "constancy_score": self.constancy_score,
            "domain": self.domain,
            "num_observations": self.num_observations,
            "elapsed_seconds": self.elapsed_seconds,
            "error": self.error,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Self-play loop
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_ITERATIONS_PER_LEVEL = 1000
CURRICULUM_SUCCESS_THRESHOLD = 0.80
DEFAULT_NOISE_FRAC = 0.01
DEFAULT_SEED = 42
DEFAULT_NUM_CONFIGS = 20


class CurriculumSelfPlayLoop:
    """Orchestrate self-play physics discovery with curriculum learning.

    Parameters
    ----------
    levels : list[int]
        Complexity levels to run (default [1, 2, 3, 4]).
    iterations_per_level : int
        Maximum iterations per level before advancing curriculum.
    noise_frac : float
        Measurement noise fraction for simulation.
    seed : int
        Random seed for reproducibility.
    output_path : str or Path
        Path for JSONL results file.
    """

    def __init__(
        self,
        *,
        levels: list[int] | None = None,
        iterations_per_level: int = DEFAULT_ITERATIONS_PER_LEVEL,
        noise_frac: float = DEFAULT_NOISE_FRAC,
        seed: int = DEFAULT_SEED,
        output_path: str | Path = "data/self_play_results.jsonl",
    ) -> None:
        self.levels = levels if levels is not None else [1, 2, 3, 4]
        self.iterations_per_level = iterations_per_level
        self.noise_frac = noise_frac
        self.seed = seed
        self.output_path = Path(output_path)
        self.results: list[SelfPlayResult] = []

        # Initialize components
        self._generator = SelfPlayExpressionGenerator(
            seed=seed,
            include_hidden_vars=True,
            hidden_var_probability=0.3,
        )
        self._evaluator = ExpressionEvaluator()

    def run(self) -> list[SelfPlayResult]:
        """Run the self-play loop across all levels with curriculum learning."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        total_successes = 0
        total_trials = 0

        for level in self.levels:
            print(f"\n{'='*60}")
            print(f"Level {level}: starting curriculum phase")
            print(f"{'='*60}")

            level_successes = 0
            level_trials = 0

            for iteration in range(1, self.iterations_per_level + 1):
                result = self._run_single_trial(level, iteration)

                if result is not None:
                    self.results.append(result)
                    self._append_jsonl(result)
                    level_trials += 1

                    if result.comparison_score >= CURRICULUM_SUCCESS_THRESHOLD:
                        level_successes += 1

                # Progress reporting
                if iteration % 100 == 0:
                    sr = (
                        level_successes / level_trials * 100
                        if level_trials > 0 else 0
                    )
                    print(
                        f"  [{iteration:4d}] trials={level_trials} "
                        f"successes={level_successes} rate={sr:.1f}%"
                    )

                # Curriculum check every 20 trials after minimum 50
                if (
                    level_trials >= 50
                    and level_trials % 20 == 0
                    and level_trials > 0
                ):
                    success_rate = level_successes / level_trials
                    if success_rate > CURRICULUM_SUCCESS_THRESHOLD:
                        print(
                            f"  >> Curriculum advance: success rate "
                            f"{success_rate*100:.1f}% > "
                            f"{CURRICULUM_SUCCESS_THRESHOLD*100:.0f}%"
                        )
                        break

            # End-of-level summary
            final_rate = (
                level_successes / level_trials * 100
                if level_trials > 0 else 0
            )
            print(
                f"Level {level} complete: {level_successes}/{level_trials} "
                f"({final_rate:.1f}%)"
            )
            total_successes += level_successes
            total_trials += level_trials

        overall = total_successes / total_trials * 100 if total_trials > 0 else 0
        print(
            f"\nAll levels complete: {total_successes}/{total_trials} "
            f"({overall:.1f}%)"
        )
        print(f"Results saved to: {self.output_path}")
        return self.results

    def _run_single_trial(
        self, level: int, iteration: int
    ) -> SelfPlayResult | None:
        """Run a single generate → simulate → discover → compare trial."""
        t_start = time.time()
        expression_str = ""
        ground_truth = ""
        domain = ""
        discovered = ""
        comparison = "fail"
        comparison_score = 0.0
        constancy_score = 0.0
        num_observations = 0
        error = None

        try:
            # ── Generate ──────────────────────────────────────────────────
            gen_result = self._generator.generate(level)
            expression_str = gen_result.expression_str
            ground_truth = gen_result.ground_truth_expression or expression_str
            domain = gen_result.domain_label

            # ── Simulate ──────────────────────────────────────────────────
            # Convert Dimension objects to dimension name strings for simulator
            sim_quantities: dict[str, str] = {
                q: str(d) for q, d in gen_result.quantities_dict.items()
            }

            observations = simulate_observations(
                expression=ground_truth,
                quantities=sim_quantities,
                num_configs=DEFAULT_NUM_CONFIGS,
                noise_frac=self.noise_frac,
                seed=self.seed + iteration,
            )
            num_observations = len(observations)

            # ── Discover ──────────────────────────────────────────────────
            discovery_result = auto_discover(
                quantities=gen_result.quantities_dict,
                observations=observations,
                known_invariant=None,
                discovery_threshold=0.70,
                beam_expansions=2000,
                _no_neural_templates=True,
                _no_regime_split=True,
            )

            discovered = discovery_result.expression
            constancy_score = discovery_result.score

            # ── Compare ───────────────────────────────────────────────────
            if discovered:
                comparison, comparison_score = compare_expressions(
                    discovered, expression_str, evaluator=self._evaluator
                )

        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc(file=sys.stderr)

        elapsed = time.time() - t_start

        return SelfPlayResult(
            level=level,
            iteration=iteration,
            expression=expression_str,
            ground_truth=ground_truth,
            discovered=discovered,
            comparison=comparison,
            comparison_score=comparison_score,
            constancy_score=constancy_score,
            domain=domain,
            num_observations=num_observations,
            elapsed_seconds=round(elapsed, 3),
            error=error,
        )

    def _append_jsonl(self, result: SelfPlayResult) -> None:
        """Append a single result as a JSON line to the output file."""
        with open(self.output_path, "a") as f:
            json.dump(result.to_dict(), f)
            f.write("\n")

    def per_level_stats(self) -> dict[int, dict[str, Any]]:
        """Return per-level success statistics."""
        stats: dict[int, dict[str, Any]] = {}
        for level in self.levels:
            level_results = [r for r in self.results if r.level == level]
            successes = sum(
                1 for r in level_results
                if r.comparison_score >= CURRICULUM_SUCCESS_THRESHOLD
            )
            stats[level] = {
                "total": len(level_results),
                "successes": successes,
                "rate": successes / len(level_results) if level_results else 0.0,
            }
        return stats


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Self-play physics discovery loop with curriculum learning",
    )
    parser.add_argument(
        "--levels", type=int, nargs="+", default=[1, 2, 3, 4],
        help="Complexity levels to run (default: 1 2 3 4)",
    )
    parser.add_argument(
        "--iterations-per-level", type=int,
        default=DEFAULT_ITERATIONS_PER_LEVEL,
        help=f"Max iterations per level (default: {DEFAULT_ITERATIONS_PER_LEVEL})",
    )
    parser.add_argument(
        "--noise", type=float, default=DEFAULT_NOISE_FRAC,
        help=f"Measurement noise fraction (default: {DEFAULT_NOISE_FRAC})",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"Random seed for reproducibility (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--output", type=str, default="data/self_play_results.jsonl",
        help="Output JSONL path (default: data/self_play_results.jsonl)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    loop = CurriculumSelfPlayLoop(
        levels=args.levels,
        iterations_per_level=args.iterations_per_level,
        noise_frac=args.noise,
        seed=args.seed,
        output_path=args.output,
    )

    results = loop.run()
    stats = loop.per_level_stats()

    print("\nPer-level summary:")
    for level, s in sorted(stats.items()):
        print(
            f"  Level {level}: {s['successes']}/{s['total']} "
            f"({s['rate']*100:.1f}%)"
        )

    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
