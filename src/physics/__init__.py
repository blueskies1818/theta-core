"""Physics expression system — grammar, dimensions, generator, observations, evaluator.

Phase A: Type system, expression grammar, and breadth-first combinatorial
expression builder for self-play physics discovery.

Phase B: Observation database loader and constancy-based expression evaluator.
"""

from src.physics.dimensions import Dimension
from src.physics.grammar import Expression
from src.physics.generator import ExpressionGenerator
from src.physics.observations import Observation, ObservationDatabase
from src.physics.evaluator import Evaluator, ExpressionEvaluator, score_expression

__all__ = [
    "Dimension",
    "Expression",
    "ExpressionGenerator",
    "Observation",
    "ObservationDatabase",
    "Evaluator",
    "ExpressionEvaluator",
    "score_expression",
]
