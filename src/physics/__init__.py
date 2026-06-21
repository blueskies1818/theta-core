"""Physics expression system — grammar, dimensions, generator.

Phase A: Type system, expression grammar, and breadth-first combinatorial
expression builder for self-play physics discovery.
"""

from src.physics.dimensions import Dimension
from src.physics.grammar import Expression
from src.physics.generator import ExpressionGenerator

__all__ = ["Dimension", "Expression", "ExpressionGenerator"]
