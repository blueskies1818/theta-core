"""Test Phase E evaluator capabilities."""
import json
from src.physics.observations import ObservationDatabase
from src.physics.evaluator import ExpressionEvaluator

db = ObservationDatabase('data/observations/phase2_extended.json')
ev = ExpressionEvaluator()

# Test score_conditional
print("=== Conditional Scoring ===")
energy_expr = "m*g*h + 0.5*m*v^2"
result = ev.score_conditional(energy_expr, db)
print(f"Expression: {energy_expr}")
print(f"  conservative_score:   {result['conservative_score']:.4f} ({result['conservative_count']} scenarios)")
print(f"  nonconservative_score: {result['nonconservative_score']:.4f} ({result['nonconservative_count']} scenarios)")
print(f"  conditional_pattern:   {result['conditional_pattern']}")
print()

# Test a non-conserved expression
noncons = "m*v"
result2 = ev.score_conditional(noncons, db)
print(f"Expression: {noncons}")
print(f"  conservative_score:   {result2['conservative_score']:.4f}")
print(f"  nonconservative_score: {result2['nonconservative_score']:.4f}")
print(f"  conditional_pattern:   {result2['conditional_pattern']}")
print()

# Test piecewise on a collision scenario
print("=== Piecewise Scoring ===")
collision_expr = "0.5*m*v^2"
for obs in db:
    if obs.phase_regions:
        pw = ev.score_piecewise(collision_expr, obs)
        print(f"Scenario: {obs.id}")
        print(f"  {json.dumps(pw, indent=2)}")
        break
print()

# Test score_with_context
print("=== Comprehensive Scoring ===")
ctx = ev.score_with_context(energy_expr, db)
print(f"Overall: {ctx['overall']:.4f}")
print(f"Conditional pattern: {ctx['conditional']['conditional_pattern']}")
print(f"Piecewise scenarios: {list(ctx['piecewise_scores'].keys())[:3]}")
print(f"Piecewise mean: {ctx['piecewise_mean']}")
print()

# Quick check: spring energy with conditonal
spring_expr = "0.5*m*v^2 + 0.5*k*h^2"
ctx2 = ev.score_with_context(spring_expr, db)
print(f"Spring energy: overall={ctx2['overall']:.4f}, pattern={ctx2['conditional']['conditional_pattern']}")
print(f"  cons={ctx2['conditional']['conservative_score']:.4f}, noncons={ctx2['conditional']['nonconservative_score']:.4f}")
