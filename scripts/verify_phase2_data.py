"""Verify phase2_extended.json loads correctly."""
from src.physics.observations import ObservationDatabase

db = ObservationDatabase('data/observations/phase2_extended.json')
print(f'Loaded {len(db)} scenarios')
ids = db.scenario_ids
print(f'IDs sample: {ids[:5]}...{ids[-3:]}')
issues = db.validate()
print(f'Validation issues: {issues}')

with_inv = [o.id for o in db if o.known_invariant]
without_inv = [o.id for o in db if not o.known_invariant]
print(f'With invariant ({len(with_inv)})')
print(f'Without invariant ({len(without_inv)})')
print(f'Without: {without_inv}')

# Test expression evaluation
from src.physics.evaluator import ExpressionEvaluator
ev = ExpressionEvaluator()

# Test on conservative scenario
score = ev.score("m*g*h + 0.5*m*v^2", db.get("freefall_moon_drop"))
print(f'Moon drop energy score: {score:.4f}')

# Test on non-conservative scenario
score2 = ev.score("m*g*h + 0.5*m*v^2", db.get("freefall_air_resistance_linear"))
print(f'Air resistance energy score: {score2:.4f} (should be < 0.95)')

# Test spring energy on undamped
score3 = ev.score("0.5*m*v^2 + 0.5*k*h^2", db.get("spring_stiff"))
print(f'Stiff spring energy score: {score3:.4f}')

# Test cross-domain
score4 = ev.score("m*g*h + 0.5*m*v^2 + 0.5*k*h^2", db.get("mass_spring_gravity"))
print(f'Mass-spring-gravity score: {score4:.4f}')
