"""Debug mass_spring_gravity invariant."""
from src.physics.observations import ObservationDatabase
from src.physics.evaluator import ExpressionEvaluator
import math

db = ObservationDatabase('data/observations/phase2_extended.json')
obs = db.get('mass_spring_gravity')

print(f"h_eq = m*g/k = {obs.parameters['m']}*{obs.parameters['g']}/{obs.parameters['k']} = {obs.parameters['m']*obs.parameters['g']/obs.parameters['k']}")
print(f"A = {obs.parameters['A']}, omega = sqrt(k/m) = {math.sqrt(obs.parameters['k']/obs.parameters['m'])}")
print()

ev = ExpressionEvaluator()

# Test the expression
expr = "0.5*m*v^2 + 0.5*k*h^2 - m*g*h"
print(f"Testing expression: {expr}")

for ts in obs.timesteps:
    ctx = {**obs.parameters, **ts}
    val = ev.evaluate(expr, ctx)
    print(f"  t={ts['t']:.4f} h={ts['h']:.6f} v={ts['v']:.6f} => expr={val:.8f}")

# Also test using the evaluator's score function
score = ev.score(expr, obs)
print(f"\nScore: {score:.6f}")

# Test alternative: maybe the sign convention is reversed?
print("\nTry: 0.5*m*v^2 + 0.5*k*h^2")
for ts in obs.timesteps:
    ctx = {**obs.parameters, **ts}
    val = ev.evaluate("0.5*m*v^2 + 0.5*k*h^2", ctx)
    print(f"  t={ts['t']:.4f} h={ts['h']:.6f} v={ts['v']:.6f} => {val:.8f}")

print(f"Score: {ev.score('0.5*m*v^2 + 0.5*k*h^2', obs):.6f}")

# Try with positive sign for mgh
print("\nTry: 0.5*m*v^2 + 0.5*k*h^2 + m*g*h")
for ts in obs.timesteps:
    ctx = {**obs.parameters, **ts}
    val = ev.evaluate("0.5*m*v^2 + 0.5*k*h^2 + m*g*h", ctx)
    print(f"  t={ts['t']:.4f} h={ts['h']:.6f} v={ts['v']:.6f} => {val:.8f}")

print(f"Score: {ev.score('0.5*m*v^2 + 0.5*k*h^2 + m*g*h', obs):.6f}")
