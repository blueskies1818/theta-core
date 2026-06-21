from src.physics.observations import ObservationDatabase
from src.physics.evaluator import ExpressionEvaluator
db = ObservationDatabase('data/observations/phase2_extended.json')
ev = ExpressionEvaluator()
obs = db.get('mass_spring_damped_gravity')
s = ev.score('0.5*m*v^2 + 0.5*k*h^2 - m*g*h', obs)
print(f'Damped vertical spring score: {s:.4f}')
for ts in obs.timesteps:
    ctx = {**obs.parameters, **ts}
    val = ev.evaluate('0.5*m*v^2 + 0.5*k*h^2 - m*g*h', ctx)
    print(f'  t={ts["t"]:.4f} h={ts["h"]:.6f} v={ts["v"]:.6f} => {val:.6f}')
