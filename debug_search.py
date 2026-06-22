"""Debug script for search."""
import sys
sys.path.insert(0, "/home/blueman1818/Projects/theta-core")

from src.physics.dimensions import Dimension
from src.physics.observations import ObservationDatabase
from src.physics.search import ExpressionSearch

db = ObservationDatabase("/home/blueman1818/Projects/theta-core/data/observations/phase1_falling.json")
train_ids = [
    "falling_ball_straight_drop", "falling_ball_upward_throw",
    "falling_ball_varying_mass", "pendulum_small_angle",
    "pendulum_large_angle", "projectile_45deg",
    "projectile_90deg", "sliding_block_incline",
]
train_obs = [db.get(oid) for oid in train_ids]

quantities = {}
for obs in train_obs:
    for name, dim_name in obs.quantities.items():
        if name not in quantities:
            quantities[name] = Dimension.named(dim_name)
    for pname in obs.parameters:
        if pname not in quantities:
            quantities[pname] = Dimension.scalar()

search = ExpressionSearch(
    quantities=quantities, train_observations=train_obs,
    max_depth=10, max_expansions=20000, discovery_threshold=0.95, top_k=50,
)

dyn = search._get_dynamic_quantities()
print("Dynamic quantities:", dyn)
print("Initial candidates:", sorted(set(dyn) | set(search.scalar_constants)))
print("m in initial?", "m" in dyn)

result = search.run()
print(f"\nResult: {result.expression!r} score={result.score:.4f}")
print(f"expansions={result.expansions}, is_discovery={result.is_discovery}")
