"""Smoke test for symmetry module."""
import json
import sys
sys.path.insert(0, ".")
from src.physics.symmetry import run_symmetry_smoke_test

results = run_symmetry_smoke_test()
print(json.dumps(results, indent=2))
