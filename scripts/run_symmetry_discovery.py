#!/usr/bin/env python3
"""Run the full symmetry discovery pipeline: train + evaluate + save."""
import sys
sys.path.insert(0, '/home/blueman1818/Projects/theta-core')

from src.physics.symmetry_discovery import run_symmetry_discovery_pipeline

results = run_symmetry_discovery_pipeline(
    checkpoint_path='checkpoints/symmetry_discoverer.pt',
    results_path='data/symmetry_discovery_results.json',
    max_candidates=500,
)

print()
print('ACCEPTANCE CHECKS:')
for name, check in results['acceptance_checks'].items():
    status = "PASS" if check.get("passed") else "FAIL"
    score = check.get("score", "N/A")
    print(f'  {name}: {status}  (score={score})')
print(f'All passed: {results["all_passed"]}')
