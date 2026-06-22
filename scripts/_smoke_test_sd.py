#!/usr/bin/env python3
"""Quick smoke test for symmetry_discovery module."""
import sys
sys.path.insert(0, '/home/blueman1818/Projects/theta-core')

from src.physics.symmetry_discovery import (
    SymmetryDiscoverer, SymmetryScorer, CandidateGroup,
    generate_candidate_groups, candidate_to_symmetry_group,
    generate_discovery_training_data, DISCOVERY_GENERATOR_POOL
)
print('Imports OK')
print(f'Generator pool size: {len(DISCOVERY_GENERATOR_POOL)}')

# Test candidate generation
candidates = generate_candidate_groups(max_groups=100)
print(f'Generated {len(candidates)} candidates')
print(f'First candidate: {[g.name for g in candidates[0]]}')

# Test training data generation
obs, gt = generate_discovery_training_data()
print(f'Training scenarios: {len(obs)}')
for o in obs:
    gen_count = len(gt[o.id])
    print(f'  {o.id}: {gen_count} generators')

# Test scorer creation
scorer = SymmetryScorer()
print(f'Scorer params: {scorer.count_parameters()}')

# Test feature building
features = scorer._build_features([candidates[0][0]], obs[0])
print(f'Feature vector size: {len(features)}')

# Test discoverer creation
discoverer = SymmetryDiscoverer(scorer=scorer, max_candidates=50)
print('Discoverer created OK')
print('All basic tests passed!')
