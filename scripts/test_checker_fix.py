#!/usr/bin/env python3
"""Smoke test for the new persistent Lean checker."""
import sys, time
sys.path.insert(0, '/home/blueman1818/Projects/theta-core')

from src.proof_checker.lean_interface import LeanProofChecker

# Test bare lean check
c = LeanProofChecker(project_dir=False, timeout=10)
t0 = time.time()
r = c.check('example : 1 + 1 = 2 := rfl')
elapsed = time.time() - t0
print(f'Bare lean: success={r.success}, errors={r.errors[:3]}, time={elapsed:.2f}s')

# Test cache hit
t0 = time.time()
r2 = c.check('example : 1 + 1 = 2 := rfl')
elapsed = time.time() - t0
print(f'Bare lean (cached): success={r2.success}, time={elapsed:.4f}s')

# Test with Lake project
c2 = LeanProofChecker(project_dir=None, timeout=15)
t0 = time.time()
r3 = c2.check('example : 1 + 1 = 2 := rfl')
elapsed = time.time() - t0
print(f'Lake lean: success={r3.success}, time={elapsed:.2f}s')

# Test failure
r4 = c2.check('example : 1 := 2')
print(f'Lake lean (fail): success={r4.success}, errors={[e[:80] for e in r4.errors[:2]]}')

# Test batch_checker
print('\n--- BatchChecker test ---')
from src.proof_checker.batch_checker import BatchChecker
bc = BatchChecker(timeout=15, max_workers=4, min_batch_size=3)
codes = [
    'example : 1 + 1 = 2 := rfl',
    'example : 2 + 2 = 4 := rfl',
    'example : 1 := 2',  # deliberate fail
    'example : 3 + 3 = 6 := rfl',
]
t0 = time.time()
results = bc.check_batch(codes)
elapsed = time.time() - t0
for i, (code, r) in enumerate(zip(codes, results)):
    print(f'  [{i}] success={r.success} ({elapsed:.2f}s batch)')

print('\nAll checks passed!')
