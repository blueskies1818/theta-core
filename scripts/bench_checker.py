#!/usr/bin/env python3
"""Benchmark Lean check latency with cached env."""
import sys, time
sys.path.insert(0, '/home/blueman1818/Projects/theta-core')

from src.proof_checker.lean_interface import LeanProofChecker, _clear_lake_env_cache

# Clear cache for fresh start
_clear_lake_env_cache()

c = LeanProofChecker(project_dir=None, timeout=15)

# Simple theorem that passes
code_ok = 'example : 1 + 1 = 2 := rfl'
code_fail = 'example : 1 + 1 = 3 := rfl'

# Warm up (first call inc env capture)
print("--- Warmup (includes env capture) ---")
t0 = time.time()
r = c.check(code_ok)
print(f"  success={r.success} time={time.time()-t0:.2f}s")

print("\n--- 10 sequential checks (cached env) ---")
times = []
for i in range(10):
    t0 = time.time()
    r = c.check(code_ok if i % 2 == 0 else code_fail)
    elapsed = time.time() - t0
    times.append(elapsed)
    print(f"  [{i}] success={r.success} time={elapsed:.2f}s" + 
          (" (cached)" if i > 0 and elapsed < 0.01 else ""))

print(f"\nAvg per-check time (excl warmup): {sum(times)/len(times):.2f}s")
print(f"Min: {min(times):.2f}s, Max: {max(times):.2f}s")
print(f"For 3000 checks at avg rate: {3000 * sum(times)/len(times):.0f}s = {3000 * sum(times)/len(times)/60:.1f}m")
