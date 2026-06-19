#!/usr/bin/env python3
"""Verify that the generated physics theorems are valid Lean 4 proofs."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from src.proof_checker.batch_checker import BatchChecker
from src.proof_checker.formats import wrap_theorem_with_proof


def main():
    theorems_path = Path("data/raw/physics_theorems.jsonl")
    if not theorems_path.exists():
        print(f"File not found: {theorems_path}")
        print("Run: python scripts/build/build_physics_theorems.py")
        sys.exit(1)

    with open(theorems_path) as f:
        theorems = [json.loads(line) for line in f]

    print(f"Testing {len(theorems)} physics theorems...\n")

    checker = BatchChecker(timeout=30, max_workers=4, cache_size=128)
    codes = [wrap_theorem_with_proof(t['statement'], t['proof']) for t in theorems]
    results = checker.check_batch(codes)

    passed = 0
    failed = []
    for t, r, code in zip(theorems, results, codes):
        if r.success:
            passed += 1
        else:
            err = r.errors[0][:120] if r.errors else "unknown error"
            failed.append((t['name'], err, t['frontier_zone']))

    print(f"\nPassed: {passed}/{len(theorems)}")
    print(f"Failed: {len(failed)}")
    if failed:
        print("\nFailures:")
        for name, err, zone in failed:
            print(f"  [{zone:25s}] {name}")
            print(f"    {err}")

    try:
        checker.shutdown()
    except Exception:
        pass


if __name__ == '__main__':
    main()
