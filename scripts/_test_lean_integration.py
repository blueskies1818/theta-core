#!/usr/bin/env python3
"""Test the import chain for the new Lean proof integration."""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Test 1: Import the new function
from src.physics.auto_lean import generate_dimensional_constancy_proof, get_available_dim_proofs

proofs = get_available_dim_proofs()
print(f"Available dim constancy proofs: {proofs}")

# Test 2: Generate a proof
expr, code, ok, err = generate_dimensional_constancy_proof("E*lambda")
print(f"E*lambda: ok={ok}, err='{err}'")
print(f"Code length: {len(code)}")

# Test 3: Generate all proofs
for p in proofs:
    _, _, ok, err = generate_dimensional_constancy_proof(p)
    status = "PASS" if ok else f"FAIL: {err[:60]}"
    print(f"  {p}: {status}")

# Test 4: Unknown expression
_, _, ok, err = generate_dimensional_constancy_proof("nonexistent")
print(f"Unknown expr: ok={ok}, err='{err}'")

# Test 5: E^2-p^2 (bonus proof)
_, _, ok, err = generate_dimensional_constancy_proof("E^2-p^2")
print(f"E^2-p^2 (bonus): ok={ok}")
