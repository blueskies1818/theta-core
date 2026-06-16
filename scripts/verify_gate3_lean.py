#!/usr/bin/env python3
"""Verify only gate3_lemma_novelty.jsonl with Lean."""
import sys, json, subprocess, re
from pathlib import Path

PROJECT = Path("/home/blueman1818/Projects/theta-core")
LAKE_DIR = PROJECT / "proof_checker_env"

LEAN_PREAMBLE = """import Mathlib
open Polynomial
open Real
"""

def verify_one(statement, proof, timeout=20):
    stmt = statement.strip()
    stmt = re.sub(r'\s*:=.*$', '', stmt)
    stmt = re.sub(r'^(lemma|theorem)\s+\S+\s+', 'example ', stmt, count=1)
    proof = proof.strip()
    if '\n' in proof:
        lines = [l.strip() for l in proof.split('\n') if l.strip()]
        indented = '\n'.join(f"  {line}" for line in lines)
        code = f"{LEAN_PREAMBLE}\n{stmt} := by\n{indented}"
    elif proof.split()[0].rstrip(':') in ('by', 'intro', 'intros', 'apply', 'exact', 'refine',
            'rcases', 'rw', 'rwa', 'erw', 'simp', 'simpa', 'have', 'calc', 'linarith', 'nlinarith',
            'omega', 'ring', 'ring_nf', 'field_simp', 'norm_num', 'positivity', 'native_decide', 'trivial'):
        code = f"{LEAN_PREAMBLE}\n{stmt} := {proof}" if proof.startswith('by ') else f"{LEAN_PREAMBLE}\n{stmt} := by {proof}"
    else:
        code = f"{LEAN_PREAMBLE}\n{stmt} := {proof}"
    try:
        proc = subprocess.run(["lake", "env", "lean", "--stdin"], input=code, capture_output=True,
                              text=True, timeout=timeout, cwd=str(LAKE_DIR))
        return proc.returncode == 0, (proc.stderr or proc.stdout or "").strip()[:200]
    except Exception as e:
        return False, str(e)[:200]

fpath = PROJECT / "data/raw/gate3_lemma_novelty.jsonl"
total = passed = failed = 0
failures = []

with open(fpath) as f:
    for line_no, line in enumerate(f, 1):
        line = line.strip()
        if not line: continue
        obj = json.loads(line)
        name = obj.get("name", f"L{line_no}")
        ok, err = verify_one(obj["statement"], obj["proof"])
        total += 1
        if ok:
            passed += 1
            if passed % 5 == 0:
                print(f"  ... {passed}/{total} passed ...")
        else:
            failed += 1
            print(f"  FAIL {name}: {err[:150]}")
            failures.append((name, err))

print(f"\n{'='*60}")
print(f"Gate 3 Lean Verification: {passed}/{total} PASS")
print(f"{'='*60}")
if failures:
    print(f"\nFAILURES ({len(failures)}):")
    for name, err in failures:
        print(f"  {name}: {err}")
sys.exit(0 if failed == 0 else 1)
