#!/usr/bin/env python3
"""Verify all Gate 2 theorems pass Lean proof checking.

Reads gate2_training.jsonl and gate2_test_pairs.jsonl, runs lake env lean --stdin
on each theorem's statement+proof, reports pass/fail counts.

Usage:
    python scripts/gates/verify_gate2_all.py [--timeout 15]
"""
import sys
import json
import argparse
import subprocess
import tempfile
from pathlib import Path

LEAN_PREAMBLE_MATHLIB = """import Mathlib
open Real
open Set
open Filter
open Function
open Nat
"""


def build_lean_code(statement: str, proof: str) -> str:
    """Build a self-contained Lean 4 code block for --stdin verification."""
    # Strip existing lemma/theorem names, convert to example
    import re
    stmt = statement.strip()
    # Remove existing :=" + proof part if present
    stmt = re.sub(r'\s*:=.*$', '', stmt)
    # Convert "lemma/theorem name" to "example"
    stmt = re.sub(r'^(lemma|theorem)\s+\S+\s+', 'example ', stmt, count=1)
    
    proof = proof.strip()
    
    # All single-word Lean tactics that need `by`
    _single_tactics = ('by ', 'intro', 'intros', 'apply', 'exact', 'refine',
                       'rcases', 'rinvoke', 'rw', 'rwa', 'erw', 'simp', 'simpa',
                       'have', 'calc', 'linarith', 'nlinarith', 'omega',
                       'ring', 'ring_nf', 'field_simp', 'norm_num', 'positivity',
                       'native_decide', 'trivial')
    # Build the full code
    if '\n' in proof or proof.split()[0].rstrip(':') in _single_tactics:
        # Tactic proof
        if proof.startswith('by '):
            return f"{LEAN_PREAMBLE_MATHLIB}\n{stmt} := {proof}"
        elif '\n' in proof:
            lines = [l.strip() for l in proof.split('\n') if l.strip()]
            indented = '\n'.join(f"  {line}" for line in lines)
            return f"{LEAN_PREAMBLE_MATHLIB}\n{stmt} := by\n{indented}"
        else:
            return f"{LEAN_PREAMBLE_MATHLIB}\n{stmt} := by {proof}"
    else:
        # Term proof
        return f"{LEAN_PREAMBLE_MATHLIB}\n{stmt} := {proof}"


def check_lean(code: str, timeout: float, project_dir: str) -> tuple[bool, str]:
    """Run lake env lean --stdin on the code. Returns (success, error_message)."""
    try:
        proc = subprocess.run(
            ["lake", "env", "lean", "--stdin"],
            input=code,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=project_dir,
        )
        if proc.returncode == 0:
            return True, ""
        err = (proc.stderr or proc.stdout or "").strip()
        # Take first meaningful error line
        err_lines = [l for l in err.split('\n') if l.strip() and 'error' in l.lower()]
        return False, err_lines[0][:200] if err_lines else err[:200]
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except Exception as e:
        return False, str(e)[:200]


def main():
    parser = argparse.ArgumentParser(description="Verify all Gate 2 theorems")
    parser.add_argument("--timeout", type=float, default=15.0)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data" / "raw"
    lake_dir = project_root / "proof_checker_env"

    if not (lake_dir / "lakefile.lean").exists():
        print("ERROR: proof_checker_env Lake project not found")
        sys.exit(1)

    files_to_check = [
        data_dir / "gate2_training.jsonl",
        data_dir / "gate2_test_pairs.jsonl",
    ]

    total = 0
    passed = 0
    failed = 0
    failures = []

    for fpath in files_to_check:
        if not fpath.exists():
            print(f"SKIP: {fpath} not found")
            continue

        print(f"\n{'='*70}")
        print(f"CHECKING: {fpath.name}")
        print(f"{'='*70}")

        with open(fpath) as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    print(f"  SKIP line {line_no}: invalid JSON")
                    continue

                name = obj.get("name", f"line_{line_no}")
                statement = obj.get("statement", "")
                proof = obj.get("proof", obj.get("ground_truth", ""))

                if not statement or not proof:
                    print(f"  SKIP {name}: missing statement or proof")
                    continue

                total += 1
                code = build_lean_code(statement, proof)
                ok, err = check_lean(code, args.timeout, str(lake_dir))

                if ok:
                    passed += 1
                    if total % 20 == 0:
                        print(f"  ... {passed}/{total} passed so far ...")
                else:
                    failed += 1
                    print(f"  FAIL {name}: {err}")
                    failures.append((fpath.name, line_no, name, err))

    print(f"\n{'='*70}")
    print(f"RESULTS: {passed}/{total} PASS, {failed}/{total} FAIL")
    print(f"{'='*70}")

    if failures:
        print(f"\nFAILURES ({len(failures)}):")
        for fname, line_no, name, err in failures:
            print(f"  {fname}:{line_no} {name}: {err}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
