#!/usr/bin/env python3
"""Build proof-step pretraining dataset from mathlib4 theorem proofs.

Extracts (goal_text, lemma_used) pairs for supervised pretraining of the GNN
on proof-step prediction: "given this goal, which lemma should I use?"

Each mathlib4 theorem has a proof body. We extract the first lemma/identifier
referenced and pair it with the goal extracted from the theorem statement.

Usage:
    python scripts/build_proof_step_data.py --max-pairs 50000
"""

import argparse, json, re, sys
from pathlib import Path
from collections import Counter

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))


# Common Lean tactics — these are NOT lemmas, they're proof structure
_TACTIC_KEYWORDS = {
    "apply", "exact", "refine", "intro", "intros", "rcases", "rintro",
    "rw", "rwa", "erw", "simp", "simpa", "simp_rw", "dsimp",
    "cases", "case", "induction", "constructor", "left", "right", "split",
    "have", "let", "show", "suffices", "calc", "convert", "gcongr",
    "by_contra", "exfalso", "push_neg", "obtain", "set", "choose",
    "positivity", "linarith", "nlinarith", "omega", "norm_num", "norm_cast",
    "field_simp", "ring", "ring_nf", "native_decide", "dec_trivial",
    "repeat", "try", "all_goals", "any_goals", "filter_upwards",
    "specialize", "generalize", "rename", "revert", "clear",
    "apply_mod_cast", "exact_mod_cast", "rw_mod_cast",
    "assumption", "trivial", "contradiction", "done", "skip",
    "first", "solve", "fail", "admit",
    "rfl", "rfl'",  # Definitional equality — not a lemma but often the right answer
}


def extract_lemma_from_proof(proof_text: str) -> str | None:
    """Extract the first meaningful lemma name from a Lean proof.

    Skips tactics and built-in keywords to find actual lemma references.
    """
    if not proof_text:
        return None

    # Clean the proof text
    text = proof_text.strip()

    # Try to find lemma references in various patterns:
    #   apply <lemma>
    #   exact <lemma>
    #   rw [<lemma>]
    #   rw [<lemma>, ...]
    #   refine <lemma>
    #   have ... := <lemma>

    # Pattern 1: rw [lemma1, lemma2, ...]
    rw_match = re.findall(r'rw\s*\[([^\]]+)\]', text)
    for match in rw_match:
        lemmas = [l.strip() for l in match.split(",")]
        for lemma in lemmas:
            # Skip "← h" patterns (hypothesis rewrites)
            if lemma and not lemma.startswith("←") and not lemma.startswith("h ") and lemma not in _TACTIC_KEYWORDS:
                # Extract just the lemma name (before any space or argument)
                lemma_name = lemma.split()[0] if " " in lemma else lemma
                result = _filter_lemma(lemma_name) if lemma_name else None
                if result:
                    return result

    # Pattern 2: apply/exact/refine <lemma>
    for tactic in ["apply", "exact", "refine"]:
        pattern = r'\b' + tactic + r'\s+([^\s\{\[\(]+)'
        match = re.findall(pattern, text)
        for m in match:
            result = _filter_lemma(m)
            if result and result not in _TACTIC_KEYWORDS:
                return result

    # Pattern 3: have ... := <lemma>
    have_match = re.findall(r':=\s*([^\s,]+)', text)
    for m in have_match:
        result = _filter_lemma(m)
        if result and result not in _TACTIC_KEYWORDS:
            return result

    # Pattern 4: simpa [...] using <lemma>
    using_match = re.findall(r'using\s+([^\s]+)', text)
    for m in using_match:
        result = _filter_lemma(m)
        if result and result not in _TACTIC_KEYWORDS:
            return result

    return None


def _is_valid_lemma(name: str) -> bool:
    """Filter out non-lemma tokens from extraction."""
    if len(name) <= 1:
        return False
    if name in ("by", "fun", "this", "show", "calc", "let", "have",
                "suffices", "obtain", "set", "choose", "H", "h₁", "h₂", "h₃"):
        return False
    if name.startswith("?") or name.startswith("⟨") or name.startswith("--"):
        return False
    if name[0] == "h" and len(name) <= 3:
        # Single-letter hypothesis names: h, h0, h1, h', etc
        return False
    try:
        int(name)
        return False
    except ValueError:
        pass
    return True


def _filter_lemma(name: str | None) -> str | None:
    """Apply validity filter to extracted lemma."""
    if name is None:
        return None
    return name if _is_valid_lemma(name) else None


def extract_goal_from_statement(statement: str) -> str:
    """Extract the goal part from a theorem statement.

    "theorem name (args) : A → B := by ..." → "A → B"
    "lemma name (args) : A = B := ..." → "A = B"
    """
    stmt = statement.strip()
    # Remove trailing proof delimiter
    for suffix in [" := by", " : by", " :=", " := ", " :="]:
        if stmt.endswith(suffix):
            stmt = stmt[:-len(suffix)]
            break
    # Find the last ':' which separates the name/args from the type
    if " : " in stmt:
        # Split on the last " : "
        parts = stmt.rsplit(" : ", 1)
        if len(parts) == 2:
            return parts[1].strip()
    elif ":" in stmt:
        parts = stmt.rsplit(":", 1)
        if len(parts) == 2:
            return parts[1].strip()
    return stmt


def main():
    parser = argparse.ArgumentParser(description="Build proof-step pretraining dataset")
    parser.add_argument("--input", default="data/raw/mathlib4_theorems.jsonl")
    parser.add_argument("--output", default="data/raw/proof_step_pairs.jsonl")
    parser.add_argument("--max-pairs", type=int, default=50000)
    parser.add_argument("--min-proof-length", type=int, default=5)
    args = parser.parse_args()

    input_path = _project_root / args.input
    if not input_path.exists():
        print(f"Error: {input_path} not found. Run scripts/prepare_data.py first.")
        sys.exit(1)

    output_path = _project_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pairs = []
    lemma_counts = Counter()
    skipped_no_lemma = 0
    skipped_short = 0
    total = 0

    print(f"Reading {input_path}...")
    with open(input_path) as f:
        for line in f:
            total += 1
            if len(pairs) >= args.max_pairs:
                break
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue

            statement = d.get("statement", "")
            proof = d.get("proof", "")

            if len(proof) < args.min_proof_length:
                skipped_short += 1
                continue

            goal = extract_goal_from_statement(statement)
            if not goal or len(goal) < 3:
                continue

            lemma = extract_lemma_from_proof(proof)
            if lemma is None:
                skipped_no_lemma += 1
                continue

            lemma_counts[lemma] += 1
            pairs.append({
                "goal": goal,
                "lemma": lemma,
                "name": d.get("name", ""),
                "domain": d.get("source_file", "").split("/Mathlib/")[-1].split("/")[0] if "/Mathlib/" in d.get("source_file", "") else "",
            })

    # Write output
    with open(output_path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    print(f"\nResults:")
    print(f"  Total theorems scanned: {total:,}")
    print(f"  Skipped (short proof):  {skipped_short:,}")
    print(f"  Skipped (no lemma):     {skipped_no_lemma:,}")
    print(f"  Valid pairs extracted:  {len(pairs):,}")
    print(f"  Unique lemmas:          {len(lemma_counts):,}")
    print(f"\nTop 20 lemmas used:")
    for lemma, count in lemma_counts.most_common(20):
        print(f"  {lemma:40s}: {count:5d}")
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
