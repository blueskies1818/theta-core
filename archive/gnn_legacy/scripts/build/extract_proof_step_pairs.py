"""
Extract (goal, lemma_used) pairs from Mathlib4 proof text for contrastive pretraining.

Reads mathlib4_theorems.jsonl, parses proof text to find lemma applications,
and outputs (goal, lemma, name, domain) pairs as proof_step_pairs.jsonl.

Uses the same filtering as graph_builder.extract_references() but adapted to
produce per-step pairs rather than graph edges.

Usage:
    python scripts/build/extract_proof_step_pairs.py \
        --input data/raw/mathlib4_theorems.jsonl \
        --output data/raw/proof_step_pairs.jsonl \
        --min-pairs 50000
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path


# ---------------------------------------------------------------------------
# Lean identifier filtering (mirrors src/explorer/graph_builder.py)
# ---------------------------------------------------------------------------

_LEAN_KEYWORDS: set[str] = {
    "fun", "forall", "exists", "have", "let", "show", "from", "using",
    "with", "where", "do", "match", "if", "then", "else", "by", "open",
    "set_option", "noncomputable", "partial", "mutual", "private",
    "protected", "scoped", "export", "initialize", "syntax", "macro",
    "elab", "deriving", "extends", "in", "at", "as", "hiding",
    "renaming", "def", "instance", "return",
}

_LEAN_TACTICS: set[str] = {
    "apply", "exact", "refine", "intro", "intros", "assumption",
    "rw", "rwa", "erw", "simp", "simpa", "simp_rw", "dsimp",
    "cases", "rcases", "obtain", "induction", "case", "rename",
    "constructor", "left", "right", "split", "ext", "infer_instance",
    "dec_trivial", "native_decide", "omega", "positivity", "linarith",
    "nlinarith", "norm_num", "norm_cast", "ring", "field_simp", "gcongr",
    "calc", "convert", "ac_rfl", "tauto", "aesop",
    "simp_all", "simp_all_arith", "apply_rules", "solve_by_elim",
    "trivial", "rfl", "done", "skip", "admit", "abort",
    "first", "any_goals", "all_goals", "try", "repeat",
    "focus", "rotate", "swap", "on_goal", "conv", "conv_lhs", "conv_rhs",
    "change", "apply_mod_cast", "exact_mod_cast", "push_cast",
    "filter_upwards", "specialize", "generalize",
    "suffices", "revert", "clear", "subst", "injection",
    "contradiction", "exfalso", "by_contra", "push_neg", "choose",
    "set", "trans", "symm", "exacts", "refine'",
    "fail_if_success", "success_if_fail", "guard_hyp", "guard_target",
    "unfold", "delta", "trace", "trace_state",
    "simp_intro", "dsimp_result", "simp_result",
    # Additional tactics / non-lemma identifiers found in extraction
    "only", "this", "use", "mpr", "by_cases", "grind", "classical",
    "rintro", "congr", "contrapose", "split_ifs", "call", "eq_self_iff_true",
    "hcomp", "congrArg", "haves", "show_iff", "fun_prop",
    "also", "haveI", "letI", "applyI",
}

_MODULE_PREFIXES: tuple[str, ...] = (
    "Std.Tactic.", "Lean.", "Lean.Parser.", "Lean.Elab.",
    "Init.", "Tactic.",
)

# Patterns for local variable names
_LOCAL_VAR_RE = re.compile(
    r"^[a-z][\d']*$|"  # single letter + digits/primes
    r"^h[A-Z]?[\d']*$|"  # hypothesis names: h, hA, h1
    r"^h[a-z]{1,4}$|"  # hypothesis names: hx, hxy, hstd, hne
    r"^h[₁₂₃₄₅₆₇₈₉₀]+$|"  # unicode subscript hypotheses
    r"^ih\d*$|"
    r"^IH\d*$|"
    r"^H\d*$|"
    r"^[a-z]+_\d+$"  # indexed vars like x_1
)

# Patterns to strip from matched identifiers (trailing punctuation artifacts)
_TRAILING_PUNCT_RE = re.compile(r'[)}\],;:]+$')
_LEADING_PUNCT_RE = re.compile(r'^[{(\[]+')
# Regex for Lean identifiers: starts with a letter (including Greek),
# then letters, digits, underscores, dots (namespacing), primes, subscripts
_LEAN_IDENT_RE = re.compile(
    r"[a-zA-Zα-ωΑ-Ωλμπφψθα-ωᴀ-᷿]"
    r"[\w.'₀-₉α-ω]*"
    r"[a-zA-Zα-ωΑ-Ωλμπφψθα-ω₀-₉']"
)


def _is_likely_tactic_or_keyword(name: str) -> bool:
    if name in _LEAN_KEYWORDS or name in _LEAN_TACTICS:
        return True
    if name.startswith(_MODULE_PREFIXES):
        return True
    if name.startswith("«"):
        return True
    return False


def _is_local_variable(name: str) -> bool:
    return bool(_LOCAL_VAR_RE.match(name))


def _is_numeric_constant(name: str) -> bool:
    cleaned = name.replace("_", "").replace(".", "").replace("'", "")
    return cleaned.replace(".", "").isdigit() or len(cleaned) == 0


def extract_references(text: str) -> set[str]:
    """Extract Lean identifier references from proof or statement text."""
    if not text:
        return set()
    identifiers: set[str] = set()
    for match in _LEAN_IDENT_RE.finditer(text):
        name = match.group(0)

        # Strip trailing/leading punctuation artifacts
        name = _LEADING_PUNCT_RE.sub('', name)
        name = _TRAILING_PUNCT_RE.sub('', name)

        # Skip very short names
        if len(name) < 3:
            continue
        # Skip known tactics/keywords
        if _is_likely_tactic_or_keyword(name):
            continue
        # Skip local variable patterns
        if _is_local_variable(name):
            continue
        # Skip numeric-looking identifiers
        if _is_numeric_constant(name):
            continue
        # Skip overly long identifiers (likely artifacts)
        if len(name) > 120:
            continue

        identifiers.add(name)

    return identifiers


# ---------------------------------------------------------------------------
# Domain inference (mirrors graph_builder._infer_domain)
# ---------------------------------------------------------------------------

_TWO_LEVEL_DOMAINS = {
    "Algebra": {
        "Polynomial", "MvPolynomial", "SkewPolynomial",
        "Module", "Lie", "Ring", "Group", "Order",
        "Homology", "QuadraticAlgebra", "Quaternion",
    },
    "RingTheory": {
        "Polynomial", "Ideal", "UniqueFactorizationDomain",
        "DedekindDomain", "Valuation", "WittVector",
    },
    "FieldTheory": {
        "Polynomial", "Galois", "SplittingField",
        "AlgebraicClosure", "Separable",
    },
    "Data": {
        "Polynomial", "Real", "Complex", "Int", "Nat",
        "Rat", "Fin", "Finset", "Fintype", "Matrix",
        "Set", "List", "Multiset", "NNReal", "ENNReal",
        "Sigma", "Prod", "Bool", "Option", "Array",
    },
    "Geometry": {
        "Manifold", "Euclidean",
    },
    "Analysis": {
        "Calculus", "InnerProductSpace", "Fourier",
        "Complex", "Convex", "Normed", "ODE",
    },
    "Topology": {
        "Algebra", "MetricSpace", "Instances",
        "UniformSpace", "Compactification",
    },
    "NumberTheory": {
        "Polynomial", "Cyclotomic", "Zeta",
    },
    "LinearAlgebra": {
        "Matrix", "Basis", "BilinearForm",
        "QuadraticForm", "AffineSpace",
    },
}


def infer_domain(source_file: str) -> str:
    """Infer mathematical domain from source file path."""
    rel = None
    if "Mathlib/" in source_file:
        rel = source_file.split("Mathlib/", 1)[1]
    elif "../mathlib4/Mathlib/" in source_file:
        rel = source_file.split("../mathlib4/Mathlib/", 1)[1]
    if not rel:
        return "Unknown"
    parts = rel.split("/")
    if not parts:
        return "Unknown"
    top_domain = parts[0]
    sub_candidate = parts[1] if len(parts) >= 2 else None
    if sub_candidate and sub_candidate.endswith(".lean"):
        sub_candidate = sub_candidate[:-5]
    sub_map = _TWO_LEVEL_DOMAINS.get(top_domain)
    if sub_map and sub_candidate and sub_candidate in sub_map:
        return f"{top_domain}/{sub_candidate}"
    return top_domain


# ---------------------------------------------------------------------------
# Goal extraction: strip theorem statement to a concise goal
# ---------------------------------------------------------------------------

def extract_goal(statement: str) -> str:
    """Extract the goal (type/proposition) from a theorem statement.

    Strips 'theorem/lemma name (args) :' prefix. Removes formatting.
    Retains the mathematical proposition.
    """
    if not statement:
        return ""
    # Keep everything after the final ':'
    # But be careful with nested colons
    # Pattern: "theorem name (args) : goal :="
    s = statement.strip()
    # Remove trailing :=
    if s.endswith(':='):
        s = s[:-2].strip()

    # Find the declaration colon (first occurrence after keyword+name+args)
    # Simple heuristic: split on ' : ' and take everything after
    parts = s.split(' : ', 1)
    if len(parts) >= 2:
        goal_part = parts[1].strip()
    else:
        goal_part = s.strip()

    # Clean up newlines
    goal_part = goal_part.replace('\u2192', ' → ').replace('\u2200', '∀').replace('\u2203', '∃')
    # Collapse whitespace
    goal_part = re.sub(r'\s+', ' ', goal_part).strip()

    # Truncate if too long (some goals are massive)
    if len(goal_part) > 500:
        goal_part = goal_part[:500] + "..."

    return goal_part


# ---------------------------------------------------------------------------
# Main extraction loop
# ---------------------------------------------------------------------------

def extract_pairs(
    theorems_path: Path,
    domain_filter: set[str] | None = None,
    max_pairs: int | None = None,
    verbose: bool = True,
) -> list[dict]:
    """Extract (goal, lemma_used) pairs from theorem proofs."""
    pairs: list[dict] = []
    processed = 0
    skipped_no_proof = 0
    skipped_no_refs = 0
    skipped_filtered = 0
    parse_errors = 0

    if verbose:
        print(f"Reading theorems from {theorems_path}...", file=sys.stderr)

    with open(theorems_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_pairs and len(pairs) >= max_pairs:
                break

            line = line.strip()
            if not line:
                continue

            try:
                theorem = json.loads(line)
            except json.JSONDecodeError:
                parse_errors += 1
                continue

            processed += 1

            # Domain filter
            domain = infer_domain(theorem.get("source_file", ""))
            if domain_filter and domain not in domain_filter:
                skipped_filtered += 1
                continue

            # Must have proof text
            proof = theorem.get("proof", "")
            if not proof:
                skipped_no_proof += 1
                continue

            # Extract references from proof
            refs = extract_references(proof)
            if not refs:
                skipped_no_refs += 1
                continue

            # Extract goal from statement
            goal = extract_goal(theorem.get("statement", ""))
            if not goal:
                skipped_no_proof += 1
                continue

            name = theorem.get("name", "unknown")

            # Create pairs: one per reference
            for ref in refs:
                if max_pairs and len(pairs) >= max_pairs:
                    break
                # Skip self-references
                if ref == name:
                    continue
                # Skip if ref matches the short name of the theorem
                if "." in name and name.split(".")[-1] == ref:
                    continue

                pairs.append({
                    "goal": goal,
                    "lemma": ref,
                    "name": name,
                    "domain": domain,
                })

            if verbose and processed % 10000 == 0:
                print(
                    f"  Processed {processed} theorems, "
                    f"{len(pairs)} pairs extracted...",
                    file=sys.stderr,
                )

    if verbose:
        print(f"\nDone. Processed {processed} theorems:", file=sys.stderr)
        print(f"  {len(pairs)} pairs extracted", file=sys.stderr)
        print(f"  {skipped_no_proof} skipped (no proof text)", file=sys.stderr)
        print(f"  {skipped_no_refs} skipped (no references found)", file=sys.stderr)
        print(f"  {skipped_filtered} filtered by domain", file=sys.stderr)
        print(f"  {parse_errors} parse errors", file=sys.stderr)

    return pairs


# ---------------------------------------------------------------------------
# Deduplication and merging
# ---------------------------------------------------------------------------

def deduplicate_pairs(pairs: list[dict]) -> list[dict]:
    """Remove duplicate (goal, lemma, domain) tuples."""
    seen: set[tuple[str, str, str]] = set()
    unique: list[dict] = []
    dupes = 0
    for p in pairs:
        key = (p["goal"], p["lemma"], p["domain"])
        if key not in seen:
            seen.add(key)
            unique.append(p)
        else:
            dupes += 1
    if dupes:
        print(f"  Removed {dupes} duplicate pairs", file=sys.stderr)
    return unique


def merge_pairs(existing: list[dict], new: list[dict]) -> list[dict]:
    """Merge new pairs into existing, preserving order and deduplicating."""
    seen: set[tuple[str, str, str]] = set()
    merged: list[dict] = []

    # Existing first
    for p in existing:
        key = (p["goal"], p["lemma"], p.get("domain", "unknown"))
        if key not in seen:
            seen.add(key)
            merged.append(p)

    # New pairs
    added = 0
    for p in new:
        key = (p["goal"], p["lemma"], p.get("domain", "unknown"))
        if key not in seen:
            seen.add(key)
            merged.append(p)
            added += 1

    print(f"  Merged: {len(existing)} existing + {added} new = {len(merged)} total",
          file=sys.stderr)
    return merged


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def print_stats(pairs: list[dict]):
    """Print statistics about the extracted pairs."""
    domains: Counter = Counter()
    lemmas: Counter = Counter()
    for p in pairs:
        domains[p.get("domain", "unknown")] += 1
        lemmas[p.get("lemma", "?")] += 1

    print(f"\n=== Pair statistics ===", file=sys.stderr)
    print(f"Total pairs: {len(pairs)}", file=sys.stderr)
    print(f"Unique lemmas: {len(lemmas)}", file=sys.stderr)
    print(f"Domains: {len(domains)}", file=sys.stderr)
    for domain, count in domains.most_common(25):
        print(f"  {domain}: {count}", file=sys.stderr)
    print(f"\nTop 20 lemmas:", file=sys.stderr)
    for lemma, count in lemmas.most_common(20):
        print(f"  {lemma}: {count}", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract (goal, lemma_used) pairs from Mathlib4 proofs"
    )
    parser.add_argument(
        "--input", type=Path,
        default=Path("data/raw/mathlib4_theorems.jsonl"),
        help="Path to mathlib4_theorems.jsonl",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/raw/proof_step_pairs.jsonl"),
        help="Output path for proof_step_pairs.jsonl",
    )
    parser.add_argument(
        "--merge-existing", action="store_true",
        help="Merge with existing pairs instead of overwriting",
    )
    parser.add_argument(
        "--domain", type=str, nargs="*",
        help="Only extract from specific domains (e.g., Algebra RingTheory)",
    )
    parser.add_argument(
        "--min-pairs", type=int, default=50000,
        help="Minimum number of pairs to extract (default: 50000)",
    )
    parser.add_argument(
        "--max-pairs", type=int, default=0,
        help="Maximum pairs to extract (0 = no limit)",
    )
    parser.add_argument(
        "--no-dedup", action="store_true",
        help="Skip deduplication step",
    )
    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = Path(__file__).resolve().parent.parent
    input_path = project_root / args.input if not args.input.is_absolute() else args.input
    output_path = project_root / args.output if not args.output.is_absolute() else args.output

    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    domain_set = set(args.domain) if args.domain else None

    print(f"=== Proof Step Pair Extraction ===", file=sys.stderr)
    print(f"Input:  {input_path}", file=sys.stderr)
    print(f"Output: {output_path}", file=sys.stderr)
    if domain_set:
        print(f"Domains: {sorted(domain_set)}", file=sys.stderr)
    print(f"Target: {args.min_pairs} pairs", file=sys.stderr)

    # Extract
    max_pairs = args.max_pairs if args.max_pairs > 0 else max(args.min_pairs * 3, 200000)
    pairs = extract_pairs(
        input_path,
        domain_filter=domain_set,
        max_pairs=max_pairs,
        verbose=True,
    )

    # Deduplicate new pairs
    if not args.no_dedup:
        pairs = deduplicate_pairs(pairs)

    # Merge with existing if requested
    if args.merge_existing and output_path.exists():
        existing = []
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        existing.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        pairs = merge_pairs(existing, pairs)

    # Print stats
    print_stats(pairs)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(pairs)} pairs to {output_path}", file=sys.stderr)
    if args.min_pairs and len(pairs) < args.min_pairs:
        print(
            f"WARNING: Only got {len(pairs)} pairs, target was {args.min_pairs}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
