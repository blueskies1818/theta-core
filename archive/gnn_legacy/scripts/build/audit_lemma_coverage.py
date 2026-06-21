"""
STEP 1: Audit lemma coverage — measure what fraction of proof-step lemmas
resolve to a GNN node index, and produce a mismatch report.

Usage:
    python scripts/build/audit_lemma_coverage.py \
        --pairs data/raw/proof_step_pairs.jsonl \
        --index data/graph/dependency_graph_full.index.json \
        --output data/lemma_coverage_audit.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def load_index(index_path: Path) -> dict[str, int]:
    """Load the node name → index mapping, with both dotted and short names."""
    raw = json.loads(index_path.read_text())
    # raw is {name: name_value} — we need the actual node integer index
    # If values are strings (same as keys), we need to build an inverse mapping
    # Actually, the index.json maps name → node_id (integer index)
    name_to_idx: dict[str, int] = {}
    for k, v in raw.items():
        if isinstance(v, int):
            name_to_idx[k] = v
        elif isinstance(v, str):
            # The value is just the name itself — convert to sequential index
            name_to_idx[k] = len(name_to_idx)
    return name_to_idx


def load_pairs(pairs_path: Path) -> list[dict]:
    """Load proof step pairs."""
    pairs = []
    with open(pairs_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pairs.append(json.loads(line))
    return pairs


# ---------------------------------------------------------------------------
# Exact match
# ---------------------------------------------------------------------------
def exact_match(lemma: str, index: dict[str, int]) -> int | None:
    """Return node index if lemma exactly matches an index key."""
    return index.get(lemma)


# ---------------------------------------------------------------------------
# Fuzzy matching strategies
# ---------------------------------------------------------------------------
def suffix_match(lemma: str, index: dict[str, int]) -> int | None:
    """Match by taking the last dot-separated component (short name)."""
    short = lemma.rsplit(".", 1)[-1]
    return index.get(short)


def dot_path_folding(lemma: str, index: dict[str, int]) -> int | None:
    """Try progressive dot-splitting: Math.Algebra.Ring.basic_add → try various fragments."""
    parts = lemma.split(".")
    # Try last N parts, dropping from the front
    for i in range(len(parts)):
        candidate = ".".join(parts[i:])
        if candidate in index:
            return index[candidate]
    return None


def underscore_token_jaccard(lemma: str, index_keys: list[str], threshold: float = 0.8) -> int | None:
    """Match by Jaccard similarity on underscore tokens."""
    lemma_tokens = set(lemma.replace(".", "_").split("_"))
    if not lemma_tokens:
        return None

    best_score = 0.0
    best_key = None

    for key in index_keys:
        key_tokens = set(key.split("_"))
        if not key_tokens:
            continue
        intersection = lemma_tokens & key_tokens
        union = lemma_tokens | key_tokens
        score = len(intersection) / len(union) if union else 0.0
        if score > best_score:
            best_score = score
            best_key = key

    if best_score >= threshold and best_key is not None:
        return index.get(best_key)
    return None


def resolve_lemma(lemma: str, index: dict[str, int], index_keys: list[str]) -> tuple[int | None, str]:
    """Try all strategies in order. Returns (node_index, strategy_used)."""
    # Strategy 1: exact match
    result = exact_match(lemma, index)
    if result is not None:
        return result, "exact"

    # Strategy 2: suffix (last dot-component)
    result = suffix_match(lemma, index)
    if result is not None:
        return result, "suffix"

    # Strategy 3: dot-path folding
    result = dot_path_folding(lemma, index)
    if result is not None:
        return result, "dot_path"

    # Strategy 4: Jaccard token matching (expensive, do last!)
    result = underscore_token_jaccard(lemma, index_keys)
    if result is not None:
        return result, "jaccard"

    return None, "unresolved"


def main():
    parser = argparse.ArgumentParser(description="Audit lemma coverage against GNN index")
    parser.add_argument("--pairs", type=Path, default=Path("data/raw/proof_step_pairs.jsonl"))
    parser.add_argument("--index", type=Path, default=Path("data/graph/dependency_graph_full.index.json"))
    parser.add_argument("--output", type=Path, default=Path("data/lemma_coverage_audit.json"))
    parser.add_argument("--jaccard-threshold", type=float, default=0.8)
    args = parser.parse_args()

    # Resolve paths: script is in scripts/build/, project root is ../../ 
    project_root = Path(__file__).resolve().parent.parent.parent
    pairs_path = project_root / args.pairs if not args.pairs.is_absolute() else args.pairs
    index_path = project_root / args.index if not args.index.is_absolute() else args.index
    output_path = project_root / args.output if not args.output.is_absolute() else args.output

    print(f"=== Lemma Coverage Audit ===", file=sys.stderr)
    print(f"Pairs: {pairs_path}", file=sys.stderr)
    print(f"Index: {index_path}", file=sys.stderr)

    # Load data
    print("Loading index...", file=sys.stderr)
    index = load_index(index_path)
    index_keys = list(index.keys())
    print(f"  Index has {len(index)} nodes", file=sys.stderr)

    print("Loading pairs...", file=sys.stderr)
    pairs = load_pairs(pairs_path)
    print(f"  {len(pairs)} pairs loaded", file=sys.stderr)

    # Audit
    unique_lemmas = sorted(set(p["lemma"] for p in pairs))
    print(f"  {len(unique_lemmas)} unique lemmas to check", file=sys.stderr)

    strategy_counts: Counter = Counter()
    match_results: dict[str, dict] = {}  # lemma → {index, strategy}
    unmatched: list[str] = []

    print("Resolving lemmas...", file=sys.stderr)
    for i, lemma in enumerate(unique_lemmas):
        if i % 10000 == 0 and i > 0:
            print(f"  {i}/{len(unique_lemmas)}...", file=sys.stderr)

        node_idx, strategy = resolve_lemma(lemma, index, index_keys)
        strategy_counts[strategy] += 1

        if node_idx is not None:
            match_results[lemma] = {"index": node_idx, "strategy": strategy}
        else:
            unmatched.append(lemma)

    # Aggregate pair-level stats
    total_pairs = len(pairs)
    resolved_pairs = sum(1 for p in pairs if p["lemma"] in match_results)
    unresolved_pairs = total_pairs - resolved_pairs

    # Per-domain stats
    domain_stats = defaultdict(lambda: {"total": 0, "resolved": 0})
    for p in pairs:
        domain = p.get("domain", "Unknown")
        domain_stats[domain]["total"] += 1
        if p["lemma"] in match_results:
            domain_stats[domain]["resolved"] += 1

    # Build report
    report = {
        "summary": {
            "total_pairs": total_pairs,
            "unique_lemmas": len(unique_lemmas),
            "index_size": len(index),
            "exact_recall": strategy_counts["exact"] / len(unique_lemmas) if unique_lemmas else 0,
            "overall_recall": (len(unique_lemmas) - len(unmatched)) / len(unique_lemmas) if unique_lemmas else 0,
            "pair_level_recall": resolved_pairs / total_pairs if total_pairs else 0,
            "resolved_lemmas": len(match_results),
            "unresolved_lemmas": len(unmatched),
        },
        "strategy_breakdown": dict(strategy_counts),
        "domain_breakdown": {
            domain: {
                "total": s["total"],
                "resolved": s["resolved"],
                "recall": s["resolved"] / s["total"] if s["total"] else 0,
            }
            for domain, s in sorted(domain_stats.items())
        },
        "unmatched_sample": unmatched[:100],  # First 100 for inspection
        "unmatched_sample_size": len(unmatched),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nReport saved to {output_path}", file=sys.stderr)

    # Print summary
    print(f"\n=== Results ===", file=sys.stderr)
    print(f"  Pair-level recall: {report['summary']['pair_level_recall']:.1%}", file=sys.stderr)
    print(f"  Lemma-level recall: {report['summary']['overall_recall']:.1%}", file=sys.stderr)
    print(f"  Exact matches: {strategy_counts['exact']} ({strategy_counts['exact']/len(unique_lemmas):.1%})", file=sys.stderr)
    print(f"  Suffix matches: {strategy_counts.get('suffix', 0)}", file=sys.stderr)
    print(f"  Dot-path matches: {strategy_counts.get('dot_path', 0)}", file=sys.stderr)
    print(f"  Jaccard matches: {strategy_counts.get('jaccard', 0)}", file=sys.stderr)
    print(f"  Unresolved: {len(unmatched)}", file=sys.stderr)

    if unmatched:
        print(f"\n  First 20 unmatched:", file=sys.stderr)
        for u in unmatched[:20]:
            print(f"    {u}", file=sys.stderr)


if __name__ == "__main__":
    main()
