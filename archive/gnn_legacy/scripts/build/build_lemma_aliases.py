"""
STEP 2: Build lemma alias map — map every proof-step lemma name to exactly one
GNN index key, using exact, suffix, dot-path folding, and Jaccard matching.

Also applies aggressive artifact cleaning for parser noise (leading/trailing
parens, @, brackets).

Output: data/lemma_aliases.json — {lemma_name: {"target": index_key, "strategy": str}}
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


def load_index(path: Path) -> tuple[set[str], list[str], dict[str, list[str]]]:
    """Load index and build token acceleration structure."""
    raw = json.loads(path.read_text())
    keys = list(raw.keys())
    key_set = set(keys)
    
    # Token → keys lookup for Jaccard acceleration
    token_to_keys: dict[str, list[str]] = defaultdict(list)
    for key in keys:
        seen_tokens = set()
        for tok in key.split("_"):
            if tok not in seen_tokens:
                token_to_keys[tok].append(key)
                seen_tokens.add(tok)
    
    return key_set, keys, dict(token_to_keys)


def load_pairs(path: Path) -> list[dict]:
    """Load proof step pairs."""
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pairs.append(json.loads(line))
    return pairs


# ---------------------------------------------------------------------------
# Artifact cleaning
# ---------------------------------------------------------------------------

def clean_artifact(name: str) -> str:
    """Aggressively clean parser artifacts from lemma names."""
    # Strip leading: parens, @, brackets, dashes
    name = re.sub(r'^[\(\)@\[\]\{\}\-]+', '', name)
    # Strip trailing: parens, @, brackets, digits, commas, colons
    name = re.sub(r'[\(\)@\[\]\{\}\,\:\;\d]+$', '', name)
    return name


# ---------------------------------------------------------------------------
# Matching strategies
# ---------------------------------------------------------------------------

def match_suffix(lemma: str, key_set: set[str]) -> str | None:
    """Last dot-segment match."""
    short = lemma.rsplit(".", 1)[-1]
    return short if short in key_set else None


def match_dotpath(lemma: str, key_set: set[str]) -> str | None:
    """Try all suffix paths from dot-separated components."""
    parts = lemma.split(".")
    for i in range(len(parts)):
        cand = ".".join(parts[i:])
        if cand in key_set:
            return cand
    return None


def match_jaccard(
    lemma: str, 
    key_set: set[str], 
    token_to_keys: dict[str, list[str]],
    threshold: float = 0.7,
) -> tuple[str, float] | None:
    """Jaccard similarity on underscore tokens with token index acceleration."""
    # Tokenize
    cleaned = re.sub(r'[\(\)@\[\]\{\}\-]', '', lemma)
    tokens = set(cleaned.replace(".", "_").split("_"))
    tokens = {t for t in tokens if t and len(t) > 1}  # Skip single-char tokens
    if not tokens:
        return None
    
    # Collect candidate keys (keys sharing at least one token)
    candidates: set[str] = set()
    for tok in tokens:
        if tok in token_to_keys:
            candidates.update(token_to_keys[tok])
    
    if not candidates:
        return None
    
    # Find best match
    best_score = 0.0
    best_key = None
    for key in candidates:
        key_tokens = set(key.split("_"))
        key_tokens = {t for t in key_tokens if t and len(t) > 1}
        if not key_tokens:
            continue
        inter = tokens & key_tokens
        uni = tokens | key_tokens
        score = len(inter) / len(uni)
        if score > best_score:
            best_score = score
            best_key = key
    
    if best_score >= threshold and best_key is not None:
        return best_key, best_score
    return None


def resolve_lemma(
    lemma: str,
    key_set: set[str],
    token_to_keys: dict[str, list[str]],
) -> tuple[str, str] | None:
    """Try all strategies in order. Returns (target_key, strategy) or None."""
    
    # Try raw lemma first
    if lemma in key_set:
        return lemma, "exact"
    
    # Clean artifacts
    cleaned = clean_artifact(lemma)
    if cleaned and cleaned in key_set:
        return cleaned, "clean_exact"
    
    # Suffix on original
    result = match_suffix(lemma, key_set)
    if result:
        return result, "suffix"
    
    # Suffix on cleaned
    if cleaned and cleaned != lemma:
        result = match_suffix(cleaned, key_set)
        if result:
            return result, "clean_suffix"
    
    # Dot-path on original
    result = match_dotpath(lemma, key_set)
    if result:
        return result, "dotpath"
    
    # Dot-path on cleaned
    if cleaned and cleaned != lemma:
        result = match_dotpath(cleaned, key_set)
        if result:
            return result, "clean_dotpath"
    
    # Jaccard on cleaned (lower threshold for aliases)
    jresult = match_jaccard(cleaned if cleaned else lemma, key_set, token_to_keys, 0.7)
    if jresult:
        return jresult[0], f"jaccard_{jresult[1]:.2f}"
    
    return None


def main():
    parser = argparse.ArgumentParser(description="Build lemma alias map")
    parser.add_argument("--pairs", type=Path, default=Path("data/raw/proof_step_pairs.jsonl"))
    parser.add_argument("--index", type=Path, default=Path("data/graph/dependency_graph_full.index.json"))
    parser.add_argument("--output", type=Path, default=Path("data/lemma_aliases.json"))
    parser.add_argument("--coverage-report", type=Path, default=Path("data/lemma_coverage_audit.json"))
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    pairs_path = project_root / args.pairs if not args.pairs.is_absolute() else args.pairs
    index_path = project_root / args.index if not args.index.is_absolute() else args.index
    output_path = project_root / args.output if not args.output.is_absolute() else args.output
    report_path = project_root / args.coverage_report if not args.coverage_report.is_absolute() else args.coverage_report

    print(f"=== Lemma Alias Builder ===", file=sys.stderr)
    
    # Load
    print("Loading index...", file=sys.stderr)
    key_set, key_list, token_to_keys = load_index(index_path)
    print(f"  {len(key_set)} index keys, {len(token_to_keys)} unique tokens", file=sys.stderr)

    print("Loading pairs...", file=sys.stderr)
    pairs = load_pairs(pairs_path)
    unique_lemmas = sorted(set(p["lemma"] for p in pairs))
    print(f"  {len(unique_lemmas)} unique lemmas", file=sys.stderr)

    # Resolve
    print("Building alias map...", file=sys.stderr)
    aliases: dict[str, dict] = {}
    strategy_counts: Counter = Counter()
    unresolved: list[str] = []

    for i, lemma in enumerate(unique_lemmas):
        if i % 10000 == 0 and i > 0:
            print(f"  {i}/{len(unique_lemmas)}...", file=sys.stderr)

        result = resolve_lemma(lemma, key_set, token_to_keys)
        if result:
            target, strategy = result
            aliases[lemma] = {"target": target, "strategy": strategy}
            strategy_counts[strategy] += 1
        else:
            unresolved.append(lemma)
            strategy_counts["unresolved"] += 1

    # Pair-level stats
    pair_resolved = sum(1 for p in pairs if p["lemma"] in aliases)
    pair_unresolved = len(pairs) - pair_resolved

    # Summary
    total = len(unique_lemmas)
    resolved = total - len(unresolved)
    print(f"\n=== Alias Map Built ===", file=sys.stderr)
    print(f"  Resolved: {resolved}/{total} ({resolved/total:.1%})", file=sys.stderr)
    print(f"  Unresolved: {len(unresolved)}", file=sys.stderr)
    print(f"  Pair-level resolved: {pair_resolved}/{len(pairs)} ({pair_resolved/len(pairs):.1%})", file=sys.stderr)
    for s, c in strategy_counts.most_common():
        print(f"    {s}: {c}", file=sys.stderr)

    # Save alias map
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(aliases, f, indent=2, ensure_ascii=False)
    print(f"\nAlias map saved to {output_path} ({len(aliases)} entries)", file=sys.stderr)

    # Save/update coverage report
    coverage = {
        "summary": {
            "total_pairs": len(pairs),
            "unique_lemmas": total,
            "index_size": len(key_set),
            "overall_lemma_recall": resolved / total if total else 0,
            "pair_level_recall": pair_resolved / len(pairs) if len(pairs) else 0,
            "resolved_lemmas": resolved,
            "unresolved_lemmas": len(unresolved),
        },
        "strategy_breakdown": dict(strategy_counts),
        "unresolved_sample": unresolved[:500],
        "unresolved_count": len(unresolved),
    }
    with open(report_path, "w") as f:
        json.dump(coverage, f, indent=2, ensure_ascii=False)
    print(f"Coverage report saved to {report_path}", file=sys.stderr)

    if unresolved:
        print(f"\nFirst 20 unresolved:", file=sys.stderr)
        for u in unresolved[:20]:
            print(f"  {u}", file=sys.stderr)


if __name__ == "__main__":
    main()
