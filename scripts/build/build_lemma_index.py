"""
Build lemma_index.json for enriched graph (fast version).
Works directly with saved graph files rather than the DependencyGraph API.
"""
import json
import sys
from collections import OrderedDict
from pathlib import Path

def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    graph_dir = project_root / "data/graph"
    
    print("=== Building Lemma Index (fast) ===", file=sys.stderr)
    
    # Load index.json (name → node_id)
    index_path = graph_dir / "dependency_graph_full_v2.index.json"
    with open(index_path) as f:
        name_to_id = json.load(f)
    print(f"Index entries: {len(name_to_id)}", file=sys.stderr)
    
    # Build integer indices from the ordered keys
    # NetworkX nodes maintain insertion order, so we can assign sequential indices
    name_to_idx: dict[str, int] = {}
    
    # First pass: assign integer indices sequentially
    for i, (name, node_id) in enumerate(name_to_id.items()):
        name_to_idx[name] = i
    
    print(f"  Built sequential indices for {len(name_to_idx)} names", file=sys.stderr)
    
    # Add short name fallbacks
    short_adds = 0
    for name, idx in list(name_to_idx.items()):
        if "." in name:
            short = name.rsplit(".", 1)[-1]
            if short not in name_to_idx:
                name_to_idx[short] = idx
                short_adds += 1
    print(f"  Added {short_adds} short name fallbacks", file=sys.stderr)
    
    # Add aliases
    aliases_path = project_root / "data/lemma_aliases.json"
    with open(aliases_path) as f:
        aliases = json.load(f)
    
    alias_adds = 0
    for alias_name, info in aliases.items():
        target = info["target"]
        if target in name_to_idx:
            idx = name_to_idx[target]
            if alias_name not in name_to_idx:
                name_to_idx[alias_name] = idx
                alias_adds += 1
    
    print(f"  Added {alias_adds} alias entries", file=sys.stderr)
    print(f"  Total entries: {len(name_to_idx)}", file=sys.stderr)
    
    # Save
    output_path = graph_dir / "dependency_graph_full_v2.lemma_index.json"
    with open(output_path, "w") as f:
        json.dump(name_to_idx, f, indent=2, ensure_ascii=False)
    print(f"Saved to {output_path}", file=sys.stderr)
    
    # Verify pair-level recall
    pairs_path = project_root / "data/raw/proof_step_pairs.jsonl"
    total = 0
    resolved = 0
    with open(pairs_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = json.loads(line)
            total += 1
            if p["lemma"] in name_to_idx:
                resolved += 1
    
    print(f"\nVerification: {resolved}/{total} pairs resolve ({resolved/total:.1%})", file=sys.stderr)
    
    # Warn if below target
    if resolved / total < 0.90:
        print(f"WARNING: Below 90% target!", file=sys.stderr)
    else:
        print(f"SUCCESS: Above 90% target!", file=sys.stderr)


if __name__ == "__main__":
    main()
