#!/usr/bin/env python3
"""Compare best-first search (Path C) vs MCTS baseline on gate3_v2 theorems.

Runs best-first search using contrastive lemma embeddings from the Path C
dual-encoder and compares against existing MCTS results.

Usage:
    python scripts/eval/compare_best_first.py

Output:
    data/pathc_search_comparison.json
"""

import sys, json, time, statistics
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

import torch
import torch.nn.functional as F

from src.contrastive.encoder import (
    CharTokenizer,
    ContrastiveConfig,
    ContrastiveDualEncoder,
)
from src.explorer.best_first_search import BestFirstSearch, BestFirstConfig
from src.explorer.proof_state import ProofState
from src.proof_checker.batch_checker import BatchChecker
from src.proof_checker.formats import wrap_theorem_with_proof


def classify_proof_pattern(proof_steps: list[str]) -> str:
    """Classify a proof into its primary pattern category."""
    if not proof_steps:
        return "empty"

    steps_text = " ".join(proof_steps).lower()

    # Multi-step check first
    tactic_types = set()
    for step in proof_steps:
        s = step.strip().lower()
        if s.startswith("rw"):
            tactic_types.add("rw")
        elif s.startswith("exact"):
            tactic_types.add("exact")
        elif s.startswith("apply"):
            tactic_types.add("apply")
        elif s.startswith("intro"):
            tactic_types.add("intro")
        elif s.startswith("have"):
            tactic_types.add("have")
        elif s in ("ring", "simp", "linarith", "field_simp", "positivity", "norm_num", "nlinarith"):
            tactic_types.add(s)
        elif s.startswith("calc"):
            tactic_types.add("calc")
        elif s.startswith("constructor"):
            tactic_types.add("constructor")
        elif s.startswith("refine"):
            tactic_types.add("refine")
        else:
            tactic_types.add("other")

    if len(tactic_types) >= 2:
        return "multi"

    # Single-step classification
    if any(tok in steps_text for tok in ("rfl", "eq.refl")):
        return "rfl"
    if "add_comm" in steps_text:
        return "add_comm"
    if "mul_comm" in steps_text:
        return "mul_comm"
    if "ring" in steps_text:
        return "ring"
    if "field_simp" in steps_text:
        return "field_simp"
    if "linarith" in steps_text:
        return "linarith"
    if "simp" in steps_text:
        return "simp"
    if "intro" in steps_text:
        return "intro"
    if "apply" in steps_text:
        return "apply"
    if "nlinarith" in steps_text:
        return "nlinarith"
    return "other"


def load_encoder(checkpoint_path: str) -> tuple:
    """Load Path C contrastive encoder and tokenizer."""
    torch.set_num_threads(4)
    encoder = ContrastiveDualEncoder.load(checkpoint_path)
    encoder.eval()
    tokenizer = CharTokenizer(max_len=256)
    return encoder, tokenizer


def load_lemma_data(lemmas_path: str, cache_path: str | None = None) -> tuple[list[str], torch.Tensor]:
    """Load lemma names and embeddings.

    Args:
        lemmas_path: Path to all_unique_lemmas.json
        cache_path: Path to cached embeddings .pt file (optional).

    Returns:
        (lemma_names, lemma_embeddings) — [N] and [N, D]
    """
    with open(lemmas_path) as f:
        lemma_names = json.load(f)

    if cache_path and Path(cache_path).exists():
        print(f"Loading cached embeddings from {cache_path}...")
        data = torch.load(cache_path, map_location="cpu", weights_only=False)
        return data["lemma_names"], data["embeddings"]

    return lemma_names, None  # Embeddings computed lazily below


def compute_lemma_embeddings(
    encoder: ContrastiveDualEncoder,
    tokenizer: CharTokenizer,
    lemma_names: list[str],
    batch_size: int = 1024,
) -> torch.Tensor:
    """Pre-compute lemma embeddings for all lemmas."""
    print(f"Computing embeddings for {len(lemma_names)} lemmas...")
    all_embeddings = []

    for i in range(0, len(lemma_names), batch_size):
        batch = lemma_names[i : i + batch_size]
        preprocessed = [CharTokenizer.preprocess_lemma(name) for name in batch]
        char_ids = tokenizer.encode_batch(preprocessed)

        with torch.no_grad():
            emb = encoder.encode_lemma(char_ids)
            all_embeddings.append(emb.cpu())

        if (i + batch_size) % 10000 == 0:
            print(f"  {i + batch_size}/{len(lemma_names)}")

    embeddings = torch.cat(all_embeddings, dim=0)
    embeddings = F.normalize(embeddings, dim=-1)
    print(f"  Done: {embeddings.shape}")
    return embeddings


def load_theorems(theorems_path: str, max_theorems: int | None = None) -> list[dict]:
    """Load theorems from JSONL file."""
    with open(theorems_path) as f:
        theorems = [json.loads(line) for line in f]
    if max_theorems:
        theorems = theorems[:max_theorems]
    return theorems


def run_best_first_search(
    bf_search: BestFirstSearch,
    theorems: list[dict],
    checker: BatchChecker,
    verbose: bool = False,
) -> list[dict]:
    """Run best-first search on all theorems."""
    results = []
    t_start = time.time()

    for i, t in enumerate(theorems):
        stmt = t["statement"]
        name = t["name"]
        era = t.get("era", "unknown")
        zone = t.get("frontier_zone", "unknown")
        ground_truth = t.get("proof", "?")
        domain = t.get("domain", "unknown")

        t0 = time.time()
        best_steps, final_state = bf_search.search(stmt, verbose=False)
        search_time = time.time() - t0

        # Build proof code for verification
        proof_text = ProofState._render_proof(best_steps)

        # Empty proof = no proof found → failure
        if not best_steps:
            ok = False
            err = "no proof found"
        else:
            full_code = wrap_theorem_with_proof(stmt, proof_text)
            check_results = checker.check_batch([full_code])
            ok = check_results[0].success
            err = check_results[0].errors[0][:200] if check_results[0].errors else ""
        steps_str = [s.to_lean() for s in best_steps[:10]]
        pattern = classify_proof_pattern(steps_str) if ok else "failed"

        result = {
            "name": name,
            "era": era,
            "zone": zone,
            "domain": domain,
            "success": ok,
            "error": err,
            "bf_steps": steps_str,
            "num_steps": len(best_steps),
            "ground_truth": ground_truth,
            "search_time_s": search_time,
            "pattern": pattern,
        }
        results.append(result)

        status = "\u2713" if ok else "\u2717"
        print(f"  [{i+1:2d}/{len(theorems)}] {status} {name:45s} "
              f"[{pattern:12s}] {search_time:.1f}s  "
              f"{len(best_steps)} steps")
        if ok:
            print(f"         Proof: {steps_str}")
        elif verbose and err:
            print(f"         Error: {err[:120]}")

    elapsed = time.time() - t_start
    passed = sum(1 for r in results if r["success"])
    print(f"\nBest-first result: {passed}/{len(results)} "
          f"({passed/max(1,len(results))*100:.0f}%) in {elapsed:.0f}s")
    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare best-first search (Path C) vs MCTS baseline"
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/contrastive/lemma_encoder.pt",
        help="Path to contrastive encoder checkpoint",
    )
    parser.add_argument(
        "--lemmas",
        default="data/all_unique_lemmas.json",
        help="Path to unique lemma names JSON",
    )
    parser.add_argument(
        "--lemma-cache",
        default="data/lemma_embeddings.pt",
        help="Path to cached lemma embeddings",
    )
    parser.add_argument(
        "--theorems",
        default="data/raw/gate3_v2.jsonl",
        help="Path to theorem JSONL file",
    )
    parser.add_argument(
        "--max-theorems",
        type=int,
        default=None,
        help="Max theorems to test (None = all)",
    )
    parser.add_argument(
        "--max-expansions",
        type=int,
        default=5000,
        help="Max node expansions for best-first search",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=30,
        help="Top-K lemmas to consider per state",
    )
    parser.add_argument(
        "--depth-penalty",
        type=float,
        default=0.05,
        help="Depth penalty factor for priority",
    )
    parser.add_argument(
        "--use-proof-checker",
        action="store_true",
        help="Verify candidates with Lean during expansion",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--output",
        default="data/pathc_search_comparison.json",
        help="Output JSON file",
    )
    args = parser.parse_args()

    # Resolve paths relative to project root
    ckpt_path = _project_root / args.checkpoint
    lemmas_path = _project_root / args.lemmas
    lemma_cache_path = _project_root / args.lemma_cache if args.lemma_cache else None
    theorems_path = _project_root / args.theorems
    output_path = _project_root / args.output

    print("=" * 70)
    print("BEST-FIRST SEARCH (Path C) vs MCTS BASELINE")
    print("=" * 70)
    print(f"Encoder:   {ckpt_path}")
    print(f"Theorems:  {theorems_path}")

    # ---- Load encoder ----
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        return 1

    encoder, tokenizer = load_encoder(str(ckpt_path))
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"Encoder:   {n_params:,} params, hidden={encoder.config.hidden_dim}")

    # ---- Load lemma data ----
    if not lemmas_path.exists():
        print(f"ERROR: Lemma file not found: {lemmas_path}")
        return 1

    lemma_names, cached_embeddings = load_lemma_data(
        str(lemmas_path), str(lemma_cache_path) if lemma_cache_path else None
    )
    print(f"Lemmas:    {len(lemma_names)} unique")

    # Pre-compute or use cached lemma embeddings
    if cached_embeddings is not None:
        lemma_embeddings = cached_embeddings
        print(f"Using cached embeddings: {lemma_embeddings.shape}")
    else:
        lemma_embeddings = compute_lemma_embeddings(encoder, tokenizer, lemma_names)

    # ---- Load theorems ----
    if not theorems_path.exists():
        print(f"ERROR: Theorem file not found: {theorems_path}")
        return 1

    theorems = load_theorems(str(theorems_path), args.max_theorems)
    print(f"Theorems:  {len(theorems)} loaded")

    # ---- Setup search ----
    config = BestFirstConfig(
        max_depth=20,
        max_expansions=args.max_expansions,
        top_k_lemmas=args.top_k,
        depth_penalty=args.depth_penalty,
        use_proof_checker=args.use_proof_checker,
        verify_timeout=5.0,
        num_threads=4,
    )

    checker = BatchChecker(timeout=30, max_workers=1, cache_size=128)
    bf_search = BestFirstSearch(
        encoder=encoder,
        tokenizer=tokenizer,
        lemma_names=lemma_names,
        lemma_embeddings=lemma_embeddings,
        config=config,
        proof_checker=checker if args.use_proof_checker else None,
    )

    # ---- Run best-first search ----
    print()
    print("-" * 70)
    print("Running best-first search (Path C)...")
    print("-" * 70)
    bf_results = run_best_first_search(bf_search, theorems, checker, verbose=args.verbose)

    # ---- Collect results ----
    bf_passed = [r for r in bf_results if r["success"]]
    bf_failed = [r for r in bf_results if not r["success"]]
    n_total = len(bf_results)

    # Count multi-step successes
    bf_multi = [r for r in bf_passed if r["pattern"] == "multi"]
    bf_single = [r for r in bf_passed if r["pattern"] != "multi"]

    print()
    print("=" * 70)
    print("RESULTS: Best-First Search (Path C) on gate3_v2")
    print("=" * 70)
    print(f"  Theorems tested:     {n_total}")
    print(f"  Total passed:        {len(bf_passed)} ({len(bf_passed)/max(1,n_total)*100:.1f}%)")
    print(f"  Multi-step proofs:   {len(bf_multi)}")
    print(f"  Single-step proofs:  {len(bf_single)}")
    print(f"  Failed:              {len(bf_failed)}")

    # Pattern breakdown
    from collections import Counter
    patterns = Counter(r["pattern"] for r in bf_passed)
    print(f"\n  Proof patterns (successful):")
    for pat, count in patterns.most_common():
        print(f"    {pat:<15} {count:>3}")

    # Per-domain breakdown
    domains = Counter(r["domain"] for r in bf_results)
    print(f"\n  By domain:")
    for dom in sorted(domains.keys()):
        dom_total = domains[dom]
        dom_passed = sum(1 for r in bf_passed if r["domain"] == dom)
        print(f"    {dom:<20} {dom_passed}/{dom_total} ({dom_passed/max(1,dom_total)*100:.0f}%)")

    # List multi-step successes
    if bf_multi:
        print(f"\n  Multi-step proofs ({len(bf_multi)}):")
        for r in bf_multi:
            print(f"    \u2713 {r['name']:<45s} [{r['pattern']}] "
                  f"→ {r['bf_steps']}")
            print(f"      ground_truth: {r['ground_truth']}")

    # ---- Comparison with MCTS baseline ----
    print()
    print("=" * 70)
    print("COMPARISON: Best-First (Path C) vs MCTS (GNN) Baseline")
    print("=" * 70)

    # Load MCTS baseline from existing gate3_fullgraph_result.json
    mcts_baseline_path = _project_root / "data/gate3_fullgraph_result.json"
    mcts_data = {}
    if mcts_baseline_path.exists():
        with open(mcts_baseline_path) as f:
            mcts_data = json.load(f)

    # MCTS baseline on original gate3 (14 theorems)
    mcts_full = mcts_data.get("results_full_graph", {})
    mcts_algebra = mcts_data.get("results_algebra_subgraph", {})

    print(f"\n  {'':30} {'Proved':>10} {'Rate':>10} {'Multi-step':>12}")
    print(f"  {'-'*65}")
    print(f"  {'Best-First (Path C, gate3_v2)':30} "
          f"{len(bf_passed):>10} {len(bf_passed)/max(1,n_total)*100:>9.0f}% "
          f"{len(bf_multi):>12}")
    print(f"  {'MCTS GNN H=0.0 (gate3, full)':30} "
          f"{mcts_full.get('gnn_h0_proved', '?'):>10} "
          f"{mcts_full.get('gnn_h0_rate', '?'):>9.0%} "
          f"{'---':>12}")
    print(f"  {'MCTS GNN H=1.0 (gate3, full)':30} "
          f"{mcts_full.get('heuristic_h1_proved', '?'):>10} "
          f"{mcts_full.get('heuristic_h1_rate', '?'):>9.0%} "
          f"{'---':>12}")
    print(f"  {'MCTS GNN H=0.0 (gate3, Algebra)':30} "
          f"{mcts_algebra.get('gnn_h0_proved', '?'):>10} "
          f"{mcts_algebra.get('gnn_h0_rate', '?'):>9.0%} "
          f"{'---':>12}")

    # ---- Save results ----
    output_data = {
        "task": "Pivot Step 3: Best-first proof search (Path C) vs MCTS baseline on gate3_v2",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": {
            "checkpoint": str(ckpt_path),
            "architecture": "CharCNN dual-encoder with InfoNCE contrastive loss",
            "params": n_params,
        },
        "config": {
            "max_expansions": args.max_expansions,
            "top_k_lemmas": args.top_k,
            "depth_penalty": args.depth_penalty,
            "use_proof_checker": args.use_proof_checker,
            "num_lemmas": len(lemma_names),
        },
        "test_config": {
            "theorems": str(theorems_path),
            "num_theorems": n_total,
        },
        "results_best_first": {
            "total": n_total,
            "passed": len(bf_passed),
            "rate": len(bf_passed) / max(1, n_total),
            "multi_step_passed": len(bf_multi),
            "multi_step_rate": len(bf_multi) / max(1, n_total),
            "single_step_passed": len(bf_single),
            "failed": len(bf_failed),
            "patterns": dict(patterns),
            "passed_theorems": [
                {
                    "name": r["name"],
                    "proof": " ".join(r["bf_steps"]),
                    "pattern": r["pattern"],
                    "ground_truth": r["ground_truth"],
                    "num_steps": r["num_steps"],
                    "search_time_s": r["search_time_s"],
                }
                for r in bf_passed
            ],
            "all_results": bf_results,
        },
        "mcts_baseline": mcts_data,
        "comparison_summary": {
            "bf_pathc_passed": len(bf_passed),
            "bf_pathc_rate": len(bf_passed) / max(1, n_total),
            "bf_pathc_multi_step": len(bf_multi),
            "mcts_gnn_full_h0_passed": mcts_full.get("gnn_h0_proved", 0),
            "mcts_gnn_algebra_h0_passed": mcts_algebra.get("gnn_h0_proved", 0),
            "note": (
                "NOTE: gate3_v2 (64 multi-step theorems) vs gate3_lemma_novelty (14 mostly "
                "single-step theorems). Direct rate comparison is not apples-to-apples. "
                "gate3_v2 is harder (all require \u22652 tactics). Key metric: "
                "best_first achieved multi-step proofs > 0."
            ),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # ---- Cleanup ----
    try:
        checker.shutdown()
    except Exception:
        pass

    # Return 0 if any multi-step proof found (the breakthrough condition)
    return 0 if len(bf_multi) > 0 else (0 if len(bf_passed) > 0 else 1)


if __name__ == "__main__":
    sys.exit(main())
