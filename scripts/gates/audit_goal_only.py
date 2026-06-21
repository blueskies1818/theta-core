#!/usr/bin/env python3
"""
Gate audits for Goal-Only GNN.

Runs:
  1. Gate 1 (purity): Verify no era contamination in training data
  2. Gate B (MRR): Compute retrieval MRR on held-out test set
  3. Gate 3: Full 64-theorem benchmark eval vs 15.6% baseline
  4. Gate C (embeddings): Embedding health check (no collapse, diversity)

Output: data/goal_only_gate_audit.json

Usage:
  python scripts/gates/audit_goal_only.py \
    --model checkpoints/gnn/goal_only/goal_only_encoder.pt \
    --pairs data/raw/proof_step_pairs.jsonl \
    --gate3 data/raw/gate3_v2.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))

from src.explorer.goal_only_encoder import GoalOnlyEncoder


# ---------------------------------------------------------------------------
# Gate 1: Purity audit (simplified — check training data for era leaks)
# ---------------------------------------------------------------------------


POST_1904_KEYWORDS = [
    "quantum", "photon", "planck", "heisenberg", "schrodinger", "schrödinger",
    "wavefunction", "wave_function", "dirac", "pauli", "bose", "fermi", "spinor",
    "hilbert_space", "relativity", "lorentz", "minkowski", "einstein",
    "time_dilation", "spacetime", "spacelike", "quark", "gluon", "higgs",
    "electroweak", "standard_model", "w_boson", "z_boson", "neutrino", "lepton",
    "hadron", "dark_matter", "dark_energy", "gauge_theory", "supersymmetry",
    "entanglement", "bell_inequality", "hubble", "big_bang", "cmb",
    "inflation", "cosmological_constant", "holographic", "string_theory",
]


def audit_purity(pairs_path: str) -> dict:
    """Scan training pairs for post-1904 physics keywords."""
    print("\n--- Gate 1: Purity Audit ---")
    hits: list[dict] = []
    total = 0
    with open(pairs_path) as f:
        for line in f:
            total += 1
            pair = json.loads(line)
            text = (pair.get("goal", "") + " " + pair.get("lemma", "")).lower()
            for kw in POST_1904_KEYWORDS:
                if kw.replace("_", " ") in text or kw in text:
                    hits.append({
                        "keyword": kw,
                        "goal": pair["goal"][:100],
                        "lemma": pair["lemma"],
                    })
                    break

    passed = len(hits) == 0
    result = {
        "gate": "G1_purity",
        "passed": passed,
        "total_pairs_scanned": total,
        "hits": len(hits),
        "details": hits[:10] if hits else [],
    }
    status = "PASS" if passed else "FAIL"
    print(f"  {status}: {len(hits)} post-1904 hits in {total} pairs")
    return result


# ---------------------------------------------------------------------------
# Gate B: MRR (retrieval quality)
# ---------------------------------------------------------------------------


def audit_mrr(
    model: GoalOnlyEncoder,
    pairs_path: str,
    device: torch.device,
    val_split: float = 0.1,
    max_val: int = 500,
    top_k: int = 50,
) -> dict:
    """Compute MRR on held-out validation pairs."""
    print("\n--- Gate B: MRR Audit ---")

    # Load pairs
    pairs = []
    with open(pairs_path) as f:
        for line in f:
            pairs.append(json.loads(line))

    # Build goal→lemmas
    goal_to_lemmas: dict[str, list[str]] = defaultdict(list)
    for p in pairs:
        goal_to_lemmas[p["goal"]].append(p["lemma"])

    all_goals = list(goal_to_lemmas.keys())
    print(f"  Total unique goals: {len(all_goals)}")

    # Split
    import random
    random.seed(42)
    random.shuffle(all_goals)
    split = int(len(all_goals) * (1 - val_split))
    train_goals = all_goals[:split]
    val_goals = all_goals[split:split + max_val]

    # Pre-encode training goals
    print(f"  Pre-encoding {len(train_goals)} training goals...")
    model.eval()
    train_embs = []
    with torch.no_grad():
        for start in range(0, len(train_goals), 2000):
            chunk = train_goals[start:start + 2000]
            embs = model(chunk, device)
            train_embs.append(embs)
    train_embs = F.normalize(torch.cat(train_embs, dim=0), dim=-1)

    # Build index: goal_idx → lemmas
    train_goal_to_lemmas = {g: goal_to_lemmas[g] for g in train_goals}

    # Compute MRR
    print(f"  Computing MRR on {len(val_goals)} validation goals...")
    reciprocal_ranks = []

    model.eval()
    with torch.no_grad():
        for val_goal in val_goals:
            correct_lemmas = set(goal_to_lemmas.get(val_goal, []))
            if not correct_lemmas:
                continue

            # Encode validation goal
            goal_emb = model.encode_single(val_goal, device)
            goal_emb = F.normalize(goal_emb, dim=-1)

            # Similarity to all training goals
            sims = (goal_emb @ train_embs.T)  # [N]

            # Top-K
            topk_sims, topk_indices = torch.topk(sims, min(top_k, len(train_goals)))

            # Aggregate lemma scores
            lemma_scores: dict[str, float] = defaultdict(float)
            for i in range(len(topk_indices)):
                idx = topk_indices[i].item()
                sim = max(0.0, topk_sims[i].item())
                goal = train_goals[idx]
                for lemma in train_goal_to_lemmas.get(goal, []):
                    lemma_scores[lemma] += sim

            # Rank correct lemmas
            sorted_lemmas = sorted(lemma_scores.items(), key=lambda x: -x[1])
            best_rank = len(sorted_lemmas) + 1
            for rank, (lem, _) in enumerate(sorted_lemmas, 1):
                if lem in correct_lemmas:
                    best_rank = min(best_rank, rank)
                    break

            reciprocal_ranks.append(1.0 / best_rank)

    mrr = sum(reciprocal_ranks) / max(1, len(reciprocal_ranks))
    passed = mrr > 0.3
    status = "PASS" if passed else "FAIL"

    result = {
        "gate": "G2_MRR",
        "passed": passed,
        "mrr": round(mrr, 4),
        "threshold": 0.3,
        "num_val_goals": len(val_goals),
        "num_valid_ranks": len(reciprocal_ranks),
        "top_k": top_k,
    }
    print(f"  {status}: MRR={mrr:.4f} (threshold=0.3)")
    return result


# ---------------------------------------------------------------------------
# Gate C: Embedding health
# ---------------------------------------------------------------------------


def audit_embedding_health(model: GoalOnlyEncoder, device: torch.device) -> dict:
    """Check for embedding collapse and diversity."""
    print("\n--- Gate C: Embedding Health ---")

    # Generate embeddings for random goals
    import random
    test_goals = [
        "a + b = b + a",
        "∀ x : ℝ, x + 0 = x",
        "lim_{n→∞} a_n = L",
        "∫_a^b f(x) dx = F(b) - F(a)",
        "x * (y + z) = x * y + x * z",
        "sin²θ + cos²θ = 1",
        "det(A * B) = det(A) * det(B)",
        "∀ ε > 0, ∃ δ > 0, |x - c| < δ → |f(x) - L| < ε",
    ]

    model.eval()
    with torch.no_grad():
        embs = model(test_goals, device)
        embs_norm = F.normalize(embs, dim=-1)

    # Check normalization
    norms = embs_norm.norm(dim=-1)
    norms_ok = bool(torch.allclose(norms, torch.ones_like(norms), atol=1e-4))

    # Check pairwise cosine similarities
    cos_sim = embs_norm @ embs_norm.T
    # Off-diagonal
    mask = ~torch.eye(len(test_goals), dtype=torch.bool, device=device)
    off_diag_cos = cos_sim[mask]

    # All embeddings should NOT be identical (collapse check)
    mean_cos = off_diag_cos.mean().item()
    std_cos = off_diag_cos.std().item()
    max_cos = off_diag_cos.max().item()
    min_cos = off_diag_cos.min().item()

    # Collapse: all embeddings identical → mean_cos ≈ 1.0, std_cos ≈ 0
    collapsed = mean_cos > 0.99 and std_cos < 0.01
    diverse = not collapsed and max_cos < 1.0 and min_cos < max_cos

    # Rank of embedding matrix
    try:
        U, S, V = torch.svd(embs_norm)
        effective_rank = (S > S.max().item() * 0.01).sum().item()
    except Exception:
        effective_rank = 0

    passed = not collapsed and effective_rank >= 2

    result = {
        "gate": "GC_embedding_health",
        "passed": passed,
        "norms_ok": norms_ok,
        "mean_cosine_similarity": round(mean_cos, 4),
        "std_cosine_similarity": round(std_cos, 4),
        "max_cosine": round(max_cos, 4),
        "min_cosine": round(min_cos, 4),
        "collapsed": collapsed,
        "effective_rank": effective_rank,
        "num_test_goals": len(test_goals),
    }
    status = "PASS" if passed else "FAIL"
    print(f"  {status}: collapsed={collapsed}, rank={effective_rank}, "
          f"mean_cos={mean_cos:.3f}, std_cos={std_cos:.3f}")
    return result


# ---------------------------------------------------------------------------
# Gate 3: Theorem benchmark
# ---------------------------------------------------------------------------


def audit_gate3(
    model: GoalOnlyEncoder,
    gate3_path: str,
    pairs_path: str,
    device: torch.device,
    top_k: int = 50,
    top_n: int = 30,
    use_lean: bool = True,
) -> dict:
    """Run full Gate 3 64-theorem benchmark."""
    print("\n--- Gate 3: 64-Theorem Benchmark ---")

    # Load theorems
    with open(gate3_path) as f:
        theorems = [json.loads(line) for line in f]
    print(f"  Loaded {len(theorems)} theorems")

    # Build retrieval index
    from scripts.eval.infer_goal_only import GoalRetrievalIndex, evaluate_gate3

    pairs = []
    with open(pairs_path) as f:
        for line in f:
            pairs.append(json.loads(line))

    print(f"  Building retrieval index from {len(pairs)} pairs...")
    index = GoalRetrievalIndex(model, pairs, device, chunk_size=2000)

    # Run eval
    results = evaluate_gate3(
        model, index, theorems,
        top_k=top_k, top_n=top_n,
        use_lean=use_lean,
    )

    passed = results["passed"]
    total = results["total"]
    rate = results["rate"]
    baseline = 0.156
    beats_baseline = rate >= baseline

    result = {
        "gate": "G3_theorem_benchmark",
        "passed": beats_baseline,
        "num_passed": passed,
        "num_total": total,
        "pass_rate": round(rate, 4),
        "baseline": baseline,
        "delta": round(rate - baseline, 4),
        "use_lean": use_lean,
        "top_k": top_k,
        "top_n": top_n,
        "elapsed_s": results.get("elapsed_s", 0),
        "theorem_results": results.get("results", []),
    }

    status = "PASS" if beats_baseline else "FAIL"
    print(f"  {status}: {passed}/{total} ({rate:.1%}) vs baseline {baseline:.1%}")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Goal-Only GNN Gate Audits"
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Path to GoalOnlyEncoder checkpoint")
    parser.add_argument("--pairs", type=str,
                        default="data/raw/proof_step_pairs.jsonl")
    parser.add_argument("--gate3", type=str,
                        default="data/raw/gate3_v2.jsonl")
    parser.add_argument("--output", type=str,
                        default="data/goal_only_gate_audit.json")
    parser.add_argument("--skip-gate3", action="store_true",
                        help="Skip Gate 3 (64-theorem eval, slow)")
    parser.add_argument("--skip-mrr", action="store_true")
    parser.add_argument("--no-lean", action="store_true")

    args = parser.parse_args()

    def resolve(p: str) -> Path:
        path = Path(p)
        if not path.is_absolute():
            path = _project_root / path
        return path

    device = torch.device("cpu")

    # Load model
    print(f"Loading model: {args.model}")
    model = GoalOnlyEncoder.load(str(resolve(args.model)))
    model = model.to(device)
    model.eval()
    print(f"  Params: {model.count_parameters():,}")

    results = []
    all_passed = True

    # Gate 1: Purity
    g1 = audit_purity(str(resolve(args.pairs)))
    results.append(g1)
    all_passed = all_passed and g1["passed"]

    # Gate B: MRR
    if not args.skip_mrr:
        g2 = audit_mrr(model, str(resolve(args.pairs)), device)
        results.append(g2)
        all_passed = all_passed and g2["passed"]

    # Gate C: Embedding health
    gc = audit_embedding_health(model, device)
    results.append(gc)
    all_passed = all_passed and gc["passed"]

    # Gate 3: Theorem benchmark
    if not args.skip_gate3:
        g3 = audit_gate3(
            model,
            str(resolve(args.gate3)),
            str(resolve(args.pairs)),
            device,
            use_lean=not args.no_lean,
        )
        results.append(g3)
        all_passed = all_passed and g3["passed"]

    # Summary
    summary = {
        "all_passed": all_passed,
        "timestamp": time.time(),
        "model": args.model,
        "gates": results,
    }

    output_path = resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'='*50}")
    print(f"AUDIT SUMMARY: {'ALL PASSED' if all_passed else 'FAILURES FOUND'}")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  {r['gate']}: {status}")
    print(f"Results saved to {output_path}")
    print(f"{'='*50}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
