#!/usr/bin/env python3
"""Evaluate hard-negative contrastive encoder on gate3_v2 benchmark.

Measures:
  1. MRR (Mean Reciprocal Rank) on lemma retrieval
  2. Top-K accuracy (top-1/5/10/50)
  3. End-to-end proof success (verified by Lean proof checker)

Compares against the 15.6% baseline from GNN+best-first hybrid architecture.
Target: beat 15.6% proof success rate.

Usage:
    python scripts/eval_hard_negative.py \\
        --model checkpoints/contrastive/hard_negative_encoder.pt \\
        --gate3 data/raw/gate3_v2.jsonl \\
        --output data/hard_neg_result.json
"""

import argparse, json, sys, time, re, subprocess, tempfile, os
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from src.contrastive.encoder import (
    ContrastiveDualEncoder,
    CharTokenizer,
)


# ---------------------------------------------------------------------------
# Goal and lemma extraction (adapted from eval_contrastive_retrieval.py)
# ---------------------------------------------------------------------------


def extract_goal_from_statement(statement: str) -> str:
    """Extract the goal proposition from a full theorem statement."""
    s = statement.strip()
    if ":=" in s:
        s = s.split(":=")[0].strip()

    for kw in ["theorem ", "lemma ", "def ", "example "]:
        if s.startswith(kw):
            s = s[len(kw):]
            break

    depth = 0
    for i, c in enumerate(s):
        if c in "({[":
            depth += 1
        elif c in ")}]":
            depth -= 1
        elif c == ":" and depth == 0:
            return s[i + 1:].strip()

    return s


def extract_lemmas_from_proof(proof_text: str) -> list[str]:
    """Extract REAL lemma names from a Lean proof script."""
    HYP_PATTERN = re.compile(r'^h[a-zA-Z0-9_]*$')
    NOT_LEMMA = {'←', '←', '▸', '⟨', '⟩', 'this', 'ih', 'rfl', 'rfl',
                 'Eq.refl', 'by'}

    lemmas = []

    for pattern in [r'simp\s*\[([^\]]+)\]', r'rw\s*\[([^\]]+)\]',
                    r'simp_rw\s*\[([^\]]+)\]', r'rw_search\s*\[([^\]]+)\]']:
        for match in re.finditer(pattern, proof_text):
            items = match.group(1).split(",")
            for item in items:
                name = item.strip()
                name = re.sub(r'^←\s*', '', name)
                name = name.strip()
                if (name and not name.startswith("*") and not name.startswith("-")
                    and name not in NOT_LEMMA
                    and not HYP_PATTERN.match(name)):
                    lemmas.append(name)

    for match in re.finditer(r'exact\s+(\S+)', proof_text):
        name = match.group(1).rstrip(".,;()[]{}\"'")
        if name and not name.startswith("h") and name not in NOT_LEMMA:
            if not HYP_PATTERN.match(name):
                lemmas.append(name)

    for match in re.finditer(r'apply\s+(\S+)', proof_text):
        name = match.group(1).rstrip(".,;()[]{}\"'")
        if name and not name.startswith("h") and name not in NOT_LEMMA:
            if not HYP_PATTERN.match(name):
                lemmas.append(name)

    for match in re.finditer(r':=\s*(\S+)', proof_text):
        name = match.group(1).rstrip(".,;()[]{}\"'")
        if (name and len(name) > 1 and name not in NOT_LEMMA
            and not HYP_PATTERN.match(name)):
            lemmas.append(name)

    if not lemmas:
        for tactic in ["ring", "linarith", "nlinarith", "field_simp",
                       "positivity", "norm_num", "simp"]:
            if tactic in proof_text.lower():
                lemmas.append(tactic)
                break

    return lemmas


# ---------------------------------------------------------------------------
# MRR computation
# ---------------------------------------------------------------------------


def compute_mrr(
    model: ContrastiveDualEncoder,
    tokenizer: CharTokenizer,
    theorems: list[dict],
    lemma_names: list[str],
    precomputed_lemma_embs: torch.Tensor,
    device: torch.device,
) -> dict:
    """Compute MRR and top-K accuracy."""
    lemma_to_idx = {name: i for i, name in enumerate(lemma_names)}

    results = []
    reciprocal_ranks = []
    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    top10_correct = 0
    top50_correct = 0
    total_evaluated = 0

    for theorem in theorems:
        name = theorem.get("name", "unknown")
        statement = theorem["statement"]
        proof = theorem.get("proof", "")

        goal = extract_goal_from_statement(statement)
        correct_lemmas = extract_lemmas_from_proof(proof)

        if not correct_lemmas:
            continue

        correct_indices = []
        for lemma in correct_lemmas:
            if lemma in lemma_to_idx:
                correct_indices.append(lemma_to_idx[lemma])
            elif "." in lemma:
                short = lemma.split(".")[-1]
                if short in lemma_to_idx:
                    correct_indices.append(lemma_to_idx[short])

        if not correct_indices:
            continue

        goal_text = tokenizer.preprocess_goal(goal)
        goal_ids = tokenizer.encode(goal_text).unsqueeze(0).to(device)
        goal_emb = model.encode_goal(goal_ids)

        scores = (goal_emb @ precomputed_lemma_embs.T).squeeze(0)
        sorted_indices = torch.argsort(scores, descending=True).cpu().tolist()

        best_rank = min(
            sorted_indices.index(idx) + 1 for idx in correct_indices
        )
        rr = 1.0 / best_rank
        reciprocal_ranks.append(rr)
        total_evaluated += 1

        if best_rank <= 1:
            top1_correct += 1
        if best_rank <= 3:
            top3_correct += 1
        if best_rank <= 5:
            top5_correct += 1
        if best_rank <= 10:
            top10_correct += 1
        if best_rank <= 50:
            top50_correct += 1

        top5_names = [lemma_names[idx] for idx in sorted_indices[:5]]
        top5_scores = [float(scores[idx].item()) for idx in sorted_indices[:5]]

        results.append({
            "name": name,
            "goal": goal[:200],
            "correct_lemmas": correct_lemmas,
            "best_rank": best_rank,
            "reciprocal_rank": rr,
            "top5_lemma_names": top5_names,
            "top5_scores": top5_scores,
        })

    mrr = sum(reciprocal_ranks) / max(1, len(reciprocal_ranks))

    return {
        "mrr": mrr,
        "top1_accuracy": top1_correct / max(1, total_evaluated),
        "top3_accuracy": top3_correct / max(1, total_evaluated),
        "top5_accuracy": top5_correct / max(1, total_evaluated),
        "top10_accuracy": top10_correct / max(1, total_evaluated),
        "top50_accuracy": top50_correct / max(1, total_evaluated),
        "num_theorems": len(theorems),
        "num_evaluated": total_evaluated,
        "num_candidates": len(lemma_names),
        "per_theorem": results,
    }


# ---------------------------------------------------------------------------
# End-to-end proof success (via Lean proof checker)
# ---------------------------------------------------------------------------


def compute_proof_success(
    model: ContrastiveDualEncoder,
    tokenizer: CharTokenizer,
    theorems: list[dict],
    lemma_names: list[str],
    precomputed_lemma_embs: torch.Tensor,
    device: torch.device,
    top_k: int = 10,
    project_dir: str | None = None,
) -> dict:
    """Compute end-to-end proof success rate.

    For each theorem, retrieves top-K lemmas and checks if any proves
    the goal via Lean proof checker.
    """
    lemma_to_idx = {name: i for i, name in enumerate(lemma_names)}

    results = []
    successes = 0
    total = 0

    for theorem in theorems:
        name = theorem.get("name", "unknown")
        statement = theorem["statement"]

        goal = extract_goal_from_statement(statement)
        goal_text = tokenizer.preprocess_goal(goal)
        goal_ids = tokenizer.encode(goal_text).unsqueeze(0).to(device)
        goal_emb = model.encode_goal(goal_ids)

        scores = (goal_emb @ precomputed_lemma_embs.T).squeeze(0)
        sorted_indices = torch.argsort(scores, descending=True).cpu().tolist()
        top_lemmas = [lemma_names[idx] for idx in sorted_indices[:top_k]]

        proved = False
        best_lemma = None

        for lemma in top_lemmas:
            # Build Lean proof script
            proof_script = f"{statement}\n  exact {lemma}"

            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".lean", delete=False
                ) as f:
                    f.write(f"import Mathlib\n\n{proof_script}\n")
                    temp_path = f.name

                cwd = project_dir or "proof_checker_env"
                result = subprocess.run(
                    ["lake", "env", "lean", temp_path],
                    capture_output=True, text=True, timeout=15,
                    cwd=_project_root / cwd if Path(_project_root / cwd).exists() else None,
                )

                if result.returncode == 0:
                    proved = True
                    best_lemma = lemma
                    os.unlink(temp_path)
                    break

                os.unlink(temp_path)
            except Exception:
                pass

        if proved:
            successes += 1
        total += 1

        result_entry = {
            "name": name,
            "goal": goal[:200],
            "proved": proved,
            "best_lemma": best_lemma if proved else None,
            "top_lemmas_tried": top_lemmas[:top_k],
        }
        results.append(result_entry)

        status = "PROVED" if proved else "FAILED"
        print(f"  {name}: {status}" + (f" (via {best_lemma})" if proved else ""))

    return {
        "proof_success_rate": successes / max(1, total),
        "successes": successes,
        "total": total,
        "top_k": top_k,
        "per_theorem": results,
    }


# ---------------------------------------------------------------------------
# Domain-filtered: rebuild using domain info from gate3
# ---------------------------------------------------------------------------


def load_lemma_candidates_from_pairs(pairs_path: Path) -> list[str]:
    """Load unique lemma names from proof-step pairs."""
    names = set()
    with open(pairs_path) as f:
        for line in f:
            d = json.loads(line)
            names.add(d["lemma"])
    return sorted(names)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate hard-negative contrastive encoder on gate3_v2"
    )
    parser.add_argument("--model", default="checkpoints/contrastive/hard_negative_encoder.pt")
    parser.add_argument("--gate3", default="data/raw/gate3_v2.jsonl")
    parser.add_argument("--pairs", default="data/raw/proof_step_pairs.jsonl")
    parser.add_argument("--output", default="data/hard_neg_result.json")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Top-K for proof success check")
    parser.add_argument("--skip-proof-check", action="store_true",
                        help="Skip Lean proof checker (MRR only)")
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--project-dir", default=None,
                        help="Lake project directory")
    args = parser.parse_args()

    torch.set_num_threads(args.num_threads)
    device = torch.device(args.device)
    print(f"Device: {device} (threads: {args.num_threads})")

    # ---- Load model ---------------------------------------------------------
    model_path = _project_root / args.model
    print(f"Loading model from {model_path}...")
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        sys.exit(1)

    model = ContrastiveDualEncoder.load(model_path).to(device)
    model.eval()
    tokenizer = CharTokenizer(max_len=model.config.max_seq_len)
    print(f"  Params: {model.num_params:,} "
          f"(goal: {model.goal_encoder_params:,}, "
          f"lemma: {model.lemma_encoder_params:,})")

    # ---- Load lemma candidates ----------------------------------------------
    pairs_path = _project_root / args.pairs
    print(f"Loading lemma candidates from {pairs_path}...")
    lemma_names = load_lemma_candidates_from_pairs(pairs_path)
    print(f"  {len(lemma_names)} unique lemma candidates")

    # ---- Pre-encode all lemmas ---------------------------------------------
    print("Encoding all lemma candidates...", end=" ", flush=True)
    t0 = time.time()
    lemma_embs_list = []
    batch_size = 512
    for i in range(0, len(lemma_names), batch_size):
        batch_names = lemma_names[i:i + batch_size]
        batch_text = [tokenizer.preprocess_lemma(n) for n in batch_names]
        batch_ids = tokenizer.encode_batch(batch_text).to(device)
        with torch.no_grad():
            batch_embs = model.encode_lemma(batch_ids)
        lemma_embs_list.append(batch_embs)
    lemma_embs = torch.cat(lemma_embs_list, dim=0)
    print(f"done ({time.time() - t0:.1f}s). Shape: {list(lemma_embs.shape)}")

    # ---- Load gate3 theorems ------------------------------------------------
    gate3_path = _project_root / args.gate3
    print(f"\nLoading gate3_v2 theorems from {gate3_path}...")
    theorems = []
    with open(gate3_path) as f:
        for line in f:
            theorems.append(json.loads(line))
    print(f"  {len(theorems)} theorems loaded")

    # ---- MRR Evaluation ----------------------------------------------------
    print(f"\n{'='*60}")
    print(f"MRR Evaluation on gate3_v2 ({len(theorems)} theorems)")
    print(f"{'='*60}")
    mrr_results = compute_mrr(
        model, tokenizer, theorems,
        lemma_names, lemma_embs, device,
    )

    print(f"\nMRR Results:")
    print(f"  MRR:               {mrr_results['mrr']:.4f}")
    print(f"  Top-1 accuracy:    {mrr_results['top1_accuracy']:.4f}")
    print(f"  Top-3 accuracy:    {mrr_results['top3_accuracy']:.4f}")
    print(f"  Top-5 accuracy:    {mrr_results['top5_accuracy']:.4f}")
    print(f"  Top-10 accuracy:   {mrr_results['top10_accuracy']:.4f}")
    print(f"  Top-50 accuracy:   {mrr_results['top50_accuracy']:.4f}")
    print(f"  Theorems evaluated: {mrr_results['num_evaluated']}/{mrr_results['num_theorems']}")
    print(f"  Candidate lemmas:   {mrr_results['num_candidates']}")

    # ---- Proof Success -----------------------------------------------------
    proof_results = None
    if not args.skip_proof_check:
        print(f"\n{'='*60}")
        print(f"End-to-End Proof Success (top-{args.top_k})")
        print(f"{'='*60}")
        proof_results = compute_proof_success(
            model, tokenizer, theorems,
            lemma_names, lemma_embs, device,
            top_k=args.top_k,
            project_dir=args.project_dir,
        )
        print(f"\nProof Success Results:")
        print(f"  Success rate: {proof_results['proof_success_rate']:.1%} "
              f"({proof_results['successes']}/{proof_results['total']})")
    else:
        proof_results = {"proof_success_rate": None, "note": "skipped"}

    # ---- Save results ------------------------------------------------------
    output_path = _project_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    final_results = {
        "model": str(model_path),
        "model_params": model.num_params,
        "mrr": mrr_results["mrr"],
        "top1_accuracy": mrr_results["top1_accuracy"],
        "top5_accuracy": mrr_results["top5_accuracy"],
        "top10_accuracy": mrr_results["top10_accuracy"],
        "num_candidates": mrr_results["num_candidates"],
        "num_evaluated": mrr_results["num_evaluated"],
        "proof_success_rate": proof_results.get("proof_success_rate"),
        "proof_successes": proof_results.get("successes", 0),
        "proof_total": proof_results.get("total", 0),
        "per_theorem_mrr": mrr_results["per_theorem"],
        "per_theorem_proof": proof_results.get("per_theorem", []),
        "baseline": 0.156,
        "target": "beat 15.6%",
    }

    with open(output_path, "w") as f:
        json.dump(final_results, f, indent=2)

    # ---- Verdict -----------------------------------------------------------
    success_rate = proof_results.get("proof_success_rate")
    if success_rate is not None:
        baseline = 0.156
        if success_rate > baseline:
            print(f"\n✓ SUCCESS: Hard-negative encoder beats baseline")
            print(f"  {success_rate:.1%} > {baseline:.1%} (delta: +{success_rate - baseline:.1%})")
        elif success_rate == baseline:
            print(f"\n= TIED: Hard-negative encoder matches baseline ({success_rate:.1%})")
        else:
            print(f"\n✗ BELOW: Hard-negative encoder below baseline")
            print(f"  {success_rate:.1%} < {baseline:.1%} (delta: {success_rate - baseline:.1%})")
    else:
        print(f"\nProof success check skipped. MRR: {mrr_results['mrr']:.4f}")

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
