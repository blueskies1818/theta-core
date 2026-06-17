#!/usr/bin/env python3
"""Evaluate contrastive dual-encoder on gate3 lemma-novelty retrieval.

Measures:
  1. MRR (Mean Reciprocal Rank): how high does the correct lemma rank?
  2. Top-K accuracy: is the correct lemma in top-1/5/10/20/50?
  3. End-to-end proof success: can the top-ranked lemma prove the theorem?

Compares against baseline cosine-similarity retrieval (GNN-based).
Target: MRR > 0.786 AND proof success > 28.6%.

Usage:
    python scripts/eval_contrastive_retrieval.py \
        --model checkpoints/contrastive/lemma_encoder.pt \
        --gate3 data/raw/gate3_lemma_novelty.jsonl
"""

import argparse, json, sys, time, re
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
# Goal extraction (adapted from eval_gnn_prover.py)
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


# ---------------------------------------------------------------------------
# Lemma extraction from proofs
# ---------------------------------------------------------------------------


def extract_lemmas_from_proof(proof_text: str) -> list[str]:
    """Extract REAL lemma names from a Lean proof script.

    Filters out hypotheses (h, h1, ha, hP, etc.) which are local variables,
    not actual lemmas from the library.

    Handles:
      - simp [lemma1, lemma2]  → ['lemma1', 'lemma2']
      - rw [lemma1, ← lemma2]  → ['lemma1', 'lemma2']
      - exact lemma arg1       → ['lemma']
      - apply lemma            → ['lemma']
      - have ... := lemma ...  → ['lemma']
    """
    # Patterns that indicate hypotheses, not real lemmas
    HYP_PATTERN = re.compile(r'^h[a-zA-Z0-9_]*$')  # h, h1, ha, hP, h_ac, etc.
    # Patterns that are definitely NOT lemma names
    NOT_LEMMA = {'←', '←', '▸', '⟨', '⟩', 'this', 'ih', 'rfl', 'rfl', 
                 'Eq.refl', 'by'}
    
    lemmas = []

    # Match simp [...] / rw [...] / simp_rw [...] / calc [...] 
    for pattern in [r'simp\s*\[([^\]]+)\]', r'rw\s*\[([^\]]+)\]',
                    r'simp_rw\s*\[([^\]]+)\]', r'rw_search\s*\[([^\]]+)\]']:
        for match in re.finditer(pattern, proof_text):
            items = match.group(1).split(",")
            for item in items:
                name = item.strip()
                # Remove leading ← (rewrite direction indicator)
                name = re.sub(r'^←\s*', '', name)
                name = name.strip()
                if (name and not name.startswith("*") and not name.startswith("-")
                    and name not in NOT_LEMMA
                    and not HYP_PATTERN.match(name)):
                    lemmas.append(name)

    # Match exact <lemma> (but not hypotheses)
    for match in re.finditer(r'exact\s+(\S+)', proof_text):
        name = match.group(1).rstrip(".,;()[]{}\"'")
        # Skip hypotheses and known non-lemmas
        if name and not name.startswith("h") and name not in NOT_LEMMA:
            if not HYP_PATTERN.match(name):
                lemmas.append(name)

    # Match apply <lemma> (but not hypotheses)
    for match in re.finditer(r'apply\s+(\S+)', proof_text):
        name = match.group(1).rstrip(".,;()[]{}\"'")
        if name and not name.startswith("h") and name not in NOT_LEMMA:
            if not HYP_PATTERN.match(name):
                lemmas.append(name)

    # Match have ... := <lemma> or ... := <lemma>
    for match in re.finditer(r':=\s*(\S+)', proof_text):
        name = match.group(1).rstrip(".,;()[]{}\"'")
        if (name and len(name) > 1 and name not in NOT_LEMMA
            and not HYP_PATTERN.match(name)):
            lemmas.append(name)

    # If no lemmas extracted, the tactic itself may be relevant
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
    gate3_path: Path,
    lemma_names: list[str],
    precomputed_lemma_embs: torch.Tensor,
    device: torch.device,
) -> dict:
    """Compute MRR and top-K accuracy on gate3 lemma-novelty set.

    Returns:
        Dict with MRR, top-K accuracies, and per-theorem details.
    """
    # Load gate3 theorems
    theorems = []
    with open(gate3_path) as f:
        for line in f:
            theorems.append(json.loads(line))

    # Build lemma name → index mapping
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

        # Extract goal
        goal = extract_goal_from_statement(statement)

        # Extract correct lemmas from proof
        correct_lemmas = extract_lemmas_from_proof(proof)

        if not correct_lemmas:
            print(f"  WARNING: No lemmas extracted from proof of {name}: {proof[:80]}")
            continue

        # Find which correct lemmas are in our candidate set
        correct_indices = []
        for lemma in correct_lemmas:
            if lemma in lemma_to_idx:
                correct_indices.append(lemma_to_idx[lemma])
            # Also try with different separators
            elif "." in lemma:
                short = lemma.split(".")[-1]
                if short in lemma_to_idx:
                    correct_indices.append(lemma_to_idx[short])

        if not correct_indices:
            print(f"  WARNING: None of {correct_lemmas} in candidate set for {name}")
            continue

        # Encode goal
        goal_text = tokenizer.preprocess_goal(goal)
        goal_ids = tokenizer.encode(goal_text).unsqueeze(0).to(device)  # [1, L]
        goal_emb = model.encode_goal(goal_ids)  # [1, D]

        # Score all lemmas: [1, D] @ [D, N] = [1, N]
        scores = (goal_emb @ precomputed_lemma_embs.T).squeeze(0)  # [N]

        # Rank by score (descending)
        sorted_indices = torch.argsort(scores, descending=True).cpu().tolist()

        # Compute best rank among correct lemmas
        best_rank = min(
            sorted_indices.index(idx) + 1 for idx in correct_indices
        )
        rr = 1.0 / best_rank
        reciprocal_ranks.append(rr)
        total_evaluated += 1

        # Top-K accuracy
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

        # Get top-5 lemma names for reporting
        top5_names = [lemma_names[idx] for idx in sorted_indices[:5]]
        top5_scores = [scores[idx].item() for idx in sorted_indices[:5]]

        results.append({
            "name": name,
            "goal": goal[:200],
            "proof": proof[:200],
            "correct_lemmas": correct_lemmas,
            "correct_indices": correct_indices,
            "best_rank": best_rank,
            "reciprocal_rank": rr,
            "top5_lemma_names": top5_names,
            "top5_scores": top5_scores,
        })

        print(f"  {name}: rank={best_rank}/{len(lemma_names)}, "
              f"RR={rr:.4f}, top-5={top5_names}")

    mrr = sum(reciprocal_ranks) / max(1, len(reciprocal_ranks))

    summary = {
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

    return summary


# ---------------------------------------------------------------------------
# End-to-end proof success (requires Lean proof checker)
# ---------------------------------------------------------------------------


def compute_proof_success(
    model: ContrastiveDualEncoder,
    tokenizer: CharTokenizer,
    gate3_path: Path,
    lemma_names: list[str],
    precomputed_lemma_embs: torch.Tensor,
    device: torch.device,
    top_k: int = 10,
) -> dict:
    """Compute end-to-end proof success rate.

    For each gate3 theorem:
    1. Rank all lemmas
    2. Try top-K lemmas as single-step proofs
    3. Check if any proves the theorem

    Returns:
        Dict with success rate and per-theorem details.
    """
    import subprocess, tempfile, os

    # Load gate3 theorems
    theorems = []
    with open(gate3_path) as f:
        for line in f:
            theorems.append(json.loads(line))

    lemma_to_idx = {name: i for i, name in enumerate(lemma_names)}

    results = []
    successes = 0
    total = 0

    for theorem in theorems:
        name = theorem.get("name", "unknown")
        statement = theorem["statement"]
        proof = theorem.get("proof", "")

        # Extract goal
        goal = extract_goal_from_statement(statement)

        # Encode goal
        goal_text = tokenizer.preprocess_goal(goal)
        goal_ids = tokenizer.encode(goal_text).unsqueeze(0).to(device)
        goal_emb = model.encode_goal(goal_ids)

        # Score and rank
        scores = (goal_emb @ precomputed_lemma_embs.T).squeeze(0)
        sorted_indices = torch.argsort(scores, descending=True).cpu().tolist()
        top_lemmas = [lemma_names[idx] for idx in sorted_indices[:top_k]]

        # Try each top lemma as a single-step proof
        proved = False
        best_lemma = None

        for lemma in top_lemmas:
            # Build Lean proof script
            # For lemma names, try different forms
            proof_step = f"  exact {lemma}"
            proof_script = f"{statement}\n{proof_step}"

            # Write to temp file
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".lean", delete=False
                ) as f:
                    f.write(f"import Mathlib\n\n{proof_script}\n")
                    temp_path = f.name

                # Try to check with lake
                result = subprocess.run(
                    ["lake", "env", "lean", temp_path],
                    capture_output=True, text=True, timeout=30,
                    cwd=_project_root,
                )

                if result.returncode == 0:
                    proved = True
                    best_lemma = lemma
                    os.unlink(temp_path)
                    break

                os.unlink(temp_path)
            except Exception as e:
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

    success_rate = successes / max(1, total)

    return {
        "proof_success_rate": success_rate,
        "successes": successes,
        "total": total,
        "top_k": top_k,
        "per_theorem": results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate contrastive dual-encoder on gate3 retrieval"
    )
    parser.add_argument("--model", default="checkpoints/contrastive/lemma_encoder.pt")
    parser.add_argument("--gate3", default="data/raw/gate3_lemma_novelty.jsonl")
    parser.add_argument("--pairs", default="data/raw/proof_step_pairs.jsonl")
    parser.add_argument("--output", default="data/pathc_retrieval_results.json")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Top-K for proof success check")
    parser.add_argument("--skip-proof-check", action="store_true",
                        help="Skip Lean proof checker (faster, MRR only)")
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    torch.set_num_threads(args.num_threads)
    device = torch.device(args.device)
    print(f"Device: {device} (threads: {args.num_threads})")

    # ---- Load model ---------------------------------------------------------
    model_path = _project_root / args.model
    print(f"Loading model from {model_path}...")
    model = ContrastiveDualEncoder.load(model_path).to(device)
    model.eval()
    tokenizer = CharTokenizer(max_len=model.config.max_seq_len)
    print(f"  Params: {model.num_params:,} "
          f"(goal: {model.goal_encoder_params:,}, "
          f"lemma: {model.lemma_encoder_params:,})")

    # ---- Load lemma candidates ----------------------------------------------
    pairs_path = _project_root / args.pairs
    print(f"Loading lemma candidates from {pairs_path}...")
    lemma_names_set = set()
    with open(pairs_path) as f:
        for line in f:
            d = json.loads(line)
            lemma_names_set.add(d["lemma"])
    lemma_names = sorted(lemma_names_set)
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
    lemma_embs = torch.cat(lemma_embs_list, dim=0)  # [N, D]
    print(f"done ({time.time() - t0:.1f}s). Shape: {list(lemma_embs.shape)}")

    # ---- MRR Evaluation ----------------------------------------------------
    gate3_path = _project_root / args.gate3
    print(f"\n--- MRR Evaluation on {gate3_path} ---")
    mrr_results = compute_mrr(
        model, tokenizer, gate3_path,
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

    # ---- Proof Success (optional) ------------------------------------------
    proof_results = None
    if not args.skip_proof_check:
        print(f"\n--- End-to-End Proof Success (top-{args.top_k}) ---")
        proof_results = compute_proof_success(
            model, tokenizer, gate3_path,
            lemma_names, lemma_embs, device,
            top_k=args.top_k,
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
        "mrr": mrr_results["mrr"],
        "top1_accuracy": mrr_results["top1_accuracy"],
        "top5_accuracy": mrr_results["top5_accuracy"],
        "top10_accuracy": mrr_results["top10_accuracy"],
        "num_candidates": mrr_results["num_candidates"],
        "num_evaluated": mrr_results["num_evaluated"],
        "proof_success_rate": proof_results.get("proof_success_rate"),
        "per_theorem_mrr": mrr_results["per_theorem"],
        "per_theorem_proof": proof_results.get("per_theorem", []),
        "baseline_mrr_target": 0.786,
        "baseline_proof_success_target": 0.286,
    }

    with open(output_path, "w") as f:
        json.dump(final_results, f, indent=2)

    print(f"\nResults saved to {output_path}")

    # ---- Comparison to baseline --------------------------------------------
    print(f"\n--- Baseline Comparison ---")
    mrr_pass = mrr_results["mrr"] > 0.786
    proof_pass = (
        proof_results.get("proof_success_rate", 0) or 0
    ) > 0.286

    print(f"  MRR:              {mrr_results['mrr']:.4f} vs baseline 0.786 → "
          f"{'PASS' if mrr_pass else 'FAIL'}")
    if proof_results.get("proof_success_rate") is not None:
        print(f"  Proof success:    {proof_results['proof_success_rate']:.1%} "
              f"vs baseline 28.6% → {'PASS' if proof_pass else 'FAIL'}")
    else:
        print(f"  Proof success:    skipped")

    return 0 if (mrr_pass and (proof_pass or args.skip_proof_check)) else 1


if __name__ == "__main__":
    sys.exit(main())
