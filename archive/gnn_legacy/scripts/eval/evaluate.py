#!/usr/bin/env python3
"""Evaluate a trained model on held-out theorems.

Measures proof success rate on theorems not seen during training.

Usage:
    python scripts/eval/evaluate.py --checkpoint checkpoints/grpo/checkpoint-1000 --data-dir data
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.mathlib_extractor import load_theorems_jsonl
from src.model.generation import generate_proofs
from src.proof_checker.batch_checker import BatchChecker
from src.proof_checker.formats import wrap_theorem_with_proof
from src.utils.config import load_model_config, GenerationConfig
from src.utils.xpu_utils import get_device


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on theorem proving")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--num-theorems", type=int, default=100)
    parser.add_argument("--num-proofs-per-theorem", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    device = get_device()
    model_config = load_model_config()
    torch_dtype = getattr(torch, model_config.precision.mixed_precision)

    print(f"Loading model from {args.checkpoint}")
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load evaluation theorems (from tail of dataset = held out)
    data_dir = Path(args.data_dir)
    raw_path = data_dir / "raw" / "mathlib4_theorems.jsonl"

    if not raw_path.exists():
        print(f"No data found at {raw_path}")
        sys.exit(1)

    all_theorems = load_theorems_jsonl(raw_path)
    eval_theorems = all_theorems[-args.num_theorems :]
    print(f"Evaluating on {len(eval_theorems)} held-out theorems")

    # Setup proof checker
    checker = BatchChecker(max_workers=8)

    gen_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=True,
    )

    total_checks = 0
    total_success = 0
    theorem_success = 0  # Theorems with at least one valid proof

    for theorem in eval_theorems:
        statement = theorem["statement"]
        prompt = f"Theorem: {statement}\nProof:"

        proofs = generate_proofs(
            model, tokenizer, [prompt], gen_config, args.num_proofs_per_theorem
        )[0]

        codes = []
        for proof in proofs:
            codes.append(wrap_theorem_with_proof(statement, proof))

        results = checker.check_batch(codes)

        any_success = False
        for r in results:
            total_checks += 1
            if r.success:
                total_success += 1
                any_success = True

        if any_success:
            theorem_success += 1

        print(
            f"  {statement[:60]:<60} | "
            f"Success: {sum(1 for r in results if r.success)}/{len(results)}"
        )

    print(f"\nEvaluation Results:")
    print(f"  Proof-level: {total_success}/{total_checks} ({total_success/total_checks:.1%})")
    print(f"  Theorem-level (pass@K): {theorem_success}/{len(eval_theorems)} ({theorem_success/len(eval_theorems):.1%})")


if __name__ == "__main__":
    main()
