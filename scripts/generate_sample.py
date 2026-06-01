#!/usr/bin/env python3
"""Interactive proof generation — test the model on a single theorem.

Usage:
    python scripts/generate_sample.py --checkpoint checkpoints/sft/final
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.model.generation import generate_single_proof
from src.proof_checker.lean_interface import LeanProofChecker
from src.proof_checker.formats import wrap_theorem_with_proof
from src.utils.config import GenerationConfig
from src.utils.xpu_utils import get_device


def main():
    parser = argparse.ArgumentParser(description="Generate a proof for a theorem")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--theorem", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--num-samples", type=int, default=4)
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    checker = LeanProofChecker()
    gen_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=True,
    )

    if args.theorem:
        statement = args.theorem
    else:
        statement = "theorem add_comm (a b : Nat) : a + b = b + a"

    print(f"\nTheorem: {statement}")
    print("=" * 60)

    for i in range(args.num_samples):
        prompt = f"Theorem: {statement}\nProof:"
        proof = generate_single_proof(model, tokenizer, prompt, gen_config)
        full_code = wrap_theorem_with_proof(statement, proof)
        result = checker.check(full_code)

        status = "VALID" if result.success else "INVALID"
        print(f"\n--- Sample {i+1} [{status}] ({result.check_time_ms:.0f}ms) ---")
        print(proof[:300] + ("..." if len(proof) > 300 else ""))
        if result.errors:
            print(f"  Error: {result.errors[0][:120]}")


if __name__ == "__main__":
    main()
