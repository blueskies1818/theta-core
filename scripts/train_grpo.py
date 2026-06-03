#!/usr/bin/env python3
"""Run GRPO self-play training for theorem proving.

This is the AlphaGo Zero analog:
1. Model generates K proofs per theorem
2. Proof checker verifies them
3. Group-relative advantages compute reward signal
4. Policy is updated via GRPO

Usage:
    python scripts/train_grpo.py --sft-checkpoint checkpoints/sft/final --output-dir checkpoints/grpo
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.data.mathlib_extractor import load_theorems_jsonl
from src.data.dataset import TheoremProofDataset
from src.model.loader import load_model_for_grpo, apply_lora
from src.proof_checker.batch_checker import BatchChecker
from src.training.grpo_trainer import GRPOTrainer
from src.reward.config import load_reward_config
from src.utils.config import load_grpo_config, load_model_config
from src.utils.logging import MetricsLogger
from src.utils.xpu_utils import get_device, print_device_info


def main():
    parser = argparse.ArgumentParser(description="Run GRPO self-play training")
    parser.add_argument("--sft-checkpoint", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="checkpoints/grpo")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--max-theorems", type=int, default=1000)
    parser.add_argument("--data-file", type=str, default=None,
                        help="Specific JSONL file to load (overrides data-dir/mathlib4_theorems.jsonl)")
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--use-wandb", action="store_true")
    args = parser.parse_args()

    print_device_info()
    device = get_device()

    # Load configs
    grpo_config = load_grpo_config()
    model_config = load_model_config()
    reward_config = load_reward_config()  # from YAML via src/reward/config.py

    if args.use_lora:
        model_config.lora.use_lora = True

    # Load data
    data_dir = Path(args.data_dir)
    if args.data_file:
        raw_path = Path(args.data_file)
        if not raw_path.is_absolute():
            raw_path = data_dir / "raw" / args.data_file
    else:
        raw_path = data_dir / "raw" / "mathlib4_theorems.jsonl"

    if not raw_path.exists():
        print(f"No training data found at {raw_path}")
        print("Run scripts/prepare_data.py first")
        sys.exit(1)

    theorems = load_theorems_jsonl(raw_path)[: args.max_theorems]
    print(f"Loaded {len(theorems)} theorems for GRPO training")

    # Split into train/val
    split = int(len(theorems) * 0.9)
    train_theorems = theorems[:split]
    val_theorems = theorems[split:]

    # Load model: use SFT checkpoint if provided, else base model
    if args.sft_checkpoint:
        print(f"Loading SFT checkpoint from {args.sft_checkpoint}")
        # Load tokenizer from base model (checkpoint tokenizer may have
        # version-incompatible extra_special_tokens format)
        base_model = model_config.base_model.name
        print(f"Loading tokenizer from {base_model}")
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"  # Required for batched generation

        import torch as _torch
        torch_dtype = getattr(_torch, model_config.precision.mixed_precision)

        # Load base model, then apply LoRA adapters from SFT checkpoint
        print(f"Loading base model {base_model}")
        base_policy = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        policy_model = PeftModel.from_pretrained(
            base_policy, args.sft_checkpoint
        )
        policy_model = policy_model.to(device)
        # PEFT loads adapters in inference mode — explicitly enable training
        for n, p in policy_model.named_parameters():
            if "lora" in n:
                p.requires_grad = True
        policy_model.train()

        # Reference model: loaded on CPU to avoid 2-model XPU loading issues.
        # KL computation is slower but stable.
        print(f"Loading reference model on CPU")
        base_ref = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        ref_model = PeftModel.from_pretrained(base_ref, args.sft_checkpoint)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False
    else:
        policy_model, ref_model, tokenizer = load_model_for_grpo(model_config, device)

    if model_config.lora.use_lora and not args.sft_checkpoint:
        # Apply LoRA only when starting from base model (not SFT checkpoint,
        # which already has LoRA adapters applied)
        policy_model = apply_lora(policy_model, model_config)

    # Count parameters
    trainable = sum(p.numel() for p in policy_model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,}")

    # Setup proof checker
    checker = BatchChecker(
        timeout=grpo_config.proof_checker.timeout_seconds,
        max_workers=grpo_config.proof_checker.max_workers,
        cache_size=grpo_config.proof_checker.cache_size,
    )

    # Setup logging
    logger = MetricsLogger(
        log_dir=Path(args.log_dir),
        use_wandb=args.use_wandb,
    )

    # Create trainer
    trainer = GRPOTrainer(
        policy_model=policy_model,
        reference_model=ref_model,
        tokenizer=tokenizer,
        proof_checker=checker,
        config=grpo_config,
        reward_config=reward_config,
        logger=logger,
        device=device,
    )

    # Run training
    metrics = trainer.train(
        train_dataset=train_theorems,
        val_dataset=val_theorems,
        output_dir=args.output_dir,
    )

    logger.close()
    print(f"GRPO training complete. {len(metrics['metrics'])} steps logged.")


if __name__ == "__main__":
    main()
