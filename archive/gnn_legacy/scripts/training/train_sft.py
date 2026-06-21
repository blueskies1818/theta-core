#!/usr/bin/env python3
"""Run supervised fine-tuning on Mathlib4 theorem-proof pairs.

Usage:
    python scripts/training/train_sft.py --data-dir data --output-dir checkpoints/sft
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoTokenizer

from src.data.mathlib_extractor import load_theorems_jsonl
from src.data.dataset import create_datasets
from src.model.loader import load_model_for_sft, apply_lora
from src.training.sft_trainer import SFTTrainer
from src.utils.config import load_model_config, load_sft_config
from src.utils.xpu_utils import get_device, print_device_info


def main():
    parser = argparse.ArgumentParser(description="Run SFT on Mathlib4 theorems")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="checkpoints/sft")
    parser.add_argument("--max-theorems", type=int, default=None)
    parser.add_argument("--use-lora", action="store_true")
    args = parser.parse_args()

    print_device_info()
    device = get_device()

    # Load configs
    sft_config = load_sft_config()
    model_config = load_model_config()

    if args.use_lora:
        model_config.lora.use_lora = True

    # Load data
    data_dir = Path(args.data_dir)
    raw_path = data_dir / "raw" / "mathlib4_theorems.jsonl"

    if not raw_path.exists():
        print(f"No training data found at {raw_path}")
        print("Run scripts/build/prepare_data.py first")
        sys.exit(1)

    print(f"Loading theorems from {raw_path}")
    theorems = load_theorems_jsonl(raw_path)
    if args.max_theorems:
        theorems = theorems[: args.max_theorems]
    print(f"Loaded {len(theorems)} theorems")

    # Load model
    model, tokenizer = load_model_for_sft(model_config, device)

    if model_config.lora.use_lora:
        model = apply_lora(model, model_config)

    # Count parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.1f}%)")

    # Create datasets
    train_ds, val_ds = create_datasets(
        theorems,
        tokenizer,
        train_split=sft_config.data.train_split,
        max_seq_length=sft_config.training.max_seq_length,
    )
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    # Train
    trainer = SFTTrainer(model, tokenizer, sft_config, device)
    metrics = trainer.train(train_ds, val_ds, output_dir=args.output_dir)

    print(f"Training complete. Best val loss: {metrics['best_val_loss']:.4f}")


if __name__ == "__main__":
    main()
