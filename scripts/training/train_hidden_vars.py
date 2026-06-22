#!/usr/bin/env python3
"""Train the Hidden Variable Proposer v3.

v3 over v2: proposer emits EXPRESSION FRAGMENTS (e.g. "gamma = 1/sqrt(1 - beta^2)")
instead of just variable names. The model learns the RELATIONSHIP, not just the label.

Training data: ~149 pre-1905 physics examples, each now includes the full expression.

RUN: python scripts/training/train_hidden_vars.py
OUTPUT: checkpoints/hidden_var_proposer_v3.pt
        data/era_gate_template_results.json (via scripts/extended_era_gate.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.physics.hidden_variables import (
    HiddenVariableProposer,
    train_hidden_var_proposer,
    generate_synthetic_training_examples,
    NUM_SHAPES, NUM_VAR_TYPES, NUM_TRANSFORMS,
    NUM_HV_DOMAINS, NUM_HV_QUANTITIES, NUM_EXPR_TEMPLATES,
)

CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "hidden_var_proposer_v3.pt"


def main() -> None:
    examples = generate_synthetic_training_examples()
    print(f"Training examples: {len(examples)}")

    input_dim = NUM_SHAPES + NUM_HV_QUANTITIES + NUM_HV_DOMAINS
    output_dim = NUM_VAR_TYPES + NUM_TRANSFORMS + 1 + NUM_EXPR_TEMPLATES
    print(f"Input dim: {input_dim} ({NUM_SHAPES} shapes + {NUM_HV_QUANTITIES} qtys + {NUM_HV_DOMAINS} domains)")
    print(f"Output dim: {output_dim} ({NUM_VAR_TYPES} var types + {NUM_TRANSFORMS} transforms + 1 conf + {NUM_EXPR_TEMPLATES} expr templates)")

    model = train_hidden_var_proposer(
        epochs=400,
        lr=0.003,
        device="cpu",
        checkpoint_path=str(CHECKPOINT_PATH),
    )

    print(f"\nModel params: {model.count_parameters()}")
    print(f"Checkpoint saved to: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
