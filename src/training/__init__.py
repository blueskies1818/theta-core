"""Training loops.

Phase 1: SFT pretraining + GRPO self-play.
- sft_trainer.py — supervised fine-tuning on Mathlib4 theorem-proof pairs
- grpo_trainer.py — GRPO self-play with proof checker as environment
- losses.py — GRPO loss with KL penalty

See mathematical_ai_system.md § Training Methodology.
"""