"""theta-core — Autonomous Mathematical Physics AI

Phase 1: Validate the self-play loop.
Small-scale system that proves the model can learn theorem proving from
proof-checker feedback alone, without human-labeled proofs.

What Phase 1 includes:
- Transformer proof generator (Qwen2.5-1.5B placeholder for future GNN+MCTS)
- Lean 4 proof checker interface with parallel batch verification
- SFT pretraining on Mathlib4 theorem-proof pairs
- GRPO self-play training with group-relative advantages
- Binary reward + optional length bonus

What Phase 1 does NOT yet include (planned for Phase 2–5):
- GNN + MCTS architecture for the Mathematical Explorer
- Physical prediction scorer (Component 2)
- Translation layer for human physicists (Component 3)
- Curiosity/exploration reward
- Formal frontier map with zone-weighted rewards
- Layer 1 / Layer 2 data architecture for physical measurements
- Domain-level holdout strategy
- Correspondence checks against GR/QFT limits
- Solution enumeration and anomaly flagging (Dirac mechanism)

See mathematical_ai_system.md for the full system design.
See model_structure_and_data.md for the detailed technical specification.
See IMPROVEMENT_IDEAS.md for the running list of planned improvements.
"""
