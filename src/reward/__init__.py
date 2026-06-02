"""Proof reward computation.

Phase 1: Binary proof-check reward with optional length bonus.
Phase 3+ will add predictive compression, correspondence, curiosity,
and simplicity-penalty reward components.

See src/reward/config.py for configuration.
See model_structure_and_data.md § Steps 7–10 for the full reward pipeline.
"""