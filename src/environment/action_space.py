"""Definition of valid proof actions for the mathematical explorer.

In Phase 1, the action space is the space of Lean 4 tactic sequences.
This is effectively unbounded (any string up to max tokens), so we
use the tokenizer's vocabulary as the discrete action space.
"""

import torch


class ProofActionSpace:
    """Action space for proof generation.

    The model generates tokens auto-regressively from the LLM vocabulary.
    This is a simple wrapper that defines the space constraints.
    """

    def __init__(
        self,
        vocab_size: int,
        max_tokens: int = 512,
        stop_tokens: list[int] | None = None,
    ):
        self.vocab_size = vocab_size
        self.max_tokens = max_tokens
        self.stop_tokens = stop_tokens or []

    def is_valid_action(self, token_id: int) -> bool:
        """Check if a token is a valid action."""
        return 0 <= token_id < self.vocab_size

    def is_terminal(self, token_id: int) -> bool:
        """Check if the token terminates the proof."""
        return token_id in self.stop_tokens

    @property
    def shape(self) -> tuple:
        return (self.vocab_size,)
