"""Experience replay buffer for GRPO self-play training.

Stores generated proofs and their rewards for experience reuse.
Each generated proof can be sampled for multiple training updates.
"""

import random
from collections import deque
from dataclasses import dataclass


@dataclass
class ProofExperience:
    """A single experience: theorem statement + generated proof + reward."""

    statement: str
    generated_proof: str
    reward: float
    success: bool


class ProofReplayBuffer:
    """Rolling buffer of proof generation experiences.

    Stores (statement, proof, reward) tuples with a maximum capacity.
    Provides sampling for training updates.
    """

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffer: deque[ProofExperience] = deque(maxlen=max_size)
        self.total_added = 0
        self.total_success = 0

    def add(self, statement: str, proof: str, reward: float) -> None:
        """Add a generated proof experience to the buffer."""
        self.buffer.append(
            ProofExperience(
                statement=statement,
                generated_proof=proof,
                reward=reward,
                success=reward > 0.0,
            )
        )
        self.total_added += 1
        if reward > 0.0:
            self.total_success += 1

    def sample(self, batch_size: int) -> list[ProofExperience]:
        """Sample a random batch of experiences."""
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        return random.sample(list(self.buffer), batch_size)

    @property
    def success_rate(self) -> float:
        """Fraction of total experiences with positive reward."""
        if self.total_added == 0:
            return 0.0
        return self.total_success / self.total_added

    @property
    def size(self) -> int:
        return len(self.buffer)

    def clear(self) -> None:
        self.buffer.clear()
        self.total_added = 0
        self.total_success = 0
