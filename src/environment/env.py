"""Gym-like environment interface for the self-play proof generation loop.

Encapsulates: observation (theorem statement), action (proof generation),
reward (proof checker output).

This is the direct analog of the AlphaGo Zero game environment.
The "game" is theorem proving against the Lean 4 proof checker.
"""

from dataclasses import dataclass

from src.proof_checker.lean_interface import LeanProofChecker
from src.proof_checker.formats import ProofResult, wrap_theorem_with_proof
from src.reward.base import compute_reward
from src.reward.config import RewardConfig


@dataclass
class EnvironmentStep:
    """Result of one step in the proof generation environment."""

    statement: str
    generated_proof: str
    result: ProofResult
    reward: float
    done: bool = True  # Single-step: theorem is proven or not


class ProofEnvironment:
    """Self-play environment for theorem proving.

    Single-step environment:
    - Observation: theorem statement
    - Action: generated proof text
    - Reward: proof checker output (binary + optional bonus)
    """

    def __init__(
        self,
        proof_checker: LeanProofChecker | None = None,
        reward_config: RewardConfig | None = None,
    ):
        self.checker = proof_checker or LeanProofChecker()
        self.reward_config = reward_config or RewardConfig()

    def step(self, statement: str, proof: str) -> EnvironmentStep:
        """Evaluate a generated proof against the proof checker.

        Args:
            statement: Theorem statement (e.g., "theorem add_comm (a b : Nat) : a + b = b + a").
            proof: Generated proof body (tactic block).

        Returns:
            EnvironmentStep with result and reward.
        """
        full_code = wrap_theorem_with_proof(statement, proof)
        result = self.checker.check(full_code)
        reward = compute_reward(result, self.reward_config)

        return EnvironmentStep(
            statement=statement,
            generated_proof=proof,
            result=result,
            reward=reward,
        )

    def evaluate_proofs(
        self, statement: str, proofs: list[str]
    ) -> list[EnvironmentStep]:
        """Evaluate multiple proofs for the same theorem.

        Used for GRPO group generation: K proofs per theorem statement.
        """
        return [self.step(statement, proof) for proof in proofs]
