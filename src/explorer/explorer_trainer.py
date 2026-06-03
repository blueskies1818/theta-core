"""GRPO self-play trainer using MCTS + GNN for proof generation (Phase 2.4).

Replaces the LLM-based proof generator from Phase 1 with the MCTS explorer.
The GNN is the policy network — its embeddings guide MCTS search, and GRPO
updates the GNN based on proof checker feedback.

Architecture:
    1. Sample theorem batch
    2. MCTS.search(goal) → proof candidates (using GNN for guidance)
    3. Proof checker validates each candidate
    4. Compute rewards (correctness + curiosity + length)
    5. Group-relative advantages over the search tree
    6. Update GNN via policy gradient + value loss

This is the core AlphaGo Zero training loop applied to theorem proving.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn.functional as F

from src.explorer.dependency_graph import DependencyGraph
from src.explorer.gnn_config import GNNConfig
from src.explorer.gnn_encoder import GNNEncoder, prepare_graph_tensors, extract_initial_features
from src.explorer.mcts import MCTS, MCTSConfig
from src.explorer.proof_state import ProofState
from src.proof_checker.batch_checker import BatchChecker
from src.proof_checker.formats import wrap_theorem_with_proof
from src.reward.base import (
    compute_rewards_batch,
    compute_group_advantages,
    record_proof_signature,
    get_curiosity_stats,
)
from src.reward.config import RewardConfig
from src.utils.checkpoint import save_checkpoint


class ExplorerTrainer:
    """GRPO trainer for the GNN+MCTS explorer.

    Replaces the LLM `generate_proofs()` call with MCTS search.
    The GNN is the trainable component — MCTS is a fixed inference
    algorithm that uses the GNN's scores.

    Training signal:
    - Policy: link prediction between goals and successful lemmas
    - Value: MSE between predicted and actual proof success
    - GRPO: group-relative advantages over proof candidates
    """

    def __init__(
        self,
        gnn_encoder: GNNEncoder,
        dependency_graph: DependencyGraph,
        proof_checker: BatchChecker,
        config: "ExplorerConfig | None" = None,
        mcts_config: MCTSConfig | None = None,
        reward_config: RewardConfig | None = None,
        device: torch.device | None = None,
    ):
        self.gnn = gnn_encoder
        self.graph = dependency_graph
        self.checker = proof_checker
        self.config = config or ExplorerConfig()
        self.mcts_config = mcts_config or MCTSConfig()
        self.reward_config = reward_config or RewardConfig()

        if device is None:
            device = torch.device("xpu:0" if torch.xpu.is_available() else "cpu")
        self.device = device

        self.gnn = self.gnn.to(device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.gnn.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Pre-compute graph tensors for GNN forward pass
        self._sources, self._targets, self._edge_types, self._num_nodes = (
            prepare_graph_tensors(self.graph, device=torch.device("cpu"))
        )
        self._initial_features: torch.Tensor | None = None

        # MCTS instance
        self._mcts: MCTS | None = None

        self.global_step = 0

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        train_theorems: list[dict],
        val_theorems: list[dict] | None = None,
        output_dir: str | Path = "checkpoints/explorer",
        num_epochs: int = 100,
    ) -> dict:
        """Run the GRPO+MCTS training loop.

        Args:
            train_theorems: List of theorem dicts with 'statement' key.
            val_theorems: Optional validation theorems.
            output_dir: Checkpoint save directory.
            num_epochs: Number of passes through the training data.

        Returns:
            dict with training metrics history.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Pre-compute initial node features
        if self._initial_features is None:
            self._initial_features = extract_initial_features(
                self.graph, self.gnn.config, device=torch.device("cpu")
            )

        # Initialize MCTS
        self._mcts = MCTS(
            gnn_encoder=self.gnn,
            dependency_graph=self.graph,
            config=self.mcts_config,
        )

        print(f"Explorer GRPO training: {num_epochs} epochs")
        print(f"Train theorems: {len(train_theorems)}")
        print(f"MCTS sims: {self.mcts_config.num_simulations}")
        print(f"GNN layers: {self.gnn.config.num_layers}, "
              f"hidden: {self.gnn.config.hidden_dim}")
        print(f"Learning rate: {self.config.learning_rate}")

        all_metrics = []

        for epoch in range(num_epochs):
            epoch_start = time.time()

            # Sample batch of theorems
            batch = self._sample_batch(train_theorems, self.config.batch_size)

            # ---- Phase A: Compute GNN embeddings ----
            features = self._initial_features.to(self.device)
            sources = self._sources.to(self.device)
            targets = self._targets.to(self.device)
            edge_types = self._edge_types.to(self.device)

            embeddings = self.gnn(features, sources, targets, edge_types, self._num_nodes)
            self._mcts.set_embeddings(embeddings, sorted(self.graph.node_ids))

            # ---- Phase B: MCTS search + proof checking ----
            all_codes = []
            all_trees = []

            for theorem in batch:
                statement = theorem["statement"]

                # Run MCTS to find proof
                best_steps, root = self._mcts.search(
                    statement, node_embeddings=embeddings, verbose=False
                )

                # Convert steps to Lean code
                proof_text = ProofState._render_proof(best_steps)
                full_code = wrap_theorem_with_proof(statement, proof_text or "sorry")
                all_codes.append(full_code)
                all_trees.append(root)

            # ---- Phase C: Proof checking ----
            results = self.checker.check_batch(all_codes)

            # ---- Phase D: Rewards ----
            rewards = compute_rewards_batch(results, self.reward_config, proof_texts=all_codes)
            for code in all_codes:
                record_proof_signature(code, self.reward_config)

            # ---- Phase E: GRPO loss ----
            # Group advantages across the batch
            advantages = compute_group_advantages(rewards, self.config.group_size)

            # Compute training loss
            loss = self._compute_explorer_loss(
                embeddings, all_trees, all_codes, results, advantages
            )

            # ---- Phase F: Backward pass ----
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.gnn.parameters(), self.config.max_grad_norm)
            self.optimizer.step()

            self.global_step += 1

            # ---- Phase G: Metrics ----
            success_rate = (rewards >= 1.0).float().mean().item()
            avg_reward = rewards.mean().item()

            metrics = {
                "epoch": epoch,
                "loss": loss.item(),
                "success_rate": success_rate,
                "avg_reward": avg_reward,
                "epoch_time_s": time.time() - epoch_start,
            }
            all_metrics.append(metrics)

            if epoch % self.config.log_every == 0 or epoch == num_epochs - 1:
                curiosity = get_curiosity_stats()
                print(
                    f"Epoch {epoch}/{num_epochs} | "
                    f"Success: {success_rate:.2%} | "
                    f"Reward: {avg_reward:.3f} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Novel: {curiosity['unique_signatures']}"
                )

            # Save checkpoint
            if epoch % self.config.save_every == 0 and epoch > 0:
                self.gnn.save(output_dir / f"gnn_epoch_{epoch}.pt")

            # Clear GPU
            if self.device.type == "xpu":
                torch.xpu.empty_cache()

        # Final save
        self.gnn.save(output_dir / "gnn_final.pt")
        self.graph.save(output_dir / "dependency_graph")

        return {"metrics": all_metrics}

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def _compute_explorer_loss(
        self,
        embeddings: torch.Tensor,
        trees: list["MCTSNode"],
        codes: list[str],
        results: list,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the explorer training loss.

        Combines:
        1. Policy loss: encourage GNN to score successful lemmas higher
        2. Value loss: MSE between GNN value and actual outcome
        3. (Optional) KL penalty against previous GNN state

        Args:
            embeddings: [N, D] current GNN node embeddings.
            trees: MCTS root nodes for each theorem.
            codes: Proof code strings.
            results: Proof checker results.
            advantages: [batch_size] advantage values from GRPO.

        Returns:
            Scalar loss tensor.
        """
        policy_losses = []
        value_losses = []

        for i, (root, result, advantage) in enumerate(zip(trees, results, advantages)):
            # Policy loss: for each MCTS node, push priors toward actual outcomes
            if root.children:
                # Get visit count distribution (target policy)
                total_visits = sum(c.visit_count for c in root.children.values())
                if total_visits > 0:
                    target_probs = []
                    prior_probs = []

                    for action, child in root.children.items():
                        target_probs.append(child.visit_count / total_visits)
                        prior_probs.append(child.prior)

                    if target_probs:
                        target = torch.tensor(target_probs, device=self.device)
                        prior = torch.tensor(prior_probs, device=self.device)
                        # Cross-entropy: push priors toward visit distribution
                        # Weighted by advantage: successful proofs get stronger signal
                        weight = torch.sigmoid(advantage)
                        policy_loss = F.kl_div(
                            torch.log_softmax(prior, dim=0),
                            torch.softmax(target, dim=0),
                            reduction="batchmean",
                        )
                        policy_losses.append(weight * policy_loss)

            # Value loss: compare GNN value estimate vs actual outcome
            actual_value = 1.0 if result.success else 0.0
            predicted_value = root.value_estimate
            value_loss = F.mse_loss(
                torch.tensor(predicted_value, device=self.device),
                torch.tensor(actual_value, device=self.device),
            )
            value_losses.append(value_loss)

        # Combine losses
        if policy_losses:
            total_policy_loss = torch.stack(policy_losses).mean()
        else:
            total_policy_loss = torch.tensor(0.0, device=self.device)

        if value_losses:
            total_value_loss = torch.stack(value_losses).mean()
        else:
            total_value_loss = torch.tensor(0.0, device=self.device)

        loss = (
            self.config.policy_weight * total_policy_loss
            + self.config.value_weight * total_value_loss
        )

        return loss

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sample_batch(
        self, theorems: list[dict], batch_size: int
    ) -> list[dict]:
        """Sample a random batch of theorems."""
        import random

        return random.sample(
            theorems, min(batch_size, len(theorems))
        )

    def evaluate(self, val_theorems: list[dict]) -> dict:
        """Evaluate the explorer on validation theorems."""
        self.gnn.eval()

        features = self._initial_features.to(self.device)
        sources = self._sources.to(self.device)
        targets = self._targets.to(self.device)
        edge_types = self._edge_types.to(self.device)

        with torch.no_grad():
            embeddings = self.gnn(features, sources, targets, edge_types, self._num_nodes)

        self._mcts.set_embeddings(embeddings, sorted(self.graph.node_ids))

        codes = []
        for theorem in val_theorems:
            best_steps, _ = self._mcts.search(
                theorem["statement"], node_embeddings=embeddings, verbose=False
            )
            proof_text = ProofState._render_proof(best_steps)
            codes.append(
                wrap_theorem_with_proof(theorem["statement"], proof_text or "sorry")
            )

        results = self.checker.check_batch(codes)
        success_rate = (
            sum(1 for r in results if r.success) / len(results) if results else 0.0
        )

        self.gnn.train()
        return {"success_rate": success_rate, "num_evaluated": len(codes)}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

from dataclasses import dataclass


@dataclass
class ExplorerConfig:
    """Configuration for the explorer trainer."""

    # Batch size (number of theorems per training step)
    batch_size: int = 4

    # Group size for GRPO advantages (proofs per theorem)
    group_size: int = 2

    # Learning rate for GNN optimizer
    learning_rate: float = 1e-3

    # Weight decay
    weight_decay: float = 1e-5

    # Max gradient norm
    max_grad_norm: float = 1.0

    # Loss weights
    policy_weight: float = 1.0
    value_weight: float = 0.5

    # Logging frequency
    log_every: int = 5

    # Checkpoint saving frequency
    save_every: int = 50
