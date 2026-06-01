"""GRPO self-play training loop.

The core AlphaGo Zero analog:
1. Model generates K proofs per theorem statement
2. Proof checker verifies them (CPU, parallel)
3. Rewards computed (binary + length bonus)
4. Group-relative advantages computed
5. Policy updated via GRPO loss + KL penalty
"""

import time
from pathlib import Path

import torch

from src.utils.xpu_utils import clear_gpu_memory, get_device

from src.data.replay_buffer import ProofReplayBuffer
from src.model.generation import generate_proofs
from src.proof_checker.batch_checker import BatchChecker
from src.proof_checker.formats import wrap_theorem_with_proof
from src.reward.base import compute_rewards_batch, compute_group_advantages
from src.reward.config import RewardConfig
from src.training.losses import compute_grpo_loss, compute_sequence_logprob
from src.utils.checkpoint import save_checkpoint
from src.utils.config import GRPOConfig
from src.utils.logging import MetricsLogger


class GRPOTrainer:
    """Group Relative Policy Optimization trainer for theorem proving.

    Trains a policy model to generate valid Lean 4 proofs through
    self-play against the Lean proof checker.
    """

    def __init__(
        self,
        policy_model,
        reference_model,
        tokenizer,
        proof_checker: BatchChecker,
        config: GRPOConfig | None = None,
        reward_config: RewardConfig | None = None,
        logger: MetricsLogger | None = None,
        device: torch.device | None = None,
    ):
        self.policy_model = policy_model
        self.reference_model = reference_model
        self.tokenizer = tokenizer
        self.checker = proof_checker
        self.config = config or GRPOConfig()
        self.reward_config = reward_config or RewardConfig()
        self.logger = logger
        self.device = device or get_device()

        self.replay_buffer = ProofReplayBuffer(
            max_size=self.config.replay_buffer.max_size,
        )

        self.optimizer = torch.optim.AdamW(
            self.policy_model.parameters(),
            lr=self.config.training.learning_rate,
            betas=(
                self.config.optimizer.beta1,
                self.config.optimizer.beta2,
            ),
            weight_decay=self.config.optimizer.weight_decay,
        )

        self.global_step = 0

    def train(
        self,
        train_dataset,
        val_dataset=None,
        output_dir: str | Path = "checkpoints/grpo",
    ) -> dict:
        """Run the full GRPO self-play training loop."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cfg = self.config.training
        gen_cfg = self.config.generation

        print(f"Starting GRPO training: {cfg.max_steps} steps")
        print(f"Group size K: {cfg.group_size}")
        print(f"Batch theorems: {cfg.batch_theorems}")
        print(f"KL beta: {cfg.kl_beta}")

        all_metrics = []

        theorem_idx = 0
        for step in range(cfg.max_steps):
            step_start = time.time()

            # 1. Sample batch of theorem statements
            batch_theorems = self._sample_theorems(
                train_dataset, cfg.batch_theorems, theorem_idx
            )
            theorem_idx += cfg.batch_theorems

            # 2. Generate K proofs per theorem (GPU)
            prompts = [
                f"Theorem: {t['statement']}\nProof:" for t in batch_theorems
            ]
            all_proofs_flat = generate_proofs(
                self.policy_model,
                self.tokenizer,
                prompts,
                gen_cfg,
                num_return_sequences=cfg.group_size,
            )

            # Flatten: [batch_theorems * K] code strings
            all_codes = []
            for i, proofs in enumerate(all_proofs_flat):
                for proof in proofs:
                    full_code = wrap_theorem_with_proof(
                        batch_theorems[i]["statement"], proof
                    )
                    all_codes.append(full_code)

            # 3. Verify all proofs (CPU, parallel)
            results = self.checker.check_batch(all_codes)
            rewards = compute_rewards_batch(results, self.reward_config)

            # 4. Add to replay buffer
            for i, code in enumerate(all_codes):
                theorem = batch_theorems[i // cfg.group_size]
                self.replay_buffer.add(
                    theorem["statement"], code, rewards[i].item()
                )

            # 5. Compute group-relative advantages
            advantages = compute_group_advantages(rewards, cfg.group_size)

            # 6. Compute log probabilities for policy and reference
            device_rewards = rewards.to(self.device)
            device_advantages = advantages.to(self.device)

            code_texts_for_model = [
                f"Theorem: {batch_theorems[i // cfg.group_size]['statement']}\nProof: {all_codes[i].split(':= by\n  ')[-1] if ':= by\n  ' in all_codes[i] else all_codes[i]}"
                for i in range(len(all_codes))
            ]

            # Tokenize generated proofs
            encodings = self.tokenizer(
                code_texts_for_model,
                truncation=True,
                max_length=1024,
                padding=True,
                return_tensors="pt",
            )
            encodings = {k: v.to(self.device) for k, v in encodings.items()}

            # 7. Compute logprobs
            policy_logprobs = compute_sequence_logprob(
                self.policy_model,
                encodings["input_ids"],
                encodings["attention_mask"],
            )

            with torch.no_grad():
                ref_logprobs = compute_sequence_logprob(
                    self.reference_model,
                    encodings["input_ids"],
                    encodings["attention_mask"],
                )

            # 8. GRPO loss
            loss_dict = compute_grpo_loss(
                policy_logprobs, ref_logprobs, device_advantages, cfg.kl_beta
            )

            # 9. Backward pass
            loss_dict["loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                self.policy_model.parameters(),
                self.config.optimizer.max_grad_norm,
            )
            self.optimizer.step()
            self.optimizer.zero_grad()

            self.global_step += 1

            # 10. Logging
            step_time = time.time() - step_start
            success_rate = (rewards > 0).float().mean().item()
            avg_reward = rewards.mean().item()

            metrics = {
                "loss": loss_dict["loss"].item(),
                "pg_loss": loss_dict["pg_loss"].item(),
                "kl_div": loss_dict["kl_div"].item(),
                "ratio_mean": loss_dict["ratio_mean"].item(),
                "success_rate": success_rate,
                "avg_reward": avg_reward,
                "step_time_s": step_time,
            }

            all_metrics.append(metrics)

            if self.logger:
                self.logger.log_step(step, metrics)

            if step % cfg.log_every == 0 or step == cfg.max_steps - 1:
                print(
                    f"Step {step}/{cfg.max_steps} | "
                    f"Success: {success_rate:.2%} | "
                    f"Reward: {avg_reward:.3f} | "
                    f"KL: {loss_dict['kl_div'].item():.4f} | "
                    f"Loss: {loss_dict['loss'].item():.4f} | "
                    f"Time: {step_time:.2f}s"
                )

            # 11. Checkpoint
            if step % cfg.save_every == 0 and step > 0:
                save_checkpoint(
                    self.policy_model,
                    self.tokenizer,
                    self.optimizer,
                    step,
                    output_dir,
                    {"step": step, **metrics},
                )

            # 12. Evaluation
            if val_dataset and step % cfg.eval_every == 0 and step > 0:
                eval_metrics = self.evaluate(val_dataset, num_theorems=20)
                metrics["eval_success_rate"] = eval_metrics["success_rate"]
                print(f"  Eval success rate: {eval_metrics['success_rate']:.2%}")

            clear_gpu_memory()

        # Final save
        save_checkpoint(
            self.policy_model,
            self.tokenizer,
            self.optimizer,
            cfg.max_steps,
            output_dir,
        )

        return {"metrics": all_metrics}

    def evaluate(self, dataset, num_theorems: int = 20) -> dict:
        """Evaluate the policy on held-out theorems."""
        self.policy_model.eval()

        theorems = self._sample_theorems(dataset, num_theorems, 0)

        prompts = [f"Theorem: {t['statement']}\nProof:" for t in theorems]
        gen_cfg = self.config.generation
        gen_cfg.do_sample = True
        gen_cfg.temperature = 0.6  # Lower temperature for evaluation

        all_proofs = generate_proofs(
            self.policy_model,
            self.tokenizer,
            prompts,
            gen_cfg,
            num_return_sequences=1,
        )

        codes = []
        for i, proofs in enumerate(all_proofs):
            codes.append(wrap_theorem_with_proof(theorems[i]["statement"], proofs[0]))

        results = self.checker.check_batch(codes)
        success_rate = sum(1 for r in results if r.success) / len(results)

        self.policy_model.train()
        return {"success_rate": success_rate, "num_evaluated": len(codes)}

    @staticmethod
    def _sample_theorems(dataset, batch_size: int, start_idx: int) -> list[dict]:
        """Sample a batch of theorems from the dataset, cycling through."""
        indices = [
            (start_idx + i) % len(dataset) for i in range(batch_size)
        ]
        return [dataset[idx] for idx in indices]
