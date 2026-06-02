"""Supervised fine-tuning loop for initial model training on Mathlib4 theorems."""

import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.utils.config import SFTConfig
from src.utils.xpu_utils import clear_gpu_memory, get_device


class SFTTrainer:
    """Simple supervised fine-tuning trainer for theorem-proof pairs.

    Trains the model to generate proofs given theorem statements.
    This is pretraining before GRPO self-play.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: SFTConfig | None = None,
        device: torch.device | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or SFTConfig()
        self.device = device or get_device()

    def train(
        self,
        train_dataset,
        val_dataset=None,
        output_dir: str | Path = "checkpoints/sft",
    ) -> dict:
        """Run SFT training loop."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cfg = self.config.training

        # num_workers=0 required for XPU compatibility:
        # forking after XPU context init causes deadlocks.
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.per_device_batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=(self.device.type != "xpu"),
        )

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.learning_rate,
            betas=(self.config.optimizer.beta1, self.config.optimizer.beta2),
            weight_decay=cfg.weight_decay,
        )

        total_steps = len(train_loader) * cfg.num_epochs
        warmup_steps = int(total_steps * cfg.warmup_ratio)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg.learning_rate,
            total_steps=total_steps,
            pct_start=warmup_steps / total_steps if total_steps > 0 else 0,
        )

        global_step = 0
        best_val_loss = float("inf")
        metrics_history = []

        print(f"Starting SFT: {cfg.num_epochs} epochs, {total_steps} total steps")
        print(f"Warmup steps: {warmup_steps}")

        for epoch in range(cfg.num_epochs):
            self.model.train()
            epoch_loss = 0.0

            for batch in train_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.model(**batch)
                loss = outputs.loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.optimizer.max_grad_norm,
                )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                epoch_loss += loss.item()

                if global_step % 50 == 0:
                    metrics_history.append(
                        {
                            "step": global_step,
                            "loss": loss.item(),
                            "lr": scheduler.get_last_lr()[0],
                        }
                    )
                    print(
                        f"Step {global_step}/{total_steps} | Loss: {loss.item():.4f} "
                        f"| LR: {scheduler.get_last_lr()[0]:.2e}"
                    )

                clear_gpu_memory()

            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{cfg.num_epochs} | Avg Loss: {avg_loss:.4f}")

            # Validation
            if val_dataset is not None:
                val_loss = self.evaluate(val_dataset)
                print(f"Validation Loss: {val_loss:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save(output_dir / "best")

        # Final save
        self.save(output_dir / "final")
        return {"metrics": metrics_history, "best_val_loss": best_val_loss}

    def evaluate(self, dataset) -> float:
        """Evaluate on validation set."""
        self.model.eval()
        loader = DataLoader(
            dataset,
            batch_size=self.config.training.per_device_batch_size,
            shuffle=False,
            num_workers=0,
        )

        total_loss = 0.0
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                total_loss += outputs.loss.item()

        self.model.train()
        return total_loss / len(loader) if len(loader) > 0 else float("inf")

    def save(self, save_dir: Path) -> None:
        """Save SFT checkpoint."""
        save_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        print(f"Saved SFT checkpoint to {save_dir}")
