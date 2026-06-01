"""Logging utilities for training metrics."""

import time
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel


class MetricsLogger:
    """Logs training metrics to console and optionally to TensorBoard/WandB."""

    def __init__(
        self,
        log_dir: Path,
        use_wandb: bool = False,
        wandb_project: str = "math-physics-ai",
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.console = Console()
        self.use_wandb = use_wandb
        self.metrics_history: list[dict] = []
        self.start_time = time.time()
        self.step_times: list[float] = []

        if use_wandb:
            try:
                import wandb

                wandb.init(project=wandb_project, dir=str(self.log_dir))
                self.wandb = wandb
            except Exception as e:
                print(f"Warning: WandB init failed: {e}")
                self.use_wandb = False

    def log_step(self, step: int, metrics: dict) -> None:
        """Log metrics for a single training step."""
        elapsed = time.time() - self.start_time
        self.step_times.append(elapsed)

        record = {"step": step, "elapsed": elapsed, **metrics}
        self.metrics_history.append(record)

        if self.use_wandb:
            self.wandb.log(record, step=step)

    def print_status(self, step: int, metrics: dict, total_steps: int) -> None:
        """Print a rich-formatted status table."""
        table = Table(title=f"GRPO Training — Step {step}/{total_steps}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        for key, value in metrics.items():
            if isinstance(value, float):
                table.add_row(key, f"{value:.4f}")
            else:
                table.add_row(key, str(value))

        steps_per_sec = (
            step / (time.time() - self.start_time) if self.step_times else 0
        )
        table.add_row("steps/sec", f"{steps_per_sec:.2f}")

        self.console.clear()
        self.console.print(table)

    def save_metrics(self, filename: str = "metrics.jsonl") -> None:
        """Save metrics history to JSONL file."""
        import json

        path = self.log_dir / filename
        with open(path, "w") as f:
            for record in self.metrics_history:
                f.write(json.dumps(record) + "\n")
        print(f"Metrics saved to {path}")

    def close(self) -> None:
        """Clean up logging."""
        self.save_metrics()
        if self.use_wandb:
            self.wandb.finish()
