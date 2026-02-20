"""WandB integration utilities."""

import os
from pathlib import Path
from typing import Any

import wandb

from .config import Config


class WandBLogger:
    """Abstraction for WandB experiment tracking."""

    def __init__(self, config: Config, config_path: Path):
        """Initialize WandB logger.

        Args:
            config: Configuration object
            config_path: Path to the config YAML file (for artifact upload)
        """
        self.config = config
        self.config_path = config_path
        self.enabled = config.wandb.enabled
        self.run = None
        self.note = os.environ.get("EXPERIMENT_NOTE", "")
        self.name = os.environ.get("EXPERIMENT_NAME", "")

        if self.enabled:
            self._init_wandb()

    def _init_wandb(self):
        """Initialize WandB run."""
        self.run = wandb.init(
            project=self.config.wandb.project,
            entity=self.config.wandb.entity,
            config=self.config.to_dict(),
            notes=self.note if self.note else None,
            name=self.name if self.name else None,
        )

        # Upload config file as artifact
        if self.config_path.exists():
            artifact = wandb.Artifact(
                name="training-config",
                type="config",
                description="Training configuration YAML",
            )
            artifact.add_file(str(self.config_path))
            wandb.log_artifact(artifact)

    def log(self, metrics: dict[str, Any], step: int | None = None):
        """Log metrics to WandB.

        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number
        """
        if self.enabled and self.run:
            wandb.log(metrics, step=step)

    def log_predictions(self, samples: list[tuple[str, str, float]], step: int | None = None):
        """Log prediction samples as a wandb Table.

        Args:
            samples: List of (prediction, target, cer) tuples
            step: Optional step number
        """
        if self.enabled and self.run:
            table = wandb.Table(columns=["prediction", "target", "cer"])
            for pred, target, cer_val in samples:
                table.add_data(pred, target, cer_val)
            wandb.log({"val/predictions": table}, step=step)
            print(f"Logged {len(samples)} predictions to wandb")

    def finish(self):
        """Finish WandB run."""
        if self.enabled and self.run:
            wandb.finish()
