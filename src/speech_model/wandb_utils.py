"""WandB integration utilities."""

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

        if self.enabled:
            self._init_wandb()

    def _init_wandb(self):
        """Initialize WandB run."""
        self.run = wandb.init(
            project=self.config.wandb.project,
            entity=self.config.wandb.entity,
            config=self.config.to_dict(),
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

    def finish(self):
        """Finish WandB run."""
        if self.enabled and self.run:
            wandb.finish()
