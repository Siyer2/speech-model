"""Configuration loading and validation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    encoder_dim: int
    hidden_dim: int
    num_classes: int
    dropout: float


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    batch_size: int
    epochs: int
    learning_rate: float
    seed: int


@dataclass
class DataConfig:
    """Data configuration."""

    data_dir: str
    sample_rate: int


@dataclass
class WandBConfig:
    """WandB configuration."""

    project: str
    entity: Optional[str]
    enabled: bool


@dataclass
class Config:
    """Main configuration container."""

    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    wandb: WandBConfig

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "Config":
        """Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            Config object with loaded settings
        """
        yaml_path = Path(yaml_path)

        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(yaml_path) as f:
            config_dict = yaml.safe_load(f)

        return cls(
            model=ModelConfig(**config_dict["model"]),
            training=TrainingConfig(**config_dict["training"]),
            data=DataConfig(**config_dict["data"]),
            wandb=WandBConfig(**config_dict["wandb"]),
        )

    def to_dict(self) -> dict:
        """Convert config to dictionary for logging."""
        return {
            "model": {
                "encoder_dim": self.model.encoder_dim,
                "hidden_dim": self.model.hidden_dim,
                "num_classes": self.model.num_classes,
                "dropout": self.model.dropout,
            },
            "training": {
                "batch_size": self.training.batch_size,
                "epochs": self.training.epochs,
                "learning_rate": self.training.learning_rate,
                "seed": self.training.seed,
            },
            "data": {
                "data_dir": self.data.data_dir,
                "sample_rate": self.data.sample_rate,
            },
            "wandb": {
                "project": self.wandb.project,
                "entity": self.wandb.entity,
                "enabled": self.wandb.enabled,
            },
        }
