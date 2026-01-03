"""Configuration loading and validation."""

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml


def _set_global_seed(seed: int):
    """Set seed for all random number generators.

    Called automatically when loading config to ensure reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    encoder_name: str
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
    k_folds: int
    threshold: float
    num_workers: int
    early_stopping_patience: int
    save_best_only: bool


@dataclass
class DataConfig:
    """Data configuration."""

    parquet_path: str
    ontology_path: str
    embeddings_dir: str
    sample_rate: int
    checkpoint_dir: str
    clean_labels: bool = True  # Remove substitution_error when other patterns exist


@dataclass
class WandBConfig:
    """WandB configuration."""

    project: str
    entity: str | None
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
        """Load configuration from YAML file and set global seed.

        This ensures reproducibility is enforced for ALL experiments.

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

        config = cls(
            model=ModelConfig(**config_dict["model"]),
            training=TrainingConfig(**config_dict["training"]),
            data=DataConfig(**config_dict["data"]),
            wandb=WandBConfig(**config_dict["wandb"]),
        )

        # Automatically set seed for reproducibility
        _set_global_seed(config.training.seed)

        return config

    def to_dict(self) -> dict:
        """Convert config to dictionary for logging."""
        return {
            "model": {
                "encoder_name": self.model.encoder_name,
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
                "k_folds": self.training.k_folds,
                "threshold": self.training.threshold,
                "num_workers": self.training.num_workers,
                "early_stopping_patience": self.training.early_stopping_patience,
                "save_best_only": self.training.save_best_only,
            },
            "data": {
                "parquet_path": self.data.parquet_path,
                "ontology_path": self.data.ontology_path,
                "embeddings_dir": self.data.embeddings_dir,
                "sample_rate": self.data.sample_rate,
                "checkpoint_dir": self.data.checkpoint_dir,
                "clean_labels": self.data.clean_labels,
            },
            "wandb": {
                "project": self.wandb.project,
                "entity": self.wandb.entity,
                "enabled": self.wandb.enabled,
            },
        }
