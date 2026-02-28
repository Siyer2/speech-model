"""Configuration loading and validation."""

import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import yaml


def _set_global_seed(seed: int):
    """Set seed for all random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    seed: int
    val_split: float
    num_workers: int
    early_stopping_patience: int
    resume_checkpoint: str = ""


@dataclass
class DataConfig:
    """Data configuration."""

    parquet_path: str
    audio_base_path: str
    checkpoint_dir: str
    sample_rate: int


@dataclass
class WandBConfig:
    """WandB configuration."""

    project: str
    entity: str | None
    enabled: bool


@dataclass
class Config:
    """Main configuration container."""

    training: TrainingConfig
    data: DataConfig
    wandb: WandBConfig

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "Config":
        """Load configuration from YAML file and set global seed."""
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(yaml_path) as f:
            config_dict = yaml.safe_load(f)

        config = cls(
            training=TrainingConfig(**config_dict["training"]),
            data=DataConfig(**config_dict["data"]),
            wandb=WandBConfig(**config_dict["wandb"]),
        )

        _set_global_seed(config.training.seed)
        return config

    def to_dict(self) -> dict:
        """Convert config to dictionary for logging."""
        return asdict(self)
