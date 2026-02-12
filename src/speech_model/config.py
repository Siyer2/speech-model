"""Configuration loading and validation."""

import random
from dataclasses import dataclass
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
class ModelConfig:
    """Model configuration."""

    d_model: int
    num_heads: int
    ff_dim: int
    num_layers: int
    conv_kernel_size: int
    dropout: float
    backbone: str | None = None  # None = original CNN, "hubert_base" = pretrained
    freeze_backbone: bool = True


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
class DecodingConfig:
    """Decoding configuration for validation."""

    method: str = "greedy"  # "greedy" or "beam_search"
    beam_width: int = 10


@dataclass
class Config:
    """Main configuration container."""

    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    wandb: WandBConfig
    decoding: DecodingConfig

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "Config":
        """Load configuration from YAML file and set global seed."""
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(yaml_path) as f:
            config_dict = yaml.safe_load(f)

        config = cls(
            model=ModelConfig(**config_dict.get("model", {})),
            training=TrainingConfig(**config_dict["training"]),
            data=DataConfig(**config_dict["data"]),
            wandb=WandBConfig(**config_dict["wandb"]),
            decoding=DecodingConfig(**config_dict.get("decoding", {})),
        )

        _set_global_seed(config.training.seed)
        return config

    def to_dict(self) -> dict:
        """Convert config to dictionary for logging."""
        return {
            "model": {
                "d_model": self.model.d_model,
                "num_heads": self.model.num_heads,
                "ff_dim": self.model.ff_dim,
                "num_layers": self.model.num_layers,
                "conv_kernel_size": self.model.conv_kernel_size,
                "dropout": self.model.dropout,
                "backbone": self.model.backbone,
                "freeze_backbone": self.model.freeze_backbone,
            },
            "training": {
                "batch_size": self.training.batch_size,
                "epochs": self.training.epochs,
                "learning_rate": self.training.learning_rate,
                "weight_decay": self.training.weight_decay,
                "seed": self.training.seed,
                "val_split": self.training.val_split,
                "num_workers": self.training.num_workers,
                "early_stopping_patience": self.training.early_stopping_patience,
            },
            "data": {
                "parquet_path": self.data.parquet_path,
                "audio_base_path": self.data.audio_base_path,
                "checkpoint_dir": self.data.checkpoint_dir,
                "sample_rate": self.data.sample_rate,
            },
            "wandb": {
                "project": self.wandb.project,
                "entity": self.wandb.entity,
                "enabled": self.wandb.enabled,
            },
            "decoding": {
                "method": self.decoding.method,
                "beam_width": self.decoding.beam_width,
            },
        }
