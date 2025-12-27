"""Test configuration loading."""

from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest

from speech_model.config import Config


def test_config_from_yaml():
    """Test loading config from YAML file."""
    yaml_content = """
model:
  encoder_dim: 256
  hidden_dim: 128
  num_classes: 5
  dropout: 0.1

training:
  batch_size: 16
  epochs: 10
  learning_rate: 0.001
  seed: 42

data:
  data_dir: "data/audio"
  sample_rate: 16000

wandb:
  project: "test-project"
  entity: null
  enabled: false
"""

    with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        temp_path = f.name

    try:
        config = Config.from_yaml(temp_path)

        assert config.model.encoder_dim == 256
        assert config.model.hidden_dim == 128
        assert config.model.num_classes == 5
        assert config.training.batch_size == 16
        assert config.training.epochs == 10
        assert config.wandb.project == "test-project"
        assert config.wandb.enabled is False
    finally:
        Path(temp_path).unlink()


def test_config_to_dict():
    """Test converting config to dictionary."""
    yaml_content = """
model:
  encoder_dim: 128
  hidden_dim: 64
  num_classes: 3
  dropout: 0.2

training:
  batch_size: 8
  epochs: 5
  learning_rate: 0.0001
  seed: 123

data:
  data_dir: "test/data"
  sample_rate: 8000

wandb:
  project: "test"
  entity: "team"
  enabled: true
"""

    with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        temp_path = f.name

    try:
        config = Config.from_yaml(temp_path)
        config_dict = config.to_dict()

        assert config_dict["model"]["encoder_dim"] == 128
        assert config_dict["training"]["learning_rate"] == 0.0001
        assert config_dict["wandb"]["entity"] == "team"
    finally:
        Path(temp_path).unlink()


def test_config_file_not_found():
    """Test error handling for missing config file."""
    with pytest.raises(FileNotFoundError):
        Config.from_yaml("nonexistent.yaml")
