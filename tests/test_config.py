"""Test configuration loading."""

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from speech_model.config import Config


def test_config_from_yaml():
    """Test loading config from YAML file."""
    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create a test ontology
        ontology_content = """
error_patterns:
  pattern_1:
    description: "Test pattern 1"
  pattern_2:
    description: "Test pattern 2"
  pattern_3:
    description: "Test pattern 3"
  pattern_4:
    description: "Test pattern 4"
  pattern_5:
    description: "Test pattern 5"
"""
        ontology_path = tmpdir / "ontology.yaml"
        ontology_path.write_text(ontology_content)

        yaml_content = f"""
model:
  encoder_name: "test-encoder"
  encoder_dim: 256
  hidden_dim: 128
  dropout: 0.1

training:
  batch_size: 16
  epochs: 10
  learning_rate: 0.001
  seed: 42
  k_folds: 5
  threshold: 0.5
  num_workers: 2
  early_stopping_patience: 10
  save_best_only: true
  use_class_weights: false

data:
  parquet_path: "data/test.parquet"
  ontology_path: "{ontology_path}"
  embeddings_dir: "data/embeddings"
  sample_rate: 16000
  checkpoint_dir: "checkpoints"
  clean_labels: true

wandb:
  project: "test-project"
  entity: null
  enabled: false
"""

        config_path = tmpdir / "config.yaml"
        config_path.write_text(yaml_content)

        config = Config.from_yaml(config_path)

        assert config.model.encoder_dim == 256
        assert config.model.hidden_dim == 128
        assert config.model.num_classes == 5  # Derived from ontology
        assert config.training.batch_size == 16
        assert config.training.epochs == 10
        assert config.wandb.project == "test-project"
        assert config.wandb.enabled is False


def test_config_to_dict():
    """Test converting config to dictionary."""
    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create a test ontology
        ontology_content = """
error_patterns:
  pattern_1:
    description: "Test pattern 1"
  pattern_2:
    description: "Test pattern 2"
  pattern_3:
    description: "Test pattern 3"
"""
        ontology_path = tmpdir / "ontology.yaml"
        ontology_path.write_text(ontology_content)

        yaml_content = f"""
model:
  encoder_name: "test-encoder"
  encoder_dim: 128
  hidden_dim: 64
  dropout: 0.2

training:
  batch_size: 8
  epochs: 5
  learning_rate: 0.0001
  seed: 123
  k_folds: 3
  threshold: 0.5
  num_workers: 1
  early_stopping_patience: 5
  save_best_only: false
  use_class_weights: true

data:
  parquet_path: "test/data.parquet"
  ontology_path: "{ontology_path}"
  embeddings_dir: "test/embeddings"
  sample_rate: 8000
  checkpoint_dir: "test/checkpoints"
  clean_labels: false

wandb:
  project: "test"
  entity: "team"
  enabled: true
"""

        config_path = tmpdir / "config.yaml"
        config_path.write_text(yaml_content)

        config = Config.from_yaml(config_path)
        config_dict = config.to_dict()

        assert config_dict["model"]["encoder_dim"] == 128
        assert config_dict["model"]["num_classes"] == 3  # Derived from ontology
        assert config_dict["training"]["learning_rate"] == 0.0001
        assert config_dict["wandb"]["entity"] == "team"


def test_config_file_not_found():
    """Test error handling for missing config file."""
    with pytest.raises(FileNotFoundError):
        Config.from_yaml("nonexistent.yaml")
