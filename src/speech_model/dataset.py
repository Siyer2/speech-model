"""Dataset for loading pre-computed embeddings and labels."""

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset


class SpeechErrorDataset(Dataset[tuple[torch.Tensor, torch.Tensor, dict]]):
    """Dataset that loads pre-computed embeddings and labels."""

    def __init__(
        self,
        df: pd.DataFrame,
        embeddings_path: str,
        ontology_path: str,
    ):
        """Initialize dataset.

        Args:
            df: DataFrame with utterance data (already filtered and indexed)
            embeddings_path: Path to pre-computed embeddings .pt file
            ontology_path: Path to ontology.yaml with error patterns
        """
        self.df = df

        # Load pre-computed embeddings
        self.embeddings = torch.load(embeddings_path)

        # Load ontology and create label encoder
        with open(ontology_path) as f:
            ontology = yaml.safe_load(f)

        # Create mapping from pattern_id to index
        self.pattern_ids = sorted(ontology["error_patterns"].keys())
        self.pattern_to_idx = {pattern: idx for idx, pattern in enumerate(self.pattern_ids)}
        self.num_classes = len(self.pattern_ids)

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.df)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Get item at index.

        Args:
            index: Index into dataset

        Returns:
            Tuple of (embedding, labels, metadata)
            - embedding: Pre-computed audio embedding tensor (encoder_dim,)
            - labels: Binary multilabel tensor (num_classes,)
            - metadata: Dict with utterance_id, participant_id, etc.
        """
        row = self.df.iloc[index]
        utterance_id = row["utterance_id"]

        # Get pre-computed embedding
        embedding = self.embeddings[utterance_id]

        # Convert error_patterns to binary multilabel tensor
        error_patterns = row["error_patterns"]
        labels = torch.zeros(self.num_classes, dtype=torch.float32)

        if isinstance(error_patterns, (list | np.ndarray)):
            for pattern in error_patterns:
                if pattern in self.pattern_to_idx:
                    labels[self.pattern_to_idx[pattern]] = 1.0

        # Metadata for logging/debugging
        metadata = {
            "utterance_id": utterance_id,
            "participant_id": row["participant_id"],
            "dataset": row["dataset"],
            "word": row["word"],
        }

        return embedding, labels, metadata

    def get_label_names(self) -> list[str]:
        """Return list of error pattern names in order."""
        return self.pattern_ids
