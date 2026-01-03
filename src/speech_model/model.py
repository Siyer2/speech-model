"""Multilabel classifier for pre-computed embeddings."""

import torch
import torch.nn as nn

from .config import ModelConfig


class ClassificationHead(nn.Module):
    """Classification head for phonological process detection."""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float):
        """Initialize classification head.

        Args:
            input_dim: Input feature dimension (from encoder)
            hidden_dim: Hidden layer dimension
            num_classes: Number of classification targets
            dropout: Dropout probability
        """
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, input_dim)

        Returns:
            Logits of shape (batch, num_classes)
        """
        return self.classifier(x)


class SpeechClassifier(nn.Module):
    """Multilabel classifier for pre-computed embeddings."""

    def __init__(self, config: ModelConfig):
        """Initialize speech classifier.

        Args:
            config: Model configuration
        """
        super().__init__()
        self.classifier = ClassificationHead(
            config.encoder_dim,
            config.hidden_dim,
            config.num_classes,
            config.dropout,
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            embeddings: Pre-computed audio embeddings (batch, encoder_dim)

        Returns:
            Logits for multilabel classification (batch, num_classes)
        """
        return self.classifier(embeddings)
