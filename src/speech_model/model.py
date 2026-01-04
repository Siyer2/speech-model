"""Multilabel classifier for pre-computed embeddings."""

import torch
import torch.nn as nn

from .config import ModelConfig


class ClassificationHead(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float, num_layers: int = 3
    ):
        """Initialize classification head.

        Args:
            input_dim: Input feature dimension (from encoder)
            hidden_dim: Hidden layer dimension (kept constant)
            num_classes: Number of classification targets
            dropout: Dropout probability
            num_layers: Number of hidden layers with residuals
        """
        super().__init__()

        # Input projection to hidden_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Residual blocks
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

        # Output projection
        self.output = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connections.

        Args:
            x: Input tensor of shape (batch, input_dim)

        Returns:
            Logits of shape (batch, num_classes)
        """
        # Project to hidden_dim
        x = self.input_proj(x)

        # Residual blocks
        for layer, norm in zip(self.layers, self.norms, strict=True):
            residual = x
            x = norm(x)
            x = torch.relu(x)
            x = self.dropout(x)
            x = layer(x)
            x = x + residual  # Residual connection

        # Final norm and output
        x = torch.relu(x)
        x = self.dropout(x)
        return self.output(x)


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
            config.num_layers,
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            embeddings: Pre-computed audio embeddings (batch, encoder_dim)

        Returns:
            Logits for multilabel classification (batch, num_classes)
        """
        return self.classifier(embeddings)
