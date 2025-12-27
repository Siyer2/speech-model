"""Simple demo model: encoder + classification head."""

import torch
import torch.nn as nn

from .config import ModelConfig


class SimpleEncoder(nn.Module):
    """Simple encoder that simulates a pretrained audio encoder."""

    def __init__(self, input_dim: int, output_dim: int):
        """Initialize encoder.

        Args:
            input_dim: Input feature dimension
            output_dim: Output embedding dimension
        """
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)

        Returns:
            Encoded tensor of shape (batch, output_dim)
        """
        # Simple mean pooling over sequence
        x = self.encoder(x)
        x = x.mean(dim=1)  # (batch, seq_len, output_dim) -> (batch, output_dim)
        return x


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
    """Complete speech classification model."""

    def __init__(self, config: ModelConfig, input_dim: int = 80):
        """Initialize speech classifier.

        Args:
            config: Model configuration
            input_dim: Input feature dimension (e.g., mel spectrogram features)
        """
        super().__init__()
        self.encoder = SimpleEncoder(input_dim, config.encoder_dim)
        self.classifier = ClassificationHead(
            config.encoder_dim,
            config.hidden_dim,
            config.num_classes,
            config.dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)

        Returns:
            Logits of shape (batch, num_classes)
        """
        embeddings = self.encoder(x)
        logits = self.classifier(embeddings)
        return logits
