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


class PhoneticEncoder(nn.Module):
    """Character-level encoder for IPA phonetic transcriptions."""

    def __init__(self, embed_dim: int = 64, hidden_dim: int = 128, num_chars: int = 512):
        """Initialize phonetic encoder.

        Args:
            embed_dim: Character embedding dimension
            hidden_dim: Output hidden dimension
            num_chars: Vocabulary size (IPA has ~150 chars, padding for safety)
        """
        super().__init__()
        self.embed = nn.Embedding(num_chars, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim // 2, batch_first=True, bidirectional=True)
        self.output_dim: int = hidden_dim

    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:
        """Encode phonetic sequence.

        Args:
            char_ids: Character IDs (batch, seq_len)

        Returns:
            Encoding (batch, hidden_dim)
        """
        emb = self.embed(char_ids)
        _, (h, _) = self.lstm(emb)
        # Concatenate forward and backward final hidden states
        return torch.cat([h[0], h[1]], dim=-1)


class SpeechClassifier(nn.Module):
    """Multilabel classifier for pre-computed embeddings."""

    def __init__(self, config: ModelConfig):
        """Initialize speech classifier.

        Args:
            config: Model configuration
        """
        super().__init__()
        # phonetic_mode: "none" or "target_only"
        self.phonetic_mode: str = getattr(config, "phonetic_mode", "none")
        phonetic_dim = getattr(config, "phonetic_dim", 128)

        if self.phonetic_mode == "target_only":
            self.target_encoder = PhoneticEncoder(hidden_dim=phonetic_dim)
            input_dim = config.encoder_dim + phonetic_dim
        else:
            input_dim = config.encoder_dim

        self.classifier = ClassificationHead(
            input_dim,
            config.hidden_dim,
            config.num_classes,
            config.dropout,
            config.num_layers,
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        target_phonetic: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            embeddings: Pre-computed audio embeddings (batch, encoder_dim)
            target_phonetic: Target phonetic char IDs (batch, seq_len) or None

        Returns:
            Logits for multilabel classification (batch, num_classes)
        """
        if self.phonetic_mode == "target_only" and target_phonetic is not None:
            target_enc = self.target_encoder(target_phonetic)
            x = torch.cat([embeddings, target_enc], dim=-1)
        else:
            x = embeddings
        return self.classifier(x)
