import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Residual 1D convolution block."""

    def __init__(self, channels: int, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.norm = nn.BatchNorm1d(channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x))) + x


class SimplePhoneticModel(nn.Module):
    """1D convolutional model for CTC-based phonetic transcription."""

    def __init__(self, vocab_size: int):
        """Initialize model.

        Args:
            vocab_size: Number of output classes (including CTC blank at index 0)
        """
        super().__init__()
        self.feature_encoder = nn.Sequential(
            nn.Conv1d(1, 128, 10, 5),
            nn.GELU(),
            nn.Conv1d(128, 256, 8, 4),
            nn.GELU(),
            nn.Conv1d(256, 512, 4, 2),
            nn.GELU(),
            nn.Conv1d(512, 512, 4, 2),
            nn.GELU(),
        )  # total stride = 80
        self.context = nn.Sequential(*[ConvBlock(512) for _ in range(4)])
        self.head = nn.Linear(512, vocab_size)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            audio: Raw audio waveform (batch, samples) at 16kHz

        Returns:
            Logits (batch, time, vocab_size)
        """
        x = audio.unsqueeze(1)  # (batch, 1, samples)
        x = self.feature_encoder(x)  # (batch, 512, time)
        x = self.context(x)  # (batch, 512, time)
        x = x.transpose(1, 2)  # (batch, time, 512)
        return self.head(x)  # (batch, time, vocab_size)
