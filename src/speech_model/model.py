import torch
import torch.nn as nn
from transformers import Wav2Vec2Model


class SimplePhoneticModel(nn.Module):
    """Wav2Vec2 encoder with linear head for CTC-based phonetic transcription."""

    def __init__(self, vocab_size: int, encoder_name: str = "facebook/wav2vec2-base"):
        """Initialize model.

        Args:
            vocab_size: Number of output classes (including CTC blank at index 0)
            encoder_name: HuggingFace model name for the encoder
        """
        super().__init__()
        self.encoder = Wav2Vec2Model.from_pretrained(encoder_name)
        self.encoder.freeze_feature_encoder()
        hidden_size = self.encoder.config.hidden_size  # 768 for base, 1024 for large
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            audio: Raw audio waveform (batch, samples) at 16kHz

        Returns:
            Logits (batch, time, vocab_size)
        """
        outputs = self.encoder(audio)
        hidden_states = outputs.last_hidden_state  # (batch, time, hidden)
        return self.head(hidden_states)
