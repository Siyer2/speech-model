"""Audio encoder abstraction for swappable audio encoders."""

from abc import ABC, abstractmethod

import soundfile as sf
import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model, Wav2Vec2Processor, WavLMModel


class AudioEncoder(ABC):
    """Abstract base class for audio encoders."""

    @abstractmethod
    def encode(self, audio_path: str) -> torch.Tensor:
        """Encode audio file to fixed-size embedding.

        Args:
            audio_path: Path to audio file

        Returns:
            Fixed-size embedding tensor
        """
        pass

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return embedding dimension."""
        pass


class Wav2Vec2Encoder(AudioEncoder):
    """Wav2Vec2 encoder implementation."""

    def __init__(self, model_name: str = "facebook/wav2vec2-base", device: str = "cpu"):
        """Initialize Wav2Vec2 encoder.

        Args:
            model_name: HuggingFace model identifier
            device: Device to run model on (cpu, cuda, mps)
        """
        self.device = device
        self.model = Wav2Vec2Model.from_pretrained(model_name).to(device)
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model.eval()  # Set to evaluation mode
        self._embedding_dim = self.model.config.hidden_size

    def encode(self, audio_path: str) -> torch.Tensor:
        """Encode audio file to fixed-size embedding.

        Args:
            audio_path: Path to audio file

        Returns:
            Mean-pooled embedding tensor of shape (embedding_dim,)
        """
        # Load audio using soundfile
        waveform, sample_rate = sf.read(audio_path)

        # Convert to mono if stereo
        if len(waveform.shape) > 1:
            waveform = waveform.mean(axis=1)

        # Resample to 16kHz if needed (Wav2Vec2 expects 16kHz)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform_tensor = torch.from_numpy(waveform).float()
            waveform = resampler(waveform_tensor).numpy()

        # Process audio
        inputs = self.processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Encode
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Get hidden states and mean pool over time dimension
            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()

        return embeddings.cpu()

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension."""
        return self._embedding_dim


class WavLMEncoder(AudioEncoder):
    """WavLM encoder implementation."""

    def __init__(self, model_name: str = "microsoft/wavlm-large", device: str = "cpu"):
        """Initialize WavLM encoder.

        Args:
            model_name: HuggingFace model identifier
            device: Device to run model on (cpu, cuda, mps)
        """
        self.device = device
        self.model = WavLMModel.from_pretrained(model_name).to(device)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model.eval()  # Set to evaluation mode
        self._embedding_dim = self.model.config.hidden_size

    def encode(self, audio_path: str) -> torch.Tensor:
        """Encode audio file to fixed-size embedding.

        Args:
            audio_path: Path to audio file

        Returns:
            Mean-pooled embedding tensor of shape (embedding_dim,)
        """
        # Load audio using soundfile
        waveform, sample_rate = sf.read(audio_path)

        # Convert to mono if stereo
        if len(waveform.shape) > 1:
            waveform = waveform.mean(axis=1)

        # Resample to 16kHz if needed (WavLM expects 16kHz)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform_tensor = torch.from_numpy(waveform).float()
            waveform = resampler(waveform_tensor).numpy()

        # Process audio
        inputs = self.processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Encode
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Get hidden states and mean pool over time dimension
            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()

        return embeddings.cpu()

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension."""
        return self._embedding_dim
