import torch
import torch.nn as nn
import torchaudio.models
import torchaudio.pipelines
import torchaudio.transforms as t


class ConformerPhoneticModel(nn.Module):
    """Conformer model for CTC-based phonetic transcription.

    Combines convolutional feature encoding with Conformer blocks
    that use self-attention (global context) and depthwise convolution
    (local context).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_heads: int = 4,
        ff_dim: int = 1024,
        num_layers: int = 4,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
        backbone: str | None = None,
        freeze_backbone: bool = True,
        spec_augment: bool = False,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.spec_augment = spec_augment

        if backbone in ("hubert_base", "wavlm_base"):
            bundle = (
                torchaudio.pipelines.WAVLM_BASE
                if backbone == "wavlm_base"
                else torchaudio.pipelines.HUBERT_BASE
            )
            self.hubert = bundle.get_model()
            if freeze_backbone:
                for param in self.hubert.parameters():
                    param.requires_grad = False
            self.input_proj = nn.Linear(768, d_model)
            # HuBERT feature extractor conv specs for length computation
            self._hubert_conv_kernels = [10, 3, 3, 3, 3, 2, 2]
            self._hubert_conv_strides = [5, 2, 2, 2, 2, 2, 2]
        else:
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
            self.input_proj = nn.Linear(512, d_model)

        self.conformer = torchaudio.models.Conformer(
            input_dim=d_model,
            num_heads=num_heads,
            ffn_dim=ff_dim,
            num_layers=num_layers,
            depthwise_conv_kernel_size=conv_kernel_size,
            dropout=dropout,
        )
        self.head = nn.Linear(d_model, vocab_size)

        if spec_augment:
            self.spec_augment_transforms = nn.Sequential(
                t.FrequencyMasking(freq_mask_param=27),
                t.TimeMasking(time_mask_param=100),
            )

    def compute_output_lengths(self, audio_lengths: torch.Tensor) -> torch.Tensor:
        """Compute output sequence lengths from input audio lengths."""
        lengths = audio_lengths.clone()
        if self.backbone_name in ("hubert_base", "wavlm_base"):
            for k, s in zip(self._hubert_conv_kernels, self._hubert_conv_strides, strict=True):
                lengths = (lengths - k) // s + 1
        else:
            for module in self.feature_encoder:
                if isinstance(module, nn.Conv1d):
                    lengths = (lengths - module.kernel_size[0]) // module.stride[0] + 1
        return lengths.clamp(min=1)

    def forward(self, audio: torch.Tensor, audio_lengths: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            audio: Raw audio waveform (batch, samples) at 16kHz
            audio_lengths: Length of each audio in samples (batch,)

        Returns:
            Logits (batch, time, vocab_size)
        """
        if self.backbone_name in ("hubert_base", "wavlm_base"):
            # WavLM attention doesn't support attention masks, so only pass lengths for HuBERT
            lengths_arg = (
                audio_lengths.to(audio.device) if self.backbone_name == "hubert_base" else None
            )
            x, _ = self.hubert(audio, lengths_arg)  # (batch, time, 768)
        else:
            x = audio.unsqueeze(1)  # (batch, 1, samples)
            x = self.feature_encoder(x)  # (batch, 512, time)
            x = x.transpose(1, 2)  # (batch, time, 512)

        if self.spec_augment and self.training:
            # spec augment expects (batch, features, time)
            x = self.spec_augment_transforms(x.transpose(1, 2)).transpose(1, 2)

        x = self.input_proj(x)  # (batch, time, d_model)

        lengths = self.compute_output_lengths(audio_lengths.to(audio.device))
        x, _ = self.conformer(x, lengths)

        return self.head(x)  # (batch, time, vocab_size)
