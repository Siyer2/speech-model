"""Multilabel classifier for pre-computed embeddings."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from .config import ModelConfig


class GradientReversalFunction(Function):
    """Gradient Reversal Layer for Domain-Adversarial training.

    Forward pass: identity function
    Backward pass: negates the gradient and scales by lambda
    """

    @staticmethod
    def forward(ctx, x, lambda_val):
        ctx.lambda_val = lambda_val
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_val * grad_output, None


class GradientReversalLayer(nn.Module):
    """Wrapper for gradient reversal function."""

    def __init__(self, lambda_val: float = 1.0):
        super().__init__()
        self.lambda_val = lambda_val

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_val)

    def set_lambda(self, lambda_val: float):
        self.lambda_val = lambda_val


class ParticipantClassifier(nn.Module):
    """Classifier for predicting participant ID (used for adversarial training)."""

    def __init__(self, input_dim: int, num_participants: int, hidden_dim: int = 128):
        super().__init__()
        self.grl = GradientReversalLayer()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_participants)
        )

    def forward(self, x, lambda_val: float = 1.0):
        self.grl.set_lambda(lambda_val)
        x = self.grl(x)
        return self.classifier(x)


class CrossAttentionFusion(nn.Module):
    """Cross-attention module for fusing audio and phonetic features.

    Audio features attend to phonetic features to learn the comparison
    between what was said vs what should have been said.
    """

    def __init__(self, audio_dim: int, phonetic_dim: int, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Project audio to query
        self.q_proj = nn.Linear(audio_dim, hidden_dim)
        # Project phonetic to key/value
        self.k_proj = nn.Linear(phonetic_dim, hidden_dim)
        self.v_proj = nn.Linear(phonetic_dim, hidden_dim)

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

        # Final projection to combine
        self.output_dim = hidden_dim

    def forward(self, audio: torch.Tensor, phonetic_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio: Audio embedding (batch, audio_dim)
            phonetic_seq: Phonetic sequence embeddings (batch, seq_len, phonetic_dim)

        Returns:
            Fused features (batch, hidden_dim)
        """
        batch_size = audio.size(0)

        # Audio as query (batch, 1, hidden_dim) -> attend to phonetic sequence
        q = self.q_proj(audio).unsqueeze(1)  # (batch, 1, hidden_dim)
        k = self.k_proj(phonetic_seq)  # (batch, seq_len, hidden_dim)
        v = self.v_proj(phonetic_seq)  # (batch, seq_len, hidden_dim)

        # Reshape for multi-head attention
        q = q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = (attn @ v).transpose(1, 2).contiguous().view(batch_size, 1, -1)
        out = self.out_proj(out).squeeze(1)  # (batch, hidden_dim)
        out = self.norm(out)

        return out


class BidirectionalCrossAttention(nn.Module):
    """Bidirectional cross-attention for comparing audio and phonetic features.

    This module implements:
    1. Audio attends to phonetic sequence (what was said attending to what should be said)
    2. Phonetic sequence attends to audio (what should be said attending to what was said)
    3. Fuses both directions for rich comparison features

    This is particularly useful for error detection where we need to identify
    mismatches between actual and expected pronunciations.
    """

    def __init__(self, audio_dim: int, phonetic_dim: int, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Project audio and phonetic to same dimension
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.phonetic_proj = nn.Linear(phonetic_dim, hidden_dim)

        # Audio -> Phonetic attention (Q from audio, K/V from phonetic)
        self.a2p_q = nn.Linear(hidden_dim, hidden_dim)
        self.a2p_k = nn.Linear(hidden_dim, hidden_dim)
        self.a2p_v = nn.Linear(hidden_dim, hidden_dim)
        self.a2p_out = nn.Linear(hidden_dim, hidden_dim)

        # Phonetic -> Audio attention (Q from phonetic, K/V from audio)
        self.p2a_q = nn.Linear(hidden_dim, hidden_dim)
        self.p2a_k = nn.Linear(hidden_dim, hidden_dim)
        self.p2a_v = nn.Linear(hidden_dim, hidden_dim)
        self.p2a_out = nn.Linear(hidden_dim, hidden_dim)

        # Layer norms
        self.norm_a2p = nn.LayerNorm(hidden_dim)
        self.norm_p2a = nn.LayerNorm(hidden_dim)

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # audio_proj + a2p + p2a_pooled
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden_dim

    def _multi_head_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Apply multi-head attention."""
        batch_size = q.size(0)
        seq_len_q = q.size(1)
        seq_len_kv = k.size(1)

        # Reshape for multi-head
        q = q.view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply to values
        out = (attn @ v).transpose(1, 2).contiguous().view(batch_size, seq_len_q, -1)
        return out

    def forward(self, audio: torch.Tensor, phonetic_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio: Audio embedding (batch, audio_dim)
            phonetic_seq: Phonetic sequence embeddings (batch, seq_len, phonetic_dim)

        Returns:
            Fused comparison features (batch, hidden_dim)
        """
        batch_size = audio.size(0)

        # Project to common dimension
        audio_h = self.audio_proj(audio).unsqueeze(1)  # (batch, 1, hidden_dim)
        phonetic_h = self.phonetic_proj(phonetic_seq)  # (batch, seq_len, hidden_dim)

        # Audio -> Phonetic attention: audio attends to phonetic sequence
        a2p_q = self.a2p_q(audio_h)  # (batch, 1, hidden_dim)
        a2p_k = self.a2p_k(phonetic_h)  # (batch, seq_len, hidden_dim)
        a2p_v = self.a2p_v(phonetic_h)  # (batch, seq_len, hidden_dim)
        a2p_out = self._multi_head_attention(a2p_q, a2p_k, a2p_v)  # (batch, 1, hidden_dim)
        a2p_out = self.a2p_out(a2p_out)
        a2p_out = self.norm_a2p(a2p_out + audio_h)  # Residual connection
        a2p_out = a2p_out.squeeze(1)  # (batch, hidden_dim)

        # Phonetic -> Audio attention: each phonetic position attends to audio
        p2a_q = self.p2a_q(phonetic_h)  # (batch, seq_len, hidden_dim)
        p2a_k = self.p2a_k(audio_h)  # (batch, 1, hidden_dim)
        p2a_v = self.p2a_v(audio_h)  # (batch, 1, hidden_dim)
        p2a_out = self._multi_head_attention(p2a_q, p2a_k, p2a_v)  # (batch, seq_len, hidden_dim)
        p2a_out = self.p2a_out(p2a_out)
        p2a_out = self.norm_p2a(p2a_out + phonetic_h)  # Residual connection
        p2a_pooled = p2a_out.mean(dim=1)  # Pool over sequence (batch, hidden_dim)

        # Fuse all representations
        audio_proj = audio_h.squeeze(1)  # (batch, hidden_dim)
        fused = torch.cat([audio_proj, a2p_out, p2a_pooled], dim=-1)  # (batch, hidden_dim * 3)
        output = self.fusion(fused)  # (batch, hidden_dim)

        return output


class LabelWiseAttention(nn.Module):
    """Label-wise attention (Query2Label style) for multilabel classification.

    Each label has a learnable query that attends to the input features,
    allowing different labels to focus on different aspects of the input.
    """

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 256, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Learnable label queries - one per class
        self.label_queries = nn.Parameter(torch.randn(num_classes, hidden_dim))

        # Project input features to key/value
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

        # Output projection per label
        self.output_proj = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features (batch, input_dim)

        Returns:
            Logits per class (batch, num_classes)
        """
        batch_size = x.size(0)

        # Project input
        x = self.input_proj(x)  # (batch, hidden_dim)
        x = x.unsqueeze(1)  # (batch, 1, hidden_dim) - single "token" to attend to

        # Label queries as Q, input as K and V
        q = self.label_queries.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, num_classes, hidden_dim)
        k = self.k_proj(x)  # (batch, 1, hidden_dim)
        v = self.v_proj(x)  # (batch, 1, hidden_dim)

        # Reshape for multi-head attention
        q = q.view(batch_size, self.num_classes, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention: each label query attends to input
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Get label-specific representations
        out = (attn @ v).transpose(1, 2).contiguous().view(batch_size, self.num_classes, -1)
        out = self.norm(out)

        # Project each label representation to logit
        logits = self.output_proj(out).squeeze(-1)  # (batch, num_classes)

        return logits


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
        self.embed_dim: int = embed_dim

    def forward(self, char_ids: torch.Tensor, return_sequence: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Encode phonetic sequence.

        Args:
            char_ids: Character IDs (batch, seq_len)
            return_sequence: If True, also return full sequence embeddings

        Returns:
            If return_sequence=False: Encoding (batch, hidden_dim)
            If return_sequence=True: (pooled (batch, hidden_dim), sequence (batch, seq_len, hidden_dim))
        """
        emb = self.embed(char_ids)
        outputs, (h, _) = self.lstm(emb)
        # Concatenate forward and backward final hidden states
        pooled = torch.cat([h[0], h[1]], dim=-1)

        if return_sequence:
            return pooled, outputs  # outputs is (batch, seq_len, hidden_dim)
        return pooled


class SpeechClassifier(nn.Module):
    """Multilabel classifier for pre-computed embeddings.

    Supports multiple architecture modes:
    - "mlp": Simple concatenation of audio + phonetic, then MLP (baseline)
    - "cross_attention": Cross-attention fusion between audio and phonetic features
    - "bidirectional_attention": Bidirectional cross-attention (audio<->phonetic)
    - "label_attention": Label-wise attention (Query2Label) for multilabel
    - "cross_label_attention": Cross-attention fusion + label-wise attention
    """

    def __init__(self, config: ModelConfig):
        """Initialize speech classifier.

        Args:
            config: Model configuration
        """
        super().__init__()
        # phonetic_mode: "none" or "target_only"
        self.phonetic_mode: str = getattr(config, "phonetic_mode", "none")
        phonetic_dim = getattr(config, "phonetic_dim", 128)

        # architecture: "mlp", "cross_attention", "bidirectional_attention", "label_attention", "cross_label_attention"
        self.architecture: str = getattr(config, "architecture", "mlp")

        if self.phonetic_mode == "target_only":
            self.target_encoder = PhoneticEncoder(hidden_dim=phonetic_dim)

        # Setup architecture-specific components
        if self.architecture == "cross_attention":
            # Cross-attention fusion: audio attends to phonetic sequence
            self.cross_attention = CrossAttentionFusion(
                audio_dim=config.encoder_dim,
                phonetic_dim=phonetic_dim,
                hidden_dim=config.hidden_dim,
                num_heads=4,
                dropout=config.dropout,
            )
            # Combine audio + cross-attended features
            input_dim = config.encoder_dim + config.hidden_dim
            self.classifier = ClassificationHead(
                input_dim,
                config.hidden_dim,
                config.num_classes,
                config.dropout,
                config.num_layers,
            )

        elif self.architecture == "bidirectional_attention":
            # Bidirectional cross-attention: audio<->phonetic
            self.bidirectional_attention = BidirectionalCrossAttention(
                audio_dim=config.encoder_dim,
                phonetic_dim=phonetic_dim,
                hidden_dim=config.hidden_dim,
                num_heads=4,
                dropout=config.dropout,
            )
            # Combine audio + bidirectional features
            input_dim = config.encoder_dim + config.hidden_dim
            self.classifier = ClassificationHead(
                input_dim,
                config.hidden_dim,
                config.num_classes,
                config.dropout,
                config.num_layers,
            )

        elif self.architecture == "label_attention":
            # Label-wise attention (Query2Label style)
            if self.phonetic_mode == "target_only":
                input_dim = config.encoder_dim + phonetic_dim
            else:
                input_dim = config.encoder_dim
            self.label_attention = LabelWiseAttention(
                input_dim=input_dim,
                num_classes=config.num_classes,
                hidden_dim=config.hidden_dim,
                num_heads=4,
                dropout=config.dropout,
            )

        elif self.architecture == "cross_label_attention":
            # Cross-attention + label-wise attention
            self.cross_attention = CrossAttentionFusion(
                audio_dim=config.encoder_dim,
                phonetic_dim=phonetic_dim,
                hidden_dim=config.hidden_dim,
                num_heads=4,
                dropout=config.dropout,
            )
            # Label attention on combined features
            input_dim = config.encoder_dim + config.hidden_dim
            self.label_attention = LabelWiseAttention(
                input_dim=input_dim,
                num_classes=config.num_classes,
                hidden_dim=config.hidden_dim,
                num_heads=4,
                dropout=config.dropout,
            )

        else:  # "mlp" - default baseline
            if self.phonetic_mode == "target_only":
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
        if self.architecture == "cross_attention":
            # Cross-attention requires phonetic sequence
            if target_phonetic is not None and self.phonetic_mode == "target_only":
                _, phonetic_seq = self.target_encoder(target_phonetic, return_sequence=True)
                cross_attended = self.cross_attention(embeddings, phonetic_seq)
                x = torch.cat([embeddings, cross_attended], dim=-1)
            else:
                # Fallback to just audio if no phonetic
                x = embeddings
            return self.classifier(x)

        elif self.architecture == "bidirectional_attention":
            # Bidirectional cross-attention
            if target_phonetic is not None and self.phonetic_mode == "target_only":
                _, phonetic_seq = self.target_encoder(target_phonetic, return_sequence=True)
                bidir_features = self.bidirectional_attention(embeddings, phonetic_seq)
                x = torch.cat([embeddings, bidir_features], dim=-1)
            else:
                x = embeddings
            return self.classifier(x)

        elif self.architecture == "label_attention":
            # Label-wise attention
            if self.phonetic_mode == "target_only" and target_phonetic is not None:
                target_enc = self.target_encoder(target_phonetic)
                x = torch.cat([embeddings, target_enc], dim=-1)
            else:
                x = embeddings
            return self.label_attention(x)

        elif self.architecture == "cross_label_attention":
            # Cross-attention + label attention
            if target_phonetic is not None and self.phonetic_mode == "target_only":
                _, phonetic_seq = self.target_encoder(target_phonetic, return_sequence=True)
                cross_attended = self.cross_attention(embeddings, phonetic_seq)
                x = torch.cat([embeddings, cross_attended], dim=-1)
            else:
                x = embeddings
            return self.label_attention(x)

        else:  # "mlp" - default
            if self.phonetic_mode == "target_only" and target_phonetic is not None:
                target_enc = self.target_encoder(target_phonetic)
                x = torch.cat([embeddings, target_enc], dim=-1)
            else:
                x = embeddings
            return self.classifier(x)

    def get_features(
        self,
        embeddings: torch.Tensor,
        target_phonetic: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Get intermediate features before classification (for adversarial training).

        Args:
            embeddings: Pre-computed audio embeddings (batch, encoder_dim)
            target_phonetic: Target phonetic char IDs (batch, seq_len) or None

        Returns:
            Features tensor (batch, feature_dim)
        """
        if self.phonetic_mode == "target_only" and target_phonetic is not None:
            target_enc = self.target_encoder(target_phonetic)
            return torch.cat([embeddings, target_enc], dim=-1)
        return embeddings
