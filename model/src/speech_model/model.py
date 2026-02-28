"""Pretrained model loading and downstream model setup."""

import torch.nn as nn
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

PRETRAINED_MODEL = "facebook/wav2vec2-lv-60-espeak-cv-ft"
HEAD_LR_MULTIPLIER = 10


def create_model(vocab_size: int) -> tuple[Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor]:
    """Load pretrained Wav2Vec2ForCTC and replace the head for a custom vocab.

    Replaces lm_head with a new Linear layer (Xavier init, zeros bias)
    and freezes the feature encoder (CNN layers).
    """
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(PRETRAINED_MODEL)
    model = Wav2Vec2ForCTC.from_pretrained(PRETRAINED_MODEL)

    model.lm_head = nn.Linear(model.config.hidden_size, vocab_size)
    nn.init.xavier_uniform_(model.lm_head.weight)
    nn.init.zeros_(model.lm_head.bias)

    model.freeze_feature_encoder()
    return model, feature_extractor


def get_param_groups(model: Wav2Vec2ForCTC, base_lr: float) -> list[dict]:
    """Build optimizer parameter groups with differential learning rates.

    The lm_head gets HEAD_LR_MULTIPLIER times the base learning rate.
    All other trainable parameters (transformer layers) get base_lr.
    """
    head_params = list(model.lm_head.parameters())
    head_param_ids = {id(p) for p in head_params}
    backbone_params = [
        p for p in model.parameters() if p.requires_grad and id(p) not in head_param_ids
    ]
    return [
        {"params": backbone_params, "lr": base_lr},
        {"params": head_params, "lr": base_lr * HEAD_LR_MULTIPLIER},
    ]
