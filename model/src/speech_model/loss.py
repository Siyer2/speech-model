"""CTC loss and decoding utilities."""

import torch.nn as nn

from .dataset import Vocab


def ctc_decode(ids: list[int], vocab: Vocab) -> str:
    """Decode CTC output by collapsing repeats and removing blanks."""
    collapsed = []
    prev = None
    for i in ids:
        if i != prev:
            collapsed.append(i)
            prev = i
    return vocab.decode([i for i in collapsed if i != 0])


def create_ctc_loss() -> nn.CTCLoss:
    """Create CTC loss with blank=0 and zero_infinity=True."""
    return nn.CTCLoss(blank=0, zero_infinity=True)
