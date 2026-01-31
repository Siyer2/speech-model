"""CTC loss utilities."""

from .dataset import Vocab


def ctc_decode(ids: list[int], vocab: Vocab) -> str:
    """Decode CTC output by collapsing repeats and removing blanks.

    Args:
        ids: List of predicted token indices
        vocab: Vocabulary for decoding

    Returns:
        Decoded string
    """
    # Collapse consecutive duplicates
    collapsed = []
    prev = None
    for i in ids:
        if i != prev:
            collapsed.append(i)
            prev = i

    # Remove blanks (index 0) and decode
    return vocab.decode([i for i in collapsed if i != 0])
