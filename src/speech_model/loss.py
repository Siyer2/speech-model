"""CTC loss utilities."""

import math
from collections import defaultdict

import torch

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


def _log_sum_exp(a: float, b: float) -> float:
    """Numerically stable log(exp(a) + exp(b))."""
    if a == float("-inf"):
        return b
    if b == float("-inf"):
        return a
    if a >= b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


def ctc_prefix_beam_search(
    log_probs: torch.Tensor,
    vocab: Vocab,
    beam_width: int = 10,
) -> str:
    """CTC prefix beam search decoding.

    Args:
        log_probs: Log probabilities (time, vocab_size) for a single utterance.
        vocab: Vocabulary for decoding.
        beam_width: Number of beams to keep at each step.

    Returns:
        Decoded string from the best beam.
    """
    time, vocab_size = log_probs.shape
    neg_inf = float("-inf")

    # Each beam: prefix tuple -> (log_p_blank, log_p_non_blank)
    beams: dict[tuple[int, ...], tuple[float, float]] = {(): (0.0, neg_inf)}

    for t in range(time):
        new_beams: dict[tuple[int, ...], tuple[float, float]] = defaultdict(
            lambda: (neg_inf, neg_inf)
        )
        lp_t = log_probs[t]

        # Prune to top beam_width before expanding
        scored = [(pfx, _log_sum_exp(pb, pnb)) for pfx, (pb, pnb) in beams.items()]
        scored.sort(key=lambda x: x[1], reverse=True)
        active = scored[:beam_width]

        for prefix, _ in active:
            pb, pnb = beams[prefix]
            p_total = _log_sum_exp(pb, pnb)

            # Blank extension
            lp_blank = lp_t[0].item()
            old_pb, old_pnb = new_beams[prefix]
            new_beams[prefix] = (_log_sum_exp(old_pb, p_total + lp_blank), old_pnb)

            # Non-blank extensions
            for c in range(1, vocab_size):
                lp_c = lp_t[c].item()
                new_prefix = prefix + (c,)

                if len(prefix) > 0 and c == prefix[-1]:
                    # Same char as end of prefix: extend only via blank path
                    old_pb_n, old_pnb_n = new_beams[new_prefix]
                    new_beams[new_prefix] = (
                        old_pb_n,
                        _log_sum_exp(old_pnb_n, pb + lp_c),
                    )
                    # Repeat on same prefix via non-blank path
                    old_pb_s, old_pnb_s = new_beams[prefix]
                    new_beams[prefix] = (
                        old_pb_s,
                        _log_sum_exp(old_pnb_s, pnb + lp_c),
                    )
                else:
                    # Different char: extend prefix
                    old_pb_n, old_pnb_n = new_beams[new_prefix]
                    new_beams[new_prefix] = (
                        old_pb_n,
                        _log_sum_exp(old_pnb_n, p_total + lp_c),
                    )

        beams = dict(new_beams)

    best_prefix = max(beams, key=lambda p: _log_sum_exp(*beams[p]))
    return vocab.decode(list(best_prefix))
