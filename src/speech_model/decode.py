"""CTC prefix beam search decoding for phoneme transcription."""

import math
from collections import defaultdict

import torch

from .dataset import Vocab

BLANK_IDX = 0


def beam_search_decode(
    log_probs: torch.Tensor,
    vocab: Vocab,
    beam_width: int = 10,
) -> str:
    """Decode a single utterance via CTC prefix beam search.

    Args:
        log_probs: (T, V) log-softmax probabilities.
        vocab: The phoneme vocabulary.
        beam_width: Number of beams to keep at each timestep.

    Returns:
        Decoded phoneme string.
    """
    n_frames, n_vocab = log_probs.shape
    lp = log_probs.detach().cpu()

    neg_inf = float("-inf")
    beams: dict[tuple[int, ...], list[float]] = {(): [0.0, neg_inf]}

    for t in range(n_frames):
        next_beams: dict[tuple[int, ...], list[float]] = defaultdict(lambda: [neg_inf, neg_inf])
        frame = lp[t]

        for prefix, (pb, pnb) in beams.items():
            p_total = _log_add(pb, pnb)

            # Extend with blank — stays at same prefix
            s = next_beams[prefix]
            s[0] = _log_add(s[0], p_total + frame[BLANK_IDX].item())

            for c in range(1, n_vocab):
                lp_c = frame[c].item()

                if prefix and c == prefix[-1]:
                    # Same label as last: only extend from blank ending
                    s_same = next_beams[prefix]
                    s_same[1] = _log_add(s_same[1], pb + lp_c)
                    # Extend prefix (non-blank ending adds new token)
                    new_prefix = prefix + (c,)
                    s_new = next_beams[new_prefix]
                    s_new[1] = _log_add(s_new[1], pnb + lp_c)
                else:
                    new_prefix = prefix + (c,)
                    s_new = next_beams[new_prefix]
                    s_new[1] = _log_add(s_new[1], p_total + lp_c)

        # Prune to top-k
        beams = dict(
            sorted(next_beams.items(), key=lambda x: _log_add(x[1][0], x[1][1]), reverse=True)[
                :beam_width
            ]
        )

    if not beams:
        return ""

    best_prefix = max(beams, key=lambda k: _log_add(beams[k][0], beams[k][1]))
    return vocab.decode(list(best_prefix))


def _log_add(a: float, b: float) -> float:
    """Numerically stable log(exp(a) + exp(b))."""
    if a == float("-inf"):
        return b
    if b == float("-inf"):
        return a
    mx = max(a, b)
    return mx + math.log1p(math.exp(-abs(a - b)))
