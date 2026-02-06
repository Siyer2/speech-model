"""Simple metrics for phonetic transcription."""

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

if TYPE_CHECKING:
    from .dataset import Vocab
    from .wandb_utils import WandBLogger


def _edit_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return _edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row

    return prev_row[-1]


def cer(pred: str, target: str) -> float:
    """Character Error Rate: edit_distance / len(target)."""
    if len(target) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    return _edit_distance(pred, target) / len(target)


def wer(pred: str, target: str) -> float:
    """Word Error Rate: edit_distance on words / num_target_words."""
    pred_words = pred.split()
    target_words = target.split()
    if len(target_words) == 0:
        return 0.0 if len(pred_words) == 0 else 1.0
    return _edit_distance(" ".join(pred_words), " ".join(target_words)) / len(target_words)


def log_topk_confusion_heatmap(
    all_true_phones: list[int],
    all_pred_phones: list[int],
    vocab: "Vocab",
    wandb_logger: "WandBLogger",
    k: int = 15,
) -> None:
    """Create k×k confusion matrix heatmap for phones with most errors.

    Args:
        all_true_phones: List of ground truth phone indices
        all_pred_phones: List of predicted phone indices
        vocab: Vocabulary mapping indices to phones
        wandb_logger: WandB logger instance
        k: Number of most confused phones to display
    """

    # Build full confusion matrix (excluding blank at index 0)
    confusion_matrix_full = confusion_matrix(
        all_true_phones, all_pred_phones, labels=range(1, vocab.size)
    )

    # Find top-k phones with highest total errors
    # Error = all misclassifications (row sum - correct predictions)
    errors_per_phone = confusion_matrix_full.sum(axis=1) - np.diag(confusion_matrix_full)
    topk_indices = np.argsort(errors_per_phone)[-k:][::-1]

    # Extract k×k submatrix for these phones
    sub_confusion_matrix = confusion_matrix_full[np.ix_(topk_indices, topk_indices)]
    phone_labels = [vocab.idx_to_phone[i + 1] for i in topk_indices]

    # Normalize by row (show as percentages of true class)
    row_sums = sub_confusion_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    sub_confusion_matrix_normalized = sub_confusion_matrix / row_sums

    # Create heatmap with matplotlib/seaborn
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        sub_confusion_matrix_normalized,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        xticklabels=phone_labels,
        yticklabels=phone_labels,
        cbar_kws={"label": "Proportion"},
        vmin=0,
        vmax=1,
        ax=ax,
    )
    ax.set_xlabel("Predicted Phone")
    ax.set_ylabel("True Phone")
    ax.set_title(f"Confusion Matrix - Top {k} Most Confused Phones (Row-Normalized)")
    plt.tight_layout()

    # Log to WandB
    if wandb_logger.enabled:
        import wandb

        wandb_logger.log({"confusion/topk_heatmap": wandb.Image(fig)})

    plt.close(fig)
