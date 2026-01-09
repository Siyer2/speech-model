"""Evaluation metrics for multilabel classification."""

from collections import defaultdict

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def compute_participant_level_metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
    participant_ids: list[str],
    class_names: list[str],
    threshold: float = 0.3,
    aggregation: str = "mean",
) -> dict:
    """Compute participant-level aggregated metrics.

    Aggregates utterance-level predictions to participant level and computes metrics.
    This reflects the actual clinical use case where we assess a whole participant.

    Args:
        probabilities: Predicted probabilities (n_samples, n_classes)
        targets: Ground truth labels (n_samples, n_classes)
        participant_ids: List of participant IDs for each sample
        class_names: List of error pattern names
        threshold: Threshold for binary prediction after aggregation
        aggregation: Aggregation method - "mean", "max", or "any" (proportion > 0)

    Returns:
        Dictionary with participant-level f1_macro, per_class metrics, and counts
    """
    # Group by participant
    participant_probs = defaultdict(list)
    participant_targets = defaultdict(list)

    for i, pid in enumerate(participant_ids):
        participant_probs[pid].append(probabilities[i])
        participant_targets[pid].append(targets[i])

    # Aggregate per participant
    agg_probs = []
    agg_targets = []
    pids_ordered = []

    for pid in sorted(participant_probs.keys()):
        probs = np.array(participant_probs[pid])
        targs = np.array(participant_targets[pid])

        # Aggregate probabilities
        if aggregation == "mean":
            agg_prob = probs.mean(axis=0)
        elif aggregation == "max":
            agg_prob = probs.max(axis=0)
        elif aggregation == "any":
            # Proportion of utterances with prob > threshold
            agg_prob = (probs > threshold).mean(axis=0)
        else:
            agg_prob = probs.mean(axis=0)

        # Aggregate targets: if ANY utterance has the pattern, participant has it
        agg_target = targs.max(axis=0)

        agg_probs.append(agg_prob)
        agg_targets.append(agg_target)
        pids_ordered.append(pid)

    agg_probs = np.array(agg_probs)
    agg_targets = np.array(agg_targets)

    # Apply threshold for predictions
    agg_preds = (agg_probs > threshold).astype(float)

    # Compute metrics at participant level
    f1_macro = f1_score(agg_targets, agg_preds, average="macro", zero_division=0)
    precision_per_class = precision_score(agg_targets, agg_preds, average=None, zero_division=0)
    recall_per_class = recall_score(agg_targets, agg_preds, average=None, zero_division=0)

    # Per-class metrics
    per_class_metrics = {}
    for idx, class_name in enumerate(class_names):
        per_class_metrics[class_name] = {
            "precision": float(precision_per_class[idx]),
            "recall": float(recall_per_class[idx]),
            "support": int(agg_targets[:, idx].sum()),  # Number of participants with this pattern
        }

    return {
        "f1_macro": float(f1_macro),
        "per_class": per_class_metrics,
        "num_participants": len(pids_ordered),
        "aggregation": aggregation,
    }


def compute_metrics(predictions: np.ndarray, targets: np.ndarray, class_names: list[str]) -> dict:
    """Compute multilabel classification metrics.

    Args:
        predictions: Binary predictions (n_samples, n_classes)
        targets: Ground truth labels (n_samples, n_classes)
        class_names: List of error pattern names

    Returns:
        Dictionary with f1_macro, precision_per_class, recall_per_class
    """
    # Compute macro F1 (average across all classes)
    f1_macro = f1_score(targets, predictions, average="macro", zero_division=0)

    # Compute per-class precision and recall
    precision_per_class = precision_score(targets, predictions, average=None, zero_division=0)
    recall_per_class = recall_score(targets, predictions, average=None, zero_division=0)

    # Create per-class metrics dict
    per_class_metrics = {}
    for idx, class_name in enumerate(class_names):
        per_class_metrics[class_name] = {
            "precision": float(precision_per_class[idx]),
            "recall": float(recall_per_class[idx]),
        }

    return {
        "f1_macro": float(f1_macro),
        "per_class": per_class_metrics,
    }


def aggregate_fold_results(fold_results: list[dict]) -> dict:
    """Aggregate metrics across folds.

    Args:
        fold_results: List of metric dicts from each fold

    Returns:
        Dictionary with mean and std for each metric
    """
    # Extract f1_macro from each fold
    f1_scores = [result["f1_macro"] for result in fold_results]

    aggregated = {
        "f1_macro_mean": float(np.mean(f1_scores)),
        "f1_macro_std": float(np.std(f1_scores)),
        "fold_results": fold_results,
    }

    return aggregated
