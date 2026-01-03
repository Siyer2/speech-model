"""Evaluation metrics for multilabel classification."""

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


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
