"""Participant-aware k-fold splits for cross-validation."""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def create_participant_aware_folds(
    df: pd.DataFrame, k_folds: int, seed: int
) -> list[tuple[list[int], list[int]]]:
    """Create stratified k-fold splits ensuring no participant leakage.

    Args:
        df: DataFrame with columns [participant_id, error_patterns]
        k_folds: Number of folds
        seed: Random seed for reproducibility

    Returns:
        List of (train_indices, val_indices) tuples
    """
    # Get participant IDs as groups
    groups = df["participant_id"].values

    # Create stratification labels based on most frequent error pattern per participant
    # For each participant, find their most common error pattern
    participant_strata = []

    for _, row in df.iterrows():
        error_patterns = row["error_patterns"]

        # Handle different error_patterns types
        if error_patterns is None:
            participant_strata.append("no_error")
        elif isinstance(error_patterns, (np.ndarray | list)):
            if len(error_patterns) == 0:
                participant_strata.append("no_error")
            else:
                participant_strata.append(error_patterns[0])
        else:
            participant_strata.append("no_error")

    # Convert to numeric labels for sklearn
    unique_strata = sorted(set(participant_strata))
    strata_to_idx = {s: i for i, s in enumerate(unique_strata)}
    y = np.array([strata_to_idx[s] for s in participant_strata])

    # Create stratified group k-fold splitter
    sgkf = StratifiedGroupKFold(n_splits=k_folds, shuffle=True, random_state=seed)

    # Generate folds
    folds = []
    for train_idx, val_idx in sgkf.split(X=np.zeros(len(df)), y=y, groups=groups):
        folds.append((train_idx.tolist(), val_idx.tolist()))

    # Print fold statistics
    print(f"\nCreated {k_folds} participant-aware folds:")
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        train_participants = df.iloc[train_idx]["participant_id"].nunique()
        val_participants = df.iloc[val_idx]["participant_id"].nunique()
        print(
            f"  Fold {fold_idx + 1}: Train={len(train_idx)} samples "
            f"({train_participants} participants), "
            f"Val={len(val_idx)} samples ({val_participants} participants)"
        )

    # Verify no participant leakage
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        train_participants = set(df.iloc[train_idx]["participant_id"])
        val_participants = set(df.iloc[val_idx]["participant_id"])
        overlap = train_participants & val_participants
        if overlap:
            raise ValueError(f"Fold {fold_idx} has participant leakage: {overlap}")

    print("✓ No participant leakage detected\n")

    return folds
