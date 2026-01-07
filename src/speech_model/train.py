"""Main training script with k-fold cross-validation."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import Config
from .dataset import SpeechErrorDataset, apply_label_merges
from .metrics import aggregate_fold_results, compute_metrics
from .model import SpeechClassifier
from .splits import create_participant_aware_folds
from .wandb_utils import WandBLogger


class FocalLoss(nn.Module):
    """Focal loss for multilabel classification with class imbalance."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha: float = alpha
        self.gamma: float = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        bce_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_term = (1 - p_t) ** self.gamma
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * focal_term * bce_loss).mean()


class SoftF1Loss(nn.Module):
    """Differentiable macro F1 loss - directly optimizes the F1 metric."""

    def __init__(self, epsilon: float = 1e-7):
        super().__init__()
        self.epsilon: float = epsilon

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)

        # Soft versions of TP, FP, FN per class
        tp = (probs * targets).sum(dim=0)
        fp = (probs * (1 - targets)).sum(dim=0)
        fn = ((1 - probs) * targets).sum(dim=0)

        # Per-class F1
        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)
        f1 = 2 * precision * recall / (precision + recall + self.epsilon)

        # Macro F1 (average across classes)
        macro_f1 = f1.mean()

        # Return 1 - F1 as loss (minimize loss = maximize F1)
        return 1 - macro_f1


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    phonetic_mode: str = "none",
) -> float:
    """Train for one epoch.

    Args:
        model: Model to train
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        phonetic_mode: "none" or "target_only"

    Returns:
        Average loss
    """
    model.train()
    total_loss = 0.0

    for embeddings, targets, metadata in tqdm(dataloader, desc="Training", leave=False):
        embeddings, targets = embeddings.to(device), targets.to(device)

        optimizer.zero_grad()

        target_phonetic = None
        if phonetic_mode == "target_only":
            target_phonetic = metadata["target_phonetic_ids"].to(device)

        outputs = model(embeddings, target_phonetic)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
    class_names: list[str],
    phonetic_mode: str = "none",
) -> dict:
    """Validate for one epoch.

    Args:
        model: Model to validate
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        threshold: Fixed threshold for binary prediction
        class_names: List of error pattern names
        phonetic_mode: "none" or "target_only"

    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for embeddings, targets, metadata in tqdm(dataloader, desc="Validation", leave=False):
            embeddings, targets = embeddings.to(device), targets.to(device)

            target_phonetic = None
            if phonetic_mode == "target_only":
                target_phonetic = metadata["target_phonetic_ids"].to(device)

            outputs = model(embeddings, target_phonetic)

            loss = criterion(outputs, targets)

            total_loss += loss.item()

            # Apply sigmoid and threshold for predictions
            probs = torch.sigmoid(outputs)
            predictions = (probs > threshold).float()

            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Compute metrics
    all_predictions = np.vstack(all_predictions)
    all_targets = np.vstack(all_targets)

    # Apply hardcoded label merging for metrics
    all_predictions, all_targets = apply_label_merges(all_predictions, all_targets, class_names)

    metrics = compute_metrics(all_predictions, all_targets, class_names)
    metrics["loss"] = total_loss / len(dataloader)

    return metrics


def save_checkpoint(
    model: nn.Module,
    config: Config,
    fold: int,
    epoch: int,
    metrics: dict,
    checkpoint_dir: Path,
):
    """Save model checkpoint.

    Args:
        model: Model to save
        config: Configuration object
        fold: Fold number
        epoch: Epoch number
        metrics: Validation metrics
        checkpoint_dir: Directory to save checkpoint
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"fold{fold}_best.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "fold": fold,
            "epoch": epoch,
            "metrics": metrics,
            "config": config.to_dict(),
        },
        checkpoint_path,
    )


def train_fold(
    fold_idx: int,
    train_dataset: SpeechErrorDataset,
    val_dataset: SpeechErrorDataset,
    config: Config,
    device: torch.device,
    wandb_logger: WandBLogger,
    global_step: int = 0,
) -> tuple[dict | None, int]:
    """Train a single fold.

    Args:
        fold_idx: Fold index
        train_dataset: Training dataset
        val_dataset: Validation dataset
        config: Configuration object
        device: Device to train on
        wandb_logger: WandB logger
        global_step: Global step counter for logging

    Returns:
        Tuple of (best validation metrics, updated global_step)
    """
    print(f"\n{'=' * 60}")
    print(f"Fold {fold_idx + 1}/{config.training.k_folds}")
    print(f"{'=' * 60}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=True,  # Add this
        persistent_workers=True,  # Add this
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=True,  # Add this
        persistent_workers=True,  # Add this
    )

    # Create model
    model = SpeechClassifier(config.model).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup training
    if config.training.loss_type == "focal":
        criterion = FocalLoss(alpha=config.training.focal_alpha, gamma=config.training.focal_gamma)
    elif config.training.loss_type == "soft_f1":
        criterion = SoftF1Loss()
    else:
        criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=10)

    # Training loop with early stopping
    best_val_f1 = 0.0
    best_val_metrics: dict | None = None
    patience_counter = 0

    phonetic_mode = config.model.phonetic_mode

    for epoch in range(config.training.epochs):
        print(f"\nEpoch {epoch + 1}/{config.training.epochs}")

        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, phonetic_mode=phonetic_mode
        )

        # Validate
        val_metrics = validate_epoch(
            model,
            val_loader,
            criterion,
            device,
            config.training.threshold,
            train_dataset.get_label_names(),
            phonetic_mode=phonetic_mode,
        )

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val F1-Macro: {val_metrics['f1_macro']:.4f}"
        )

        # Log to WandB
        wandb_logger.log(
            {
                f"fold{fold_idx}/train/loss": train_loss,
                f"fold{fold_idx}/val/loss": val_metrics["loss"],
                f"fold{fold_idx}/val/f1_macro": val_metrics["f1_macro"],
            },
            step=global_step,
        )
        global_step += 1

        # Save best model (initialize best_val_metrics on first epoch)
        if best_val_metrics is None or val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            best_val_metrics = val_metrics
            patience_counter = 0

            if config.training.save_best_only:
                checkpoint_dir = Path(config.data.checkpoint_dir)
                save_checkpoint(model, config, fold_idx, epoch, val_metrics, checkpoint_dir)
                print(f"✓ Saved best model (F1={best_val_f1:.4f})")
        else:
            patience_counter += 1

        # Step LR scheduler
        scheduler.step(val_metrics["f1_macro"])

        # Early stopping
        if patience_counter >= config.training.early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break

    print(f"\nBest validation F1-Macro: {best_val_f1:.4f}")
    return best_val_metrics, global_step


def main():
    """Main training function."""
    # Load configuration (auto-seeds)
    config_path = Path(__file__).parent.parent.parent / "trains.yaml"
    config = Config.from_yaml(config_path)

    print(f"Loaded configuration from {config_path}")
    print(f"Seed: {config.training.seed} (automatically set)")

    # Initialize WandB
    wandb_logger = WandBLogger(config, config_path)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    df = pd.read_parquet(config.data.parquet_path)
    print(f"\nLoaded {len(df)} utterances from {len(df['participant_id'].unique())} participants")

    # Get embeddings path
    encoder_name_safe = config.model.encoder_name.replace("/", "_")
    embeddings_path = Path(config.data.embeddings_dir) / f"{encoder_name_safe}_embeddings.pt"

    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"Embeddings not found at {embeddings_path}\n"
            f"Run: uv run python scripts/compute_embeddings.py"
        )

    print(f"Using embeddings: {embeddings_path}")

    # Load embeddings and filter dataframe to only include utterances with embeddings
    embeddings = torch.load(embeddings_path)
    df = df[df["utterance_id"].isin(embeddings.keys())].reset_index(drop=True)
    print(f"Using {len(df)} utterances with valid embeddings")

    # Create folds
    folds = create_participant_aware_folds(df, config.training.k_folds, config.training.seed)

    # Train each fold
    fold_results = []
    global_step = 0
    for fold_idx, (train_indices, val_indices) in enumerate(folds):
        # Create datasets - pass the pre-filtered dataframe
        train_dataset = SpeechErrorDataset(
            df.iloc[train_indices].reset_index(drop=True),
            str(embeddings_path),
            config.data.ontology_path,
            clean_labels=config.data.clean_labels,
            phonetic_mode=config.model.phonetic_mode,
        )
        val_dataset = SpeechErrorDataset(
            df.iloc[val_indices].reset_index(drop=True),
            str(embeddings_path),
            config.data.ontology_path,
            clean_labels=config.data.clean_labels,
            phonetic_mode=config.model.phonetic_mode,
        )

        # Train fold
        best_metrics, global_step = train_fold(
            fold_idx, train_dataset, val_dataset, config, device, wandb_logger, global_step
        )
        fold_results.append(best_metrics)

    # Aggregate results
    print(f"\n{'=' * 60}")
    print("Cross-Validation Results")
    print(f"{'=' * 60}")

    aggregated = aggregate_fold_results(fold_results)
    print(f"\nF1-Macro: {aggregated['f1_macro_mean']:.4f} ± {aggregated['f1_macro_std']:.4f}")

    # Aggregate per-class metrics across folds
    class_names = list(fold_results[0]["per_class"].keys())
    per_class_f1_aggregated = {}

    for class_name in class_names:
        # Collect metrics for this class across all folds
        precisions = []
        recalls = []

        for fold_result in fold_results:
            p = fold_result["per_class"][class_name]["precision"]
            r = fold_result["per_class"][class_name]["recall"]
            precisions.append(p)
            recalls.append(r)

        # Average precision and recall across folds
        avg_precision = np.mean(precisions)
        avg_recall = np.mean(recalls)

        # Compute F1 from averaged precision and recall
        if (avg_precision + avg_recall) > 0:
            avg_f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall)
        else:
            avg_f1 = 0.0

        per_class_f1_aggregated[class_name] = avg_f1

    # Log aggregated metrics to WandB
    log_dict = {
        "cv/f1_macro_mean": aggregated["f1_macro_mean"],
        "cv/f1_macro_std": aggregated["f1_macro_std"],
    }

    # Add per-class F1 scores (averaged across folds)
    for class_name, avg_f1 in per_class_f1_aggregated.items():
        log_dict[f"cv/per_class/{class_name}/f1"] = avg_f1

    wandb_logger.log(log_dict)

    print("\n✓ Training complete!")
    wandb_logger.finish()


if __name__ == "__main__":
    main()
