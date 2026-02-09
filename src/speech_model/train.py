"""Training script for CTC-based audio-to-phonetic model."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import Config
from .dataset import PhoneticDataset, Vocab
from .loss import ctc_decode
from .metrics import cer
from .model import SimplePhoneticModel
from .wandb_utils import WandBLogger


def split_by_participant(
    df: pd.DataFrame, val_split: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data by participant (no participant in both train and val)."""
    participants = df["participant_id"].unique()
    np.random.seed(seed)
    np.random.shuffle(participants)
    split_idx = int(len(participants) * (1 - val_split))
    train_participants = set(participants[:split_idx])
    mask = df["participant_id"].isin(train_participants)
    return df[mask], df[~mask]


def collate_fn(
    batch: list,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str], torch.Tensor]:
    """Collate variable-length audio and targets for CTC."""
    audios, targets, texts, has_errors = zip(*batch, strict=False)

    # Pad audio
    audio_lengths = torch.tensor([a.size(0) for a in audios])
    audios_padded = pad_sequence(audios, batch_first=True)

    # Pad targets
    target_lengths = torch.tensor([t.size(0) for t in targets])
    targets_padded = pad_sequence(targets, batch_first=True)

    has_errors = torch.tensor(has_errors, dtype=torch.bool)

    return audios_padded, targets_padded, audio_lengths, target_lengths, list(texts), has_errors


def train_epoch(
    model: SimplePhoneticModel,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    wandb_logger: WandBLogger,
    global_step: int,
    log_every: int = 50,
) -> tuple[float, int]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    for batch_idx, (audios, targets, audio_lengths, target_lengths, _, _) in enumerate(
        tqdm(dataloader, desc="Training", leave=False)
    ):
        audios = audios.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(audios)  # (batch, time, vocab)

        # CTC expects (time, batch, vocab)
        log_probs = nn.functional.log_softmax(logits, dim=-1).transpose(0, 1)

        input_lengths = model.compute_output_lengths(audio_lengths)

        loss = criterion(log_probs, targets, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % log_every == 0:
            wandb_logger.log({"train/batch_loss": loss.item()}, step=global_step)
        global_step += 1

    return total_loss / len(dataloader), global_step


def validate_epoch(
    model: SimplePhoneticModel,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    vocab: Vocab,
    max_samples_to_log: int = 10,
) -> tuple[float, float, float, list[tuple[str, str, float]]]:
    """Validate and compute CER.

    Returns:
        Tuple of (avg_loss, avg_cer, avg_cer_errors, sample_preds) where
        avg_cer_errors is CER computed only on samples with error patterns,
        and sample_preds is a list of (pred, target, cer) tuples for logging.
    """
    model.eval()
    total_loss = 0.0
    total_cer = 0.0
    total_cer_errors = 0.0
    num_samples = 0
    num_error_samples = 0
    sample_preds: list[tuple[str, str, float]] = []

    with torch.no_grad():
        for audios, targets, audio_lengths, target_lengths, texts, has_errors in tqdm(
            dataloader, desc="Validation", leave=False
        ):
            audios = audios.to(device)
            targets = targets.to(device)

            logits = model(audios)
            log_probs = nn.functional.log_softmax(logits, dim=-1).transpose(0, 1)

            input_lengths = model.compute_output_lengths(audio_lengths)

            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            total_loss += loss.item()

            # Decode predictions and compute CER
            preds = logits.argmax(dim=-1)  # (batch, time)
            for i, (pred, target_text) in enumerate(zip(preds, texts, strict=False)):
                pred_text = ctc_decode(pred[: input_lengths[i]].tolist(), vocab)
                sample_cer = cer(pred_text, target_text)
                total_cer += sample_cer
                num_samples += 1

                if has_errors[i]:
                    total_cer_errors += sample_cer
                    num_error_samples += 1

                if len(sample_preds) < max_samples_to_log:
                    sample_preds.append((pred_text, target_text, sample_cer))

    avg_loss = total_loss / len(dataloader)
    avg_cer = total_cer / num_samples if num_samples > 0 else 0.0
    avg_cer_errors = total_cer_errors / num_error_samples if num_error_samples > 0 else 0.0
    return avg_loss, avg_cer, avg_cer_errors, sample_preds


def save_checkpoint(model: nn.Module, epoch: int, val_loss: float, checkpoint_dir: Path):
    """Save model checkpoint."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state_dict": model.state_dict(), "epoch": epoch, "val_loss": val_loss},
        checkpoint_dir / "best.pt",
    )


def main():
    config_path = Path(__file__).parent.parent.parent / "trains.yaml"
    config = Config.from_yaml(config_path)

    print(f"Loaded config from {config_path}")
    print(f"Seed: {config.training.seed}")

    wandb_logger = WandBLogger(config, config_path)
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")

    print(f"Using device: {device}")

    # Load data
    df = pd.read_parquet(config.data.parquet_path)
    print(f"Loaded {len(df)} utterances")

    # Build vocab and split data
    vocab = Vocab.from_texts(df["actual_phonetic"].dropna().tolist())
    print(f"Vocab size: {vocab.size} (including blank)")

    train_df, val_df = split_by_participant(df, config.training.val_split, config.training.seed)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")

    # Create datasets and loaders
    train_dataset = PhoneticDataset(
        train_df, vocab, config.data.audio_base_path, config.data.sample_rate
    )
    val_dataset = PhoneticDataset(
        val_df, vocab, config.data.audio_base_path, config.data.sample_rate
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Create model
    model = SimplePhoneticModel(vocab.size).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    # Training loop
    best_loss = float("inf")
    patience = 0
    checkpoint_dir = Path(config.data.checkpoint_dir)
    global_step = 0

    for epoch in range(config.training.epochs):
        print(f"\nEpoch {epoch + 1}/{config.training.epochs}")

        train_loss, global_step = train_epoch(
            model, train_loader, criterion, optimizer, device, wandb_logger, global_step
        )
        val_loss, val_cer, val_cer_errors, sample_preds = validate_epoch(
            model, val_loader, criterion, device, vocab
        )

        print(
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val CER: {val_cer:.4f} | Val CER (errors): {val_cer_errors:.4f}"
        )

        wandb_logger.log(
            {
                "train/loss": train_loss,
                "val/loss": val_loss,
                "val/cer": val_cer,
                "val/cer_errors": val_cer_errors,
            },
            step=global_step,
        )
        wandb_logger.log_predictions(sample_preds, step=global_step)

        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            patience = 0
            save_checkpoint(model, epoch, val_loss, checkpoint_dir)
            print(f"Saved best model (loss={best_loss:.4f})")
        else:
            patience += 1
            if patience >= config.training.early_stopping_patience:
                print(f"Early stopping after {epoch + 1} epochs")
                break

    print("\nTraining complete!")
    wandb_logger.finish()


if __name__ == "__main__":
    main()
