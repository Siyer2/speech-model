"""Training script for fine-tuning Wav2Vec2ForCTC on phonetic transcription data."""

from __future__ import annotations

import os
import random
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

from .config import Config
from .dataset import PhoneticDataset, Vocab, load_target_words, normalize_for_cer
from .decode import beam_search_decode
from .loss import create_ctc_loss
from .metrics import cer
from .model import PRETRAINED_MODEL, create_model, get_param_groups
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


def collate_fn_w2v(
    batch: list[tuple[torch.Tensor, torch.Tensor, str, bool, str, str, str, str, str] | None],
    feature_extractor: Wav2Vec2FeatureExtractor,
) -> dict | None:
    """Collate for Wav2Vec2 training: normalize audio + pad targets."""
    filtered = [b for b in batch if b is not None]
    if not filtered:
        return None
    audios = [b[0].numpy() for b in filtered]
    targets = [b[1] for b in filtered]
    texts = [b[2] for b in filtered]
    has_errors = torch.tensor([b[3] for b in filtered], dtype=torch.bool)
    error_patterns = [b[4] for b in filtered]
    target_phonetics = [b[5] for b in filtered]
    utterance_ids = [b[6] for b in filtered]
    audio_paths = [b[7] for b in filtered]
    words = [b[8] for b in filtered]

    processed = feature_extractor(audios, sampling_rate=16000, return_tensors="pt", padding=True)

    target_lengths = torch.tensor([t.size(0) for t in targets], dtype=torch.long)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=0)

    return {
        "input_values": processed["input_values"],
        "attention_mask": processed["attention_mask"],
        "targets": targets_padded,
        "target_lengths": target_lengths,
        "texts": texts,
        "has_errors": has_errors,
        "error_patterns": error_patterns,
        "target_phonetics": target_phonetics,
        "utterance_ids": utterance_ids,
        "audio_paths": audio_paths,
        "words": words,
    }


def train_epoch(
    model: Wav2Vec2ForCTC,
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

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training", leave=False)):
        if batch is None:
            continue
        input_values = batch["input_values"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["targets"].to(device)
        target_lengths = batch["target_lengths"].to(device)

        optimizer.zero_grad()

        logits = model(input_values, attention_mask=attention_mask).logits
        # CTC expects (time, batch, vocab)
        log_probs = nn.functional.log_softmax(logits, dim=-1).transpose(0, 1)

        input_lengths = model._get_feat_extract_output_lengths(attention_mask.sum(dim=-1)).long()  # pyright: ignore[reportAttributeAccessIssue]

        loss = criterion(log_probs, targets, input_lengths, target_lengths)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % log_every == 0:
            wandb_logger.log({"train/batch_loss": loss.item()}, step=global_step)
        global_step += 1

    return total_loss / len(dataloader), global_step


def validate_epoch(
    model: Wav2Vec2ForCTC,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    vocab: Vocab,
    target_words: set[str] | None = None,
) -> tuple[float, float, float, float, list[tuple[str, str, float, str]], list[dict]]:
    """Validate and compute CER.

    Returns:
        Tuple of (avg_loss, avg_cer_target_word, avg_cer, avg_cer_errors,
        all_preds, instance_records) where avg_cer_target_word is CER computed
        only on samples whose word is in the target word set,
        avg_cer_errors is CER computed only on samples with error patterns,
        all_preds contains (pred, target, cer, error_patterns) for every sample,
        and instance_records contains per-instance dicts for parquet logging.
    """
    model.eval()
    total_loss = 0.0
    total_cer = 0.0
    total_cer_errors = 0.0
    total_cer_target_word = 0.0
    num_samples = 0
    num_error_samples = 0
    num_target_word_samples = 0
    all_preds: list[tuple[str, str, float, str]] = []
    instance_records: list[dict] = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", leave=False):
            if batch is None:
                continue
            input_values = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["targets"].to(device)
            target_lengths = batch["target_lengths"].to(device)
            texts = batch["texts"]
            has_errors = batch["has_errors"]
            error_patterns = batch["error_patterns"]
            utterance_ids = batch["utterance_ids"]
            audio_paths = batch["audio_paths"]
            words = batch["words"]

            logits = model(input_values, attention_mask=attention_mask).logits
            log_probs_ctc = nn.functional.log_softmax(logits, dim=-1)

            input_lengths = model._get_feat_extract_output_lengths(
                attention_mask.sum(dim=-1)
            ).long()  # pyright: ignore[reportAttributeAccessIssue]

            loss = criterion(log_probs_ctc.transpose(0, 1), targets, input_lengths, target_lengths)
            total_loss += loss.item()

            for i, target_text in enumerate(texts):
                seq_len = input_lengths[i]

                sample_log_probs = log_probs_ctc[i, :seq_len]  # (T, V)
                pred_text = beam_search_decode(sample_log_probs, vocab)

                norm_pred = normalize_for_cer(pred_text)
                norm_target = normalize_for_cer(target_text)

                sample_cer = cer(norm_pred, norm_target)
                total_cer += sample_cer
                num_samples += 1

                if has_errors[i]:
                    total_cer_errors += sample_cer
                    num_error_samples += 1

                if target_words is not None and words[i].lower() in target_words:
                    total_cer_target_word += sample_cer
                    num_target_word_samples += 1

                all_preds.append((norm_pred, norm_target, sample_cer, error_patterns[i]))
                instance_records.append(
                    {
                        "utterance_id": utterance_ids[i],
                        "actual_phonetic": norm_target,
                        "predicted_phonetic": norm_pred,
                        "audio_path": audio_paths[i],
                        "cer": sample_cer,
                    }
                )

    avg_loss = total_loss / len(dataloader)
    avg_cer_target_word = (
        total_cer_target_word / num_target_word_samples if num_target_word_samples > 0 else 0.0
    )
    avg_cer = total_cer / num_samples if num_samples > 0 else 0.0
    avg_cer_errors = total_cer_errors / num_error_samples if num_error_samples > 0 else 0.0
    return avg_loss, avg_cer_target_word, avg_cer, avg_cer_errors, all_preds, instance_records


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.ReduceLROnPlateau,
    epoch: int,
    val_cer: float,
    best_cer: float,
    global_step: int,
    checkpoint_dir: Path,
    leaf_path: str,
):
    """Save model checkpoint with full training state."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "val_cer": val_cer,
            "best_cer": best_cer,
            "global_step": global_step,
        },
        checkpoint_dir / leaf_path,
    )


def save_instance_parquet(records: list[dict], train_name: str) -> Path:
    """Save per-instance evaluation results to tmp/{train_name}.parquet."""
    project_root = Path(__file__).parent.parent.parent
    tmp_dir = project_root / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_path = tmp_dir / f"{train_name}.parquet"
    df = pd.DataFrame(records)
    df.to_parquet(out_path, index=False)
    print(f"Saved {len(records)} instance records to {out_path}")
    return out_path


def main():
    config_path = Path(__file__).parent.parent.parent / "trains.yaml"
    config = Config.from_yaml(config_path)

    eval_only = os.environ.get("EVAL_ONLY", "").strip() == "1"
    train_name = os.environ.get("EXPERIMENT_NAME", "eval")

    print(f"Loaded config from {config_path}")
    print(f"Seed: {config.training.seed}")
    if eval_only:
        print("*** EVAL-ONLY MODE ***")

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

    # Exclude McAllister dataset (outlier: multi-word, 4x duplicated, CER ~0.72)
    mcallister_mask = df["audio_path"].str.contains("McAllister", na=False)
    print(f"Excluding {mcallister_mask.sum()} McAllister utterances")
    df = df[~mcallister_mask].reset_index(drop=True)

    # Build vocab and split data
    vocab = Vocab.from_phones()
    print(f"Vocab size: {vocab.size} (including blank and unk)")

    train_df, val_df = split_by_participant(df, config.training.val_split, config.training.seed)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")

    val_dataset = PhoneticDataset(
        val_df, vocab, config.data.audio_base_path, config.data.sample_rate
    )

    # Load pretrained Wav2Vec2 and replace head for our vocab
    print(f"Loading pretrained model: {PRETRAINED_MODEL}")
    model, feature_extractor = create_model(vocab.size)

    model = model.to(device)  # type: ignore

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    collate_fn = partial(collate_fn_w2v, feature_extractor=feature_extractor)

    def worker_init_fn(worker_id: int):
        """Seed numpy/random per worker so audiomentations augmentations are deterministic."""
        seed = torch.initial_seed() % 2**32
        np.random.seed(seed)
        random.seed(seed)

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    criterion = create_ctc_loss()
    target_words = {w.lower() for w in load_target_words()}
    print(f"Target words ({len(target_words)}): {sorted(target_words)}")

    # Resume from checkpoint if specified
    if config.training.resume_checkpoint:
        ckpt_path = Path(config.training.resume_checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        print(f"Loading checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint (epoch {checkpoint.get('epoch', '?')})")

    # ---------- Eval-only mode ----------
    if eval_only:
        if not config.training.resume_checkpoint:
            raise ValueError("EVAL_ONLY=1 requires resume_checkpoint to be set in config")

        val_loss, val_cer_tw, val_cer, val_cer_errors, all_preds, instance_records = validate_epoch(
            model,
            val_loader,
            criterion,
            device,
            vocab,
            target_words=target_words,
        )
        print(
            f"Val Loss: {val_loss:.4f} | Val CER: {val_cer:.4f} | "
            f"Val CER (target words): {val_cer_tw:.4f} | Val CER (errors): {val_cer_errors:.4f}"
        )
        wandb_logger.log_predictions(all_preds, step=0)
        save_instance_parquet(instance_records, train_name)
        wandb_logger.finish()
        return

    # ---------- Training mode ----------
    train_dataset = PhoneticDataset(
        train_df,
        vocab,
        config.data.audio_base_path,
        config.data.sample_rate,
        train=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    param_groups = get_param_groups(model, config.training.learning_rate)
    optimizer = optim.AdamW(param_groups, weight_decay=config.training.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_cer = float("inf")
    patience_counter = 0
    checkpoint_dir = Path(config.data.checkpoint_dir)
    global_step = 0
    start_epoch = 0

    # Restore full training state if resuming
    if config.training.resume_checkpoint:
        assert checkpoint  # pyright: ignore[reportPossiblyUnboundVariable] — set in checkpoint loading above
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_cer = checkpoint.get("best_cer", float("inf"))
        global_step = checkpoint.get("global_step", 0)
        print(f"Resumed training at epoch {start_epoch}, best_cer={best_cer:.4f}")

    for epoch in range(start_epoch, config.training.epochs):
        print(f"\nEpoch {epoch + 1}/{config.training.epochs}")

        train_loss, global_step = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            wandb_logger,
            global_step,
        )
        val_loss, val_cer_tw, val_cer, val_cer_errors, all_preds, instance_records = validate_epoch(
            model,
            val_loader,
            criterion,
            device,
            vocab,
            target_words=target_words,
        )

        print(
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val CER: {val_cer:.4f} | Val CER (target words): {val_cer_tw:.4f} | "
            f"Val CER (errors): {val_cer_errors:.4f}"
        )

        wandb_logger.log(
            {
                "train/loss": train_loss,
                "val/loss": val_loss,
                "val/cer": val_cer,
                "val/cer-target-word": val_cer_tw,
                "val/cer_errors": val_cer_errors,
                "lr": optimizer.param_groups[0]["lr"],
            },
            step=global_step,
        )
        wandb_logger.log_predictions(all_preds, step=global_step)

        scheduler.step(val_loss)

        if val_cer < best_cer:
            best_cer = val_cer
            patience_counter = 0

            leaf = f"{train_name}-best.pt" if train_name else "best.pt"

            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                val_cer,
                best_cer,
                global_step,
                checkpoint_dir,
                leaf_path=leaf,
            )
            print(f"Saved best model (cer={best_cer:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.training.early_stopping_patience:
                print(f"Early stopping after {epoch + 1} epochs")
                break

    print("\nTraining complete!")
    wandb_logger.finish()


if __name__ == "__main__":
    main()
