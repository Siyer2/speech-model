"""Quantise a checkpoint to INT8 (dynamic) and save to a new file."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.ao.quantization
import torch.nn as nn

from speech_model.dataset import Vocab
from speech_model.model import create_model

# ── Configuration ────────────────────────────────────────────────────────────
CHECKPOINT_PATH = "checkpoints/drop-mcallister-best.pt"
QUANTISED_OUTPUT_PATH = "checkpoints/drop-mcallister-int8.pt"
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def get_file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def main():
    ckpt_path = PROJECT_ROOT / CHECKPOINT_PATH
    out_path = PROJECT_ROOT / QUANTISED_OUTPUT_PATH

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if out_path.exists():
        raise FileExistsError(
            f"Output already exists: {out_path} — delete it or change QUANTISED_OUTPUT_PATH"
        )

    # Load model on CPU (required for dynamic quantisation)
    vocab = Vocab.from_phones()
    model, _ = create_model(vocab.size)

    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    # Save FP32 model-only size for fair comparison (original checkpoint
    # includes optimizer/scheduler state which inflates it).
    tmp_path = PROJECT_ROOT / "checkpoints" / "_tmp_fp32_model_only.pt"
    torch.save({"model_state_dict": model.state_dict()}, tmp_path)
    model_only_size = get_file_size_mb(tmp_path)
    tmp_path.unlink()

    # Quantise all Linear layers to INT8
    print("Applying INT8 dynamic quantisation...")
    quantised_model = torch.ao.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8,
    )

    # Save quantised state_dict (keys differ from FP32 due to quantised
    # Linear modules, so we flag it so the loader knows to quantise the
    # architecture before loading).
    torch.save(
        {"model_state_dict": quantised_model.state_dict(), "quantised": True},
        out_path,
    )
    print(f"Saved quantised checkpoint: {out_path}")

    # ── Size comparison ──────────────────────────────────────────────────────
    original_size = get_file_size_mb(ckpt_path)
    quantised_size = get_file_size_mb(out_path)

    print(f"\nOriginal checkpoint (with optimizer): {original_size:,.1f} MB")
    print(f"FP32 model-only:                      {model_only_size:,.1f} MB")
    print(f"INT8 quantised:                        {quantised_size:,.1f} MB")
    print(
        f"Model size reduction:                  {model_only_size - quantised_size:,.1f} MB "
        f"({(1 - quantised_size / model_only_size) * 100:.1f}%)"
    )


if __name__ == "__main__":
    main()
