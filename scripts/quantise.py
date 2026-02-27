"""Export a PyTorch checkpoint to ONNX and quantise to INT8."""

from __future__ import annotations

import sys
from pathlib import Path

import onnx
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

from speech_model.dataset import Vocab
from speech_model.model import create_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# ── Configuration ────────────────────────────────────────────────────────────
CHECKPOINT_PATH = "checkpoints/drop-mcallister-best.pt"
ONNX_FP32_PATH = "checkpoints/model-fp32.onnx"
ONNX_INT8_PATH = "checkpoints/model-int8.onnx"
# ─────────────────────────────────────────────────────────────────────────────


def size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def main():
    ckpt = PROJECT_ROOT / CHECKPOINT_PATH
    fp32_out = PROJECT_ROOT / ONNX_FP32_PATH
    int8_out = PROJECT_ROOT / ONNX_INT8_PATH

    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    # Load model
    vocab = Vocab.from_phones()
    model, _ = create_model(vocab.size)
    checkpoint = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Dummy input for tracing (1 second of 16kHz audio, batch=1)
    dummy_input = torch.randn(1, 16000)
    dummy_mask = torch.ones(1, 16000, dtype=torch.long)

    # Step 1: Export to ONNX (FP32)
    print("Exporting to ONNX (FP32)...")
    torch.onnx.export(
        model,
        (dummy_input, dummy_mask),
        str(fp32_out),
        input_names=["input_values", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_values": {0: "batch", 1: "time"},
            "attention_mask": {0: "batch", 1: "time"},
            "logits": {0: "batch", 1: "frames"},
        },
        opset_version=17,
    )

    # Merge external data into a single .onnx file (torch exports weights
    # as a separate .onnx.data file for large models).
    external_data = Path(str(fp32_out) + ".data")
    if external_data.exists():
        print("  Merging external weights into single file...")
        model_proto = onnx.load(str(fp32_out), load_external_data=True)
        onnx.save_model(model_proto, str(fp32_out), save_as_external_data=False)
        external_data.unlink()

    print(f"  Saved: {fp32_out}")

    # Step 2: Quantise to INT8 (MatMul only — Conv layers use weight_norm
    # which produces dynamic weights the quantiser can't handle)
    print("Quantising to INT8...")
    quantize_dynamic(
        str(fp32_out), str(int8_out), weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"]
    )
    print(f"  Saved: {int8_out}")

    # Size comparison
    fp32_size = size_mb(fp32_out)
    int8_size = size_mb(int8_out)
    reduction = (1 - int8_size / fp32_size) * 100

    print(f"\nFP32 ONNX:  {fp32_size:,.1f} MB")
    print(f"INT8 ONNX:  {int8_size:,.1f} MB")
    print(f"Reduction:  {fp32_size - int8_size:,.1f} MB ({reduction:.1f}%)")


if __name__ == "__main__":
    main()
