"""Pre-compute audio embeddings for fast training iteration.

Usage:
    uv run python scripts/compute_embeddings.py
"""

import argparse
import json

# Add src to path for imports
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from speech_model.config import Config
from speech_model.encoders import Wav2Vec2Encoder


def compute_embeddings(config: Config):
    """Pre-compute and save audio embeddings.

    Args:
        config: Configuration object
    """
    print(f"Loading data from {config.data.parquet_path}")
    df = pd.read_parquet(config.data.parquet_path)
    print(f"Loaded {len(df)} utterances")

    # Create embeddings directory
    embeddings_dir = Path(config.data.embeddings_dir)
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    # Initialize encoder
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Initializing encoder: {config.model.encoder_name}")

    encoder = Wav2Vec2Encoder(model_name=config.model.encoder_name, device=device)
    print(f"Encoder embedding dimension: {encoder.embedding_dim}")

    # Compute embeddings
    embeddings_dict = {}
    failed_files = []

    print("\nComputing embeddings...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Encoding audio"):
        utterance_id = row["utterance_id"]
        audio_path = row["audio_path"]

        # Fix relative path (prepend data/ if needed)
        if not audio_path.startswith("data/"):
            audio_path = "data/" + audio_path

        try:
            # Check if audio file exists
            if not Path(audio_path).exists():
                failed_files.append({"utterance_id": utterance_id, "reason": "File not found"})
                continue

            # Encode audio
            embedding = encoder.encode(audio_path)
            embeddings_dict[utterance_id] = embedding

        except Exception as e:
            failed_files.append({"utterance_id": utterance_id, "reason": str(e)})

    # Save embeddings
    encoder_name_safe = config.model.encoder_name.replace("/", "_")
    embeddings_path = embeddings_dir / f"{encoder_name_safe}_embeddings.pt"
    metadata_path = embeddings_dir / f"{encoder_name_safe}_metadata.json"

    print(f"\nSaving embeddings to {embeddings_path}")
    torch.save(embeddings_dict, embeddings_path)

    # Save metadata
    metadata = {
        "encoder_name": config.model.encoder_name,
        "encoder_dim": encoder.embedding_dim,
        "num_samples": len(embeddings_dict),
        "num_failed": len(failed_files),
        "timestamp": datetime.now().isoformat(),
        "failed_files": failed_files[:10],  # Save first 10 failures
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved metadata to {metadata_path}")
    print(f"\nSuccessfully encoded: {len(embeddings_dict)} utterances")
    print(f"Failed: {len(failed_files)} utterances")

    if failed_files:
        print("\nFirst few failures:")
        for fail in failed_files[:5]:
            print(f"  - {fail['utterance_id']}: {fail['reason']}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Pre-compute audio embeddings")
    parser.add_argument(
        "--config",
        type=str,
        default="trains.yaml",
        help="Path to config file (default: trains.yaml)",
    )
    args = parser.parse_args()

    # Load config
    config = Config.from_yaml(args.config)

    # Compute embeddings
    compute_embeddings(config)

    print("\n✓ Embeddings computation complete!")


if __name__ == "__main__":
    main()
