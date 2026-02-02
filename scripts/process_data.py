import json
from pathlib import Path

import pandas as pd
from parse_cha import extract_participant_info, parse_cha_file, validate_cha_data
from segment_audio import segment_audio

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PARQUET_PATH = PROCESSED_DIR / "utterances.parquet"
AUDIO_SEGMENTS_DIR = PROCESSED_DIR / "audio_segments"

DATASET_NAMES = [
    "Preston",
    "McAllister",
    "PreKHistorySSD",
    "SuspectedSSD",
    "TDChildrenandAdults",
    # Cummings
    "PD01",
    "PD02",
    "PD06",
    "PD08",
    "PD10",
    "PD11",
    "PD13",
    "PD15",
    "PD16",
    "PD21",
    "PD23",
    "PD27",
    "PD28",
    "PD39",
    "PD54",
    "PD55",
    "PD59",
    "PD66",
    "PD68",
    "PD69",
    "PD71",
    "McAllister",
]


def process_dataset(dataset_name: str, raw_dir: Path, output_dir: Path) -> pd.DataFrame:
    """Process a dataset: parse .cha files and segment audio.

    Args:
        dataset_name: Name of dataset (e.g., "Preston")
        raw_dir: Directory containing raw .cha and .wav files
        output_dir: Directory to save processed data

    Returns:
        DataFrame with all processed utterances
    """
    print(f"Processing dataset: {dataset_name}")

    if not raw_dir.exists():
        print(f"Raw data directory not found: {raw_dir}")
        return pd.DataFrame()

    # Find all .cha files
    cha_files = sorted(raw_dir.glob("*.cha"))
    if not cha_files:
        print(f"No .cha files found in {raw_dir}")
        return pd.DataFrame()

    print(f"Found {len(cha_files)} participants in {dataset_name}")

    all_utterances = []
    audio_segments_dir = output_dir / "audio_segments"

    for cha_file in cha_files:
        participant_id = cha_file.stem
        wav_file = cha_file.with_suffix(".wav")

        try:
            # Parse .cha file
            cha_data = parse_cha_file(cha_file)

            if not validate_cha_data(cha_data):
                print(f"Invalid .cha data for {participant_id}")
                continue

            # Segment audio if available
            audio_paths = []
            if wav_file.exists():
                audio_paths = segment_audio(
                    wav_file,
                    cha_data["utterances"],
                    audio_segments_dir,
                    dataset_name,
                    participant_id,
                )
            else:
                print(f"Audio file not found for {participant_id}: {wav_file}")
                audio_paths = [None] * len(cha_data["utterances"])

            # Extract participant info
            participant_info = extract_participant_info(cha_data["metadata"])

            # Create rows for each utterance
            for idx, utterance in enumerate(cha_data["utterances"]):
                utterance_id = f"{dataset_name}_{participant_id}_{idx}"

                row = {
                    "utterance_id": utterance_id,
                    "dataset": dataset_name,
                    "participant_id": participant_id,
                    "audio_path": audio_paths[idx] if idx < len(audio_paths) else None,
                    "word": utterance["word"],
                    "target_phonetic": utterance["target_phonetic"],
                    "actual_phonetic": utterance["actual_phonetic"],
                    "start_time_ms": utterance["start_time_ms"],
                    "end_time_ms": utterance["end_time_ms"],
                    "cha_metadata": json.dumps({**cha_data["metadata"], **participant_info}),
                    "error_patterns": None,  # For future LLM labeling
                    "comment": utterance.get("comment"),
                }

                all_utterances.append(row)

        except Exception as e:
            print(f"Failed to process {participant_id}: {e}")
            continue

    # Create DataFrame
    df = pd.DataFrame(all_utterances)

    if not df.empty:
        print(
            f"Processed {len(df)} utterances from {len(cha_files)} participants in {dataset_name}"
        )
    else:
        print(f"No utterances processed for {dataset_name}")

    return df


def process_all_datasets() -> pd.DataFrame:
    """Process all datasets and combine into single DataFrame.

    Returns:
        Combined DataFrame with all utterances
    """

    # Load existing data if parquet exists
    existing_df = pd.read_parquet(PARQUET_PATH) if PARQUET_PATH.exists() else pd.DataFrame()
    existing_datasets = set(existing_df["dataset"].unique()) if not existing_df.empty else set()

    all_dfs = [existing_df] if not existing_df.empty else []

    for dataset_name in DATASET_NAMES:
        if dataset_name in existing_datasets:
            print(f"Skipping {dataset_name} (already in parquet)")
            continue

        dataset_raw_dir = RAW_DIR / dataset_name
        if dataset_raw_dir.exists():
            df = process_dataset(dataset_name, dataset_raw_dir, PROCESSED_DIR)
            if not df.empty:
                all_dfs.append(df)
        else:
            print(f"Dataset directory not found: {dataset_raw_dir}")

    if not all_dfs:
        print("No data processed from any dataset")
        return pd.DataFrame()

    # Combine all datasets
    combined_df = pd.concat(all_dfs, ignore_index=True)

    print("\n=== Processing Complete ===")
    print(f"Total utterances: {len(combined_df)}")
    print("Utterances by dataset:")
    for dataset_name in DATASET_NAMES:
        count = len(combined_df[combined_df["dataset"] == dataset_name])
        print(f"  {dataset_name}: {count}")

    return combined_df


def save_to_parquet(df: pd.DataFrame) -> None:
    """Save DataFrame to Parquet with compression.

    Args:
        df: DataFrame to save
        output_path: Path to save Parquet file
    """
    PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(
        PARQUET_PATH,
        engine="pyarrow",
        compression="snappy",
        index=False,
    )

    print(f"Saved {len(df)} utterances to {PARQUET_PATH}")

    size_mb = PARQUET_PATH.stat().st_size / (1024 * 1024)
    print(f"Parquet file size: {size_mb:.2f} MB")


if __name__ == "__main__":
    df = process_all_datasets()
    save_to_parquet(df)
