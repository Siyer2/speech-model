"""Segment audio files based on timestamps from .cha files."""

from pathlib import Path

import soundfile as sf


def segment_audio(
    wav_path: Path,
    utterances: list[dict],
    output_dir: Path,
    dataset_name: str,
    participant_id: str,
) -> list[str | None]:
    """Segment an audio file into individual utterances.

    Args:
        wav_path: Path to full audio file
        utterances: List of utterance dicts with start_time_ms and end_time_ms
        output_dir: Directory to save audio segments (dataset subdirectory will be created)
        dataset_name: Name of dataset (e.g., "Preston")
        participant_id: Participant ID (e.g., "P01")

    Returns:
        List of paths to created audio segments (relative to data/ directory).
        None values indicate failed/invalid segments.
    """
    if not wav_path.exists():
        print(f"Audio file not found: {wav_path}")
        return []

    try:
        # Read full audio file
        audio_data, sample_rate = sf.read(wav_path)

        # Create dataset-specific subdirectory
        dataset_output_dir = output_dir / dataset_name
        dataset_output_dir.mkdir(parents=True, exist_ok=True)
        segment_paths = []

        for idx, utterance in enumerate(utterances):
            segment_filename = f"{participant_id}_{idx}.wav"
            segment_path = dataset_output_dir / segment_filename
            relative_path = f"processed/audio_segments/{dataset_name}/{segment_filename}"

            # Skip if segment already exists
            if segment_path.exists() and segment_path.stat().st_size > 0:
                segment_paths.append(relative_path)
                print(f"{segment_path} exists, skipping processing")
                continue

            start_ms = utterance["start_time_ms"]
            end_ms = utterance["end_time_ms"]

            # Convert milliseconds to sample indices
            start_sample = int(start_ms * sample_rate / 1000)
            end_sample = int(end_ms * sample_rate / 1000)

            # Ensure we don't go out of bounds
            start_sample = max(0, start_sample)
            end_sample = min(len(audio_data), end_sample)

            if start_sample >= end_sample:
                print(
                    f"Invalid segment: {participant_id} utterance {idx} ({start_ms}ms - {end_ms}ms)"
                )
                segment_paths.append(None)
                continue

            # Extract segment
            segment = audio_data[start_sample:end_sample]

            # Save segment
            sf.write(segment_path, segment, sample_rate)
            segment_paths.append(relative_path)

        print(f"Segmented {len(segment_paths)} utterances from {wav_path.name}")
        return segment_paths

    except Exception as e:
        print(f"Failed to segment audio {wav_path}: {e}")
        return [None] * len(utterances)


def validate_audio_segment(segment_path: Path) -> bool:
    """Validate that an audio segment is readable and non-empty.

    Args:
        segment_path: Path to audio segment

    Returns:
        True if valid, False otherwise
    """
    try:
        if not segment_path.exists():
            return False

        audio_data, sample_rate = sf.read(segment_path)

        # Check that we have data
        if len(audio_data) == 0:
            return False

        # Check that sample rate is reasonable
        return not (sample_rate < 8000 or sample_rate > 192000)

    except Exception:
        return False
