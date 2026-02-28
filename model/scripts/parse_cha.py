"""Parse .cha files to extract metadata and phonetic transcriptions."""

import re
from pathlib import Path


def parse_cha_file(filepath: Path) -> dict:
    """Parse a .cha file and extract all metadata and utterances.

    Args:
        filepath: Path to .cha file

    Returns:
        Dict containing:
            - metadata: dict of all @-header fields
            - utterances: list of dicts with word, target_phonetic, actual_phonetic, timestamps
    """
    with open(filepath, encoding="utf-8") as f:
        lines = f.readlines()

    metadata = {}
    utterances = []
    current_utterance = None

    for line in lines:
        # Remove control characters (some files have \x15)
        line = "".join(char for char in line if char.isprintable() or char in "\t\n")
        line = line.strip()

        # Parse metadata headers
        if line.startswith("@"):
            # Split on first colon to get key and value
            if ":" in line:
                key, value = line[1:].split(":", 1)
                metadata[key.strip()] = value.strip()
            else:
                # Some headers like @Begin, @End don't have values
                metadata[line[1:]] = True

        # Parse utterances
        elif line.startswith("*CHI:"):
            # Extract word and timestamp
            # Format: *CHI: word . timestamp
            match = re.match(r"\*CHI:\s+(.+?)\s+\.\s+(\d+)_(\d+)", line)
            if match:
                word = match.group(1)
                start_ms = int(match.group(2))
                end_ms = int(match.group(3))

                current_utterance = {
                    "word": word,
                    "start_time_ms": start_ms,
                    "end_time_ms": end_ms,
                    "target_phonetic": None,
                    "actual_phonetic": None,
                    "comment": None,
                }

        # Parse phonetic transcriptions
        elif (
            line.startswith("%xmod:") or line.startswith("%mod:")
        ) and current_utterance is not None:
            # Target phonetic (model/target pronunciation)
            current_utterance["target_phonetic"] = (
                line.replace("%xmod:", "").replace("%mod:", "").strip()
            )

        elif (
            line.startswith("%xpho:") or line.startswith("%pho:")
        ) and current_utterance is not None:
            # Actual phonetic pronunciation
            current_utterance["actual_phonetic"] = (
                line.replace("%xpho:", "").replace("%pho:", "").strip()
            )

            # Add to utterances list (xpho/pho is typically the last line)
            utterances.append(current_utterance)
            current_utterance = None

        # Parse optional comment
        elif line.startswith("%com:") and current_utterance is not None:
            current_utterance["comment"] = line.replace("%com:", "").strip()

    result = {"metadata": metadata, "utterances": utterances}

    return result


def validate_cha_data(data: dict) -> bool:
    """Validate parsed .cha data.

    Args:
        data: Parsed data from parse_cha_file

    Returns:
        True if valid, False otherwise
    """
    if "metadata" not in data or "utterances" not in data:
        return False

    # Check that we have some metadata
    if not data["metadata"]:
        return False

    # Check that utterances have required fields
    for utt in data["utterances"]:
        required_fields = [
            "word",
            "target_phonetic",
            "actual_phonetic",
            "start_time_ms",
            "end_time_ms",
        ]
        if not all(field in utt for field in required_fields):
            return False

        # Check that phonetic fields are not None
        if utt["target_phonetic"] is None or utt["actual_phonetic"] is None:
            return False

        # Check timestamp validity
        if utt["start_time_ms"] >= utt["end_time_ms"]:
            return False

    return True


def extract_participant_info(metadata: dict) -> dict:
    """Extract participant information from metadata.

    Args:
        metadata: Metadata dict from parse_cha_file

    Returns:
        Dict with extracted participant info (age, sex, language, etc.)
    """
    info = {}

    # Extract from @ID field
    # Format: @ID: lang|corpus|role|age|sex|group|SES|role|custom|custom|
    if "ID" in metadata:
        id_parts = metadata["ID"].split("|")
        if len(id_parts) >= 5:
            info["language"] = id_parts[0]
            info["corpus"] = id_parts[1]
            info["role"] = id_parts[2]
            info["age"] = id_parts[3]
            info["sex"] = id_parts[4]

    # Extract other useful fields
    if "Languages" in metadata:
        info["languages"] = metadata["Languages"]

    if "Date" in metadata:
        info["date"] = metadata["Date"]

    if "Media" in metadata:
        info["media"] = metadata["Media"]

    if "PID" in metadata:
        info["pid"] = metadata["PID"]

    return info
