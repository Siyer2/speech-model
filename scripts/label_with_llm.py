import json
import os
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
PARQUET_PATH = PROCESSED_DIR / "utterances.parquet"
ONTOLOGY_PATH = Path("ontology.yaml")
MODEL = "gemini-3-flash-preview"

client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
)


def load_error_patterns(ontology_path: Path) -> dict:
    """Load error patterns from ontology.yaml.

    Args:
        ontology_path: Path to ontology.yaml

    Returns:
        Dict of error patterns with their IDs as keys
    """
    with open(ontology_path) as f:
        ontology = yaml.safe_load(f)

    return ontology.get("error_patterns", {})


def label_utterance_with_llm(
    target_phonetic: str, actual_phonetic: str, error_patterns: dict
) -> list[str]:
    """Label an utterance with error patterns using an LLM.

    Call LLM with:
    - target_phonetic: the correct pronunciation
    - actual_phonetic: what the child actually said
    - error_patterns: dict of possible error patterns from ontology

    Args:
        target_phonetic: Target pronunciation (IPA)
        actual_phonetic: Actual pronunciation (IPA)
        error_patterns: Dict of error patterns from ontology

    Returns:
        List of error pattern IDs present in this utterance
        Example: ["gliding_l", "cluster_reduction"]
    """

    prompt = f"""
    Given the following phonetic transcriptions:
    Target: {target_phonetic}
    Actual: {actual_phonetic}
    
    Identify which of these error patterns are present:
    {json.dumps(error_patterns, indent=2)}
    
    Return ONLY a JSON array of pattern IDs. Do not include any markdown or explanations.
    Example format: ["pattern_id_1", "pattern_id_2"]
    """

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )

    # # todo: remove these 2
    # print(json.loads(response.choices[0].message.content))
    # exit(0)

    return json.loads(response.choices[0].message.content)


def add_error_pattern_labels(parquet_path: Path, ontology_path: Path) -> None:
    """Add error pattern labels to utterances parquet file.

    Args:
        parquet_path: Path to utterances.parquet
        ontology_path: Path to ontology.yaml
    """
    print(f"Loading parquet from {parquet_path}...")
    df = pd.read_parquet(parquet_path)

    print(f"Loading error patterns from {ontology_path}...")
    error_patterns = load_error_patterns(ontology_path)

    print(f"Loaded {len(error_patterns)} error patterns")
    print(f"Processing {len(df)} utterances...")

    # Add labels for each utterance
    error_pattern_labels = []

    for idx, row in df.iterrows():
        # Call LLM to get labels
        labels = label_utterance_with_llm(
            row["target_phonetic"], row["actual_phonetic"], error_patterns
        )

        # Store as JSON string for Parquet compatibility
        error_pattern_labels.append(json.dumps(labels) if labels else None)

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(df)} utterances...")

    # Add to DataFrame
    df["error_patterns"] = error_pattern_labels

    # Add labeling metadata
    df["labeling_metadata"] = json.dumps(
        {
            "model": MODEL,
            "timestamp": pd.Timestamp.now().isoformat(),
        }
    )

    df.to_parquet(
        parquet_path,
        engine="pyarrow",
        compression="snappy",
        index=False,
    )

    print(f"Saved updated parquet to {parquet_path}")
    num_labeled = sum(df["error_patterns"].notna())
    print(f"Added error_patterns column with labels for {num_labeled} utterances")


def main():
    add_error_pattern_labels(PARQUET_PATH, ONTOLOGY_PATH)
    return 0


if __name__ == "__main__":
    exit(main())
