import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv
from google import genai
from google.genai.files import types
from openai import OpenAI

load_dotenv()

DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
PARQUET_PATH = PROCESSED_DIR / "utterances.parquet"
BATCH_DIR = DATA_DIR / "batch_jobs"
BATCH_DIR.mkdir(parents=True, exist_ok=True)
BATCH_RESULT_DIR = BATCH_DIR / "result_batches"
BATCH_RESULT_DIR.mkdir(parents=True, exist_ok=True)
ONTOLOGY_PATH = Path("ontology.yaml")

MODEL = "gemini-3-flash-preview"

client = genai.Client(api_key=os.getenv("API_KEY"))
openai_client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
)


def create_prompt(target_phonetic: str, actual_phonetic: str, error_patterns: dict) -> str:
    """Create prompt optimized for caching.

    Static content (error_patterns) goes first to enable prompt caching.
    Dynamic content (phonetics) goes last.

    Args:
        target_phonetic: Target pronunciation (IPA)
        actual_phonetic: Actual pronunciation (IPA)
        error_patterns: Dict of error patterns from ontology

    Returns:
        Optimized prompt string
    """
    return f"""You are a speech pathology assistant analyzing phonetic transcriptions.

Here are the error patterns you can identify:
{json.dumps(error_patterns)}

Now analyze this specific utterance:
Target pronunciation: {target_phonetic}
Actual pronunciation: {actual_phonetic}

Identify which error patterns from the list above are present in this utterance.
Return ONLY a JSON object with a "patterns" key containing an array of pattern IDs.
Example format: {{"patterns": ["pattern_id_1", "pattern_id_2"]}}
"""


def load_error_patterns() -> dict:
    with open(ONTOLOGY_PATH) as f:
        ontology = yaml.safe_load(f)

    error_patterns = ontology.get("error_patterns", {})

    stripped = {}
    for pattern_id, pattern_data in error_patterns.items():
        entry = {"desc": pattern_data.get("description", "")}

        # Only include non-empty sounds_affected
        sounds = pattern_data.get("sounds_affected", [])
        if sounds:
            entry["sounds"] = sounds

        # Only include up to 2 examples to save tokens
        examples = pattern_data.get("examples", [])
        if examples:
            entry["ex"] = examples[:2]

        stripped[pattern_id] = entry

    return stripped


def create_batch_file():
    print("Creating batch file")
    df = pd.read_parquet(PARQUET_PATH)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_file_path = BATCH_DIR / f"batch_{timestamp}.jsonl"
    error_patterns = load_error_patterns()

    with open(batch_file_path, "w") as f:
        for _, row in df.iterrows():
            prompt = create_prompt(row["target_phonetic"], row["actual_phonetic"], error_patterns)

            request = {
                "custom_id": row["utterance_id"],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "response_format": {"type": "json_object"},
                },
            }
            f.write(json.dumps(request) + "\n")

    print(f"Created batch file at {batch_file_path}")


def submit_batch(batch_file_path: Path) -> str:
    """Submit a batch file and return the batch ID."""
    print(f"Submitting batch using {batch_file_path}")

    # Upload the JSONL file in OpenAI input format, using regular genai SDK
    uploaded_file = client.files.upload(
        file=batch_file_path,
        config=types.UploadFileConfig(display_name="my-batch-requests", mime_type="jsonl"),
    )

    batch_input_file_id = uploaded_file.name
    if batch_input_file_id is None:
        raise ValueError("batch_input_file_id is None. Upload may have failed.")

    # Create batch
    batch = openai_client.batches.create(
        input_file_id=batch_input_file_id, endpoint="/v1/chat/completions", completion_window="24h"
    )

    print(f"Created batch with id {batch.id}")
    return batch.id


def retrieve_batch(batch_input_id: str) -> bool:
    """Retrieve batch results and update parquet. Returns True if successful."""
    batch = openai_client.batches.retrieve(batch_input_id)
    if batch.status != "completed":
        print(f"Batch status: {batch.status}")
        return False

    print(f"{batch_input_id} completed. Retrieving results...")

    if batch.output_file_id is None:
        raise ValueError("batch.output_file_id is None. Batch result may not be available yet.")

    file_content = client.files.download(file=batch.output_file_id).decode("utf-8")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file_path = BATCH_DIR / f"result_{batch_input_id}_{timestamp}.jsonl"

    with open(result_file_path, "w") as f:
        f.write(file_content)

    print(f"Results saved to {result_file_path}")

    # Update parquet with error patterns
    df = pd.read_parquet(PARQUET_PATH)

    df = df.set_index("utterance_id")

    for line in file_content.strip().split("\n"):
        result = json.loads(line)
        utterance_id = result["custom_id"]

        response_body = result.get("response", {}).get("body", {})
        choices = response_body.get("choices", [])
        if not choices:
            continue

        content = choices[0].get("message", {}).get("content", "{}")
        patterns_data = json.loads(content)
        patterns = patterns_data.get("patterns", [])

        df.at[utterance_id, "error_patterns"] = patterns

    df = df.reset_index()

    df.to_parquet(PARQUET_PATH, index=False)
    print(f"Updated {PARQUET_PATH} with error patterns")
    return True


def batch_submit_all(batch_file_path: Path):
    """Process a batch file in chunks of 45, waiting for each to complete.

    Args:
        batch_file_path: Path to the batch JSONL file to process
    """
    # Read all requests from the batch file
    with open(batch_file_path) as f:
        all_requests = [line.strip() for line in f if line.strip()]

    total = len(all_requests)
    print(f"Processing {total} requests in chunks of 45")

    chunk_size = 45
    batch_num = 1

    for i in range(0, total, chunk_size):
        chunk = all_requests[i : i + chunk_size]
        total_batches = (total + chunk_size - 1) // chunk_size
        print(f"\n=== Batch {batch_num}/{total_batches} ({len(chunk)} requests) ===")

        # Create temp batch file for this chunk
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chunk_file = BATCH_DIR / f"batch_chunk_{batch_num}_{timestamp}.jsonl"

        with open(chunk_file, "w") as f:
            for line in chunk:
                f.write(line + "\n")

        print(f"Created chunk file: {chunk_file}")

        # Submit batch
        batch_id = submit_batch(chunk_file)

        # Poll until complete
        print("Polling for completion (checking every 60s)...")
        while True:
            time.sleep(60)
            if retrieve_batch(batch_id):
                print(f"Batch {batch_num} complete!")
                break

        batch_num += 1

    print("\nAll done!")


def main():
    command = sys.argv[1]

    if command == "create":
        create_batch_file()
    elif command == "submit":
        batch_file_path = Path(sys.argv[2])
        if not batch_file_path.exists():
            print(f"Batch file not found: {batch_file_path}")
            return

        submit_batch(batch_file_path)

    elif command == "retrieve":
        batch_input_id = sys.argv[2]
        if not batch_input_id:
            print("No batch_input_id provided")
            return

        retrieve_batch(batch_input_id)

    elif command == "batch_submit":
        batch_file_path = Path(sys.argv[2])
        if not batch_file_path.exists():
            print(f"Batch file not found: {batch_file_path}")
            return

        batch_submit_all(batch_file_path)


if __name__ == "__main__":
    exit(main())
