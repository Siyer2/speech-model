"""TALKBANK_COOKIE="..." uv run python scripts/acquire_speech_data.py"""

import os
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

TALKBANK_COOKIE = os.getenv("TALKBANK_COOKIE")
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
SUMMARY_PATH = DATA_DIR / "summary.json"
MAX_RETRIES = 3
DELAY = 2.0


def get_participant_list(response_text: str) -> list[str]:
    soup = BeautifulSoup(response_text, "html.parser")
    participants = []

    # Find all links where the text ends in .cha
    for link in soup.find_all("a", href=True):
        link_text = link.get_text()
        if link_text.endswith(".cha"):
            participant_id = Path(link_text).stem
            participants.append(participant_id)

    print(f"Found {len(participants)} participants")
    return participants


def _exponential_backoff(attempt: int) -> float:
    return 2**attempt


def download_file(url: str, output_path: Path) -> bool:
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"Skipping existing file: {output_path}")
        return True

    output_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(MAX_RETRIES):
        try:
            payload = ""
            headers = {
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",  # noqa: E501
                "Accept-Language": "en-US,en;q=0.9",
                "Cache-Control": "max-age=0",
                "Connection": "keep-alive",
                "DNT": "1",
                "If-None-Match": 'W/"3790-ymAo/5caHmWPgL/ttGYKsMTEIYg"',
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-User": "?1",
                "Upgrade-Insecure-Requests": "1",
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36",  # noqa: E501
                "sec-ch-ua": '"Chromium";v="139", "Not;A=Brand";v="99"',
                "sec-ch-ua-mobile": "?0",
                "sec-ch-ua-platform": '"macOS"',
                "Cookie": TALKBANK_COOKIE,
            }
            response = requests.request("GET", url, data=payload, headers=headers)

            if response.status_code == 429:
                wait_time = 60
                print(f"Rate limited. Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
                continue

            response.raise_for_status()

            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            print(f"Downloaded: {url} -> {output_path}")
            return True

        except requests.RequestException as e:
            backoff = _exponential_backoff(attempt)
            print(f"Attempt {attempt + 1} failed for {url}: {e}")

            if attempt < MAX_RETRIES - 1:
                print(f"Retrying in {backoff}s...")
                time.sleep(backoff)
            else:
                print(f"Failed to download {url} after {MAX_RETRIES} attempts")
                return False

    return False


def download_preston() -> dict:
    print("Starting Preston dataset download...")

    url = "https://git.talkbank.org/phon/data-orig/Clinical/Preston"
    payload = ""
    headers = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",  # noqa: E501
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "max-age=0",
        "Connection": "keep-alive",
        "DNT": "1",
        "If-None-Match": 'W/"3790-ymAo/5caHmWPgL/ttGYKsMTEIYg"',
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Upgrade-Insecure-Requests": "1",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36",  # noqa: E501
        "sec-ch-ua": '"Chromium";v="139", "Not;A=Brand";v="99"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"macOS"',
        "Cookie": TALKBANK_COOKIE,
    }
    response = requests.request("GET", url, data=payload, headers=headers)
    participants = get_participant_list(response.text)

    output_dir = RAW_DIR / "Preston"
    success_count = 0
    failed = []

    for participant in tqdm(participants, desc="Preston"):
        cha_url = f"https://git.talkbank.org/phon/data-orig/Clinical/Preston/{participant}.cha"
        wav_url = f"https://media.talkbank.org/phon/Clinical/Preston/0wav//{participant}.wav?f=save"

        cha_path = output_dir / f"{participant}.cha"
        wav_path = output_dir / f"{participant}.wav"

        cha_ok = download_file(cha_url, cha_path)
        wav_ok = download_file(wav_url, wav_path)

        if cha_ok and wav_ok:
            success_count += 1
        else:
            failed.append(participant)

    return {"success": success_count, "failed": failed, "total": len(participants)}


def download_percept(dataset_name: str) -> dict:
    print(f"Starting {dataset_name} dataset download...")

    url = f"https://git.talkbank.org/phon/data-orig/Clinical/PERCEPT-GFTA/{dataset_name}"
    payload = ""
    headers = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",  # noqa: E501
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "max-age=0",
        "Connection": "keep-alive",
        "DNT": "1",
        "If-None-Match": 'W/"3790-ymAo/5caHmWPgL/ttGYKsMTEIYg"',
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Upgrade-Insecure-Requests": "1",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36",  # noqa: E501
        "sec-ch-ua": '"Chromium";v="139", "Not;A=Brand";v="99"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"macOS"',
        "Cookie": TALKBANK_COOKIE,
    }
    response = requests.request("GET", url, data=payload, headers=headers)
    participants = get_participant_list(response.text)

    output_dir = RAW_DIR / dataset_name
    success_count = 0
    failed = []

    for participant in tqdm(participants, desc=dataset_name):
        cha_url = f"https://git.talkbank.org/phon/data-orig/Clinical/PERCEPT-GFTA/{dataset_name}/{participant}.cha?f=save"
        wav_url = f"https://media.talkbank.org/phon/Clinical/PERCEPT-GFTA/{dataset_name}/{participant}.wav?f=save"

        cha_path = output_dir / f"{participant}.cha"
        wav_path = output_dir / f"{participant}.wav"

        cha_ok = download_file(cha_url, cha_path)
        wav_ok = download_file(wav_url, wav_path)

        if cha_ok and wav_ok:
            success_count += 1
        else:
            failed.append(participant)

    return {"success": success_count, "failed": failed, "total": len(participants)}


def main():
    if TALKBANK_COOKIE is None:
        print("Missing TALKBANK_COOKIE")
        return 1

    results = {}
    results["Preston"] = download_preston()
    for dataset_name in ["PreKHistorySSD", "SuspectedSSD", "TDChildrenandAdults"]:
        results[dataset_name] = download_percept(dataset_name)

    total_success = sum(r["success"] for r in results.values())
    total_files = sum(r["total"] for r in results.values())
    print("\n=== Download Complete ===")
    print(f"Total: {total_success}/{total_files} participants successful")
    return 0


if __name__ == "__main__":
    main()
