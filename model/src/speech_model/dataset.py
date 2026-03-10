import json
import logging
import unicodedata
from pathlib import Path

import audiomentations as am
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


def load_target_words() -> list[str]:
    """Load the target word list from frontend/src/target_words.json."""
    path = _PROJECT_ROOT / "frontend" / "src" / "target_words.json"
    with open(path) as f:
        return json.load(f)


_INVALID_AUDIO = {
    "processed/audio_segments/Preston/P15_101.wav",
    "processed/audio_segments/PD21/041027_215.wav",
}


def normalize_phonetic(text: str) -> str:
    """Strip combining diacritics by NFD decomposing then removing combining chars."""
    return "".join(c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c))


def normalize_for_cer(text: str) -> str:
    """Normalize phonetic text for CER comparison with Wav2Vec2Phoneme output.

    Strips characters the pretrained model cannot predict (stress, aspiration,
    standalone length marks) and maps ligature affricates to digraphs so that
    both model output and target text are in the same character space.
    """
    text = normalize_phonetic(text)
    # Map ligature affricates to digraphs (model uses digraphs)
    replacements = {
        "ʤ": "dʒ",
        "ʧ": "tʃ",
        "ʦ": "ts",
        "ʨ": "tɕ",
        "g": "ɡ",  # ASCII g -> IPA ɡ (U+0261)
        "ɝ": "ɚ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    # Strip characters the model cannot produce
    text = "".join(c for c in text if c not in "ˈˌːʰ* ")
    return text


class Vocab:
    """Character vocab for phonetic transcription.

    Index 0 = CTC blank, index 1 = UNK, index 2+ = phones.
    Uses a fixed set of 53 phone characters covering all English IPA phones
    needed for speech pathology diagnosis.
    """

    UNK_IDX = 1

    # fmt: off
    PHONES: list[str] = [
        # Consonants
        "b", "d", "f", "g", "h", "j", "k", "l", "m", "n", "p", "r", "s", "t",
        "v", "w", "z", "ð", "ŋ", "ɡ", "ɫ", "ɹ", "ɾ", "ʃ", "ʒ", "ʔ", "ʤ", "ʧ", "θ",
        # Vowels
        "a", "e", "i", "o", "u", "æ", "ɑ", "ɔ", "ɚ", "ɛ", "ɝ", "ɪ", "ə", "ʊ", "ʌ",
        # Modifiers and other
        "ʰ", "ˈ", "ˌ", "ː", "\u0329", "\u032F", "\u0361", "*", " ",
    ]
    # fmt: on

    def __init__(self, phones: list[str]):
        """Initialize vocab from list of phone characters."""
        self.phones = sorted(set(phones))
        # 0 = CTC blank, 1 = UNK, 2+ = phones
        self.phone_to_idx = {p: i + 2 for i, p in enumerate(self.phones)}
        self.idx_to_phone = {i + 2: p for i, p in enumerate(self.phones)}

    @classmethod
    def from_phones(cls) -> "Vocab":
        """Build vocab from the fixed phone set."""
        return cls(cls.PHONES)

    @property
    def size(self) -> int:
        """Vocab size including blank (0) and UNK (1)."""
        return len(self.phones) + 2

    def encode(self, text: str) -> list[int]:
        """Encode text to list of indices. Unknown chars map to UNK_IDX."""
        return [self.phone_to_idx.get(c, self.UNK_IDX) for c in text]

    def decode(self, ids: list[int]) -> str:
        """Decode indices to text (excludes blank and UNK tokens)."""
        return "".join(self.idx_to_phone.get(i, "") for i in ids if i not in (0, self.UNK_IDX))


class PhoneticDataset(
    Dataset[tuple[torch.Tensor, torch.Tensor, str, bool, str, str, str, str, str]]
):
    """Dataset that loads raw audio and phonetic transcriptions."""

    def __init__(
        self,
        df: pd.DataFrame,
        vocab: Vocab,
        audio_base_path: str,
        sample_rate: int = 16000,
        train: bool = False,
    ):
        """Initialize dataset.

        Args:
            df: DataFrame with audio_path and actual_phonetic columns
            vocab: Vocabulary for encoding phonetic text
            audio_base_path: Base path to prepend to audio_path
            sample_rate: Target sample rate (16kHz)
            train: Whether to apply augmentations (training only)
        """
        self.vocab = vocab
        self.audio_base_path = Path(audio_base_path)
        self.sample_rate = sample_rate
        self.augment = (
            am.Compose(
                [
                    am.TimeStretch(min_rate=0.9, max_rate=1.1, p=0.5),
                    am.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
                    am.Gain(min_gain_db=-6, max_gain_db=6, p=0.5),
                ]
            )
            if train
            else None
        )

        mask = ~df["audio_path"].isin(_INVALID_AUDIO)
        n_filtered = (~mask).sum()
        if n_filtered > 0:
            logging.warning(f"Filtered {n_filtered}/{len(df)} known-invalid audio files")
        self.df = df[mask].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index):
        """Get audio and encoded target.

        Returns:
            Tuple of (audio_waveform, target_ids, target_text, has_errors,
                       error_patterns_str, target_phonetic, utterance_id, audio_path_str, word),
            or None if the sample is broken.
        """
        row = self.df.iloc[index]
        audio_path = self.audio_base_path / row["audio_path"]

        try:
            # Use soundfile directly (avoids torchcodec dependency issues with torchaudio)
            data, sr = sf.read(audio_path)
            if data.ndim > 1:
                data = data.mean(axis=-1)

            # Resample if needed
            if sr != self.sample_rate:
                waveform = torch.from_numpy(data).float()
                waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
                data = waveform.numpy()

            # Apply augmentations (training only)
            if self.augment is not None:
                data = self.augment(
                    samples=np.asarray(data, dtype=np.float32),
                    sample_rate=self.sample_rate,
                )

            waveform = torch.from_numpy(data).float()

            # Encode target
            target_text = row["actual_phonetic"] if isinstance(row["actual_phonetic"], str) else ""
            target_text = normalize_phonetic(target_text)
            target_ids = torch.tensor(self.vocab.encode(target_text), dtype=torch.long)

            raw_patterns = row["error_patterns"]
            has_errors = raw_patterns is not None and len(raw_patterns) > 0
            error_patterns_str = ", ".join(raw_patterns) if has_errors else ""

            # Additional fields for constrained decoding and logging
            raw_target = row["target_phonetic"] if isinstance(row["target_phonetic"], str) else ""
            target_phonetic = normalize_phonetic(raw_target)
            utterance_id = str(row.get("utterance_id", ""))
            audio_path_str = str(row["audio_path"])
            word = str(row.get("word", ""))

            return (
                waveform,
                target_ids,
                target_text,
                has_errors,
                error_patterns_str,
                target_phonetic,
                utterance_id,
                audio_path_str,
                word,
            )
        except Exception:
            logging.warning(f"Skipping broken sample {index}: {audio_path}")
            return None
