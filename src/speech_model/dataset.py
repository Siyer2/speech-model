import logging
import sys
import unicodedata
from pathlib import Path

import pandas as pd
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset

_IS_MAC = sys.platform == "darwin"


def normalize_phonetic(text: str) -> str:
    """Strip combining diacritics by NFD decomposing then removing combining chars."""
    return "".join(c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c))


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


class PhoneticDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str, bool]]):
    """Dataset that loads raw audio and phonetic transcriptions."""

    def __init__(
        self, df: pd.DataFrame, vocab: Vocab, audio_base_path: str, sample_rate: int = 16000
    ):
        """Initialize dataset.

        Args:
            df: DataFrame with audio_path and actual_phonetic columns
            vocab: Vocabulary for encoding phonetic text
            audio_base_path: Base path to prepend to audio_path
            sample_rate: Target sample rate (16kHz)
        """
        self.vocab = vocab
        self.audio_base_path = Path(audio_base_path)
        self.sample_rate = sample_rate

        # Pre-filter invalid audio files
        valid_mask = []
        for _, row in df.iterrows():
            audio_path = self.audio_base_path / row["audio_path"]
            try:
                info = sf.info(audio_path)
                is_valid = info.frames > 0
            except Exception:
                is_valid = False

            if not is_valid:
                logging.warning(f"Filtering invalid audio: {audio_path}")
            valid_mask.append(is_valid)

        n_filtered = len(df) - sum(valid_mask)
        if n_filtered > 0:
            logging.warning(f"Filtered {n_filtered}/{len(df)} invalid audio files")

        self.df = df[valid_mask].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index):
        """Get audio and encoded target.

        Returns:
            Tuple of (audio_waveform, target_ids, target_text, has_errors)
        """
        row = self.df.iloc[index]
        audio_path = self.audio_base_path / row["audio_path"]

        # TODO: Remove Mac workaround once torchcodec/ffmpeg issues are resolved
        # Mac: Use soundfile to bypass torchaudio/torchcodec issues
        # Linux: Use torchaudio (more efficient, native C++ bindings)
        if _IS_MAC:
            data, sr = sf.read(audio_path)
            waveform = torch.from_numpy(data).float()
            if waveform.ndim > 1:
                waveform = waveform.mean(dim=-1)
        else:
            waveform, sr = torchaudio.load(audio_path)
            waveform = waveform[0]  # Take first channel

        # Resample if needed
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        # Encode target
        target_text = row["actual_phonetic"] if isinstance(row["actual_phonetic"], str) else ""
        target_text = normalize_phonetic(target_text)
        target_ids = torch.tensor(self.vocab.encode(target_text), dtype=torch.long)

        has_errors = row["error_patterns"] is not None and len(row["error_patterns"]) > 0

        return waveform, target_ids, target_text, has_errors
