import logging
import sys
from pathlib import Path

import pandas as pd
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset

_IS_MAC = sys.platform == "darwin"


class Vocab:
    """Character vocabulary for phonetic transcription. Index 0 is reserved for CTC blank."""

    def __init__(self, phones: list[str] | None = None):
        """Initialize vocab from list of phone characters.

        Args:
            phones: List of unique characters (blank not included). If None, loads IPA base vocab.
        """
        if phones is None:
            ipa_path = Path(__file__).parent / "ipa-data.csv"
            ipa_df = pd.read_csv(ipa_path)
            phones = ipa_df["Symbol"].dropna().tolist()
        self.phones = sorted(set(phones))
        self.phone_to_idx = {p: i + 1 for i, p in enumerate(self.phones)}
        self.idx_to_phone = {i + 1: p for i, p in enumerate(self.phones)}

    @classmethod
    def from_texts(cls, texts: list[str]) -> "Vocab":
        """Build vocab from list of phonetic transcriptions, starting with IPA base."""
        base_vocab = cls()  # Load IPA base
        chars = set(base_vocab.phones)

        for text in texts:
            if isinstance(text, str):
                new_chars = set(text) - chars
                if new_chars:
                    logging.warning(f"Adding phones not in IPA base: {new_chars}")
                    chars.update(new_chars)

        return cls(list(chars))

    @property
    def size(self) -> int:
        """Vocab size including blank token at index 0."""
        return len(self.phones) + 1

    def encode(self, text: str) -> list[int]:
        """Encode text to list of indices."""
        return [self.phone_to_idx.get(c, 0) for c in text]

    def decode(self, ids: list[int]) -> str:
        """Decode indices to text (excludes blank tokens)."""
        return "".join(self.idx_to_phone.get(i, "") for i in ids if i != 0)


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
        target_ids = torch.tensor(self.vocab.encode(target_text), dtype=torch.long)

        has_errors = len(row["error_patterns"]) > 0

        return waveform, target_ids, target_text, has_errors
