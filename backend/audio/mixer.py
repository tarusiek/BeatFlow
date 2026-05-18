import numpy as np
from backend.audio.loudness import normalize_to_lufs

TARGET_BEAT_LUFS  = -16.0
TARGET_VOCAL_LUFS = -20.0


def _pad_to_length(audio: np.ndarray, length: int) -> np.ndarray:
    if audio.shape[1] >= length:
        return audio
    pad_width = length - audio.shape[1]
    return np.pad(audio, ((0, 0), (0, pad_width)))


def balance_and_sum(beat: np.ndarray, vocal: np.ndarray,
                    sample_rate: int,
                    beat_target_lufs: float = TARGET_BEAT_LUFS,
                    vocal_target_lufs: float = TARGET_VOCAL_LUFS,
                    return_stems: bool = False):
    """
    Normalize beat and vocal to target LUFS values, pad to same length, sum.

    Returns:
      - mix (2, N) if return_stems=False
      - (beat_norm, vocal_norm, mix) if return_stems=True
    """
    beat_norm  = normalize_to_lufs(beat,  sample_rate, beat_target_lufs)
    vocal_norm = normalize_to_lufs(vocal, sample_rate, vocal_target_lufs)

    max_len    = max(beat_norm.shape[1], vocal_norm.shape[1])
    beat_pad   = _pad_to_length(beat_norm,  max_len)
    vocal_pad  = _pad_to_length(vocal_norm, max_len)
    mix        = (beat_pad + vocal_pad).astype(np.float32)

    if return_stems:
        return beat_norm, vocal_norm, mix
    return mix
