import numpy as np
import pyloudnorm as pyln


def measure_lufs(audio: np.ndarray, sample_rate: int) -> float:
    """
    Measure integrated loudness (ITU-R BS.1770-4).
    audio: (2, N) or (N,) float32
    Returns LUFS as float; -inf if signal too quiet to measure.
    """
    meter = pyln.Meter(sample_rate)
    if audio.ndim == 2:
        loudness = meter.integrated_loudness(audio.T)
    else:
        loudness = meter.integrated_loudness(audio)
    return float(loudness)


def normalize_to_lufs(audio: np.ndarray, sample_rate: int,
                      target_lufs: float) -> np.ndarray:
    """Apply linear gain to reach target_lufs. Returns audio unchanged if silent."""
    current = measure_lufs(audio, sample_rate)
    if np.isinf(current):
        return audio
    gain_db = target_lufs - current
    gain_linear = 10.0 ** (gain_db / 20.0)
    return (audio * gain_linear).astype(np.float32)
