import numpy as np
import pytest
from backend.audio.loudness import measure_lufs, normalize_to_lufs

SAMPLE_RATE = 44100


def _sine_stereo(freq_hz: float = 440.0, duration_s: float = 3.0,
                 amplitude: float = 0.5) -> np.ndarray:
    """Generate a stereo sine wave as (2, N) float32."""
    t = np.linspace(0, duration_s, int(SAMPLE_RATE * duration_s), endpoint=False)
    mono = (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)
    return np.stack([mono, mono])


def test_measure_lufs_returns_float():
    audio = _sine_stereo()
    result = measure_lufs(audio, SAMPLE_RATE)
    assert isinstance(result, float)


def test_measure_lufs_silent_signal():
    audio = np.zeros((2, SAMPLE_RATE * 3), dtype=np.float32)
    result = measure_lufs(audio, SAMPLE_RATE)
    assert np.isinf(result) or result < -70.0


def test_measure_lufs_reasonable_range_for_half_amplitude_sine():
    # A 440 Hz stereo sine at 0.5 amplitude (≈ -6 dBFS) should be roughly -6.75 LUFS ±5
    # RMS = 0.5/sqrt(2) = 0.354, stereo power sums both channels → ~-6.75 LUFS
    audio = _sine_stereo(amplitude=0.5, duration_s=5.0)
    result = measure_lufs(audio, SAMPLE_RATE)
    assert -10.0 < result < -3.0, f"Unexpected LUFS: {result}"


def test_normalize_to_lufs_reaches_target():
    audio = _sine_stereo(amplitude=0.1, duration_s=5.0)
    target = -20.0
    normalized = normalize_to_lufs(audio, SAMPLE_RATE, target)
    result = measure_lufs(normalized, SAMPLE_RATE)
    assert abs(result - target) < 1.0, f"Expected ~{target} LUFS, got {result:.2f}"


def test_normalize_to_lufs_silent_returns_unchanged():
    audio = np.zeros((2, SAMPLE_RATE * 3), dtype=np.float32)
    result = normalize_to_lufs(audio, SAMPLE_RATE, -20.0)
    assert np.allclose(result, audio)


def test_normalize_to_lufs_preserves_shape():
    audio = _sine_stereo()
    result = normalize_to_lufs(audio, SAMPLE_RATE, -18.0)
    assert result.shape == audio.shape
