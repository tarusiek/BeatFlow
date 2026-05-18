import numpy as np
import pytest
from backend.audio.loudness import measure_lufs
from backend.audio.mixer import balance_and_sum
from backend.audio.master import apply_master_chain

SAMPLE_RATE = 44100


def _stereo_sine(freq: float = 440.0, duration_s: float = 4.0,
                 amplitude: float = 0.4) -> np.ndarray:
    t = np.linspace(0, duration_s, int(SAMPLE_RATE * duration_s), endpoint=False)
    mono = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return np.stack([mono, mono])


def test_balance_and_sum_returns_stereo():
    beat  = _stereo_sine(freq=80.0)
    vocal = _stereo_sine(freq=440.0)
    result = balance_and_sum(beat, vocal, SAMPLE_RATE)
    assert result.ndim == 2
    assert result.shape[0] == 2


def test_balance_and_sum_handles_length_mismatch():
    beat  = _stereo_sine(duration_s=4.0)
    vocal = _stereo_sine(duration_s=2.5)  # shorter vocal
    result = balance_and_sum(beat, vocal, SAMPLE_RATE)
    # Output length should equal longer input
    expected_len = beat.shape[1]
    assert result.shape[1] == expected_len


def test_balance_and_sum_beat_louder_than_vocal():
    beat  = _stereo_sine(freq=80.0,  amplitude=0.5)
    vocal = _stereo_sine(freq=440.0, amplitude=0.1)
    beat_norm, vocal_norm, _ = balance_and_sum(
        beat, vocal, SAMPLE_RATE, return_stems=True
    )
    beat_lufs  = measure_lufs(beat_norm,  SAMPLE_RATE)
    vocal_lufs = measure_lufs(vocal_norm, SAMPLE_RATE)
    # Beat should be louder by roughly 4 dB
    assert beat_lufs > vocal_lufs, "Beat should be louder than vocal in mix"


def test_master_chain_reaches_target_lufs():
    mix = _stereo_sine(amplitude=0.3, duration_s=5.0)
    mastered = apply_master_chain(mix, SAMPLE_RATE, target_lufs=-14.0)
    result_lufs = measure_lufs(mastered, SAMPLE_RATE)
    assert abs(result_lufs - (-14.0)) < 1.5, f"Expected ~-14 LUFS, got {result_lufs:.2f}"


def test_master_chain_true_peak_under_limit():
    mix = _stereo_sine(amplitude=0.9, duration_s=4.0)
    mastered = apply_master_chain(mix, SAMPLE_RATE, target_lufs=-14.0)
    # Allow up to -0.5 dBFS true peak (limiter accuracy tolerance)
    true_peak_dbfs = 20.0 * np.log10(float(np.abs(mastered).max()) + 1e-9)
    assert true_peak_dbfs <= -0.5, f"True peak too high: {true_peak_dbfs:.2f} dBFS"


def test_master_chain_returns_float32():
    mix = _stereo_sine()
    result = apply_master_chain(mix, SAMPLE_RATE, target_lufs=-14.0)
    assert result.dtype == np.float32
