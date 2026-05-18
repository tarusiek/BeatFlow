import numpy as np
import pytest
import librosa
from backend.audio.vocal_chain import apply_vocal_chain

SAMPLE_RATE = 44100


def _noisy_vocal(duration_s: float = 3.0) -> np.ndarray:
    """Simulate a noisy vocal: quiet noise floor + 440 Hz signal + harsh highs."""
    rng = np.random.default_rng(99)
    t = np.linspace(0, duration_s, int(SAMPLE_RATE * duration_s), endpoint=False)
    noise  = 0.02 * rng.standard_normal(len(t)).astype(np.float32)
    voice  = 0.35 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    sib    = 0.15 * np.sin(2 * np.pi * 8000 * t).astype(np.float32)
    mono   = noise + voice + sib
    return np.stack([mono, mono])


_DEFAULT_PARAMS = {
    'gate_threshold_db':  -46.0,
    'comp_ratio':         3.0,
    'comp_threshold':     -20.0,
    'comp_attack_ms':     12.0,
    'comp_release_ms':    120.0,
    'presence_freq':      4000.0,
    'presence_gain':      2.0,
    'deess_enabled':      True,
    'deess_freq':         7000.0,
    'deess_threshold':    -16.0,
    'deess_reduction':    -6.0,
}


def test_vocal_chain_returns_same_shape():
    audio = _noisy_vocal()
    result = apply_vocal_chain(audio, SAMPLE_RATE, _DEFAULT_PARAMS)
    assert result.shape == audio.shape


def test_vocal_chain_returns_float32():
    audio = _noisy_vocal()
    result = apply_vocal_chain(audio, SAMPLE_RATE, _DEFAULT_PARAMS)
    assert result.dtype == np.float32


def test_vocal_chain_no_clipping():
    audio = _noisy_vocal()
    result = apply_vocal_chain(audio, SAMPLE_RATE, _DEFAULT_PARAMS)
    assert float(np.abs(result).max()) <= 1.05, "Output clips significantly"


def test_vocal_chain_reduces_noise_floor():
    audio = _noisy_vocal()
    result = apply_vocal_chain(audio, SAMPLE_RATE, _DEFAULT_PARAMS)
    mono_in  = librosa.to_mono(audio)
    mono_out = librosa.to_mono(result)
    rms_in  = np.percentile(librosa.feature.rms(y=mono_in)[0],  10)
    rms_out = np.percentile(librosa.feature.rms(y=mono_out)[0], 10)
    assert rms_out <= rms_in * 1.1, "Noise floor did not decrease"


def test_vocal_chain_without_deesser():
    params = {**_DEFAULT_PARAMS, 'deess_enabled': False}
    audio  = _noisy_vocal()
    result = apply_vocal_chain(audio, SAMPLE_RATE, params)
    assert result.shape == audio.shape
