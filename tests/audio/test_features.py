import numpy as np
import pytest
from backend.audio.features import extract_features

SAMPLE_RATE = 44100


def _white_noise_stereo(duration_s: float = 3.0, amplitude: float = 0.3) -> np.ndarray:
    rng = np.random.default_rng(42)
    mono = (amplitude * rng.standard_normal(int(SAMPLE_RATE * duration_s))).astype(np.float32)
    mono = np.clip(mono, -1.0, 1.0)
    return np.stack([mono, mono])


def _sine_with_noise(duration_s: float = 3.0) -> np.ndarray:
    """A 440 Hz sine with a low-level noise floor."""
    t = np.linspace(0, duration_s, int(SAMPLE_RATE * duration_s), endpoint=False)
    signal = 0.4 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    rng = np.random.default_rng(7)
    noise = 0.005 * rng.standard_normal(len(t)).astype(np.float32)
    mono = signal + noise
    return np.stack([mono, mono])


def test_extract_features_returns_all_keys():
    audio = _sine_with_noise()
    features = extract_features(audio, SAMPLE_RATE)
    expected_keys = {
        'lufs', 'noise_floor_db', 'dynamic_range_db',
        'sibilance_ratio', 'spectral_centroid', 'zcr_mean',
    }
    assert expected_keys.issubset(features.keys())


def test_extract_features_all_values_are_floats():
    audio = _sine_with_noise()
    features = extract_features(audio, SAMPLE_RATE)
    for k, v in features.items():
        assert isinstance(v, float), f"{k} is not float: {type(v)}"


def test_dynamic_range_positive():
    audio = _sine_with_noise()
    features = extract_features(audio, SAMPLE_RATE)
    assert features['dynamic_range_db'] > 0.0


def test_noise_floor_below_peak():
    audio = _sine_with_noise()
    features = extract_features(audio, SAMPLE_RATE)
    assert features['noise_floor_db'] < features['lufs']


def test_sibilance_ratio_positive():
    audio = _white_noise_stereo()
    features = extract_features(audio, SAMPLE_RATE)
    # White noise has roughly equal energy across bands, ratio near 1.0
    assert 0.5 < features['sibilance_ratio'] < 3.0


def test_high_frequency_signal_has_high_sibilance():
    """A 8kHz sine should have very high sibilance ratio."""
    t = np.linspace(0, 3.0, int(SAMPLE_RATE * 3.0), endpoint=False)
    mono = (0.3 * np.sin(2 * np.pi * 8000 * t)).astype(np.float32)
    audio = np.stack([mono, mono])
    features = extract_features(audio, SAMPLE_RATE)
    assert features['sibilance_ratio'] > 2.0


def test_spectral_centroid_higher_for_bright_signal():
    t = np.linspace(0, 3.0, int(SAMPLE_RATE * 3.0), endpoint=False)
    low_mono  = (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    high_mono = (0.3 * np.sin(2 * np.pi * 4000 * t)).astype(np.float32)
    low_audio  = np.stack([low_mono,  low_mono])
    high_audio = np.stack([high_mono, high_mono])
    low_feat  = extract_features(low_audio,  SAMPLE_RATE)
    high_feat = extract_features(high_audio, SAMPLE_RATE)
    assert high_feat['spectral_centroid'] > low_feat['spectral_centroid']
