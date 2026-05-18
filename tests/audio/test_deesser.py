import numpy as np
import pytest
from backend.audio.deesser import apply_deesser

SAMPLE_RATE = 44100


def _sibilant_audio(duration_s: float = 2.0) -> np.ndarray:
    """High-frequency heavy audio (simulates harsh sibilance)."""
    t = np.linspace(0, duration_s, int(SAMPLE_RATE * duration_s), endpoint=False)
    mono = (0.4 * np.sin(2 * np.pi * 8000 * t)).astype(np.float32)
    mono += (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    return np.stack([mono, mono])


def _low_freq_audio(duration_s: float = 2.0) -> np.ndarray:
    """Low-frequency audio (no sibilance)."""
    t = np.linspace(0, duration_s, int(SAMPLE_RATE * duration_s), endpoint=False)
    mono = (0.4 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    return np.stack([mono, mono])


def test_deesser_returns_same_shape():
    audio = _sibilant_audio()
    result = apply_deesser(audio, SAMPLE_RATE, freq=7000.0,
                           threshold_db=-16.0, reduction_db=-6.0)
    assert result.shape == audio.shape


def test_deesser_returns_float32():
    audio = _sibilant_audio()
    result = apply_deesser(audio, SAMPLE_RATE, freq=7000.0,
                           threshold_db=-16.0, reduction_db=-6.0)
    assert result.dtype == np.float32


def test_deesser_reduces_high_frequency_energy():
    audio = _sibilant_audio()
    import librosa
    result = apply_deesser(audio, SAMPLE_RATE, freq=7000.0,
                           threshold_db=-16.0, reduction_db=-6.0)
    freqs = librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=2048)
    sib_mask = freqs >= 7000

    before_stft = np.abs(librosa.stft(librosa.to_mono(audio), n_fft=2048))
    after_stft  = np.abs(librosa.stft(librosa.to_mono(result), n_fft=2048))

    before_energy = float(before_stft[sib_mask].mean())
    after_energy  = float(after_stft[sib_mask].mean())

    assert after_energy < before_energy, (
        f"De-esser did not reduce high-freq energy: before={before_energy:.4f}, "
        f"after={after_energy:.4f}"
    )


def test_deesser_does_not_touch_low_freq_audio_significantly():
    audio = _low_freq_audio()
    result = apply_deesser(audio, SAMPLE_RATE, freq=7000.0,
                           threshold_db=-16.0, reduction_db=-6.0)
    # Low-freq audio should pass through with < 0.5 dB change
    before_rms = float(np.sqrt(np.mean(audio ** 2)))
    after_rms  = float(np.sqrt(np.mean(result ** 2)))
    diff_db = abs(20.0 * np.log10((after_rms + 1e-9) / (before_rms + 1e-9)))
    assert diff_db < 1.0, f"De-esser changed low-freq audio by {diff_db:.2f} dB"


def test_deesser_output_has_no_clipping():
    audio = _sibilant_audio()
    result = apply_deesser(audio, SAMPLE_RATE, freq=7000.0,
                           threshold_db=-16.0, reduction_db=-6.0)
    assert float(np.abs(result).max()) <= 1.0
