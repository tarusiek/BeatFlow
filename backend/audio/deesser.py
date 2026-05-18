import numpy as np
import librosa
from scipy.ndimage import uniform_filter1d


def apply_deesser(audio: np.ndarray, sample_rate: int, freq: float,
                  threshold_db: float, reduction_db: float) -> np.ndarray:
    """
    Spectral de-esser: detect frames where energy above `freq` Hz exceeds
    `threshold_db`, apply `reduction_db` gain to those frames on all channels.

    audio:          (2, N) float32
    freq:           frequency above which sibilance is measured (e.g. 7000.0)
    threshold_db:   level above which de-essing activates (e.g. -16.0)
    reduction_db:   gain reduction applied (negative, e.g. -6.0)
    Returns:        (2, N) float32, same shape as input
    """
    n_fft = 2048
    hop   = 512

    mono  = librosa.to_mono(audio)
    stft  = librosa.stft(mono, n_fft=n_fft, hop_length=hop)
    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)

    sib_mask    = freqs >= freq
    magnitudes  = np.abs(stft)
    sib_energy  = magnitudes[sib_mask].mean(axis=0)       # (n_frames,)

    threshold_linear = 10.0 ** (threshold_db / 20.0)
    reduction_linear = 10.0 ** (reduction_db / 20.0)      # < 1.0

    # Gain = 1.0 below threshold, reduction_linear above
    gain_curve = np.where(sib_energy > threshold_linear, reduction_linear, 1.0)

    # Smooth to prevent zipper noise
    gain_curve = uniform_filter1d(gain_curve.astype(np.float64), size=9).astype(np.float32)

    # Map frame-level gain to sample-level gain via linear interpolation
    frame_centers = np.arange(len(gain_curve)) * hop + n_fft // 2
    sample_indices = np.arange(audio.shape[1], dtype=np.float32)
    gain_samples = np.interp(sample_indices, frame_centers, gain_curve)
    gain_samples = np.clip(gain_samples, float(reduction_linear), 1.0).astype(np.float32)

    return (audio * gain_samples[np.newaxis, :]).astype(np.float32)
