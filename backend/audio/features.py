import numpy as np
import librosa
from backend.audio.loudness import measure_lufs


def extract_features(audio: np.ndarray, sample_rate: int) -> dict:
    """
    Extract audio features needed for rule-based processing decisions.
    audio: (2, N) float32
    Returns dict with float values.
    """
    mono = librosa.to_mono(audio)

    # Dynamic range via RMS frame analysis
    rms_frames = librosa.feature.rms(
        y=mono, frame_length=2048, hop_length=512
    )[0]
    noise_floor_rms = float(np.percentile(rms_frames, 10))
    peak_rms        = float(np.percentile(rms_frames, 95))

    noise_floor_db = 20.0 * np.log10(noise_floor_rms + 1e-9)
    dynamic_range  = 20.0 * np.log10((peak_rms + 1e-9) / (noise_floor_rms + 1e-9))

    # Sibilance: energy ratio 6–10 kHz vs 2–6 kHz
    stft  = np.abs(librosa.stft(mono, n_fft=2048))
    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=2048)
    sib_mask  = (freqs >= 6000) & (freqs <= 10000)
    pres_mask = (freqs >= 2000) & (freqs <=  6000)
    sib_energy  = float(stft[sib_mask].mean())
    pres_energy = float(stft[pres_mask].mean())
    sibilance_ratio = sib_energy / (pres_energy + 1e-9)

    # Spectral centroid (brightness)
    centroid = float(
        librosa.feature.spectral_centroid(y=mono, sr=sample_rate).mean()
    )

    # Zero crossing rate
    zcr = float(librosa.feature.zero_crossing_rate(mono).mean())

    return {
        'lufs':              measure_lufs(audio, sample_rate),
        'noise_floor_db':    float(noise_floor_db),
        'dynamic_range_db':  float(dynamic_range),
        'sibilance_ratio':   float(sibilance_ratio),
        'spectral_centroid': float(centroid),
        'zcr_mean':          float(zcr),
    }
