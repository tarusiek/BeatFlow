import numpy as np
import librosa
import soundfile as sf

TARGET_SAMPLE_RATE = 44100


def load_audio(path: str, target_sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """
    Load audio file, resample to target_sr, convert to (2, N) float32.
    Handles mono input by duplicating to stereo.
    """
    audio, sr = librosa.load(path, sr=target_sr, mono=False)
    if audio.ndim == 1:
        audio = np.stack([audio, audio])
    return audio.astype(np.float32)


def save_audio(path: str, audio: np.ndarray, sample_rate: int) -> None:
    """Save (2, N) float32 as 24-bit stereo WAV."""
    sf.write(path, audio.T, sample_rate, subtype='PCM_24')
