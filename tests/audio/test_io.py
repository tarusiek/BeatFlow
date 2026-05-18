import numpy as np
import pytest
import tempfile
import os
from backend.audio.io import load_audio, save_audio

SAMPLE_RATE = 44100


def _make_temp_wav(duration_s: float = 1.0) -> str:
    """Write a temp 440 Hz sine WAV at 44100 Hz stereo, return path."""
    import soundfile as sf
    t = np.linspace(0, duration_s, int(SAMPLE_RATE * duration_s), endpoint=False)
    mono = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    stereo = np.stack([mono, mono]).T  # soundfile wants (N, 2)
    tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    tmp.close()  # close before writing on Windows
    sf.write(tmp.name, stereo, SAMPLE_RATE, subtype='PCM_16')
    return tmp.name


def test_load_audio_returns_stereo_float32():
    path = _make_temp_wav()
    try:
        audio = load_audio(path)
        assert audio.ndim == 2
        assert audio.shape[0] == 2
        assert audio.dtype == np.float32
    finally:
        os.unlink(path)


def test_load_audio_resamples_to_44100():
    import soundfile as sf
    # Write a 22050 Hz file
    t = np.linspace(0, 1.0, 22050, endpoint=False)
    mono = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    tmp.close()  # close before writing on Windows
    sf.write(tmp.name, mono, 22050)
    try:
        audio = load_audio(tmp.name)
        assert audio.shape[1] == SAMPLE_RATE  # resampled to 44100 samples for 1s
    finally:
        os.unlink(tmp.name)


def test_save_audio_creates_file():
    t = np.linspace(0, 1.0, SAMPLE_RATE, endpoint=False)
    mono = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    audio = np.stack([mono, mono])
    tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    tmp.close()  # close before unlinking on Windows
    os.unlink(tmp.name)
    try:
        save_audio(tmp.name, audio, SAMPLE_RATE)
        assert os.path.exists(tmp.name)
        assert os.path.getsize(tmp.name) > 1000
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)


def test_load_mono_converts_to_stereo():
    import soundfile as sf
    t = np.linspace(0, 1.0, SAMPLE_RATE, endpoint=False)
    mono = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    tmp.close()  # close before writing on Windows
    sf.write(tmp.name, mono, SAMPLE_RATE)
    try:
        audio = load_audio(tmp.name)
        assert audio.shape[0] == 2, "Mono input should be converted to stereo"
    finally:
        os.unlink(tmp.name)
