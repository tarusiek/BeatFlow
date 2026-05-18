import numpy as np
import os
import tempfile
import soundfile as sf
import pytest
from scripts.prototype import run

SAMPLE_RATE = 44100


def _write_wav(path: str, freq: float = 440.0, duration_s: float = 4.0,
               amplitude: float = 0.35) -> None:
    t = np.linspace(0, duration_s, int(SAMPLE_RATE * duration_s), endpoint=False)
    mono = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    # Add a noise floor
    rng = np.random.default_rng(42)
    mono += 0.01 * rng.standard_normal(len(mono)).astype(np.float32)
    stereo = np.stack([mono, mono]).T
    sf.write(path, stereo, SAMPLE_RATE, subtype='PCM_16')


def test_prototype_run_produces_output_file():
    with tempfile.TemporaryDirectory() as tmp:
        beat_path   = os.path.join(tmp, 'beat.wav')
        vocal_path  = os.path.join(tmp, 'vocal.wav')
        output_path = os.path.join(tmp, 'output.wav')

        _write_wav(beat_path,  freq=80.0,  amplitude=0.5)
        _write_wav(vocal_path, freq=440.0, amplitude=0.3)

        run(beat_path, vocal_path, output_path)

        assert os.path.exists(output_path), "Output file was not created"
        assert os.path.getsize(output_path) > 10_000, "Output file is suspiciously small"


def test_prototype_output_lufs_near_target():
    import pyloudnorm as pyln
    with tempfile.TemporaryDirectory() as tmp:
        beat_path   = os.path.join(tmp, 'beat.wav')
        vocal_path  = os.path.join(tmp, 'vocal.wav')
        output_path = os.path.join(tmp, 'output.wav')

        _write_wav(beat_path,  freq=80.0,  amplitude=0.5, duration_s=5.0)
        _write_wav(vocal_path, freq=440.0, amplitude=0.3, duration_s=5.0)

        run(beat_path, vocal_path, output_path)

        audio, sr = sf.read(output_path)
        meter = pyln.Meter(sr)
        lufs = meter.integrated_loudness(audio)
        assert abs(lufs - (-14.0)) < 2.0, f"Final LUFS {lufs:.1f} not near -14"


def test_prototype_output_no_clipping():
    with tempfile.TemporaryDirectory() as tmp:
        beat_path   = os.path.join(tmp, 'beat.wav')
        vocal_path  = os.path.join(tmp, 'vocal.wav')
        output_path = os.path.join(tmp, 'output.wav')

        _write_wav(beat_path,  freq=80.0,  amplitude=0.8, duration_s=5.0)
        _write_wav(vocal_path, freq=440.0, amplitude=0.8, duration_s=5.0)

        run(beat_path, vocal_path, output_path)

        audio, _ = sf.read(output_path)
        max_sample = float(np.abs(audio).max())
        assert max_sample <= 1.0, f"Output clips at {max_sample:.4f}"
