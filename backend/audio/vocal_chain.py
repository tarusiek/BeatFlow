import numpy as np
import noisereduce as nr
from pedalboard import (
    Pedalboard, HighpassFilter, LowShelfFilter, HighShelfFilter,
    Compressor, NoiseGate,
)
from backend.audio.deesser import apply_deesser


def apply_vocal_chain(audio: np.ndarray, sample_rate: int,
                      params: dict) -> np.ndarray:
    """
    Full vocal processing chain. Order matters:
    1. Noise reduction
    2. Highpass filter (80 Hz)
    3. Noise gate
    4. Low-mid cut (300 Hz mud reduction)
    5. Presence boost (freq/gain from params)
    6. Air shelf (+1.5 dB at 12 kHz)
    7. Compressor
    8. De-esser (if enabled)

    audio:   (2, N) float32
    params:  dict from decide_params()
    Returns: (2, N) float32
    """
    # 1. Noise reduction — use first 300 ms as noise profile
    noise_len = min(int(sample_rate * 0.3), audio.shape[1] // 4)
    noise_sample = audio[:, :noise_len]
    audio = nr.reduce_noise(
        y=audio, sr=sample_rate, y_noise=noise_sample,
        prop_decrease=0.75, n_fft=2048,
    ).astype(np.float32)

    # 2–7. EQ, gate, compressor via pedalboard
    chain = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=80.0),
        NoiseGate(
            threshold_db=params['gate_threshold_db'],
            ratio=10.0,
            attack_ms=1.0,
            release_ms=80.0,
        ),
        LowShelfFilter(
            cutoff_frequency_hz=300.0,
            gain_db=-2.0,
        ),
        HighShelfFilter(
            cutoff_frequency_hz=params['presence_freq'],
            gain_db=params['presence_gain'],
        ),
        HighShelfFilter(
            cutoff_frequency_hz=12000.0,
            gain_db=1.5,
        ),
        Compressor(
            threshold_db=params['comp_threshold'],
            ratio=params['comp_ratio'],
            attack_ms=params['comp_attack_ms'],
            release_ms=params['comp_release_ms'],
        ),
    ])
    audio = chain(audio, sample_rate).astype(np.float32)

    # 8. De-esser
    if params.get('deess_enabled'):
        audio = apply_deesser(
            audio, sample_rate,
            freq=params['deess_freq'],
            threshold_db=params['deess_threshold'],
            reduction_db=params['deess_reduction'],
        )

    return audio
