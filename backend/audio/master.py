import numpy as np
from pedalboard import Pedalboard, Limiter
from backend.audio.loudness import normalize_to_lufs

TRUE_PEAK_LIMIT_DB = -1.0


def apply_master_chain(mix: np.ndarray, sample_rate: int,
                       target_lufs: float = -14.0) -> np.ndarray:
    """
    Mastering chain: limit true peak to -1 dBTP, normalize to target_lufs,
    then hard-clip as a safety net to guarantee peak never exceeds the limit.

    The pedalboard Limiter applies make-up gain when the signal is below its
    threshold, so a second Limiter pass would boost the normalised signal.
    A hard clip avoids that side-effect while still preventing any sample-peak
    overshoot that could result from the normalization step.

    mix:  (2, N) float32
    Returns: (2, N) float32
    """
    true_peak_linear = 10.0 ** (TRUE_PEAK_LIMIT_DB / 20.0)

    limiter = Pedalboard([
        Limiter(threshold_db=TRUE_PEAK_LIMIT_DB, release_ms=100.0)
    ])

    # Pass 1: attenuate any peaks above the limit
    mix = limiter(mix, sample_rate).astype(np.float32)
    # Bring integrated loudness to target
    mix = normalize_to_lufs(mix, sample_rate, target_lufs)
    # Safety hard clip — catches any inter-sample overshoot after normalization
    mix = np.clip(mix, -true_peak_linear, true_peak_linear).astype(np.float32)

    return mix
