import numpy as np


def decide_params(features: dict) -> dict:
    """
    Map extracted audio features to DSP processing parameters.
    All rules are explicit and deterministic — no model inference.
    Returns dict of parameter names to values.
    """
    p: dict = {}

    # ── Noise gate ──────────────────────────────────────────────────────────
    p['gate_threshold_db'] = float(
        np.clip(features['noise_floor_db'] + 6.0, -70.0, -30.0)
    )

    # ── Compressor ──────────────────────────────────────────────────────────
    dr = features['dynamic_range_db']
    if dr > 22.0:
        p['comp_ratio']      = 4.0
        p['comp_threshold']  = -20.0
        p['comp_attack_ms']  = 8.0
        p['comp_release_ms'] = 80.0
    elif dr > 15.0:
        p['comp_ratio']      = 3.0
        p['comp_threshold']  = -20.0
        p['comp_attack_ms']  = 12.0
        p['comp_release_ms'] = 120.0
    else:
        p['comp_ratio']      = 2.0
        p['comp_threshold']  = -24.0
        p['comp_attack_ms']  = 18.0
        p['comp_release_ms'] = 160.0

    # High transient density → faster attack (rap/hip-hop vocals typically 0.10–0.11)
    if features['zcr_mean'] > 0.10:
        p['comp_attack_ms'] = max(4.0, p['comp_attack_ms'] - 6.0)

    # ── Presence EQ ─────────────────────────────────────────────────────────
    # Thresholds calibrated from reference vocal centroids (3011–4220 Hz range)
    centroid = features['spectral_centroid']
    if centroid < 3000.0:
        p['presence_freq'] = 3500.0
        p['presence_gain'] = 3.0
    elif centroid < 4000.0:
        p['presence_freq'] = 4000.0
        p['presence_gain'] = 2.0
    else:
        p['presence_freq'] = 5000.0
        p['presence_gain'] = 1.5

    # ── De-esser ────────────────────────────────────────────────────────────
    sr = features['sibilance_ratio']
    if sr > 1.5:
        p['deess_enabled']   = True
        p['deess_freq']      = 7000.0
        p['deess_threshold'] = -16.0
        p['deess_reduction'] = -6.0
    elif sr > 1.2:
        p['deess_enabled']   = True
        p['deess_freq']      = 8000.0
        p['deess_threshold'] = -22.0
        p['deess_reduction'] = -4.0
    else:
        p['deess_enabled'] = False

    return p
