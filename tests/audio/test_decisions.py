import pytest
from backend.audio.decisions import decide_params


def _features(noise_floor_db=-50.0, dynamic_range_db=18.0, sibilance_ratio=1.0,
               spectral_centroid=2500.0, zcr_mean=0.08, lufs=-20.0) -> dict:
    return {
        'noise_floor_db':    noise_floor_db,
        'dynamic_range_db':  dynamic_range_db,
        'sibilance_ratio':   sibilance_ratio,
        'spectral_centroid': spectral_centroid,
        'zcr_mean':          zcr_mean,
        'lufs':              lufs,
    }


def test_decide_params_returns_required_keys():
    params = decide_params(_features())
    required = {
        'gate_threshold_db', 'comp_ratio', 'comp_threshold',
        'comp_attack_ms', 'comp_release_ms', 'presence_freq',
        'presence_gain', 'deess_enabled',
    }
    assert required.issubset(params.keys())


def test_gate_threshold_above_noise_floor():
    features = _features(noise_floor_db=-48.0)
    params = decide_params(features)
    assert params['gate_threshold_db'] > features['noise_floor_db']


def test_gate_threshold_clamped_to_valid_range():
    # Even with very high noise floor, clamp to -30
    params = decide_params(_features(noise_floor_db=-10.0))
    assert params['gate_threshold_db'] <= -30.0
    # Very clean signal, clamp to -70
    params = decide_params(_features(noise_floor_db=-80.0))
    assert params['gate_threshold_db'] >= -70.0


def test_high_dynamic_range_gives_higher_ratio():
    loud_dr  = decide_params(_features(dynamic_range_db=25.0))
    quiet_dr = decide_params(_features(dynamic_range_db=10.0))
    assert loud_dr['comp_ratio'] > quiet_dr['comp_ratio']


def test_deesser_enabled_for_high_sibilance():
    params = decide_params(_features(sibilance_ratio=2.0))
    assert params['deess_enabled'] is True


def test_deesser_disabled_for_low_sibilance():
    params = decide_params(_features(sibilance_ratio=0.8))
    assert params['deess_enabled'] is False


def test_dark_vocal_gets_lower_presence_freq():
    dark   = decide_params(_features(spectral_centroid=1500.0))
    bright = decide_params(_features(spectral_centroid=4000.0))
    assert dark['presence_freq'] < bright['presence_freq']


def test_dark_vocal_gets_higher_presence_gain():
    dark   = decide_params(_features(spectral_centroid=1500.0))
    bright = decide_params(_features(spectral_centroid=4000.0))
    assert dark['presence_gain'] > bright['presence_gain']


def test_high_zcr_reduces_attack_time():
    low_zcr  = decide_params(_features(zcr_mean=0.05))
    high_zcr = decide_params(_features(zcr_mean=0.25))
    assert high_zcr['comp_attack_ms'] < low_zcr['comp_attack_ms']
