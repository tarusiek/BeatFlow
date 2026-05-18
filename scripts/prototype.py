#!/usr/bin/env python3
"""
BeatFlow Stage 1 Prototype
Usage: python scripts/prototype.py beat.wav vocal.wav output.wav
"""
import sys
from pathlib import Path

# Add project root to path so backend imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from backend.audio.io        import load_audio, save_audio, TARGET_SAMPLE_RATE
from backend.audio.loudness  import measure_lufs
from backend.audio.features  import extract_features
from backend.audio.decisions import decide_params
from backend.audio.vocal_chain import apply_vocal_chain
from backend.audio.mixer     import balance_and_sum
from backend.audio.master    import apply_master_chain


def _print_section(title: str) -> None:
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print('─' * 50)


def run(beat_path: str, vocal_path: str, output_path: str) -> None:
    SR = TARGET_SAMPLE_RATE

    _print_section("Loading audio")
    beat  = load_audio(beat_path)
    vocal = load_audio(vocal_path)
    print(f"  Beat  : {beat.shape[1] / SR:.1f}s  "
          f"LUFS={measure_lufs(beat, SR):.1f}")
    print(f"  Vocal : {vocal.shape[1] / SR:.1f}s  "
          f"LUFS={measure_lufs(vocal, SR):.1f}")

    _print_section("Analyzing vocal")
    features = extract_features(vocal, SR)
    print(f"  LUFS             : {features['lufs']:.1f}")
    print(f"  Noise floor      : {features['noise_floor_db']:.1f} dB")
    print(f"  Dynamic range    : {features['dynamic_range_db']:.1f} dB")
    print(f"  Sibilance ratio  : {features['sibilance_ratio']:.2f}")
    print(f"  Spectral centroid: {features['spectral_centroid']:.0f} Hz")
    print(f"  ZCR mean         : {features['zcr_mean']:.4f}")

    _print_section("Deciding parameters")
    params = decide_params(features)
    print(f"  Gate threshold   : {params['gate_threshold_db']:.1f} dB")
    print(f"  Comp ratio       : {params['comp_ratio']:.1f}:1")
    print(f"  Comp threshold   : {params['comp_threshold']:.1f} dB")
    print(f"  Comp attack      : {params['comp_attack_ms']:.0f} ms")
    print(f"  Presence boost   : +{params['presence_gain']:.1f} dB @ {params['presence_freq']:.0f} Hz")
    print(f"  De-esser         : {'ON' if params['deess_enabled'] else 'OFF'}")
    if params.get('deess_enabled'):
        print(f"    Freq           : {params['deess_freq']:.0f} Hz")
        print(f"    Threshold      : {params['deess_threshold']:.1f} dB")
        print(f"    Reduction      : {params['deess_reduction']:.1f} dB")

    _print_section("Processing vocal")
    vocal_processed = apply_vocal_chain(vocal, SR, params)

    _print_section("Balancing and summing")
    mix = balance_and_sum(beat, vocal_processed, SR)

    _print_section("Mastering")
    master = apply_master_chain(mix, SR, target_lufs=-14.0)

    _print_section("Exporting")
    save_audio(output_path, master, SR)

    # ── Before/after report ───────────────────────────────────────────
    after_features = extract_features(vocal_processed, SR)
    _print_section("Before / After Report")
    print(f"  {'Metric':<25}  {'Before':>10}  {'After':>10}")
    print(f"  {'─'*25}  {'─'*10}  {'─'*10}")
    print(f"  {'Vocal LUFS':<25}  {features['lufs']:>10.1f}  "
          f"{measure_lufs(vocal_processed, SR):>10.1f}")
    print(f"  {'Dynamic range (dB)':<25}  {features['dynamic_range_db']:>10.1f}  "
          f"{after_features['dynamic_range_db']:>10.1f}")
    print(f"  {'Sibilance ratio':<25}  {features['sibilance_ratio']:>10.2f}  "
          f"{after_features['sibilance_ratio']:>10.2f}")
    print(f"\n  Final mix LUFS   : {measure_lufs(master, SR):.1f}")
    print(f"  True peak (dBFS) : {20 * np.log10(float(np.abs(master).max()) + 1e-9):.2f}")
    print(f"  Saved to         : {output_path}")
    print()


def main() -> None:
    if len(sys.argv) != 4:
        print("Usage: python scripts/prototype.py beat.wav vocal.wav output.wav")
        sys.exit(1)
    run(sys.argv[1], sys.argv[2], sys.argv[3])


if __name__ == "__main__":
    main()
