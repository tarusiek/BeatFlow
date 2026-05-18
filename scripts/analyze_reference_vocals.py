"""
Analyze vocal stems from reference tracks to establish target feature ranges
for the decision engine. Results saved to tests/fixtures/reference_tracks/reference_features.json
"""
import json
import pathlib
import numpy as np
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from backend.audio.io import load_audio
from backend.audio.features import extract_features

SEPARATED_DIR = pathlib.Path("D:/APKI/BeatFlow/tests/fixtures/reference_tracks/separated")
OUT_FILE = pathlib.Path("D:/APKI/BeatFlow/tests/fixtures/reference_tracks/reference_features.json")
SAMPLE_RATE = 44100


def main():
    results = {}

    track_dirs = sorted(SEPARATED_DIR.iterdir())
    for track_dir in track_dirs:
        vocal_path = track_dir / "vocals.wav"
        if not vocal_path.exists():
            print(f"  Skipping {track_dir.name}: no vocals.wav")
            continue

        print(f"\nAnalyzing {track_dir.name}/vocals.wav...", flush=True)
        audio = load_audio(str(vocal_path), SAMPLE_RATE)
        feats = extract_features(audio, SAMPLE_RATE)
        results[track_dir.name] = feats

        print(f"  lufs:              {feats['lufs']:.2f} LUFS")
        print(f"  noise_floor_db:    {feats['noise_floor_db']:.2f} dBFS")
        print(f"  dynamic_range_db:  {feats['dynamic_range_db']:.2f} dB")
        print(f"  sibilance_ratio:   {feats['sibilance_ratio']:.4f}")
        print(f"  spectral_centroid: {feats['spectral_centroid']:.1f} Hz")
        print(f"  zcr_mean:          {feats['zcr_mean']:.4f}")

    # Aggregate statistics
    all_keys = list(next(iter(results.values())).keys())
    stats = {}
    for key in all_keys:
        vals = [v[key] for v in results.values() if not (isinstance(v[key], float) and np.isinf(v[key]))]
        if not vals:
            continue
        stats[key] = {
            "min":    round(float(np.min(vals)), 4),
            "max":    round(float(np.max(vals)), 4),
            "mean":   round(float(np.mean(vals)), 4),
            "median": round(float(np.median(vals)), 4),
        }

    print("\n" + "="*60)
    print("REFERENCE VOCAL FEATURE RANGES")
    print("="*60)
    for key, s in stats.items():
        print(f"  {key:25s}: min={s['min']:.4f}  max={s['max']:.4f}  mean={s['mean']:.4f}  median={s['median']:.4f}")

    output = {
        "tracks": results,
        "stats": stats,
    }
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(output, indent=2))
    print(f"\nSaved to {OUT_FILE}")


if __name__ == "__main__":
    main()
