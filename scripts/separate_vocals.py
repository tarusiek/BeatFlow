"""
Vocal separation using demucs htdemucs model.
Loads audio with soundfile to bypass torchaudio/torchcodec dependency.
Outputs vocals.wav and no_vocals.wav alongside the input file.
"""
import sys
import pathlib
import numpy as np
import soundfile as sf
import torch
from demucs.pretrained import get_model
from demucs.apply import apply_model

MODEL_NAME = "htdemucs"
TARGET_SR = 44100


def load_audio_sf(path: str) -> tuple[torch.Tensor, int]:
    """Load audio as (channels, samples) float32 tensor using soundfile."""
    audio, sr = sf.read(path, always_2d=True)  # (samples, channels)
    audio = audio.T.astype(np.float32)          # (channels, samples)
    if audio.shape[0] == 1:
        audio = np.concatenate([audio, audio], axis=0)
    return torch.from_numpy(audio), sr


def save_stem(path: pathlib.Path, audio_tensor: torch.Tensor, sr: int) -> None:
    arr = audio_tensor.numpy().T  # (samples, channels)
    sf.write(str(path), arr, sr, subtype="PCM_24")


def separate_file(model, track_path: pathlib.Path, out_dir: pathlib.Path) -> None:
    print(f"  Loading {track_path.name}...", flush=True)
    audio, sr = load_audio_sf(str(track_path))

    # Resample to model's expected sample rate if needed
    if sr != model.samplerate:
        import torchaudio.transforms as T
        resampler = T.Resample(sr, model.samplerate)
        audio = resampler(audio)
        sr = model.samplerate

    # apply_model expects (batch, channels, samples)
    audio_batch = audio.unsqueeze(0)

    print(f"  Separating {track_path.name} ({audio.shape[1]/sr:.1f}s)...", flush=True)
    with torch.no_grad():
        sources = apply_model(model, audio_batch, progress=True)
    # sources: (batch=1, num_sources, channels, samples)
    sources = sources[0]  # (num_sources, channels, samples)

    stem_names = model.sources  # e.g. ['drums', 'bass', 'other', 'vocals']
    track_out = out_dir / track_path.stem
    track_out.mkdir(parents=True, exist_ok=True)

    for i, name in enumerate(stem_names):
        if name in ("vocals", "no_vocals"):
            out_path = track_out / f"{name}.wav"
            save_stem(out_path, sources[i], sr)
            print(f"  Saved {out_path.relative_to(out_dir.parent.parent.parent)}", flush=True)

    # Also save no_vocals as sum of all non-vocal stems
    if "vocals" in stem_names and "no_vocals" not in stem_names:
        vocal_idx = stem_names.index("vocals")
        no_vocal = sum(sources[i] for i in range(len(stem_names)) if i != vocal_idx)
        save_stem(track_out / "no_vocals.wav", no_vocal, sr)
        print(f"  Saved {(track_out / 'no_vocals.wav').relative_to(out_dir.parent.parent.parent)}", flush=True)


def main():
    tracks_dir = pathlib.Path("D:/APKI/BeatFlow/tests/fixtures/reference_tracks")
    out_dir = tracks_dir / "separated"
    out_dir.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(tracks_dir.glob("*.wav"))
    if not wav_files:
        print("No .wav files found in", tracks_dir)
        sys.exit(1)

    print(f"Loading model {MODEL_NAME}...", flush=True)
    model = get_model(MODEL_NAME)
    model.eval()

    for track in wav_files:
        print(f"\n[{track.name}]", flush=True)
        separate_file(model, track, out_dir)

    print("\nDone. All tracks separated.")


if __name__ == "__main__":
    main()
