"""
Runs the full audio pipeline for a Job and updates the DB row.
Called from FastAPI BackgroundTasks so it runs off the request thread.
"""
import json
import logging
import traceback
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.orm import Session

from backend.audio.io import load_audio, save_audio
from backend.audio.features import extract_features
from backend.audio.decisions import decide_params
from backend.audio.vocal_chain import apply_vocal_chain
from backend.audio.mixer import balance_and_sum
from backend.audio.master import apply_master_chain
from backend.audio.loudness import measure_lufs
from backend.config import settings

log = logging.getLogger(__name__)


def _now():
    return datetime.now(timezone.utc)


def run_pipeline(job_id: str, db: Session) -> None:
    from backend.db.models import Job

    job = db.get(Job, job_id)
    if job is None:
        log.error("Job %s not found", job_id)
        return

    try:
        job.status = "processing"
        job.updated_at = _now()
        db.commit()

        sr = 44100
        beat = load_audio(job.beat_path, sr)
        vocal = load_audio(job.vocal_path, sr)

        before_vocal_lufs = measure_lufs(vocal, sr)
        job.before_vocal_lufs = before_vocal_lufs

        features = extract_features(vocal, sr)
        params = decide_params(features)

        # Apply reference track adjustment if provided
        if job.reference_path:
            ref = load_audio(job.reference_path, sr)
            ref_feats = extract_features(ref, sr)
            _adjust_params_for_reference(params, features, ref_feats)

        # Apply preset overrides
        if job.preset and job.preset != "default":
            _apply_preset(params, job.preset)

        job.params_json = json.dumps(params)

        vocal_processed = apply_vocal_chain(vocal, sr, params)
        mix = balance_and_sum(beat, vocal_processed, sr)
        master = apply_master_chain(mix, sr)

        out_path = settings.outputs_dir / f"{job_id}_output.wav"
        save_audio(str(out_path), master, sr)

        job.output_path = str(out_path)
        job.after_vocal_lufs = measure_lufs(vocal_processed, sr)
        job.before_mix_lufs = before_vocal_lufs  # raw mix would be this
        job.after_mix_lufs = measure_lufs(master, sr)
        job.status = "done"
        job.updated_at = _now()
        db.commit()

    except Exception:
        log.exception("Pipeline failed for job %s", job_id)
        try:
            job.status = "error"
            job.error_message = traceback.format_exc()
            job.updated_at = _now()
            db.commit()
        except Exception:
            pass


# ── Preset & reference adjustments ────────────────────────────────────────────

_PRESETS: dict[str, dict] = {
    "hip_hop": {
        "comp_ratio": 4.0,
        "comp_threshold": -18.0,
        "comp_attack_ms": 6.0,
        "comp_release_ms": 60.0,
        "presence_freq": 4000.0,
        "presence_gain": 2.5,
    },
    "rnb": {
        "comp_ratio": 2.5,
        "comp_threshold": -22.0,
        "comp_attack_ms": 15.0,
        "comp_release_ms": 150.0,
        "presence_freq": 3500.0,
        "presence_gain": 2.0,
    },
    "pop": {
        "comp_ratio": 3.0,
        "comp_threshold": -20.0,
        "comp_attack_ms": 10.0,
        "comp_release_ms": 100.0,
        "presence_freq": 5000.0,
        "presence_gain": 1.5,
    },
}


def _apply_preset(params: dict, preset: str) -> None:
    overrides = _PRESETS.get(preset, {})
    params.update(overrides)


def _adjust_params_for_reference(params: dict, vocal_feats: dict, ref_feats: dict) -> None:
    """Nudge compressor and EQ toward reference track characteristics."""
    dr_diff = ref_feats["dynamic_range_db"] - vocal_feats["dynamic_range_db"]
    if abs(dr_diff) > 3.0:
        ratio_delta = -0.3 * (dr_diff / abs(dr_diff))
        params["comp_ratio"] = max(1.5, min(6.0, params["comp_ratio"] + ratio_delta))

    centroid_diff = ref_feats["spectral_centroid"] - vocal_feats["spectral_centroid"]
    if abs(centroid_diff) > 500.0:
        gain_delta = 0.5 * (centroid_diff / abs(centroid_diff))
        params["presence_gain"] = max(0.5, min(5.0, params["presence_gain"] + gain_delta))
