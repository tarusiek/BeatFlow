import uuid
import shutil
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from backend.api.schemas import UploadResponse
from backend.config import settings
from backend.db.models import Job
from backend.db.session import get_db
from backend.workers.pipeline import run_pipeline

router = APIRouter(prefix="/api", tags=["upload"])

ALLOWED_TYPES = {"audio/wav", "audio/x-wav", "audio/mpeg", "audio/flac",
                 "audio/ogg", "application/octet-stream"}


def _save_upload(file: UploadFile, dest: Path) -> None:
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)


@router.post("/upload", response_model=UploadResponse)
async def upload_files(
    background_tasks: BackgroundTasks,
    beat: UploadFile = File(..., description="Beat WAV/MP3"),
    vocal: UploadFile = File(..., description="Raw vocal WAV/MP3"),
    preset: str = Form("default"),
    db: Session = Depends(get_db),
):
    job_id = str(uuid.uuid4())
    job_dir = settings.uploads_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    beat_ext = Path(beat.filename or "beat.wav").suffix or ".wav"
    vocal_ext = Path(vocal.filename or "vocal.wav").suffix or ".wav"

    beat_path = job_dir / f"beat{beat_ext}"
    vocal_path = job_dir / f"vocal{vocal_ext}"

    _save_upload(beat, beat_path)
    _save_upload(vocal, vocal_path)

    job = Job(
        id=job_id,
        beat_path=str(beat_path),
        vocal_path=str(vocal_path),
        preset=preset if preset in ("default", "hip_hop", "rnb", "pop") else "default",
    )
    db.add(job)
    db.commit()

    # Run pipeline in background (off request thread)
    background_tasks.add_task(_run_in_new_session, job_id)

    return UploadResponse(job_id=job_id, message="Job queued. Poll /api/jobs/{job_id} for status.")


@router.post("/upload/reference/{job_id}", response_model=UploadResponse)
async def upload_reference(
    job_id: str,
    background_tasks: BackgroundTasks,
    reference: UploadFile = File(..., description="Reference track WAV/MP3"),
    db: Session = Depends(get_db),
):
    job = db.get(Job, job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job.status == "done":
        raise HTTPException(409, "Job already completed. Create a new job with the reference.")

    ref_ext = Path(reference.filename or "ref.wav").suffix or ".wav"
    ref_path = settings.uploads_dir / job_id / f"reference{ref_ext}"
    _save_upload(reference, ref_path)

    job.reference_path = str(ref_path)
    db.commit()

    return UploadResponse(job_id=job_id, message="Reference uploaded. Pipeline will use it.")


def _run_in_new_session(job_id: str, session_factory=None) -> None:
    """Create a fresh DB session for the background task (can't share across threads)."""
    if session_factory is None:
        from backend.db.session import SessionLocal as _SL
        session_factory = _SL
    db = session_factory()
    try:
        run_pipeline(job_id, db)
    finally:
        db.close()
