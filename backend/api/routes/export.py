from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from backend.db.models import Job
from backend.db.session import get_db

router = APIRouter(prefix="/api", tags=["export"])


@router.get("/export/{job_id}/output.wav")
def download_output(job_id: str, db: Session = Depends(get_db)):
    job = db.get(Job, job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job.status != "done" or not job.output_path:
        raise HTTPException(425, "Processing not finished yet")
    path = Path(job.output_path)
    if not path.exists():
        raise HTTPException(500, "Output file missing on server")
    return FileResponse(str(path), media_type="audio/wav",
                        filename="beatflow_mix.wav")


@router.get("/export/{job_id}/beat.wav")
def download_beat(job_id: str, db: Session = Depends(get_db)):
    job = db.get(Job, job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    path = Path(job.beat_path)
    if not path.exists():
        raise HTTPException(500, "Beat file missing")
    return FileResponse(str(path), media_type="audio/wav", filename="beat.wav")


@router.get("/export/{job_id}/vocal_raw.wav")
def download_vocal_raw(job_id: str, db: Session = Depends(get_db)):
    job = db.get(Job, job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    path = Path(job.vocal_path)
    if not path.exists():
        raise HTTPException(500, "Vocal file missing")
    return FileResponse(str(path), media_type="audio/wav", filename="vocal_raw.wav")
