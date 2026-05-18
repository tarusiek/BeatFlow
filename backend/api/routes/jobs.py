import json
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.api.schemas import JobResponse
from backend.db.models import Job
from backend.db.session import get_db

router = APIRouter(prefix="/api", tags=["jobs"])


def _job_to_response(job: Job) -> JobResponse:
    params = None
    if job.params_json:
        try:
            params = json.loads(job.params_json)
        except Exception:
            pass
    return JobResponse(
        id=job.id,
        status=job.status,
        preset=job.preset,
        before_vocal_lufs=job.before_vocal_lufs,
        after_vocal_lufs=job.after_vocal_lufs,
        before_mix_lufs=job.before_mix_lufs,
        after_mix_lufs=job.after_mix_lufs,
        params=params,
        error_message=job.error_message,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


@router.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str, db: Session = Depends(get_db)):
    job = db.get(Job, job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return _job_to_response(job)
