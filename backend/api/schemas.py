from __future__ import annotations
from datetime import datetime
from typing import Any
from pydantic import BaseModel


class JobResponse(BaseModel):
    id: str
    status: str
    preset: str | None
    before_vocal_lufs: float | None
    after_vocal_lufs: float | None
    before_mix_lufs: float | None
    after_mix_lufs: float | None
    params: dict[str, Any] | None
    error_message: str | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class UploadResponse(BaseModel):
    job_id: str
    message: str
