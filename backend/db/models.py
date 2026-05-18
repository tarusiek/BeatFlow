import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Float, DateTime, Text
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class Job(Base):
    __tablename__ = "jobs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    status = Column(String(20), nullable=False, default="pending")
    # pending | processing | done | error

    beat_path = Column(String(512), nullable=False)
    vocal_path = Column(String(512), nullable=False)
    output_path = Column(String(512), nullable=True)

    preset = Column(String(32), nullable=True, default="default")
    reference_path = Column(String(512), nullable=True)

    before_vocal_lufs = Column(Float, nullable=True)
    after_vocal_lufs = Column(Float, nullable=True)
    before_mix_lufs = Column(Float, nullable=True)
    after_mix_lufs = Column(Float, nullable=True)

    params_json = Column(Text, nullable=True)  # JSON-encoded decide_params result
    error_message = Column(Text, nullable=True)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc),
                        onupdate=lambda: datetime.now(timezone.utc))
