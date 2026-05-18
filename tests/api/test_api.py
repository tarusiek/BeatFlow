"""Integration tests for the BeatFlow REST API."""
import io
import time
import numpy as np
import soundfile as sf
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.main import app
from backend.db.models import Base
from backend.db.session import get_db
import backend.api.routes.upload as upload_module

# ── Shared in-memory SQLite for all tests ───────────────────────────────────
TEST_DATABASE_URL = "sqlite://"

test_engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


def override_get_db():
    db = TestSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db

# Patch background task session factory to use test DB
upload_module._TEST_SESSION_FACTORY = TestSessionLocal


def _patched_run_in_new_session(job_id: str, session_factory=None) -> None:
    upload_module._original_run_in_new_session(
        job_id, session_factory=TestSessionLocal
    )


upload_module._original_run_in_new_session = upload_module._run_in_new_session
upload_module._run_in_new_session = _patched_run_in_new_session


@pytest.fixture(autouse=True, scope="module")
def setup_db():
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)


@pytest.fixture(scope="module")
def client(setup_db):
    with TestClient(app) as c:
        yield c


# ── Helpers ──────────────────────────────────────────────────────────────────

def _wav_bytes(duration_s: float = 2.0, sr: int = 44100) -> bytes:
    """Generate a minimal valid stereo WAV in memory."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    audio = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    stereo = np.stack([audio, audio], axis=1)  # (N, 2)
    buf = io.BytesIO()
    sf.write(buf, stereo, sr, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


def _wait_for_done(client, job_id: str, timeout: int = 30) -> dict:
    for _ in range(timeout * 2):
        r = client.get(f"/api/jobs/{job_id}")
        assert r.status_code == 200
        if r.json()["status"] in ("done", "error"):
            return r.json()
        time.sleep(0.5)
    return client.get(f"/api/jobs/{job_id}").json()


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_upload_returns_job_id(client):
    wav = _wav_bytes()
    r = client.post(
        "/api/upload",
        files={"beat": ("beat.wav", wav, "audio/wav"),
                "vocal": ("vocal.wav", wav, "audio/wav")},
        data={"preset": "default"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "job_id" in body
    assert len(body["job_id"]) == 36  # UUID


def test_get_job_status_completes(client):
    wav = _wav_bytes()
    r = client.post(
        "/api/upload",
        files={"beat": ("beat.wav", wav, "audio/wav"),
                "vocal": ("vocal.wav", wav, "audio/wav")},
        data={"preset": "default"},
    )
    job_id = r.json()["job_id"]
    final = _wait_for_done(client, job_id)
    assert final["status"] == "done", f"Job failed: {final.get('error_message')}"
    assert final["after_mix_lufs"] is not None


def test_get_job_not_found(client):
    r = client.get("/api/jobs/00000000-0000-0000-0000-000000000000")
    assert r.status_code == 404


def test_preset_hip_hop(client):
    wav = _wav_bytes()
    r = client.post(
        "/api/upload",
        files={"beat": ("beat.wav", wav, "audio/wav"),
                "vocal": ("vocal.wav", wav, "audio/wav")},
        data={"preset": "hip_hop"},
    )
    job_id = r.json()["job_id"]
    final = _wait_for_done(client, job_id)
    assert final["status"] == "done"
    assert final["preset"] == "hip_hop"


def test_download_output(client):
    wav = _wav_bytes()
    r = client.post(
        "/api/upload",
        files={"beat": ("beat.wav", wav, "audio/wav"),
                "vocal": ("vocal.wav", wav, "audio/wav")},
        data={"preset": "default"},
    )
    job_id = r.json()["job_id"]
    _wait_for_done(client, job_id)
    dl = client.get(f"/api/export/{job_id}/output.wav")
    assert dl.status_code == 200
    assert "audio" in dl.headers["content-type"]
    assert len(dl.content) > 1000
