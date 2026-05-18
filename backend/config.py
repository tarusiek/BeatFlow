from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    project_root: Path = Path(__file__).parent.parent
    uploads_dir: Path = Path(__file__).parent.parent / "uploads"
    outputs_dir: Path = Path(__file__).parent.parent / "outputs"
    database_url: str = "sqlite:///./beatflow.db"
    target_vocal_lufs: float = -20.0
    target_beat_lufs: float = -16.0
    target_mix_lufs: float = -14.0
    max_upload_bytes: int = 200 * 1024 * 1024  # 200 MB per file

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    def ensure_dirs(self) -> None:
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
