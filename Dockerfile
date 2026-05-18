FROM python:3.12-slim

# System deps for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml requirements.txt ./
RUN pip install --no-cache-dir -e ".[dev]" \
    && pip install --no-cache-dir uvicorn[standard] fastapi sqlalchemy pydantic-settings aiofiles python-multipart

COPY backend/ ./backend/
COPY configs/ ./configs/
COPY conftest.py ./

RUN mkdir -p uploads outputs

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
