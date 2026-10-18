# syntax=docker/dockerfile:1

# ==========================================
# Builder — install dependencies only
# ==========================================
FROM python:3.11-slim AS builder

WORKDIR /build

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir \
        torch torchvision torchaudio \
        --index-url ${TORCH_INDEX}

RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/venv/bin/pip install --no-cache-dir \
        -r requirements.txt

COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/venv/bin/pip install --no-cache-dir .

# ==========================================
# Runtime — minimal production image
# ==========================================
FROM python:3.11-slim AS runtime

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --system --gid 10001 agent \
    && useradd \
        --system \
        --uid 10001 \
        --gid 10001 \
        --no-create-home \
        agent

COPY --from=builder /opt/venv /opt/venv

COPY --chown=agent:agent vendor/ ./vendor/

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PICKLEBALL_CHECKPOINT=/app/src/weights/best.pt \
    PICKLEBALL_BEATS_CHECKPOINT=/app/src/weights/BEATs_iter3_plus_AS2M.pt \
    PICKLEBALL_BEATS_ROOT=/app/vendor/unilm \
    PICKLEBALL_VIDEOMAE=/app/src/weights/videomae-base

USER 10001:10001

EXPOSE 8888

HEALTHCHECK \
    --interval=30s \
    --timeout=10s \
    --start-period=60s \
    --retries=3 \
    CMD python -c \
    "import urllib.request; urllib.request.urlopen('http://localhost:8888/ready', timeout=5)" \
    || exit 1

CMD ["uvicorn", "pickleball_track.api.main:app", "--host", "0.0.0.0", "--port", "8888", "--workers", "1"]
