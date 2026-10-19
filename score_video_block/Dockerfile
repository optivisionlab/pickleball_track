# syntax=docker/dockerfile:1

# ==========================================
# Builder — install dependencies only
# ==========================================
FROM python:3.11-slim AS builder

WORKDIR /build

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# CUDA 12.8 wheels (bundle their CUDA runtime; the host only needs the NVIDIA
# driver). CPU-only image: --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
ARG TORCH_VERSION=2.10.0
ARG TORCHVISION_VERSION=0.25.0
ARG TORCHAUDIO_VERSION=2.10.0

# PyTorch first: the largest layer, rebuilt only when these arguments change.
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m venv /opt/venv \
    && /opt/venv/bin/pip install \
        torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION} \
        --index-url ${TORCH_INDEX}

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/venv/bin/pip install -r requirements.txt

# ==========================================
# Runtime — minimal production image
# ==========================================
FROM python:3.11-slim AS runtime

WORKDIR /app

# ffmpeg/ffprobe decode audio; glib is needed by opencv-python-headless.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
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

# BEATs source (vendor/unilm, cloned before building; see README) and the
# application, laid out as in the repository so relative paths in .env
# (src/weights/..., vendor/unilm) resolve from /app. Weights are mounted.
COPY --chown=agent:agent vendor/ ./vendor/
COPY --chown=agent:agent src/ ./src/

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app/src \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/tmp/huggingface \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PICKLEBALL_CHECKPOINT=/app/src/weights/best.pt \
    PICKLEBALL_BEATS_CHECKPOINT=/app/src/weights/BEATs_iter3_plus_AS2M.pt \
    PICKLEBALL_BEATS_ROOT=/app/vendor/unilm \
    PICKLEBALL_VIDEOMAE=/app/src/weights/videomae-base

USER 10001:10001

EXPOSE 8888

# Startup loads three models, warms up the GPU and spawns preprocessing workers.
HEALTHCHECK \
    --interval=30s \
    --timeout=10s \
    --start-period=120s \
    --retries=3 \
    CMD python -c \
    "import urllib.request; urllib.request.urlopen('http://localhost:8888/ready', timeout=5)" \
    || exit 1

# One worker per GPU: the models are loaded once per process.
CMD ["uvicorn", "pickleball_track.api.main:app", "--host", "0.0.0.0", "--port", "8888", "--workers", "1"]
