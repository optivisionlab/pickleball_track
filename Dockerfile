FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
# For GPU hosts, install the CUDA build of torch first, e.g.
#   --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu121
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir torch torchvision --index-url ${TORCH_INDEX} \
    && pip install --no-cache-dir .

# Model artifacts are not part of the image; mount them at runtime:
#   -v $PWD/src/weights:/app/src/weights -v $PWD/vendor/unilm:/app/vendor/unilm
ENV PICKLEBALL_CHECKPOINT=/app/src/weights/best.pt \
    PICKLEBALL_BEATS_CHECKPOINT=/app/src/weights/BEATs_iter3_plus_AS2M.pt \
    PICKLEBALL_BEATS_ROOT=/app/vendor/unilm \
    PICKLEBALL_VIDEOMAE=/app/src/weights/videomae-base

RUN useradd --create-home appuser
USER appuser
EXPOSE 8000
CMD ["uvicorn", "pickleball_track.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
