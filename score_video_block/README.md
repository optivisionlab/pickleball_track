# Pickleball Track Inference API

FastAPI service for scoring pickleball videos stored in MinIO. An external
backend calls `POST /api/v1/process`; the service downloads the referenced
video, runs temporal hit/miss inference and returns the job score and metrics.

All paths and commands below are relative to this folder
(`pickleball_track/score_video_block/`); run them from here.

## Architecture

```text
src/pickleball_track/
  api/
    main.py              create_app(): lifespan loads models once, registers routes
    dependencies.py      request access to settings / services held in app.state
    errors.py            domain error -> HTTP status mapping ({"detail": ...})
    routes/health.py     GET /health, GET /ready
    routes/inference.py  POST /api/v1/process
    schemas/             Pydantic request / response contract
  core/
    config.py            Settings (env vars) + model constants fixed by training
    exceptions.py        domain errors (no FastAPI dependency)
    logging.py           JSON logging
  integrations/minio.py  MinIO client construction and endpoint validation
  models/
    architecture.py      AVFeatureModel (trained fusion model)
    components/          projection, cross-attention fusion, temporal transformer,
                         prediction head, VideoMAE and BEATs wrappers
    model_loader.py      backbone + checkpoint loading, ModelBundle
  pipelines/
    frames.py            sequential decoding, preprocessing (threads or ProcessFramePool), window batches
    frame_workers.py     exact resize/crop/normalize + worker process (NumPy/Pillow only)
    preprocessing.py     video metadata, audio decoding, sliding windows, audio windows
    postprocessing.py    window aggregation, smoothing, hit/miss event decoding
  services/
    inference_service.py VideoInferenceService: whole-video inference
    video_jobs.py        VideoJobService: download, size checks, scoring
scripts/infer_video.py   command-line inference on a local video (replaces inference.py)
tests/unit, tests/integration, tests/fixtures
```

### Model

The trained model is audiovisual:

1. VideoMAE (local `src/weights/videomae-base`, frozen) encodes 32-frame windows.
   Its temporal position embeddings are interpolated from 16 to 32 frames; the
   spatial tokens are mean-pooled into 16 temporal tokens (one per tubelet).
2. BEATs (`BEATs_iter3_plus_AS2M.pt`, frozen, source from microsoft/unilm)
   encodes the window's 16 kHz mono audio.
3. The trained `AVFeatureModel` (`best.pt`) projects both modalities to 768-d,
   fuses them with cross-attention, adds temporal positions, runs a 2-layer
   temporal transformer and a 3-class head per temporal token.

Labels are `no_event=0`, `hit=1`, `miss=2`; the checkpoint's `label2id` is
checked when present. Audio is decoded with FFmpeg at its native rate,
averaged to mono and resampled to 16 kHz with a port of
`torchaudio.functional.resample` (`pipelines/resample.py`), reproducing the
original `torchaudio.load` + `resample` steps without depending on torchaudio. Overlapping token probabilities are averaged, smoothed,
thresholded and merged into hit/miss events using the inference
hyperparameters below.

### Model lifecycle

All models are loaded once per worker in the FastAPI lifespan, put in `eval()`
mode with gradients disabled, and inference runs under `torch.inference_mode()`.
The fusion checkpoint is loaded with `strict=True`; its projection input sizes
are read from the checkpoint tensors and verified against VideoMAE's hidden
size and BEATs' feature size at startup. If loading fails the process keeps
running, the error is logged, `/ready` returns `503` and `/api/v1/process`
returns `503`. Inference is serialized per worker with a lock; the sync route
runs in FastAPI's thread pool so the event loop is never blocked. Run one
uvicorn worker per GPU.

## Setup

Requirements: Python ≥ 3.10, FFmpeg, PyTorch/TorchVision matching your
accelerator, the model artifacts and the BEATs source checkout. Large weights
are not tracked by Git; place them as described in `src/weights/README.md`.

```bash
# from score_video_block/
git clone --depth 1 --filter=blob:none --sparse https://github.com/microsoft/unilm.git vendor/unilm
git -C vendor/unilm sparse-checkout set beats   # BEATs source only (~4 MB)

conda activate score-video
python -m pip install -r requirements-gpu.txt   # CUDA 12.8 PyTorch; requirements.txt for CPU
python -m pip install -e ".[dev]"
cp .env.example .env   # then fill in MinIO credentials
```

## Configuration

`.env` in this folder (`score_video_block/.env`) is loaded at startup; process environment
variables take precedence.

| Variable | Default | Purpose |
| --- | --- | --- |
| `PICKLEBALL_CHECKPOINT` | `src/weights/best.pt` | Trained fusion checkpoint |
| `PICKLEBALL_BEATS_CHECKPOINT` | `src/weights/BEATs_iter3_plus_AS2M.pt` | Pretrained BEATs weights |
| `PICKLEBALL_BEATS_ROOT` | `vendor/unilm` | BEATs source checkout |
| `PICKLEBALL_VIDEOMAE` | `src/weights/videomae-base` | Local model directory or model id |
| `PICKLEBALL_VIDEOMAE_REPO` | `MCG-NJU/videomae-base` | Hub repo whose files missing from the local directory are downloaded at startup; empty = offline |
| `PICKLEBALL_DEVICE` | `auto` | `auto`, `cpu`, or `cuda` |
| `PICKLEBALL_BATCH_SIZE` | `0` | Sliding windows per forward pass; `0` = 16 on CUDA, 4 on CPU |
| `PICKLEBALL_PRECISION` | `auto` | VideoMAE autocast: `auto` (fp16 on CUDA, fp32 on CPU), `fp32`, `fp16`, `bf16` |
| `PICKLEBALL_GPU_PREPROCESS` | `false` | Resize/normalize frames on the GPU (pixels may differ by one 8-bit level) |
| `PICKLEBALL_PREPROCESS_WORKERS` | `8` | Exact frame preprocessing processes (shared memory); `0` = threads in the API process |
| `PICKLEBALL_MAX_VIDEO_BYTES` | `536870912` | Maximum MinIO object size |
| `PICKLEBALL_INFERENCE_SLOTS` | `2` | Videos analysed at the same time; each slot has its own `PICKLEBALL_PREPROCESS_WORKERS` processes (~210 MB of `/dev/shm` at 1080p) and shares the loaded models |
| `PICKLEBALL_MAX_PENDING_JOBS` | `8` | Requests accepted at once (downloading, waiting for a slot or running); more get `429` |
| `PICKLEBALL_BUSY_RETRY_AFTER_SECONDS` | `10` | `Retry-After` sent with `429` |
| `PICKLEBALL_DOWNLOAD_PARTS` | `4` | Parallel range requests per video download |
| `PICKLEBALL_DOWNLOAD_PART_MIN_BYTES` | `8388608` | Smallest part; smaller videos use one request |
| `PICKLEBALL_RESULT_CACHE_SECONDS` | `600` | Completed results kept by `job_id` for client retries; `0` disables |
| `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY` | unset | MinIO credentials (or `MINIO_ROOT_USER` / `MINIO_ROOT_PASSWORD`) |
| `MINIO_SECURE` | `false` | TLS for an endpoint given without scheme |
| `MINIO_PUBLIC_ENDPOINT` | unset | If set, always download from this endpoint instead of the request's `minio_endpoint` |
| `LOG_LEVEL` | `INFO` | Python logging level |

### Inference hyperparameters

Defined in `src/pickleball_track/core/config.py` (section *INFERENCE
HYPERPARAMETERS*). Edit the defaults there, or override per deployment with
environment variables; the CLI flags override both for one run. Defaults equal
the original `inference.py`.

| Constant | Env variable | Default | CLI flag | Effect |
| --- | --- | --- | --- | --- |
| `HIT_THRESHOLD` | `PICKLEBALL_HIT_THRESHOLD` | `0.8` | `--hit-threshold` | min. smoothed probability for `hit` |
| `MISS_THRESHOLD` | `PICKLEBALL_MISS_THRESHOLD` | `0.8` | `--miss-threshold` | min. smoothed probability for `miss` |
| `EVENT_MIN_GAP` | `PICKLEBALL_EVENT_MIN_GAP` | `0.5` | `--event-gap` | seconds; joins/merges same-class events |
| `TEMPORAL_SMOOTH_RADIUS` | `PICKLEBALL_SMOOTH_RADIUS` | `1` | `--smooth-radius` | moving-average radius, `0` disables |
| `OVERLAP` | `PICKLEBALL_OVERLAP` | `0.5` | `--overlap` | sliding-window overlap in `[0, 1)` |

`NUM_VIDEO_FRAMES`, `NUM_TEMPORAL_TOKENS`, `AUDIO_SR` and `LABEL2ID` in the
*MODEL CONSTANTS* section are fixed by the trained checkpoint and must not be
changed.

Credentials are never logged and are hidden from `repr(Settings)`.

### GPU performance

On CUDA the pipeline is built for throughput:

- The video is decoded **once, sequentially**, straight into a shared-memory ring.
  Per-window seeking was removed: on variable-frame-rate phone videos
  `CAP_PROP_POS_FRAMES` lands up to a second away from the requested frame. On
  such videos the audio window follows the decoded frame timestamps;
  constant-frame-rate videos keep `frame / fps` as in training.
- `PICKLEBALL_PREPROCESS_WORKERS` worker processes rotate (portrait phone
  videos), resize, crop and normalize each frame once (overlapping windows share
  frames) while the GPU runs the previous batch. The result is **bit-identical**
  to `VideoMAEImageProcessor` on OpenCV's frames (tested, including all display
  rotations). Each worker has its own pipes; a crashed worker fails that request
  with a `502` and is restarted for the next one.
- `PICKLEBALL_BATCH_SIZE` windows go through VideoMAE, BEATs and the fusion model
  together. VideoMAE runs under fp16 autocast (`PICKLEBALL_PRECISION`); BEATs
  (its fbank overflows fp16) and the fusion model stay fp32.
- Models are warmed up at startup, so the first request is not slower.

Measured on an NVIDIA A30 (24-core Xeon Silver 4310, shared) through
`POST /api/v1/process` with a real MinIO download:

| Video | Before | Threads (`PREPROCESS_WORKERS=0`) | 8 worker processes (default) |
| --- | --- | --- | --- |
| 3 s, 1080×1920 (91 frames, one keyframe) | 4.6–4.8 s | 1.0–1.2 s | 1.1–1.2 s |
| 47 s, 1080×1920, constant frame rate | 82 s | 8.6–9.8 s | 7.5–9.7 s |
| 65 s phone video, variable frame rate, rotated | 108 s | 13.5–15.9 s | 11.4–11.8 s |

Hit/miss counts on constant-frame-rate videos are unchanged; token
probabilities differ by ≤ 0.003 from fp32 (`PICKLEBALL_PRECISION=fp32` for
exact fp32). On variable-frame-rate videos results change because the frames
are now the right ones. Remaining time is CPU: decoding plus exact resizing
costs ~25 ms of CPU per 1080p frame; batch sizes above 16 do not help.

Worker processes exchange frames through `/dev/shm` (~10 MB per worker for
1080p). Docker's default 64 MB is too small: run containers with
`--shm-size=1g`; otherwise the service logs a warning and uses threads.

## Run

```bash
python -m uvicorn pickleball_track.api.main:app --host 0.0.0.0 --port 8000
# or: python -m pickleball_track.api
```

Docker (GPU; needs the NVIDIA Container Toolkit). `vendor/unilm` must be cloned
first (it is copied into the image); the weights are mounted read-only:

```bash
docker compose up -d --build      # builds pickleball-track:gpu, serves :8888
docker compose logs -f            # wait for "Models loaded"
curl localhost:8888/ready
```

`docker-compose.yml` sets `--gpus`, `shm_size: 1gb` (preprocessing workers),
`env_file: .env` and the weights volume. The image keeps the repository layout
under `/app` (`PYTHONPATH=/app/src`), so relative paths in `.env` work unchanged.
CPU-only image: `docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu -t pickleball-track .`
and run it with `--shm-size=1g -e PICKLEBALL_DEVICE=cpu`.

Command-line inference on a local file (writes `<video>.inference.json`):

```bash
python scripts/infer_video.py --video match.mp4 [--checkpoint best.pt --hit-threshold 0.8 --miss-threshold 0.8 --event-gap 0.5 --smooth-radius 1 --overlap 0.5]
```

## API Contract

| Endpoint | Success | Failure |
| --- | --- | --- |
| `GET /health` | `200 {"status": "ok"}` | – |
| `GET /ready` | `200 {"status": "ready"}` when models are loaded | `503` |
| `POST /api/v1/process` | `200` job response | see below |
| `GET /stats` | `200` slots, pending jobs and cached results | – |

Request (only `.mp4` / `.mov` object keys are accepted):

```json
{
  "job_id": "job-1",
  "result_id": "result-1",
  "task_id": "task-1",
  "course_id": "course-1",
  "sport_id": "sport-1",
  "sport_code": "PB",
  "mode": "VIDEO",
  "bucket": "videos",
  "object_key": "matches/match.MOV",
  "video_url": "https://storage.example/matches/match.MOV",
  "minio_endpoint": "https://minio.example:9000",
  "user_id": "user-1",
  "attempt": 1,
  "schema_version": 1,
  "detail": {"title": "Match", "category": "serve", "gradingMethod": "ai", "timeLimit": "60"}
}
```

Response:

```json
{
  "jobId": "job-1",
  "resultId": "result-1",
  "status": "completed",
  "score": 8.0,
  "feedback": "Thực hiện động tác đạt yêu cầu.",
  "metrics": {"accuracy": 0.8, "hitCount": 8, "missCount": 2}
}
```

`accuracy = hitCount / (hitCount + missCount)` (0 when there are no actions),
`score = accuracy × 10`; `no_event` is not counted as a failed action.
Errors use FastAPI's `{"detail": "..."}` body: `422` invalid request, unsupported
video type or invalid `minio_endpoint`; `413` video too large; `503` model not
loaded or MinIO not configured; `502` download or inference failure (including
unreadable video or missing audio track); `429` with `Retry-After` when
`PICKLEBALL_MAX_PENDING_JOBS` requests are already in progress. Failures are never
reported as `no_event`.

### Concurrency

The endpoint is `async`: each job runs on a dedicated executor (download, then a free
inference slot), so waiting requests hold no event-loop thread. Up to
`PICKLEBALL_INFERENCE_SLOTS` videos are analysed at once; frame decoding is
CPU-bound, so a second slot keeps the GPU busy while the first decodes. Beyond
`PICKLEBALL_MAX_PENDING_JOBS` the service answers `429` + `Retry-After` immediately
instead of letting requests queue until the client times out.

A request whose `job_id` (same bucket/object) is already running waits for that run,
and completed results are cached for `PICKLEBALL_RESULT_CACHE_SECONDS`: a client that
retries after a timeout or a dropped connection gets the result without a second
inference. Jobs are not cancelled when the client disconnects.

The backend (`haui_sict_fitness_score`) treats `429`/`503` with `Retry-After` as busy
(waits and resends without spending its retry budget) and scales its timeout with
video duration and file size. Keep its `AI_MAX_CONCURRENCY` × worker replicas at or
below `PICKLEBALL_MAX_PENDING_JOBS`.

## Tests

```bash
python -m pytest -q
python -m ruff check src tests scripts
```

- `tests/unit`: labels, settings, schemas, preprocessing shapes (with the
  repository's VideoMAE processor config), event decoding, checkpoint loading
  (strict key/label checks on synthetic checkpoints), VideoMAE wrapper, job
  scoring and error handling, MinIO endpoint validation.
- `tests/integration`: API with injected services, the full pipeline on a small
  real video with tiny VideoMAE/fusion weights and a controlled BEATs fake, and
  regression against outputs recorded from the original `inference.py`
  (`tests/fixtures/golden_reference.json`).
- `tests/integration/test_real_artifacts.py` runs only when the real artifacts
  exist; set `PICKLEBALL_TEST_VIDEO` to also run real inference.
