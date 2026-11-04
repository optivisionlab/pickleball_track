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
    architecture.py      AVTFeatureModel (trained v4 model) + FoldEnsemble
    components/          projection, cross-modal attention + gated fusion, temporal
                         transformer, text-anchored head, VideoMAE and BEATs wrappers
    model_loader.py      backbone + checkpoint loading, ModelBundle
  pipelines/
    frames.py            sequential decoding, 30 fps sampling, preprocessing (threads or ProcessFramePool), window batches
    frame_workers.py     training resize/normalize + worker process (NumPy/OpenCV only)
    preprocessing.py     video metadata, audio decoding, sliding windows
    postprocessing.py    window aggregation, hit/miss event decoding
  services/
    inference_service.py VideoInferenceService: whole-video inference
    video_jobs.py        VideoJobService: download, size checks, scoring
scripts/infer_video.py   command-line inference on a local video
scripts/export_checkpoint.py  pack a training run (5 folds) into pickleball_v4.pt
tests/unit, tests/integration, tests/fixtures
```

### Model

The served model is **v4** of `baseline/` (VideoMAE + BEATs + CLIP class
descriptions), the best of v3/v4/v5 in the 5-fold user-grouped evaluation
(event macro-F1 ±0.5 s: v3 0.931, **v4 0.951**, v5 0.838; v3 costs the same time,
v5 only skips BEATs, ~0.5% of the processing time). The pipeline reproduces
`baseline/pickleball_common.py`:

1. Frames are sampled by time at 30 fps (source frame `round(i * fps / 30)`,
   so 60 fps videos skip every other frame), auto-rotated like OpenCV, resized to
   224×224 with `cv2.INTER_AREA` (no crop) and ImageNet-normalized.
2. VideoMAE fine-tuned on Kinetics (`src/weights/videomae-base-finetuned-kinetics`,
   frozen; temporal position embeddings interpolated from 16 to 32 frames)
   encodes 32-frame windows (1.07 s, stride 16). Each of the 16 temporal tokens
   is the mean of all patches plus a 2×2 grid of patch means (3840-d).
3. BEATs (`BEATs_iter3_plus_AS2M.pt`, frozen) runs once over the whole 16 kHz
   mono recording in 10 s chunks; its 0.16 s time steps are linearly
   interpolated at every video token's centre time. A video without audio is
   scored from the video alone (as in training).
4. The 5 per-fold `AVTFeatureModel`s of `pickleball_v4.pt` each standardize the
   features with their own training statistics; their softmax outputs are
   averaged. Overlapping windows are averaged per token.
5. Runs of at least `MIN_EVENT_TOKENS` event tokens become hit/miss events
   (class by summed probability, time = run centre).

On 12 training videos (each scored by the fold model that held it out) served
token probabilities match the training test predictions to within 0.07 (99.9%
same argmax) and the events are identical.

Labels are `no_event=0`, `hit=1`, `miss=2`; the checkpoint's `label2id` is checked.

### Model lifecycle

All models are loaded once per worker in the FastAPI lifespan, put in `eval()`
mode with gradients disabled, and inference runs under `torch.inference_mode()`.
Every fold is loaded with `strict=True`; its projection input sizes are read
from the checkpoint tensors and verified against the VideoMAE token size
(hidden × 5) and BEATs' feature size at startup. If loading fails the process keeps
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
| `PICKLEBALL_CHECKPOINT` | `src/weights/pickleball_v4.pt` | Trained fold ensemble (`scripts/export_checkpoint.py`) |
| `PICKLEBALL_BEATS_CHECKPOINT` | `src/weights/BEATs_iter3_plus_AS2M.pt` | Pretrained BEATs weights |
| `PICKLEBALL_BEATS_ROOT` | `vendor/unilm` | BEATs source checkout |
| `PICKLEBALL_VIDEOMAE` | `src/weights/videomae-base-finetuned-kinetics` | Local model directory or model id |
| `PICKLEBALL_VIDEOMAE_REPO` | `MCG-NJU/videomae-base-finetuned-kinetics` | Hub repo whose files missing from the local directory are downloaded at startup; empty = offline |
| `PICKLEBALL_DEVICE` | `auto` | `auto`, `cpu`, or `cuda` |
| `PICKLEBALL_BATCH_SIZE` | `0` | Sliding windows per forward pass; `0` = 16 on CUDA, 4 on CPU |
| `PICKLEBALL_PRECISION` | `auto` | VideoMAE autocast: `auto` (fp16 on CUDA, fp32 on CPU), `fp32`, `fp16`, `bf16` |
| `PICKLEBALL_GPU_PREPROCESS` | `false` | Resize/normalize frames on the GPU (area interpolation only approximates `INTER_AREA`) |
| `PICKLEBALL_PREPROCESS_WORKERS` | `8` | Training-identical frame preprocessing processes (shared memory); `0` = threads in the API process |
| `PICKLEBALL_MAX_VIDEO_BYTES` | `536870912` | Maximum MinIO object size |
| `PICKLEBALL_INFERENCE_SLOTS` | `3` | Videos analysed at the same time; each slot has its own `PICKLEBALL_PREPROCESS_WORKERS` processes (~210 MB of `/dev/shm` at 1080p) and shares the loaded models |
| `PICKLEBALL_RESERVED_SHORT_SLOTS` | `1` | Slots only short videos may use (capped to slots - 1), so short clips never wait behind long ones |
| `PICKLEBALL_SHORT_VIDEO_SECONDS` | `20` | Longest video counted as short |
| `PICKLEBALL_SLOT_AGING_SECONDS` | `60` | Waiting videos get a slot shortest first; one waiting this long goes first |
| `PICKLEBALL_MAX_PENDING_JOBS` | `16` | Requests accepted at once (downloading, waiting for a slot or running); more get `429` |
| `PICKLEBALL_MAX_CONCURRENT_DOWNLOADS` | `4` | Accepted jobs downloading from MinIO at once |
| `PICKLEBALL_BUSY_RETRY_AFTER_SECONDS` | `10` | Minimum `Retry-After` sent with `429` (raised to half the average job time) |
| `PICKLEBALL_DECODER` | `opencv` | `opencv` (as in training) or `ffmpeg` (ffmpeg CLI: ~1.5x faster on 10-bit HEVC, same speed on H.264, but iPhone HDR frames come out ~15% darker: event macro-F1 0.971 → 0.961 on 12 training videos) |
| `PICKLEBALL_FFMPEG_THREADS` | `0` | ffmpeg decoding threads per video (`0` = automatic) |
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
environment variables; the CLI flags override both for one run.

| Constant | Env variable | Default | CLI flag | Effect |
| --- | --- | --- | --- | --- |
| `EVENT_THRESHOLD` | `PICKLEBALL_EVENT_THRESHOLD` | `0.4` | `--event-threshold` / `--argmax` | a token is an event when P(hit) + P(miss) exceeds it; `argmax` = argmax is hit/miss |
| `MIN_EVENT_TOKENS` | `PICKLEBALL_MIN_EVENT_TOKENS` | `8` | `--min-event-tokens` | shortest run of event tokens kept as an event (1 token = 1/15 s) |
| `EVENT_SMOOTH_TOKENS` | `PICKLEBALL_EVENT_SMOOTH_TOKENS` | `1` | `--smooth-tokens` | moving average of P(event) in tokens; `1` disables |
| `EVENT_AT_PEAK` | `PICKLEBALL_EVENT_AT_PEAK` | `false` | `--peak` | event time at the most confident token instead of the run centre |

The defaults were tuned on real inference (2026-11-04) over two scenarios:
2685 single-stroke clips of `data/video_capcut` (21 players, fore- and backhand,
1785 hits / 900 misses; score: count-based hit/miss macro-F1 and the share of
clips with exactly one event of the right class) and the 115 long training
videos (held-out fold predictions, event macro-F1 with ±0.5 s matching).

| Decoder | Clips exactly right | Clip macro-F1 | Long-video event macro-F1 |
| --- | --- | --- | --- |
| training (argmax, ≥ 3 tokens) | 96.8% | 0.981 | 0.951 |
| **default (P(event) > 0.4, ≥ 8 tokens)** | **98.4%** | **0.990** | **0.952** |

Misses gain the most (94.3% → 98.0% of miss clips exactly right). Chosen on
other players, the setting holds on held-out players (leave-players-out, 4
groups). Joining nearby runs looked best on single-stroke clips (0.997) but
merged consecutive strokes of long videos (event F1 ≤ 0.63), so it is not used.

`NUM_VIDEO_FRAMES`, `NUM_TEMPORAL_TOKENS`, `AUDIO_SR` and `LABEL2ID` in the
*MODEL CONSTANTS* section are fixed by the trained checkpoint and must not be
changed.

Credentials are never logged and are hidden from `repr(Settings)`.

### GPU performance

On CUDA the pipeline is built for throughput:

- The video is decoded **once, sequentially**, straight into a shared-memory ring;
  frames that are not sampled (every other one at 60 fps) are only grabbed, and
  decoding stops after the last sampled frame. Per-window seeking is avoided: on
  variable-frame-rate phone videos `CAP_PROP_POS_FRAMES` lands up to a second
  away from the requested frame.
- `PICKLEBALL_PREPROCESS_WORKERS` worker processes rotate (portrait phone
  videos), resize and normalize each sampled frame once (overlapping windows
  share frames) while the GPU runs the previous batch. The result is
  **bit-identical** to the training preprocessing (tested). Each worker has its
  own pipes; a crashed worker fails that request with a `502` and is restarted
  for the next one.
- `PICKLEBALL_BATCH_SIZE` windows go through VideoMAE and the fusion ensemble
  together; BEATs runs once per video before them. VideoMAE runs under fp16
  autocast (`PICKLEBALL_PRECISION`); BEATs (its fbank overflows fp16) and the
  fusion models stay fp32.
- Models are warmed up at startup, so the first request is not slower.

Measured with the previous model (native-fps windows, PIL preprocessing) on an
NVIDIA A30 (24-core Xeon Silver 4310, shared) through `POST /api/v1/process`
with a real MinIO download:

| Video | Before | Threads (`PREPROCESS_WORKERS=0`) | 8 worker processes (default) |
| --- | --- | --- | --- |
| 3 s, 1080×1920 (91 frames, one keyframe) | 4.6–4.8 s | 1.0–1.2 s | 1.1–1.2 s |
| 47 s, 1080×1920, constant frame rate | 82 s | 8.6–9.8 s | 7.5–9.7 s |
| 65 s phone video, variable frame rate, rotated | 108 s | 13.5–15.9 s | 11.4–11.8 s |

Remaining time is CPU: decoding plus exact resizing costs ~25 ms of CPU per
1080p frame; batch sizes above 16 do not help. With v4 one GPU process scores a
135 s 60 fps iPhone HEVC video in ~65 s and a 53 s H.264 video in ~9 s.

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

`docker-compose.yml` runs `AI_REPLICAS` (default 3) model servers on the GPU behind an
nginx load balancer (`deploy/nginx-lb.conf`) on port 8888. nginx sends each job to the
replica with the fewest open connections (a job holds its connection until scored)
and skips a replica that answers `429`. Scale with
`AI_REPLICAS=3 docker compose up -d` (each replica loads the models once, ~3 GB of
VRAM). The weights directory is `PICKLEBALL_WEIGHTS_DIR` (default `./src/weights`).

`restart: unless-stopped` does not restart a container someone stopped by hand, so
`deploy/watchdog.sh` runs every minute from cron and calls `docker compose up -d` when
fewer than `AI_REPLICAS` model servers or no load balancer are running:

```bash
( crontab -l; echo "* * * * * $PWD/deploy/watchdog.sh >> \$HOME/pickleball-ai-watchdog.log 2>&1" ) | crontab -
touch deploy/.watchdog-pause   # planned maintenance: pause it; rm the file to resume
```
It also sets `--gpus`, `shm_size: 2gb` (preprocessing workers) and `env_file: .env`. The image keeps the repository layout
under `/app` (`PYTHONPATH=/app/src`), so relative paths in `.env` work unchanged.
CPU-only image: `docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu -t pickleball-track .`
and run it with `--shm-size=1g -e PICKLEBALL_DEVICE=cpu`.

Command-line inference on a local file (writes `<video>.inference.json`):

```bash
python scripts/infer_video.py --video match.mp4 [--checkpoint pickleball_v4.pt --event-threshold 0.5 --min-event-tokens 3 --smooth-tokens 1 --peak]

# Pack a new training run (baseline/run_pickleball.sh) for the service:
python scripts/export_checkpoint.py --run-dir /data/Cuong/pickleball/artifacts/weights_pickleball/v4 \
    --output src/weights/pickleball_v4.pt
```

## API Contract

| Endpoint | Success | Failure |
| --- | --- | --- |
| `GET /health` | `200 {"status": "ok"}` | – |
| `GET /ready` | `200 {"status": "ready"}` when models are loaded | `503` |
| `POST /api/v1/process` | `200` job response | see below |
| `GET /stats` | `200` slots, pending jobs and cached results | – |

Request (only `.mp4` / `.mov` object keys are accepted). Only `job_id`, `bucket` and
`object_key` are required; `minio_endpoint` is needed unless `MINIO_PUBLIC_ENDPOINT`
is set. The other fields are optional context (validated when present):

```json
{"job_id": "job-1", "result_id": "job-1", "bucket": "videos", "object_key": "matches/match.MOV", "mode": "VIDEO", "attempt": 1}
```

Full form accepted for older clients:

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
below `AI_REPLICAS` × `PICKLEBALL_MAX_PENDING_JOBS`.

Measured on the A30 server (3 s 1080p clips; 60 s iPhone HDR clips): the model is
~17% of the time of a long video; decoding 10-bit HEVC with OpenCV takes the rest
and ~28 CPU cores, so long-video throughput (~1.2/min) does not grow with more
slots or replicas. Short clips scale with replicas: 3 replicas x 2 slots score
~120 clips/min, and short clips stay under ~9 s (p95) while long videos run.

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
