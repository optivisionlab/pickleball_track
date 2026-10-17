# Pickleball Track Inference API

FastAPI service for processing pickleball videos stored in MinIO. The product
contract remains `POST /api/v1/process`; the API downloads the requested object,
runs temporal inference, and returns the existing job score/metrics response.

## Architecture

```text
src/pickleball_track/
  api/          FastAPI app, routes, and Pydantic request/response schemas
  core/         Environment configuration and JSON logging
  integrations/ MinIO client construction
  models/       Fusion architecture and cached pretrained/checkpoint loaders
  pipelines/    Video/audio decoding, windowing, and feature preparation
  services/     Inference orchestration and video-job business logic
```

The trained model is audiovisual: local VideoMAE and Microsoft BEATs backbones
feed the trained projection, cross-attention fusion, temporal transformer, and
three-class prediction head. Checkpoint metadata confirms the output order:
`no_event=0`, `hit=1`, `miss=2`. Inference uses 32 video frames per window,
16 temporal tokens, and 16 kHz mono audio. Overlapping windows are averaged and
the existing temporal smoothing/event-merging behavior is retained.

The VideoMAE, BEATs, and trained fusion model are loaded lazily and cached once
per worker process. The fusion checkpoint is loaded strictly and its label map
is checked when present. All models are put in evaluation mode; inference uses
`torch.inference_mode()`. Requests are serialized per worker while sharing the
cached model/device resources.

## Setup

The runtime needs compatible PyTorch/TorchVision packages, FFmpeg, the model
artifacts, and the BEATs source checkout. Keep large weights outside source
control. The default artifacts are `src/weights/best.pt`,
`src/weights/BEATs_iter3_plus_AS2M.pt`, and `src/weights/videomae-base/`.

Clone the BEATs source once if it is not already available:

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/microsoft/unilm.git vendor/unilm
git -C vendor/unilm sparse-checkout set beats
```

Install the project into the selected environment when needed:

```bash
conda run -n baseline python -m pip install -e ".[dev]"
```

For accelerator deployments, ensure the baseline environment already contains
the PyTorch build for that accelerator before installing the project.

## Configuration

The package loads `.env` from the repository root. Copy `.env.example` as a
starting point; it contains no credentials. Settings can also be provided as
process environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `PICKLEBALL_CHECKPOINT` | `src/weights/best.pt` | Trained fusion checkpoint |
| `PICKLEBALL_BEATS_CHECKPOINT` | `src/weights/BEATs_iter3_plus_AS2M.pt` | Pretrained BEATs weights |
| `PICKLEBALL_BEATS_ROOT` | `vendor/unilm` | BEATs source checkout |
| `PICKLEBALL_VIDEOMAE` | `src/weights/videomae-base` | Local model directory or model id |
| `PICKLEBALL_DEVICE` | `auto` | `auto`, `cpu`, or `cuda` |
| `PICKLEBALL_MAX_VIDEO_BYTES` | `536870912` | Maximum MinIO object size |
| `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY` | unset | MinIO credentials |
| `MINIO_SECURE` | `false` | TLS for endpoint without scheme |
| `LOG_LEVEL` | `INFO` | Python logging level |

MinIO credentials may alternatively use `MINIO_ROOT_USER` and
`MINIO_ROOT_PASSWORD`. Never place live credentials in source control or logs.

## Run

Start the API using the Conda baseline environment:

```bash
conda run -n baseline python -m uvicorn pickleball_track.api.main:app --host 0.0.0.0 --port 8000
```

`GET /health` is a process health check. `GET /ready` returns `200` when the
configured model artifacts are present, otherwise `503`. Models load on the
first accepted inference request.

## API Contract

`POST /api/v1/process` accepts a job with the current field names and downloads
the video from `bucket` and `object_key` (only `.mp4` and `.mov` are accepted):

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
  "detail": {
    "title": "Match",
    "category": "serve",
    "gradingMethod": "ai",
    "timeLimit": "60"
  }
}
```

The response remains `jobId`, `resultId`, `status`, `score`, `feedback`, and
`metrics` (`accuracy`, `hitCount`, `missCount`). Accuracy preserves the existing
product calculation `hitCount / (hitCount + missCount)`; `no_event` is not
counted as a failed action. Input/schema errors return `422`, oversized videos
return `413`, unavailable configured checkpoints return `503`, and download or
inference failures return `502`. Processing failures are not converted to a
`no_event` prediction.

## Tests

Run the complete suite in the Conda baseline environment:

```bash
conda run -n baseline python -m pytest -q
```

The API workflow tests mock MinIO and inference. A real inference additionally
requires the video fixture, FFmpeg, all three model artifacts, and an available
supported device. Checkpoint loading can be verified independently by loading
the trained fusion model with the dimensions recorded in its tensors.