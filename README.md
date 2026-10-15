# Pickleball Track API

FastAPI service for processing pickleball videos stored in MinIO. Runtime code
uses a `src/` package layout under `src/pickleball_track`. Training scripts and
temporary reference files are not part of the product package.

## Run

Install a PyTorch and TorchAudio build compatible with the deployment hardware,
then install this project:

```bash
python -m pip install -e ".[dev]"
```

The application loads `.env` from the repository root. MinIO credentials use
`MINIO_ACCESS_KEY` and `MINIO_SECRET_KEY` when set, otherwise the configured
`MINIO_ROOT_USER` and `MINIO_ROOT_PASSWORD`. Do not commit `.env`.

Model files live in one directory: `src/weights/best.pt` (trained fusion),
`src/weights/BEATs_iter3_plus_AS2M.pt` (BEATs), and
`src/weights/videomae-base/` (VideoMAE). The BEATs source code is separate from
its weights and should be cloned once into `vendor/unilm`:

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/microsoft/unilm.git vendor/unilm
git -C vendor/unilm sparse-checkout set beats
```

The model loaders run lazily on the first video request and cache all three
models in the API worker process; later requests reuse the loaded models.
Environment variables `PICKLEBALL_CHECKPOINT`, `PICKLEBALL_BEATS_CHECKPOINT`,
`PICKLEBALL_BEATS_ROOT`, and `PICKLEBALL_VIDEOMAE` can override these defaults.
Start the service with:

```bash
python -m uvicorn pickleball_track.api.main:app --host 0.0.0.0 --port 8000
```

The API listens on port `8000`. Health check: `GET /health`.

## Video Job Contract

`POST /api/v1/process` accepts the request schema below. Field names, nesting,
types, and the positive constraints on `attempt` and `schema_version` match the
provided product API contract.

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

The video is downloaded from `bucket` and `object_key`; `object_key` must end in
`.mp4` or `.mov`. `MINIO_SECURE=true` enables TLS when `minio_endpoint` is given
without a URL scheme. The response contains `jobId`, `resultId`, `status`,
`score`, `feedback`, and `metrics` (`accuracy`, `hitCount`, `missCount`).

Run contract and simulated workflow tests with `python -m pytest -q`. The
workflow test mocks MinIO and model inference; real inference also requires
the BEATs source checkout, FFmpeg on `PATH`, and compatible PyTorch and
TorchVision builds for the deployment hardware.