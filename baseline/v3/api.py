import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = BASE_DIR / "checkpoints" / "fold_1" / "best.pt"
CHECKPOINT_PATH = Path(
    os.getenv("PICKLEBALL_CHECKPOINT", str(DEFAULT_CHECKPOINT))
).expanduser()
if not CHECKPOINT_PATH.is_absolute():
    CHECKPOINT_PATH = (BASE_DIR / CHECKPOINT_PATH).resolve()


class VideoDetail(BaseModel):
    title: str
    category: str
    gradingMethod: str
    timeLimit: str


class VideoJobRequest(BaseModel):
    job_id: str
    result_id: str
    task_id: str
    course_id: str
    sport_id: str
    sport_code: str
    mode: Literal["VIDEO"]
    bucket: str
    object_key: str
    video_url: str
    minio_endpoint: str
    user_id: str
    attempt: int = Field(ge=1)
    schema_version: int = Field(ge=1)
    detail: VideoDetail


class VideoMetrics(BaseModel):
    accuracy: float
    hitCount: int
    missCount: int


class VideoJobResponse(BaseModel):
    jobId: str
    resultId: str
    status: Literal["completed"]
    score: float
    feedback: str
    metrics: VideoMetrics


app = FastAPI(title="Pickleball AI Pipeline", version="1.0.0")


@lru_cache(maxsize=1)
def _load_inference_module():
    try:
        from . import inference as inference_module
    except ImportError:
        import inference as inference_module

    beats_checkpoint = Path(inference_module.BEATS_CHECKPOINT)
    if not beats_checkpoint.is_absolute():
        beats_checkpoint = BASE_DIR / beats_checkpoint
    inference_module.BEATS_CHECKPOINT = str(beats_checkpoint.resolve())
    return inference_module


def _minio_client(endpoint: str):
    try:
        from minio import Minio
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail="MinIO SDK is missing; install v3/requirements-api.txt",
        ) from exc

    parsed = urlsplit(endpoint if "://" in endpoint else f"//{endpoint}")
    if not parsed.hostname or parsed.path not in ("", "/"):
        raise HTTPException(status_code=422, detail="Invalid minio_endpoint")
    if parsed.scheme and parsed.scheme not in ("http", "https"):
        raise HTTPException(status_code=422, detail="minio_endpoint must use HTTP or HTTPS")

    secure = parsed.scheme == "https" or (
        not parsed.scheme and os.getenv("MINIO_SECURE", "false").lower() == "true"
    )
    return Minio(
        parsed.netloc or parsed.path,
        access_key=os.getenv("MINIO_ACCESS_KEY"),
        secret_key=os.getenv("MINIO_SECRET_KEY"),
        secure=secure,
    )


def _feedback(accuracy: float) -> str:
    if accuracy >= 0.8:
        return "Thực hiện động tác đạt yêu cầu."
    if accuracy >= 0.5:
        return "Cần luyện tập thêm để cải thiện độ chính xác."
    return "Chưa đạt yêu cầu, hãy luyện tập lại các động tác."


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/api/v1/process", response_model=VideoJobResponse)
def process_video_job(request: VideoJobRequest) -> VideoJobResponse:
    if not CHECKPOINT_PATH.is_file():
        raise HTTPException(
            status_code=503,
            detail=f"Model checkpoint not found: {CHECKPOINT_PATH}",
        )

    suffix = Path(request.object_key).suffix.lower()
    if suffix not in {".mp4", ".mov"}:
        raise HTTPException(status_code=422, detail="object_key must identify an MP4 or MOV video")

    try:
        client = _minio_client(request.minio_endpoint)
        with tempfile.TemporaryDirectory(prefix="pickleball-video-") as temp_dir:
            local_video = Path(temp_dir) / f"input{suffix}"
            client.fget_object(request.bucket, request.object_key, str(local_video))
            inference_module = _load_inference_module()
            result = inference_module.inference(
                video_path=local_video,
                checkpoint_path=CHECKPOINT_PATH,
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Could not download video or run inference: {exc}",
        ) from exc

    hits = int(result["total_hits"])
    misses = int(result["total_misses"])
    attempts = hits + misses
    accuracy = hits / attempts if attempts else 0.0

    return VideoJobResponse(
        jobId=request.job_id,
        resultId=request.result_id,
        status="completed",
        score=round(accuracy * 10.0, 2),
        feedback=_feedback(accuracy),
        metrics=VideoMetrics(
            accuracy=round(accuracy, 4),
            hitCount=hits,
            missCount=misses,
        ),
    )