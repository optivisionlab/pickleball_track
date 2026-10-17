import tempfile
import logging
import time
from collections.abc import Callable
from threading import Lock
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any

from fastapi import HTTPException

from ..core.config import DEFAULT_CHECKPOINT_PATH, MAX_VIDEO_BYTES
from ..integrations.minio import create_minio_client
from ..api.schemas import VideoJobRequest, VideoJobResponse, VideoMetrics

logger = logging.getLogger(__name__)
_inference_lock = Lock()


@lru_cache(maxsize=1)
def _load_inference_module() -> ModuleType:
    from . import inference

    return inference


def _feedback(accuracy: float) -> str:
    if accuracy >= 0.8:
        return "Thực hiện động tác đạt yêu cầu."
    if accuracy >= 0.5:
        return "Cần luyện tập thêm để cải thiện độ chính xác."
    return "Chưa đạt yêu cầu, hãy luyện tập lại các động tác."


def process_video_job(
    request: VideoJobRequest,
    *,
    minio_client_factory: Callable[[str], Any] | None = None,
    inference_module_loader: Callable[[], Any] | None = None,
    checkpoint_path: Path | None = None,
) -> VideoJobResponse:
    started_at = time.perf_counter()
    configured_checkpoint = checkpoint_path or DEFAULT_CHECKPOINT_PATH
    create_client = minio_client_factory or create_minio_client
    load_inference = inference_module_loader or _load_inference_module
    suffix = Path(request.object_key).suffix.lower()
    if suffix not in {".mp4", ".mov"}:
        raise HTTPException(
            status_code=422,
            detail="object_key must identify an MP4 or MOV video",
        )

    if not configured_checkpoint.is_file():
        raise HTTPException(
            status_code=503,
            detail="Model checkpoint is not configured or unavailable",
        )

    try:
        client = create_client(request.minio_endpoint)
        with tempfile.TemporaryDirectory(prefix="pickleball-video-") as temp_dir:
            local_video = Path(temp_dir) / f"input{suffix}"
            object_info = client.stat_object(request.bucket, request.object_key)
            if object_info.size > MAX_VIDEO_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail="Video exceeds the configured size limit",
                )
            client.fget_object(request.bucket, request.object_key, str(local_video))
            if local_video.stat().st_size > MAX_VIDEO_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail="Video exceeds the configured size limit",
                )
            with _inference_lock:
                result = load_inference().inference(
                    video_path=local_video,
                    checkpoint_path=configured_checkpoint,
                )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(
            "Video job failed",
            extra={
                "job_id": request.job_id,
                "duration_seconds": round(time.perf_counter() - started_at, 4),
            },
        )
        raise HTTPException(
            status_code=502,
            detail="Could not download the video or complete inference",
        ) from exc

    hits = int(result["total_hits"])
    misses = int(result["total_misses"])
    attempts = hits + misses
    accuracy = hits / attempts if attempts else 0.0
    score = round(accuracy * 10.0, 2)
    duration_seconds = round(time.perf_counter() - started_at, 4)
    logger.info(
        "Video job completed",
        extra={
            "job_id": request.job_id,
            "hit_count": hits,
            "miss_count": misses,
            "score": score,
            "duration_seconds": duration_seconds,
        },
    )

    return VideoJobResponse(
        jobId=request.job_id,
        resultId=request.result_id,
        status="completed",
        score=score,
        feedback=_feedback(accuracy),
        metrics=VideoMetrics(
            accuracy=round(accuracy, 4),
            hitCount=hits,
            missCount=misses,
        ),
    )