import tempfile
import logging
from threading import Lock
from functools import lru_cache
from pathlib import Path

from fastapi import HTTPException

from ..core.config import DEFAULT_CHECKPOINT_PATH
from ..integrations.minio import create_minio_client
from ..api.schemas import VideoJobRequest, VideoJobResponse, VideoMetrics

logger = logging.getLogger(__name__)
_inference_lock = Lock()


@lru_cache(maxsize=1)
def _load_inference_module():
    from . import inference

    return inference


def _feedback(accuracy: float) -> str:
    if accuracy >= 0.8:
        return "Thực hiện động tác đạt yêu cầu."
    if accuracy >= 0.5:
        return "Cần luyện tập thêm để cải thiện độ chính xác."
    return "Chưa đạt yêu cầu, hãy luyện tập lại các động tác."


def process_video_job(request: VideoJobRequest) -> VideoJobResponse:
    suffix = Path(request.object_key).suffix.lower()
    if suffix not in {".mp4", ".mov"}:
        raise HTTPException(
            status_code=422,
            detail="object_key must identify an MP4 or MOV video",
        )

    if not DEFAULT_CHECKPOINT_PATH.is_file():
        raise HTTPException(
            status_code=503,
            detail="Model checkpoint is not configured or unavailable",
        )

    try:
        client = create_minio_client(request.minio_endpoint)
        with tempfile.TemporaryDirectory(prefix="pickleball-video-") as temp_dir:
            local_video = Path(temp_dir) / f"input{suffix}"
            client.fget_object(request.bucket, request.object_key, str(local_video))
            with _inference_lock:
                result = _load_inference_module().inference(
                    video_path=local_video,
                    checkpoint_path=DEFAULT_CHECKPOINT_PATH,
                )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Video job %s failed", request.job_id)
        raise HTTPException(
            status_code=502,
            detail="Could not download the video or complete inference",
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