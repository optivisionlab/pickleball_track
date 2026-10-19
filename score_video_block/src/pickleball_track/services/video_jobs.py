"""Video job business logic: fetch from MinIO, run inference, score the result."""

from __future__ import annotations

import logging
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ..core.config import SUPPORTED_VIDEO_SUFFIXES, Settings
from ..core.exceptions import (
    InvalidInputError,
    PayloadTooLargeError,
    PickleballError,
    ProcessingError,
    ServiceUnavailableError,
)
from ..integrations.minio import create_minio_client

logger = logging.getLogger(__name__)


class VideoPredictor(Protocol):
    def predict(self, video_path: str | Path) -> dict[str, Any]: ...


@dataclass(frozen=True)
class JobScore:
    hit_count: int
    miss_count: int
    accuracy: float
    score: float
    feedback: str


def feedback_for(accuracy: float) -> str:
    if accuracy >= 0.8:
        return "Thực hiện động tác đạt yêu cầu."
    if accuracy >= 0.5:
        return "Cần luyện tập thêm để cải thiện độ chính xác."
    return "Chưa đạt yêu cầu, hãy luyện tập lại các động tác."


def score_counts(hits: int, misses: int) -> JobScore:
    """Product scoring: accuracy = hits / (hits + misses); no_event is not counted."""
    attempts = hits + misses
    accuracy = hits / attempts if attempts else 0.0
    return JobScore(
        hit_count=hits,
        miss_count=misses,
        accuracy=round(accuracy, 4),
        score=round(accuracy * 10.0, 2),
        feedback=feedback_for(accuracy),
    )


class VideoJobService:
    def __init__(
        self,
        settings: Settings,
        predictor: VideoPredictor | None,
        minio_client_factory: Callable[[str, Settings], Any] = create_minio_client,
    ):
        self.settings = settings
        self.predictor = predictor
        self.minio_client_factory = minio_client_factory

    def _check_size(self, size: int) -> None:
        if size > self.settings.max_video_bytes:
            raise PayloadTooLargeError("Video exceeds the configured size limit")

    def process(
        self, *, job_id: str, bucket: str, object_key: str, minio_endpoint: str
    ) -> JobScore:
        started_at = time.perf_counter()
        suffix = Path(object_key).suffix.lower()
        if suffix not in SUPPORTED_VIDEO_SUFFIXES:
            raise InvalidInputError("object_key must identify an MP4 or MOV video")
        if self.predictor is None:
            raise ServiceUnavailableError("Model checkpoint is not configured or unavailable")

        endpoint = self.settings.minio_public_endpoint or minio_endpoint
        if endpoint != minio_endpoint:
            logger.info(
                "Using configured MinIO endpoint",
                extra={"job_id": job_id, "requested_endpoint": minio_endpoint, "endpoint": endpoint},
            )
        try:
            client = self.minio_client_factory(endpoint, self.settings)
            with tempfile.TemporaryDirectory(prefix="pickleball-video-") as temp_dir:
                local_video = Path(temp_dir) / f"input{suffix}"
                self._check_size(client.stat_object(bucket, object_key).size)
                download_started_at = time.perf_counter()
                client.fget_object(bucket, object_key, str(local_video))
                download_seconds = time.perf_counter() - download_started_at
                self._check_size(local_video.stat().st_size)
                result = self.predictor.predict(local_video)
        except PickleballError:
            raise
        except Exception as exc:
            logger.exception(
                "Video job failed",
                extra={
                    "job_id": job_id,
                    "duration_seconds": round(time.perf_counter() - started_at, 4),
                },
            )
            raise ProcessingError("Could not download the video or complete inference") from exc

        job_score = score_counts(int(result["total_hits"]), int(result["total_misses"]))
        logger.info(
            "Video job completed",
            extra={
                "job_id": job_id,
                "hit_count": job_score.hit_count,
                "miss_count": job_score.miss_count,
                "score": job_score.score,
                "download_seconds": round(download_seconds, 4),
                "duration_seconds": round(time.perf_counter() - started_at, 4),
            },
        )
        return job_score
