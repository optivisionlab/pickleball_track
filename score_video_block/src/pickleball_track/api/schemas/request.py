from typing import Literal

from pydantic import BaseModel, Field


class VideoDetail(BaseModel):
    title: str
    category: str
    gradingMethod: str
    timeLimit: str


class VideoJobRequest(BaseModel):
    """Scoring request. Only ``job_id``, ``bucket`` and ``object_key`` are needed to
    score a video; the other fields are optional context kept for compatibility with
    clients that still send the full payload (they are validated when present)."""

    job_id: str
    bucket: str
    object_key: str
    result_id: str | None = None
    mode: Literal["VIDEO"] = "VIDEO"
    # Ignored when the service sets MINIO_PUBLIC_ENDPOINT.
    minio_endpoint: str | None = None
    attempt: int = Field(default=1, ge=1)
    schema_version: int = Field(default=1, ge=1)
    task_id: str | None = None
    course_id: str | None = None
    sport_id: str | None = None
    sport_code: str | None = None
    video_url: str | None = None
    user_id: str | None = None
    detail: VideoDetail | None = None
