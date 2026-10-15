from typing import Literal

from pydantic import BaseModel, Field


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