from typing import Literal

from pydantic import BaseModel


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


class HealthResponse(BaseModel):
    status: Literal["ok"]


class ReadinessResponse(BaseModel):
    status: Literal["ready"]
