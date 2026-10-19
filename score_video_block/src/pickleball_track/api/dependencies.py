"""Request-scoped access to objects created in the application lifespan."""

from fastapi import Request

from ..core.config import Settings
from ..services.inference_service import VideoInferenceService
from ..services.video_jobs import VideoJobService


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_inference_service(request: Request) -> VideoInferenceService | None:
    return request.app.state.inference_service


def get_video_job_service(request: Request) -> VideoJobService:
    return VideoJobService(
        settings=request.app.state.settings,
        predictor=request.app.state.inference_service,
        minio_client_factory=request.app.state.minio_client_factory,
    )
