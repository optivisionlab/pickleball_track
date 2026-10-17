from fastapi import APIRouter, HTTPException

from .schemas import VideoJobRequest, VideoJobResponse
from ..models.model_loader import model_assets_available
from ..services.video_jobs import process_video_job


router = APIRouter()


@router.get("/health")
def health_check():
    return {"status": "ok"}


@router.get("/ready")
def readiness_check():
    if not model_assets_available():
        raise HTTPException(status_code=503, detail="Model assets are unavailable")
    return {"status": "ready"}


@router.post("/api/v1/process", response_model=VideoJobResponse)
def process_video(request: VideoJobRequest) -> VideoJobResponse:
    return process_video_job(request)