from fastapi import APIRouter

from .schemas import VideoJobRequest, VideoJobResponse
from ..services.video_jobs import process_video_job


router = APIRouter()


@router.get("/health")
def health_check():
    return {"status": "ok"}


@router.post("/api/v1/process", response_model=VideoJobResponse)
def process_video(request: VideoJobRequest) -> VideoJobResponse:
    return process_video_job(request)