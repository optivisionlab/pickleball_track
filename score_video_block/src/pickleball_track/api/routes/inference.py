from fastapi import APIRouter, Depends

from ...services.video_jobs import VideoJobService
from ..dependencies import get_video_job_service
from ..schemas import VideoJobRequest, VideoJobResponse, VideoMetrics

router = APIRouter()


# Sync handler: FastAPI runs it in a worker thread, so inference never blocks the event loop.
@router.post("/api/v1/process", response_model=VideoJobResponse)
def process_video(
    request: VideoJobRequest,
    service: VideoJobService = Depends(get_video_job_service),
) -> VideoJobResponse:
    job_score = service.process(
        job_id=request.job_id,
        bucket=request.bucket,
        object_key=request.object_key,
        minio_endpoint=request.minio_endpoint,
    )
    return VideoJobResponse(
        jobId=request.job_id,
        resultId=request.result_id,
        status="completed",
        score=job_score.score,
        feedback=job_score.feedback,
        metrics=VideoMetrics(
            accuracy=job_score.accuracy,
            hitCount=job_score.hit_count,
            missCount=job_score.miss_count,
        ),
    )
