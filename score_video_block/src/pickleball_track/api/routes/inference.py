from fastapi import APIRouter, Depends

from ...services.job_gate import JobGate
from ...services.video_jobs import VideoJobService
from ..dependencies import get_job_gate, get_video_job_service
from ..schemas import VideoJobRequest, VideoJobResponse, VideoMetrics

router = APIRouter()


# Async handler: the job runs on the JobGate executor (download, then a free inference
# slot), so waiting requests hold no event-loop or threadpool resources, overload is
# answered with 429 + Retry-After, and a retried job_id reuses the running/cached result.
@router.post("/api/v1/process", response_model=VideoJobResponse)
async def process_video(
    request: VideoJobRequest,
    service: VideoJobService = Depends(get_video_job_service),
    gate: JobGate = Depends(get_job_gate),
) -> VideoJobResponse:
    job_score = await gate.run(
        f"{request.job_id}:{request.bucket}/{request.object_key}",
        lambda: service.process(
            job_id=request.job_id,
            bucket=request.bucket,
            object_key=request.object_key,
            minio_endpoint=request.minio_endpoint,
        ),
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
