from fastapi import APIRouter, Depends, HTTPException

from ...services.inference_service import VideoInferenceService
from ...services.job_gate import JobGate
from ..dependencies import get_inference_service, get_job_gate
from ..schemas import HealthResponse, ReadinessResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/ready", response_model=ReadinessResponse)
def readiness_check(
    inference_service: VideoInferenceService | None = Depends(get_inference_service),
) -> ReadinessResponse:
    if inference_service is None:
        raise HTTPException(status_code=503, detail="Model assets are unavailable")
    return ReadinessResponse(status="ready")


@router.get("/stats")
def stats(
    inference_service: VideoInferenceService | None = Depends(get_inference_service),
    gate: JobGate = Depends(get_job_gate),
) -> dict[str, int | float | bool | str | None]:
    """Load for monitoring and client tuning (not part of the scoring contract)."""
    scheduler = getattr(inference_service, "scheduler", None)
    waiting_short, waiting_long = scheduler.waiting() if scheduler is not None else (0, 0)
    avg = gate.avg_job_seconds
    return {
        "ready": inference_service is not None,
        "inference_slots": getattr(inference_service, "slots", 0),
        "busy_slots": getattr(inference_service, "busy_slots", 0),
        "reserved_short_slots": getattr(scheduler, "reserved_short", 0),
        "short_video_seconds": getattr(scheduler, "short_seconds", 0),
        "waiting_short": waiting_short,
        "waiting_long": waiting_long,
        "pending_jobs": gate.pending,
        "max_pending_jobs": gate.max_pending,
        "cached_results": gate.cached,
        "avg_job_seconds": None if avg is None else round(avg, 2),
        "decoder": getattr(inference_service, "decoder", None),
    }
