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
) -> dict[str, int | bool]:
    """Load for monitoring and client tuning (not part of the scoring contract)."""
    return {
        "ready": inference_service is not None,
        "inference_slots": getattr(inference_service, "slots", 0),
        "busy_slots": getattr(inference_service, "busy_slots", 0),
        "pending_jobs": gate.pending,
        "max_pending_jobs": gate.max_pending,
        "cached_results": gate.cached,
    }
