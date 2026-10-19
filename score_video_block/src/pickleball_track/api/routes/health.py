from fastapi import APIRouter, Depends, HTTPException

from ...services.inference_service import VideoInferenceService
from ..dependencies import get_inference_service
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
