import pytest
from pydantic import ValidationError

from pickleball_track.api.schemas import VideoJobRequest, VideoJobResponse


def test_request_matches_product_contract(valid_job_request):
    request = VideoJobRequest(**valid_job_request)

    assert request.model_dump() == valid_job_request
    assert set(VideoJobRequest.model_fields) == set(valid_job_request)


@pytest.mark.parametrize(
    "override",
    [{"mode": "IMAGE"}, {"attempt": 0}, {"schema_version": 0}, {"detail": {}}],
)
def test_request_rejects_invalid_contract_values(valid_job_request, override):
    with pytest.raises(ValidationError):
        VideoJobRequest(**(valid_job_request | override))


def test_minimal_request_needs_only_job_bucket_and_key():
    request = VideoJobRequest(job_id="job-1", bucket="videos", object_key="a/b.mp4")

    assert (request.mode, request.attempt, request.result_id, request.detail) == ("VIDEO", 1, None, None)


@pytest.mark.parametrize("missing", ["job_id", "bucket", "object_key"])
def test_request_requires_job_bucket_and_key(valid_job_request, missing):
    with pytest.raises(ValidationError):
        VideoJobRequest(**{k: v for k, v in valid_job_request.items() if k != missing})


def test_response_matches_product_contract():
    response = VideoJobResponse(
        jobId="job-1",
        resultId="result-1",
        status="completed",
        score=8.0,
        feedback="Thực hiện động tác đạt yêu cầu.",
        metrics={"accuracy": 0.8, "hitCount": 4, "missCount": 1},
    )

    assert response.model_dump() == {
        "jobId": "job-1",
        "resultId": "result-1",
        "status": "completed",
        "score": 8.0,
        "feedback": "Thực hiện động tác đạt yêu cầu.",
        "metrics": {"accuracy": 0.8, "hitCount": 4, "missCount": 1},
    }
