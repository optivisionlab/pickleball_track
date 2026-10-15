import pytest
from pydantic import ValidationError

from pickleball_track.api.schemas import VideoJobRequest, VideoJobResponse


@pytest.fixture
def valid_request():
    return {
        "job_id": "job-1",
        "result_id": "result-1",
        "task_id": "task-1",
        "course_id": "course-1",
        "sport_id": "sport-1",
        "sport_code": "PB",
        "mode": "VIDEO",
        "bucket": "videos",
        "object_key": "match.MOV",
        "video_url": "https://example.test/match.MOV",
        "minio_endpoint": "localhost:9000",
        "user_id": "user-1",
        "attempt": 1,
        "schema_version": 1,
        "detail": {
            "title": "Match",
            "category": "serve",
            "gradingMethod": "ai",
            "timeLimit": "60",
        },
    }


def test_request_matches_product_contract(valid_request):
    request = VideoJobRequest(**valid_request)

    assert request.model_dump() == valid_request
    assert set(VideoJobRequest.model_fields) == set(valid_request)


@pytest.mark.parametrize(
    "override",
    [{"mode": "IMAGE"}, {"attempt": 0}, {"schema_version": 0}],
)
def test_request_rejects_invalid_contract_values(valid_request, override):
    with pytest.raises(ValidationError):
        VideoJobRequest(**(valid_request | override))


def test_response_matches_product_contract():
    response = VideoJobResponse(
        jobId="job-1",
        resultId="result-1",
        status="completed",
        score=8.0,
        feedback="Thực hiện động tác đạt yêu cầu.",
        metrics={"accuracy": 0.8, "hitCount": 4, "missCount": 1},
    )

    assert set(response.model_dump()) == {
        "jobId",
        "resultId",
        "status",
        "score",
        "feedback",
        "metrics",
    }