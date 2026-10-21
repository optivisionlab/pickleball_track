from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pickleball_track.api.main import create_app
from pickleball_track.core.config import Settings


class FakeMinioClient:
    def __init__(self):
        self.downloads = []

    def stat_object(self, bucket, object_key):
        return type("ObjectInfo", (), {"size": 10})()

    def fget_object(self, bucket, object_key, destination):
        self.downloads.append((bucket, object_key))
        Path(destination).write_bytes(b"test video")


class FakeInferenceService:
    def __init__(self, result=None, error=None):
        self.result = result or {"total_hits": 8, "total_misses": 2}
        self.error = error
        self.calls = 0

    def predict(self, video_path):
        self.calls += 1
        if self.error:
            raise self.error
        return self.result


def _client(settings, inference=None, load_error=None, minio=None):
    def inference_factory(_settings):
        if load_error:
            raise load_error
        return inference

    minio = minio or FakeMinioClient()
    app = create_app(
        settings,
        inference_factory=inference_factory,
        minio_client_factory=lambda endpoint, _settings: minio,
    )
    return TestClient(app)


def test_health_is_independent_of_model_state(test_settings):
    with _client(test_settings, load_error=RuntimeError("missing weights")) as client:
        assert client.get("/health").json() == {"status": "ok"}


def test_ready_after_models_load(test_settings):
    with _client(test_settings, inference=FakeInferenceService()) as client:
        response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_not_ready_and_process_unavailable_when_model_load_fails(test_settings, valid_job_request):
    with _client(test_settings, load_error=RuntimeError("missing weights")) as client:
        ready = client.get("/ready")
        process = client.post("/api/v1/process", json=valid_job_request)

    assert ready.status_code == 503
    assert process.status_code == 503
    assert process.json() == {"detail": "Model checkpoint is not configured or unavailable"}


def test_default_app_without_artifacts_starts_but_is_not_ready(tmp_path):
    settings = Settings(checkpoint_path=tmp_path / "missing.pt", device="cpu")
    with TestClient(create_app(settings)) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503


def test_process_returns_product_contract(test_settings, valid_job_request):
    minio, inference = FakeMinioClient(), FakeInferenceService()
    with _client(test_settings, inference=inference, minio=minio) as client:
        response = client.post("/api/v1/process", json=valid_job_request)

    assert response.status_code == 200
    assert response.json() == {
        "jobId": "job-1",
        "resultId": "result-1",
        "status": "completed",
        "score": 8.0,
        "feedback": "Thực hiện động tác đạt yêu cầu.",
        "metrics": {"accuracy": 0.8, "hitCount": 8, "missCount": 2},
    }
    assert minio.downloads == [("videos", "matches/match.MOV")]
    assert inference.calls == 1


@pytest.mark.parametrize(
    "override",
    [{"mode": "IMAGE"}, {"attempt": 0}, {"job_id": None}, {"object_key": "clip.avi"}],
)
def test_invalid_requests_return_422(test_settings, valid_job_request, override):
    inference = FakeInferenceService()
    with _client(test_settings, inference=inference) as client:
        response = client.post("/api/v1/process", json=valid_job_request | override)

    assert response.status_code == 422
    assert inference.calls == 0


def test_oversized_video_returns_413(test_settings, valid_job_request):
    inference = FakeInferenceService()
    with _client(replace(test_settings, max_video_bytes=2), inference=inference) as client:
        response = client.post("/api/v1/process", json=valid_job_request)

    assert response.status_code == 413
    assert inference.calls == 0


def test_inference_failure_returns_502(test_settings, valid_job_request):
    inference = FakeInferenceService(error=RuntimeError("Cannot open video"))
    with _client(test_settings, inference=inference) as client:
        response = client.post("/api/v1/process", json=valid_job_request)

    assert response.status_code == 502
    assert response.json() == {"detail": "Could not download the video or complete inference"}


def test_invalid_minio_endpoint_returns_422(test_settings, valid_job_request):
    app = create_app(test_settings, inference_factory=lambda s: FakeInferenceService())
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/process", json=valid_job_request | {"minio_endpoint": "ftp://minio"}
        )

    assert response.status_code == 422
    assert response.json() == {"detail": "minio_endpoint must use HTTP or HTTPS"}


class BlockingInferenceService(FakeInferenceService):
    def __init__(self):
        super().__init__()
        import threading

        self.started = threading.Event()
        self.release = threading.Event()

    def predict(self, video_path):
        self.started.set()
        self.release.wait(timeout=10)
        return super().predict(video_path)


def test_overloaded_service_returns_429_with_retry_after(test_settings, valid_job_request):
    import threading

    inference = BlockingInferenceService()
    settings = replace(test_settings, inference_slots=1, max_pending_jobs=1, busy_retry_after_seconds=9)
    with _client(settings, inference=inference) as client:
        first = {}
        worker = threading.Thread(
            target=lambda: first.setdefault("response", client.post("/api/v1/process", json=valid_job_request))
        )
        worker.start()
        assert inference.started.wait(timeout=10)
        busy = client.post("/api/v1/process", json=valid_job_request | {"job_id": "job-2"})
        stats = client.get("/stats").json()
        inference.release.set()
        worker.join(timeout=10)

    assert busy.status_code == 429
    assert busy.headers["Retry-After"] == "9"
    assert busy.json() == {"detail": "Server busy: 1 videos already in progress"}
    assert stats["pending_jobs"] == 1 and stats["max_pending_jobs"] == 1
    assert first["response"].status_code == 200


def test_retried_job_id_reuses_the_result(test_settings, valid_job_request):
    inference = FakeInferenceService()
    with _client(test_settings, inference=inference) as client:
        responses = [client.post("/api/v1/process", json=valid_job_request) for _ in range(2)]

    assert [r.status_code for r in responses] == [200, 200]
    assert responses[0].json() == responses[1].json()
    assert inference.calls == 1
