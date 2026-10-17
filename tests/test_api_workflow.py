from fastapi.testclient import TestClient

from pickleball_track.api.main import app
from pickleball_track.api import routes
from pickleball_track.services import video_jobs


def _valid_video_job_request():
    return {
        "job_id": "job-1",
        "result_id": "result-1",
        "task_id": "task-1",
        "course_id": "course-1",
        "sport_id": "sport-1",
        "sport_code": "PB",
        "mode": "VIDEO",
        "bucket": "videos",
        "object_key": "matches/match.MOV",
        "video_url": "https://storage.example/matches/match.MOV",
        "minio_endpoint": "http://minio:9000",
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


def test_process_video_job_with_mocked_minio_and_inference(monkeypatch, tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"test checkpoint")
    monkeypatch.setattr(video_jobs, "DEFAULT_CHECKPOINT_PATH", checkpoint)

    calls = {}

    class FakeMinioClient:
        def stat_object(self, bucket, object_key):
            return type("ObjectInfo", (), {"size": 10})()

        def fget_object(self, bucket, object_key, destination):
            calls["download"] = (bucket, object_key)
            calls["download_count"] = calls.get("download_count", 0) + 1
            with open(destination, "wb") as video_file:
                video_file.write(b"test video")

    class FakeInference:
        @staticmethod
        def inference(video_path, checkpoint_path):
            calls["inference"] = (video_path.is_file(), checkpoint_path == checkpoint)
            return {"total_hits": 8, "total_misses": 2}

    def fake_create_minio_client(endpoint):
        calls["endpoint"] = endpoint
        return FakeMinioClient()

    monkeypatch.setattr(
        video_jobs,
        "create_minio_client",
        fake_create_minio_client,
    )
    monkeypatch.setattr(video_jobs, "_load_inference_module", lambda: FakeInference)

    response = TestClient(app).post("/api/v1/process", json=_valid_video_job_request())

    assert response.status_code == 200
    assert response.json() == {
        "jobId": "job-1",
        "resultId": "result-1",
        "status": "completed",
        "score": 8.0,
        "feedback": "Thực hiện động tác đạt yêu cầu.",
        "metrics": {"accuracy": 0.8, "hitCount": 8, "missCount": 2},
    }
    assert calls["endpoint"] == "http://minio:9000"
    assert calls["download"] == ("videos", "matches/match.MOV")
    assert calls["inference"] == (True, True)

    monkeypatch.setattr(video_jobs, "MAX_VIDEO_BYTES", 2)
    oversized = TestClient(app).post(
        "/api/v1/process", json=_valid_video_job_request()
    )
    assert oversized.status_code == 413
    assert calls["inference"] == (True, True)
    assert calls["download_count"] == 1


def test_readiness_reports_asset_availability(monkeypatch):
    monkeypatch.setattr(routes, "model_assets_available", lambda: True)
    ready = TestClient(app).get("/ready")
    assert ready.status_code == 200
    assert ready.json() == {"status": "ready"}

    monkeypatch.setattr(routes, "model_assets_available", lambda: False)
    unavailable = TestClient(app).get("/ready")
    assert unavailable.status_code == 503