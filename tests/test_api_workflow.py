from fastapi.testclient import TestClient

from pickleball_track.api.main import app
from pickleball_track.services import video_jobs


def test_process_video_job_with_mocked_minio_and_inference(monkeypatch, tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"test checkpoint")
    monkeypatch.setattr(video_jobs, "DEFAULT_CHECKPOINT_PATH", checkpoint)

    calls = {}

    class FakeMinioClient:
        def fget_object(self, bucket, object_key, destination):
            calls["download"] = (bucket, object_key)
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

    response = TestClient(app).post(
        "/api/v1/process",
        json={
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
        },
    )

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