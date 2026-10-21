from pathlib import Path

import pytest

from pickleball_track.core.exceptions import (
    InvalidInputError,
    PayloadTooLargeError,
    ProcessingError,
    ServiceUnavailableError,
)
from pickleball_track.services.video_jobs import VideoJobService, feedback_for, score_counts


class FakeMinioClient:
    def __init__(self, reported_size=10, payload=b"test video"):
        self.reported_size = reported_size
        self.payload = payload
        self.downloads = []

    def stat_object(self, bucket, object_key):
        return type("ObjectInfo", (), {"size": self.reported_size})()

    def fget_object(self, bucket, object_key, destination):
        self.downloads.append((bucket, object_key, destination))
        Path(destination).write_bytes(self.payload)


class RecordingPredictor:
    def __init__(self, result=None, error=None):
        self.result = result or {"total_hits": 8, "total_misses": 2}
        self.error = error
        self.paths = []

    def predict(self, video_path):
        self.paths.append(Path(video_path))
        assert Path(video_path).is_file()
        if self.error:
            raise self.error
        return self.result


def _service(settings, predictor, client):
    return VideoJobService(settings, predictor, minio_client_factory=lambda endpoint, s: client)


def _process(service, object_key="matches/match.MOV"):
    return service.process(
        job_id="job-1", bucket="videos", object_key=object_key, minio_endpoint="minio:9000"
    )


@pytest.mark.parametrize(
    ("hits", "misses", "accuracy", "score", "feedback"),
    [
        (8, 2, 0.8, 8.0, "Thực hiện động tác đạt yêu cầu."),
        (2, 1, 0.6667, 6.67, "Cần luyện tập thêm để cải thiện độ chính xác."),
        (1, 3, 0.25, 2.5, "Chưa đạt yêu cầu, hãy luyện tập lại các động tác."),
        (0, 0, 0.0, 0.0, "Chưa đạt yêu cầu, hãy luyện tập lại các động tác."),
    ],
)
def test_scoring_preserves_product_formula(hits, misses, accuracy, score, feedback):
    job_score = score_counts(hits, misses)

    assert (job_score.accuracy, job_score.score, job_score.feedback) == (accuracy, score, feedback)
    assert (job_score.hit_count, job_score.miss_count) == (hits, misses)


def test_feedback_uses_unrounded_accuracy():
    assert feedback_for(0.79999) == "Cần luyện tập thêm để cải thiện độ chính xác."


def test_process_downloads_runs_inference_and_cleans_up(test_settings):
    client, predictor = FakeMinioClient(), RecordingPredictor()

    job_score = _process(_service(test_settings, predictor, client))

    assert job_score.score == 8.0
    assert client.downloads[0][:2] == ("videos", "matches/match.MOV")
    assert predictor.paths[0].suffix == ".mov"
    assert not predictor.paths[0].parent.exists()


@pytest.mark.parametrize("object_key", ["clip.avi", "clip", "folder.mp4/clip.txt"])
def test_unsupported_video_type_is_rejected_before_download(test_settings, object_key):
    client = FakeMinioClient()

    with pytest.raises(InvalidInputError):
        _process(_service(test_settings, RecordingPredictor(), client), object_key)
    assert client.downloads == []


def test_missing_model_is_service_unavailable(test_settings):
    client = FakeMinioClient()

    with pytest.raises(ServiceUnavailableError):
        _process(_service(test_settings, None, client))
    assert client.downloads == []


def test_oversized_object_is_rejected_before_download(test_settings):
    client = FakeMinioClient(reported_size=test_settings.max_video_bytes + 1)

    with pytest.raises(PayloadTooLargeError):
        _process(_service(test_settings, RecordingPredictor(), client))
    assert client.downloads == []


def test_oversized_download_is_rejected_before_inference(test_settings):
    client = FakeMinioClient(payload=b"x" * (test_settings.max_video_bytes + 1))
    predictor = RecordingPredictor()

    with pytest.raises(PayloadTooLargeError):
        _process(_service(test_settings, predictor, client))
    assert predictor.paths == []
    assert not Path(client.downloads[0][2]).exists()


def test_inference_failure_is_a_processing_error_not_no_event(test_settings):
    predictor = RecordingPredictor(error=RuntimeError("Video contains no decodable audio stream"))

    with pytest.raises(ProcessingError):
        _process(_service(test_settings, predictor, FakeMinioClient()))
    assert not predictor.paths[0].parent.exists()


@pytest.mark.parametrize(
    ("public_endpoint", "expected"),
    [(None, "minio:9000"), ("100.116.40.89:9000", "100.116.40.89:9000")],
)
def test_configured_public_endpoint_overrides_the_request(test_settings, public_endpoint, expected):
    from dataclasses import replace

    endpoints = []
    client, predictor = FakeMinioClient(), RecordingPredictor()
    service = VideoJobService(
        replace(test_settings, minio_public_endpoint=public_endpoint),
        predictor,
        minio_client_factory=lambda endpoint, s: endpoints.append(endpoint) or client,
    )

    _process(service)

    assert endpoints == [expected]


class FakeRangeClient(FakeMinioClient):
    """Serves ranged GETs like MinIO's get_object(offset=, length=)."""

    def __init__(self, payload):
        super().__init__(reported_size=len(payload), payload=payload)
        self.ranges = []

    def get_object(self, bucket, object_key, offset=0, length=0):
        self.ranges.append((offset, length))
        data = self.payload[offset:offset + length]

        class Response:
            def stream(self, amt):
                for i in range(0, len(data), 3):
                    yield data[i:i + 3]

            def close(self):
                pass

            def release_conn(self):
                pass

        return Response()


class ContentPredictor(RecordingPredictor):
    def predict(self, video_path):
        self.content = Path(video_path).read_bytes()
        return super().predict(video_path)


def test_large_video_is_downloaded_with_parallel_ranges(test_settings):
    from dataclasses import replace

    payload = bytes(range(256)) * 4  # 1024 bytes
    client, predictor = FakeRangeClient(payload), ContentPredictor()
    settings = replace(test_settings, download_parts=4, download_part_min_bytes=100)

    _process(_service(settings, predictor, client))

    assert predictor.content == payload
    assert sorted(client.ranges) == [(0, 256), (256, 256), (512, 256), (768, 256)]
    assert client.downloads == []


def test_small_video_uses_a_single_request(test_settings):
    from dataclasses import replace

    client, predictor = FakeRangeClient(b"tiny video"), ContentPredictor()
    settings = replace(test_settings, download_parts=4, download_part_min_bytes=100)

    _process(_service(settings, predictor, client))

    assert predictor.content == b"tiny video"
    assert client.ranges == [] and len(client.downloads) == 1
