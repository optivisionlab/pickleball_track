import shutil
from contextlib import closing

import cv2
import numpy as np
import pytest

from pickleball_track.pipelines.frames import FFmpegCapture, iter_frames

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg CLI not installed")


def _read_all(capture):
    frames, stamps = [], []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                return frames, stamps
            frames.append(frame.copy())
            stamps.append(capture.get(cv2.CAP_PROP_POS_MSEC))
    finally:
        capture.release()


def test_ffmpeg_capture_matches_opencv_frames_and_timestamps(clip_path):
    opencv = cv2.VideoCapture(str(clip_path))
    opencv.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    expected, expected_stamps = _read_all(opencv)
    frames, stamps = _read_all(FFmpegCapture(clip_path))

    assert len(frames) == len(expected) > 0
    assert stamps == pytest.approx(expected_stamps, abs=1e-3)
    # 8-bit video: same picture; only YUV->BGR rounding may differ slightly.
    diff = max(int(np.abs(a.astype(int) - b.astype(int)).max()) for a, b in zip(frames, expected, strict=True))
    assert diff <= 8


def test_ffmpeg_capture_reads_into_the_given_buffer(clip_path):
    capture = FFmpegCapture(clip_path)
    try:
        buffer = np.zeros((capture.height, capture.width, 3), dtype=np.uint8)
        ok, frame = capture.read(buffer)
        assert ok and frame is buffer and buffer.any()
    finally:
        capture.release()


def test_ffmpeg_capture_reports_unreadable_files(tmp_path):
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a video")
    assert not FFmpegCapture(broken).isOpened()


def test_iter_frames_with_ffmpeg_decoder_yields_every_frame(clip_path):
    with closing(iter_frames(clip_path)) as frames:
        expected = [stamp for _, stamp in frames]
    with closing(iter_frames(clip_path, decoder="ffmpeg")) as frames:
        stamps = [stamp for _, stamp in frames]

    assert stamps == pytest.approx(expected, abs=1e-3)
