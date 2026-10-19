import subprocess

import cv2
import numpy as np
import pytest
import torch
from transformers import VideoMAEImageProcessor

from pickleball_track.core.config import NUM_VIDEO_FRAMES, WEIGHTS_DIR
from pickleball_track.pipelines.frames import (
    FrameTimeline,
    ProcessFramePool,
    ProcessorFrames,
    TorchFrames,
    iter_frames,
    iter_window_batches,
    preprocessed_chunks,
)


@pytest.fixture(scope="module")
def processor():
    return VideoMAEImageProcessor.from_pretrained(str(WEIGHTS_DIR / "videomae-base"), local_files_only=True)


def _random_frames(height, width, count=3, seed=0):
    rng = np.random.default_rng(seed)
    # Smooth content: resampling differences stay at rounding level, as on real video.
    small = rng.integers(0, 256, size=(count, height // 8, width // 8, 3), dtype=np.uint8)
    return [cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR) for frame in small]


def test_frames_are_decoded_in_order_with_timestamps(clip_path):
    decoded = list(iter_frames(clip_path))

    assert len(decoded) == 50
    assert [round(ts, 4) for _, ts in decoded[:3]] == [0.0, 0.1, 0.2]
    assert decoded[0][0].shape == (48, 64, 3)


def test_unreadable_video_raises(tmp_path):
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a video")

    with pytest.raises(RuntimeError, match="Cannot open video"):
        next(iter_frames(broken))


def test_processor_frames_match_hugging_face_processor(processor):
    frames = _random_frames(48, 64)
    rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]

    expected = processor(rgb, return_tensors="pt")["pixel_values"][0]

    assert torch.equal(ProcessorFrames(processor)(frames), expected)


@pytest.mark.parametrize(("height", "width"), [(1080, 1920), (1920, 1080), (48, 64)])
def test_torch_frames_match_processor_within_one_pixel_level(processor, height, width):
    frames = _random_frames(height, width)

    expected = ProcessorFrames(processor)(frames)
    actual = TorchFrames(processor, torch.device("cpu"))(frames)

    assert actual.shape == expected.shape == (3, 3, 224, 224)
    # One 8-bit level after normalization is 1 / 255 / min(std) ~= 0.0175.
    assert (actual - expected).abs().max() < 0.018
    assert (actual - expected).abs().mean() < 0.004


@pytest.fixture(scope="module")
def frame_pool(processor):
    pool = ProcessFramePool(processor, workers=2)
    yield pool
    pool.close()


def _hf_pixels(video_path, processor):
    """Reference: every frame through the Hugging Face processor."""
    frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame, _ in iter_frames(video_path)]
    return processor(frames, return_tensors="pt")["pixel_values"][0]


def test_threaded_chunks_are_bit_identical_to_the_processor(clip_path, processor):
    timeline = FrameTimeline(fps=10.0)

    pixels = torch.cat(list(preprocessed_chunks(
        iter_frames(clip_path), ProcessorFrames(processor), timeline, chunk=3, workers=2,
    )))

    assert torch.equal(pixels, _hf_pixels(clip_path, processor))
    assert len(timeline.timestamps) == 50


@pytest.mark.parametrize("chunk", [1, 4, 7])
def test_process_pool_chunks_are_bit_identical_to_the_processor(clip_path, processor, frame_pool, chunk):
    timeline = FrameTimeline(fps=10.0)

    chunks = list(frame_pool.chunks(clip_path, timeline, chunk=chunk))

    assert all(len(c) == chunk for c in chunks[:-1])
    assert torch.equal(torch.cat(chunks), _hf_pixels(clip_path, processor))
    assert [round(ts, 4) for ts in timeline.timestamps[:3]] == [0.0, 0.1, 0.2]
    assert len(timeline.timestamps) == 50


@pytest.mark.parametrize("display_rotation", [90, 180, 270])
def test_process_pool_matches_opencv_auto_rotation(clip_path, processor, frame_pool, tmp_path, display_rotation):
    # Phone videos store rotation metadata; OpenCV rotates frames when reading.
    rotated = tmp_path / f"rotated_{display_rotation}.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-display_rotation", str(display_rotation), "-i", str(clip_path),
         "-c", "copy", str(rotated)],
        check=True,
    )
    capture = cv2.VideoCapture(str(rotated))
    assert capture.get(cv2.CAP_PROP_ORIENTATION_META) in (90, 180, 270)
    capture.release()

    pixels = torch.cat(list(frame_pool.chunks(rotated, FrameTimeline(fps=10.0))))

    assert torch.equal(pixels, _hf_pixels(rotated, processor))


def test_process_pool_recovers_after_an_interrupted_video(clip_path, processor, frame_pool):
    chunks = frame_pool.chunks(clip_path, FrameTimeline(fps=10.0), chunk=4)
    next(chunks)
    chunks.close()  # consumer stops early, e.g. inference raised

    pixels = torch.cat(list(frame_pool.chunks(clip_path, FrameTimeline(fps=10.0))))

    assert torch.equal(pixels, _hf_pixels(clip_path, processor))


def test_process_pool_restarts_dead_workers(clip_path, processor, frame_pool):
    frame_pool._processes[0].kill()
    frame_pool._processes[0].join()

    pixels = torch.cat(list(frame_pool.chunks(clip_path, FrameTimeline(fps=10.0))))

    assert torch.equal(pixels, _hf_pixels(clip_path, processor))
    assert all(process.is_alive() for process in frame_pool._processes)


def test_worker_crash_mid_video_raises_instead_of_hanging(clip_path, processor, frame_pool):
    chunks = frame_pool.chunks(clip_path, FrameTimeline(fps=10.0), chunk=1, prefetch_frames=1)
    next(chunks)
    frame_pool._processes[1].kill()

    with pytest.raises(RuntimeError, match="exited unexpectedly|Cannot decode video"):
        list(chunks)

    # The next video gets fresh workers.
    pixels = torch.cat(list(frame_pool.chunks(clip_path, FrameTimeline(fps=10.0))))
    assert torch.equal(pixels, _hf_pixels(clip_path, processor))


def test_process_pool_unreadable_video_raises(tmp_path, frame_pool):
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a video")

    with pytest.raises(RuntimeError, match="Cannot open video"):
        next(frame_pool.chunks(broken, FrameTimeline(fps=10.0)))


def test_process_pool_checks_shared_memory_capacity(frame_pool):
    assert frame_pool.fits(48, 64)
    assert not frame_pool.fits(100_000, 100_000)


def _chunks(clip_path, processor, timeline):
    return preprocessed_chunks(iter_frames(clip_path), ProcessorFrames(processor), timeline)


def test_windows_past_the_end_repeat_last_frame_and_are_invalid(clip_path, processor):
    timeline = FrameTimeline(fps=10.0)

    batches = list(iter_window_batches(
        _chunks(clip_path, processor, timeline), [0, 16, 40], NUM_VIDEO_FRAMES, batch_size=2,
    ))

    assert [b.starts for b in batches] == [[0, 16], [40]]
    assert batches[0].pixel_values.shape == (2, NUM_VIDEO_FRAMES, 3, 224, 224)
    last = batches[1]
    # 10 frames remain after frame 40; the rest repeat the last frame and are invalid.
    assert last.frame_valid.tolist() == [[True] * 10 + [False] * 22]
    assert torch.equal(last.pixel_values[0, 9], last.pixel_values[0, -1])
    # Window 16 and window 0 share frames 16..31.
    assert torch.equal(batches[0].pixel_values[0, 16], batches[0].pixel_values[1, 0])
    assert len(timeline.timestamps) == 50


def test_window_starting_after_last_decoded_frame_raises(clip_path, processor):
    batches = iter_window_batches(
        _chunks(clip_path, processor, FrameTimeline(fps=10.0)), [60], NUM_VIDEO_FRAMES, batch_size=1,
    )

    with pytest.raises(RuntimeError, match="Cannot read frames at 60"):
        next(batches)


def test_timeline_uses_nominal_time_for_constant_frame_rate():
    timeline = FrameTimeline(fps=30.0, timestamps=[i / 30.0 + 0.001 for i in range(10)])

    assert timeline.time(4) == 4 / 30.0
    assert timeline.time(25) == 25 / 30.0


def test_timeline_follows_timestamps_on_variable_frame_rate():
    # Frames arrive every 50 ms while the container claims 30 fps.
    timeline = FrameTimeline(fps=30.0, timestamps=[i * 0.05 for i in range(10)])

    assert timeline.time(8) == pytest.approx(0.4)
    assert timeline.time(12) == pytest.approx(0.45 + 3 / 30.0)
