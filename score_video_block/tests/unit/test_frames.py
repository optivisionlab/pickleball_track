import subprocess

import cv2
import numpy as np
import pytest
import torch

from pickleball_track.core.config import IMAGENET_MEAN, IMAGENET_STD, NUM_VIDEO_FRAMES
from pickleball_track.pipelines.frames import (
    CpuFrames,
    FrameSpec,
    ProcessFramePool,
    TorchFrames,
    frame_counts,
    iter_frames,
    iter_window_batches,
    preprocessed_chunks,
    sample_frames,
)

SPEC = FrameSpec()


def _random_frames(height, width, count=3, seed=0):
    rng = np.random.default_rng(seed)
    # Smooth content: resampling differences stay at rounding level, as on real video.
    small = rng.integers(0, 256, size=(count, height // 8, width // 8, 3), dtype=np.uint8)
    return [cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR) for frame in small]


def _training_pixels(bgr_frames, size=224):
    """Training (baseline/pickleball_common.py): cv2 INTER_AREA resize, RGB, float32 /255, ImageNet norm."""
    rgb = np.stack([
        cv2.cvtColor(cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        for frame in bgr_frames
    ])
    pixels = torch.from_numpy(rgb).permute(0, 3, 1, 2).float().div_(255.0)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (pixels - mean) / std


def _training_sampled(video_path, num_frames, fps):
    """Training's decode_sampled: OpenCV (auto-rotated) frames round(i * fps / 30), repeated as needed."""
    num_sampled = max(1, int(np.floor(num_frames / fps * 30.0)))
    targets = np.minimum(np.round(np.arange(num_sampled) * fps / 30.0).astype(int), num_frames - 1)
    capture = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    return [frames[t] for t in targets]


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


@pytest.mark.parametrize(("height", "width"), [(1080, 1920), (1920, 1080), (48, 64)])
def test_cpu_frames_are_bit_identical_to_training_preprocessing(height, width):
    frames = _random_frames(height, width)

    assert torch.equal(CpuFrames(SPEC)(frames), _training_pixels(frames))


@pytest.mark.parametrize(("height", "width"), [(1080, 1920), (1920, 1080), (448, 448)])
def test_torch_frames_are_close_to_training_preprocessing(height, width):
    frames = _random_frames(height, width)

    expected = CpuFrames(SPEC)(frames)
    actual = TorchFrames(SPEC, torch.device("cpu"))(frames)

    assert actual.shape == expected.shape == (3, 3, 224, 224)
    # One 8-bit level after normalization is 1 / 255 / min(std) ~= 0.0175; torch's area
    # mode only approximates INTER_AREA for non-integer factors (opt-in GPU path).
    assert (actual - expected).abs().mean() < 0.05


@pytest.mark.parametrize(
    ("num_frames", "fps", "expected_sampled"),
    [(50, 10.0, 150), (120, 60.0, 60), (90, 30.0, 90), (1, 30.0, 1)],
)
def test_frame_counts_follow_training_sampling(num_frames, fps, expected_sampled):
    counts = frame_counts(num_frames, fps)
    num_sampled = max(1, int(np.floor(num_frames / fps * 30.0)))
    targets = np.minimum(np.round(np.arange(num_sampled) * fps / 30.0).astype(int), num_frames - 1)

    assert counts.sum() == expected_sampled
    assert np.array_equal(np.repeat(np.arange(num_frames), counts), targets)


def test_sixty_fps_keeps_every_other_frame():
    counts = frame_counts(120, 60.0)

    assert counts[:6].tolist() == [1, 0, 1, 0, 1, 0]


def test_sampled_frames_repeat_and_skip_source_frames(clip_path):
    counts = frame_counts(50, 10.0)

    frames = list(sample_frames(iter_frames(clip_path), counts))

    reference = _training_sampled(clip_path, 50, 10.0)
    assert len(frames) == len(reference) == 150
    assert all(np.array_equal(a, b) for a, b in zip(frames, reference, strict=True))


@pytest.fixture(scope="module")
def frame_pool():
    pool = ProcessFramePool(SPEC, workers=2)
    yield pool
    pool.close()


def _reference(video_path, num_frames=50, fps=10.0):
    return _training_pixels(_training_sampled(video_path, num_frames, fps))


def test_threaded_chunks_match_training(clip_path):
    pixels = torch.cat(list(preprocessed_chunks(
        sample_frames(iter_frames(clip_path), frame_counts(50, 10.0)), CpuFrames(SPEC), chunk=3, workers=2,
    )))

    assert torch.equal(pixels, _reference(clip_path))


@pytest.mark.parametrize("chunk", [1, 4, 7])
def test_process_pool_chunks_match_training(clip_path, frame_pool, chunk):
    chunks = list(frame_pool.chunks(clip_path, frame_counts(50, 10.0), chunk=chunk))

    assert torch.equal(torch.cat(chunks), _reference(clip_path))


def test_process_pool_skips_frames_that_are_not_sampled(clip_path, frame_pool):
    # As if the 10 fps clip were 60 fps: only every other frame is used.
    counts = frame_counts(50, 60.0)

    pixels = torch.cat(list(frame_pool.chunks(clip_path, counts)))

    frames = [frame for frame, _ in iter_frames(clip_path)]
    assert len(pixels) == counts.sum() == 25
    assert torch.equal(pixels, _training_pixels(frames[::2]))


def test_process_pool_stops_after_the_last_sampled_frame(clip_path, frame_pool):
    pixels = torch.cat(list(frame_pool.chunks(clip_path, frame_counts(10, 10.0))))

    assert len(pixels) == 30
    assert torch.equal(pixels, _reference(clip_path, num_frames=10))


@pytest.mark.parametrize("display_rotation", [90, 180, 270])
def test_process_pool_matches_opencv_auto_rotation(clip_path, frame_pool, tmp_path, display_rotation):
    # Phone videos store rotation metadata; OpenCV (and so training) rotates frames when reading.
    rotated = tmp_path / f"rotated_{display_rotation}.mp4"
    made = subprocess.run(
        ["ffmpeg", "-v", "error", "-display_rotation", str(display_rotation), "-i", str(clip_path),
         "-c", "copy", str(rotated)],
        capture_output=True,
    )
    if made.returncode:
        pytest.skip("this ffmpeg has no -display_rotation (FFmpeg >= 6 needed)")
    capture = cv2.VideoCapture(str(rotated))
    if capture.get(cv2.CAP_PROP_ORIENTATION_META) not in (90, 180, 270):
        capture.release()
        pytest.skip("this OpenCV build does not report the rotation metadata")
    capture.release()

    pixels = torch.cat(list(frame_pool.chunks(rotated, frame_counts(50, 10.0))))

    assert torch.equal(pixels, _reference(rotated))


def test_process_pool_recovers_after_an_interrupted_video(clip_path, frame_pool):
    chunks = frame_pool.chunks(clip_path, frame_counts(50, 10.0), chunk=4)
    next(chunks)
    chunks.close()  # consumer stops early, e.g. inference raised

    pixels = torch.cat(list(frame_pool.chunks(clip_path, frame_counts(50, 10.0))))

    assert torch.equal(pixels, _reference(clip_path))


def test_process_pool_restarts_dead_workers(clip_path, frame_pool):
    frame_pool._processes[0].kill()
    frame_pool._processes[0].join()

    pixels = torch.cat(list(frame_pool.chunks(clip_path, frame_counts(50, 10.0))))

    assert torch.equal(pixels, _reference(clip_path))
    assert all(process.is_alive() for process in frame_pool._processes)


def test_worker_crash_mid_video_raises_instead_of_hanging(clip_path, frame_pool):
    chunks = frame_pool.chunks(clip_path, frame_counts(50, 10.0), chunk=1, prefetch_frames=1)
    next(chunks)
    frame_pool._processes[1].kill()

    with pytest.raises(RuntimeError, match="exited unexpectedly|Cannot decode video"):
        list(chunks)

    # The next video gets fresh workers.
    pixels = torch.cat(list(frame_pool.chunks(clip_path, frame_counts(50, 10.0))))
    assert torch.equal(pixels, _reference(clip_path))


def test_process_pool_unreadable_video_raises(tmp_path, frame_pool):
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a video")

    with pytest.raises(RuntimeError, match="Cannot open video"):
        next(frame_pool.chunks(broken, frame_counts(50, 10.0)))


def test_process_pool_checks_shared_memory_capacity(frame_pool):
    assert frame_pool.fits(48, 64)
    assert not frame_pool.fits(100_000, 100_000)


def _chunks(clip_path, counts=None):
    counts = frame_counts(50, 10.0) if counts is None else counts
    return preprocessed_chunks(sample_frames(iter_frames(clip_path), counts), CpuFrames(SPEC))


def test_windows_past_the_end_repeat_last_frame_and_are_invalid(clip_path):
    batches = list(iter_window_batches(_chunks(clip_path), [0, 16, 128], NUM_VIDEO_FRAMES, batch_size=2))

    assert [b.starts for b in batches] == [[0, 16], [128]]
    assert batches[0].pixel_values.shape == (2, NUM_VIDEO_FRAMES, 3, 224, 224)
    last = batches[1]
    # 22 sampled frames remain after frame 128; the rest repeat the last frame and are invalid.
    assert last.frame_valid.tolist() == [[True] * 22 + [False] * 10]
    assert torch.equal(last.pixel_values[0, 21], last.pixel_values[0, -1])
    # Window 16 and window 0 share sampled frames 16..31.
    assert torch.equal(batches[0].pixel_values[0, 16], batches[0].pixel_values[1, 0])


def test_window_starting_after_last_decoded_frame_is_entirely_invalid(clip_path):
    # The container claimed more frames than could be decoded (training masks them too).
    batch = next(iter_window_batches(_chunks(clip_path), [160], NUM_VIDEO_FRAMES, batch_size=1))

    assert not batch.frame_valid.any()
    assert batch.pixel_values.shape == (1, NUM_VIDEO_FRAMES, 3, 224, 224)


def test_video_without_frames_raises(clip_path):
    with pytest.raises(RuntimeError, match="Cannot read frames at 0"):
        next(iter_window_batches(iter([]), [0], NUM_VIDEO_FRAMES, batch_size=1))
