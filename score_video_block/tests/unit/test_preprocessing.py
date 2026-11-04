import pytest
import torch

from pickleball_track.core.config import AUDIO_SR
from pickleball_track.pipelines.preprocessing import (
    VideoInfo,
    get_video_info,
    load_audio,
    num_sampled_frames,
    window_starts,
)


@pytest.mark.parametrize(
    ("num_sampled", "expected"),
    # Training's window_starts: stride 16, even starts, the last one ends at the end.
    [(64, [0, 16, 32]), (50, [0, 16, 18]), (35, [0, 2]), (33, [0]), (32, [0]), (10, [0])],
)
def test_window_starts_match_training(num_sampled, expected):
    assert window_starts(num_sampled) == expected


def test_window_starts_are_even():
    assert all(start % 2 == 0 for start in window_starts(1001))


@pytest.mark.parametrize(
    ("fps", "num_frames", "expected"), [(10.0, 50, 150), (60.0, 120, 60), (29.97, 300, 300), (30.0, 0, 1)]
)
def test_sampled_frames_cover_the_duration_at_30_fps(fps, num_frames, expected):
    assert num_sampled_frames(VideoInfo(fps=fps, num_frames=num_frames, width=1, height=1)) == expected


def test_video_info(clip_path):
    info = get_video_info(clip_path)

    assert (info.fps, info.num_frames, info.width, info.height) == (10.0, 50, 64, 48)
    assert info.duration == 5.0


def test_unreadable_video_raises(tmp_path):
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a video")

    with pytest.raises(RuntimeError, match="Cannot open video"):
        get_video_info(broken)


def test_load_audio_decodes_mono_16khz(clip_path):
    waveform = load_audio(clip_path)

    assert waveform.ndim == 1
    assert waveform.dtype == torch.float32
    assert abs(waveform.numel() / AUDIO_SR - 5.0) < 0.1


def test_video_without_audio_gives_an_empty_waveform(clip_no_audio_path):
    assert load_audio(clip_no_audio_path).numel() == 0
