import pytest
import torch
from transformers import VideoMAEImageProcessor

from pickleball_track.core.config import AUDIO_SR, NUM_VIDEO_FRAMES, WEIGHTS_DIR
from pickleball_track.pipelines.preprocessing import (
    build_windows,
    get_audio_window,
    get_video_info,
    load_audio,
    read_video_window,
)


def test_audio_window_padding_matches_expected_shape_and_mask():
    waveform = torch.tensor([[0.25, 0.5]])

    segment, padding_mask = get_audio_window(waveform, start_time=0.0, duration=0.00025)

    assert segment.shape == (1, 4)
    assert torch.equal(padding_mask, torch.tensor([[False, False, True, True]]))
    assert torch.equal(segment, torch.tensor([[0.25, 0.5, 0.0, 0.0]]))


def test_audio_window_slices_from_start_time():
    waveform = torch.arange(AUDIO_SR * 2, dtype=torch.float32).unsqueeze(0)

    segment, padding_mask = get_audio_window(waveform, start_time=1.0, duration=0.5)

    assert segment.shape == (1, AUDIO_SR // 2)
    assert segment[0, 0].item() == AUDIO_SR
    assert not padding_mask.any()


@pytest.mark.parametrize(
    ("num_frames", "expected"),
    [(64, [0, 16, 32]), (50, [0, 16, 18]), (32, [0]), (10, [0])],
)
def test_sliding_windows_preserve_half_window_overlap(num_frames, expected):
    assert build_windows(num_frames=num_frames, num_window_frames=32) == expected


def test_sliding_window_overlap_is_configurable():
    assert build_windows(num_frames=64, num_window_frames=32, overlap=0.75) == [0, 8, 16, 24, 32]
    assert build_windows(num_frames=64, num_window_frames=32, overlap=0.0) == [0, 32]


def test_video_info(clip_path):
    info = get_video_info(clip_path)

    assert (info.fps, info.num_frames, info.width, info.height) == (10.0, 50, 64, 48)
    assert info.duration == 5.0


def test_unreadable_video_raises(tmp_path):
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a video")

    with pytest.raises(RuntimeError, match="Cannot open video"):
        get_video_info(broken)


def test_read_video_window_uses_repository_videomae_preprocessing(clip_path):
    processor = VideoMAEImageProcessor.from_pretrained(
        str(WEIGHTS_DIR / "videomae-base"), local_files_only=True
    )

    pixel_values, valid = read_video_window(clip_path, 40, NUM_VIDEO_FRAMES, processor)

    assert pixel_values.shape == (1, NUM_VIDEO_FRAMES, 3, 224, 224)
    assert pixel_values.dtype == torch.float32
    # 10 frames remain after frame 40; the rest repeat the last frame and are invalid.
    assert valid.tolist() == [True] * 10 + [False] * 22


def test_load_audio_decodes_mono_16khz(clip_path):
    waveform = load_audio(clip_path)

    assert waveform.shape[0] == 1
    assert waveform.dtype == torch.float32
    assert abs(waveform.shape[1] / AUDIO_SR - 5.0) < 0.1


def test_video_without_audio_is_an_error(clip_no_audio_path):
    with pytest.raises(RuntimeError, match="audio"):
        load_audio(clip_no_audio_path)
