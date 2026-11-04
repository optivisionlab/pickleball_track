import torch

from pickleball_track.core.config import AUDIO_SR, BEATS_PATCH_SEC
from pickleball_track.models.components.audio_encoder import AudioEncoder, AudioTimeline

from ..helpers import TINY_AUDIO_DIM, build_fake_beats


def _encoder(chunk_batch=8):
    return AudioEncoder(build_fake_beats(), torch.device("cpu"), chunk_batch=chunk_batch)


def test_short_recordings_have_no_audio():
    assert _encoder().timeline(torch.zeros(AUDIO_SR // 2 - 1)) is None


def test_timeline_has_one_step_per_patch_at_its_centre_time():
    wave = torch.randn(AUDIO_SR * 3)

    timeline = _encoder().timeline(wave)

    steps = int(3 / BEATS_PATCH_SEC)
    assert timeline.features.shape == (steps, TINY_AUDIO_DIM)
    assert torch.allclose(timeline.times[:2], torch.tensor([0.08, 0.24]))


def test_chunks_cover_long_recordings_and_batching_does_not_change_features():
    torch.manual_seed(0)
    wave = torch.randn(AUDIO_SR * 23 + 3000)  # 2 full 10 s chunks + a 3.2 s tail

    batched = _encoder(chunk_batch=8).timeline(wave)
    single = _encoder(chunk_batch=1).timeline(wave)

    assert torch.allclose(batched.features, single.features, atol=1e-6)
    assert torch.equal(batched.times, single.times)
    assert (batched.times.diff() > 0).all()
    assert batched.times[-1] < wave.numel() / AUDIO_SR


def test_tail_shorter_than_a_quarter_second_is_dropped():
    timeline = _encoder().timeline(torch.randn(AUDIO_SR * 10 + AUDIO_SR // 5))

    assert timeline.times[-1] < 10.0


def test_interpolation_is_linear_and_clamped():
    timeline = AudioTimeline(torch.tensor([[0.0], [1.0], [3.0]]), torch.tensor([0.0, 1.0, 2.0]))

    values = timeline.at(torch.tensor([[-1.0, 0.5], [1.5, 9.0]]))

    assert values.shape == (2, 2, 1)
    assert values[..., 0].tolist() == [[0.0, 0.5], [2.0, 3.0]]
