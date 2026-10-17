"""Band-limited sinc resampling, ported from ``torchaudio.functional.resample``.

The original inference script resampled audio with torchaudio, which has no
release for the PyTorch version this service runs on. This is the torchaudio
2.11 implementation restricted to its default ``sinc_interp_hann`` method, kept
operation-for-operation identical so features match training.

Copyright (c) Facebook, Inc. and its affiliates. BSD 2-Clause License
(https://github.com/pytorch/audio/blob/main/LICENSE).
"""

from __future__ import annotations

import math

import torch


def _get_sinc_resample_kernel(
    orig_freq: int,
    new_freq: int,
    gcd: int,
    lowpass_filter_width: int,
    rolloff: float,
    device: torch.device,
    dtype: torch.dtype | None,
) -> tuple[torch.Tensor, int]:
    orig_freq = int(orig_freq) // gcd
    new_freq = int(new_freq) // gcd

    if lowpass_filter_width <= 0:
        raise ValueError("Low pass filter width should be positive.")
    # Antialiasing: cut the highest frequencies when down- or upsampling.
    base_freq = min(orig_freq, new_freq) * rolloff

    # y[j] = sum_i x[i] sinc(pi * orig_freq * (i / orig_freq - j / new_freq)), truncated
    # after `lowpass_filter_width` zero crossings; the filter repeats every new_freq outputs.
    width = math.ceil(lowpass_filter_width * orig_freq / base_freq)
    idx_dtype = dtype if dtype is not None else torch.float64

    idx = torch.arange(-width, width + orig_freq, dtype=idx_dtype, device=device)[None, None] / orig_freq

    t = torch.arange(0, -new_freq, -1, dtype=dtype, device=device)[:, None, None] / new_freq + idx
    t *= base_freq
    t = t.clamp_(-lowpass_filter_width, lowpass_filter_width)

    window = torch.cos(t * math.pi / lowpass_filter_width / 2) ** 2

    t *= math.pi

    scale = base_freq / orig_freq
    kernels = torch.where(t == 0, torch.tensor(1.0).to(t), t.sin() / t)
    kernels *= window * scale

    if dtype is None:
        kernels = kernels.to(dtype=torch.float32)

    return kernels, width


def _apply_sinc_resample_kernel(
    waveform: torch.Tensor,
    orig_freq: int,
    new_freq: int,
    gcd: int,
    kernel: torch.Tensor,
    width: int,
) -> torch.Tensor:
    if not waveform.is_floating_point():
        raise TypeError(f"Expected floating point type for waveform tensor, but received {waveform.dtype}.")

    orig_freq = int(orig_freq) // gcd
    new_freq = int(new_freq) // gcd

    shape = waveform.size()
    waveform = waveform.view(-1, shape[-1])

    num_wavs, length = waveform.shape
    waveform = torch.nn.functional.pad(waveform, (width, width + orig_freq))
    resampled = torch.nn.functional.conv1d(waveform[:, None], kernel, stride=orig_freq)
    resampled = resampled.transpose(1, 2).reshape(num_wavs, -1)
    target_length = torch.ceil(torch.as_tensor(new_freq * length / orig_freq)).long()
    resampled = resampled[..., :target_length]

    return resampled.view(shape[:-1] + resampled.shape[-1:])


def resample(
    waveform: torch.Tensor,
    orig_freq: int,
    new_freq: int,
    lowpass_filter_width: int = 6,
    rolloff: float = 0.99,
) -> torch.Tensor:
    """Resample ``(..., time)`` from ``orig_freq`` to ``new_freq`` (torchaudio defaults)."""
    if orig_freq <= 0.0 or new_freq <= 0.0:
        raise ValueError("Original frequency and desired frequecy should be positive")
    if orig_freq == new_freq:
        return waveform

    gcd = math.gcd(int(orig_freq), int(new_freq))
    kernel, width = _get_sinc_resample_kernel(
        orig_freq, new_freq, gcd, lowpass_filter_width, rolloff, waveform.device, waveform.dtype,
    )
    return _apply_sinc_resample_kernel(waveform, orig_freq, new_freq, gcd, kernel, width)
