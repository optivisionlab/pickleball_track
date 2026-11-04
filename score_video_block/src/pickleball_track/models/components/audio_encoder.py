"""BEATs backbone wrapper: whole waveform -> audio features at video token times."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ...core.config import AUDIO_SR, BEATS_CHUNK_SEC, BEATS_PATCH_SEC

# BEATs tokens are time-major with this many frequency patches per time step.
FREQUENCY_PATCHES = 8


class AudioTimeline:
    """BEATs features of a whole recording, ``features`` (Ta, D) at ``times`` (Ta,) seconds."""

    def __init__(self, features: torch.Tensor, times: torch.Tensor):
        self.features = features
        self.times = times

    def at(self, query: torch.Tensor) -> torch.Tensor:
        """Linear interpolation at ``query`` times (...,) -> (..., D) (training's ``interpolate_timeline``)."""
        flat = query.reshape(-1).to(self.times.device, torch.float32)
        if self.features.shape[0] == 1:
            out = self.features.expand(flat.numel(), -1)
        else:
            right = torch.searchsorted(self.times, flat).clamp(1, self.times.numel() - 1)
            left = right - 1
            span = (self.times[right] - self.times[left]).clamp_min(1e-6)
            weight = ((flat - self.times[left]) / span).clamp(0, 1)[:, None]
            out = self.features[left] * (1 - weight) + self.features[right] * weight
        return out.reshape(*query.shape, -1)


class AudioEncoder:
    """Runs ``BEATs.extract_features`` over a recording in BEATS_CHUNK_SEC chunks.

    Always runs in fp32: BEATs' Kaldi fbank overflows in fp16.
    """

    def __init__(self, model: Any, device: torch.device, chunk_batch: int = 8):
        # `model` is a microsoft/unilm BEATs instance loaded from its source checkout.
        self.model = model
        self.device = device
        # Full-length chunks per forward pass (identical inputs to one-at-a-time calls).
        self.chunk_batch = chunk_batch

    @torch.inference_mode()
    def features(self, segments: torch.Tensor) -> torch.Tensor:
        """Equal-length segments (B, S) -> per-time-step features (B, steps, D), frequency-averaged."""
        out, _ = self.model.extract_features(segments.to(self.device, non_blocking=True), padding_mask=None)
        out = out.float()
        steps = out.shape[1] // FREQUENCY_PATCHES
        return out[:, : steps * FREQUENCY_PATCHES].reshape(out.shape[0], steps, FREQUENCY_PATCHES, -1).mean(2)

    @torch.inference_mode()
    def timeline(self, wave: torch.Tensor) -> AudioTimeline | None:
        """Training's ``beats_timeline``: None for recordings shorter than half a second."""
        wave = wave.reshape(-1).float()
        if wave.numel() < AUDIO_SR // 2:
            return None
        chunk = int(BEATS_CHUNK_SEC * AUDIO_SR)
        begins = []
        for begin in range(0, wave.numel(), chunk):
            if wave.numel() - begin < AUDIO_SR // 4:
                break  # training drops a final chunk shorter than 0.25 s
            begins.append(begin)
        full = [b for b in begins if b + chunk <= wave.numel()]
        feats, times = [], []
        for i in range(0, len(full), self.chunk_batch):
            group = full[i:i + self.chunk_batch]
            out = self.features(torch.stack([wave[b:b + chunk] for b in group]))
            for b, chunk_feats in zip(group, out, strict=True):
                feats.append(chunk_feats)
                times.append(self._times(b, chunk_feats.shape[0]))
        for b in begins[len(full):]:  # the last, shorter chunk
            segment = wave[b:]
            length = segment.numel()
            if length < AUDIO_SR:
                segment = F.pad(segment, (0, AUDIO_SR - length))
            chunk_feats = self.features(segment.unsqueeze(0))[0]
            t = self._times(b, chunk_feats.shape[0])
            keep = t < (b + length) / AUDIO_SR
            feats.append(chunk_feats[keep.to(chunk_feats.device)])
            times.append(t[keep])
        if not feats or sum(len(t) for t in times) == 0:
            return None
        return AudioTimeline(torch.cat(feats), torch.cat(times).to(self.device))

    def _times(self, begin: int, steps: int) -> torch.Tensor:
        return begin / AUDIO_SR + (torch.arange(steps, dtype=torch.float32) + 0.5) * BEATS_PATCH_SEC

    def feature_dim(self) -> int:
        """Feature size, from one second of silence (it does not depend on the content)."""
        return int(self.features(torch.zeros(1, AUDIO_SR)).shape[-1])
