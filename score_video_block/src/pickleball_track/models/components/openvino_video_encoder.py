"""VideoMAE on Intel accelerators (NPU, integrated GPU) through OpenVINO.

Same contract as ``VideoEncoder``. The VideoMAE model (already adapted to
NUM_VIDEO_FRAMES) is converted once at startup with a static single-window shape
and compiled on every configured device. For a batch, each device takes the next
window that is not taken yet, so a faster device scores more windows. Inference
releases the GIL, so the audio model can run on the CPU meanwhile.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import count
from typing import Any

import numpy as np
import torch
from transformers import VideoMAEModel

from ...core.config import NUM_VIDEO_FRAMES
from ...core.exceptions import ModelLoadError

logger = logging.getLogger(__name__)


class _LastHiddenState(torch.nn.Module):
    def __init__(self, model: VideoMAEModel):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=pixel_values).last_hidden_state


@dataclass
class _Device:
    name: str
    request: Any
    # One infer request per device; concurrent slots take turns.
    lock: threading.Lock = field(default_factory=threading.Lock)
    windows: int = 0


def _match_device(wanted: str, available: Sequence[str]) -> str | None:
    """'GPU' matches 'GPU' or 'GPU.0'; an explicit 'GPU.1' must be listed."""
    if wanted in available:
        return wanted
    return next((device for device in available if device.split(".")[0] == wanted), None)


class OpenVINOVideoEncoder:
    """Mean-pools VideoMAE spatial tokens into one feature per tubelet, on OpenVINO devices."""

    # Inference runs off the CPU: callers may run CPU work while a batch is encoded.
    offloaded = True

    def __init__(
        self,
        model: VideoMAEModel,
        devices: Sequence[str],
        *,
        gpu_precision: str = "f32",
        cache_dir: str | None = None,
    ):
        try:
            import openvino as ov
        except ImportError as exc:
            raise ModelLoadError(
                "PICKLEBALL_VIDEOMAE_BACKEND=openvino needs the openvino package"
            ) from exc

        config = model.config
        self.config = config
        self.device = torch.device("cpu")
        self.temporal_tokens = NUM_VIDEO_FRAMES // int(config.tubelet_size)
        self.spatial_tokens = (int(config.image_size) // int(config.patch_size)) ** 2
        size = int(config.image_size)
        self.input_shape = (1, NUM_VIDEO_FRAMES, int(config.num_channels), size, size)

        core = ov.Core()
        if cache_dir:
            core.set_property({"CACHE_DIR": cache_dir})
        available = core.available_devices
        with torch.inference_mode():
            ov_model = ov.convert_model(
                _LastHiddenState(model).eval(), example_input=torch.zeros(self.input_shape)
            )
        ov_model.reshape({0: list(self.input_shape)})

        self.devices: list[_Device] = []
        for wanted in devices:
            name = _match_device(wanted, available)
            if name is None:
                logger.warning(
                    "OpenVINO device not available; skipped",
                    extra={"device": wanted, "available": available},
                )
                continue
            properties: dict[str, Any] = {"PERFORMANCE_HINT": "LATENCY"}
            if name.startswith("GPU"):
                properties["INFERENCE_PRECISION_HINT"] = gpu_precision
            compiled = core.compile_model(ov_model, name, properties)
            self.devices.append(_Device(name=name, request=compiled.create_infer_request()))
            logger.info("VideoMAE compiled with OpenVINO", extra={"device": name})
        if not self.devices:
            raise ModelLoadError(
                f"No OpenVINO device among {list(devices)} is available (found {available})"
            )
        # Several inference slots may encode batches at the same time.
        self._executor = ThreadPoolExecutor(
            max_workers=len(self.devices) * 4, thread_name_prefix="videomae-ov"
        )

    @property
    def hidden_size(self) -> int:
        return int(self.config.hidden_size)

    @property
    def device_names(self) -> list[str]:
        return [device.name for device in self.devices]

    def window_counts(self) -> list[tuple[str, int]]:
        """Windows encoded so far by each device."""
        return [(device.name, device.windows) for device in self.devices]

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _run_device(
        self, device: _Device, windows: np.ndarray, features: np.ndarray, next_index: Any, lock: threading.Lock
    ) -> None:
        while True:
            with lock:
                index = next(next_index)
            if index >= windows.shape[0]:
                return
            with device.lock:
                device.request.infer({0: windows[index : index + 1]})
                tokens = device.request.get_output_tensor(0).data
                features[index] = tokens.reshape(
                    self.temporal_tokens, self.spatial_tokens, -1
                ).mean(axis=1)
                device.windows += 1

    def __call__(
        self,
        pixel_values: torch.Tensor,
        frame_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Batch of windows (B, F, C, H, W) + validity (B, F) -> features (B, T, D), padding (B, T)."""
        if frame_valid.ndim == 1:
            frame_valid = frame_valid.unsqueeze(0)
        batch = pixel_values.shape[0]
        windows = np.ascontiguousarray(pixel_values.detach().cpu().numpy(), dtype=np.float32)
        features = np.empty((batch, self.temporal_tokens, self.hidden_size), dtype=np.float32)
        next_index, index_lock = count(), threading.Lock()
        workers = [
            self._executor.submit(self._run_device, device, windows, features, next_index, index_lock)
            for device in self.devices[:batch]
        ]
        for worker in workers:
            worker.result()

        frames_per_token = NUM_VIDEO_FRAMES // self.temporal_tokens
        temporal_padding = ~frame_valid.reshape(batch, self.temporal_tokens, frames_per_token).any(dim=2)
        return torch.from_numpy(features), temporal_padding
