"""Exact VideoMAE frame preprocessing and the worker process that runs it.

Kept free of torch/transformers imports: worker processes are spawned and only
need NumPy and Pillow.
"""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import shared_memory
from multiprocessing.connection import Connection

import numpy as np
import PIL.Image

# Result kinds. END is posted by the parent itself once decoding has finished.
DONE, ERROR, END = "done", "error", "end"


@dataclass(frozen=True, eq=False)
class ResizeSpec:
    """The processor's resize + center crop geometry for one input frame size, and its pixel LUT."""

    resize: tuple[int, int] | None  # (height, width); None skips resizing
    resample: int  # PIL resample code
    crop: tuple[int, int, int, int] | None  # (top, left, height, width); None skips cropping
    # (3, 256) float32: the processor's rescale + normalize of every uint8 value per channel.
    lut: np.ndarray
    # Clockwise display rotation (0/90/180/270) still to apply, as OpenCV's
    # CAP_PROP_ORIENTATION_AUTO would; resize/crop refer to the rotated frame.
    rotate: int = 0

    def output_shape(self, height: int, width: int) -> tuple[int, int, int]:
        """Pixel values (C, H, W) produced for a (height, width) frame."""
        if self.crop is not None:
            return 3, self.crop[2], self.crop[3]
        if self.resize is not None:
            return 3, self.resize[0], self.resize[1]
        return (3, width, height) if self.rotate in (90, 270) else (3, height, width)


# Clockwise rotation -> PIL transpose (PIL's ROTATE_* turn counter-clockwise).
_CLOCKWISE = {
    90: PIL.Image.Transpose.ROTATE_270,
    180: PIL.Image.Transpose.ROTATE_180,
    270: PIL.Image.Transpose.ROTATE_90,
}


def resize_and_crop(frame: np.ndarray, spec: ResizeSpec) -> np.ndarray:
    """BGR uint8 frame -> RGB uint8 (H, W, 3), pixel-identical to VideoMAEImageProcessor.

    The processor converts the uint8 RGB image to PIL, resizes it with
    ``reducing_gap=None`` and center-crops the result.
    """
    height, width = frame.shape[:2]
    # Same pixels as cvtColor(BGR2RGB) + Image.fromarray, in one cheaper copy.
    image = PIL.Image.frombuffer("RGB", (width, height), np.ascontiguousarray(frame), "raw", "BGR", 0, 1)
    if spec.rotate:
        # A pure pixel permutation, identical to OpenCV's cv::rotate.
        image = image.transpose(_CLOCKWISE[spec.rotate])
    if spec.resize is not None:
        image = image.resize((spec.resize[1], spec.resize[0]), resample=spec.resample, reducing_gap=None)
    rgb = np.asarray(image)
    if spec.crop is not None:
        top, left, crop_h, crop_w = spec.crop
        rgb = rgb[top:top + crop_h, left:left + crop_w]
    return rgb


def to_pixels(rgb: np.ndarray, lut: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """RGB uint8 (..., H, W, 3) -> float32 (..., 3, H, W) through the per-channel LUT.

    Equal to the processor's float64 rescale, float32 cast and float32 normalize,
    which are element-wise and so a function of each uint8 value alone.
    """
    if out is None:
        out = np.empty((*rgb.shape[:-3], 3, *rgb.shape[-3:-1]), dtype=np.float32)
    for channel in range(3):
        np.take(lut[channel], rgb[..., channel], out=out[..., channel, :, :])
    return out


def _attach(name: str) -> shared_memory.SharedMemory:
    # Spawned workers share the parent's resource tracker, which already tracks
    # the segment; the parent closes and unlinks it.
    return shared_memory.SharedMemory(name=name)


class _Rings:
    """Shared-memory rings of the video currently being processed."""

    def __init__(self) -> None:
        self._segments: list[shared_memory.SharedMemory] = []
        self.arrays: dict[str, np.ndarray] = {}

    def get(self, in_name: str, in_shape: tuple[int, ...], out_name: str, out_shape: tuple[int, ...]):
        if in_name not in self.arrays or out_name not in self.arrays:
            self.close()  # one video at a time: release the previous video's rings
            for name, shape, dtype in ((in_name, in_shape, np.uint8), (out_name, out_shape, np.float32)):
                shm = _attach(name)
                self._segments.append(shm)
                self.arrays[name] = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        return self.arrays[in_name], self.arrays[out_name]

    def close(self) -> None:
        self.arrays.clear()  # views must go before their segments can be closed
        for shm in self._segments:
            shm.close()
        self._segments.clear()


def worker_main(tasks: Connection, results: Connection) -> None:
    """Resize frames from an input shared-memory ring into an output ring.

    Each worker has its own pipes (no locks shared with other workers, so one
    worker dying cannot block the rest).
    Task: ``(job, frame_index, slot, in_name, in_shape, out_name, out_shape, spec)``:
    the input ring is uint8 ``(slots, H, W, 3)`` BGR, the output ring float32
    ``(slots, 3, H', W')`` pixel values; ``None`` stops the worker.
    Result: ``(DONE, job, frame_index, slot, None)`` or ``(ERROR, job, frame_index, slot, message)``.
    """
    rings = _Rings()
    try:
        while True:
            try:
                task = tasks.recv()
            except EOFError:  # parent went away
                return
            if task is None:
                return
            job, index, slot, in_name, in_shape, out_name, out_shape, spec = task
            try:
                frames, outputs = rings.get(in_name, in_shape, out_name, out_shape)
                to_pixels(resize_and_crop(frames[slot], spec), spec.lut, out=outputs[slot])
                results.send((DONE, job, index, slot, None))
            except Exception as exc:  # reported to the parent, which raises
                results.send((ERROR, job, index, slot, f"{type(exc).__name__}: {exc}"))
    finally:
        rings.close()
