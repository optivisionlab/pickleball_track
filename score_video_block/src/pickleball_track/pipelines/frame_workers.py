"""Training-identical frame preprocessing and the worker process that runs it.

Kept free of torch/transformers imports: worker processes are spawned and only
need NumPy and OpenCV.
"""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import shared_memory
from multiprocessing.connection import Connection

import cv2
import numpy as np

# Result kinds. END is posted by the parent itself once decoding has finished.
DONE, ERROR, END = "done", "error", "end"

# Clockwise display rotation -> cv2.rotate code (what CAP_PROP_ORIENTATION_AUTO applies).
_CLOCKWISE = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


@dataclass(frozen=True, eq=False)
class ResizeSpec:
    """Output size of every frame, its pixel LUT and the rotation still to apply."""

    size: tuple[int, int]  # (height, width)
    # (3, 256) float32: rescale + normalize of every uint8 value per RGB channel.
    lut: np.ndarray
    # Clockwise display rotation (0/90/180/270) still to apply, as OpenCV's
    # CAP_PROP_ORIENTATION_AUTO would, before resizing.
    rotate: int = 0

    def output_shape(self) -> tuple[int, int, int]:
        """Pixel values (C, H, W) produced for any input frame."""
        return 3, self.size[0], self.size[1]


def resize_frame(frame: np.ndarray, spec: ResizeSpec) -> np.ndarray:
    """BGR uint8 frame -> RGB uint8 (H, W, 3), as in training.

    Training reads frames with OpenCV (auto-rotated), resizes them with
    ``cv2.resize(..., INTER_AREA)`` straight to the output size (no aspect-preserving
    resize, no crop) and converts BGR to RGB. Resizing works per channel, so
    converting first or last gives the same pixels.
    """
    if spec.rotate:
        frame = cv2.rotate(frame, _CLOCKWISE[spec.rotate])
    resized = cv2.resize(frame, (spec.size[1], spec.size[0]), interpolation=cv2.INTER_AREA)
    return resized[..., ::-1]


def to_pixels(rgb: np.ndarray, lut: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """RGB uint8 (..., H, W, 3) -> float32 (..., 3, H, W) through the per-channel LUT.

    The training normalization (float32 ``x / 255``, then ``(x - mean) / std``) is
    element-wise and so a function of each uint8 value alone.
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
    cv2.setNumThreads(1)  # one frame per worker at a time; parallelism comes from the workers
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
                to_pixels(resize_frame(frames[slot], spec), spec.lut, out=outputs[slot])
                results.send((DONE, job, index, slot, None))
            except Exception as exc:  # reported to the parent, which raises
                results.send((ERROR, job, index, slot, f"{type(exc).__name__}: {exc}"))
    finally:
        rings.close()
