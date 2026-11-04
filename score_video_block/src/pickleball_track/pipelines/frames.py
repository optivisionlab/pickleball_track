"""Sequential frame decoding, time-based frame sampling, preprocessing and window batching.

The video is decoded once, front to back, in a background thread. As in training,
only the source frames ``round(i * fps / TARGET_FPS)`` are kept (``frame_counts``
says how many sampled slots each source frame fills: 0 on a 60 fps video for every
other frame, 2 for some frames below 30 fps). Each kept frame is preprocessed once
(overlapping windows share frames) and windows are stacked into batches for the
backbone. Seeking with ``CAP_PROP_POS_FRAMES`` is avoided: on variable-frame-rate
phone videos it lands up to a second away from the requested frame.

Preprocessing runs either on a thread pool (``preprocessed_chunks``) or, faster,
on worker processes that read decoded frames from shared memory
(``ProcessFramePool``); the GIL otherwise caps the thread pool at ~150 frames/s.
Both produce pixel values identical to the training preprocessing.
"""

from __future__ import annotations

import json
import logging
import multiprocessing
import os
import queue
import re
import subprocess
import threading
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from multiprocessing import shared_memory
from multiprocessing.connection import Connection, wait
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ..core.config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, TARGET_FPS
from .frame_workers import END, ERROR, ResizeSpec, resize_frame, to_pixels, worker_main

logger = logging.getLogger(__name__)


class FramePreprocessor(Protocol):
    def __call__(self, frames: Sequence[np.ndarray]) -> torch.Tensor:
        """BGR uint8 frames (H, W, 3) -> pixel values (N, C, H', W') float32."""
        ...


@dataclass(frozen=True)
class FrameSpec:
    """The training frame preprocessing: square resize + ImageNet normalization."""

    size: int = IMAGE_SIZE
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD

    def lut(self) -> np.ndarray:
        """(3, 256) float32: ``(v / 255 - mean) / std`` for every uint8 value, in float32 like training."""
        values = np.arange(256, dtype=np.float32)[None] / np.float32(255.0)
        mean = np.asarray(self.mean, dtype=np.float32)[:, None]
        std = np.asarray(self.std, dtype=np.float32)[:, None]
        return np.ascontiguousarray((values - mean) / std, dtype=np.float32)

    def resize_spec(self, rotate: int = 0) -> ResizeSpec:
        return ResizeSpec(size=(self.size, self.size), lut=self.lut(), rotate=rotate)


def _defer_rotation(capture: cv2.VideoCapture) -> int:
    """Turn off OpenCV's per-frame auto-rotation and return the rotation to apply instead.

    OpenCV rotates every decoded frame on the decoding thread (doubling decode time
    for portrait phone videos) and then cannot read into a caller's buffer;
    workers apply the same rotation in parallel instead.
    """
    meta = getattr(cv2, "CAP_PROP_ORIENTATION_META", None)
    auto = getattr(cv2, "CAP_PROP_ORIENTATION_AUTO", None)
    if meta is None or auto is None or not capture.get(auto):
        return 0
    rotate = int(round(capture.get(meta))) % 360
    if rotate not in (90, 180, 270) or not capture.set(auto, 0):
        return 0
    return rotate


class CpuFrames:
    """Training preprocessing on the CPU (cv2 INTER_AREA resize + LUT normalization)."""

    def __init__(self, spec: FrameSpec):
        self.resize = spec.resize_spec()

    def __call__(self, frames: Sequence[np.ndarray]) -> torch.Tensor:
        resized = np.stack([resize_frame(frame, self.resize) for frame in frames])
        return torch.from_numpy(to_pixels(resized, self.resize.lut))


class TorchFrames:
    """The training preprocessing as batched torch ops on the GPU.

    Area interpolation matches cv2 INTER_AREA exactly for integer downscale factors
    and closely otherwise, so features stay close to the CPU path.
    """

    def __init__(self, spec: FrameSpec, device: torch.device, chunk: int = 8):
        self.spec = spec
        self.device = device
        self.chunk = chunk
        self.mean = torch.tensor(spec.mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(spec.std, device=device).view(1, 3, 1, 1)

    def _process(self, frames: Sequence[np.ndarray]) -> torch.Tensor:
        x = torch.from_numpy(np.stack(frames)).to(self.device, non_blocking=True)
        # BGR -> RGB, (N, H, W, C) uint8 -> (N, C, H, W) float.
        x = x.flip(-1).permute(0, 3, 1, 2).float()
        x = F.interpolate(x, size=(self.spec.size, self.spec.size), mode="area")
        # cv2 returns uint8 images.
        x = x.round_().clamp_(0, 255)
        return ((x / 255.0 - self.mean) / self.std).contiguous()

    def __call__(self, frames: Sequence[np.ndarray]) -> torch.Tensor:
        # Full-resolution float frames are large; convert a few at a time.
        return torch.cat(
            [self._process(frames[i:i + self.chunk]) for i in range(0, len(frames), self.chunk)]
        )


def frame_counts(num_frames: int, fps: float, target_fps: float = TARGET_FPS) -> np.ndarray:
    """Sampled slots filled by each source frame (training's ``decode_sampled``).

    Slot ``i`` of the ``floor(duration * target_fps)`` sampled frames shows source
    frame ``min(round(i * fps / target_fps), num_frames - 1)``.
    """
    num_frames = max(1, int(num_frames))
    num_sampled = max(1, int(np.floor(num_frames / fps * target_fps)))
    targets = np.minimum(np.round(np.arange(num_sampled) * fps / target_fps).astype(np.int64), num_frames - 1)
    return np.bincount(targets, minlength=num_frames)


def sample_frames(frames: Iterator[tuple[np.ndarray, float]], counts: np.ndarray) -> Iterator[np.ndarray]:
    """Keep the sampled source frames, each repeated as often as it fills a slot."""
    for index, (frame, _) in enumerate(frames):
        if index >= len(counts):
            return  # past the last sampled frame
        for _ in range(int(counts[index])):
            yield frame


@dataclass(frozen=True)
class _Failure:
    error: BaseException


def _background(produce: Callable[[Callable[[Any], bool]], None], maxsize: int, name: str) -> Iterator[Any]:
    """Run ``produce(emit)`` in a thread and yield what it emits, in order.

    ``emit`` returns False once the consumer has stopped; ``produce`` must then
    return. Exceptions in ``produce`` are re-raised in the consumer.
    """
    items: queue.Queue = queue.Queue(maxsize=maxsize)
    stop = threading.Event()
    done = object()

    def emit(item: Any) -> bool:
        while not stop.is_set():
            try:
                items.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def run() -> None:
        try:
            produce(emit)
            emit(done)
        except BaseException as exc:  # surfaced in the consumer
            emit(_Failure(exc))

    worker = threading.Thread(target=run, name=name, daemon=True)
    worker.start()
    try:
        while True:
            item = items.get()
            if item is done:
                return
            if isinstance(item, _Failure):
                raise item.error
            yield item
    finally:
        stop.set()
        worker.join()


def _read_frames(
    capture: cv2.VideoCapture, video_path: str | Path, emit: Callable[[Any], bool]
) -> None:
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                return
            if not emit((frame, capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)):
                return
    except Exception as exc:
        raise RuntimeError(f"Cannot decode video {video_path}: {exc}") from exc


_PTS_TIME = re.compile(rb"\bpts_time:\s*(-?[0-9.]+|nan|NOPTS)")


class FFmpegCapture:
    """The part of ``cv2.VideoCapture`` this module uses, decoding with the ffmpeg CLI.

    ffmpeg decodes with its own threads and pipes BGR frames, which on 10-bit HEVC
    (iPhone HDR) costs about a third of OpenCV's CPU time. Colours of HDR video
    differ from OpenCV's conversion, so the decoder is opt-in (``decoder="ffmpeg"``).
    Timestamps come from the ``showinfo`` filter and match ``CAP_PROP_POS_MSEC``.
    """

    def __init__(self, video_path: str | Path, threads: int = 0, autorotate: bool = False):
        self.path = str(video_path)
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,start_time:stream_side_data=rotation:format=start_time",
             "-of", "json", self.path],
            capture_output=True, check=False,
        )
        info = json.loads(probe.stdout or b"{}")
        streams = info.get("streams") or []
        if probe.returncode != 0 or not streams:
            self._process = None
            return
        stream = streams[0]
        self.width, self.height = int(stream["width"]), int(stream["height"])
        rotation = 0
        for side in stream.get("side_data_list") or []:
            rotation = int(side.get("rotation", rotation) or 0)
        if autorotate and rotation % 180:
            self.width, self.height = self.height, self.width
        # OpenCV reports pts - stream start; ffmpeg shifts timestamps by the format start.
        self._offset = float(info.get("format", {}).get("start_time") or 0) - float(stream.get("start_time") or 0)
        self.frame_bytes = self.width * self.height * 3
        command = ["ffmpeg", "-hide_banner", "-nostats", "-nostdin", "-loglevel", "info"]
        if not autorotate:
            command.append("-noautorotate")
        command += [
            "-threads", str(threads), "-i", self.path, "-map", "0:v:0", "-an", "-sn", "-dn",
            "-fps_mode", "passthrough", "-vf", "showinfo=checksum=0", "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1",
        ]
        self._process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1 << 20)
        self._timestamps: queue.Queue[float] = queue.Queue()
        self._log: deque[bytes] = deque(maxlen=20)
        self._reader = threading.Thread(target=self._read_log, name="ffmpeg-log", daemon=True)
        self._reader.start()
        self._position = 0.0

    def _read_log(self) -> None:
        for line in self._process.stderr:
            match = _PTS_TIME.search(line) if b"Parsed_showinfo" in line else None
            if match is None:
                self._log.append(line.rstrip())
                continue
            try:
                value = float(match.group(1))
            except ValueError:
                value = float("nan")
            self._timestamps.put(value)

    def isOpened(self) -> bool:  # noqa: N802 (cv2 naming)
        return self._process is not None

    def read(self, image: np.ndarray | None = None) -> tuple[bool, np.ndarray | None]:
        if self._process is None:
            return False, None
        out = image if image is not None and image.nbytes == self.frame_bytes else np.empty(
            (self.height, self.width, 3), dtype=np.uint8
        )
        view = memoryview(out.reshape(-1))
        filled = 0
        while filled < self.frame_bytes:
            count = self._process.stdout.readinto(view[filled:])
            if not count:
                break
            filled += count
        if filled < self.frame_bytes:
            # End of stream: clean exit, or an error / truncated frame.
            returncode = self._process.wait()
            if filled or returncode:
                tail = b" | ".join(self._log).decode(errors="replace")[-500:]
                raise RuntimeError(f"ffmpeg failed decoding {self.path} (exit {returncode}): {tail}")
            return False, None
        try:
            stamp = self._timestamps.get(timeout=30)
        except queue.Empty:
            stamp = float("nan")
        self._position = stamp + self._offset if stamp == stamp else self._position
        return True, out

    def grab(self) -> bool:
        """Decode and drop one frame (a frame that is not sampled)."""
        if self._process is not None and getattr(self, "_scratch", None) is None:
            self._scratch = np.empty((self.height, self.width, 3), dtype=np.uint8)
        ok, _ = self.read(getattr(self, "_scratch", None))
        return ok

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_POS_MSEC:
            return self._position * 1000.0
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.width)
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.height)
        return 0.0

    def set(self, prop: int, value: float) -> bool:
        return False

    def release(self) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        if process.poll() is None:
            process.kill()
        process.wait()
        for stream in (process.stdout, process.stderr):
            if stream:
                stream.close()
        self._reader.join(timeout=5)


def open_capture(video_path: str | Path, decoder: str = "opencv", ffmpeg_threads: int = 0, autorotate: bool = True):
    """A capture for ``decoder``: ``cv2.VideoCapture`` or ``FFmpegCapture``."""
    if decoder == "ffmpeg":
        return FFmpegCapture(video_path, threads=ffmpeg_threads, autorotate=autorotate)
    if decoder != "opencv":
        raise ValueError(f"Unknown decoder: {decoder}")
    return cv2.VideoCapture(str(video_path))


def _probe_rotation(video_path: str | Path) -> int:
    """The rotation OpenCV would apply, read without decoding (for the ffmpeg decoder)."""
    capture = cv2.VideoCapture(str(video_path))
    try:
        return _defer_rotation(capture) if capture.isOpened() else 0
    finally:
        capture.release()


def iter_frames(
    video_path: str | Path, prefetch: int = 64, *, decoder: str = "opencv", ffmpeg_threads: int = 0
) -> Iterator[tuple[np.ndarray, float]]:
    """Yield ``(BGR frame, timestamp seconds)`` decoded sequentially in a background thread."""
    capture = open_capture(video_path, decoder, ffmpeg_threads, autorotate=True)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    try:
        yield from _background(lambda emit: _read_frames(capture, video_path, emit), prefetch, "video-decode")
    finally:
        capture.release()


def preprocessed_chunks(
    frames: Iterator[np.ndarray],
    preprocess: FramePreprocessor,
    chunk: int = 4,
    workers: int = 8,
    prefetch_frames: int = 512,
) -> Iterator[torch.Tensor]:
    """Preprocess consecutive chunks of (sampled) frames on a thread pool, yielding them in order.

    A feeder thread keeps submitting chunks while the consumer runs the models, so
    decoding and preprocessing overlap with GPU work.
    """
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="frame-preprocess")

    def produce(emit: Callable[[Any], bool]) -> None:
        raw: list[np.ndarray] = []
        for frame in frames:
            raw.append(frame)
            if len(raw) == chunk:
                if not emit(pool.submit(preprocess, raw)):
                    return
                raw = []
        if raw:
            emit(pool.submit(preprocess, raw))

    futures = _background(produce, max(1, prefetch_frames // chunk), "frame-feeder")
    try:
        for future in futures:
            yield future.result()
    finally:
        futures.close()
        pool.shutdown(wait=True, cancel_futures=True)


def default_preprocess_workers(requested: int) -> int:
    """Worker processes to use: ``requested`` capped to leave one CPU for decoding."""
    return max(0, min(requested, (os.cpu_count() or 1) - 1))


@dataclass
class _Worker:
    process: multiprocessing.process.BaseProcess
    tasks: Connection  # parent -> worker
    results: Connection  # worker -> parent


class ProcessFramePool:
    """Persistent worker processes that resize frames decoded into shared memory.

    The parent decodes sequentially directly into an input ring of shared-memory
    slots, skipping frames that are not sampled; workers resize and normalize each
    slot into an output ring of pixel values (identical to ``CpuFrames``), which
    the parent copies out in order, repeating frames that fill several slots.
    Only slot indices cross process boundaries, never frames. Each worker has its
    own pipes, so a crashed worker fails the current video with an error instead
    of deadlocking the others, and is restarted for the next one. Serve one video
    at a time (the inference service holds a lock).
    """

    def __init__(self, spec: FrameSpec, workers: int, slots_per_worker: int = 4):
        if workers < 1:
            raise ValueError("ProcessFramePool needs at least one worker")
        self.spec = spec
        self.workers = workers
        # Input slots hold full-resolution frames (6 MB at 1080p): keep the ring small.
        self.slots = workers * slots_per_worker
        self._context = multiprocessing.get_context("spawn")  # CUDA-safe, unlike fork
        self._workers: list[_Worker] = []
        self._job = 0
        self._ensure_workers()

    @property
    def _processes(self) -> list[multiprocessing.process.BaseProcess]:
        return [worker.process for worker in self._workers]

    def _start_worker(self, number: int) -> _Worker:
        task_recv, task_send = self._context.Pipe(duplex=False)
        result_recv, result_send = self._context.Pipe(duplex=False)
        process = self._context.Process(
            target=worker_main, args=(task_recv, result_send), name=f"frame-worker-{number}", daemon=True
        )
        process.start()
        # The child holds its own copies of these ends.
        task_recv.close()
        result_send.close()
        return _Worker(process, task_send, result_recv)

    def _stop_worker(self, worker: _Worker) -> None:
        if worker.process.is_alive():
            worker.process.terminate()
        worker.process.join(timeout=5)
        worker.tasks.close()
        worker.results.close()

    def _ensure_workers(self) -> None:
        dead = [worker for worker in self._workers if not worker.process.is_alive()]
        if dead:
            logger.warning("Restarting frame preprocessing workers", extra={"dead": len(dead)})
        for worker in dead:
            self._stop_worker(worker)
        self._workers = [worker for worker in self._workers if worker not in dead]
        while len(self._workers) < self.workers:
            self._workers.append(self._start_worker(len(self._workers)))

    def ring_bytes(self, height: int, width: int) -> int:
        """Shared memory for (height, width) frames; the same for either orientation."""
        channels, out_h, out_w = self.spec.resize_spec().output_shape()
        return self.slots * (height * width * 3 + channels * out_h * out_w * 4)

    def fits(self, height: int, width: int) -> bool:
        """Whether /dev/shm can hold the rings for this frame size.

        Shared memory is allocated lazily on Linux: an undersized /dev/shm (64 MB
        by default in Docker) would crash the process with SIGBUS on first write
        instead of raising, so callers check first and fall back to threads.
        """
        try:
            stats = os.statvfs("/dev/shm")
        except OSError:
            return True  # not Linux-style shared memory; creation raises if it fails
        return self.ring_bytes(height, width) <= stats.f_bavail * stats.f_frsize

    def close(self) -> None:
        for worker in self._workers:
            try:
                worker.tasks.send(None)
            except OSError:
                pass
        for worker in self._workers:
            worker.process.join(timeout=5)
            self._stop_worker(worker)
        self._workers = []

    def chunks(
        self,
        video_path: str | Path,
        counts: np.ndarray,
        chunk: int = 4,
        prefetch_frames: int = 512,
        *,
        decoder: str = "opencv",
        ffmpeg_threads: int = 0,
    ) -> Iterator[torch.Tensor]:
        """Yield the preprocessed sampled frames (N, C, H, W) of ``video_path`` in order.

        ``counts[i]`` is how many sampled slots source frame ``i`` fills (``frame_counts``).
        """
        # Workers rotate frames in parallel; the ffmpeg decoder emits them unrotated too.
        capture = open_capture(video_path, decoder, ffmpeg_threads, autorotate=False)
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        try:
            rotate = _defer_rotation(capture) if decoder == "opencv" else _probe_rotation(video_path)
            ok, first = capture.read()
            if not ok or len(counts) == 0:
                return
            self._ensure_workers()
            self._job += 1
            yield from _background(
                lambda emit: self._run(emit, capture, first, rotate, video_path, counts, chunk),
                max(1, prefetch_frames // chunk),
                "frame-collector",
            )
        finally:
            capture.release()

    def _run(
        self,
        emit: Callable[[Any], bool],
        capture: cv2.VideoCapture,
        first: np.ndarray,
        rotate: int,
        video_path: str | Path,
        counts: np.ndarray,
        chunk: int,
    ) -> None:
        job = self._job
        workers = list(self._workers)
        height, width = first.shape[:2]
        spec = self.spec.resize_spec(rotate)
        in_shape = (self.slots, height, width, 3)
        out_shape = (self.slots, *spec.output_shape())
        in_shm = shared_memory.SharedMemory(create=True, size=int(np.prod(in_shape)))
        out_shm = shared_memory.SharedMemory(create=True, size=int(np.prod(out_shape)) * 4)
        rings = {
            "in": np.ndarray(in_shape, dtype=np.uint8, buffer=in_shm.buf),
            "out": np.ndarray(out_shape, dtype=np.float32, buffer=out_shm.buf),
        }
        free: queue.Queue = queue.Queue()
        for slot in range(self.slots):
            free.put(slot)
        # Sampled slots each kept frame fills, by kept-frame index (appended before submit).
        repeats: list[int] = []
        # Decoder -> collector: (END, frame_count) or (ERROR, message).
        signal_recv, signal_send = multiprocessing.Pipe(duplex=False)
        stop = threading.Event()

        def submit(slot: int, index: int) -> None:
            task = (job, index, slot, in_shm.name, in_shape, out_shm.name, out_shape, spec)
            workers[index % len(workers)].tasks.send(task)

        def decode() -> None:
            # `index` counts submitted (kept) frames; `source` counts decoded frames.
            index = 0
            try:
                ring = rings["in"]
                slot = free.get()
                ring[slot] = first
                repeats.append(int(counts[0]))
                submit(slot, index)
                index, source = 1, 1
                while not stop.is_set() and source < len(counts):
                    if counts[source] == 0:
                        if not capture.grab():
                            break
                        source += 1
                        continue
                    try:
                        slot = free.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    ok, frame = capture.read(ring[slot])
                    if not ok:
                        break
                    if frame.ctypes.data != ring[slot].ctypes.data:
                        # OpenCV allocated a new array: the frame size changed mid-video.
                        if frame.shape != ring[slot].shape:
                            raise RuntimeError(f"Frame size changed at frame {source} of {video_path}")
                        ring[slot] = frame
                    repeats.append(int(counts[source]))
                    submit(slot, index)
                    index += 1
                    source += 1
                signal_send.send((END, index))
            except Exception as exc:
                signal_send.send((ERROR, f"Cannot decode video {video_path}: {exc}"))

        decoder = threading.Thread(target=decode, name="video-decode", daemon=True)
        decoder.start()
        connections = {worker.results: worker for worker in workers}
        try:
            pending: dict[int, int] = {}
            group: list[int] = []
            next_index = 0
            total: int | None = None
            while total is None or next_index < total:
                for ready in wait([signal_recv, *connections]):
                    if ready is signal_recv:
                        kind, value = signal_recv.recv()
                        if kind == ERROR:
                            raise RuntimeError(value)
                        total = value
                        continue
                    try:
                        kind, result_job, index, slot, message = ready.recv()
                    except EOFError:
                        raise RuntimeError("A frame preprocessing worker exited unexpectedly") from None
                    if result_job != job:
                        continue  # left over from an earlier, interrupted video
                    if kind == ERROR:
                        raise RuntimeError(f"Preprocessing failed at frame {index}: {message}")
                    pending[index] = slot
                while next_index in pending:
                    group.extend([pending.pop(next_index)] * repeats[next_index])
                    next_index += 1
                    if len(group) >= chunk or next_index == total:
                        pixels = torch.from_numpy(rings["out"][group])  # fancy indexing copies
                        for used in dict.fromkeys(group):
                            free.put(used)
                        group = []
                        if not emit(pixels):
                            return
                if total is not None and next_index == total and group:
                    emit(torch.from_numpy(rings["out"][group]))
                    for used in dict.fromkeys(group):
                        free.put(used)
                    group = []
        finally:
            stop.set()
            decoder.join()
            signal_recv.close()
            signal_send.close()
            rings.clear()  # views must go before their segments can be closed
            for shm in (in_shm, out_shm):
                shm.close()
                shm.unlink()


@dataclass(frozen=True)
class WindowBatch:
    starts: list[int]
    pixel_values: torch.Tensor  # (B, F, C, H, W)
    frame_valid: torch.Tensor  # (B, F) bool; False for frames past the end of the video


def iter_window_batches(
    chunks: Iterator[torch.Tensor],
    starts: Sequence[int],
    num_frames: int,
    batch_size: int,
) -> Iterator[WindowBatch]:
    """Assemble windows starting at sorted ``starts`` from in-order preprocessed frame chunks.

    Frames past the end of the video repeat the last decoded frame and are marked
    invalid (a window starting past the end is entirely invalid, as in training).
    Frames no later window needs are released as batches are emitted.
    """
    buffer: dict[int, torch.Tensor] = {}
    decoded = 0
    last: torch.Tensor | None = None

    def fill(until: int) -> None:
        nonlocal decoded, last
        while decoded < until:
            pixels = next(chunks, None)
            if pixels is None:
                return
            for offset, frame_pixels in enumerate(pixels):
                buffer[decoded + offset] = frame_pixels
            decoded += len(pixels)
            last = pixels[-1]

    for batch_begin in range(0, len(starts), batch_size):
        batch_starts = [int(s) for s in starts[batch_begin:batch_begin + batch_size]]
        windows, valid = [], []
        for start in batch_starts:
            fill(start + num_frames)
            if last is None:
                raise RuntimeError(f"Cannot read frames at {start}")
            available = max(0, min(num_frames, decoded - start))
            window = [buffer[start + i] for i in range(available)]
            window += [window[-1] if window else last] * (num_frames - available)
            windows.append(torch.stack(window))
            valid.append([True] * available + [False] * (num_frames - available))

        next_start = int(starts[batch_begin + batch_size]) if batch_begin + batch_size < len(starts) else decoded
        for index in [i for i in buffer if i < next_start]:
            del buffer[index]
        yield WindowBatch(
            starts=batch_starts,
            pixel_values=torch.stack(windows),
            frame_valid=torch.tensor(valid, dtype=torch.bool),
        )
