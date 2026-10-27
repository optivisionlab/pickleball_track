"""Admission control and job_id de-duplication for the processing endpoint."""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from ..core.exceptions import ServiceBusyError

logger = logging.getLogger(__name__)


class JobGate:
    """Runs blocking jobs off the event loop with bounded concurrency.

    - At most ``max_pending`` jobs are accepted at once (downloading, waiting for an
      inference slot or running); more raise ``ServiceBusyError`` (HTTP 429).
    - A request for a job that is already running waits for that run instead of
      starting a second one, and successful results are kept for ``cache_seconds``,
      so client retries after a timeout or a dropped connection cost no inference.
    - Jobs are not tied to the request: a disconnected client does not cancel them.
    """

    def __init__(self, max_pending: int, retry_after: int, cache_seconds: int):
        self.max_pending = max_pending
        self.retry_after = retry_after
        self.cache_seconds = cache_seconds
        self._executor = ThreadPoolExecutor(max_workers=max_pending, thread_name_prefix="video-job")
        self._running: dict[str, Future] = {}
        self._started: dict[str, float] = {}
        self._done: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        # Moving average of completed job durations (seconds), for Retry-After.
        self.avg_job_seconds: float | None = None

    @property
    def busy_retry_after(self) -> int:
        """Seconds a rejected client should wait: about half a job, at least ``retry_after``."""
        if self.avg_job_seconds is None:
            return self.retry_after
        return max(self.retry_after, min(120, math.ceil(self.avg_job_seconds / 2)))

    @property
    def pending(self) -> int:
        return len(self._running)

    @property
    def cached(self) -> int:
        return len(self._done)

    def _evict(self, now: float) -> None:
        while self._done:
            key, (expires, _) = next(iter(self._done.items()))
            if expires > now:
                break
            self._done.pop(key)

    def _finished(self, key: str, future: Future) -> None:
        self._running.pop(key, None)
        started = self._started.pop(key, None)
        ok = not future.cancelled() and future.exception() is None
        if ok and started is not None:
            seconds = time.monotonic() - started
            avg = self.avg_job_seconds
            self.avg_job_seconds = seconds if avg is None else 0.8 * avg + 0.2 * seconds
        if self.cache_seconds and ok:
            self._done[key] = (time.monotonic() + self.cache_seconds, future.result())

    async def run(self, key: str, work: Callable[[], Any]) -> Any:
        self._evict(time.monotonic())
        if key in self._done:
            logger.info("Returning cached result", extra={"job_key": key})
            return self._done[key][1]
        future = self._running.get(key)
        if future is not None:
            logger.info("Joining running job", extra={"job_key": key})
        else:
            if len(self._running) >= self.max_pending:
                raise ServiceBusyError(
                    f"Server busy: {self.max_pending} videos already in progress", self.busy_retry_after
                )
            future = self._executor.submit(work)
            self._running[key] = future
            self._started[key] = time.monotonic()
            loop = asyncio.get_running_loop()

            def done(f: Future) -> None:
                try:
                    loop.call_soon_threadsafe(self._finished, key, f)
                except RuntimeError:  # event loop already closed (shutdown)
                    pass

            future.add_done_callback(done)
        return await asyncio.shield(asyncio.wrap_future(future))

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
