"""Admission control and job_id de-duplication for the processing endpoint."""

from __future__ import annotations

import asyncio
import logging
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
        self._done: OrderedDict[str, tuple[float, Any]] = OrderedDict()

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
        if self.cache_seconds and not future.cancelled() and future.exception() is None:
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
                    f"Server busy: {self.max_pending} videos already in progress", self.retry_after
                )
            future = self._executor.submit(work)
            self._running[key] = future
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
