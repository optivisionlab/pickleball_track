"""Hand inference slots to waiting videos: short videos first, none starved.

- ``reserved_short`` slots serve only short videos (``duration <= short_seconds``),
  so a short clip never waits behind long videos; the other slots serve any video.
- Among the videos that may take a free slot, the shortest goes first; a video that
  has waited ``aging_seconds`` goes before any newer one, so a stream of short clips
  cannot starve a long video.
"""

from __future__ import annotations

import itertools
import threading
import time
from dataclasses import dataclass, field
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(order=True)
class _Waiter:
    seq: int
    duration: float = field(compare=False)
    short: bool = field(compare=False)
    since: float = field(compare=False)


class SlotScheduler(Generic[T]):
    def __init__(
        self,
        slots: list[T],
        *,
        reserved_short: int = 0,
        short_seconds: float = 0.0,
        aging_seconds: float = 60.0,
        clock=time.monotonic,
    ):
        if not slots:
            raise ValueError("SlotScheduler needs at least one slot")
        if not 0 <= reserved_short < len(slots):
            raise ValueError("reserved_short must leave at least one slot for long videos")
        self.total = len(slots)
        self.reserved_short = reserved_short
        self.short_seconds = short_seconds
        self.aging_seconds = aging_seconds
        self._clock = clock
        self._free = list(slots)
        self._long_busy = 0
        self._waiting: list[_Waiter] = []
        self._seq = itertools.count()
        self._cond = threading.Condition()

    # ---- state -------------------------------------------------------------
    @property
    def busy(self) -> int:
        with self._cond:
            return self.total - len(self._free)

    def waiting(self) -> tuple[int, int]:
        """(short, long) videos waiting for a slot."""
        with self._cond:
            short = sum(1 for w in self._waiting if w.short)
            return short, len(self._waiting) - short

    def is_short(self, duration: float) -> bool:
        return duration <= self.short_seconds

    # ---- scheduling ----------------------------------------------------------
    def _eligible(self, waiter: _Waiter) -> bool:
        if not self._free:
            return False
        return waiter.short or self._long_busy < self.total - self.reserved_short

    def _rank(self, waiter: _Waiter, now: float) -> tuple:
        if now - waiter.since >= self.aging_seconds:
            return (0, waiter.seq)  # aged: first come, first served
        return (1, waiter.duration, waiter.seq)

    def _is_next(self, waiter: _Waiter) -> bool:
        if not self._eligible(waiter):
            return False
        now = self._clock()
        mine = self._rank(waiter, now)
        return all(
            other is waiter or not self._eligible(other) or self._rank(other, now) > mine
            for other in self._waiting
        )

    def acquire(self, duration: float) -> tuple[T, bool]:
        """Block until a slot is granted; returns (slot, short) to pass to ``release``."""
        short = self.is_short(duration)
        with self._cond:
            waiter = _Waiter(next(self._seq), float(duration), short, self._clock())
            self._waiting.append(waiter)
            try:
                # Re-check periodically: ageing can change the order without a release.
                while not self._is_next(waiter):
                    self._cond.wait(timeout=1.0)
            finally:
                self._waiting.remove(waiter)
            slot = self._free.pop()
            if not short:
                self._long_busy += 1
            # Another waiter may also be able to run now (e.g. several free slots).
            self._cond.notify_all()
            return slot, short

    def release(self, slot: T, short: bool) -> None:
        with self._cond:
            self._free.append(slot)
            if not short:
                self._long_busy -= 1
            self._cond.notify_all()
