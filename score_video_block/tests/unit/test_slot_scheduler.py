import threading
import time

import pytest

from pickleball_track.services.slot_scheduler import SlotScheduler


def _start(scheduler, duration, order, hold=0.05):
    def run():
        slot, short = scheduler.acquire(duration)
        order.append(duration)
        time.sleep(hold)
        scheduler.release(slot, short)

    thread = threading.Thread(target=run)
    thread.start()
    return thread


def _wait_for_waiters(scheduler, count):
    deadline = time.monotonic() + 2
    while sum(scheduler.waiting()) < count:
        assert time.monotonic() < deadline, "waiters did not queue up"
        time.sleep(0.005)


def test_waiting_videos_get_the_slot_shortest_first():
    scheduler = SlotScheduler(["s"], aging_seconds=60)
    slot, short = scheduler.acquire(1)  # occupy the only slot
    order: list[float] = []
    threads = []
    for duration in (50, 3, 20):
        threads.append(_start(scheduler, duration, order))
        _wait_for_waiters(scheduler, len(threads))
    scheduler.release(slot, short)
    for thread in threads:
        thread.join(5)

    assert order == [3, 20, 50]


def test_long_videos_cannot_take_the_reserved_short_slot():
    scheduler = SlotScheduler(["a", "b"], reserved_short=1, short_seconds=10)
    long_slot, long_short = scheduler.acquire(60)
    assert long_short is False

    blocked: list[float] = []
    second_long = _start(scheduler, 60, blocked)
    _wait_for_waiters(scheduler, 1)
    # The free slot is reserved: the second long video waits, a short one runs at once.
    started = time.monotonic()
    short_slot, is_short = scheduler.acquire(3)
    assert is_short and time.monotonic() - started < 0.5
    assert blocked == [] and scheduler.waiting() == (0, 1)

    scheduler.release(short_slot, is_short)
    time.sleep(0.1)
    assert blocked == []  # still only the reserved slot is free
    scheduler.release(long_slot, long_short)
    second_long.join(5)
    assert blocked == [60]
    assert scheduler.busy == 0


def test_aged_waiter_goes_before_shorter_newer_ones():
    now = [0.0]
    scheduler = SlotScheduler(["s"], aging_seconds=30, clock=lambda: now[0])
    slot, short = scheduler.acquire(1)
    order: list[float] = []
    threads = [_start(scheduler, 100, order)]
    _wait_for_waiters(scheduler, 1)
    now[0] = 31.0  # the long video has waited past the ageing limit
    threads.append(_start(scheduler, 2, order))
    _wait_for_waiters(scheduler, 2)
    scheduler.release(slot, short)
    for thread in threads:
        thread.join(5)

    assert order == [100, 2]


def test_reserved_slots_must_leave_one_for_long_videos():
    with pytest.raises(ValueError):
        SlotScheduler(["a"], reserved_short=1)
