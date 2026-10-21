import asyncio
import threading
import time

import pytest

from pickleball_track.core.exceptions import ServiceBusyError
from pickleball_track.services.job_gate import JobGate


def _counting_work(calls, result="ok", delay=0.2, error=None):
    def work():
        calls.append(threading.current_thread().name)
        time.sleep(delay)
        if error:
            raise error
        return result

    return work


def test_concurrent_requests_for_one_job_run_it_once():
    async def scenario():
        gate, calls = JobGate(max_pending=4, retry_after=5, cache_seconds=60), []
        try:
            results = await asyncio.gather(*(gate.run("job-1", _counting_work(calls)) for _ in range(3)))
        finally:
            gate.close()
        return results, calls

    results, calls = asyncio.run(scenario())

    assert results == ["ok", "ok", "ok"]
    assert len(calls) == 1
    assert calls[0].startswith("video-job")


def test_completed_result_is_reused_from_cache():
    async def scenario():
        gate, calls = JobGate(max_pending=2, retry_after=5, cache_seconds=60), []
        try:
            first = await gate.run("job-1", _counting_work(calls, delay=0))
            second = await gate.run("job-1", _counting_work(calls, delay=0))
            return first, second, calls, gate.cached
        finally:
            gate.close()

    first, second, calls, cached = asyncio.run(scenario())

    assert (first, second, len(calls), cached) == ("ok", "ok", 1, 1)


def test_more_jobs_than_capacity_are_rejected_with_retry_after():
    async def scenario():
        gate, calls = JobGate(max_pending=1, retry_after=7, cache_seconds=60), []
        try:
            running = asyncio.create_task(gate.run("job-1", _counting_work(calls, delay=0.3)))
            await asyncio.sleep(0.05)
            with pytest.raises(ServiceBusyError) as busy:
                await gate.run("job-2", _counting_work(calls))
            await running
            return busy.value.retry_after, gate.pending, calls
        finally:
            gate.close()

    retry_after, pending, calls = asyncio.run(scenario())

    assert retry_after == 7
    assert pending == 0
    assert len(calls) == 1


def test_failed_job_is_not_cached():
    async def scenario():
        gate, calls = JobGate(max_pending=2, retry_after=5, cache_seconds=60), []
        try:
            with pytest.raises(RuntimeError):
                await gate.run("job-1", _counting_work(calls, delay=0, error=RuntimeError("boom")))
            await asyncio.sleep(0)
            return await gate.run("job-1", _counting_work(calls, delay=0)), calls
        finally:
            gate.close()

    result, calls = asyncio.run(scenario())

    assert result == "ok"
    assert len(calls) == 2


def test_cancelled_request_does_not_cancel_the_job():
    async def scenario():
        gate, calls = JobGate(max_pending=2, retry_after=5, cache_seconds=60), []
        try:
            request = asyncio.create_task(gate.run("job-1", _counting_work(calls, delay=0.2)))
            await asyncio.sleep(0.05)
            request.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request
            await asyncio.sleep(0.3)
            return await gate.run("job-1", _counting_work(calls)), calls
        finally:
            gate.close()

    result, calls = asyncio.run(scenario())

    assert result == "ok"
    assert len(calls) == 1
