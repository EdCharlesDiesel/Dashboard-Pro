"""Unit tests for src/services/parallel_fetch.py.

No live DB or network: ``fn`` is a plain callable exercised against small
in-memory item lists. We assert the contract callers rely on — every item is
attempted, a failing item is isolated (not fatal to the batch), callbacks fire
on the calling thread, and results/order semantics match a serial loop.
"""
from __future__ import annotations

import threading

import pytest

from src.services.parallel_fetch import run_parallel


class TestRunParallel:
    def test_empty_items_returns_empty(self):
        assert run_parallel(lambda x: x, []) == []

    def test_runs_every_item(self):
        seen = []
        run_parallel(lambda x: seen.append(x), range(10), max_workers=4)
        assert sorted(seen) == list(range(10))

    def test_returns_successful_results(self):
        out = run_parallel(lambda x: x * 2, range(5), max_workers=3)
        assert sorted(out) == [0, 2, 4, 6, 8]

    def test_failing_item_is_isolated(self):
        def fn(x):
            if x == 2:
                raise ValueError("boom")
            return x

        out = run_parallel(fn, range(5), max_workers=3)
        # the other 4 items still succeed; the failure doesn't abort the batch
        assert sorted(out) == [0, 1, 3, 4]

    def test_on_error_receives_the_item_and_exception(self):
        errors = []

        def fn(x):
            raise RuntimeError(f"fail-{x}")

        run_parallel(fn, [1, 2, 3], on_error=lambda item, exc: errors.append((item, str(exc))))
        assert sorted(errors) == [(1, "fail-1"), (2, "fail-2"), (3, "fail-3")]

    def test_on_result_receives_the_item_and_result(self):
        seen = {}
        run_parallel(lambda x: x * 10, [1, 2, 3], on_result=lambda item, r: seen.__setitem__(item, r))
        assert seen == {1: 10, 2: 20, 3: 30}

    def test_callbacks_fire_on_the_calling_thread(self):
        # Callers (e.g. st.progress updates) must be safe to call from these
        # callbacks, which requires they run on the thread that called
        # run_parallel, not inside a worker.
        caller_thread = threading.current_thread()
        callback_threads = []
        run_parallel(
            lambda x: x,
            range(6),
            max_workers=4,
            on_result=lambda item, r: callback_threads.append(threading.current_thread()),
        )
        assert all(t is caller_thread for t in callback_threads)

    def test_max_workers_does_not_exceed_item_count(self):
        # A pool sized larger than the work shouldn't error or hang.
        out = run_parallel(lambda x: x, [1], max_workers=8)
        assert out == [1]

    def test_one_slow_item_does_not_block_the_others(self):
        import time

        def fn(x):
            if x == 0:
                time.sleep(0.2)
            return x

        start = time.monotonic()
        out = run_parallel(fn, range(4), max_workers=4)
        elapsed = time.monotonic() - start
        assert sorted(out) == [0, 1, 2, 3]
        # Parallel: bounded by the one slow item, not the sum of all four.
        assert elapsed < 0.35
