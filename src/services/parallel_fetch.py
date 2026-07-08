"""Thread-pool fan-out for I/O-bound per-item fetches (yfinance/Postgres).

Scan pages (Setup Ranker, Market Overview) pull dozens of tickers one at a
time in a plain ``for`` loop — each pull blocks on network/DB I/O, so the
scan's wall-clock time is the *sum* of every round-trip. Since these calls
release the GIL while waiting on sockets, fanning them out across a thread
pool turns that sum into a max over a few concurrent waves.

The catch: ``@st.cache_data`` and ``st.session_state`` both key off Streamlit's
``ScriptRunContext``, which only exists on the thread Streamlit started the
script run on. A plain worker thread has none, so cached fetchers silently
miss and ``st.session_state`` reads raise. :func:`run_parallel` copies the
calling thread's context onto every worker so cached fetchers and
session-state reads behave exactly as they do when called serially.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable, List, Optional, TypeVar

from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

T = TypeVar("T")
R = TypeVar("R")

# Bounds concurrent network/DB round-trips. Kept modest relative to the
# Postgres pool size (src/db/cache.py, src/db/market_cache.py — maxconn=12)
# and to avoid tripping Yahoo Finance rate limits on a cold cache.
DEFAULT_MAX_WORKERS = 8


def run_parallel(
    fn: Callable[[T], R],
    items: Iterable[T],
    *,
    max_workers: int = DEFAULT_MAX_WORKERS,
    on_result: Optional[Callable[[T, R], None]] = None,
    on_error: Optional[Callable[[T, Exception], None]] = None,
) -> List[R]:
    """Run ``fn(item)`` for every item, fanned out across a thread pool.

    Mirrors a serial ``for item in items: try: fn(item) except Exception:
    ...`` loop: a failing item is isolated and reported via ``on_error``
    rather than aborting the batch. ``on_result``/``on_error`` fire on the
    calling thread as each future completes (via ``as_completed``), so it is
    safe to touch Streamlit UI calls (e.g. ``st.progress``) from those
    callbacks.

    Returns the successful results, in completion order (not input order) —
    callers that need a specific order should sort afterwards, as the
    existing scan pages already do.
    """
    items = list(items)
    if not items:
        return []

    ctx = get_script_run_ctx()
    results: List[R] = []

    def _run_with_ctx(item: T) -> R:
        add_script_run_ctx(threading.current_thread(), ctx)
        return fn(item)

    with ThreadPoolExecutor(max_workers=min(max_workers, len(items))) as pool:
        futures = {pool.submit(_run_with_ctx, item): item for item in items}
        for future in as_completed(futures):
            item = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001 - isolate per-item failures
                if on_error is not None:
                    on_error(item, exc)
                continue
            results.append(result)
            if on_result is not None:
                on_result(item, result)

    return results
