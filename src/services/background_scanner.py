"""Unattended ingest + score daemon — the app's data streams in day & night.

Streamlit fragments (``run_every=300``) only re-run while a browser session is
actually viewing a page. This module closes that gap: a single daemon thread
per **server process** runs a full cycle every few minutes — fetch fresh bars,
persist them, recompute the whole universe's house view + Setup Ranker read,
store that precomputed board, and email/journal new Grade-A setups — whether or
not anyone is looking at the app.

The point is to move the slow work **off the request path**. When a page
loads it reads the worker's already-computed board (one DB round-trip via
:mod:`src.services.precomputed`) instead of fetching three timeframes and
recomputing per pair while the user waits. From SAST the Neon DB is ~200ms
away; recomputing 24 pairs live is seconds of that latency stacked up — reading
one precomputed row is a single hop.

One cycle (``scan_once``), per instrument, drained from a rate-limited queue:

  1. **Ingest** — pull weekly/daily/4H bars through the canonical spine
     (:mod:`src.services.market_data`), which persists them to ``market_bars``
     via the Postgres read-through. This is the "streaming" — a tight 24/5
     poll (yfinance snapshots, not a live tick feed), rate-limited so Yahoo
     never throttles us.
  2. **Score** — the *pure* ``house_view`` and the page's own
     ``_SetupRankerDataFeed.score`` (both directions, best kept). Same code the
     pages use — the worker is a scheduler, never a second implementation.
  3. **Store** — the assembled board → ``app_state`` (+ JSON fallback).
  4. **Alert** — persist Grade-A to ``trade_setups`` and email newly-appearing
     Grade-A setups, sharing the page's dedupe ledgers so nothing double-fires.

Design constraints (all deliberate):

- **Fails soft everywhere.** No email config → still ingests, scores, stores.
  No DB → still emails, and the JSON board still helps a local run. Yahoo down
  for one ticker → logged, the sweep continues. A worker error can never
  surface in (or crash) the UI.
- **One thread per process**, started lazily from ``ensure_started()`` (called
  on every page load via the sidebar nav). Never started under pytest —
  ``PYTEST_CURRENT_TEST`` guards against live emails/DB writes from AppTest.
- **Runs where it's hosted.** Inside ``streamlit run`` it works whenever the
  server is up; the *same code* as a dedicated worker service in the DB's region
  (Neon ``us-east-1`` ⇒ Railway US East, see ``DEPLOY-RAILWAY.md``) gives true
  24/7 with ~2ms DB latency — that's the upgrade path, not a rewrite.

Scope: Grade-A ranker sweep + house-view board. 15M Fib Entry alerts stay
page-driven (its entries are desk-time intraday triggers).
"""
from __future__ import annotations

import logging
import os
import queue as _queue
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from src.services.precomputed import (
    build_board, serialize_house_view, store_board,
)

logger = logging.getLogger("ForexDashboard")

SCAN_INTERVAL_S = 300          # same cadence as the page fragments
GRADE_A_PCT = 80               # Grade A threshold (matches sr_email_min default)
_RR_RATIO = 2.0                # page default (sr_rr_ratio)
_RISK_PCT = 1.0                # page default (risk_pct)
_TRADES_PER_SIGNAL = 2         # page default (sr_trades_per_signal)
_STARTUP_DELAY_S = 90          # let the server (and any first page scan) settle
_PER_PAIR_DELAY_S = 0.5        # rate-limit between pairs so Yahoo doesn't throttle

_lock = threading.Lock()
_thread: Optional[threading.Thread] = None
last_scan: dict = {}           # introspection: last cycle's stats


def ensure_started(interval: int = SCAN_INTERVAL_S) -> bool:
    """Start the background daemon once per server process.

    Idempotent and cheap (a lock + liveness check), so it's safe to call on
    every page load. Returns True when the worker is running. Refuses to start
    under pytest so AppTest smoke runs can't fire live emails or DB writes.
    """
    global _thread
    if "PYTEST_CURRENT_TEST" in os.environ:
        return False
    if _thread is not None and _thread.is_alive():
        return True
    with _lock:
        if _thread is not None and _thread.is_alive():
            return True
        _thread = threading.Thread(
            target=_loop, args=(interval,), name="ingest-score-daemon", daemon=True
        )
        _thread.start()
        logger.info("[bg-scanner] started (every %ds)", interval)
        return True


def _loop(interval: int) -> None:
    # Quiet the "missing ScriptRunContext" chatter Streamlit logs when its
    # cached functions are (legitimately) called from a bare thread.
    logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)
    time.sleep(_STARTUP_DELAY_S)
    while True:
        try:
            scan_once()
        except Exception as exc:  # noqa: BLE001 — the worker must never die
            logger.warning("[bg-scanner] cycle failed: %s", exc)
        time.sleep(interval)


def _pair_frames(ticker: str):
    """Fetch the three canonical timeframes for a ticker through the spine.

    This is the ingest step — the spine's Postgres read-through persists the
    bars as a side effect. Factored out as the one network seam so tests can
    stub it (and so score() below reuses the now-warm cache).
    """
    from src.services.market_data import daily_ohlc, h4_ohlc, weekly_ohlc
    return weekly_ohlc(ticker), daily_ohlc(ticker), h4_ohlc(ticker)


def _setup_summary(r: dict) -> dict:
    """Compact, JSON-safe best-setup read for the board (what pages display)."""
    return {
        "direction": r.get("direction"),
        "grade": r.get("grade"),
        "pct": r.get("pct"),
        "score": r.get("score"),
        "max_score": r.get("max_score"),
        "close": r.get("close"),
        "sl_pips": r.get("sl_pips"),
    }


def scan_once() -> dict:
    """One full cycle: ingest → score → store board → persist/email Grade A.

    Also callable ad hoc (e.g. from a shell) for a headless cycle. Returns a
    small stats dict, mirrored to ``last_scan`` for introspection.
    """
    from src.core.bias import house_view
    from src.instruments import INSTRUMENTS
    from src.pages_lib.setup_ranker import (
        SetupRankerPage, _SetupRankerDataFeed, alert_price_bucket,
    )
    from src.services import account_state, alert_service

    # Rate-limited work queue: instruments drained one at a time so the ingest
    # step never bursts Yahoo. Single consumer by design (parallel fetch would
    # raise throttle risk); the queue keeps it easy to reason about and to
    # extend later.
    q: "_queue.Queue[str]" = _queue.Queue()
    for pair in INSTRUMENTS.keys():
        q.put(pair)

    results: List[dict] = []          # every (pair, direction) — for alerts/persist
    board_pairs: Dict[str, dict] = {}  # per-pair {hv, setup} — for the UI board

    while not q.empty():
        pair = q.get()
        info = INSTRUMENTS[pair]
        try:
            # 1. Ingest (persists bars via the read-through) + 2a. house view.
            wk, dl, h4 = _pair_frames(info["ticker"])
            hv = house_view(pair, weekly=wk, daily=dl, h4=h4)

            # 2b. Setup Ranker read, both directions (reuses the warm cache).
            pair_results = []
            for direction in ("LONG", "SHORT"):
                r = _SetupRankerDataFeed.score(pair, info, direction)
                pair_results.append(r)
                results.append(r)

            best = max(pair_results, key=lambda r: r.get("pct", 0) or 0)
            board_pairs[pair] = {
                "hv": serialize_house_view(hv),
                "setup": _setup_summary(best),
            }
        except Exception as exc:  # one bad ticker must not kill the sweep
            logger.warning("[bg-scanner] %s failed: %s", pair, exc)
        finally:
            time.sleep(_PER_PAIR_DELAY_S)  # rate limit between instruments

    # 3. Store the precomputed board (durable + JSON fallback) — this is what
    # the UI reads instead of recomputing live.
    try:
        store_board(build_board(board_pairs))
    except Exception as exc:
        logger.warning("[bg-scanner] board store failed: %s", exc)

    grade_a = [r for r in results if r.get("grade") == "A"]

    # 4a. Persist through the page's own path (shared ledger, source tag).
    saved = 0
    if grade_a:
        try:
            SetupRankerPage._persist_signals(grade_a, _RR_RATIO)
            saved = len(grade_a)
        except Exception as exc:
            logger.warning("[bg-scanner] persist failed: %s", exc)

    # 4b. Email newly-appearing Grade-A setups — page-identical keys + ledger.
    emailed = 0
    if grade_a and alert_service.email_configured():
        try:
            keys = [f"{r['pair']}|{r['direction']}|{alert_price_bucket(float(r['close']))}"
                    for r in grade_a]
            cache = alert_service.NotifyCache("setup_ranker")
            seen = cache.load()
            fresh = [(r, k) for r, k in zip(grade_a, keys) if k not in seen]
            if fresh:
                setups = [r for r, _ in fresh]
                bal = account_state.get_balance(10000.0)
                html, plain = SetupRankerPage._build_alert_email(
                    setups, _RR_RATIO, bal, _RISK_PCT, GRADE_A_PCT,
                    _TRADES_PER_SIGNAL)
                subject = (
                    f"🎰 {len(setups)} new Grade-A setup{'s' if len(setups) != 1 else ''} — "
                    f"{', '.join(r['pair'] for r in setups[:3])}{'…' if len(setups) > 3 else ''}"
                )
                ok, msg = alert_service.send_email(subject, html, plain,
                                                   source="setup_ranker_bg")
                if ok:
                    cache.filter_new([k for _, k in fresh])  # mark seen only on success
                    emailed = len(setups)
                else:
                    logger.warning("[bg-scanner] email failed: %s", msg)
        except Exception as exc:
            logger.warning("[bg-scanner] email step failed: %s", exc)

    stats = {"at": datetime.now().isoformat(timespec="seconds"),
             "scored": len(results), "pairs": len(board_pairs),
             "grade_a": len(grade_a), "emailed": emailed, "saved": saved}
    last_scan.clear()
    last_scan.update(stats)
    logger.info("[bg-scanner] %s", stats)
    return stats


def run_forever(interval: int = SCAN_INTERVAL_S) -> None:  # pragma: no cover - process entrypoint
    """Foreground ingest→score loop for a **dedicated worker process/container**
    (the Railway ``worker`` service, docker-compose ``worker``, or any always-on
    host).

    Unlike :func:`ensure_started` this does not spawn a thread or honour the
    pytest guard — it runs the cycle in the current process forever, which is
    exactly what a headless worker service wants (``python -m
    src.services.background_scanner``). Deployed in the DB's region (Neon
    ``us-east-1`` ⇒ Railway US East) it becomes the true 24/7 engine; the UI
    containers just read the board it keeps fresh.
    """
    from src.core.observability import init_logging
    try:
        init_logging()
    except Exception:
        logging.basicConfig(level=logging.INFO)
    logger.info("[bg-scanner] headless worker starting (cycle every %ds)", interval)
    while True:
        try:
            scan_once()
        except Exception as exc:  # noqa: BLE001 — the worker must never die
            logger.warning("[bg-scanner] cycle failed: %s", exc)
        time.sleep(interval)


if __name__ == "__main__":  # pragma: no cover - container/CLI entrypoint
    run_forever()
