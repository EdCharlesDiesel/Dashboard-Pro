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

from src.services import confluence_alert
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
_FIB_DAYS = 5                  # 15M history for the confluence leg (page default)
_FIB_LOOKBACK = 96             # impulse-leg lookback in 15M bars (page default)

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


def _find_confluences(grade_a: List[dict], board_pairs: Dict[str, dict]) -> List:
    """Grade-A ranker reads that ALSO carry a same-side house view and a live
    15M fib trigger. The 15M fetch is the only extra network cost and it runs
    just for Grade-A candidates — never the whole universe.

    Isolated so :func:`scan_once` stays a scheduler, and so tests can stub it.
    Never raises: a failure here must not cost the ingest/score/store work.
    """
    from src.pages_lib.fib_entry import _fetch_15m, fib_analysis
    from src.instruments import INSTRUMENTS

    out = []
    for r in grade_a:
        pair = r.get("pair")
        entry = board_pairs.get(pair) or {}
        hv = entry.get("hv") or {}
        try:
            inst = INSTRUMENTS.get(pair)
            if inst is None:
                continue
            df15 = _fetch_15m(inst.ticker, days=_FIB_DAYS)
            if df15 is None or df15.empty:
                continue
            fib = fib_analysis(df15, r["direction"], _FIB_LOOKBACK)
            c = confluence_alert.evaluate(
                pair, r, hv.get("direction"), hv.get("score"),
                fib, r["direction"], min_pct=GRADE_A_PCT)
            if c is not None:
                out.append(c)
        except Exception as exc:
            logger.warning("[bg-scanner] confluence check failed for %s: %s", pair, exc)
        finally:
            time.sleep(_PER_PAIR_DELAY_S)      # same rate limit as the sweep
    return out


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


_EXEC_ENGINE = None


def _execution_engine():
    """SQLAlchemy engine for the execution queue, or ``None`` if unavailable.

    Built from ``db_config()`` rather than ``DATABASE_URL`` directly: that is the
    repo's single resolver, so this works unchanged in the container (DB_* env)
    and on the host (secrets.toml), instead of silently falling back to
    localhost the way a raw env read would.
    """
    global _EXEC_ENGINE
    if _EXEC_ENGINE is None:
        from sqlalchemy import create_engine

        from src.core.secrets import db_config

        cfg = db_config()
        _EXEC_ENGINE = create_engine(
            f"postgresql+psycopg2://{cfg['user']}:{cfg['password']}"
            f"@{cfg['host']}:{cfg['port']}/{cfg['dbname']}",
            pool_pre_ping=True)
    return _EXEC_ENGINE


def _enqueue_for_execution(fresh: List) -> int:
    """Put fresh confluences on the execution queue. Returns how many landed.

    Two conversions the executor depends on, both silent if wrong:
    `EUR/USD` -> `EURUSD` (the gate parses the quote currency positionally) and
    `LONG` -> `buy` (it compares against lowercase, so an uppercased direction
    would be read as the opposite side).

    Nothing here decides whether to trade — `gate.run_gate()` does that on the
    executor side, against the live book and market. This only offers.
    """
    from src.execution import queue as exec_queue

    engine = _execution_engine()
    if engine is None:
        return 0

    placed = 0
    for c in fresh:
        symbol = c.pair.replace("/", "")
        side = "buy" if c.direction.upper() == "LONG" else "sell"
        signal_id = exec_queue.make_signal_id(
            symbol, side, c.entry, c.sl, datetime.now(), source="bg_scanner")
        ok = exec_queue.enqueue_signal(
            engine, signal_id=signal_id, symbol=symbol, direction=side,
            entry=c.entry, stop=c.sl, tp1=c.tp1, tp2=c.tp2,
            source="bg_scanner",
            meta={"ranker_pct": c.ranker_pct, "ranker_grade": c.ranker_grade,
                  "house_direction": c.house_direction,
                  "house_score": c.house_score, "fib_status": c.fib_status,
                  "rr1": c.rr1, "pair": c.pair})
        if ok:
            placed += 1
    return placed


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

    # 4b. Email only TRIPLE CONFLUENCE — Grade-A ranker + house view + a live
    # 15M fib trigger, all the same side. Grade A alone is a week-long opinion
    # and fired constantly; requiring the 15M leg means every email has an
    # entry to take right now. Deliberately rare (see confluence_alert docs).
    emailed = 0
    telegrammed = 0
    try:
        confluences = _find_confluences(grade_a, board_pairs)
    except Exception as exc:
        # The 15M leg touches the network and imports the fib page library —
        # neither may cost us the ingest/score/store work already done.
        logger.warning("[bg-scanner] confluence scan failed: %s", exc)
        confluences = []
    # Two independent channels. This whole block used to sit behind
    # `email_configured()`, so an owner with Telegram and no SMTP received
    # nothing at all - the alert was suppressed by a check about a different
    # delivery mechanism.
    if confluences:
        try:
            from src.core import secrets as core_secrets

            cache = alert_service.NotifyCache("confluence_alert")
            seen = cache.load()
            fresh = [c for c in confluences if c.dedupe_key() not in seen]
            if fresh:
                delivered = False

                if alert_service.email_configured():
                    html, plain = confluence_alert.build_email(fresh)
                    ok, msg = alert_service.send_email(
                        confluence_alert.subject_for(fresh), html, plain,
                        source="confluence_bg")
                    if ok:
                        delivered = True
                        emailed = len(fresh)
                    else:
                        logger.warning("[bg-scanner] confluence email failed: %s", msg)

                # Attempted unconditionally: the sender reports "not
                # configured" harmlessly rather than raising.
                ok_tg, msg_tg = core_secrets.send_telegram_message(
                    confluence_alert.build_telegram(fresh))
                if ok_tg:
                    delivered = True
                    telegrammed = len(fresh)
                else:
                    logger.warning("[bg-scanner] confluence telegram failed: %s", msg_tg)

                # Execution queue — opt-in, and in its own try/except.
                #
                # Behind EXECUTOR_ENQUEUE because deploying a build that
                # happens to contain this code must never start queueing
                # orders: automation is switched on deliberately or not at all.
                #
                # Isolated because alerting predates it and must not regress.
                # A Postgres outage may cost the queue; it may not cost the
                # message that wakes someone at 3am. That is also why this sits
                # after both channels rather than before them.
                if os.environ.get("EXECUTOR_ENQUEUE") == "1":
                    try:
                        queued = _enqueue_for_execution(fresh)
                        if queued:
                            logger.info("[bg-scanner] queued %d signal(s) for "
                                        "execution", queued)
                    except Exception as exc:
                        logger.warning("[bg-scanner] execution enqueue failed "
                                       "(alert already sent): %s", exc)

                # filter_new() records keys as seen, so it runs once and only
                # if something actually arrived. Marking on email alone would
                # let a working Telegram plus broken SMTP re-alert forever;
                # marking on neither would swallow an alert nobody received.
                if delivered:
                    cache.filter_new([c.dedupe_key() for c in fresh])
        except Exception as exc:
            logger.warning("[bg-scanner] confluence alert step failed: %s", exc)

    stats = {"at": datetime.now().isoformat(timespec="seconds"),
             "scored": len(results), "pairs": len(board_pairs),
             "grade_a": len(grade_a), "emailed": emailed,
             "telegrammed": telegrammed, "saved": saved}
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
