"""Central observability: structured logging + event recording.

One place wires up everything that "records what the app is doing":

* Python ``logging`` — a rotating file (``logs/app.log``) plus the console, so
  every ``logging.getLogger("ForexDashboard")`` call across the codebase is
  finally captured (previously nothing was configured and the records were
  dropped). Third-party WARNING/ERROR (incl. Streamlit's uncaught-exception
  logs) are captured too.
* ``logs/events.jsonl`` — one JSON object per line for every structured event
  (page views, data fetches, exceptions, custom events). Easy to grep, tail or
  load into pandas.
* Postgres ``app_events`` table — the same events, best-effort, so they can be
  queried with SQL next to ``trade_setups``.

Every write is best-effort and guarded: recording must NEVER break the app. If
the log directory is read-only or Postgres is down, the page still renders.

Public API
----------
``init_logging()``            idempotent; call once per process (cheap to repeat)
``get_logger(name)``          a configured logger
``log_event(type, **fields)`` record a structured event everywhere
``log_page_view(page)``       convenience event for a page load / rerun
``log_exception(exc, ...)``   record a traceback as an event
``track_fetch(source, ...)``  context manager that times a data fetch
``read_events(limit)``        tail the JSONL event log (for the viewer page)
``read_app_log(limit)``       tail the text log (for the viewer page)
``read_pg_events(limit)``     query the Postgres app_events table
"""
from __future__ import annotations

import inspect
import json
import logging
import logging.handlers
import os
import sys
import threading
import time
import traceback
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

from src.core import secrets

LOGGER_NAME = "ForexDashboard"

_LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
_APP_LOG = _LOG_DIR / "app.log"
_EVENTS_LOG = _LOG_DIR / "events.jsonl"

_MAX_BYTES = 5 * 1024 * 1024  # 5 MB per file
_BACKUPS = 5

_init_lock = threading.Lock()
_initialised = False
_events_lock = threading.Lock()

# Module-level Postgres connection reused across Streamlit reruns (the module is
# imported once per process). ``_pg_disabled`` short-circuits after a failure so
# a down database doesn't add latency to every event.
_pg_lock = threading.Lock()
_pg_conn = None
_pg_disabled = False


# ─────────────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────────────

def init_logging() -> None:
    """Configure handlers once. Safe (and cheap) to call on every page."""
    global _initialised
    if _initialised:
        return
    with _init_lock:
        if _initialised:
            return
        try:
            _LOG_DIR.mkdir(parents=True, exist_ok=True)
        except Exception:
            # Can't create the log dir — fall back to console-only below.
            pass

        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        root = logging.getLogger()
        # Capture third-party WARNING+ (Streamlit logs uncaught script
        # exceptions at ERROR through logging — this is how we record crashes
        # on legacy pages without wrapping every script).
        if root.level == logging.NOTSET or root.level > logging.WARNING:
            root.setLevel(logging.WARNING)

        handlers: List[logging.Handler] = []
        try:
            fh = logging.handlers.RotatingFileHandler(
                _APP_LOG, maxBytes=_MAX_BYTES, backupCount=_BACKUPS, encoding="utf-8"
            )
            handlers.append(fh)
        except Exception:
            pass
        handlers.append(logging.StreamHandler(stream=sys.stderr))

        for h in handlers:
            h.setLevel(logging.INFO)
            h.setFormatter(fmt)
            # Avoid attaching duplicate handlers on hot-reload.
            if not any(type(x) is type(h) and getattr(x, "baseFilename", None)
                       == getattr(h, "baseFilename", None) for x in root.handlers):
                root.addHandler(h)

        # Our app logger: INFO+ flows up to the root handlers above. (Root stays
        # at WARNING so third-party INFO noise is dropped at source while our
        # INFO is still written, because propagation is checked against handler
        # levels, not the ancestor logger level.)
        app_logger = logging.getLogger(LOGGER_NAME)
        app_logger.setLevel(logging.INFO)
        app_logger.propagate = True

        _install_excepthooks()
        _initialised = True
        app_logger.info("observability initialised — logs at %s", _LOG_DIR)


def get_logger(name: str = LOGGER_NAME) -> logging.Logger:
    init_logging()
    return logging.getLogger(name)


def _install_excepthooks() -> None:
    """Record uncaught exceptions on the main thread and worker threads."""
    prev = sys.excepthook

    def _hook(exc_type, exc_value, tb):
        try:
            log_exception(exc_value, where="excepthook")
        except Exception:
            pass
        prev(exc_type, exc_value, tb)

    sys.excepthook = _hook

    def _thread_hook(args):
        try:
            log_exception(args.exc_value, where="thread:%s" % args.thread.name)
        except Exception:
            pass

    try:
        threading.excepthook = _thread_hook
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Context helpers
# ─────────────────────────────────────────────────────────────────────────────

def session_id() -> str:
    """Stable per-browser-session id, or a sentinel outside a session."""
    try:
        sid = st.session_state.get("_obs_session_id")
        if not sid:
            sid = uuid.uuid4().hex[:12]
            st.session_state["_obs_session_id"] = sid
        return sid
    except Exception:
        return "no-session"


def _detect_page() -> str:
    """Best-effort name of the page being rendered, from the call stack."""
    try:
        for frame in inspect.stack():
            fn = frame.filename.replace("\\", "/")
            base = os.path.basename(fn)
            if base == "app.py" or "/pages/" in fn:
                return os.path.splitext(base)[0]
    except Exception:
        pass
    return "unknown"


def _session_context() -> Dict[str, Any]:
    """Pull a few commonly-useful selections off session_state, if present."""
    keys = ("selected_instrument", "selected_symbol", "symbol",
            "trend_timeframe", "trade_direction", "session")
    out: Dict[str, Any] = {}
    try:
        for k in keys:
            v = st.session_state.get(k)
            if v not in (None, ""):
                out[k] = v
    except Exception:
        pass
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Event recording
# ─────────────────────────────────────────────────────────────────────────────

def log_event(event_type: str, level: str = "INFO", message: str = "",
              page: Optional[str] = None, **fields: Any) -> None:
    """Record one structured event to JSONL, Postgres and the text log."""
    init_logging()
    page = page or _detect_page()
    sid = session_id()
    ts = datetime.now(timezone.utc)
    record: Dict[str, Any] = {
        "ts": ts.isoformat(),
        "session_id": sid,
        "page": page,
        "event_type": event_type,
        "level": level,
        "message": message,
        "data": _safe(fields),
    }

    # 1) JSONL file
    try:
        line = json.dumps(record, default=str)
        with _events_lock:
            with open(_EVENTS_LOG, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception:
        pass

    # 2) Text log (so events also show up in app.log alongside ad-hoc logging)
    try:
        lvl = getattr(logging, level.upper(), logging.INFO)
        compact = " ".join(f"{k}={v}" for k, v in fields.items())
        logging.getLogger(LOGGER_NAME).log(
            lvl, "[%s] %s %s", event_type, message or "", compact
        )
    except Exception:
        pass

    # 3) Postgres (best-effort)
    _pg_insert(ts, sid, page, event_type, level, message, record["data"])


def log_page_view(page: Optional[str] = None, **extra: Any) -> None:
    fields = _session_context()
    fields.update(extra)
    log_event("page_view", page=page, **fields)


def log_exception(exc: BaseException, where: str = "",
                  page: Optional[str] = None, **fields: Any) -> None:
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    log_event(
        "exception", level="ERROR",
        message=f"{type(exc).__name__}: {exc}",
        page=page, where=where, traceback=tb, **fields,
    )


@contextmanager
def track_fetch(source: str, **fields: Any):
    """Time a data fetch and record latency / success / error.

    Set ``cm.rows`` inside the block to record how many rows came back::

        with track_fetch("yfinance", ticker=t) as fetch:
            df = ...
            fetch.rows = len(df)
    """
    class _Tracker:
        rows: Optional[int] = None

    tracker = _Tracker()
    t0 = time.perf_counter()
    err: Optional[BaseException] = None
    try:
        yield tracker
    except BaseException as e:  # noqa: BLE001 — re-raised below
        err = e
        raise
    finally:
        dur_ms = round((time.perf_counter() - t0) * 1000, 1)
        log_event(
            "data_fetch",
            level="ERROR" if err else "INFO",
            source=source,
            duration_ms=dur_ms,
            ok=err is None,
            rows=tracker.rows,
            error=(f"{type(err).__name__}: {err}" if err else None),
            **fields,
        )


def _safe(obj: Any) -> Any:
    """Make a value JSON-serialisable without throwing."""
    try:
        json.dumps(obj)
        return obj
    except Exception:
        if isinstance(obj, dict):
            return {str(k): _safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_safe(v) for v in obj]
        return str(obj)


# ─────────────────────────────────────────────────────────────────────────────
# Postgres sink
# ─────────────────────────────────────────────────────────────────────────────

_CREATE_EVENTS_SQL = """
CREATE TABLE IF NOT EXISTS app_events (
    id         BIGSERIAL PRIMARY KEY,
    ts         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    session_id VARCHAR(40),
    page       VARCHAR(80),
    event_type VARCHAR(40),
    level      VARCHAR(10),
    message    TEXT,
    data       JSONB
);

-- The only read this table serves is `ORDER BY ts DESC LIMIT n` (see
-- recent_events below), and the PRIMARY KEY on id does nothing for it. Without
-- this, every read was a parallel seq scan plus a top-N heapsort over the whole
-- table: measured at 62.9 ms for LIMIT 200 across 59,123 rows, and growing with
-- every logged event. A descending index on ts answers the ordering directly.
--
-- This is the one index worth adding here. `trade_setups` documents the same
-- rule from the other side: an index nothing queries is pure write-time cost.
CREATE INDEX IF NOT EXISTS idx_app_events_ts ON app_events (ts DESC);
"""


def _ensure_pg():
    """Return a live connection or None. Disables itself after a failure."""
    global _pg_conn, _pg_disabled
    if _pg_disabled:
        return None
    with _pg_lock:
        if _pg_disabled:
            return None
        try:
            if _pg_conn is not None and getattr(_pg_conn, "closed", 1) == 0:
                return _pg_conn
            import psycopg2
            cfg = secrets.db_config()
            if not cfg.get("password") and cfg.get("user") == "postgres":
                # No real credentials configured — don't spam connection errors.
                _pg_disabled = True
                return None
            conn = psycopg2.connect(
                host=cfg["host"], port=cfg["port"], dbname=cfg["dbname"],
                user=cfg["user"], password=cfg["password"], connect_timeout=3,
            )
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute(_CREATE_EVENTS_SQL)
            _pg_conn = conn
            return _pg_conn
        except Exception:
            _pg_disabled = True
            _pg_conn = None
            return None


def reset_pg() -> None:
    """Re-enable the Postgres sink (e.g. after the user fixes credentials)."""
    global _pg_conn, _pg_disabled
    with _pg_lock:
        try:
            if _pg_conn is not None:
                _pg_conn.close()
        except Exception:
            pass
        _pg_conn = None
        _pg_disabled = False


def _pg_insert(ts, sid, page, event_type, level, message, data) -> None:
    conn = _ensure_pg()
    if conn is None:
        return
    global _pg_disabled, _pg_conn
    try:
        import psycopg2.extras
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO app_events "
                "(ts, session_id, page, event_type, level, message, data) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (ts, sid, page, event_type, level, message,
                 psycopg2.extras.Json(data)),
            )
    except Exception:
        # Drop the connection; next event re-evaluates (but stay enabled once).
        try:
            conn.close()
        except Exception:
            pass
        _pg_conn = None


# ─────────────────────────────────────────────────────────────────────────────
# Readers (used by the System Logs viewer page)
# ─────────────────────────────────────────────────────────────────────────────

def read_events(limit: int = 500) -> List[Dict[str, Any]]:
    """Return the last ``limit`` events from the JSONL log (newest last)."""
    if not _EVENTS_LOG.exists():
        return []
    try:
        with open(_EVENTS_LOG, "r", encoding="utf-8") as f:
            lines = f.readlines()[-limit:]
        out: List[Dict[str, Any]] = []
        for ln in lines:
            ln = ln.strip()
            if not ln:
                continue
            try:
                out.append(json.loads(ln))
            except Exception:
                continue
        return out
    except Exception:
        return []


def read_app_log(limit: int = 500) -> str:
    """Return the last ``limit`` lines of the text log."""
    if not _APP_LOG.exists():
        return ""
    try:
        with open(_APP_LOG, "r", encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-limit:])
    except Exception:
        return ""


def read_pg_events(limit: int = 500) -> List[Dict[str, Any]]:
    """Query the Postgres app_events table, newest first. [] if unavailable."""
    conn = _ensure_pg()
    if conn is None:
        return []
    try:
        import psycopg2.extras
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT ts, session_id, page, event_type, level, message, data "
                "FROM app_events ORDER BY ts DESC LIMIT %s",
                (limit,),
            )
            return [dict(r) for r in cur.fetchall()]
    except Exception:
        return []


def log_paths() -> Dict[str, Path]:
    return {"dir": _LOG_DIR, "app_log": _APP_LOG, "events": _EVENTS_LOG}
