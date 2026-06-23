"""Streamlit read-through cache for market data, backed by Postgres.

Keeps :mod:`src.db.market_data_repository` Streamlit-free. The flow for every
fetch (``cached_ohlc`` / ``cached_blob``):

1. Resolve :class:`DBConfig` from :func:`src.core.secrets.db_config`. With no
   usable credentials, skip the DB entirely and fetch live (graceful degrade).
2. Check the stored ``fetched_at``. **Fresh** (within the caller's ``ttl``) →
   serve from Postgres, no network. **Stale/missing** → fetch live, persist,
   return.
3. **Any DB error** → fall back to a live fetch so a Postgres outage never takes
   the app down.

A thin ``@st.cache_data`` wrapper sits on top purely to absorb the burst of
reruns a single widget interaction causes; the durable, cross-session source of
truth and the real staleness gate is Postgres.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional

import pandas as pd
import streamlit as st
import yfinance as yf

from src.core import secrets
from src.db.cache import _get_pool, _LeasedConnection  # reuse the one pool-per-target cache
from src.db.market_data_repository import MarketDataRepository, _OHLC_COLS
from src.db.trade_repository import DBConfig

# In-process cache lifetime — short, just to coalesce reruns. The Postgres layer
# enforces the real per-caller staleness via the ``ttl`` argument.
_MEM_TTL = 60

_OHLC_COL_LIST = list(_OHLC_COLS)


# ── DB config / repository wiring ─────────────────────────────────────────────
def _resolve_cfg() -> Optional[DBConfig]:
    """DBConfig for the read-through cache, or ``None`` when unconfigured.

    Prefers the live sidebar credentials (``st.session_state``, the runtime
    source of truth per the project's DB contract), then ``secrets``/env. A
    blank password means "DB not configured" — we skip the DB and fetch live,
    avoiding a slow connect attempt to a non-existent localhost server on every
    rerun.
    """
    # 1. Sidebar/session credentials — the same target TradeRepository uses.
    try:
        ss = st.session_state
        if ss.get("db_pass"):
            return DBConfig.from_mapping({
                "host": ss.get("db_host"), "port": ss.get("db_port", 5432),
                "dbname": ss.get("db_name"), "user": ss.get("db_user"),
                "password": ss.get("db_pass"),
            })
    except Exception:
        pass  # no Streamlit runtime (e.g. unit tests) — fall through to secrets
    # 2. secrets.toml / env / DATABASE_URL.
    try:
        cfg = DBConfig.from_mapping(secrets.db_config())
    except Exception:
        return None
    if not cfg.password:
        return None
    return cfg


def pooled_market_repository(cfg: DBConfig) -> MarketDataRepository:  # pragma: no cover - opens a live pool
    """A repository whose connections are borrowed from the shared pool."""
    pool = _get_pool(cfg.host, cfg.port, cfg.dbname, cfg.user, cfg.password)
    return MarketDataRepository(
        cfg, connect_factory=lambda: _LeasedConnection(pool, pool.getconn())
    )


# ── period parsing ────────────────────────────────────────────────────────────
_PERIOD_UNITS = {"d": 1, "mo": 30, "y": 365}


def _window_start(period: Optional[str], start: Optional[datetime]) -> Optional[datetime]:
    """The earliest timestamp the caller wants, used to trim a fresh DB read.

    ``start`` wins when given. A ``period`` like ``"60d"``/``"2y"``/``"1mo"`` is
    converted to ``now - delta``. ``"max"``/``"ytd"``/unparseable → ``None`` (no
    trim — return everything stored).
    """
    if start is not None:
        # Tolerate date / naive datetime / tz-aware datetime alike.
        if getattr(start, "tzinfo", None) is not None:
            return start.astimezone(timezone.utc).replace(tzinfo=None)
        return start
    if not period:
        return None
    m = re.fullmatch(r"(\d+)(d|mo|y)", period.strip().lower())
    if not m:
        return None
    days = int(m.group(1)) * _PERIOD_UNITS[m.group(2)]
    return datetime.utcnow() - timedelta(days=days)


# ── live fetch (single source of the yfinance call) ───────────────────────────
def _fetch_yf(
    symbol: str,
    period: Optional[str],
    interval: str,
    start: Optional[datetime],
    end: Optional[datetime],
    auto_adjust: bool,
) -> pd.DataFrame:
    """Live yfinance pull → flattened OHLCV frame. Reproduces the legacy call."""
    kwargs: dict = {"interval": interval, "progress": False, "auto_adjust": auto_adjust}
    if start is not None or end is not None:
        kwargs["start"] = start
        kwargs["end"] = end
    else:
        kwargs["period"] = period or "1mo"
    df = yf.download(symbol, **kwargs)  # pragma: no cover - live network
    if df is None or df.empty:
        return pd.DataFrame(columns=_OHLC_COL_LIST)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    keep = [c for c in _OHLC_COL_LIST if c in df.columns]
    out = df[keep]
    # Normalise to a tz-naive index so the live-fetch path matches the frame
    # reconstructed from Postgres (TIMESTAMP is tz-naive). Keeps wall-clock
    # values, so resample bucket boundaries are unchanged.
    if isinstance(out.index, pd.DatetimeIndex) and out.index.tz is not None:
        out = out.copy()
        out.index = out.index.tz_localize(None)
    return out


def _is_fresh(fetched_at: Optional[datetime], ttl: int) -> bool:
    if fetched_at is None:
        return False
    ref = fetched_at
    if ref.tzinfo is not None:
        ref = ref.astimezone(timezone.utc).replace(tzinfo=None)
    return (datetime.utcnow() - ref) < timedelta(seconds=ttl)


# ── read-through core (undecorated, unit-testable) ────────────────────────────
# A ``_repo_factory`` seam lets tests inject a fake repository; production passes
# ``pooled_market_repository`` (the default).
def _ohlc_impl(
    symbol: str,
    period: Optional[str],
    interval: str,
    start: Optional[datetime],
    end: Optional[datetime],
    ttl: int,
    auto_adjust: bool,
    repo_factory: Callable[[DBConfig], MarketDataRepository] = pooled_market_repository,
) -> pd.DataFrame:
    cfg = _resolve_cfg()
    if cfg is None:
        return _fetch_yf(symbol, period, interval, start, end, auto_adjust)
    try:
        repo = repo_factory(cfg)
        if _is_fresh(repo.last_fetched_at(symbol, interval), ttl):
            return repo.load_bars(symbol, interval, start=_window_start(period, start))
        df = _fetch_yf(symbol, period, interval, start, end, auto_adjust)
        repo.upsert_bars(symbol, interval, df)
        return df
    except Exception:
        # DB unreachable / pool exhausted / schema missing → never break the app.
        return _fetch_yf(symbol, period, interval, start, end, auto_adjust)


def _blob_impl(
    cache_key: str,
    ttl: int,
    fetch_fn: Callable[[], Any],
    repo_factory: Callable[[DBConfig], MarketDataRepository] = pooled_market_repository,
) -> Any:
    cfg = _resolve_cfg()
    if cfg is None:
        return fetch_fn()
    try:
        repo = repo_factory(cfg)
        if _is_fresh(repo.blob_fetched_at(cache_key), ttl):
            return repo.get_blob(cache_key)
        payload = fetch_fn()
        repo.set_blob(cache_key, payload)
        return payload
    except Exception:
        return fetch_fn()


# ── public read-through API (thin @st.cache_data wrappers) ────────────────────
@st.cache_data(ttl=_MEM_TTL, show_spinner=False)
def cached_ohlc(
    symbol: str,
    *,
    period: Optional[str] = None,
    interval: str = "1d",
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    ttl: int = 300,
    auto_adjust: bool = True,
) -> pd.DataFrame:  # pragma: no cover - thin @st.cache_data wrapper over _ohlc_impl
    """Read-through OHLC fetch. Serves fresh Postgres data, else fetches live and
    persists. Returns a flattened ``Open/High/Low/Close/Volume`` frame indexed by
    timestamp — the same shape ``yf.download`` produces after MultiIndex flatten,
    so existing page processing is unchanged.
    """
    return _ohlc_impl(symbol, period, interval, start, end, ttl, auto_adjust)


@st.cache_data(ttl=_MEM_TTL, show_spinner=False)
def cached_blob(cache_key: str, ttl: int, _fetch_fn: Callable[[], Any]) -> Any:  # pragma: no cover - thin @st.cache_data wrapper over _blob_impl
    """Read-through cache for non-OHLC payloads (FRED macro, news, forecasts).

    Serves a fresh JSONB blob from Postgres; otherwise calls ``_fetch_fn()``,
    persists the result, and returns it. ``_fetch_fn`` must return a
    JSON-serialisable value (dict/list/scalar). DB problems fall back to calling
    ``_fetch_fn`` directly.
    """
    return _blob_impl(cache_key, ttl, _fetch_fn)


def cached_closes(
    symbols, *, period: Optional[str] = None, interval: str = "1d", ttl: int = 600
) -> pd.DataFrame:
    """Read-through Close-price frame for several symbols (correlation/ratio
    pages). One column per symbol, aligned on the union of timestamps — each
    symbol's bars flow through :func:`cached_ohlc`, so they're persisted too.
    Replaces multi-ticker ``yf.download([...])`` calls.
    """
    frames = {}
    for sym in symbols:
        df = cached_ohlc(sym, period=period, interval=interval, ttl=ttl)
        if df is not None and not df.empty and "Close" in df.columns:
            frames[sym] = df["Close"]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1)


# Every cached read fn — one place so invalidation can't drift.
_MARKET_CACHES = (cached_ohlc, cached_blob)


def clear_market_caches() -> None:
    """Invalidate the in-process market caches (parallels clear_read_caches)."""
    for fn in _MARKET_CACHES:
        fn.clear()
