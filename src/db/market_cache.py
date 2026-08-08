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

import logging
import re
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Optional

import pandas as pd
import streamlit as st
import yfinance as yf

logger = logging.getLogger("ForexDashboard")

from src.core import secrets
from src.db.cache import _get_pool, _LeasedConnection  # reuse the one pool-per-target cache
from src.db.market_data_repository import MarketDataRepository, _OHLC_COLS
from src.db.trade_repository import DBConfig

# In-process cache lifetimes. The Postgres layer enforces the real per-caller
# staleness via the ``ttl`` argument; the in-process layer only decides how
# long a rerun is served from memory before re-consulting Postgres. With a
# remote DB (Neon us-east-1 from SAST ≈ 200ms/round-trip, 2 round-trips per
# symbol×interval check), a 60s memory TTL meant every interaction after a
# minute re-paid the full network cost — 20-30s on universe-wide pages. Two
# lanes keep intraday freshness intact while making the heavy daily/weekly
# pages rerun instantly:
#   • standard lane (caller ttl ≥ 300 — the canonical spine and slower): 300s
#   • fast lane (caller ttl < 300 — 15M fib, predictive): 60s, unchanged
_MEM_TTL_FAST = 60
_MEM_TTL_STD = 300
_FAST_LANE_BELOW = 300

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


# ── coverage (freshness is not sufficiency) ───────────────────────────────────
# ``market_bars`` holds whatever window the *last* caller asked for, and
# ``last_fetched_at`` records only *when* that happened — not how far back it
# reached. So a symbol first cached at ``period="2y"`` kept serving 2 years to
# every later ``period="5y"`` request, for as long as the row stayed fresh.
#
# Not hypothetical, and it did not self-heal: the stale branch *would* have
# refetched at the deeper period, but `sweeper` re-warms every symbol every 5
# minutes, so a row is almost never stale when a deep request arrives. Whichever
# caller warmed a symbol first therefore pinned its depth permanently. Measured
# 2026-08-08, before this fix — one interval, one table, five different depths:
#
#     CHFZAR=X  260 bars    SPY       503 bars    ^VIX  755 bars
#     USDJPY=X  780 bars    DX-Y.NYB  503 bars (yfinance holds 1257)
#
# The damage is silent by construction — a short frame is still a valid frame.
# `weekly_ohlc` asks for 2y and resamples: on CHF/ZAR it was building every
# weekly indicator from 52 weeks instead of 105.
#
# A stored window counts as covering the request when its first bar lands within
# a slack of the requested start. Slack absorbs weekends, holidays and coarse
# intervals (a monthly bar can legitimately start ~a month late), scaled to the
# span so it stays proportionate on both 60d and 10y requests.
_COVERAGE_SLACK_MIN = timedelta(days=10)
_COVERAGE_SLACK_FRAC = 0.05

# (symbol, interval) -> (earliest bar returned, window start we asked for).
#
# Recorded only when a fetch came up *short* of what was asked. Without it a
# symbol whose history genuinely starts later than the request — a young
# listing, a delisted ticker — would fail the coverage check forever and refetch
# live on every single call, turning the cache off for that symbol. Process-local
# by design: one wasted fetch per symbol per restart is the whole cost, and
# nothing stale can outlive the process.
_HISTORY_FLOOR: dict = {}


def _naive(value) -> pd.Timestamp:
    """Any timestamp-ish value as a tz-naive UTC ``pd.Timestamp``.

    ``_window_start`` passes a bare ``date`` straight through, and ``date`` does
    not subtract from ``datetime`` — without this the comparison would raise and
    be swallowed as "covered", quietly restoring the very bug this guards.
    """
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def _covers(df: Optional[pd.DataFrame], want: Optional[datetime]) -> bool:
    """Does ``df`` actually reach back to ``want``?"""
    if df is None or df.empty:
        # Fresh-but-empty is never sufficient: a stamped fetch that stored
        # nothing would otherwise hand every caller a blank frame until the TTL
        # expired.
        return False
    if want is None:
        return True  # "max"/ytd/unparseable — no window asked for, nothing to check
    try:
        first = _naive(df.index.min())
        w = _naive(want)
        slack = max(_COVERAGE_SLACK_MIN,
                    (_naive(datetime.utcnow()) - w) * _COVERAGE_SLACK_FRAC)
        return bool(first <= w + slack)
    except Exception:
        # A non-datetime index (or anything else unexpected) must not put us in
        # a refetch loop — treat it as covered and serve what we have.
        return True


def _history_exhausted(symbol: str, interval: str, want: Optional[datetime]) -> bool:
    """True when we already learned the provider cannot reach back to ``want``."""
    rec = _HISTORY_FLOOR.get((symbol, interval))
    if rec is None:
        return False
    _, asked = rec
    if asked is None:
        return True   # we asked for everything and that was all there was
    if want is None:
        return False  # now asking for everything — worth one attempt
    try:
        # No more history than the attempt that already fell short.
        return bool(_naive(want) >= _naive(asked))
    except Exception:
        return False


def _remember_history_floor(
    symbol: str, interval: str, want: Optional[datetime], df: pd.DataFrame
) -> None:
    """Note that ``symbol`` could not be fetched back as far as ``want``."""
    if _covers(df, want):
        return
    first = None
    try:
        if df is not None and not df.empty:
            first = df.index.min()
    except Exception:
        pass
    _HISTORY_FLOOR[(symbol, interval)] = (first, want)


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
    source: str = "live",
) -> pd.DataFrame:
    cfg = _resolve_cfg()
    want = _window_start(period, start)
    if source == "duka":
        if cfg is None:
            logger.warning("[market_cache] source='duka' requested but DB not "
                           "configured — falling back to live for %s", symbol)
        else:
            try:
                repo = repo_factory(cfg)
                df = repo.load_dukascopy_bars(symbol, interval, start=want)
                if _covers(df, want):
                    return df
                if not df.empty:
                    # Same failure as the live cache had: a short archive is
                    # still a valid frame, so serving it degrades the caller's
                    # window with nothing to notice. There is no live remedy on
                    # this path — the archive only deepens when somebody runs the
                    # backfill — so say exactly that and fall through rather than
                    # hand back a silently truncated series.
                    logger.warning("[market_cache] source='duka' archive for %s (%s) "
                                   "starts %s, short of the requested %s — falling "
                                   "back to live. Run src/data/dukascopy_backfill.py "
                                   "to deepen it.",
                                   symbol, interval, df.index.min(), want)
                else:
                    logger.warning("[market_cache] source='duka' has no archived bars "
                                   "for %s (%s) — falling back to live. Run "
                                   "src/data/dukascopy_backfill.py to backfill it.",
                                   symbol, interval)
            except Exception as exc:
                logger.warning("[market_cache] source='duka' read failed for %s: "
                               "%s — falling back to live", symbol, exc)
        # Falls through to the live/cache path below on any duka miss.
    if cfg is None:
        return _fetch_yf(symbol, period, interval, start, end, auto_adjust)
    try:
        repo = repo_factory(cfg)
        if _is_fresh(repo.last_fetched_at(symbol, interval), ttl):
            cached = repo.load_bars(symbol, interval, start=want)
            if _covers(cached, want) or _history_exhausted(symbol, interval, want):
                return cached
            logger.info(
                "[market_cache] %s (%s) is fresh but stored history starts %s, "
                "short of the requested %s — backfilling",
                symbol, interval,
                (cached.index.min() if not cached.empty else "nothing"), want,
            )
        df = _fetch_yf(symbol, period, interval, start, end, auto_adjust)
        repo.upsert_bars(symbol, interval, df)
        _remember_history_floor(symbol, interval, want, df)
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


# ── debug-mode source/asof resolution ──────────────────────────────────────
def _resolve_debug_defaults(
    source: Optional[str], asof: Optional[date]
) -> tuple[str, Optional[date]]:
    """Fill unspecified source/asof from the debug panel's session_state.

    This is what makes every existing ``cached_ohlc`` call site debug-aware
    for free — resolved *outside* the ``@st.cache_data``-decorated function,
    because ``st.cache_data`` keys on its arguments, not on global state read
    inside the function body. If this resolution happened inside the cached
    function instead, toggling the debug panel's source/date would never
    bust the cache and every page would keep serving whatever it first
    computed. See src/debug/panel.py.
    """
    if source is None:
        try:
            source = st.session_state.get("dbg_source", "live")
        except Exception:
            source = "live"
    if asof is None:
        try:
            asof = st.session_state.get("dbg_asof")
        except Exception:
            asof = None
    return source, asof


# ── public read-through API (thin @st.cache_data wrappers) ────────────────────
def cached_ohlc(
    symbol: str,
    *,
    period: Optional[str] = None,
    interval: str = "1d",
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    ttl: int = 300,
    auto_adjust: bool = True,
    source: Optional[str] = None,
    asof: Optional[date] = None,
) -> pd.DataFrame:
    """Read-through OHLC fetch. Serves fresh Postgres data, else fetches live and
    persists. Returns a flattened ``Open/High/Low/Close/Volume`` frame indexed by
    timestamp — the same shape ``yf.download`` produces after MultiIndex flatten,
    so existing page processing is unchanged.

    ``source``/``asof`` default to the debug panel's session_state
    (``dbg_source``/``dbg_asof``) when not passed explicitly — a page needs
    no changes to become debug-aware. ``asof`` truncates the returned frame
    to ``index <= asof`` (the "time machine" — replay a page as of a past
    date). ``source="duka"`` reads the Dukascopy archive
    (``src/data/dukascopy_backfill.py``, the ``dukascopy_bars`` table) keyed
    by the same yfinance ticker as ``symbol`` here — any symbol/interval not
    yet backfilled (or with no Dukascopy mapping, e.g. GBP/ZAR) falls back to
    live with a warning rather than erroring. Provenance (source/asof/rows/
    date range/NaN count) is attached to the returned frame's
    ``.attrs["provenance"]`` for the debug panel to show.
    """
    source, asof = _resolve_debug_defaults(source, asof)
    impl = _cached_ohlc_impl_fast if ttl < _FAST_LANE_BELOW else _cached_ohlc_impl
    return impl(symbol, period, interval, start, end, ttl, auto_adjust, source, asof)


def _ohlc_with_provenance(
    symbol: str,
    period: Optional[str],
    interval: str,
    start: Optional[datetime],
    end: Optional[datetime],
    ttl: int,
    auto_adjust: bool,
    source: str,
    asof: Optional[date],
) -> pd.DataFrame:  # pragma: no cover - exercised via the cached wrappers
    df = _ohlc_impl(symbol, period, interval, start, end, ttl, auto_adjust, source=source)
    df = _apply_asof(df, asof)
    df.attrs["provenance"] = {
        "source": source, "asof": str(asof) if asof else "live",
        "rows": len(df),
        "start": str(df.index.min()) if len(df) else None,
        "end": str(df.index.max()) if len(df) else None,
        "nans": int(df.isna().sum().sum()) if len(df) else 0,
    }
    return df


@st.cache_data(ttl=_MEM_TTL_STD, show_spinner=False)
def _cached_ohlc_impl(
    symbol: str,
    period: Optional[str],
    interval: str,
    start: Optional[datetime],
    end: Optional[datetime],
    ttl: int,
    auto_adjust: bool,
    source: str,
    asof: Optional[date],
) -> pd.DataFrame:  # pragma: no cover - thin @st.cache_data wrapper
    return _ohlc_with_provenance(symbol, period, interval, start, end, ttl,
                                 auto_adjust, source, asof)


@st.cache_data(ttl=_MEM_TTL_FAST, show_spinner=False)
def _cached_ohlc_impl_fast(
    symbol: str,
    period: Optional[str],
    interval: str,
    start: Optional[datetime],
    end: Optional[datetime],
    ttl: int,
    auto_adjust: bool,
    source: str,
    asof: Optional[date],
) -> pd.DataFrame:  # pragma: no cover - thin @st.cache_data wrapper
    return _ohlc_with_provenance(symbol, period, interval, start, end, ttl,
                                 auto_adjust, source, asof)


def _apply_asof(df: pd.DataFrame, asof: Optional[date]) -> pd.DataFrame:
    """Truncate to bars at or before ``asof`` — the replay/backtest time machine."""
    if asof is None or df.empty:
        return df
    cutoff = pd.Timestamp(asof)
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        cutoff = cutoff.tz_localize(df.index.tz)
    return df[df.index <= cutoff]


@st.cache_data(ttl=_MEM_TTL_STD, show_spinner=False)
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


# Every cached read fn — one place so invalidation can't drift. cached_ohlc
# itself is a plain resolver wrapper now (so debug-mode source/asof toggles
# actually bust the cache — see _resolve_debug_defaults); the real
# @st.cache_data-decorated functions underneath it are the two lane impls.
_MARKET_CACHES = (_cached_ohlc_impl, _cached_ohlc_impl_fast, cached_blob)


def clear_market_caches() -> None:
    """Invalidate the in-process market caches (parallels clear_read_caches)."""
    for fn in _MARKET_CACHES:
        fn.clear()
