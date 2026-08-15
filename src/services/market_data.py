"""Canonical market-data spine — the one OHLC feed every page reads from.

Pages used to pick their own period/interval/TTL per fetcher, and an EMA50
computed over 90 days is not an EMA50 computed over 300 days — so two pages
could disagree about the same instrument while each being "right" on its own
private data window. This module fixes **one canonical window per timeframe**
— the exact windows `_SetupRankerDataFeed` (and therefore ``score_setup``,
the master checklist scorer) already uses — and routes every pull through
:func:`src.db.market_cache.cached_ohlc` (the Postgres read-through cache,
debug-panel aware).

Contract:

- ``interval``, resample rule and ``ttl`` are fixed **here and nowhere else**.
- A page may widen ``period``/``days`` for chart display, but the default is
  the canonical window and indicator math should use it.
- Every function degrades to an empty frame on any fetch error — pages keep
  their existing "no data" handling.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

_OHLCV = ["Open", "High", "Low", "Close", "Volume"]

# Canonical windows (single source of truth — mirrors the Setup Ranker feed).
WEEKLY_PERIOD = "2y"     # daily bars resampled to weekly
DAILY_PERIOD = "300d"
H4_DAYS = 90             # hourly bars resampled to 4H
HOURLY_DAYS = 30
CANONICAL_TTL = 300      # seconds — one staleness policy for every timeframe


def _ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Trim to the OHLCV columns and drop incomplete bars."""
    if df is None or df.empty:
        return pd.DataFrame()
    keep = [c for c in _OHLCV if c in df.columns]
    if "Close" not in keep:
        return pd.DataFrame()
    return df[keep].dropna()


def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    return df.resample(rule).agg({
        "Open": "first", "High": "max", "Low": "min",
        "Close": "last", "Volume": "sum",
    }).dropna()


def daily_ohlc(ticker: str, period: str = DAILY_PERIOD) -> pd.DataFrame:
    """Canonical daily bars (default 300d, ttl 300)."""
    from src.db.market_cache import cached_ohlc
    try:
        return _ohlcv(cached_ohlc(ticker, period=period, interval="1d",
                                  ttl=CANONICAL_TTL))
    except Exception:
        return pd.DataFrame()


def weekly_ohlc(ticker: str, period: str = WEEKLY_PERIOD) -> pd.DataFrame:
    """Canonical weekly bars: daily bars resampled to W (default 2y, ttl 300).

    Resampling from daily (rather than pulling native ``1wk`` bars) is the
    master's convention — the in-progress week is included as a partial bar,
    identically for every page.
    """
    from src.db.market_cache import cached_ohlc
    try:
        df = _ohlcv(cached_ohlc(ticker, period=period, interval="1d",
                                ttl=CANONICAL_TTL))
        if df.empty:
            return df
        return _resample(df, "W")
    except Exception:
        return pd.DataFrame()


MONTHLY_PERIOD = "5y"    # ~60 monthly bars — enough for a 12-month lookback


def monthly_ohlc(ticker: str, period: str = MONTHLY_PERIOD) -> pd.DataFrame:
    """Canonical monthly bars: daily bars resampled to month-end (default 5y).

    Resampled from daily rather than pulled as native ``1mo`` bars — the same
    convention ``weekly_ohlc`` follows, and for the same reason. Native monthly
    bars are a *different* series from the daily one every other timeframe
    derives from, which is how a month's change comes to disagree with the weeks
    inside it. The in-progress month is included as a partial bar, identically
    for every page.
    """
    from src.db.market_cache import cached_ohlc
    try:
        df = _ohlcv(cached_ohlc(ticker, period=period, interval="1d",
                                ttl=CANONICAL_TTL))
        if df.empty:
            return df
        # "ME", not the removed "M": month-end is the pandas 3.0 spelling.
        return _resample(df, "ME")
    except Exception:
        return pd.DataFrame()


def h4_ohlc(ticker: str, days: int = H4_DAYS) -> pd.DataFrame:
    """Canonical 4-hour bars: hourly bars resampled to 4h (default 90d, ttl 300)."""
    from src.db.market_cache import cached_ohlc
    try:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days)
        df = _ohlcv(cached_ohlc(ticker, start=start, end=end, interval="1h",
                                ttl=CANONICAL_TTL))
        if df.empty:
            return df
        return _resample(df, "4h")
    except Exception:
        return pd.DataFrame()


def hourly_ohlc(ticker: str, days: int = HOURLY_DAYS) -> pd.DataFrame:
    """Canonical hourly bars (default 30d, ttl 300)."""
    from src.db.market_cache import cached_ohlc
    try:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days)
        return _ohlcv(cached_ohlc(ticker, start=start, end=end, interval="1h",
                                  ttl=CANONICAL_TTL))
    except Exception:
        return pd.DataFrame()


def data_asof(*frames: pd.DataFrame) -> Optional[pd.Timestamp]:
    """Latest bar timestamp across the given frames — the page's "data as of"."""
    stamps = []
    for df in frames:
        if df is not None and len(df) and isinstance(df.index, pd.DatetimeIndex):
            stamps.append(df.index.max())
    return max(stamps) if stamps else None


def asof_caption(*frames: pd.DataFrame) -> str:
    """Human caption for the shared strip, e.g. ``data as of 2026-07-17 08:00 UTC``."""
    ts = data_asof(*frames)
    if ts is None:
        return "no data"
    return f"data as of {ts.strftime('%Y-%m-%d %H:%M')} UTC"
