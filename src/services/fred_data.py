"""Canonical macro spine — one definition of a FRED series, for every page.

The OHLC equivalent is `src/services/market_data.py`, and the reasoning is
identical: a page that fetches FRED itself picks its own window, its own cache
policy and its own timeout, and two pages then disagree about the same series.
Before this existed, `pages/quant_models_tab.py`, `pages/disconnect_monitor_tab.py`
and `src/pages_lib/market_overview_lib.py` each held their own FRED client.

Postgres (`fred_series`, filled daily by `src.data_backbone.worker`) is the
source of truth. HTTP is the fallback for a series the worker does not watch,
and what it fetches is written back so the next reader is served locally —
the same read-through shape `market_cache` gives the OHLC spine.

Never raises. A page asking for macro data must degrade to "no data", never to
a traceback, because these series decorate a read rather than carry it.

Pure of Streamlit: no `st.cache_data` here. Callers that want request-level
caching decorate their own wrapper, exactly as they do around the OHLC spine.
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger("ForexDashboard")

# One staleness policy for every macro series. Macro data moves daily at most
# (several of the watched series are monthly), so this is deliberately far
# looser than the OHLC spine's 300s.
CANONICAL_TTL = 6 * 3600


def _read_stored(series_id: str) -> pd.Series:
    """Stored observations, oldest first. The IO seam tests stub."""
    from src.data_backbone import db
    return db.read_fred(series_id)


def _fetch_remote(series_id: str) -> pd.Series:
    """FRED over HTTP. The other IO seam tests stub."""
    from src.data_backbone import data_access
    return data_access.fetch_fred(series_id)


def _write_back(series_id: str, s: pd.Series) -> None:
    from src.data_backbone import db
    db.upsert_fred(series_id, s)


def _empty() -> pd.Series:
    return pd.Series(dtype=float)


def fred_series(series_id: str, start: Optional[str] = None) -> pd.Series:
    """A FRED series as a date-indexed Series, Postgres first.

    `start` is inclusive and trims the left edge, matching the OHLC spine's
    convention. The index is always tz-naive: `source_scorecard` and the price
    frames these series get aligned against are naive UTC, and a tz-aware index
    silently refuses to join with them.
    """
    try:
        s = _read_stored(series_id)
    except Exception as exc:  # noqa: BLE001 — a macro read must not break a page
        logger.warning("[fred_data] stored read failed for %s: %s", series_id, exc)
        s = None

    if s is None or s.empty:
        try:
            s = _fetch_remote(series_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[fred_data] remote fetch failed for %s: %s", series_id, exc)
            return _empty()
        if s is not None and not s.empty:
            try:
                _write_back(series_id, s)
            except Exception as exc:  # noqa: BLE001
                # Caching is a bonus; answering the caller is the job.
                logger.warning("[fred_data] write-back failed for %s: %s",
                               series_id, exc)

    if s is None or s.empty:
        return _empty()

    idx = pd.to_datetime(s.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    s = s.copy()
    s.index = idx
    s = s.sort_index()
    if start is not None:
        s = s[s.index >= pd.Timestamp(start)]
    return s
