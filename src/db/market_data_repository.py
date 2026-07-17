"""Market-data persistence via PostgreSQL.

A read-through cache's durable backing store. Three tables:

* ``market_bars``       — one row per OHLC bar, keyed ``(symbol, interval, ts)``.
* ``market_fetch_meta`` — last-fetch timestamp per ``(symbol, interval)``; drives
  the staleness gate (refetch only when older than the caller's TTL).
* ``market_blobs``      — generic JSONB store (``cache_key`` PK) for the non-OHLC
  long tail: FRED macro dicts, FRED series, news lists, forecast outputs.

Streamlit-free by design (mirrors :class:`src.db.trade_repository.TradeRepository`):
connections come from ``connect_factory`` so callers inject a pooled connection
(``src/db/market_cache.py``) or a fake one in unit tests. SQL uses the same
``with closing(self._connect()) as conn, conn, conn.cursor()`` idiom — closing
handles close, the middle ``conn`` handles commit/rollback.
"""
from __future__ import annotations

from contextlib import closing
from datetime import datetime
from typing import Any, Callable, List, Optional

import pandas as pd
import psycopg2
import psycopg2.extras

from src.db.trade_repository import DBConfig

# yfinance names the index "Datetime" for intraday intervals and "Date" for
# daily/weekly/monthly. Reconstructed frames match that so downstream
# reset_index()/column references behave identically to a live fetch.
_INTRADAY_SUFFIXES = ("m", "h")
_OHLC_COLS = ("Open", "High", "Low", "Close", "Volume")


def _index_name(interval: str) -> str:
    return "Datetime" if interval.endswith(_INTRADAY_SUFFIXES) else "Date"


class MarketDataRepository:
    """OHLC bar store + generic JSONB blob cache behind a single class."""

    CREATE_BARS_SQL = """
        CREATE TABLE IF NOT EXISTS market_bars (
            symbol   VARCHAR(30) NOT NULL,
            interval VARCHAR(10) NOT NULL,
            ts       TIMESTAMP   NOT NULL,
            open     FLOAT,
            high     FLOAT,
            low      FLOAT,
            close    FLOAT,
            volume   FLOAT,
            PRIMARY KEY (symbol, interval, ts)
        )
    """

    CREATE_META_SQL = """
        CREATE TABLE IF NOT EXISTS market_fetch_meta (
            symbol     VARCHAR(30) NOT NULL,
            interval   VARCHAR(10) NOT NULL,
            fetched_at TIMESTAMP   NOT NULL DEFAULT NOW(),
            rows       INT,
            PRIMARY KEY (symbol, interval)
        )
    """

    CREATE_BLOBS_SQL = """
        CREATE TABLE IF NOT EXISTS market_blobs (
            cache_key  TEXT PRIMARY KEY,
            payload    JSONB,
            fetched_at TIMESTAMP NOT NULL DEFAULT NOW()
        )
    """

    # Dukascopy historical archive (src/data/dukascopy_backfill.py) for the
    # local debug mode's source="duka" replay/time-machine (src/debug/) — a
    # separate table from market_bars, not the same one, since this data is
    # deliberately never touched by normal live-fetch traffic, only by the
    # backfill script running on demand.
    CREATE_DUKA_BARS_SQL = """
        CREATE TABLE IF NOT EXISTS dukascopy_bars (
            symbol   VARCHAR(30) NOT NULL,
            interval VARCHAR(10) NOT NULL,
            ts       TIMESTAMP   NOT NULL,
            open     FLOAT,
            high     FLOAT,
            low      FLOAT,
            close    FLOAT,
            volume   FLOAT,
            PRIMARY KEY (symbol, interval, ts)
        )
    """

    def __init__(
        self,
        cfg: DBConfig,
        connect_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        self.cfg = cfg
        self._connect_factory = connect_factory

    # ── connection helper ───────────────────────────────────────────────────
    def _connect(self):
        if self._connect_factory is not None:
            return self._connect_factory()
        return psycopg2.connect(**self.cfg.as_kwargs())

    # ── schema ──────────────────────────────────────────────────────────────
    def init_schema(self) -> tuple[bool, str]:
        """Create all four market tables. Returns (ok, message)."""
        try:
            with closing(self._connect()) as conn, conn, conn.cursor() as cur:
                cur.execute(self.CREATE_BARS_SQL)
                cur.execute(self.CREATE_META_SQL)
                cur.execute(self.CREATE_BLOBS_SQL)
                cur.execute(self.CREATE_DUKA_BARS_SQL)
                conn.commit()
            return True, "Connected"
        except Exception as exc:
            return False, str(exc)

    # ── OHLC writes ───────────────────────────────────────────────────────────
    def upsert_bars(self, symbol: str, interval: str, df: pd.DataFrame) -> int:
        """Persist an OHLC frame and stamp the fetch metadata.

        ``df`` is the flattened ``Open/High/Low/Close/Volume`` frame indexed by
        timestamp (exactly what the live fetcher returns). Existing bars are
        overwritten so a re-fetch refreshes the latest (possibly still-forming)
        bar. Returns the number of bars written.
        """
        if df is None or df.empty:
            # Still stamp the fetch so an empty result isn't retried every rerun.
            self._stamp_meta(symbol, interval, 0)
            return 0

        bar_sql = """
            INSERT INTO market_bars (symbol, interval, ts, open, high, low, close, volume)
            VALUES (%(symbol)s, %(interval)s, %(ts)s, %(open)s, %(high)s, %(low)s, %(close)s, %(volume)s)
            ON CONFLICT (symbol, interval, ts) DO UPDATE SET
                open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low,
                close = EXCLUDED.close, volume = EXCLUDED.volume
        """
        payload: List[dict] = []
        for ts, row in df.iterrows():
            payload.append({
                "symbol": symbol, "interval": interval,
                "ts": pd.Timestamp(ts).to_pydatetime(),
                "open": _f(row.get("Open")), "high": _f(row.get("High")),
                "low": _f(row.get("Low")), "close": _f(row.get("Close")),
                "volume": _f(row.get("Volume")),
            })
        with closing(self._connect()) as conn, conn, conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, bar_sql, payload)
            cur.execute(self._META_UPSERT_SQL, (symbol, interval, len(payload)))
            conn.commit()
        return len(payload)

    _META_UPSERT_SQL = """
        INSERT INTO market_fetch_meta (symbol, interval, fetched_at, rows)
        VALUES (%s, %s, NOW(), %s)
        ON CONFLICT (symbol, interval) DO UPDATE SET
            fetched_at = NOW(), rows = EXCLUDED.rows
    """

    def _stamp_meta(self, symbol: str, interval: str, rows: int) -> None:
        with closing(self._connect()) as conn, conn, conn.cursor() as cur:
            cur.execute(self._META_UPSERT_SQL, (symbol, interval, rows))
            conn.commit()

    # ── OHLC reads ──────────────────────────────────────────────────────────
    def last_fetched_at(self, symbol: str, interval: str) -> Optional[datetime]:
        sql = "SELECT fetched_at FROM market_fetch_meta WHERE symbol = %s AND interval = %s"
        try:
            with closing(self._connect()) as conn, conn, conn.cursor() as cur:
                cur.execute(sql, (symbol, interval))
                row = cur.fetchone()
            return row[0] if row else None
        except Exception:
            return None

    def load_bars(
        self, symbol: str, interval: str, start: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Return the stored bars as a flattened OHLCV frame indexed by timestamp.

        ``start`` trims to ``ts >= start`` so a fresh DB read returns the same
        window the caller asked for. Empty frame (with the right columns) when
        nothing is stored.
        """
        sql = (
            "SELECT ts, open, high, low, close, volume FROM market_bars "
            "WHERE symbol = %s AND interval = %s"
        )
        params: list = [symbol, interval]
        if start is not None:
            sql += " AND ts >= %s"
            params.append(start)
        sql += " ORDER BY ts ASC"
        with closing(self._connect()) as conn, conn, conn.cursor() as cur:
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()
        idx_name = _index_name(interval)
        if not rows:
            empty = pd.DataFrame(columns=list(_OHLC_COLS))
            empty.index = pd.DatetimeIndex([], name=idx_name)
            return empty
        df = pd.DataFrame(rows, columns=["ts", "Open", "High", "Low", "Close", "Volume"])
        df["ts"] = pd.to_datetime(df["ts"])
        df = df.set_index("ts")
        df.index.name = idx_name
        return df

    # ── Dukascopy archive (backfill writes, debug-mode source="duka" reads) ──
    def upsert_dukascopy_bars(self, symbol: str, interval: str, df: pd.DataFrame) -> int:
        """Persist a Dukascopy OHLC frame. Same shape/overwrite semantics as
        :meth:`upsert_bars`, separate table — see dukascopy_backfill.py."""
        if df is None or df.empty:
            return 0
        bar_sql = """
            INSERT INTO dukascopy_bars (symbol, interval, ts, open, high, low, close, volume)
            VALUES (%(symbol)s, %(interval)s, %(ts)s, %(open)s, %(high)s, %(low)s, %(close)s, %(volume)s)
            ON CONFLICT (symbol, interval, ts) DO UPDATE SET
                open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low,
                close = EXCLUDED.close, volume = EXCLUDED.volume
        """
        payload: List[dict] = []
        for ts, row in df.iterrows():
            payload.append({
                "symbol": symbol, "interval": interval,
                "ts": pd.Timestamp(ts).to_pydatetime(),
                "open": _f(row.get("Open")), "high": _f(row.get("High")),
                "low": _f(row.get("Low")), "close": _f(row.get("Close")),
                "volume": _f(row.get("Volume")),
            })
        with closing(self._connect()) as conn, conn, conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, bar_sql, payload)
            conn.commit()
        return len(payload)

    def load_dukascopy_bars(
        self, symbol: str, interval: str, start: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Return archived Dukascopy bars as a flattened OHLCV frame — same
        contract as :meth:`load_bars`. Empty frame when nothing is archived
        for this symbol (e.g. GBP/ZAR, which Dukascopy doesn't carry)."""
        sql = (
            "SELECT ts, open, high, low, close, volume FROM dukascopy_bars "
            "WHERE symbol = %s AND interval = %s"
        )
        params: list = [symbol, interval]
        if start is not None:
            sql += " AND ts >= %s"
            params.append(start)
        sql += " ORDER BY ts ASC"
        with closing(self._connect()) as conn, conn, conn.cursor() as cur:
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()
        idx_name = _index_name(interval)
        if not rows:
            empty = pd.DataFrame(columns=list(_OHLC_COLS))
            empty.index = pd.DatetimeIndex([], name=idx_name)
            return empty
        df = pd.DataFrame(rows, columns=["ts", "Open", "High", "Low", "Close", "Volume"])
        df["ts"] = pd.to_datetime(df["ts"])
        df = df.set_index("ts")
        df.index.name = idx_name
        return df

    # ── generic blob cache ────────────────────────────────────────────────────
    def blob_fetched_at(self, cache_key: str) -> Optional[datetime]:
        sql = "SELECT fetched_at FROM market_blobs WHERE cache_key = %s"
        try:
            with closing(self._connect()) as conn, conn, conn.cursor() as cur:
                cur.execute(sql, (cache_key,))
                row = cur.fetchone()
            return row[0] if row else None
        except Exception:
            return None

    def get_blob(self, cache_key: str) -> Optional[Any]:
        sql = "SELECT payload FROM market_blobs WHERE cache_key = %s"
        with closing(self._connect()) as conn, conn, conn.cursor() as cur:
            cur.execute(sql, (cache_key,))
            row = cur.fetchone()
        return row[0] if row else None

    def set_blob(self, cache_key: str, payload: Any) -> None:
        sql = """
            INSERT INTO market_blobs (cache_key, payload, fetched_at)
            VALUES (%s, %s, NOW())
            ON CONFLICT (cache_key) DO UPDATE SET
                payload = EXCLUDED.payload, fetched_at = NOW()
        """
        with closing(self._connect()) as conn, conn, conn.cursor() as cur:
            cur.execute(sql, (cache_key, psycopg2.extras.Json(payload)))
            conn.commit()


def _f(v: Any) -> Optional[float]:
    """Coerce a cell to float, mapping NaN/None to SQL NULL."""
    if v is None:
        return None
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return None
    return None if fv != fv else fv  # NaN -> None
