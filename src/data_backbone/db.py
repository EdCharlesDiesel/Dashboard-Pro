"""
db.py — Postgres persistence (SQLAlchemy 2.0 Core).

Tables:
  price_bars  (ticker, date) -> OHLCV          [permanent price store]
  fred_series (series_id, date) -> value        [permanent macro store]
  trades      (id) -> trade journal             [replaces the CSV journal]

Writes use INSERT ... ON CONFLICT DO UPDATE (true upserts), so re-running the
worker never duplicates rows and always patches in corrections/new bars.
"""
from __future__ import annotations

import pandas as pd
from sqlalchemy import (create_engine, MetaData, Table, Column, String, Date,
                        Float, Integer, Text, select)
from sqlalchemy.dialects.postgresql import insert as pg_insert

from .config import DB_URL

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(DB_URL, pool_pre_ping=True, future=True)
    return _engine


metadata = MetaData()

price_bars = Table(
    "price_bars", metadata,
    Column("ticker", String(32), primary_key=True),
    Column("date", Date, primary_key=True),
    Column("open", Float), Column("high", Float), Column("low", Float),
    Column("close", Float), Column("volume", Float),
)

fred_series = Table(
    "fred_series", metadata,
    Column("series_id", String(64), primary_key=True),
    Column("date", Date, primary_key=True),
    Column("value", Float),
)

trades = Table(
    "trades", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("opened", Date), Column("closed", Date),
    Column("pair", String(32)), Column("direction", String(8)),
    Column("setup", String(64)), Column("entry", Float), Column("stop", Float),
    Column("target", Float), Column("exit", Float), Column("size", Float),
    Column("status", String(16)), Column("confidence", Integer),
    Column("thesis", Text), Column("notes", Text),
)


def init_db():
    """Create tables if they don't exist."""
    metadata.create_all(get_engine())


def _f(v):
    """float or None (so NaN becomes SQL NULL)."""
    try:
        x = float(v)
        return None if pd.isna(x) else x
    except (TypeError, ValueError):
        return None


# ---------- price bars ----------

def _price_records(ticker: str, df: pd.DataFrame) -> list[dict]:
    d = df.reset_index()
    d.columns = [str(c).lower() for c in d.columns]
    datecol = "date" if "date" in d.columns else d.columns[0]
    out = []
    for _, row in d.iterrows():
        dt = pd.to_datetime(row[datecol], errors="coerce")
        if pd.isna(dt):
            continue
        out.append(dict(ticker=ticker, date=dt.date(),
                        open=_f(row.get("open")), high=_f(row.get("high")),
                        low=_f(row.get("low")), close=_f(row.get("close")),
                        volume=_f(row.get("volume"))))
    return out


def price_upsert_stmt(records: list[dict]):
    """Build the upsert statement (separated so it can be unit-tested/compiled)."""
    stmt = pg_insert(price_bars).values(records)
    return stmt.on_conflict_do_update(
        index_elements=["ticker", "date"],
        set_={c: stmt.excluded[c] for c in ("open", "high", "low", "close", "volume")})


def upsert_price_bars(ticker: str, df: pd.DataFrame) -> int:
    recs = _price_records(ticker, df)
    if not recs:
        return 0
    with get_engine().begin() as conn:
        conn.execute(price_upsert_stmt(recs))
    return len(recs)


def read_price_bars(ticker: str, start=None) -> pd.DataFrame:
    q = select(price_bars).where(price_bars.c.ticker == ticker)
    if start is not None:
        q = q.where(price_bars.c.date >= pd.to_datetime(start).date())
    with get_engine().connect() as conn:
        df = pd.read_sql(q, conn)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df[["open", "high", "low", "close", "volume"]].rename(columns=str.capitalize)


# ---------- FRED ----------

def fred_upsert_stmt(records: list[dict]):
    stmt = pg_insert(fred_series).values(records)
    return stmt.on_conflict_do_update(
        index_elements=["series_id", "date"],
        set_={"value": stmt.excluded["value"]})


def upsert_fred(series_id: str, series: pd.Series) -> int:
    recs = [dict(series_id=series_id, date=pd.to_datetime(d).date(), value=_f(v))
            for d, v in series.items() if not pd.isna(pd.to_datetime(d, errors="coerce"))]
    if not recs:
        return 0
    with get_engine().begin() as conn:
        conn.execute(fred_upsert_stmt(recs))
    return len(recs)


def read_fred(series_id: str) -> pd.Series:
    q = select(fred_series).where(fred_series.c.series_id == series_id)
    with get_engine().connect() as conn:
        df = pd.read_sql(q, conn)
    if df.empty:
        return pd.Series(dtype=float)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()["value"].rename(series_id)


# ---------- trade journal ----------

def save_trade(row: dict) -> None:
    with get_engine().begin() as conn:
        conn.execute(pg_insert(trades).values(**row))


def read_trades() -> pd.DataFrame:
    with get_engine().connect() as conn:
        return pd.read_sql(select(trades), conn)


def migrate_journal_csv(path: str) -> int:
    """One-shot import of the existing CSV journal into Postgres."""
    df = pd.read_csv(path)
    cols = [c for c in df.columns if c in
            {"opened", "closed", "pair", "direction", "setup", "entry", "stop",
             "target", "exit", "size", "status", "confidence", "thesis", "notes"}]
    n = 0
    for _, r in df.iterrows():
        save_trade({c: (None if pd.isna(r[c]) else r[c]) for c in cols})
        n += 1
    return n
