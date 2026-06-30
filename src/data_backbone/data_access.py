"""
data_access.py — the read path your tabs call.

    Postgres (durable)  ->  yfinance / FRED (source of truth)

get_ohlcv() and get_fred() are drop-in replacements for the per-tab loaders:
they serve from Postgres and only hit the external API when the stored data is
missing or stale (then write it back to Postgres).
"""
from __future__ import annotations

import io
import datetime as dt
import pandas as pd
import requests

try:
    import yfinance as yf
except Exception:
    yf = None

from . import db
from .config import STALE_DAYS

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
HEADERS = {"User-Agent": "Mozilla/5.0 (trading-dashboard)"}


def _stale(index_max, max_age_days: int = STALE_DAYS) -> bool:
    if index_max is None or pd.isna(index_max):
        return True
    last = pd.to_datetime(index_max).normalize()
    now = pd.Timestamp(dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)).normalize()
    return (now - last).days > max_age_days


# ---------- prices ----------

def fetch_yf(ticker: str, period: str) -> pd.DataFrame:
    if yf is None:
        return pd.DataFrame()
    d = yf.download(ticker, period=period, interval="1d", progress=False)
    if d is None or d.empty:
        return pd.DataFrame()
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d.index = pd.to_datetime(d.index)
    return d.dropna(how="all")


def get_ohlcv(ticker: str, period: str = "5y", force: bool = False) -> pd.DataFrame:
    df = db.read_price_bars(ticker)          # durable layer
    if force or df is None or df.empty or _stale(df.index.max() if not df.empty else None):
        fresh = fetch_yf(ticker, period)     # source of truth
        if not fresh.empty:
            db.upsert_price_bars(ticker, fresh)
            df = db.read_price_bars(ticker)

    return df if df is not None else pd.DataFrame()


# ---------- FRED ----------

def fetch_fred(series_id: str) -> pd.Series:
    r = requests.get(FRED_CSV.format(sid=series_id), headers=HEADERS, timeout=20)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    dcol, vcol = df.columns[0], df.columns[1]
    df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
    df[vcol] = pd.to_numeric(df[vcol], errors="coerce")
    s = df.dropna(subset=[dcol]).set_index(dcol)[vcol].dropna()
    s.name = series_id
    return s


def get_fred(series_id: str, force: bool = False) -> pd.Series:
    s = db.read_fred(series_id)
    # macro series update slowly; allow a longer stale window (30d).
    if force or s is None or s.empty or _stale(s.index.max() if not s.empty else None, 30):
        fresh = fetch_fred(series_id)
        if not fresh.empty:
            db.upsert_fred(series_id, fresh)
            s = db.read_fred(series_id)

    return s if s is not None else pd.Series(dtype=float)
