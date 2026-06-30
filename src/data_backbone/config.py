"""
config.py — configuration for the data_backbone pipeline.

Everything tuneable in one place. Values come from environment variables (the
same ones docker-compose already passes to the app/db services) with safe
defaults, so the pipeline runs locally without any extra setup.

This module holds no business logic — it only resolves connection strings and
the watchlists that ``db``, ``data_access``, ``worker`` and ``seed_history``
consume.
"""
from __future__ import annotations

import os


# ── Postgres (SQLAlchemy URL) ────────────────────────────────────────────────
# Mirrors the DB_* env vars docker-compose passes to the app service.
DB_HOST = os.getenv("DB_HOST", os.getenv("POSTGRES_HOST", "localhost"))
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", os.getenv("POSTGRES_DB", "trading"))
DB_USER = os.getenv("DB_USER", os.getenv("POSTGRES_USER", "postgres"))
DB_PASSWORD = os.getenv("DB_PASSWORD", os.getenv("POSTGRES_PASSWORD", "postgres"))

DB_URL = os.getenv(
    "DB_URL",
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
)


# ── staleness window ─────────────────────────────────────────────────────────
STALE_DAYS = int(os.getenv("STALE_DAYS", "1"))                 # refetch if older

# Deepest daily history the worker keeps current.
PRICE_PERIOD = os.getenv("PRICE_PERIOD", "5y")


# ── watchlists ───────────────────────────────────────────────────────────────
def _default_tickers() -> list[str]:
    """Derive the price watchlist from the instrument registry — the project's
    single source of truth — so the pipeline scans the identical universe as the
    rest of the app. Falls back to a small set if the registry is unavailable."""
    try:
        from src.instruments.registry import INSTRUMENTS
        return [inst.ticker for inst in INSTRUMENTS.values()]
    except Exception:
        return ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "GC=F", "SI=F"]


WATCH_TICKERS = _default_tickers()

# Macro series backing the market-overview / DXY-vs-Gold views.
WATCH_FRED = [
    "DFF",        # effective fed funds rate
    "DGS2",       # 2-year treasury
    "DGS10",      # 10-year treasury
    "T10Y2Y",     # 10y-2y spread
    "DTWEXBGS",   # broad USD index
    "CPIAUCSL",   # CPI
    "UNRATE",     # unemployment
    "VIXCLS",     # VIX
]
