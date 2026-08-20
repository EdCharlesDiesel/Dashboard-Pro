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
# Resolved through `src.core.secrets.db_config()`, the same path the rest of the
# app uses, so there is exactly one answer to "where is the database".
#
# This module used to read the DB_* env vars itself and default to
# localhost:5432/trading. Inside Docker that was right -- compose passes
# DB_HOST=db / DB_PORT=5432 and the container reaches the `db` service on its
# internal port. On the *host* it was wrong twice over: a native Windows
# PostgreSQL owns 5432, and no database called `trading` exists on either
# server. Every macro read therefore failed authentication and fell back to the
# FRED API -- silently, because `fred_data`'s fallback works, so the only
# symptom was a log nobody read and a `fred_series` table that stayed empty.
# Measured 2026-08-16.
#
# NOTE the precedence, which is not the obvious one: db_config() tries the
# secrets.toml value BEFORE the env var. The container is safe only because
# .dockerignore keeps secrets.toml out of the image, leaving the DB_* env vars
# as the only source there. Mount a secrets.toml into a container and every
# service will chase the host's 127.0.0.1:5433 from inside Docker.
# `tests/test_data_backbone_config.py` guards that .dockerignore line.
def _resolve_db() -> dict:
    """Connection settings, or safe defaults if secrets cannot be read.

    Importing a config module must never be what takes the app down, so a
    failure here degrades to env vars rather than raising at import time.
    """
    try:
        from src.core.secrets import db_config
        return dict(db_config())
    except Exception:                       # noqa: BLE001 — import must not fail
        return {"host": os.getenv("DB_HOST", "localhost"),
                "port": int(os.getenv("DB_PORT", "5432")),
                # `postgres` always exists; `trading` never did on either
                # server, so that default turned a misconfiguration into a
                # confusing "database does not exist".
                "dbname": os.getenv("DB_NAME", "postgres"),
                "user": os.getenv("DB_USER", "postgres"),
                "password": os.getenv("DB_PASSWORD", "")}


_DB = _resolve_db()

DB_HOST: str = str(_DB.get("host", "localhost"))
DB_PORT: int = int(_DB.get("port", 5432))
DB_NAME: str = str(_DB.get("dbname", "postgres"))
DB_USER: str = str(_DB.get("user", "postgres"))
DB_PASSWORD: str = str(_DB.get("password", ""))

DB_URL: str = os.getenv(
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
