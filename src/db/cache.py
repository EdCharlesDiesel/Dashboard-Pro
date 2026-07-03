"""Connection pooling for the Postgres trade repository.

Keeps ``src/db/trade_repository.py`` free of Streamlit. Provides one layer:

**Connection pooling** — one ``ThreadedConnectionPool`` per DB target, cached
with ``@st.cache_resource`` for the app's lifetime. Streamlit reruns the whole
script on every widget interaction; without a pool each query opens a fresh
Postgres socket. ``pooled_repository()`` hands the repository a connection
factory that borrows from the pool and returns the connection on ``close()``.

Reads (``cached_*``) and writes go **straight to Postgres** through the pool —
there is no intermediate cache. :func:`clear_read_caches` is retained as a no-op
so callers that invalidate after a write don't need to change; with no read
cache there is nothing to invalidate.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import psycopg2.pool
import streamlit as st

from src.db.trade_repository import DBConfig, TradeRepository


class _LeasedConnection:
    """Proxy over a pooled psycopg2 connection.

    Delegates the exact connection contract the repository relies on — cursor
    creation and the transaction context manager (``with conn:``) — to the real
    connection, but repurposes ``close()`` as "return to pool". That lets the
    repository keep using ``contextlib.closing(self._connect())`` unchanged: the
    connection is recycled instead of having its socket torn down.
    """

    __slots__ = ("_pool", "_conn")

    def __init__(self, pool: psycopg2.pool.AbstractConnectionPool, conn: Any) -> None:
        self._pool = pool
        self._conn = conn

    def cursor(self, *args, **kwargs):
        return self._conn.cursor(*args, **kwargs)

    def commit(self):
        return self._conn.commit()

    def rollback(self):
        return self._conn.rollback()

    def __enter__(self):
        # psycopg2's connection context manager wraps a transaction (commit on
        # success, rollback on error) — it does NOT close the connection.
        return self._conn.__enter__()

    def __exit__(self, exc_type, exc, tb):
        return self._conn.__exit__(exc_type, exc, tb)

    def close(self) -> None:
        self._pool.putconn(self._conn)


# Sized to cover a scan page's parallel fetch fan-out (src/services/
# parallel_fetch.py, DEFAULT_MAX_WORKERS=8) plus headroom for other
# concurrent reads/writes (repository + market-data pools share this cache
# per DB target, and both trade_repository and market_data_repository draw
# from it at once during a scan).
_POOL_MAXCONN = 12


@st.cache_resource(show_spinner=False)
def _get_pool(host: str, port: int, dbname: str, user: str, password: str):  # pragma: no cover - opens a live Postgres pool
    """One pool per distinct DB target, cached for the app's lifetime."""
    return psycopg2.pool.ThreadedConnectionPool(
        minconn=1,
        maxconn=_POOL_MAXCONN,
        host=host,
        port=int(port),
        dbname=dbname,
        user=user,
        password=password,
    )


def _cfg(host: str, port: int, dbname: str, user: str, password: str) -> DBConfig:
    return DBConfig.from_mapping({
        "host": host, "port": port, "dbname": dbname,
        "user": user, "password": password,
    })


def pooled_repository(cfg: DBConfig) -> TradeRepository:
    """A repository whose connections are borrowed from the cached pool."""
    pool = _get_pool(cfg.host, cfg.port, cfg.dbname, cfg.user, cfg.password)
    return TradeRepository(
        cfg, connect_factory=lambda: _LeasedConnection(pool, pool.getconn())
    )


# ── direct reads (straight to Postgres via the pool) ──────────────────────────
# Thin pooled passthroughs to the repository (fully unit-tested in
# tests/test_trade_repository.py); they need a live DB to execute, so the DB path
# is excluded from coverage.
def cached_load_setups(
    host: str, port: int, dbname: str, user: str, password: str, limit: int = 100
) -> List[Dict[str, Any]]:
    return pooled_repository(_cfg(host, port, dbname, user, password)).load_setups(limit=limit)  # pragma: no cover - needs live DB


def cached_load_open(
    host: str, port: int, dbname: str, user: str, password: str
) -> List[Dict[str, Any]]:
    return pooled_repository(_cfg(host, port, dbname, user, password)).load_open()  # pragma: no cover - needs live DB


def cached_daily_losses(
    host: str, port: int, dbname: str, user: str, password: str, max_losses: int = 2
) -> Dict[str, Any]:
    return pooled_repository(_cfg(host, port, dbname, user, password)).daily_losses(max_losses=max_losses)  # pragma: no cover - needs live DB


def cached_performance_stats(
    host: str, port: int, dbname: str, user: str, password: str, n: int = 20
) -> Optional[Dict[str, Any]]:
    return pooled_repository(_cfg(host, port, dbname, user, password)).performance_stats(n=n)  # pragma: no cover - needs live DB


def cached_realized_pnl(
    host: str, port: int, dbname: str, user: str, password: str
) -> Dict[str, Any]:
    return pooled_repository(_cfg(host, port, dbname, user, password)).realized_pnl()  # pragma: no cover - needs live DB


# ── app_state key/value (direct read + write to Postgres) ─────────────────────
# Used for durable app-level state such as the live account balance.
def cached_get_state(
    host: str, port: int, dbname: str, user: str, password: str, state_key: str
) -> Optional[Any]:
    return pooled_repository(_cfg(host, port, dbname, user, password)).get_state(state_key)  # pragma: no cover - needs live DB


def set_state(
    host: str, port: int, dbname: str, user: str, password: str,
    state_key: str, value: Any
) -> None:
    """Persist ``value`` to Postgres."""
    pooled_repository(_cfg(host, port, dbname, user, password)).set_state(state_key, value)  # pragma: no cover - needs live DB


# Every read function — kept in one place for documentation/tests.
_READ_CACHES = (
    cached_load_setups,
    cached_load_open,
    cached_daily_losses,
    cached_performance_stats,
    cached_realized_pnl,
)


def clear_read_caches() -> None:
    """No-op retained for API compatibility.

    Reads now go straight to Postgres, so there is no cache to invalidate after a
    write. Callers (the signal store and the journal/stats pages) still call this
    after a write; keeping it as a no-op means none of them need to change.
    """
    return None
