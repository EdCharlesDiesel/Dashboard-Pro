"""Streamlit caching layer for the Postgres trade repository.

Keeps ``src/db/trade_repository.py`` free of Streamlit. Two independent layers:

1. **Connection pooling** — one ``ThreadedConnectionPool`` per DB target, cached
   with ``@st.cache_resource`` for the app's lifetime. Streamlit reruns the whole
   script on every widget interaction; without a pool each query opens a fresh
   Postgres socket. ``pooled_repository()`` hands the repository a connection
   factory that borrows from the pool and returns the connection on ``close()``.

2. **Read caching** — query results cached with ``@st.cache_data(ttl=READ_TTL)``
   so repeated reads inside the TTL don't hit Postgres. Writes (save/close/
   delete) MUST call :func:`clear_read_caches` so the next read re-queries —
   you never cache a write, you invalidate after one.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import psycopg2.pool
import streamlit as st

from src.db.trade_repository import DBConfig, TradeRepository

# Reads are cached this long. Short enough that a second device's trades show up
# quickly; long enough to absorb the burst of reruns a single interaction causes.
READ_TTL = 60


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


@st.cache_resource(show_spinner=False)
def _get_pool(host: str, port: int, dbname: str, user: str, password: str):  # pragma: no cover - opens a live Postgres pool
    """One pool per distinct DB target, cached for the app's lifetime."""
    return psycopg2.pool.ThreadedConnectionPool(
        minconn=1,
        maxconn=5,
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


# ── cached reads (keyed on connection params so a DB change invalidates) ──────
# The bodies are thin pooled passthroughs to the repository (which is fully unit-
# tested in tests/test_trade_repository.py) and need a live DB to execute, so
# they are excluded from coverage; the wiring around them is tested.
@st.cache_data(ttl=READ_TTL, show_spinner=False)
def cached_load_setups(
    host: str, port: int, dbname: str, user: str, password: str, limit: int = 100
) -> List[Dict[str, Any]]:  # pragma: no cover - needs live DB
    return pooled_repository(_cfg(host, port, dbname, user, password)).load_setups(limit=limit)


@st.cache_data(ttl=READ_TTL, show_spinner=False)
def cached_load_open(
    host: str, port: int, dbname: str, user: str, password: str
) -> List[Dict[str, Any]]:  # pragma: no cover - needs live DB
    return pooled_repository(_cfg(host, port, dbname, user, password)).load_open()


@st.cache_data(ttl=READ_TTL, show_spinner=False)
def cached_daily_losses(
    host: str, port: int, dbname: str, user: str, password: str, max_losses: int = 2
) -> Dict[str, Any]:  # pragma: no cover - needs live DB
    return pooled_repository(_cfg(host, port, dbname, user, password)).daily_losses(max_losses=max_losses)


@st.cache_data(ttl=READ_TTL, show_spinner=False)
def cached_performance_stats(
    host: str, port: int, dbname: str, user: str, password: str, n: int = 20
) -> Optional[Dict[str, Any]]:  # pragma: no cover - needs live DB
    return pooled_repository(_cfg(host, port, dbname, user, password)).performance_stats(n=n)


@st.cache_data(ttl=READ_TTL, show_spinner=False)
def cached_realized_pnl(
    host: str, port: int, dbname: str, user: str, password: str
) -> Dict[str, Any]:  # pragma: no cover - needs live DB
    return pooled_repository(_cfg(host, port, dbname, user, password)).realized_pnl()


# Every cached read function — kept in one place so invalidation can't drift.
_READ_CACHES = (
    cached_load_setups,
    cached_load_open,
    cached_daily_losses,
    cached_performance_stats,
    cached_realized_pnl,
)


def clear_read_caches() -> None:
    """Invalidate all cached reads. Call after any write so the journal, stats,
    open-trades and P/L re-query Postgres on the next rerun."""
    for fn in _READ_CACHES:
        fn.clear()
