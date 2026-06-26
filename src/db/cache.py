"""Caching layer for the Postgres trade repository.

Keeps ``src/db/trade_repository.py`` free of Streamlit/Redis. Two independent
layers:

1. **Connection pooling** — one ``ThreadedConnectionPool`` per DB target, cached
   with ``@st.cache_resource`` for the app's lifetime. Streamlit reruns the whole
   script on every widget interaction; without a pool each query opens a fresh
   Postgres socket. ``pooled_repository()`` hands the repository a connection
   factory that borrows from the pool and returns the connection on ``close()``.

2. **Read caching (Redis cache-aside)** — query results are cached in Redis under
   a TTL (:data:`READ_TTL`). Each ``cached_*`` read is a textbook cache-aside:

       check Redis ─hit→ return            (no DB hit)
                   └miss→ query Postgres → store in Redis → return

   Writes (save/close/delete) MUST call :func:`clear_read_caches` so the next
   read re-queries — you never cache a write, you invalidate after one. Redis is
   shared across processes, so a second app instance (or a restart) sees the same
   cache, and invalidation from any writer is seen by every reader.

   Every Redis operation degrades gracefully: if Redis is unreachable the reads
   fall straight through to Postgres and the writes/invalidations no-op, so the
   app keeps working off the database alone.
"""
from __future__ import annotations

import os
import pickle
from typing import Any, Dict, List, Optional

import psycopg2.pool
import streamlit as st

from src.db.trade_repository import DBConfig, TradeRepository

try:  # redis is an optional runtime dependency; reads fall back to Postgres without it
    import redis
except Exception:  # pragma: no cover - exercised only when redis isn't installed
    redis = None

# Reads are cached this long. Short enough that a second device's trades show up
# quickly; long enough to absorb the burst of reruns a single interaction causes.
READ_TTL = 60

# All trade-read keys share this prefix so invalidation can wipe them in one scan.
_KEY_PREFIX = "dbcache:"


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


# ── Redis cache-aside store ───────────────────────────────────────────────────
# Sentinel distinguishing "key absent in Redis" from "cached value that is None"
# (``performance_stats`` legitimately returns None when there are no closed
# trades, and we still want to cache that to avoid re-querying every rerun).
_MISS = object()


def _redis_settings() -> tuple[str, int]:
    """Resolve the Redis target: ``[redis]`` in secrets.toml, else REDIS_* env
    vars (the names ``src/data_backbone`` already uses), else localhost:6379."""
    host, port = "localhost", 6379
    try:
        sec = st.secrets.get("redis", {})
        host = str(sec.get("host", host))
        port = int(sec.get("port", port))
    except Exception:
        pass
    host = os.getenv("REDIS_HOST", host)
    port = int(os.getenv("REDIS_PORT", port))
    return host, port


@st.cache_resource(show_spinner=False)
def _redis_client():  # pragma: no cover - opens a live Redis socket
    """One Redis client for the app's lifetime, or None if redis is unavailable.

    Short socket timeouts so a missing/unreachable Redis fails fast and the read
    falls through to Postgres rather than hanging the rerun.
    """
    if redis is None:
        return None
    host, port = _redis_settings()
    try:
        client = redis.Redis(
            host=host, port=port,
            socket_connect_timeout=2, socket_timeout=2,
        )
        client.ping()
        return client
    except Exception:
        return None


def _safe(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def _key(name: str, host: str, port: int, dbname: str, user: str, *args: Any) -> str:
    """Cache key scoped to the DB target (so switching DBs can't cross-read) plus
    the call's distinguishing args. Password is intentionally omitted from the key."""
    parts = [name, host, str(port), dbname, user, *(str(a) for a in args)]
    return _KEY_PREFIX + ":".join(parts)


def _cache_get(key: str):
    """Return the cached value for ``key``, or :data:`_MISS` on absence/Redis-down."""
    client = _redis_client()
    if client is None:
        return _MISS
    raw = _safe(lambda: client.get(key))
    if raw is None:
        return _MISS
    return _safe(lambda: pickle.loads(raw), _MISS)


def _cache_set(key: str, value: Any, ttl: int = READ_TTL) -> None:
    """Store ``value`` under ``key`` for ``ttl`` seconds. No-ops if Redis is down."""
    client = _redis_client()
    if client is None:
        return
    _safe(lambda: client.setex(key, ttl, pickle.dumps(value)))


# ── cached reads (cache-aside; keyed on DB target + args) ─────────────────────
# The loader bodies are thin pooled passthroughs to the repository (which is
# fully unit-tested in tests/test_trade_repository.py) and need a live DB to
# execute, so the DB path is excluded from coverage; the cache-aside wiring is
# tested in tests/test_db_cache.py against a fake Redis.
def cached_load_setups(
    host: str, port: int, dbname: str, user: str, password: str, limit: int = 100
) -> List[Dict[str, Any]]:
    key = _key("load_setups", host, port, dbname, user, limit)
    hit = _cache_get(key)
    if hit is not _MISS:
        return hit
    val = pooled_repository(_cfg(host, port, dbname, user, password)).load_setups(limit=limit)  # pragma: no cover - needs live DB
    _cache_set(key, val)  # pragma: no cover - needs live DB
    return val  # pragma: no cover - needs live DB


def cached_load_open(
    host: str, port: int, dbname: str, user: str, password: str
) -> List[Dict[str, Any]]:
    key = _key("load_open", host, port, dbname, user)
    hit = _cache_get(key)
    if hit is not _MISS:
        return hit
    val = pooled_repository(_cfg(host, port, dbname, user, password)).load_open()  # pragma: no cover - needs live DB
    _cache_set(key, val)  # pragma: no cover - needs live DB
    return val  # pragma: no cover - needs live DB


def cached_daily_losses(
    host: str, port: int, dbname: str, user: str, password: str, max_losses: int = 2
) -> Dict[str, Any]:
    key = _key("daily_losses", host, port, dbname, user, max_losses)
    hit = _cache_get(key)
    if hit is not _MISS:
        return hit
    val = pooled_repository(_cfg(host, port, dbname, user, password)).daily_losses(max_losses=max_losses)  # pragma: no cover - needs live DB
    _cache_set(key, val)  # pragma: no cover - needs live DB
    return val  # pragma: no cover - needs live DB


def cached_performance_stats(
    host: str, port: int, dbname: str, user: str, password: str, n: int = 20
) -> Optional[Dict[str, Any]]:
    key = _key("performance_stats", host, port, dbname, user, n)
    hit = _cache_get(key)
    if hit is not _MISS:
        return hit
    val = pooled_repository(_cfg(host, port, dbname, user, password)).performance_stats(n=n)  # pragma: no cover - needs live DB
    _cache_set(key, val)  # pragma: no cover - needs live DB
    return val  # pragma: no cover - needs live DB


def cached_realized_pnl(
    host: str, port: int, dbname: str, user: str, password: str
) -> Dict[str, Any]:
    key = _key("realized_pnl", host, port, dbname, user)
    hit = _cache_get(key)
    if hit is not _MISS:
        return hit
    val = pooled_repository(_cfg(host, port, dbname, user, password)).realized_pnl()  # pragma: no cover - needs live DB
    _cache_set(key, val)  # pragma: no cover - needs live DB
    return val  # pragma: no cover - needs live DB


# ── app_state key/value (cache-aside read + write-through) ────────────────────
# Used for durable app-level state such as the live account balance. Reads are
# cached like the trade reads; writes go straight to Postgres and prime the cache
# with the new value so the next read is immediate.
def cached_get_state(
    host: str, port: int, dbname: str, user: str, password: str, state_key: str
) -> Optional[Any]:
    key = _key("app_state", host, port, dbname, user, state_key)
    hit = _cache_get(key)
    if hit is not _MISS:
        return hit
    val = pooled_repository(_cfg(host, port, dbname, user, password)).get_state(state_key)  # pragma: no cover - needs live DB
    _cache_set(key, val)  # pragma: no cover - needs live DB
    return val  # pragma: no cover - needs live DB


def set_state(
    host: str, port: int, dbname: str, user: str, password: str,
    state_key: str, value: Any
) -> None:
    """Persist ``value`` to Postgres and prime the Redis cache with it."""
    pooled_repository(_cfg(host, port, dbname, user, password)).set_state(state_key, value)  # pragma: no cover - needs live DB
    _cache_set(_key("app_state", host, port, dbname, user, state_key), value)  # pragma: no cover - needs live DB


# Every cached read function — kept in one place for documentation/tests.
_READ_CACHES = (
    cached_load_setups,
    cached_load_open,
    cached_daily_losses,
    cached_performance_stats,
    cached_realized_pnl,
)


def clear_read_caches() -> None:
    """Invalidate all cached reads in Redis. Call after any write so the journal,
    stats, open-trades and P/L re-query Postgres on the next read. No-ops if Redis
    is unavailable (there is nothing cached to invalidate)."""
    client = _redis_client()
    if client is None:
        return

    def _wipe() -> None:
        keys = list(client.scan_iter(match=_KEY_PREFIX + "*"))
        if keys:
            client.delete(*keys)

    _safe(_wipe)
