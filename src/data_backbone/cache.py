"""
cache.py — Redis cache layer.

Stores any picklable object (DataFrame, Series) under a TTL. Every operation
degrades gracefully: if Redis is unreachable, reads return None and writes
no-op, so the app keeps working straight off Postgres/yfinance.
"""
from __future__ import annotations

import pickle

try:
    import redis
except Exception:
    redis = None

from .config import REDIS

_client = None


def get_client():
    global _client
    if _client is None and redis is not None:
        _client = redis.Redis(host=REDIS["host"], port=REDIS["port"],
                              socket_connect_timeout=2, socket_timeout=2)
    return _client


def _safe(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def get(key: str):
    c = get_client()
    if c is None:
        return None
    raw = _safe(lambda: c.get(key))
    return pickle.loads(raw) if raw else None


def set(key: str, value, ttl: int = 3600) -> bool:
    c = get_client()
    if c is None:
        return False
    return bool(_safe(lambda: c.setex(key, ttl, pickle.dumps(value)), False))


def delete(key: str) -> None:
    c = get_client()
    if c is None:
        return
    _safe(lambda: c.delete(key))


def cached(key: str, ttl: int, loader):
    """Return cached value, else call loader(), cache it, and return it."""
    hit = get(key)
    if hit is not None:
        return hit
    value = loader()
    if value is not None:
        set(key, value, ttl)
    return value
