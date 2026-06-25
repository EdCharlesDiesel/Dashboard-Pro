"""Unit tests for src/db/cache.py — pooling proxy + read-cache invalidation."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.db import cache
from src.db.trade_repository import DBConfig


class FakePool:
    def __init__(self, conn):
        self._conn = conn
        self.got = 0
        self.put = []

    def getconn(self):
        self.got += 1
        return self._conn

    def putconn(self, conn):
        self.put.append(conn)


class TestLeasedConnection:
    def test_delegates_cursor_and_txn(self):
        conn = MagicMock()
        pool = FakePool(conn)
        leased = cache._LeasedConnection(pool, conn)

        leased.cursor("arg", k=1)
        conn.cursor.assert_called_once_with("arg", k=1)

        leased.commit()
        conn.commit.assert_called_once()

        with leased:
            pass
        conn.__enter__.assert_called_once()
        conn.__exit__.assert_called_once()

    def test_close_returns_to_pool(self):
        conn = MagicMock()
        pool = FakePool(conn)
        leased = cache._LeasedConnection(pool, conn)
        leased.close()
        assert pool.put == [conn]
        conn.close.assert_not_called()  # NOT actually closed


class TestPooledRepository:
    def test_connect_borrows_and_close_returns(self, monkeypatch):
        conn = MagicMock()
        pool = FakePool(conn)
        monkeypatch.setattr(cache, "_get_pool", lambda *a, **k: pool)

        repo = cache.pooled_repository(DBConfig())
        leased = repo._connect()
        assert isinstance(leased, cache._LeasedConnection)
        assert pool.got == 1

        leased.close()
        assert pool.put == [conn]

    def test_pooled_repository_runs_a_read(self, monkeypatch):
        # Wire a fake connection so a real read borrows from the pool, runs, and
        # returns the connection — end-to-end through the pooling layer.
        cur = MagicMock()
        cur.__enter__.return_value = cur
        cur.fetchall.return_value = [{"id": 1}]
        conn = MagicMock()
        conn.cursor.return_value = cur
        conn.__enter__.return_value = conn
        pool = FakePool(conn)
        monkeypatch.setattr(cache, "_get_pool", lambda *a, **k: pool)

        repo = cache.pooled_repository(DBConfig())
        rows = repo.load_open()
        assert rows == [{"id": 1}]
        assert pool.put == [conn]  # connection returned to the pool


class TestCfgHelper:
    def test_builds_dbconfig(self):
        cfg = cache._cfg("h", 5433, "trading", "u", "p")
        assert isinstance(cfg, DBConfig)
        assert cfg.host == "h"
        assert cfg.port == 5433
        assert cfg.dbname == "trading"
        assert cfg.user == "u"
        assert cfg.password == "p"


class FakeRedis:
    """In-memory stand-in for the redis client contract cache.py uses:
    get / setex / scan_iter / delete / ping."""

    def __init__(self):
        self.store: dict = {}

    def ping(self):
        return True

    def get(self, key):
        return self.store.get(key)

    def setex(self, key, ttl, value):
        self.store[key] = value
        return True

    def scan_iter(self, match="*"):
        prefix = match.rstrip("*")
        return [k for k in list(self.store) if k.startswith(prefix)]

    def delete(self, *keys):
        for k in keys:
            self.store.pop(k, None)
        return len(keys)


class TestCacheAside:
    def test_miss_then_hit(self, monkeypatch):
        """First call misses (loader runs, value stored); second call hits Redis
        and never touches the repository."""
        fake = FakeRedis()
        monkeypatch.setattr(cache, "_redis_client", lambda: fake)

        calls = {"n": 0}

        class Repo:
            def load_open(self):
                calls["n"] += 1
                return [{"id": 1}]

        monkeypatch.setattr(cache, "pooled_repository", lambda cfg: Repo())

        first = cache.cached_load_open("h", 5432, "trading", "u", "p")
        second = cache.cached_load_open("h", 5432, "trading", "u", "p")

        assert first == [{"id": 1}] == second
        assert calls["n"] == 1  # loader ran exactly once; second call was a cache hit

    def test_caches_none_result(self, monkeypatch):
        """A legitimate None (no closed trades) is cached, not re-queried."""
        fake = FakeRedis()
        monkeypatch.setattr(cache, "_redis_client", lambda: fake)

        calls = {"n": 0}

        class Repo:
            def performance_stats(self, n=20):
                calls["n"] += 1
                return None

        monkeypatch.setattr(cache, "pooled_repository", lambda cfg: Repo())

        assert cache.cached_performance_stats("h", 5432, "trading", "u", "p") is None
        assert cache.cached_performance_stats("h", 5432, "trading", "u", "p") is None
        assert calls["n"] == 1  # None was cached; loader did not run twice

    def test_redis_down_falls_through_to_db(self, monkeypatch):
        """When Redis is unavailable every call hits the repository (no caching),
        but nothing errors."""
        monkeypatch.setattr(cache, "_redis_client", lambda: None)

        calls = {"n": 0}

        class Repo:
            def load_open(self):
                calls["n"] += 1
                return [{"id": 9}]

        monkeypatch.setattr(cache, "pooled_repository", lambda cfg: Repo())

        assert cache.cached_load_open("h", 5432, "trading", "u", "p") == [{"id": 9}]
        assert cache.cached_load_open("h", 5432, "trading", "u", "p") == [{"id": 9}]
        assert calls["n"] == 2  # no cache, so the loader runs each time


class TestRedisSettings:
    def test_env_overrides_default(self, monkeypatch):
        monkeypatch.setenv("REDIS_HOST", "redis.internal")
        monkeypatch.setenv("REDIS_PORT", "6380")
        # st.secrets access raises outside a Streamlit runtime; the helper must
        # swallow that and fall back to env vars.
        assert cache._redis_settings() == ("redis.internal", 6380)

    def test_defaults_when_unset(self, monkeypatch):
        monkeypatch.delenv("REDIS_HOST", raising=False)
        monkeypatch.delenv("REDIS_PORT", raising=False)
        assert cache._redis_settings() == ("localhost", 6379)


class TestInvalidation:
    def test_clear_read_caches_deletes_keys(self, monkeypatch):
        fake = FakeRedis()
        fake.store[cache._KEY_PREFIX + "load_open:h:5432:trading:u"] = b"x"
        fake.store["unrelated:key"] = b"y"
        monkeypatch.setattr(cache, "_redis_client", lambda: fake)

        cache.clear_read_caches()

        assert not any(k.startswith(cache._KEY_PREFIX) for k in fake.store)
        assert "unrelated:key" in fake.store  # only our prefix is wiped

    def test_clear_read_caches_no_redis_is_noop(self, monkeypatch):
        monkeypatch.setattr(cache, "_redis_client", lambda: None)
        cache.clear_read_caches()  # must not raise

    def test_read_cache_set_is_complete(self):
        names = {fn.__name__ for fn in cache._READ_CACHES}
        assert names == {
            "cached_load_setups", "cached_load_open", "cached_daily_losses",
            "cached_performance_stats", "cached_realized_pnl",
        }
