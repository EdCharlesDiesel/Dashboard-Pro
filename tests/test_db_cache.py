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


class TestInvalidation:
    def test_all_reads_are_cache_wrapped(self):
        for fn in cache._READ_CACHES:
            assert hasattr(fn, "clear"), f"{fn} is not @st.cache_data wrapped"

    def test_clear_read_caches_runs(self):
        # Should clear every read cache without error.
        cache.clear_read_caches()

    def test_read_cache_set_is_complete(self):
        names = {fn.__name__ for fn in cache._READ_CACHES}
        assert names == {
            "cached_load_setups", "cached_load_open", "cached_daily_losses",
            "cached_performance_stats", "cached_realized_pnl",
        }
