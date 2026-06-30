"""Unit tests for src/db/market_data_repository.py.

No live Postgres — a fake connection (injected via ``connect_factory``) records
every SQL string and parameter set, so we can assert exactly what is written and
read on the OHLC + blob paths.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from src.db.market_data_repository import MarketDataRepository, _index_name
from src.db.trade_repository import DBConfig


# ── fakes ─────────────────────────────────────────────────────────────────────
class FakeCursor:
    def __init__(self, fetchall_rows=None, fetchone_row=None):
        self.executed = []          # [(sql, params), ...]
        self._fetchall = fetchall_rows if fetchall_rows is not None else []
        self._fetchone = fetchone_row

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def fetchall(self):
        return self._fetchall

    def fetchone(self):
        return self._fetchone

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor
        self.committed = False
        self.closed = False

    def cursor(self, *args, **kwargs):
        return self._cursor

    def commit(self):
        self.committed = True

    def rollback(self):
        pass

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _repo(cursor):
    conn = FakeConnection(cursor)
    return MarketDataRepository(DBConfig(), connect_factory=lambda: conn), conn


def _ohlc_frame():
    idx = pd.to_datetime(["2026-06-20", "2026-06-21", "2026-06-22"])
    return pd.DataFrame(
        {
            "Open": [1.10, 1.11, 1.12], "High": [1.12, 1.13, 1.14],
            "Low": [1.09, 1.10, 1.11], "Close": [1.11, 1.12, 1.13],
            "Volume": [0.0, 0.0, 0.0],
        },
        index=idx,
    )


# ── schema ────────────────────────────────────────────────────────────────────
class TestSchema:
    def test_init_schema_creates_all_three_tables(self):
        cur = FakeCursor()
        repo, conn = _repo(cur)
        ok, msg = repo.init_schema()
        assert ok is True and msg == "Connected"
        sql = " ".join(s for s, _ in cur.executed)
        assert "CREATE TABLE IF NOT EXISTS market_bars" in sql
        assert "CREATE TABLE IF NOT EXISTS market_fetch_meta" in sql
        assert "CREATE TABLE IF NOT EXISTS market_blobs" in sql

    def test_init_schema_reports_failure(self):
        repo = MarketDataRepository(DBConfig(), connect_factory=lambda: (_ for _ in ()).throw(RuntimeError("refused")))
        ok, msg = repo.init_schema()
        assert ok is False and "refused" in msg


# ── OHLC writes ────────────────────────────────────────────────────────────────
class TestUpsertBars:
    def test_upserts_bars_and_stamps_meta(self, monkeypatch):
        captured = {}

        def fake_execute_batch(cur, sql, payload):
            captured["sql"] = sql
            captured["payload"] = payload

        monkeypatch.setattr(
            "src.db.market_data_repository.psycopg2.extras.execute_batch",
            fake_execute_batch,
        )
        cur = FakeCursor()
        repo, conn = _repo(cur)
        n = repo.upsert_bars("EURUSD=X", "1d", _ohlc_frame())

        assert n == 3
        assert "ON CONFLICT (symbol, interval, ts) DO UPDATE" in captured["sql"]
        assert len(captured["payload"]) == 3
        first = captured["payload"][0]
        assert first["symbol"] == "EURUSD=X" and first["interval"] == "1d"
        assert first["open"] == pytest.approx(1.10)
        assert isinstance(first["ts"], datetime)
        # meta upsert ran on the same connection + committed
        meta_sql = cur.executed[-1][0]
        assert "INSERT INTO market_fetch_meta" in meta_sql
        assert cur.executed[-1][1] == ("EURUSD=X", "1d", 3)
        assert conn.committed is True

    def test_empty_frame_stamps_meta_zero_no_batch(self, monkeypatch):
        called = {"batch": False}
        monkeypatch.setattr(
            "src.db.market_data_repository.psycopg2.extras.execute_batch",
            lambda *a, **k: called.__setitem__("batch", True),
        )
        cur = FakeCursor()
        repo, conn = _repo(cur)
        n = repo.upsert_bars("EURUSD=X", "1d", pd.DataFrame())
        assert n == 0
        assert called["batch"] is False           # no bar insert attempted
        assert "market_fetch_meta" in cur.executed[-1][0]  # but meta still stamped
        assert cur.executed[-1][1] == ("EURUSD=X", "1d", 0)

    def test_nan_volume_coerced_to_none(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            "src.db.market_data_repository.psycopg2.extras.execute_batch",
            lambda cur, sql, payload: captured.__setitem__("payload", payload),
        )
        df = _ohlc_frame()
        df.loc[df.index[0], "Volume"] = float("nan")
        repo, _ = _repo(FakeCursor())
        repo.upsert_bars("EURUSD=X", "1d", df)
        assert captured["payload"][0]["volume"] is None


# ── OHLC reads ──────────────────────────────────────────────────────────────────
class TestReads:
    def test_last_fetched_at_returns_value(self):
        ts = datetime(2026, 6, 22, 10, 0, 0)
        cur = FakeCursor(fetchone_row=(ts,))
        repo, _ = _repo(cur)
        assert repo.last_fetched_at("EURUSD=X", "1d") == ts
        sql, params = cur.executed[0]
        assert "FROM market_fetch_meta" in sql and params == ("EURUSD=X", "1d")

    def test_last_fetched_at_none_when_missing(self):
        repo, _ = _repo(FakeCursor(fetchone_row=None))
        assert repo.last_fetched_at("EURUSD=X", "1d") is None

    def test_last_fetched_at_swallows_errors(self):
        repo = MarketDataRepository(DBConfig(), connect_factory=lambda: (_ for _ in ()).throw(OSError()))
        assert repo.last_fetched_at("X", "1d") is None

    def test_load_bars_builds_frame_with_named_index(self):
        rows = [
            (datetime(2026, 6, 20), 1.10, 1.12, 1.09, 1.11, 0.0),
            (datetime(2026, 6, 21), 1.11, 1.13, 1.10, 1.12, 0.0),
        ]
        cur = FakeCursor(fetchall_rows=rows)
        repo, _ = _repo(cur)
        df = repo.load_bars("EURUSD=X", "1d")
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert df.index.name == "Date"        # daily interval
        assert len(df) == 2
        assert df["Close"].iloc[-1] == pytest.approx(1.12)

    def test_load_bars_start_filter_in_sql(self):
        cur = FakeCursor(fetchall_rows=[])
        repo, _ = _repo(cur)
        start = datetime(2026, 1, 1)
        df = repo.load_bars("EURUSD=X", "1h", start=start)
        sql, params = cur.executed[0]
        assert "ts >= %s" in sql
        assert start in params
        # empty result still has the right columns + intraday index name
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert df.index.name == "Datetime"

    def test_index_name_intraday_vs_daily(self):
        assert _index_name("5m") == "Datetime"
        assert _index_name("1h") == "Datetime"
        assert _index_name("1d") == "Date"
        assert _index_name("1wk") == "Date"


# ── blob cache ──────────────────────────────────────────────────────────────────
class TestBlobs:
    def test_set_blob_wraps_payload_as_json(self, monkeypatch):
        sentinel = object()
        monkeypatch.setattr(
            "src.db.market_data_repository.psycopg2.extras.Json",
            lambda payload: ("JSON", payload),
        )
        cur = FakeCursor()
        repo, conn = _repo(cur)
        repo.set_blob("macro:USD", {"GDP": 2.5})
        sql, params = cur.executed[0]
        assert "INSERT INTO market_blobs" in sql
        assert params[0] == "macro:USD"
        assert params[1] == ("JSON", {"GDP": 2.5})
        assert conn.committed is True

    def test_get_blob_returns_payload(self):
        cur = FakeCursor(fetchone_row=({"GDP": 2.5},))
        repo, _ = _repo(cur)
        assert repo.get_blob("macro:USD") == {"GDP": 2.5}

    def test_get_blob_none_when_missing(self):
        repo, _ = _repo(FakeCursor(fetchone_row=None))
        assert repo.get_blob("nope") is None

    def test_blob_fetched_at_swallows_errors(self):
        repo = MarketDataRepository(DBConfig(), connect_factory=lambda: (_ for _ in ()).throw(OSError()))
        assert repo.blob_fetched_at("k") is None
