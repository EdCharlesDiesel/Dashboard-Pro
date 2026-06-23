"""Unit tests for src/db/market_cache.py — the read-through contract.

No live DB or network: ``_fetch_yf`` is monkeypatched and a fake repository is
injected via the ``repo_factory`` seam. We assert the three branches that matter:
fresh → serve from DB (no fetch); stale/missing → fetch + persist; DB error →
fall back to a live fetch.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.db import market_cache as mc
from src.db.trade_repository import DBConfig


# ── fakes ─────────────────────────────────────────────────────────────────────
class FakeRepo:
    def __init__(self, fetched_at=None, blob_at=None, blob=None):
        self._fetched_at = fetched_at
        self._blob_at = blob_at
        self._blob = blob
        self.upserted = []
        self.set_blobs = []
        self.loaded = []

    def last_fetched_at(self, symbol, interval):
        return self._fetched_at

    def load_bars(self, symbol, interval, start=None):
        self.loaded.append((symbol, interval, start))
        return pd.DataFrame({"Close": [1.0]})

    def upsert_bars(self, symbol, interval, df):
        self.upserted.append((symbol, interval, len(df)))

    def blob_fetched_at(self, key):
        return self._blob_at

    def get_blob(self, key):
        return self._blob

    def set_blob(self, key, payload):
        self.set_blobs.append((key, payload))


@pytest.fixture
def cfg_set(monkeypatch):
    """Make _resolve_cfg return a usable config (password present)."""
    monkeypatch.setattr(mc.secrets, "db_config", lambda: {
        "host": "h", "port": 5432, "dbname": "d", "user": "u", "password": "secret",
    })


def _track_fetch(monkeypatch):
    calls = {"n": 0}

    def fake(symbol, period, interval, start, end, auto_adjust):
        calls["n"] += 1
        return pd.DataFrame({"Close": [9.9]})

    monkeypatch.setattr(mc, "_fetch_yf", fake)
    return calls


# ── _resolve_cfg ───────────────────────────────────────────────────────────────
class TestResolveCfg:
    def test_returns_cfg_when_password_set(self, cfg_set):
        cfg = mc._resolve_cfg()
        assert isinstance(cfg, DBConfig) and cfg.password == "secret"

    def test_none_when_no_password(self, monkeypatch):
        monkeypatch.setattr(mc.secrets, "db_config", lambda: {
            "host": "h", "port": 5432, "dbname": "d", "user": "u", "password": "",
        })
        assert mc._resolve_cfg() is None

    def test_none_when_db_config_raises(self, monkeypatch):
        monkeypatch.setattr(mc.secrets, "db_config", lambda: (_ for _ in ()).throw(RuntimeError()))
        assert mc._resolve_cfg() is None


# ── _window_start ────────────────────────────────────────────────────────────
class TestWindowStart:
    def test_start_wins(self):
        s = datetime(2026, 1, 1)
        assert mc._window_start("2y", s) == s

    def test_tz_aware_start_made_naive_utc(self):
        s = datetime(2026, 1, 1, tzinfo=timezone.utc)
        out = mc._window_start(None, s)
        assert out.tzinfo is None

    @pytest.mark.parametrize("period,days", [("60d", 60), ("2y", 730), ("1mo", 30)])
    def test_period_to_cutoff(self, period, days):
        out = mc._window_start(period, None)
        delta = datetime.utcnow() - out
        assert abs(delta.days - days) <= 1

    def test_unparseable_or_max_returns_none(self):
        assert mc._window_start("max", None) is None
        assert mc._window_start("ytd", None) is None
        assert mc._window_start(None, None) is None


# ── _is_fresh ────────────────────────────────────────────────────────────────
class TestIsFresh:
    def test_none_is_stale(self):
        assert mc._is_fresh(None, 300) is False

    def test_recent_is_fresh(self):
        assert mc._is_fresh(datetime.utcnow() - timedelta(seconds=10), 300) is True

    def test_old_is_stale(self):
        assert mc._is_fresh(datetime.utcnow() - timedelta(seconds=600), 300) is False

    def test_tz_aware_handled(self):
        recent = datetime.now(timezone.utc) - timedelta(seconds=5)
        assert mc._is_fresh(recent, 300) is True


# ── _fetch_yf ────────────────────────────────────────────────────────────────
class TestFetchYf:
    def test_flattens_multiindex_and_selects_ohlcv(self, monkeypatch):
        idx = pd.to_datetime(["2026-06-20", "2026-06-21"])
        cols = pd.MultiIndex.from_product([["Open", "High", "Low", "Close", "Volume"], ["EURUSD=X"]])
        raw = pd.DataFrame(1.0, index=idx, columns=cols)
        monkeypatch.setattr(mc.yf, "download", lambda *a, **k: raw)
        out = mc._fetch_yf("EURUSD=X", "60d", "1d", None, None, True)
        assert list(out.columns) == ["Open", "High", "Low", "Close", "Volume"]

    def test_empty_returns_empty_ohlc_frame(self, monkeypatch):
        monkeypatch.setattr(mc.yf, "download", lambda *a, **k: pd.DataFrame())
        out = mc._fetch_yf("EURUSD=X", "60d", "1d", None, None, True)
        assert out.empty and list(out.columns) == ["Open", "High", "Low", "Close", "Volume"]


# ── _ohlc_impl read-through contract ──────────────────────────────────────────
class TestOhlcImpl:
    def test_no_cfg_fetches_live(self, monkeypatch):
        monkeypatch.setattr(mc, "_resolve_cfg", lambda: None)
        calls = _track_fetch(monkeypatch)
        mc._ohlc_impl("EURUSD=X", "60d", "1d", None, None, 300, True)
        assert calls["n"] == 1

    def test_fresh_serves_from_db_no_fetch(self, cfg_set, monkeypatch):
        calls = _track_fetch(monkeypatch)
        repo = FakeRepo(fetched_at=datetime.utcnow() - timedelta(seconds=5))
        out = mc._ohlc_impl("EURUSD=X", "60d", "1d", None, None, 300, True,
                            repo_factory=lambda cfg: repo)
        assert calls["n"] == 0                 # no live fetch
        assert repo.loaded and out["Close"].iloc[0] == 1.0  # came from load_bars

    def test_stale_fetches_and_upserts(self, cfg_set, monkeypatch):
        calls = _track_fetch(monkeypatch)
        repo = FakeRepo(fetched_at=datetime.utcnow() - timedelta(seconds=9999))
        out = mc._ohlc_impl("EURUSD=X", "60d", "1d", None, None, 300, True,
                            repo_factory=lambda cfg: repo)
        assert calls["n"] == 1
        assert repo.upserted == [("EURUSD=X", "1d", 1)]
        assert out["Close"].iloc[0] == 9.9     # came from _fetch_yf

    def test_missing_meta_fetches(self, cfg_set, monkeypatch):
        calls = _track_fetch(monkeypatch)
        repo = FakeRepo(fetched_at=None)
        mc._ohlc_impl("EURUSD=X", "60d", "1d", None, None, 300, True,
                      repo_factory=lambda cfg: repo)
        assert calls["n"] == 1 and repo.upserted

    def test_db_error_falls_back_to_fetch(self, cfg_set, monkeypatch):
        calls = _track_fetch(monkeypatch)

        def boom(cfg):
            raise RuntimeError("pool exhausted")

        out = mc._ohlc_impl("EURUSD=X", "60d", "1d", None, None, 300, True,
                            repo_factory=boom)
        assert calls["n"] == 1 and out["Close"].iloc[0] == 9.9


# ── _blob_impl read-through contract ──────────────────────────────────────────
class TestBlobImpl:
    def test_no_cfg_calls_fetch_fn(self, monkeypatch):
        monkeypatch.setattr(mc, "_resolve_cfg", lambda: None)
        out = mc._blob_impl("k", 300, lambda: {"v": 1})
        assert out == {"v": 1}

    def test_fresh_serves_blob_no_fetch(self, cfg_set):
        calls = {"n": 0}

        def fetch():
            calls["n"] += 1
            return {"fresh": False}

        repo = FakeRepo(blob_at=datetime.utcnow() - timedelta(seconds=5), blob={"fresh": True})
        out = mc._blob_impl("k", 300, fetch, repo_factory=lambda cfg: repo)
        assert calls["n"] == 0 and out == {"fresh": True}

    def test_stale_fetches_and_persists(self, cfg_set):
        repo = FakeRepo(blob_at=datetime.utcnow() - timedelta(seconds=9999))
        out = mc._blob_impl("k", 300, lambda: {"v": 2}, repo_factory=lambda cfg: repo)
        assert out == {"v": 2}
        assert repo.set_blobs == [("k", {"v": 2})]

    def test_db_error_falls_back(self, cfg_set):
        def boom(cfg):
            raise RuntimeError()

        out = mc._blob_impl("k", 300, lambda: {"v": 3}, repo_factory=boom)
        assert out == {"v": 3}


# ── invalidation ───────────────────────────────────────────────────────────────
class TestInvalidation:
    def test_all_caches_are_wrapped(self):
        for fn in mc._MARKET_CACHES:
            assert hasattr(fn, "clear")

    def test_clear_runs(self):
        mc.clear_market_caches()
