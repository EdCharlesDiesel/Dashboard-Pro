"""Unit tests for src/db/market_cache.py — the read-through contract.

No live DB or network: ``_fetch_yf`` is monkeypatched and a fake repository is
injected via the ``repo_factory`` seam. We assert the three branches that matter:
fresh → serve from DB (no fetch); stale/missing → fetch + persist; DB error →
fall back to a live fetch.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

from src.db import market_cache as mc
from src.db.trade_repository import DBConfig


# ── fakes ─────────────────────────────────────────────────────────────────────
class FakeRepo:
    def __init__(self, fetched_at=None, blob_at=None, blob=None, duka_bars=None,
                duka_raises=None):
        self._fetched_at = fetched_at
        self._blob_at = blob_at
        self._blob = blob
        self._duka_bars = duka_bars if duka_bars is not None else pd.DataFrame()
        self._duka_raises = duka_raises
        self.upserted = []
        self.set_blobs = []
        self.loaded = []
        self.duka_loaded = []

    def last_fetched_at(self, symbol, interval):
        return self._fetched_at

    def load_bars(self, symbol, interval, start=None):
        self.loaded.append((symbol, interval, start))
        return pd.DataFrame({"Close": [1.0]})

    def upsert_bars(self, symbol, interval, df):
        self.upserted.append((symbol, interval, len(df)))

    def load_dukascopy_bars(self, symbol, interval, start=None):
        self.duka_loaded.append((symbol, interval, start))
        if self._duka_raises:
            raise self._duka_raises
        return self._duka_bars

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


# ── debug mode: source/asof resolution + time-machine truncation ─────────────
class TestResolveDebugDefaults:
    def test_explicit_values_win(self):
        # No Streamlit runtime in tests, so st.session_state access raises —
        # explicit args must short-circuit that, not attempt the lookup.
        source, asof = mc._resolve_debug_defaults("duka", date(2026, 1, 1))
        assert (source, asof) == ("duka", date(2026, 1, 1))

    def test_none_falls_back_to_live_no_asof_without_runtime(self):
        # Outside a Streamlit runtime, st.session_state access raises and the
        # resolver must degrade to the safe defaults rather than propagate.
        source, asof = mc._resolve_debug_defaults(None, None)
        assert source == "live"
        assert asof is None


class TestApplyAsof:
    def _frame(self):
        idx = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"])
        return pd.DataFrame({"Close": [1.0, 2.0, 3.0]}, index=idx)

    def test_none_asof_is_noop(self):
        df = self._frame()
        out = mc._apply_asof(df, None)
        assert len(out) == 3

    def test_truncates_to_asof_inclusive(self):
        out = mc._apply_asof(self._frame(), date(2026, 1, 2))
        assert list(out["Close"]) == [1.0, 2.0]

    def test_asof_before_all_data_empties_frame(self):
        out = mc._apply_asof(self._frame(), date(2025, 1, 1))
        assert out.empty

    def test_empty_frame_is_noop(self):
        out = mc._apply_asof(pd.DataFrame(), date(2026, 1, 1))
        assert out.empty

    def test_tz_aware_index_handled(self):
        idx = pd.to_datetime(["2026-01-01", "2026-01-02"]).tz_localize("UTC")
        df = pd.DataFrame({"Close": [1.0, 2.0]}, index=idx)
        out = mc._apply_asof(df, date(2026, 1, 1))
        assert len(out) == 1


class TestOhlcImplDuka:
    def test_no_cfg_falls_back_to_live(self, monkeypatch):
        monkeypatch.setattr(mc, "_resolve_cfg", lambda: None)
        calls = _track_fetch(monkeypatch)
        out = mc._ohlc_impl("EURUSD=X", "60d", "1d", None, None, 300, True,
                            source="duka")
        assert calls["n"] == 1                 # fell back to live fetch
        assert out["Close"].iloc[0] == 9.9

    def test_archived_bars_served_no_live_fetch(self, cfg_set, monkeypatch):
        calls = _track_fetch(monkeypatch)
        archived = pd.DataFrame({"Close": [1.2345]})
        repo = FakeRepo(duka_bars=archived)
        out = mc._ohlc_impl("EURUSD=X", "60d", "1d", None, None, 300, True,
                            repo_factory=lambda cfg: repo, source="duka")
        assert calls["n"] == 0                 # no live fetch — served from archive
        # "60d" period resolves to a ~60-days-ago cutoff via _window_start,
        # same trimming _ohlc_impl's live path already applies to load_bars.
        loaded_symbol, loaded_interval, loaded_start = repo.duka_loaded[0]
        assert (loaded_symbol, loaded_interval) == ("EURUSD=X", "1d")
        assert loaded_start is not None
        assert out["Close"].iloc[0] == pytest.approx(1.2345)

    def test_empty_archive_falls_back_to_live(self, cfg_set, monkeypatch):
        calls = _track_fetch(monkeypatch)
        repo = FakeRepo(duka_bars=pd.DataFrame())  # symbol never backfilled
        out = mc._ohlc_impl("GBPZAR=X", "60d", "1d", None, None, 300, True,
                            repo_factory=lambda cfg: repo, source="duka")
        assert calls["n"] == 1
        assert out["Close"].iloc[0] == 9.9

    def test_duka_read_error_falls_back_to_live(self, cfg_set, monkeypatch):
        calls = _track_fetch(monkeypatch)
        repo = FakeRepo(duka_raises=RuntimeError("table missing"))
        out = mc._ohlc_impl("EURUSD=X", "60d", "1d", None, None, 300, True,
                            repo_factory=lambda cfg: repo, source="duka")
        assert calls["n"] == 1
        assert out["Close"].iloc[0] == 9.9

    def test_falls_back_still_goes_through_normal_live_cache_path(self, cfg_set, monkeypatch):
        # A duka miss shouldn't just live-fetch and discard — it should still
        # hit the same fresh-from-DB / upsert-on-stale contract as source="live".
        calls = _track_fetch(monkeypatch)
        repo = FakeRepo(duka_bars=pd.DataFrame(),
                        fetched_at=datetime.utcnow() - timedelta(seconds=5))
        out = mc._ohlc_impl("EURUSD=X", "60d", "1d", None, None, 300, True,
                            repo_factory=lambda cfg: repo, source="duka")
        assert calls["n"] == 0                 # fresh market_bars served instead
        assert repo.loaded                     # went through the live-cache path
        assert out["Close"].iloc[0] == 1.0     # came from load_bars, not live fetch
