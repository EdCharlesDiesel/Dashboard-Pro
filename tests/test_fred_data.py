"""The canonical macro spine: one definition of a FRED series for every page.

`src/services/fred_data.py` is to macro data what `market_data.py` is to OHLC.
Postgres (`fred_series`, filled by the data_backbone worker) is the source;
HTTP is the fallback, and whatever it fetches is written back so the next
reader is served locally.

Every test here stubs the two IO seams (`_read_stored`, `_fetch_remote`), so
none of this touches Postgres or the network.
"""
from __future__ import annotations

import pandas as pd

from src.services import fred_data


def _series(dates, values):
    return pd.Series(values, index=pd.to_datetime(dates))


class TestReadPath:
    def test_serves_from_postgres_without_touching_the_network(self, monkeypatch):
        stored = _series(["2026-01-01", "2026-01-02"], [1.0, 2.0])
        monkeypatch.setattr(fred_data, "_read_stored", lambda sid: stored)

        def boom(sid):
            raise AssertionError("network hit despite stored data being present")

        monkeypatch.setattr(fred_data, "_fetch_remote", boom)
        out = fred_data.fred_series("DFF")
        assert list(out.values) == [1.0, 2.0]

    def test_falls_back_to_http_and_writes_back(self, monkeypatch):
        fetched = _series(["2026-01-03"], [3.0])
        written = {}
        monkeypatch.setattr(fred_data, "_read_stored", lambda sid: pd.Series(dtype=float))
        monkeypatch.setattr(fred_data, "_fetch_remote", lambda sid: fetched)
        monkeypatch.setattr(fred_data, "_write_back",
                            lambda sid, s: written.setdefault(sid, s))
        out = fred_data.fred_series("DFF")
        assert list(out.values) == [3.0]
        assert "DFF" in written, "a remote fetch must be persisted for the next reader"

    def test_a_failed_write_back_still_serves_the_caller(self, monkeypatch):
        # Caching is a bonus; answering the question is the job.
        monkeypatch.setattr(fred_data, "_read_stored", lambda sid: pd.Series(dtype=float))
        monkeypatch.setattr(fred_data, "_fetch_remote",
                            lambda sid: _series(["2026-01-03"], [3.0]))

        def boom(sid, s):
            raise RuntimeError("db read-only")

        monkeypatch.setattr(fred_data, "_write_back", boom)
        assert list(fred_data.fred_series("DFF").values) == [3.0]


class TestShape:
    def test_index_is_naive(self, monkeypatch):
        aware = _series(["2026-01-01"], [1.0]).tz_localize("UTC")
        monkeypatch.setattr(fred_data, "_read_stored", lambda sid: aware)
        assert fred_data.fred_series("DFF").index.tz is None

    def test_start_trims_the_left_edge_inclusively(self, monkeypatch):
        stored = _series(["2026-01-01", "2026-01-02", "2026-01-03"], [1.0, 2.0, 3.0])
        monkeypatch.setattr(fred_data, "_read_stored", lambda sid: stored)
        out = fred_data.fred_series("DFF", start="2026-01-02")
        assert list(out.values) == [2.0, 3.0], "start is inclusive, matching the OHLC spine"

    def test_no_start_returns_the_whole_series(self, monkeypatch):
        stored = _series(["2026-01-01", "2026-01-02"], [1.0, 2.0])
        monkeypatch.setattr(fred_data, "_read_stored", lambda sid: stored)
        assert len(fred_data.fred_series("DFF")) == 2


class TestFailureModes:
    def test_a_dead_backend_returns_empty_rather_than_raising(self, monkeypatch):
        # A page must degrade to "no data", never to a traceback.
        def boom(*a, **k):
            raise RuntimeError("db down")

        monkeypatch.setattr(fred_data, "_read_stored", boom)
        monkeypatch.setattr(fred_data, "_fetch_remote", boom)
        out = fred_data.fred_series("DFF")
        assert isinstance(out, pd.Series) and out.empty

    def test_empty_everywhere_is_an_empty_series_not_none(self, monkeypatch):
        monkeypatch.setattr(fred_data, "_read_stored", lambda sid: pd.Series(dtype=float))
        monkeypatch.setattr(fred_data, "_fetch_remote", lambda sid: pd.Series(dtype=float))
        out = fred_data.fred_series("DFF")
        assert isinstance(out, pd.Series) and out.empty

    def test_a_none_from_a_seam_is_tolerated(self, monkeypatch):
        # db.read_fred is defensive but a future backend might hand back None.
        monkeypatch.setattr(fred_data, "_read_stored", lambda sid: None)
        monkeypatch.setattr(fred_data, "_fetch_remote", lambda sid: None)
        assert fred_data.fred_series("DFF").empty
