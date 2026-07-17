"""Unit tests for src/services/market_data.py — the canonical data spine.

cached_ohlc is monkeypatched at its source module so no network/DB is touched.
"""
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

import src.db.market_cache as market_cache
from src.services import market_data


def _bars(n: int, freq: str = "D") -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=n, freq=freq)
    close = np.linspace(100.0, 110.0, n)
    return pd.DataFrame(
        {
            "Open": close,
            "High": close + 0.5,
            "Low": close - 0.5,
            "Close": close,
            "Volume": np.full(n, 1000.0),
        },
        index=idx,
    )


@pytest.fixture
def capture(monkeypatch):
    """Patch cached_ohlc; records the kwargs of every call."""
    calls = []

    def fake(symbol, **kwargs):
        calls.append({"symbol": symbol, **kwargs})
        freq = "h" if kwargs.get("interval") == "1h" else "D"
        n = 48 if freq == "h" else 60
        return _bars(n, freq=freq)

    monkeypatch.setattr(market_cache, "cached_ohlc", fake)
    return calls


class TestCanonicalParams:
    def test_daily_uses_canonical_window(self, capture):
        df = market_data.daily_ohlc("EURUSD=X")
        assert not df.empty
        call = capture[0]
        assert call["period"] == market_data.DAILY_PERIOD
        assert call["interval"] == "1d"
        assert call["ttl"] == market_data.CANONICAL_TTL

    def test_weekly_resamples_daily_bars(self, capture):
        df = market_data.weekly_ohlc("EURUSD=X")
        call = capture[0]
        assert call["period"] == market_data.WEEKLY_PERIOD
        assert call["interval"] == "1d"
        # 60 daily bars → ~9 weekly buckets
        assert 7 <= len(df) <= 10
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]

    def test_h4_resamples_hourly_bars(self, capture):
        df = market_data.h4_ohlc("EURUSD=X")
        call = capture[0]
        assert call["interval"] == "1h"
        assert isinstance(call["start"], datetime)
        # 48 hourly bars → 12 4H buckets
        assert len(df) == 12

    def test_hourly_window(self, capture):
        df = market_data.hourly_ohlc("EURUSD=X")
        call = capture[0]
        assert call["interval"] == "1h"
        assert len(df) == 48

    def test_page_may_widen_period_but_not_interval(self, capture):
        market_data.daily_ohlc("EURUSD=X", period="5y")
        call = capture[0]
        assert call["period"] == "5y"
        assert call["interval"] == "1d"
        assert call["ttl"] == market_data.CANONICAL_TTL


class TestGracefulDegrade:
    def test_fetch_error_returns_empty(self, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("network down")

        monkeypatch.setattr(market_cache, "cached_ohlc", boom)
        for fn in (market_data.daily_ohlc, market_data.weekly_ohlc,
                   market_data.h4_ohlc, market_data.hourly_ohlc):
            out = fn("EURUSD=X")
            assert isinstance(out, pd.DataFrame) and out.empty

    def test_empty_fetch_returns_empty(self, monkeypatch):
        monkeypatch.setattr(market_cache, "cached_ohlc",
                            lambda *a, **k: pd.DataFrame())
        assert market_data.weekly_ohlc("EURUSD=X").empty

    def test_missing_close_column_returns_empty(self, monkeypatch):
        junk = pd.DataFrame({"Foo": [1, 2, 3]})
        monkeypatch.setattr(market_cache, "cached_ohlc", lambda *a, **k: junk)
        assert market_data.daily_ohlc("EURUSD=X").empty


class TestAsof:
    def test_data_asof_takes_freshest(self):
        older = _bars(10)
        newer = _bars(20)
        assert market_data.data_asof(older, newer) == newer.index[-1]

    def test_data_asof_none_when_no_frames(self):
        assert market_data.data_asof(pd.DataFrame(), None) is None

    def test_asof_caption(self):
        df = _bars(5)
        cap = market_data.asof_caption(df)
        assert cap.startswith("data as of 2025-01-05")
        assert cap.endswith("UTC")
        assert market_data.asof_caption(pd.DataFrame()) == "no data"
