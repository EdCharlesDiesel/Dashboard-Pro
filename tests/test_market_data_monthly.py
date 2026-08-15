"""Monthly bars come from the same daily series as everything else.

Pulling native ``1mo`` bars would give a *different* series from the daily one
every other timeframe is resampled from, which is how a monthly change comes to
disagree with the weeks inside it.

The spine imports ``cached_ohlc`` *inside* each function, so these patch it at
its source module (``src.db.market_cache``). Patching the attribute on
``market_data`` would set something the function never reads, and every test
here would pass against the real network.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.services import market_data as md


def _daily(days: int = 400, start: str = "2025-01-01") -> pd.DataFrame:
    idx = pd.date_range(start=start, periods=days, freq="D")
    return pd.DataFrame({"Open": range(days), "High": range(days),
                         "Low": range(days), "Close": range(days),
                         "Volume": [1] * days}, index=idx)


class TestMonthlyOhlc:
    def test_resamples_from_daily_not_native_monthly(self, monkeypatch):
        seen: dict = {}

        def fake(ticker, **kw):
            seen.update(kw)
            return _daily()

        monkeypatch.setattr("src.db.market_cache.cached_ohlc", fake)
        md.monthly_ohlc("EURUSD=X")
        assert seen.get("interval") == "1d", "must fetch daily and resample"
        assert seen.get("ttl") == md.CANONICAL_TTL

    def test_returns_monthly_bars(self, monkeypatch):
        monkeypatch.setattr("src.db.market_cache.cached_ohlc",
                            lambda t, **kw: _daily())
        out = md.monthly_ohlc("EURUSD=X")
        assert 12 <= len(out) <= 14          # ~13 months in 400 days
        assert list(out.columns) == ["Open", "High", "Low", "Close", "Volume"]

    def test_ohlc_aggregation_is_correct(self, monkeypatch):
        monkeypatch.setattr("src.db.market_cache.cached_ohlc",
                            lambda t, **kw: _daily(days=62, start="2025-01-01"))
        january = md.monthly_ohlc("EURUSD=X").iloc[0]
        assert january["Open"] == 0          # first daily open of the month
        assert january["Close"] == 30        # last daily close of January
        assert january["High"] == 30         # max over the month
        assert january["Low"] == 0           # min over the month

    def test_the_in_progress_month_is_included_as_a_partial_bar(self, monkeypatch):
        # 32 days from 1 Jan = all of January plus one day of February.
        monkeypatch.setattr("src.db.market_cache.cached_ohlc",
                            lambda t, **kw: _daily(days=32, start="2025-01-01"))
        out = md.monthly_ohlc("EURUSD=X")
        assert len(out) == 2
        assert out.iloc[-1]["Close"] == 31   # the single February day

    def test_empty_input_degrades_to_empty(self, monkeypatch):
        monkeypatch.setattr("src.db.market_cache.cached_ohlc",
                            lambda t, **kw: pd.DataFrame())
        assert md.monthly_ohlc("EURUSD=X").empty

    def test_failure_never_raises(self, monkeypatch):
        def boom(t, **kw):
            raise RuntimeError("network")

        monkeypatch.setattr("src.db.market_cache.cached_ohlc", boom)
        assert md.monthly_ohlc("EURUSD=X").empty

    def test_the_period_is_widenable_through_the_argument(self, monkeypatch):
        seen: dict = {}

        def fake(ticker, **kw):
            seen.update(kw)
            return _daily()

        monkeypatch.setattr("src.db.market_cache.cached_ohlc", fake)
        md.monthly_ohlc("EURUSD=X", period="10y")
        assert seen.get("period") == "10y"


class TestItMatchesTheWeeklyConvention:
    def test_both_resample_from_the_same_daily_fetch(self, monkeypatch):
        """Weekly and monthly must agree on the underlying series.

        If one pulled native bars and the other resampled, a month's change and
        the weeks inside it would come from different data.
        """
        calls: list = []

        def fake(ticker, **kw):
            calls.append(kw.get("interval"))
            return _daily()

        monkeypatch.setattr("src.db.market_cache.cached_ohlc", fake)
        md.weekly_ohlc("EURUSD=X")
        md.monthly_ohlc("EURUSD=X")
        assert calls == ["1d", "1d"]
