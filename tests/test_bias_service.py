"""Unit tests for src/services/bias_service.py — spine→engine wiring.

The spine fetchers are monkeypatched in the service's namespace, so no
network/DB/Streamlit-runtime is touched. Distinct pair/ticker args are used
per test because st.cache_data may memoize across calls even in bare mode.
"""
import numpy as np
import pandas as pd

from src.core.bias import BEARISH, BULLISH
from src.services import bias_service


def _ohlc(n: int = 120, slope: float = 0.5, start: float = 100.0) -> pd.DataFrame:
    close = np.arange(n, dtype=float) * slope + start
    idx = pd.date_range("2025-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "Open": close, "High": close + 0.5, "Low": close - 0.5,
            "Close": close, "Volume": np.full(n, 1000.0),
        },
        index=idx,
    )


def test_get_house_view_unknown_pair_is_none():
    assert bias_service.get_house_view("NOT/APAIR") is None


def test_get_house_view_combines_spine_timeframes(monkeypatch):
    up = _ohlc(slope=0.5)
    monkeypatch.setattr(bias_service, "weekly_ohlc", lambda t: up)
    monkeypatch.setattr(bias_service, "daily_ohlc", lambda t: up)
    monkeypatch.setattr(bias_service, "h4_ohlc", lambda t: up)

    view = bias_service.get_house_view("EUR/USD")
    assert view is not None
    assert view.pair == "EUR/USD"
    assert view.direction == BULLISH
    assert view.aligned
    assert {tf.timeframe for tf in view.timeframes} == {"Weekly", "Daily", "4H"}


def test_get_house_view_bearish_and_partial_data(monkeypatch):
    dn = _ohlc(slope=-0.5, start=200.0)
    empty = pd.DataFrame()
    monkeypatch.setattr(bias_service, "weekly_ohlc", lambda t: dn)
    monkeypatch.setattr(bias_service, "daily_ohlc", lambda t: dn)
    monkeypatch.setattr(bias_service, "h4_ohlc", lambda t: empty)

    view = bias_service.get_house_view("GBP/USD")
    assert view is not None
    assert view.direction == BEARISH
    assert view.timeframe("4H") is None


def test_get_house_view_swallows_fetch_errors(monkeypatch):
    def boom(t):
        raise RuntimeError("network down")

    monkeypatch.setattr(bias_service, "weekly_ohlc", boom)
    monkeypatch.setattr(bias_service, "daily_ohlc", boom)
    monkeypatch.setattr(bias_service, "h4_ohlc", boom)
    # house_view is never reached — the service returns None, page renders nothing.
    assert bias_service.get_house_view("USD/JPY") is None
