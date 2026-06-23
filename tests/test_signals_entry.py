"""Unit tests for the entry-signal helpers in src/core/signals.py.

Covers safe_get, EntrySignalGenerator.get_entry_signal, and
generate_supertrend_signals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core.signals import (
    EntrySignalGenerator,
    generate_supertrend_signals,
    safe_get,
)


def _ohlc(n: int, start: float = 1.10, stop: float = 1.12) -> pd.DataFrame:
    close = np.linspace(start, stop, n)
    return pd.DataFrame(
        {"Open": close, "High": close + 0.001, "Low": close - 0.001,
         "Close": close, "Volume": np.full(n, 1000.0)}
    )


class TestSafeGet:
    def test_returns_value(self):
        row = pd.Series({"Close": 1.2345})
        assert safe_get(row, "Close") == pytest.approx(1.2345)

    def test_missing_column_returns_default(self):
        row = pd.Series({"Close": 1.0})
        assert safe_get(row, "RSI", default=50.0) == 50.0

    def test_none_returns_default(self):
        row = pd.Series({"RSI": None})
        assert safe_get(row, "RSI", default=42.0) == 42.0

    def test_nan_returns_default(self):
        row = pd.Series({"RSI": np.nan})
        assert safe_get(row, "RSI", default=42.0) == 42.0

    def test_non_numeric_returns_default(self):
        row = pd.Series({"X": "not-a-number"})
        assert safe_get(row, "X", default=7.0) == 7.0


class TestGetEntrySignal:
    def test_empty_frame(self):
        res = EntrySignalGenerator.get_entry_signal(pd.DataFrame(), "Long")
        assert res["signal"] == 0
        assert res["confidence"] == 0
        assert "Insufficient" in res["reasons"][0]

    def test_too_few_rows(self):
        res = EntrySignalGenerator.get_entry_signal(_ohlc(3), "Long")
        assert res["signal"] == 0

    def test_long_bias_returns_contract(self):
        res = EntrySignalGenerator.get_entry_signal(_ohlc(120), "Long")
        assert res["signal"] in (0, 1)
        assert 0 <= res["confidence"] <= 5
        assert res["reasons"]  # always at least an "awaiting" reason
        assert "price" in res and res["price"] > 0

    def test_short_bias_returns_contract(self):
        res = EntrySignalGenerator.get_entry_signal(_ohlc(120, 1.12, 1.10), "Short")
        assert res["signal"] in (0, -1)
        assert 0 <= res["confidence"] <= 5

    def test_neutral_bias_message(self):
        res = EntrySignalGenerator.get_entry_signal(_ohlc(120), "Neutral")
        assert res["signal"] == 0
        assert "Neutral" in res["reasons"][0]

    def test_oversold_long_trigger_adds_confidence(self):
        # Sharp sell-off into the last bars drives RSI/Stoch into oversold so the
        # Long branch records at least one confluence reason.
        n = 120
        close = np.concatenate([np.linspace(1.20, 1.21, n - 20),
                                np.linspace(1.21, 1.10, 20)])
        df = pd.DataFrame({"Open": close, "High": close + 0.001,
                           "Low": close - 0.001, "Close": close,
                           "Volume": np.full(n, 1000.0)})
        res = EntrySignalGenerator.get_entry_signal(df, "Long")
        assert res["confidence"] >= 1
        assert any("RSI" in r or "Bollinger" in r or "Stochastic" in r
                   for r in res["reasons"])


class TestSupertrendSignals:
    def test_empty_frame_returns_empty_series(self):
        out = generate_supertrend_signals(pd.DataFrame())
        assert isinstance(out, pd.Series)
        assert out.empty

    def test_returns_signal_series_in_range(self):
        out = generate_supertrend_signals(_ohlc(260, 1.05, 1.30))
        assert len(out) == 260
        assert set(out.unique()).issubset({-1, 0, 1})
