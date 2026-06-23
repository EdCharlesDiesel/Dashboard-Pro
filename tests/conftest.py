"""Shared pytest fixtures for the src/ logic suite.

These build deterministic OHLC frames so indicator/scoring tests never touch
yfinance or a Streamlit runtime.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _ohlc_from_close(close: "list[float] | np.ndarray") -> pd.DataFrame:
    """Wrap a close-price array in a minimal OHLCV frame.

    High/Low are nudged off Close by a small band so true-range based
    indicators (ATR/ADX) have non-degenerate inputs.
    """
    close = np.asarray(close, dtype=float)
    high = close + 0.5
    low = close - 0.5
    return pd.DataFrame(
        {
            "Open": close,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": np.full(len(close), 1000.0),
        }
    )


@pytest.fixture
def ohlc_from_close():
    """Factory fixture: callable that turns a close array into an OHLCV frame."""
    return _ohlc_from_close


@pytest.fixture
def uptrend_frame():
    """A long, steadily rising series — enough bars for EMA200-based logic."""
    return _ohlc_from_close(np.linspace(100.0, 200.0, 260))


@pytest.fixture
def rising_zigzag():
    """Zigzag with both swing highs and swing lows trending up (BULLISH)."""
    hi, lo, base = [], [], 0.0
    for _ in range(5):
        hi += [base + 1, base + 3, base + 5, base + 3, base + 1]
        lo += [base + 0.5, base + 1.5, base + 2.5, base + 1.5, base + 0.5]
        base += 2
    return pd.DataFrame({"High": hi, "Low": lo, "Close": hi, "Open": hi,
                         "Volume": [1000.0] * len(hi)})


@pytest.fixture
def falling_zigzag():
    """Zigzag with both swing highs and swing lows trending down (BEARISH)."""
    hi, lo, base = [], [], 20.0
    for _ in range(5):
        hi += [base + 1, base + 3, base + 5, base + 3, base + 1]
        lo += [base + 0.5, base + 1.5, base + 2.5, base + 1.5, base + 0.5]
        base -= 2
    return pd.DataFrame({"High": hi, "Low": lo, "Close": hi, "Open": hi,
                         "Volume": [1000.0] * len(hi)})
