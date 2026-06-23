"""Unit tests for src/indicators/technical.py — EMA/RSI/MACD/ADX/ATR math."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.indicators.technical import TechnicalIndicators as TI


class TestEMA:
    def test_constant_series_returns_constant(self):
        s = pd.Series([5.0] * 50)
        out = TI.ema(s, 10)
        assert np.allclose(out.values, 5.0)

    def test_span_one_returns_series_itself(self):
        # alpha=1 (span=1, adjust=False) → EMA collapses to the input.
        s = pd.Series([1.0, 4.0, 9.0, 16.0])
        assert np.allclose(TI.ema(s, 1).values, s.values)

    def test_first_value_seeds_with_first_observation(self):
        s = pd.Series([10.0, 20.0, 30.0])
        assert TI.ema(s, 5).iloc[0] == pytest.approx(10.0)

    def test_length_preserved(self):
        s = pd.Series(np.arange(100, dtype=float))
        assert len(TI.ema(s, 20)) == 100


class TestRSI:
    def test_bounded_zero_to_hundred(self):
        rng = np.random.default_rng(42)
        s = pd.Series(np.cumsum(rng.normal(size=300)) + 100)
        rsi = TI.rsi(s).dropna()
        assert ((rsi >= 0) & (rsi <= 100)).all()

    def test_downtrend_below_fifty(self):
        # Amplitude > per-step drift so there are genuine up-ticks (non-zero
        # avg_gain) — otherwise RSI is NaN rather than a low number.
        s = pd.Series(np.linspace(200, 100, 100) + 3 * np.sin(np.arange(100)))
        assert TI.rsi(s).iloc[-1] < 50

    def test_uptrend_above_fifty(self):
        s = pd.Series(np.linspace(100, 200, 100) + 3 * np.sin(np.arange(100)))
        assert TI.rsi(s).iloc[-1] > 50


class TestMACD:
    def test_returns_three_aligned_series(self):
        s = pd.Series(np.linspace(100, 120, 80))
        macd, sig, hist = TI.macd(s)
        assert len(macd) == len(sig) == len(hist) == 80
        # hist is, by definition, macd - signal.
        assert np.allclose((macd - sig).values, hist.values, equal_nan=True)

    def test_uptrend_macd_positive(self):
        s = pd.Series(np.linspace(100, 200, 120))
        macd, _, _ = TI.macd(s)
        assert macd.iloc[-1] > 0


class TestATR:
    def test_seeded_with_nan_until_period(self):
        df = _trending_ohlc(60)
        atr = TI.atr(df["High"], df["Low"], df["Close"], 14)
        # min_periods=14 → first 13 entries are NaN.
        assert atr.iloc[:13].isna().all()
        assert not np.isnan(atr.iloc[-1])

    def test_atr_positive(self):
        df = _trending_ohlc(60)
        atr = TI.atr(df["High"], df["Low"], df["Close"], 14)
        assert atr.iloc[-1] > 0

    def test_constant_band_atr_equals_band(self):
        # High-Low fixed at 1.0, no gaps → ATR converges to 1.0.
        n = 60
        close = pd.Series(np.full(n, 100.0))
        high = close + 0.5
        low = close - 0.5
        atr = TI.atr(high, low, close, 14)
        assert atr.iloc[-1] == pytest.approx(1.0, abs=1e-9)


class TestADX:
    def test_returns_three_series(self):
        df = _trending_ohlc(80)
        adx, plus_di, minus_di = TI.adx(df)
        assert len(adx) == len(plus_di) == len(minus_di) == 80

    def test_strong_uptrend_plus_di_dominates(self):
        df = _trending_ohlc(120, slope=1.0)
        _, plus_di, minus_di = TI.adx(df)
        assert plus_di.iloc[-1] > minus_di.iloc[-1]


class TestEnrichTrendFrame:
    def test_adds_expected_columns_without_mutating_input(self):
        df = _trending_ohlc(120)
        original_cols = set(df.columns)
        out = TI.enrich_trend_frame(df)
        for col in ("EMA50", "EMA200", "RSI", "MACD", "MACDSig", "MACDHist",
                    "ADX", "PlusDI", "MinusDI"):
            assert col in out.columns
        # Input frame is untouched (function returns a copy).
        assert set(df.columns) == original_cols


def _trending_ohlc(n: int, slope: float = 0.5) -> pd.DataFrame:
    close = np.arange(n, dtype=float) * slope + 100
    return pd.DataFrame(
        {
            "Open": close,
            "High": close + 0.5,
            "Low": close - 0.5,
            "Close": close,
            "Volume": np.full(n, 1000.0),
        }
    )
