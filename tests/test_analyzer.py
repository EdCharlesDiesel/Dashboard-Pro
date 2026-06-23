"""Unit tests for src/core/analyzer.py — TechnicalAnalyzer (pure math)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core.analyzer import TechnicalAnalyzer as TA


def _ohlc(n: int, start: float = 100.0, stop: float = 150.0) -> pd.DataFrame:
    close = np.linspace(start, stop, n)
    return pd.DataFrame(
        {"Open": close, "High": close + 1.0, "Low": close - 1.0,
         "Close": close, "Volume": np.full(n, 1000.0)}
    )


# ── add_indicators ───────────────────────────────────────────────────────────
class TestAddIndicators:
    def test_short_frame_returned_unchanged(self):
        df = _ohlc(10)
        out = TA.add_indicators(df)
        assert "RSI" not in out.columns
        assert out is df  # untouched, same object

    def test_empty_frame_returned_unchanged(self):
        df = pd.DataFrame()
        assert TA.add_indicators(df).empty

    def test_missing_columns_returned_unchanged(self):
        df = pd.DataFrame({"Close": np.linspace(1, 2, 60)})
        out = TA.add_indicators(df)
        assert "RSI" not in out.columns

    def test_full_indicator_set_added(self):
        out = TA.add_indicators(_ohlc(120))
        assert TA.INDICATOR_COLS.issubset(out.columns)

    def test_idempotent_when_already_enriched(self):
        enriched = TA.add_indicators(_ohlc(120))
        again = TA.add_indicators(enriched)
        # No duplicate work; returns a frame with the same columns.
        assert TA.INDICATOR_COLS.issubset(again.columns)


# ── Pivots & Fibonacci ───────────────────────────────────────────────────────
class TestPivots:
    def _ref(self):
        # Two rows; pivots use the *previous* candle (iloc[-2]): H=11, L=9, C=10.
        return pd.DataFrame({"High": [11.0, 99.0], "Low": [9.0, 1.0], "Close": [10.0, 50.0]})

    def test_classic(self):
        p = TA.calculate_pivots(self._ref(), "Classic")
        assert p["Pivot"] == pytest.approx(10.0)
        assert p["R1"] == pytest.approx(11.0)
        assert p["S1"] == pytest.approx(9.0)
        assert p["R2"] == pytest.approx(12.0)
        assert p["S2"] == pytest.approx(8.0)
        assert p["R3"] == pytest.approx(13.0)
        assert p["S3"] == pytest.approx(7.0)

    def test_fibonacci(self):
        p = TA.calculate_pivots(self._ref(), "Fibonacci")
        assert p["Pivot"] == pytest.approx(10.0)
        assert p["R1"] == pytest.approx(10.764)
        assert p["S1"] == pytest.approx(9.236)
        assert p["R3"] == pytest.approx(12.0)

    def test_woodie(self):
        p = TA.calculate_pivots(self._ref(), "Woodie")
        assert p["Pivot"] == pytest.approx(10.0)
        assert "R3" not in p  # Woodie only returns R1/R2/S1/S2

    def test_unknown_method_empty(self):
        assert TA.calculate_pivots(self._ref(), "Nope") == {}

    def test_empty_frame(self):
        assert TA.calculate_pivots(pd.DataFrame()) == {}

    def test_single_row_uses_that_row(self):
        df = pd.DataFrame({"High": [11.0], "Low": [9.0], "Close": [10.0]})
        assert TA.calculate_pivots(df, "Classic")["Pivot"] == pytest.approx(10.0)


class TestExpandedPivots:
    def test_levels_and_midpoints(self):
        df = pd.DataFrame({"High": [11.0, 99.0], "Low": [9.0, 1.0], "Close": [10.0, 50.0]})
        p = TA.calculate_expanded_pivots(df)
        assert p["PP"] == pytest.approx(10.0)
        assert p["R1"] == pytest.approx(11.0)
        assert p["S1"] == pytest.approx(9.0)
        assert p["M2"] == pytest.approx(10.5)  # (PP + R1) / 2
        assert p["M1"] == pytest.approx(9.5)   # (S1 + PP) / 2

    def test_empty_frame(self):
        assert TA.calculate_expanded_pivots(pd.DataFrame()) == {}


class TestFibonacci:
    def test_levels(self):
        df = pd.DataFrame({"High": [20.0, 15.0], "Low": [12.0, 10.0]})
        f = TA.calculate_fibonacci(df)
        assert f["0.0%"] == pytest.approx(20.0)
        assert f["100.0%"] == pytest.approx(10.0)
        assert f["50.0%"] == pytest.approx(15.0)
        assert f["38.2%"] == pytest.approx(16.18)

    def test_empty_frame(self):
        assert TA.calculate_fibonacci(pd.DataFrame()) == {}


# ── Sentiment ────────────────────────────────────────────────────────────────
class TestSentiment:
    def test_bullish(self):
        df = pd.DataFrame({"Close": [10.0], "EMA_20": [9.0], "RSI": [60.0]})
        assert TA.get_sentiment(df) == "Bullish"

    def test_bearish(self):
        df = pd.DataFrame({"Close": [8.0], "EMA_20": [9.0], "RSI": [40.0]})
        assert TA.get_sentiment(df) == "Bearish"

    def test_neutral_mixed(self):
        df = pd.DataFrame({"Close": [10.0], "EMA_20": [9.0], "RSI": [40.0]})
        assert TA.get_sentiment(df) == "Neutral"

    def test_neutral_without_ema(self):
        assert TA.get_sentiment(pd.DataFrame({"Close": [10.0]})) == "Neutral"

    def test_neutral_empty(self):
        assert TA.get_sentiment(pd.DataFrame()) == "Neutral"


# ── Sessions & daily bias ────────────────────────────────────────────────────
class TestSessions:
    def test_get_sessions_keys(self):
        s = TA.get_sessions("2026-06-23")
        assert set(s) == {"Asian", "London", "NY"}
        assert s["London"][0].hour == 7

    def test_session_analysis_empty(self):
        assert TA.session_analysis(pd.DataFrame()) == {}

    def test_session_analysis_segments(self):
        idx = pd.date_range("2026-06-23 00:00", periods=24, freq="h")
        up = np.linspace(1.10, 1.12, 24)
        df = pd.DataFrame({"datetime": idx, "Open": up, "High": up + 0.001,
                           "Low": up - 0.001, "Close": up})
        res = TA.session_analysis(df)
        assert set(res) == {"Asian", "London", "NY"}
        assert res["London"]["direction"].endswith("Bull")


class TestDailyBias:
    def test_too_few_rows_neutral(self):
        idx = pd.date_range("2026-06-23", periods=3, freq="h")
        df = pd.DataFrame({"datetime": idx, "Open": [1.1] * 3, "High": [1.1] * 3,
                           "Low": [1.1] * 3, "Close": [1.1] * 3})
        assert TA.compute_daily_bias(df) == ("Neutral", 50.0)

    def test_strong_uptrend_bullish(self):
        idx = pd.date_range("2026-06-23", periods=24, freq="h")
        up = np.linspace(1.10, 1.12, 24)
        df = pd.DataFrame({"datetime": idx, "Open": up, "High": up + 0.001,
                           "Low": up - 0.001, "Close": up})
        label, pct = TA.compute_daily_bias(df)
        assert label == "Bullish"
        assert 0 <= pct <= 100


# ── DEMA / Supertrend / QQE ──────────────────────────────────────────────────
class TestDEMA:
    def test_constant_series(self):
        s = pd.Series([5.0] * 50)
        assert np.allclose(TA.calculate_dema(s, 10).values, 5.0)


class TestSupertrend:
    def test_short_frame_zero_trend(self):
        df = _ohlc(5)
        st = TA.calculate_supertrend(df, period=10)
        assert (st["Trend"] == 0).all()

    def test_returns_trend_and_line(self):
        st = TA.calculate_supertrend(_ohlc(60), period=10)
        assert set(st.columns) == {"Trend", "Line"}
        assert len(st) == 60


class TestQQE:
    def test_short_series_neutral(self):
        s = pd.Series(np.linspace(1, 2, 5))
        q = TA.calculate_qqe(s, rsi_period=14)
        assert (q["Trend"] == 0).all()

    def test_returns_expected_columns(self):
        s = pd.Series(np.linspace(100, 200, 80))
        q = TA.calculate_qqe(s)
        assert set(q.columns) == {"RSI_Smooth", "Line", "Trend"}


# ── Heikin Ashi ──────────────────────────────────────────────────────────────
class TestHeikinAshi:
    def _df(self):
        return pd.DataFrame({"Open": [10.0, 11.0, 12.0], "High": [12.0, 13.0, 14.0],
                             "Low": [9.0, 10.0, 11.0], "Close": [11.0, 12.0, 13.0]})

    def test_empty_frame(self):
        assert TA.calculate_heikin_ashi(pd.DataFrame()).empty

    def test_close_is_ohlc_mean(self):
        ha = TA.calculate_heikin_ashi(self._df())
        assert ha["Close"].iloc[0] == pytest.approx(10.5)  # (10+12+9+11)/4
        assert ha["High"].iloc[0] == pytest.approx(12.0)

    def test_first_open_is_midpoint(self):
        ha = TA.calculate_heikin_ashi(self._df())
        assert ha["Open"].iloc[0] == pytest.approx(10.5)  # (Open0 + Close0) / 2

    def test_open_recursion(self):
        # ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2
        ha = TA.calculate_heikin_ashi(self._df())
        assert ha["Open"].iloc[1] == pytest.approx(10.5)  # (10.5 + 10.5)/2
        assert ha["Open"].iloc[2] == pytest.approx(11.0)  # (10.5 + 11.5)/2

    def test_no_zero_open_regression(self):
        # The pandas-3 chained-assignment bug left HA Open at 0.0; guard against it.
        ha = TA.calculate_heikin_ashi(self._df())
        assert (ha["Open"] != 0.0).all()


# ── Volume profile ───────────────────────────────────────────────────────────
class TestVolumeProfile:
    def _clustered(self):
        # Volume concentrated in the 14–16 band → clear POC there.
        return pd.DataFrame({"Low": [14.0] * 30 + [10.0] * 10,
                             "High": [16.0] * 30 + [20.0] * 10,
                             "Volume": [100.0] * 30 + [5.0] * 10})

    def test_empty_or_no_volume(self):
        assert TA.compute_volume_profile(pd.DataFrame()).empty
        no_vol = pd.DataFrame({"Low": [1.0], "High": [2.0]})
        assert TA.compute_volume_profile(no_vol).empty

    def test_shape_and_columns(self):
        vp = TA.compute_volume_profile(self._clustered(), n_bins=20)
        assert list(vp.columns) == ["price", "volume"]
        assert len(vp) == 20

    def test_detect_levels_finds_poc(self):
        vp = TA.compute_volume_profile(self._clustered(), n_bins=20)
        levels = TA.detect_levels(vp)
        # POC sits in the high-volume cluster (bin edges pull it to ~13.75),
        # well away from the sparse 16–20 region.
        assert 13.0 <= levels["poc"] <= 16.5
        assert isinstance(levels["hvns"], list)

    def test_detect_levels_empty(self):
        levels = TA.detect_levels(pd.DataFrame(columns=["price", "volume"]))
        assert levels["poc"] is None

    def test_value_area(self):
        vp = TA.compute_volume_profile(self._clustered(), n_bins=20)
        lo, hi = TA.value_area(vp)
        assert lo is not None and hi is not None
        assert lo <= hi

    def test_value_area_empty(self):
        assert TA.value_area(pd.DataFrame(columns=["price", "volume"])) == (None, None)


# ── Pro signals & MTF sentiment ──────────────────────────────────────────────
class TestProSignals:
    def test_empty_frame(self):
        assert TA.generate_pro_signals(pd.DataFrame(), {}).empty

    def test_signal_columns_present(self):
        df = _ohlc(120)
        pivots = TA.calculate_expanded_pivots(df)
        out = TA.generate_pro_signals(df, pivots)
        assert "Signal" in out.columns
        assert "Signal_Score" in out.columns
        assert out["Signal"].isin(
            ["HOLD", "BUY", "STRONG BUY", "SELL", "STRONG SELL"]
        ).all()


class TestMTFSentiment:
    def test_missing_pair_is_na(self):
        data = {"Daily": {}}
        assert TA.get_mtf_sentiment(data, "EUR/USD") == {"Daily": "N/A"}

    def test_present_pair_scored(self):
        data = {"Daily": {"EUR/USD": _ohlc(120)}}
        res = TA.get_mtf_sentiment(data, "EUR/USD")
        assert res["Daily"] in {"Bullish", "Bearish", "Neutral"}
