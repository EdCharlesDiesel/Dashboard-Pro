"""Unit tests for src/indicators/leading.py."""
import numpy as np
import pandas as pd
import pytest

from src.indicators.leading import (
    cci,
    demarker,
    divergence_pivot_read,
    near_pivot,
    pivot_points,
    rsi_divergence,
    stochastic,
    volume_delta,
    williams_r,
)


def _ohlc(close, volume=None, spread=0.5):
    close = np.asarray(close, dtype=float)
    n = len(close)
    idx = pd.date_range("2026-01-01", periods=n, freq="D")
    opens = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame({
        "Open": opens,
        "High": np.maximum(opens, close) + spread,
        "Low": np.minimum(opens, close) - spread,
        "Close": close,
        "Volume": volume if volume is not None else np.full(n, 1000.0),
    }, index=idx)


def _uptrend(n=60):
    return _ohlc(100 + 0.5 * np.arange(n))


def _downtrend(n=60):
    return _ohlc(200 - 0.5 * np.arange(n))


class TestOscillators:
    def test_stochastic_range_and_trend(self):
        k, d = stochastic(_uptrend())
        valid_k = k.dropna()
        assert ((valid_k >= 0) & (valid_k <= 100)).all()
        assert valid_k.iloc[-1] > 70          # persistent uptrend → near top
        assert d.dropna().iloc[-1] > 70

    def test_williams_r_range(self):
        wr = williams_r(_uptrend()).dropna()
        assert ((wr >= -100) & (wr <= 0)).all()
        assert wr.iloc[-1] > -30              # uptrend → near 0
        assert williams_r(_downtrend()).dropna().iloc[-1] < -70

    def test_cci_sign_follows_trend(self):
        assert cci(_uptrend()).dropna().iloc[-1] > 0
        assert cci(_downtrend()).dropna().iloc[-1] < 0

    def test_demarker_range_and_exhaustion(self):
        dm_up = demarker(_uptrend()).dropna()
        assert ((dm_up >= 0) & (dm_up <= 1)).all()
        assert dm_up.iloc[-1] > 0.7           # one-way buying
        assert demarker(_downtrend()).dropna().iloc[-1] < 0.3


class TestVolumeDelta:
    def test_green_dominance_positive_imbalance(self):
        df = _uptrend()                        # every candle closes above open
        vd = volume_delta(df)
        assert vd["imbalance"].dropna().iloc[-1] > 0.9
        assert vd["cum_delta"].iloc[-1] > 0

    def test_red_dominance_negative_imbalance(self):
        vd = volume_delta(_downtrend())
        assert vd["imbalance"].dropna().iloc[-1] < -0.9

    def test_zero_volume_falls_back_to_range_proxy(self):
        df = _uptrend()
        df["Volume"] = 0.0
        vd = volume_delta(df)
        # Spot-FX case: still a meaningful positive lean, not NaN/zero.
        assert vd["imbalance"].dropna().iloc[-1] > 0.9


class TestPivots:
    def test_classic_formulas(self):
        p = pivot_points(high=110.0, low=90.0, close=100.0)
        assert p["P"] == pytest.approx(100.0)
        assert p["R1"] == pytest.approx(110.0)
        assert p["S1"] == pytest.approx(90.0)
        assert p["R2"] == pytest.approx(120.0)
        assert p["S2"] == pytest.approx(80.0)
        assert p["R3"] == pytest.approx(130.0)
        assert p["S3"] == pytest.approx(70.0)

    def test_near_pivot_matches_closest_within_tolerance(self):
        p = pivot_points(110.0, 90.0, 100.0)
        assert near_pivot(100.05, p, tolerance_pct=0.15) == "P"
        assert near_pivot(150.0, p, tolerance_pct=0.15) is None


class TestRsiDivergence:
    def _bullish_divergence_frame(self):
        # Sharp fall to a low, weak bounce, then a marginally lower low on a
        # much milder decline — price LL, RSI HL (classic bullish divergence).
        seg1 = np.linspace(120, 100, 20)      # hard sell-off → RSI crushed
        seg2 = np.linspace(100, 106, 8)       # bounce
        seg3 = np.linspace(106, 99.5, 20)     # grind to a LOWER low
        seg4 = np.linspace(99.5, 101, 6)      # confirmation bars after swing
        return _ohlc(np.concatenate([seg1, seg2, seg3, seg4]), spread=0.05)

    def test_bullish_divergence_detected(self):
        out = rsi_divergence(self._bullish_divergence_frame())
        assert out["bullish"] is True
        assert "RSI higher low" in out["detail"]

    def test_clean_trend_has_no_divergence(self):
        out = rsi_divergence(_uptrend(80))
        assert out["bullish"] is False and out["bearish"] is False

    def test_short_frame_never_guesses(self):
        out = rsi_divergence(_uptrend(6))
        assert out["bullish"] is False and out["bearish"] is False


class TestDivergencePivotRead:
    def test_direction_requires_both_conditions(self):
        df = _uptrend(80)
        out = divergence_pivot_read(df)
        assert out["direction"] is None       # no divergence on a clean trend
        assert set(out["pivots"]) == {"P", "R1", "S1", "R2", "S2", "R3", "S3"}

    def test_bullish_divergence_at_pivot_fires_long(self):
        # Reuse the divergence frame, then force the last close onto the
        # prior bar's pivot so the confluence condition is met exactly.
        base = TestRsiDivergence()._bullish_divergence_frame()
        prev = base.iloc[-2]
        pivot = (prev["High"] + prev["Low"] + prev["Close"]) / 3
        base.iloc[-1, base.columns.get_loc("Close")] = pivot
        out = divergence_pivot_read(base, tolerance_pct=0.2)
        assert out["at_level"] is not None
        assert out["direction"] == "Long"
