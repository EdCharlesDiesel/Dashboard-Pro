"""Unit tests for src/services/index_analysis.py."""
import numpy as np
import pandas as pd
import pytest

from src.services import index_analysis as ia


def _series(vals, start="2026-01-05", freq="D"):
    idx = pd.date_range(start, periods=len(vals), freq=freq)
    return pd.Series(vals, index=idx, dtype=float)


class TestAlignment:
    def test_common_dates_only(self):
        a = _series([1, 2, 3, 4, 5])                 # 01-05 .. 01-09
        b = _series([1, 2, 3], start="2026-01-06")   # 01-06 .. 01-08
        df = ia._aligned(a, b)
        assert list(df.index.strftime("%m-%d")) == ["01-06", "01-07", "01-08"]
        assert list(df.columns) == ["a", "b"]


class TestCorrelation:
    def test_perfectly_correlated_returns(self):
        base = np.linspace(100, 120, 80)
        a = _series(base)
        b = _series(base * 2.0)          # exact scalar multiple → returns identical
        s = ia.correlation_summary(a, b, windows=(20, 60))
        assert s["full"] == pytest.approx(1.0, abs=1e-6)
        assert s["w20"] == pytest.approx(1.0, abs=1e-6)
        assert s["n"] > 60

    def test_anti_correlated_returns(self):
        rng = np.random.default_rng(3)
        r = rng.standard_normal(120) * 0.01
        a = _series(100 * np.cumprod(1 + r))
        b = _series(100 * np.cumprod(1 - r))   # mirror returns
        s = ia.correlation_summary(a, b)
        assert s["full"] < -0.9

    def test_beta_recovers_scalar(self):
        # b's returns = 0.5 × a's returns → beta(a on b) ≈ 2.0
        rng = np.random.default_rng(7)
        ra = rng.standard_normal(200) * 0.01
        a = _series(100 * np.cumprod(1 + ra))
        b = _series(100 * np.cumprod(1 + ra * 0.5))
        s = ia.correlation_summary(a, b)
        assert s["beta"] == pytest.approx(2.0, rel=0.05)

    def test_rolling_correlation_shape_and_range(self):
        rng = np.random.default_rng(1)
        a = _series(100 + np.cumsum(rng.standard_normal(100)))
        b = _series(100 + np.cumsum(rng.standard_normal(100)))
        rc = ia.rolling_correlation(a, b, 20).dropna()
        assert len(rc) > 0
        assert ((rc >= -1.0001) & (rc <= 1.0001)).all()

    def test_short_series_is_safe(self):
        a = _series([1.0, 2.0])
        b = _series([1.0, 2.0])
        s = ia.correlation_summary(a, b)
        assert s["full"] is None and s["beta"] is None
        assert s["n"] <= 1


class TestRatio:
    def test_ratio_rises_when_a_outperforms(self):
        a = _series(np.linspace(100, 130, 30))   # +30%
        b = _series(np.linspace(100, 110, 30))   # +10%
        r = ia.ratio(a, b).dropna()
        assert r.iloc[-1] > r.iloc[0]

    def test_ratio_handles_zero_denominator(self):
        a = _series([100.0, 101.0, 102.0])
        b = _series([1.0, 0.0, 2.0])
        r = ia.ratio(a, b)
        assert np.isnan(r.iloc[1])              # /0 → NaN, not inf/raise
