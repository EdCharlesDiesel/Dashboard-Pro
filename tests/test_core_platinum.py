"""The platinum monitor engine — pure numpy/pandas, so it is testable exactly.

`src/core/platinum.py` has no Streamlit, no DB and no network, which is why it
stays inside the coverage gate while its collector (`data_backbone`) and its tab
(`pages_lib`) are omitted.

The claim the module exists to support is narrow and worth pinning down: does
USD/ZAR carry information about XPTUSD *after* the dollar factor is removed? A
raw correlation cannot answer that, because DXY moves both. So the tests below
construct series where the answer is known by construction, rather than
asserting on whatever live data happens to do today.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core.platinum import (
    disconnect_state,
    log_returns,
    nested_incremental_test,
    ols,
    producer_margin,
    ratio_series,
    realised_vol,
    zscore,
)


def _idx(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2024-01-01", periods=n)


class TestLogReturns:
    def test_a_constant_series_has_zero_return(self):
        s = pd.Series([100.0] * 20, index=_idx(20))
        assert log_returns(s).dropna().abs().max() == pytest.approx(0.0)

    def test_a_known_doubling_gives_ln_two(self):
        s = pd.Series([100.0, 200.0], index=_idx(2))
        assert log_returns(s).dropna().iloc[0] == pytest.approx(np.log(2))

    def test_returns_are_additive_across_steps(self):
        # The property that makes log returns worth using at all.
        s = pd.Series([100.0, 110.0, 121.0], index=_idx(3))
        r = log_returns(s).dropna()
        assert r.sum() == pytest.approx(np.log(121.0 / 100.0))


class TestZScore:
    def test_a_constant_series_does_not_divide_by_zero(self):
        s = pd.Series([5.0] * 40, index=_idx(40))
        z = zscore(s, window=20)
        assert not np.isinf(z.dropna()).any()

    def test_a_value_at_the_mean_scores_zero(self):
        s = pd.Series([1.0, 2.0, 3.0, 2.0], index=_idx(4))
        assert zscore(s).iloc[-1] == pytest.approx(0.0, abs=1e-9)


class TestRealisedVol:
    def test_a_flat_series_has_no_volatility(self):
        s = pd.Series([100.0] * 60, index=_idx(60))
        assert realised_vol(s, window=20).dropna().abs().max() == pytest.approx(0.0)

    def test_a_noisier_series_scores_higher(self):
        rng = np.random.default_rng(0)
        calm = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.001, 200))), index=_idx(200))
        wild = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 200))), index=_idx(200))
        assert realised_vol(wild).dropna().mean() > realised_vol(calm).dropna().mean()

    def test_annualising_scales_up(self):
        rng = np.random.default_rng(1)
        s = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 120))), index=_idx(120))
        ann = realised_vol(s, annualise=True).dropna().mean()
        raw = realised_vol(s, annualise=False).dropna().mean()
        assert ann > raw


class TestRatioAndMargin:
    def test_the_ratio_is_elementwise(self):
        a = pd.Series([10.0, 20.0], index=_idx(2))
        b = pd.Series([2.0, 5.0], index=_idx(2))
        assert list(ratio_series(a, b).round(6)) == [5.0, 4.0]

    def test_producer_margin_rises_with_a_weaker_rand(self):
        # A South African producer sells platinum in dollars and pays costs in
        # rand, so a weaker rand (higher USD/ZAR) widens the margin. If this
        # inverts, the tab's headline reads backwards.
        idx = _idx(3)
        xpt = pd.Series([1000.0, 1000.0, 1000.0], index=idx)
        zar_weak = pd.Series([17.0, 18.0, 19.0], index=idx)
        m = producer_margin(xpt, zar_weak).dropna()
        assert m.is_monotonic_increasing


class TestDisconnectState:
    def test_none_is_handled_rather_than_crashing(self):
        assert isinstance(disconnect_state(None), str)

    @pytest.mark.parametrize("z", [3.0, -3.0])
    def test_a_wide_dislocation_is_not_reported_as_calm(self, z):
        assert disconnect_state(z, entry=2.0, warn=1.5) != disconnect_state(0.0)

    def test_the_magnitude_picks_the_tier_and_the_sign_picks_the_side(self):
        # |z| decides how stretched; the sign decides rich vs cheap. Both
        # matter, and conflating them would make the tab's headline meaningless.
        rich, cheap = disconnect_state(3.0), disconnect_state(-3.0)
        assert rich != cheap, "sign must distinguish rich from cheap"
        assert "STRETCHED" in rich and "STRETCHED" in cheap, "same severity tier"
        assert disconnect_state(0.0) == "IN LINE"
        assert disconnect_state(1.7) == "EXTENDED"


class TestOLSAndIncrementalInformation:
    def test_ols_recovers_a_known_slope(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=300)
        y = 2.5 * x + rng.normal(0, 0.01, size=300)
        # ols(add_const=True) supplies its own intercept - passing another
        # makes a duplicate constant column and the slope shifts to beta[2].
        res = ols(y, x.reshape(-1, 1))
        assert res.beta[1] == pytest.approx(2.5, abs=0.05)

    def test_noise_adds_no_information_over_the_base_factor(self):
        # The null case the module's whole premise depends on: a variable that
        # is pure noise must not appear significant once the base factor is in.
        rng = np.random.default_rng(11)
        n = 400
        dollar = pd.Series(rng.normal(size=n), index=_idx(n))
        noise = pd.Series(rng.normal(size=n), index=_idx(n))
        y = 1.8 * dollar + pd.Series(rng.normal(0, 0.1, n), index=_idx(n))
        out = nested_incremental_test(y, dollar.to_frame("dxy"), noise.to_frame("zar"))
        assert out["p_F"] > 0.05, "pure noise must not look informative"
        assert out["delta_r2"] < 0.01
        assert out["max_abs_t_extra"] < 2.0

    def test_a_genuine_extra_factor_is_detectable(self):
        rng = np.random.default_rng(13)
        n = 400
        dollar = pd.Series(rng.normal(size=n), index=_idx(n))
        zar = pd.Series(rng.normal(size=n), index=_idx(n))
        y = 1.5 * dollar + 2.0 * zar + pd.Series(rng.normal(0, 0.05, n), index=_idx(n))
        out = nested_incremental_test(y, dollar.to_frame("dxy"), zar.to_frame("zar"))
        assert out["p_F"] < 0.01, "a real factor must be detected"
        assert out["delta_r2"] > 0.5
        assert out["extra_coefs"].loc["zar", "coef"] == pytest.approx(2.0, abs=0.05)
