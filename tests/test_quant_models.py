"""Unit tests for src/core/quant_models.py — OU, GARCH, GBM null test,
UIP/carry, Kalman beta, and Engle-Granger cointegration. All synthetic data
with known ground-truth parameters; no network, no Streamlit."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core.quant_models import (
    OUFit, fit_ou, ou_signal, fit_garch, garch_position_size, gbm_null_test,
    uip_carry_analysis, kalman_dynamic_beta, engle_granger, full_pair_analysis,
)


def _ou_fit(**overrides) -> OUFit:
    defaults = dict(theta=0.1, mu=0.0, sigma=1.0, sigma_eq=1.0, half_life=6.9,
                    zscore=0.0, ar1_beta=0.9, r_squared=0.5, n_obs=100)
    defaults.update(overrides)
    return OUFit(**defaults)

N = 1500
_IDX = pd.bdate_range("2020-01-01", periods=N)


@pytest.fixture
def rng():
    return np.random.default_rng(7)


@pytest.fixture
def ou_series(rng):
    """AR(1)/OU process with known theta=0.05, mu=0, sigma=0.5."""
    theta, mu, sigma = 0.05, 0.0, 0.5
    x = np.zeros(N)
    for t in range(1, N):
        x[t] = x[t - 1] + theta * (mu - x[t - 1]) + sigma * rng.standard_normal()
    return pd.Series(x, index=_IDX)


@pytest.fixture
def cointegrated_pair(rng):
    """y = 1.5 * common + noise, x = common + noise -> cointegrated, hedge ratio 1.5."""
    common = np.cumsum(rng.standard_normal(N))
    x = pd.Series(common + 0.5 * rng.standard_normal(N), index=_IDX)
    y = pd.Series(1.5 * common + 0.5 * rng.standard_normal(N), index=_IDX)
    return y, x


@pytest.fixture
def random_walk_pair(rng):
    """Two independent random walks -> NOT cointegrated."""
    x = pd.Series(np.cumsum(rng.standard_normal(N)), index=_IDX)
    y = pd.Series(np.cumsum(rng.standard_normal(N)), index=_IDX)
    return y, x


class TestFitOU:
    def test_recovers_mean_reversion_params(self, ou_series):
        fit = fit_ou(ou_series)
        assert fit.theta == pytest.approx(0.05, abs=0.02)
        # mu's sampling error is wide for a slow-reverting process over a
        # finite sample (it wanders for long stretches) - just sanity-check
        # it's in the right ballpark, not pinned to true mu=0.
        assert fit.mu == pytest.approx(0.0, abs=1.0)
        assert fit.half_life == pytest.approx(np.log(2) / 0.05, rel=0.5)
        assert fit.n_obs == N

    def test_zscore_matches_last_value(self, ou_series):
        fit = fit_ou(ou_series)
        expected = (ou_series.values[-1] - fit.mu) / (fit.sigma_eq + 1e-12)
        assert fit.zscore == pytest.approx(expected, rel=1e-6)

    def test_too_few_observations_raises(self):
        with pytest.raises(ValueError):
            fit_ou(pd.Series(np.random.randn(10)))

    def test_explosive_series_gives_degenerate_fit(self):
        # Deterministic AR(1) slope of 1.001 (b >= 1, unit root or explosive)
        # -> fit_ou takes the degenerate branch: theta=0, infinite half-life.
        explosive = pd.Series(100 * (1.001 ** np.arange(500)))
        fit = fit_ou(explosive)
        assert fit.theta == 0.0
        assert fit.half_life == np.inf
        assert fit.ar1_beta >= 1.0


class TestOUSignal:
    def test_entry_threshold_triggers_long(self):
        sig = ou_signal(_ou_fit(zscore=-2.5), entry_z=2.0)
        assert sig["signal"] == 1
        assert sig["tradeable"] is True

    def test_entry_threshold_triggers_short(self):
        sig = ou_signal(_ou_fit(zscore=2.5), entry_z=2.0)
        assert sig["signal"] == -1

    def test_long_half_life_not_tradeable(self):
        sig = ou_signal(_ou_fit(half_life=500.0, zscore=3.0),
                        entry_z=2.0, max_half_life=60.0)
        assert sig["tradeable"] is False
        assert sig["signal"] == 0


class TestEngleGranger:
    def test_cointegrated_pair_detected(self, cointegrated_pair):
        y, x = cointegrated_pair
        eg = engle_granger(y, x)
        assert eg["hedge_ratio"] == pytest.approx(1.5, abs=0.1)
        assert eg["cointegrated_5pct"] is True
        assert eg["ec_alpha"] < 0

    def test_random_walk_pair_not_cointegrated(self, random_walk_pair):
        y, x = random_walk_pair
        eg = engle_granger(y, x)
        assert eg["cointegrated_5pct"] is False

    def test_spread_is_regression_residual(self, cointegrated_pair):
        y, x = cointegrated_pair
        eg = engle_granger(y, x)
        assert len(eg["spread"]) == len(y)
        assert eg["spread"].index.equals(y.index)


class TestKalmanDynamicBeta:
    def test_beta_tracks_drifting_true_beta(self, rng):
        x = pd.Series(np.cumsum(rng.standard_normal(N)), index=_IDX)
        beta_path = np.linspace(1.0, 2.0, N)
        y = pd.Series(beta_path * x.values + 0.5 * rng.standard_normal(N), index=_IDX)
        kf = kalman_dynamic_beta(y, x, delta=1e-4)
        assert kf["beta"].iloc[-1] == pytest.approx(2.0, abs=0.3)
        assert set(kf.columns) == {"alpha", "beta", "innovation",
                                   "innovation_var", "innovation_z"}

    def test_output_aligned_to_input_index(self, cointegrated_pair):
        y, x = cointegrated_pair
        kf = kalman_dynamic_beta(y, x)
        assert len(kf) == len(y)


class TestFullPairAnalysis:
    def test_gates_on_cointegration(self, random_walk_pair):
        y, x = random_walk_pair
        out = full_pair_analysis(y, x)
        assert out["cointegrated"] is False
        assert out["tradeable"] is False

    def test_tradeable_pair_produces_signal(self, cointegrated_pair):
        y, x = cointegrated_pair
        out = full_pair_analysis(y, x)
        assert out["cointegrated"] is True
        assert out["signal"] in (-1, 0, 1)
        assert "hedge_ratio_kalman" in out


class TestGBMNullTest:
    def test_extreme_observed_value_is_detected(self, rng):
        prices = pd.Series(
            100 * np.exp(np.cumsum(0.0002 + 0.01 * rng.standard_normal(N))),
            index=_IDX)

        def mean_daily_ret(p):
            return float(np.log(p).diff().mean())

        # An observed mean daily return no GBM path with this vol would
        # plausibly produce - deterministically "beats" the null regardless
        # of which way the specific simulated realization drifts.
        observed = 0.05
        res = gbm_null_test(prices, observed, mean_daily_ret, n_sims=300, seed=1)
        assert res["beats_null"] is True
        assert res["p_value_one_sided"] < 0.05

    def test_matching_the_null_mean_does_not_beat_it(self, rng):
        prices = pd.Series(
            100 * np.exp(np.cumsum(0.0002 + 0.01 * rng.standard_normal(N))),
            index=_IDX)

        def mean_daily_ret(p):
            return float(np.log(p).diff().mean())

        res = gbm_null_test(prices, mean_daily_ret(prices), mean_daily_ret,
                            n_sims=300, seed=1)
        assert res["beats_null"] is False

    def test_deterministic_with_seed(self, rng):
        prices = pd.Series(100 * np.exp(np.cumsum(rng.standard_normal(N) * 0.01)),
                           index=_IDX)

        def metric(p):
            return float(np.log(p).diff().std())

        r1 = gbm_null_test(prices, 0.01, metric, n_sims=200, seed=42)
        r2 = gbm_null_test(prices, 0.01, metric, n_sims=200, seed=42)
        assert r1 == r2


class TestUIPCarryAnalysis:
    def test_output_columns(self, rng):
        i_dom = pd.Series(7.0 + 0.5 * np.sin(np.arange(N) / 120), index=_IDX)
        i_for = pd.Series(2.0 + 0.3 * np.cos(np.arange(N) / 150), index=_IDX)
        spot = pd.Series(
            15 * np.exp(np.cumsum(0.00005 + 0.008 * rng.standard_normal(N))),
            index=_IDX)
        df = uip_carry_analysis(spot, i_dom, i_for).dropna()
        assert {"rate_diff_ann", "uip_expected_ret", "realized_ret",
                "carry_deviation", "carry_z"} <= set(df.columns)
        assert not df.empty

    def test_rate_diff_matches_inputs(self, rng):
        i_dom = pd.Series([7.0] * N, index=_IDX)
        i_for = pd.Series([2.0] * N, index=_IDX)
        spot = pd.Series(100 + np.cumsum(rng.standard_normal(N) * 0.01), index=_IDX)
        df = uip_carry_analysis(spot, i_dom, i_for)
        assert (df["rate_diff_ann"] == 5.0).all()


class TestGARCH:
    def test_fit_returns_expected_keys(self, rng):
        omega, a_true, b_true = 0.02, 0.08, 0.90
        eps = np.zeros(N)
        sig2 = np.full(N, omega / (1 - a_true - b_true))
        for t in range(1, N):
            sig2[t] = omega + a_true * eps[t - 1] ** 2 + b_true * sig2[t - 1]
            eps[t] = np.sqrt(sig2[t]) * rng.standard_normal()
        returns = pd.Series(eps / 100.0, index=_IDX)

        g = fit_garch(returns)
        assert g["persistence"] == pytest.approx(a_true + b_true, abs=0.1)
        assert g["regime"] in ("high", "low", "normal")
        assert len(g["forecast_vol_annual_pct"]) == 5

    def test_too_short_series_raises(self):
        with pytest.raises(ValueError):
            fit_garch(pd.Series(np.random.randn(50) / 100.0))


class TestGarchPositionSize:
    def test_stop_and_leverage(self):
        sizing = garch_position_size(1.0, 0.5, stop_vol_multiple=2.0)
        assert sizing["stop_distance_pct"] == pytest.approx(1.0)
        assert sizing["notional_leverage"] == pytest.approx(1.0)
