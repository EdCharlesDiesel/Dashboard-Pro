"""Correctness checks for src/core/stochastic.py.

Converted from a print-based script to real pytest assertions: the original
`ok(name, cond)` helper printed PASS/FAIL but never failed a run, and its bare
`from stochastic import ...` broke collection for the whole suite.

The maths checks are kept verbatim in spirit — each verifies an *identity*, an
*inversion*, or a *known limit* rather than a magic number, so they stay valid
under refactoring:

  QV -> t, realised vol recovers sigma, Ito identity mu = nu + sigma^2/2,
  Hurst = 0.5 on GBM, signature-plot slope shift = 2q, gambler's ruin as the
  zero-drift limit, Kelly peak at mu/sigma^2, E[S_T] = S_0 e^{mu T},
  put-call parity, Greeks vs finite differences, implied-vol round trip.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats as _st

from src.core.stochastic import (
    barrier_probabilities,
    black_scholes,
    calibrate_gbm,
    implied_vol,
    kelly_leverage,
    log_returns,
    realised_variance,
    signature_plot,
    simulate_gbm,
    variance_scaling,
    volatility_estimators,
)

TRUE_MU, TRUE_SIG, N = 0.12, 0.28, 4000
DT = 1 / 252
NU = TRUE_MU - 0.5 * TRUE_SIG**2


@pytest.fixture(scope="module")
def gbm_path():
    """A GBM price path with known mu/sigma — the workhorse fixture."""
    rng = np.random.default_rng(3)
    lr = NU * DT + TRUE_SIG * np.sqrt(DT) * rng.standard_normal(N)
    return 1900 * np.exp(np.concatenate(([0.0], np.cumsum(lr))))


@pytest.fixture(scope="module")
def ohlc_frame():
    """OHLC built from a finely sub-sampled path, so ranges are meaningful."""
    rng = np.random.default_rng(23)
    n_bars, spb = 900, 400
    sub = (NU * (DT / spb)
           + TRUE_SIG * np.sqrt(DT / spb) * rng.standard_normal((n_bars, spb)))
    path = 1900 * np.exp(np.cumsum(sub.ravel())).reshape(n_bars, spb)
    return pd.DataFrame({"open": path[:, 0], "high": path.max(1),
                         "low": path.min(1), "close": path[:, -1]})


class TestQuadraticVariation:
    def test_qv_of_brownian_converges_to_t(self):
        # [W,W]_t -> t. On [0,1] the realised variation must approach 1.
        rng = np.random.default_rng(3)
        n = 2**14
        w = np.concatenate(([0.0], np.cumsum(rng.standard_normal(n) / np.sqrt(n))))
        assert abs(np.sum(np.diff(w) ** 2) - 1.0) < 0.05

    def test_realised_vol_recovers_sigma(self, gbm_path):
        rv = realised_variance(log_returns(gbm_path), annualise=252)
        assert abs(np.sqrt(rv) - TRUE_SIG) < 0.02


class TestCalibration:
    def test_recovers_sigma(self, gbm_path):
        assert abs(calibrate_gbm(gbm_path).sigma - TRUE_SIG) < 0.02

    def test_recovers_nu_within_three_standard_errors(self, gbm_path):
        # Drift is barely identifiable from price history — that's the point.
        p = calibrate_gbm(gbm_path)
        se_nu = TRUE_SIG / np.sqrt(N / 252)
        assert abs(p.nu - NU) < 3 * se_nu

    def test_ito_identity(self, gbm_path):
        p = calibrate_gbm(gbm_path)
        assert abs(p.mu - (p.nu + p.drag)) < 1e-12

    def test_drag_is_half_sigma_squared(self, gbm_path):
        p = calibrate_gbm(gbm_path)
        assert abs(p.drag - 0.5 * p.sigma**2) < 1e-12

    def test_hurst_is_one_half_on_gbm(self, gbm_path):
        vs = variance_scaling(gbm_path)
        assert abs(vs["hurst"] - 0.5) < 0.06


class TestSignaturePlot:
    def test_injected_noise_shifts_slope_by_2q(self, gbm_path):
        # Only the PAIRED difference on the same path is meaningful — a single
        # path carries its own random tilt.
        rng = np.random.default_rng(101)
        q = 0.004
        noisy = gbm_path * np.exp(rng.standard_normal(gbm_path.size) * q)
        _, clean_fit = signature_plot(gbm_path, 25)
        _, noisy_fit = signature_plot(noisy, 25)
        d_slope = noisy_fit["slope"] - clean_fit["slope"]
        assert abs(d_slope - 2 * q**2) < 0.15 * 2 * q**2

    def test_realistic_daily_spread_is_not_flagged(self, gbm_path):
        # A ~1bp spread is invisible against daily bar variance. Flagging it
        # would be a false positive, so the fit must stay insignificant.
        rng = np.random.default_rng(202)
        realistic = gbm_path * np.exp(rng.standard_normal(gbm_path.size) * 0.0001)
        _, fit = signature_plot(realistic, 25, n_boot=150)
        assert not fit["significant"]

    def test_bootstrap_se_exceeds_naive_ols(self, gbm_path):
        # Overlapping rows are dependent, so the OLS SE understates the truth.
        curve, _ = signature_plot(gbm_path, 25)
        naive = _st.linregress(1 / curve["skip"], curve["rv_per_period"]).stderr
        _, boot = signature_plot(gbm_path, 25, n_boot=150)
        assert boot["slope_boot_sd"] > naive


class TestVolatilityEstimators:
    ESTIMATORS = ("close_to_close", "parkinson", "garman_klass",
                  "rogers_satchell", "yang_zhang")

    @pytest.mark.parametrize("name", ESTIMATORS)
    def test_each_estimator_recovers_sigma(self, ohlc_frame, name):
        assert abs(volatility_estimators(ohlc_frame)[name] - TRUE_SIG) < 0.045

    def test_range_estimator_is_more_efficient(self, ohlc_frame):
        # The whole reason to use Parkinson: less dispersion on the same data.
        def spread(est):
            vals = [volatility_estimators(ohlc_frame.iloc[i:i + 60])[est]
                    for i in range(0, 840, 60)]
            return float(np.std(vals))
        assert spread("parkinson") < spread("close_to_close")

    def test_missing_columns_raise(self):
        with pytest.raises(ValueError):
            volatility_estimators(pd.DataFrame({"close": [1.0] * 10}))


class TestFirstPassage:
    S0, TP, SL = 1900.0, 1957.0, 1881.0

    def _ruin(self):
        u, l = np.log(self.TP / self.S0), -np.log(self.SL / self.S0)
        return l / (u + l)

    def test_zero_log_drift_reproduces_gamblers_ruin(self):
        # mu = sigma^2/2  =>  nu = 0  =>  pure gambler's ruin.
        b = barrier_probabilities(self.S0, self.TP, self.SL,
                                  sigma=0.20, mu=0.5 * 0.20**2)
        assert abs(b["p_take_profit"] - self._ruin()) < 1e-9

    def test_zero_log_drift_gives_zero_expected_log_return(self):
        b = barrier_probabilities(self.S0, self.TP, self.SL,
                                  sigma=0.20, mu=0.5 * 0.20**2)
        assert abs(b["expected_log_return"]) < 1e-12

    def test_price_ev_stays_positive_by_jensen(self):
        # A log-martingale is a price SUBmartingale — the two don't vanish
        # together, which is the whole Jensen point.
        b = barrier_probabilities(self.S0, self.TP, self.SL,
                                  sigma=0.20, mu=0.5 * 0.20**2)
        assert b["expected_value_price"] > 0

    def test_monte_carlo_matches_closed_form(self):
        b = barrier_probabilities(self.S0, self.TP, self.SL, sigma=0.20,
                                  mu=0.10, horizon_years=3.0, n_sims=60_000)
        assert abs(b["mc_p_tp_given_resolved"] - b["p_take_profit"]) < 0.012

    def test_drag_can_flip_the_log_drift_sign(self):
        # Positive arithmetic drift, negative log drift — the trap.
        b = barrier_probabilities(self.S0, self.TP, self.SL, sigma=0.60, mu=0.10)
        assert b["log_drift_nu"] < 0 < b["arith_drift_mu"]


class TestKellyAndSimulation:
    def test_growth_peaks_at_mu_over_sigma_squared(self, gbm_path):
        p = calibrate_gbm(gbm_path)
        kdf = kelly_leverage(p)
        peak = kdf.loc[kdf["log_growth"].idxmax(), "leverage"]
        assert abs(peak - p.kelly) < 0.15

    def test_growth_is_zero_at_twice_kelly(self, gbm_path):
        p = calibrate_gbm(gbm_path)
        f = 2 * p.kelly
        assert abs(f * p.mu - 0.5 * f**2 * p.sigma**2) < 1e-9

    def test_expected_terminal_price(self):
        paths = simulate_gbm(1900.0, 0.10, 0.25, 1.0, n_paths=200_000)
        assert abs(paths[:, -1].mean() / (1900.0 * np.exp(0.10)) - 1) < 0.01


class TestBlackScholes:
    S, K, T, R, SIG = 100.0, 105.0, 0.75, 0.04, 0.22

    def test_put_call_parity(self):
        c = black_scholes(self.S, self.K, self.T, self.R, self.SIG, "call")
        p = black_scholes(self.S, self.K, self.T, self.R, self.SIG, "put")
        resid = c["price"] - p["price"] - (self.S - self.K * np.exp(-self.R * self.T))
        assert abs(resid) < 1e-10

    def test_delta_parity(self):
        c = black_scholes(self.S, self.K, self.T, self.R, self.SIG, "call")
        p = black_scholes(self.S, self.K, self.T, self.R, self.SIG, "put")
        assert abs(c["delta"] - p["delta"] - 1.0) < 1e-10

    @pytest.mark.parametrize("greek,tol", [("delta", 1e-6), ("gamma", 1e-3),
                                           ("vega", 1e-4)])
    def test_greeks_match_finite_differences(self, greek, tol):
        h = 1e-5
        base = black_scholes(self.S, self.K, self.T, self.R, self.SIG)
        if greek == "vega":
            up = black_scholes(self.S, self.K, self.T, self.R, self.SIG + h)["price"]
            dn = black_scholes(self.S, self.K, self.T, self.R, self.SIG - h)["price"]
            fd = (up - dn) / (2 * h)
        else:
            up = black_scholes(self.S + h, self.K, self.T, self.R, self.SIG)["price"]
            dn = black_scholes(self.S - h, self.K, self.T, self.R, self.SIG)["price"]
            fd = ((up - dn) / (2 * h) if greek == "delta"
                  else (up - 2 * base["price"] + dn) / h**2)
        assert abs(fd - base[greek]) < tol

    def test_implied_vol_round_trips(self):
        c = black_scholes(self.S, self.K, self.T, self.R, self.SIG, "call")
        iv = implied_vol(c["price"], self.S, self.K, self.T, self.R, "call")
        assert abs(iv - self.SIG) < 1e-8

    def test_arbitrage_violating_quote_returns_nan(self):
        # A call quoted above spot has no implied vol.
        assert np.isnan(implied_vol(150.0, self.S, self.K, self.T, self.R, "call"))

    def test_low_but_admissible_quote_still_solves(self):
        assert implied_vol(0.001, self.S, self.K, self.T, self.R, "call") > 0
