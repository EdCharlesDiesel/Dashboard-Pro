"""Probability forecasting engine — Monte Carlo on daily closes.

All the math lives here so pages only render results:

- Drift: sample mean of log returns, shrunk 75% toward zero (daily FX drift
  is mostly noise; shrinkage stops the cone from tilting on a lucky run).
- Volatility: RiskMetrics EWMA (lambda = 0.94) for the current regime,
  long-run sample vol for comparison.
- Paths, two families combined 50/50:
    1. Parametric: EWMA-vol geometric paths with Student-t(5) innovations
       (fat tails a Gaussian would miss).
    2. Stationary block bootstrap of the actual return history (mean block
       5 days) — keeps real-world autocorrelation and skew.
- Touch probabilities: first close beyond TP vs first close beyond SL per
  path, using the house risk model (SL = mult x ATR14, TP = R:R x SL).
  Close-based touches slightly understate intraday hits on both sides.

Results are deterministic for a given input (fixed RNG seed) so the UI
doesn't flicker between reruns.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

_LAMBDA = 0.94          # RiskMetrics decay
_DRIFT_SHRINK = 0.25    # keep 25% of sample drift
_T_DOF = 5              # Student-t tail heaviness
_N_PARAMETRIC = 4000
_N_BOOTSTRAP = 4000
_MEAN_BLOCK = 5         # stationary bootstrap mean block length (days)
_SEED = 20260612


@dataclass(frozen=True)
class ForecastResult:
    last: float
    horizon: int
    p_up: float                     # P(close above last at horizon)
    median_price: float
    q05: float
    q95: float
    exp_move_pct: float             # median terminal move, percent
    cone: Dict[str, np.ndarray]     # per-step quantiles: q05/q25/q50/q75/q95
    terminal_sample: np.ndarray     # decimated terminal prices for histogram
    # House risk model (SL = mult x ATR, TP = rr x SL), close-touch basis
    sl_dist: float
    tp_dist: float
    p_tp_first_long: float
    p_sl_first_long: float
    p_tp_first_short: float
    p_sl_first_short: float
    ewma_vol_ann: float
    longrun_vol_ann: float

    @property
    def vol_ratio(self) -> float:
        return self.ewma_vol_ann / self.longrun_vol_ann if self.longrun_vol_ann else 1.0

    def edge(self, rr: float, side: str) -> float:
        """P(TP first) minus the breakeven probability for the given R:R."""
        breakeven = 1.0 / (1.0 + rr)
        p = self.p_tp_first_long if side == "LONG" else self.p_tp_first_short
        return p - breakeven


def _ewma_vol(returns: np.ndarray, lam: float = _LAMBDA) -> float:
    var = returns.var()
    for r in returns:
        var = lam * var + (1 - lam) * r * r
    return float(np.sqrt(var))


def _parametric_paths(rng, last, mu, sig, horizon, n) -> np.ndarray:
    # t(5) innovations rescaled to unit variance: Var[t_v] = v/(v-2)
    eps = rng.standard_t(_T_DOF, size=(n, horizon))
    eps *= np.sqrt((_T_DOF - 2) / _T_DOF)
    steps = mu + sig * eps
    return last * np.exp(np.cumsum(steps, axis=1))


def _bootstrap_paths(rng, last, returns, horizon, n) -> np.ndarray:
    m = len(returns)
    out = np.empty((n, horizon))
    p_new = 1.0 / _MEAN_BLOCK
    for i in range(n):
        steps = np.empty(horizon)
        j = rng.integers(0, m)
        for t in range(horizon):
            steps[t] = returns[j]
            # stationary bootstrap: restart block with prob 1/mean_block
            j = rng.integers(0, m) if rng.random() < p_new else (j + 1) % m
        out[i] = steps
    return last * np.exp(np.cumsum(out, axis=1))


def _first_touch(paths: np.ndarray, level: float, above: bool) -> np.ndarray:
    """Index of first close beyond `level` per path; horizon+1 if never."""
    hit = paths >= level if above else paths <= level
    never = ~hit.any(axis=1)
    idx = hit.argmax(axis=1).astype(float)
    idx[never] = paths.shape[1] + 1
    return idx


@st.cache_data(ttl=600, show_spinner=False)
def run_forecast(closes: pd.Series, atr: float, horizon: int,
                 sl_mult: float = 1.5, rr: float = 2.0) -> ForecastResult:
    """Full Monte Carlo forecast. Cached; deterministic per input."""
    px = closes.dropna().to_numpy(dtype=float)
    last = float(px[-1])
    log_r = np.diff(np.log(px))
    log_r = log_r[np.isfinite(log_r)]

    mu = float(log_r.mean()) * _DRIFT_SHRINK
    sig_ewma = _ewma_vol(log_r)
    sig_lr = float(log_r.std())

    rng = np.random.default_rng(_SEED)
    paths = np.vstack([
        _parametric_paths(rng, last, mu, sig_ewma, horizon, _N_PARAMETRIC),
        _bootstrap_paths(rng, last, log_r, horizon, _N_BOOTSTRAP),
    ])

    terminal = paths[:, -1]
    cone = {
        f"q{q:02d}": np.percentile(paths, q, axis=0)
        for q in (5, 25, 50, 75, 95)
    }

    sl_dist = sl_mult * atr
    tp_dist = rr * sl_dist

    tp_long = _first_touch(paths, last + tp_dist, above=True)
    sl_long = _first_touch(paths, last - sl_dist, above=False)
    tp_short = _first_touch(paths, last - tp_dist, above=False)
    sl_short = _first_touch(paths, last + sl_dist, above=True)

    n = len(paths)
    return ForecastResult(
        last=last,
        horizon=horizon,
        p_up=float((terminal > last).mean()),
        median_price=float(np.median(terminal)),
        q05=float(np.percentile(terminal, 5)),
        q95=float(np.percentile(terminal, 95)),
        exp_move_pct=float((np.median(terminal) / last - 1) * 100),
        cone=cone,
        terminal_sample=terminal[:: max(1, n // 1500)].copy(),
        sl_dist=float(sl_dist),
        tp_dist=float(tp_dist),
        p_tp_first_long=float((tp_long < sl_long).mean()),
        p_sl_first_long=float((sl_long < tp_long).mean()),
        p_tp_first_short=float((tp_short < sl_short).mean()),
        p_sl_first_short=float((sl_short < tp_short).mean()),
        ewma_vol_ann=float(sig_ewma * np.sqrt(252) * 100),
        longrun_vol_ann=float(sig_lr * np.sqrt(252) * 100),
    )


# ══════════════════════════════════════════════════════════════════
# STATISTICAL FORECASTS  (point trajectory + analytic interval)
# ══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class StatForecast:
    model: str
    forecast: pd.Series           # point forecast, business-day index
    lower: pd.Series              # lower prediction band
    upper: pd.Series              # upper prediction band


@st.cache_data(ttl=1800, show_spinner=False)
def run_statistical_forecast(series: pd.Series, periods: int, model_type: str,
                             ci: float = 0.90) -> StatForecast:
    """Fit a classical time-series model and project `periods` business days,
    with an analytic prediction interval at confidence `ci`."""
    s = series.dropna().astype(float)
    z = float(_norm_ppf(0.5 + ci / 2.0))

    if model_type == "ARIMA":
        from statsmodels.tsa.arima.model import ARIMA
        fit = ARIMA(s, order=(1, 1, 1)).fit()
        fc = fit.get_forecast(periods)
        mean = np.asarray(fc.predicted_mean)
        se = np.asarray(fc.se_mean)
        lower, upper = mean - z * se, mean + z * se
    else:  # Exponential Smoothing (Holt, damped)
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        fit = ExponentialSmoothing(s, trend="add", damped_trend=True, seasonal=None).fit()
        mean = np.asarray(fit.forecast(periods))
        # ETS has no closed-form SE here — widen by residual std × √step.
        resid = np.asarray(s) - np.asarray(fit.fittedvalues)
        sigma = float(np.nanstd(resid))
        steps = np.sqrt(np.arange(1, periods + 1))
        lower, upper = mean - z * sigma * steps, mean + z * sigma * steps

    idx = pd.bdate_range(s.index[-1] + pd.Timedelta(days=1), periods=periods)
    return StatForecast(
        model=model_type,
        forecast=pd.Series(mean, index=idx, name="Forecast"),
        lower=pd.Series(lower, index=idx, name="Lower"),
        upper=pd.Series(upper, index=idx, name="Upper"),
    )


def _norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF (Acklam's rational approximation) — avoids a
    SciPy dependency for the handful of quantiles we need."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = np.sqrt(-2 * np.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = np.sqrt(-2 * np.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


# ══════════════════════════════════════════════════════════════════
# FORECAST ACCURACY BACKTEST  (rolling out-of-sample evaluation)
# ══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class AccuracyResult:
    model: str
    horizon: int
    n_windows: int
    mape: float                   # mean abs % error of the h-step point forecast
    rmse_pct: float               # RMS % error
    directional_hit: float        # P(sign of forecast move == sign of realised)
    naive_mape: float             # random-walk benchmark (forecast = last price)
    skill: float                  # 1 − model_mape / naive_mape  (>0 = beats RW)
    cone_coverage: float          # share of realised terminals inside the 90% MC cone
    errors: np.ndarray            # per-window % errors (for the histogram)
    pred: np.ndarray              # forecast terminal prices
    actual: np.ndarray            # realised terminal prices


def _quick_cone_bounds(log_r: np.ndarray, last: float, horizon: int,
                       rng, lo: float = 5, hi: float = 95, n: int = 1200) -> Tuple[float, float]:
    """Light Student-t Monte Carlo terminal band for calibration (cheaper than
    the full 8k-path engine but the same shape: EWMA vol, fat tails, shrunk drift)."""
    mu = float(log_r.mean()) * _DRIFT_SHRINK
    sig = _ewma_vol(log_r)
    eps = rng.standard_t(_T_DOF, size=(n, horizon)) * np.sqrt((_T_DOF - 2) / _T_DOF)
    terminal = last * np.exp(np.cumsum(mu + sig * eps, axis=1))[:, -1]
    return float(np.percentile(terminal, lo)), float(np.percentile(terminal, hi))


@st.cache_data(ttl=1800, show_spinner=False)
def backtest_forecast_accuracy(closes: pd.Series, horizon: int, model_type: str,
                               n_windows: int = 40, seed: int = 20260614) -> Optional[AccuracyResult]:
    """Rolling-origin evaluation: step back through history, forecast `horizon`
    days ahead from each origin using only prior data, and score against what
    actually happened. Also measures how often the realised price landed inside
    the Monte-Carlo 90% cone (probabilistic calibration)."""
    s = closes.dropna().astype(float)
    min_train = 120
    last_origin = len(s) - horizon - 1
    if last_origin <= min_train:
        return None

    origins = np.linspace(min_train, last_origin, num=min(n_windows, last_origin - min_train),
                          dtype=int)
    origins = np.unique(origins)
    rng = np.random.default_rng(seed)

    preds, actuals, origin_px, naive_err, cone_in = [], [], [], [], 0
    for o in origins:
        train = s.iloc[:o + 1]
        actual = float(s.iloc[o + horizon])
        last = float(train.iloc[-1])

        try:
            if model_type == "ARIMA":
                from statsmodels.tsa.arima.model import ARIMA
                pred = float(np.asarray(ARIMA(train, order=(1, 1, 1)).fit().forecast(horizon))[-1])
            elif model_type == "Exponential Smoothing":
                from statsmodels.tsa.holtwinters import ExponentialSmoothing
                pred = float(np.asarray(ExponentialSmoothing(train, trend="add", damped_trend=True,
                                                             seasonal=None).fit().forecast(horizon))[-1])
            else:  # Monte Carlo median
                lr = np.diff(np.log(train.to_numpy(dtype=float)))
                lr = lr[np.isfinite(lr)]
                pred = last * float(np.exp((lr.mean() * _DRIFT_SHRINK) * horizon))
        except Exception:
            continue

        preds.append(pred)
        actuals.append(actual)
        origin_px.append(last)
        naive_err.append(abs(actual - last) / actual * 100)

        lr = np.diff(np.log(train.to_numpy(dtype=float)))
        lr = lr[np.isfinite(lr)]
        if len(lr) > 20:
            lo, hi = _quick_cone_bounds(lr, last, horizon, rng)
            if lo <= actual <= hi:
                cone_in += 1

    if len(preds) < 5:
        return None

    preds = np.array(preds)
    actuals = np.array(actuals)
    errs = (preds - actuals) / actuals * 100
    mape = float(np.mean(np.abs(errs)))
    rmse = float(np.sqrt(np.mean(errs ** 2)))
    naive_mape = float(np.mean(naive_err)) or 1e-9

    # Directional hit: did the forecast get the *direction* right vs the origin?
    origin_px = np.array(origin_px)
    directional = float(np.mean(np.sign(preds - origin_px) == np.sign(actuals - origin_px)) * 100)

    return AccuracyResult(
        model=model_type,
        horizon=horizon,
        n_windows=len(preds),
        mape=mape,
        rmse_pct=rmse,
        directional_hit=directional,
        naive_mape=naive_mape,
        skill=float(1 - mape / naive_mape),
        cone_coverage=cone_in / len(preds) * 100,
        errors=errs,
        pred=preds,
        actual=actuals,
    )
