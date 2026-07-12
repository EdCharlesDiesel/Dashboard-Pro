"""
quant_models.py
================
Six quant models for the trading dashboard, designed to slot into the
existing Streamlit + PostgreSQL architecture (same style as confluence_engine.py).

Models:
    1. Ornstein-Uhlenbeck  -> mean reversion fit, half-life, z-score bands
    2. GARCH(1,1)          -> volatility forecast, vol regime
    3. GBM Monte Carlo     -> null model / significance test for any signal
    4. UIP / Carry         -> rate differential vs realized FX return deviation
    5. Kalman Filter       -> dynamic (time-varying) alpha & beta between two series
    6. Engle-Granger       -> cointegration test + error-correction speed

All functions take pandas Series/DataFrames and return plain dicts or
DataFrames so results can be persisted to PostgreSQL or fed into the
confluence scoring engine.

Dependencies: numpy, pandas, scipy, statsmodels (required), arch (optional —
only `fit_garch`/`full_pair_analysis`'s GARCH overlay need it; everything
else works without it, same self-disabling pattern as ai_reports.py).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant


# ----------------------------------------------------------------------------
# 1. ORNSTEIN-UHLENBECK  (mean reversion)
# ----------------------------------------------------------------------------
@dataclass
class OUFit:
    theta: float          # speed of mean reversion (per bar)
    mu: float             # long-run mean
    sigma: float          # instantaneous volatility of the process
    sigma_eq: float       # stationary (equilibrium) std dev of the process
    half_life: float      # bars until half of a deviation decays
    zscore: float         # current z-score vs long-run mean
    ar1_beta: float       # AR(1) slope (sanity: must be < 1 for mean reversion)
    r_squared: float      # fit quality of the AR(1) regression
    n_obs: int

    def to_dict(self) -> dict:
        return asdict(self)


def fit_ou(series: pd.Series, dt: float = 1.0) -> OUFit:
    """
    Fit an Ornstein-Uhlenbeck process to a (presumed stationary) series --
    e.g. the residuals from your rolling OLS disconnect monitor, or a
    cointegration spread. NOT raw price.

        dX = theta * (mu - X) dt + sigma dW

    Discretised, this is an AR(1):  X_{t+1} = a + b * X_t + eps
        b     = exp(-theta * dt)     ->  theta = -ln(b) / dt
        a     = mu * (1 - b)         ->  mu    = a / (1 - b)
        sigma = std(eps) * sqrt( 2*theta / (1 - b^2) )

    half_life = ln(2) / theta   (your expected holding period)
    """
    x = series.dropna().values
    if len(x) < 30:
        raise ValueError("Need at least 30 observations to fit OU")

    x_lag, x_now = x[:-1], x[1:]
    X = add_constant(x_lag)
    model = OLS(x_now, X).fit()
    a, b = model.params

    if b <= 0 or b >= 1:
        # No mean reversion detected -- return degenerate fit with inf half-life
        return OUFit(theta=0.0, mu=float(np.mean(x)), sigma=float(np.std(x)),
                     sigma_eq=float(np.std(x)), half_life=np.inf,
                     zscore=float((x[-1] - np.mean(x)) / (np.std(x) + 1e-12)),
                     ar1_beta=float(b), r_squared=float(model.rsquared),
                     n_obs=len(x))

    theta = -np.log(b) / dt
    mu = a / (1.0 - b)
    resid_std = np.std(model.resid, ddof=2)
    sigma = resid_std * np.sqrt(2.0 * theta / (1.0 - b ** 2))
    sigma_eq = sigma / np.sqrt(2.0 * theta)          # stationary std dev
    half_life = np.log(2.0) / theta
    z = (x[-1] - mu) / (sigma_eq + 1e-12)

    return OUFit(theta=float(theta), mu=float(mu), sigma=float(sigma),
                 sigma_eq=float(sigma_eq), half_life=float(half_life),
                 zscore=float(z), ar1_beta=float(b),
                 r_squared=float(model.rsquared), n_obs=len(x))


def ou_signal(fit: OUFit, entry_z: float = 2.0, exit_z: float = 0.5,
              max_half_life: float = 60.0) -> dict:
    """
    Turn an OU fit into a discrete signal for the confluence engine.

    Returns dict with:
        signal:  -1 (short the spread), 0 (flat), +1 (long the spread)
        score:   continuous -1..+1 (negative z -> long expected)
        tradeable: False if half-life too long or no mean reversion
    """
    tradeable = bool(np.isfinite(fit.half_life) and fit.half_life <= max_half_life)
    signal = 0
    if tradeable:
        if fit.zscore >= entry_z:
            signal = -1                       # spread rich -> expect fall
        elif fit.zscore <= -entry_z:
            signal = +1                       # spread cheap -> expect rise
    # continuous score, clipped: z of -2 -> +1.0 (bullish spread)
    score = float(np.clip(-fit.zscore / entry_z, -1, 1)) if tradeable else 0.0
    return {"signal": signal, "score": score, "tradeable": tradeable,
            "zscore": fit.zscore, "half_life": fit.half_life,
            "exit_z": exit_z}


# ----------------------------------------------------------------------------
# 2. GARCH(1,1)  (volatility forecasting)
# ----------------------------------------------------------------------------
def fit_garch(returns: pd.Series, horizon: int = 5,
              ann_factor: int = 252) -> dict:
    """
    Fit GARCH(1,1) on percentage returns and forecast volatility.

        sigma^2_t = omega + alpha * eps^2_{t-1} + beta * sigma^2_{t-1}

    Input:  daily returns as decimals (0.001 = 0.1%). Rescaled x100
            internally, as the arch package prefers.
    Output: next-`horizon` daily vol forecasts (annualised %), persistence
            (alpha+beta), and current vol percentile vs its own history.

    Uses:
        - position sizing:  size ~ target_risk / forecast_vol
        - stop distance:    stop ~ k * forecast_daily_vol
        - regime filter:    percentile > 80 -> breakouts more credible,
                            mean reversion riskier
    """
    try:
        from arch import arch_model
    except ImportError as exc:
        raise ImportError(
            "fit_garch() needs the 'arch' package: pip install arch"
        ) from exc

    r = returns.dropna() * 100.0                       # arch likes % units
    if len(r) < 250:
        raise ValueError("Need ~1y of daily returns for a stable GARCH fit")

    am = arch_model(r, vol="GARCH", p=1, q=1, mean="Constant", dist="t")
    res = am.fit(disp="off")

    omega = float(res.params["omega"])
    alpha = float(res.params["alpha[1]"])
    beta = float(res.params["beta[1]"])
    persistence = alpha + beta

    # conditional (in-sample) vol history, daily %
    cond_vol = pd.Series(res.conditional_volatility, index=r.index)
    current_vol = float(cond_vol.iloc[-1])
    vol_percentile = float(stats.percentileofscore(cond_vol, current_vol))

    # h-step ahead forecasts
    fc = res.forecast(horizon=horizon, reindex=False)
    var_path = fc.variance.values[-1]                  # daily variance, %^2
    vol_path_daily = np.sqrt(var_path)                 # daily vol %
    vol_path_annual = vol_path_daily * np.sqrt(ann_factor)

    # long-run (unconditional) annualised vol implied by the fit
    lr_daily_var = omega / max(1e-12, (1.0 - persistence))
    lr_annual_vol = float(np.sqrt(lr_daily_var * ann_factor))

    return {
        "omega": omega, "alpha": alpha, "beta": beta,
        "persistence": persistence,
        "current_vol_daily_pct": current_vol,
        "current_vol_annual_pct": current_vol * np.sqrt(ann_factor),
        "vol_percentile": vol_percentile,
        "forecast_vol_annual_pct": [float(v) for v in vol_path_annual],
        "long_run_vol_annual_pct": lr_annual_vol,
        "regime": ("high" if vol_percentile > 80 else
                   "low" if vol_percentile < 20 else "normal"),
    }


def garch_position_size(account_risk_pct: float,
                        forecast_daily_vol_pct: float,
                        stop_vol_multiple: float = 2.0) -> dict:
    """
    Vol-targeted sizing: place the stop at k * forecast daily vol and size
    so that a stop-out loses exactly account_risk_pct.

        stop_distance_pct = stop_vol_multiple * forecast_daily_vol_pct
        position_notional = account_risk_pct / stop_distance_pct   (as leverage)
    """
    stop_pct = stop_vol_multiple * forecast_daily_vol_pct
    leverage = account_risk_pct / max(1e-9, stop_pct)
    return {"stop_distance_pct": stop_pct, "notional_leverage": leverage}


# ----------------------------------------------------------------------------
# 3. GBM MONTE CARLO  (null model / significance testing)
# ----------------------------------------------------------------------------
def gbm_null_test(prices: pd.Series,
                  observed_metric: float,
                  metric_fn,
                  n_sims: int = 2000,
                  seed: Optional[int] = 42) -> dict:
    """
    Test whether a signal's performance beats pure luck under GBM.

    You provide:
        prices          : the actual price series the signal was tested on
        observed_metric : the number your backtest produced (Sharpe, mean
                          overnight return, hit rate, ...)
        metric_fn       : function(price_series) -> float, the SAME metric
                          computed on a simulated series

    We estimate drift/vol from the real series, simulate n_sims GBM paths of
    equal length, compute metric_fn on each, and report where the observed
    value sits in the null distribution.

        p_value < 0.05  -> unlikely to be luck (at least vs the GBM null)

    Note: GBM has no autocorrelation, fat tails or vol clustering, so this is
    a WEAK null -- failing it is damning, passing it is encouraging but not
    proof. For a harder test, use block-bootstrap of real returns.
    """
    p = prices.dropna().values
    log_ret = np.diff(np.log(p))
    mu = log_ret.mean()
    sigma = log_ret.std(ddof=1)
    n = len(p)

    rng = np.random.default_rng(seed)
    null_metrics = np.empty(n_sims)
    for i in range(n_sims):
        z = rng.standard_normal(n - 1)
        sim_log = np.concatenate([[np.log(p[0])],
                                  np.log(p[0]) + np.cumsum(mu + sigma * z)])
        sim_prices = pd.Series(np.exp(sim_log), index=prices.dropna().index)
        null_metrics[i] = metric_fn(sim_prices)

    pct = float(stats.percentileofscore(null_metrics, observed_metric))
    p_value = 1.0 - pct / 100.0        # one-sided: "is observed unusually high?"

    return {
        "observed": observed_metric,
        "null_mean": float(null_metrics.mean()),
        "null_std": float(null_metrics.std(ddof=1)),
        "null_p95": float(np.percentile(null_metrics, 95)),
        "percentile": pct,
        "p_value_one_sided": p_value,
        "beats_null": p_value < 0.05,
        "n_sims": n_sims,
    }


# ----------------------------------------------------------------------------
# 4. UIP / CARRY  (rate differential vs realized return)
# ----------------------------------------------------------------------------
def uip_carry_analysis(spot: pd.Series,
                       rate_domestic: pd.Series,
                       rate_foreign: pd.Series,
                       window: int = 63) -> pd.DataFrame:
    """
    Uncovered Interest Parity says expected depreciation of the domestic
    currency equals the rate differential:

        E[ds] = i_dom - i_for        (annualised, s = log spot)

    Empirically it FAILS at short horizons (the 'forward premium puzzle') --
    high-yield currencies tend to appreciate. The deviation IS the carry trade.

    Inputs:
        spot          : price of the pair, quoted as DOM/FOR units
                        (e.g. USDZAR: domestic=ZAR rate, foreign=USD rate)
        rate_domestic : annualised policy/short rate of quote currency, in %
        rate_foreign  : annualised rate of base currency, in %
        window        : rolling window (63 = one quarter) for realized moves

    Output columns:
        rate_diff_ann      : i_dom - i_for (%)
        uip_expected_ret   : what UIP predicts for spot over the window (%)
        realized_ret       : actual spot change over the window (%)
        carry_deviation    : realized - UIP-expected. Persistent sign here is
                             the carry premium your dashboard can score.
        carry_z            : z-score of current deviation vs its own history
    """
    df = pd.DataFrame({"spot": spot,
                       "i_dom": rate_domestic,
                       "i_for": rate_foreign}).dropna()

    ann_factor = 252.0
    df["rate_diff_ann"] = df["i_dom"] - df["i_for"]
    # UIP-implied expected % change of spot over `window` days
    df["uip_expected_ret"] = df["rate_diff_ann"] * (window / ann_factor)
    df["realized_ret"] = (df["spot"].shift(-window) / df["spot"] - 1.0) * 100.0
    df["carry_deviation"] = df["realized_ret"] - df["uip_expected_ret"]

    dev = df["carry_deviation"]
    df["carry_z"] = (dev - dev.expanding(min_periods=window).mean()) / \
                    (dev.expanding(min_periods=window).std() + 1e-12)
    return df


# ----------------------------------------------------------------------------
# 5. KALMAN FILTER  (dynamic alpha/beta -- upgrade of rolling OLS)
# ----------------------------------------------------------------------------
def kalman_dynamic_beta(y: pd.Series, x: pd.Series,
                        delta: float = 1e-4,
                        obs_var: Optional[float] = None) -> pd.DataFrame:
    """
    Time-varying regression  y_t = alpha_t + beta_t * x_t + v_t
    where the state [alpha_t, beta_t] follows a random walk.

    This replaces rolling-window OLS: no arbitrary lookback, adapts
    continuously, and produces a residual (innovation) series you can feed
    straight into fit_ou().

    delta controls how fast the state can move (bigger = faster adaptation,
    noisier beta). 1e-4 to 1e-5 is a sensible starting range for daily data.

    Returns DataFrame indexed like the inputs with:
        alpha, beta        : filtered state estimates
        innovation         : y_t - predicted y_t   (THE spread to trade)
        innovation_var     : predicted variance of the innovation
        innovation_z       : standardised innovation (instant z-score)
    """
    df = pd.DataFrame({"y": y, "x": x}).dropna()
    n = len(df)
    yv, xv = df["y"].values, df["x"].values

    if obs_var is None:
        # crude initial estimate of observation noise from a static OLS
        X0 = add_constant(xv)
        obs_var = float(np.var(OLS(yv, X0).fit().resid, ddof=2))

    # State: [alpha, beta]; random-walk transition
    state = np.zeros(2)
    P = np.eye(2) * 1.0                    # state covariance (diffuse-ish start)
    Q = np.eye(2) * delta                  # state noise
    R = obs_var                            # observation noise

    alphas = np.empty(n); betas = np.empty(n)
    innov = np.empty(n); innov_var = np.empty(n)

    for t in range(n):
        H = np.array([1.0, xv[t]])         # observation matrix

        # Predict
        P = P + Q
        y_hat = H @ state
        S = H @ P @ H + R                  # innovation variance
        e = yv[t] - y_hat                  # innovation

        # Update
        K = (P @ H) / S                    # Kalman gain
        state = state + K * e
        P = P - np.outer(K, H @ P)

        alphas[t], betas[t] = state
        innov[t], innov_var[t] = e, S

    out = pd.DataFrame({
        "alpha": alphas, "beta": betas,
        "innovation": innov,
        "innovation_var": innov_var,
        "innovation_z": innov / np.sqrt(innov_var),
    }, index=df.index)
    return out


# ----------------------------------------------------------------------------
# 6. ENGLE-GRANGER COINTEGRATION + ERROR CORRECTION
# ----------------------------------------------------------------------------
def engle_granger(y: pd.Series, x: pd.Series) -> dict:
    """
    Two-step Engle-Granger:
      Step 1: OLS  y = c + gamma * x + resid   -> hedge ratio gamma
      Step 2: ADF test on resid. Stationary residuals => cointegrated.
      Step 3: Error-correction speed:  dy_t = alpha * resid_{t-1} + e
              alpha < 0 and significant => y actively corrects toward
              equilibrium; |alpha| is the correction speed per bar.

    Trade only pairs where adf_pvalue < 0.05 AND alpha is negative/significant.
    The residual series is what you hand to fit_ou() for z-bands & half-life.
    """
    df = pd.DataFrame({"y": y, "x": x}).dropna()
    X = add_constant(df["x"].values)
    step1 = OLS(df["y"].values, X).fit()
    const, gamma = step1.params
    resid = pd.Series(step1.resid, index=df.index, name="spread")

    adf_stat, adf_p, *_ = adfuller(resid.values, autolag="AIC")

    # error-correction regression
    d_y = df["y"].diff().dropna()
    lag_resid = resid.shift(1).dropna()
    common = d_y.index.intersection(lag_resid.index)
    ecm = OLS(d_y.loc[common].values,
              add_constant(lag_resid.loc[common].values)).fit()
    alpha_ec = float(ecm.params[1])
    alpha_t = float(ecm.tvalues[1])

    return {
        "hedge_ratio": float(gamma),
        "intercept": float(const),
        "adf_stat": float(adf_stat),
        "adf_pvalue": float(adf_p),
        "cointegrated_5pct": bool(adf_p < 0.05),
        "ec_alpha": alpha_ec,               # want negative
        "ec_alpha_tstat": alpha_t,          # want < -2
        "spread": resid,                    # feed this to fit_ou()
    }


# ----------------------------------------------------------------------------
# COMBINED: one call that chains 6 -> 5 -> 1 -> 2 into a confluence dict
# ----------------------------------------------------------------------------
def full_pair_analysis(y: pd.Series, x: pd.Series,
                       returns_for_garch: Optional[pd.Series] = None,
                       entry_z: float = 2.0) -> dict:
    """
    The whole pipeline for a candidate pair (e.g. gold vs real yields,
    AUDUSD vs copper):

        1. Engle-Granger: are they cointegrated at all? (gate)
        2. Kalman filter: dynamic beta + innovation z-score (fast signal)
        3. OU fit on the EG spread: half-life + z-bands (slow signal)
        4. GARCH on y's returns: vol regime for sizing (risk overlay)

    Returns a flat dict ready for the confluence engine / PostgreSQL.
    """
    eg = engle_granger(y, x)
    kf = kalman_dynamic_beta(y, x)
    ou = fit_ou(eg["spread"])
    sig = ou_signal(ou, entry_z=entry_z)

    out = {
        # gates
        "cointegrated": eg["cointegrated_5pct"],
        "adf_pvalue": eg["adf_pvalue"],
        "ec_alpha": eg["ec_alpha"],
        "ec_alpha_tstat": eg["ec_alpha_tstat"],
        # structure
        "hedge_ratio_static": eg["hedge_ratio"],
        "hedge_ratio_kalman": float(kf["beta"].iloc[-1]),
        # signals
        "ou_zscore": ou.zscore,
        "ou_half_life": ou.half_life,
        "kalman_innovation_z": float(kf["innovation_z"].iloc[-1]),
        "signal": sig["signal"],
        "score": sig["score"],
        "tradeable": sig["tradeable"] and eg["cointegrated_5pct"],
    }

    if returns_for_garch is not None and len(returns_for_garch.dropna()) >= 250:
        g = fit_garch(returns_for_garch)
        out.update({
            "vol_regime": g["regime"],
            "vol_percentile": g["vol_percentile"],
            "forecast_vol_annual_pct": g["forecast_vol_annual_pct"][0],
        })

    return out