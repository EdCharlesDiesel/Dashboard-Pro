"""
Stochastic calculus toolkit for price series.

Pure numpy/scipy/pandas. No Streamlit, no DB.

Each block is one chapter of the standard construction, turned into something
you can point at a real instrument:

    Ch 1  Brownian motion, Var(W_t) = t   ->  does realised variance actually
                                              scale linearly in horizon?
    Ch 2  quadratic variation [W,W]_t = t ->  realised volatility estimators;
                                              the signature plot that exposes
                                              microstructure noise
    Ch 3  Ito's lemma, mu - sigma^2/2     ->  volatility drag, optimal leverage,
                                              first-passage (target vs stop)
    Ch 4  Black-Scholes                   ->  pricing, Greeks, implied vol

Companion to concentration.py: `barrier_probabilities` with zero log-drift
reproduces the gambler's-ruin ratio l/(u+l), which is the continuous-time
statement of the same optional-stopping result.

References
----------
Yang & Zhang (2000), J. Business 73(3), 477-491.
Rogers & Satchell (1991), Ann. Appl. Probab. 1(4), 504-512.
Karatzas & Shreve, *Brownian Motion and Stochastic Calculus*, ch. 5 (scale
    function and first-passage).
Glasserman, *Monte Carlo Methods in Financial Engineering*, sec. 6.4
    (Brownian-bridge barrier correction).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from scipy import optimize, stats

__all__ = [
    "GBMParams",
    "log_returns",
    "realised_variance",
    "signature_plot",
    "variance_scaling",
    "volatility_estimators",
    "calibrate_gbm",
    "kelly_leverage",
    "barrier_probabilities",
    "simulate_gbm",
    "black_scholes",
    "implied_vol",
]

SQRT_2PI = np.sqrt(2.0 * np.pi)


def _phi(x):
    """Standard normal pdf."""
    return np.exp(-0.5 * np.asarray(x, float) ** 2) / SQRT_2PI


def _Phi(x):
    """Standard normal cdf."""
    return stats.norm.cdf(x)


def _as_1d(x, name="x") -> np.ndarray:
    a = np.asarray(x, dtype=float).ravel()
    if a.size == 0:
        raise ValueError(f"{name} is empty")
    return a


# --------------------------------------------------------------------------
# Ch 1-2: quadratic variation and realised volatility
# --------------------------------------------------------------------------

def log_returns(prices: Sequence[float] | np.ndarray) -> np.ndarray:
    """Log returns, with a check that prices are strictly positive."""
    p = _as_1d(prices, "prices")
    if (p <= 0).any():
        raise ValueError("prices must be strictly positive to take logs")
    return np.diff(np.log(p))


def realised_variance(returns: Sequence[float], annualise: int | None = None) -> float:
    """Sum of squared log returns — the empirical [X, X]_T.

    This is the *only* volatility estimator that follows directly from the
    theory: quadratic variation is a pathwise limit, not a statistical
    estimate, so it needs no assumption about the mean. Note it does not
    subtract a sample mean, deliberately.

    annualise : periods per year; if given, returns an annualised variance.
    """
    r = _as_1d(returns, "returns")
    rv = float(np.sum(r**2))
    return rv * (annualise / r.size) if annualise else rv


def _rv_curve(logp: np.ndarray, max_skip: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Realised variance per period at each sampling interval."""
    skips, rvs, ns = [], [], []
    for k in range(1, int(max_skip) + 1):
        r = logp[k:] - logp[:-k]
        if r.size < 5:
            break
        skips.append(k)
        rvs.append(np.sum(r**2) / (r.size * k))
        ns.append(r.size)
    return np.array(skips, float), np.array(rvs), np.array(ns)


def _noise_fit(skips: np.ndarray, rvs: np.ndarray) -> tuple[float, float]:
    """OLS of RV_per_period on 1/k -> (intercept, slope)."""
    x = 1.0 / skips
    A = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(A, rvs, rcond=None)
    return float(beta[0]), float(beta[1])


def signature_plot(
    prices: Sequence[float],
    max_skip: int = 30,
    n_boot: int = 0,
    block: int | None = None,
    seed: int | None = 11,
) -> tuple[pd.DataFrame, dict]:
    """Realised variance vs sampling interval, with a microstructure-noise fit.

    Chapter 2 says refining the partition makes the sum of squared increments
    converge to a constant. Real price series do not do this. If the observed
    log price is the efficient price plus i.i.d. microstructure noise of
    variance q (bid-ask bounce, tick rounding, stale quotes), then realised
    variance measured at sampling interval k satisfies

        RV_per_period(k)  =  true_var  +  2q / k

    so the curve inflates as k -> 1 and flattens as k grows. Regressing the
    measured curve on 1/k separates the two: the intercept estimates the
    noise-free variance and half the slope estimates q.

    Two warnings that matter more than the formula.

    First, this is an *intraday* diagnostic. At daily sampling, bid-ask noise
    is a negligible fraction of per-bar variance and the fit will return
    essentially nothing — correctly. Point it at tick or minute data.

    Second, the rows of this curve are not independent observations: they come
    from one path via overlapping windows, so textbook regression standard
    errors are wildly overconfident, and a single sample path carries a random
    tilt of its own. Inference therefore comes from a moving-block bootstrap
    on the log returns (n_boot > 0) or not at all. With n_boot = 0 the fit is
    reported as a point estimate with no significance claim.

    Returns (curve, fit).
    """
    p = _as_1d(prices, "prices")
    if p.size < max_skip * 3:
        max_skip = max(2, p.size // 3)

    logp = np.log(p)
    skips, rvs, ns = _rv_curve(logp, max_skip)
    curve = pd.DataFrame({"skip": skips.astype(int), "n_obs": ns.astype(int),
                          "rv_per_period": rvs})
    if len(curve) < 4:
        return curve, {}

    intercept, slope = _noise_fit(skips, rvs)
    curve["fitted"] = intercept + slope / skips

    noise_var = max(slope / 2.0, 0.0)
    true_var = max(intercept, 1e-18)
    fit = {
        "noise_free_var_per_period": float(true_var),
        "noise_var": float(noise_var),
        "implied_noise_sd": float(np.sqrt(noise_var)),
        "inflation_at_skip1": float(rvs[0] / true_var),
        "noise_share_at_skip1": float(2 * noise_var / (true_var + 2 * noise_var)),
        "slope": float(slope),
        "n_boot": int(n_boot),
    }

    if n_boot <= 0:
        fit.update({"slope_ci": None, "significant": None,
                    "note": "no inference — set n_boot > 0 for a bootstrap CI"})
        return curve, fit

    # moving-block bootstrap on log returns, preserving short-range dependence
    r = np.diff(logp)
    N = r.size
    b = int(block or max(10, round(N ** (1 / 3))))
    b = min(b, max(2, N // 4))
    n_blocks = int(np.ceil(N / b))
    starts_pool = N - b + 1
    rng = np.random.default_rng(seed)

    boot = np.empty(n_boot)
    for i in range(n_boot):
        st = rng.integers(0, starts_pool, size=n_blocks)
        idx = (st[:, None] + np.arange(b)[None, :]).ravel()[:N]
        lp = np.concatenate(([0.0], np.cumsum(r[idx])))
        sk, rv, _ = _rv_curve(lp, max_skip)
        boot[i] = _noise_fit(sk, rv)[1]

    lo, hi = np.percentile(boot, [2.5, 97.5])
    fit.update({
        "slope_ci": (float(lo), float(hi)),
        "slope_boot_sd": float(boot.std(ddof=1)),
        "significant": bool(lo > 0),
        "note": "significant = bootstrap 95% CI for the slope lies above zero",
    })
    return curve, fit


def variance_scaling(prices: Sequence[float], horizons=(1, 2, 3, 5, 8, 13, 21)) -> dict:
    """Fit Var(r_h) ~ h^(2H) and report the scaling exponent.

    Brownian motion forces H = 0.5 — variance linear in horizon. H > 0.5 means
    variance grows faster than time (trend persistence); H < 0.5 means slower
    (mean reversion). This is the continuous-time twin of the variance-ratio
    test in concentration.py, and it is the assumption you are implicitly
    making every time you scale a daily vol by sqrt(21) to get a monthly one.
    """
    p = _as_1d(prices, "prices")
    logp = np.log(p)
    hs, vs = [], []
    for h in horizons:
        h = int(h)
        if p.size < 4 * h + 10:
            continue
        r = logp[h:] - logp[:-h]
        hs.append(h)
        vs.append(float(np.var(r, ddof=1)))

    if len(hs) < 3:
        raise ValueError("not enough data for a scaling fit")

    x, y = np.log(hs), np.log(vs)
    slope, intercept, r_val, p_val, stderr = stats.linregress(x, y)
    H = slope / 2.0
    return {
        "hurst": float(H),
        "hurst_stderr": float(stderr / 2.0),
        "slope": float(slope),
        "r_squared": float(r_val**2),
        "p_value": float(p_val),
        "horizons": hs,
        "variances": vs,
        "interpretation": (
            "trending / persistent" if H > 0.55 else
            "mean-reverting / anti-persistent" if H < 0.45 else
            "consistent with Brownian scaling"
        ),
    }


def volatility_estimators(
    df: pd.DataFrame, periods_per_year: int = 252, window: int | None = None
) -> dict:
    """Annualised volatility by five estimators, from OHLC.

    Close-to-close throws away everything that happened intraday. The range
    estimators do not, and are several times more efficient per observation
    for the same sample. Yang-Zhang is the one to prefer by default: it is the
    only one here that handles both overnight gaps and intraday drift.

    df must have columns open, high, low, close (case-insensitive).
    """
    d = df.rename(columns=str.lower)
    need = {"open", "high", "low", "close"}
    missing = need - set(d.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if window:
        d = d.tail(int(window))
    d = d.dropna(subset=list(need))
    n = len(d)
    if n < 5:
        raise ValueError(f"need at least 5 bars, got {n}")

    O, H, L, C = (d[c].to_numpy(float) for c in ("open", "high", "low", "close"))
    ann = float(periods_per_year)

    # close-to-close
    rc = np.diff(np.log(C))
    v_cc = np.var(rc, ddof=1)

    # Parkinson — high-low range only
    hl = np.log(H / L)
    v_park = np.sum(hl**2) / (4.0 * n * np.log(2.0))

    # Garman-Klass — range plus open-close
    co = np.log(C / O)
    v_gk = np.mean(0.5 * hl**2 - (2.0 * np.log(2.0) - 1.0) * co**2)

    # Rogers-Satchell — drift-independent
    v_rs = np.mean(np.log(H / C) * np.log(H / O) + np.log(L / C) * np.log(L / O))

    # Yang-Zhang — overnight + open-to-close + Rogers-Satchell
    o_t = np.log(O[1:] / C[:-1])          # overnight jump
    c_t = co[1:]                           # open-to-close, aligned
    m = o_t.size
    if m > 1:
        v_o = np.var(o_t, ddof=1)
        v_c = np.var(c_t, ddof=1)
        k = 0.34 / (1.34 + (m + 1) / (m - 1))
        v_rs_al = np.mean(
            np.log(H[1:] / C[1:]) * np.log(H[1:] / O[1:])
            + np.log(L[1:] / C[1:]) * np.log(L[1:] / O[1:])
        )
        v_yz = v_o + k * v_c + (1.0 - k) * v_rs_al
    else:
        v_yz = v_rs

    clip = lambda v: float(np.sqrt(max(v, 0.0) * ann))
    return {
        "close_to_close": clip(v_cc),
        "parkinson": clip(v_park),
        "garman_klass": clip(v_gk),
        "rogers_satchell": clip(v_rs),
        "yang_zhang": clip(v_yz),
        "n_bars": int(n),
        "periods_per_year": int(periods_per_year),
    }


# --------------------------------------------------------------------------
# Ch 3: Ito's correction made concrete
# --------------------------------------------------------------------------

@dataclass
class GBMParams:
    """Calibrated dS = mu S dt + sigma S dW, all annualised."""

    mu: float             # arithmetic drift of the price
    sigma: float          # volatility
    nu: float             # log-drift = mu - sigma^2/2   <- the Ito correction
    drag: float           # sigma^2/2, the cost of volatility
    n_obs: int
    periods_per_year: int

    @property
    def kelly(self) -> float:
        """Growth-optimal leverage f* = mu / sigma^2."""
        return float(self.mu / self.sigma**2) if self.sigma > 0 else float("nan")

    def summary(self) -> str:
        return (
            f"mu={self.mu:.2%}  sigma={self.sigma:.2%}  "
            f"drag={self.drag:.2%}  log-drift nu={self.nu:.2%}  "
            f"Kelly f*={self.kelly:.2f}x"
        )


def calibrate_gbm(
    prices: Sequence[float], periods_per_year: int = 252
) -> GBMParams:
    """Fit GBM by maximum likelihood on log returns.

    The MLE on log returns estimates nu = mu - sigma^2/2 directly, since
    log S is arithmetic BM with that drift. Recovering mu means *adding* the
    correction back: mu = nu + sigma^2/2. Getting this backwards is the single
    most common way to overstate an expected return.
    """
    r = log_returns(prices)
    if r.size < 20:
        raise ValueError(f"need at least 20 returns, got {r.size}")
    ann = float(periods_per_year)

    sigma = float(np.std(r, ddof=1) * np.sqrt(ann))
    nu = float(np.mean(r) * ann)
    drag = 0.5 * sigma**2
    return GBMParams(
        mu=nu + drag, sigma=sigma, nu=nu, drag=drag,
        n_obs=int(r.size), periods_per_year=int(periods_per_year),
    )


def kelly_leverage(params: GBMParams, leverages=None) -> pd.DataFrame:
    """Log-growth rate as a function of leverage: g(f) = f*mu - f^2*sigma^2/2.

    Pure Ito. The parabola is the whole argument against over-leverage: past
    f* = mu/sigma^2 the quadratic drag term dominates the linear return term,
    and at f = 2f* expected log growth is back to zero no matter how good the
    edge is. Full Kelly is already aggressive; most practitioners size at a
    fraction of it because mu is estimated with far more error than sigma.
    """
    if leverages is None:
        top = max(3.0, 2.5 * abs(params.kelly)) if np.isfinite(params.kelly) else 5.0
        leverages = np.linspace(0.0, min(top, 20.0), 120)
    f = np.asarray(leverages, float)
    g = f * params.mu - 0.5 * f**2 * params.sigma**2
    return pd.DataFrame({"leverage": f, "log_growth": g})


# --------------------------------------------------------------------------
# Ch 3: first passage — target vs stop
# --------------------------------------------------------------------------

def barrier_probabilities(
    spot: float,
    take_profit: float,
    stop_loss: float,
    sigma: float,
    mu: float = 0.0,
    horizon_years: float | None = None,
    n_sims: int = 50_000,
    periods_per_year: int = 252,
    seed: int | None = 7,
) -> dict:
    """P(hit take-profit before stop-loss) for GBM, closed form and simulated.

    Working in log space, X = ln(S/S_0) is arithmetic BM with drift
    nu = mu - sigma^2/2 (the Ito correction again) and volatility sigma. With
    u = ln(TP/S_0) > 0 and l = -ln(SL/S_0) > 0, the scale function
    s(x) = exp(-2 nu x / sigma^2) gives

        P(TP first) = (1 - e^{2 nu l / sigma^2}) / (e^{-2 nu u / sigma^2} - e^{2 nu l / sigma^2})

    and when nu = 0 this collapses to the gambler's ruin ratio l / (u + l) —
    the same optional-stopping result as in concentration.py, now in
    continuous time. Note it is nu, not mu, that has to vanish: a positive
    arithmetic drift can still leave a negative log-drift once volatility is
    large enough, which is precisely the volatility drag.

    The closed form assumes an unlimited horizon. Passing horizon_years adds a
    Monte Carlo with a Brownian-bridge correction, which catches crossings
    that happen between simulation steps — without it a coarse grid
    systematically understates barrier-hit rates.
    """
    if spot <= 0:
        raise ValueError("spot must be positive")
    if take_profit <= spot:
        raise ValueError("take_profit must be above spot (long convention)")
    if not 0 < stop_loss < spot:
        raise ValueError("stop_loss must be between 0 and spot")
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    u = np.log(take_profit / spot)
    l = -np.log(stop_loss / spot)
    nu = mu - 0.5 * sigma**2
    s2 = sigma**2

    if abs(nu) < 1e-12:
        p_tp = l / (u + l)
    else:
        a = 2.0 * nu / s2
        # s(x) = exp(-a x); P = (s(0) - s(-l)) / (s(u) - s(-l))
        p_tp = (1.0 - np.exp(a * l)) / (np.exp(-a * u) - np.exp(a * l))
    p_tp = float(np.clip(p_tp, 0.0, 1.0))

    rr = (take_profit - spot) / (spot - stop_loss)
    ev_price = p_tp * (take_profit - spot) - (1 - p_tp) * (spot - stop_loss)
    ev_log = p_tp * u - (1 - p_tp) * l

    out = {
        "p_take_profit": p_tp,
        "p_stop_loss": 1.0 - p_tp,
        "reward_risk": float(rr),
        "breakeven_win_rate": float(1.0 / (1.0 + rr)),
        # Two different expectations, and they do not vanish together.
        # nu = 0 makes log-price a martingale, so expected_log_return is
        # exactly 0 — but by Jensen the price itself is then a *sub*martingale
        # and expected_value_price is strictly positive. Compounded returns
        # live in log space; the price figure is what hits the account.
        "expected_value_price": float(ev_price),
        "expected_log_return": float(ev_log),
        "log_drift_nu": float(nu),
        "arith_drift_mu": float(mu),
        "u_log": float(u),
        "l_log": float(l),
    }

    if horizon_years is None:
        return out

    # --- finite-horizon Monte Carlo with Brownian-bridge crossing correction
    n_steps = max(20, int(round(horizon_years * periods_per_year)))
    dt = horizon_years / n_steps
    rng = np.random.default_rng(seed)

    x = np.zeros(n_sims)
    hit_tp = np.zeros(n_sims, bool)
    hit_sl = np.zeros(n_sims, bool)
    exit_step = np.full(n_sims, n_steps, int)
    live = np.ones(n_sims, bool)

    sd = sigma * np.sqrt(dt)
    for step in range(n_steps):
        idx = np.flatnonzero(live)
        if idx.size == 0:
            break
        x0 = x[idx]
        x1 = x0 + nu * dt + sd * rng.standard_normal(idx.size)

        # endpoint crossings
        up = x1 >= u
        dn = x1 <= -l

        # bridge crossings between the endpoints (Glasserman 6.4)
        p_up = np.where(
            ~up & (x0 < u),
            np.exp(-2.0 * (u - x0) * (u - x1) / (s2 * dt)),
            0.0,
        )
        p_dn = np.where(
            ~dn & (x0 > -l),
            np.exp(-2.0 * (x0 + l) * (x1 + l) / (s2 * dt)),
            0.0,
        )
        rr_u = rng.random(idx.size)
        rr_d = rng.random(idx.size)
        up = up | (rr_u < p_up)
        dn = dn | (rr_d < p_dn)

        both = up & dn                      # tie-break by proximity
        if both.any():
            closer_up = (u - x0[both]) < (x0[both] + l)
            up_b, dn_b = up[both], dn[both]
            up_b[:] = closer_up
            dn_b[:] = ~closer_up
            up[both], dn[both] = up_b, dn_b

        x[idx] = x1
        done = up | dn
        if done.any():
            gidx = idx[done]
            hit_tp[gidx] = up[done]
            hit_sl[gidx] = dn[done]
            exit_step[gidx] = step + 1
            live[gidx] = False

    resolved = hit_tp | hit_sl
    out.update({
        "mc_p_take_profit": float(hit_tp.mean()),
        "mc_p_stop_loss": float(hit_sl.mean()),
        "mc_p_unresolved": float((~resolved).mean()),
        "mc_p_tp_given_resolved": float(hit_tp.sum() / max(resolved.sum(), 1)),
        "mc_mean_periods_to_exit": float(exit_step[resolved].mean()) if resolved.any() else float("nan"),
        "horizon_years": float(horizon_years),
        "n_sims": int(n_sims),
    })
    return out


def simulate_gbm(
    spot: float, mu: float, sigma: float, horizon_years: float,
    n_paths: int = 2000, periods_per_year: int = 252, seed: int | None = 7,
) -> np.ndarray:
    """Exact GBM paths via the closed-form solution (no Euler discretisation).

    S_t = S_0 exp[(mu - sigma^2/2) t + sigma W_t] is exact for any step size,
    so there is no discretisation bias in the marginal distribution — only in
    path-dependent quantities like barrier hits.
    """
    n_steps = max(2, int(round(horizon_years * periods_per_year)))
    dt = horizon_years / n_steps
    rng = np.random.default_rng(seed)
    nu = mu - 0.5 * sigma**2
    incr = nu * dt + sigma * np.sqrt(dt) * rng.standard_normal((n_paths, n_steps))
    return spot * np.exp(np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(incr, axis=1)], axis=1))


# --------------------------------------------------------------------------
# Ch 4: Black-Scholes
# --------------------------------------------------------------------------

def black_scholes(
    spot: float, strike: float, time_to_expiry: float, rate: float,
    sigma: float, kind: Literal["call", "put"] = "call", dividend: float = 0.0,
) -> dict:
    """European option price and Greeks under Black-Scholes.

    Note what is absent: the real-world drift mu. Two traders who disagree
    completely about direction must still quote the same price given the same
    sigma, because the delta hedge cancels the dW term and what remains has to
    earn r. Only sigma carries a forecast — which is why the entire options
    market is, in effect, a market in volatility opinions.
    """
    S, K, T, r, q = map(float, (spot, strike, time_to_expiry, rate, dividend))
    sig = float(sigma)
    if S <= 0 or K <= 0:
        raise ValueError("spot and strike must be positive")
    if kind not in ("call", "put"):
        raise ValueError("kind must be 'call' or 'put'")

    if T <= 0 or sig <= 0:  # degenerate: intrinsic value only
        intrinsic = max(S - K, 0.0) if kind == "call" else max(K - S, 0.0)
        return {"price": intrinsic, "delta": float(
            (S > K) if kind == "call" else -(S < K)), "gamma": 0.0,
            "vega": 0.0, "theta": 0.0, "rho": 0.0, "d1": np.nan, "d2": np.nan}

    srt = sig * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sig**2) * T) / srt
    d2 = d1 - srt
    dfq, dfr = np.exp(-q * T), np.exp(-r * T)

    if kind == "call":
        price = S * dfq * _Phi(d1) - K * dfr * _Phi(d2)
        delta = dfq * _Phi(d1)
        theta = (-S * dfq * _phi(d1) * sig / (2 * np.sqrt(T))
                 - r * K * dfr * _Phi(d2) + q * S * dfq * _Phi(d1))
        rho = K * T * dfr * _Phi(d2)
    else:
        price = K * dfr * _Phi(-d2) - S * dfq * _Phi(-d1)
        delta = -dfq * _Phi(-d1)
        theta = (-S * dfq * _phi(d1) * sig / (2 * np.sqrt(T))
                 + r * K * dfr * _Phi(-d2) - q * S * dfq * _Phi(-d1))
        rho = -K * T * dfr * _Phi(-d2)

    return {
        "price": float(price),
        "delta": float(delta),
        "gamma": float(dfq * _phi(d1) / (S * srt)),
        "vega": float(S * dfq * _phi(d1) * np.sqrt(T)),   # per 1.00 of vol
        "theta": float(theta),                            # per year
        "rho": float(rho),
        "d1": float(d1),
        "d2": float(d2),
    }


def implied_vol(
    price: float, spot: float, strike: float, time_to_expiry: float,
    rate: float, kind: Literal["call", "put"] = "call", dividend: float = 0.0,
    lo: float = 1e-6, hi: float = 5.0,
) -> float:
    """Invert Black-Scholes for sigma by Brent's method.

    Returns nan if the quoted price is outside no-arbitrage bounds, which is
    the honest answer — an implied vol does not exist for such a quote.
    """
    S, K, T, r, q = map(float, (spot, strike, time_to_expiry, rate, dividend))
    dfq, dfr = np.exp(-q * T), np.exp(-r * T)
    if kind == "call":
        lower, upper = max(S * dfq - K * dfr, 0.0), S * dfq
    else:
        lower, upper = max(K * dfr - S * dfq, 0.0), K * dfr
    if not lower - 1e-12 <= price <= upper + 1e-12:
        return float("nan")

    f = lambda s: black_scholes(S, K, T, r, s, kind, q)["price"] - price
    try:
        if f(lo) * f(hi) > 0:
            return float("nan")
        return float(optimize.brentq(f, lo, hi, xtol=1e-10, maxiter=200))
    except (ValueError, RuntimeError):
        return float("nan")