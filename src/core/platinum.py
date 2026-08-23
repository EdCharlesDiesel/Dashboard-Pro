"""
Platinum monitor engine.

Pure numpy/pandas/scipy. No Streamlit, no DB, no I/O — so it is unit-testable
in isolation and reusable from the scheduler as well as the UI.

Lives at: src/core/platinum.py

Core question this module is built to answer: does the rand carry any
information about XPTUSD *after* the dollar factor is removed? The naive
correlation is heavily confounded — DXY weakness lifts EM currencies and
dollar-denominated metals simultaneously, so a raw corr(XPTUSD, USDZAR)
loads on a common factor rather than a causal link. Everything here is
built around stripping that factor out first.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "log_returns",
    "newey_west_cov",
    "ols",
    "OLSResult",
    "rolling_factor_model",
    "nested_incremental_test",
    "cross_correlation",
    "ratio_series",
    "realised_vol",
    "zscore",
    "producer_margin",
    "disconnect_state",
]

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# basics
# ---------------------------------------------------------------------------

def log_returns(s: pd.Series) -> pd.Series:
    """Log returns, NaNs dropped. Guards against non-positive prices."""
    s = pd.to_numeric(s, errors="coerce")
    s = s.where(s > 0)
    return np.log(s).diff().dropna()


def zscore(s: pd.Series, window: int | None = None, min_periods: int | None = None) -> pd.Series:
    """Z-score. Rolling if window given, else full-sample.

    Rolling is the honest choice for anything you would trade off, because a
    full-sample z-score peeks at data the signal would not have had.
    """
    if window is None:
        sd = s.std(ddof=1)
        return (s - s.mean()) / sd if sd and np.isfinite(sd) else s * np.nan
    mp = min_periods or max(20, window // 3)
    mu = s.rolling(window, min_periods=mp).mean()
    sd = s.rolling(window, min_periods=mp).std(ddof=1)
    return (s - mu) / sd.replace(0.0, np.nan)


def realised_vol(close: pd.Series, window: int = 20, annualise: bool = True) -> pd.Series:
    """Close-to-close realised volatility.

    If you have OHLC, prefer the Yang-Zhang estimator already in
    src/core/stochastic.py — it is materially more efficient. This exists so
    the platinum tab works off a close-only series.
    """
    r = log_returns(close)
    v = r.rolling(window, min_periods=max(5, window // 2)).std(ddof=1)
    return v * np.sqrt(TRADING_DAYS) if annualise else v


def ratio_series(a: pd.Series, b: pd.Series) -> pd.Series:
    """Aligned ratio, e.g. Pt/Au or Pt/Pd."""
    df = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    return (df["a"] / df["b"].replace(0.0, np.nan)).dropna()


def producer_margin(xpt_usd: pd.Series, usd_zar: pd.Series) -> pd.Series:
    """Platinum in rand — the South African producer revenue proxy.

    This is the series that actually matters for SA mining economics: revenue
    is dollar-denominated, costs are rand-denominated. A firm rand with flat
    dollar platinum compresses local margin even though the USD chart looks
    fine. Rising XPTZAR is the genuine "good for SA producers" condition.
    """
    df = pd.concat([xpt_usd.rename("pt"), usd_zar.rename("zar")], axis=1).dropna()
    return (df["pt"] * df["zar"]).dropna()


# ---------------------------------------------------------------------------
# OLS with HAC standard errors
# ---------------------------------------------------------------------------

@dataclass
class OLSResult:
    beta: np.ndarray
    se: np.ndarray
    tstat: np.ndarray
    pvalue: np.ndarray
    resid: np.ndarray
    r2: float
    adj_r2: float
    nobs: int
    k: int
    names: list[str] = field(default_factory=list)

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"coef": self.beta, "hac_se": self.se, "t": self.tstat, "p": self.pvalue},
            index=self.names or [f"x{i}" for i in range(len(self.beta))],
        )


def newey_west_cov(X: np.ndarray, resid: np.ndarray, lags: int | None = None) -> np.ndarray:
    """Newey-West HAC covariance matrix.

    Daily financial residuals are heteroskedastic and mildly autocorrelated;
    plain OLS standard errors will be too small and you will over-reject.
    Default lag follows the usual 4*(n/100)^(2/9) rule of thumb.
    """
    n, k = X.shape
    if lags is None:
        lags = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    lags = max(0, min(lags, n - 1))

    XtX_inv = np.linalg.pinv(X.T @ X)
    u = X * resid[:, None]

    S = u.T @ u
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)  # Bartlett kernel
        G = u[L:].T @ u[:-L]
        S += w * (G + G.T)

    cov = XtX_inv @ S @ XtX_inv
    return cov * (n / max(n - k, 1))  # small-sample correction


def ols(y: np.ndarray, X: np.ndarray, names: list[str] | None = None,
        hac_lags: int | None = None, add_const: bool = True) -> OLSResult:
    """OLS with Newey-West standard errors."""
    y = np.asarray(y, dtype=float).ravel()
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if X.shape[0] != y.shape[0]:
        X = X.T
    if add_const:
        X = np.column_stack([np.ones(len(y)), X])
        names = ["const"] + list(names or [f"x{i}" for i in range(X.shape[1] - 1)])
    else:
        names = list(names or [f"x{i}" for i in range(X.shape[1])])

    n, k = X.shape
    beta = np.linalg.pinv(X.T @ X) @ X.T @ y
    resid = y - X @ beta

    cov = newey_west_cov(X, resid, hac_lags)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, beta / se, 0.0)
    p = 2.0 * (1.0 - stats.t.cdf(np.abs(t), df=max(n - k, 1)))

    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    adj = 1.0 - (1.0 - r2) * (n - 1) / max(n - k, 1)

    return OLSResult(beta, se, t, p, resid, r2, adj, n, k, names)


# ---------------------------------------------------------------------------
# rolling factor model / disconnect
# ---------------------------------------------------------------------------

def rolling_factor_model(y: pd.Series, factors: pd.DataFrame, window: int = 120,
                         z_window: int = 60) -> pd.DataFrame:
    """Rolling OLS of y on factors; returns betas, residual and residual z.

    Same shape as the existing disconnect monitor: the residual is the part of
    the platinum move that the factor set does not explain, and its rolling
    z-score is the stretch measure. A |z| > 2 means platinum has decoupled from
    its usual dollar/rates relationship — which is information, but note it is
    a *statement about the residual*, not a directional forecast.
    """
    df = pd.concat([y.rename("_y"), factors], axis=1).dropna()
    if len(df) < window + 5:
        return pd.DataFrame(columns=["resid", "resid_z", "r2"] + list(factors.columns))

    cols = list(factors.columns)
    yv = df["_y"].to_numpy(float)
    Xv = df[cols].to_numpy(float)

    n = len(df)
    out_resid = np.full(n, np.nan)
    out_r2 = np.full(n, np.nan)
    out_beta = np.full((n, len(cols)), np.nan)

    for i in range(window, n + 1):
        sl = slice(i - window, i)
        yw, Xw = yv[sl], Xv[sl]
        Xd = np.column_stack([np.ones(window), Xw])
        try:
            b = np.linalg.pinv(Xd.T @ Xd) @ Xd.T @ yw
        except np.linalg.LinAlgError:
            continue
        rw = yw - Xd @ b
        j = i - 1
        out_resid[j] = rw[-1]
        out_beta[j] = b[1:]
        sst = float(((yw - yw.mean()) ** 2).sum())
        out_r2[j] = 1.0 - float(rw @ rw) / sst if sst > 0 else np.nan

    res = pd.DataFrame(index=df.index)
    for c, arr in zip(cols, out_beta.T):
        res[f"beta_{c}"] = arr
    res["resid"] = out_resid
    res["r2"] = out_r2
    res["resid_z"] = zscore(res["resid"], window=z_window)
    return res


def disconnect_state(z: float | None, entry: float = 2.0, warn: float = 1.5) -> str:
    """Label the residual stretch. Deliberately non-directional."""
    if z is None or not np.isfinite(z):
        return "NO DATA"
    a = abs(z)
    if a >= entry:
        return "STRETCHED RICH" if z > 0 else "STRETCHED CHEAP"
    if a >= warn:
        return "EXTENDED"
    return "IN LINE"


# ---------------------------------------------------------------------------
# the ZAR question: nested incremental test
# ---------------------------------------------------------------------------

def nested_incremental_test(y: pd.Series, base: pd.DataFrame, extra: pd.DataFrame,
                            hac_lags: int | None = None) -> dict:
    """Does `extra` add explanatory power for y on top of `base`?

    This is the formal version of "is the rand telling me anything about
    platinum that the dollar has not already told me". Fit the base model
    (DXY, and whatever else), fit the full model (base + USDZAR), and test the
    restriction that the extra coefficients are jointly zero.

    Reported both ways because they can disagree:
      - F-test on the incremental R^2 (classical, assumes iid errors)
      - HAC t-stats on the extra coefficients (robust, what you should trust)

    A small delta_r2 with a HAC |t| under 2 is the expected result. If you get
    the opposite, block-bootstrap it before believing it.
    """
    cols_b, cols_e = list(base.columns), list(extra.columns)
    df = pd.concat([y.rename("_y"), base, extra], axis=1).dropna()
    n = len(df)
    if n < 40:
        return {"error": f"insufficient overlap ({n} obs)"}

    yv = df["_y"].to_numpy(float)
    Xb = df[cols_b].to_numpy(float)
    Xf = df[cols_b + cols_e].to_numpy(float)

    m_base = ols(yv, Xb, names=cols_b, hac_lags=hac_lags)
    m_full = ols(yv, Xf, names=cols_b + cols_e, hac_lags=hac_lags)

    q = len(cols_e)
    ssr_b = float(m_base.resid @ m_base.resid)
    ssr_f = float(m_full.resid @ m_full.resid)
    dfree = n - m_full.k

    if ssr_f > 0 and dfree > 0:
        F = ((ssr_b - ssr_f) / q) / (ssr_f / dfree)
        p_F = 1.0 - stats.f.cdf(F, q, dfree)
    else:
        F, p_F = np.nan, np.nan

    tbl = m_full.as_frame()
    extra_rows = tbl.loc[cols_e]

    return {
        "nobs": n,
        "r2_base": m_base.r2,
        "r2_full": m_full.r2,
        "delta_r2": m_full.r2 - m_base.r2,
        "F": F,
        "p_F": p_F,
        "extra_coefs": extra_rows,
        "max_abs_t_extra": float(extra_rows["t"].abs().max()),
        "full_table": tbl,
        "base_cols": cols_b,
        "extra_cols": cols_e,
    }


# ---------------------------------------------------------------------------
# lead-lag
# ---------------------------------------------------------------------------

def cross_correlation(x: pd.Series, y: pd.Series, max_lag: int = 10,
                      n_boot: int = 500, block: int = 10,
                      seed: int | None = 7) -> pd.DataFrame:
    """Cross-correlogram of x and y with moving-block-bootstrap bands.

    Sign convention: lag k > 0 is corr(x[t], y[t+k]) — x LEADS y.
    So a significant spike at positive lag with x=platinum, y=USDZAR means
    platinum moves first and the rand follows, which is the direction the
    terms-of-trade story predicts.

    Bands come from a block bootstrap that resamples y in blocks, destroying
    the cross-dependence while preserving y's own autocorrelation. Naive
    +/-1.96/sqrt(n) bands are far too tight on autocorrelated series.
    """
    df = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    n = len(df)
    if n < max(40, 4 * max_lag):
        return pd.DataFrame(columns=["lag", "corr", "lo", "hi", "significant"])

    xv = df["x"].to_numpy(float)
    yv = df["y"].to_numpy(float)
    lags = np.arange(-max_lag, max_lag + 1)

    def _cc(a, b):
        out = np.full(len(lags), np.nan)
        for i, k in enumerate(lags):
            if k >= 0:
                aa, bb = a[: n - k], b[k:]
            else:
                aa, bb = a[-k:], b[: n + k]
            if len(aa) > 5 and aa.std() > 0 and bb.std() > 0:
                out[i] = float(np.corrcoef(aa, bb)[0, 1])
        return out

    obs = _cc(xv, yv)

    rng = np.random.default_rng(seed)
    nb = max(1, int(np.ceil(n / block)))
    boot = np.full((n_boot, len(lags)), np.nan)
    for b in range(n_boot):
        starts = rng.integers(0, max(1, n - block), size=nb)
        yb = np.concatenate([yv[s: s + block] for s in starts])[:n]
        if len(yb) < n:
            yb = np.concatenate([yb, yv[: n - len(yb)]])
        boot[b] = _cc(xv, yb)

    lo = np.nanpercentile(boot, 2.5, axis=0)
    hi = np.nanpercentile(boot, 97.5, axis=0)

    return pd.DataFrame({
        "lag": lags,
        "corr": obs,
        "lo": lo,
        "hi": hi,
        "significant": (obs < lo) | (obs > hi),
    })
