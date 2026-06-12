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
from typing import Dict, Tuple

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
