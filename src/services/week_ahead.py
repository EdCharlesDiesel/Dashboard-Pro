"""Week Ahead — the probabilistic range for the coming week.

Pure maths for the "what might next week look like?" question. The honest
answer is **not** a direction forecast (the house-view walk-forward in
``src/services/house_view_backtest.py`` showed the composite has no standalone
entry edge); it's a *volatility* forecast, which is the one thing that
genuinely predicts — vol clusters and is persistent.

:func:`cone_from_returns` builds the 5-trading-day cone from a GARCH(1,1)
term-structure forecast (``src.core.quant_models.fit_garch``), falling back to
an EWMA/RiskMetrics constant-sigma estimate when ``arch`` is unavailable or the
fit is unstable — the same fallback ladder
``src/services/swing_playbook_service.daily_sigma`` uses, so the two can't
disagree about an instrument's volatility.

Cumulative sigma over h days is ``sqrt(sum(sigma_i^2))`` — identical to the
``garch_cone`` in ``pages/forecast_tab.py``, but usable from a service because
that page executes its Streamlit body at import and can't be imported.

Streamlit-free and I/O-free: callers pass returns + spot, so this is unit
tested directly.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Set

import numpy as np
import pandas as pd

# One trading week. Weekly vol scales as sqrt(5) under the random-walk
# assumption the cone is built on.
WEEK_DAYS = 5

# 1 sigma ~= 68% of outcomes, 1.96 sigma ~= 95% (normal approximation; the
# GARCH fit itself uses a t distribution, so the tails are fatter in reality —
# the 95% band is the conservative read, not a guarantee).
_Z68 = 1.0
_Z95 = 1.96


def _ewma_sigma(returns: pd.Series, lam: float = 0.94, window: int = 60) -> Optional[float]:
    """RiskMetrics EWMA daily sigma (decimal). Fallback when GARCH is out."""
    r = returns.dropna()
    if len(r) < 30:
        return None
    s2 = float(r.var())
    for x in r.values[-window:]:
        s2 = lam * s2 + (1.0 - lam) * float(x) ** 2
    sigma = math.sqrt(s2)
    return sigma if math.isfinite(sigma) and sigma > 0 else None


def cone_from_returns(
    returns: pd.Series,
    spot: float,
    horizon: int = WEEK_DAYS,
) -> Optional[Dict]:
    """Probabilistic price cone ``horizon`` trading days out.

    ``returns`` are daily **decimal** returns (0.001 = 0.1%), ``spot`` the last
    price. Returns ``None`` when there isn't enough data to say anything.

    The result carries both bands, the cumulative weekly sigma, the expected
    move as a percentage of spot, and which estimator produced it (``method``)
    so the UI can be honest about precision.
    """
    if returns is None or spot is None or not np.isfinite(spot) or spot <= 0:
        return None
    r = returns.dropna()
    r = r[np.isfinite(r)]
    if len(r) < 30:
        return None

    cum_path: Optional[List[float]] = None   # cumulative sigma per day ahead
    method = "ewma"
    vol_percentile: Optional[float] = None
    regime: Optional[str] = None

    # Preferred: GARCH(1,1) term structure — each of the next `horizon` days
    # gets its own forecast vol, so a currently-calm or currently-stressed
    # market decays toward its long-run level instead of being held flat.
    try:
        from src.core.quant_models import fit_garch
        fit = fit_garch(r, horizon=horizon)
        daily_dec = [float(v) / math.sqrt(252) / 100.0
                     for v in fit["forecast_vol_annual_pct"]]
        if daily_dec and all(np.isfinite(daily_dec)) and any(d > 0 for d in daily_dec):
            running = 0.0
            cum_path = []
            for d in daily_dec:
                running += d * d
                cum_path.append(math.sqrt(running))
            method = "garch"
            vol_percentile = float(fit.get("vol_percentile")) if fit.get("vol_percentile") is not None else None
            regime = fit.get("regime")
    except Exception:
        cum_path = None  # fall through to EWMA

    if not cum_path or not math.isfinite(cum_path[-1]) or cum_path[-1] <= 0:
        sigma_d = _ewma_sigma(r)
        if sigma_d is None:
            return None
        cum_path = [sigma_d * math.sqrt(i + 1) for i in range(horizon)]
        method = "ewma"

    cum_sigma = cum_path[-1]

    return {
        "spot": float(spot),
        "horizon": int(horizon),
        "sigma_week": float(cum_sigma),
        # Log-space bands keep the cone positive and asymmetric in price terms,
        # matching pages/forecast_tab.py's garch_cone.
        "hi68": float(spot * math.exp(_Z68 * cum_sigma)),
        "lo68": float(spot * math.exp(-_Z68 * cum_sigma)),
        "hi95": float(spot * math.exp(_Z95 * cum_sigma)),
        "lo95": float(spot * math.exp(-_Z95 * cum_sigma)),
        # Per-day widening bands — what the cone chart plots.
        "hi68_path": [float(spot * math.exp(_Z68 * c)) for c in cum_path],
        "lo68_path": [float(spot * math.exp(-_Z68 * c)) for c in cum_path],
        "hi95_path": [float(spot * math.exp(_Z95 * c)) for c in cum_path],
        "lo95_path": [float(spot * math.exp(-_Z95 * c)) for c in cum_path],
        "range_pct": float(cum_sigma * 100.0),   # ±1 sigma as % of spot
        "vol_percentile": vol_percentile,
        "regime": regime,
        "method": method,
    }


def classify_range(range_pct: float) -> str:
    """Plain-language size of the expected weekly move (±1 sigma, % of spot).

    Thresholds are deliberately instrument-agnostic: FX majors typically run
    ~1-1.5%/week, metals ~2-3%, so "wide" means wide *for anything*.
    """
    if range_pct < 1.0:
        return "tight"
    if range_pct < 2.0:
        return "normal"
    if range_pct < 3.5:
        return "wide"
    return "very wide"


def currencies_for(instrument: str) -> Set[str]:
    """Currency codes whose news moves this instrument.

    ``"EUR/USD"`` -> ``{"EUR", "USD"}``. Metals/oil quote in dollars and have no
    second currency leg, so ``"XAU/USD"`` -> ``{"USD"}`` — their driver is the
    dollar and US rates, which is exactly what the calendar filter should show.
    """
    if not instrument or "/" not in instrument:
        return set()
    base, quote = instrument.split("/", 1)
    metals = {"XAU", "XAG", "XPT", "WTI"}
    out = {quote.strip().upper()}
    b = base.strip().upper()
    if b not in metals:
        out.add(b)
    return {c for c in out if c}


def risk_flags(
    cone: Optional[Dict],
    cot_percentile: Optional[float],
    n_high_impact: int,
) -> List[str]:
    """Short, plain-language warnings for the week — the things that make a
    plan fail. Pure so the wording is testable and consistent."""
    flags: List[str] = []
    if cone is not None:
        rng = cone["range_pct"]
        label = classify_range(rng)
        if label in ("wide", "very wide"):
            flags.append(f"{label.capitalize()} expected range (±{rng:.1f}%) — size down")
        if cone.get("regime") == "high":
            flags.append("Vol regime HIGH — breakouts more credible, mean reversion riskier")
        elif cone.get("regime") == "low":
            flags.append("Vol regime LOW — expect chop; breakouts fail more often")
    if cot_percentile is not None:
        if cot_percentile > 90:
            flags.append(f"COT crowded long ({cot_percentile:.0f}th pctile) — unwind risk")
        elif cot_percentile < 10:
            flags.append(f"COT crowded short ({cot_percentile:.0f}th pctile) — squeeze risk")
    if n_high_impact >= 3:
        flags.append(f"{n_high_impact} high-impact events this week — event-driven week")
    return flags
