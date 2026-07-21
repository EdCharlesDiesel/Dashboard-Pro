"""Leading indicators — oscillators and flow proxies that turn before price.

Deliberately orthogonal to the trend/structure stack (bias engine, ABR
toolkit, EMA lenses): DeMarker measures buying/selling *exhaustion*, the
volume-delta proxy measures *flow imbalance*, and RSI divergence is the real
leading RSI signal (not the 70/30 levels). Session pivots are leading by
construction — computed from the prior bar, they exist before price gets
there.

Domain note (XAU/XAG): gold trends hard, so pure overbought/oversold reads
get run over — :func:`rsi_divergence` + :func:`pivot_points` confluence is
the combination that earns attention, with the oscillators as context.

Pure module: no Streamlit, no network, no DB.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.indicators.technical import TechnicalIndicators


# ── Oscillators ─────────────────────────────────────────────────────────────

def stochastic(df: pd.DataFrame, k: int = 14, d: int = 3,
               smooth: int = 3) -> Tuple[pd.Series, pd.Series]:
    """Slow Stochastic %K/%D in [0, 100]."""
    ll = df["Low"].rolling(k).min()
    hh = df["High"].rolling(k).max()
    fast_k = 100 * (df["Close"] - ll) / (hh - ll).replace(0, np.nan)
    slow_k = fast_k.rolling(smooth).mean()
    slow_d = slow_k.rolling(d).mean()
    return slow_k, slow_d


def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Williams %R in [-100, 0] — a faster, unsmoothed Stochastic."""
    hh = df["High"].rolling(period).max()
    ll = df["Low"].rolling(period).min()
    return -100 * (hh - df["Close"]) / (hh - ll).replace(0, np.nan)


def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Commodity Channel Index (typical-price deviation, 0.015 scale)."""
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(),
                                   raw=True)
    return (tp - sma) / (0.015 * mad.replace(0, np.nan))


def demarker(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """DeMarker in [0, 1] — buying/selling exhaustion. >0.7 exhausted buying,
    <0.3 exhausted selling."""
    de_max = (df["High"] - df["High"].shift(1)).clip(lower=0)
    de_min = (df["Low"].shift(1) - df["Low"]).clip(lower=0)
    sma_max = de_max.rolling(period).mean()
    sma_min = de_min.rolling(period).mean()
    return sma_max / (sma_max + sma_min).replace(0, np.nan)


# ── Flow ────────────────────────────────────────────────────────────────────

def volume_delta(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Signed-volume flow proxy: each bar's volume signed by candle direction,
    plus a rolling imbalance ratio in [-1, +1].

    Spot FX (``=X`` tickers) reports zero volume on Yahoo — in that case the
    bar's *true range* substitutes as the activity weight (the same proxy the
    AMD scanner uses), so the imbalance still means "which side did the
    activity lean" rather than collapsing to zero.
    """
    direction = np.sign(df["Close"] - df["Open"])
    volume = df.get("Volume")
    has_volume = volume is not None and float(volume.fillna(0).sum()) > 0
    if has_volume:
        activity = volume.fillna(0).astype(float)
    else:
        prev_close = df["Close"].shift(1)
        activity = pd.concat([
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ], axis=1).max(axis=1)
    delta = direction * activity
    rolling_delta = delta.rolling(window).sum()
    rolling_activity = activity.rolling(window).sum()
    ratio = rolling_delta / rolling_activity.replace(0, np.nan)
    return pd.DataFrame({
        "delta": delta,
        "cum_delta": delta.cumsum(),
        "imbalance": ratio,
    }, index=df.index)


# ── Leading levels ──────────────────────────────────────────────────────────

def pivot_points(high: float, low: float, close: float) -> Dict[str, float]:
    """Classic floor-trader pivots from the prior bar — leading levels that
    exist before the session trades them."""
    p = (high + low + close) / 3
    return {
        "P": p,
        "R1": 2 * p - low, "S1": 2 * p - high,
        "R2": p + (high - low), "S2": p - (high - low),
        "R3": high + 2 * (p - low), "S3": low - 2 * (high - p),
    }


def near_pivot(price: float, pivots: Dict[str, float],
               tolerance_pct: float = 0.15) -> Optional[str]:
    """Name of the pivot level ``price`` is within ``tolerance_pct``% of, or
    ``None``. Ties go to the closest level."""
    best, best_dist = None, None
    for name, level in pivots.items():
        if not level:
            continue
        dist = abs(price - level) / abs(level) * 100
        if dist <= tolerance_pct and (best_dist is None or dist < best_dist):
            best, best_dist = name, dist
    return best


# ── RSI divergence (the real leading RSI signal) ────────────────────────────

def _swing_indices(series: pd.Series, kind: str, strength: int) -> list:
    """Index positions of confirmed swing highs/lows (strength bars each side)."""
    vals = series.values
    n = len(vals)
    out = []
    for i in range(strength, n - strength):
        window = vals[i - strength: i + strength + 1]
        if np.isnan(window).any():
            continue
        if kind == "low" and vals[i] == window.min():
            out.append(i)
        elif kind == "high" and vals[i] == window.max():
            out.append(i)
    # Collapse runs of consecutive indices (flat plateaus register every bar
    # of the plateau as a swing) into their last bar, so "the last two swings"
    # are genuinely distinct turning points and never a bar vs itself.
    collapsed: list = []
    for i in out:
        if collapsed and i == collapsed[-1] + 1:
            collapsed[-1] = i
        else:
            collapsed.append(i)
    return collapsed


def rsi_divergence(df: pd.DataFrame, rsi_period: int = 14,
                   lookback: int = 60, strength: int = 3) -> Dict[str, object]:
    """Detect regular RSI divergence over the last ``lookback`` bars.

    - **Bullish**: price prints a lower low while RSI prints a higher low —
      selling momentum fading into the new low.
    - **Bearish**: price prints a higher high while RSI prints a lower high.

    Compares the last two confirmed swing points (``strength`` bars each
    side). Returns ``{"bullish", "bearish", "rsi", "detail"}`` — both flags
    False when there aren't two swings to compare. Never guesses.
    """
    close = df["Close"]
    rsi = TechnicalIndicators.rsi(close, rsi_period)
    tail = close.iloc[-lookback:]
    rsi_tail = rsi.iloc[-lookback:]
    result: Dict[str, object] = {
        "bullish": False, "bearish": False,
        "rsi": float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else None,
        "detail": "no confirmed swing pair in window",
    }
    if len(tail) < 2 * strength + 2:
        return result

    lows = _swing_indices(tail, "low", strength)
    if len(lows) >= 2:
        a, b = lows[-2], lows[-1]
        price_ll = tail.iloc[b] < tail.iloc[a]
        rsi_hl = rsi_tail.iloc[b] > rsi_tail.iloc[a]
        if price_ll and rsi_hl:
            result["bullish"] = True
            result["detail"] = (
                f"price lower low ({tail.iloc[a]:.5g}→{tail.iloc[b]:.5g}) with "
                f"RSI higher low ({rsi_tail.iloc[a]:.1f}→{rsi_tail.iloc[b]:.1f})"
            )

    highs = _swing_indices(tail, "high", strength)
    if len(highs) >= 2:
        a, b = highs[-2], highs[-1]
        price_hh = tail.iloc[b] > tail.iloc[a]
        rsi_lh = rsi_tail.iloc[b] < rsi_tail.iloc[a]
        if price_hh and rsi_lh:
            result["bearish"] = True
            result["detail"] = (
                f"price higher high ({tail.iloc[a]:.5g}→{tail.iloc[b]:.5g}) with "
                f"RSI lower high ({rsi_tail.iloc[a]:.1f}→{rsi_tail.iloc[b]:.1f})"
            )
    return result


# ── The combined gold read ──────────────────────────────────────────────────

def divergence_pivot_read(df: pd.DataFrame, rsi_period: int = 14,
                          lookback: int = 60, strength: int = 3,
                          tolerance_pct: float = 0.15) -> Dict[str, object]:
    """RSI divergence + pivot confluence — the combination that works on
    hard-trending instruments (XAU/XAG) where naked overbought/oversold gets
    run over. Fires ``direction`` (``"Long"``/``"Short"``) only when a
    divergence exists AND price sits at a pivot level from the prior bar;
    otherwise ``direction`` is ``None`` and the parts are reported for
    context."""
    div = rsi_divergence(df, rsi_period, lookback, strength)
    prev = df.iloc[-2] if len(df) >= 2 else None
    pivots = (pivot_points(float(prev["High"]), float(prev["Low"]),
                           float(prev["Close"])) if prev is not None else {})
    price = float(df["Close"].iloc[-1])
    at_level = near_pivot(price, pivots, tolerance_pct) if pivots else None

    direction = None
    if at_level and div["bullish"]:
        direction = "Long"
    elif at_level and div["bearish"]:
        direction = "Short"
    return {
        "direction": direction,
        "divergence": div,
        "pivots": pivots,
        "at_level": at_level,
        "price": price,
    }
