"""Walk-forward validation of the canonical house view (src/core/bias.py).

Answers the question the consensus board can't: *is the house view a real
filter, or decoration?* For every historical day it recomputes the composite
direction using only data available **at that day's close** and measures what
price did over the following days.

No-lookahead construction (and why it's exact, not approximate):

- **Daily EMAs** — ``ewm(span, adjust=False)`` is a pure recursion, so the
  EMA value at bar *t* computed over the full series is byte-identical to the
  EMA computed over the series truncated at *t*. Vectorizing over the full
  frame therefore leaks nothing.
- **Weekly EMAs** — the weekly frame at day *t* is [completed weeks…, partial
  current week whose close is day *t*'s close]. The EMA over that series is
  one recursion step on top of the EMA through the last *completed* week:
  ``ema_t = α·close_t + (1−α)·ema_prev_completed_week`` — again exact.
- The equivalence is locked in by a unit test that compares this module's
  per-day direction against :func:`src.core.bias.house_view` run on truncated
  resampled frames.

Scope: Weekly + Daily only. Hourly history (for the 4H leg) only reaches back
~730 days, so a multi-year replay can't include it — the bias engine's weight
renormalization handles a missing timeframe by design, exactly as it does live
for a pair with no 4H data.

Caveats (also shown in the UI): daily observations with multi-day horizons
**overlap**, so effective sample sizes are much smaller than row counts — read
hit rates and mean returns as descriptive evidence, not significance tests.
No spread/slippage; close-to-close returns.

Pure module: no Streamlit, no network, no DB.
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

from src.core.bias import (
    BEARISH,
    BULLISH,
    NEUTRAL,
    TF_WEIGHTS,
    _COMPOSITE_THRESHOLD,
    EMA_FAST,
    EMA_SLOW,
)

DEFAULT_HORIZONS = (5, 10, 20)
ALIGNED = "ALIGNED"          # summary pseudo-state: weekly & daily agree, directional
ALL_DAYS = "ALL DAYS"        # unconditional baseline


def _tf_scores(close: pd.Series, e_fast: pd.Series, e_slow: pd.Series,
               available: pd.Series) -> pd.Series:
    """Per-day timeframe score — (bull votes − bear votes)/3, exactly the
    :func:`src.core.bias.timeframe_bias` rule; NaN where the TF is unavailable."""
    bull = (
        (e_fast > e_slow).astype(int)
        + (close > e_fast).astype(int)
        + (close > e_slow).astype(int)
    )
    bear = (
        (e_fast < e_slow).astype(int)
        + (close < e_fast).astype(int)
        + (close < e_slow).astype(int)
    )
    score = (bull - bear) / 3.0
    return score.where(available)


def walk_forward(
    daily: pd.DataFrame,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    fast: int = EMA_FAST,
    slow: int = EMA_SLOW,
) -> pd.DataFrame:
    """Per-day house view + forward returns for one instrument.

    ``daily`` — OHLC frame with a DatetimeIndex (spine ``daily_ohlc`` shape).
    Returns a frame indexed by date with ``direction``, ``aligned``,
    ``composite`` and one ``fwd_{h}`` column (close-to-close return, unsigned)
    per horizon. Days where no timeframe has enough history are dropped.
    """
    if daily is None or len(daily) < slow or "Close" not in daily.columns:
        return pd.DataFrame()
    close = daily["Close"].astype(float)
    idx = close.index

    # ── Daily timeframe (vectorized; recursion ⇒ no lookahead) ──────────────
    d_fast = close.ewm(span=fast, adjust=False).mean()
    d_slow = close.ewm(span=slow, adjust=False).mean()
    n_bars = pd.Series(np.arange(1, len(close) + 1), index=idx)
    d_avail = n_bars >= slow                       # timeframe_bias: len >= slow
    d_score = _tf_scores(close, d_fast, d_slow, d_avail)

    # ── Weekly timeframe (completed-week EMA + one partial-week step) ───────
    week = idx.to_period("W")
    wk_last_close = close.groupby(week).last()     # completed-week closes
    wk_fast_ema = wk_last_close.ewm(span=fast, adjust=False).mean()
    wk_slow_ema = wk_last_close.ewm(span=slow, adjust=False).mean()

    week_series = pd.Series(week, index=idx)
    prev_fast = week_series.map(wk_fast_ema.shift(1))   # EMA through prior week
    prev_slow = week_series.map(wk_slow_ema.shift(1))

    a_fast = 2.0 / (fast + 1)
    a_slow = 2.0 / (slow + 1)
    # First-ever week: the truncated weekly series is [close_t] → EMA seeds at close_t.
    w_fast = (a_fast * close + (1 - a_fast) * prev_fast).fillna(close)
    w_slow = (a_slow * close + (1 - a_slow) * prev_slow).fillna(close)

    # Weeks seen so far at day t = completed weeks before t's week + the partial one.
    week_rank = week_series.map(
        pd.Series(np.arange(1, len(wk_last_close) + 1), index=wk_last_close.index)
    )
    w_avail = week_rank >= slow
    w_score = _tf_scores(close, w_fast, w_slow, w_avail)

    # ── Composite (weights renormalized over available TFs, as house_view) ──
    w_w = TF_WEIGHTS["Weekly"]
    w_d = TF_WEIGHTS["Daily"]
    weight_sum = w_w * w_score.notna() + w_d * d_score.notna()
    weighted = (w_w * w_score.fillna(0.0) + w_d * d_score.fillna(0.0))
    composite = (weighted / weight_sum).where(weight_sum > 0)

    direction = pd.Series(NEUTRAL, index=idx, dtype=object)
    direction[composite >= _COMPOSITE_THRESHOLD] = BULLISH
    direction[composite <= -_COMPOSITE_THRESHOLD] = BEARISH

    def _tf_dir(score: pd.Series) -> pd.Series:
        out = pd.Series(NEUTRAL, index=idx, dtype=object)
        out[score == 1.0] = BULLISH
        out[score == -1.0] = BEARISH
        return out.where(score.notna())

    w_dir = _tf_dir(w_score)
    d_dir = _tf_dir(d_score)
    # Aligned mirrors HouseView.aligned: every available TF directional + same side.
    both = w_dir.notna() & d_dir.notna()
    one_w = w_dir.notna() & ~d_dir.notna()
    one_d = d_dir.notna() & ~w_dir.notna()
    aligned = (
        (both & (w_dir == d_dir) & (w_dir != NEUTRAL))
        | (one_w & (w_dir != NEUTRAL))
        | (one_d & (d_dir != NEUTRAL))
    ).fillna(False)

    out = pd.DataFrame({
        "direction": direction,
        "aligned": aligned,
        "composite": composite,
    })
    for h in horizons:
        out[f"fwd_{h}"] = close.shift(-h) / close - 1.0
    return out[composite.notna()]


def summarize(
    bt: pd.DataFrame, horizons: Sequence[int] = DEFAULT_HORIZONS
) -> pd.DataFrame:
    """Per-state report over one or more concatenated ``walk_forward`` frames.

    Rows: BULLISH, BEARISH, ALIGNED (both TFs agree), NEUTRAL, ALL DAYS.
    Columns per horizon: ``hit_{h}`` (% of days the signed forward return was
    positive — directional states only) and ``avg_{h}_bps`` (mean signed
    forward return in basis points; raw/unsigned for NEUTRAL and ALL DAYS).
    """
    if bt is None or bt.empty:
        return pd.DataFrame()

    sign = bt["direction"].map({BULLISH: 1.0, BEARISH: -1.0, NEUTRAL: np.nan})

    def _stats(mask: pd.Series, signed: bool) -> Dict[str, object]:
        sub = bt[mask]
        row: Dict[str, object] = {"n": int(len(sub))}
        for h in horizons:
            fwd = sub[f"fwd_{h}"].dropna()
            if signed:
                s = sign[fwd.index] * fwd
                row[f"hit_{h}"] = round(float((s > 0).mean() * 100), 1) if len(s) else None
                row[f"avg_{h}_bps"] = round(float(s.mean() * 10_000), 1) if len(s) else None
            else:
                row[f"hit_{h}"] = None
                row[f"avg_{h}_bps"] = round(float(fwd.mean() * 10_000), 1) if len(fwd) else None
        return row

    directional = bt["direction"] != NEUTRAL
    rows = {
        BULLISH: _stats(bt["direction"] == BULLISH, signed=True),
        BEARISH: _stats(bt["direction"] == BEARISH, signed=True),
        ALIGNED: _stats(bt["aligned"] & directional, signed=True),
        NEUTRAL: _stats(bt["direction"] == NEUTRAL, signed=False),
        ALL_DAYS: _stats(pd.Series(True, index=bt.index), signed=False),
    }
    return pd.DataFrame.from_dict(rows, orient="index")


def run_universe(
    bars_by_pair: Mapping[str, pd.DataFrame],
    horizons: Sequence[int] = DEFAULT_HORIZONS,
) -> tuple:
    """Backtest several instruments and pool them.

    Returns ``(pooled_summary, per_pair_summaries)`` where ``per_pair_summaries``
    maps pair → summary frame (empty frames for pairs without enough history).
    """
    frames: List[pd.DataFrame] = []
    per_pair: Dict[str, pd.DataFrame] = {}
    for pair, daily in bars_by_pair.items():
        bt = walk_forward(daily, horizons)
        per_pair[pair] = summarize(bt, horizons)
        if not bt.empty:
            frames.append(bt)
    pooled = summarize(pd.concat(frames), horizons) if frames else pd.DataFrame()
    return pooled, per_pair
