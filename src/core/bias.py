"""Canonical directional bias — the single definition of BULLISH / BEARISH /
NEUTRAL for one instrument on one timeframe, and the weighted "house view"
across timeframes.

Pages used to each define "trend" their own way (EMA20/50 here, EMA50/200
there, slope majority votes elsewhere), so the same pair could be BUY on one
page and SELL on the next. The rule below is ``score_setup``'s Weekly-EMA
criterion (EMA20 above EMA50 with price above both) generalized to every
timeframe, so the master checklist and the house view can never disagree
about what an uptrend *is*. Pages keep their specialized lenses (MACD detail,
structure, COT…) — but the headline direction they show comes from here.

Pure module: no Streamlit, no network, no DB. ``src/services/bias_service.py``
wires it to the canonical data spine.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pandas as pd

from src.core.config import default_config as _config
from src.indicators.technical import TechnicalIndicators

BULLISH = "BULLISH"
BEARISH = "BEARISH"
NEUTRAL = "NEUTRAL"

# Derived from the canonical lens parameters (src/core/config.py) — one home
# for the numbers, so sliders, score_setup and the house view move together.
EMA_FAST = _config.ema_fast
EMA_SLOW = _config.ema_slow

# Higher timeframes carry more weight in the composite (weekly bias is the
# regime; 4H is timing). Renormalized over the *available* timeframes so a
# missing frame never silently drags the composite toward zero — same rule as
# prediction_service.aggregate().
TF_WEIGHTS = {"Weekly": 3.0, "Daily": 2.0, "4H": 1.0}

# Composite thresholds: one directional timeframe outvoted by the others
# lands inside (-0.5, +0.5) → NEUTRAL; weekly + daily agreeing clears it.
_COMPOSITE_THRESHOLD = 0.5

_BULL_WORDS = {"bullish", "buy", "strong_buy", "strong buy", "long", "up", "uptrend"}
_BEAR_WORDS = {"bearish", "sell", "strong_sell", "strong sell", "short", "down", "downtrend"}


@dataclass(frozen=True)
class TimeframeBias:
    """One timeframe's canonical read: three votes, unanimous or NEUTRAL."""
    timeframe: str                 # "Weekly" | "Daily" | "4H"
    direction: str                 # BULLISH | BEARISH | NEUTRAL
    score: float                   # (bull votes − bear votes) / 3 ∈ [−1, +1]
    votes: Dict[str, bool]         # bull-side truth of each canonical check
    price: float
    ema_fast: float
    ema_slow: float
    asof: Optional[pd.Timestamp]   # last bar timestamp the read is based on


@dataclass(frozen=True)
class HouseView:
    """The composite directional read for one instrument."""
    pair: str
    timeframes: Tuple[TimeframeBias, ...]   # available timeframes only
    direction: str                          # BULLISH | BEARISH | NEUTRAL
    score: float                            # weighted composite ∈ [−1, +1]
    aligned: bool                           # every available TF directional + same side

    def timeframe(self, name: str) -> Optional[TimeframeBias]:
        for tf in self.timeframes:
            if tf.timeframe == name:
                return tf
        return None

    @property
    def asof(self) -> Optional[pd.Timestamp]:
        stamps = [tf.asof for tf in self.timeframes if tf.asof is not None]
        return max(stamps) if stamps else None


def timeframe_bias(
    df: Optional[pd.DataFrame],
    timeframe: str,
    fast: int = EMA_FAST,
    slow: int = EMA_SLOW,
) -> Optional[TimeframeBias]:
    """The canonical read for one OHLC frame, or ``None`` when there isn't
    enough history for the slow EMA to mean anything (< ``slow`` bars)."""
    if df is None or len(df) < slow or "Close" not in df.columns:
        return None
    close = df["Close"]
    price = float(close.iloc[-1])
    e_fast = float(TechnicalIndicators.ema(close, fast).iloc[-1])
    e_slow = float(TechnicalIndicators.ema(close, slow).iloc[-1])

    votes = {
        f"EMA{fast} > EMA{slow}": e_fast > e_slow,
        f"Price > EMA{fast}": price > e_fast,
        f"Price > EMA{slow}": price > e_slow,
    }
    bull = sum(votes.values())
    # Bear votes are strict inverses; an exact tie (==) counts for neither side.
    bear = sum([e_fast < e_slow, price < e_fast, price < e_slow])

    if bull == 3:
        direction = BULLISH
    elif bear == 3:
        direction = BEARISH
    else:
        direction = NEUTRAL

    asof = df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else None
    return TimeframeBias(
        timeframe=timeframe, direction=direction, score=(bull - bear) / 3.0,
        votes=votes, price=price, ema_fast=e_fast, ema_slow=e_slow, asof=asof,
    )


def house_view(
    pair: str,
    weekly: Optional[pd.DataFrame] = None,
    daily: Optional[pd.DataFrame] = None,
    h4: Optional[pd.DataFrame] = None,
) -> HouseView:
    """Combine the canonical timeframe reads into one weighted house view."""
    frames = (("Weekly", weekly), ("Daily", daily), ("4H", h4))
    available = tuple(
        tf for tf in (timeframe_bias(df, name) for name, df in frames)
        if tf is not None
    )

    if not available:
        return HouseView(pair=pair, timeframes=(), direction=NEUTRAL,
                         score=0.0, aligned=False)

    total_w = sum(TF_WEIGHTS[tf.timeframe] for tf in available)
    composite = sum(TF_WEIGHTS[tf.timeframe] * tf.score for tf in available) / total_w

    if composite >= _COMPOSITE_THRESHOLD:
        direction = BULLISH
    elif composite <= -_COMPOSITE_THRESHOLD:
        direction = BEARISH
    else:
        direction = NEUTRAL

    directions = {tf.direction for tf in available}
    aligned = directions in ({BULLISH}, {BEARISH})

    return HouseView(pair=pair, timeframes=available, direction=direction,
                     score=composite, aligned=aligned)


def timeframe_agreement(view: HouseView, trade_direction: Optional[str]) -> Tuple[Dict, ...]:
    """Per-timeframe verdict on a proposed trade direction.

    Answers the question a Grade-A signal can't: *which* timeframe disagrees?
    A setup scored on weight-of-evidence (``score_setup``) can be strongly
    directional while the board is NEUTRAL because the timeframes are fighting
    — typically a counter-trend bounce where the fast frame has already flipped
    but the weekly hasn't. Surfacing the dissenter turns "the board says
    NEUTRAL" into "the 4H is against you", which is actionable.

    Returns one dict per *available* timeframe with its own direction, weight,
    and whether it agrees with / opposes ``trade_direction``. NEUTRAL frames do
    neither — an absent opinion is not opposition.
    """
    want = normalize_direction(trade_direction)
    rows = []
    for tf in view.timeframes:
        rows.append({
            "timeframe": tf.timeframe,
            "direction": tf.direction,
            "score": tf.score,
            "weight": TF_WEIGHTS[tf.timeframe],
            "agrees": want != NEUTRAL and tf.direction == want,
            "opposes": is_conflict(tf.direction, trade_direction),
        })
    return tuple(rows)


def dissenting_timeframes(view: HouseView, trade_direction: Optional[str]) -> Tuple[str, ...]:
    """Names of the timeframes actively pointing the other way — the short
    answer to "what's holding this back?"."""
    return tuple(r["timeframe"] for r in timeframe_agreement(view, trade_direction)
                 if r["opposes"])


def normalize_direction(text: Optional[str]) -> str:
    """Map a page's own vocabulary (Long/Short, BUY/SELL, Bullish/Bearish,
    STRONG_BUY…) onto the canonical direction constants."""
    if not text:
        return NEUTRAL
    t = str(text).strip().lower()
    if t in _BULL_WORDS:
        return BULLISH
    if t in _BEAR_WORDS:
        return BEARISH
    return NEUTRAL


def is_conflict(house_direction: str, page_direction: Optional[str]) -> bool:
    """True only when both reads are directional and on opposite sides —
    a NEUTRAL on either side is a disagreement of conviction, not direction."""
    page = normalize_direction(page_direction)
    return (
        {house_direction, page} == {BULLISH, BEARISH}
    )
