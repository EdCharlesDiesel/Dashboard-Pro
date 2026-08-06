"""FX Bootcamp "Biased Pivots" — the pivot levels from the desk's own charts.

The Python twin of `BiasedPivots.mq5` (header: `WyattsPivots.mq4`, "Copyright,
FX BOOTCAMP TRAINING LLC"), which sits on the live EUR/CAD chart. Ported from
that source line by line, not from a textbook — two details differ from classic
pivots and both are preserved deliberately:

**1. The close comes from one period EARLIER than the high/low.** The source
reads::

    highValue = iHigh (NULL, tf, shft);
    lowValue  = iLow  (NULL, tf, shft);
    closeValue= iClose(NULL, tf, shft + 1);      // one period OLDER

Textbook pivots take H, L and C from the same completed period. This does not.
It is not a bug to tidy: correcting it would move every level and the dashboard
would stop agreeing with what is drawn on the chart, which is the whole point of
having a twin.

**2. "Bias mode" is a display filter, not different maths.** `bullishMode` and
`bearishMode` change which levels get *drawn* (R2/M4 vs S2/M1); PP, R1, R2, S1,
S2 and the midpoints are computed identically either way. So this module
computes every level once and lets the caller decide what to show.

The midpoints carry the trading meaning, straight from the source's own labels:

    M2 = 0.5*(PP+S1)   "Bullish Buying Zone"
    M3 = 0.5*(PP+R1)   "Bearish Selling Zone"
    M4 = 0.5*(R1+R2)   "Bullish Profit Target - Conservative"
    M1 = 0.5*(S1+S2)   "Bearish Profit Target - Conservative"
    R2                 "Bullish Profit Zone - Agressive"  [sic]
    S2                 "Bearish Profit Zone - Agressive"

R3, S3, M0 and M5 are assigned 0 in the source and are not levels — they are
omitted here rather than reproduced as fake zeros.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import pandas as pd

LONG, SHORT, NEUTRAL = "Long", "Short", "Neutral"

# Zone half-width as a fraction of the PP→S1 (or PP→R1) leg. The source draws a
# label offset rather than a band, so the width is ours; expressed relative to
# the pivot geometry so it scales with the instrument instead of hard-coding pips.
DEFAULT_ZONE_FRAC = 0.25


@dataclass(frozen=True)
class Pivots:
    """One period's levels, plus where price sits relative to them."""

    pp: float
    r1: float
    r2: float
    s1: float
    s2: float
    m1: float      # bearish profit target, conservative
    m2: float      # bullish buying zone
    m3: float      # bearish selling zone
    m4: float      # bullish profit target, conservative
    price: float
    direction: str
    zone: str
    bar_time: Optional[pd.Timestamp]
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def levels(high: float, low: float, close_prev: float) -> Dict[str, float]:
    """The raw levels. ``close_prev`` is the close of the period BEFORE h/l.

    Pure arithmetic, mirroring `BiasedPivots.mq5` exactly.
    """
    pp = (high + low + close_prev) / 3.0
    r1 = 2.0 * pp - low
    r2 = pp + (high - low)
    s1 = 2.0 * pp - high
    s2 = pp - (high - low)
    return {
        "pp": pp, "r1": r1, "r2": r2, "s1": s1, "s2": s2,
        "m1": 0.5 * (s1 + s2),
        "m2": 0.5 * (pp + s1),
        "m3": 0.5 * (pp + r1),
        "m4": 0.5 * (r1 + r2),
    }


def read(df: pd.DataFrame, zone_frac: float = DEFAULT_ZONE_FRAC) -> Optional[Pivots]:
    """Pivot read for the current period from a completed daily frame.

    Uses the last **closed** bar as the pivot period and the bar before it for
    the close, matching the indicator's ``shft`` / ``shft+1`` pairing. The
    still-forming bar supplies the price being judged, never the levels — levels
    that move intrabar would repaint.
    """
    if df is None or df.empty or len(df) < 3:
        return None
    if not {"High", "Low", "Close"}.issubset(df.columns):
        return None

    period = df.iloc[-2]           # last completed period -> high/low
    earlier = df.iloc[-3]          # one older -> close (the source's shft+1)
    price = float(df["Close"].iloc[-1])

    high, low = float(period["High"]), float(period["Low"])
    close_prev = float(earlier["Close"])
    if any(pd.isna(v) for v in (high, low, close_prev, price)):
        return None

    lv = levels(high, low, close_prev)

    # Zones are centred on the source's own "buying"/"selling" midpoints, with a
    # half-width taken from the pivot geometry so it scales per instrument.
    buy_half = abs(lv["pp"] - lv["s1"]) * zone_frac
    sell_half = abs(lv["r1"] - lv["pp"]) * zone_frac

    direction, zone, reason = NEUTRAL, "—", "price is not in a pivot zone"
    if abs(price - lv["m2"]) <= buy_half:
        direction, zone = LONG, "Bullish Buying Zone"
        reason = ("price {0:.5f} is in the Bullish Buying Zone around M2 "
                  "{1:.5f}".format(price, lv["m2"]))
    elif abs(price - lv["m3"]) <= sell_half:
        direction, zone = SHORT, "Bearish Selling Zone"
        reason = ("price {0:.5f} is in the Bearish Selling Zone around M3 "
                  "{1:.5f}".format(price, lv["m3"]))

    return Pivots(
        price=price, direction=direction, zone=zone, reason=reason,
        bar_time=df.index[-2] if isinstance(df.index, pd.DatetimeIndex) else None,
        **{k: float(v) for k, v in lv.items()},
    )


def targets(p: Pivots) -> Dict[str, Optional[float]]:
    """Conservative and aggressive targets for the read, per the source labels."""
    if p.direction == LONG:
        return {"conservative": p.m4, "aggressive": p.r2, "stop_ref": p.s1}
    if p.direction == SHORT:
        return {"conservative": p.m1, "aggressive": p.s2, "stop_ref": p.r1}
    return {"conservative": None, "aggressive": None, "stop_ref": None}
