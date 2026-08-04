"""The desk's chart MA stack as a directional read — pure, no I/O.

The Python twin of `mt5/FiboRibbon.mq5`, which was itself recreated from the
trader's own saved MT5 chart profile. Same five averages, same gate, same
closed-bar rule, so the dashboard and the chart cannot disagree about whether a
signal fired:

    EMA200 / EMA55 / EMA21 / EMA5   on CLOSE
    SMA8                            on OPEN

**The mixed applied price is deliberate.** The 8-period SMA reads the *open*
while everything else reads the close — that came out of the chart file, and the
5-EMA-vs-8-SMA cross only behaves as a trigger while the two series are drawn
from different prices. Aligning them collapses the pair into noise around one
series.

**The gate is the point.** A bare 5/8 cross fires constantly in chop. Requiring
the ribbon to be ordered (EMA21 > EMA55 > EMA200, price above EMA21) is what
distinguishes a fast cross going *with* the larger trend from one fighting it.

**Closed bars only.** The canonical spine deliberately includes the still-forming
bar (see `market_data.weekly_ohlc`), so the newest row is partial. Reading a
cross off it would repaint: a signal that appears intrabar and vanishes by the
close back-tests as a winner and trades as a loss. Every read here is taken from
the last *completed* bar.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

import pandas as pd

from src.indicators.technical import TechnicalIndicators

# Periods read from MQL5\Profiles\Charts\(usd) AUDUSD\chart02.chr — identical on
# (usd) NZDUSD, so this is the desk's standard template, not a per-symbol tweak.
SLOW, MID, FAST = 200, 55, 21
TRIGGER_EMA, TRIGGER_SMA = 5, 8

LONG, SHORT, NEUTRAL = "Long", "Short", "Neutral"


@dataclass(frozen=True)
class RibbonRead:
    """One instrument's ribbon state on its last closed bar."""

    direction: str          # Long | Short | Neutral
    crossed: bool           # trigger pair crossed on that bar
    stacked: bool           # ribbon ordered in the crossed direction
    close: float
    ema5: float
    sma8: float
    ema21: float
    ema55: float
    ema200: float
    bar_time: Optional[pd.Timestamp]
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def ribbon(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the five averages. Returns a copy; never mutates the caller's frame.

    Needs ``Open`` as well as ``Close`` — the 8-SMA is an average of the open.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    if not {"Open", "Close"}.issubset(df.columns):
        return pd.DataFrame()

    out = df.copy()
    close, open_ = out["Close"], out["Open"]
    out["ema200"] = TechnicalIndicators.ema(close, SLOW)
    out["ema55"] = TechnicalIndicators.ema(close, MID)
    out["ema21"] = TechnicalIndicators.ema(close, FAST)
    out["ema5"] = TechnicalIndicators.ema(close, TRIGGER_EMA)
    # Simple, and on the OPEN — matching the chart. Not an EMA, not on close.
    out["sma8"] = open_.rolling(TRIGGER_SMA).mean()
    return out


def _stacked(row: pd.Series, bullish: bool) -> bool:
    """Ribbon ordered in the given direction, with price on the right side.

    Coerced to a real ``bool``: pandas comparisons yield ``numpy.bool_``, which
    fails an ``is True`` identity check and does not serialize to JSON — and this
    value reaches Postgres inside the signal's ``checks_detail`` blob.
    """
    if bullish:
        return bool(row["ema21"] > row["ema55"] > row["ema200"]
                    and row["Close"] > row["ema21"])
    return bool(row["ema21"] < row["ema55"] < row["ema200"]
                and row["Close"] < row["ema21"])


def read(df: pd.DataFrame, require_stack: bool = True) -> Optional[RibbonRead]:
    """The ribbon read on the last **closed** bar, or ``None`` without enough history.

    ``require_stack=False`` drops the trend gate and reports the raw 5/8 cross —
    useful for measuring how much the gate is actually worth, not for trading.
    """
    frame = ribbon(df)
    # Need the slow EMA defined, plus a prior bar to cross from, plus the
    # forming bar we are about to discard.
    if frame.empty or len(frame) < SLOW + 3:
        return None

    cur, prev = frame.iloc[-2], frame.iloc[-3]   # -1 is the forming bar
    if pd.isna(cur[["ema200", "ema55", "ema21", "ema5", "sma8"]]).any():
        return None
    if pd.isna(prev[["ema5", "sma8"]]).any():
        return None

    # bool(...) for the same reason as _stacked: numpy scalars leak into JSON.
    cross_up = bool(cur["ema5"] > cur["sma8"] and prev["ema5"] <= prev["sma8"])
    cross_dn = bool(cur["ema5"] < cur["sma8"] and prev["ema5"] >= prev["sma8"])

    direction, crossed, stacked, reason = NEUTRAL, False, False, "no cross on the last closed bar"
    if cross_up or cross_dn:
        crossed = True
        bullish = bool(cross_up)
        stacked = _stacked(cur, bullish)
        if stacked or not require_stack:
            direction = LONG if bullish else SHORT
            reason = ("EMA5 crossed {0} SMA8(open){1}".format(
                "above" if bullish else "below",
                ", ribbon aligned" if stacked else ", ribbon NOT aligned (gate off)"))
        else:
            reason = ("EMA5 crossed {0} SMA8(open) but the ribbon disagrees "
                      "— gated out".format("above" if bullish else "below"))

    return RibbonRead(
        direction=direction,
        crossed=crossed,
        stacked=stacked,
        close=float(cur["Close"]),
        ema5=float(cur["ema5"]),
        sma8=float(cur["sma8"]),
        ema21=float(cur["ema21"]),
        ema55=float(cur["ema55"]),
        ema200=float(cur["ema200"]),
        bar_time=frame.index[-2] if isinstance(frame.index, pd.DatetimeIndex) else None,
        reason=reason,
    )
