"""Source scorecard — which signal sources actually predict winners?

Every signal page persists its reads to ``trade_setups`` with a ``source``
tag, but nothing ever asked whether those signals worked. This module answers
that: each row is resolved against what price did *after* it was saved, then
aggregated per source into win rate, average R and expectancy — the evidence
for trusting (or ignoring) each page.

Resolution rules, most reliable evidence first:

1. **Recorded outcome** — a row closed by the journal / MT4 import
   (``is_open == False`` with an ``r_multiple``) uses its recorded R.
   Ground truth; no price replay needed.
2. **TP/SL replay** — an open signal with entry + ``sl_pips`` + ``tp1_pips``
   is walked forward bar by bar: stop hit → −1R, TP1 hit → +(tp1/sl)R.
   A bar that spans both levels counts as the **loss** (conservative,
   same convention as the 20-day-breakout backtest).
3. **Horizon mark** — entry + stop but no target: the R-multiple is marked
   to the close ``horizon`` bars after the signal (capped at −1R if the stop
   was hit first).
4. **Directional hit** — entry but no stop: did price close in the signal's
   direction ``horizon`` bars later? Hit-rate only; no R is invented.

Rows that can't be evaluated (Neutral bias, missing entry, no bars yet) are
counted but never guessed. Pure module: no DB, no Streamlit, no network —
the caller supplies rows and OHLC frames (spine ``daily_ohlc``).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

import pandas as pd

from src.instruments.registry import INSTRUMENTS
from src.services.signal_expiry import _bias_sign, extract_entry_price

DEFAULT_HORIZON = 10  # trading days for horizon marks / directional hits

# Row statuses
WIN = "win"                    # TP1 before stop, or recorded WIN
LOSS = "loss"                  # stop before TP1, or recorded LOSS
MARKED = "marked"              # no TP — R marked to close at horizon
HIT = "hit"                    # no stop — direction correct at horizon
MISS = "miss"                  # no stop — direction wrong at horizon
OPEN = "open"                  # levels intact, horizon not reached yet
UNRESOLVED = "unresolved"      # not evaluable (Neutral / missing data)

_RESOLVED = {WIN, LOSS, MARKED}


@dataclass(frozen=True)
class RowResult:
    source: str
    status: str
    r: Optional[float] = None      # signed R-multiple (WIN/LOSS/MARKED)
    hit: Optional[bool] = None     # directional call (HIT/MISS)


def _pip_size_for(row: Mapping[str, Any]) -> Optional[float]:
    inst = INSTRUMENTS.get(str(row.get("instrument") or ""))
    if inst is not None and inst.pip_size:
        return float(inst.pip_size)
    ticker = row.get("ticker")
    if ticker:
        for i in INSTRUMENTS.values():
            if i.ticker == ticker and i.pip_size:
                return float(i.pip_size)
    return None


def _detail(row: Mapping[str, Any]) -> Dict[str, Any]:
    """``checks_detail`` as a dict — JSONB arrives deserialized, fixtures as text."""
    detail = row.get("checks_detail")
    if isinstance(detail, str):
        try:
            detail = json.loads(detail)
        except (TypeError, ValueError):
            return {}
    return detail if isinstance(detail, dict) else {}


def row_horizon(row: Mapping[str, Any], default: int = DEFAULT_HORIZON) -> int:
    """The page's own intended horizon in trading days, else ``default``.

    A weekly-swing read and a 15-minute fib trigger should not be marked on the
    same clock: judging the swing at 10 days cuts it off before its thesis plays
    out, and judging the trigger at 10 days credits it with a move it never aimed
    at. Pages that know their horizon record it; the rest keep the default.
    """
    value = _detail(row).get("horizon_days")
    try:
        horizon = int(value)
    except (TypeError, ValueError):
        return default
    return horizon if horizon > 0 else default


def quality_passed(row: Mapping[str, Any]) -> Optional[bool]:
    """The tradeability gate recorded at signal time, or ``None`` if unrecorded.

    Stored rather than enforced, so the scorecard can segment on it and show
    whether the gate actually separates winners from losers.
    """
    value = _detail(row).get("quality_passed")
    return bool(value) if isinstance(value, bool) else None


def _as_float(value: Any) -> Optional[float]:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _bars_after(bars: Optional[pd.DataFrame], logged_at: Any) -> Optional[pd.DataFrame]:
    if bars is None or len(bars) == 0 or not isinstance(bars.index, pd.DatetimeIndex):
        return None
    try:
        ts = pd.Timestamp(logged_at)
    except (TypeError, ValueError):
        return None
    if ts is pd.NaT:
        return None
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    idx = bars.index
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    after = bars[idx > ts]
    return after if len(after) else None


def evaluate_row(
    row: Mapping[str, Any],
    bars: Optional[pd.DataFrame],
    horizon: int = DEFAULT_HORIZON,
) -> RowResult:
    """Resolve one ``trade_setups`` row into a :class:`RowResult`.

    ``horizon`` is the caller's default; a row that recorded its own
    ``horizon_days`` overrides it.
    """
    source = str(row.get("source") or "checklist")
    horizon = row_horizon(row, horizon)

    # 1. Recorded outcome — ground truth from a real close.
    if row.get("is_open") is False:
        r = _as_float(row.get("r_multiple"))
        if r is not None:
            status = WIN if r > 0 else LOSS
            return RowResult(source, status, r=r)
        outcome = str(row.get("outcome") or "").upper()
        if outcome in ("WIN", "LOSS"):
            return RowResult(source, WIN if outcome == "WIN" else LOSS,
                             r=1.0 if outcome == "WIN" else -1.0)
        return RowResult(source, UNRESOLVED)

    sign = _bias_sign(row.get("direction"))
    entry = extract_entry_price(dict(row))
    pip = _pip_size_for(row)
    if sign is None or entry is None:
        return RowResult(source, UNRESOLVED)

    after = _bars_after(bars, row.get("logged_at"))
    if after is None:
        return RowResult(source, OPEN)

    sl_pips = _as_float(row.get("sl_pips"))
    tp1_pips = _as_float(row.get("tp1_pips"))

    # 2. TP/SL replay.
    if sl_pips and tp1_pips and pip:
        sl_level = entry - sign * sl_pips * pip
        tp_level = entry + sign * tp1_pips * pip
        win_r = tp1_pips / sl_pips
        for _, bar in after.iterrows():
            lo, hi = float(bar["Low"]), float(bar["High"])
            stop_hit = lo <= sl_level if sign == 1 else hi >= sl_level
            tp_hit = hi >= tp_level if sign == 1 else lo <= tp_level
            if stop_hit:            # conservative: stop first on same-bar ties
                return RowResult(source, LOSS, r=-1.0)
            if tp_hit:
                return RowResult(source, WIN, r=win_r)
        return RowResult(source, OPEN)

    # 3. Horizon mark (stop known, no target).
    if sl_pips and pip:
        sl_level = entry - sign * sl_pips * pip
        window = after.iloc[:horizon]
        for _, bar in window.iterrows():
            lo, hi = float(bar["Low"]), float(bar["High"])
            if (lo <= sl_level if sign == 1 else hi >= sl_level):
                return RowResult(source, LOSS, r=-1.0)
        if len(after) < horizon:
            return RowResult(source, OPEN)
        close_h = float(window["Close"].iloc[-1])
        r = sign * (close_h - entry) / (sl_pips * pip)
        return RowResult(source, MARKED, r=r)

    # 4. Directional hit (no stop to size R against).
    if len(after) < horizon:
        return RowResult(source, OPEN)
    close_h = float(after["Close"].iloc[horizon - 1])
    moved = sign * (close_h - entry)
    return RowResult(source, HIT if moved > 0 else MISS, hit=moved > 0)


def build_scorecard(
    rows: List[Mapping[str, Any]],
    bars_by_ticker: Mapping[str, pd.DataFrame],
    horizon: int = DEFAULT_HORIZON,
) -> pd.DataFrame:
    """Aggregate per-source performance. Returns a frame indexed by source with:

    ``signals`` (rows seen), ``resolved`` (WIN+LOSS+MARKED), ``wins``,
    ``losses``, ``win_rate`` (wins vs losses, %), ``avg_r`` / ``total_r`` /
    ``expectancy_r`` (mean R over resolved rows), ``dir_hit_rate`` (% of
    horizon directional calls that went the right way), ``open`` and
    ``unresolved`` counts — sorted by expectancy, best first.
    """
    per_source: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        res = evaluate_row(row, bars_by_ticker.get(row.get("ticker")), horizon)
        agg = per_source.setdefault(res.source, {
            "signals": 0, "wins": 0, "losses": 0, "r_values": [],
            "dir_hits": 0, "dir_calls": 0, "open": 0, "unresolved": 0,
        })
        agg["signals"] += 1
        if res.status in _RESOLVED and res.r is not None:
            agg["r_values"].append(res.r)
            if res.status == WIN or (res.status == MARKED and res.r > 0):
                agg["wins"] += 1
            elif res.status == LOSS or (res.status == MARKED and res.r <= 0):
                agg["losses"] += 1
        elif res.status in (HIT, MISS):
            agg["dir_calls"] += 1
            agg["dir_hits"] += int(bool(res.hit))
        elif res.status == OPEN:
            agg["open"] += 1
        else:
            agg["unresolved"] += 1

    out = []
    for source, agg in per_source.items():
        r_vals = agg["r_values"]
        resolved = len(r_vals)
        decided = agg["wins"] + agg["losses"]
        out.append({
            "source": source,
            "signals": agg["signals"],
            "resolved": resolved,
            "wins": agg["wins"],
            "losses": agg["losses"],
            "win_rate": round(agg["wins"] / decided * 100, 1) if decided else None,
            "avg_r": round(sum(r_vals) / resolved, 2) if resolved else None,
            "total_r": round(sum(r_vals), 2) if resolved else None,
            "expectancy_r": round(sum(r_vals) / resolved, 2) if resolved else None,
            "dir_calls": agg["dir_calls"],
            "dir_hit_rate": (round(agg["dir_hits"] / agg["dir_calls"] * 100, 1)
                             if agg["dir_calls"] else None),
            "open": agg["open"],
            "unresolved": agg["unresolved"],
        })
    if not out:
        return pd.DataFrame()
    df = pd.DataFrame(out).set_index("source")
    return df.sort_values(
        by=["expectancy_r", "dir_hit_rate"],
        ascending=[False, False],
        na_position="last",
    )
