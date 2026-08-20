"""Percentage change over a horizon, computed the same way for every timeframe.

Market Overview previously reported ``Close.pct_change().iloc[-1]`` on whichever
tab was open, so "change" meant a different span depending on where you were
standing — and there was no way to see a month, a week, a day and four hours
side by side.

These functions take the frames the spine already produces. Every one of them is
a resample of the same daily series (see ``src/services/market_data.py``), so a
1W figure and the 1D figures inside it are arithmetically consistent with each
other and with every other page. Pure: no Streamlit, no I/O, no network.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

# Column label -> the `config.timeframes` key whose frame supplies it.
# Ordered shortest to longest so the panel reads left to right as
# "just happened" -> "this month".
HORIZONS: Tuple[Tuple[str, str], ...] = (
    ("4H %", "4 Hour"),
    ("1D %", "Daily"),
    ("1W %", "Weekly"),
    ("1M %", "Monthly"),
)

# The column the panel sorts on: the panel's job is to show what moved, and
# alphabetical order buries that.
SORT_COLUMN = "1D %"


def period_change(df: Optional[pd.DataFrame]) -> Optional[float]:
    """Last close against the previous one, in percent. ``None`` when unanswerable.

    Returns ``None`` rather than ``NaN`` for every degenerate case. NaN renders
    as "nan%" in a table and is invalid JSONB if anything ever persists it —
    the failure that silently destroyed whole signal rows before ``_finite``
    was added to ``src/core/signals.py``.
    """
    if df is None or not hasattr(df, "empty") or df.empty:
        return None
    if "Close" not in getattr(df, "columns", ()):
        return None
    closes = df["Close"].dropna()
    if len(closes) < 2:
        return None
    previous, latest = float(closes.iloc[-2]), float(closes.iloc[-1])
    if not previous or not math.isfinite(previous) or not math.isfinite(latest):
        return None
    return (latest / previous - 1.0) * 100.0


def horizon_row(pair: str, frames: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """One table row: the pair, then its change over each horizon."""
    row: Dict[str, Any] = {"Pair": pair}
    for label, timeframe in HORIZONS:
        row[label] = period_change(frames.get(timeframe))
    return row


def sort_by_mover(rows: Sequence[Dict[str, Any]],
                  column: str = SORT_COLUMN) -> List[Dict[str, Any]]:
    """Biggest absolute move first; rows with no reading sink to the bottom."""
    def key(row: Dict[str, Any]) -> Tuple[int, float]:
        value = row.get(column)
        if value is None or not math.isfinite(float(value)):
            return (1, 0.0)          # unreadable rows last, order preserved
        return (0, -abs(float(value)))

    return sorted(rows, key=key)
