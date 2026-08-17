"""Annualised carry per registry pair, from policy rates.

`carry_pct` is what you EARN for being LONG the pair: rate(base) - rate(quote).
Long USD/ZAR earns the US rate and pays the South African one, so it is deeply
negative and the short earns the mirror. This sign is the whole point of the
module, and getting it backwards would invert the read on every EM pair.

Why it exists: GBM drift estimated from price history was statistically
indistinguishable from zero on every instrument in this universe (t = 0.13 to
1.64, measured 2026-08-15). Carry is the one component of expected return that
does not have to be inferred from noisy prices — it is read off the policy
rates, and at a 5-20 day horizon it is the dominant driver.

Rates come through `src/services/fred_data.py`, which is Postgres-first, so a
page asking for carry does not add a network round-trip.
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Dict, Optional

from src.core.data_provider import FRED_SERIES
from src.services.fred_data import fred_series

logger = logging.getLogger("ForexDashboard")

# Currency -> FRED policy-rate series, derived from FRED_SERIES so the mapping
# is not duplicated. Metals and oil are absent by construction: they have no
# policy rate, and `carry_pct` returns None for them rather than a misleading 0.
RATE_SERIES: Dict[str, str] = {
    ccy: block["Rates"] for ccy, block in FRED_SERIES.items() if block.get("Rates")
}

# Differentials smaller than this are noise, not an edge: policy rates move in
# 25bp steps and the underlying series are monthly.
DEAD_BAND_PCT = 0.25


def _latest_rate(currency: str) -> Optional[float]:
    """Most recent policy rate for ``currency``, or None if unavailable."""
    series_id = RATE_SERIES.get(currency)
    if not series_id:
        return None
    try:
        series = fred_series(series_id)
    except Exception as exc:                      # noqa: BLE001 — never break a page
        logger.warning("[carry] %s (%s) unavailable: %s", currency, series_id, exc)
        return None
    if series is None or len(series) == 0:
        return None
    try:
        cleaned = series.dropna()
        if len(cleaned) == 0:
            return None
        return float(cleaned.iloc[-1])
    except Exception:                             # noqa: BLE001
        return None


def carry_pct(pair: str, *, now: Optional[date] = None) -> Optional[float]:
    """Annualised carry, in percent, for being LONG ``pair``.

    ``None`` when either leg has no policy rate — metals and oil always, and any
    FX pair whose series is missing. None rather than 0.0 deliberately: zero is
    a claim that there is no carry, which is a different statement from "not
    known".
    """
    base, _, quote = str(pair or "").partition("/")
    if not base or not quote:
        return None
    base_rate, quote_rate = _latest_rate(base), _latest_rate(quote)
    if base_rate is None or quote_rate is None:
        return None
    return base_rate - quote_rate


def favours(pair: str, direction: str) -> Optional[bool]:
    """Does carry pay this direction? ``None`` when unknown or negligible.

    None is not "no": it means the question cannot be answered (no policy rate)
    or the differential is inside the dead band. Callers must not score None as
    a failure — doing so would dock every metal signal a point for a question
    that does not apply to it.
    """
    value = carry_pct(pair)
    # `<=`, not `<`: a differential sitting exactly on the dead band is treated
    # as noise, matching how the oscillator bands treat a reading exactly on 70
    # as not-yet-overbought. The threshold belongs to the neutral side in both,
    # and at the boundary the conservative call is the right one.
    if value is None or abs(value) <= DEAD_BAND_PCT:
        return None
    long_side = str(direction or "").strip().upper().startswith(("L", "B"))
    return (value > 0) if long_side else (value < 0)
