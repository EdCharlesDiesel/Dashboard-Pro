"""Price-based signal invalidation — is a saved ``trade_setups`` row still live?

Pure: no DB, no Streamlit, no network. ``pages/trade-journal.py`` supplies the
row data (already loaded) and current prices (already fetched via the
read-through market cache) and calls :func:`find_newly_invalidated`.

A signal is invalidated when price has crossed its stop level — this single
check also covers "price ran past entry without ever being filled" (the stop
sits beyond entry in the adverse direction, so blowing through the stop means
the whole zone, filled or not, is gone). A row is only ever a candidate when
both its entry price and stop distance are known:

- ``entry_price`` (the DB column) is only populated once a trade is actually
  closed (``TradeRepository.close_trade`` / MT4 import) — while a signal is
  still open, the only place its entry price lives is
  ``checks_detail->>'entry'``, and only for rows saved via
  :func:`src.core.signals.signal_to_setup_row` (the auto-signal pages).
  Checklist-saved rows have no entry price anywhere until manually closed, so
  they're simply not evaluable — :func:`is_invalidated` returns ``None``
  rather than guessing.
- ``direction`` strings are inconsistent across pages (``"LONG"``, ``"Long"``,
  ``"BUY"``, ``"STRONG_SELL"``, ``"Neutral"``, ...) — :func:`_bias_sign`
  normalizes case-insensitively rather than assuming one convention.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

# Substring markers, matched case-insensitively, so every vocabulary the app
# speaks resolves to a side: pages persist "Long"/"Short", the checklist and MT4
# import use "LONG"/"SHORT", the trend evaluator emits "BUY"/"SELL" and
# "STRONG_BUY"/"STRONG_SELL", and `bias.house_view` speaks "BULLISH"/"BEARISH".
# BULL/BEAR were missing until 2026-08-04: no stored row used them yet, but any
# page wiring a house-view read straight into a signal would have produced rows
# the Source Scorecard could never resolve — silently, since an unresolvable
# direction looks identical to a signal still waiting on price.
_LONG_MARKERS = ("LONG", "BUY", "BULL")
_SHORT_MARKERS = ("SHORT", "SELL", "BEAR")


def extract_entry_price(row: Dict[str, Any]) -> Optional[float]:
    """``entry_price`` column if set, else ``checks_detail->>'entry'``.

    ``checks_detail`` may already be a dict (psycopg2 auto-deserializes
    JSONB) or a JSON string (e.g. in test fixtures) — both are handled.
    """
    entry_price = row.get("entry_price")
    if entry_price is not None:
        try:
            return float(entry_price)
        except (TypeError, ValueError):
            return None

    detail = row.get("checks_detail")
    if isinstance(detail, str):
        try:
            detail = json.loads(detail)
        except (TypeError, ValueError):
            detail = None
    if not isinstance(detail, dict):
        return None

    entry = detail.get("entry")
    if entry is None:
        return None
    try:
        value = float(entry)
    except (TypeError, ValueError):
        return None
    return value or None  # signal_to_setup_row defaults a missing entry to 0.0


def _bias_sign(direction: Any) -> Optional[int]:
    """+1 long-biased, -1 short-biased, ``None`` if unresolved (e.g. Neutral)."""
    if not direction:
        return None
    d = str(direction).strip().upper()
    if any(marker in d for marker in _LONG_MARKERS):
        return 1
    if any(marker in d for marker in _SHORT_MARKERS):
        return -1
    return None


def stop_price(entry: float, sl_pips: float, pip_size: float, sign: int) -> float:
    """Stop-loss price level: below entry for longs, above for shorts."""
    return entry - sign * sl_pips * pip_size


def is_invalidated(
    entry: Optional[float],
    sl_pips: Optional[float],
    pip_size: Optional[float],
    direction: Any,
    current_price: Optional[float],
) -> Optional[bool]:
    """Has price crossed the stop level? ``None`` when not evaluable (missing
    input) — a candidate that can't be checked is never flagged."""
    sign = _bias_sign(direction)
    if (
        sign is None
        or entry is None
        or sl_pips is None
        or not pip_size
        or current_price is None
    ):
        return None
    sl = stop_price(entry, sl_pips, pip_size, sign)
    return current_price <= sl if sign == 1 else current_price >= sl


def find_newly_invalidated(
    rows: List[Dict[str, Any]], price_by_ticker: Dict[str, float]
) -> List[Tuple[Any, float]]:
    """For each candidate row (must carry ``id``, ``direction``, ``sl_pips``,
    ``pip_size``, ``ticker``, plus entry via :func:`extract_entry_price`),
    return ``(id, current_price)`` for those that just crossed their stop.
    Rows that aren't evaluable, or whose ticker has no known price, are
    silently skipped — never flagged on missing data."""
    out: List[Tuple[Any, float]] = []
    for row in rows:
        current_price = price_by_ticker.get(row.get("ticker"))
        if current_price is None:
            continue
        entry = extract_entry_price(row)
        verdict = is_invalidated(
            entry, row.get("sl_pips"), row.get("pip_size"),
            row.get("direction"), current_price,
        )
        if verdict:
            out.append((row["id"], current_price))
    return out
