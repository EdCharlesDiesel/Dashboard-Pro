"""Parse an MT4 'Save as Report' / 'Detailed Report' HTML statement into rows
the Trade Journal can store.

MT4 statements are a single HTML table with merged-header rows. The closed-trade
rows always carry an exact `buy`/`sell` type cell, two timestamps (open + close)
and a profit; open positions carry only one timestamp and pending orders use
`buy limit`/`sell stop` etc. — we key off those facts rather than fragile column
indices, so the parser tolerates the per-broker column drift (commission / taxes
/ swap columns appear or vanish between brokers).

Times in MT4 are *broker server* time, not UTC — the caller supplies the broker's
UTC offset so sessions and timestamps land correctly.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from io import StringIO
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.instruments.registry import INSTRUMENTS

# Display name (e.g. "EUR/USD") keyed by its compact form ("EURUSD").
_SYMBOL_KEYS: Dict[str, str] = {name.replace("/", ""): name for name in INSTRUMENTS}
_ALIASES = {"GOLD": "XAU/USD", "SILVER": "XAG/USD", "PLATINUM": "XPT/USD"}

_TRADE_TYPES = {"buy", "sell"}


# ══════════════════════════════════════════════════════════════════
# SYMBOL MAPPING
# ══════════════════════════════════════════════════════════════════

def normalize_symbol(item: str) -> Optional[str]:
    """Map an MT4 instrument string ('EURUSD', 'XAUUSD.r', 'GOLD', 'EURUSDm')
    to an app instrument name, or None if it can't be matched automatically."""
    if not item:
        return None
    u = str(item).upper()
    core = re.sub(r"[^A-Z]", "", u)[:6]
    if core in _SYMBOL_KEYS:
        return _SYMBOL_KEYS[core]
    for alias, name in _ALIASES.items():
        if alias in u:
            return name
    return None


def build_symbol_map(items: List[str]) -> Dict[str, Optional[str]]:
    """Auto-map every distinct raw symbol seen in the statement."""
    return {it: normalize_symbol(it) for it in sorted(set(items))}


# ══════════════════════════════════════════════════════════════════
# SESSIONS  (UTC bands — must match the rest of the app)
# ══════════════════════════════════════════════════════════════════

def session_for_hour(hour: int) -> str:
    if 7 <= hour <= 9:
        return "London KZ"
    if 12 <= hour <= 14:
        return "NY KZ"
    if 15 <= hour <= 17:
        return "London Close"
    if 0 <= hour <= 3:
        return "Tokyo"
    return "Dead Zone"


# ══════════════════════════════════════════════════════════════════
# PARSING
# ══════════════════════════════════════════════════════════════════

def _to_float(val) -> Optional[float]:
    if val is None:
        return None
    s = str(val).strip().replace(" ", "").replace("\xa0", "").replace(",", "")
    if s in ("", "-", "nan"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _to_dt(val) -> Optional[datetime]:
    s = str(val).strip()
    for fmt in ("%Y.%m.%d %H:%M:%S", "%Y.%m.%d %H:%M", "%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def parse_mt4_html(content: bytes) -> List[dict]:
    """Return one dict per *closed* trade with the raw broker fields."""
    text = None
    for enc in ("utf-8", "utf-16", "cp1252", "latin-1"):
        try:
            text = content.decode(enc)
            break
        except (UnicodeDecodeError, LookupError):
            continue
    if text is None:
        raise ValueError("Could not decode the uploaded file as text.")

    try:
        tables = pd.read_html(StringIO(text))
    except Exception as exc:  # noqa: BLE001 — surface a friendly message to the UI
        raise ValueError(f"No HTML tables found — is this an MT4 statement? ({exc})")

    trades: List[dict] = []
    seen_tickets = set()

    for tbl in tables:
        for row in tbl.itertuples(index=False, name=None):
            cells = [c for c in row]
            # Locate the exact buy/sell type cell.
            t_idx = next((i for i, c in enumerate(cells)
                          if str(c).strip().lower() in _TRADE_TYPES), None)
            if t_idx is None or t_idx < 2 or t_idx + 7 >= len(cells):
                continue

            close_dt = _to_dt(cells[t_idx + 6])
            if close_dt is None:
                continue  # open position / pending order — no close timestamp

            ticket = _to_float(cells[t_idx - 2])
            if ticket is None or int(ticket) in seen_tickets:
                continue
            seen_tickets.add(int(ticket))

            # Profit is the last finite numeric cell on the row.
            profit = next((f for c in reversed(cells) if (f := _to_float(c)) is not None), None)

            trades.append(dict(
                ticket=int(ticket),
                open_time=_to_dt(cells[t_idx - 1]),
                close_time=close_dt,
                type=str(cells[t_idx]).strip().lower(),
                size=_to_float(cells[t_idx + 1]),
                item=str(cells[t_idx + 2]).strip(),
                open_price=_to_float(cells[t_idx + 3]),
                sl=_to_float(cells[t_idx + 4]),
                tp=_to_float(cells[t_idx + 5]),
                close_price=_to_float(cells[t_idx + 7]),
                profit=profit,
            ))

    return trades


# ══════════════════════════════════════════════════════════════════
# MAP TO JOURNAL ROWS
# ══════════════════════════════════════════════════════════════════

def to_journal_rows(trades: List[dict], symbol_map: Dict[str, Optional[str]],
                    broker_utc_offset: float = 0.0) -> Tuple[List[dict], Dict[str, int]]:
    """Convert parsed trades to journal rows. Returns (rows, skipped) where
    `skipped` counts trades dropped per unmapped raw symbol."""
    rows: List[dict] = []
    skipped: Dict[str, int] = {}

    for t in trades:
        inst = symbol_map.get(t["item"])
        if inst is None or inst not in INSTRUMENTS:
            skipped[t["item"]] = skipped.get(t["item"], 0) + 1
            continue
        if t["open_price"] is None or t["close_price"] is None:
            skipped[t["item"]] = skipped.get(t["item"], 0) + 1
            continue

        pip_size = INSTRUMENTS[inst]["pip_size"]
        ticker = INSTRUMENTS[inst]["ticker"]
        direction = "LONG" if t["type"] == "buy" else "SHORT"
        entry, close = t["open_price"], t["close_price"]

        if direction == "LONG":
            pips = (close - entry) / pip_size
        else:
            pips = (entry - close) / pip_size

        sl_pips = None
        r_mult = None
        if t["sl"] and t["sl"] > 0:
            sl_pips = abs(entry - t["sl"]) / pip_size
            if sl_pips > 0:
                r_mult = pips / sl_pips

        profit = t["profit"] if t["profit"] is not None else 0.0
        outcome = "WIN" if profit > 0 else "LOSS" if profit < 0 else "BE"

        # Broker server time → UTC for session bucketing and storage.
        entry_dt = t["open_time"] or t["close_time"]
        entry_utc = entry_dt - timedelta(hours=broker_utc_offset)

        rows.append(dict(
            ticket=t["ticket"],
            logged_at=entry_utc,
            instrument=inst,
            ticker=ticker,
            direction=direction,
            session=session_for_hour(entry_utc.hour),
            lot_size=t["size"],
            entry_price=entry,
            close_price=close,
            outcome=outcome,
            pips_gained=round(pips, 1),
            r_multiple=round(r_mult, 2) if r_mult is not None else None,
            sl_pips=round(sl_pips, 1) if sl_pips is not None else None,
            profit=profit,
            notes=f"MT4 #{t['ticket']}",
        ))

    return rows, skipped
