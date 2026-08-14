"""Import closed MT5 deals into ``trade_setups`` as graded, closed rows.

The gap this fills: `mt4_import` parses an uploaded MT4 *statement file*, and
`open_positions` syncs the *open* book — so nothing ever wrote a closed MT5
trade to the database. That is why `outcome` and `r_multiple` were empty on
every row except the 42 historical MT4 imports, and why the Source Scorecard
had no ground truth of its own to grade against: rule 1 of
:func:`src.services.source_scorecard.evaluate_row` wants a real close, and no
real close was ever stored.

MT5 models a round trip as two or more *deals* sharing a ``position_id`` — one
or more entries (``entry == 0``) and one or more exits (``entry == 1``). This
module reassembles them, which is what makes partial closes and scale-ins come
out as a single honest row rather than several half-trades.

Pure mapping (``pair_deals`` / ``trips_to_journal_rows``) is separated from IO
(``import_closed_trades``) so the arithmetic is testable without a terminal.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from src.instruments.registry import INSTRUMENTS
from src.services.mt4_import import session_for_hour

logger = logging.getLogger("ForexDashboard")

SOURCE = "mt5_import"

# Broker stop-out / take-profit comments look like "[sl 18.76307]".
_LEVEL_RE = re.compile(r"\[(sl|tp)\s+([0-9]*\.?[0-9]+)\]", re.IGNORECASE)

DEAL_ENTRY_IN = 0
DEAL_ENTRY_OUT = 1
DEAL_TYPE_BUY = 0
DEAL_TYPE_SELL = 1


def parse_close_level(comment: str) -> Tuple[Optional[str], Optional[float]]:
    """``("sl", 18.76307)`` from a broker close comment, else ``(None, None)``."""
    m = _LEVEL_RE.search(comment or "")
    if not m:
        return None, None
    try:
        return m.group(1).lower(), float(m.group(2))
    except (TypeError, ValueError):
        return None, None


def _weighted(pairs: List[Tuple[float, float]]) -> Optional[float]:
    """Volume-weighted average price; plain mean if volumes are all zero."""
    if not pairs:
        return None
    vol = sum(v for v, _ in pairs)
    if vol > 0:
        return sum(v * p for v, p in pairs) / vol
    return sum(p for _, p in pairs) / len(pairs)


def pair_deals(deals: List[Dict[str, Any]], days: int = 7,
               now: Optional[datetime] = None) -> List[Dict[str, Any]]:
    """Reassemble deals into round trips closed within the last ``days``.

    A trip needs both legs. Positions still open (entries with no exit) are
    skipped — they are the live book's business, not the journal's — and so are
    exits whose entry fell outside the fetch window, because a round trip with
    no entry price cannot be scored and a guessed one would be worse than none.
    """
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=int(days))

    grouped: Dict[int, Dict[str, List[Dict[str, Any]]]] = {}
    for d in deals:
        pid = d.get("position_id") or 0
        if not pid:
            continue
        slot = grouped.setdefault(pid, {"in": [], "out": []})
        if d.get("entry") == DEAL_ENTRY_IN:
            slot["in"].append(d)
        elif d.get("entry") == DEAL_ENTRY_OUT:
            slot["out"].append(d)

    trips: List[Dict[str, Any]] = []
    for pid, slot in grouped.items():
        ins, outs = slot["in"], slot["out"]
        if not ins or not outs:
            continue
        close_time = max(d["time"] for d in outs)
        if close_time < cutoff:
            continue

        first_in = min(ins, key=lambda d: d["time"])
        # Direction comes from the *entry* deal: the exit is always its
        # opposite, so reading the exit would invert every trade.
        direction = "Long" if first_in.get("type") == DEAL_TYPE_BUY else "Short"

        stop_price = None
        for d in sorted(outs, key=lambda x: x["time"]):
            kind, level = parse_close_level(d.get("comment", ""))
            if kind == "sl" and level:
                stop_price = level
                break

        trips.append({
            "position_id": pid,
            "symbol": first_in.get("symbol", ""),
            "direction": direction,
            "volume": sum(d.get("volume", 0.0) for d in outs),
            "entry_price": _weighted([(d.get("volume", 0.0), d.get("price", 0.0))
                                      for d in ins]),
            "close_price": _weighted([(d.get("volume", 0.0), d.get("price", 0.0))
                                      for d in outs]),
            "open_time": first_in["time"],
            "close_time": close_time,
            # Commission and swap are part of what the trade actually returned,
            # so they belong in the figure the scorecard grades.
            "profit": sum(d.get("profit", 0.0) + d.get("commission", 0.0)
                          + d.get("swap", 0.0) for d in outs),
            "stop_price": stop_price,
        })

    trips.sort(key=lambda t: t["close_time"])
    return trips


def trips_to_journal_rows(
    trips: List[Dict[str, Any]], symbol_map: Dict[str, Optional[str]],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Map round trips to ``trade_setups`` rows. Returns ``(rows, skipped)``."""
    rows: List[Dict[str, Any]] = []
    skipped: Dict[str, int] = {}

    for t in trips:
        raw_symbol = t.get("symbol", "")
        inst = symbol_map.get(raw_symbol)
        entry, close = t.get("entry_price"), t.get("close_price")
        if inst is None or inst not in INSTRUMENTS or entry is None or close is None:
            skipped[raw_symbol] = skipped.get(raw_symbol, 0) + 1
            continue

        meta = INSTRUMENTS.get(inst)
        pip_size = (meta.pip_size if meta else 0.0001) or 0.0001
        sign = 1.0 if t["direction"] == "Long" else -1.0
        pips = sign * (close - entry) / pip_size

        # The broker comment records where the stop sat *at the close*, which is
        # not the same thing as the risk originally taken. For a genuine
        # stop-out that distinction does not matter: price reached the stop, so
        # the trade lost exactly the R it was risking and -1.0 is the standard
        # reading. For a winner closed by a *trailed* stop it matters entirely —
        # the stop had been dragged into profit, so `|entry - stop|` describes
        # where the trade ended, not what it risked, and dividing by it yields a
        # meaningless +1.0 for every such trade. Those keep a NULL R rather than
        # a number that looks measured and is not.
        #
        # Real R for the winners needs the stop the *signal* proposed, which now
        # lives in trade_setups.sl_pips — matching an import back to its signal
        # row is the follow-up that would make this column fully honest.
        sl_pips = r_mult = None
        if t.get("stop_price") and pips < 0:
            sl_pips = abs(entry - float(t["stop_price"])) / pip_size
            if sl_pips > 0:
                r_mult = pips / sl_pips
            else:
                sl_pips = None

        profit = float(t.get("profit") or 0.0)
        rows.append({
            "ticket": t["position_id"],
            "logged_at": t["open_time"].replace(tzinfo=None),
            "instrument": inst,
            "ticker": meta.ticker if meta else None,
            # Canonical "Long"/"Short" — deliberately not mt4_import's
            # "LONG"/"SHORT", which is why the column already holds three
            # spellings for two directions (see `_canonical_direction`).
            "direction": t["direction"],
            "session": session_for_hour(t["open_time"].hour),
            "lot_size": round(float(t.get("volume") or 0.0), 2),
            "entry_price": entry,
            "close_price": close,
            "outcome": "WIN" if profit > 0 else "LOSS" if profit < 0 else "BE",
            "pips_gained": round(pips, 1),
            "r_multiple": round(r_mult, 2) if r_mult is not None else None,
            "sl_pips": round(sl_pips, 1) if sl_pips is not None else None,
            "profit": round(profit, 2),
            "notes": "MT5 #{0}".format(t["position_id"]),
        })

    return rows, skipped


def import_closed_trades(days: int = 7, cfg=None, mt5=None) -> Dict[str, Any]:
    """Read closed MT5 deals and store the new ones. Never raises.

    Returns ``{"ok", "message", "imported", "skipped", "seen"}``.
    """
    out: Dict[str, Any] = {"ok": False, "message": "", "imported": 0,
                           "skipped": {}, "seen": 0}
    from src.services import mt5_link

    read = mt5_link.closed_deals(days=days, mt5=mt5)
    if not read.get("ok"):
        out["message"] = read.get("message", "MT5 history unavailable")
        return out

    trips = pair_deals(read["deals"], days=days)
    out["seen"] = len(trips)
    if not trips:
        out.update(ok=True, message="No closed trades in the last {0}d".format(days))
        return out

    try:
        from src.db.cache import pooled_repository
        from src.db.market_cache import _resolve_cfg
        target = cfg if cfg is not None else _resolve_cfg()
        if target is None:
            out["message"] = "No database configured — nothing imported."
            return out
        repo = pooled_repository(target)

        from src.services.broker_symbols import build_symbol_map
        smap = build_symbol_map([t["symbol"] for t in trips])
        rows, skipped = trips_to_journal_rows(trips, smap)
        out["skipped"] = skipped

        # Dedupe on position id, so re-running the job is safe at any cadence.
        already = repo.imported_tickets(source=SOURCE, tag="MT5")
        fresh = [r for r in rows if r["ticket"] not in already]
        if not fresh:
            out.update(ok=True, message="Already up to date ({0} trip(s) seen)"
                       .format(len(trips)))
            return out

        out["imported"] = repo.import_closed_rows(fresh, source=SOURCE)
        out.update(ok=True, message="Imported {0} closed trade(s)".format(
            out["imported"]))
        logger.info("[mt5_import] %s", out["message"])
        return out
    except Exception as exc:  # noqa: BLE001 — a scheduled job must not crash
        out["message"] = "MT5 import failed: {0}".format(exc)
        logger.warning("[mt5_import] %s", out["message"])
        return out
