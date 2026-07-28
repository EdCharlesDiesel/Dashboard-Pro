"""Durable store for the positions you are actually holding.

The exposure guard needs to know your live book, and until now nothing did:
``import_mt4_rows`` writes imported trades as ``is_open = FALSE`` (it's a
closed-history importer for the balance/stats), and the ``is_open IS TRUE``
rows in ``trade_setups`` are overwhelmingly **auto-persisted scanner signals**,
not positions — thousands of them. Netting exposure over those would produce
noise, not a guard.

So open positions get their own key: a single ``app_state`` blob written by the
Trade Journal's MT4 import (see `mt4_import.parse_mt4_open_positions`) and read
by the Setup Ranker. Same dual-write pattern as ``mt4_watch`` / ``account_state``
— Postgres is the source of truth, a local JSON file is the offline fallback —
so it survives a restart and a Railway redeploy.

Streamlit-free; every DB call is best-effort and never raises.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

_PATH = os.path.join(os.getcwd(), "open_positions.json")
_STATE_KEY = "open_positions"

# Fields the guard and the UI actually read. Anything else in a row is dropped
# on save so the blob can't silently grow into a second trade log.
_FIELDS = ("ticket", "pair", "direction", "lot_size", "entry_price",
           "stop_loss", "take_profit", "has_stop", "opened_at", "label")


def make_row(
    pair: str,
    direction: str,
    lot_size: float,
    ticket: Optional[int] = None,
    entry_price: Optional[float] = None,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
    opened_at: Optional[str] = None,
    label: Optional[str] = None,
) -> Dict[str, Any]:
    """Build one stored position row — the single definition of the shape.

    Both feeds go through here (the MT4 statement parser and the live MT5
    link) so a field can never mean one thing from a file and another from the
    terminal. In particular ``has_stop`` is derived **here**, from a single
    rule: a stop is real only if it is present and non-zero. Both platforms
    report "no stop" as ``0.0``, not as null, and a naive truthiness check on
    the price would call that a stop and silently under-report the one risk
    that has no bound.
    """
    has_stop = stop_loss is not None and float(stop_loss) > 0
    has_tp = take_profit is not None and float(take_profit) > 0
    return {
        "ticket": int(ticket) if ticket is not None else None,
        "pair": str(pair),
        "direction": str(direction),
        "lot_size": float(lot_size),
        "entry_price": float(entry_price) if entry_price is not None else None,
        "stop_loss": float(stop_loss) if has_stop else None,
        "take_profit": float(take_profit) if has_tp else None,
        "has_stop": has_stop,
        "opened_at": str(opened_at) if opened_at else None,
        "label": str(label) if label else "",
    }


def _db_cfg():
    try:
        from src.db.market_cache import _resolve_cfg
        return _resolve_cfg()
    except Exception:
        return None


def _db_read() -> Optional[Dict]:
    cfg = _db_cfg()
    if cfg is None:
        return None
    try:
        from src.db import cache as dbcache
        val = dbcache.cached_get_state(
            cfg.host, cfg.port, cfg.dbname, cfg.user, cfg.password, _STATE_KEY)
        return val if isinstance(val, dict) else None
    except Exception:
        return None


def _db_write(payload: Dict) -> None:
    cfg = _db_cfg()
    if cfg is None:
        return
    try:
        from src.db import cache as dbcache
        dbcache.set_state(cfg.host, cfg.port, cfg.dbname, cfg.user,
                          cfg.password, _STATE_KEY, payload)
    except Exception:
        pass


def _json_read() -> Dict:
    try:
        if os.path.exists(_PATH):
            with open(_PATH) as fh:
                data = json.load(fh)
                return data if isinstance(data, dict) else {}
    except Exception:
        pass
    return {}


def _json_write(payload: Dict) -> None:
    try:
        with open(_PATH, "w") as fh:
            json.dump(payload, fh)
    except Exception:
        pass


def _clean(rows: Any) -> List[Dict[str, Any]]:
    """Keep only well-formed rows with the fields the guard needs."""
    out: List[Dict[str, Any]] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        if not row.get("pair") or not row.get("direction"):
            continue
        out.append({k: row.get(k) for k in _FIELDS})
    return out


def save(rows: List[Dict[str, Any]],
         account: Optional[Dict[str, Any]] = None) -> int:
    """Replace the stored book with ``rows``. Returns how many were kept.

    A **replace**, not an append: a statement or a terminal read is a full
    snapshot of what is open, so merging would resurrect positions you have
    since closed — and a guard that warns about trades you no longer hold gets
    ignored, which is worse than no guard.

    ``account`` optionally carries the broker's own balance/equity/margin
    snapshot (the live MT5 link supplies it) so the UI can show margin level
    next to the exposure it is warning about.
    """
    cleaned = _clean(rows)
    payload = {
        "positions": cleaned,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    if account:
        payload["account"] = dict(account)
    _db_write(payload)
    _json_write(payload)
    return len(cleaned)


def account_snapshot() -> Optional[Dict[str, Any]]:
    """The broker account figures stored alongside the book, if the feed
    supplied them (live MT5 does; an MT4 statement doesn't)."""
    stored = _db_read() or _json_read() or {}
    acct = stored.get("account")
    return dict(acct) if isinstance(acct, dict) else None


def load() -> List[Dict[str, Any]]:
    """The stored book (DB preferred, then local JSON). Empty list when unset."""
    stored = _db_read() or _json_read() or {}
    return _clean(stored.get("positions"))


def saved_at() -> Optional[str]:
    """ISO timestamp of the last save, for an honest "as of" caption — a stale
    book is a misleading guard, so the UI must be able to show its age."""
    stored = _db_read() or _json_read() or {}
    ts = stored.get("saved_at")
    return str(ts) if ts else None


def clear() -> None:
    """Drop the stored book (flat account, or a bad import)."""
    save([])


def unstopped(rows: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """Positions carrying no stop loss — unbounded risk, worth its own callout."""
    book = load() if rows is None else _clean(rows)
    return [r for r in book if not r.get("has_stop")]
