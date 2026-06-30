"""Persistent store for the live account balance.

The Trade Journal reads the balance straight off the latest MT4 statement and
writes it here; the Setup Ranker (and any other page that sizes positions) reads
it back.

Two layers, both written on every update so the value is durable and shared:

1. **Postgres (source of truth)** — via the ``app_state`` key/value store
   (``src/db/cache.py``). This survives a restart and is visible across devices.
2. **Local JSON file** — a fallback that keeps the balance available when the DB
   is unconfigured or unreachable (the app must never hard-depend on Postgres).

Reads prefer the DB value and fall back to the JSON file; writes update both.
The module stays Streamlit-free — the DB target is resolved lazily and
every DB call is best-effort, so importing/using this outside a Streamlit runtime
(e.g. unit tests) degrades to the JSON file.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, Optional

_PATH = os.path.join(os.getcwd(), "account_state.json")
_STATE_KEY = "account_balance"


# ── DB layer (best-effort; resolves the same target the pages' data uses) ──────
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
            cfg.host, cfg.port, cfg.dbname, cfg.user, cfg.password, _STATE_KEY
        )
        return val if isinstance(val, dict) else None
    except Exception:
        return None


def _db_write(payload: Dict) -> None:
    cfg = _db_cfg()
    if cfg is None:
        return
    try:
        from src.db import cache as dbcache
        dbcache.set_state(
            cfg.host, cfg.port, cfg.dbname, cfg.user, cfg.password,
            _STATE_KEY, payload,
        )
    except Exception:
        pass


# ── local JSON fallback ────────────────────────────────────────────────────────
def _json_read() -> Dict:
    try:
        if os.path.exists(_PATH):
            with open(_PATH) as fh:
                return json.load(fh)
    except Exception:
        pass
    return {}


def _json_write(payload: Dict) -> None:
    try:
        with open(_PATH, "w") as fh:
            json.dump(payload, fh)
    except Exception:
        pass


# ── public API ─────────────────────────────────────────────────────────────────
def get() -> Dict:
    """Return {balance, source, updated_at} or {} if nothing has been stored.

    Prefers the DB value; falls back to the local JSON file.
    """
    db = _db_read()
    if db and "balance" in db:
        return db
    return _json_read()


def get_balance(default: float = 10000.0) -> float:
    try:
        return float(get().get("balance", default))
    except (TypeError, ValueError):
        return default


def set_balance(balance: float, source: str = "manual") -> None:
    """Store the balance durably (Postgres) and to the local JSON fallback, so it
    survives a restart and is visible across devices."""
    payload = {
        "balance": float(balance),
        "source": source,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    _json_write(payload)   # always — offline fallback
    _db_write(payload)     # best-effort — durable + cached
