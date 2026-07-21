"""Durable config for the MT4 statement auto-importer.

The Trade Journal's auto-import watches a file MT4 saves its report to and
re-reads it when it changes (see ``pages/trade-journal.py``). The *path*, the
*auto-on* toggle and the broker's *UTC offset* used to live only in
``st.session_state``, so they reset on every restart — you'd re-type the path
and re-enable auto-import each session. This stores them durably so the
balance link is genuinely set-and-forget.

Same dual-write pattern as ``account_state`` / ``score_history``: Postgres
``app_state`` is the source of truth (survives restart, shared across
devices), a local JSON file is the offline fallback. Streamlit-free — the DB
target is resolved lazily and every DB call is best-effort.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

_PATH = os.path.join(os.getcwd(), "mt4_watch_config.json")
_STATE_KEY = "mt4_watch_config"

_DEFAULTS: Dict[str, Any] = {"path": "", "auto_on": False, "offset": 0.0}


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


def load() -> Dict[str, Any]:
    """Stored watch config, DB-preferred then local JSON, with defaults filled."""
    stored = _db_read() or _json_read() or {}
    out = dict(_DEFAULTS)
    for k in _DEFAULTS:
        if k in stored and stored[k] is not None:
            out[k] = stored[k]
    return out


def save(path: str, auto_on: bool, offset: float) -> None:
    """Persist the watch config durably (Postgres + local JSON fallback)."""
    payload = {"path": str(path or ""), "auto_on": bool(auto_on),
               "offset": float(offset)}
    _db_write(payload)
    _json_write(payload)
