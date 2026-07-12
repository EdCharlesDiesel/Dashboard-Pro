"""Persistent store for Setup Ranker's per-pair/direction score history.

Setup Ranker recomputes every setup's score from scratch on each scan (see
``SetupRankerPage._scan``) — there's no built-in notion of a setup going
"stale." This module remembers each pair+direction's last-seen score/grade so
a scan can detect when a previously Grade-A setup falls out of grade, or just
report how much a still-qualifying score moved since the last look.

Same two-layer pattern as ``account_state.py``: Postgres ``app_state`` is the
source of truth (survives restarts, shared across devices); a local JSON file
is the offline fallback. Reads prefer the DB; writes update both. The module
stays Streamlit-free so it degrades to the JSON file outside a DB-configured
runtime (e.g. unit tests).
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional

_PATH = os.path.join(os.getcwd(), "setup_ranker_score_history.json")
_STATE_KEY = "setup_ranker_score_history"


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
def load() -> Dict[str, Dict]:
    """``{"EUR/USD|LONG": {"score", "grade", "pct", "ts"}, ...}`` from the last
    scan. Prefers the DB value; falls back to the local JSON file."""
    db = _db_read()
    if db:
        return db
    return _json_read()


def save(history: Dict[str, Dict]) -> None:
    """Store the full history dict durably (Postgres) and to the local JSON
    fallback, so it survives a restart and is visible across devices."""
    _json_write(history)   # always — offline fallback
    _db_write(history)     # best-effort — durable + shared
