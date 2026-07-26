"""Precomputed score board — the worker's output the UI reads instead of
computing live.

The background daemon (:mod:`src.services.background_scanner`) recomputes the
whole universe's **house view** and best **Setup Ranker** read every few
minutes and writes the result here as one JSONB row. Pages then read that one
row — a single DB round-trip for the entire consensus board — rather than
fetching three timeframes and recomputing per pair on the request path. This
is what turns the WAN-latency-bound universe pages (MTF Matrix, every
house-view strip) from "recompute 24 pairs while the user waits" into "read
one already-computed snapshot".

Two layers, same durable pattern as :mod:`src.services.account_state`:

1. **Postgres (source of truth)** — the ``app_state`` key/value store, so the
   board survives a restart and is shared across every device/session.
2. **Local JSON file** — an offline fallback so a single-process ``streamlit
   run`` with no DB still benefits from the worker's precompute.

The module is **Streamlit-free** (no ``st`` import): the worker runs in a bare
thread and unit tests import it directly. The one-cached-read-per-rerun wrapper
lives in :mod:`src.services.bias_service` (the UI side). Every DB call is
best-effort — a missing/stale/unreadable board just means the caller falls
back to a live compute, so this can only make the app faster, never break it.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

_PATH = os.path.join(os.getcwd(), "worker_board.json")
_STATE_KEY = "worker_board"

# How old the board may be before a reader ignores it and computes live. The
# worker cycles every 5 min; three missed cycles (15 min) means it's down or
# the market is closed — either way, live compute is the honest fallback.
DEFAULT_MAX_AGE_S = 900


# ── serialization (pure — unit-tested) ──────────────────────────────────────
def serialize_house_view(view) -> Dict:
    """A :class:`src.core.bias.HouseView` → JSON-safe dict (round-trips via
    :func:`deserialize_house_view`). Timestamps become ISO strings."""
    return {
        "pair": view.pair,
        "direction": view.direction,
        "score": view.score,
        "aligned": view.aligned,
        "timeframes": [
            {
                "timeframe": tf.timeframe,
                "direction": tf.direction,
                "score": tf.score,
                "votes": tf.votes,
                "price": tf.price,
                "ema_fast": tf.ema_fast,
                "ema_slow": tf.ema_slow,
                "asof": tf.asof.isoformat() if tf.asof is not None else None,
            }
            for tf in view.timeframes
        ],
    }


def deserialize_house_view(data: Dict):
    """Reconstruct a full :class:`HouseView` (with ``.timeframe()`` / ``.asof``)
    from :func:`serialize_house_view` output, so the shared strip and the MTF
    board render identically whether the view was computed live or read back."""
    import pandas as pd

    from src.core.bias import HouseView, TimeframeBias

    tfs = tuple(
        TimeframeBias(
            timeframe=t["timeframe"],
            direction=t["direction"],
            score=float(t.get("score", 0.0)),
            votes=dict(t.get("votes", {})),
            price=float(t.get("price", 0.0)),
            ema_fast=float(t.get("ema_fast", 0.0)),
            ema_slow=float(t.get("ema_slow", 0.0)),
            asof=pd.Timestamp(t["asof"]) if t.get("asof") else None,
        )
        for t in data.get("timeframes", [])
    )
    return HouseView(
        pair=data["pair"],
        timeframes=tfs,
        direction=data.get("direction", "NEUTRAL"),
        score=float(data.get("score", 0.0)),
        aligned=bool(data.get("aligned", False)),
    )


def build_board(pairs: Dict[str, Dict]) -> Dict:
    """Wrap per-pair entries in the board envelope with a UTC computed-at stamp.

    ``pairs`` maps registry pair → ``{"hv": <serialized house view>, "setup":
    <summary dict|None>}``. Kept pure so the worker's assembly is testable.
    """
    return {
        "computed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pairs": pairs,
    }


def board_age_seconds(board: Optional[Dict]) -> Optional[float]:
    """Seconds since the board was computed, or ``None`` if unknown/unparseable."""
    if not board or "computed_at" not in board:
        return None
    try:
        ts = datetime.fromisoformat(board["computed_at"])
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - ts).total_seconds()
    except Exception:
        return None


def board_is_fresh(board: Optional[Dict], max_age_s: int = DEFAULT_MAX_AGE_S) -> bool:
    """True when the board exists and was computed within ``max_age_s``."""
    age = board_age_seconds(board)
    return age is not None and age <= max_age_s


def house_view_from_board(pair: str, board: Optional[Dict]):
    """The precomputed :class:`HouseView` for ``pair``, or ``None`` when the
    board is empty/missing the pair (caller then computes live)."""
    if not board:
        return None
    entry = board.get("pairs", {}).get(pair)
    if not entry or not entry.get("hv"):
        return None
    try:
        return deserialize_house_view(entry["hv"])
    except Exception:
        return None


def setup_from_board(pair: str, board: Optional[Dict]) -> Optional[Dict]:
    """The precomputed best-direction Setup Ranker summary for ``pair``."""
    if not board:
        return None
    entry = board.get("pairs", {}).get(pair)
    return entry.get("setup") if entry else None


def board_pairs(board: Optional[Dict]) -> List[str]:
    return list(board.get("pairs", {}).keys()) if board else []


# ── DB layer (best-effort; same target the pages' data uses) ─────────────────
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


def _db_write(board: Dict) -> None:
    cfg = _db_cfg()
    if cfg is None:
        return
    try:
        from src.db import cache as dbcache
        dbcache.set_state(
            cfg.host, cfg.port, cfg.dbname, cfg.user, cfg.password,
            _STATE_KEY, board,
        )
    except Exception:
        pass


# ── local JSON fallback ──────────────────────────────────────────────────────
def _json_read() -> Optional[Dict]:
    try:
        if os.path.exists(_PATH):
            with open(_PATH) as fh:
                data = json.load(fh)
            return data if isinstance(data, dict) else None
    except Exception:
        pass
    return None


def _json_write(board: Dict) -> None:
    try:
        with open(_PATH, "w") as fh:
            json.dump(board, fh)
    except Exception:
        pass


# ── public store / read ──────────────────────────────────────────────────────
def store_board(board: Dict) -> None:
    """Persist the board durably (Postgres) and to the local JSON fallback.
    Best-effort — never raises, so a storage failure can't kill the worker."""
    _json_write(board)   # always — offline fallback
    _db_write(board)     # best-effort — durable + shared


def read_board() -> Optional[Dict]:
    """The latest board: prefers the DB value, falls back to the JSON file.
    Returns ``None`` when nothing has been computed yet (caller goes live)."""
    return _db_read() or _json_read()
