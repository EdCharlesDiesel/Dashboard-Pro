"""
Execution queue over Postgres.

The sentry (Railway) calls enqueue_signal(). The executor (Windows, MT5)
calls claim_batch(). Neither needs to reach the other over the network.

Lives at: src/execution/queue.py
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import text

log = logging.getLogger(__name__)

__all__ = [
    "make_signal_id",
    "enqueue_signal",
    "claim_batch",
    "release_claim",
    "mark_placed",
    "mark_filled",
    "mark_rejected",
    "expire_stale",
    "get_executor_state",
    "set_executor_state",
    "log_event",
]


# ---------------------------------------------------------------------------
# idempotency
# ---------------------------------------------------------------------------

def make_signal_id(symbol: str, direction: str, entry: float, stop: float,
                   leg_ts: datetime | str, source: str = "evening_sentry") -> str:
    """Stable hash identifying one distinct trade idea.

    `leg_ts` should be the timestamp of the fib leg (or whatever structure the
    setup is anchored to) — NOT the alert time. That is the whole point: the
    09:19 and 09:22 alerts describe the same leg, so they collapse to the same
    id and the second insert is discarded.

    Prices are rounded to 6dp before hashing so float noise cannot split one
    signal into two.
    """
    if isinstance(leg_ts, datetime):
        leg_ts = leg_ts.replace(microsecond=0).isoformat()
    payload = "|".join([
        source, symbol.upper(), direction.upper(),
        f"{float(entry):.6f}", f"{float(stop):.6f}", str(leg_ts),
    ])
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


# ---------------------------------------------------------------------------
# producer side (sentry)
# ---------------------------------------------------------------------------

_INSERT = """
    INSERT INTO pending_signals
        (signal_id, source, symbol, direction, entry, stop, tp1, tp2,
         risk_pct, meta, expires_at)
    VALUES
        (:signal_id, :source, :symbol, :direction, :entry, :stop, :tp1, :tp2,
         :risk_pct, CAST(:meta AS jsonb), :expires_at)
    ON CONFLICT (signal_id) DO NOTHING
    RETURNING id
"""


def enqueue_signal(engine, *, signal_id: str, symbol: str, direction: str,
                   entry: float, stop: float, tp1: float | None = None,
                   tp2: float | None = None, risk_pct: float | None = None,
                   meta: dict | None = None, source: str = "evening_sentry",
                   ttl_minutes: int = 15) -> bool:
    """Insert a signal. Returns True if newly queued, False if it was a dupe.

    Call this from the sentry at the point where the signal object is built —
    BEFORE it is formatted into the Telegram message. Never parse the message
    text back into a trade; the structured object already exists upstream.
    """
    params = {
        "signal_id": signal_id,
        "source": source,
        "symbol": symbol.upper(),
        "direction": direction.upper(),
        "entry": entry,
        "stop": stop,
        "tp1": tp1,
        "tp2": tp2,
        "risk_pct": risk_pct,
        "meta": json.dumps(meta or {}),
        "expires_at": datetime.now(timezone.utc) + timedelta(minutes=ttl_minutes),
    }
    with engine.begin() as conn:
        row = conn.execute(text(_INSERT), params).fetchone()

    if row is None:
        log.info("duplicate signal suppressed: %s %s %s",
                 signal_id, symbol, direction)
        return False
    log.info("queued signal %s: %s %s @ %s", signal_id, symbol, direction, entry)
    return True


# ---------------------------------------------------------------------------
# consumer side (executor)
# ---------------------------------------------------------------------------

_CLAIM = """
    WITH claimable AS (
        SELECT id
        FROM pending_signals
        WHERE status = 'PENDING'
          AND expires_at > now()
          AND attempts < :max_attempts
        ORDER BY created_at
        FOR UPDATE SKIP LOCKED
        LIMIT :limit
    )
    UPDATE pending_signals p
    SET status = 'CLAIMED',
        claimed_at = now(),
        claimed_by = :worker,
        attempts = p.attempts + 1
    FROM claimable c
    WHERE p.id = c.id
    RETURNING p.id, p.signal_id, p.symbol, p.direction, p.entry, p.stop,
              p.tp1, p.tp2, p.risk_pct, p.meta, p.attempts, p.created_at
"""


def claim_batch(engine, worker: str, limit: int = 5,
                max_attempts: int = 3) -> list[dict]:
    """Atomically claim pending signals.

    SKIP LOCKED means two executors can run side by side without ever handing
    the same signal to both — useful when you are testing a new build against
    the live one.
    """
    with engine.begin() as conn:
        rows = conn.execute(text(_CLAIM), {
            "worker": worker, "limit": limit, "max_attempts": max_attempts,
        }).mappings().all()
    return [dict(r) for r in rows]


def release_claim(engine, signal_id: str) -> None:
    """Return a claimed signal to PENDING (transient failure, will retry)."""
    with engine.begin() as conn:
        conn.execute(text("""
            UPDATE pending_signals
            SET status = 'PENDING', claimed_at = NULL, claimed_by = NULL
            WHERE signal_id = :sid AND status = 'CLAIMED'
        """), {"sid": signal_id})


def mark_placed(engine, signal_id: str, ticket: int | None, order_type: str,
                lots: float) -> None:
    with engine.begin() as conn:
        conn.execute(text("""
            UPDATE pending_signals
            SET status = 'PLACED', ticket = :ticket, order_type = :ot,
                lots = :lots, resolved_at = now()
            WHERE signal_id = :sid
        """), {"sid": signal_id, "ticket": ticket, "ot": order_type, "lots": lots})


def mark_filled(engine, signal_id: str, ticket: int, fill_price: float,
                lots: float) -> None:
    with engine.begin() as conn:
        conn.execute(text("""
            UPDATE pending_signals
            SET status = 'FILLED', ticket = :ticket, fill_price = :px,
                lots = :lots, resolved_at = now()
            WHERE signal_id = :sid
        """), {"sid": signal_id, "ticket": ticket, "px": fill_price, "lots": lots})


def mark_rejected(engine, signal_id: str, reason: str,
                  status: str = "REJECTED") -> None:
    with engine.begin() as conn:
        conn.execute(text("""
            UPDATE pending_signals
            SET status = :st, reject_reason = :reason, resolved_at = now()
            WHERE signal_id = :sid
        """), {"sid": signal_id, "reason": reason[:500], "st": status})


def expire_stale(engine) -> int:
    """Expire anything past its TTL, including claims a crashed worker held."""
    with engine.begin() as conn:
        n = conn.execute(text("""
            UPDATE pending_signals
            SET status = 'EXPIRED',
                reject_reason = COALESCE(reject_reason, 'ttl elapsed'),
                resolved_at = now()
            WHERE status IN ('PENDING', 'CLAIMED')
              AND expires_at <= now()
        """)).rowcount
    if n:
        log.info("expired %d stale signals", n)
    return n


# ---------------------------------------------------------------------------
# control + audit
# ---------------------------------------------------------------------------

def get_executor_state(engine) -> dict:
    """Read the kill switch. Also rolls the daily loss counter over midnight."""
    with engine.begin() as conn:
        conn.execute(text("""
            UPDATE executor_state
            SET daily_loss_r = 0, daily_loss_date = CURRENT_DATE,
                updated_at = now()
            WHERE id = 1 AND daily_loss_date < CURRENT_DATE
        """))
        row = conn.execute(text(
            "SELECT enabled, dry_run, halt_reason, daily_loss_r "
            "FROM executor_state WHERE id = 1"
        )).mappings().fetchone()
    return dict(row) if row else {
        "enabled": False, "dry_run": True,
        "halt_reason": "executor_state row missing", "daily_loss_r": 0.0,
    }


def set_executor_state(engine, *, enabled: bool | None = None,
                       dry_run: bool | None = None,
                       halt_reason: str | None = None) -> None:
    """Flip the kill switch. Wire this to a button on the dashboard."""
    sets, params = [], {}
    if enabled is not None:
        sets.append("enabled = :enabled")
        params["enabled"] = enabled
    if dry_run is not None:
        sets.append("dry_run = :dry_run")
        params["dry_run"] = dry_run
    if halt_reason is not None:
        sets.append("halt_reason = :halt_reason")
        params["halt_reason"] = halt_reason
    if not sets:
        return
    sets.append("updated_at = now()")
    with engine.begin() as conn:
        conn.execute(text(
            f"UPDATE executor_state SET {', '.join(sets)} WHERE id = 1"), params)


def log_event(engine, *, event: str, signal_id: str | None = None,
              worker: str | None = None, dry_run: bool = True,
              detail: dict | None = None) -> None:
    """Append to the audit log. Never raises — logging must not break trading."""
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO execution_log (signal_id, worker, event, dry_run, detail)
                VALUES (:sid, :worker, :event, :dry, CAST(:detail AS jsonb))
            """), {"sid": signal_id, "worker": worker, "event": event,
                   "dry": dry_run, "detail": json.dumps(detail or {}, default=str)})
    except Exception:  # noqa: BLE001
        log.exception("failed to write execution_log event=%s", event)
