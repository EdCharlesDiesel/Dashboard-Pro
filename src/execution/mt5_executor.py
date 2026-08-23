"""
MT5 executor worker.

Runs on the Windows box with the MetaTrader 5 terminal open and algo trading
enabled. Polls the Postgres queue, gates, sizes, and (only when DRY_RUN is
off AND the kill switch is on) sends the order.

Lives at: src/execution/mt5_executor.py

Run:
    set DATABASE_URL=postgresql://...
    set EXECUTOR_DRY_RUN=1
    python -m src.execution.mt5_executor

DEFAULTS ARE SAFE. DRY_RUN is on and the kill switch is off until you
explicitly change both. Leave it that way for at least a month.
"""

from __future__ import annotations

import logging
import os
import signal as _signal
import socket
import sys
import time
from datetime import datetime, timezone

from sqlalchemy import create_engine, text

from src.execution.gate import (
    AccountState,
    GateConfig,
    MarketSnapshot,
    Signal,
    classify_order_type,
    compute_lots,
    run_gate,
)
from src.execution.queue import (
    claim_batch,
    expire_stale,
    get_executor_state,
    log_event,
    mark_filled,
    mark_placed,
    mark_rejected,
    release_claim,
)

log = logging.getLogger("mt5_executor")

POLL_SECONDS = float(os.environ.get("EXECUTOR_POLL_SECONDS", "3"))
MAGIC = int(os.environ.get("EXECUTOR_MAGIC", "770315"))
DEVIATION = int(os.environ.get("EXECUTOR_DEVIATION", "20"))
WORKER = f"{socket.gethostname()}:{os.getpid()}"

_running = True


def _stop(signum, frame):  # noqa: ARG001
    global _running
    log.info("shutdown signal received; finishing current cycle")
    _running = False


# ---------------------------------------------------------------------------
# MT5 plumbing
# ---------------------------------------------------------------------------

def init_mt5():
    """Connect to the running terminal. Windows only — this is why the
    executor cannot live on Railway alongside the sentry."""
    try:
        import MetaTrader5 as mt5
    except ImportError:
        log.error("MetaTrader5 package unavailable. This worker must run on "
                  "Windows with the terminal installed.")
        sys.exit(1)

    if not mt5.initialize():
        log.error("mt5.initialize() failed: %s", mt5.last_error())
        sys.exit(1)

    info = mt5.account_info()
    if info is None:
        log.error("no account_info — is the terminal logged in?")
        mt5.shutdown()
        sys.exit(1)

    log.info("connected: login=%s server=%s balance=%.2f %s",
             info.login, info.server, info.balance, info.currency)
    if not getattr(info, "trade_allowed", True):
        log.warning("ACCOUNT REPORTS trade_allowed=False — enable algo trading "
                    "in the terminal or nothing will execute")
    return mt5


def snapshot(mt5, symbol: str) -> MarketSnapshot | None:
    info = mt5.symbol_info(symbol)
    if info is None:
        log.warning("symbol %s not found", symbol)
        return None
    if not info.visible and not mt5.symbol_select(symbol, True):
        log.warning("could not select %s in Market Watch", symbol)
        return None

    tick = mt5.symbol_info_tick(symbol)
    if tick is None or tick.bid <= 0 or tick.ask <= 0:
        log.warning("no valid tick for %s", symbol)
        return None

    margin_per_lot = 0.0
    try:
        m = mt5.order_calc_margin(mt5.ORDER_TYPE_BUY, symbol, 1.0, tick.ask)
        margin_per_lot = float(m) if m else 0.0
    except Exception:  # noqa: BLE001
        pass

    return MarketSnapshot(
        symbol=symbol,
        bid=float(tick.bid),
        ask=float(tick.ask),
        point=float(info.point),
        digits=int(info.digits),
        tick_value=float(info.trade_tick_value),
        tick_size=float(info.trade_tick_size),
        volume_min=float(info.volume_min),
        volume_step=float(info.volume_step),
        volume_max=float(info.volume_max),
        trade_allowed=info.trade_mode != mt5.SYMBOL_TRADE_MODE_DISABLED,
        stops_level_points=float(getattr(info, "trade_stops_level", 0) or 0),
        margin_per_lot=margin_per_lot,
    )


def account_state(mt5, enabled: bool, dry_run: bool, daily_loss_r: float) -> AccountState:
    info = mt5.account_info()
    positions = mt5.positions_get() or ()
    mine = [p for p in positions if p.magic == MAGIC]
    return AccountState(
        balance=float(info.balance),
        equity=float(info.equity),
        free_margin=float(info.margin_free),
        open_positions=len(mine),
        open_symbols=tuple(p.symbol for p in mine),
        daily_loss_r=daily_loss_r,
        enabled=enabled,
        dry_run=dry_run,
    )


# ---------------------------------------------------------------------------
# order construction
# ---------------------------------------------------------------------------

def build_request(mt5, sig: Signal, snap: MarketSnapshot, lots: float,
                  order_type: str) -> dict:
    tp = sig.tp1 if sig.tp1 is not None else 0.0
    d = snap.digits

    if order_type == "MARKET":
        px = snap.ask if sig.is_buy else snap.bid
        return {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": sig.symbol,
            "volume": lots,
            "type": mt5.ORDER_TYPE_BUY if sig.is_buy else mt5.ORDER_TYPE_SELL,
            "price": round(px, d),
            "sl": round(sig.stop, d),
            "tp": round(tp, d) if tp else 0.0,
            "deviation": DEVIATION,
            "magic": MAGIC,
            "comment": f"sentry:{sig.signal_id[:12]}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

    type_map = {
        "BUY_LIMIT": mt5.ORDER_TYPE_BUY_LIMIT,
        "BUY_STOP": mt5.ORDER_TYPE_BUY_STOP,
        "SELL_LIMIT": mt5.ORDER_TYPE_SELL_LIMIT,
        "SELL_STOP": mt5.ORDER_TYPE_SELL_STOP,
    }
    return {
        "action": mt5.TRADE_ACTION_PENDING,
        "symbol": sig.symbol,
        "volume": lots,
        "type": type_map[order_type],
        "price": round(sig.entry, d),
        "sl": round(sig.stop, d),
        "tp": round(tp, d) if tp else 0.0,
        "magic": MAGIC,
        "comment": f"sentry:{sig.signal_id[:12]}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_RETURN,
    }


# ---------------------------------------------------------------------------
# processing
# ---------------------------------------------------------------------------

def process(mt5, engine, row: dict, cfg: GateConfig, state: dict) -> None:
    sid = row["signal_id"]
    sig = Signal(
        signal_id=sid,
        symbol=row["symbol"],
        direction=row["direction"],
        entry=float(row["entry"]),
        stop=float(row["stop"]),
        tp1=float(row["tp1"]) if row.get("tp1") is not None else None,
        tp2=float(row["tp2"]) if row.get("tp2") is not None else None,
        risk_pct=float(row["risk_pct"]) if row.get("risk_pct") is not None else None,
        meta=row.get("meta") or {},
    )
    dry = bool(state["dry_run"])

    log_event(engine, event="CLAIMED", signal_id=sid, worker=WORKER, dry_run=dry,
              detail={"symbol": sig.symbol, "direction": sig.direction,
                      "entry": sig.entry, "stop": sig.stop,
                      "attempts": row.get("attempts")})

    snap = snapshot(mt5, sig.symbol)
    if snap is None:
        release_claim(engine, sid)
        log_event(engine, event="ERROR", signal_id=sid, worker=WORKER,
                  dry_run=dry, detail={"reason": "no market snapshot"})
        return

    acct = account_state(mt5, state["enabled"], dry, float(state["daily_loss_r"]))
    now_t = datetime.now(timezone.utc).time()
    gate = run_gate(sig, snap, acct, cfg, now=now_t)

    if not gate.ok:
        reason = "; ".join(gate.reasons)
        log.warning("BLOCKED %s %s: %s", sig.symbol, sig.direction, reason)
        mark_rejected(engine, sid, reason)
        log_event(engine, event="GATE_BLOCK", signal_id=sid, worker=WORKER,
                  dry_run=dry, detail={"reasons": gate.reasons,
                                       "spread_pts": snap.spread_points,
                                       "bid": snap.bid, "ask": snap.ask})
        return

    for w in gate.warnings:
        log.warning("gate warning %s: %s", sid, w)

    risk_pct = sig.risk_pct if sig.risk_pct is not None else cfg.default_risk_pct
    sizing = compute_lots(sig, snap, acct, risk_pct)
    if not sizing.ok:
        mark_rejected(engine, sid, f"sizing: {sizing.reason}")
        log_event(engine, event="GATE_BLOCK", signal_id=sid, worker=WORKER,
                  dry_run=dry, detail={"reasons": [sizing.reason]})
        return

    order_type = classify_order_type(sig, snap)
    req = build_request(mt5, sig, snap, sizing.lots, order_type)

    detail = {
        "symbol": sig.symbol, "direction": sig.direction, "order_type": order_type,
        "lots": sizing.lots, "entry": sig.entry, "stop": sig.stop, "tp": sig.tp1,
        "risk_amount": sizing.risk_amount, "risk_per_lot": sizing.risk_per_lot,
        "raw_lots": sizing.raw_lots, "bid": snap.bid, "ask": snap.ask,
        "spread_pts": snap.spread_points, "sizing_note": sizing.reason,
    }

    # -------------------- the one line that risks money --------------------
    if dry:
        log.info("[DRY RUN] would send %s %s %s lots @ %s sl=%s tp=%s",
                 order_type, sig.symbol, sizing.lots, req["price"],
                 req["sl"], req["tp"])
        mark_rejected(engine, sid, "dry run — not sent", status="CANCELLED")
        log_event(engine, event="DRY_RUN", signal_id=sid, worker=WORKER,
                  dry_run=True, detail=detail)
        return

    result = mt5.order_send(req)
    if result is None:
        release_claim(engine, sid)
        log_event(engine, event="ERROR", signal_id=sid, worker=WORKER,
                  dry_run=False, detail={**detail, "last_error": str(mt5.last_error())})
        return

    detail.update({"retcode": result.retcode, "comment": result.comment,
                   "order": getattr(result, "order", None)})

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        log.error("order_send rejected retcode=%s %s", result.retcode, result.comment)
        mark_rejected(engine, sid, f"retcode {result.retcode}: {result.comment}")
        log_event(engine, event="REJECTED", signal_id=sid, worker=WORKER,
                  dry_run=False, detail=detail)
        return

    ticket = int(getattr(result, "order", 0) or getattr(result, "deal", 0))
    if order_type == "MARKET":
        mark_filled(engine, sid, ticket, float(result.price), sizing.lots)
        log_event(engine, event="FILLED", signal_id=sid, worker=WORKER,
                  dry_run=False, detail={**detail, "fill_price": result.price})
        log.info("FILLED %s %s %s lots @ %s (ticket %s)",
                 sig.symbol, sig.direction, sizing.lots, result.price, ticket)
    else:
        mark_placed(engine, sid, ticket, order_type, sizing.lots)
        log_event(engine, event="SENT", signal_id=sid, worker=WORKER,
                  dry_run=False, detail=detail)
        log.info("PLACED %s %s %s lots @ %s (ticket %s)",
                 order_type, sig.symbol, sizing.lots, sig.entry, ticket)


# ---------------------------------------------------------------------------
# reconciliation
# ---------------------------------------------------------------------------

def reconcile(mt5, engine, lookback_hours: int = 48) -> None:
    """Pull closed deals for our magic number into executed_trades.

    Without this your journal silently diverges from the account, and every
    R-multiple downstream is wrong.
    """
    since = datetime.now(timezone.utc).timestamp() - lookback_hours * 3600
    deals = mt5.history_deals_get(datetime.fromtimestamp(since, tz=timezone.utc),
                                  datetime.now(timezone.utc))
    if not deals:
        return

    closers = [d for d in deals
               if d.magic == MAGIC and d.entry == mt5.DEAL_ENTRY_OUT]
    if not closers:
        return

    with engine.begin() as conn:
        for d in closers:
            sid = None
            if d.comment and d.comment.startswith("sentry:"):
                prefix = d.comment.split(":", 1)[1]
                r = conn.execute(text(
                    "SELECT signal_id, entry, stop FROM pending_signals "
                    "WHERE signal_id LIKE :p LIMIT 1"), {"p": f"{prefix}%"}
                ).mappings().fetchone()
                if r:
                    sid = r["signal_id"]
                    risk = abs(float(r["entry"]) - float(r["stop"]))

            r_mult = None
            if sid and risk > 0 and d.volume > 0:
                # R in money terms: pnl divided by what one R was worth.
                info = mt5.symbol_info(d.symbol)
                if info and info.trade_tick_size > 0:
                    r_money = (risk / info.trade_tick_size) * \
                              info.trade_tick_value * d.volume
                    if r_money > 0:
                        r_mult = float(d.profit) / r_money

            conn.execute(text("""
                INSERT INTO executed_trades
                    (signal_id, ticket, symbol, direction, lots, close_price,
                     closed_at, pnl, r_multiple, dry_run, meta)
                VALUES
                    (:sid, :ticket, :symbol, :direction, :lots, :px,
                     :closed_at, :pnl, :r, FALSE, CAST(:meta AS jsonb))
                ON CONFLICT (ticket) DO UPDATE SET
                    pnl = EXCLUDED.pnl,
                    r_multiple = EXCLUDED.r_multiple,
                    close_price = EXCLUDED.close_price,
                    closed_at = EXCLUDED.closed_at
            """), {
                "sid": sid, "ticket": int(d.ticket), "symbol": d.symbol,
                "direction": "BUY" if d.type == mt5.DEAL_TYPE_SELL else "SELL",
                "lots": float(d.volume), "px": float(d.price),
                "closed_at": datetime.fromtimestamp(d.time, tz=timezone.utc),
                "pnl": float(d.profit), "r": r_mult,
                "meta": "{}",
            })

            # Feed the daily loss counter that the gate reads.
            if r_mult is not None and r_mult < 0:
                conn.execute(text("""
                    UPDATE executor_state
                    SET daily_loss_r = daily_loss_r + :loss, updated_at = now()
                    WHERE id = 1 AND daily_loss_date = CURRENT_DATE
                """), {"loss": abs(r_mult)})


# ---------------------------------------------------------------------------
# main loop
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s %(message)s")

    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        log.error("DATABASE_URL not set. Secrets come from the environment, "
                  "never from source.")
        sys.exit(1)

    engine = create_engine(db_url, pool_pre_ping=True, pool_size=2)
    mt5 = init_mt5()

    cfg = GateConfig(
        symbol_whitelist=tuple(
            s.strip().upper()
            for s in os.environ.get("EXECUTOR_SYMBOLS", "XAUUSD,XAGUSD").split(",")
            if s.strip()),
        default_risk_pct=float(os.environ.get("EXECUTOR_RISK_PCT", "0.5")),
        max_concurrent_positions=int(os.environ.get("EXECUTOR_MAX_POSITIONS", "3")),
        max_daily_loss_r=float(os.environ.get("EXECUTOR_MAX_DAILY_LOSS_R", "3")),
        max_spread_points=float(os.environ.get("EXECUTOR_MAX_SPREAD_PTS", "40")),
    )

    _signal.signal(_signal.SIGINT, _stop)
    _signal.signal(_signal.SIGTERM, _stop)

    log.info("worker %s up | whitelist=%s risk=%.2f%%",
             WORKER, cfg.symbol_whitelist, cfg.default_risk_pct)

    last_reconcile = 0.0
    while _running:
        try:
            state = get_executor_state(engine)
            expire_stale(engine)

            if not state["enabled"]:
                time.sleep(POLL_SECONDS)
                continue

            for row in claim_batch(engine, WORKER, limit=5):
                try:
                    process(mt5, engine, row, cfg, state)
                except Exception:  # noqa: BLE001
                    log.exception("processing failed for %s", row.get("signal_id"))
                    release_claim(engine, row["signal_id"])

            if not state["dry_run"] and time.time() - last_reconcile > 60:
                reconcile(mt5, engine)
                last_reconcile = time.time()

        except Exception:  # noqa: BLE001
            log.exception("cycle failed; backing off")
            time.sleep(10)

        time.sleep(POLL_SECONDS)

    mt5.shutdown()
    log.info("worker %s stopped", WORKER)


if __name__ == "__main__":
    main()
