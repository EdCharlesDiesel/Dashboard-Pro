"""Live MetaTrader 5 order execution for the Trading Journal's UI — the
Streamlit-safe, opt-in counterpart to ``mt5_link.py``'s read-only link.

**This is a deliberate reversal of a documented design decision.** ``mt5_link``
says outright: "a page has no legitimate reason to place an order, so it keeps
no route to one." This module exists because the desk explicitly asked for one
anyway, with its own independent safety gate — see ``TRADE_JOURNAL_ALLOW_TRADING``
below. Nothing here is reachable unless that flag is set AND the caller passes
``confirm=True`` AND the UI's own two-step Review/Confirm flow was completed.

Why this isn't just ``src/services/mt5_mcp.py`` imported directly: that module
holds one MT5 connection open for its entire process lifetime (it's a standalone
MCP server answering many tool calls a minute) and is explicitly documented as
"never imported by a page" — importing it here would fight ``mt5_link``'s
``initialize()``/``shutdown()``-per-call pattern for the same terminal handle
(their own docs: "either module's `shutdown()` would kill the other's
session"). So this module is a **third, independent copy** of the guardrail
logic (confirm flag, tradability checks, volume rounding, filling-mode
selection, ``order_check`` dry-run), ported onto ``mt5_link``'s connect-per-call
discipline instead of ``mt5_mcp``'s persistent connection. The guardrail logic
is intentionally duplicated, not shared — a known, accepted maintenance cost of
keeping this module safe to run inside the Streamlit process. If you change a
guardrail in one module, check whether the other needs the same fix.

**Never raises** — same contract as ``mt5_link``: every function returns a dict
with at least ``{"success": bool, "error": Optional[str]}``, because a page
must not crash on a broker hiccup. This is a deliberate difference from
``mt5_mcp``, which raises ``MT5Error`` (fine for an MCP client, which can
inspect an exception; a Streamlit page just wants a dict to render).

**Every call attempt is audited**, success or failure, via
``src/services/tool_log.py``'s ``log_tool_usage`` — tool name
``"trade_journal_order"`` — so a rejected or fat-fingered attempt leaves a
record even when nothing reached the broker.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# ── Configuration — a dedicated gate, independent of the MCP server's own
# MT5_ALLOW_TRADING. Enabling agent-driven trading (the MCP server) must never
# silently also enable UI-driven trading, or vice versa. ─────────────────────


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _reload_config() -> None:
    """Re-read env vars. Called at import time and exposed for tests — a page
    process doesn't restart to pick up a changed .env, but this keeps the
    knobs testable without reloading the module."""
    global ALLOW_TRADING, MAX_VOLUME, MAGIC
    ALLOW_TRADING = _flag("TRADE_JOURNAL_ALLOW_TRADING")
    # Deliberately conservative and independent of MT5_MAX_VOLUME (the MCP
    # server's own cap, currently 0.5 lots) — this surface has a weaker
    # human-verification step than the conversational agent one, so it gets a
    # smaller ceiling by default.
    MAX_VOLUME = float(os.environ.get("TRADE_JOURNAL_MAX_VOLUME", "0.1"))
    # Distinct from mt5_mcp's 770001 so orders placed via this UI are
    # identifiable in the broker's own history apart from agent-placed ones.
    MAGIC = int(os.environ.get("TRADE_JOURNAL_MAGIC", "770002"))


ALLOW_TRADING: bool = False
MAX_VOLUME: float = 0.1
MAGIC: int = 770002
_reload_config()


# ── Optional dependency — same lazy-import pattern as mt5_link ──────────────


def _import_mt5():
    try:
        import MetaTrader5 as mt5  # noqa: N813 — vendor's capitalisation
        return mt5
    except Exception:
        return None


def available() -> bool:
    """True when the package can be imported at all (Windows + installed)."""
    return _import_mt5() is not None


def trading_enabled() -> bool:
    """True when the UI's own gate is on. Check this before rendering any
    order-entry control — a disabled gate should degrade to an explanation,
    not a broken form."""
    return ALLOW_TRADING


def _fmt_error(mt5) -> str:
    try:
        code, text = mt5.last_error()
        return "{0} ({1})".format(text, code)
    except Exception:
        return "unknown error"


# ── Pure helpers — ported from mt5_mcp.py, kept in lockstep by hand ─────────


def round_volume(volume: float, step: float, vol_min: float, vol_max: float) -> float:
    """Snap a lot size to the symbol's volume step and clamp to its limits."""
    step = step or 0.01
    snapped = round(round(volume / step) * step, 8)
    return max(vol_min, min(vol_max, snapped))


def pick_filling(filling_mode: int) -> int:
    """Choose a filling policy the symbol actually supports (mt5 constants
    resolved lazily since the package may not be importable at module load)."""
    mt5 = _import_mt5()
    if filling_mode & 1:
        return mt5.ORDER_FILLING_FOK
    if filling_mode & 2:
        return mt5.ORDER_FILLING_IOC
    return mt5.ORDER_FILLING_RETURN


def _resolve_broker_symbol(mt5, pair: str) -> Optional[str]:
    """Registry pair name ('EUR/USD') -> the broker's own symbol string.

    mt5_link/mt5_mcp only ever need the reverse direction (broker -> registry,
    from an already-open position). Opening a *new* position needs this one,
    so it scans the broker's live symbol list and normalises each until one
    matches — no hardcoded suffix guessing, reuses the one shared mapping.
    """
    from src.services.broker_symbols import normalize_symbol

    all_syms = mt5.symbols_get()
    if not all_syms:
        return None
    for s in all_syms:
        if normalize_symbol(s.name) == pair:
            return s.name
    return None


# ── Audit logging — every attempt, not just successes ───────────────────────


def _audit(action: str, params: Dict[str, Any], result: Dict[str, Any]) -> None:
    try:
        from src.services.tool_log import log_tool_usage
        payload = {
            "action": action,
            **params,
            "success": result.get("success"),
            "error": result.get("error"),
            "logged_at": datetime.now(timezone.utc).isoformat(),
        }
        log_tool_usage("trade_journal_order", payload)
    except Exception:
        pass  # logging must never break the trading path either way


# ── Guardrails ────────────────────────────────────────────────────────────


def _guard(mt5, symbol_name: Optional[str], volume: float, confirm: bool) -> Optional[str]:
    """Every check except the broker's own `order_check`. Returns an error
    string, or ``None`` when clear to proceed."""
    if not confirm:
        return ("refused: this places or alters a real order. The UI must "
                "complete its Review step and pass confirm=True.")
    if not ALLOW_TRADING:
        return ("Trade Control is disabled on this deployment. Set "
                "TRADE_JOURNAL_ALLOW_TRADING=1 (and restart the app) to enable it — "
                "off by default, independent of the MCP server's own trading gate.")
    if volume > MAX_VOLUME:
        return ("volume {0} exceeds this UI's TRADE_JOURNAL_MAX_VOLUME cap of {1} "
                "lots (independent of any broker/MCP-side cap).".format(volume, MAX_VOLUME))
    term = mt5.terminal_info()
    if term is not None and not term.trade_allowed:
        return ("the terminal refuses automated trading. Enable Tools > Options > "
                "Expert Advisors > 'Allow algorithmic trading', and the Algo "
                "Trading toolbar button.")
    if symbol_name is not None:
        info = mt5.symbol_info(symbol_name)
        if info is None:
            return "unknown broker symbol {0!r}.".format(symbol_name)
    return None


def _send(mt5, request: Dict[str, Any]) -> Dict[str, Any]:
    """`order_check` as a dry run, then `order_send`. Mirrors mt5_mcp's
    `_send`, but returns a dict instead of raising."""
    check = mt5.order_check(request)
    if check is None:
        return {"success": False, "error": "order_check() failed: " + _fmt_error(mt5)}
    if check.retcode not in (0, mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED):
        return {"success": False,
                "error": "pre-trade check rejected the order: [{0}] {1}".format(
                    check.retcode, check.comment)}

    result = mt5.order_send(request)
    if result is None:
        return {"success": False, "error": "order_send() failed: " + _fmt_error(mt5)}
    out: Dict[str, Any] = {
        "success": result.retcode in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED),
        "retcode": result.retcode,
        "comment": result.comment,
        "order": getattr(result, "order", None),
        "deal": getattr(result, "deal", None),
        "price": getattr(result, "price", None),
        "volume": getattr(result, "volume", None),
    }
    if not out["success"]:
        out["error"] = "[{0}] {1}".format(result.retcode, result.comment)
    else:
        out["error"] = None
    return out


# ── Public API ───────────────────────────────────────────────────────────


def estimate_margin(pair: str, direction: str, volume: float) -> Dict[str, Any]:
    """Margin required for a hypothetical order — read-only, sends nothing,
    no confirm needed. Feeds the Review step's "estimated margin" line."""
    mt5 = _import_mt5()
    out: Dict[str, Any] = {"success": False, "error": None, "margin": None}
    if mt5 is None:
        out["error"] = "MetaTrader5 package not installed."
        return out
    if not mt5.initialize():
        out["error"] = "Terminal not reachable: " + _fmt_error(mt5)
        return out
    try:
        broker_symbol = _resolve_broker_symbol(mt5, pair)
        if broker_symbol is None:
            out["error"] = "no broker symbol found for {0!r}.".format(pair)
            return out
        info = mt5.symbol_info(broker_symbol)
        if info is not None and not info.visible:
            mt5.symbol_select(broker_symbol, True)
        tick = mt5.symbol_info_tick(broker_symbol)
        if tick is None:
            out["error"] = "no live tick for {0!r}.".format(broker_symbol)
            return out
        is_buy = direction == "buy"
        margin = mt5.order_calc_margin(
            mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL,
            broker_symbol, volume, tick.ask if is_buy else tick.bid)
        if margin is None:
            out["error"] = "order_calc_margin() failed: " + _fmt_error(mt5)
            return out
        acct = mt5.account_info()
        out.update(success=True, margin=round(float(margin), 2),
                   broker_symbol=broker_symbol,
                   price=tick.ask if is_buy else tick.bid,
                   margin_free=float(acct.margin_free) if acct else None,
                   currency=str(acct.currency) if acct else None)
        return out
    except Exception as exc:  # noqa: BLE001 — a link read must never break a page
        out["error"] = "margin estimate failed: {0}".format(exc)
        return out
    finally:
        try:
            mt5.shutdown()
        except Exception:
            pass


def open_position(pair: str, direction: str, volume: float,
                  stop_loss: float = 0.0, take_profit: float = 0.0,
                  confirm: bool = False, comment: str = "") -> Dict[str, Any]:
    """Open a market position. REAL MONEY. `pair` is the registry name
    ('EUR/USD'); resolved to the broker's own symbol internally.
    `stop_loss`/`take_profit` are absolute prices, 0 means none."""
    params = {"pair": pair, "direction": direction, "volume": volume,
              "stop_loss": stop_loss, "take_profit": take_profit}
    mt5 = _import_mt5()
    out: Dict[str, Any] = {"success": False, "error": None}
    if mt5 is None:
        out["error"] = "MetaTrader5 package not installed."
        _audit("open", params, out)
        return out
    if not mt5.initialize():
        out["error"] = "Terminal not reachable: " + _fmt_error(mt5)
        _audit("open", params, out)
        return out
    try:
        err = _guard(mt5, None, volume, confirm)
        if err:
            out["error"] = err
            return out
        broker_symbol = _resolve_broker_symbol(mt5, pair)
        if broker_symbol is None:
            out["error"] = "no broker symbol found for {0!r}.".format(pair)
            return out
        err = _guard(mt5, broker_symbol, volume, confirm)
        if err:
            out["error"] = err
            return out
        info = mt5.symbol_info(broker_symbol)
        if not info.visible and not mt5.symbol_select(broker_symbol, True):
            out["error"] = "symbol_select({0!r}) failed: {1}".format(
                broker_symbol, _fmt_error(mt5))
            return out
        info = mt5.symbol_info(broker_symbol)
        tick = mt5.symbol_info_tick(broker_symbol)
        is_buy = direction == "buy"
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": broker_symbol,
            "volume": round_volume(volume, info.volume_step, info.volume_min, info.volume_max),
            "type": mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL,
            "price": tick.ask if is_buy else tick.bid,
            "sl": round(stop_loss, info.digits),
            "tp": round(take_profit, info.digits),
            "deviation": 20,
            "magic": MAGIC,
            "comment": (comment or "trade_journal")[:31],
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": pick_filling(info.filling_mode),
        }
        out = _send(mt5, request)
        out["broker_symbol"] = broker_symbol
        return out
    except Exception as exc:  # noqa: BLE001
        out["success"] = False
        out["error"] = "open_position failed: {0}".format(exc)
        return out
    finally:
        _audit("open", params, out)
        try:
            mt5.shutdown()
        except Exception:
            pass


def close_position(ticket: int, confirm: bool = False,
                   volume: float = 0.0) -> Dict[str, Any]:
    """Close an open position by ticket. REAL MONEY. `volume=0` closes the
    whole position; a smaller value closes it partially."""
    params = {"ticket": ticket, "volume": volume}
    mt5 = _import_mt5()
    out: Dict[str, Any] = {"success": False, "error": None}
    if mt5 is None:
        out["error"] = "MetaTrader5 package not installed."
        _audit("close", params, out)
        return out
    if not mt5.initialize():
        out["error"] = "Terminal not reachable: " + _fmt_error(mt5)
        _audit("close", params, out)
        return out
    try:
        found = mt5.positions_get(ticket=ticket)
        if not found:
            out["error"] = "no open position with ticket {0}.".format(ticket)
            return out
        pos = found[0]
        close_vol = volume or pos.volume
        err = _guard(mt5, pos.symbol, close_vol, confirm)
        if err:
            out["error"] = err
            return out
        info = mt5.symbol_info(pos.symbol)
        close_vol = (round_volume(volume, info.volume_step, info.volume_min,
                                  info.volume_max) if volume else pos.volume)
        if close_vol > pos.volume:
            out["error"] = "cannot close {0} lots of a {1} lot position.".format(
                close_vol, pos.volume)
            return out
        was_buy = pos.type == mt5.POSITION_TYPE_BUY
        tick = mt5.symbol_info_tick(pos.symbol)
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": ticket,
            "symbol": pos.symbol,
            "volume": close_vol,
            "type": mt5.ORDER_TYPE_SELL if was_buy else mt5.ORDER_TYPE_BUY,
            "price": tick.bid if was_buy else tick.ask,
            "deviation": 20,
            "magic": MAGIC,
            "comment": "closed via trade_journal",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": pick_filling(info.filling_mode),
        }
        out = _send(mt5, request)
        if out.get("success"):
            out["closed_volume"] = close_vol
            out["remaining_volume"] = round(pos.volume - close_vol, 8)
        return out
    except Exception as exc:  # noqa: BLE001
        out["success"] = False
        out["error"] = "close_position failed: {0}".format(exc)
        return out
    finally:
        _audit("close", params, out)
        try:
            mt5.shutdown()
        except Exception:
            pass


def modify_position(ticket: int, confirm: bool = False,
                    stop_loss: float = -1.0, take_profit: float = -1.0) -> Dict[str, Any]:
    """Change the stop-loss / take-profit of an open position. REAL MONEY.
    Absolute prices; leave a field at -1 to keep it unchanged, 0 to remove it."""
    params = {"ticket": ticket, "stop_loss": stop_loss, "take_profit": take_profit}
    mt5 = _import_mt5()
    out: Dict[str, Any] = {"success": False, "error": None}
    if mt5 is None:
        out["error"] = "MetaTrader5 package not installed."
        _audit("modify", params, out)
        return out
    if not mt5.initialize():
        out["error"] = "Terminal not reachable: " + _fmt_error(mt5)
        _audit("modify", params, out)
        return out
    try:
        found = mt5.positions_get(ticket=ticket)
        if not found:
            out["error"] = "no open position with ticket {0}.".format(ticket)
            return out
        pos = found[0]
        # Modifying SL/TP carries no new market risk from size, but the
        # confirm/ALLOW_TRADING/Algo-Trading checks still apply — a wrong stop
        # is still a real mistake. Volume isn't the relevant cap here, so pass
        # 0.0 through the shared guard rather than skip it.
        err = _guard(mt5, pos.symbol, 0.0, confirm)
        if err:
            out["error"] = err
            return out
        info = mt5.symbol_info(pos.symbol)
        new_sl = pos.sl if stop_loss < 0 else round(stop_loss, info.digits)
        new_tp = pos.tp if take_profit < 0 else round(take_profit, info.digits)
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol": pos.symbol,
            "sl": new_sl,
            "tp": new_tp,
        }
        out = _send(mt5, request)
        out["stop_loss"] = new_sl
        out["take_profit"] = new_tp
        return out
    except Exception as exc:  # noqa: BLE001
        out["success"] = False
        out["error"] = "modify_position failed: {0}".format(exc)
        return out
    finally:
        _audit("modify", params, out)
        try:
            mt5.shutdown()
        except Exception:
            pass
