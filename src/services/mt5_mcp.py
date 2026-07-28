"""MCP server exposing the live MetaTrader 5 terminal to an AI client.

The sibling of `src/services/mt5_link.py`, and the deliberate counterpart to it:

- **`mt5_link` is read-only by design** and stays that way. It is imported *by
  the Streamlit app* (exposure guard, Trade Journal sync), where an accidental
  order would be a catastrophe and nothing legitimately needs to place one.
- **This module can trade**, because it is a different surface with a different
  consumer: an AI client driving the terminal on explicit instruction, behind
  four independent gates (see "Guardrails" below). It is never imported by a
  page — it is a standalone entrypoint, in the same spirit as
  `evening_sentry` / `news_fetcher`.

Run it as its own process (this is how the registered MCP client launches it)::

    python -m src.services.mt5_mcp

**Why this one holds a connection open and `mt5_link` does not.** `mt5_link`
calls `initialize()`/`shutdown()` around every read, which is right for a page
that touches the terminal occasionally. A tool server answers many calls per
minute, so it initialises once and keeps the handle. The two would fight over
one process — `shutdown()` from either kills the other's session — which is
exactly why this is a **separate process** and why no page may import it.

Same optional-dependency rules as `mt5_link`: `MetaTrader5` is Windows-only and
needs a running, logged-in terminal on the same machine, so the Railway Linux
deploy neither installs nor runs this.
"""
from __future__ import annotations

import os
import sys
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Literal, Optional

# A standalone entrypoint may be launched from any working directory (an MCP
# client picks its own), so anchor `import src...` to the repo root the same way
# `pyproject.toml`'s `pythonpath = ["."]` does for pytest.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import MetaTrader5 as mt5  # noqa: E402,N813 — vendor's capitalisation
from mcp.server.mcpserver import MCPServer  # noqa: E402

from src.instruments.registry import INSTRUMENTS  # noqa: E402
from src.services.broker_symbols import normalize_symbol  # noqa: E402
from src.services.mt5_link import margin_warning  # noqa: E402

# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


TERMINAL_PATH = os.environ.get("MT5_TERMINAL_PATH") or None
LOGIN = os.environ.get("MT5_LOGIN") or None
PASSWORD = os.environ.get("MT5_PASSWORD") or None
SERVER = os.environ.get("MT5_SERVER") or None

ALLOW_TRADING = _flag("MT5_ALLOW_TRADING")
MAX_VOLUME = float(os.environ.get("MT5_MAX_VOLUME", "1.0"))
MAGIC = int(os.environ.get("MT5_MAGIC", "770001"))
_allowed = os.environ.get("MT5_ALLOWED_SYMBOLS", "").strip()
ALLOWED_SYMBOLS = set(s.strip().upper() for s in _allowed.split(",") if s.strip())

TIMEFRAMES = {
    "M1": mt5.TIMEFRAME_M1, "M2": mt5.TIMEFRAME_M2, "M3": mt5.TIMEFRAME_M3,
    "M4": mt5.TIMEFRAME_M4, "M5": mt5.TIMEFRAME_M5, "M6": mt5.TIMEFRAME_M6,
    "M10": mt5.TIMEFRAME_M10, "M12": mt5.TIMEFRAME_M12, "M15": mt5.TIMEFRAME_M15,
    "M20": mt5.TIMEFRAME_M20, "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1,
    "H2": mt5.TIMEFRAME_H2, "H3": mt5.TIMEFRAME_H3, "H4": mt5.TIMEFRAME_H4,
    "H6": mt5.TIMEFRAME_H6, "H8": mt5.TIMEFRAME_H8, "H12": mt5.TIMEFRAME_H12,
    "D1": mt5.TIMEFRAME_D1, "W1": mt5.TIMEFRAME_W1, "MN1": mt5.TIMEFRAME_MN1,
}

Timeframe = Literal[
    "M1", "M2", "M3", "M4", "M5", "M6", "M10", "M12", "M15", "M20", "M30",
    "H1", "H2", "H3", "H4", "H6", "H8", "H12", "D1", "W1", "MN1",
]

# The MetaTrader5 extension is one connection per process and is not
# thread-safe, so every call funnels through this lock.
_lock = threading.RLock()
_connected = False


# --------------------------------------------------------------------------
# pure helpers (unit-tested in tests/test_mt5_mcp.py)
# --------------------------------------------------------------------------


def iso(epoch: float) -> str:
    """MT5 epoch seconds → ISO 8601 UTC. Terminal times are broker server time."""
    return datetime.fromtimestamp(float(epoch), tz=timezone.utc).isoformat()


def stamp(row: Dict[str, Any]) -> Dict[str, Any]:
    """Add `*_iso` alongside each epoch-second time field MT5 returns."""
    for key in ("time", "time_setup", "time_done", "time_expiration", "time_update"):
        val = row.get(key)
        if isinstance(val, (int, float)) and not isinstance(val, bool) and val > 0:
            row[key + "_iso"] = iso(val)
    return row


def round_volume(volume: float, step: float, vol_min: float, vol_max: float) -> float:
    """Snap a lot size to the symbol's volume step and clamp to its limits.

    A broker rejects 0.037 lots outright; rounding here turns a hard rejection
    into the nearest legal size.
    """
    step = step or 0.01
    snapped = round(round(volume / step) * step, 8)
    return max(vol_min, min(vol_max, snapped))


def pick_filling(filling_mode: int) -> int:
    """Choose a filling policy the symbol actually supports.

    `SYMBOL_FILLING_FOK` is bit 0, `SYMBOL_FILLING_IOC` is bit 1. A request
    carrying an unsupported policy fails with "Unsupported filling mode", which
    is one of the most common reasons a correct-looking order never fills.
    """
    if filling_mode & 1:
        return mt5.ORDER_FILLING_FOK
    if filling_mode & 2:
        return mt5.ORDER_FILLING_IOC
    return mt5.ORDER_FILLING_RETURN


def instrument_for(symbol: str) -> Optional[str]:
    """Broker symbol → registry display name ('EURUSDm' → 'EUR/USD'), or None.

    Delegates to `broker_symbols.normalize_symbol`, the same mapping the
    statement importer and `mt5_link` use — never a second copy.
    """
    return normalize_symbol(symbol)


# --------------------------------------------------------------------------
# connection + guardrails
# --------------------------------------------------------------------------


class MT5Error(RuntimeError):
    """Raised with the terminal's own error code and message attached."""


def _fail(what: str) -> MT5Error:
    try:
        code, text = mt5.last_error()
    except Exception:
        code, text = "?", "unknown error"
    return MT5Error("{0} failed: [{1}] {2}".format(what, code, text))


def _connect() -> None:
    """Attach to (or launch) the terminal. Idempotent; holds the handle open."""
    global _connected
    if _connected and mt5.terminal_info() is not None:
        return

    kwargs: Dict[str, Any] = {}
    if LOGIN:
        kwargs["login"] = int(LOGIN)
    if PASSWORD:
        kwargs["password"] = PASSWORD
    if SERVER:
        kwargs["server"] = SERVER

    ok = mt5.initialize(TERMINAL_PATH, **kwargs) if TERMINAL_PATH else mt5.initialize(**kwargs)
    if not ok:
        raise _fail("mt5.initialize()")
    _connected = True


def _asdict(obj: Any) -> Dict[str, Any]:
    return obj._asdict() if hasattr(obj, "_asdict") else dict(obj)


def _symbol(name: str) -> Any:
    """Resolve a symbol, pulling it into Market Watch if the terminal hides it."""
    name = str(name).strip()
    info = mt5.symbol_info(name)
    if info is None:
        raise MT5Error(
            "unknown symbol {0!r} — brokers add suffixes, so call list_symbols "
            "to find the exact spelling".format(name))
    if not info.visible and not mt5.symbol_select(name, True):
        raise _fail("symbol_select({0!r})".format(name))
    return mt5.symbol_info(name)


def _confirmed(confirm: bool) -> None:
    if not confirm:
        raise MT5Error(
            "refused: this places or alters a real order. Re-call with confirm=true "
            "once the user has agreed to the symbol, direction, volume and stop.")


def _check_tradable(symbol: str, volume: float) -> None:
    """Every guardrail except `confirm`, applied before anything reaches the broker."""
    if not ALLOW_TRADING:
        raise MT5Error(
            "trading is disabled on this server. Set MT5_ALLOW_TRADING=1 in the "
            "MCP server config and restart it to enable order execution.")
    if ALLOWED_SYMBOLS and symbol.upper() not in ALLOWED_SYMBOLS:
        raise MT5Error("symbol {0!r} is not in MT5_ALLOWED_SYMBOLS ({1})".format(
            symbol, sorted(ALLOWED_SYMBOLS)))
    if volume > MAX_VOLUME:
        raise MT5Error("volume {0} exceeds the MT5_MAX_VOLUME cap of {1} lots".format(
            volume, MAX_VOLUME))
    term = mt5.terminal_info()
    if term is not None and not term.trade_allowed:
        raise MT5Error(
            "the terminal refuses automated trading. Enable Tools > Options > "
            "Expert Advisors > 'Allow algorithmic trading', and the Algo Trading "
            "toolbar button.")


def _send(request: Dict[str, Any]) -> Dict[str, Any]:
    """`order_check` as a dry run, then `order_send`. Never sends a doomed request."""
    check = mt5.order_check(request)
    if check is None:
        raise _fail("mt5.order_check()")
    if check.retcode not in (0, mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED):
        raise MT5Error("pre-trade check rejected the order: [{0}] {1}".format(
            check.retcode, check.comment))

    result = mt5.order_send(request)
    if result is None:
        raise _fail("mt5.order_send()")
    out = _asdict(result)
    if hasattr(out.get("request"), "_asdict"):
        out["request"] = _asdict(out["request"])
    out["success"] = result.retcode in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED)
    if not out["success"]:
        out["error"] = "[{0}] {1}".format(result.retcode, result.comment)
    return out


def _position(ticket: int) -> Any:
    found = mt5.positions_get(ticket=ticket)
    if not found:
        raise MT5Error("no open position with ticket {0}".format(ticket))
    return found[0]


mcp = MCPServer(
    name="mt5",
    version="0.1.0",
    instructions=(
        "The live MetaTrader 5 terminal behind the Dashboard-Pro desk. Read tools "
        "(account, positions, quotes, candles, history) are safe. Trading tools move "
        "real money: always state symbol, direction, volume and stop-loss back to the "
        "user and get agreement before calling them, and never invent a lot size the "
        "user did not give. Positions carry a `pair` field naming the app's registry "
        "instrument, which is how they line up with the dashboard's own pages."
    ),
)


# --------------------------------------------------------------------------
# read tools
# --------------------------------------------------------------------------


@mcp.tool()
def mt5_status() -> Dict[str, Any]:
    """Connection health: terminal build, broker link, and this server's guardrails.

    Call this first whenever something is not working.
    """
    with _lock:
        try:
            _connect()
        except MT5Error as exc:
            return {"connected": False, "error": str(exc),
                    "hint": "Is the MT5 terminal running and logged in?",
                    "trading_enabled": ALLOW_TRADING}
        term = _asdict(mt5.terminal_info())
        acct = mt5.account_info()
        return {
            "connected": True,
            "package_version": mt5.__version__,
            "terminal": {
                "name": term.get("name"),
                "company": term.get("company"),
                "build": term.get("build"),
                "path": term.get("path"),
                "connected_to_broker": term.get("connected"),
                "algo_trading_allowed": term.get("trade_allowed"),
            },
            "account": ({
                "login": acct.login,
                "server": acct.server,
                "currency": acct.currency,
                "type": "demo" if acct.trade_mode == mt5.ACCOUNT_TRADE_MODE_DEMO else "REAL",
            } if acct else None),
            "guardrails": {
                "trading_enabled": ALLOW_TRADING,
                "max_volume_lots": MAX_VOLUME,
                "allowed_symbols": sorted(ALLOWED_SYMBOLS) or "any",
                "magic_number": MAGIC,
            },
        }


@mcp.tool()
def account_info() -> Dict[str, Any]:
    """Balance, equity, margin, leverage — the broker's own figures.

    Includes the shared margin-level warning the dashboard's exposure guard uses.
    """
    with _lock:
        _connect()
        acct = mt5.account_info()
        if acct is None:
            raise _fail("mt5.account_info()")
        out = _asdict(acct)
        out["is_demo"] = acct.trade_mode == mt5.ACCOUNT_TRADE_MODE_DEMO
        out["margin_warning"] = margin_warning({
            "margin_level": acct.margin_level,
            "margin_free": acct.margin_free,
            "currency": acct.currency,
        })
        return out


@mcp.tool()
def list_symbols(search: str = "", limit: int = 50) -> Dict[str, Any]:
    """Search tradable broker symbols by case-insensitive substring.

    `pair` gives the app's registry name where the symbol maps to one, so you can
    tie a broker symbol to the dashboard's own instrument list.
    """
    with _lock:
        _connect()
        symbols = mt5.symbols_get()
        if symbols is None:
            raise _fail("mt5.symbols_get()")
        needle = search.strip().upper()
        matches = [s for s in symbols if not needle or needle in s.name.upper()]
        return {
            "total_matching": len(matches),
            "returned": min(len(matches), limit),
            "symbols": [{
                "name": s.name,
                "pair": instrument_for(s.name),
                "description": s.description,
                "digits": s.digits,
                "spread": s.spread,
                "volume_min": s.volume_min,
                "volume_max": s.volume_max,
                "volume_step": s.volume_step,
                "in_market_watch": s.visible,
            } for s in matches[:limit]],
        }


@mcp.tool()
def symbol_info(symbol: str) -> Dict[str, Any]:
    """Full contract spec plus the current tick.

    Read this before sizing an order — `volume_min`/`volume_step`/`digits` and
    `stops_level_points` (the minimum distance a stop may sit from price) are
    what decide whether an order is legal.
    """
    with _lock:
        _connect()
        info = _symbol(symbol)
        tick = mt5.symbol_info_tick(info.name)
        pair = instrument_for(info.name)
        return {
            "symbol": info.name,
            "pair": pair,
            "registry_pip_size": (INSTRUMENTS[pair]["pip_size"]
                                  if pair and pair in INSTRUMENTS else None),
            "description": info.description,
            "digits": info.digits,
            "point": info.point,
            "spread": info.spread,
            "contract_size": info.trade_contract_size,
            "volume_min": info.volume_min,
            "volume_max": info.volume_max,
            "volume_step": info.volume_step,
            "stops_level_points": info.trade_stops_level,
            "swap_long": info.swap_long,
            "swap_short": info.swap_short,
            "currency_base": info.currency_base,
            "currency_profit": info.currency_profit,
            "currency_margin": info.currency_margin,
            "tick": stamp(_asdict(tick)) if tick else None,
        }


@mcp.tool()
def get_quote(symbols: list[str]) -> Dict[str, Any]:
    """Current bid/ask/spread for one or more symbols."""
    with _lock:
        _connect()
        quotes = []
        for name in symbols:
            info = _symbol(name)
            tick = mt5.symbol_info_tick(info.name)
            if tick is None:
                raise _fail("symbol_info_tick({0!r})".format(name))
            quotes.append({
                "symbol": info.name,
                "pair": instrument_for(info.name),
                "bid": tick.bid,
                "ask": tick.ask,
                "spread_points": (round((tick.ask - tick.bid) / info.point)
                                  if info.point else None),
                "time": iso(tick.time),
            })
        return {"quotes": quotes}


@mcp.tool()
def get_candles(symbol: str, timeframe: Timeframe = "H1", count: int = 100,
                start_pos: int = 0) -> Dict[str, Any]:
    """OHLCV candles by bar position. `start_pos=0` is the still-forming bar.

    Broker terminal data, not yfinance — this is the execution venue's own
    prices, so it can legitimately differ slightly from the dashboard's charts.
    """
    with _lock:
        _connect()
        info = _symbol(symbol)
        count = max(1, min(count, 5000))
        rates = mt5.copy_rates_from_pos(info.name, TIMEFRAMES[timeframe], start_pos, count)
        if rates is None:
            raise _fail("copy_rates_from_pos({0!r})".format(symbol))
        return {
            "symbol": info.name,
            "pair": instrument_for(info.name),
            "timeframe": timeframe,
            "count": len(rates),
            "candles": [{
                "time": iso(r["time"]),
                "open": float(r["open"]), "high": float(r["high"]),
                "low": float(r["low"]), "close": float(r["close"]),
                "tick_volume": int(r["tick_volume"]),
                "spread": int(r["spread"]),
                "real_volume": int(r["real_volume"]),
            } for r in rates],
        }


@mcp.tool()
def get_candles_range(symbol: str, timeframe: Timeframe, date_from: str,
                      date_to: str) -> Dict[str, Any]:
    """OHLCV candles between two UTC timestamps (ISO 8601, '2026-07-01T00:00:00')."""
    with _lock:
        _connect()
        info = _symbol(symbol)
        rates = mt5.copy_rates_range(info.name, TIMEFRAMES[timeframe],
                                     datetime.fromisoformat(date_from),
                                     datetime.fromisoformat(date_to))
        if rates is None:
            raise _fail("copy_rates_range({0!r})".format(symbol))
        return {
            "symbol": info.name,
            "pair": instrument_for(info.name),
            "timeframe": timeframe,
            "count": len(rates),
            "candles": [{
                "time": iso(r["time"]),
                "open": float(r["open"]), "high": float(r["high"]),
                "low": float(r["low"]), "close": float(r["close"]),
                "tick_volume": int(r["tick_volume"]),
            } for r in rates],
        }


@mcp.tool()
def get_positions(symbol: str = "") -> Dict[str, Any]:
    """Open positions with floating P/L, each tagged with its registry `pair`.

    This is the live book — the same read the Trade Journal's MT5 sync persists,
    but without writing anything.
    """
    with _lock:
        _connect()
        positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
        positions = positions or ()
        rows = []
        for p in positions:
            row = stamp(_asdict(p))
            row["direction"] = "buy" if p.type == mt5.POSITION_TYPE_BUY else "sell"
            row["pair"] = instrument_for(p.symbol)
            rows.append(row)
        return {
            "count": len(rows),
            "total_profit": round(sum(p.profit for p in positions), 2),
            "positions": rows,
        }


@mcp.tool()
def get_pending_orders(symbol: str = "") -> Dict[str, Any]:
    """Working (not yet filled) pending orders."""
    with _lock:
        _connect()
        orders = mt5.orders_get(symbol=symbol) if symbol else mt5.orders_get()
        orders = orders or ()
        rows = []
        for o in orders:
            row = stamp(_asdict(o))
            row["pair"] = instrument_for(o.symbol)
            rows.append(row)
        return {"count": len(rows), "orders": rows}


@mcp.tool()
def get_history(days: int = 7, symbol: str = "") -> Dict[str, Any]:
    """Closed deals from the last N days, with realised P/L, commission and swap."""
    with _lock:
        _connect()
        end = datetime.now(timezone.utc) + timedelta(days=1)
        start = end - timedelta(days=max(1, days) + 1)
        deals = (mt5.history_deals_get(start, end, group="*{0}*".format(symbol))
                 if symbol else mt5.history_deals_get(start, end))
        deals = deals or ()
        return {
            "days": days,
            "count": len(deals),
            "realized_profit": round(sum(d.profit for d in deals), 2),
            "total_commission": round(sum(d.commission for d in deals), 2),
            "total_swap": round(sum(d.swap for d in deals), 2),
            "deals": [stamp(_asdict(d)) for d in deals],
        }


@mcp.tool()
def calc_margin(symbol: str, direction: Literal["buy", "sell"],
                volume: float) -> Dict[str, Any]:
    """Margin required for a hypothetical order. A sizing aid — sends nothing."""
    with _lock:
        _connect()
        info = _symbol(symbol)
        tick = mt5.symbol_info_tick(info.name)
        is_buy = direction == "buy"
        margin = mt5.order_calc_margin(
            mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL,
            info.name, volume, tick.ask if is_buy else tick.bid)
        if margin is None:
            raise _fail("mt5.order_calc_margin()")
        acct = mt5.account_info()
        return {
            "symbol": info.name,
            "pair": instrument_for(info.name),
            "direction": direction,
            "volume": volume,
            "price": tick.ask if is_buy else tick.bid,
            "margin_required": margin,
            "account_currency": acct.currency if acct else None,
            "free_margin": acct.margin_free if acct else None,
            "affordable": bool(acct and margin <= acct.margin_free),
        }


# --------------------------------------------------------------------------
# trade tools — gated on MT5_ALLOW_TRADING, confirm=True, MT5_MAX_VOLUME,
# and the terminal's own Algo Trading switch
# --------------------------------------------------------------------------


@mcp.tool()
def open_position(symbol: str, direction: Literal["buy", "sell"], volume: float,
                  confirm: bool = False, stop_loss: float = 0.0,
                  take_profit: float = 0.0, deviation_points: int = 20,
                  comment: str = "") -> Dict[str, Any]:
    """Open a position at market. REAL MONEY.

    State the symbol, direction, volume and stop-loss to the user and get
    agreement, then call with confirm=true. `stop_loss`/`take_profit` are
    absolute prices; 0 means none. The desk's risk model is SL = 1.5 x ATR14.
    """
    with _lock:
        _connect()
        _confirmed(confirm)
        info = _symbol(symbol)
        _check_tradable(info.name, volume)
        is_buy = direction == "buy"
        tick = mt5.symbol_info_tick(info.name)
        return _send({
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": info.name,
            "volume": round_volume(volume, info.volume_step, info.volume_min, info.volume_max),
            "type": mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL,
            "price": tick.ask if is_buy else tick.bid,
            "sl": round(stop_loss, info.digits),
            "tp": round(take_profit, info.digits),
            "deviation": deviation_points,
            "magic": MAGIC,
            "comment": comment[:31],
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": pick_filling(info.filling_mode),
        })


@mcp.tool()
def close_position(ticket: int, confirm: bool = False,
                   volume: float = 0.0) -> Dict[str, Any]:
    """Close an open position by ticket. REAL MONEY.

    `volume=0` closes the whole position; a smaller value closes it partially.
    """
    with _lock:
        _connect()
        _confirmed(confirm)
        pos = _position(ticket)
        info = _symbol(pos.symbol)
        _check_tradable(info.name, volume or pos.volume)
        close_vol = (round_volume(volume, info.volume_step, info.volume_min,
                                  info.volume_max) if volume else pos.volume)
        if close_vol > pos.volume:
            raise MT5Error("cannot close {0} lots of a {1} lot position".format(
                close_vol, pos.volume))
        was_buy = pos.type == mt5.POSITION_TYPE_BUY
        tick = mt5.symbol_info_tick(info.name)
        out = _send({
            "action": mt5.TRADE_ACTION_DEAL,
            "position": ticket,
            "symbol": info.name,
            "volume": close_vol,
            "type": mt5.ORDER_TYPE_SELL if was_buy else mt5.ORDER_TYPE_BUY,
            "price": tick.bid if was_buy else tick.ask,
            "deviation": 20,
            "magic": MAGIC,
            "comment": "closed via mcp",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": pick_filling(info.filling_mode),
        })
        out["closed_volume"] = close_vol
        out["remaining_volume"] = round(pos.volume - close_vol, 8)
        return out


@mcp.tool()
def modify_position(ticket: int, confirm: bool = False, stop_loss: float = -1.0,
                    take_profit: float = -1.0) -> Dict[str, Any]:
    """Change the stop-loss / take-profit of an open position. REAL MONEY.

    Absolute prices. Leave a field at -1 to keep it unchanged; pass 0 to remove it.
    """
    with _lock:
        _connect()
        _confirmed(confirm)
        pos = _position(ticket)
        info = _symbol(pos.symbol)
        _check_tradable(info.name, 0.0)
        new_sl = pos.sl if stop_loss < 0 else round(stop_loss, info.digits)
        new_tp = pos.tp if take_profit < 0 else round(take_profit, info.digits)
        out = _send({
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol": info.name,
            "sl": new_sl,
            "tp": new_tp,
        })
        out["stop_loss"] = new_sl
        out["take_profit"] = new_tp
        return out


@mcp.tool()
def place_pending_order(symbol: str,
                        order_type: Literal["buy_limit", "sell_limit", "buy_stop", "sell_stop"],
                        volume: float, price: float, confirm: bool = False,
                        stop_loss: float = 0.0, take_profit: float = 0.0,
                        comment: str = "") -> Dict[str, Any]:
    """Place a pending limit/stop order. REAL MONEY once it triggers."""
    with _lock:
        _connect()
        _confirmed(confirm)
        info = _symbol(symbol)
        _check_tradable(info.name, volume)
        types = {
            "buy_limit": mt5.ORDER_TYPE_BUY_LIMIT,
            "sell_limit": mt5.ORDER_TYPE_SELL_LIMIT,
            "buy_stop": mt5.ORDER_TYPE_BUY_STOP,
            "sell_stop": mt5.ORDER_TYPE_SELL_STOP,
        }
        return _send({
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": info.name,
            "volume": round_volume(volume, info.volume_step, info.volume_min, info.volume_max),
            "type": types[order_type],
            "price": round(price, info.digits),
            "sl": round(stop_loss, info.digits),
            "tp": round(take_profit, info.digits),
            "magic": MAGIC,
            "comment": comment[:31],
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": pick_filling(info.filling_mode),
        })


@mcp.tool()
def cancel_pending_order(ticket: int, confirm: bool = False) -> Dict[str, Any]:
    """Cancel a working pending order by ticket."""
    with _lock:
        _connect()
        _confirmed(confirm)
        orders = mt5.orders_get(ticket=ticket)
        if not orders:
            raise MT5Error("no working pending order with ticket {0}".format(ticket))
        _check_tradable(orders[0].symbol, 0.0)
        return _send({"action": mt5.TRADE_ACTION_REMOVE, "order": ticket})


if __name__ == "__main__":
    mcp.run("stdio")
