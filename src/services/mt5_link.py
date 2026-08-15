"""Live MetaTrader 5 terminal link — read the real book, not a saved file.

The statement importer (`mt4_import`) reads whatever you last exported. This
reads the terminal *now*: open positions, and the broker's own
balance/equity/margin. That closes the two gaps the exposure guard was left
with — a snapshot that goes stale the moment you open a trade, and a position
book with no account context to size against.

**Read-only by design.** Nothing here places, modifies or closes an order —
the package can, and this module deliberately does not expose it. Terminal
"Algo Trading" being off does not affect anything here.

Optional dependency, three separate reasons it must stay optional:

1. ``MetaTrader5`` is **Windows-only**. The deployment of record is Railway
   Linux containers (see ``DEPLOY-RAILWAY.md``), where the wheel does not
   exist. ``requirements.txt`` carries a ``sys_platform == "win32"`` marker
   and every import here is lazy, so the Linux worker/UI import this module
   happily and simply report "not available".
2. It needs a **running terminal on the same machine**. A server process has
   no terminal to talk to.
3. It only speaks to **MetaTrader 5** terminals. An MT4 terminal will not
   connect no matter what is installed — for MT4, the statement path stays the
   only route.

Same self-disabling pattern as ``arch`` in ``quant_models``: import inside the
function, degrade to a message, never raise at module scope.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

# Position type constants as MT5 reports them (mt5.POSITION_TYPE_BUY/SELL).
_BUY, _SELL = 0, 1


@dataclass(frozen=True)
class AccountSnapshot:
    """The broker's own account figures — authoritative, unlike a typed-in
    balance. ``margin_level`` is the number that decides whether you get to
    keep your positions, so it is carried explicitly rather than derived."""

    login: Optional[int]
    server: str
    currency: str
    balance: float
    equity: float
    margin: float
    margin_free: float
    margin_level: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _import_mt5():
    """The lazy import. Returns the module or ``None`` — never raises, so a
    Linux container or a machine without the package just gets 'unavailable'."""
    try:
        import MetaTrader5 as mt5  # noqa: N813 — vendor's capitalisation
        return mt5
    except Exception:
        return None


def available() -> bool:
    """True when the package can be imported at all (Windows + installed).
    Says nothing about whether a terminal is running — see `probe`."""
    return _import_mt5() is not None


def _fmt_error(mt5) -> str:
    try:
        code, text = mt5.last_error()
        return "{0} ({1})".format(text, code)
    except Exception:
        return "unknown error"


def probe(mt5=None) -> Tuple[bool, str]:
    """``(connected, message)`` — a one-shot connectivity check for the UI.

    Initialises, reads terminal info, and shuts down again. ``mt5`` may be
    injected for testing; production passes nothing.
    """
    mod = mt5 or _import_mt5()
    if mod is None:
        return False, ("MetaTrader5 package not installed (Windows only — "
                       "on Linux the statement import is the only route).")
    if not mod.initialize():
        return False, "Terminal not reachable: {0}".format(_fmt_error(mod))
    try:
        info = mod.terminal_info()
        acct = mod.account_info()
        if info is None:
            return False, "Terminal info unavailable: {0}".format(_fmt_error(mod))
        if acct is None:
            return False, ("Terminal running but not logged in: {0}"
                           .format(_fmt_error(mod)))
        return True, "{0} · build {1} · account {2} @ {3}".format(
            getattr(info, "name", "MT5"), getattr(info, "build", "?"),
            acct.login, acct.server)
    finally:
        try:
            mod.shutdown()
        except Exception:
            pass


def _snapshot_from(acct) -> AccountSnapshot:
    return AccountSnapshot(
        login=getattr(acct, "login", None),
        server=str(getattr(acct, "server", "")),
        currency=str(getattr(acct, "currency", "")),
        balance=float(getattr(acct, "balance", 0.0)),
        equity=float(getattr(acct, "equity", 0.0)),
        margin=float(getattr(acct, "margin", 0.0)),
        margin_free=float(getattr(acct, "margin_free", 0.0)),
        # MT5 reports 0.0 for margin_level when nothing is open; that is "not
        # applicable", not "0% and about to be liquidated".
        margin_level=(float(acct.margin_level)
                      if getattr(acct, "margin_level", 0) else None),
    )


def positions_to_rows(positions, symbol_map: Dict[str, Optional[str]]
                      ) -> Tuple[List[dict], Dict[str, int]]:
    """Map raw MT5 position objects onto stored rows (pure given the objects).

    Built through `open_positions.make_row`, the same constructor the statement
    parser uses — including its "a stop is only real if non-zero" rule, which
    matters here because MT5 reports an absent stop as ``sl = 0.0``.
    """
    from src.instruments.registry import INSTRUMENTS
    from src.services.open_positions import make_row

    rows: List[dict] = []
    skipped: Dict[str, int] = {}

    for p in positions or []:
        symbol = str(getattr(p, "symbol", ""))
        inst = symbol_map.get(symbol)
        if inst is None or inst not in INSTRUMENTS:
            skipped[symbol] = skipped.get(symbol, 0) + 1
            continue

        opened = getattr(p, "time", None)
        opened_iso = None
        if opened:
            try:
                opened_iso = datetime.fromtimestamp(
                    int(opened), tz=timezone.utc).isoformat()
            except Exception:
                opened_iso = None

        rows.append(make_row(
            pair=inst,
            direction="LONG" if getattr(p, "type", _BUY) == _BUY else "SHORT",
            lot_size=getattr(p, "volume", 0.0),
            ticket=getattr(p, "ticket", None),
            entry_price=getattr(p, "price_open", None),
            stop_loss=getattr(p, "sl", None),
            take_profit=getattr(p, "tp", None),
            opened_at=opened_iso,
            label="MT5 #{0}".format(getattr(p, "ticket", "?")),
        ))

    return rows, skipped


def read_terminal(mt5=None) -> Dict[str, Any]:
    """Read open positions + account from the running terminal.

    Returns ``{"ok", "message", "rows", "account", "skipped"}``. Never raises:
    every failure mode (no package, no terminal, not logged in, odd symbol)
    comes back as ``ok=False`` with a message the UI can show.
    """
    mod = mt5 or _import_mt5()
    out: Dict[str, Any] = {"ok": False, "message": "", "rows": [],
                           "account": None, "skipped": {}}
    if mod is None:
        out["message"] = ("MetaTrader5 package not installed (Windows only).")
        return out
    if not mod.initialize():
        out["message"] = "Terminal not reachable: {0}".format(_fmt_error(mod))
        return out
    try:
        acct = mod.account_info()
        if acct is None:
            out["message"] = ("Terminal running but not logged in: {0}"
                              .format(_fmt_error(mod)))
            return out
        snapshot = _snapshot_from(acct)

        raw = mod.positions_get()
        if raw is None:
            # Distinguish "no positions" (empty) from a real read failure.
            code = 0
            try:
                code = mod.last_error()[0]
            except Exception:
                pass
            if code not in (0, 1):
                out["message"] = "Could not read positions: {0}".format(
                    _fmt_error(mod))
                out["account"] = snapshot.to_dict()
                return out
            raw = []

        from src.services.broker_symbols import build_symbol_map
        smap = build_symbol_map([str(getattr(p, "symbol", "")) for p in raw])
        rows, skipped = positions_to_rows(raw, smap)

        out.update(ok=True, rows=rows, account=snapshot.to_dict(),
                   skipped=skipped,
                   message="{0} position(s) from account {1} @ {2}".format(
                       len(rows), snapshot.login, snapshot.server))
        return out
    except Exception as exc:  # noqa: BLE001 — a link read must never break a page
        out["message"] = "MT5 read failed: {0}".format(exc)
        return out
    finally:
        try:
            mod.shutdown()
        except Exception:
            pass


def sync(mt5=None, set_balance: bool = True) -> Dict[str, Any]:
    """Read the terminal and persist it: positions → the exposure store,
    balance → `account_state`.

    This is the whole integration in one call — the Trade Journal button and
    any refresh path use it. Returns `read_terminal`'s dict plus ``saved``.

    Setting the balance is the fix for sizing off a stale default: the ranker
    prints lot sizes from whatever balance it has, and the broker's own figure
    is the only one that can't be wrong.
    """
    result = read_terminal(mt5=mt5)
    result["saved"] = 0
    if not result["ok"]:
        return result

    from src.services.open_positions import save
    result["saved"] = save(result["rows"], account=result["account"])

    if set_balance and result["account"]:
        try:
            from src.services import account_state
            account_state.set_balance(float(result["account"]["balance"]),
                                      source="MT5 terminal")
        except Exception:
            pass  # balance is a bonus; the book is the point
    return result


def margin_warning(account: Optional[Dict[str, Any]],
                   danger: float = 200.0) -> Optional[str]:
    """A plain-language margin-level warning, or ``None`` when healthy.

    ``margin_level`` is equity/margin × 100. Brokers typically margin-call
    around 100% and stop out below that, so anything under ``danger`` is worth
    saying out loud on a page that is about to suggest *more* exposure.
    """
    if not account:
        return None
    level = account.get("margin_level")
    if level is None or level <= 0:
        return None
    if level >= danger:
        return None
    free = account.get("margin_free")
    tail = (" Free margin {0:,.2f} {1}.".format(free, account.get("currency", ""))
            if free is not None else "")
    return ("⚠ Margin level {0:.0f}% — below {1:.0f}%.{2} New positions reduce "
            "it further.".format(level, danger, tail))


def detect_server_utc_offset(mt5=None, symbol: str = "EURUSD",
                             max_tick_age_s: int = 600) -> Dict[str, Any]:
    """Measure the broker's clock against UTC. ``{"ok", "hours", "message"}``.

    Deliberately **not** wired into :func:`closed_deals` automatically. The
    measurement compares the last tick's timestamp to the system clock, so it is
    only meaningful while the market is quoting: run it at a weekend and the
    freshest tick is Friday's close, which would read as a ~48h offset and
    silently rewrite every timestamp in the journal. The weekly import job runs
    Sunday 20:00, squarely inside that dead window — exactly when naive
    auto-detection is at its most confidently wrong.

    So this is a one-shot tool for *finding* the number during trading hours;
    the number itself is then passed explicitly. ``max_tick_age_s`` is the guard:
    a stale tick returns ``ok=False`` rather than a plausible-looking lie.
    """
    mod = mt5 or _import_mt5()
    out: Dict[str, Any] = {"ok": False, "hours": 0.0, "message": ""}
    if mod is None:
        out["message"] = "MetaTrader5 package not installed (Windows only)."
        return out
    if not mod.initialize():
        out["message"] = "Terminal not reachable: {0}".format(_fmt_error(mod))
        return out
    try:
        tick = mod.symbol_info_tick(symbol)
        if tick is None or not getattr(tick, "time", 0):
            out["message"] = "No tick for {0}: {1}".format(symbol, _fmt_error(mod))
            return out
        now = datetime.now(timezone.utc)
        server = datetime.fromtimestamp(int(tick.time), tz=timezone.utc)
        delta_h = (server - now).total_seconds() / 3600.0
        # Broker offsets are whole hours; anything else is tick staleness, not
        # a timezone. Rounding first means the staleness guard measures the
        # residual rather than the offset itself.
        hours = round(delta_h)
        residual_s = abs((delta_h - hours) * 3600.0)
        if residual_s > max_tick_age_s:
            # Plain ASCII: this string reaches a Windows console, which is not
            # always UTF-8, and a mangled byte in a diagnostic is a diagnostic
            # people stop reading.
            out["message"] = (
                "Last {0} tick is {1:.0f}s off a whole-hour offset - market "
                "likely closed. Re-measure during trading hours.".format(
                    symbol, residual_s))
            return out
        out.update(ok=True, hours=float(hours),
                   message="Broker clock is UTC{0:+.0f}".format(hours))
        return out
    except Exception as exc:  # noqa: BLE001 — a link read must never break a page
        out["message"] = "Offset probe failed: {0}".format(exc)
        return out
    finally:
        try:
            mod.shutdown()
        except Exception:
            pass


def closed_deals(days: int = 7, lookback_days: int = 180,
                 mt5=None, broker_utc_offset: float = 0.0) -> Dict[str, Any]:
    """Closed deals whose *exit* falls in the last ``days``, as plain dicts.

    Returns ``{"ok", "message", "deals"}``. Read-only and never raises — same
    contract as :func:`read_terminal`.

    ``lookback_days`` is deliberately much wider than ``days``. A position
    closed yesterday may have been opened months ago, and the round trip is
    only reconstructable with *both* of its deals. Fetching just ``days`` would
    silently drop every such trade — the long holds, which are exactly the ones
    worth grading. The window is filtered down to ``days`` by exit time later,
    in :func:`src.services.mt5_trade_import.pair_deals`, so the wider fetch
    costs a little IO and buys correctness.

    ``broker_utc_offset`` is the broker's timezone in hours east of UTC, the
    same convention as ``mt4_import.to_journal_rows``. MT5 reports deal times as
    the *server's* wall clock packed into a Unix timestamp, so reading them as
    UTC is only correct when the server runs UTC — measured true for this
    account (Exness-MT5Trial9) but not for brokers on UTC+2/+3. Subtracting the
    offset here, at the boundary, means everything downstream — the ``days``
    cutoff in ``pair_deals``, ``session_for_hour``, and the naive-UTC
    ``logged_at`` that ``source_scorecard._bars_after`` compares against price
    bars — sees true UTC and needs no knowledge of the broker.
    Find the value with :func:`detect_server_utc_offset` during trading hours.
    """
    mod = mt5 or _import_mt5()
    out: Dict[str, Any] = {"ok": False, "message": "", "deals": []}
    if mod is None:
        out["message"] = "MetaTrader5 package not installed (Windows only)."
        return out
    if not mod.initialize():
        out["message"] = "Terminal not reachable: {0}".format(_fmt_error(mod))
        return out
    try:
        now = datetime.now(timezone.utc)
        start = now - timedelta(days=max(int(lookback_days), int(days)))
        # The terminal reads these bounds as server time, so convert out of true
        # UTC on the way in — the mirror of the per-deal correction below. The
        # day of padding on the upper bound would hide a few hours' skew, but a
        # window that means what it says is worth two additions.
        skew = timedelta(hours=float(broker_utc_offset or 0.0))
        raw = mod.history_deals_get(start + skew, now + skew + timedelta(days=1))
        if raw is None:
            code = 0
            try:
                code = mod.last_error()[0]
            except Exception:
                pass
            if code not in (0, 1):
                out["message"] = "Could not read history: {0}".format(
                    _fmt_error(mod))
                return out
            raw = []

        deals = []
        for d in raw:
            deals.append({
                "ticket": int(getattr(d, "ticket", 0) or 0),
                "position_id": int(getattr(d, "position_id", 0) or 0),
                "symbol": str(getattr(d, "symbol", "") or ""),
                # 0 = buy, 1 = sell.
                "type": int(getattr(d, "type", -1)),
                # 0 = entry (in), 1 = exit (out), 2 = reversal, 3 = out-by.
                "entry": int(getattr(d, "entry", -1)),
                "volume": float(getattr(d, "volume", 0.0) or 0.0),
                "price": float(getattr(d, "price", 0.0) or 0.0),
                "profit": float(getattr(d, "profit", 0.0) or 0.0),
                "commission": float(getattr(d, "commission", 0.0) or 0.0),
                "swap": float(getattr(d, "swap", 0.0) or 0.0),
                "comment": str(getattr(d, "comment", "") or ""),
                # Server wall clock -> true UTC. Same direction as
                # mt4_import.to_journal_rows: a broker on UTC+3 stamps 15:00 on
                # a fill that happened at 12:00 UTC, so the offset subtracts.
                "time": (datetime.fromtimestamp(
                    int(getattr(d, "time", 0) or 0), tz=timezone.utc)
                    - timedelta(hours=float(broker_utc_offset or 0.0))),
            })
        out.update(ok=True, deals=deals,
                   message="{0} deal(s) over {1}d (broker UTC{2:+.0f})".format(
                       len(deals), lookback_days, float(broker_utc_offset or 0.0)))
        return out
    except Exception as exc:  # noqa: BLE001 — a link read must never break a page
        out["message"] = "MT5 history read failed: {0}".format(exc)
        return out
    finally:
        try:
            mod.shutdown()
        except Exception:
            pass
