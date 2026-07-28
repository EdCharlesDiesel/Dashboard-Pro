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
from datetime import datetime, timezone
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
