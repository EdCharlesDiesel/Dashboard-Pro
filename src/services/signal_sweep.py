"""Run every signal page headlessly so its own code persists its signals.

The problem this solves: `signal_store.persist_signals` is called from inside
each page's render body, so a signal is only ever saved when a human opens that
page. The background scanner covers exactly one page (`setup_ranker`, Grade-A
only) plus the house-view board — everything else persists nothing, which is why
the Trade Journal's Source Scorecard has no history to resolve.

This module is a **scheduler, not a second implementation**. It does not
re-derive any signal. It runs the real page under Streamlit's `AppTest` harness
(the same mechanism `tests/test_pages_smoke.py` uses), so the page's own
scoring, its own `persist_signals` call, and its own dedupe ledger all execute
byte-for-byte as they do in a browser. Adding a scoring rule here would be a
bug; the only thing that belongs in this file is *which pages to run and when*.

Run it::

    python -m src.services.signal_sweep              # sync, then every page
    python -m src.services.signal_sweep --only daily_trend,weekly_ema
    python -m src.services.signal_sweep --no-sync    # pages only
    python -m src.services.signal_sweep --list

Each run first calls `sync_broker_state()` to refresh the stored balance and
open-position book from the live MT5 terminal, so the pages score against the
real account rather than whatever was last written. See that function for why
this belongs in the scheduled job rather than being left to a human.

Notes on what it deliberately does and does not touch:

- **`PYTEST_CURRENT_TEST` is set** for the duration. That is the flag
  `background_scanner` checks before starting its daemon thread — without it,
  every page render in the sweep would try to spawn a scanner inside this
  process. It does *not* gate `persist_signals`, which is the whole point.
- **Emails stay live by default**, because alerts-on is the standing preference
  and the alert pages share one `NotifyCache` ledger with the worker, so a swept
  page cannot double-send. Set ``SIGNAL_SWEEP_NO_EMAIL=1`` to stub sends; a
  stubbed send returns ``False``, which per the alert contract leaves the ledger
  untouched so the real alert still fires later.
- **Pages hit live yfinance**, 5-60s each, so a full sweep takes ~10-20 minutes.
  That makes this an hourly/daily cron job, not something to call from a page.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from contextlib import closing
from typing import Dict, List, Optional, Tuple

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

logger = logging.getLogger("ForexDashboard")

# source tag -> page file. The tag must match the string the page passes to
# persist_signals(), because that is what lands in trade_setups.source and what
# the Source Scorecard groups by. tests/test_signal_sweep.py asserts this table
# and the call sites can never drift apart.
PAGES: Tuple[Tuple[str, str], ...] = (
    ("market_overview", "pages/market-overview.py"),
    ("setup_ranker", "pages/setup-ranker.py"),
    ("weekly_ema", "pages/weekly-ema.py"),
    ("weekly_swing", "pages/weekly-swing.py"),
    ("daily_trend", "pages/daily-trend.py"),
    ("daily_macd", "pages/daily-macd.py"),
    ("trend_signals", "pages/trend-signals.py"),
    ("market_structure", "pages/market-structure.py"),
    ("confluence_zone_4h", "pages/4H-confluence-zone.py"),
    ("confluence_checker", "pages/confluence-checker.py"),
    ("twenty_day_breakout", "pages/twenty_day_breakout_tab.py"),
    ("daily_cockpit", "pages/daily_cockpit_tab.py"),
    ("amd_scanner", "pages/amd-scanner.py"),
    ("vwap_ema_gold", "pages/vwap-ema-gold.py"),
    ("fib_entry", "pages/15m-fib-entry.py"),
    ("predictive", "pages/predictive.py"),
    ("dxy_gold", "pages/dxy-gold.py"),
    ("currency_strength", "pages/currency-strength.py"),
    ("instrument_predictor", "pages/instrument-predictor.py"),
    ("cot_composite", "pages/cot_composite_trade_signal.py"),
    ("disconnect_mon", "pages/disconnect_monitor_tab.py"),
    # Imports `arch` at module scope, which has no Python 3.14 wheel and needs
    # MSVC to build — so this page cannot run in the local Windows venv and will
    # report FAILED there. It runs normally on the Linux/3.11 deploy container.
    ("forecast_dashboard", "pages/forecast_tab.py"),
    # Restored from archive/ on 2026-08-04; already carried persist_signals
    # before the 2026-07-03 cleanup archived them.
    ("seasonality", "pages/seasonality.py"),
    ("risk_reversal", "pages/risk_reversal_tab.py"),
    ("smart_money", "pages/smart_money_tab.py"),
    # The desk's own chart stack (mt5/FiboRibbon.mq5) scored in Python.
    ("fibo_ribbon", "pages/fibo-ribbon.py"),
)

DEFAULT_TIMEOUT = 300


def _rr_use_free_proxy(at) -> None:
    """Switch Risk Reversals off its synthetic-demo default.

    The page ships defaulted to "Demo (synthetic)" and deliberately refuses to
    persist a signal derived from made-up data — correct behaviour, but it means
    the sweep's default render can never produce anything. Selecting the free
    return-skew proxy gives it a real (if approximate) input. There is no
    dishonesty here: the page labels the proxy as such in its own caption.
    """
    for box in at.selectbox:
        if "RR data source" in (box.label or ""):
            box.select("Free proxy (return-skew)")
            at.run()
            return


# Swept like the rest, but they write to `tool_usage_log` instead of
# `trade_setups`: their read is a *location* (price at a confluence zone), with no
# direction for the Source Scorecard to resolve. They still need running so the
# audit trail fills — being unscoreable is not the same as being uninteresting.
AUDIT_ONLY = frozenset({"confluence_zone_4h", "confluence_checker"})


# source -> callable(AppTest) run after the first render, for pages whose default
# widget state can't produce a signal. Keep these to *selecting a data source or
# mode*; anything that changes what the page concludes belongs in the page.
PREPARE = {
    "risk_reversal": _rr_use_free_proxy,
}


def _counts_by_source() -> Dict[str, int]:
    """Rows per source currently in ``trade_setups``, or {} when the DB is down.

    Used only to report what a sweep achieved — the sweep itself neither reads
    nor needs this, so a DB hiccup here must never fail the run.
    """
    try:
        from src.db.cache import pooled_repository
        from src.db.market_cache import _resolve_cfg
        cfg = _resolve_cfg()
        if cfg is None:
            return {}
        repo = pooled_repository(cfg)
        # The repository's own connection idiom (see CLAUDE.md): closing handles
        # close, the middle `conn` handles commit/rollback.
        with closing(repo._connect()) as conn, conn, conn.cursor() as cur:  # noqa: SLF001
            cur.execute("SELECT source, COUNT(*) FROM trade_setups GROUP BY source")
            return {str(r[0]): int(r[1]) for r in cur.fetchall()}
    except Exception as exc:
        logger.debug("[sweep] count query failed: %s", exc)
        return {}


def _suppress_email() -> None:
    """Stub `send_email` to a refusal, which leaves the dedupe ledger untouched."""
    try:
        from src.services import alert_service
        alert_service.send_email = lambda *a, **k: (False, "suppressed by signal_sweep")
    except Exception as exc:
        logger.warning("[sweep] could not suppress email: %s", exc)


def run_page(source: str, path: str, timeout: int = DEFAULT_TIMEOUT) -> Dict[str, object]:
    """Run one page under AppTest. Returns a result dict; never raises.

    A page that throws is reported and skipped — one broken page must not cost
    the sweep the other twenty.
    """
    from streamlit.testing.v1 import AppTest

    full = os.path.join(_REPO_ROOT, path)
    result: Dict[str, object] = {"source": source, "page": path, "ok": False,
                                 "saved": 0, "seconds": 0.0, "error": ""}
    if not os.path.exists(full):
        result["error"] = "page file not found"
        return result

    started = time.time()
    try:
        at = AppTest.from_file(full, default_timeout=timeout)
        at.run()
        prepare = PREPARE.get(source)
        if prepare is not None and not at.exception:
            prepare(at)
        errs = [str(e.value) for e in at.exception]
        result["ok"] = not errs
        if errs:
            result["error"] = errs[0][:300]
    except Exception as exc:
        result["error"] = "{0}: {1}".format(type(exc).__name__, exc)[:300]
    result["seconds"] = round(time.time() - started, 1)
    return result


def sync_broker_state() -> Dict[str, object]:
    """Refresh the stored balance and open-position book from the live terminal.

    Runs **before** the pages so they score against the real account: the ranker
    prints lot sizes from whatever balance it has, and the exposure guard checks
    correlation against whatever book it has. Both are durable stores that go
    stale silently — on 2026-08-04 the balance was a 7-week-old MT4 statement
    reading $5,182.84 against a real $1,989 (every lot size 2.6x too large) while
    the position book still held four trades from a different broker entirely.
    Nothing in the app notices that on its own, which is exactly why it belongs
    in the scheduled job.

    Delegates to `mt5_link.sync()` — the read-only terminal link, not the MCP
    server — so this never places, modifies or closes anything. Fails soft for
    the same three reasons `mt5_link` is optional: the package is Windows-only,
    it needs a running logged-in terminal on the same machine, and it only speaks
    MT5. On the Linux deploy container this is a no-op with a message.
    """
    out: Dict[str, object] = {"ok": False, "message": "", "balance": None, "positions": 0}
    try:
        from src.services import mt5_link
    except Exception as exc:
        out["message"] = "mt5_link unavailable: {0}".format(exc)
        return out
    try:
        res = mt5_link.sync()
    except Exception as exc:  # noqa: BLE001 — a stale balance must not abort the sweep
        out["message"] = "sync failed: {0}".format(exc)
        return out
    account = res.get("account") or {}
    out.update(ok=bool(res.get("ok")), message=str(res.get("message") or ""),
               balance=account.get("balance"), positions=int(res.get("saved") or 0))
    return out


def sweep_once(only: Optional[List[str]] = None,
               timeout: int = DEFAULT_TIMEOUT,
               sync: bool = True) -> List[Dict[str, object]]:
    """Run every registered signal page once and report what each persisted."""
    # Keep the background scanner from starting a daemon inside this process;
    # this flag does not gate persist_signals.
    os.environ.setdefault("PYTEST_CURRENT_TEST", "signal_sweep")
    if os.environ.get("SIGNAL_SWEEP_NO_EMAIL", "").strip().lower() in ("1", "true", "yes"):
        _suppress_email()

    if sync:
        state = sync_broker_state()
        print("[sync] {0}{1}".format(
            state["message"] or "no terminal",
            "  balance={0} positions={1}".format(state["balance"], state["positions"])
            if state["ok"] else "  (skipped — balance/book left as-is)"), flush=True)

    targets = [(s, p) for s, p in PAGES if not only or s in only]
    before = _counts_by_source()
    results: List[Dict[str, object]] = []

    for idx, (source, path) in enumerate(targets, 1):
        print("[{0}/{1}] {2:22} {3}".format(idx, len(targets), source, path), flush=True)
        res = run_page(source, path, timeout)
        after = _counts_by_source()
        res["saved"] = after.get(source, 0) - before.get(source, 0)
        before = after
        results.append(res)
        print("        -> {0} in {1}s, saved {2}{3}".format(
            "ok" if res["ok"] else "FAILED", res["seconds"], res["saved"],
            "" if res["ok"] else "  [" + str(res["error"])[:160] + "]"), flush=True)

    return results


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Run every signal page so it persists its signals.")
    ap.add_argument("--only", help="comma-separated source tags to run")
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="per-page seconds")
    ap.add_argument("--list", action="store_true", help="list registered pages and exit")
    ap.add_argument("--no-sync", action="store_true",
                    help="skip the pre-run balance/position sync from the MT5 terminal")
    args = ap.parse_args(argv)

    if args.list:
        for source, path in PAGES:
            print("{0:22} {1}".format(source, path))
        return 0

    only = [s.strip() for s in args.only.split(",")] if args.only else None
    started = time.time()
    results = sweep_once(only=only, timeout=args.timeout, sync=not args.no_sync)

    ok = [r for r in results if r["ok"]]
    saved = sum(int(r["saved"]) for r in results)
    print("\n=== sweep complete in {0}s ===".format(round(time.time() - started)))
    print("pages ok: {0}/{1}   signals saved: {2}".format(len(ok), len(results), saved))
    for r in results:
        if not r["ok"]:
            print("  FAILED {0:22} {1}".format(r["source"], str(r["error"])[:160]))
    return 0 if len(ok) == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
