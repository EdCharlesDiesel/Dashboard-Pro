"""
threat_sentry_hook.py — evening_sentry v2 integration.

Same state-change pattern sentry already uses: evaluate, compare with the
last journaled state, alert Telegram only on transitions (green->amber,
amber->red, or any de-escalation), journal every run.

Wiring inside evening_sentry.py:

    from src.core.threat_sentry_hook import run_threat_check
    ...
    # inside the 17:00-20:00 SAST monitoring loop, e.g. every 15 min:
    run_threat_check(engine, equity=get_current_equity())

Environment (same vars sentry already uses for Telegram):
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
"""

from __future__ import annotations

import os

import requests

from src.core import threat_core as tc
from src.services import open_positions

# Matches the pages. A book older than this is not judged at all.
STALE_AFTER_MIN = 15

ESCALATION_ORDER = {"green": 0, "amber": 1, "red": 2}


def _send_telegram(msg: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat:
        print("[threat] Telegram env vars missing; alert not sent:\n" + msg)
        return
    requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={"chat_id": chat, "text": msg, "parse_mode": "Markdown"},
        timeout=10,
    )


def _format_alert(prev: str | None, rep: tc.ThreatReport) -> str:
    d = rep.detail
    arrow = f"{prev or 'none'} → {rep.state}"
    icon = {"green": "🟢", "amber": "🟠", "red": "🔴"}[rep.state]
    lines = [
        f"{icon} *Threat Board: {arrow}*  ({rep.score}/100)",
        f"USDJPY {d['usdjpy_last']}  ({d['usdjpy_roc5_pct']:+.2f}%/5d)",
        f"Worst correlated stop-out: ${d['worst_cluster_usd']:,.0f} "
        f"({d['worst_cluster_pct_equity']}% eq, {d['worst_cluster_ccy']})",
    ]
    # Prefer the components that actually set the headline. Since 1.10.26 the
    # state follows the worst component, so the highest raw score and the
    # driver can differ - two sources for one fact drift apart.
    driver = d.get("state_driver")
    if driver:
        lines.append("Driver: " + ", ".join(
            f"{n.title()} ({rep.components.get(n, 0):.0f})" for n in driver))
    else:                                   # reports journaled before 1.10.26
        top = max(rep.components, key=rep.components.get)
        lines.append(f"Top driver: {top} ({rep.components[top]:.0f})")
    if d.get("unstopped"):
        # Unbounded risk is the loudest thing this sentry can have to say.
        lines.append("⚠️ NO STOP on: " + ", ".join(d["unstopped"]))
    if d["headline_hits"]:
        lines.append("⚠️ Verbal intervention: " + d["headline_hits"][0][:120])
    if d["red_events"]:
        lines.append(f"📅 {len(d['red_events'])} red event(s) on your pairs in 7d")
    return "\n".join(lines)


def run_threat_check(engine, equity: float | None = None,
                     zone=(tc.JPY_ZONE_LOW, tc.JPY_ZONE_HIGH)) -> tc.ThreatReport | None:
    """One evaluation cycle. Alerts only on state transition. Always journals.

    Reads the same MT5 book and the same equity the Threat Board page reads.
    It used to read `threat_positions`, the hand-typed table the page stopped
    using on 2026-08-20 - which by then was empty, so every run returned None
    and the sentry was silent by construction.

    ``equity`` defaults to the stored account snapshot. If neither an argument
    nor a snapshot supplies one, this returns None rather than falling back to
    a constant: the hardcoded $935 that sat in this file's smoke test was wrong
    by 4x and had been for months.

    Returns the report, or None when there is nothing trustworthy to judge.
    """
    book = open_positions.load()
    positions, unstopped = tc.positions_from_book(book)

    if equity is None:
        snap = open_positions.account_snapshot() or {}
        equity = float(snap.get("equity") or 0.0) or None
    if not equity:
        print("[threat] no equity available - skipping rather than guessing")
        return None

    # A stale book must not be judged. Evaluating a ten-hour-old book can
    # report green about positions that were closed hours ago - and a sentry
    # that simply goes quiet when its feed dies is worse, because silence is
    # indistinguishable from "all clear". So: skip, and say so once.
    age = open_positions.age_minutes()
    if age is not None and age > STALE_AFTER_MIN:
        with engine.connect() as conn:
            tc.ensure_tables(conn)
            prev = tc.last_state(conn)
        if prev != "stale":
            _send_telegram(
                f"⚠️ *Threat Board: feed dead*\n"
                f"MT5 book is {age:.0f} min old - the sync loop is not running, "
                f"so no threat check was made. See logs/mt5_sync.log.")
        return None

    if not positions:
        return None

    with engine.connect() as conn:
        tc.ensure_tables(conn)
        try:
            rep = tc.build_report(positions, equity, zone)
        except Exception as exc:
            print(f"[threat] evaluation failed: {exc}")
            return None

        rep.detail["unstopped"] = [str(r.get("pair")) for r in unstopped]
        prev = tc.last_state(conn)
        tc.journal(conn, rep)

    if prev != rep.state:
        _send_telegram(_format_alert(prev, rep))
    return rep


if __name__ == "__main__":
    # Standalone smoke test:  python threat_sentry_hook.py
    from sqlalchemy import create_engine
    url = os.getenv("DATABASE_URL", "postgresql://localhost/trading")
    # No EQUITY default: the live snapshot is the only honest source.
    rep = run_threat_check(create_engine(url))
    if rep:
        print(rep.state, rep.score, rep.components)
