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
    top = max(rep.components, key=rep.components.get)
    lines.append(f"Top driver: {top} ({rep.components[top]:.0f})")
    if d["headline_hits"]:
        lines.append("⚠️ Verbal intervention: " + d["headline_hits"][0][:120])
    if d["red_events"]:
        lines.append(f"📅 {len(d['red_events'])} red event(s) on your pairs in 7d")
    return "\n".join(lines)


def run_threat_check(engine, equity: float,
                     zone=(tc.JPY_ZONE_LOW, tc.JPY_ZONE_HIGH)) -> tc.ThreatReport | None:
    """One evaluation cycle. Alerts only on state transition. Always journals.
    Returns the report (or None if no positions / data failure)."""
    with engine.connect() as conn:
        tc.ensure_tables(conn)
        positions = tc.load_positions(conn)
        if not positions:
            return None
        try:
            rep = tc.build_report(positions, equity, zone)
        except Exception as exc:
            print(f"[threat] evaluation failed: {exc}")
            return None

        prev = tc.last_state(conn)
        tc.journal(conn, rep)

    if prev != rep.state:
        _send_telegram(_format_alert(prev, rep))
    return rep


if __name__ == "__main__":
    # Standalone smoke test:  python threat_sentry_hook.py
    from sqlalchemy import create_engine
    url = os.getenv("DATABASE_URL", "postgresql://localhost/trading")
    rep = run_threat_check(create_engine(url), equity=float(os.getenv("EQUITY", "935")))
    if rep:
        print(rep.state, rep.score, rep.components)