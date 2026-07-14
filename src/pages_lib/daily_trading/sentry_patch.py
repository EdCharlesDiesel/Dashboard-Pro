"""
sentry_patch.py — wiring for evening_sentry.py (v2)
====================================================
Add these three calls to your existing poll loop. Nothing else changes.
"""

from surprise_tab import (
    compute_surprises,
    event_gate,
    sentry_surprise_alerts,
    shock_check,
)

# Inside your existing 17:00-20:00 SAST loop, per cycle:


def sentry_cycle(conn, send_telegram, garch_sigma_xau, xau_returns_15m,
                 last_gate_state: dict) -> dict:
    """Returns updated gate state (so you only alert on STATE CHANGES,
    matching your existing sentry pattern)."""

    # 1. Refresh surprises + alert on |z| >= 1.5 (last 90 min)
    compute_surprises(conn)
    sentry_surprise_alerts(conn, send_telegram)

    # 2. Event gate — alert only on transitions, like your other monitors
    gate = event_gate(conn, "XAUUSD", hours_before=4.0, hours_after=2.0)
    if gate["blocked"] != last_gate_state.get("blocked"):
        if gate["blocked"]:
            send_telegram(f"GATE ON — XAUUSD signals suppressed. {gate['reason']}")
        else:
            send_telegram("GATE OFF — XAUUSD event window clear.")

    # 3. Shock check against your GARCH module's conditional sigma
    shock = shock_check(xau_returns_15m, garch_sigma_xau, k=2.5)
    if shock["shock"] and not last_gate_state.get("shock"):
        send_telegram(
            f"SHOCK — XAUUSD 15m move {shock['return']:+.3%} is "
            f"{shock['ratio']:.1f}x GARCH sigma. Stand aside / manage risk."
        )

    return {"blocked": gate["blocked"], "shock": shock["shock"]}


# Confluence engine: one extra condition in your regime gating —
#
#   gate = event_gate(conn, instrument)
#   if gate["blocked"]:
#       confluence_score = 0          # or downweight, your call
#       journal(reason=gate["reason"])