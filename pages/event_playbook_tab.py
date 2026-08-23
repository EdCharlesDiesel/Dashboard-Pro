"""
Macro Event Playbook — decide the response before the number prints.

Entry point follows the house pattern: render(conn).
Service logic lives in src/services/event_playbook_service.py.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

try:
    from src.services.event_playbook_service import (
        SAST,
        SESSION_END,
        SESSION_START,
        SPEECH_LADDER,
        EventKind,
        SessionFit,
        alert_text,
        breakout_plan,
        ensure_schema,
        estimate_metal_usd_beta,
        log_plan,
        retrace_plan,
        scenario_ladder,
        score_surprise,
        session_fit,
        size_from_stop,
        vol_scaled_size,
        week_events,
    )
except ImportError:  # flat-layout fallback
    from event_playbook_service import (  # type: ignore
        SAST,
        SESSION_END,
        SESSION_START,
        SPEECH_LADDER,
        EventKind,
        SessionFit,
        alert_text,
        breakout_plan,
        ensure_schema,
        estimate_metal_usd_beta,
        log_plan,
        retrace_plan,
        scenario_ladder,
        score_surprise,
        session_fit,
        size_from_stop,
        vol_scaled_size,
        week_events,
    )


FIT_BADGE = {
    SessionFit.IN_WINDOW: ("🟢", "Inside your window"),
    SessionFit.PRE_WINDOW: ("🟡", "Fires before your window"),
    SessionFit.OUT_OF_SESSION: ("⚪", "Outside your session"),
}

INSTRUMENTS = {
    "XAUUSD": {"contract_value": 100.0, "ref_price": 4000.0, "daily_vol": 1.2},
    "XAGUSD": {"contract_value": 5000.0, "ref_price": 48.0, "daily_vol": 2.1},
}


def render(conn=None) -> None:
    st.title("Macro Event Playbook")
    st.caption(
        "Every response is written down before the release. The number decides "
        "which line you execute, not how you feel when the candle prints."
    )

    events = week_events()
    now_sast = datetime.now(SAST)

    # ---------------------------------------------------------------- sidebar
    with st.sidebar:
        st.subheader("Playbook settings")
        instrument = st.selectbox("Focus instrument", list(INSTRUMENTS), index=0)
        st.caption("One instrument per event. Split attention is how the plan gets abandoned.")

        default_beta, beta_note = estimate_metal_usd_beta(conn, metal_symbol=instrument)
        metal_beta = st.slider(
            "Metal-to-dollar beta",
            min_value=-2.0,
            max_value=2.0,
            value=float(round(default_beta, 2)),
            step=0.05,
            help="Negative means a stronger dollar pushes the metal down. "
                 "Since Feb 2026 this has been the prevailing regime.",
        )
        st.caption(beta_note)

        st.divider()
        equity = st.number_input("Account equity ($)", min_value=100.0, value=10_000.0, step=500.0)
        risk_pct = st.number_input("Risk per trade (%)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
        atr_m15 = st.number_input(
            f"M15 ATR(14) for {instrument}",
            min_value=0.0,
            value=6.0 if instrument == "XAUUSD" else 0.12,
            step=0.01,
            format="%.3f",
            help="Read this off the chart before the release. Everything downstream scales off it.",
        )

    # ------------------------------------------------------------- week board
    st.subheader("This week")
    rows = []
    for ev in events:
        icon, fit_label = FIT_BADGE[session_fit(ev)]
        rows.append(
            {
                "When (SAST)": ev.when_sast.strftime("%a %d %b %H:%M"),
                "Event": ev.name + ("  ⏱" if not ev.timing_confirmed else ""),
                "Ccy": ev.currency,
                "Forecast": "—" if ev.forecast is None else f"{ev.forecast}{ev.unit}",
                "Prev": "—" if ev.previous is None else f"{ev.previous}{ev.unit}",
                "Tier": ev.tier,
                "Session": f"{icon} {fit_label}",
            }
        )
    board = pd.DataFrame(rows)
    st.dataframe(board, use_container_width=True, hide_index=True)
    st.caption(
        f"Your window is {SESSION_START:%H:%M}–{SESSION_END:%H:%M} SAST. "
        "⏱ marks a time that is not yet officially published — confirm it on the day."
    )

    incomplete = [e.name for e in events if not e.is_complete]
    if incomplete:
        st.warning(
            "Missing a forecast, so no ladder can be built for: "
            + ", ".join(incomplete)
            + ". Fill these in week_events() from your calendar feed."
        )

    st.divider()

    # ---------------------------------------------------------- event chooser
    labels = {f"{e.when_sast:%a %H:%M} · {e.name}": e for e in events}
    picked = st.selectbox("Build the plan for", list(labels))
    ev = labels[picked]
    fit = session_fit(ev)
    icon, fit_label = FIT_BADGE[fit]

    left, right = st.columns([2, 1])
    with left:
        st.markdown(f"### {ev.name}")
        st.markdown(f"{icon} **{fit_label}** — {ev.when_sast:%A %d %B, %H:%M} SAST")
        if ev.note:
            st.info(ev.note)
    with right:
        if ev.forecast is not None:
            st.metric("Forecast", f"{ev.forecast}{ev.unit}",
                      delta=None if ev.previous is None else f"{ev.forecast - ev.previous:+.2f} vs prev")
        st.metric("Surprise σ", f"{ev.sigma:.3f}")

    # ------------------------------------------------------------- the ladder
    st.subheader("Scenario ladder")
    if ev.kind is EventKind.SPEECH:
        st.dataframe(pd.DataFrame(SPEECH_LADDER), use_container_width=True, hide_index=True)
        st.caption(
            "A speech has no forecast to standardise against, so the ladder is scored on "
            "language rather than a number. Decide which row you heard before you touch the platform."
        )
    elif ev.kind is EventKind.EARNINGS:
        st.info(
            "No directional ladder. Treat this as a position-management event: reduce or "
            "flatten exposure held across the print rather than trading it."
        )
    else:
        ladder = scenario_ladder(ev, metal_beta=metal_beta)
        if ladder:
            st.dataframe(pd.DataFrame(ladder), use_container_width=True, hide_index=True)
            st.caption(
                f"Rungs sit at ±1σ and ±2σ of the surprise distribution "
                f"(σ = {ev.sigma:.3f}). Metal column already has the "
                f"{metal_beta:+.2f} beta applied."
            )
        else:
            st.warning("Add a forecast for this event to generate the ladder.")

    st.divider()

    # ------------------------------------------------------- the actual print
    st.subheader("When the number lands")
    bias = None
    actual = None
    if ev.kind is EventKind.PRINT and ev.forecast is not None:
        c1, c2, c3 = st.columns(3)
        actual = c1.number_input(
            "Actual", value=float(ev.forecast), step=0.1, format="%.2f",
            help="Leave on forecast until the release. Then type what printed, not what you expected.",
        )
        bias = score_surprise(ev, actual, metal_beta=metal_beta)
        c2.metric("Surprise (z)", f"{bias.z:+.2f}", delta=f"{actual - ev.forecast:+.2f}{ev.unit}")
        c3.metric(f"{ev.currency}", bias.label)
        direction = bias.direction
        if direction == "flat":
            st.info(
                f"Conviction {abs(bias.metal_bias):.2f} is below the 0.25 floor. "
                "In-line prints do not pay — the plan is no trade."
            )
        else:
            st.success(f"{instrument} bias: **{direction.upper()}** "
                       f"(conviction {abs(bias.metal_bias):.2f})")
    else:
        direction = st.radio(
            "Read from the ladder", ["long", "short", "flat"],
            horizontal=True, index=2,
            help="Pick the row you actually heard, then build the plan off it.",
        )

    # -------------------------------------------------------------- the plan
    st.subheader("Trade construction")
    plan = None

    if fit is SessionFit.IN_WINDOW:
        st.markdown(
            "**Retracement plan.** The release fires while you are at the desk. "
            "Let the first impulse finish, then enter the pullback."
        )
        c1, c2 = st.columns(2)
        imp_high = c1.number_input("Impulse high", value=0.0, step=0.1, format="%.3f")
        imp_low = c2.number_input("Impulse low", value=0.0, step=0.1, format="%.3f")
        if imp_high > imp_low > 0 and atr_m15 > 0:
            plan = retrace_plan(direction, imp_high, imp_low, atr_m15)
        else:
            st.caption(
                "Mark the high and low of the first 30 minutes after the release, "
                "then the entry zone appears here."
            )

    elif fit is SessionFit.PRE_WINDOW:
        gap = (ev.when_sast.replace(hour=SESSION_START.hour, minute=SESSION_START.minute)
               - ev.when_sast).total_seconds() / 3600
        st.markdown(
            f"**Breakout plan.** The print lands {gap:.1f}h before you open. "
            "The spike is gone by the time you sit down, so the object is the "
            "consolidation that formed afterwards."
        )
        c1, c2 = st.columns(2)
        rng_high = c1.number_input("Consolidation high", value=0.0, step=0.1, format="%.3f")
        rng_low = c2.number_input("Consolidation low", value=0.0, step=0.1, format="%.3f")
        if rng_high > rng_low > 0 and atr_m15 > 0:
            plan = breakout_plan(direction, rng_high, rng_low, atr_m15)
        else:
            st.caption(
                "Mark the high and low of the block between the release and your open, "
                "then the break level appears here."
            )

    else:
        st.markdown(
            "**No session plan.** This release fires while you are away from the desk. "
            "It is a risk-management event, not a trade: decide now what exposure you "
            "are willing to hold through it."
        )
        st.caption(
            "If you are flat, you have nothing to do. If you are holding, either halve "
            "the position or move the stop outside the expected gap."
        )

    if plan is not None:
        for reason in plan.reasons:
            st.caption(f"· {reason}")
        if plan.valid:
            g1, g2, g3, g4 = st.columns(4)
            zone = (f"{plan.entry_low}" if plan.entry_low == plan.entry_high
                    else f"{plan.entry_low} – {plan.entry_high}")
            g1.metric("Entry", zone)
            g2.metric("Stop", f"{plan.stop}")
            g3.metric("Target 1", f"{plan.target_1}", delta=f"{plan.r_multiple_t1}R")
            g4.metric("Target 2", f"{plan.target_2}", delta=f"{plan.r_multiple_t2}R")

            anchor = (plan.entry_low + plan.entry_high) / 2
            spec = INSTRUMENTS[instrument]
            sizing = size_from_stop(equity, risk_pct, anchor, plan.stop, spec["contract_value"])
            vol_sizing = vol_scaled_size(
                equity, risk_pct, spec["ref_price"], spec["daily_vol"], horizon_days=0.25
            )
            s1, s2, s3 = st.columns(3)
            s1.metric("Risk", f"${sizing['risk_cash']:,.2f}")
            s2.metric("Stop distance", f"{sizing['stop_distance']}")
            s3.metric("Size (lots)", f"{sizing.get('lots', 0):.2f}")
            st.caption(
                f"Volatility-scaled cross-check at a 2σ√T intraday stop: "
                f"{vol_sizing['units']:.3f} units. If that is far below the stop-based "
                f"size, the market is wider than your plan assumes — take the smaller one."
            )
            if (plan.r_multiple_t1 or 0) < 1.0:
                st.warning(
                    f"Target 1 is only {plan.r_multiple_t1}R. Below 1R the trade needs a "
                    "win rate you have not demonstrated. Skip it."
                )
        else:
            st.error("No trade. " + (plan.reasons[-1] if plan.reasons else ""))

    st.divider()

    # ------------------------------------------------------------- journaling
    st.subheader("Thesis and invalidation")
    thesis = st.text_area(
        "Thesis",
        placeholder="Why this direction, in one sentence, referencing the print and the regime.",
    )
    invalidation = st.text_area(
        "Invalidation",
        placeholder="The condition that proves this wrong — a level, a close, or a time. "
                    "Not a feeling.",
    )

    can_log = bool(thesis.strip() and invalidation.strip() and plan is not None and plan.valid)
    if st.button("Save plan", disabled=not can_log, type="primary"):
        if conn is None:
            st.error("No database connection. The plan was not saved.")
        else:
            try:
                ensure_schema(conn)
                fp = log_plan(
                    conn, ev, instrument,
                    actual=actual,
                    bias=bias, plan=plan, metal_beta=metal_beta,
                    thesis=thesis.strip(), invalidation=invalidation.strip(),
                )
                st.success(f"Saved as {fp}. Re-saving the same event updates the row.")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Save failed: {type(exc).__name__}: {exc}")
    if not can_log:
        st.caption(
            "Saving needs a valid plan plus a written thesis and invalidation. "
            "A trade you cannot describe is a trade you cannot review."
        )

    if plan is not None and plan.valid and bias is not None:
        with st.expander("Sentry alert text"):
            st.code(alert_text(ev, bias, plan, instrument), language="text")

    st.caption(f"Rendered {now_sast:%Y-%m-%d %H:%M} SAST")


def _connect():
    """Reuse the app's connection helper; fall back to a read-only page."""
    try:
        from src.db import get_conn  # type: ignore
        return get_conn()
    except Exception:
        return None


# Streamlit executes pages as scripts, so render on import.
render(_connect())