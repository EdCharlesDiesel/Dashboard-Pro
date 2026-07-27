"""
week_ahead_tab.py
=================
The Sunday pre-flight: one screen that answers "what might the coming week
look like?" for every instrument you care about.

The honest framing baked into this page: **direction is not forecastable** —
the house-view walk-forward (src/services/house_view_backtest.py) showed the
composite has no standalone entry edge. What *is* forecastable is
**volatility** (it clusters and persists), and what's *knowable* is the event
calendar and positioning. So this page gives you a probabilistic range, the
week's landmines, and crowding — then makes you write a thesis + invalidation
before Monday.

Composed entirely from existing services (no new maths in the page):
  • expected range  — src/services/week_ahead.cone_from_returns (GARCH(1,1)
    term structure, EWMA fallback)
  • house view      — src/services/bias_service.get_house_view (reads the
    background worker's precomputed board, so the whole board is one DB hop)
  • COT crowding + supply/demand — src/services/swing_playbook_service
  • event calendar  — src/services/econ_calendar (importable twin of
    news-filter's fetch, which can't be imported)
  • thesis storage  — swing_playbook_service.save_thesis (the same
    ``swing_theses`` table the Swing Playbook writes, so a thesis written here
    shows up there and vice versa)

Audit-only: a range forecast is not a directional pair+bias call, so this page
logs to ``tool_usage_log`` and never writes ``trade_setups``.

Entry point: render_week_ahead_tab()
"""
from __future__ import annotations

import streamlit as st

from src.ui.theme import BloombergTheme

st.set_page_config(
    page_title="WKAH · Week Ahead",
    page_icon="🗓️",
    layout="wide",
    initial_sidebar_state="expanded",
)
BloombergTheme.apply()

from datetime import date, timedelta

import pandas as pd

from src.instruments.registry import INSTRUMENTS
from src.pages_lib.navigation import render_sidebar_nav
from src.services import econ_calendar as cal
from src.services import swing_playbook_service as sp
from src.services import week_ahead as wa
from src.services.bias_service import get_house_view
from src.services.market_data import daily_ohlc
from src.services.parallel_fetch import run_parallel
from src.ui.charts import ChartKit
from src.ui.components import MetricCell, render_metric_row
from src.ui.theme import BloombergTheme as T

DEFAULT_INSTRUMENTS = ["EUR/USD", "GBP/USD", "USD/JPY", "XAU/USD", "WTI/USD"]
_CHART_HISTORY = 60          # trading days of context behind the cone


@st.cache_data(ttl=900, show_spinner=False)
def _calendar(week_key: str):
    """Normalised events for the chosen week (15-min cache — the feed is weekly)."""
    return cal.normalise(cal.fetch_week(week_key))


def _fmt(price, pair: str) -> str:
    """Project price convention: 5dp under 100 (FX), else 2-3dp."""
    if price is None:
        return "—"
    return f"{price:.5f}" if abs(price) < 100 else f"{price:.2f}"


def _week_label(week_key: str) -> str:
    today = date.today()
    monday = today - timedelta(days=today.weekday())
    if week_key == "nextweek":
        monday = monday + timedelta(days=7)
    return f"{monday:%d %b} – {monday + timedelta(days=4):%d %b %Y}"


# ---------------------------------------------------------------------------
# Per-instrument assembly (pure-ish: all I/O via existing services)
# ---------------------------------------------------------------------------
def _build_row(pair: str, events: list) -> dict:
    inst = INSTRUMENTS.get(pair)
    ticker = inst.ticker if inst else None
    row: dict = {"pair": pair}

    df = daily_ohlc(ticker) if ticker else None
    if df is None or df.empty or "Close" not in df.columns:
        return row

    close = df["Close"].dropna()
    spot = float(close.iloc[-1])
    row["spot"] = spot
    row["cone"] = wa.cone_from_returns(close.pct_change().dropna(), spot)
    row["close"] = close

    view = get_house_view(pair)
    if view is not None:
        row["house"] = view.direction
        row["house_score"] = view.score
        row["aligned"] = view.aligned

    cot = sp.cot_read(pair)
    row["cot_pctile"] = cot["percentile"] if cot else None

    zones = sp.nearest_zones(pair, spot)
    if zones:
        row["supply"], row["demand"] = zones["supply"], zones["demand"]

    ccys = wa.currencies_for(pair)
    row["events"] = cal.filter_events(events, ccys, ("High",))
    row["flags"] = wa.risk_flags(row.get("cone"), row.get("cot_pctile"),
                                 len(row["events"]))
    return row


# ---------------------------------------------------------------------------
# Cone chart
# ---------------------------------------------------------------------------
def _cone_chart(pair: str, close: pd.Series, cone: dict):
    """History + the forward probability cone, in the house chart stencil."""
    hist = close.tail(_CHART_HISTORY)
    last_dt = hist.index[-1]
    future = pd.bdate_range(last_dt + pd.Timedelta(days=1), periods=cone["horizon"])

    # Anchor each band at spot so the cone opens from the last close rather
    # than jumping on day 1.
    x = [last_dt] + list(future)
    hi95 = [cone["spot"]] + cone["hi95_path"]
    lo95 = [cone["spot"]] + cone["lo95_path"]
    hi68 = [cone["spot"]] + cone["hi68_path"]
    lo68 = [cone["spot"]] + cone["lo68_path"]

    fig = ChartKit.panels([f"{pair} — last {_CHART_HISTORY}d and the week ahead"])
    ChartKit.line(fig, hist.index, hist.values, "Close", color=T.WHITE, width=1.8)
    ChartKit.band(fig, x, hi95, lo95, "rgba(0,255,65,0.08)", name="95%")
    ChartKit.band(fig, x, hi68, lo68, "rgba(0,255,65,0.18)", name="68%")
    ChartKit.line(fig, x, [cone["spot"]] * len(x), "spot", color=T.GREY,
                  width=1, dash="dot")
    return ChartKit.finish(fig, height=420)


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------
def render_week_ahead_tab() -> None:
    st.markdown("## 🗓️ Week Ahead — Sunday pre-flight")
    st.caption(
        "Direction is **not** forecastable (the house-view backtest proved it "
        "has no standalone entry edge). Volatility is. This page gives you the "
        "probable range, the week's scheduled landmines, and who's crowded — "
        "then asks you to commit a thesis + invalidation while you're calm. "
        "Research context, not a trade signal: nothing here writes trade_setups."
    )

    week_key = st.session_state.get("wa_week", "nextweek")
    pairs = st.session_state.get("wa_pairs", DEFAULT_INSTRUMENTS)
    if not pairs:
        st.warning("⚠ Select at least one instrument in the sidebar.")
        return

    events = _calendar(week_key)
    high = cal.filter_events(events, None, ("High",))

    render_metric_row([
        MetricCell("Week of", _week_label(week_key), "amber"),
        MetricCell("High-impact events", str(len(high)),
                   "red" if len(high) >= 8 else "yellow" if len(high) >= 4 else "green"),
        MetricCell("Instruments", str(len(pairs)), "cyan"),
        MetricCell("Calendar", "live" if events else "unavailable",
                   "green" if events else "red"),
    ])
    if not events:
        st.info("Economic calendar feed unavailable — the range and positioning "
                "columns below are still valid; event counts show 0.")

    # ── the board ────────────────────────────────────────────────────────────
    rows: dict = {}
    prog = st.progress(0.0, text="Building the week-ahead board…")

    def _done(pair, res):
        rows[pair] = res
        prog.progress(len(rows) / len(pairs),
                      text=f"{pair} ({len(rows)}/{len(pairs)})")

    run_parallel(lambda p: _build_row(p, events), pairs,
                 on_result=_done, on_error=lambda p, e: rows.setdefault(p, {"pair": p}))
    prog.empty()

    table = []
    for pair in pairs:                       # keep the user's ordering
        r = rows.get(pair) or {"pair": pair}
        cone = r.get("cone")
        table.append({
            "Instrument": pair,
            "Price": _fmt(r.get("spot"), pair),
            "House view": r.get("house", "N/A"),
            "Expected range (68%)": (
                f"{_fmt(cone['lo68'], pair)} – {_fmt(cone['hi68'], pair)}"
                if cone else "N/A"),
            "±%": f"{cone['range_pct']:.2f}%" if cone else "—",
            "Size": wa.classify_range(cone["range_pct"]) if cone else "—",
            "Vol regime": (cone.get("regime") or "—") if cone else "—",
            "COT %ile": (f"{r['cot_pctile']:.0f}"
                         if r.get("cot_pctile") is not None else "—"),
            "Supply": _fmt(r.get("supply"), pair),
            "Demand": _fmt(r.get("demand"), pair),
            "Events": len(r.get("events", [])),
        })
    st.dataframe(pd.DataFrame(table), width="stretch", hide_index=True)
    st.caption(
        "**Expected range** = ±1σ over 5 trading days from a GARCH(1,1) "
        "forecast — roughly 2 weeks in 3 finish inside it, so a move beyond it "
        "is genuinely unusual (fade territory), and a move inside it is noise "
        "(not a breakout). **Events** counts high-impact releases hitting that "
        "instrument's currencies. **COT %ile** >90 or <10 = crowded, the "
        "setup for a violent unwind."
    )

    # ── risk flags ───────────────────────────────────────────────────────────
    flagged = [(p, rows[p]["flags"]) for p in pairs
               if rows.get(p, {}).get("flags")]
    if flagged:
        st.markdown("#### ⚠ What could break your week")
        for pair, flags in flagged:
            st.markdown(f"**{pair}** — " + " · ".join(flags))

    # ── cone chart ───────────────────────────────────────────────────────────
    st.divider()
    chartable = [p for p in pairs if rows.get(p, {}).get("cone") is not None]
    if chartable:
        focus = st.selectbox("Cone detail", chartable, key="wa_focus")
        r = rows[focus]
        st.plotly_chart(_cone_chart(focus, r["close"], r["cone"]),
                        width="stretch", config=ChartKit.PLOTLY_CONFIG)
        c = r["cone"]
        st.caption(
            f"Cone built from a **{c['method'].upper()}** volatility forecast. "
            f"Inner band = 68% of outcomes, outer = 95%. The cone is centred on "
            f"spot on purpose — over one week, drift is negligible next to "
            f"noise, so a random walk is the honest centre. Bands widen with "
            f"√time, not linearly."
        )

    # ── the week's calendar ──────────────────────────────────────────────────
    if high:
        st.divider()
        st.markdown("#### 📰 High-impact events this week")
        for day, evs in cal.group_by_day(high).items():
            with st.expander(f"{day:%A %d %b} — {len(evs)} event(s)",
                             expanded=(day == date.today())):
                st.dataframe(pd.DataFrame([{
                    "Time (ET)": e["time"], "Ccy": e["currency"],
                    "Event": e["title"], "Forecast": e["forecast"],
                    "Previous": e["previous"],
                } for e in evs]), width="stretch", hide_index=True)

    # ── thesis ───────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("#### ✍ This week's thesis")
    st.caption(
        "The forecast isn't the deliverable — this is. Write the bias and, "
        "more importantly, **what would prove it wrong**, before Monday. "
        "Saved to the same `swing_theses` table the Swing Playbook reads, so "
        "one thesis serves both pages."
    )
    t_pair = st.selectbox("Instrument", pairs, key="wa_thesis_pair")
    existing = sp.load_thesis(t_pair) or {}
    with st.form("wa_thesis"):
        c1, c2 = st.columns([1, 3])
        with c1:
            bias_opts = ["Long", "Short", "Neutral"]
            prev = (existing.get("bias") or "Neutral").capitalize()
            bias = st.selectbox("Bias", bias_opts,
                                index=bias_opts.index(prev) if prev in bias_opts else 2)
        with c2:
            inval = st.text_input(
                "Invalidation — what kills this thesis?",
                value=existing.get("invalidation", ""),
                placeholder="e.g. daily close below 4180, or CPI above 3.4%",
            )
        if st.form_submit_button("💾 Save thesis", type="primary"):
            if not inval.strip():
                st.error("An invalidation is required — a thesis you can't be "
                         "wrong about isn't a thesis.")
            elif sp.save_thesis(t_pair, bias, inval.strip()):
                st.success(f"Saved {t_pair}: {bias} — invalidated by {inval.strip()}")
            else:
                st.warning("Database not configured — thesis not saved.")
    if existing:
        st.caption(f"Current: **{existing.get('bias', '—')}** · invalidation: "
                   f"{existing.get('invalidation', '—')}")

    # ── audit log (research read, not a trade signal) ────────────────────────
    try:
        from src.services.alert_service import NotifyCache
        from src.services.tool_log import log_tool_usage
        key = f"{_week_label(week_key)}|{','.join(sorted(pairs))}"
        if NotifyCache("week_ahead_log").filter_new([key]):
            log_tool_usage("week_ahead", {
                "week": _week_label(week_key),
                "high_impact_events": len(high),
                "instruments": sorted(pairs),
                "ranges": {p: round(rows[p]["cone"]["range_pct"], 2)
                           for p in pairs if rows.get(p, {}).get("cone")},
            })
    except Exception:
        pass


# ===========================================================================
# PAGE ENTRY
# ===========================================================================
with st.sidebar:
    st.markdown("### 🗓️ Week Ahead")
    st.session_state.wa_week = st.radio(
        "Which week", ["nextweek", "thisweek"],
        format_func=lambda k: "Next week" if k == "nextweek" else "This week",
        help="Run this on a Sunday with 'Next week' selected.",
    )
    st.session_state.wa_pairs = st.multiselect(
        "Instruments", sorted(INSTRUMENTS.keys()),
        default=[p for p in DEFAULT_INSTRUMENTS if p in INSTRUMENTS],
    )
    st.divider()
    render_sidebar_nav()

render_week_ahead_tab()
