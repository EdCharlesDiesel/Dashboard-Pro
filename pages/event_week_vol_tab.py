"""
event_week_vol_tab.py
=====================
Empirical test: is there a "busy week of the month" pattern in FX volatility,
and does anchoring to events (NFP / CPI / FOMC / BoJ / SARB) explain it
better than the raw week number?

Uses the canonical OHLC spine (src/services/market_data.py) for daily bars.

Two competing groupings are computed for every instrument:
  1. NAIVE:  week-of-month (1..5) -> mean daily ATR%
  2. EVENT-ANCHORED: days-to-nearest-event (-5..+5) -> mean daily ATR%

If the event-anchored profile shows a sharper, cleaner peak than the
week-of-month profile, the "busy week" is an artifact of the event calendar,
not the calendar week itself. That is the hypothesis being tested.

Entry point: render_event_week_vol_tab()
Standalone:  streamlit run pages/event_week_vol_tab.py (also auto-registers as
a multipage page — see src/pages_lib/navigation.py)
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.core.event_calendar import EVENT_RELEVANCE, build_event_calendar
from src.instruments import INSTRUMENTS as _REGISTRY_INSTRUMENTS
from src.services.alert_service import NotifyCache
from src.services.market_data import daily_ohlc
from src.services.tool_log import log_tool_usage

# ---------------------------------------------------------------------------
# Instruments — sourced from the shared registry (src/instruments/registry.py)
# so this page covers the same universe as every other page.
# ---------------------------------------------------------------------------
INSTRUMENTS: dict[str, str] = {
    name: info.ticker for name, info in _REGISTRY_INSTRUMENTS.items()
}

# ---------------------------------------------------------------------------
# Event calendars live in src/core/event_calendar.py -- one list, shared with
# the Event Reaction Map (pages/nfp_reaction.py). Add or correct a release
# date there, never here.
# ---------------------------------------------------------------------------


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def fetch_daily(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Daily OHLC over [start, end), through the canonical spine.

    The spine takes a `period`, so the span back to `start` is converted to
    days and both edges are then trimmed here. The right edge is **exclusive**
    (`< finish`), matching `yf.download`'s `end` semantics — closing that
    interval instead would quietly add a bar to every window, and this page
    slices windows around event dates where one extra bar is a wrong answer.

    The index is forced tz-naive conditionally: `tz_localize(None)` raises on
    an already-naive index, and the spine returns either depending on whether
    the bars came from Postgres or straight from Yahoo.
    """
    begin, finish = pd.Timestamp(start), pd.Timestamp(end)
    # +7d of slack: `period="Nd"` measures back from today and can land just
    # inside `begin`, dropping a bar sitting exactly on it (measured: the
    # 2024-01-01 bar vanished at an exact 957d span). Over-fetch a little and
    # let the trim below define the left edge, not the fetch window.
    days = max((pd.Timestamp.today().normalize() - begin).days, 1) + 7
    df = daily_ohlc(ticker, period=f"{days}d")
    cols = ["Open", "High", "Low", "Close"]
    if df.empty or not all(c in df.columns for c in cols):
        return pd.DataFrame()
    df = df[cols].dropna()
    idx = pd.to_datetime(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx
    return df.loc[(df.index >= begin) & (df.index < finish)]


def add_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Daily ATR% (true range / previous close) plus calendar features."""
    out = df.copy()
    prev_close = out["Close"].shift(1)
    tr = pd.concat([
        out["High"] - out["Low"],
        (out["High"] - prev_close).abs(),
        (out["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    out["atr_pct"] = 100.0 * tr / prev_close
    out["week_of_month"] = ((out.index.day - 1) // 7) + 1  # 1..5
    out["month"] = out.index.month
    return out.dropna(subset=["atr_pct"])


def days_to_nearest_event(index: pd.DatetimeIndex,
                          event_dates: pd.Series) -> np.ndarray:
    """Signed trading-agnostic calendar distance to the nearest event.
    Negative = event is ahead (approaching), positive = event has passed."""
    ev = np.sort(event_dates.values.astype("datetime64[D]"))
    days = index.values.astype("datetime64[D]")
    pos = np.searchsorted(ev, days)
    dist = np.full(len(days), np.inf)
    signed = np.zeros(len(days))
    for i, (d, p) in enumerate(zip(days, pos)):
        candidates = []
        if p < len(ev):
            candidates.append((ev[p] - d).astype(int))      # >= 0, upcoming
        if p > 0:
            candidates.append((ev[p - 1] - d).astype(int))  # <= 0, passed
        best = min(candidates, key=abs)
        # best = (event - today): positive when the event is upcoming.
        # Convention: event day = 0, days BEFORE are negative -> flip sign.
        signed[i] = -best
        dist[i] = abs(best)
    return signed


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------
def render_event_week_vol_tab() -> None:
    st.header("📅 Busy-Week Anatomy: week-of-month vs event-anchored volatility")
    st.caption(
        "Hypothesis: the 'busy week' in FX is the event calendar in disguise. "
        "If days-to-event shows a sharper volatility peak than week-of-month, "
        "trade the calendar, not the week number."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        years = st.slider("Lookback (years)", 1, 10, 4)
    with col2:
        window = st.slider("Event window (± days)", 3, 10, 5)
    with col3:
        chosen = st.multiselect("Instruments", sorted(INSTRUMENTS),
                                default=["USD/JPY", "USD/ZAR", "EUR/USD"])

    uploaded = st.file_uploader(
        "Optional: extend the event calendar (CSV with columns date,event)",
        type="csv")
    extra = pd.read_csv(uploaded) if uploaded else None

    if not chosen:
        st.info("Pick at least one instrument.")
        return

    end = dt.date.today()
    start = end - dt.timedelta(days=int(years * 365.25))
    cal_df = build_event_calendar(start, end, extra)

    naive_rows, event_rows, month_rows = [], [], []

    for name in chosen:
        raw = fetch_daily(INSTRUMENTS[name], str(start), str(end))
        if raw.empty:
            st.warning(f"No data for {name} — skipped.")
            continue
        df = add_metrics(raw)

        # 1) naive week-of-month
        wom = df.groupby("week_of_month")["atr_pct"].mean()
        for w, v in wom.items():
            naive_rows.append({"instrument": name, "week": int(w), "atr_pct": v})

        # 2) event-anchored, per relevant event type
        for ev_name in EVENT_RELEVANCE.get(name, ["NFP", "FOMC"]):
            ev_dates = cal_df.loc[cal_df["event"] == ev_name, "date"]
            if ev_dates.empty:
                continue
            d2e = days_to_nearest_event(df.index, ev_dates)
            sub = df.assign(d2e=d2e)
            sub = sub[sub["d2e"].abs() <= window]
            prof = sub.groupby("d2e")["atr_pct"].mean()
            for d, v in prof.items():
                event_rows.append({"instrument": name, "event": ev_name,
                                   "days_to_event": int(d), "atr_pct": v})

        # 3) seasonal sanity check: month-of-year
        mom = df.groupby("month")["atr_pct"].mean()
        for m, v in mom.items():
            month_rows.append({"instrument": name, "month": int(m), "atr_pct": v})

    if not naive_rows:
        st.error("No data fetched.")
        return

    naive = pd.DataFrame(naive_rows)
    events = pd.DataFrame(event_rows)
    months = pd.DataFrame(month_rows)

    # ---- Chart 1: week-of-month heatmap -----------------------------------
    st.subheader("1 · Naive view: mean daily ATR% by week of month")
    pivot = naive.pivot(index="instrument", columns="week", values="atr_pct")
    fig = px.imshow(pivot, text_auto=".2f", aspect="auto",
                    color_continuous_scale="RdYlGn_r",
                    labels=dict(color="ATR %"))
    st.plotly_chart(fig, use_container_width=True)

    # ---- Chart 2: event-anchored profiles ----------------------------------
    st.subheader("2 · Event-anchored view: ATR% vs days to event")
    st.caption("Day 0 = release/decision day. Negative = days before.")
    if not events.empty:
        for name in events["instrument"].unique():
            sub = events[events["instrument"] == name]
            fig2 = go.Figure()
            for ev_name, grp in sub.groupby("event"):
                grp = grp.sort_values("days_to_event")
                fig2.add_trace(go.Scatter(
                    x=grp["days_to_event"], y=grp["atr_pct"],
                    mode="lines+markers", name=ev_name))
            fig2.add_vline(x=0, line_dash="dash", line_color="gray")
            fig2.update_layout(title=name, height=320,
                               xaxis_title="days to event",
                               yaxis_title="mean ATR %",
                               margin=dict(t=40, b=30))
            st.plotly_chart(fig2, use_container_width=True)

    # ---- Chart 3: month-of-year seasonality --------------------------------
    st.subheader("3 · Seasonality check: ATR% by calendar month")
    pivot_m = months.pivot(index="instrument", columns="month", values="atr_pct")
    fig3 = px.imshow(pivot_m, text_auto=".2f", aspect="auto",
                     color_continuous_scale="RdYlGn_r",
                     labels=dict(color="ATR %"))
    st.plotly_chart(fig3, use_container_width=True)

    # ---- Verdict ------------------------------------------------------------
    st.subheader("4 · Verdict: which grouping explains more?")
    verdict_rows = []
    for name in naive["instrument"].unique():
        n = naive[naive["instrument"] == name]["atr_pct"]
        naive_spread = (n.max() - n.min()) / n.mean() * 100
        e = events[events["instrument"] == name]["atr_pct"] \
            if not events.empty else pd.Series(dtype=float)
        ev_spread = ((e.max() - e.min()) / e.mean() * 100) if len(e) else np.nan
        verdict_rows.append({
            "instrument": name,
            "week-of-month spread %": round(naive_spread, 1),
            "event-anchored spread %": round(ev_spread, 1),
            "calendar beats week#": bool(ev_spread > naive_spread)
            if not np.isnan(ev_spread) else None,
        })
    st.dataframe(pd.DataFrame(verdict_rows), use_container_width=True)
    st.caption(
        "Spread = (max − min) / mean of the grouped ATR% profile. A larger "
        "spread means that grouping separates quiet from busy days better."
    )

    # Log this research run to Postgres (audit trail, not trade_setups — this
    # is a seasonality/calendar-pattern study, not a directional pair+bias
    # signal). Deduped per instrument-set+settings via NotifyCache.
    _ewv_key = f"{','.join(sorted(chosen))}|{years}|{window}"
    if NotifyCache("event_week_vol_log").filter_new([_ewv_key]):
        log_tool_usage("event_week_vol", {
            "instruments": chosen, "years": years, "window": window,
            "verdict": verdict_rows,
        })


# ===========================================================================
# PAGE ENTRY — multipage (auto-run) + standalone (streamlit run)
# ===========================================================================

def _page() -> None:
    """Wire the tab into the Bloomberg-terminal multipage shell: shared theme,
    the uniform sidebar logo + navigation, then the tab body."""
    from src.ui.theme import BloombergTheme
    from src.pages_lib.navigation import render_sidebar_nav

    st.set_page_config(
        page_title="EWKV · Busy-Week Anatomy",
        page_icon="📅",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    BloombergTheme.apply()
    with st.sidebar:
        st.markdown("### 📅 Busy-Week Anatomy")
        st.caption("Week-of-month vs event-anchored volatility")
        st.divider()
        render_sidebar_nav()
    render_event_week_vol_tab()


# Streamlit executes a page module top-to-bottom on every run (the same way the
# other legacy pages self-initialise), so call the entry unconditionally.
_page()