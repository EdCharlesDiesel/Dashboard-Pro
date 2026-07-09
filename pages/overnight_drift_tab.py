"""
overnight_drift_tab.py
======================
Maps the NY Fed Staff Report 917 ("The Overnight Drift", Boyarchenko,
Larsen & Whelan) onto live ES futures data so the findings can be
empirically re-tested rather than taken on faith.

Paper's core claims, and where each is tested in this module:

  1. Returns concentrate in the 02:00-03:00 ET hour (European open)
        -> Section 1: hour-of-day return decomposition
  2. The drift is a *reversal* of prior US-close selling pressure
        -> Section 2: condition the drift hour on the prior US session's
           final-hour return (proxy for closing order imbalance)
  3. The effect scales with volatility (dealer VaR constraints)
        -> Section 3: interaction with VIX terciles
  4. Unconditional drift is not tradable after costs; the conditional
     "buy the dip" version may be
        -> Section 4: pre-cost strategy equity curves + hit rates

Data: yfinance ES=F 1h bars (~730 days max), ^VIX daily.
Timezone: all bars converted to America/New_York, matching the paper.

Entry point: render_overnight_drift_tab()
Standalone:  streamlit run pages/overnight_drift_tab.py (also auto-registers as
a multipage page — see src/pages_lib/navigation.py)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from src.services.alert_service import NotifyCache
from src.services.tool_log import log_tool_usage

ET = "America/New_York"
DRIFT_HOUR = 2          # 02:00-03:00 ET bar (bar start time)
US_LAST_HOUR = 15       # 15:00-16:00 ET, closing-pressure proxy
TRADING_DAYS = 252


# ----------------------------------------------------------------------
# Data layer
# ----------------------------------------------------------------------

@st.cache_data(ttl=6 * 3600, show_spinner="Fetching ES=F hourly bars...")
def fetch_es_hourly(period: str = "730d") -> pd.DataFrame:
    """ES futures 1h bars, ET-indexed, with per-bar open->close returns."""
    df = yf.download("ES=F", period=period, interval="1h",
                     auto_adjust=False, progress=False)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.tz_convert(ET) if df.index.tz is not None else df.tz_localize("UTC").tz_convert(ET)
    df = df[["Open", "Close"]].dropna()
    # Per-bar return isolates the hour itself (no gap contamination).
    df["ret"] = df["Close"] / df["Open"] - 1.0
    df["hour"] = df.index.hour
    df["date"] = df.index.date
    return df


@st.cache_data(ttl=6 * 3600, show_spinner="Fetching VIX...")
def fetch_vix(period: str = "730d") -> pd.Series:
    vix = yf.download("^VIX", period=period, interval="1d",
                      auto_adjust=False, progress=False)
    if vix.empty:
        return pd.Series(dtype=float)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    s = vix["Close"].copy()
    s.index = pd.to_datetime(s.index).date
    s.name = "vix"
    return s


def _tstat(x: pd.Series) -> float:
    x = x.dropna()
    if len(x) < 3 or x.std(ddof=1) == 0:
        return np.nan
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


# ----------------------------------------------------------------------
# Section builders
# ----------------------------------------------------------------------

def hour_of_day_table(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("hour")["ret"]
    out = pd.DataFrame({
        "mean_bps": g.mean() * 1e4,
        "ann_pct": g.mean() * TRADING_DAYS * 100,
        "t_stat": g.apply(_tstat),
        "hit_rate": g.apply(lambda s: (s > 0).mean()),
        "n": g.size(),
    })
    return out


def build_daily_panel(df: pd.DataFrame, vix: pd.Series) -> pd.DataFrame:
    """One row per date: drift-hour return + previous US last-hour return + VIX."""
    drift = df[df["hour"] == DRIFT_HOUR].set_index("date")["ret"].rename("drift_ret")
    close_pressure = df[df["hour"] == US_LAST_HOUR].set_index("date")["ret"].rename("close_ret")
    # The 02:00 bar on date D follows the US close of the *previous* trading day.
    close_shifted = close_pressure.copy()
    close_shifted.index = pd.Index(close_shifted.index, name="date")
    panel = pd.DataFrame(drift)
    panel["prev_close_ret"] = close_shifted.reindex(panel.index, method=None)
    # align: for each drift date, take the most recent prior close_ret
    cp = close_pressure.sort_index()
    prev_vals = []
    cp_dates = np.array(cp.index)
    cp_vals = cp.values
    for d in panel.index:
        mask = cp_dates < d
        prev_vals.append(cp_vals[mask][-1] if mask.any() else np.nan)
    panel["prev_close_ret"] = prev_vals
    if not vix.empty:
        vix_sorted = vix.sort_index()
        vd = np.array(vix_sorted.index)
        vv = vix_sorted.values
        vprev = []
        for d in panel.index:
            mask = vd < d
            vprev.append(float(vv[mask][-1]) if mask.any() else np.nan)
        panel["vix_prev"] = vprev
    return panel.dropna(subset=["drift_ret", "prev_close_ret"])


# ----------------------------------------------------------------------
# Render
# ----------------------------------------------------------------------

def render_overnight_drift_tab() -> None:
    st.header("Overnight Drift — NY Fed SR 917, re-tested on live ES data")
    st.caption(
        "Paper: 1998-2020, e-mini tick data. Here: last ~730 days of 1h bars, "
        "so treat this as an out-of-sample sanity check, not a replication."
    )

    period = st.selectbox("Lookback", ["180d", "365d", "730d"], index=2)
    df = fetch_es_hourly(period)
    if df.empty:
        st.error("No ES=F data returned. Check connectivity / yfinance.")
        return
    vix = fetch_vix(period)

    # ---------------- Section 1: hour-of-day decomposition -------------
    st.subheader("1 · Where do returns accrue around the clock?")
    hod = hour_of_day_table(df)
    colors = ["#2ca02c" if h == DRIFT_HOUR else "#4c78a8" for h in hod.index]
    fig = go.Figure(go.Bar(x=hod.index, y=hod["ann_pct"], marker_color=colors))
    fig.update_layout(
        xaxis_title="Bar start hour (ET)", yaxis_title="Annualized mean return (%)",
        height=380, margin=dict(t=20, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    drift_row = hod.loc[DRIFT_HOUR] if DRIFT_HOUR in hod.index else None
    if drift_row is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("02:00-03:00 ET annualized", f"{drift_row['ann_pct']:.1f}%")
        c2.metric("t-stat", f"{drift_row['t_stat']:.2f}")
        c3.metric("Hit rate", f"{drift_row['hit_rate']:.0%}")
        st.caption(
            "Paper found ~3.7% annualized for this hour over 23 years. "
            "A short sample will be noisy — the t-stat matters more than the level."
        )

    with st.expander("Full hour-of-day table"):
        st.dataframe(hod.style.format({
            "mean_bps": "{:.2f}", "ann_pct": "{:.2f}",
            "t_stat": "{:.2f}", "hit_rate": "{:.0%}",
        }))

    # ---------------- Section 2: reversal conditioning -----------------
    st.subheader("2 · Is the drift a reversal of US closing pressure?")
    panel = build_daily_panel(df, vix)
    if panel.empty:
        st.warning("Not enough overlapping drift-hour / close-hour observations.")
        return

    neg = panel[panel["prev_close_ret"] < 0]["drift_ret"]
    pos = panel[panel["prev_close_ret"] >= 0]["drift_ret"]
    comp = pd.DataFrame({
        "condition": ["Prev US last hour DOWN", "Prev US last hour UP"],
        "mean_bps": [neg.mean() * 1e4, pos.mean() * 1e4],
        "t_stat": [_tstat(neg), _tstat(pos)],
        "hit_rate": [(neg > 0).mean(), (pos > 0).mean()],
        "n": [len(neg), len(pos)],
    })
    st.dataframe(comp.style.format({
        "mean_bps": "{:.2f}", "t_stat": "{:.2f}", "hit_rate": "{:.0%}",
    }), hide_index=True)
    st.caption(
        "Paper's asymmetry: strong positive drift after sell pressure, weak/none "
        "after rallies. Note the proxy here is the prior 15:00-16:00 ET return, "
        "not true order-flow imbalance — direction should match, magnitude won't."
    )

    # ---------------- Section 3: VIX interaction -----------------------
    if "vix_prev" in panel.columns and panel["vix_prev"].notna().sum() > 30:
        st.subheader("3 · Does volatility amplify the effect? (dealer VaR story)")
        p = panel.dropna(subset=["vix_prev"]).copy()
        p["vix_bucket"] = pd.qcut(p["vix_prev"], 3, labels=["Low VIX", "Mid VIX", "High VIX"])
        vb = p.groupby("vix_bucket", observed=True).apply(
            lambda g: pd.Series({
                "all_bps": g["drift_ret"].mean() * 1e4,
                "after_selloff_bps": g.loc[g["prev_close_ret"] < 0, "drift_ret"].mean() * 1e4,
                "n": len(g),
            })
        )
        st.dataframe(vb.style.format({"all_bps": "{:.2f}",
                                      "after_selloff_bps": "{:.2f}",
                                      "n": "{:.0f}"}))
        st.caption(
            "Paper: sell imbalance at VIX≈30 → ~6 bps overnight; at VIX≈12 → ~0. "
            "Look for the after-selloff column rising across buckets."
        )

    # ---------------- Section 4: strategy comparison -------------------
    st.subheader("4 · Naive vs conditional 'buy the dip' (pre-cost)")
    uncond = panel["drift_ret"]
    cond = panel["drift_ret"].where(panel["prev_close_ret"] < 0, 0.0)

    def _sharpe(x: pd.Series) -> float:
        traded = x[x != 0] if (x == 0).any() else x
        if traded.std(ddof=1) == 0 or len(traded) < 3:
            return np.nan
        return float(traded.mean() / traded.std(ddof=1) * np.sqrt(TRADING_DAYS))

    eq = pd.DataFrame({
        "Unconditional (every night)": (1 + uncond).cumprod(),
        "Conditional (only after US sell-off)": (1 + cond).cumprod(),
    }, index=pd.to_datetime(panel.index))
    fig2 = go.Figure()
    for col in eq.columns:
        fig2.add_trace(go.Scatter(x=eq.index, y=eq[col], name=col, mode="lines"))
    fig2.update_layout(height=380, yaxis_title="Growth of 1",
                       margin=dict(t=20, b=40), legend=dict(orientation="h"))
    st.plotly_chart(fig2, use_container_width=True)

    c1, c2 = st.columns(2)
    uncond_sharpe = _sharpe(uncond)
    cond_sharpe = _sharpe(cond)
    c1.metric("Unconditional Sharpe (pre-cost)", f"{uncond_sharpe:.2f}")
    c2.metric("Conditional Sharpe (pre-cost)", f"{cond_sharpe:.2f}")

    # Log this research run to Postgres (audit trail, not trade_setups — ES
    # futures isn't a registry forex/metals pair, and this is a backtest
    # study, not a directional pair+bias signal). Deduped per lookback period
    # via NotifyCache, since re-running with the same period recomputes the
    # same read from the same cached data.
    if NotifyCache("overnight_drift_log").filter_new([period]):
        log_tool_usage("overnight_drift", {
            "period": period,
            "drift_hour_ann_pct": float(drift_row["ann_pct"]) if drift_row is not None else None,
            "drift_hour_t_stat": float(drift_row["t_stat"]) if drift_row is not None else None,
            "drift_hour_hit_rate": float(drift_row["hit_rate"]) if drift_row is not None else None,
            "unconditional_sharpe": None if np.isnan(uncond_sharpe) else float(uncond_sharpe),
            "conditional_sharpe": None if np.isnan(cond_sharpe) else float(cond_sharpe),
        })

    st.warning(
        "Paper's punchline: the unconditional version dies after bid-ask costs "
        "(Sharpe 1.1 → -0.5). Only the conditional version survived, because it "
        "pays the spread on ~half the nights. Before trading this, subtract a "
        "realistic ES spread+commission (~0.25-0.5 index pts round trip) per trade."
    )

    st.info(
        "SAST note: 02:00-03:00 ET = 08:00-09:00 SAST (US winter) / "
        "09:00-10:00 SAST (US summer) — the Frankfurt/pre-London open, "
        "before your work day locks you out. The paper's DST test showed the "
        "effect follows *European traders' clocks*, so anchor to the European "
        "open, not the ET wall clock."
    )


# ===========================================================================
# PAGE ENTRY — multipage (auto-run) + standalone (streamlit run)
# ===========================================================================

def _page() -> None:
    from src.ui.theme import BloombergTheme
    from src.pages_lib.navigation import render_sidebar_nav

    st.set_page_config(
        page_title="ONDR · Overnight Drift",
        page_icon="🌙",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    BloombergTheme.apply()
    with st.sidebar:
        st.markdown("### 🌙 Overnight Drift")
        st.caption("NY Fed SR 917 re-tested on live ES data")
        st.divider()
        render_sidebar_nav()
    render_overnight_drift_tab()


_page()