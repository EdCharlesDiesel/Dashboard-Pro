"""
holding_period_tab.py
=====================
Implements the 'optimal holding period' methodology from:

  "The predictability of technical analysis in foreign exchange market
   using forward return" (2024) — 497 TTRs, 10 currencies, 2000-2022,
   swing windows of 1-7 days.

Core question it answers for YOUR checklist items #05-#07:
  When a weekly signal fires, how many days later does the edge peak,
  and when does it decay to noise?

Method:
  1. Build signal series (+1 long / -1 short / 0 flat) from weekly-
     timeframe rules, evaluated on daily bars.
  2. For each horizon h = 1..H days, compute the signal-aligned forward
     return:  fwd(h) = sign(signal_t) * (close_{t+h}/close_t - 1)
  3. Plot mean fwd(h), Newey-West t-stat, and hit rate per horizon.
     The horizon where the t-stat peaks is the optimal holding period.
  4. Benchmark every rule against a plain time-series momentum baseline
     (sign of trailing 12-week return). Per Hutchinson et al. (2022),
     TSMOM explains virtually all TTR returns — if your rule's edge
     curve doesn't beat this line, the rule adds nothing.

Signals implemented (mapped to your 18-point checklist):
  #05  weekly_ema_align   : close vs weekly EMA(10)  [state-based]
  #06  weekly_rsi_room    : weekly RSI(14) in 30-70 band, direction from EMA
  #07  weekly_ema_cross   : weekly EMA(5) x EMA(10) crossover [event-based]
  BASE tsmom_12w          : sign of 12-week trailing return

Modes:
  STATE  — every day the condition holds counts (overlapping windows,
           hence Newey-West errors with lag = h-1).
  EVENT  — only the day the signal flips counts (cleaner, fewer obs).

Entry point: render_holding_period_tab()
Standalone:  streamlit run pages/holding_period_tab.py (also auto-registers
as a multipage page — see src/pages_lib/navigation.py)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from src.services.alert_service import NotifyCache
from src.services.tool_log import log_tool_usage

MAX_H = 15  # horizons 1..15 days (paper found decay ~7)

PAIRS = {
    "EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X", "USD/JPY": "USDJPY=X",
    "USD/ZAR": "USDZAR=X", "AUD/USD": "AUDUSD=X", "Gold (GC=F)": "GC=F",
    "WTI (CL=F)": "CL=F",
}


# ----------------------------------------------------------------------
# Data + indicators
# ----------------------------------------------------------------------

@st.cache_data(ttl=12 * 3600, show_spinner="Fetching daily bars...")
def fetch_daily(ticker: str, years: int) -> pd.DataFrame:
    df = yf.download(ticker, period=f"{years}y", interval="1d",
                     auto_adjust=False, progress=False)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Close"]].dropna()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def build_signals(daily: pd.DataFrame) -> pd.DataFrame:
    """Weekly-timeframe signals, forward-filled onto daily bars.

    Weekly values are shifted by one week before mapping to daily to
    avoid lookahead: a weekly bar's value is only known at its close.
    """
    c = daily["Close"]
    wk = c.resample("W-FRI").last()

    w_ema10 = ema(wk, 10).shift(1)
    w_ema5 = ema(wk, 5).shift(1)
    w_rsi = rsi(wk, 14).shift(1)
    w_ret12 = wk.pct_change(12).shift(1)

    sig = pd.DataFrame(index=daily.index)

    # #05 — weekly EMA alignment (state)
    ema_align_w = np.sign(wk.shift(1) - w_ema10)
    sig["weekly_ema_align"] = ema_align_w.reindex(daily.index, method="ffill")

    # #06 — RSI has room: same direction as EMA, but only while 30<RSI<70
    room = ((w_rsi > 30) & (w_rsi < 70)).astype(float)
    rsi_sig_w = ema_align_w * room
    sig["weekly_rsi_room"] = rsi_sig_w.reindex(daily.index, method="ffill")

    # #07 — weekly EMA(5)/EMA(10) cross (state; events derived later)
    cross_w = np.sign(w_ema5 - w_ema10)
    sig["weekly_ema_cross"] = cross_w.reindex(daily.index, method="ffill")

    # baseline — 12-week time-series momentum
    sig["tsmom_12w"] = np.sign(w_ret12).reindex(daily.index, method="ffill")

    return sig.fillna(0.0)


# ----------------------------------------------------------------------
# Forward-return engine
# ----------------------------------------------------------------------

def nw_tstat(x: np.ndarray, lag: int) -> float:
    """Newey-West t-stat of the mean, HAC lag = `lag` (overlap correction)."""
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 10:
        return np.nan
    e = x - x.mean()
    g0 = float(e @ e) / n
    var = g0
    for k in range(1, min(lag, n - 1) + 1):
        gk = float(e[k:] @ e[:-k]) / n
        var += 2 * (1 - k / (lag + 1)) * gk
    if var <= 0:
        return np.nan
    return float(x.mean() / np.sqrt(var / n))


def edge_curve(close: pd.Series, signal: pd.Series,
               mode: str, max_h: int = MAX_H) -> pd.DataFrame:
    """Mean signal-aligned forward return, NW t-stat, hit rate per horizon."""
    s = signal.reindex(close.index).fillna(0.0)
    if mode == "event":
        flips = s.diff().fillna(0) != 0
        s = s.where(flips & (s != 0), 0.0)

    rows = []
    for h in range(1, max_h + 1):
        fwd = close.shift(-h) / close - 1.0
        aligned = (np.sign(s) * fwd).where(s != 0)
        a = aligned.dropna().values
        if len(a) == 0:
            rows.append((h, np.nan, np.nan, np.nan, 0))
            continue
        rows.append((
            h,
            a.mean() * 1e4,                              # bps
            nw_tstat(a, lag=h - 1),
            float((a > 0).mean()),
            len(a),
        ))
    return pd.DataFrame(rows, columns=["h", "mean_bps", "t_nw",
                                       "hit_rate", "n"]).set_index("h")


# ----------------------------------------------------------------------
# Render
# ----------------------------------------------------------------------

SIGNAL_LABELS = {
    "weekly_ema_align": "#05 Weekly EMA alignment",
    "weekly_rsi_room": "#06 Weekly RSI has room",
    "weekly_ema_cross": "#07 Weekly EMA 5/10 cross",
    "tsmom_12w": "Baseline: 12w TS-momentum",
}


def render_holding_period_tab() -> None:
    st.header("Optimal Holding Period — forward-return edge curves")
    st.caption(
        "Methodology from the 2024 FX forward-return paper: for each weekly "
        "signal, how does the edge evolve 1-15 days after the signal? "
        "Rules are benchmarked against plain time-series momentum "
        "(Hutchinson et al. 2022: if you don't beat that line, the rule is "
        "just momentum in a costume)."
    )

    c1, c2, c3 = st.columns(3)
    pair = c1.selectbox("Instrument", list(PAIRS.keys()))
    years = c2.slider("History (years)", 5, 20, 10)
    mode = c3.radio("Sampling", ["state", "event"], horizontal=True,
                    help="state = every day in signal; event = signal flips only")

    daily = fetch_daily(PAIRS[pair], years)
    if daily.empty:
        st.error("No data returned for this ticker.")
        return

    sigs = build_signals(daily)
    close = daily["Close"]

    curves = {name: edge_curve(close, sigs[name], mode)
              for name in SIGNAL_LABELS}

    # ---- edge curves: mean forward return ----
    st.subheader("Edge curve — mean signal-aligned forward return (bps)")
    fig = go.Figure()
    for name, label in SIGNAL_LABELS.items():
        dash = "dot" if name == "tsmom_12w" else "solid"
        fig.add_trace(go.Scatter(
            x=curves[name].index, y=curves[name]["mean_bps"],
            name=label, mode="lines+markers", line=dict(dash=dash)))
    fig.add_hline(y=0, line_color="grey", line_width=1)
    fig.update_layout(height=420, xaxis_title="Holding period (days)",
                      yaxis_title="Mean forward return (bps)",
                      legend=dict(orientation="h"), margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)

    # ---- t-stat curves ----
    st.subheader("Significance — Newey-West t-stat per horizon")
    fig2 = go.Figure()
    for name, label in SIGNAL_LABELS.items():
        dash = "dot" if name == "tsmom_12w" else "solid"
        fig2.add_trace(go.Scatter(
            x=curves[name].index, y=curves[name]["t_nw"],
            name=label, mode="lines+markers", line=dict(dash=dash)))
    for lvl in (1.96, -1.96):
        fig2.add_hline(y=lvl, line_color="red", line_dash="dash", line_width=1)
    fig2.update_layout(height=420, xaxis_title="Holding period (days)",
                       yaxis_title="NW t-stat",
                       legend=dict(orientation="h"), margin=dict(t=20))
    st.plotly_chart(fig2, use_container_width=True)

    # ---- verdicts ----
    st.subheader("Verdict per rule")
    base = curves["tsmom_12w"]
    rows = []
    for name, label in SIGNAL_LABELS.items():
        cv = curves[name]
        if cv["t_nw"].isna().all():
            continue
        h_star = int(cv["t_nw"].idxmax())
        beats_base = (cv["mean_bps"] > base["mean_bps"]).mean() > 0.5
        rows.append({
            "rule": label,
            "optimal_h_days": h_star,
            "mean_bps@h*": round(float(cv.loc[h_star, "mean_bps"]), 2),
            "t_nw@h*": round(float(cv.loc[h_star, "t_nw"]), 2),
            "hit@h*": f"{cv.loc[h_star, 'hit_rate']:.0%}",
            "n@h*": int(cv.loc[h_star, "n"]),
            "beats_tsmom": "yes" if (beats_base and name != "tsmom_12w")
                           else ("—" if name == "tsmom_12w" else "no"),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # Log this research run to Postgres (audit trail, not trade_setups — a
    # per-rule edge-curve verdict, not a single directional pair+bias read).
    # Deduped per pair+years+mode via NotifyCache, since re-running with the
    # same settings recomputes the same read from the same cached data.
    _hp_key = f"{pair}|{years}|{mode}"
    if NotifyCache("holding_period_log").filter_new([_hp_key]):
        log_tool_usage("holding_period", {
            "pair": pair, "years": years, "mode": mode, "verdict": rows,
        })

    st.info(
        "How to read this: a rule earns its checklist slot only if "
        "(a) its t-stat clears ±1.96 somewhere in 1-15 days, AND "
        "(b) its edge curve sits above the dotted momentum baseline. "
        "The paper found signals stay effective up to ~7 days — if your "
        "optimal h* lands there too, that independently corroborates it. "
        "Pre-cost numbers: subtract spread before drawing trading conclusions."
    )


# ===========================================================================
# PAGE ENTRY — multipage (auto-run) + standalone (streamlit run)
# ===========================================================================

def _page() -> None:
    from src.ui.theme import BloombergTheme
    from src.pages_lib.navigation import render_sidebar_nav

    st.set_page_config(
        page_title="HOLD · Optimal Holding Period",
        page_icon="⏱️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    BloombergTheme.apply()
    with st.sidebar:
        st.markdown("### ⏱️ Optimal Holding Period")
        st.caption("Forward-return edge curves vs TS-momentum baseline")
        st.divider()
        render_sidebar_nav()
    render_holding_period_tab()


_page()