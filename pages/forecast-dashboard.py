"""Forecast Lab — the single home for everything forecasting.

Backtesting moved to its own page (Backtest Lab). This page consolidates the
three forecasting surfaces that used to be scattered across two pages:

  • PROBABILITY — 8,000-path Monte Carlo cone, terminal distribution, and the
                  house-risk-model touch probabilities / best-bet verdict.
  • STATISTICAL — ARIMA / Exponential-Smoothing point trajectory with a
                  prediction interval, plus FRED macro context.
  • ACCURACY    — rolling out-of-sample backtest of the forecast itself
                  (MAPE / RMSE, skill vs random walk, directional hit-rate,
                  Monte-Carlo cone calibration).

All math lives in src/services/forecast_service.py.
"""
import warnings
from datetime import datetime, timedelta
from io import StringIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

from src.ui.theme import BloombergTheme as T
from src.pages_lib.navigation import render_sidebar_nav
from src.instruments.registry import INSTRUMENTS, TREND_COMMODITIES
from src.pages_lib.setup_ranker import fmt_price
from src.services import backtest_service as bt
from src.services.forecast_service import (
    run_forecast, run_statistical_forecast, backtest_forecast_accuracy)

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Forecast Lab · Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)
T.apply()

st.markdown("""
<style>
    .stApp { --border: color-mix(in srgb, var(--text-color) 12%, transparent); }
    html,body,[class*="css"]{font-family:'JetBrains Mono','Fira Code',monospace;}
    .metric-box{background:var(--background-color);border:1px solid var(--border,#2a2a2a);border-radius:8px;padding:12px;text-align:center;}
    .metric-value{font-size:21px;font-weight:700;color:#e6e6e6;}
    .metric-label{font-size:10px;color:#9a9a9a;margin-top:2px;font-weight:500;letter-spacing:.05em;text-transform:uppercase;}
    [data-testid="stSidebarNav"]{display:none;}
    .block-container{padding-top:1.5rem;max-width:1500px;}
</style>
""", unsafe_allow_html=True)

GREEN, RED, AMBER, GREY = "#00ff66", "#ff3344", "#ffcc00", "#9a9a9a"

CURRENCY_PAIRS = {name: INSTRUMENTS[name]["ticker"] for name in INSTRUMENTS}
METALS = sorted(TREND_COMMODITIES)
FOREX = [n for n in CURRENCY_PAIRS if n not in TREND_COMMODITIES]

FRED_SERIES = {
    "US 10Y Treasury Yield (%)": "DGS10",
    "US Effective Fed Funds Rate (%)": "DFF",
    "Germany 10Y Govt Bond Yield (%)": "IRLTLT01DEM156N",
    "US Dollar Index (Broad, DXY proxy)": "DTWEXBGS",
}


def metric(col, value, label, color="#e6e6e6"):
    col.markdown(
        f'<div class="metric-box"><div class="metric-value" style="color:{color}">{value}</div>'
        f'<div class="metric-label">{label}</div></div>', unsafe_allow_html=True)


def _dark(fig, height=360, **kw):
    fig.update_layout(paper_bgcolor="#000000", plot_bgcolor="#0f0f0f",
                      font=dict(color="#e6e6e6", size=10), height=height,
                      margin=dict(l=30, r=20, t=20, b=30), **kw)
    fig.update_xaxes(gridcolor="#2a2a2a")
    fig.update_yaxes(gridcolor="#2a2a2a")
    return fig


# ══════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════

@st.cache_data(ttl=1800, show_spinner=False)
def load_history(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna(subset=["Close"])


@st.cache_data(ttl=3600, show_spinner=False)
def load_fred_series(series_id: str, start: datetime, end: datetime) -> pd.DataFrame:
    try:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        date_col = df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.set_index(date_col)
        df.columns = [series_id]
        df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
        df = df.dropna()
        return df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
    except Exception:
        return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 📈 Forecast Lab")
    st.markdown("---")

    asset_class = st.radio("Asset class", ["All", "Forex", "Metals"], index=0, horizontal=True)
    choices = FOREX if asset_class == "Forex" else METALS if asset_class == "Metals" else list(CURRENCY_PAIRS)
    pair = st.selectbox("Instrument", choices, index=0)
    ticker = CURRENCY_PAIRS[pair]
    pip_sz = INSTRUMENTS[pair]["pip_size"]

    period = st.selectbox("History window", ["6mo", "1y", "2y", "5y", "10y"], index=3)

    st.markdown("---")
    st.caption("RISK MODEL (for touch probabilities)")
    sl_mult = st.slider("SL × ATR", 1.0, 3.0, 1.5, 0.1)
    rr = st.slider("TP R:R", 1.5, 4.0, 2.0, 0.5)

    st.divider()
    render_sidebar_nav()

with st.spinner("Loading market data…"):
    df = load_history(ticker, period, "1d")

if df.empty or len(df) < 60:
    st.error("Could not retrieve enough data for this instrument. Try again later.")
    st.stop()

close = df["Close"]
last_price = float(close.iloc[-1])
atr14 = float(bt.calc_atr(df, 14).iloc[-1])

_kind = "Metal" if pair in TREND_COMMODITIES else "FX"
st.markdown(f"## 📈 {pair} {_kind} — Forecast Lab")

tab_prob, tab_stat, tab_acc = st.tabs(
    ["🔮 PROBABILITY", "📊 STATISTICAL + MACRO", "🎯 ACCURACY"])


# ══════════════════════════════════════════════════════════════════
# TAB 1 — PROBABILITY (Monte Carlo)
# ══════════════════════════════════════════════════════════════════

with tab_prob:
    horizon = st.radio("Forecast horizon", [5, 10, 21], index=1, horizontal=True,
                       format_func=lambda d: f"{d} trading days", key="fc_horizon")
    with st.spinner("Running 8,000 Monte Carlo paths…"):
        fc = run_forecast(close, atr14, int(horizon), float(sl_mult), float(rr))

    rng_pips = (fc.q95 - fc.q05) / pip_sz
    vol_tag = ("ELEVATED" if fc.vol_ratio > 1.15 else
               "SUPPRESSED" if fc.vol_ratio < 0.85 else "NORMAL")
    k = st.columns(5)
    metric(k[0], f"{fc.p_up*100:.0f}%", f"P(higher in {horizon}d)",
           GREEN if fc.p_up >= 0.5 else RED)
    metric(k[1], f"{fc.exp_move_pct:+.2f}%", "Median move")
    metric(k[2], f"{fc.median_price:,.5f}", "Median price")
    metric(k[3], f"{rng_pips:,.0f}p", "90% range")
    metric(k[4], vol_tag, f"Vol regime ({fc.vol_ratio:.2f}×)")

    # ── Best bet under the house risk model ───────────────────────────
    be = 1.0 / (1.0 + rr)
    edge_long = fc.p_tp_first_long - be
    edge_short = fc.p_tp_first_short - be
    best_side = "LONG" if edge_long >= edge_short else "SHORT"
    best_edge = max(edge_long, edge_short)
    edge_col = GREEN if best_edge > 0.03 else AMBER if best_edge > -0.03 else RED
    verdict = ("EDGE" if best_edge > 0.03 else "MARGINAL" if best_edge > -0.03 else "NO EDGE — STAY FLAT")
    st.markdown(f"""
<div style="border:1px solid {edge_col};background:#0f0f0f;padding:12px 16px;margin:12px 0;">
  <span style="color:{edge_col};font-weight:700;font-size:15px;letter-spacing:.1em;">
    BEST BET · {pair} {best_side} — {verdict}</span><br>
  <span style="color:#e6e6e6;font-size:12px;">
    P(TP before SL): LONG <b style="color:{GREEN if edge_long > 0 else RED}">{fc.p_tp_first_long*100:.0f}%</b>
    · SHORT <b style="color:{GREEN if edge_short > 0 else RED}">{fc.p_tp_first_short*100:.0f}%</b>
    · breakeven at {rr}:1 = {be*100:.0f}%
    · SL {fc.sl_dist/pip_sz:.0f}p / TP {fc.tp_dist/pip_sz:.0f}p</span><br>
  <span style="color:#9a9a9a;font-size:11px;">Close-touch estimates from 8,000 simulated paths; intraday wicks hit both levels slightly more often.</span>
</div>""", unsafe_allow_html=True)

    cc1, cc2 = st.columns([3, 2])
    hist_tail = close.iloc[-60:]
    future_x = list(range(1, int(horizon) + 1))
    with cc1:
        st.markdown("**Probability cone**")
        fig_c = go.Figure()
        fig_c.add_trace(go.Scatter(x=list(range(-len(hist_tail) + 1, 1)), y=hist_tail.values,
                                   name="History", line=dict(color="#e6e6e6", width=1.4)))
        for hi, lo, alpha in (("q95", "q05", 0.10), ("q75", "q25", 0.18)):
            fig_c.add_trace(go.Scatter(x=future_x + future_x[::-1],
                                       y=list(fc.cone[hi]) + list(fc.cone[lo])[::-1],
                                       fill="toself", fillcolor=f"rgba(0,255,65,{alpha})",
                                       line=dict(width=0), hoverinfo="skip", showlegend=False))
        fig_c.add_trace(go.Scatter(x=future_x, y=fc.cone["q50"], name="Median path",
                                   line=dict(color="#00ff41", width=2)))
        fig_c.add_hline(y=fc.last + fc.tp_dist, line_dash="dash", line_color=GREEN, line_width=1)
        fig_c.add_hline(y=fc.last - fc.sl_dist, line_dash="dash", line_color=RED, line_width=1)
        st.plotly_chart(_dark(fig_c, 380, legend=dict(orientation="h", y=1.05, bgcolor="rgba(0,0,0,0)"),
                              xaxis=dict(title="Trading days (0 = today)")),
                        width="stretch", config=dict(displayModeBar=False))
    with cc2:
        st.markdown(f"**Where price lands in {horizon} days**")
        term = fc.terminal_sample
        fig_h = go.Figure(go.Histogram(x=term, nbinsx=60,
                                       marker=dict(color=[GREEN if t > fc.last else RED for t in term])))
        fig_h.add_vline(x=fc.last, line_color="#e6e6e6", line_width=1.5)
        st.plotly_chart(_dark(fig_h, 380, showlegend=False),
                        width="stretch", config=dict(displayModeBar=False))

    with st.expander("◇ The math under the hood"):
        st.markdown(f"""
- **8,000 Monte Carlo paths**: half parametric (EWMA volatility λ=0.94, Student-t(5) fat tails, drift shrunk 75% to zero), half stationary block bootstrap of the real return history (mean block 5 days).
- **Volatility regime**: {fc.ewma_vol_ann:.1f}% current vs {fc.longrun_vol_ann:.1f}% long-run annualised → {fc.vol_ratio:.2f}×.
- **Touch probabilities** use your risk model (SL = {sl_mult}×ATR14 = {fc.sl_dist/pip_sz:.0f}p, TP = {rr}R = {fc.tp_dist/pip_sz:.0f}p): per path, which level does the close cross first?
- Deterministic seed — numbers only change when the data, horizon or risk settings change.
""")


# ══════════════════════════════════════════════════════════════════
# TAB 2 — STATISTICAL + MACRO
# ══════════════════════════════════════════════════════════════════

with tab_stat:
    cset = st.columns([1, 1, 2])
    model_choice = cset[0].radio("Model", ["Exponential Smoothing", "ARIMA"], key="stat_model")
    stat_h = cset[1].slider("Horizon (business days)", 5, 20, 5, key="stat_h")

    with st.spinner("Fitting model…"):
        try:
            sf = run_statistical_forecast(close, stat_h, model_choice, ci=0.90)
            stat_ok = True
        except Exception as e:
            stat_ok = False
            st.warning(f"Model failed to fit: {e}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=close.index, y=close.values, name="Historical", mode="lines",
                             line=dict(color=T.CYAN, width=1.5), fill="tozeroy",
                             fillcolor="rgba(0,224,255,0.10)"))
    if stat_ok:
        bx = [close.index[-1]] + list(sf.forecast.index)
        fig.add_trace(go.Scatter(x=bx, y=[last_price] + list(sf.forecast.values),
                                 name=f"Forecast ({model_choice})", mode="lines",
                                 line=dict(color=T.YELLOW, width=2, dash="dash")))
        fig.add_trace(go.Scatter(x=list(sf.upper.index) + list(sf.lower.index)[::-1],
                                 y=list(sf.upper.values) + list(sf.lower.values)[::-1],
                                 fill="toself", fillcolor="rgba(255,204,0,0.12)",
                                 line=dict(width=0), name="90% interval", hoverinfo="skip"))
        yvals = list(close.values) + list(sf.lower.values) + list(sf.upper.values)
    else:
        yvals = list(close.values)
    ymin, ymax = float(np.nanmin(yvals)), float(np.nanmax(yvals))
    pad = (ymax - ymin) * 0.08 or abs(ymax) * 0.001
    st.plotly_chart(_dark(fig, 460,
                          legend=dict(orientation="h", y=1.02, x=1, xanchor="right", bgcolor="rgba(0,0,0,0)"),
                          yaxis=dict(title="Price", range=[ymin - pad, ymax + pad]),
                          hovermode="x unified"),
                    width="stretch")

    if stat_ok:
        end_val = float(sf.forecast.iloc[-1])
        end_pct = (end_val - last_price) / last_price * 100
        mc = st.columns(4)
        metric(mc[0], fmt_price(last_price), "Current price")
        metric(mc[1], fmt_price(end_val), f"{stat_h}d forecast",
               GREEN if end_pct >= 0 else RED)
        metric(mc[2], f"{end_pct:+.2f}%", "Projected move", GREEN if end_pct >= 0 else RED)
        metric(mc[3], f"{fmt_price(float(sf.lower.iloc[-1]))} – {fmt_price(float(sf.upper.iloc[-1]))}", "90% interval")
    st.caption("Classical time-series extrapolation of historical prices — not investment advice.")

    # ── Macro context ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Macro Context (FRED)")
    fred_start = datetime.today() - timedelta(days=365 * 5)
    fred_cols = st.columns(2)
    any_fred = False
    for i, (name, sid) in enumerate(FRED_SERIES.items()):
        fred_df = load_fred_series(sid, fred_start, datetime.today())
        with fred_cols[i % 2]:
            if not fred_df.empty:
                any_fred = True
                st.markdown(f"**{name}**  \nLatest: {fred_df.iloc[-1, 0]:.2f} ({fred_df.index[-1]:%b %Y})")
                st.line_chart(fred_df, height=170, width="stretch")
            else:
                st.markdown(f"**{name}**")
                st.caption("Data unavailable from FRED right now.")
    if not any_fred:
        st.info("FRED data could not be loaded (no API key required — check your connection).")


# ══════════════════════════════════════════════════════════════════
# TAB 3 — ACCURACY
# ══════════════════════════════════════════════════════════════════

with tab_acc:
    st.markdown("#### Does this forecast actually work? — rolling out-of-sample backtest")
    st.caption("Steps back through history, forecasts ahead from each origin using only prior data, "
               "and scores every forecast against what really happened.")

    cset = st.columns([1, 1])
    acc_model = cset[0].selectbox("Model to evaluate",
                                  ["Monte Carlo median", "Exponential Smoothing", "ARIMA"])
    acc_h = cset[1].slider("Forecast horizon (days)", 5, 21, 10, key="acc_h")

    if st.button("▶ Run accuracy backtest", type="primary"):
        with st.spinner("Backtesting forecasts across history…"):
            acc = backtest_forecast_accuracy(close, int(acc_h), acc_model, n_windows=40)
        st.session_state["acc_result"] = acc

    if st.session_state.get("acc_result") is None and "acc_result" in st.session_state:
        st.warning("Not enough history for this horizon. Use a longer history window in the sidebar.")
    elif "acc_result" in st.session_state:
        acc = st.session_state["acc_result"]
        k = st.columns(5)
        metric(k[0], f"{acc.mape:.2f}%", "MAPE (point error)")
        metric(k[1], f"{acc.rmse_pct:.2f}%", "RMSE %")
        metric(k[2], f"{acc.skill*100:+.0f}%", "Skill vs random walk",
               GREEN if acc.skill > 0 else RED)
        metric(k[3], f"{acc.directional_hit:.0f}%", "Directional hit rate",
               GREEN if acc.directional_hit > 50 else RED)
        metric(k[4], f"{acc.cone_coverage:.0f}%", "90% cone coverage",
               GREEN if 80 <= acc.cone_coverage <= 98 else AMBER)

        st.markdown(
            f"Evaluated **{acc.n_windows}** out-of-sample windows. "
            + ("The model **beats** a naive random-walk baseline. " if acc.skill > 0
               else "The model does **not** beat a naive random-walk baseline — point forecasts add little. ")
            + (f"Cone calibration is healthy ({acc.cone_coverage:.0f}% vs 90% target)."
               if 80 <= acc.cone_coverage <= 98 else
               f"Cone is { 'too wide' if acc.cone_coverage > 98 else 'too narrow' } "
               f"({acc.cone_coverage:.0f}% vs 90% target).")
        )

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Forecast vs actual (terminal price)**")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=acc.actual, y=acc.pred, mode="markers",
                                     marker=dict(color=GREEN, size=6, opacity=0.6),
                                     hovertemplate="actual %{x:.5f}<br>pred %{y:.5f}<extra></extra>"))
            lo, hi = float(min(acc.actual.min(), acc.pred.min())), float(max(acc.actual.max(), acc.pred.max()))
            fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                                     line=dict(color=GREY, dash="dash"), hoverinfo="skip"))
            st.plotly_chart(_dark(fig, 320, showlegend=False,
                                  xaxis=dict(title="Actual"), yaxis=dict(title="Forecast")),
                            width="stretch", config=dict(displayModeBar=False))
        with c2:
            st.markdown("**Forecast error distribution (%)**")
            fig_h = go.Figure(go.Histogram(x=acc.errors, nbinsx=40,
                                           marker=dict(color=AMBER)))
            fig_h.add_vline(x=0, line_color="#e6e6e6", line_width=1.5)
            st.plotly_chart(_dark(fig_h, 320, showlegend=False, xaxis=dict(title="Error %")),
                            width="stretch", config=dict(displayModeBar=False))
