"""Leading Indicators — oscillators & flow that turn before price.

DeMarker (exhaustion), volume-delta flow proxy, Stochastic / Williams %R /
CCI context, classic pivots, and the combined **RSI divergence + pivot
confluence** read that suits hard-trending metals (XAU/XAG) better than
naked overbought/oversold. Deliberately orthogonal to the trend/structure
stack — an early-warning lens, not another trend definition.

Math lives in src/indicators/leading.py (pure, unit-tested); data comes from
the canonical spine. Audit-only: reads journal to tool_usage_log, not
trade_setups (oscillator reads are context, not standalone trade signals).
"""
import streamlit as st

from src.ui.theme import BloombergTheme
from src.ui.theme import BloombergTheme as T
from src.pages_lib.navigation import render_sidebar_nav
from src.instruments.registry import INSTRUMENTS

st.set_page_config(
    page_title="Leading Indicators · Trading System",
    page_icon="🧲",
    layout="wide",
    initial_sidebar_state="expanded",
)
BloombergTheme.apply()

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.core.config import CANDLE_STYLE, default_config as cfg
from src.indicators.leading import (
    cci, demarker, divergence_pivot_read, stochastic, volume_delta, williams_r,
)
from src.indicators.technical import TechnicalIndicators
from src.services.market_data import daily_ohlc, h4_ohlc, asof_caption
from src.ui.components import MetricCell, Panel, render_metric_row

# ── Sidebar (contract: header → controls, symbol first → divider → nav) ─────
with st.sidebar:
    st.markdown("### 🧲 Leading Indicators")
    inst_keys = list(INSTRUMENTS.keys())
    default_ix = inst_keys.index("XAU/USD") if "XAU/USD" in inst_keys else 0
    pair = st.selectbox("Instrument", inst_keys, index=default_ix)
    timeframe = st.selectbox("Timeframe", ["Daily", "4H"], index=0)
    rsi_period = st.number_input("RSI period", 5, 50, 14)
    lookback = st.slider("Divergence lookback (bars)", 30, 120, 60, step=10)
    tol = st.slider("Pivot tolerance (%)", 0.05, 0.50, 0.15, step=0.05)
    st.divider()
    render_sidebar_nav()

ticker = INSTRUMENTS[pair]["ticker"]

# ── Data (canonical spine) ──────────────────────────────────────────────────
with st.spinner(f"Fetching {timeframe} bars for {pair}…"):
    df = daily_ohlc(ticker) if timeframe == "Daily" else h4_ohlc(ticker)

st.markdown(f"## 🧲 Leading Indicators — {pair} · {timeframe}")
st.caption(
    "Exhaustion, flow and divergence — signals that can turn *before* price. "
    "On hard-trending instruments (gold especially), naked overbought/oversold "
    "gets run over; the read that earns attention is RSI divergence landing on "
    "a pivot level. Context, not a standalone trigger — confirm with the "
    "house view and your setup process. · " + asof_caption(df)
)

if df is None or len(df) < 40:
    st.error(f"Not enough {timeframe} history for {pair}.")
    st.stop()

# ── Shared house view (this page's read may disagree — that's the point of a
#    leading lens; the strip flags a direct conflict) ────────────────────────
read = divergence_pivot_read(df, rsi_period=int(rsi_period),
                             lookback=int(lookback), tolerance_pct=float(tol))
from src.services.bias_service import show_house_view
show_house_view(pair, page_read=read["direction"], page_label="Leading Indicators")

# ── Compute the oscillator set ──────────────────────────────────────────────
k, d = stochastic(df)
wr = williams_r(df)
cci_s = cci(df)
dm = demarker(df)
rsi_s = TechnicalIndicators.rsi(df["Close"], int(rsi_period))
vd = volume_delta(df)

def _last(s):
    v = s.dropna()
    return float(v.iloc[-1]) if len(v) else None

vals = {
    "DeMarker": _last(dm), "Stoch %K": _last(k), "Stoch %D": _last(d),
    "Williams %R": _last(wr), "CCI": _last(cci_s), "RSI": _last(rsi_s),
    "Flow (20)": _last(vd["imbalance"]),
}

def _zone(name, v):
    """(display, color, read) per oscillator convention."""
    if v is None:
        return "—", "white", "no data"
    if name == "DeMarker":
        return (f"{v:.2f}", "red" if v > 0.7 else "green" if v < 0.3 else "white",
                "buying exhausted" if v > 0.7 else "selling exhausted" if v < 0.3 else "neutral")
    if name.startswith("Stoch"):
        return (f"{v:.0f}", "red" if v > 80 else "green" if v < 20 else "white",
                "overbought" if v > 80 else "oversold" if v < 20 else "neutral")
    if name == "Williams %R":
        return (f"{v:.0f}", "red" if v > -20 else "green" if v < -80 else "white",
                "overbought" if v > -20 else "oversold" if v < -80 else "neutral")
    if name == "CCI":
        return (f"{v:+.0f}", "red" if v > 100 else "green" if v < -100 else "white",
                "stretched up" if v > 100 else "stretched down" if v < -100 else "neutral")
    if name == "RSI":
        return (f"{v:.0f}", "red" if v > 70 else "green" if v < 30 else "white",
                "overbought" if v > 70 else "oversold" if v < 30 else "neutral")
    if name == "Flow (20)":
        return (f"{v:+.2f}", "green" if v > 0.2 else "red" if v < -0.2 else "white",
                "buy-side flow" if v > 0.2 else "sell-side flow" if v < -0.2 else "balanced")
    return f"{v:.2f}", "white", ""

cells, reads = [], {}
for name, v in vals.items():
    disp, colr, zone = _zone(name, v)
    reads[name] = zone
    cells.append(MetricCell(name, disp, color=colr, delta=zone))
render_metric_row(cells[:4])
render_metric_row(cells[4:])

# ── The combined gold read ──────────────────────────────────────────────────
div = read["divergence"]
if read["direction"]:
    verdict = ("🟢 LONG watch — bullish RSI divergence at pivot "
               f"{read['at_level']}" if read["direction"] == "Long" else
               f"🔴 SHORT watch — bearish RSI divergence at pivot {read['at_level']}")
    v_color = T.GREEN if read["direction"] == "Long" else T.RED
elif div["bullish"] or div["bearish"]:
    verdict = ("🟡 Divergence present "
               f"({'bullish' if div['bullish'] else 'bearish'}) but price is "
               "not at a pivot — watch the levels below")
    v_color = T.YELLOW
else:
    verdict = "⚪ No RSI divergence in the window — nothing leading to act on"
    v_color = T.GREY

Panel(
    title="RSI DIVERGENCE + PIVOT CONFLUENCE",
    tag="THE GOLD READ",
    body_html=(
        f'<div style="font-size:14px;font-weight:700;color:{v_color};">{verdict}</div>'
        f'<div style="color:{T.GREY};font-size:11px;margin-top:6px;">'
        f'{div["detail"]} · price {read["price"]:.5g}'
        + (f' at {read["at_level"]}' if read["at_level"] else " (between levels)")
        + '</div>'
    ),
).show()

# ── Chart: candles + pivots · RSI · flow ────────────────────────────────────
view = df.tail(120)
fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                    row_heights=[0.55, 0.22, 0.23], vertical_spacing=0.03,
                    subplot_titles=[f"{pair} {timeframe} + prior-bar pivots",
                                    f"RSI ({int(rsi_period)})",
                                    "Volume-delta flow (imbalance ±1)"])
fig.add_trace(go.Candlestick(
    x=view.index, open=view["Open"], high=view["High"],
    low=view["Low"], close=view["Close"], name="Price",
    showlegend=False, **CANDLE_STYLE), row=1, col=1)
_pivot_colors = {"P": T.AMBER, "R1": T.RED, "S1": T.GREEN, "R2": T.RED,
                 "S2": T.GREEN, "R3": T.RED, "S3": T.GREEN}
for name, level in read["pivots"].items():
    fig.add_hline(y=level, line_dash="dot", line_width=1,
                  line_color=_pivot_colors.get(name, T.GREY),
                  annotation_text=name, annotation_position="right",
                  annotation_font_color=_pivot_colors.get(name, T.GREY),
                  row=1, col=1)
fig.add_trace(go.Scatter(x=view.index, y=rsi_s.tail(120), name="RSI",
                         line=dict(color=T.CYAN, width=1.5),
                         showlegend=False), row=2, col=1)
fig.add_hline(y=70, line_dash="dot", line_color=T.RED, line_width=1, row=2, col=1)
fig.add_hline(y=30, line_dash="dot", line_color=T.GREEN, line_width=1, row=2, col=1)
imb = vd["imbalance"].tail(120)
fig.add_trace(go.Bar(
    x=view.index, y=imb, name="Flow",
    marker_color=[T.GREEN if (v or 0) > 0 else T.RED for v in imb],
    showlegend=False), row=3, col=1)
fig.add_hline(y=0, line_color=T.BORDER, line_width=1, row=3, col=1)
fig.update_layout(
    height=680, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=T.BG,
    margin=dict(l=10, r=60, t=30, b=10), xaxis_rangeslider_visible=False,
    font=dict(family="JetBrains Mono, monospace", color=T.WHITE, size=10),
)
for row in (1, 2, 3):
    fig.update_xaxes(showgrid=False, linecolor=T.BORDER,
                     tickfont=dict(color=T.GREY, size=9), row=row, col=1)
    fig.update_yaxes(showgrid=True, gridcolor=T.BORDER,
                     tickfont=dict(color=T.GREY, size=9), row=row, col=1)
fig.update_annotations(font_color=T.GREY, font_size=10)
st.plotly_chart(fig, width="stretch", config=dict(displayModeBar=False))

st.caption(
    "DeMarker >0.70 = buying exhausted / <0.30 = selling exhausted · "
    "Flow is candle-direction-signed volume (true-range proxy on zero-volume "
    "spot FX) · Pivots are computed from the prior bar — leading by "
    "construction. Not financial advice."
)

# ── Audit trail: one tool_usage_log row per distinct read (never per rerun) ─
try:
    from src.services.alert_service import NotifyCache
    from src.services.tool_log import log_tool_usage
    _key = (f"{pair}|{timeframe}|{df.index[-1].date()}|"
            f"{read['direction'] or 'none'}|{div['bullish']}|{div['bearish']}")
    _cache = NotifyCache("leading_ind_log")
    if _key not in _cache.load():
        if log_tool_usage("leading_ind", {
            "pair": pair, "timeframe": timeframe,
            "direction": read["direction"],
            "divergence_bullish": bool(div["bullish"]),
            "divergence_bearish": bool(div["bearish"]),
            "at_level": read["at_level"], "price": read["price"],
            "readings": {k_: (round(v_, 3) if v_ is not None else None)
                         for k_, v_ in vals.items()},
        }):
            _cache.add([_key])
except Exception:
    pass
