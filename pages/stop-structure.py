import streamlit as st
from src.ui.theme import BloombergTheme
from src.core.config import CANDLE_STYLE
from src.pages_lib.navigation import render_sidebar_nav
from src.services.alert_service import NotifyCache
from src.services.tool_log import log_tool_usage
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime

# ── Page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stop Below/Above Structure",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)
BloombergTheme.apply()

# ── CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Theme-adaptive layout vars */
    .stApp {
      --border: color-mix(in srgb, var(--text-color) 12%, transparent);
      --muted:  color-mix(in srgb, var(--text-color) 55%, transparent);
    }
    html,body,[class*="css"]{ font-family:'JetBrains Mono','Fira Code',monospace; }
    .stApp{ background:var(--background-color); }
    section[data-testid="stSidebar"]{ background:var(--secondary-background-color)!important; border-right:1px solid var(--border,#2a2a2a); }
    #MainMenu,footer{ visibility:hidden; }
    [data-testid="stSidebarCollapsedControl"]{visibility:visible !important;}
    [data-testid="stSidebarNav"]{ display:none; }
    .block-container{ padding-top:1.5rem; max-width:1380px; }

    .card{ background:var(--secondary-background-color); border:1px solid var(--border,#2a2a2a); border-radius:12px;
           padding:20px; margin-bottom:16px; }
    .card-header{ font-size:13px; font-weight:600; letter-spacing:.08em;
                  text-transform:uppercase; color:#9a9a9a; margin-bottom:14px; }
    .metric-box{ background:var(--background-color); border:1px solid var(--border,#2a2a2a); border-radius:8px;
                 padding:14px; text-align:center; }
    .metric-value{ font-size:22px; font-weight:700; color:var(--text-color); }
    .metric-label{ font-size:11px; color:var(--muted,#9a9a9a); margin-top:2px; font-weight:500;
                   letter-spacing:.04em; text-transform:uppercase; }
    .section-title{ font-size:16px; font-weight:700; color:var(--text-color);
                    margin:24px 0 12px 0; padding-left:4px; border-left:3px solid #00ff41; }
    .prog-track{ background:var(--border,#2a2a2a); border-radius:8px; height:10px;
                 margin:6px 0 2px 0; overflow:hidden; }

    /* SL level cards */
    .sl-card{ border-radius:12px; padding:20px 24px; margin-bottom:12px; }
    .sl-card-long { background:linear-gradient(135deg,#0a2918,#0d3a1f);
                    border:1px solid #00a32a; border-left:4px solid #00ff66; }
    .sl-card-short{ background:linear-gradient(135deg,#2a0a0a,#3a0d0d);
                    border:1px solid #8b2d2d; border-left:4px solid #ff3344; }

    /* Structure level table rows */
    .str-row{ display:flex; justify-content:space-between; align-items:center;
              padding:9px 14px; border-bottom:1px solid #2a2a2a;
              font-size:13px; }
    .str-row:last-child{ border-bottom:none; }
    .str-label{ color:#9a9a9a; }
    .str-val  { font-family:monospace; font-weight:600; color:#e6e6e6; }

    /* Verdict banners */
    .verdict-long { background:linear-gradient(135deg,#0d3a1f,#0d5e32);
                    border:2px solid #00a32a; border-radius:14px;
                    padding:22px 28px; text-align:center; margin-bottom:18px; }
    .verdict-short{ background:linear-gradient(135deg,#3a0d0d,#5e1414);
                    border:2px solid #8b2d2d; border-radius:14px;
                    padding:22px 28px; text-align:center; margin-bottom:18px; }
    .verdict-none { background:linear-gradient(135deg,#0a0a0a,#1c2128);
                    border:1px solid #2a2a2a; border-radius:14px;
                    padding:22px 28px; text-align:center; margin-bottom:18px; }

    .explainer{ background:var(--background-color); border:1px solid #1e3a5f;
                border-left:3px solid #00ff41; border-radius:8px;
                padding:14px 18px; font-size:13px; color:var(--muted,#9a9a9a); line-height:1.7; }
    .formula-box{ background:#000000; border:1px solid #2a2a2a; border-radius:8px;
                  padding:14px 18px; font-family:monospace; font-size:13px;
                  color:#e6e6e6; margin:10px 0; line-height:2; }
</style>
""", unsafe_allow_html=True)


# ── Instrument registry ────────────────────────────────────────────
INSTRUMENTS = {
    "EUR/USD":    {"ticker": "EURUSD=X", "pip": 10.0,  "pip_size": 0.0001},
    "GBP/USD":    {"ticker": "GBPUSD=X", "pip": 10.0,  "pip_size": 0.0001},
    "AUD/USD":    {"ticker": "AUDUSD=X", "pip": 10.0,  "pip_size": 0.0001},
    "NZD/USD":    {"ticker": "NZDUSD=X", "pip": 10.0,  "pip_size": 0.0001},
    "USD/JPY":    {"ticker": "USDJPY=X", "pip": 9.09,  "pip_size": 0.01},
    "USD/CHF":    {"ticker": "USDCHF=X", "pip": 10.8,  "pip_size": 0.0001},
    "USD/CAD":    {"ticker": "USDCAD=X", "pip": 7.4,   "pip_size": 0.0001},
    "EUR/GBP":    {"ticker": "EURGBP=X", "pip": 12.5,  "pip_size": 0.0001},
    "EUR/JPY":    {"ticker": "EURJPY=X", "pip": 9.09,  "pip_size": 0.01},
    "GBP/JPY":    {"ticker": "GBPJPY=X", "pip": 9.09,  "pip_size": 0.01},
    "AUD/JPY":    {"ticker": "AUDJPY=X", "pip": 9.09,  "pip_size": 0.01},
    "EUR/AUD":    {"ticker": "EURAUD=X", "pip": 6.3,   "pip_size": 0.0001},
    "GBP/AUD":    {"ticker": "GBPAUD=X", "pip": 6.3,   "pip_size": 0.0001},
    "EUR/CAD":    {"ticker": "EURCAD=X", "pip": 7.4,   "pip_size": 0.0001},
    "GBP/CAD":    {"ticker": "GBPCAD=X", "pip": 7.4,   "pip_size": 0.0001},
    "USD/ZAR":    {"ticker": "USDZAR=X", "pip": 0.55,  "pip_size": 0.0001},
    "EUR/ZAR":    {"ticker": "EURZAR=X", "pip": 0.55,  "pip_size": 0.0001},
    "GBP/ZAR":    {"ticker": "GBPZAR=X", "pip": 0.55,  "pip_size": 0.0001},
    "XAU/USD":    {"ticker": "GC=F",     "pip": 10.0,  "pip_size": 0.10},
    "XAG/USD":  {"ticker": "SI=F",     "pip": 10.0,  "pip_size": 0.01},
    "XPT/USD":{"ticker": "PL=F",     "pip": 10.0,  "pip_size": 0.10},
    "WTI/USD":    {"ticker": "CL=F",     "pip": 10.0,  "pip_size": 0.01},
}


# ══════════════════════════════════════════════════════════════════
# DATA & CALCULATIONS
# ══════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner=False)
def fetch_data(ticker: str) -> pd.DataFrame:
    from src.db.market_cache import cached_ohlc
    try:
        df = cached_ohlc(ticker, interval="1d", period="1y", ttl=300)
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df[["Open","High","Low","Close","Volume"]].dropna()
    except Exception:
        return pd.DataFrame()


def calc_atr(df: pd.DataFrame, period: int) -> pd.Series:
    prev  = df["Close"].shift(1)
    tr    = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev).abs(),
        (df["Low"]  - prev).abs(),
        ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()


def find_structure_levels(df: pd.DataFrame, pivot_lb: int = 5) -> dict:
    """
    Identify the nearest swing high and swing low using a pivot lookback.
    These act as the structural levels to place the SL beyond.
    """
    H = df["High"].values
    L = df["Low"].values
    n = len(df)

    swing_high_idx, swing_low_idx = [], []
    lb = min(pivot_lb, 3)

    for i in range(lb, n - lb):
        if all(H[i] >= H[i-j] for j in range(1, lb+1)) and \
                all(H[i] >= H[i+j] for j in range(1, lb+1)):
            swing_high_idx.append(i)
        if all(L[i] <= L[i-j] for j in range(1, lb+1)) and \
                all(L[i] <= L[i+j] for j in range(1, lb+1)):
            swing_low_idx.append(i)

    # Most recent confirmed swing high & low
    recent_sh = df.iloc[swing_high_idx[-1]] if swing_high_idx else None
    recent_sl = df.iloc[swing_low_idx[-1]]  if swing_low_idx  else None

    # Build full lists for charting
    sh_df = df.iloc[swing_high_idx] if swing_high_idx else pd.DataFrame()
    sl_df = df.iloc[swing_low_idx]  if swing_low_idx  else pd.DataFrame()

    return {
        "recent_sh": recent_sh,
        "recent_sl": recent_sl,
        "sh_df":     sh_df,
        "sl_df":     sl_df,
        "sh_idx":    swing_high_idx,
        "sl_idx":    swing_low_idx,
    }


def compute_stops(df: pd.DataFrame, atr_mult: float,
                  pip_size: float, pip_val: float,
                  account_bal: float, risk_pct: float,
                  struct: dict, atr_period: int = 14) -> dict:

    atr14    = calc_atr(df, atr_period)
    atr_now  = float(atr14.iloc[-1])
    price    = float(df["Close"].iloc[-1])
    sl_dist  = atr_now * atr_mult          # price distance for SL
    sl_pips  = round(sl_dist / pip_size, 1)
    tp1_pips = round(sl_pips * 2.0, 1)
    tp2_pips = round(sl_pips * 3.0, 1)
    tp1_dist = tp1_pips * pip_size
    tp2_dist = tp2_pips * pip_size

    # ATR buffer applied beyond structure
    atr_buf  = atr_now * 0.20             # 20% of ATR as extra buffer beyond level

    # LONG: SL goes below nearest swing low
    sh_price = float(struct["recent_sh"]["High"]) if struct["recent_sh"] is not None else None
    sl_price = float(struct["recent_sl"]["Low"])  if struct["recent_sl"] is not None else None

    sl_long_struct  = (sl_price - atr_buf)        if sl_price else (price - sl_dist)
    sl_short_struct = (sh_price + atr_buf)         if sh_price else (price + sl_dist)

    sl_long_atr     = price - sl_dist
    sl_short_atr    = price + sl_dist

    # Use structure-based SL if it's within 2× ATR (sanity check)
    use_struct_long  = sl_price and abs(price - sl_long_struct)  <= atr_now * 3
    use_struct_short = sh_price and abs(sl_short_struct - price) <= atr_now * 3

    sl_long_final  = sl_long_struct  if use_struct_long  else sl_long_atr
    sl_short_final = sl_short_struct if use_struct_short else sl_short_atr

    sl_long_pips   = round(abs(price - sl_long_final)  / pip_size, 1)
    sl_short_pips  = round(abs(sl_short_final - price) / pip_size, 1)

    # TP levels
    tp1_long  = price + tp1_dist
    tp2_long  = price + tp2_dist
    tp1_short = price - tp1_dist
    tp2_short = price - tp2_dist

    # Position sizing
    risk_amount  = account_bal * (risk_pct / 100)
    lot_long     = round(risk_amount / (sl_long_pips  * pip_val), 2) if sl_long_pips  else 0
    lot_short    = round(risk_amount / (sl_short_pips * pip_val), 2) if sl_short_pips else 0

    rr_tp1_long  = round(tp1_pips / sl_long_pips,  2) if sl_long_pips  else 0
    rr_tp2_long  = round(tp2_pips / sl_long_pips,  2) if sl_long_pips  else 0
    rr_tp1_short = round(tp1_pips / sl_short_pips, 2) if sl_short_pips else 0
    rr_tp2_short = round(tp2_pips / sl_short_pips, 2) if sl_short_pips else 0

    return {
        "price":         price,
        "atr14":         round(atr_now, 5),
        "sl_pips":       sl_pips,
        "atr_buf":       atr_buf,

        # LONG levels
        "sl_long":       round(sl_long_final,  5),
        "sl_long_pips":  sl_long_pips,
        "tp1_long":      round(tp1_long, 5),
        "tp2_long":      round(tp2_long, 5),
        "lot_long":      lot_long,
        "rr_tp1_long":   rr_tp1_long,
        "rr_tp2_long":   rr_tp2_long,
        "use_struct_long": use_struct_long,

        # SHORT levels
        "sl_short":      round(sl_short_final, 5),
        "sl_short_pips": sl_short_pips,
        "tp1_short":     round(tp1_short, 5),
        "tp2_short":     round(tp2_short, 5),
        "lot_short":     lot_short,
        "rr_tp1_short":  rr_tp1_short,
        "rr_tp2_short":  rr_tp2_short,
        "use_struct_short": use_struct_short,

        "risk_amount":   round(risk_amount, 2),
        "tp1_pips":      tp1_pips,
        "tp2_pips":      tp2_pips,

        # Raw structure prices
        "struct_sh": sh_price,
        "struct_sl": sl_price,
        "atr_series": atr14,
    }


# ══════════════════════════════════════════════════════════════════
# CHART
# ══════════════════════════════════════════════════════════════════

def build_chart(df: pd.DataFrame, pair: str,
                calc: dict, struct: dict,
                direction: str, show_n: int = 80) -> go.Figure:

    plot  = df.tail(show_n).copy()
    price = calc["price"]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.78, 0.22], vertical_spacing=0.03,
    )

    # Candles
    fig.add_trace(go.Candlestick(
        x=plot.index,
        open=plot["Open"], high=plot["High"],
        low=plot["Low"],   close=plot["Close"],
        **CANDLE_STYLE,
        name="Daily", showlegend=False,
    ), row=1, col=1)

    # Volume
    vcol = ["rgba(63,185,80,0.33)" if c >= o else "rgba(248,81,73,0.33)"
            for c, o in zip(plot["Close"], plot["Open"])]
    fig.add_trace(go.Bar(
        x=plot.index, y=plot["Volume"],
        marker_color=vcol, name="Volume", showlegend=False,
    ), row=2, col=1)

    # ATR line on volume panel
    atr_plot = calc["atr_series"].reindex(plot.index).dropna()
    fig.add_trace(go.Scatter(
        x=atr_plot.index, y=atr_plot.values / atr_plot.values.max() * plot["Volume"].max(),
        mode="lines", line=dict(color="rgba(56,139,253,0.33)", width=1.5),
        name="ATR (scaled)", showlegend=False,
    ), row=2, col=1)

    # ── Swing highs / lows ────────────────────────────────────────
    sh_plot = struct["sh_df"].reindex(
        [i for i in struct["sh_df"].index if i in plot.index])
    sl_plot = struct["sl_df"].reindex(
        [i for i in struct["sl_df"].index if i in plot.index])

    if not sh_plot.empty:
        fig.add_trace(go.Scatter(
            x=sh_plot.index,
            y=sh_plot["High"] * 1.0005,
            mode="markers+text",
            marker=dict(symbol="triangle-down", size=10, color="#ff3344",
                        line=dict(color="#000000", width=1)),
            text=["SH"] * len(sh_plot),
            textposition="top center",
            textfont=dict(size=8, color="#ff3344"),
            name="Swing High",
        ), row=1, col=1)

    if not sl_plot.empty:
        fig.add_trace(go.Scatter(
            x=sl_plot.index,
            y=sl_plot["Low"] * 0.9995,
            mode="markers+text",
            marker=dict(symbol="triangle-up", size=10, color="#00ff66",
                        line=dict(color="#000000", width=1)),
            text=["SL"] * len(sl_plot),
            textposition="bottom center",
            textfont=dict(size=8, color="#00ff66"),
            name="Swing Low",
        ), row=1, col=1)

    # ── Key horizontal lines ──────────────────────────────────────
    # Current price
    fig.add_hline(
        y=price, line_color="#ffffff", line_dash="solid",
        line_width=1.5, opacity=0.85,
        annotation_text=f"  Price {price:.5f}",
        annotation_position="right",
        annotation_font=dict(size=10, color="#ffffff"),
        row=1, col=1,
    )

    def hline(y, color, dash, width, label, pos="right"):
        fig.add_hline(
            y=y, line_color=color, line_dash=dash,
            line_width=width, opacity=0.9,
            annotation_text=f"  {label}",
            annotation_position=pos,
            annotation_font=dict(size=9, color=color),
            row=1, col=1,
        )

    if direction in ("LONG", "BOTH"):
        # SL zone band for LONG
        fig.add_hrect(
            y0=calc["sl_long"] - calc["atr_buf"],
            y1=calc["sl_long"] + calc["atr_buf"],
            fillcolor="#ff3344", opacity=0.08, line_width=0, row=1, col=1,
        )
        hline(calc["sl_long"],   "#ff3344", "dash",  2.0,
              f"SL LONG  {calc['sl_long']:.5f}  ({calc['sl_long_pips']:.0f} pips)")
        hline(calc["tp1_long"],  "#00ff66", "dot",   1.5,
              f"TP1  {calc['tp1_long']:.5f}  ({calc['tp1_pips']:.0f} pips)")
        hline(calc["tp2_long"],  "#56d364", "dot",   1.0,
              f"TP2  {calc['tp2_long']:.5f}  ({calc['tp2_pips']:.0f} pips)")

        # Structural SL anchor
        if calc["struct_sl"]:
            hline(calc["struct_sl"], "#ffcc00", "longdash", 1.0,
                  f"Swing Low  {calc['struct_sl']:.5f}", pos="left")

    if direction in ("SHORT", "BOTH"):
        fig.add_hrect(
            y0=calc["sl_short"] - calc["atr_buf"],
            y1=calc["sl_short"] + calc["atr_buf"],
            fillcolor="#ff3344", opacity=0.08, line_width=0, row=1, col=1,
        )
        hline(calc["sl_short"],  "#ff3344", "dash",  2.0,
              f"SL SHORT  {calc['sl_short']:.5f}  ({calc['sl_short_pips']:.0f} pips)")
        hline(calc["tp1_short"], "#00ff66", "dot",   1.5,
              f"TP1  {calc['tp1_short']:.5f}  ({calc['tp1_pips']:.0f} pips)")
        hline(calc["tp2_short"], "#56d364", "dot",   1.0,
              f"TP2  {calc['tp2_short']:.5f}  ({calc['tp2_pips']:.0f} pips)")

        if calc["struct_sh"]:
            hline(calc["struct_sh"], "#ffcc00", "longdash", 1.0,
                  f"Swing High  {calc['struct_sh']:.5f}", pos="left")

    fig.update_layout(
        height=620,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0a0a0a",
        font=dict(family="JetBrains Mono, monospace", size=11, color="#9a9a9a"),
        xaxis_rangeslider_visible=False,
        legend=dict(
            bgcolor="rgba(13,17,23,0.60)", bordercolor="#2a2a2a", borderwidth=1,
            font=dict(size=10), orientation="h",
            yanchor="bottom", y=1.01, xanchor="left", x=0,
        ),
        margin=dict(l=10, r=170, t=10, b=10),
    )
    fig.update_yaxes(gridcolor="#2a2a2a", zeroline=False)
    fig.update_xaxes(gridcolor="#2a2a2a", showgrid=False)
    return fig


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 🛡️ Stop Below/Above Structure")

    inst_keys    = list(INSTRUMENTS.keys())
    default_inst = st.session_state.get("selected_instrument", "EUR/USD")
    if default_inst not in inst_keys:
        default_inst = inst_keys[0]

    selected_pair = st.selectbox("Instrument", inst_keys,
                                 index=inst_keys.index(default_inst))

    direction = st.radio("Trade direction", ["LONG", "SHORT", "BOTH"],
                         horizontal=True)

    st.divider()
    st.markdown("**📐 ATR Settings**")
    atr_period = st.number_input("ATR Period",     value=14, min_value=5, max_value=50)
    atr_mult   = st.number_input("ATR Multiplier (SL)", value=1.5,
                                 min_value=0.5, max_value=5.0, step=0.1, format="%.1f")

    st.divider()
    st.markdown("**💰 Position Sizing**")
    account_bal = st.number_input("Account Balance ($)",
                                  value=float(st.session_state.get("account_bal", 10000.0)),
                                  step=500.0, format="%.2f")
    risk_pct    = st.slider("Risk per Trade (%)", 0.25, 3.0,
                            float(st.session_state.get("risk_pct", 1.0)), 0.25)

    st.divider()
    st.markdown("**📏 Structure Pivot**")
    pivot_lb = st.slider("Pivot lookback (bars)", 2, 10, 5,
                         help="Bars each side needed to confirm a swing pivot")
    show_n   = st.slider("Candles on chart", 40, 200, 80, step=10)

    st.divider()
    if st.button("🔄 Refresh Data", width="stretch", type="primary"):
        st.cache_data.clear()
        st.rerun()

    st.caption(f"🕐 {datetime.now().strftime('%H:%M:%S')} local")


    st.divider()
    render_sidebar_nav()
# ══════════════════════════════════════════════════════════════════
# MAIN PAGE
# ══════════════════════════════════════════════════════════════════

inst     = INSTRUMENTS[selected_pair]
ticker   = inst["ticker"]
pip_size = inst["pip_size"]
pip_val  = inst["pip"]

st.markdown(f"""
<div style="background:linear-gradient(135deg,#000000 0%,#0a0a0a 50%,#000000 100%);
            border:1px solid #2a2a2a; border-radius:16px; padding:24px 28px;
            margin-bottom:20px;">
  <div style="font-size:24px; font-weight:700; color:#e6e6e6;">
    🛡️ Stop Below / Above Structure
  </div>
  <div style="color:#9a9a9a; font-size:13px; margin-top:4px;">
    SL = {atr_mult}× ATR({int(atr_period)}) placed beyond the nearest swing structure · {selected_pair}
  </div>
  <div style="font-size:12px; color:#00ff41; margin-top:6px;">
    Check #14 — Stop placement · {datetime.now().strftime('%A %d %B %Y  |  %H:%M')}
  </div>
</div>
""", unsafe_allow_html=True)

def level_row(label, value, extra=""):
    return (f'<div class="str-row"><span class="str-label">{label}</span>'
            f'<span class="str-val">{value}&nbsp;&nbsp;<span style="color:#9a9a9a;'
            f'font-size:11px;">{extra}</span></span></div>')


# ─────────────────────────────────────────────
#  Tabs
# ─────────────────────────────────────────────
tab_scan, tab_detail = st.tabs(["📊 All-Pairs Scanner", "🔍 Pair Detail"])

# ══════════════════════════════════════════════
#  TAB 1 — All-Pairs Scanner
# ══════════════════════════════════════════════
with tab_scan:
    st.markdown(
        f"Scanning all 21 instruments · ATR({int(atr_period)}) × {atr_mult} · "
        f"pivot lookback {int(pivot_lb)} bars …"
    )
    prog = st.progress(0)
    all_instruments = list(INSTRUMENTS.items())
    n_inst = len(all_instruments)
    scan_results: list[dict] = []

    for idx, (pair_name, info) in enumerate(all_instruments):
        prog.progress((idx + 1) / n_inst, text=f"Scanning {pair_name} …")
        try:
            df_s = fetch_data(info["ticker"])
            if df_s.empty or len(df_s) < int(atr_period) + 10:
                scan_results.append({"pair": pair_name, "ok": False})
                continue
            struct_s = find_structure_levels(df_s, pivot_lb=int(pivot_lb))
            calc_s   = compute_stops(
                df_s, atr_mult=float(atr_mult),
                pip_size=info["pip_size"], pip_val=info["pip"],
                account_bal=account_bal, risk_pct=risk_pct,
                struct=struct_s, atr_period=int(atr_period),
            )
            scan_results.append({
                "pair":         pair_name,
                "ok":           True,
                "price":        calc_s["price"],
                "atr14":        calc_s["atr14"],
                "sl_long_pips": calc_s["sl_long_pips"],
                "sl_short_pips":calc_s["sl_short_pips"],
                "use_struct_l": calc_s["use_struct_long"],
                "use_struct_s": calc_s["use_struct_short"],
                "rr_tp1_long":  calc_s["rr_tp1_long"],
                "rr_tp1_short": calc_s["rr_tp1_short"],
            })
        except Exception:
            scan_results.append({"pair": pair_name, "ok": False})

    prog.empty()

    # ── Summary counts ──────────────────────────
    cnt_both    = sum(1 for r in scan_results if r.get("ok") and r.get("use_struct_l") and r.get("use_struct_s"))
    cnt_long_s  = sum(1 for r in scan_results if r.get("ok") and r.get("use_struct_l") and not r.get("use_struct_s"))
    cnt_short_s = sum(1 for r in scan_results if r.get("ok") and not r.get("use_struct_l") and r.get("use_struct_s"))
    cnt_atr     = sum(1 for r in scan_results if r.get("ok") and not r.get("use_struct_l") and not r.get("use_struct_s"))

    sc1, sc2, sc3, sc4 = st.columns(4)
    for col_, val_, lbl_, c_ in [
        (sc1, cnt_both,    "Both Structures",  "#00ff66"),
        (sc2, cnt_long_s,  "Long Struct Only",  "#56d364"),
        (sc3, cnt_short_s, "Short Struct Only", "#ff3344"),
        (sc4, cnt_atr,     "ATR Fallback",      "#555555"),
    ]:
        with col_:
            st.markdown(
                f'<div class="metric-box">'
                f'<div class="metric-value" style="color:{c_};font-size:28px;">{val_}</div>'
                f'<div class="metric-label">{lbl_}</div>'
                f'</div>',
                unsafe_allow_html=True)

    st.markdown("---")

    # ── Card grid ───────────────────────────────
    cols3 = st.columns(3)
    for i, r in enumerate(scan_results):
        pair_name = r["pair"]
        is_sel    = pair_name == selected_pair
        border    = "border:2px solid #00ff41;" if is_sel else "border:1px solid #2a2a2a;"

        if not r.get("ok"):
            card_body = '<span style="font-size:11px;color:#ff3344;">No data</span>'
        else:
            l_note = "struct" if r["use_struct_l"] else "ATR"
            s_note = "struct" if r["use_struct_s"] else "ATR"
            l_col  = "#00ff66" if r["use_struct_l"] else "#555555"
            s_col  = "#ff3344" if r["use_struct_s"] else "#555555"
            rr_l   = r["rr_tp1_long"]
            rr_s   = r["rr_tp1_short"]
            rr_l_c = "#00ff66" if rr_l >= 2 else "#ffcc00"
            rr_s_c = "#00ff66" if rr_s >= 2 else "#ffcc00"
            card_body = (
                f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">'
                f'<span style="font-size:13px;font-weight:600;color:#e6e6e6;">{pair_name}</span>'
                f'<span style="font-size:11px;color:#9a9a9a;font-family:monospace;">{r["price"]:,.4f}</span>'
                f'</div>'
                f'<div style="display:flex;gap:10px;flex-wrap:wrap;">'
                f'<span style="font-size:11px;color:#9a9a9a;">🔼 SL <span style="color:{l_col};font-family:monospace;">{r["sl_long_pips"]:.0f}p</span> '
                f'<span style="font-size:10px;color:#555555;">({l_note})</span> '
                f'<span style="color:{rr_l_c};font-size:10px;">R:{rr_l:.1f}</span></span>'
                f'<span style="font-size:11px;color:#9a9a9a;">🔽 SL <span style="color:{s_col};font-family:monospace;">{r["sl_short_pips"]:.0f}p</span> '
                f'<span style="font-size:10px;color:#555555;">({s_note})</span> '
                f'<span style="color:{rr_s_c};font-size:10px;">R:{rr_s:.1f}</span></span>'
                f'</div>'
                f'<div style="font-size:10px;color:#555555;margin-top:4px;">ATR {r["atr14"]:.5f}</div>'
            )

        with cols3[i % 3]:
            st.markdown(
                f'<div style="background:#0a0a0a;{border}border-radius:8px;'
                f'padding:12px 14px;margin-bottom:8px;">{card_body}</div>',
                unsafe_allow_html=True)

    # ── Expander table ───────────────────────────
    with st.expander("📋 Full Scanner Results"):
        table_rows = []
        for r in scan_results:
            table_rows.append({
                "Pair":          r["pair"],
                "Price":         f'{r["price"]:,.5f}'      if r.get("ok") else "—",
                f"ATR({int(atr_period)})": f'{r["atr14"]:.5f}' if r.get("ok") else "—",
                "SL Long (pips)":  f'{r["sl_long_pips"]:.1f}'  if r.get("ok") else "—",
                "SL Short (pips)": f'{r["sl_short_pips"]:.1f}' if r.get("ok") else "—",
                "Long Struct":   "✅" if r.get("use_struct_l") else "ATR" if r.get("ok") else "—",
                "Short Struct":  "✅" if r.get("use_struct_s") else "ATR" if r.get("ok") else "—",
                "R:R TP1 L":     f'{r["rr_tp1_long"]:.2f}'  if r.get("ok") else "—",
                "R:R TP1 S":     f'{r["rr_tp1_short"]:.2f}' if r.get("ok") else "—",
            })
        st.dataframe(pd.DataFrame(table_rows), width="stretch", hide_index=True)

# ══════════════════════════════════════════════
#  TAB 2 — Pair Detail
# ══════════════════════════════════════════════
with tab_detail:
    detail_ok = True

    with st.spinner(f"Loading data for {selected_pair}…"):
        df = fetch_data(ticker)

    if df.empty or len(df) < int(atr_period) + 10:
        st.error(f"⚠️ Not enough data for **{selected_pair}**.")
        detail_ok = False

    if detail_ok:
        struct = find_structure_levels(df, pivot_lb=int(pivot_lb))
        calc   = compute_stops(df, atr_mult=float(atr_mult),
                               pip_size=pip_size, pip_val=pip_val,
                               account_bal=account_bal, risk_pct=risk_pct,
                               struct=struct, atr_period=int(atr_period))

        # Log this stop-structure read to Postgres (audit trail). Deduped on
        # the pair + settings shape via NotifyCache — every widget touch
        # reruns this tab, so without dedupe an unrelated tweak would re-log
        # an unchanged read.
        _ss_key = (f"{selected_pair}|{direction}|{atr_mult}|{atr_period}|"
                  f"{pivot_lb}|{calc['sl_pips']:.1f}")
        if NotifyCache("stop_structure_log").filter_new([_ss_key]):
            log_tool_usage("stop_structure", {
                "pair": selected_pair, "direction": direction,
                "atr_mult": atr_mult, "atr_period": atr_period,
                "pivot_lb": pivot_lb, "price": calc["price"],
                "atr14": calc["atr14"], "sl_pips": calc["sl_pips"],
                "tp1_pips": calc["tp1_pips"], "tp2_pips": calc["tp2_pips"],
                "risk_amount": calc["risk_amount"],
            })

        # ── KPI strip ────────────────────────────────
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        for col_, val_, lbl_, c_ in [
            (k1, f"{calc['price']:.5f}",      "Current Price",           "#e6e6e6"),
            (k2, f"{calc['atr14']:.5f}",      f"ATR ({int(atr_period)})", "#00ff41"),
            (k3, f"{calc['sl_pips']:.1f}",    f"SL Pips ({atr_mult}×ATR)","#ff3344"),
            (k4, f"{calc['tp1_pips']:.1f}",   "TP1 Pips (2:1)",          "#00ff66"),
            (k5, f"{calc['tp2_pips']:.1f}",   "TP2 Pips (3:1)",          "#56d364"),
            (k6, f"${calc['risk_amount']:.2f}","Risk Amount",             "#ffcc00"),
        ]:
            with col_:
                st.markdown(
                    f'<div class="metric-box">'
                    f'<div class="metric-value" style="color:{c_};font-size:17px;">{val_}</div>'
                    f'<div class="metric-label">{lbl_}</div>'
                    f'</div>',
                    unsafe_allow_html=True)

        st.markdown("---")

        # ── SL level cards ───────────────────────────
        st.markdown('<div class="section-title">🎯 Stop Loss Levels</div>', unsafe_allow_html=True)
        col_l, col_r = st.columns(2)

        if direction in ("LONG", "BOTH"):
            struct_note = "📐 Structure-based" if calc["use_struct_long"] else "📐 ATR-based"
            with col_l:
                st.markdown(f"""
                <div class="sl-card sl-card-long">
                  <div style="font-size:15px;font-weight:700;color:#00ff66;margin-bottom:14px;">🔼 LONG Setup</div>
                  {level_row("Entry (current price)", f"{calc['price']:.5f}")}
                  {level_row("Nearest swing low", f"{calc['struct_sl']:.5f}" if calc['struct_sl'] else "—", "structural anchor")}
                  {level_row(f"Stop Loss  ({calc['sl_long_pips']:.0f} pips)", f"{calc['sl_long']:.5f}", struct_note + " + buffer")}
                  {level_row(f"TP1  ({calc['tp1_pips']:.0f} pips)", f"{calc['tp1_long']:.5f}", f"R:R {calc['rr_tp1_long']:.1f}:1")}
                  {level_row(f"TP2  ({calc['tp2_pips']:.0f} pips)", f"{calc['tp2_long']:.5f}", f"R:R {calc['rr_tp2_long']:.1f}:1")}
                  {level_row("Position size", f"{calc['lot_long']:.2f} lots", f"${calc['risk_amount']:.2f} risk at {risk_pct}%")}
                </div>
                """, unsafe_allow_html=True)

        if direction in ("SHORT", "BOTH"):
            struct_note = "📐 Structure-based" if calc["use_struct_short"] else "📐 ATR-based"
            with col_r if direction == "BOTH" else col_l:
                st.markdown(f"""
                <div class="sl-card sl-card-short">
                  <div style="font-size:15px;font-weight:700;color:#ff3344;margin-bottom:14px;">🔽 SHORT Setup</div>
                  {level_row("Entry (current price)", f"{calc['price']:.5f}")}
                  {level_row("Nearest swing high", f"{calc['struct_sh']:.5f}" if calc['struct_sh'] else "—", "structural anchor")}
                  {level_row(f"Stop Loss  ({calc['sl_short_pips']:.0f} pips)", f"{calc['sl_short']:.5f}", struct_note + " + buffer")}
                  {level_row(f"TP1  ({calc['tp1_pips']:.0f} pips)", f"{calc['tp1_short']:.5f}", f"R:R {calc['rr_tp1_short']:.1f}:1")}
                  {level_row(f"TP2  ({calc['tp2_pips']:.0f} pips)", f"{calc['tp2_short']:.5f}", f"R:R {calc['rr_tp2_short']:.1f}:1")}
                  {level_row("Position size", f"{calc['lot_short']:.2f} lots", f"${calc['risk_amount']:.2f} risk at {risk_pct}%")}
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # ── Chart ────────────────────────────────────
        st.markdown('<div class="section-title">📈 Daily Chart — Structure · SL · TP Levels</div>',
                    unsafe_allow_html=True)
        fig = build_chart(df, selected_pair, calc, struct, direction, show_n=show_n)
        st.plotly_chart(fig, width="stretch")

        st.markdown("---")

        # ── ATR history table ────────────────────────
        with st.expander("📋 ATR History — Last 15 Sessions"):
            atr_hist = calc["atr_series"].tail(15).copy()[::-1]
            atr_df   = pd.DataFrame({
                "Date":               atr_hist.index.strftime("%Y-%m-%d"),
                f"ATR({int(atr_period)})": atr_hist.values.round(5),
                "SL Dist (price)":    (atr_hist.values * float(atr_mult)).round(5),
                "SL Pips":            (atr_hist.values * float(atr_mult) / pip_size).round(1),
                "TP1 Pips":           (atr_hist.values * float(atr_mult) / pip_size * 2).round(1),
                "TP2 Pips":           (atr_hist.values * float(atr_mult) / pip_size * 3).round(1),
            })
            st.dataframe(atr_df, width="stretch", hide_index=True)

        st.markdown("---")

        # ── Explainer ────────────────────────────────
        st.markdown('<div class="section-title">📖 Stop Placement Logic</div>', unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"""
            <div class="explainer">
            <b style="color:#00ff66;">🔼 LONG — SL below structure</b><br><br>
            <b>Step 1 — Identify swing low</b><br>
            Find the most recent confirmed swing low on the daily chart.
            This is the structural level the market has already rejected — it acts as natural support.<br><br>
            <b>Step 2 — Add ATR buffer</b><br>
            Place the SL <em>below</em> the swing low by 20% of ATR({int(atr_period)}).
            This prevents a wick through the level from stopping you out prematurely.<br><br>
            <b>Step 3 — Verify R:R</b><br>
            TP1 = 2× SL distance. TP2 = 3× SL distance.
            If R:R to TP1 is below 2:1, the zone is too far — skip the trade.
            </div>
            """, unsafe_allow_html=True)
        with col_b:
            st.markdown(f"""
            <div class="explainer">
            <b style="color:#ff3344;">🔽 SHORT — SL above structure</b><br><br>
            <b>Step 1 — Identify swing high</b><br>
            Find the most recent confirmed swing high on the daily chart.
            This is where the market rejected higher prices — it acts as natural resistance.<br><br>
            <b>Step 2 — Add ATR buffer</b><br>
            Place the SL <em>above</em> the swing high by 20% of ATR({int(atr_period)}).
            Gives the trade room to breathe without being clipped by a false breakout wick.<br><br>
            <b>Step 3 — Verify R:R</b><br>
            TP1 = 2× SL distance. TP2 = 3× SL distance.
            A valid short trade needs TP1 R:R ≥ 2:1 before entry.
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="formula-box">
        ATR SL distance &nbsp;= &nbsp;ATR({int(atr_period)}) &nbsp;×&nbsp; {atr_mult} &nbsp;= &nbsp;
        {calc['atr14']:.5f} × {atr_mult} &nbsp;= &nbsp;
        <b style="color:#ff3344;">{calc['atr14'] * float(atr_mult):.5f} &nbsp; ({calc['sl_pips']:.1f} pips)</b>
        <br>
        SL LONG  &nbsp;= &nbsp;Entry &nbsp;−&nbsp; ATR SL distance
        &nbsp;=&nbsp; {calc['price']:.5f} − {calc['atr14'] * float(atr_mult):.5f}
        &nbsp;=&nbsp; <b style="color:#ff3344;">{calc['sl_long']:.5f}</b>
        <br>
        SL SHORT &nbsp;= &nbsp;Entry &nbsp;+&nbsp; ATR SL distance
        &nbsp;=&nbsp; {calc['price']:.5f} + {calc['atr14'] * float(atr_mult):.5f}
        &nbsp;=&nbsp; <b style="color:#ff3344;">{calc['sl_short']:.5f}</b>
        <br>
        TP1 &nbsp;= &nbsp;SL distance × 2.0 &nbsp;= &nbsp;
        <b style="color:#00ff66;">{calc['tp1_pips']:.1f} pips</b> &nbsp;&nbsp;
        TP2 &nbsp;= &nbsp;SL distance × 3.0 &nbsp;= &nbsp;
        <b style="color:#56d364;">{calc['tp2_pips']:.1f} pips</b>
        <br>
        Lot size &nbsp;= &nbsp;Risk $ ÷ (SL pips × pip value)
        &nbsp;= &nbsp;${calc['risk_amount']:.2f} ÷ ({calc['sl_pips']:.1f} × {pip_val})
        &nbsp;= &nbsp;<b style="color:#00ff41;">{calc['lot_long']:.2f} lots</b>
        </div>
        <div style="font-size:11px;color:#555555;margin-top:6px;">
          ⚠️ If structure-based SL is more than 3× ATR away from price the calculator falls back to
          the pure ATR-based distance. Adjust the pivot lookback in the sidebar if needed.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="text-align:center;color:#555555;font-size:11px;margin-top:32px;
                    padding-top:16px;border-top:1px solid #2a2a2a;">
          🛡️ Stop Below/Above Structure · Check #14 · ATR-based SL beyond swing structure · For educational purposes only
        </div>
        """, unsafe_allow_html=True)