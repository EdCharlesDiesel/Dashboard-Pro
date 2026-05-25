import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime

# ── Page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Weekly Swing Alignment",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS — matches main app dark theme ─────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html,body,[class*="css"]{ font-family:'Inter',sans-serif; }
    .stApp{ background:#0d1117; }
    section[data-testid="stSidebar"]{ background:#161b22!important; border-right:1px solid #21262d; }

    .card{ background:#161b22; border:1px solid #21262d; border-radius:12px;
           padding:20px; margin-bottom:16px; }
    .card-header{ font-size:13px; font-weight:600; letter-spacing:.08em;
                  text-transform:uppercase; color:#8b949e; margin-bottom:14px; }
    .metric-box{ background:#0d1117; border:1px solid #21262d; border-radius:8px;
                 padding:14px; text-align:center; }
    .metric-value{ font-size:22px; font-weight:700; color:#c9d1d9; }
    .metric-label{ font-size:11px; color:#8b949e; margin-top:2px; font-weight:500;
                   letter-spacing:.04em; text-transform:uppercase; }
    .section-title{ font-size:16px; font-weight:700; color:#e6edf3;
                    margin:24px 0 12px 0; padding-left:4px; border-left:3px solid #388bfd; }
    .prog-track{ background:#21262d; border-radius:8px; height:10px;
                 margin:6px 0 2px 0; overflow:hidden; }
    #MainMenu,footer,header{ visibility:hidden; }
    [data-testid="stSidebarCollapsedControl"]{visibility:visible !important;}
    .block-container{ padding-top:1.5rem; max-width:1380px; }

    /* Verdict banners */
    .verdict-aligned  { background:linear-gradient(135deg,#0d3a1f,#0d5e32);
                        border:2px solid #238636; border-radius:14px;
                        padding:22px 28px; text-align:center; margin-bottom:18px; }
    .verdict-conflict { background:linear-gradient(135deg,#3a0d0d,#5e1414);
                        border:2px solid #8b2d2d; border-radius:14px;
                        padding:22px 28px; text-align:center; margin-bottom:18px; }
    .verdict-neutral  { background:linear-gradient(135deg,#1c1c0d,#2e2a0d);
                        border:2px solid #9e6a03; border-radius:14px;
                        padding:22px 28px; text-align:center; margin-bottom:18px; }
    .verdict-unclear  { background:linear-gradient(135deg,#161b22,#1c2128);
                        border:1px solid #30363d; border-radius:14px;
                        padding:22px 28px; text-align:center; margin-bottom:18px; }

    /* Swing structure cards */
    .swing-card{ background:#0d1117; border:1px solid #21262d;
                 border-radius:10px; padding:16px 20px; margin-bottom:10px; }
    .swing-card-bull{ border-left:4px solid #3fb950; }
    .swing-card-bear{ border-left:4px solid #f85149; }
    .swing-card-neutral{ border-left:4px solid #8b949e; }

    /* HH/HL LL/LH tags */
    .hh-tag{ display:inline-block; background:#0d4a2f; color:#3fb950;
             border:1px solid #238636; border-radius:4px;
             padding:2px 8px; font-size:11px; font-weight:700; margin:2px; }
    .hl-tag{ display:inline-block; background:#0a3522; color:#56d364;
             border:1px solid #2ea043; border-radius:4px;
             padding:2px 8px; font-size:11px; font-weight:700; margin:2px; }
    .ll-tag{ display:inline-block; background:#4a0d0d; color:#f85149;
             border:1px solid #8b2d2d; border-radius:4px;
             padding:2px 8px; font-size:11px; font-weight:700; margin:2px; }
    .lh-tag{ display:inline-block; background:#3a0d0d; color:#ff7b72;
             border:1px solid #6e1414; border-radius:4px;
             padding:2px 8px; font-size:11px; font-weight:700; margin:2px; }

    /* Alignment row */
    .align-row{ display:flex; align-items:center; gap:14px; padding:12px 16px;
                background:#0d1117; border:1px solid #21262d;
                border-radius:8px; margin-bottom:8px; font-size:13px; }
    .align-icon{ font-size:20px; flex-shrink:0; }
    .align-label{ color:#8b949e; min-width:160px; }
    .align-value{ font-weight:600; color:#c9d1d9; }
    .align-badge-ok  { margin-left:auto; background:#0d4a2f; color:#3fb950;
                       border:1px solid #238636; border-radius:20px;
                       padding:3px 12px; font-size:11px; font-weight:700; }
    .align-badge-fail{ margin-left:auto; background:#4a0d0d; color:#f85149;
                       border:1px solid #8b2d2d; border-radius:20px;
                       padding:3px 12px; font-size:11px; font-weight:700; }
    .align-badge-warn{ margin-left:auto; background:#2a1f00; color:#e3b341;
                       border:1px solid #9e6a03; border-radius:20px;
                       padding:3px 12px; font-size:11px; font-weight:700; }

    .explainer{ background:#0d1117; border:1px solid #1e3a5f;
                border-left:3px solid #388bfd; border-radius:8px;
                padding:14px 18px; font-size:13px; color:#8b949e; line-height:1.7; }
</style>
""", unsafe_allow_html=True)


# ── Instrument registry ────────────────────────────────────────────
INSTRUMENTS = {
    "EUR/USD":    {"ticker": "EURUSD=X", "pip_size": 0.0001},
    "GBP/USD":    {"ticker": "GBPUSD=X", "pip_size": 0.0001},
    "AUD/USD":    {"ticker": "AUDUSD=X", "pip_size": 0.0001},
    "NZD/USD":    {"ticker": "NZDUSD=X", "pip_size": 0.0001},
    "USD/JPY":    {"ticker": "USDJPY=X", "pip_size": 0.01},
    "USD/CHF":    {"ticker": "USDCHF=X", "pip_size": 0.0001},
    "USD/CAD":    {"ticker": "USDCAD=X", "pip_size": 0.0001},
    "EUR/GBP":    {"ticker": "EURGBP=X", "pip_size": 0.0001},
    "EUR/JPY":    {"ticker": "EURJPY=X", "pip_size": 0.01},
    "GBP/JPY":    {"ticker": "GBPJPY=X", "pip_size": 0.01},
    "AUD/JPY":    {"ticker": "AUDJPY=X", "pip_size": 0.01},
    "EUR/AUD":    {"ticker": "EURAUD=X", "pip_size": 0.0001},
    "GBP/AUD":    {"ticker": "GBPAUD=X", "pip_size": 0.0001},
    "EUR/CAD":    {"ticker": "EURCAD=X", "pip_size": 0.0001},
    "GBP/CAD":    {"ticker": "GBPCAD=X", "pip_size": 0.0001},
    "USD/ZAR":    {"ticker": "USDZAR=X", "pip_size": 0.0001},
    "EUR/ZAR":    {"ticker": "EURZAR=X", "pip_size": 0.0001},
    "GBP/ZAR":    {"ticker": "GBPZAR=X", "pip_size": 0.0001},
    "🥇 Gold":    {"ticker": "GC=F",     "pip_size": 0.10},
    "🥈 Silver":  {"ticker": "SI=F",     "pip_size": 0.01},
    "🪙 Platinum":{"ticker": "PL=F",     "pip_size": 0.10},
}


# ══════════════════════════════════════════════════════════════════
# DATA FETCHING
# ══════════════════════════════════════════════════════════════════

@st.cache_data(ttl=600, show_spinner=False)
def fetch_weekly(ticker: str) -> pd.DataFrame:
    try:
        df = yf.download(ticker, interval="1wk", period="2y",
                         progress=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def fetch_daily(ticker: str) -> pd.DataFrame:
    try:
        df = yf.download(ticker, interval="1d", period="6mo",
                         progress=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    except Exception:
        return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════
# SWING ANALYSIS ENGINE
# ══════════════════════════════════════════════════════════════════

def calc_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def find_swing_pivots(df: pd.DataFrame, left: int = 2, right: int = 2) -> pd.DataFrame:
    """Identify swing highs and lows using a left/right lookback window."""
    df = df.copy()
    n = len(df)
    H = df["High"].values
    L = df["Low"].values

    swing_high = np.zeros(n, bool)
    swing_low = np.zeros(n, bool)

    for i in range(left, n - right):
        if all(H[i] >= H[i - j] for j in range(1, left + 1)) and \
                all(H[i] >= H[i + j] for j in range(1, right + 1)):
            swing_high[i] = True
        if all(L[i] <= L[i - j] for j in range(1, left + 1)) and \
                all(L[i] <= L[i + j] for j in range(1, right + 1)):
            swing_low[i] = True

    df["swing_high"] = swing_high
    df["swing_low"] = swing_low
    return df


def classify_weekly_swing(df_w: pd.DataFrame, pivot_lb: int) -> dict:
    """
    Classify the weekly swing structure as BULLISH, BEARISH, or RANGING
    using the last N confirmed swing highs and lows.
    """
    df_w = find_swing_pivots(df_w, left=pivot_lb, right=pivot_lb)

    sh_idx = df_w.index[df_w["swing_high"]].tolist()
    sl_idx = df_w.index[df_w["swing_low"]].tolist()

    sh_prices = [float(df_w.loc[i, "High"]) for i in sh_idx]
    sl_prices = [float(df_w.loc[i, "Low"]) for i in sl_idx]

    if len(sh_prices) < 2 or len(sl_prices) < 2:
        return {
            "bias": "UNCLEAR", "label": "⚠️ Insufficient pivots",
            "sh": sh_prices, "sl": sl_prices,
            "sh_idx": sh_idx, "sl_idx": sl_idx,
            "hh": False, "hl": False, "ll": False, "lh": False,
            "last_sh": None, "last_sl": None,
            "df": df_w,
        }

    last_sh = sh_prices[-1]
    prev_sh = sh_prices[-2]
    last_sl = sl_prices[-1]
    prev_sl = sl_prices[-2]

    hh = last_sh > prev_sh
    hl = last_sl > prev_sl
    ll = last_sl < prev_sl
    lh = last_sh < prev_sh

    if hh and hl:
        bias = "BULLISH"
        label = "📈 Bullish Swing Structure"
    elif ll and lh:
        bias = "BEARISH"
        label = "📉 Bearish Swing Structure"
    elif hh and ll:
        bias = "RANGING"
        label = "↔️ Ranging / Expanding"
    elif hl and lh:
        bias = "RANGING"
        label = "↔️ Ranging / Contracting"
    else:
        bias = "UNCLEAR"
        label = "⚠️ Mixed — No clear structure"

    return {
        "bias": bias, "label": label,
        "sh": sh_prices, "sl": sl_prices,
        "sh_idx": sh_idx, "sl_idx": sl_idx,
        "hh": hh, "hl": hl, "ll": ll, "lh": lh,
        "last_sh": last_sh, "prev_sh": prev_sh,
        "last_sl": last_sl, "prev_sl": prev_sl,
        "df": df_w,
    }


def classify_daily_trend(df_d: pd.DataFrame, fast: int, slow: int) -> dict:
    """
    Determine daily trend via EMA crossover and recent candle structure.
    """
    df_d = df_d.copy()
    df_d["ema_f"] = calc_ema(df_d["Close"], fast)
    df_d["ema_s"] = calc_ema(df_d["Close"], slow)

    last = df_d.iloc[-1]
    price = float(last["Close"])
    ema_f_now = float(last["ema_f"])
    ema_s_now = float(last["ema_s"])

    df_d = find_swing_pivots(df_d, left=3, right=3)
    sh_d = [float(df_d.loc[i, "High"]) for i in df_d.index[df_d["swing_high"]]]
    sl_d = [float(df_d.loc[i, "Low"]) for i in df_d.index[df_d["swing_low"]]]

    daily_hh = len(sh_d) >= 2 and sh_d[-1] > sh_d[-2]
    daily_hl = len(sl_d) >= 2 and sl_d[-1] > sl_d[-2]
    daily_ll = len(sl_d) >= 2 and sl_d[-1] < sl_d[-2]
    daily_lh = len(sh_d) >= 2 and sh_d[-1] < sh_d[-2]

    ema_bull = ema_f_now > ema_s_now
    price_above = price > ema_s_now

    if ema_bull and price_above:
        trend = "BULLISH"
    elif not ema_bull and not price_above:
        trend = "BEARISH"
    else:
        trend = "MIXED"

    return {
        "trend": trend,
        "price": price,
        "ema_f": ema_f_now,
        "ema_s": ema_s_now,
        "ema_bull": ema_bull,
        "daily_hh": daily_hh,
        "daily_hl": daily_hl,
        "daily_ll": daily_ll,
        "daily_lh": daily_lh,
        "df": df_d,
    }


def determine_alignment(weekly_bias: str, daily_trend: str) -> dict:
    """
    Cross-check weekly swing bias with daily trend direction.
    Returns overall alignment verdict.
    """
    aligned_combos = {("BULLISH", "BULLISH"), ("BEARISH", "BEARISH")}
    conflict_combos = {("BULLISH", "BEARISH"), ("BEARISH", "BULLISH")}

    pair = (weekly_bias, daily_trend)

    if pair in aligned_combos:
        return {
            "status": "ALIGNED",
            "verdict": "✅ Aligned",
            "grade": "aligned",
            "desc": "Weekly swing and daily trend point in the same direction — "
                    "trade in the direction of both timeframes for maximum confluence.",
            "color": "#3fb950",
        }
    elif pair in conflict_combos:
        return {
            "status": "CONFLICT",
            "verdict": "❌ Conflict",
            "grade": "conflict",
            "desc": "Weekly swing and daily trend are in opposite directions — "
                    "counter-trend setups carry elevated risk; avoid or size down significantly.",
            "color": "#f85149",
        }
    elif weekly_bias == "RANGING":
        return {
            "status": "RANGING",
            "verdict": "↔️ Weekly Ranging",
            "grade": "neutral",
            "desc": "Weekly structure is ranging or transitioning — "
                    "no clear swing bias. Wait for a confirmed weekly breakout before trading.",
            "color": "#e3b341",
        }
    else:
        return {
            "status": "UNCLEAR",
            "verdict": "⚠️ Unclear",
            "grade": "unclear",
            "desc": "Not enough pivot data to determine weekly swing structure. "
                    "Try increasing the lookback period or reducing the pivot sensitivity.",
            "color": "#8b949e",
        }


# ══════════════════════════════════════════════════════════════════
# CHART BUILDERS
# ══════════════════════════════════════════════════════════════════

def build_weekly_chart(sw: dict, pair: str, show_n: int = 52) -> go.Figure:
    df = sw["df"].tail(show_n)
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.78, 0.22], vertical_spacing=0.03,
    )

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        increasing_line_color="#3fb950",
        decreasing_line_color="#f85149",
        increasing_fillcolor="rgba(63, 185, 80, 0.20)",
        decreasing_fillcolor="rgba(248, 81, 73, 0.20)",
        name="Weekly",
        showlegend=False,
    ), row=1, col=1)

    vcol = ["rgba(63, 185, 80, 0.33)" if c >= o else "rgba(248, 81, 73, 0.33)"
            for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"],
        marker_color=vcol, name="Vol", showlegend=False,
    ), row=2, col=1)

    sh_df = df[df["swing_high"]]
    if not sh_df.empty:
        fig.add_trace(go.Scatter(
            x=sh_df.index,
            y=sh_df["High"] * 1.0008,
            mode="markers+text",
            marker=dict(symbol="triangle-down", size=11,
                        color="#f85149", line=dict(color="#0d1117", width=1)),
            text=["SH"] * len(sh_df),
            textposition="top center",
            textfont=dict(size=8, color="#f85149"),
            name="Swing High",
        ), row=1, col=1)

    sl_df = df[df["swing_low"]]
    if not sl_df.empty:
        fig.add_trace(go.Scatter(
            x=sl_df.index,
            y=sl_df["Low"] * 0.9992,
            mode="markers+text",
            marker=dict(symbol="triangle-up", size=11,
                        color="#3fb950", line=dict(color="#0d1117", width=1)),
            text=["SL"] * len(sl_df),
            textposition="bottom center",
            textfont=dict(size=8, color="#3fb950"),
            name="Swing Low",
        ), row=1, col=1)

    if len(sw["sh_idx"]) >= 2:
        sh_plot = [i for i in sw["sh_idx"] if i in df.index]
        if len(sh_plot) >= 2:
            fig.add_trace(go.Scatter(
                x=sh_plot,
                y=[float(df.loc[i, "High"]) for i in sh_plot],
                mode="lines",
                line=dict(color="rgba(248, 81, 73, 0.40)", width=1.5, dash="dot"),
                name="SH Trend", showlegend=False,
            ), row=1, col=1)

    if len(sw["sl_idx"]) >= 2:
        sl_plot = [i for i in sw["sl_idx"] if i in df.index]
        if len(sl_plot) >= 2:
            fig.add_trace(go.Scatter(
                x=sl_plot,
                y=[float(df.loc[i, "Low"]) for i in sl_plot],
                mode="lines",
                line=dict(color="rgba(63, 185, 80, 0.40)", width=1.5, dash="dot"),
                name="SL Trend", showlegend=False,
            ), row=1, col=1)

    fig.update_layout(
        height=500,
        paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
        font=dict(family="Inter, sans-serif", size=11, color="#8b949e"),
        xaxis_rangeslider_visible=False,
        title=dict(text=f"<b>{pair}</b> — Weekly Swing Structure",
                   font=dict(color="#e6edf3", size=14), x=0.01),
        legend=dict(bgcolor="rgba(13, 17, 23, 0.60)", bordercolor="#21262d", borderwidth=1,
                    font=dict(size=10), orientation="h",
                    yanchor="bottom", y=1.01, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    fig.update_yaxes(gridcolor="#21262d", zeroline=False)
    fig.update_xaxes(gridcolor="#21262d", showgrid=False)
    return fig


def build_daily_chart(dt: dict, pair: str, fast: int, slow: int,
                      show_n: int = 90) -> go.Figure:
    df = dt["df"].tail(show_n)
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.78, 0.22], vertical_spacing=0.03,
    )

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        increasing_line_color="#3fb950",
        decreasing_line_color="#f85149",
        increasing_fillcolor="rgba(63, 185, 80, 0.20)",
        decreasing_fillcolor="rgba(248, 81, 73, 0.20)",
        name="Daily",
        showlegend=False,
    ), row=1, col=1)

    vcol = ["rgba(63, 185, 80, 0.33)" if c >= o else "rgba(248, 81, 73, 0.33)"
            for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"],
        marker_color=vcol, name="Vol", showlegend=False,
    ), row=2, col=1)

    if "ema_f" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["ema_f"],
            mode="lines", line=dict(color="#f59e0b", width=2),
            name=f"EMA {fast}",
        ), row=1, col=1)
    if "ema_s" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["ema_s"],
            mode="lines", line=dict(color="#a29bfe", width=2, dash="dash"),
            name=f"EMA {slow}",
        ), row=1, col=1)

    sh_d = df[df["swing_high"]]
    sl_d = df[df["swing_low"]]
    if not sh_d.empty:
        fig.add_trace(go.Scatter(
            x=sh_d.index, y=sh_d["High"],
            mode="markers",
            marker=dict(symbol="x", size=9, color="#f85149",
                        line=dict(width=2, color="#f85149")),
            name="Daily SH",
        ), row=1, col=1)
    if not sl_d.empty:
        fig.add_trace(go.Scatter(
            x=sl_d.index, y=sl_d["Low"],
            mode="markers",
            marker=dict(symbol="x", size=9, color="#3fb950",
                        line=dict(width=2, color="#3fb950")),
            name="Daily SL",
        ), row=1, col=1)

    fig.add_hline(
        y=dt["price"], line_dash="solid",
        line_color="#ffffff", line_width=1, opacity=0.7,
        annotation_text=f"  {dt['price']:.5f}",
        annotation_position="right",
        annotation_font=dict(size=10, color="#ffffff"),
        row=1, col=1,
    )

    fig.update_layout(
        height=500,
        paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
        font=dict(family="Inter, sans-serif", size=11, color="#8b949e"),
        xaxis_rangeslider_visible=False,
        title=dict(text=f"<b>{pair}</b> — Daily Trend Confirmation",
                   font=dict(color="#e6edf3", size=14), x=0.01),
        legend=dict(bgcolor="rgba(13, 17, 23, 0.60)", bordercolor="#21262d", borderwidth=1,
                    font=dict(size=10), orientation="h",
                    yanchor="bottom", y=1.01, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    fig.update_yaxes(gridcolor="#21262d", zeroline=False)
    fig.update_xaxes(gridcolor="#21262d", showgrid=False)
    return fig


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 🔄 Weekly Swing Alignment")

    inst_keys = list(INSTRUMENTS.keys())
    default_inst = st.session_state.get("selected_instrument", "EUR/USD")
    if default_inst not in inst_keys:
        default_inst = inst_keys[0]

    selected_pair = st.selectbox("Instrument", inst_keys,
                                 index=inst_keys.index(default_inst))

    st.divider()
    st.page_link("daily-trading-checklist.py", label="00. Checklist", icon="📋")
    st.page_link("pages/macro-bias.py", label="01. Macro Bias", icon="🌐")
    st.page_link("pages/news-filter.py", label="02. News Filter", icon="📰")
    st.page_link("pages/correlations.py", label="03. Correlations", icon="🔗")
    st.page_link("pages/atr-volatility.py", label="04. ATR Volatility", icon="📊")
    st.page_link("pages/weekly-ema.py", label="05. Weekly EMA", icon="📉")
    st.page_link("pages/weekly-rsi.py", label="06. Weekly RSI", icon="📡")
    st.page_link("pages/weekly-swing.py", label="07. Weekly Swing", icon="🔄")
    st.page_link("pages/daily-trend.py", label="08. Daily Trend", icon="📈")
    st.page_link("pages/daily-macd.py", label="09. Daily MACD", icon="📊")
    st.page_link("pages/4H-confluence-zone.py", label="10. 4H Confluence Zone", icon="🎯")
    st.page_link("pages/confluence-checker.py", label="11. 2/3 Confluence Check", icon="🔀")
    st.page_link("pages/15m-rejection.py", label="12. 15M Rejection", icon="🕯️")
    st.page_link("pages/15m-entry-signal.py", label="13. 15M Entry Signal", icon="⚡")
    st.page_link("pages/stop-structure.py", label="14. Stop Structure", icon="🛡️")
    st.page_link("pages/rr-calculator.py", label="15. R:R Calculator", icon="⚖️")
    st.divider()
    st.markdown("**📅 Weekly Settings**")
    pivot_lb = st.slider("Swing pivot sensitivity (weeks)", 1, 5, 2,
                         help="Left/right candles needed to confirm a swing pivot. "
                              "Higher = fewer but stronger pivots.")
    weekly_n = st.slider("Weekly candles shown", 26, 104, 52, step=13)

    st.divider()
    st.markdown("**📆 Daily Settings**")
    daily_fast = st.number_input("Daily EMA Fast", value=20, min_value=5, max_value=100)
    daily_slow = st.number_input("Daily EMA Slow", value=50, min_value=10, max_value=200)
    daily_n = st.slider("Daily candles shown", 30, 180, 90, step=15)

    st.divider()
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.caption(f"🕐 {datetime.now().strftime('%H:%M:%S')} local")

    st.divider()
    st.markdown("""
    <div style='font-size:11px;color:#484f58;line-height:1.9;'>
    <b style='color:#8b949e;'>Bullish swing =</b><br>
    &nbsp;&nbsp;Higher Highs (HH)<br>
    &nbsp;&nbsp;+ Higher Lows (HL)<br><br>
    <b style='color:#8b949e;'>Bearish swing =</b><br>
    &nbsp;&nbsp;Lower Lows (LL)<br>
    &nbsp;&nbsp;+ Lower Highs (LH)<br><br>
    <b style='color:#8b949e;'>✅ Aligned =</b><br>
    &nbsp;&nbsp;Weekly swing matches<br>
    &nbsp;&nbsp;the daily EMA trend
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# MAIN PAGE
# ══════════════════════════════════════════════════════════════════

ticker = INSTRUMENTS[selected_pair]["ticker"]

st.markdown(f"""
<div style="background:linear-gradient(135deg,#0d1117 0%,#161b22 50%,#0d1117 100%);
            border:1px solid #21262d; border-radius:16px; padding:24px 28px;
            margin-bottom:20px; position:relative; overflow:hidden;">
  <div style="font-size:24px; font-weight:700; color:#e6edf3;">
    🔄 Weekly Swing Alignment
  </div>
  <div style="color:#8b949e; font-size:13px; margin-top:4px;">
    Weekly swing structure vs daily trend confirmation · {selected_pair}
  </div>
  <div style="font-size:12px; color:#388bfd; margin-top:6px;">
    Check #7 — Weekly Swing ✅ Aligned · {datetime.now().strftime('%A %d %B %Y  |  %H:%M')}
  </div>
</div>
""", unsafe_allow_html=True)

tab_scan, tab_detail = st.tabs(["📊 All-Pairs Scanner", "🔍 Pair Detail"])


# ══════════════════════════════════════════════════════════════════
# TAB 1 — ALL-PAIRS SCANNER
# ══════════════════════════════════════════════════════════════════
with tab_scan:
    scan_results = {}
    prog = st.progress(0, text="📡 Scanning all pairs…")
    for idx, (pair, cfg) in enumerate(INSTRUMENTS.items()):
        prog.progress((idx + 1) / len(INSTRUMENTS),
                      text=f"📡 {pair} ({idx+1}/{len(INSTRUMENTS)})…")
        try:
            df_w_s = fetch_weekly(cfg["ticker"])
            df_d_s = fetch_daily(cfg["ticker"])
            if df_w_s.empty or len(df_w_s) < 10 or df_d_s.empty or len(df_d_s) < 20:
                scan_results[pair] = None
                continue
            sw_s  = classify_weekly_swing(df_w_s, pivot_lb=int(pivot_lb))
            dt_s  = classify_daily_trend(df_d_s, fast=int(daily_fast), slow=int(daily_slow))
            aln_s = determine_alignment(sw_s["bias"], dt_s["trend"])
            scan_results[pair] = {"sw": sw_s, "dt": dt_s, "aln": aln_s}
        except Exception:
            scan_results[pair] = None
    prog.empty()

    loaded_s = {p: v for p, v in scan_results.items() if v is not None}
    aligned  = [p for p, v in loaded_s.items() if v["aln"]["status"] == "ALIGNED"]
    conflict = [p for p, v in loaded_s.items() if v["aln"]["status"] == "CONFLICT"]
    ranging  = [p for p, v in loaded_s.items() if v["aln"]["status"] == "RANGING"]
    unclear  = [p for p, v in loaded_s.items() if v["aln"]["status"] == "UNCLEAR"]
    no_data  = [p for p in INSTRUMENTS if p not in loaded_s]

    s1, s2, s3, s4, s5 = st.columns(5)
    for col_, val_, lbl_, c_ in [
        (s1, len(aligned),  "✅ Aligned",  "#3fb950"),
        (s2, len(conflict), "❌ Conflict", "#f85149"),
        (s3, len(ranging),  "↔️ Ranging",  "#e3b341"),
        (s4, len(unclear),  "⚠️ Unclear",  "#8b949e"),
        (s5, len(no_data),  "⛔ No Data",  "#484f58"),
    ]:
        with col_:
            st.markdown(
                f'<div class="metric-box">'
                f'<div class="metric-value" style="color:{c_};font-size:20px;">{val_}</div>'
                f'<div class="metric-label">{lbl_}</div></div>',
                unsafe_allow_html=True)

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    SCAN_ORDER = {"ALIGNED": 0, "CONFLICT": 1, "RANGING": 2, "UNCLEAR": 3}
    sorted_scan = sorted(
        INSTRUMENTS.keys(),
        key=lambda p: (SCAN_ORDER.get(
            loaded_s[p]["aln"]["status"] if p in loaded_s else "UNCLEAR", 9), p)
    )

    BORDER_COL = {"ALIGNED": "#3fb950", "CONFLICT": "#f85149",
                  "RANGING": "#e3b341", "UNCLEAR":  "#484f58"}
    BADGE_STYLE = {
        "ALIGNED":  ("background:#0d2f1a;color:#3fb950;border:1px solid #238636;", "✅ Aligned"),
        "CONFLICT": ("background:#2f0d0d;color:#f85149;border:1px solid #8b2d2d;", "❌ Conflict"),
        "RANGING":  ("background:#2f1f0d;color:#e3b341;border:1px solid #9e6a03;", "↔️ Ranging"),
        "UNCLEAR":  ("background:#21262d;color:#8b949e;border:1px solid #30363d;", "⚠️ Unclear"),
    }

    def _tag(css_class, text):
        return (f'<span style="border-radius:3px;padding:1px 5px;font-size:9px;'
                f'font-weight:700;margin-right:2px;{css_class}">{text}</span>')

    HH_CSS = "background:#0d4a2f;color:#3fb950;border:1px solid #238636;"
    HL_CSS = "background:#0a3522;color:#56d364;border:1px solid #2ea043;"
    LL_CSS = "background:#4a0d0d;color:#f85149;border:1px solid #8b2d2d;"
    LH_CSS = "background:#3a0d0d;color:#ff7b72;border:1px solid #6e1414;"

    cards_html = '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:9px;">'
    for pair in sorted_scan:
        if pair not in loaded_s:
            cards_html += (
                f'<div style="border:1px solid #21262d;border-left:4px solid #30363d;'
                f'background:#161b22;border-radius:10px;padding:11px 14px;opacity:.5;">'
                f'<div style="font-size:13px;font-weight:700;color:#8b949e;">{pair}</div>'
                f'<div style="font-size:10px;color:#484f58;margin-top:4px;">No data</div></div>'
            )
            continue

        r   = loaded_s[pair]
        st_ = r["aln"]["status"]
        bdr = BORDER_COL.get(st_, "#30363d")
        bdg_css, bdg_txt = BADGE_STYLE.get(st_, BADGE_STYLE["UNCLEAR"])
        w_bias  = r["sw"]["bias"]
        d_trend = r["dt"]["trend"]
        w_col   = "#3fb950" if w_bias == "BULLISH" else "#f85149" if w_bias == "BEARISH" else "#e3b341"
        d_col   = "#3fb950" if d_trend == "BULLISH" else "#f85149" if d_trend == "BEARISH" else "#e3b341"
        tags    = ((_tag(HH_CSS, "HH") if r["sw"].get("hh") else "") +
                   (_tag(HL_CSS, "HL") if r["sw"].get("hl") else "") +
                   (_tag(LL_CSS, "LL") if r["sw"].get("ll") else "") +
                   (_tag(LH_CSS, "LH") if r["sw"].get("lh") else ""))
        outline = "outline:2px solid #388bfd;" if pair == selected_pair else ""

        cards_html += (
            f'<div style="border:1px solid #21262d;border-left:4px solid {bdr};'
            f'background:#161b22;border-radius:10px;padding:11px 14px;{outline}">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">'
            f'<span style="font-size:13px;font-weight:700;color:#e6edf3;">{pair}</span>'
            f'<span style="border-radius:5px;padding:2px 8px;font-size:10px;font-weight:700;{bdg_css}">{bdg_txt}</span>'
            f'</div>'
            f'<div style="display:flex;gap:10px;font-size:11px;margin-bottom:5px;">'
            f'<span style="color:#484f58;">W:</span>'
            f'<span style="color:{w_col};font-weight:600;">{w_bias}</span>'
            f'<span style="color:#484f58;">D:</span>'
            f'<span style="color:{d_col};font-weight:600;">{d_trend}</span>'
            f'</div>'
            f'<div>{tags}</div>'
            f'</div>'
        )
    cards_html += "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    with st.expander("📋 Full Scan Table"):
        tbl_rows = []
        for pair in INSTRUMENTS:
            r = loaded_s.get(pair)
            if r is None:
                tbl_rows.append({"Pair": pair, "W Bias": "—", "D Trend": "—",
                                 "HH": "—", "HL": "—", "LL": "—", "LH": "—",
                                 "Alignment": "No Data"})
            else:
                tbl_rows.append({
                    "Pair":      pair,
                    "W Bias":    r["sw"]["bias"],
                    "D Trend":   r["dt"]["trend"],
                    "HH":        "✅" if r["sw"].get("hh") else "❌",
                    "HL":        "✅" if r["sw"].get("hl") else "❌",
                    "LL":        "✅" if r["sw"].get("ll") else "❌",
                    "LH":        "✅" if r["sw"].get("lh") else "❌",
                    "Alignment": r["aln"]["verdict"],
                })
        st.dataframe(pd.DataFrame(tbl_rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════
# TAB 2 — PAIR DETAIL
# ══════════════════════════════════════════════════════════════════
with tab_detail:
    with st.spinner(f"Fetching weekly & daily data for {selected_pair}…"):
        df_w = fetch_weekly(ticker)
        df_d = fetch_daily(ticker)

    detail_ok = True
    if df_w.empty or len(df_w) < 10:
        st.error(f"⚠️ Could not load weekly data for **{selected_pair}**.")
        detail_ok = False
    elif df_d.empty or len(df_d) < 20:
        st.error(f"⚠️ Could not load daily data for **{selected_pair}**.")
        detail_ok = False

    if detail_ok:
        sw  = classify_weekly_swing(df_w, pivot_lb=int(pivot_lb))
        dt  = classify_daily_trend(df_d, fast=int(daily_fast), slow=int(daily_slow))
        aln = determine_alignment(sw["bias"], dt["trend"])

        GRADE_CSS = {
            "aligned":  "verdict-aligned",
            "conflict": "verdict-conflict",
            "neutral":  "verdict-neutral",
            "unclear":  "verdict-unclear",
        }
        vcls  = GRADE_CSS[aln["grade"]]
        color = aln["color"]

        w_arrow    = "📈" if sw["bias"] == "BULLISH" else "📉" if sw["bias"] == "BEARISH" else "↔️"
        d_arrow    = "📈" if dt["trend"] == "BULLISH" else "📉" if dt["trend"] == "BEARISH" else "〰️"
        match_icon = "✅" if aln["status"] == "ALIGNED" else "❌" if aln["status"] == "CONFLICT" else "⚠️"

        st.markdown(f"""
        <div class="{vcls}">
          <div style="font-size:26px; font-weight:800; color:{color};
                      letter-spacing:1px; margin-bottom:10px;">{aln['verdict']}</div>
          <div style="display:flex; justify-content:center; align-items:center;
                      gap:20px; margin:12px 0; flex-wrap:wrap;">
            <div style="background:rgba(13,17,23,.40); border:1px solid #30363d;
                        border-radius:10px; padding:12px 24px; text-align:center;">
              <div style="font-size:10px; letter-spacing:.1em; text-transform:uppercase;
                          color:#8b949e; margin-bottom:6px;">Weekly Swing</div>
              <div style="font-size:22px;">{w_arrow}</div>
              <div style="font-size:13px; font-weight:700;
                          color:{'#3fb950' if sw['bias']=='BULLISH' else '#f85149' if sw['bias']=='BEARISH' else '#e3b341'};">
                {sw["bias"]}
              </div>
            </div>
            <div style="font-size:28px;">{match_icon}</div>
            <div style="background:rgba(13,17,23,.40); border:1px solid #30363d;
                        border-radius:10px; padding:12px 24px; text-align:center;">
              <div style="font-size:10px; letter-spacing:.1em; text-transform:uppercase;
                          color:#8b949e; margin-bottom:6px;">Daily Trend</div>
              <div style="font-size:22px;">{d_arrow}</div>
              <div style="font-size:13px; font-weight:700;
                          color:{'#3fb950' if dt['trend']=='BULLISH' else '#f85149' if dt['trend']=='BEARISH' else '#e3b341'};">
                {dt["trend"]}
              </div>
            </div>
          </div>
          <div style="font-size:13px; color:#8b949e; margin-top:10px;">{aln['desc']}</div>
        </div>
        """, unsafe_allow_html=True)

        num_sh    = len(sw["sh"])
        num_sl    = len(sw["sl"])
        ema_cross = "✅ Bull" if dt["ema_bull"] else "❌ Bear"

        k1, k2, k3, k4, k5 = st.columns(5)
        for col, val, lbl, c in [
            (k1, sw["bias"], "Weekly Bias", color),
            (k2, dt["trend"], "Daily Trend",
             "#3fb950" if dt["trend"]=="BULLISH" else "#f85149" if dt["trend"]=="BEARISH" else "#e3b341"),
            (k3, str(num_sh), "Weekly Swing Highs", "#f85149"),
            (k4, str(num_sl), "Weekly Swing Lows",  "#3fb950"),
            (k5, ema_cross, f"EMA {int(daily_fast)}/{int(daily_slow)} Cross",
             "#3fb950" if dt["ema_bull"] else "#f85149"),
        ]:
            with col:
                st.markdown(
                    f'<div class="metric-box">'
                    f'<div class="metric-value" style="color:{c};font-size:17px;">{val}</div>'
                    f'<div class="metric-label">{lbl}</div></div>',
                    unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<div class="section-title">🧩 Alignment Breakdown</div>',
                    unsafe_allow_html=True)

        def align_row(icon, label, value, badge_text, badge_ok):
            badge_cls = ("align-badge-ok"   if badge_ok is True  else
                         "align-badge-fail" if badge_ok is False else
                         "align-badge-warn")
            st.markdown(f"""
            <div class="align-row">
              <div class="align-icon">{icon}</div>
              <div class="align-label">{label}</div>
              <div class="align-value">{value}</div>
              <span class="{badge_cls}">{badge_text}</span>
            </div>""", unsafe_allow_html=True)

        hh_tag = '<span class="hh-tag">HH</span>' if sw.get("hh") else ""
        hl_tag = '<span class="hl-tag">HL</span>' if sw.get("hl") else ""
        ll_tag = '<span class="ll-tag">LL</span>' if sw.get("ll") else ""
        lh_tag = '<span class="lh-tag">LH</span>' if sw.get("lh") else ""
        struct_tags = (hh_tag + hl_tag + ll_tag + lh_tag) or "—"

        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown('<div class="card"><div class="card-header">📅 Weekly Swing Structure</div>',
                        unsafe_allow_html=True)
            align_row("📊", "Swing structure", sw["label"],
                      "Bullish" if sw["bias"]=="BULLISH" else "Bearish" if sw["bias"]=="BEARISH" else "Ranging",
                      True if sw["bias"]=="BULLISH" else False if sw["bias"]=="BEARISH" else None)
            align_row("🔺", "Last swing high",
                      f'{sw["last_sh"]:.5f}' if sw.get("last_sh") else "—",
                      "HH ✅" if sw.get("hh") else "LH ❌", sw.get("hh", False))
            align_row("🔻", "Last swing low",
                      f'{sw["last_sl"]:.5f}' if sw.get("last_sl") else "—",
                      "HL ✅" if sw.get("hl") else "LL ❌", sw.get("hl", False))
            align_row("🔢", "Confirmed pivots", f'{num_sh} SH · {num_sl} SL',
                      "Sufficient" if num_sh >= 2 and num_sl >= 2 else "Need more",
                      num_sh >= 2 and num_sl >= 2)
            st.markdown(f'<div style="padding:10px 0 4px 0;font-size:12px;color:#8b949e;">'
                        f'Structure tags: &nbsp;{struct_tags}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_r:
            st.markdown('<div class="card"><div class="card-header">📆 Daily Trend Confirmation</div>',
                        unsafe_allow_html=True)
            align_row("〰️", "Daily trend", dt["trend"],
                      "Bullish" if dt["trend"]=="BULLISH" else "Bearish" if dt["trend"]=="BEARISH" else "Mixed",
                      True if dt["trend"]=="BULLISH" else False if dt["trend"]=="BEARISH" else None)
            align_row("📈", f"EMA {int(daily_fast)} vs EMA {int(daily_slow)}",
                      f'{dt["ema_f"]:.5f} vs {dt["ema_s"]:.5f}',
                      "Bull cross ✅" if dt["ema_bull"] else "Bear cross ❌", dt["ema_bull"])
            align_row("💵", "Price vs slow EMA",
                      f'{dt["price"]:.5f} {">" if dt["price"] > dt["ema_s"] else "<"} {dt["ema_s"]:.5f}',
                      "Above ✅" if dt["price"] > dt["ema_s"] else "Below ❌",
                      dt["price"] > dt["ema_s"])
            align_row("🔺", "Daily swing highs",
                      "HH ✅" if dt["daily_hh"] else "LH ❌",
                      "Higher Highs" if dt["daily_hh"] else "Lower Highs", dt["daily_hh"])
            align_row("🔻", "Daily swing lows",
                      "HL ✅" if dt["daily_hl"] else "LL ❌",
                      "Higher Lows" if dt["daily_hl"] else "Lower Lows", dt["daily_hl"])
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<div class="section-title">📊 Weekly & Daily Charts</div>',
                    unsafe_allow_html=True)

        tab_w, tab_d = st.tabs(["📅 Weekly Swing Chart", "📆 Daily Trend Chart"])
        with tab_w:
            fig_w = build_weekly_chart(sw, selected_pair, show_n=weekly_n)
            st.plotly_chart(fig_w, use_container_width=True)
        with tab_d:
            fig_d = build_daily_chart(dt, selected_pair,
                                      fast=int(daily_fast), slow=int(daily_slow),
                                      show_n=daily_n)
            st.plotly_chart(fig_d, use_container_width=True)

        st.markdown("---")

        with st.expander("📋 Weekly Pivot History"):
            rows = []
            df_tmp  = sw["df"]
            sh_list = [(i, float(df_tmp.loc[i, "High"])) for i in sw["sh_idx"]]
            sl_list = [(i, float(df_tmp.loc[i, "Low"]))  for i in sw["sl_idx"]]
            for idx, (ts, val) in enumerate(sh_list[-10:]):
                prev_val = sh_list[idx - 1][1] if idx > 0 else None
                tag = "HH ✅" if (prev_val and val > prev_val) else ("LH ❌" if prev_val else "—")
                rows.append({"Date": ts.strftime("%Y-%m-%d"), "Type": "Swing High",
                             "Price": round(val, 5), "Structure": tag})
            for idx, (ts, val) in enumerate(sl_list[-10:]):
                prev_val = sl_list[idx - 1][1] if idx > 0 else None
                tag = "HL ✅" if (prev_val and val > prev_val) else ("LL ❌" if prev_val else "—")
                rows.append({"Date": ts.strftime("%Y-%m-%d"), "Type": "Swing Low",
                             "Price": round(val, 5), "Structure": tag})
            if rows:
                piv_df = pd.DataFrame(rows).sort_values("Date", ascending=False)
                st.dataframe(piv_df, use_container_width=True, hide_index=True)
            else:
                st.info("No pivot data available.")

        st.markdown("---")
        st.markdown('<div class="section-title">📖 Alignment Logic Explained</div>',
                    unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("""
            <div class="explainer">
            <b style="color:#3fb950;">📅 Weekly Swing Structure</b><br><br>
            The weekly chart defines the <em>primary bias</em> — the dominant force driving price.<br><br>
            <b>Bullish structure (HH + HL):</b><br>
            Each successive swing high and swing low is higher than the previous.
            Buyers are in control across the macro timeframe.<br><br>
            <b>Bearish structure (LH + LL):</b><br>
            Each successive swing high and swing low is lower than the previous.
            Sellers dominate the macro picture.<br><br>
            Pivots require a configurable number of candles on each side to be confirmed,
            reducing false signals from noise.
            </div>
            """, unsafe_allow_html=True)
        with col_b:
            st.markdown(f"""
            <div class="explainer">
            <b style="color:#388bfd;">📆 Daily Trend & Alignment Rule</b><br><br>
            The daily trend acts as the <em>execution timeframe filter</em>.<br><br>
            <b>EMA crossover:</b><br>
            EMA {int(daily_fast)} above EMA {int(daily_slow)} = daily bullish.<br>
            EMA {int(daily_fast)} below EMA {int(daily_slow)} = daily bearish.<br><br>
            <b>✅ Aligned:</b> Weekly bias matches daily EMA direction → trade with both.<br>
            <b>❌ Conflict:</b> Weekly and daily point opposite ways → skip or size down.<br>
            <b>↔️ Ranging:</b> Weekly structure is not trending → wait for breakout.<br><br>
            This is <em>Check #7</em>. A confirmed ✅ Aligned result here means the
            primary macro trend supports the entry direction you identified on the lower timeframes.
            </div>
            """, unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center;color:#484f58;font-size:11px;margin-top:32px;
            padding-top:16px;border-top:1px solid #21262d;">
  🔄 Weekly Swing Alignment · Check #7 · Weekly structure + Daily confirmation · For educational purposes only
</div>
""", unsafe_allow_html=True)