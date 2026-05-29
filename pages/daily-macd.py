import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime

# ── Page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Daily MACD Momentum",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    /* Theme-adaptive layout vars */
    .stApp {
      --border: color-mix(in srgb, var(--text-color) 12%, transparent);
      --muted:  color-mix(in srgb, var(--text-color) 55%, transparent);
    }
    html,body,[class*="css"]{ font-family:'Inter',sans-serif; }
    .stApp{ background:var(--background-color); }
    section[data-testid="stSidebar"]{ background:var(--secondary-background-color)!important; border-right:1px solid var(--border,#21262d); }
    #MainMenu,footer,header{ visibility:hidden; }
    [data-testid="stSidebarCollapsedControl"]{visibility:visible !important;}
    [data-testid="stSidebarNav"]{ display:none; }
    .block-container{ padding-top:1.5rem; max-width:1380px; }

    .card{ background:var(--secondary-background-color); border:1px solid var(--border,#21262d); border-radius:12px;
           padding:20px; margin-bottom:16px; }
    .card-header{ font-size:13px; font-weight:600; letter-spacing:.08em;
                  text-transform:uppercase; color:#8b949e; margin-bottom:14px; }
    .metric-box{ background:var(--background-color); border:1px solid var(--border,#21262d); border-radius:8px;
                 padding:14px; text-align:center; }
    .metric-value{ font-size:22px; font-weight:700; color:var(--text-color); }
    .metric-label{ font-size:11px; color:var(--muted,#8b949e); margin-top:2px; font-weight:500;
                   letter-spacing:.04em; text-transform:uppercase; }
    .section-title{ font-size:16px; font-weight:700; color:var(--text-color);
                    margin:24px 0 12px 0; padding-left:4px; border-left:3px solid #388bfd; }
    .prog-track{ background:var(--border,#21262d); border-radius:8px; height:10px;
                 margin:6px 0 2px 0; overflow:hidden; }

    /* Verdict banners */
    .verdict-bull { background:linear-gradient(135deg,#0d3a1f,#0d5e32);
                    border:2px solid #238636; border-radius:14px;
                    padding:22px 28px; text-align:center; margin-bottom:18px; }
    .verdict-bear { background:linear-gradient(135deg,#3a0d0d,#5e1414);
                    border:2px solid #8b2d2d; border-radius:14px;
                    padding:22px 28px; text-align:center; margin-bottom:18px; }
    .verdict-flat { background:linear-gradient(135deg,#1c1c0d,#2e2a0d);
                    border:2px solid #9e6a03; border-radius:14px;
                    padding:22px 28px; text-align:center; margin-bottom:18px; }
    .verdict-none { background:linear-gradient(135deg,#161b22,#1c2128);
                    border:1px solid #30363d; border-radius:14px;
                    padding:22px 28px; text-align:center; margin-bottom:18px; }
    .verdict-turning-bull { background:linear-gradient(135deg,#162812,#1f3d18);
                            border:2px solid #4d8c2a; border-radius:14px;
                            padding:22px 28px; text-align:center; margin-bottom:18px; }
    .verdict-turning-bear { background:linear-gradient(135deg,#28160d,#3d2214);
                            border:2px solid #8c5a2a; border-radius:14px;
                            padding:22px 28px; text-align:center; margin-bottom:18px; }

    /* Histogram bars table */
    .hist-row{ display:flex; align-items:center; gap:12px; padding:8px 14px;
               border-bottom:1px solid #21262d; font-size:13px; }
    .hist-row:last-child{ border-bottom:none; }
    .hist-label{ color:#8b949e; min-width:100px; font-size:12px; }
    .hist-bar-wrap{ flex:1; background:#21262d; border-radius:4px; height:8px;
                    overflow:hidden; }
    .hist-val{ font-family:monospace; font-weight:600; min-width:80px;
               text-align:right; font-size:12px; }

    /* Condition rows */
    .cond-row{ display:flex; justify-content:space-between; align-items:center;
               padding:10px 14px; background:#0d1117; border:1px solid #21262d;
               border-radius:8px; margin-bottom:8px; font-size:13px; }
    .cond-text{ color:#c9d1d9; }
    .cond-badge-ok  { background:#0d4a2f; color:#3fb950; border:1px solid #238636;
                      border-radius:20px; padding:3px 12px; font-size:11px; font-weight:700; }
    .cond-badge-fail{ background:#4a0d0d; color:#f85149; border:1px solid #8b2d2d;
                      border-radius:20px; padding:3px 12px; font-size:11px; font-weight:700; }
    .cond-badge-warn{ background:#2a1f00; color:#e3b341; border:1px solid #9e6a03;
                      border-radius:20px; padding:3px 12px; font-size:11px; font-weight:700; }

    .explainer{ background:var(--background-color); border:1px solid #1e3a5f;
                border-left:3px solid #388bfd; border-radius:8px;
                padding:14px 18px; font-size:13px; color:var(--muted,#8b949e); line-height:1.7; }
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
# DATA & INDICATORS
# ══════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner=False)
def fetch_daily(ticker: str) -> pd.DataFrame:
    try:
        df = yf.download(ticker, interval="1d", period="1y",
                         progress=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df[["Open","High","Low","Close","Volume"]].dropna()
    except Exception:
        return pd.DataFrame()


def calc_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def calc_macd(series: pd.Series,
              fast: int = 12, slow: int = 26, sig: int = 9
              ) -> tuple[pd.Series, pd.Series, pd.Series]:
    e_fast   = calc_ema(series, fast)
    e_slow   = calc_ema(series, slow)
    macd_line = e_fast - e_slow
    sig_line  = calc_ema(macd_line, sig)
    histogram = macd_line - sig_line
    return macd_line, sig_line, histogram


def analyse_momentum(df: pd.DataFrame,
                     fast: int, slow: int, sig: int,
                     lookback: int) -> dict:
    """
    Evaluate MACD histogram momentum direction.

    Rules:
    BULLISH  — histogram is positive AND has been rising for ≥ 2 of last N bars
    BEARISH  — histogram is negative AND has been falling for ≥ 2 of last N bars
    TURNING BULLISH — histogram was negative, now crossing zero or climbing
    TURNING BEARISH — histogram was positive, now crossing zero or falling
    FLAT     — histogram near zero or not showing clear direction
    """
    df = df.copy()
    macd_l, sig_l, hist = calc_macd(df["Close"], fast, slow, sig)
    df["macd"]    = macd_l
    df["signal"]  = sig_l
    df["hist"]    = hist
    df["ema_fast"]= calc_ema(df["Close"], fast)
    df["ema_slow"]= calc_ema(df["Close"], slow)

    tail      = df.tail(lookback + 1)
    hist_vals = tail["hist"].values          # oldest → newest
    h_now     = float(hist_vals[-1])
    h_prev    = float(hist_vals[-2])
    h_prev2   = float(hist_vals[-3]) if len(hist_vals) >= 3 else h_prev

    macd_now  = float(df["macd"].iloc[-1])
    sig_now   = float(df["signal"].iloc[-1])
    price_now = float(df["Close"].iloc[-1])
    ema_f_now = float(df["ema_fast"].iloc[-1])
    ema_s_now = float(df["ema_slow"].iloc[-1])

    # Direction of last N histogram bars (True = rising, False = falling)
    hist_deltas   = [hist_vals[i] - hist_vals[i-1]
                     for i in range(1, len(hist_vals))]
    rising_count  = sum(1 for d in hist_deltas[-lookback:] if d > 0)
    falling_count = sum(1 for d in hist_deltas[-lookback:] if d < 0)

    above_zero    = h_now > 0
    below_zero    = h_now < 0
    rising_now    = h_now > h_prev
    falling_now   = h_now < h_prev
    accelerating  = rising_now  and (h_prev > h_prev2)
    decelerating  = falling_now and (h_prev < h_prev2)

    # Cross-zero events
    crossed_up    = h_prev <= 0 < h_now
    crossed_down  = h_prev >= 0 > h_now

    # Momentum verdict
    if above_zero and rising_count >= 2:
        verdict = "BULLISH"
        label   = "📈 Bullish Momentum"
        color   = "#3fb950"
        grade   = "bull"
        desc    = "Histogram is positive and rising — bullish momentum is building."
    elif below_zero and falling_count >= 2:
        verdict = "BEARISH"
        label   = "📉 Bearish Momentum"
        color   = "#f85149"
        grade   = "bear"
        desc    = "Histogram is negative and falling — bearish momentum is building."
    elif crossed_up or (below_zero and rising_count >= 2):
        verdict = "TURNING BULLISH"
        label   = "🟡 Turning Bullish"
        color   = "#6ac43f"
        grade   = "turning-bull"
        desc    = "Histogram is climbing from negative territory — possible bullish shift forming."
    elif crossed_down or (above_zero and falling_count >= 2):
        verdict = "TURNING BEARISH"
        label   = "🟡 Turning Bearish"
        color   = "#e07b39"
        grade   = "turning-bear"
        desc    = "Histogram is falling from positive territory — possible bearish shift forming."
    else:
        verdict = "FLAT"
        label   = "⏳ No Clear Momentum"
        color   = "#8b949e"
        grade   = "none"
        desc    = "Histogram is flat or mixed — wait for a directional move before acting."

    # Confirmation conditions
    conds = {
        "MACD line above zero":        macd_now > 0,
        "MACD above signal line":      macd_now > sig_now,
        "Histogram positive":          above_zero,
        "Histogram rising (last bar)": rising_now,
        f"Rising ≥ 2 of last {lookback} bars": rising_count >= 2,
        "Accelerating (3 bars up)":    accelerating,
        "EMA fast > EMA slow":         ema_f_now > ema_s_now,
    }
    bear_conds = {
        "MACD line below zero":         macd_now < 0,
        "MACD below signal line":       macd_now < sig_now,
        "Histogram negative":           below_zero,
        "Histogram falling (last bar)": falling_now,
        f"Falling ≥ 2 of last {lookback} bars": falling_count >= 2,
        "Accelerating (3 bars down)":   decelerating,
        "EMA fast < EMA slow":          ema_f_now < ema_s_now,
    }

    # Recent histogram bars for sparkline display
    recent = df.tail(20)[["hist"]].copy()
    recent.index = recent.index.strftime("%b %d")

    return {
        "verdict":      verdict,
        "label":        label,
        "color":        color,
        "grade":        grade,
        "desc":         desc,
        "h_now":        h_now,
        "h_prev":       h_prev,
        "macd_now":     macd_now,
        "sig_now":      sig_now,
        "price":        price_now,
        "ema_fast":     ema_f_now,
        "ema_slow":     ema_s_now,
        "rising_count": rising_count,
        "falling_count":falling_count,
        "accelerating": accelerating,
        "decelerating": decelerating,
        "crossed_up":   crossed_up,
        "crossed_down": crossed_down,
        "conds":        conds,
        "bear_conds":   bear_conds,
        "recent_hist":  recent,
        "df":           df,
    }


# ══════════════════════════════════════════════════════════════════
# CHART
# ══════════════════════════════════════════════════════════════════

def build_chart(m: dict, pair: str,
                fast: int, slow: int, sig: int,
                show_n: int = 120) -> go.Figure:

    df   = m["df"].tail(show_n)
    hist = df["hist"].values

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.50, 0.25, 0.25],
        vertical_spacing=0.03,
        subplot_titles=[
            f"{pair} — Daily Price + EMAs",
            f"MACD ({fast} / {slow} / {sig})",
            "MACD Histogram — Momentum Direction",
        ],
    )

    # ── Row 1: Price + EMAs ───────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"],   close=df["Close"],
        increasing_line_color="#3fb950", decreasing_line_color="#f85149",
        increasing_fillcolor="rgba(63,185,80,0.20)", decreasing_fillcolor="rgba(248,81,73,0.20)",
        name="Daily", showlegend=False,
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["ema_fast"],
        mode="lines", line=dict(color="#f59e0b", width=2),
        name=f"EMA {fast}",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["ema_slow"],
        mode="lines", line=dict(color="#a29bfe", width=2, dash="dash"),
        name=f"EMA {slow}",
    ), row=1, col=1)

    # ── Row 2: MACD lines ─────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df.index, y=df["macd"],
        mode="lines", line=dict(color="#0984e3", width=2),
        name="MACD",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["signal"],
        mode="lines", line=dict(color="#e17055", width=1.8, dash="dot"),
        name="Signal",
    ), row=2, col=1)
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.13)", line_dash="dot",
                  line_width=1, row=2, col=1)

    # MACD crossover dots
    macd_v = df["macd"].values
    sig_v  = df["signal"].values
    cross_bull_idx = [df.index[i] for i in range(1, len(df))
                      if macd_v[i] > sig_v[i] and macd_v[i-1] <= sig_v[i-1]]
    cross_bear_idx = [df.index[i] for i in range(1, len(df))
                      if macd_v[i] < sig_v[i] and macd_v[i-1] >= sig_v[i-1]]

    if cross_bull_idx:
        cross_y = [float(df.loc[i,"macd"]) for i in cross_bull_idx]
        fig.add_trace(go.Scatter(
            x=cross_bull_idx, y=cross_y, mode="markers",
            marker=dict(symbol="circle", size=10, color="#3fb950",
                        line=dict(color="#0d1117", width=1.5)),
            name="Bull Cross", showlegend=True,
        ), row=2, col=1)
    if cross_bear_idx:
        cross_y = [float(df.loc[i,"macd"]) for i in cross_bear_idx]
        fig.add_trace(go.Scatter(
            x=cross_bear_idx, y=cross_y, mode="markers",
            marker=dict(symbol="circle", size=10, color="#f85149",
                        line=dict(color="#0d1117", width=1.5)),
            name="Bear Cross", showlegend=True,
        ), row=2, col=1)

    # ── Row 3: Histogram — colour-coded by direction ───────────────
    # Green rising, green fading, red falling, red fading
    bar_colors = []
    for i, v in enumerate(hist):
        if i == 0:
            bar_colors.append("#3fb950" if v >= 0 else "#f85149")
            continue
        prev_v = hist[i-1]
        if v >= 0:
            bar_colors.append("#3fb950" if v >= prev_v else "rgba(86,211,100,0.53)")   # bright if rising, faded if falling
        else:
            bar_colors.append("#f85149" if v <= prev_v else "rgba(248,81,73,0.53)")   # bright if falling, faded if rising

    fig.add_trace(go.Bar(
        x=df.index, y=df["hist"],
        marker_color=bar_colors,
        name="Histogram", showlegend=False,
    ), row=3, col=1)
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.20)", line_dash="dot",
                  line_width=1, row=3, col=1)

    # Highlight last bar
    fig.add_trace(go.Bar(
        x=[df.index[-1]], y=[float(df["hist"].iloc[-1])],
        marker_color="#ffffff",
        marker_line_color="#ffffff", marker_line_width=2,
        opacity=0.9, name="Current bar", showlegend=True,
    ), row=3, col=1)

    fig.update_layout(
        height=700,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#161b22",
        font=dict(family="Inter, sans-serif", size=11, color="#8b949e"),
        xaxis_rangeslider_visible=False,
        legend=dict(
            bgcolor="rgba(13,17,23,0.60)", bordercolor="#21262d", borderwidth=1,
            font=dict(size=10), orientation="h",
            yanchor="bottom", y=1.01, xanchor="left", x=0,
        ),
        margin=dict(l=10, r=20, t=10, b=10),
        bargap=0.15,
    )
    fig.update_yaxes(gridcolor="#21262d", zeroline=False)
    fig.update_xaxes(gridcolor="#21262d", showgrid=False)
    return fig


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 📊 Daily MACD Momentum")
    st.page_link("daily-trading-checklist.py", label="00. Checklist", icon="📋")
    st.page_link("pages/macro-bias.py", label="02. Macro Bias", icon="🌐")
    st.page_link("pages/news-filter.py", label="03. News Filter", icon="📰")
    st.page_link("pages/correlations.py", label="04. Correlations", icon="🔗")
    st.page_link("pages/atr-volatility.py", label="05. ATR Volatility", icon="📊")
    st.page_link("pages/weekly-ema.py", label="06. Weekly EMA", icon="📉")
    st.page_link("pages/weekly-rsi.py", label="07. Weekly RSI", icon="📡")
    st.page_link("pages/weekly-swing.py", label="08. Weekly Swing", icon="🔄")
    st.page_link("pages/daily-trend.py", label="09. Daily Trend", icon="📈")
    st.page_link("pages/daily-macd.py", label="10. Daily MACD", icon="📊")
    st.page_link("pages/4H-confluence-zone.py", label="11. 4H Confluence Zone", icon="🎯")
    st.page_link("pages/confluence-checker.py", label="12. 2/3 Confluence Check", icon="🔀")
    st.page_link("pages/15m-rejection.py", label="13. 15M Rejection", icon="🕯️")
    st.page_link("pages/15m-entry-signal.py", label="14. 15M Entry Signal", icon="⚡")
    st.page_link("pages/stop-structure.py", label="15. Stop Structure", icon="🛡️")
    st.page_link("pages/rr-calculator.py", label="16. R:R Calculator", icon="⚖️")
    st.page_link("pages/trade-journal.py",    label="17. Trade Journal",     icon="📓")
    st.page_link("pages/market-structure.py",  label="18. Market Structure",  icon="🏗️")
    st.page_link("pages/setup-ranker.py",      label="01. Setup Ranker",      icon="🏆")
    st.page_link("pages/backtest-workflow.py",    label="19. Backtest Workflow",    icon="🧪")
    st.divider()

    inst_keys    = list(INSTRUMENTS.keys())
    default_inst = st.session_state.get("selected_instrument", "EUR/USD")
    if default_inst not in inst_keys:
        default_inst = inst_keys[0]

    selected_pair = st.selectbox("Instrument", inst_keys,
                                 index=inst_keys.index(default_inst))

    show_candles = st.slider("Candles on chart", 60, 252, 120, step=20)

    st.divider()
    st.markdown("**📐 MACD Settings**")
    fast = st.number_input("Fast EMA",   value=12, min_value=2,  max_value=50)
    slow = st.number_input("Slow EMA",   value=26, min_value=5,  max_value=100)
    sig  = st.number_input("Signal EMA", value=9,  min_value=2,  max_value=30)
    lb   = st.slider("Momentum lookback (bars)", 2, 10, 4,
                     help="How many recent histogram bars to assess direction over")

    st.divider()
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.caption(f"🕐 {datetime.now().strftime('%H:%M:%S')} local")

    st.divider()
    st.markdown("""
    <div style='font-size:11px;color:#484f58;line-height:1.9;'>
    <b style='color:#8b949e;'>Histogram colours</b><br>
    🟢 Bright green = rising above zero<br>
    🟢 Faded green  = falling but above zero<br>
    🔴 Bright red   = falling below zero<br>
    🔴 Faded red    = rising but below zero<br><br>
    ⬜ White bar = current candle
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# MAIN PAGE
# ══════════════════════════════════════════════════════════════════

ticker = INSTRUMENTS[selected_pair]["ticker"]

GRADE_CSS = {
    "bull":         "verdict-bull",
    "bear":         "verdict-bear",
    "turning-bull": "verdict-turning-bull",
    "turning-bear": "verdict-turning-bear",
    "flat":         "verdict-flat",
    "none":         "verdict-none",
}

# ── Hero ───────────────────────────────────────────────────────────
st.markdown(f"""
<div style="background:linear-gradient(135deg,#0d1117 0%,#161b22 50%,#0d1117 100%);
            border:1px solid #21262d; border-radius:16px; padding:24px 28px;
            margin-bottom:20px; position:relative; overflow:hidden;">
  <div style="font-size:24px; font-weight:700; color:#e6edf3;">📊 Daily MACD Momentum</div>
  <div style="color:#8b949e; font-size:13px; margin-top:4px;">
    MACD histogram turning in the direction of the trade · Daily chart · {selected_pair}
  </div>
  <div style="font-size:12px; color:#388bfd; margin-top:6px;">
    Check #09 — Daily MACD momentum · {datetime.now().strftime('%A %d %B %Y  |  %H:%M')}
  </div>
</div>
""", unsafe_allow_html=True)

tab_scan, tab_detail = st.tabs(["📊 All-Pairs Scanner", "🔍 Pair Detail"])


# ══════════════════════════════════════════════════════════════════
# TAB 1 — ALL-PAIRS SCANNER
# ══════════════════════════════════════════════════════════════════
with tab_scan:
    scan_results = {}
    prog = st.progress(0, text="📡 Scanning MACD momentum for all pairs…")
    for idx, (scan_pair, cfg) in enumerate(INSTRUMENTS.items()):
        prog.progress((idx + 1) / len(INSTRUMENTS),
                      text=f"📡 {scan_pair} ({idx+1}/{len(INSTRUMENTS)})…")
        try:
            df_s = fetch_daily(cfg["ticker"])
            if df_s.empty or len(df_s) < int(slow) + int(sig) + 5:
                scan_results[scan_pair] = None
                continue
            scan_results[scan_pair] = analyse_momentum(
                df_s, int(fast), int(slow), int(sig), int(lb))
        except Exception:
            scan_results[scan_pair] = None
    prog.empty()

    loaded_s    = {p: v for p, v in scan_results.items() if v is not None}
    bull_s      = [p for p, v in loaded_s.items() if v["verdict"] == "BULLISH"]
    t_bull_s    = [p for p, v in loaded_s.items() if v["verdict"] == "TURNING BULLISH"]
    t_bear_s    = [p for p, v in loaded_s.items() if v["verdict"] == "TURNING BEARISH"]
    bear_s      = [p for p, v in loaded_s.items() if v["verdict"] == "BEARISH"]
    flat_s      = [p for p, v in loaded_s.items() if v["verdict"] == "FLAT"]
    no_data_s   = [p for p in INSTRUMENTS if p not in loaded_s]

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    for col_, val_, lbl_, c_ in [
        (c1, len(bull_s),    "📈 Bullish",       "#3fb950"),
        (c2, len(t_bull_s),  "🟡 Turning Bull",  "#6ac43f"),
        (c3, len(t_bear_s),  "🟡 Turning Bear",  "#e07b39"),
        (c4, len(bear_s),    "📉 Bearish",        "#f85149"),
        (c5, len(flat_s),    "⏳ Flat",            "#484f58"),
        (c6, len(no_data_s), "⛔ No Data",        "#30363d"),
    ]:
        with col_:
            st.markdown(
                f'<div class="metric-box">'
                f'<div class="metric-value" style="color:{c_};font-size:18px;">{val_}</div>'
                f'<div class="metric-label">{lbl_}</div></div>',
                unsafe_allow_html=True)

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    SCAN_ORDER = {"BULLISH": 0, "TURNING BULLISH": 1,
                  "TURNING BEARISH": 2, "BEARISH": 3, "FLAT": 4}
    sorted_scan = sorted(
        INSTRUMENTS.keys(),
        key=lambda p: (SCAN_ORDER.get(
            loaded_s[p]["verdict"] if p in loaded_s else "FLAT", 9), p)
    )

    SCAN_BORDER = {
        "BULLISH":        "#3fb950",
        "TURNING BULLISH":"#6ac43f",
        "TURNING BEARISH":"#e07b39",
        "BEARISH":        "#f85149",
        "FLAT":           "#484f58",
    }
    SCAN_BADGE = {
        "BULLISH":        ("background:#0d2f1a;color:#3fb950;border:1px solid #238636;",  "📈 Bullish"),
        "TURNING BULLISH":("background:#162812;color:#6ac43f;border:1px solid #4d8c2a;",  "🟡 Turning Bull"),
        "TURNING BEARISH":("background:#28160d;color:#e07b39;border:1px solid #8c5a2a;",  "🟡 Turning Bear"),
        "BEARISH":        ("background:#2f0d0d;color:#f85149;border:1px solid #8b2d2d;",  "📉 Bearish"),
        "FLAT":           ("background:#21262d;color:#8b949e;border:1px solid #30363d;",  "⏳ Flat"),
    }

    cards_html = '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:9px;">'
    for scan_pair in sorted_scan:
        if scan_pair not in loaded_s:
            cards_html += (
                f'<div style="border:1px solid #21262d;border-left:4px solid #30363d;'
                f'background:#161b22;border-radius:10px;padding:11px 14px;opacity:.5;">'
                f'<div style="font-size:13px;font-weight:700;color:#8b949e;">{scan_pair}</div>'
                f'<div style="font-size:10px;color:#484f58;margin-top:4px;">No data</div></div>'
            )
            continue

        r       = loaded_s[scan_pair]
        verdict = r["verdict"]
        bdr     = SCAN_BORDER.get(verdict, "#30363d")
        bdg_css, bdg_txt = SCAN_BADGE.get(verdict, SCAN_BADGE["FLAT"])
        h_col   = "#3fb950" if r["h_now"] > 0 else "#f85149"
        outline = "outline:2px solid #388bfd;" if scan_pair == selected_pair else ""

        accel_html = ""
        if r["accelerating"]:
            accel_html = '<span style="color:#3fb950;font-weight:600;">▲ Accel</span>'
        elif r["decelerating"]:
            accel_html = '<span style="color:#f85149;font-weight:600;">▼ Decel</span>'

        cross_html = ""
        if r["crossed_up"]:
            cross_html = '<span style="color:#e3b341;font-weight:600;">⬆ Cross</span>'
        elif r["crossed_down"]:
            cross_html = '<span style="color:#e3b341;font-weight:600;">⬇ Cross</span>'

        cards_html += (
            f'<div style="border:1px solid #21262d;border-left:4px solid {bdr};'
            f'background:#161b22;border-radius:10px;padding:11px 14px;{outline}">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">'
            f'<span style="font-size:13px;font-weight:700;color:#e6edf3;">{scan_pair}</span>'
            f'<span style="border-radius:5px;padding:2px 8px;font-size:10px;font-weight:700;{bdg_css}">{bdg_txt}</span>'
            f'</div>'
            f'<div style="display:flex;gap:8px;font-size:11px;align-items:center;flex-wrap:wrap;">'
            f'<span style="color:#484f58;">H:</span>'
            f'<span style="color:{h_col};font-weight:600;font-family:monospace;">{r["h_now"]:+.5f}</span>'
            f'<span style="color:#484f58;">↑{r["rising_count"]} ↓{r["falling_count"]}</span>'
            f'{accel_html}{cross_html}'
            f'</div></div>'
        )
    cards_html += "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    with st.expander("📋 Full Scan Table"):
        tbl_rows = []
        for scan_pair in INSTRUMENTS:
            r = loaded_s.get(scan_pair)
            if r is None:
                tbl_rows.append({"Pair": scan_pair, "Verdict": "No Data",
                                 "H Now": None, "MACD": None, "Signal": None,
                                 "Rising": None, "Falling": None, "Accel": "—"})
            else:
                tbl_rows.append({
                    "Pair":    scan_pair,
                    "Verdict": r["verdict"],
                    "H Now":   round(r["h_now"], 5),
                    "MACD":    round(r["macd_now"], 5),
                    "Signal":  round(r["sig_now"], 5),
                    "Rising":  r["rising_count"],
                    "Falling": r["falling_count"],
                    "Accel":   "▲" if r["accelerating"] else ("▼" if r["decelerating"] else "—"),
                })
        st.dataframe(pd.DataFrame(tbl_rows), use_container_width=True, hide_index=True,
                     column_config={
                         "H Now":   st.column_config.NumberColumn("H Now",   format="%+.5f"),
                         "MACD":    st.column_config.NumberColumn("MACD",    format="%.5f"),
                         "Signal":  st.column_config.NumberColumn("Signal",  format="%.5f"),
                     })


# ══════════════════════════════════════════════════════════════════
# TAB 2 — PAIR DETAIL
# ══════════════════════════════════════════════════════════════════
with tab_detail:
    with st.spinner(f"Loading daily data for {selected_pair}…"):
        df_raw = fetch_daily(ticker)

    detail_ok = True
    if df_raw.empty or len(df_raw) < int(slow) + int(sig) + 5:
        st.error(f"⚠️ Not enough daily data for **{selected_pair}**.")
        detail_ok = False

    if detail_ok:
        m     = analyse_momentum(df_raw, int(fast), int(slow), int(sig), int(lb))
        vcls  = GRADE_CSS[m["grade"]]
        color = m["color"]

        # Histogram sparkline
        spark_bars = ""
        hist_tail  = list(m["df"]["hist"].tail(8).values)
        max_abs    = max(abs(v) for v in hist_tail) or 1
        for v in hist_tail:
            h_pct = int(abs(v) / max_abs * 32)
            sc    = "#3fb950" if v >= 0 else "#f85149"
            align = "flex-end" if v >= 0 else "flex-start"
            spark_bars += (
                f'<div style="display:flex;flex-direction:column;justify-content:{align};'
                f'width:10px;height:40px;">'
                f'<div style="background:{sc};width:10px;height:{max(h_pct,2)}px;'
                f'border-radius:2px;opacity:0.85;"></div></div>'
            )

        st.markdown(f"""
        <div class="{vcls}">
          <div style="font-size:23px;font-weight:800;color:{color};
                      letter-spacing:1px;margin-bottom:6px;">{m['label']}</div>
          <div style="font-size:13px;color:#8b949e;margin-bottom:14px;">{m['desc']}</div>
          <div style="display:flex;justify-content:center;align-items:flex-end;
                      gap:3px;margin:10px 0;">{spark_bars}</div>
          <div style="font-size:10px;color:#484f58;margin-top:4px;">Last 8 daily histogram bars</div>
          <div style="display:flex;gap:28px;justify-content:center;flex-wrap:wrap;
                      font-size:13px;color:#c9d1d9;margin-top:14px;">
            <div>MACD &nbsp;<code style="color:#0984e3;">{m['macd_now']:.5f}</code></div>
            <div>Signal &nbsp;<code style="color:#e17055;">{m['sig_now']:.5f}</code></div>
            <div>Hist &nbsp;<code style="color:{color};">{m['h_now']:.5f}</code></div>
            <div>Close &nbsp;<code style="color:#e6edf3;">{m['price']:.5f}</code></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── KPI strip ──────────────────────────────────────────────
        accel_val = "▲ Accel" if m["accelerating"] else ("▼ Decel" if m["decelerating"] else "—")
        accel_col = "#3fb950" if m["accelerating"] else "#f85149" if m["decelerating"] else "#484f58"
        k1, k2, k3, k4, k5 = st.columns(5)
        for col, val, lbl, c in [
            (k1, f'{m["h_now"]:+.5f}',   "Histogram Now",        color),
            (k2, f'{m["h_prev"]:+.5f}',  "Histogram Prev Bar",   "#8b949e"),
            (k3, str(m["rising_count"]),  f"Rising / Last {lb}",  "#3fb950"),
            (k4, str(m["falling_count"]), f"Falling / Last {lb}", "#f85149"),
            (k5, accel_val,               "Momentum",             accel_col),
        ]:
            with col:
                st.markdown(
                    f'<div class="metric-box">'
                    f'<div class="metric-value" style="color:{c};font-size:16px;">{val}</div>'
                    f'<div class="metric-label">{lbl}</div></div>',
                    unsafe_allow_html=True)

        st.markdown("---")

        # ── Condition breakdown ────────────────────────────────────
        st.markdown('<div class="section-title">🧩 Momentum Conditions</div>',
                    unsafe_allow_html=True)

        def render_conds(title, conds):
            html = f'<div class="card"><div class="card-header">{title}</div>'
            for text, met in conds.items():
                badge_cls = "cond-badge-ok" if met else "cond-badge-fail"
                badge_txt = "✅ Yes" if met else "❌ No"
                html += (f'<div class="cond-row">'
                         f'<div class="cond-text">{text}</div>'
                         f'<span class="{badge_cls}">{badge_txt}</span></div>')
            score    = sum(conds.values())
            total    = len(conds)
            pct      = int(score / total * 100)
            prog_col = "#3fb950" if pct >= 70 else "#e3b341" if pct >= 40 else "#f85149"
            html += (f'<div style="margin-top:14px;">'
                     f'<div style="display:flex;justify-content:space-between;margin-bottom:4px;">'
                     f'<span style="font-size:11px;color:#8b949e;">Score</span>'
                     f'<span style="font-size:11px;font-weight:700;color:{prog_col};">{score}/{total}</span>'
                     f'</div><div class="prog-track">'
                     f'<div style="background:{prog_col};width:{pct}%;height:100%;border-radius:8px;"></div>'
                     f'</div></div></div>')
            return html

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown(render_conds("📈 Bullish Momentum Conditions", m["conds"]),
                        unsafe_allow_html=True)
        with col_r:
            st.markdown(render_conds("📉 Bearish Momentum Conditions", m["bear_conds"]),
                        unsafe_allow_html=True)

        st.markdown("---")

        # ── Chart ──────────────────────────────────────────────────
        st.markdown('<div class="section-title">📈 Daily Chart — Price · MACD · Histogram</div>',
                    unsafe_allow_html=True)
        fig = build_chart(m, selected_pair,
                          fast=int(fast), slow=int(slow), sig=int(sig),
                          show_n=show_candles)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # ── Recent histogram log ───────────────────────────────────
        st.markdown('<div class="section-title">🗂️ Recent Histogram Values (last 20 days)</div>',
                    unsafe_allow_html=True)
        recent = m["df"].tail(20)[["Close","macd","signal","hist"]].copy()
        recent = recent[::-1]
        recent.index = recent.index.strftime("%Y-%m-%d")
        hist_arr   = recent["hist"].values
        directions = []
        for i, v in enumerate(hist_arr):
            if i == len(hist_arr) - 1:
                directions.append("—")
                continue
            prev = hist_arr[i + 1]
            if v > prev:
                directions.append("↑ Rising" if v > 0 else "↑ Recovering")
            elif v < prev:
                directions.append("↓ Falling" if v < 0 else "↓ Fading")
            else:
                directions.append("→ Flat")
        recent["Direction"] = directions
        recent.columns      = ["Close", "MACD", "Signal", "Histogram", "Direction"]
        st.dataframe(recent.round(5), use_container_width=True,
                     column_config={
                         "Histogram": st.column_config.NumberColumn("Histogram", format="%.5f"),
                         "MACD":      st.column_config.NumberColumn("MACD",      format="%.5f"),
                         "Signal":    st.column_config.NumberColumn("Signal",    format="%.5f"),
                         "Close":     st.column_config.NumberColumn("Close",     format="%.5f"),
                         "Direction": st.column_config.TextColumn("Direction"),
                     })

        st.markdown("---")

        # ── Explainer ──────────────────────────────────────────────
        st.markdown('<div class="section-title">📖 How to Read MACD Histogram Momentum</div>',
                    unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("""
            <div class="explainer">
            <b style="color:#3fb950;">📈 Bullish histogram setup</b><br><br>
            <b>1. Histogram turns positive</b><br>
            Crosses from negative to positive — MACD line has crossed above the signal line.
            The histogram bar is now above zero.<br><br>
            <b>2. Histogram bars grow taller</b><br>
            Each bar is larger than the previous — momentum is <em>accelerating</em> to the upside.
            Bright green bars in the chart.<br><br>
            <b>3. Histogram stays positive while pulling back</b><br>
            Bars shrink (faded green) but stay above zero — healthy pullback within an uptrend.
            Look for entry when bars start growing again.
            </div>
            """, unsafe_allow_html=True)
        with col_b:
            st.markdown("""
            <div class="explainer">
            <b style="color:#f85149;">📉 Bearish histogram setup</b><br><br>
            <b>1. Histogram turns negative</b><br>
            Crosses from positive to negative — MACD line has crossed below the signal line.
            The histogram bar is now below zero.<br><br>
            <b>2. Histogram bars grow deeper</b><br>
            Each bar is more negative than the previous — bearish momentum is <em>accelerating</em>.
            Bright red bars in the chart.<br><br>
            <b>3. Histogram stays negative while recovering</b><br>
            Bars shrink (faded red) but stay below zero — a pullback within a downtrend.
            Look for short entry when bars start deepening again.
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="explainer" style="margin-top:12px;">
        <b style="color:#388bfd;">⚙️ Settings used on this page</b>&nbsp;·&nbsp;
        MACD Fast: <code>{fast}</code> &nbsp;·&nbsp;
        Slow: <code>{slow}</code> &nbsp;·&nbsp;
        Signal: <code>{sig}</code> &nbsp;·&nbsp;
        Lookback: <code>{lb} bars</code><br><br>
        This is <em>Check #09</em>. Confirm the histogram is turning in your intended trade direction
        <em>before</em> moving to Check #10 (session window) and the 4H confluence checks.
        Use alongside Check #08 (Daily trend intact — EMA crossover) for full daily timeframe alignment.
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align:center;color:#484f58;font-size:11px;margin-top:32px;
            padding-top:16px;border-top:1px solid #21262d;">
  📊 Daily MACD Momentum · Check #09 · Histogram direction · For educational purposes only
</div>
""", unsafe_allow_html=True)