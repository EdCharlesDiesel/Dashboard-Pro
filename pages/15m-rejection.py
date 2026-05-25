import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import pytz

# ── Page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="15M Rejection Scanner",
    page_icon="🕯️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS — matches the main app dark theme ─────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html,body,[class*="css"]{ font-family:'Inter',sans-serif; }
    .stApp { background:#0d1117; }
    section[data-testid="stSidebar"]{ background:#161b22!important; border-right:1px solid #21262d; }

    .card{ background:#161b22; border:1px solid #21262d; border-radius:12px; padding:20px; margin-bottom:16px; }
    .card-header{ font-size:13px; font-weight:600; letter-spacing:.08em; text-transform:uppercase; color:#8b949e; margin-bottom:14px; }
    .metric-box{ background:#0d1117; border:1px solid #21262d; border-radius:8px; padding:14px; text-align:center; }
    .metric-value{ font-size:22px; font-weight:700; color:#c9d1d9; }
    .metric-label{ font-size:11px; color:#8b949e; margin-top:2px; font-weight:500; letter-spacing:.04em; text-transform:uppercase; }
    .section-title{ font-size:16px; font-weight:700; color:#e6edf3; margin:24px 0 12px 0; padding-left:4px; border-left:3px solid #388bfd; }
    .prog-track{ background:#21262d; border-radius:8px; height:10px; margin:6px 0 2px 0; overflow:hidden; }
    #MainMenu,footer,header{ visibility:hidden; }
    [data-testid="stSidebarCollapsedControl"]{visibility:visible !important;}
    [data-testid="stSidebarNav"]{ display:none; }
    .block-container{ padding-top:1.5rem; max-width:1380px; }

    /* Pattern signal pills */
    .pill-bull{ display:inline-block; background:#0d4a2f; color:#3fb950; border:1px solid #238636;
                border-radius:20px; padding:3px 12px; font-size:12px; font-weight:600; margin:2px; }
    .pill-bear{ display:inline-block; background:#4a0d0d; color:#f85149; border:1px solid #8b2d2d;
                border-radius:20px; padding:3px 12px; font-size:12px; font-weight:600; margin:2px; }
    .pill-neutral{ display:inline-block; background:#1c2128; color:#8b949e; border:1px solid #30363d;
                   border-radius:20px; padding:3px 12px; font-size:12px; font-weight:600; margin:2px; }

    /* Signal verdict banner */
    .verdict-bull{ background:linear-gradient(135deg,#0d4a2f,#0d6634); border:1px solid #238636;
                   border-radius:12px; padding:18px 24px; text-align:center; margin-bottom:16px; }
    .verdict-bear{ background:linear-gradient(135deg,#4a0d0d,#6e1414); border:1px solid #8b2d2d;
                   border-radius:12px; padding:18px 24px; text-align:center; margin-bottom:16px; }
    .verdict-none{ background:linear-gradient(135deg,#161b22,#1c2128); border:1px solid #30363d;
                   border-radius:12px; padding:18px 24px; text-align:center; margin-bottom:16px; }

    /* Pattern info cards */
    .pattern-card{ background:#0d1117; border:1px solid #21262d; border-radius:8px;
                   padding:12px 16px; margin:6px 0; }
    .pattern-name{ font-weight:700; font-size:14px; }
    .pattern-desc{ font-size:12px; color:#8b949e; margin-top:4px; }
    .pattern-rule{ font-size:11px; color:#58a6ff; margin-top:6px; font-family:monospace; }
</style>
""", unsafe_allow_html=True)


# ── Instrument registry (mirrors main app) ─────────────────────────
INSTRUMENTS = {
    "EUR/USD": {"ticker": "EURUSD=X", "pip_size": 0.0001},
    "GBP/USD": {"ticker": "GBPUSD=X", "pip_size": 0.0001},
    "AUD/USD": {"ticker": "AUDUSD=X", "pip_size": 0.0001},
    "NZD/USD": {"ticker": "NZDUSD=X", "pip_size": 0.0001},
    "USD/JPY": {"ticker": "USDJPY=X", "pip_size": 0.01},
    "USD/CHF": {"ticker": "USDCHF=X", "pip_size": 0.0001},
    "USD/CAD": {"ticker": "USDCAD=X", "pip_size": 0.0001},
    "EUR/GBP": {"ticker": "EURGBP=X", "pip_size": 0.0001},
    "EUR/JPY": {"ticker": "EURJPY=X", "pip_size": 0.01},
    "GBP/JPY": {"ticker": "GBPJPY=X", "pip_size": 0.01},
    "AUD/JPY": {"ticker": "AUDJPY=X", "pip_size": 0.01},
    "EUR/AUD": {"ticker": "EURAUD=X", "pip_size": 0.0001},
    "GBP/AUD": {"ticker": "GBPAUD=X", "pip_size": 0.0001},
    "EUR/CAD": {"ticker": "EURCAD=X", "pip_size": 0.0001},
    "GBP/CAD": {"ticker": "GBPCAD=X", "pip_size": 0.0001},
    "USD/ZAR": {"ticker": "USDZAR=X", "pip_size": 0.0001},
    "EUR/ZAR": {"ticker": "EURZAR=X", "pip_size": 0.0001},
    "GBP/ZAR": {"ticker": "GBPZAR=X", "pip_size": 0.0001},
    "🥇 Gold":    {"ticker": "GC=F",    "pip_size": 0.10},
    "🥈 Silver":  {"ticker": "SI=F",    "pip_size": 0.01},
    "🪙 Platinum":{"ticker": "PL=F",    "pip_size": 0.10},
}


# ══════════════════════════════════════════════════════════════════
# DATA FETCHING
# ══════════════════════════════════════════════════════════════════

@st.cache_data(ttl=180, show_spinner=False)
def fetch_15m(ticker: str, days: int = 5) -> pd.DataFrame:
    """Fetch 15-minute OHLCV. yfinance supports 15m up to 60 days."""
    try:
        df = yf.download(ticker, interval="15m", period=f"{days}d", progress=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        return df
    except Exception:
        return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════
# PATTERN DETECTION ENGINE
# ══════════════════════════════════════════════════════════════════

def _body(o, h, l, c):
    return abs(c - o)

def _range(h, l):
    return h - l

def _upper_wick(o, h, l, c):
    return h - max(o, c)

def _lower_wick(o, h, l, c):
    return min(o, c) - l

def detect_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect 8 rejection / structural confirmation patterns on each candle.
    Returns the DataFrame with boolean flag columns added.
    """
    O = df["Open"].values
    H = df["High"].values
    L = df["Low"].values
    C = df["Close"].values
    n = len(df)

    hammer        = np.zeros(n, bool)
    shooting_star = np.zeros(n, bool)
    bullish_engulf= np.zeros(n, bool)
    bearish_engulf= np.zeros(n, bool)
    pin_bar_bull  = np.zeros(n, bool)
    pin_bar_bear  = np.zeros(n, bool)
    doji          = np.zeros(n, bool)
    tweezer_bot   = np.zeros(n, bool)
    tweezer_top   = np.zeros(n, bool)

    for i in range(1, n):
        body  = _body(O[i], H[i], L[i], C[i])
        rng   = _range(H[i], L[i])
        uw    = _upper_wick(O[i], H[i], L[i], C[i])
        lw    = _lower_wick(O[i], H[i], L[i], C[i])
        bullish = C[i] > O[i]
        bearish = C[i] < O[i]

        if rng == 0:
            continue
        body_ratio = body / rng

        # ── Doji ───────────────────────────────────────────────
        if body_ratio < 0.10:
            doji[i] = True

        # ── Hammer (bullish rejection at lows) ─────────────────
        # Small body near top, long lower wick ≥ 2× body, tiny upper wick
        if (lw >= 2 * body and uw <= body * 0.5
                and body_ratio < 0.40 and bullish):
            hammer[i] = True

        # ── Shooting Star (bearish rejection at highs) ──────────
        # Small body near bottom, long upper wick ≥ 2× body
        if (uw >= 2 * body and lw <= body * 0.5
                and body_ratio < 0.40 and bearish):
            shooting_star[i] = True

        # ── Pin Bar Bullish ─────────────────────────────────────
        # Lower wick ≥ 60% of total range, body in upper third
        if lw / rng >= 0.60 and min(O[i], C[i]) > L[i] + rng * 0.60:
            pin_bar_bull[i] = True

        # ── Pin Bar Bearish ─────────────────────────────────────
        # Upper wick ≥ 60% of total range, body in lower third
        if uw / rng >= 0.60 and max(O[i], C[i]) < H[i] - rng * 0.60:
            pin_bar_bear[i] = True

        # ── Bullish Engulfing ───────────────────────────────────
        prev_bear = C[i-1] < O[i-1]
        if (prev_bear and bullish
                and C[i] > O[i-1] and O[i] < C[i-1]):
            bullish_engulf[i] = True

        # ── Bearish Engulfing ───────────────────────────────────
        prev_bull = C[i-1] > O[i-1]
        if (prev_bull and bearish
                and C[i] < O[i-1] and O[i] > C[i-1]):
            bearish_engulf[i] = True

        # ── Tweezer Bottom ─────────────────────────────────────
        if abs(L[i] - L[i-1]) / rng < 0.05 and C[i] > O[i] and C[i-1] < O[i-1]:
            tweezer_bot[i] = True

        # ── Tweezer Top ─────────────────────────────────────────
        if abs(H[i] - H[i-1]) / rng < 0.05 and C[i] < O[i] and C[i-1] > O[i-1]:
            tweezer_top[i] = True

    df = df.copy()
    df["hammer"]         = hammer
    df["shooting_star"]  = shooting_star
    df["bullish_engulf"] = bullish_engulf
    df["bearish_engulf"] = bearish_engulf
    df["pin_bar_bull"]   = pin_bar_bull
    df["pin_bar_bear"]   = pin_bar_bear
    df["doji"]           = doji
    df["tweezer_bot"]    = tweezer_bot
    df["tweezer_top"]    = tweezer_top

    # Composite signals
    df["bull_signal"] = (
            df["hammer"] | df["bullish_engulf"] |
            df["pin_bar_bull"] | df["tweezer_bot"]
    )
    df["bear_signal"] = (
            df["shooting_star"] | df["bearish_engulf"] |
            df["pin_bar_bear"] | df["tweezer_top"]
    )
    return df


def detect_structure(df: pd.DataFrame, lookback: int = 10) -> pd.DataFrame:
    """
    Mark swing highs/lows and break-of-structure (BOS) points.
    """
    df = df.copy()
    df["swing_high"] = False
    df["swing_low"]  = False
    df["bos_bull"]   = False
    df["bos_bear"]   = False

    H = df["High"].values
    L = df["Low"].values
    C = df["Close"].values
    n = len(df)

    lb = min(lookback, 3)   # pivot look-left/right
    for i in range(lb, n - lb):
        if all(H[i] >= H[i-j] for j in range(1, lb+1)) and all(H[i] >= H[i+j] for j in range(1, lb+1)):
            df.iloc[i, df.columns.get_loc("swing_high")] = True
        if all(L[i] <= L[i-j] for j in range(1, lb+1)) and all(L[i] <= L[i+j] for j in range(1, lb+1)):
            df.iloc[i, df.columns.get_loc("swing_low")] = True

    # BOS: close breaks previous swing high/low
    prev_sh = None
    prev_sl = None
    for i in range(n):
        if df["swing_high"].iloc[i]:
            prev_sh = H[i]
        if df["swing_low"].iloc[i]:
            prev_sl = L[i]
        if prev_sh is not None and C[i] > prev_sh:
            df.iloc[i, df.columns.get_loc("bos_bull")] = True
        if prev_sl is not None and C[i] < prev_sl:
            df.iloc[i, df.columns.get_loc("bos_bear")] = True

    return df


def latest_verdict(df: pd.DataFrame) -> dict:
    """Summarise signals on the last 3 candles."""
    tail = df.tail(3)
    bull_patterns = []
    bear_patterns = []

    pattern_map = {
        "hammer":        ("🔨 Hammer",         "bull"),
        "bullish_engulf":("🟢 Bull Engulf",     "bull"),
        "pin_bar_bull":  ("📌 Pin Bar ↑",       "bull"),
        "tweezer_bot":   ("🔧 Tweezer Bottom",  "bull"),
        "shooting_star": ("⭐ Shooting Star",   "bear"),
        "bearish_engulf":("🔴 Bear Engulf",     "bear"),
        "pin_bar_bear":  ("📌 Pin Bar ↓",       "bear"),
        "tweezer_top":   ("🔩 Tweezer Top",     "bear"),
        "doji":          ("⚖️ Doji",            "neutral"),
        "bos_bull":      ("📈 BOS Break Up",    "bull"),
        "bos_bear":      ("📉 BOS Break Down",  "bear"),
    }

    for _, row in tail.iterrows():
        for col, (label, side) in pattern_map.items():
            if col in row and row[col]:
                if side == "bull":
                    bull_patterns.append(label)
                elif side == "bear":
                    bear_patterns.append(label)

    # Remove duplicates preserving order
    bull_patterns = list(dict.fromkeys(bull_patterns))
    bear_patterns = list(dict.fromkeys(bear_patterns))

    if len(bull_patterns) > len(bear_patterns):
        verdict = "BULL"
    elif len(bear_patterns) > len(bull_patterns):
        verdict = "BEAR"
    elif bull_patterns or bear_patterns:
        verdict = "MIXED"
    else:
        verdict = "NONE"

    return {
        "verdict": verdict,
        "bull_patterns": bull_patterns,
        "bear_patterns": bear_patterns,
        "last_close": float(df["Close"].iloc[-1]),
        "last_time":  df.index[-1],
    }


# ══════════════════════════════════════════════════════════════════
# CHART BUILDER
# ══════════════════════════════════════════════════════════════════

def build_chart(df: pd.DataFrame, pair: str, show_candles: int = 80) -> go.Figure:
    plot_df = df.tail(show_candles).copy()

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.80, 0.20], vertical_spacing=0.03,
    )

    # Candles
    fig.add_trace(go.Candlestick(
        x=plot_df.index,
        open=plot_df["Open"], high=plot_df["High"],
        low=plot_df["Low"],  close=plot_df["Close"],
        increasing_line_color="#3fb950", decreasing_line_color="#f85149",
        increasing_fillcolor="#3fb95044", decreasing_fillcolor="#f8514944",
        name="15M Candles", showlegend=False,
    ), row=1, col=1)

    # Volume
    vol_colors = [
        "#3fb95066" if c >= o else "#f8514966"
        for c, o in zip(plot_df["Close"], plot_df["Open"])
    ]
    fig.add_trace(go.Bar(
        x=plot_df.index, y=plot_df["Volume"],
        marker_color=vol_colors, name="Volume", showlegend=False,
    ), row=2, col=1)

    # ── Pattern markers ───────────────────────────────────────────
    def add_markers(mask_col, symbol, color, yref, offset_sign, label):
        mask = plot_df[mask_col] if mask_col in plot_df.columns else pd.Series(False, index=plot_df.index)
        pts  = plot_df[mask]
        if pts.empty:
            return
        if yref == "low":
            y_vals = pts["Low"] - offset_sign * (pts["High"] - pts["Low"]) * 0.3
        else:
            y_vals = pts["High"] + offset_sign * (pts["High"] - pts["Low"]) * 0.3
        fig.add_trace(go.Scatter(
            x=pts.index, y=y_vals,
            mode="markers+text",
            marker=dict(symbol=symbol, size=14, color=color, line=dict(color="#0d1117", width=1)),
            text=[label] * len(pts),
            textposition="top center" if yref == "high" else "bottom center",
            textfont=dict(size=9, color=color),
            name=label, showlegend=True,
        ), row=1, col=1)

    add_markers("hammer",         "triangle-up",   "#3fb950", "low",  1, "Hammer")
    add_markers("pin_bar_bull",   "triangle-up",   "#00d4ff", "low",  1, "PinBar↑")
    add_markers("bullish_engulf", "square",        "#3fb950", "low",  1, "Bull Engulf")
    add_markers("tweezer_bot",    "diamond",       "#74b9ff", "low",  1, "Tweezer Bot")
    add_markers("shooting_star",  "triangle-down", "#f85149", "high", 1, "Shoot★")
    add_markers("pin_bar_bear",   "triangle-down", "#ff6b35", "high", 1, "PinBar↓")
    add_markers("bearish_engulf", "square",        "#f85149", "high", 1, "Bear Engulf")
    add_markers("tweezer_top",    "diamond",       "#fdcb6e", "high", 1, "Tweezer Top")
    add_markers("doji",           "circle",        "#8b949e", "high", 1, "Doji")

    # BOS lines
    for i, row in plot_df.iterrows():
        if row.get("bos_bull"):
            fig.add_vline(x=i, line_color="#3fb95066", line_dash="dot", line_width=1, row=1, col=1)
        if row.get("bos_bear"):
            fig.add_vline(x=i, line_color="#f8514966", line_dash="dot", line_width=1, row=1, col=1)

    # Swing high/low labels
    swing_highs = plot_df[plot_df.get("swing_high", False) == True] if "swing_high" in plot_df.columns else pd.DataFrame()
    swing_lows  = plot_df[plot_df.get("swing_low",  False) == True] if "swing_low"  in plot_df.columns else pd.DataFrame()

    if not swing_highs.empty:
        fig.add_trace(go.Scatter(
            x=swing_highs.index, y=swing_highs["High"],
            mode="markers", marker=dict(symbol="x", size=8, color="#e3b341"),
            name="Swing High", showlegend=True,
        ), row=1, col=1)
    if not swing_lows.empty:
        fig.add_trace(go.Scatter(
            x=swing_lows.index, y=swing_lows["Low"],
            mode="markers", marker=dict(symbol="x", size=8, color="#58a6ff"),
            name="Swing Low", showlegend=True,
        ), row=1, col=1)

    fig.update_layout(
        height=600,
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        font=dict(family="Inter, sans-serif", size=11, color="#8b949e"),
        xaxis_rangeslider_visible=False,
        legend=dict(
            bgcolor="#0d111799", bordercolor="#21262d", borderwidth=1,
            font=dict(size=10), orientation="h",
            yanchor="bottom", y=1.01, xanchor="left", x=0,
        ),
        margin=dict(l=10, r=10, t=10, b=10),
        title=dict(text=f"<b>{pair}</b> — 15M Candlestick Rejection Scanner",
                   font=dict(color="#e6edf3", size=15), x=0.01),
    )
    fig.update_yaxes(gridcolor="#21262d", zeroline=False)
    fig.update_xaxes(gridcolor="#21262d", showgrid=False)
    return fig


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 🕯️ 15M Rejection Scanner")
    st.page_link("daily-trading-checklist.py", label="Checklist", icon="📋")
    st.page_link("pages/macro-bias.py", label=" 01. Macro Bias", icon="🌐")
    st.page_link("pages/news-filter.py", label="02. News Filter", icon="📰")
    st.page_link("pages/correlations.py", label="03. Correlations", icon="🔗")
    st.page_link("pages/atr-volatility.py", label="04. ATR Volatility", icon="📊")
    st.page_link("pages/weekly-ema.py", label="05. Weekly EMA", icon="📉")
    st.page_link("pages/weekly-rsi.py", label="06. Weekly RSI", icon="📡")
    st.page_link("pages/daily-trend.py", label="08. Daily Trend", icon="📈")
    st.divider()

    inst_keys = list(INSTRUMENTS.keys())
    default_inst = st.session_state.get("selected_instrument", "EUR/USD")
    if default_inst not in inst_keys:
        default_inst = inst_keys[0]

    selected_pair = st.selectbox(
        "Instrument",
        inst_keys,
        index=inst_keys.index(default_inst),
    )

    lookback_days = st.slider("Data lookback (days)", 1, 7, 3,
                              help="yfinance supports 15m data up to 60 days")
    show_candles  = st.slider("Candles shown on chart", 30, 200, 80, step=10)
    pivot_lb      = st.slider("Swing pivot sensitivity", 2, 8, 4,
                              help="Higher = fewer but more significant pivots")

    st.divider()
    st.markdown("**📋 Patterns detected**")
    st.markdown("""
    <div style='font-size:12px;color:#8b949e;line-height:1.8;'>
    🔨 Hammer &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⭐ Shooting Star<br>
    📌 Pin Bar ↑↓ &nbsp;&nbsp;&nbsp;🟢🔴 Engulfing<br>
    🔧 Tweezer Bot &nbsp;&nbsp;🔩 Tweezer Top<br>
    ⚖️ Doji &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;📈📉 BOS Break
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.caption(f"🕐 {datetime.now().strftime('%H:%M:%S')} local")


# ══════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════════════════════

ticker    = INSTRUMENTS[selected_pair]["ticker"]
pip_size  = INSTRUMENTS[selected_pair]["pip_size"]

# Header
st.markdown(f"""
<div style="background:linear-gradient(135deg,#0d1117 0%,#161b22 50%,#0d1117 100%);
            border:1px solid #21262d;border-radius:16px;padding:24px 28px;
            margin-bottom:20px;position:relative;overflow:hidden;">
  <div style="font-size:24px;font-weight:700;color:#e6edf3;">
    🕯️ 15M Rejection Scanner
  </div>
  <div style="color:#8b949e;font-size:13px;margin-top:4px;">
    Rejection candle &amp; structural confirmation on 15-minute chart · {selected_pair}
  </div>
  <div style="font-size:12px;color:#388bfd;margin-top:6px;">
    Check #13 — Candlestick rejection on 15M · {datetime.now().strftime('%A %d %B %Y  |  %H:%M')}
  </div>
</div>
""", unsafe_allow_html=True)

# Fetch & process
with st.spinner(f"Loading 15M data for {selected_pair}…"):
    df_raw = fetch_15m(ticker, lookback_days)

if df_raw.empty or len(df_raw) < 10:
    st.error(f"⚠️ Could not load 15M data for **{selected_pair}**. "
             "Try a different instrument or reduce the lookback.")
    st.stop()

df = detect_patterns(df_raw)
df = detect_structure(df, lookback=pivot_lb)
v  = latest_verdict(df)

# ── Verdict banner ─────────────────────────────────────────────────
verdict_cfg = {
    "BULL":  ("verdict-bull", "🟢 BULLISH REJECTION", "#3fb950",
              "Rejection candle(s) suggest upward structural confirmation"),
    "BEAR":  ("verdict-bear", "🔴 BEARISH REJECTION", "#f85149",
              "Rejection candle(s) suggest downward structural confirmation"),
    "MIXED": ("verdict-none", "⚠️ MIXED SIGNALS",    "#e3b341",
              "Both bullish and bearish patterns present — wait for clarity"),
    "NONE":  ("verdict-none", "⏳ NO CLEAR PATTERN", "#8b949e",
              "No significant rejection candles detected on the last 3 candles"),
}
vcls, vtitle, vcolor, vdesc = verdict_cfg[v["verdict"]]

bull_pills = "".join(f'<span class="pill-bull">{p}</span>' for p in v["bull_patterns"])
bear_pills = "".join(f'<span class="pill-bear">{p}</span>' for p in v["bear_patterns"])
all_pills  = bull_pills + bear_pills or '<span class="pill-neutral">No patterns on last 3 candles</span>'

st.markdown(f"""
<div class="{vcls}">
  <div style="font-size:22px;font-weight:800;color:{vcolor};letter-spacing:1px;">{vtitle}</div>
  <div style="font-size:13px;color:#8b949e;margin:6px 0 10px 0;">{vdesc}</div>
  <div>{all_pills}</div>
  <div style="margin-top:10px;font-size:12px;color:#484f58;">
    Last candle: {v['last_time'].strftime('%Y-%m-%d %H:%M')} &nbsp;·&nbsp;
    Close: <code style="color:#c9d1d9;">{v['last_close']:.5f}</code>
  </div>
</div>
""", unsafe_allow_html=True)

# ── KPI row ────────────────────────────────────────────────────────
total_bull = int(df["bull_signal"].sum())
total_bear = int(df["bear_signal"].sum())
total_bos  = int(df["bos_bull"].sum() + df["bos_bear"].sum())
total_doji = int(df["doji"].sum())
last_3_patterns = len(v["bull_patterns"]) + len(v["bear_patterns"])

k1, k2, k3, k4, k5 = st.columns(5)
kpis = [
    (k1, str(total_bull),       "Bull Signals",     "#3fb950"),
    (k2, str(total_bear),       "Bear Signals",     "#f85149"),
    (k3, str(total_bos),        "BOS Events",       "#388bfd"),
    (k4, str(total_doji),       "Doji Candles",     "#8b949e"),
    (k5, str(last_3_patterns),  "Last 3 Patterns",  "#e3b341"),
]
for col, val, lbl, color in kpis:
    with col:
        st.markdown(
            f'<div class="metric-box">'
            f'<div class="metric-value" style="color:{color};">{val}</div>'
            f'<div class="metric-label">{lbl}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.markdown("---")

# ── Chart ──────────────────────────────────────────────────────────
st.markdown('<div class="section-title">📊 15M Chart — Pattern Overlay</div>', unsafe_allow_html=True)
fig = build_chart(df, selected_pair, show_candles)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ── Recent signals table ───────────────────────────────────────────
st.markdown('<div class="section-title">🗂️ Recent Pattern Log (last 30 candles)</div>',
            unsafe_allow_html=True)

tail = df.tail(30).copy()
pattern_cols = [
    ("hammer",         "🔨 Hammer",        "Bullish"),
    ("shooting_star",  "⭐ Shooting Star", "Bearish"),
    ("bullish_engulf", "🟢 Bull Engulf",   "Bullish"),
    ("bearish_engulf", "🔴 Bear Engulf",   "Bearish"),
    ("pin_bar_bull",   "📌 PinBar↑",       "Bullish"),
    ("pin_bar_bear",   "📌 PinBar↓",       "Bearish"),
    ("tweezer_bot",    "🔧 Tweezer Bot",   "Bullish"),
    ("tweezer_top",    "🔩 Tweezer Top",   "Bearish"),
    ("doji",           "⚖️ Doji",          "Neutral"),
    ("bos_bull",       "📈 BOS Up",        "Bullish"),
    ("bos_bear",       "📉 BOS Down",      "Bearish"),
]

log_rows = []
for ts, row in tail.iterrows():
    patterns_found = []
    for col, label, side in pattern_cols:
        if col in row and row[col]:
            patterns_found.append((label, side))
    if patterns_found:
        for label, side in patterns_found:
            log_rows.append({
                "Time":    ts.strftime("%m-%d %H:%M"),
                "Pattern": label,
                "Side":    side,
                "Open":    round(float(row["Open"]),  5),
                "High":    round(float(row["High"]),  5),
                "Low":     round(float(row["Low"]),   5),
                "Close":   round(float(row["Close"]), 5),
            })

if log_rows:
    log_df = pd.DataFrame(log_rows)
    st.dataframe(
        log_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Side": st.column_config.TextColumn("Side", width="small"),
            "Pattern": st.column_config.TextColumn("Pattern", width="medium"),
        },
    )
else:
    st.info("No notable patterns detected in the last 30 candles. "
            "Try reducing the swing pivot sensitivity or refreshing data.")

st.markdown("---")

# ── Pattern reference cards ────────────────────────────────────────
st.markdown('<div class="section-title">📖 Pattern Reference Guide</div>', unsafe_allow_html=True)

patterns_ref = [
    ("🔨 Hammer", "Bullish", "#3fb950",
     "Small body at top of range, lower wick ≥ 2× body. Signals buyers absorbing selling pressure.",
     "Lower wick ≥ 2× body · Upper wick ≤ 0.5× body · Bullish close"),

    ("⭐ Shooting Star", "Bearish", "#f85149",
     "Small body at bottom of range, upper wick ≥ 2× body. Signals sellers rejecting higher prices.",
     "Upper wick ≥ 2× body · Lower wick ≤ 0.5× body · Bearish close"),

    ("📌 Pin Bar ↑", "Bullish", "#00d4ff",
     "Long lower wick covering ≥ 60% of the candle range. Classic liquidity sweep / stop hunt reversal.",
     "Lower wick ≥ 60% of range · Body in upper 40% of candle"),

    ("📌 Pin Bar ↓", "Bearish", "#ff6b35",
     "Long upper wick covering ≥ 60% of the candle range. Bearish rejection at key resistance.",
     "Upper wick ≥ 60% of range · Body in lower 40% of candle"),

    ("🟢 Bullish Engulfing", "Bullish", "#3fb950",
     "Current bullish candle's body fully engulfs the previous bearish candle. Strong reversal signal.",
     "Close > Prev Open AND Open < Prev Close · Current must be bullish"),

    ("🔴 Bearish Engulfing", "Bearish", "#f85149",
     "Current bearish candle's body fully engulfs the previous bullish candle. Strong reversal signal.",
     "Close < Prev Open AND Open > Prev Close · Current must be bearish"),

    ("🔧 Tweezer Bottom", "Bullish", "#74b9ff",
     "Two candles with matching lows — previous bearish, current bullish. Double rejection of lows.",
     "|Low[i] − Low[i−1]| < 5% range · Bear candle then Bull candle"),

    ("🔩 Tweezer Top", "Bearish", "#fdcb6e",
     "Two candles with matching highs — previous bullish, current bearish. Double rejection of highs.",
     "|High[i] − High[i−1]| < 5% range · Bull candle then Bear candle"),

    ("⚖️ Doji", "Neutral", "#8b949e",
     "Body < 10% of candle range. Market indecision. High confluence with other signals.",
     "Body / Range < 0.10 · Confirm with next candle direction"),

    ("📈 BOS Break Up", "Bullish", "#3fb950",
     "Price closes above a prior swing high, confirming a bullish break of structure.",
     "Close > prior Swing High · Structural shift in buyer control"),

    ("📉 BOS Break Down", "Bearish", "#f85149",
     "Price closes below a prior swing low, confirming a bearish break of structure.",
     "Close < prior Swing Low · Structural shift in seller control"),
]

cols = st.columns(2)
for idx, (name, side, color, desc, rule) in enumerate(patterns_ref):
    with cols[idx % 2]:
        st.markdown(f"""
        <div class="pattern-card">
            <div class="pattern-name" style="color:{color};">{name}
                <span style="font-size:11px;font-weight:400;color:#8b949e;margin-left:8px;">— {side}</span>
            </div>
            <div class="pattern-desc">{desc}</div>
            <div class="pattern-rule">📐 {rule}</div>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align:center;color:#484f58;font-size:11px;margin-top:32px;
            padding-top:16px;border-top:1px solid #21262d;">
  🕯️ 15M Rejection Scanner · Checklist Item #13 · For educational purposes only
</div>
""", unsafe_allow_html=True)