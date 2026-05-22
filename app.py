"""
╔══════════════════════════════════════════════════════════════╗
║          📈 Forex Trend Following Signal Alert App           ║
║  Strategy: 50 EMA / 200 EMA + RSI + MACD + ADX Confirmation ║
╚══════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import time

# ── Try importing autorefresh (gracefully degrade if missing) ──
try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_AVAILABLE = True
except ImportError:
    AUTOREFRESH_AVAILABLE = False

# ──────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Forex Trend Signal Alerts",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
#  CUSTOM CSS
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Signal banner cards */
    .card-buy {
        background: linear-gradient(135deg, #00b894 0%, #00cec9 100%);
        color: white; padding: 18px; border-radius: 14px;
        text-align: center; margin-bottom: 10px;
        box-shadow: 0 4px 15px rgba(0,184,148,0.35);
    }
    .card-sell {
        background: linear-gradient(135deg, #d63031 0%, #e17055 100%);
        color: white; padding: 18px; border-radius: 14px;
        text-align: center; margin-bottom: 10px;
        box-shadow: 0 4px 15px rgba(214,48,49,0.35);
    }
    .card-neutral {
        background: linear-gradient(135deg, #636e72 0%, #b2bec3 100%);
        color: white; padding: 18px; border-radius: 14px;
        text-align: center; margin-bottom: 10px;
        box-shadow: 0 4px 12px rgba(99,110,114,0.25);
    }
    .pair-name   { font-size: 1.0rem; font-weight: 600; opacity: 0.9; }
    .sig-text    { font-size: 1.7rem; font-weight: 800; letter-spacing: 1px; margin: 4px 0; }
    .price-text  { font-size: 1.0rem; font-family: monospace; }
    .score-text  { font-size: 0.78rem; opacity: 0.85; margin-top: 4px; }

    /* Condition list */
    .cond-met   { color: #00b894; font-weight: 600; }
    .cond-unmet { color: #d63031; font-weight: 600; }

    /* Alert banner */
    .alert-buy  { background:#e8f8f5; border-left:5px solid #00b894;
                  padding:12px 18px; border-radius:8px; margin:8px 0; }
    .alert-sell { background:#fdf2f2; border-left:5px solid #d63031;
                  padding:12px 18px; border-radius:8px; margin:8px 0; }

    /* Sidebar polish */
    [data-testid="stSidebar"] { background: #0f1117; }
    [data-testid="stSidebar"] h2 { color: #fdcb6e; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
#  CONSTANTS
# ──────────────────────────────────────────────────────────────
PAIRS = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "USDJPY=X",
    "AUD/USD": "AUDUSD=X",
    "USD/CHF": "USDCHF=X",
    "NZD/USD": "NZDUSD=X",
    "USD/CAD": "USDCAD=X",
    "EUR/GBP": "EURGBP=X",
}

# (yfinance interval, yfinance period, resample_rule or None)
TIMEFRAMES = {
    "1 Hour":  ("1h", "59d",  None),
    "4 Hours": ("1h", "59d",  "4h"),
    "Daily":   ("1d", "2y",   None),
}

# Sessions (UTC hours)
SESSIONS = {
    "London":        (7,  16),
    "New York":      (12, 21),
    "London+NY":     (12, 16),
    "Asia":          (23, 8),
}

ALERT_HISTORY_KEY = "alert_history"

# ──────────────────────────────────────────────────────────────
#  TECHNICAL INDICATORS
# ──────────────────────────────────────────────────────────────
def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast=12, slow=26, sig=9):
    e_fast   = ema(series, fast)
    e_slow   = ema(series, slow)
    macd_line= e_fast - e_slow
    sig_line = ema(macd_line, sig)
    hist     = macd_line - sig_line
    return macd_line, sig_line, hist

def adx(df: pd.DataFrame, period: int = 14):
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)

    plus_dm  = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    # zero out when the other direction is larger
    mask = plus_dm > minus_dm;  minus_dm[mask]  = 0
    mask = minus_dm > plus_dm;  plus_dm[mask]   = 0

    atr_      = tr.rolling(period).mean()
    plus_di   = 100 * (plus_dm.rolling(period).mean() / atr_.replace(0, np.nan))
    minus_di  = 100 * (minus_dm.rolling(period).mean() / atr_.replace(0, np.nan))
    dx        = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx_val   = dx.rolling(period).mean()
    return adx_val, plus_di, minus_di

# ──────────────────────────────────────────────────────────────
#  DATA FETCHING & PREPARATION
# ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def fetch_and_prepare(ticker: str, interval: str, period: str, resample: str | None) -> pd.DataFrame | None:
    try:
        df = yf.download(ticker, interval=interval, period=period, progress=False, auto_adjust=True)
    except Exception:
        return None

    if df is None or df.empty:
        return None

    # Flatten MultiIndex columns (yfinance ≥ 0.2.38 wraps single tickers too)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)

    if resample:
        df = df.resample(resample).agg({
            "Open":   "first",
            "High":   "max",
            "Low":    "min",
            "Close":  "last",
            "Volume": "sum",
        }).dropna()

    # Indicators
    df["EMA50"]    = ema(df["Close"], 50)
    df["EMA200"]   = ema(df["Close"], 200)
    df["RSI"]      = rsi(df["Close"])
    df["MACD"], df["MACDSig"], df["MACDHist"] = macd(df["Close"])
    df["ADX"], df["PlusDI"], df["MinusDI"]    = adx(df)

    return df.dropna()

# ──────────────────────────────────────────────────────────────
#  SIGNAL ENGINE
# ──────────────────────────────────────────────────────────────
def evaluate_signal(df: pd.DataFrame, min_conditions: int = 4):
    """
    Returns (signal_label, score, max_score, conditions_dict, direction)
    direction: 'BUY' | 'SELL' | 'NEUTRAL'
    """
    if len(df) < 10:
        return "NEUTRAL ⏳", 0, 6, {}, "NEUTRAL"

    r = df.iloc[-1]
    close   = float(r["Close"])
    e50     = float(r["EMA50"])
    e200    = float(r["EMA200"])
    rsi_val = float(r["RSI"])
    macd_v  = float(r["MACD"])
    macd_s  = float(r["MACDSig"])
    adx_v   = float(r["ADX"])

    buy_conds = {
        "Price above 200 EMA":            close > e200,
        "50 EMA above 200 EMA (Golden X)":e50 > e200,
        "Price at/above 50 EMA":          close >= e50 * 0.9985,
        "RSI 45–70 (bullish momentum)":   45 <= rsi_val <= 70,
        "MACD above Signal line":         macd_v > macd_s,
        "Strong trend (ADX > 25)":        adx_v  > 25,
    }
    sell_conds = {
        "Price below 200 EMA":            close < e200,
        "50 EMA below 200 EMA (Death X)": e50 < e200,
        "Price at/below 50 EMA":          close <= e50 * 1.0015,
        "RSI 30–55 (bearish momentum)":   30 <= rsi_val <= 55,
        "MACD below Signal line":         macd_v < macd_s,
        "Strong trend (ADX > 25)":        adx_v  > 25,
    }

    bs = sum(buy_conds.values())
    ss = sum(sell_conds.values())

    if bs >= min_conditions and bs >= ss:
        label = ("🚀 STRONG BUY" if bs >= 5 else "📈 BUY")
        return label, bs, 6, buy_conds, "BUY"
    elif ss >= min_conditions and ss > bs:
        label = ("🔻 STRONG SELL" if ss >= 5 else "📉 SELL")
        return label, ss, 6, sell_conds, "SELL"
    else:
        return "⏳ NEUTRAL", max(bs, ss), 6, {}, "NEUTRAL"

# ──────────────────────────────────────────────────────────────
#  CHART
# ──────────────────────────────────────────────────────────────
def build_chart(df: pd.DataFrame, pair: str) -> go.Figure:
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.58, 0.21, 0.21],
        subplot_titles=[f"{pair} — Price with EMAs", "RSI (14)", "MACD (12 / 26 / 9)"],
    )

    # ── Candlestick ──────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"],  close=df["Close"],
        name="Price",
        increasing_line_color="#00b894",
        decreasing_line_color="#d63031",
        increasing_fillcolor="#00b894",
        decreasing_fillcolor="#d63031",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["EMA50"],
        name="50 EMA", line=dict(color="#fdcb6e", width=2),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["EMA200"],
        name="200 EMA", line=dict(color="#a29bfe", width=2.5, dash="dash"),
    ), row=1, col=1)

    # Shade area between EMAs
    fig.add_trace(go.Scatter(
        x=pd.concat([df.index.to_series(), df.index.to_series()[::-1]]),
        y=pd.concat([df["EMA50"], df["EMA200"][::-1]]),
        fill="toself",
        fillcolor="rgba(253,203,110,0.08)",
        line=dict(color="rgba(255,255,255,0)"),
        showlegend=False, name="EMA band",
    ), row=1, col=1)

    # ── RSI ──────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df.index, y=df["RSI"],
        name="RSI", line=dict(color="#74b9ff", width=1.8),
    ), row=2, col=1)

    for level, color in [(70, "rgba(214,48,49,0.6)"), (30, "rgba(0,184,148,0.6)"), (50, "rgba(255,255,255,0.25)")]:
        fig.add_hline(y=level, line_dash="dash", line_color=color, row=2, col=1)

    # RSI fill zones
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(214,48,49,0.07)", line_width=0, row=2, col=1)
    fig.add_hrect(y0=0,  y1=30,  fillcolor="rgba(0,184,148,0.07)", line_width=0, row=2, col=1)

    # ── MACD ─────────────────────────────────────────────────
    hist_colors = ["#00b894" if v >= 0 else "#d63031" for v in df["MACDHist"]]
    fig.add_trace(go.Bar(
        x=df.index, y=df["MACDHist"],
        name="Histogram", marker_color=hist_colors, opacity=0.7,
    ), row=3, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["MACD"],
        name="MACD", line=dict(color="#0984e3", width=1.8),
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["MACDSig"],
        name="Signal", line=dict(color="#e17055", width=1.8),
    ), row=3, col=1)

    # ── Layout ───────────────────────────────────────────────
    fig.update_layout(
        height=700,
        template="plotly_dark",
        paper_bgcolor="#0f1117",
        plot_bgcolor="#0f1117",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=10, t=40, b=0),
        font=dict(family="monospace", size=11),
    )
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
    return fig

# ──────────────────────────────────────────────────────────────
#  SOUND ALERT  (browser Web Audio API)
# ──────────────────────────────────────────────────────────────
def play_sound(direction: str):
    # BUY → two ascending tones; SELL → two descending tones
    if direction == "BUY":
        tones = [(600, 0.0, 0.25), (900, 0.3, 0.25)]
    else:
        tones = [(900, 0.0, 0.25), (600, 0.3, 0.25)]

    tone_js = "\n".join(
        f"""
        var o{i}=ctx.createOscillator(), g{i}=ctx.createGain();
        o{i}.connect(g{i}); g{i}.connect(ctx.destination);
        o{i}.frequency.value={freq}; o{i}.type='sine';
        g{i}.gain.setValueAtTime(0.35, ctx.currentTime+{start});
        g{i}.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime+{start}+{dur});
        o{i}.start(ctx.currentTime+{start}); o{i}.stop(ctx.currentTime+{start}+{dur}+0.05);
        """
        for i, (freq, start, dur) in enumerate(tones)
    )
    st.components.v1.html(f"""
    <script>
    (function(){{
        var ctx = new (window.AudioContext || window.webkitAudioContext)();
        {tone_js}
    }})();
    </script>""", height=0)

# ──────────────────────────────────────────────────────────────
#  SESSION STATE INIT
# ──────────────────────────────────────────────────────────────
if ALERT_HISTORY_KEY not in st.session_state:
    st.session_state[ALERT_HISTORY_KEY] = []

# ──────────────────────────────────────────────────────────────
#  SIDEBAR
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📈 Forex Signals")
    st.caption("Trend Following Strategy")
    st.divider()

    st.subheader("🌐 Pairs & Timeframe")
    selected_pairs = st.multiselect(
        "Currency Pairs",
        list(PAIRS.keys()),
        default=["EUR/USD", "GBP/USD", "USD/JPY"],
    )
    timeframe = st.selectbox("Timeframe", list(TIMEFRAMES.keys()), index=1)

    st.divider()
    st.subheader("🔔 Alerts")
    sound_on    = st.toggle("🔊 Sound Alert",    value=True)
    toast_on    = st.toggle("💬 Toast Popup",     value=True)
    banner_on   = st.toggle("🚨 Alert Banner",    value=True)

    st.divider()
    st.subheader("⚙️ Signal Sensitivity")
    min_conds = st.slider(
        "Min conditions to trigger",
        min_value=3, max_value=6, value=4,
        help="Lower = more signals (more noise). Higher = fewer but stronger signals."
    )

    st.divider()
    st.subheader("🔄 Auto Refresh")
    auto_refresh = st.toggle("Enable Auto Refresh", value=AUTOREFRESH_AVAILABLE)
    refresh_mins = st.slider("Interval (minutes)", 1, 60, 15,
                             disabled=not auto_refresh)

    if st.button("⚡ Refresh Now", use_container_width=True, type="primary"):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.caption(f"🕐 {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC")
    if AUTOREFRESH_AVAILABLE and auto_refresh:
        st.caption(f"Next refresh in ~{refresh_mins} min")

# ──────────────────────────────────────────────────────────────
#  AUTO REFRESH
# ──────────────────────────────────────────────────────────────
if AUTOREFRESH_AVAILABLE and auto_refresh:
    count = st_autorefresh(interval=refresh_mins * 60_000, limit=None, key="autorefresh_counter")

# ──────────────────────────────────────────────────────────────
#  HEADER
# ──────────────────────────────────────────────────────────────
st.title("📈 Forex Trend Following — Live Signal Dashboard")
st.caption(
    "**Strategy:** 50 EMA / 200 EMA crossover · RSI (14) · MACD · ADX  |  "
    f"**Timeframe:** {timeframe}  |  "
    f"**Sensitivity:** {min_conds}/6 conditions required"
)
st.divider()

# ──────────────────────────────────────────────────────────────
#  LOAD DATA + GENERATE SIGNALS
# ──────────────────────────────────────────────────────────────
if not selected_pairs:
    st.warning("⚠️ Please select at least one currency pair in the sidebar.")
    st.stop()

interval, period, resample = TIMEFRAMES[timeframe]
pair_results = {}
new_alerts   = []

progress = st.progress(0, text="Fetching market data…")
for idx, pair in enumerate(selected_pairs):
    ticker = PAIRS[pair]
    with st.spinner(f"Loading {pair}…"):
        df = fetch_and_prepare(ticker, interval, period, resample)
    progress.progress((idx + 1) / len(selected_pairs), text=f"Processed {pair}")

    if df is None or df.empty:
        pair_results[pair] = {"error": True}
        continue

    signal, score, max_score, conds, direction = evaluate_signal(df, min_conds)
    last = df.iloc[-1]

    pair_results[pair] = {
        "error":     False,
        "df":        df,
        "signal":    signal,
        "score":     score,
        "max_score": max_score,
        "conds":     conds,
        "direction": direction,
        "close":     float(last["Close"]),
        "ema50":     float(last["EMA50"]),
        "ema200":    float(last["EMA200"]),
        "rsi":       float(last["RSI"]),
        "macd":      float(last["MACD"]),
        "adx":       float(last["ADX"]),
    }

    if direction in ("BUY", "SELL"):
        new_alerts.append((pair, signal, direction, float(last["Close"])))

progress.empty()

# ──────────────────────────────────────────────────────────────
#  FIRE ALERTS
# ──────────────────────────────────────────────────────────────
if new_alerts:
    # Sound (first alert only to avoid cacophony)
    if sound_on:
        play_sound(new_alerts[0][2])

    # Toasts
    if toast_on:
        for pair, signal, direction, price in new_alerts:
            icon = "📈" if direction == "BUY" else "📉"
            st.toast(f"{icon} **{pair}** — {signal}  @ {price:.5f}", icon=icon)

    # Alert banner
    if banner_on:
        for pair, signal, direction, price in new_alerts:
            css_cls = "alert-buy" if direction == "BUY" else "alert-sell"
            emoji   = "🚀" if direction == "BUY" else "🔻"
            st.markdown(f"""
            <div class="{css_cls}">
                {emoji} <strong>{pair}</strong> — <strong>{signal}</strong>
                — Entry price: <code>{price:.5f}</code>
                — {datetime.utcnow().strftime('%H:%M UTC')}
            </div>
            """, unsafe_allow_html=True)

    # Save to history (keep last 50)
    for pair, signal, direction, price in new_alerts:
        st.session_state[ALERT_HISTORY_KEY].append({
            "time":      datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            "pair":      pair,
            "signal":    signal,
            "direction": direction,
            "price":     f"{price:.5f}",
        })
    st.session_state[ALERT_HISTORY_KEY] = st.session_state[ALERT_HISTORY_KEY][-50:]

# ──────────────────────────────────────────────────────────────
#  SIGNAL CARDS GRID
# ──────────────────────────────────────────────────────────────
st.subheader("🎯 Live Signals")
cols = st.columns(min(len(selected_pairs), 4))

for idx, pair in enumerate(selected_pairs):
    r = pair_results.get(pair, {})
    with cols[idx % 4]:
        if r.get("error"):
            st.error(f"❌ {pair}\nFailed to load")
            continue

        direction = r["direction"]
        css_cls   = {"BUY": "card-buy", "SELL": "card-sell"}.get(direction, "card-neutral")
        pct_bar   = int(r["score"] / r["max_score"] * 100)

        st.markdown(f"""
        <div class="{css_cls}">
            <div class="pair-name">{pair}</div>
            <div class="sig-text">{r["signal"]}</div>
            <div class="price-text">Price: {r["close"]:.5f}</div>
            <div class="score-text">✦ Conditions: {r["score"]}/{r["max_score"]}</div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(pct_bar, text=f"Signal strength {pct_bar}%")

# ──────────────────────────────────────────────────────────────
#  DETAILED CHART VIEW
# ──────────────────────────────────────────────────────────────
st.divider()
st.subheader("📊 Detailed Chart & Analysis")

valid_pairs = [p for p in selected_pairs if not pair_results.get(p, {}).get("error")]
if not valid_pairs:
    st.error("No valid data available. Try refreshing.")
    st.stop()

# Pre-select the first pair that has a signal
default_pair = next(
    (p for p in valid_pairs if pair_results[p]["direction"] != "NEUTRAL"),
    valid_pairs[0],
)
selected_detail = st.selectbox("Choose pair to inspect:", valid_pairs,
                               index=valid_pairs.index(default_pair))

r = pair_results[selected_detail]

# ── Key metrics ──────────────────────────────────────────────
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Close",   f"{r['close']:.5f}")
m2.metric("50 EMA",  f"{r['ema50']:.5f}",
          delta=f"{((r['close']-r['ema50'])/r['ema50']*100):+.2f}%")
m3.metric("200 EMA", f"{r['ema200']:.5f}",
          delta=f"{((r['close']-r['ema200'])/r['ema200']*100):+.2f}%")
m4.metric("RSI (14)", f"{r['rsi']:.1f}",
          delta="Overbought" if r['rsi'] > 70 else ("Oversold" if r['rsi'] < 30 else "Neutral"))
m5.metric("MACD",    f"{r['macd']:.5f}")
m6.metric("ADX",     f"{r['adx']:.1f}",
          delta="Trending" if r['adx'] > 25 else "Ranging")

# ── Chart ────────────────────────────────────────────────────
st.plotly_chart(build_chart(r["df"], selected_detail), use_container_width=True)

# ── Conditions breakdown ─────────────────────────────────────
if r["conds"]:
    st.subheader(f"📋 Signal Conditions — {selected_detail}")
    col_a, col_b = st.columns(2)
    items = list(r["conds"].items())
    half  = (len(items) + 1) // 2
    for col, chunk in zip([col_a, col_b], [items[:half], items[half:]]):
        with col:
            for cond, met in chunk:
                icon = "✅" if met else "❌"
                st.markdown(f"{icon} {cond}")
else:
    st.info("📋 No active signal conditions — market is neutral / ranging on this timeframe.")

# ──────────────────────────────────────────────────────────────
#  STRATEGY REFERENCE
# ──────────────────────────────────────────────────────────────
with st.expander("📖 Strategy Reference — Trend Following Rules"):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 🟢 BUY Setup
        - Price **above** 200 EMA
        - 50 EMA **above** 200 EMA (Golden Cross)
        - Price pulling back toward / holding above 50 EMA
        - RSI between **45–70** (bullish zone)
        - MACD line **above** signal line
        - ADX **> 25** (trend strength confirmed)

        ### 🔴 SELL Setup
        - Price **below** 200 EMA
        - 50 EMA **below** 200 EMA (Death Cross)
        - Price pulling back toward / below 50 EMA
        - RSI between **30–55** (bearish zone)
        - MACD line **below** signal line
        - ADX **> 25** (trend strength confirmed)
        """)
    with col2:
        st.markdown("""
        ### ⚙️ Recommended Settings
        | Parameter | Value |
        |-----------|-------|
        | Timeframe | 4H or Daily |
        | EMA Fast  | 50 periods |
        | EMA Slow  | 200 periods |
        | RSI       | 14 periods |
        | MACD      | 12 / 26 / 9 |
        | ADX       | 14 periods |

        ### 💡 Risk Management
        - Risk **≤ 1–2%** per trade
        - Target **≥ 1:2** risk:reward ratio
        - Best sessions: **London** & **NY overlap**
        - Best pairs: **EUR/USD · GBP/USD · USD/JPY**
        """)

# ──────────────────────────────────────────────────────────────
#  ALERT HISTORY
# ──────────────────────────────────────────────────────────────
st.divider()
with st.expander(f"🔔 Alert History ({len(st.session_state[ALERT_HISTORY_KEY])} alerts this session)"):
    hist = st.session_state[ALERT_HISTORY_KEY]
    if hist:
        df_hist = pd.DataFrame(hist[::-1])  # newest first
        st.dataframe(
            df_hist,
            column_config={
                "direction": st.column_config.TextColumn("Dir"),
                "pair":      st.column_config.TextColumn("Pair"),
                "signal":    st.column_config.TextColumn("Signal"),
                "price":     st.column_config.TextColumn("Price"),
                "time":      st.column_config.TextColumn("Time (UTC)"),
            },
            hide_index=True,
            use_container_width=True,
        )
        if st.button("🗑️ Clear History"):
            st.session_state[ALERT_HISTORY_KEY] = []
            st.rerun()
    else:
        st.info("No alerts triggered yet this session.")

# ──────────────────────────────────────────────────────────────
#  FOOTER
# ──────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "⚠️ **Disclaimer:** This tool is for educational and informational purposes only. "
    "It does not constitute financial advice. Forex trading involves significant risk of loss. "
    "Always use a demo account before trading real capital and consult a licensed financial advisor."
)
