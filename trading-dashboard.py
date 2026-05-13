"""
Trading Signal Dashboard
Pivot Points + FRED Macro Data + Yahoo Finance Prices + QuantConnect-style signals
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Pivot Point Signal Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Syne:wght@400;600;800&display=swap');
html, body, [class*="css"] { font-family: 'JetBrains Mono', monospace; background-color: #0a0e17; color: #c9d1e0; }
.stApp { background: #0a0e17; }
section[data-testid="stSidebar"] { background: #0d1220; border-right: 1px solid #1e2d45; }
section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] .stMarkdown { color: #8899bb !important; }
[data-testid="metric-container"] { background: #0f1826; border: 1px solid #1e2d45; border-radius: 4px; padding: 12px 16px; }
[data-testid="metric-container"] label { color: #5577aa !important; font-size: 11px !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #e0e8ff !important; font-family: 'JetBrains Mono', monospace !important; }
h1, h2, h3 { font-family: 'Syne', sans-serif !important; color: #e0e8ff !important; }
h1 { font-size: 22px !important; font-weight: 800; }
h3 { font-size: 13px !important; font-weight: 600; color: #5577aa !important; letter-spacing: 2px; text-transform: uppercase; }
.stTabs [data-baseweb="tab-list"] { background: #0d1220; border-bottom: 1px solid #1e2d45; }
.stTabs [data-baseweb="tab"] { color: #5577aa; font-family: 'JetBrains Mono', monospace; font-size: 12px; padding: 8px 20px; border-bottom: 2px solid transparent; }
.stTabs [aria-selected="true"] { color: #4af0c4 !important; border-bottom: 2px solid #4af0c4 !important; background: transparent !important; }
.stSelectbox > div > div { background: #0f1826 !important; border-color: #1e2d45 !important; color: #c9d1e0 !important; }
.sig-buy  { background:#0d3b2a; border:1px solid #1a7a55; color:#4af0c4; padding:4px 12px; border-radius:3px; font-size:11px; font-family:'JetBrains Mono',monospace; display:inline-block; }
.sig-sell { background:#3b0d16; border:1px solid #7a1a2a; color:#f04a6a; padding:4px 12px; border-radius:3px; font-size:11px; font-family:'JetBrains Mono',monospace; display:inline-block; }
.sig-hold { background:#1a1e2d; border:1px solid #2d3a55; color:#8899bb; padding:4px 12px; border-radius:3px; font-size:11px; font-family:'JetBrains Mono',monospace; display:inline-block; }
hr { border-color: #1e2d45; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
FRED_SERIES = {
    "Fed Funds Rate": "FEDFUNDS",
    "CPI YoY":        "CPIAUCSL",
    "10Y Treasury":   "DGS10",
    "2Y Treasury":    "DGS2",
    "Unemployment":   "UNRATE",
    "GDP Growth":     "A191RL1Q225SBEA",
    "DXY Index":      "DTWEXBGS",
    "VIX":            "VIXCLS",
}

POPULAR_SYMBOLS = [
    "EURUSD=X","GBPUSD=X","USDJPY=X","USDCHF=X","AUDUSD=X","NZDUSD=X","USDCAD=X",
    "SPY","QQQ","DIA","IWM","GLD","SLV","USO",
    "AAPL","MSFT","GOOGL","AMZN","NVDA","TSLA","META",
    "BTC-USD","ETH-USD","SOL-USD",
]

TIMEFRAMES = {
    "1m":"1m","5m":"5m","15m":"15m","30m":"30m",
    "1h":"1h","4h":"4h","1d":"1d","1wk":"1wk","1mo":"1mo",
}

PIVOT_LOOKBACKS = {"Daily":"1d","Weekly":"1wk","Monthly":"1mo"}


# ── Pivot calculation ─────────────────────────────────────────────────────────
def calculate_pivots(high, low, close):
    PP = (high + low + close) / 3
    R1 = 2*PP - low
    R2 = PP + (high - low)
    R3 = high + 2*(PP - low)
    S1 = 2*PP - high
    S2 = PP - (high - low)
    S3 = low - 2*(high - PP)
    return {
        "R3":R3,"M4":(R2+R3)/2,"R2":R2,"M3":(R1+R2)/2,"R1":R1,
        "M2":(PP+R1)/2,"PP":PP,"M1":(S1+PP)/2,"S1":S1,
        "M0":(S1+S2)/2,"S2":S2,"S3":S3,
    }


# ── FRED fetch ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_fred(series_id, fred_key, limit=36):
    import requests
    if not fred_key:
        return None
    url = (f"https://api.stlouisfed.org/fred/series/observations"
           f"?series_id={series_id}&api_key={fred_key}&file_type=json&sort_order=desc&limit={limit}")
    try:
        r = requests.get(url, timeout=8)
        data = r.json()
        obs = data.get("observations", [])
        df = pd.DataFrame(obs)[["date","value"]]
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["date"] = pd.to_datetime(df["date"])
        return df.dropna().sort_values("date")
    except Exception:
        return None


# ── Yahoo Finance fetch ───────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch_price_data(symbol, interval, start, end):
    try:
        df = yf.download(symbol, start=start, end=end, interval=interval,
                         auto_adjust=True, progress=False)
        if df.empty:
            return None
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        return df[["Open","High","Low","Close","Volume"]].dropna()
    except Exception as e:
        st.error(f"Yahoo Finance error: {e}")
        return None


# ── RSI / ATR ─────────────────────────────────────────────────────────────────
def compute_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - 100/(1+rs)

def compute_atr(df, period=14):
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"]  - df["Close"].shift()).abs()
    tr = pd.concat([hl,hc,lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ── Signal engine ─────────────────────────────────────────────────────────────
def generate_signals(df, pivots):
    df = df.copy()
    for name, period in [("MA5",5),("MA10",10),("MA20",20),("MA50",50)]:
        df[name] = df["Close"].rolling(period).mean()
    df["RSI"] = compute_rsi(df["Close"], 14)
    df["ATR"] = compute_atr(df, 14)
    df["Signal"] = "HOLD"
    df["Signal_Score"] = 0

    if pivots:
        pp, r1, s1, r2, s2 = pivots["PP"], pivots["R1"], pivots["S1"], pivots["R2"], pivots["S2"]
        close = df["Close"]
        buy  = (close > pp) & (close < r1) & (df["MA5"] > df["MA10"]) & (df["MA10"] > df["MA20"]) & (df["RSI"] < 70) & (close > df["MA20"])
        sell = (close < pp) & (close > s1) & (df["MA5"] < df["MA10"]) & (df["MA10"] < df["MA20"]) & (df["RSI"] > 30) & (close < df["MA20"])
        sbuy  = buy  & (close > df["MA50"]) & (df["RSI"] < 60)
        ssell = sell & (close < df["MA50"]) & (df["RSI"] > 40)

        df.loc[buy,  "Signal"] = "BUY"
        df.loc[sbuy, "Signal"] = "STRONG BUY"
        df.loc[sell, "Signal"] = "SELL"
        df.loc[ssell,"Signal"] = "STRONG SELL"

        scores = {"STRONG BUY":2,"BUY":1,"HOLD":0,"SELL":-1,"STRONG SELL":-2}
        df["Signal_Score"] = df["Signal"].map(scores)
    return df


# ── Chart builder ─────────────────────────────────────────────────────────────
def build_chart(df, pivots, symbol, show_vol):
    rows = 3 if show_vol else 2
    row_heights = [0.65,0.15,0.20] if show_vol else [0.72,0.28]
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                        vertical_spacing=0.02, row_heights=row_heights)

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Price",
        increasing_fillcolor="#1a4d2e", increasing_line_color="#4af0c4",
        decreasing_fillcolor="#4d1a22", decreasing_line_color="#f04a6a",
        line_width=1,
    ), row=1, col=1)

    for col, color, width in [("MA5","#ffffff",1.2),("MA10","#e05050",1.5),("MA20","#5090e0",1.8),("MA50","#50c878",2.2)]:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col,
                                     line=dict(color=color, width=width), opacity=0.85), row=1, col=1)

    if pivots:
        pivot_style = {
            "R3":("#c0392b","dash"),"M4":("#e74c3c","dot"),"R2":("#e74c3c","dash"),
            "M3":("#e74c3c","dot"),"R1":("#c0392b","dashdot"),"M2":("#e74c3c","dot"),
            "PP":("#e0e8ff","solid"),"M1":("#27ae60","dot"),"S1":("#27ae60","dashdot"),
            "M0":("#27ae60","dot"),"S2":("#1e8449","dash"),"S3":("#1e8449","dash"),
        }
        for level,(color,dash) in pivot_style.items():
            val = pivots.get(level)
            if val is None: continue
            lw = 1.8 if level in ("PP","R3","S3") else 1.2
            fig.add_hline(y=val, line_dash=dash, line_color=color, line_width=lw,
                          annotation_text=f"{level} {val:.5f}",
                          annotation_position="right",
                          annotation_font=dict(size=9, color=color, family="JetBrains Mono"),
                          row=1, col=1)
        fig.add_hrect(y0=pivots["R1"], y1=pivots["R2"], fillcolor="rgba(192,57,43,0.10)", line_width=0, row=1, col=1)
        fig.add_hrect(y0=pivots["S2"], y1=pivots["S1"], fillcolor="rgba(39,174,96,0.10)",  line_width=0, row=1, col=1)
        spread = (pivots["R1"] - pivots["S1"]) * 0.04
        fig.add_hrect(y0=pivots["PP"]-spread, y1=pivots["PP"]+spread, fillcolor="rgba(142,68,173,0.12)", line_width=0, row=1, col=1)

    if "Signal" in df.columns:
        for sig, color, sym, size in [("STRONG BUY","#4af0c4","triangle-up",12),("BUY","#27ae60","triangle-up",9),
                                       ("STRONG SELL","#ff2255","triangle-down",12),("SELL","#e05050","triangle-down",9)]:
            mask = df["Signal"] == sig
            if mask.any():
                y_pos = df.loc[mask,"Low"]*0.9998 if "BUY" in sig else df.loc[mask,"High"]*1.0002
                fig.add_trace(go.Scatter(x=df.index[mask], y=y_pos, mode="markers",
                                         marker=dict(symbol=sym, color=color, size=size,
                                                     line=dict(color="#0a0e17",width=1)),
                                         name=sig, showlegend=True), row=1, col=1)

    rsi_row = 3 if show_vol else 2
    if "RSI" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI",
                                  line=dict(color="#f0a030", width=1.2),
                                  fill="tozeroy", fillcolor="rgba(240,160,48,0.07)"), row=rsi_row, col=1)
        for lvl, col in [(70,"rgba(240,80,80,0.4)"),(30,"rgba(74,240,196,0.4)")]:
            fig.add_hline(y=lvl, line_dash="dot", line_color=col, line_width=1, row=rsi_row, col=1)

    if show_vol and "Volume" in df.columns:
        colors_vol = ["#1a4d2e" if c >= o else "#4d1a22" for c,o in zip(df["Close"], df["Open"])]
        fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume",
                              marker_color=colors_vol, opacity=0.7), row=2, col=1)

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0a0e17", plot_bgcolor="#0a0e17",
        font=dict(family="JetBrains Mono", size=10, color="#8899bb"),
        title=dict(text=f"<b>{symbol}</b>  Pivot Signal Dashboard",
                   font=dict(size=16, color="#e0e8ff", family="Syne"), x=0.01),
        legend=dict(bgcolor="#0d1220", bordercolor="#1e2d45", borderwidth=1,
                    font=dict(size=9), orientation="h", x=0, y=1.02),
        xaxis_rangeslider_visible=False, hovermode="x unified",
        margin=dict(l=10, r=90, t=50, b=10), height=640,
    )
    for i in range(1, rows+1):
        fig.update_xaxes(gridcolor="#1e2d45", zeroline=False, showspikes=True,
                         spikecolor="#2d4070", spikethickness=1, row=i, col=1)
        fig.update_yaxes(gridcolor="#1e2d45", zeroline=False, showspikes=True,
                         spikecolor="#2d4070", spikethickness=1, row=i, col=1)
    return fig


# ── FRED chart ────────────────────────────────────────────────────────────────
def build_fred_chart(dfs):
    if not dfs: return None
    fig = make_subplots(rows=2, cols=4, subplot_titles=list(dfs.keys()),
                        vertical_spacing=0.18, horizontal_spacing=0.08)
    colors = ["#4af0c4","#f04a6a","#f0a030","#5090e0","#c878e0","#e05050","#50c878","#e0d050"]
    for i, (name, df_f) in enumerate(dfs.items()):
        if df_f is None or df_f.empty: continue
        row, col = divmod(i, 4)
        hex_c = colors[i % len(colors)].lstrip('#')
        r_c,g_c,b_c = int(hex_c[0:2],16), int(hex_c[2:4],16), int(hex_c[4:6],16)
        fig.add_trace(go.Scatter(
            x=df_f["date"], y=df_f["value"], name=name,
            line=dict(color=colors[i%len(colors)], width=1.5),
            fill="tozeroy", fillcolor=f"rgba({r_c},{g_c},{b_c},0.08)",
        ), row=row+1, col=col+1)
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0a0e17", plot_bgcolor="#0a0e17",
        font=dict(family="JetBrains Mono", size=9, color="#8899bb"),
        showlegend=False, height=320, margin=dict(l=10,r=10,t=35,b=10),
    )
    for ann in fig.layout.annotations:
        ann.font = dict(size=9, color="#8899bb", family="JetBrains Mono")
    fig.update_xaxes(gridcolor="#1e2d45")
    fig.update_yaxes(gridcolor="#1e2d45")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚡ SIGNAL DASHBOARD")
    st.markdown("---")
    st.markdown("### 📊 INSTRUMENT")
    custom_sym = st.text_input("Custom symbol", placeholder="e.g. EURUSD=X, AAPL")
    symbol = custom_sym.upper() if custom_sym else st.selectbox("Popular symbols", POPULAR_SYMBOLS, index=0)

    st.markdown("### ⏱ TIMEFRAME")
    interval = st.selectbox("Chart interval", list(TIMEFRAMES.keys()), index=6)
    interval_val = TIMEFRAMES[interval]
    pivot_tf_label = st.selectbox("Pivot lookback", list(PIVOT_LOOKBACKS.keys()), index=0)
    pivot_tf = PIVOT_LOOKBACKS[pivot_tf_label]

    st.markdown("### 📅 DATE RANGE")
    default_end   = datetime.today()
    default_start = default_end - timedelta(days=180)
    start_date = st.date_input("From", default_start)
    end_date   = st.date_input("To",   default_end)

    st.markdown("### 🎛 DISPLAY")
    show_volume = st.toggle("Show volume", value=True)
    show_fred   = st.toggle("Show FRED macro", value=True)

    st.markdown("### 🔑 API KEYS")
    fred_key = st.text_input("FRED API key", type="password",
                              help="Free key at fred.stlouisfed.org/docs/api/api_key.html")

    st.markdown("---")
    run_btn = st.button("▶  LOAD DATA", use_container_width=True, type="primary")

    st.markdown("""
    <div style='font-size:10px;color:#334466;margin-top:16px;line-height:1.8'>
    <b>Data sources</b><br>
    · Yahoo Finance — OHLCV prices<br>
    · FRED — macro indicators<br>
    · Classic pivot formula (PP/R/S/M)<br>
    · QuantConnect-style signal logic<br><br>
    <b>MAs shown</b><br>
    White=5 · Red=10 · Blue=20 · Green=50
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"## 📈 {symbol}  |  Pivot Signal Dashboard")
st.markdown(f"`{interval}` chart  ·  `{pivot_tf_label}` pivots  ·  {start_date} → {end_date}")

if not run_btn:
    st.info("Configure your instrument in the sidebar and click **▶ LOAD DATA**")
    st.stop()

with st.spinner("Fetching price data from Yahoo Finance..."):
    df = fetch_price_data(symbol, interval_val, str(start_date), str(end_date))

if df is None or df.empty:
    st.error(f"No data for **{symbol}** at `{interval}` interval. Try a different symbol or timeframe.")
    st.stop()

with st.spinner("Computing pivot points..."):
    df_piv = fetch_price_data(symbol, pivot_tf, str(start_date - timedelta(days=30)), str(end_date))
    pivots = None
    if df_piv is not None and len(df_piv) >= 2:
        last_piv = df_piv.iloc[-2]
        pivots = calculate_pivots(float(last_piv["High"]), float(last_piv["Low"]), float(last_piv["Close"]))

df = generate_signals(df, pivots)

# ── Metrics ───────────────────────────────────────────────────────────────────
latest = df["Close"].iloc[-1]
prev   = df["Close"].iloc[-2] if len(df) > 1 else latest
chg, pct = latest - prev, (latest - prev) / prev * 100
last_signal = df["Signal"].iloc[-1]
rsi_val = df["RSI"].iloc[-1] if "RSI" in df.columns else float("nan")
atr_val = df["ATR"].iloc[-1] if "ATR" in df.columns else float("nan")
buy_c   = df["Signal"].str.contains("BUY").sum()
sell_c  = df["Signal"].str.contains("SELL").sum()
bias    = "BULLISH 🟢" if buy_c > sell_c else ("BEARISH 🔴" if sell_c > buy_c else "NEUTRAL ⚪")

c1,c2,c3,c4,c5,c6 = st.columns(6)
c1.metric("Last Price",  f"{latest:.5f}", f"{chg:+.5f} ({pct:+.2f}%)")
c2.metric("Last Signal", last_signal)
c3.metric("RSI (14)",    f"{rsi_val:.1f}", delta_color="off")
c4.metric("ATR (14)",    f"{atr_val:.5f}", delta_color="off")
c5.metric("Signal Bias", bias)
c6.metric("PP Level",    f"{pivots['PP']:.5f}" if pivots else "N/A", delta_color="off")

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊  CHART", "🎯  SIGNALS", "🏛  FRED MACRO", "📋  DATA TABLE"])

with tab1:
    fig = build_chart(df, pivots, symbol, show_volume)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})

    if pivots:
        st.markdown("#### Pivot Level Summary")
        cols_p = st.columns(len(pivots))
        color_map = {"R3":"#c0392b","M4":"#e74c3c","R2":"#e74c3c","M3":"#e74c3c","R1":"#c0392b",
                     "M2":"#e74c3c","PP":"#e0e8ff","M1":"#27ae60","S1":"#27ae60",
                     "M0":"#27ae60","S2":"#1e8449","S3":"#1e8449"}
        for i, (level, val) in enumerate(pivots.items()):
            c = color_map.get(level, "#8899bb")
            dist = (latest - val) / val * 100
            cols_p[i].markdown(
                f"<div style='text-align:center;background:#0f1826;border:1px solid #1e2d45;"
                f"border-top:2px solid {c};border-radius:3px;padding:8px 4px'>"
                f"<div style='color:{c};font-size:10px;font-weight:700'>{level}</div>"
                f"<div style='color:#e0e8ff;font-size:11px;margin:3px 0'>{val:.5f}</div>"
                f"<div style='color:#5577aa;font-size:9px'>{dist:+.2f}%</div></div>",
                unsafe_allow_html=True)

with tab2:
    st.markdown("#### Signal Distribution")
    sig_counts = df["Signal"].value_counts()
    sig_cols = st.columns(min(5, len(sig_counts)+1))
    for col_w, (sig, cnt) in zip(sig_cols, sig_counts.items()):
        if "BUY" in sig:   pill = f'<span class="sig-buy">▲ {sig}: {cnt}</span>'
        elif "SELL" in sig: pill = f'<span class="sig-sell">▼ {sig}: {cnt}</span>'
        else:               pill = f'<span class="sig-hold">— {sig}: {cnt}</span>'
        col_w.markdown(pill, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Signal Score Timeline")
    fig_score = go.Figure(go.Bar(
        x=df.index, y=df["Signal_Score"],
        marker=dict(color=df["Signal_Score"],
                    colorscale=[[0,"#c0392b"],[0.25,"#e74c3c"],[0.5,"#2d3a55"],[0.75,"#27ae60"],[1,"#4af0c4"]],
                    cmin=-2, cmax=2),
    ))
    fig_score.update_layout(
        template="plotly_dark", paper_bgcolor="#0a0e17", plot_bgcolor="#0a0e17",
        height=160, margin=dict(l=10,r=10,t=10,b=10),
        font=dict(family="JetBrains Mono", size=9),
        xaxis=dict(gridcolor="#1e2d45"),
        yaxis=dict(gridcolor="#1e2d45", tickvals=[-2,-1,0,1,2],
                   ticktext=["STR SELL","SELL","HOLD","BUY","STR BUY"]),
    )
    st.plotly_chart(fig_score, use_container_width=True)

    st.markdown("#### Recent Signal Events")
    sig_df = df[df["Signal"] != "HOLD"][["Close","MA5","MA10","MA20","RSI","Signal"]].tail(30).round(5)
    if hasattr(sig_df.index, 'strftime'):
        sig_df.index = sig_df.index.strftime("%Y-%m-%d %H:%M")
    st.dataframe(sig_df.iloc[::-1], use_container_width=True, height=300)

    st.markdown("---")
    st.markdown("#### Signal Logic *(QuantConnect-style)*")
    st.markdown("""
    <div style='font-size:11px;line-height:2.2;color:#8899bb;font-family:JetBrains Mono,monospace'>
    <span style='color:#4af0c4'>▲ BUY</span>       Close &gt; PP &amp; Close &lt; R1 &amp; MA5 &gt; MA10 &gt; MA20 &amp; RSI &lt; 70 &amp; Close &gt; MA20<br>
    <span style='color:#4af0c4'>▲▲ STRONG BUY</span>  BUY + Close &gt; MA50 &amp; RSI &lt; 60<br>
    <span style='color:#f04a6a'>▼ SELL</span>       Close &lt; PP &amp; Close &gt; S1 &amp; MA5 &lt; MA10 &lt; MA20 &amp; RSI &gt; 30 &amp; Close &lt; MA20<br>
    <span style='color:#f04a6a'>▼▼ STRONG SELL</span>  SELL + Close &lt; MA50 &amp; RSI &gt; 40<br>
    <span style='color:#8899bb'>— HOLD</span>       No conditions met
    </div>""", unsafe_allow_html=True)

with tab3:
    if not show_fred:
        st.info("Enable **Show FRED macro** in the sidebar.")
    elif not fred_key:
        st.warning("Enter your **FRED API key** in the sidebar to load macro data.  "
                   "Get a free key at [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html)")
        st.table(pd.DataFrame([(k,v) for k,v in FRED_SERIES.items()], columns=["Indicator","Series ID"]))
    else:
        with st.spinner("Fetching FRED macro data..."):
            fred_data = {name: fetch_fred(sid, fred_key, 36) for name, sid in FRED_SERIES.items()}
        loaded = {k:v for k,v in fred_data.items() if v is not None and not v.empty}
        if not loaded:
            st.error("Could not fetch FRED data. Check your API key.")
        else:
            st.success(f"Loaded {len(loaded)}/{len(FRED_SERIES)} FRED series")
            fig_fred = build_fred_chart(loaded)
            if fig_fred:
                st.plotly_chart(fig_fred, use_container_width=True)

            macro_rows = []
            for name, df_f in loaded.items():
                if df_f is not None and not df_f.empty:
                    last_row = df_f.iloc[-1]
                    prev_row = df_f.iloc[-2] if len(df_f) > 1 else last_row
                    macro_rows.append({"Indicator":name,"Series":FRED_SERIES[name],
                                       "Date":last_row["date"].strftime("%Y-%m-%d"),
                                       "Value":round(last_row["value"],4),
                                       "Change":round(last_row["value"]-prev_row["value"],4)})
            if macro_rows:
                st.dataframe(pd.DataFrame(macro_rows), use_container_width=True, hide_index=True)

            st.markdown("#### Macro Regime Assessment")
            macro_score, notes = 0, []
            if "Fed Funds Rate" in loaded:
                rate = loaded["Fed Funds Rate"]["value"].iloc[-1]
                if rate > 4.5:
                    notes.append("🔴 High Fed Funds Rate — restrictive monetary policy"); macro_score -= 1
                else:
                    notes.append("🟢 Moderate Fed Funds Rate"); macro_score += 1
            if "10Y Treasury" in loaded and "2Y Treasury" in loaded:
                t10 = loaded["10Y Treasury"]["value"].iloc[-1]
                t2  = loaded["2Y Treasury"]["value"].iloc[-1]
                if t10 - t2 < 0:
                    notes.append("🔴 Inverted yield curve — recession signal"); macro_score -= 1
                else:
                    notes.append("🟢 Normal yield curve"); macro_score += 1
            overall = "MACRO BULLISH 🟢" if macro_score > 0 else ("MACRO BEARISH 🔴" if macro_score < 0 else "MACRO NEUTRAL ⚪")
            st.metric("Macro Regime", overall)
            for n in notes:
                st.markdown(f"<div style='font-size:11px;font-family:JetBrains Mono,monospace;color:#8899bb;margin:3px 0'>{n}</div>", unsafe_allow_html=True)

with tab4:
    st.markdown("#### Raw OHLCV + Indicators")
    display_df = df.copy().round(6)
    if hasattr(display_df.index, 'strftime'):
        display_df.index = display_df.index.strftime("%Y-%m-%d %H:%M")
    st.dataframe(display_df.iloc[::-1], use_container_width=True, height=500)
    csv = display_df.to_csv().encode("utf-8")
    st.download_button("⬇  Download CSV", csv, file_name=f"{symbol}_{interval}_signals.csv", mime="text/csv")

st.markdown("---")
st.markdown("<div style='font-size:10px;color:#334466;text-align:center'>"
            "Data: Yahoo Finance · FRED (St. Louis Fed) · Signal logic inspired by QuantConnect Lean · "
            "Not financial advice · Educational use only</div>", unsafe_allow_html=True)