import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from pathlib import Path
import warnings
from src.core.analyzer import TechnicalAnalyzer as analyzer
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EUR/USD Backtest Lab",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stApp { background-color: #0e1117; }
    .metric-card {
        background: #1a1d27;
        border: 1px solid #2d3148;
        border-radius: 10px;
        padding: 14px 18px;
        text-align: center;
    }
    .metric-label { font-size: 11px; color: #8b8fa8; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { font-size: 22px; font-weight: 700; margin-top: 4px; }
    .bull { color: #26a69a; }
    .bear { color: #ef5350; }
    .neutral { color: #ffa726; }
    .badge-bull { background:#1b3a38; color:#26a69a; padding:3px 10px; border-radius:12px; font-size:13px; font-weight:700; }
    .badge-bear { background:#3a1b1b; color:#ef5350; padding:3px 10px; border-radius:12px; font-size:13px; font-weight:700; }
    .badge-neutral { background:#3a2f1a; color:#ffa726; padding:3px 10px; border-radius:12px; font-size:13px; font-weight:700; }
    .idea-card {
        background: #1a1d27;
        border-left: 3px solid #26a69a;
        border-radius: 6px;
        padding: 10px 14px;
        margin-bottom: 8px;
        font-size: 13px;
    }
    .idea-card.bear { border-left-color: #ef5350; }
    .idea-card.neutral { border-left-color: #ffa726; }
    h1, h2, h3 { color: #e0e0e0 !important; }
    .stSelectbox label, .stDateInput label, .stSlider label { color: #8b8fa8; }
</style>
""", unsafe_allow_html=True)


# ─── DATA LOADING ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading data…")
def load_data(file_path):
    try:
        df = pd.read_csv(
            file_path,
            sep=";",
            header=None,
            names=["datetime", "open", "high", "low", "close", "volume"],
            parse_dates=["datetime"],
            date_format="%Y%m%d %H%M%S",
        )
        df = df.dropna(subset=["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)
        df["date"] = df["datetime"].dt.date
        # Rename columns to standard OHLC for the modular analyzer
        df.columns = [col.capitalize() if col != 'datetime' and col != 'date' else col for col in df.columns]
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def resample(df_day, tf):
    rules = {"M1": "1min", "M5": "5min", "M15": "15min", "M30": "30min", "H1": "1h"}
    rule = rules[tf]
    r = df_day.set_index("datetime").resample(rule).agg(
        Open=("Open", "first"),
        High=("High", "max"),
        Low=("Low", "min"),
        Close=("Close", "last"),
        Volume=("Volume", "sum"),
    ).dropna()
    r = r.reset_index()
    return r


def get_sessions(date):
    base = pd.Timestamp(date)
    sessions = {
        "Asian": (base.replace(hour=0, minute=0), base.replace(hour=8, minute=59)),
        "London": (base.replace(hour=7, minute=0), base.replace(hour=15, minute=59)),
        "NY": (base.replace(hour=13, minute=0), base.replace(hour=21, minute=59)),
    }
    return sessions


def session_analysis(df_day):
    results = {}
    for name, (s, e) in get_sessions(df_day["datetime"].dt.date.iloc[0]).items():
        seg = df_day[(df_day["datetime"] >= s) & (df_day["datetime"] <= e)]
        if len(seg) == 0:
            results[name] = None
            continue
        o = seg["Open"].iloc[0]
        c = seg["Close"].iloc[-1]
        h = seg["High"].max()
        l = seg["Low"].min()
        direction = "🟢 Bull" if c > o else ("🔴 Bear" if c < o else "⚪ Flat")
        results[name] = {
            "open": o, "close": c, "high": h, "low": l,
            "range_pips": round((h - l) * 10000, 1),
            "direction": direction,
        }
    return results


def compute_daily_bias(df_day):
    if len(df_day) < 5:
        return "Neutral", 50.0

    opens = df_day["Open"].iloc[0]
    closes = df_day["Close"].iloc[-1]
    highs = df_day["High"].max()
    lows = df_day["Low"].min()
    mid = (highs + lows) / 2
    pct_change = (closes - opens) / opens * 100

    # Check higher highs / lower lows structure
    h4 = df_day.set_index("datetime").resample("4h").agg(
        Open=("Open", "first"), High=("High", "max"),
        Low=("Low", "min"), Close=("Close", "last")
    ).dropna()

    bull_score = 0
    if closes > opens: bull_score += 2
    if closes > mid: bull_score += 1
    if len(h4) >= 2:
        if h4["Close"].iloc[-1] > h4["Close"].iloc[0]: bull_score += 2
        if h4["High"].iloc[-1] > h4["High"].iloc[-2]: bull_score += 1
        if h4["Low"].iloc[-1] > h4["Low"].iloc[-2]: bull_score += 1
    if pct_change > 0.1: bull_score += 1

    total = 8
    bull_pct = (bull_score / total) * 100
    if bull_pct >= 62:
        return "Bullish", bull_pct
    elif bull_pct <= 38:
        return "Bearish", 100 - bull_pct
    else:
        return "Neutral", 50.0


def generate_trading_ideas(df_day, bias, indicators):
    ideas = []
    c = df_day["Close"].iloc[-1]
    h = df_day["High"].max()
    l = df_day["Low"].min()
    rng = h - l
    mid = (h + l) / 2
    pp = (h + l + c) / 3  # pivot point
    r1 = 2 * pp - l
    r2 = pp + rng
    s1 = 2 * pp - h
    s2 = pp - rng

    last = indicators.iloc[-1] if len(indicators) > 0 else None
    rsi_val = last["RSI"] if last is not None and "RSI" in last.index else None
    ema20_val = last["EMA_20"] if last is not None and "EMA_20" in last.index else None

    # Key levels
    ideas.append({"type": "level", "icon": "📍", "text": f"Pivot Point: {pp:.5f} | R1: {r1:.5f} | R2: {r2:.5f}"})
    ideas.append({"type": "level", "icon": "📍", "text": f"S1: {s1:.5f} | S2: {s2:.5f} | Mid Range: {mid:.5f}"})

    # Bias-based ideas
    if bias == "Bullish":
        ideas.append({"type": "bull", "icon": "🟢",
                      "text": f"BUY on pullback to S1 ({s1:.5f}) or mid-range ({mid:.5f}). Target R1 ({r1:.5f})."})
    elif bias == "Bearish":
        ideas.append({"type": "bear", "icon": "🔴",
                      "text": f"SELL on bounce to R1 ({r1:.5f}) or mid-range ({mid:.5f}). Target S1 ({s1:.5f})."})
    else:
        ideas.append({"type": "neutral", "icon": "⚪",
                      "text": f"Range day. BUY near lows ({l:.5f}) / SELL near highs ({h:.5f}). Fade the extremes."})

    if rsi_val:
        if rsi_val > 70:
            ideas.append({"type": "bear", "icon": "⚠️", "text": f"RSI overbought ({rsi_val:.1f}). Caution on longs."})
        elif rsi_val < 30:
            ideas.append({"type": "bull", "icon": "⚠️", "text": f"RSI oversold ({rsi_val:.1f}). Caution on shorts."})

    return ideas, {"pp": pp, "r1": r1, "r2": r2, "s1": s1, "s2": s2}


def build_chart(df_tf, bias, levels, tf_label, show_ema, show_bb, show_sessions, selected_date):
    df_tf = analyzer.add_indicators(df_tf.copy())

    rows = 3
    row_heights = [0.55, 0.23, 0.22]
    subplot_titles = [f"Backtest {tf_label} — {selected_date}", "RSI (14)", "MACD"]
    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        row_heights=row_heights,
        vertical_spacing=0.04,
        subplot_titles=subplot_titles,
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df_tf["datetime"],
        open=df_tf["Open"], high=df_tf["High"],
        low=df_tf["Low"], close=df_tf["Close"],
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        name="Price", showlegend=False,
    ), row=1, col=1)

    if show_ema:
        for col, color in [("EMA_20", "#ff9800"), ("EMA_50", "#ab47bc")]:
            if col in df_tf.columns:
                fig.add_trace(
                    go.Scatter(x=df_tf["datetime"], y=df_tf[col], line=dict(color=color, width=1.2), name=col), row=1,
                    col=1)

    if show_bb:
        bb_styles = [("BB_Upper", "#4fc3f7", "dot"), ("BB_Middle", "#90a4ae", "solid"), ("BB_Lower", "#4fc3f7", "dot")]
        for col, color, dash in bb_styles:
            if col in df_tf.columns:
                fig.add_trace(
                    go.Scatter(x=df_tf["datetime"], y=df_tf[col],
                               line=dict(color=color, width=1, dash=dash),
                               name=col, opacity=0.7), row=1, col=1)
        if "BB_Upper" in df_tf.columns and "BB_Lower" in df_tf.columns:
            fig.add_trace(
                go.Scatter(x=pd.concat([df_tf["datetime"], df_tf["datetime"][::-1]]),
                           y=pd.concat([df_tf["BB_Upper"], df_tf["BB_Lower"][::-1]]),
                           fill="toself", fillcolor="rgba(79,195,247,0.05)",
                           line=dict(color="rgba(0,0,0,0)"), showlegend=False, name="BB Band"), row=1, col=1)

    # Key Levels
    x_range = [df_tf["datetime"].min(), df_tf["datetime"].max()]
    level_colors = {"pp": "#9e9e9e", "r1": "#ef5350", "r2": "#e53935", "s1": "#26a69a", "s2": "#00897b"}
    for key, price in levels.items():
        fig.add_shape(type="line", x0=x_range[0], x1=x_range[1], y0=price, y1=price,
                      line=dict(color=level_colors[key], width=0.8, dash="dot"), row=1, col=1)

    # Sessions
    if show_sessions:
        sess_colors = {"Asian": "rgba(255,193,7,0.06)", "London": "rgba(30,136,229,0.07)", "NY": "rgba(239,83,80,0.06)"}
        for name, (s, e) in get_sessions(selected_date).items():
            fig.add_vrect(x0=s, x1=e, fillcolor=sess_colors[name], layer="below", line_width=0, annotation_text=name,
                          row=1, col=1)

    # RSI
    if "RSI" in df_tf.columns:
        fig.add_trace(
            go.Scatter(x=df_tf["datetime"], y=df_tf["RSI"], line=dict(color="#7986cb", width=1.5), name="RSI"), row=2,
            col=1)

    # MACD
    if "MACD" in df_tf.columns:
        fig.add_trace(go.Bar(x=df_tf["datetime"], y=df_tf["MACD_Histogram"], name="MACD Hist"), row=3, col=1)

    fig.update_layout(height=720, plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font=dict(color="#c0c0c0", size=11),
                      xaxis_rangeslider_visible=False)
    return fig


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    st.markdown("## 📊 Modular Backtest Lab")

    with st.sidebar:
        st.markdown("### 📁 Data Source")
        data_file = st.text_input("Path to CSV (M1, semi-colon separated)", "DAT_NT_EURUSD_M1_2025.csv")

        if not Path(data_file).exists():
            st.warning("Please provide a valid path to the data file.")
            return

        df = load_data(data_file)
        if df.empty: return

        all_dates = sorted(df["date"].unique())
        date_options = [str(d) for d in all_dates]

        st.markdown("### ⚙️ Controls")
        selected_str = st.selectbox("📅 Select Trading Day", date_options, index=0)
        selected_date = pd.Timestamp(selected_str).date()
        tf = st.selectbox("⏱ Timeframe", ["M5", "M15", "M30", "H1", "M1"], index=1)
        show_ema = st.checkbox("EMA (20 / 50)", value=True)
        show_bb = st.checkbox("Bollinger Bands", value=False)
        show_sessions = st.checkbox("Session Shading", value=True)

    df_day = df[df["date"] == selected_date].copy()
    if df_day.empty:
        st.warning("No data for selected date.")
        return

    df_tf = resample(df_day, tf) if tf != "M1" else df_day.copy()
    df_tf_ind = analyzer.add_indicators(df_tf.copy())

    bias, confidence = compute_daily_bias(df_day)
    sess = session_analysis(df_day)
    ideas, levels = generate_trading_ideas(df_day, bias, df_tf_ind)

    st.markdown(f"### {selected_str} | Bias: {bias} ({confidence:.0f}%)")

    # ── Session summary row ───────────────────────────────────────────────────
    sess_cols = st.columns(3)
    for i, (name, data) in enumerate(sess.items()):
        with sess_cols[i]:
            if data:
                st.markdown(f"**{name}** {data['direction']}")
                st.caption(f"Range: {data['range_pips']} pips | H: {data['high']:.5f} L: {data['low']:.5f}")
            else:
                st.markdown(f"**{name}** — no data")

    chart_col, ideas_col = st.columns([3, 1])
    with chart_col:
        fig = build_chart(df_tf, bias, levels, tf, show_ema, show_bb, show_sessions, selected_date)
        st.plotly_chart(fig, use_container_width=True)

    with ideas_col:
        st.markdown("#### 💡 Trading Ideas")
        for idea in ideas:
            st.info(f"{idea['icon']} {idea['text']}")


if __name__ == "__main__":
    main()