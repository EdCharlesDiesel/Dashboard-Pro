import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime

st.set_page_config(
    page_title="Weekly EMA · Trading System",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    /* Theme-adaptive layout vars */
    .stApp {
      --border: color-mix(in srgb, var(--text-color) 12%, transparent);
      --muted:  color-mix(in srgb, var(--text-color) 55%, transparent);
    }
    html,body,[class*="css"]{font-family:'Inter',sans-serif;}
    .stApp{background:var(--background-color);}
    section[data-testid="stSidebar"]{background:var(--secondary-background-color)!important;border-right:1px solid var(--border,#21262d);}
    .card{background:var(--secondary-background-color);border:1px solid var(--border,#21262d);border-radius:12px;padding:18px 20px;margin-bottom:14px;}
    .card-header{font-size:12px;font-weight:600;letter-spacing:.09em;text-transform:uppercase;color:#8b949e;margin-bottom:14px;}
    .hero{background:linear-gradient(135deg,#0d1117 0%,#161b22 50%,#0d1117 100%);border:1px solid #21262d;border-radius:16px;padding:24px 32px;margin-bottom:20px;position:relative;overflow:hidden;}
    .hero::before{content:'';position:absolute;top:-40%;right:-5%;width:300px;height:300px;background:radial-gradient(circle,rgba(56,139,253,.07) 0%,transparent 70%);border-radius:50%;}

    .metric-box{background:var(--background-color);border:1px solid var(--border,#21262d);border-radius:8px;padding:12px 14px;text-align:center;}
    .metric-value{font-size:20px;font-weight:700;color:#c9d1d9;}
    .metric-label{font-size:10px;color:#8b949e;margin-top:2px;font-weight:500;letter-spacing:.05em;text-transform:uppercase;}

    /* Pair grid cards */
    .ema-card{background:#161b22;border:1px solid #21262d;border-radius:10px;padding:14px 15px;margin-bottom:8px;}
    .ema-card-bull{border-left:4px solid #3fb950;}
    .ema-card-bear{border-left:4px solid #f85149;}
    .ema-card-neut{border-left:4px solid #e3b341;}
    .ema-card-na  {border-left:4px solid #30363d;opacity:.5;}

    .pair-label{font-size:15px;font-weight:700;color:#e6edf3;}
    .price-tag{font-size:12px;font-family:monospace;color:#8b949e;margin-left:8px;}

    .badge{border-radius:20px;padding:4px 13px;font-size:12px;font-weight:700;display:inline-block;}
    .badge-bull {background:#0d4a2f;color:#3fb950;border:1px solid #238636;}
    .badge-bear {background:#4a0d0d;color:#f85149;border:1px solid #8b2d2d;}
    .badge-neut {background:#21262d;color:#e3b341;border:1px solid #9e6a03;}
    .badge-na   {background:#21262d;color:#484f58;border:1px solid #30363d;}

    .align-pill{border-radius:6px;padding:2px 9px;font-size:11px;font-weight:700;display:inline-block;margin-left:6px;}
    .align-yes{background:#0d2f1a;color:#3fb950;border:1px solid #238636;}
    .align-no {background:#2f0d0d;color:#f85149;border:1px solid #8b2d2d;}

    .ema-row{display:flex;justify-content:space-between;align-items:center;font-size:12px;padding:3px 0;}
    .ema-lbl{color:#8b949e;}
    .ema-val{font-family:monospace;font-weight:600;color:#e6edf3;}
    .ema-above{color:#3fb950;}
    .ema-below{color:#f85149;}

    .slope-arrow-up  {color:#3fb950;font-size:14px;font-weight:900;}
    .slope-arrow-down{color:#f85149;font-size:14px;font-weight:900;}
    .slope-arrow-flat{color:#e3b341;font-size:14px;font-weight:900;}

    .prog-track{background:var(--border,#21262d);border-radius:6px;height:5px;overflow:hidden;margin:4px 0;}

    [data-testid="stSidebarNav"]{display:none;}
    #MainMenu,footer,header{visibility:hidden;}
    [data-testid="stSidebarCollapsedControl"]{visibility:visible !important;}
    .block-container{padding-top:1.5rem;max-width:1420px;}
</style>
""", unsafe_allow_html=True)

# ── Instruments ────────────────────────────────────────────────────────────────
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
    "🥇 Gold":  {"ticker": "GC=F",    "pip_size": 0.10},
    "🥈 Silver":  {"ticker": "SI=F",  "pip_size": 0.01},
    "🪙 Platinum":{"ticker": "PL=F",  "pip_size": 0.10},
}

EMA_PERIODS = [10, 20, 50]   # Weekly EMAs calculated
SLOPE_BARS  = 3              # How many bars back to measure slope

# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════
def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def slope_pct(series: pd.Series, n: int = SLOPE_BARS) -> float:
    """% change of EMA over last n bars — positive = rising."""
    if len(series) < n + 1:
        return 0.0
    return float((series.iloc[-1] - series.iloc[-(n+1)]) / series.iloc[-(n+1)] * 100)

def slope_label(pct: float):
    if   pct >  0.10: return "↑ Rising",  "slope-arrow-up",   "Bullish"
    elif pct < -0.10: return "↓ Falling", "slope-arrow-down", "Bearish"
    else:             return "→ Flat",     "slope-arrow-flat", "Neutral"

@st.cache_data(ttl=600, show_spinner=False)
def fetch_weekly(ticker: str, pip_size: float, lookback: str = "2y"):
    try:
        df = yf.download(ticker, period=lookback, interval="1wk",
                         progress=False, auto_adjust=True)
        if df.empty or len(df) < 55:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]

        close = df["Close"]

        # Calculate all EMAs
        for p in EMA_PERIODS:
            df[f"ema{p}"] = ema(close, p)

        df = df.dropna(subset=[f"ema{p}" for p in EMA_PERIODS])
        if df.empty:
            return None

        latest = df.iloc[-1]
        price  = float(latest["Close"])

        ema_data = {}
        for p in EMA_PERIODS:
            ev      = float(latest[f"ema{p}"])
            sl      = slope_pct(df[f"ema{p}"])
            sl_lbl, sl_cls, sl_bias = slope_label(sl)
            above   = price > ev
            dist_pct= round((price - ev) / ev * 100, 3)

            ema_data[p] = {
                "value":     round(ev, 5),
                "slope_pct": round(sl, 4),
                "slope_lbl": sl_lbl,
                "slope_cls": sl_cls,
                "bias":      sl_bias,
                "above":     above,
                "dist_pct":  dist_pct,
            }

        # Overall weekly bias — majority vote of EMA slopes
        biases = [ema_data[p]["bias"] for p in EMA_PERIODS]
        bull_count = biases.count("Bullish")
        bear_count = biases.count("Bearish")
        if   bull_count >= 2: overall = "Bullish"
        elif bear_count >= 2: overall = "Bearish"
        else:                 overall = "Neutral"

        # Price vs EMA stack alignment
        ema_vals  = [ema_data[p]["value"] for p in EMA_PERIODS]
        stack_bull = price > ema_vals[0] > ema_vals[1] > ema_vals[2]   # price > EMA10 > EMA20 > EMA50
        stack_bear = price < ema_vals[0] < ema_vals[1] < ema_vals[2]

        return {
            "price":      round(price, 5),
            "ema_data":   ema_data,
            "overall":    overall,
            "stack_bull": stack_bull,
            "stack_bear": stack_bear,
            "history":    df.tail(80),
        }
    except Exception:
        return None

# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 📉 Weekly EMA")
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

    trade_direction = st.radio("Your Trade Direction", ["LONG", "SHORT"], horizontal=True)
    focus_pair      = st.selectbox("Focus Pair (Chart)", list(INSTRUMENTS.keys()), index=0)
    lookback        = st.selectbox("Lookback Period", ["1y","2y","3y","5y"], index=1)

    st.markdown("---")
    ema_primary = st.selectbox("Primary EMA for Alignment Check",
                               [f"EMA {p}" for p in EMA_PERIODS], index=1)
    primary_period = int(ema_primary.split()[1])

    show_filter = st.radio("Show Pairs",
                           ["All", "✅ Aligned with direction", "❌ Against direction"], index=0)

    st.markdown("---")
    if st.button("🔄 Refresh Data", use_container_width=True, type="primary"):
        st.cache_data.clear()
        st.rerun()

# ══════════════════════════════════════════════════════════════════
# FETCH ALL
# ══════════════════════════════════════════════════════════════════
bar = st.progress(0, text="📡 Loading weekly EMA data…")
results = {}
for idx, (pair, cfg) in enumerate(INSTRUMENTS.items()):
    bar.progress((idx + 1) / len(INSTRUMENTS), text=f"📡 Fetching weekly data — {pair}…")
    results[pair] = fetch_weekly(cfg["ticker"], cfg["pip_size"], lookback)
bar.empty()

loaded  = {p: d for p, d in results.items() if d}
bull    = {p: d for p, d in loaded.items() if d["overall"] == "Bullish"}
bear    = {p: d for p, d in loaded.items() if d["overall"] == "Bearish"}
neutral = {p: d for p, d in loaded.items() if d["overall"] == "Neutral"}

# Alignment with user's direction
aligned     = bull if trade_direction == "LONG" else bear
not_aligned = bear if trade_direction == "LONG" else bull

# ══════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════
dir_icon  = "🔼" if trade_direction == "LONG" else "🔽"
dir_color = "#3fb950" if trade_direction == "LONG" else "#f85149"
st.markdown(f"""
<div class="hero">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:12px;">
    <div>
      <div style="font-size:26px;font-weight:700;color:#e6edf3;">📉 Weekly EMA Alignment</div>
      <div style="color:#8b949e;font-size:14px;margin-top:6px;">
        EMA {', '.join(str(p) for p in EMA_PERIODS)} · Weekly Timeframe · Checklist Item #5
      </div>
      <div style="font-size:13px;color:#388bfd;font-weight:500;margin-top:4px;">
        🕐 {datetime.now().strftime('%A, %d %B %Y  |  %H:%M')} · {lookback} lookback · Weekly bars
      </div>
    </div>
    <div style="display:flex;gap:10px;flex-wrap:wrap;align-items:flex-start;">
      <div class="metric-box">
        <div style="font-size:14px;font-weight:700;color:{dir_color};">{dir_icon} {trade_direction}</div>
        <div class="metric-label">Your Direction</div>
      </div>
      <div class="metric-box" style="min-width:80px;">
        <div class="metric-value" style="color:#3fb950;">{len(bull)}</div>
        <div class="metric-label">🟢 Bullish</div>
      </div>
      <div class="metric-box" style="min-width:80px;">
        <div class="metric-value" style="color:#f85149;">{len(bear)}</div>
        <div class="metric-label">🔴 Bearish</div>
      </div>
      <div class="metric-box" style="min-width:80px;">
        <div class="metric-value" style="color:#e3b341;">{len(neutral)}</div>
        <div class="metric-label">🟡 Neutral</div>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# KPI strip
# ══════════════════════════════════════════════════════════════════
k1,k2,k3,k4,k5,k6 = st.columns(6)
kpis = [
    (k1, len(aligned),     f"✅ Aligned ({trade_direction})", "#3fb950"),
    (k2, len(not_aligned), f"❌ Against ({trade_direction})", "#f85149"),
    (k3, len(neutral),     "⚪ Neutral",                      "#e3b341"),
    (k4, sum(1 for d in loaded.values() if d["stack_bull"]),  "Full Bull Stack", "#3fb950"),
    (k5, sum(1 for d in loaded.values() if d["stack_bear"]),  "Full Bear Stack", "#f85149"),
    (k6, len(loaded),      "Pairs Loaded",                   "#c9d1d9"),
]
for col, val, lbl, color in kpis:
    with col:
        st.markdown(f'<div class="metric-box"><div class="metric-value" style="color:{color};">{val}</div><div class="metric-label">{lbl}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ══════════════════════════════════════════════════════════════════
left, right = st.columns([2, 3], gap="large")

# ── Which pairs to show ────────────────────────────────────────────────────────
def passes_filter(pair, data):
    if show_filter == "✅ Aligned with direction":
        return data["overall"] == ("Bullish" if trade_direction == "LONG" else "Bearish")
    elif show_filter == "❌ Against direction":
        return data["overall"] == ("Bearish" if trade_direction == "LONG" else "Bullish")
    return True

# Sort: aligned first, then neutral, then against
def sort_key(pair):
    d = loaded.get(pair)
    if not d:
        return 3
    if d["overall"] == ("Bullish" if trade_direction == "LONG" else "Bearish"):
        return 0
    if d["overall"] == "Neutral":
        return 1
    return 2

sorted_pairs = sorted(INSTRUMENTS.keys(), key=sort_key)

# ══════════════════════════════════════════════════════════════════
# LEFT: PAIR CARDS
# ══════════════════════════════════════════════════════════════════
with left:
    st.markdown("### 🗂️ All Pairs — Weekly EMA Status")

    for pair in sorted_pairs:
        d = loaded.get(pair)

        if d is None:
            st.markdown(f"""
            <div class="ema-card ema-card-na">
              <span class="pair-label">{pair}</span>
              <span class="badge badge-na" style="float:right;">⚠️ No Data</span>
            </div>
            """, unsafe_allow_html=True)
            continue

        if not passes_filter(pair, d):
            continue

        overall    = d["overall"]
        is_aligned = overall == ("Bullish" if trade_direction == "LONG" else "Bearish")
        card_cls   = "ema-card-bull" if overall == "Bullish" else \
            "ema-card-bear" if overall == "Bearish" else "ema-card-neut"
        badge_cls  = "badge-bull" if overall == "Bullish" else \
            "badge-bear" if overall == "Bearish" else "badge-neut"
        overall_icon = "🟢" if overall == "Bullish" else "🔴" if overall == "Bearish" else "🟡"
        align_html = f'<span class="align-pill align-yes">✅ Aligned</span>' if is_aligned \
            else f'<span class="align-pill align-no">❌ Against</span>'

        # Stack alignment
        stack_ok   = d["stack_bull"] if trade_direction == "LONG" else d["stack_bear"]
        stack_html = '<span style="font-size:10px;color:#3fb950;margin-left:6px;">⬆ Full stack</span>' \
            if stack_ok else ""

        # EMA rows
        ema_rows_html = ""
        for p in EMA_PERIODS:
            ev   = d["ema_data"][p]
            bold = "font-weight:800;" if p == primary_period else ""
            pri  = " ★" if p == primary_period else ""
            above_cls = "ema-above" if ev["above"] else "ema-below"
            above_txt = f"Price {'above' if ev['above'] else 'below'}"
            dist_col  = "#3fb950" if ev["dist_pct"] > 0 else "#f85149"
            ema_rows_html += f"""
            <div class="ema-row" style="{bold}">
              <span class="ema-lbl">EMA {p}{pri}</span>
              <span class="ema-val">{ev['value']:.5f}</span>
              <span class="{above_cls}">{above_txt}</span>
              <span style="font-size:11px;color:{dist_col};">{ev['dist_pct']:+.2f}%</span>
              <span class="{ev['slope_cls']}">{ev['slope_lbl']}</span>
            </div>
            """

        # Highlight if it's the focus pair
        focus_border = "border-color:#388bfd!important;" if pair == focus_pair else ""

        st.markdown(f"""
        <div class="ema-card {card_cls}" style="{focus_border}">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
            <div>
              <span class="pair-label">{pair}</span>
              <span class="price-tag">@ {d['price']:.5f}</span>
              {align_html}{stack_html}
            </div>
            <span class="badge {badge_cls}">{overall_icon} {overall}</span>
          </div>
          {ema_rows_html}
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# RIGHT: FOCUS PAIR CHART
# ══════════════════════════════════════════════════════════════════
with right:
    d = loaded.get(focus_pair)
    st.markdown(f"### 📈 {focus_pair} — Weekly EMA Chart")

    if d is None:
        st.warning(f"No weekly data available for {focus_pair}. Try refreshing.")
    else:
        overall   = d["overall"]
        is_aligned= overall == ("Bullish" if trade_direction == "LONG" else "Bearish")
        ok_color  = "#3fb950" if is_aligned else "#f85149"
        ok_icon   = "✅" if is_aligned else "❌"
        ok_txt    = f"Weekly EMA is {overall} — {'Aligned' if is_aligned else 'Against'} your {trade_direction}"

        # Stack note
        stack_ok  = d["stack_bull"] if trade_direction == "LONG" else d["stack_bear"]
        stack_note = "· ⬆ Full EMA stack alignment" if stack_ok else "· ⚠️ EMA stack not fully aligned"

        st.markdown(f"""
        <div style="background:{'#0d2f1a' if is_aligned else '#2f0d0d'};
             border:1px solid {'#238636' if is_aligned else '#8b2d2d'};
             border-radius:10px;padding:14px 18px;margin-bottom:16px;">
          <div style="font-size:16px;font-weight:700;color:{ok_color};">{ok_icon} {ok_txt}</div>
          <div style="font-size:12px;color:#8b949e;margin-top:4px;">
            Primary EMA {primary_period} slope: {d['ema_data'][primary_period]['slope_lbl']} &nbsp;{stack_note}
          </div>
        </div>
        """, unsafe_allow_html=True)

        hist = d["history"]
        cfg  = INSTRUMENTS[focus_pair]

        # ── Candlestick + EMA chart ────────────────────────────────
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.72, 0.28],
            vertical_spacing=0.04,
            subplot_titles=[
                f"{focus_pair} — Weekly Candlestick + EMA {', '.join(str(p) for p in EMA_PERIODS)}",
                "EMA Slope % (5-bar rate of change)",
            ],
        )

        # Candlesticks
        fig.add_trace(go.Candlestick(
            x=hist.index,
            open=hist["Open"], high=hist["High"],
            low=hist["Low"],   close=hist["Close"],
            name="Price",
            increasing=dict(line=dict(color="#3fb950"), fillcolor="#0d2f1a"),
            decreasing=dict(line=dict(color="#f85149"), fillcolor="#2f0d0d"),
            hoverinfo="x+y",
        ), row=1, col=1)

        # EMA lines
        ema_colors = {10: "#f0c040", 20: "#388bfd", 50: "#a371f7"}
        ema_widths = {10: 1.5,       20: 2.5,       50: 2.0}
        for p in EMA_PERIODS:
            fig.add_trace(go.Scatter(
                x=hist.index, y=hist[f"ema{p}"],
                name=f"EMA {p}",
                mode="lines",
                line=dict(color=ema_colors[p], width=ema_widths[p]),
                hovertemplate=f"EMA{p}: %{{y:.5f}}<extra></extra>",
            ), row=1, col=1)

        # Current price line
        fig.add_hline(
            y=d["price"], line_dash="dot",
            line_color="#e3b341", line_width=1,
            annotation_text=f"Now {d['price']:.5f}",
            annotation_font_color="#e3b341",
            annotation_position="right",
            row=1, col=1,
        )

        # EMA slope panel — show slope % of primary EMA
        slopes = []
        for i in range(SLOPE_BARS, len(hist)):
            seg = hist[f"ema{primary_period}"].iloc[i-SLOPE_BARS:i+1]
            if len(seg) > SLOPE_BARS:
                s = float((seg.iloc[-1] - seg.iloc[0]) / seg.iloc[0] * 100)
            else:
                s = 0.0
            slopes.append(s)

        slope_dates  = hist.index[SLOPE_BARS:]
        slope_colors = ["#3fb950" if s > 0.10 else "#f85149" if s < -0.10 else "#e3b341"
                        for s in slopes]

        fig.add_trace(go.Bar(
            x=slope_dates, y=slopes,
            name=f"EMA{primary_period} Slope",
            marker_color=slope_colors,
            hovertemplate="Slope: %{y:.3f}%<extra></extra>",
        ), row=2, col=1)

        fig.add_hline(y=0, line_dash="dot", line_color="#484f58", line_width=1, row=2, col=1)
        fig.add_hline(y=0.10,  line_dash="dot", line_color="#238636", line_width=1, row=2, col=1,
                      annotation_text="Rising",   annotation_font_color="#238636", annotation_position="right")
        fig.add_hline(y=-0.10, line_dash="dot", line_color="#8b2d2d", line_width=1, row=2, col=1,
                      annotation_text="Falling",  annotation_font_color="#8b2d2d", annotation_position="right")

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1117",
            legend=dict(font=dict(color="#8b949e", size=11), bgcolor="#161b22",
                        bordercolor="#21262d", orientation="h", x=0, y=1.06),
            margin=dict(l=10, r=80, t=30, b=10),
            height=530,
            hovermode="x unified",
            xaxis_rangeslider_visible=False,
        )
        for row in [1, 2]:
            fig.update_xaxes(showgrid=False, tickfont=dict(color="#8b949e", size=9),
                             linecolor="#21262d", row=row, col=1)
            fig.update_yaxes(showgrid=True, gridcolor="#21262d",
                             tickfont=dict(color="#8b949e", size=9), row=row, col=1)
        fig.update_annotations(font_color="#8b949e", font_size=11)

        st.plotly_chart(fig, use_container_width=True, config=dict(displayModeBar=False))

        # ── EMA Snapshot Table ─────────────────────────────────────
        st.markdown("### 📋 EMA Snapshot")
        snap_rows = []
        for p in EMA_PERIODS:
            ev = d["ema_data"][p]
            snap_rows.append({
                "EMA":       f"EMA {p}{'  ★' if p == primary_period else ''}",
                "Value":     ev["value"],
                "Price vs":  f"{'Above ▲' if ev['above'] else 'Below ▼'}",
                "Dist %":    ev["dist_pct"],
                "Slope %":   ev["slope_pct"],
                "Direction": ev["slope_lbl"],
                "Bias":      ev["bias"],
            })
        df_snap = pd.DataFrame(snap_rows)
        st.dataframe(
            df_snap, use_container_width=True, hide_index=True,
            column_config={
                "EMA":       st.column_config.TextColumn("EMA"),
                "Value":     st.column_config.NumberColumn("Value",    format="%.5f"),
                "Price vs":  st.column_config.TextColumn("Price vs EMA"),
                "Dist %":    st.column_config.NumberColumn("Distance %", format="%+.3f"),
                "Slope %":   st.column_config.NumberColumn("Slope %",  format="%+.4f"),
                "Direction": st.column_config.TextColumn("Direction"),
                "Bias":      st.column_config.TextColumn("Bias"),
            }
        )

        # ── Alignment Summary for all pairs ───────────────────────
        st.markdown("### 🗺️ All-Pair EMA Alignment Summary")

        summary_rows = []
        for pair2 in INSTRUMENTS:
            d2 = loaded.get(pair2)
            if not d2:
                continue
            is_al = d2["overall"] == ("Bullish" if trade_direction == "LONG" else "Bearish")
            primary_ev = d2["ema_data"][primary_period]
            summary_rows.append({
                "Pair":      pair2,
                "Overall":   d2["overall"],
                "Aligned":   "✅ Yes" if is_al else "❌ No",
                f"EMA{primary_period}": primary_ev["value"],
                "Price vs":  "Above" if primary_ev["above"] else "Below",
                "Slope":     primary_ev["slope_lbl"],
                "Stack":     "✅ Full" if (d2["stack_bull"] if trade_direction=="LONG" else d2["stack_bear"]) else "—",
                "Price":     d2["price"],
            })

        df_sum = pd.DataFrame(summary_rows)
        st.dataframe(
            df_sum, use_container_width=True, hide_index=True,
            column_config={
                "Pair":      st.column_config.TextColumn("Pair"),
                "Overall":   st.column_config.TextColumn("Weekly Bias"),
                "Aligned":   st.column_config.TextColumn("Aligned?"),
                f"EMA{primary_period}": st.column_config.NumberColumn(f"EMA{primary_period}", format="%.5f"),
                "Price vs":  st.column_config.TextColumn("Price vs EMA"),
                "Slope":     st.column_config.TextColumn("Slope"),
                "Stack":     st.column_config.TextColumn("Full Stack"),
                "Price":     st.column_config.NumberColumn("Price", format="%.5f"),
            }
        )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;color:#484f58;font-size:11px;margin-top:32px;padding-top:16px;border-top:1px solid #21262d;">
  📉 Weekly EMA · EMA 10 / 20 / 50 · Weekly bars · Slope = 3-bar rate of change · Not financial advice
</div>
""", unsafe_allow_html=True)