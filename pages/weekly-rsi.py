import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime

st.set_page_config(
    page_title="Weekly RSI · Trading System",
    page_icon="📡",
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

    /* RSI gauge colours */
    .rsi-ob   {background:#2f0d0d;border-left:4px solid #f85149;}   /* overbought  */
    .rsi-os   {background:#0d1a2f;border-left:4px solid #388bfd;}   /* oversold    */
    .rsi-bull {background:#0d1e0d;border-left:4px solid #3fb950;}   /* bullish room*/
    .rsi-bear {background:#1e0d0d;border-left:4px solid #e3b341;}   /* bearish room*/
    .rsi-neut {background:#161b22;border-left:4px solid #30363d;}   /* neutral     */

    .pair-card{border-radius:12px;padding:14px 16px;margin-bottom:9px;border:1px solid #21262d;}
    .pair-name{font-size:15px;font-weight:700;color:#e6edf3;}

    .badge{border-radius:6px;padding:3px 10px;font-size:11px;font-weight:700;display:inline-block;}
    .badge-room   {background:#0d2f1a;color:#3fb950;border:1px solid #238636;}
    .badge-ob     {background:#2f0d0d;color:#f85149;border:1px solid #8b2d2d;}
    .badge-os     {background:#0d1a2f;color:#388bfd;border:1px solid #1f6feb;}
    .badge-warn   {background:#2f1f0d;color:#e3b341;border:1px solid #9e6a03;}
    .badge-neut   {background:#21262d;color:#8b949e;border:1px solid #30363d;}

    /* RSI gauge bar */
    .gauge-track{position:relative;background:linear-gradient(90deg,
        #1f4f8a 0%,#1f4f8a 30%,
        #21262d 30%,#21262d 40%,
        #0d2f1a 40%,#0d2f1a 60%,
        #21262d 60%,#21262d 70%,
        #4a0d0d 70%,#4a0d0d 100%);
        border-radius:8px;height:10px;margin:8px 0;}
    .gauge-needle{position:absolute;top:-3px;width:4px;height:16px;
        background:#e6edf3;border-radius:2px;transform:translateX(-50%);}

    .metric-box{background:var(--background-color);border:1px solid var(--border,#21262d);border-radius:8px;padding:12px 14px;text-align:center;}
    .metric-value{font-size:20px;font-weight:700;color:#c9d1d9;}
    .metric-label{font-size:10px;color:#8b949e;margin-top:2px;font-weight:500;letter-spacing:.05em;text-transform:uppercase;}

    .zone-label{font-size:9px;color:#484f58;display:flex;justify-content:space-between;margin-top:2px;}

    [data-testid="stSidebarNav"]{display:none;}
    #MainMenu,footer,header{visibility:hidden;}
    [data-testid="stSidebarCollapsedControl"]{visibility:visible !important;}
    .block-container{padding-top:1.5rem;max-width:1400px;}
</style>
""", unsafe_allow_html=True)

# ── Instruments ────────────────────────────────────────────────────────────────
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

# ── RSI Calculation ────────────────────────────────────────────────────────────
def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

# ── Zone classifier ────────────────────────────────────────────────────────────
def classify_rsi(rsi_val: float, ob: float, os: float, direction: str) -> dict:
    if rsi_val >= ob:
        return {"zone": "Overbought", "card_cls": "rsi-ob", "badge_cls": "badge-ob",
                "color": "#f85149", "icon": "🔴",
                "has_room": False,
                "detail": f"RSI {rsi_val:.1f} ≥ {ob} — exhausted, do NOT long"}
    if rsi_val <= os:
        return {"zone": "Oversold", "card_cls": "rsi-os", "badge_cls": "badge-os",
                "color": "#388bfd", "icon": "🔵",
                "has_room": False,
                "detail": f"RSI {rsi_val:.1f} ≤ {os} — exhausted, do NOT short"}
    # Has room — now check direction fit
    if direction == "LONG":
        if rsi_val >= 50:
            return {"zone": "Bullish Room", "card_cls": "rsi-bull", "badge_cls": "badge-room",
                    "color": "#3fb950", "icon": "✅",
                    "has_room": True,
                    "detail": f"RSI {rsi_val:.1f} — above 50, room to run higher"}
        else:
            return {"zone": "Weak (below 50)", "card_cls": "rsi-bear", "badge_cls": "badge-warn",
                    "color": "#e3b341", "icon": "⚠️",
                    "has_room": True,
                    "detail": f"RSI {rsi_val:.1f} — has room but below 50, weak long"}
    else:  # SHORT
        if rsi_val <= 50:
            return {"zone": "Bearish Room", "card_cls": "rsi-bull", "badge_cls": "badge-room",
                    "color": "#3fb950", "icon": "✅",
                    "has_room": True,
                    "detail": f"RSI {rsi_val:.1f} — below 50, room to fall further"}
        else:
            return {"zone": "Weak (above 50)", "card_cls": "rsi-bear", "badge_cls": "badge-warn",
                    "color": "#e3b341", "icon": "⚠️",
                    "has_room": True,
                    "detail": f"RSI {rsi_val:.1f} — has room but above 50, weak short"}

# ── Data fetch ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=600, show_spinner=False)
def fetch_rsi_data(ticker: str, rsi_period: int, weeks: int):
    try:
        df = yf.download(ticker, period=f"{weeks * 8}d",
                         interval="1wk", progress=False, auto_adjust=True)
        if df.empty or len(df) < rsi_period + 5:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]

        df = df[["Open","High","Low","Close","Volume"]].dropna(subset=["Close"])
        df["rsi"]      = calc_rsi(df["Close"], rsi_period)
        df["rsi_ma"]   = df["rsi"].rolling(5).mean()   # smoothed RSI
        df["close_ema21"] = df["Close"].ewm(span=21, adjust=False).mean()

        df = df.dropna(subset=["rsi"])
        if df.empty:
            return None

        latest   = df.iloc[-1]
        prev     = df.iloc[-2] if len(df) >= 2 else latest
        rsi_now  = float(latest["rsi"])
        rsi_prev = float(prev["rsi"])
        rsi_chg  = round(rsi_now - rsi_prev, 2)

        # Distance to OB/OS in RSI points
        dist_ob = round(70 - rsi_now, 1)
        dist_os = round(rsi_now - 30, 1)

        # Divergence: price going up but RSI going down (bearish div) or vice versa
        price_chg = float(latest["Close"]) - float(prev["Close"])
        div = None
        if price_chg > 0 and rsi_chg < -2:
            div = "⚠️ Bearish Divergence"
        elif price_chg < 0 and rsi_chg > 2:
            div = "⚠️ Bullish Divergence"

        return {
            "rsi":       round(rsi_now, 2),
            "rsi_prev":  round(rsi_prev, 2),
            "rsi_chg":   rsi_chg,
            "dist_ob":   dist_ob,
            "dist_os":   dist_os,
            "close":     round(float(latest["Close"]), 5),
            "divergence": div,
            "history":   df,
        }
    except Exception:
        return None

# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 📡 Weekly RSI")
    st.page_link("daily-trading-checklist.py", label="00. Checklist", icon="📋")
    st.page_link("pages/setup-ranker.py",      label="01. Setup Ranker",      icon="🎰")
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
    st.page_link("pages/backtest-workflow.py",    label="19. Backtest Workflow",    icon="🧪")
    st.divider()

    direction = st.radio("🎯 Trade Direction", ["LONG", "SHORT"], horizontal=True)

    rsi_period = st.select_slider("RSI Period", options=[7, 9, 14, 21], value=14)

    col_ob, col_os = st.columns(2)
    with col_ob: ob_level = st.number_input("Overbought", value=70, min_value=60, max_value=85, step=1)
    with col_os: os_level = st.number_input("Oversold",   value=30, min_value=15, max_value=40, step=1)

    focus_pair = st.selectbox("Focus Pair (Detail)", list(INSTRUMENTS.keys()), index=0)
    lookback_w = st.slider("History (weeks)", 52, 260, 104, 26)

    sort_by = st.radio("Sort By", ["RSI Value", "Has Room First", "Pair Name"], index=1)
    show_only_room = st.checkbox("Show only ✅ Has Room", value=False)

    st.divider()
    if st.button("🔄 Refresh RSI Data", use_container_width=True, type="primary"):
        st.cache_data.clear()
        st.rerun()

# ══════════════════════════════════════════════════════════════════
# FETCH
# ══════════════════════════════════════════════════════════════════
prog = st.progress(0, text="📡 Fetching weekly RSI for all pairs…")
all_data = {}
for idx, (pair, cfg) in enumerate(INSTRUMENTS.items()):
    prog.progress((idx+1)/len(INSTRUMENTS), text=f"📡 {pair} ({idx+1}/{len(INSTRUMENTS)})…")
    all_data[pair] = fetch_rsi_data(cfg["ticker"], rsi_period, lookback_w)
prog.empty()

loaded   = {p: d for p, d in all_data.items() if d is not None}
classes  = {p: classify_rsi(d["rsi"], ob_level, os_level, direction) for p, d in loaded.items()}

has_room = [p for p, c in classes.items() if c["has_room"]]
obought  = [p for p, c in classes.items() if c["zone"] == "Overbought"]
osold    = [p for p, c in classes.items() if c["zone"] == "Oversold"]
divs     = [p for p, d in loaded.items() if d["divergence"]]

# ══════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════
dir_color = "#3fb950" if direction == "LONG" else "#f85149"
dir_icon  = "🔼" if direction == "LONG" else "🔽"

st.markdown(f"""
<div class="hero">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:12px;">
    <div>
      <div style="font-size:26px;font-weight:700;color:#e6edf3;">📡 Weekly RSI Scanner</div>
      <div style="color:#8b949e;font-size:14px;margin-top:6px;">
        Checklist Item #6 · RSI({rsi_period}) Weekly · OB ≥ {ob_level} · OS ≤ {os_level}
      </div>
      <div style="font-size:13px;color:#388bfd;font-weight:500;margin-top:4px;">
        🕐 {datetime.now().strftime('%A, %d %B %Y  |  %H:%M')} · "Has Room" = RSI not exhausted at extremes
      </div>
    </div>
    <div style="display:flex;gap:10px;flex-wrap:wrap;">
      <div class="metric-box">
        <div style="font-size:22px;font-weight:700;color:{dir_color};">{dir_icon} {direction}</div>
        <div class="metric-label">Direction</div>
      </div>
      <div class="metric-box">
        <div class="metric-value" style="color:#3fb950;">{len(has_room)}</div>
        <div class="metric-label">✅ Has Room</div>
      </div>
      <div class="metric-box">
        <div class="metric-value" style="color:#f85149;">{len(obought)}</div>
        <div class="metric-label">🔴 Overbought</div>
      </div>
      <div class="metric-box">
        <div class="metric-value" style="color:#388bfd;">{len(osold)}</div>
        <div class="metric-label">🔵 Oversold</div>
      </div>
      <div class="metric-box">
        <div class="metric-value" style="color:#e3b341;">{len(divs)}</div>
        <div class="metric-label">⚠️ Divergence</div>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Health bar ─────────────────────────────────────────────────────────────────
if loaded:
    room_pct = int(len(has_room) / len(loaded) * 100)
    ob_pct   = int(len(obought)  / len(loaded) * 100)
    os_pct   = int(len(osold)    / len(loaded) * 100)
    bar_col  = "#3fb950" if room_pct >= 60 else "#e3b341" if room_pct >= 40 else "#f85149"
    st.markdown(f"""
    <div style="margin-bottom:22px;">
      <div style="display:flex;justify-content:space-between;margin-bottom:5px;">
        <span style="font-size:12px;color:#8b949e;font-weight:500;">RSI ROOM AVAILABILITY — {direction}</span>
        <span style="font-size:12px;color:{bar_col};font-weight:700;">{len(has_room)}/{len(loaded)} pairs have room</span>
      </div>
      <div style="background:#21262d;border-radius:8px;height:12px;overflow:hidden;display:flex;">
        <div style="background:#3fb950;width:{room_pct}%;height:100%;border-radius:8px 0 0 8px;"></div>
        <div style="background:#f85149;width:{ob_pct}%;height:100%;"></div>
        <div style="background:#388bfd;width:{os_pct}%;height:100%;border-radius:0 8px 8px 0;"></div>
      </div>
      <div style="display:flex;gap:16px;font-size:10px;color:#484f58;margin-top:3px;">
        <span>🟢 Has Room {room_pct}%</span>
        <span>🔴 Overbought {ob_pct}%</span>
        <span>🔵 Oversold {os_pct}%</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ══════════════════════════════════════════════════════════════════
left, right = st.columns([2, 3], gap="large")

# ── Sort ───────────────────────────────────────────────────────────────────────
def sort_key(pair):
    d = loaded.get(pair)
    c = classes.get(pair, {})
    if d is None: return (99, 99, pair)
    if sort_by == "RSI Value":
        return (0, d["rsi"], pair)
    elif sort_by == "Has Room First":
        room_order = 0 if c.get("has_room") else 1
        return (room_order, d["rsi"] if direction=="LONG" else -d["rsi"], pair)
    return (0, 0, pair)

sorted_pairs = sorted(INSTRUMENTS.keys(), key=sort_key)
if show_only_room:
    sorted_pairs = [p for p in sorted_pairs if classes.get(p, {}).get("has_room")]

# ══════════════════════════════════════════════════════════════════
# LEFT: INSTRUMENT CARDS
# ══════════════════════════════════════════════════════════════════
with left:
    st.markdown("### 🗂️ All Pairs — Weekly RSI Status")

    for pair in sorted_pairs:
        d = loaded.get(pair)
        c = classes.get(pair)

        if d is None:
            st.markdown(f"""
            <div class="pair-card rsi-neut" style="opacity:.45;">
              <div style="display:flex;justify-content:space-between;align-items:center;">
                <span class="pair-name">{pair}</span>
                <span class="badge badge-neut">⚠️ No Data</span>
              </div>
            </div>""", unsafe_allow_html=True)
            continue

        rsi      = d["rsi"]
        chg      = d["rsi_chg"]
        dist_ob  = d["dist_ob"]
        dist_os  = d["dist_os"]
        div      = d["divergence"] or ""
        chg_col  = "#3fb950" if chg > 0 else "#f85149" if chg < 0 else "#8b949e"
        chg_icon = "▲" if chg > 0 else "▼" if chg < 0 else "●"

        # Needle position (0–100 mapped to 0–100%)
        needle_pct = min(max(rsi, 0), 100)
        highlight  = "border-color:#388bfd!important;" if pair == focus_pair else ""

        st.markdown(f"""
        <div class="pair-card {c['card_cls']}" style="{highlight}">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px;">
            <div>
              <span class="pair-name">{pair}</span>
              <span style="font-size:11px;color:#8b949e;margin-left:6px;">@ {d['close']}</span>
              {f'<span style="font-size:11px;color:#e3b341;margin-left:8px;">{div}</span>' if div else ''}
            </div>
            <span class="badge {c['badge_cls']}">{c['icon']} {c['zone']}</span>
          </div>

          <!-- RSI Gauge -->
          <div style="position:relative;background:linear-gradient(90deg,
              #1f4f8a 0%,#1f4f8a 30%,
              #21262d 30%,#21262d 40%,
              #0d2f1a 40%,#0d2f1a 60%,
              #21262d 60%,#21262d 70%,
              #4a0d0d 70%,#4a0d0d 100%);
              border-radius:8px;height:10px;margin:6px 0 2px;">
            <div style="position:absolute;top:-3px;left:{needle_pct}%;
                width:3px;height:16px;background:#e6edf3;border-radius:2px;transform:translateX(-50%);"></div>
          </div>
          <div style="display:flex;justify-content:space-between;font-size:9px;color:#484f58;margin-bottom:8px;">
            <span>0</span><span style="color:#388bfd;">OS {os_level}</span>
            <span>50</span>
            <span style="color:#f85149;">OB {ob_level}</span><span>100</span>
          </div>

          <!-- RSI value grid -->
          <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:5px;margin-bottom:8px;">
            <div style="background:#0d1117;border-radius:6px;padding:7px;text-align:center;">
              <div style="font-size:9px;color:#8b949e;">RSI({rsi_period})</div>
              <div style="font-size:18px;font-weight:700;color:{c['color']};">{rsi:.1f}</div>
            </div>
            <div style="background:#0d1117;border-radius:6px;padding:7px;text-align:center;">
              <div style="font-size:9px;color:#8b949e;">Change</div>
              <div style="font-size:14px;font-weight:700;color:{chg_col};">{chg_icon}{abs(chg):.1f}</div>
            </div>
            <div style="background:#0d1117;border-radius:6px;padding:7px;text-align:center;">
              <div style="font-size:9px;color:#8b949e;">→ OB</div>
              <div style="font-size:14px;font-weight:700;color:#{'3fb950' if dist_ob>15 else 'e3b341' if dist_ob>5 else 'f85149'};">{dist_ob:.1f}</div>
            </div>
            <div style="background:#0d1117;border-radius:6px;padding:7px;text-align:center;">
              <div style="font-size:9px;color:#8b949e;">→ OS</div>
              <div style="font-size:14px;font-weight:700;color:#{'3fb950' if dist_os>15 else 'e3b341' if dist_os>5 else '388bfd'};">{dist_os:.1f}</div>
            </div>
          </div>

          <div style="font-size:11px;color:{c['color']};font-weight:600;">{c['detail']}</div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# RIGHT: DETAIL + CHARTS
# ══════════════════════════════════════════════════════════════════
with right:
    d  = loaded.get(focus_pair)
    c  = classes.get(focus_pair, {})

    st.markdown(f"### 📈 {focus_pair} — Weekly RSI Detail")

    if d is None:
        st.warning(f"No data for {focus_pair}. Try refreshing.")
    else:
        rsi_val  = d["rsi"]
        ok_bg    = "#0d2f1a" if c["has_room"] else "#2f0d0d" if c["zone"]=="Overbought" else "#0d1a2f"
        ok_bdr   = "#238636" if c["has_room"] else "#8b2d2d" if c["zone"]=="Overbought" else "#1f6feb"

        # ── Status banner ──────────────────────────────────────────
        st.markdown(f"""
        <div style="background:{ok_bg};border:1px solid {ok_bdr};border-radius:10px;
             padding:14px 18px;margin-bottom:16px;">
          <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;">
            <div>
              <div style="font-size:17px;font-weight:700;color:{c.get('color','#8b949e')};">
                {c.get('icon','—')} {c.get('zone','—')} — {c.get('detail','—')}
              </div>
              <div style="font-size:12px;color:#8b949e;margin-top:5px;">
                {d['divergence'] or 'No divergence detected'} &nbsp;·&nbsp;
                Prev RSI: {d['rsi_prev']:.1f} &nbsp;·&nbsp;
                Change: <span style="color:{'#3fb950' if d['rsi_chg']>0 else '#f85149'};">{d['rsi_chg']:+.2f}</span>
              </div>
            </div>
            <div style="text-align:center;background:#0d1117;border-radius:8px;padding:10px 18px;">
              <div style="font-size:38px;font-weight:700;color:{c.get('color','#8b949e')};
                   font-family:monospace;line-height:1;">{rsi_val:.1f}</div>
              <div style="font-size:10px;color:#8b949e;margin-top:2px;">RSI({rsi_period})</div>
            </div>
          </div>

          <!-- Full gauge -->
          <div style="position:relative;background:linear-gradient(90deg,
              #1f4f8a 0%,#1f4f8a 30%,
              #21262d 30%,#21262d 40%,
              #0d2f1a 40%,#0d2f1a 60%,
              #21262d 60%,#21262d 70%,
              #4a0d0d 70%,#4a0d0d 100%);
              border-radius:10px;height:16px;margin:14px 0 4px;">
            <div style="position:absolute;top:-4px;left:{min(max(rsi_val,0),100)}%;
                width:4px;height:24px;background:#fff;border-radius:2px;
                box-shadow:0 0 6px rgba(255,255,255,.5);transform:translateX(-50%);"></div>
          </div>
          <div style="display:flex;justify-content:space-between;font-size:11px;color:#8b949e;">
            <span>0 — Extreme OS</span>
            <span style="color:#388bfd;">{os_level} Oversold</span>
            <span>50 — Neutral</span>
            <span style="color:#f85149;">{ob_level} Overbought</span>
            <span>100</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Price + RSI chart ──────────────────────────────────────
        hist = d["history"].tail(min(lookback_w, 150))

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.55, 0.45], vertical_spacing=0.04,
            subplot_titles=[f"{focus_pair} — Weekly Price", f"RSI({rsi_period}) Weekly"],
        )

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=hist.index, open=hist["Open"], high=hist["High"],
            low=hist["Low"], close=hist["Close"], name="Price",
            increasing_line_color="#3fb950", decreasing_line_color="#f85149",
            increasing_fillcolor="#0d2f1a",  decreasing_fillcolor="#2f0d0d",
        ), row=1, col=1)

        # EMA21 on price
        if "close_ema21" in hist.columns:
            fig.add_trace(go.Scatter(
                x=hist.index, y=hist["close_ema21"], name="EMA21",
                mode="lines", line=dict(color="#388bfd", width=1.5, dash="dot"),
                hovertemplate="EMA21: %{y:.5f}<extra></extra>",
            ), row=1, col=1)

        # RSI line
        fig.add_trace(go.Scatter(
            x=hist.index, y=hist["rsi"], name=f"RSI({rsi_period})",
            mode="lines", line=dict(color="#a371f7", width=2.5),
            hovertemplate=f"RSI: %{{y:.2f}}<extra></extra>",
        ), row=2, col=1)

        # Smoothed RSI
        if "rsi_ma" in hist.columns:
            fig.add_trace(go.Scatter(
                x=hist.index, y=hist["rsi_ma"], name="RSI MA5",
                mode="lines", line=dict(color="#e3b341", width=1.2, dash="dot"),
                hovertemplate="RSI MA: %{y:.2f}<extra></extra>",
            ), row=2, col=1)

        # OB / OS / 50 lines
        for level, color, lbl in [
            (ob_level, "#f85149", f"OB {ob_level}"),
            (os_level, "#388bfd", f"OS {os_level}"),
            (50,       "#484f58", "50"),
        ]:
            fig.add_hline(y=level, line_dash="dot", line_color=color, line_width=1.2,
                          annotation_text=lbl, annotation_font_color=color,
                          annotation_position="right", row=2, col=1)

        # OB / OS fill bands
        fig.add_hrect(y0=ob_level, y1=100, fillcolor="#f85149", opacity=0.06,
                      line_width=0, row=2, col=1)
        fig.add_hrect(y0=0, y1=os_level, fillcolor="#388bfd", opacity=0.06,
                      line_width=0, row=2, col=1)
        fig.add_hrect(y0=os_level, y1=ob_level, fillcolor="#3fb950", opacity=0.03,
                      line_width=0, row=2, col=1)

        # Current RSI marker
        fig.add_hline(y=rsi_val, line_dash="dash", line_color=c.get("color","#8b949e"),
                      line_width=1.5, row=2, col=1,
                      annotation_text=f"  Now {rsi_val:.1f}",
                      annotation_font_color=c.get("color","#8b949e"),
                      annotation_position="right")

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1117",
            legend=dict(font=dict(color="#8b949e", size=11), bgcolor="#161b22",
                        bordercolor="#21262d", orientation="h", x=0, y=1.06),
            margin=dict(l=10, r=80, t=30, b=10),
            height=520,
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

        # ── All-pair RSI bar chart ─────────────────────────────────
        st.markdown("### 📊 All Pairs — RSI Snapshot")

        pairs_sorted = sorted(loaded.keys(), key=lambda p: loaded[p]["rsi"])
        rsi_vals   = [loaded[p]["rsi"] for p in pairs_sorted]
        bar_colors = []
        for p in pairs_sorted:
            rv = loaded[p]["rsi"]
            if   rv >= ob_level: bar_colors.append("#f85149")
            elif rv <= os_level: bar_colors.append("#388bfd")
            elif p == focus_pair: bar_colors.append("#388bfd")
            elif direction=="LONG" and rv >= 50:  bar_colors.append("#3fb950")
            elif direction=="SHORT" and rv <= 50: bar_colors.append("#3fb950")
            else: bar_colors.append("#e3b341")

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=pairs_sorted, y=rsi_vals,
            marker_color=bar_colors,
            text=[f"{v:.1f}" for v in rsi_vals],
            textposition="outside",
            textfont=dict(color="#c9d1d9", size=10),
            hovertemplate="<b>%{x}</b><br>RSI: %{y:.2f}<extra></extra>",
        ))
        for level, color, lbl in [(ob_level,"#f85149",f"OB {ob_level}"),
                                  (os_level,"#388bfd",f"OS {os_level}"),
                                  (50,"#484f58","50")]:
            fig2.add_hline(y=level, line_dash="dot", line_color=color, line_width=1.2,
                           annotation_text=lbl, annotation_font_color=color,
                           annotation_position="right")
        fig2.add_hrect(y0=ob_level, y1=105, fillcolor="#f85149", opacity=0.05, line_width=0)
        fig2.add_hrect(y0=0, y1=os_level, fillcolor="#388bfd", opacity=0.05, line_width=0)
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1117",
            margin=dict(l=10, r=60, t=10, b=10), height=300,
            xaxis=dict(tickfont=dict(color="#8b949e", size=9), tickangle=-40, showgrid=False),
            yaxis=dict(tickfont=dict(color="#8b949e"), showgrid=True, gridcolor="#21262d",
                       range=[0, 105]),
            showlegend=False, bargap=0.2,
        )
        st.plotly_chart(fig2, use_container_width=True, config=dict(displayModeBar=False))

        # ── Summary table ──────────────────────────────────────────
        st.markdown("### 📋 Full RSI Summary Table")
        rows = []
        for pair in INSTRUMENTS:
            dd = loaded.get(pair)
            cc = classes.get(pair, {})
            if dd is None:
                rows.append({"Pair": pair, "RSI": None, "Prev RSI": None,
                             "Chg": None, "→OB": None, "→OS": None,
                             "Zone": "No Data", "Status": "—", "Divergence": "—"})
                continue
            rows.append({
                "Pair":      pair,
                "RSI":       dd["rsi"],
                "Prev RSI":  dd["rsi_prev"],
                "Chg":       dd["rsi_chg"],
                "→OB":       dd["dist_ob"],
                "→OS":       dd["dist_os"],
                "Zone":      cc.get("zone", "—"),
                "Status":    cc.get("icon","—") + " " + ("Has Room" if cc.get("has_room") else "Exhausted"),
                "Divergence":dd["divergence"] or "—",
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True, column_config={
            "Pair":      st.column_config.TextColumn("Pair"),
            "RSI":       st.column_config.NumberColumn("RSI",      format="%.2f"),
            "Prev RSI":  st.column_config.NumberColumn("Prev RSI", format="%.2f"),
            "Chg":       st.column_config.NumberColumn("Change",   format="%+.2f"),
            "→OB":       st.column_config.NumberColumn("Pts → OB", format="%.1f"),
            "→OS":       st.column_config.NumberColumn("Pts → OS", format="%.1f"),
            "Zone":      st.column_config.TextColumn("Zone"),
            "Status":    st.column_config.TextColumn("Status"),
            "Divergence":st.column_config.TextColumn("Divergence"),
        })

st.markdown("""
<div style="text-align:center;color:#484f58;font-size:11px;margin-top:32px;
     padding-top:16px;border-top:1px solid #21262d;">
  📡 Weekly RSI · Wilder RSI (EWM) · Weekly bars · OB/OS levels configurable · Not financial advice
</div>
""", unsafe_allow_html=True)