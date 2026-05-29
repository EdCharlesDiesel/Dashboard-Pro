import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime

st.set_page_config(
    page_title="Daily Trend · Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
    .hero{background:linear-gradient(135deg,#0d1117 0%,#161b22 50%,#0d1117 100%);border:1px solid #21262d;border-radius:16px;padding:24px 32px;margin-bottom:20px;position:relative;overflow:hidden;}
    .hero::before{content:'';position:absolute;top:-40%;right:-5%;width:300px;height:300px;background:radial-gradient(circle,rgba(56,139,253,.07) 0%,transparent 70%);border-radius:50%;}
    .metric-box{background:var(--background-color);border:1px solid var(--border,#21262d);border-radius:8px;padding:12px 14px;text-align:center;}
    .metric-value{font-size:20px;font-weight:700;color:#c9d1d9;}
    .metric-label{font-size:10px;color:#8b949e;margin-top:2px;font-weight:500;letter-spacing:.05em;text-transform:uppercase;}

    /* Pair cards */
    .pair-card{border-radius:12px;padding:14px 16px;margin-bottom:9px;border:1px solid #21262d;background:#161b22;}
    .card-bull{border-left:4px solid #3fb950;}
    .card-bear{border-left:4px solid #f85149;}
    .card-neut{border-left:4px solid #e3b341;}
    .card-none{border-left:4px solid #30363d;opacity:.5;}

    .pair-name{font-size:15px;font-weight:700;color:#e6edf3;}
    .badge{border-radius:6px;padding:3px 10px;font-size:11px;font-weight:700;display:inline-block;}
    .b-bull{background:#0d2f1a;color:#3fb950;border:1px solid #238636;}
    .b-bear{background:#2f0d0d;color:#f85149;border:1px solid #8b2d2d;}
    .b-warn{background:#2f1f0d;color:#e3b341;border:1px solid #9e6a03;}
    .b-none{background:#21262d;color:#8b949e;border:1px solid #30363d;}

    /* EMA gap bar */
    .gap-track{background:#21262d;border-radius:6px;height:8px;margin:6px 0;overflow:hidden;position:relative;}

    [data-testid="stSidebarNav"]{display:none;}
    #MainMenu,footer,header{visibility:hidden;}
    [data-testid="stSidebarCollapsedControl"]{visibility:visible !important;}
    .block-container{padding-top:1.5rem;max-width:1400px;}
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

ALL_EMAS = [20, 50, 100, 200]

# ── Helpers ────────────────────────────────────────────────────────────────────
def ema_series(close: pd.Series, period: int) -> pd.Series:
    return close.ewm(span=period, adjust=False).mean()

def slope_pct(series: pd.Series, bars: int = 5) -> float:
    s = series.dropna()
    if len(s) < bars + 1:
        return 0.0
    return round((s.iloc[-1] - s.iloc[-bars]) / s.iloc[-bars] * 100, 4)

def crossover_bars_ago(e20: pd.Series, e50: pd.Series) -> int | None:
    """How many bars ago did EMA20 cross above/below EMA50?"""
    diff = (e20 - e50).dropna()
    if len(diff) < 2:
        return None
    current_sign = np.sign(diff.iloc[-1])
    for i in range(2, min(len(diff), 200)):
        if np.sign(diff.iloc[-i]) != current_sign:
            return i - 1
    return None

# ── Fetch ──────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def fetch_daily_trend(ticker: str, pip_size: float, days: int = 300):
    try:
        df = yf.download(ticker, period=f"{days}d", interval="1d",
                         progress=False, auto_adjust=True)
        if df.empty or len(df) < 55:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]

        df = df[["Open","High","Low","Close","Volume"]].dropna(subset=["Close"])

        for p in ALL_EMAS:
            df[f"ema{p}"] = ema_series(df["Close"], p)

        df.dropna(subset=["ema20","ema50"], inplace=True)
        if df.empty:
            return None

        last        = df.iloc[-1]
        close       = float(last["Close"])
        e20         = float(last["ema20"])
        e50         = float(last["ema50"])
        gap_pips    = round((e20 - e50) / pip_size, 1)
        gap_pct     = round((e20 - e50) / e50 * 100, 4)
        trend_bull  = e20 > e50

        gap_series   = (df["ema20"] - df["ema50"]) / pip_size
        gap_hist_p90 = float(np.percentile(gap_series.abs().dropna(), 90)) if len(gap_series) > 10 else 50.0

        # Slopes
        s20 = slope_pct(df["ema20"], 5)
        s50 = slope_pct(df["ema50"], 5)

        # Price position
        price_above_20 = close > e20
        price_above_50 = close > e50

        # Crossover recency
        xover = crossover_bars_ago(df["ema20"], df["ema50"])

        # Additional EMAs
        e100 = float(last["ema100"]) if not np.isnan(last["ema100"]) else None
        e200 = float(last["ema200"]) if not np.isnan(last["ema200"]) else None

        return {
            "close": round(close, 5),
            "ema20": round(e20, 5), "ema50": round(e50, 5),
            "ema100": round(e100, 5) if e100 else None,
            "ema200": round(e200, 5) if e200 else None,
            "gap_pips": gap_pips, "gap_pct": gap_pct,
            "trend_bull": trend_bull,
            "slope20": s20, "slope50": s50,
            "price_above_20": price_above_20,
            "price_above_50": price_above_50,
            "crossover_bars": xover,
            "gap_hist_p90": round(gap_hist_p90, 1),
            "history": df,
        }
    except Exception:
        return None

# ── Check alignment ────────────────────────────────────────────────────────────
def check_trend(data: dict, direction: str) -> dict:
    bull      = data["trend_bull"]
    s20       = data["slope20"]
    price_a20 = data["price_above_20"]
    price_a50 = data["price_above_50"]

    if direction == "LONG":
        if bull and s20 > 0:
            status = "✅ Intact"
            detail = "EMA20 > EMA50 · EMA20 rising · Trend confirmed"
            color, card, badge = "#3fb950", "card-bull", "b-bull"
        elif bull and s20 <= 0:
            status = "⚠️ Weakening"
            detail = "EMA20 > EMA50 but EMA20 slope flattening"
            color, card, badge = "#e3b341", "card-neut", "b-warn"
        else:
            status = "❌ Broken"
            detail = "EMA20 < EMA50 — bearish structure, avoid longs"
            color, card, badge = "#f85149", "card-bear", "b-bear"
    else:  # SHORT
        if not bull and s20 < 0:
            status = "✅ Intact"
            detail = "EMA20 < EMA50 · EMA20 falling · Trend confirmed"
            color, card, badge = "#3fb950", "card-bull", "b-bull"
        elif not bull and s20 >= 0:
            status = "⚠️ Weakening"
            detail = "EMA20 < EMA50 but EMA20 slope flattening"
            color, card, badge = "#e3b341", "card-neut", "b-warn"
        else:
            status = "❌ Broken"
            detail = "EMA20 > EMA50 — bullish structure, avoid shorts"
            color, card, badge = "#f85149", "card-bear", "b-bear"

    return {"status": status, "detail": detail,
            "color": color, "card": card, "badge": badge}

# ── EMA stack alignment ────────────────────────────────────────────────────────
def ema_stack(data: dict, direction: str) -> dict:
    close = data["close"]
    e20   = data["ema20"]
    e50   = data["ema50"]
    e100  = data["ema100"]
    e200  = data["ema200"]

    if direction == "LONG":
        conds = [close > e20, e20 > e50]
        if e100 is not None:
            conds.append(e50 > e100)
        if e100 is not None and e200 is not None:
            conds.append(e100 > e200)
    else:
        conds = [close < e20, e20 < e50]
        if e100 is not None:
            conds.append(e50 < e100)
        if e100 is not None and e200 is not None:
            conds.append(e100 < e200)

    n     = len(conds)
    score = sum(conds)
    pct   = score / n if n else 0

    if pct == 1.0:    lbl, col = "Full Stack",   "#3fb950"
    elif pct >= 0.75: lbl, col = "Strong",        "#52cc6b"
    elif pct >= 0.5:  lbl, col = "Partial",       "#e3b341"
    elif pct > 0:     lbl, col = "Weak",          "#e07b39"
    else:             lbl, col = "No Alignment",  "#f85149"

    return {"score": score, "max": n, "pct": pct, "label": lbl, "color": col, "conditions": conds}

# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 📈 Daily Trend")
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

    direction = st.radio("🎯 Trade Direction", ["LONG", "SHORT"], horizontal=True)
    focus_pair = st.selectbox("Focus Pair (Detail Chart)", list(INSTRUMENTS.keys()), index=0)

    show_emas = st.multiselect("Overlay EMAs on chart",
                               ["EMA20","EMA50","EMA100","EMA200"],
                               default=["EMA20","EMA50","EMA100"])

    lookback_days = st.slider("Lookback (days)", 100, 500, 250, 50)

    sort_by = st.radio("Sort Cards By",
                       ["Trend Intact First", "EMA Gap (pips)", "Pair Name"], index=0)
    only_intact = st.checkbox("Show only ✅ Intact", value=False)

    st.divider()
    if st.button("🔄 Refresh Data", use_container_width=True, type="primary"):
        st.cache_data.clear()
        st.rerun()

# ══════════════════════════════════════════════════════════════════
# FETCH ALL
# ══════════════════════════════════════════════════════════════════
prog = st.progress(0, text="📡 Fetching daily EMA data…")
all_data = {}
for idx, (pair, cfg) in enumerate(INSTRUMENTS.items()):
    prog.progress((idx+1)/len(INSTRUMENTS), text=f"📡 {pair} ({idx+1}/{len(INSTRUMENTS)})…")
    all_data[pair] = fetch_daily_trend(cfg["ticker"], cfg["pip_size"], lookback_days)
prog.empty()

loaded  = {p: d for p, d in all_data.items() if d is not None}
checks  = {p: check_trend(d, direction) for p, d in loaded.items()}
stacks  = {p: ema_stack(d, direction)   for p, d in loaded.items()}

intact   = [p for p, c in checks.items() if c["status"].startswith("✅")]
weak     = [p for p, c in checks.items() if c["status"].startswith("⚠️")]
broken   = [p for p, c in checks.items() if c["status"].startswith("❌")]
failed   = [p for p in INSTRUMENTS if p not in loaded]

# ══════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════
dir_color = "#3fb950" if direction == "LONG" else "#f85149"
dir_icon  = "🔼" if direction == "LONG" else "🔽"
rule_txt  = "EMA20 > EMA50" if direction == "LONG" else "EMA20 < EMA50"

st.markdown(f"""
<div class="hero">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:12px;">
    <div>
      <div style="font-size:26px;font-weight:700;color:#e6edf3;">📈 Daily Trend Scanner</div>
      <div style="color:#8b949e;font-size:14px;margin-top:6px;">
        Checklist Item #8 · {rule_txt} · Daily timeframe · EMA20 slope confirmation
      </div>
      <div style="font-size:13px;color:#388bfd;font-weight:500;margin-top:4px;">
        🕐 {datetime.now().strftime('%A, %d %B %Y  |  %H:%M')} · {lookback_days}-day window
      </div>
    </div>
    <div style="display:flex;gap:10px;flex-wrap:wrap;">
      <div class="metric-box">
        <div style="font-size:22px;font-weight:700;color:{dir_color};">{dir_icon} {direction}</div>
        <div class="metric-label">Direction</div>
      </div>
      <div class="metric-box">
        <div class="metric-value" style="color:#3fb950;">{len(intact)}</div>
        <div class="metric-label">✅ Intact</div>
      </div>
      <div class="metric-box">
        <div class="metric-value" style="color:#e3b341;">{len(weak)}</div>
        <div class="metric-label">⚠️ Weakening</div>
      </div>
      <div class="metric-box">
        <div class="metric-value" style="color:#f85149;">{len(broken)}</div>
        <div class="metric-label">❌ Broken</div>
      </div>
      <div class="metric-box">
        <div class="metric-value" style="color:#484f58;">{len(failed)}</div>
        <div class="metric-label">No Data</div>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Health bar ─────────────────────────────────────────────────────────────────
if loaded:
    n = len(loaded)
    i_pct = int(len(intact)/n*100)
    w_pct = int(len(weak)/n*100)
    b_pct = int(len(broken)/n*100)
    bar_c = "#3fb950" if i_pct>=60 else "#e3b341" if i_pct>=35 else "#f85149"
    st.markdown(f"""
    <div style="margin-bottom:22px;">
      <div style="display:flex;justify-content:space-between;margin-bottom:5px;">
        <span style="font-size:12px;color:#8b949e;font-weight:500;">DAILY TREND HEALTH — {direction}</span>
        <span style="font-size:12px;color:{bar_c};font-weight:700;">{len(intact)}/{n} trends intact</span>
      </div>
      <div style="background:#21262d;border-radius:8px;height:12px;overflow:hidden;display:flex;">
        <div style="background:#3fb950;width:{i_pct}%;height:100%;border-radius:8px 0 0 8px;"></div>
        <div style="background:#e3b341;width:{w_pct}%;height:100%;"></div>
        <div style="background:#f85149;width:{b_pct}%;height:100%;border-radius:0 8px 8px 0;"></div>
      </div>
      <div style="display:flex;gap:16px;font-size:10px;color:#484f58;margin-top:3px;">
        <span>🟢 Intact {i_pct}%</span>
        <span>🟡 Weakening {w_pct}%</span>
        <span>🔴 Broken {b_pct}%</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# LAYOUT
# ══════════════════════════════════════════════════════════════════
left, right = st.columns([2, 3], gap="large")

# ── Sort ───────────────────────────────────────────────────────────────────────
STATUS_ORDER = {"✅ Intact": 0, "⚠️ Weakening": 1, "❌ Broken": 2}

def card_sort(pair):
    d = loaded.get(pair); c = checks.get(pair, {})
    if d is None: return (9, 0, pair)
    if sort_by == "Trend Intact First":
        return (STATUS_ORDER.get(c.get("status",""), 9), 0, pair)
    elif sort_by == "EMA Gap (pips)":
        return (0, -abs(d["gap_pips"]), pair)
    return (0, 0, pair)

sorted_pairs = sorted(INSTRUMENTS.keys(), key=card_sort)
if only_intact:
    sorted_pairs = [p for p in sorted_pairs if checks.get(p,{}).get("status","").startswith("✅")]

# ══════════════════════════════════════════════════════════════════
# LEFT — Cards
# ══════════════════════════════════════════════════════════════════
with left:
    st.markdown("### 🗂️ All Pairs — Daily Trend Status")

    for pair in sorted_pairs:
        d   = loaded.get(pair)
        c   = checks.get(pair)
        stk = stacks.get(pair, {})
        cfg = INSTRUMENTS[pair]

        if d is None:
            st.markdown(f"""
            <div class="pair-card card-none">
              <div style="display:flex;justify-content:space-between;">
                <span class="pair-name">{pair}</span>
                <span class="badge b-none">⚠️ No Data</span>
              </div>
            </div>""", unsafe_allow_html=True)
            continue

        gap_abs    = abs(d["gap_pips"])
        gap_col    = "#3fb950" if (direction=="LONG" and d["gap_pips"]>0) or \
                                  (direction=="SHORT" and d["gap_pips"]<0) else "#f85149"
        gap_scale  = max(d.get("gap_hist_p90", 100.0), gap_abs, 1.0)
        gap_bar    = min(int(gap_abs / gap_scale * 100), 100)
        s20_col    = "#3fb950" if d["slope20"]>0 else "#f85149"
        s50_col    = "#3fb950" if d["slope50"]>0 else "#f85149"
        s20_icon   = "▲" if d["slope20"]>0.01 else "▼" if d["slope20"]<-0.01 else "➡️"
        s50_icon   = "▲" if d["slope50"]>0.01 else "▼" if d["slope50"]<-0.01 else "➡️"
        xover_txt  = f"Cross {d['crossover_bars']}d ago" if d["crossover_bars"] else "Cross > 200d ago"
        xover_col  = "#e3b341" if d["crossover_bars"] and d["crossover_bars"] < 10 else "#484f58"
        highlight  = "border-color:#388bfd!important;" if pair == focus_pair else ""

        # Price vs EMAs
        pa20 = "▲" if d["price_above_20"] else "▼"
        pa50 = "▲" if d["price_above_50"] else "▼"
        pa20c= "#3fb950" if d["price_above_20"] else "#f85149"
        pa50c= "#3fb950" if d["price_above_50"] else "#f85149"

        # Stack dots
        dot_cols  = ["#3fb950" if v else "#f85149" for v in stk.get("conditions", [])]
        while len(dot_cols) < 4:
            dot_cols.append("#30363d")
        dots_html = "".join(
            f'<div style="width:9px;height:9px;border-radius:2px;background:{cl};'
            f'display:inline-block;margin:0 1px;"></div>'
            for cl in dot_cols
        )

        st.markdown(f"""
        <div class="pair-card {c['card']}" style="{highlight}">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px;">
            <div>
              <span class="pair-name">{pair}</span>
              <span style="font-size:11px;color:#8b949e;margin-left:8px;">@ {d['close']}</span>
            </div>
            <span class="badge {c['badge']}">{c['status']}</span>
          </div>

          <!-- EMA gap bar -->
          <div style="font-size:10px;color:#8b949e;margin-bottom:3px;">
            EMA gap: <span style="color:{gap_col};font-weight:700;">{d['gap_pips']:+.1f} pips
            ({d['gap_pct']:+.3f}%)</span>
            &nbsp;·&nbsp;<span style="color:{xover_col};">{xover_txt}</span>
          </div>
          <div style="background:#21262d;border-radius:6px;height:7px;margin-bottom:10px;overflow:hidden;">
            <div style="background:{gap_col};width:{gap_bar}%;height:100%;border-radius:6px;"></div>
          </div>

          <!-- EMA value grid -->
          <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:5px;margin-bottom:8px;">
            <div style="background:#0d1117;border-radius:6px;padding:7px;text-align:center;">
              <div style="font-size:9px;color:#e3b341;font-weight:700;">EMA20</div>
              <div style="font-size:12px;font-weight:700;color:#e6edf3;font-family:monospace;">{d['ema20']:.4f}</div>
              <div style="font-size:10px;color:{s20_col};">{s20_icon} {d['slope20']:+.3f}%</div>
            </div>
            <div style="background:#0d1117;border-radius:6px;padding:7px;text-align:center;">
              <div style="font-size:9px;color:#388bfd;font-weight:700;">EMA50</div>
              <div style="font-size:12px;font-weight:700;color:#e6edf3;font-family:monospace;">{d['ema50']:.4f}</div>
              <div style="font-size:10px;color:{s50_col};">{s50_icon} {d['slope50']:+.3f}%</div>
            </div>
            <div style="background:#0d1117;border-radius:6px;padding:7px;text-align:center;">
              <div style="font-size:9px;color:#8b949e;">P vs EMA20</div>
              <div style="font-size:16px;font-weight:700;color:{pa20c};">{pa20}</div>
            </div>
            <div style="background:#0d1117;border-radius:6px;padding:7px;text-align:center;">
              <div style="font-size:9px;color:#8b949e;">P vs EMA50</div>
              <div style="font-size:16px;font-weight:700;color:{pa50c};">{pa50}</div>
            </div>
          </div>

          <!-- Stack signal -->
          <div style="display:flex;align-items:center;gap:5px;padding:6px 0 4px;">
            <span style="font-size:9px;color:#484f58;letter-spacing:.05em;font-weight:500;">STACK</span>
            {dots_html}
            <span style="font-size:10px;font-weight:700;color:{stk.get('color','#8b949e')};">
              {stk.get('score','?')}/{stk.get('max','4')} {stk.get('label','—')}
            </span>
          </div>

          <div style="font-size:11px;color:{c['color']};font-weight:600;">{c['detail']}</div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# RIGHT — Detail charts
# ══════════════════════════════════════════════════════════════════
with right:
    d  = loaded.get(focus_pair)
    c  = checks.get(focus_pair, {})

    st.markdown(f"### 📈 {focus_pair} — Daily Trend Detail")

    if d is None:
        st.warning(f"No data for {focus_pair}.")
    else:
        ok_col = c.get("color", "#8b949e")
        ok_bg  = "#0d2f1a" if ok_col=="#3fb950" else "#2f0d0d" if ok_col=="#f85149" else "#2f1f0d"
        ok_bdr = "#238636" if ok_col=="#3fb950" else "#8b2d2d" if ok_col=="#f85149" else "#9e6a03"
        pip_size = INSTRUMENTS[focus_pair]["pip_size"]

        # ── Status banner ──────────────────────────────────────────
        xover_note = (f"Last crossover: {d['crossover_bars']} days ago"
                      if d["crossover_bars"] else "Crossover > 200 days ago")
        fp_stk    = stacks.get(focus_pair, {})
        e100_str  = f"{d['ema100']:.5f}" if d["ema100"] else "—"
        e200_str  = f"{d['ema200']:.5f}" if d["ema200"] else "—"
        st.markdown(f"""
        <div style="background:{ok_bg};border:1px solid {ok_bdr};border-radius:10px;
             padding:14px 18px;margin-bottom:16px;">
          <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;">
            <div>
              <div style="font-size:17px;font-weight:700;color:{ok_col};">
                {c.get('status','—')} — {c.get('detail','—')}
              </div>
              <div style="font-size:12px;color:#8b949e;margin-top:5px;">{xover_note}</div>
              <div style="font-size:12px;margin-top:6px;">
                <span style="color:#484f58;font-weight:500;">STACK </span>
                <span style="color:{fp_stk.get('color','#8b949e')};font-weight:700;">
                  {fp_stk.get('score','?')}/{fp_stk.get('max','4')} — {fp_stk.get('label','—')}
                </span>
              </div>
            </div>
            <div style="display:flex;gap:8px;flex-wrap:wrap;">
              <div style="background:#0d1117;border-radius:6px;padding:8px 12px;text-align:center;">
                <div style="font-size:10px;color:#e3b341;">EMA20</div>
                <div style="font-size:14px;font-weight:700;color:#e3b341;font-family:monospace;">{d['ema20']:.5f}</div>
              </div>
              <div style="background:#0d1117;border-radius:6px;padding:8px 12px;text-align:center;">
                <div style="font-size:10px;color:#388bfd;">EMA50</div>
                <div style="font-size:14px;font-weight:700;color:#388bfd;font-family:monospace;">{d['ema50']:.5f}</div>
              </div>
              <div style="background:#0d1117;border-radius:6px;padding:8px 12px;text-align:center;">
                <div style="font-size:10px;color:#a371f7;">EMA100</div>
                <div style="font-size:14px;font-weight:700;color:#a371f7;font-family:monospace;">{e100_str}</div>
              </div>
              <div style="background:#0d1117;border-radius:6px;padding:8px 12px;text-align:center;">
                <div style="font-size:10px;color:#f85149;">EMA200</div>
                <div style="font-size:14px;font-weight:700;color:#f85149;font-family:monospace;">{e200_str}</div>
              </div>
              <div style="background:#0d1117;border-radius:6px;padding:8px 12px;text-align:center;">
                <div style="font-size:10px;color:#8b949e;">Gap</div>
                <div style="font-size:14px;font-weight:700;color:{ok_col};">{d['gap_pips']:+.1f}p</div>
              </div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Candlestick + EMA chart ────────────────────────────────
        hist = d["history"].tail(min(lookback_days, 300))

        EMA_STYLE = {
            "EMA20":  {"col":"#e3b341","w":2.5},
            "EMA50":  {"col":"#388bfd","w":2.0},
            "EMA100": {"col":"#a371f7","w":1.5},
            "EMA200": {"col":"#f85149","w":1.5,"dash":"dot"},
        }

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.68, 0.32], vertical_spacing=0.04,
                            subplot_titles=[
                                f"{focus_pair} — Daily Price + EMAs",
                                "EMA20 − EMA50 Gap (pips)"
                            ])

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=hist.index, open=hist["Open"], high=hist["High"],
            low=hist["Low"], close=hist["Close"], name="Price",
            increasing_line_color="#3fb950", decreasing_line_color="#f85149",
            increasing_fillcolor="#0d2f1a",  decreasing_fillcolor="#2f0d0d",
        ), row=1, col=1)

        # EMAs
        for label in show_emas:
            col_key = label.lower()
            if col_key in hist.columns:
                style = EMA_STYLE.get(label, {"col":"#8b949e","w":1.5})
                fig.add_trace(go.Scatter(
                    x=hist.index, y=hist[col_key], name=label,
                    mode="lines",
                    line=dict(color=style["col"], width=style["w"],
                              dash=style.get("dash","solid")),
                    hovertemplate=f"{label}: %{{y:.5f}}<extra></extra>",
                ), row=1, col=1)

        # EMA gap histogram
        gap_vals  = (hist["ema20"] - hist["ema50"]) / pip_size
        gap_cols  = ["#3fb950" if v > 0 else "#f85149" for v in gap_vals]
        fig.add_trace(go.Bar(
            x=hist.index, y=gap_vals,
            marker_color=gap_cols, name="EMA Gap",
            hovertemplate="Gap: %{y:+.1f} pips<extra></extra>",
        ), row=2, col=1)
        fig.add_hline(y=0, line_dash="dot", line_color="#484f58",
                      line_width=1, row=2, col=1)

        # Current price
        fig.add_hline(y=d["close"], line_dash="dash", line_color="#8b949e",
                      line_width=1, row=1, col=1,
                      annotation_text=f"  {d['close']}",
                      annotation_font_color="#8b949e",
                      annotation_position="right")

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1117",
            legend=dict(font=dict(color="#8b949e",size=11), bgcolor="#161b22",
                        bordercolor="#21262d", orientation="h", x=0, y=1.06),
            margin=dict(l=10, r=80, t=30, b=10),
            height=520, hovermode="x unified",
            xaxis_rangeslider_visible=False,
        )
        for row in [1, 2]:
            fig.update_xaxes(showgrid=False, tickfont=dict(color="#8b949e",size=9),
                             linecolor="#21262d", row=row, col=1)
            fig.update_yaxes(showgrid=True, gridcolor="#21262d",
                             tickfont=dict(color="#8b949e",size=9), row=row, col=1)
        fig.update_annotations(font_color="#8b949e", font_size=11)
        st.plotly_chart(fig, use_container_width=True, config=dict(displayModeBar=False))

        # ── EMA gap bar chart all pairs ────────────────────────────
        st.markdown("### 📊 All Pairs — EMA20 vs EMA50 Gap (pips)")

        gap_pairs  = [(p, loaded[p]["gap_pips"]) for p in loaded]
        gap_pairs.sort(key=lambda x: x[1])
        gp_labels  = [x[0] for x in gap_pairs]
        gp_values  = [x[1] for x in gap_pairs]
        gp_colors  = []
        for p, v in gap_pairs:
            if p == focus_pair:
                gp_colors.append("#58a6ff")
            elif (direction=="LONG" and v>0) or (direction=="SHORT" and v<0):
                gp_colors.append("#3fb950")
            else:
                gp_colors.append("#f85149")

        fig2 = go.Figure(go.Bar(
            x=gp_labels, y=gp_values,
            marker_color=gp_colors,
            text=[f"{v:+.0f}" for v in gp_values],
            textposition="outside",
            textfont=dict(color="#c9d1d9", size=9),
            hovertemplate="<b>%{x}</b><br>EMA Gap: %{y:+.1f} pips<extra></extra>",
        ))
        fig2.add_hline(y=0, line_dash="solid", line_color="#484f58", line_width=1.5)
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1117",
            margin=dict(l=10,r=10,t=10,b=10), height=260,
            xaxis=dict(tickfont=dict(color="#8b949e",size=9), tickangle=-40, showgrid=False),
            yaxis=dict(tickfont=dict(color="#8b949e"), showgrid=True,
                       gridcolor="#21262d", ticksuffix=" p",
                       zeroline=True, zerolinecolor="#484f58"),
            showlegend=False, bargap=0.2,
        )
        st.plotly_chart(fig2, use_container_width=True, config=dict(displayModeBar=False))

        # ── Summary table ──────────────────────────────────────────
        st.markdown("### 📋 Full Summary Table")
        rows = []
        for pair in INSTRUMENTS:
            dd = loaded.get(pair)
            cc = checks.get(pair, {})
            if dd is None:
                rows.append({"Pair":pair,"Close":None,"EMA20":None,"EMA50":None,
                             "EMA100":None,"EMA200":None,"Gap(pips)":None,
                             "Gap%":None,"EMA20 Slope":None,"EMA50 Slope":None,
                             "Cross Ago":None,"Status":"No Data"})
                continue
            ss = stacks.get(pair, {})
            rows.append({
                "Pair":       pair,
                "Close":      dd["close"],
                "EMA20":      dd["ema20"],
                "EMA50":      dd["ema50"],
                "EMA100":     dd["ema100"],
                "EMA200":     dd["ema200"],
                "Gap (pips)": dd["gap_pips"],
                "Gap %":      dd["gap_pct"],
                "EMA20 Slope":dd["slope20"],
                "EMA50 Slope":dd["slope50"],
                "Cross Ago":  dd["crossover_bars"],
                "Status":     cc.get("status","—"),
                "Stack":      f"{ss.get('score','?')}/{ss.get('max','4')} {ss.get('label','—')}",
            })
        df_out = pd.DataFrame(rows)
        st.dataframe(df_out, use_container_width=True, hide_index=True,
                     column_config={
                         "Pair":       st.column_config.TextColumn("Pair"),
                         "Close":      st.column_config.NumberColumn("Close",       format="%.5f"),
                         "EMA20":      st.column_config.NumberColumn("EMA20",       format="%.5f"),
                         "EMA50":      st.column_config.NumberColumn("EMA50",       format="%.5f"),
                         "EMA100":     st.column_config.NumberColumn("EMA100",      format="%.5f"),
                         "EMA200":     st.column_config.NumberColumn("EMA200",      format="%.5f"),
                         "Gap (pips)": st.column_config.NumberColumn("Gap (pips)",  format="%+.1f"),
                         "Gap %":      st.column_config.NumberColumn("Gap %",       format="%+.4f"),
                         "EMA20 Slope":st.column_config.NumberColumn("EMA20 Slope", format="%+.4f"),
                         "EMA50 Slope":st.column_config.NumberColumn("EMA50 Slope", format="%+.4f"),
                         "Cross Ago":  st.column_config.NumberColumn("Cross (days ago)"),
                         "Status":     st.column_config.TextColumn("Status"),
                         "Stack":      st.column_config.TextColumn("EMA Stack"),
                     })

st.markdown("""
<div style="text-align:center;color:#484f58;font-size:11px;margin-top:32px;
     padding-top:16px;border-top:1px solid #21262d;">
  📈 Daily Trend · EMA20 vs EMA50 · Daily bars · Wilder EWM · Not financial advice
</div>
""", unsafe_allow_html=True)