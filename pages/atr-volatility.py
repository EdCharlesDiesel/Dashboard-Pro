import streamlit as st
from src.pages_lib.navigation import render_sidebar_nav
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime
import pytz

st.set_page_config(
    page_title="ATR Volatility · Trading System",
    page_icon="📊",
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

    /* ATR instrument cards */
    .atr-card{border-radius:12px;padding:15px 16px;margin-bottom:10px;border:1px solid #21262d;background:#161b22;transition:border-color .2s;}
    .atr-card-pass{border-left:4px solid #3fb950;}
    .atr-card-fail{border-left:4px solid #f85149;}
    .atr-card-warn{border-left:4px solid #e3b341;}

    .pair-name{font-size:15px;font-weight:700;color:#e6edf3;}
    .atr-values{font-size:12px;font-family:monospace;color:#8b949e;margin-top:4px;}
    .check-badge{border-radius:6px;padding:3px 10px;font-size:12px;font-weight:700;display:inline-block;}
    .check-pass{background:#0d2f1a;color:#3fb950;border:1px solid #238636;}
    .check-fail{background:#2f0d0d;color:#f85149;border:1px solid #8b2d2d;}
    .check-warn{background:#2f1f0d;color:#e3b341;border:1px solid #9e6a03;}

    /* Progress bar */
    .prog-track{background:var(--border,#21262d);border-radius:6px;height:6px;margin:6px 0;overflow:hidden;}

    /* Metric */
    .metric-box{background:var(--background-color);border:1px solid var(--border,#21262d);border-radius:8px;padding:12px 14px;text-align:center;}
    .metric-value{font-size:20px;font-weight:700;color:#c9d1d9;}
    .metric-label{font-size:10px;color:#8b949e;margin-top:2px;font-weight:500;letter-spacing:.05em;text-transform:uppercase;}

    /* Rank table rows */
    .rank-row{display:flex;align-items:center;padding:8px 4px;border-bottom:1px solid #21262d;font-size:13px;}
    .rank-row:last-child{border-bottom:none;}
    .rank-num{width:28px;font-size:11px;font-weight:700;color:#484f58;}

    [data-testid="stSidebarNav"]{display:none;}
    #MainMenu,footer,header{visibility:hidden;}
    [data-testid="stSidebarCollapsedControl"]{visibility:visible !important;}
    .block-container{padding-top:1.5rem;max-width:1400px;}
</style>
""", unsafe_allow_html=True)

# ── Instrument registry ────────────────────────────────────────────────────────
INSTRUMENTS = {
    "EUR/USD": {"ticker": "EURUSD=X", "pip_size": 0.0001, "pip_label": "0.0001"},
    "GBP/USD": {"ticker": "GBPUSD=X", "pip_size": 0.0001, "pip_label": "0.0001"},
    "AUD/USD": {"ticker": "AUDUSD=X", "pip_size": 0.0001, "pip_label": "0.0001"},
    "NZD/USD": {"ticker": "NZDUSD=X", "pip_size": 0.0001, "pip_label": "0.0001"},
    "USD/JPY": {"ticker": "USDJPY=X", "pip_size": 0.01,   "pip_label": "0.01"},
    "USD/CHF": {"ticker": "USDCHF=X", "pip_size": 0.0001, "pip_label": "0.0001"},
    "USD/CAD": {"ticker": "USDCAD=X", "pip_size": 0.0001, "pip_label": "0.0001"},
    "EUR/GBP": {"ticker": "EURGBP=X", "pip_size": 0.0001, "pip_label": "0.0001"},
    "EUR/JPY": {"ticker": "EURJPY=X", "pip_size": 0.01,   "pip_label": "0.01"},
    "GBP/JPY": {"ticker": "GBPJPY=X", "pip_size": 0.01,   "pip_label": "0.01"},
    "AUD/JPY": {"ticker": "AUDJPY=X", "pip_size": 0.01,   "pip_label": "0.01"},
    "EUR/AUD": {"ticker": "EURAUD=X", "pip_size": 0.0001, "pip_label": "0.0001"},
    "GBP/AUD": {"ticker": "GBPAUD=X", "pip_size": 0.0001, "pip_label": "0.0001"},
    "EUR/CAD": {"ticker": "EURCAD=X", "pip_size": 0.0001, "pip_label": "0.0001"},
    "GBP/CAD": {"ticker": "GBPCAD=X", "pip_size": 0.0001, "pip_label": "0.0001"},
    "USD/ZAR": {"ticker": "USDZAR=X", "pip_size": 0.0001, "pip_label": "0.0001"},
    "EUR/ZAR": {"ticker": "EURZAR=X", "pip_size": 0.0001, "pip_label": "0.0001"},
    "GBP/ZAR": {"ticker": "GBPZAR=X", "pip_size": 0.0001, "pip_label": "0.0001"},
    "🥇 Gold":  {"ticker": "GC=F",    "pip_size": 0.10,   "pip_label": "0.10"},
    "🥈 Silver":  {"ticker": "SI=F",  "pip_size": 0.01,  "pip_label": "0.01"},
    "🪙 Platinum":{"ticker": "PL=F",  "pip_size": 0.10,  "pip_label": "0.10"},
}

# Typical retail broker spreads in pips (used for Spread/ATR ratio check; >5% = caution)
TYPICAL_SPREADS = {
    "EUR/USD": 1.2, "GBP/USD": 1.5, "AUD/USD": 1.5, "NZD/USD": 2.0,
    "USD/JPY": 1.2, "USD/CHF": 2.0, "USD/CAD": 2.0, "EUR/GBP": 1.5,
    "EUR/JPY": 2.0, "GBP/JPY": 3.0, "AUD/JPY": 2.5, "EUR/AUD": 3.0,
    "GBP/AUD": 3.5, "EUR/CAD": 3.0, "GBP/CAD": 3.5, "USD/ZAR": 80.0,
    "EUR/ZAR": 90.0, "GBP/ZAR": 90.0,
    "🥇 Gold": 3.0, "🥈 Silver": 5.0, "🪙 Platinum": 8.0,
}

# ── ATR Calculation ────────────────────────────────────────────────────────────
def calc_atr(high, low, close, period):
    prev  = close.shift(1)
    tr    = pd.concat([high - low,
                       (high - prev).abs(),
                       (low  - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

# ── Data Fetcher ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def fetch_instrument_data(ticker: str, pip_size: float, period: str = "60d", interval: str = "1d"):
    try:
        df = yf.download(ticker, period=period, interval=interval,
                         progress=False, auto_adjust=True)
        if df.empty or len(df) < 25:
            return None

        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]

        df["atr14"] = calc_atr(df["High"], df["Low"], df["Close"], 14)
        df["atr20"] = calc_atr(df["High"], df["Low"], df["Close"], 20)
        df = df.dropna(subset=["atr14", "atr20"])

        if df.empty:
            return None

        latest       = df.iloc[-1]
        atr14        = float(latest["atr14"])
        atr20        = float(latest["atr20"])
        atr14_pips   = round(atr14 / pip_size, 1)
        atr20_pips   = round(atr20 / pip_size, 1)
        atr_ok       = atr14 > atr20
        ratio        = round(atr14 / atr20, 3) if atr20 else 0
        pct_diff     = round((atr14 - atr20) / atr20 * 100, 1) if atr20 else 0

        # SL / TP from ATR14
        sl_pips  = round(atr14_pips * 1.5, 1)
        tp1_pips = round(sl_pips * 2, 1)
        tp2_pips = round(sl_pips * 3, 1)

        # Last 30 rows of ATR history for sparkline
        hist = df[["atr14", "atr20"]].tail(30).copy()
        hist.index = pd.to_datetime(hist.index)

        # 5-day ATR change
        atr14_5d_ago = float(df["atr14"].iloc[-6]) if len(df) >= 6 else atr14
        atr14_chg    = round((atr14 - atr14_5d_ago) / atr14_5d_ago * 100, 1) if atr14_5d_ago else 0

        return {
            "atr14": round(atr14, 5), "atr20": round(atr20, 5),
            "atr14_pips": atr14_pips, "atr20_pips": atr20_pips,
            "atr_ok": atr_ok, "ratio": ratio, "pct_diff": pct_diff,
            "sl_pips": sl_pips, "tp1_pips": tp1_pips, "tp2_pips": tp2_pips,
            "atr14_chg": atr14_chg,
            "history": hist,
            "close": round(float(latest["Close"]), 5),
        }
    except Exception:
        return None

# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 📊 ATR Volatility")

    period_options = {"30 Days": "30d", "60 Days": "60d", "90 Days": "90d", "6 Months": "6mo"}
    selected_period = st.selectbox("Lookback Period", list(period_options.keys()), index=1)
    fetch_period    = period_options[selected_period]

    interval_options = {"Daily (1D)": "1d", "4-Hour (4H)": "4h", "1-Hour (1H)": "1h"}
    selected_interval = st.selectbox("Timeframe", list(interval_options.keys()), index=0)
    fetch_interval    = interval_options[selected_interval]

    focus_pair = st.selectbox("Focus Pair (Detail Chart)", list(INSTRUMENTS.keys()), index=0)

    st.markdown("---")
    sort_by = st.radio("Sort Cards By", ["ATR Ratio (High→Low)", "ATR Pips (High→Low)", "Pair Name"], index=0)
    show_only_pass = st.checkbox("Show only ✅ ATR14 > ATR20", value=False)
    st.markdown("---")

    if st.button("🔄 Refresh All ATR Data", use_container_width=True, type="primary"):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    render_sidebar_nav()
# ══════════════════════════════════════════════════════════════════
# FETCH ALL INSTRUMENTS
# ══════════════════════════════════════════════════════════════════
# Guard: yfinance caps intraday data at 60 days
if fetch_interval in {"4h", "1h"} and fetch_period in {"90d", "6mo"}:
    st.warning(
        f"⚠️ **{selected_interval} data is limited to 60 days by yfinance.** "
        f"Lookback auto-capped from {selected_period} → 60 Days.",
        icon="⚠️",
    )
    fetch_period = "60d"

progress_bar = st.progress(0, text="📡 Fetching ATR data for all instruments…")
results = {}
total = len(INSTRUMENTS)

for idx, (pair, cfg) in enumerate(INSTRUMENTS.items()):
    progress_bar.progress((idx + 1) / total,
                          text=f"📡 Fetching {pair} ({idx+1}/{total})…")
    data = fetch_instrument_data(cfg["ticker"], cfg["pip_size"], fetch_period, fetch_interval)
    results[pair] = data

progress_bar.empty()

# Augment results with spread/ATR ratio
for _pair, _data in results.items():
    if _data is not None:
        _spread = TYPICAL_SPREADS.get(_pair, 0)
        _atr_pips = _data.get("atr14_pips", 0)
        _data["spread_pips"] = _spread
        _data["spread_pct"] = round(_spread / _atr_pips * 100, 1) if _atr_pips > 0 else 0

# Separate loaded vs failed
loaded  = {p: d for p, d in results.items() if d is not None}
failed  = [p for p, d in results.items() if d is None]
passing = {p: d for p, d in loaded.items() if d["atr_ok"]}
failing = {p: d for p, d in loaded.items() if not d["atr_ok"]}

# ══════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════
now_str = datetime.now().strftime("%A, %d %B %Y  |  %H:%M")
st.markdown(f"""
<div class="hero">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:12px;">
    <div>
      <div style="font-size:26px;font-weight:700;color:#e6edf3;">📊 ATR Volatility Scanner</div>
      <div style="color:#8b949e;font-size:14px;margin-top:6px;">
        ATR(14) vs ATR(20) · Checklist Item #4 · {selected_interval} · {selected_period} lookback
      </div>
      <div style="font-size:13px;color:#388bfd;font-weight:500;margin-top:4px;">
        🕐 {now_str} · SL = 1.5×ATR14 · TP1 = 2:1 · TP2 = 3:1
      </div>
    </div>
    <div style="display:flex;gap:10px;flex-wrap:wrap;align-items:flex-start;">
      <div class="metric-box" style="min-width:80px;">
        <div class="metric-value" style="color:#3fb950;">{len(passing)}</div>
        <div class="metric-label">✅ ATR Pass</div>
      </div>
      <div class="metric-box" style="min-width:80px;">
        <div class="metric-value" style="color:#f85149;">{len(failing)}</div>
        <div class="metric-label">❌ Low Vol</div>
      </div>
      <div class="metric-box" style="min-width:80px;">
        <div class="metric-value" style="color:#8b949e;">{len(failed)}</div>
        <div class="metric-label">⚠️ No Data</div>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# PASS / FAIL SUMMARY BAR
# ══════════════════════════════════════════════════════════════════
if loaded:
    pass_pct = int(len(passing) / len(loaded) * 100)
    st.markdown(f"""
    <div style="margin-bottom:24px;">
      <div style="display:flex;justify-content:space-between;margin-bottom:5px;">
        <span style="font-size:12px;color:#8b949e;font-weight:500;">MARKET VOLATILITY HEALTH</span>
        <span style="font-size:12px;color:#{'3fb950' if pass_pct>=60 else 'e3b341' if pass_pct>=40 else 'f85149'};font-weight:700;">
          {len(passing)}/{len(loaded)} pairs have ATR14 &gt; ATR20
        </span>
      </div>
      <div style="background:#21262d;border-radius:8px;height:12px;overflow:hidden;">
        <div style="background:linear-gradient(90deg,#238636,#3fb950);width:{pass_pct}%;height:100%;border-radius:8px;"></div>
      </div>
      <div style="display:flex;justify-content:space-between;font-size:10px;color:#484f58;margin-top:3px;">
        <span>0% passing</span><span>50%</span><span>100% passing</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# KPI STRIP
# ══════════════════════════════════════════════════════════════════
if passing:
    best_ratio_pair = max(passing, key=lambda p: passing[p]["ratio"])
    best_ratio      = passing[best_ratio_pair]["ratio"]
    highest_pips_pair = max(loaded, key=lambda p: loaded[p]["atr14_pips"])
    highest_pips    = loaded[highest_pips_pair]["atr14_pips"]
    lowest_pips_pair = min(loaded, key=lambda p: loaded[p]["atr14_pips"])
    lowest_pips     = loaded[lowest_pips_pair]["atr14_pips"]
    avg_ratio       = round(np.mean([d["ratio"] for d in loaded.values()]), 3)

    k1,k2,k3,k4,k5,k6 = st.columns(6)
    kpis = [
        (k1, f"{len(passing)}/{len(loaded)}", "ATR14 > ATR20",   "#3fb950" if len(passing) >= len(loaded)//2 else "#f85149"),
        (k2, f"{avg_ratio:.3f}",              "Avg ATR Ratio",   "#c9d1d9"),
        (k3, f"{best_ratio:.3f}",             f"Best Ratio\n{best_ratio_pair}", "#3fb950"),
        (k4, f"{highest_pips:.0f}",           f"Most Volatile\n{highest_pips_pair}", "#e3b341"),
        (k5, f"{lowest_pips:.0f}",            f"Least Volatile\n{lowest_pips_pair}", "#8b949e"),
        (k6, f"{len(failed)}",                "No Data",         "#484f58"),
    ]
    for col, val, lbl, color in kpis:
        with col:
            lbl_short = lbl.split("\n")[0]
            lbl_sub   = lbl.split("\n")[1] if "\n" in lbl else ""
            st.markdown(f"""
            <div class="metric-box">
              <div class="metric-value" style="color:{color};">{val}</div>
              <div class="metric-label">{lbl_short}</div>
              {'<div style="font-size:9px;color:#388bfd;margin-top:1px;">'+lbl_sub+'</div>' if lbl_sub else ''}
            </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# MAIN LAYOUT: Cards left, detail chart right
# ══════════════════════════════════════════════════════════════════
left, right = st.columns([2, 3], gap="large")

# ── Sort instruments ───────────────────────────────────────────────────────────
def sort_key(pair):
    d = loaded.get(pair)
    if d is None:
        return (-999, -999, pair)
    if sort_by == "ATR Ratio (High→Low)":
        return (-d["ratio"], 0, pair)
    elif sort_by == "ATR Pips (High→Low)":
        return (-d["atr14_pips"], 0, pair)
    else:
        return (0, 0, pair)

sorted_pairs = sorted(INSTRUMENTS.keys(), key=sort_key)
if show_only_pass:
    sorted_pairs = [p for p in sorted_pairs if p in passing]

# ══════════════════════════════════════════════════════════════════
# LEFT: ATR CARDS
# ══════════════════════════════════════════════════════════════════
with left:
    st.markdown("### 🗂️ All Instruments — ATR Status")

    for pair in sorted_pairs:
        d = loaded.get(pair)
        if d is None:
            st.markdown(f"""
            <div class="atr-card" style="border-left:4px solid #30363d;opacity:.5;">
              <div style="display:flex;justify-content:space-between;align-items:center;">
                <div class="pair-name">{pair}</div>
                <span class="check-badge check-warn">⚠️ No Data</span>
              </div>
              <div class="atr-values">Could not load — check connection</div>
            </div>
            """, unsafe_allow_html=True)
            continue

        atr_ok    = d["atr_ok"]
        card_cls  = "atr-card-pass" if atr_ok else "atr-card-fail"
        badge_cls = "check-pass" if atr_ok else "check-fail"
        badge_txt = "✅ ATR14 > ATR20" if atr_ok else "❌ ATR14 ≤ ATR20"
        status    = "Volatility supports follow-through" if atr_ok else "Low volatility — caution"
        status_c  = "#3fb950" if atr_ok else "#f85149"

        pct_diff = d["pct_diff"]
        diff_txt = f"+{pct_diff}% above avg" if pct_diff >= 0 else f"{pct_diff}% below avg"
        diff_c   = "#3fb950" if pct_diff >= 0 else "#f85149"

        # Change arrow
        chg      = d["atr14_chg"]
        chg_icon = "▲" if chg > 0 else "▼" if chg < 0 else "●"
        chg_c    = "#3fb950" if chg > 0 else "#f85149" if chg < 0 else "#8b949e"

        # Ratio bar (0.8 to 1.2 range)
        bar_pct = min(100, max(0, int((d["ratio"] - 0.8) / 0.4 * 100)))
        bar_c   = "#3fb950" if atr_ok else "#f85149"

        # Spread/ATR badge
        spread_pct = d.get("spread_pct", 0)
        spread_warn = spread_pct > 5
        spread_badge_html = (
            f'<span style="background:rgba(248,81,73,.15);border:1px solid rgba(248,81,73,.5);'
            f'border-radius:4px;padding:2px 7px;font-size:10px;color:#f85149;font-weight:600;margin-left:6px;">'
            f'⚠️ Spread {spread_pct:.1f}% ATR</span>'
        ) if spread_warn else (
            f'<span style="font-size:10px;color:#484f58;margin-left:6px;">Spread {spread_pct:.1f}%</span>'
        )

        # Highlight selected focus pair
        highlight = "border-color:#388bfd!important;" if pair == focus_pair else ""

        st.markdown(f"""
        <div class="atr-card {card_cls}" style="{highlight}">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:6px;">
            <div>
              <span class="pair-name">{pair}</span>
              <span style="font-size:11px;color:#8b949e;margin-left:8px;">@ {d['close']}</span>
              <span style="font-size:11px;color:{chg_c};margin-left:6px;">{chg_icon} {chg:+.1f}% (5d)</span>
              {spread_badge_html}
            </div>
            <span class="check-badge {badge_cls}">{badge_txt}</span>
          </div>

          <!-- Ratio bar -->
          <div class="prog-track">
            <div style="background:{bar_c};width:{bar_pct}%;height:100%;border-radius:6px;"></div>
          </div>
          <div style="display:flex;justify-content:space-between;font-size:10px;color:#484f58;margin-bottom:8px;">
            <span>Low</span><span>Ratio: <strong style="color:{bar_c};">{d['ratio']:.3f}</strong></span><span>High</span>
          </div>

          <!-- ATR values grid -->
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:8px;">
            <div style="background:#0d1117;border-radius:6px;padding:8px;text-align:center;">
              <div style="font-size:11px;color:#8b949e;">ATR(14)</div>
              <div style="font-size:14px;font-weight:700;font-family:monospace;color:#e6edf3;">{d['atr14']:.5f}</div>
              <div style="font-size:11px;color:#388bfd;">{d['atr14_pips']} pips</div>
            </div>
            <div style="background:#0d1117;border-radius:6px;padding:8px;text-align:center;">
              <div style="font-size:11px;color:#8b949e;">ATR(20)</div>
              <div style="font-size:14px;font-weight:700;font-family:monospace;color:#8b949e;">{d['atr20']:.5f}</div>
              <div style="font-size:11px;color:#8b949e;">{d['atr20_pips']} pips</div>
            </div>
          </div>

          <!-- SL/TP auto levels -->
          <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:4px;margin-bottom:8px;">
            <div style="background:#1a0d0d;border-radius:5px;padding:5px;text-align:center;">
              <div style="font-size:9px;color:#8b949e;">SL</div>
              <div style="font-size:12px;font-weight:700;color:#f85149;">{d['sl_pips']}p</div>
            </div>
            <div style="background:#0d1a0d;border-radius:5px;padding:5px;text-align:center;">
              <div style="font-size:9px;color:#8b949e;">TP1</div>
              <div style="font-size:12px;font-weight:700;color:#3fb950;">{d['tp1_pips']}p</div>
            </div>
            <div style="background:#0d1a0d;border-radius:5px;padding:5px;text-align:center;">
              <div style="font-size:9px;color:#8b949e;">TP2</div>
              <div style="font-size:12px;font-weight:700;color:#3fb950;">{d['tp2_pips']}p</div>
            </div>
          </div>

          <div style="font-size:11px;color:{status_c};font-weight:600;">
            {status} &nbsp;·&nbsp; <span style="color:{diff_c};">{diff_txt}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# RIGHT: DETAIL CHARTS
# ══════════════════════════════════════════════════════════════════
with right:
    d = loaded.get(focus_pair)

    st.markdown(f"### 📈 {focus_pair} — ATR Detail")

    if d is None:
        st.warning(f"No data available for {focus_pair}. Try refreshing or selecting a different pair.")
    else:
        # ── Status header ──────────────────────────────────────────
        ok        = d["atr_ok"]
        ok_color  = "#3fb950" if ok else "#f85149"
        ok_icon   = "✅" if ok else "❌"
        ok_txt    = "ATR(14) > ATR(20) — Volatility supports follow-through" \
            if ok else "ATR(14) ≤ ATR(20) — Low volatility caution"

        st.markdown(f"""
        <div style="background:{'#0d2f1a' if ok else '#2f0d0d'};border:1px solid {'#238636' if ok else '#8b2d2d'};
             border-radius:10px;padding:14px 18px;margin-bottom:16px;display:flex;justify-content:space-between;align-items:center;">
          <div>
            <div style="font-size:16px;font-weight:700;color:{ok_color};">{ok_icon} {ok_txt}</div>
            <div style="font-size:12px;color:#8b949e;margin-top:4px;">
              ATR14 = {d['atr14']:.5f} ({d['atr14_pips']} pips) &nbsp;·&nbsp;
              ATR20 = {d['atr20']:.5f} ({d['atr20_pips']} pips) &nbsp;·&nbsp;
              Ratio = {d['ratio']:.3f} &nbsp;·&nbsp;
              Diff = {d['pct_diff']:+.1f}%
            </div>
          </div>
          <div style="text-align:right;">
            <div style="font-size:22px;font-weight:700;color:{ok_color};">{d['ratio']:.3f}</div>
            <div style="font-size:10px;color:#8b949e;">ATR14/ATR20</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── ATR History Chart ──────────────────────────────────────
        hist = d["history"]
        cfg  = INSTRUMENTS[focus_pair]

        hist_atr14 = hist["atr14"] / cfg["pip_size"]
        hist_atr20 = hist["atr20"] / cfg["pip_size"]
        dates      = hist.index

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.65, 0.35],
            vertical_spacing=0.06,
            subplot_titles=["ATR(14) vs ATR(20) — Pips", "ATR Ratio (ATR14 ÷ ATR20)"],
        )

        # Conditional fill: green where ATR14 > ATR20, red where ATR14 <= ATR20
        # Build polygon segments so the colour flips correctly at every crossover
        _d  = list(dates)
        _a  = list(hist_atr14.values)
        _b  = list(hist_atr20.values)
        _s  = 0
        for _i in range(1, len(_d) + 1):
            _end     = _i == len(_d)
            _changed = _end or ((_a[_i] >= _b[_i]) != (_a[_s] >= _b[_s]))
            if _changed:
                _sd, _sa, _sb = _d[_s:_i], _a[_s:_i], _b[_s:_i]
                _clr = "rgba(63,185,80,0.15)" if _a[_s] >= _b[_s] else "rgba(248,81,73,0.15)"
                fig.add_trace(go.Scatter(
                    x=_sd + _sd[::-1], y=_sa + _sb[::-1],
                    fill="toself", fillcolor=_clr,
                    line=dict(width=0), showlegend=False, hoverinfo="skip",
                ), row=1, col=1)
                _s = _i

        # ATR lines drawn on top of the fill
        fig.add_trace(go.Scatter(
            x=dates, y=hist_atr14, name="ATR(14)",
            mode="lines", line=dict(color="#388bfd", width=2.5),
            hovertemplate="ATR14: %{y:.1f} pips<extra></extra>",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=dates, y=hist_atr20, name="ATR(20)",
            mode="lines", line=dict(color="#8b949e", width=1.5, dash="dot"),
            hovertemplate="ATR20: %{y:.1f} pips<extra></extra>",
        ), row=1, col=1)

        # ATR Ratio
        ratio_series = hist["atr14"] / hist["atr20"]
        ratio_colors = ["#3fb950" if v >= 1 else "#f85149" for v in ratio_series]
        fig.add_trace(go.Bar(
            x=dates, y=ratio_series,
            marker_color=ratio_colors,
            name="Ratio",
            hovertemplate="Ratio: %{y:.3f}<extra></extra>",
        ), row=2, col=1)
        fig.add_hline(y=1.0, line_dash="dot", line_color="#e3b341",
                      line_width=1.5, row=2, col=1,
                      annotation_text="1.0 threshold",
                      annotation_font_color="#e3b341",
                      annotation_position="right")

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1117",
            legend=dict(font=dict(color="#8b949e", size=11),
                        bgcolor="#161b22", bordercolor="#21262d",
                        orientation="h", x=0, y=1.08),
            margin=dict(l=10, r=60, t=30, b=10),
            height=400,
            hovermode="x unified",
        )
        for row in [1, 2]:
            fig.update_xaxes(showgrid=False, tickfont=dict(color="#8b949e", size=9),
                             linecolor="#21262d", row=row, col=1)
            fig.update_yaxes(showgrid=True, gridcolor="#21262d",
                             tickfont=dict(color="#8b949e", size=9), row=row, col=1)

        fig.update_annotations(font_color="#8b949e", font_size=11)
        st.plotly_chart(fig, use_container_width=True, config=dict(displayModeBar=False))

        # ── All-pairs ATR comparison bar chart ────────────────────
        st.markdown("### 📊 All Pairs — ATR(14) Pips Comparison")

        pairs_loaded = [p for p in sorted(loaded.keys(),
                                          key=lambda x: -loaded[x]["atr14_pips"])]
        atr14_vals   = [loaded[p]["atr14_pips"] for p in pairs_loaded]
        atr20_vals   = [loaded[p]["atr20_pips"] for p in pairs_loaded]
        bar_colors   = ["#3fb950" if loaded[p]["atr_ok"] else "#f85149" for p in pairs_loaded]
        focus_colors = ["#388bfd" if p == focus_pair else c
                        for p, c in zip(pairs_loaded, bar_colors)]

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=pairs_loaded, y=atr14_vals,
            name="ATR(14)",
            marker_color=focus_colors,
            hovertemplate="<b>%{x}</b><br>ATR14: %{y:.1f} pips<extra></extra>",
        ))
        fig2.add_trace(go.Scatter(
            x=pairs_loaded, y=atr20_vals,
            name="ATR(20)",
            mode="markers+lines",
            marker=dict(color="#8b949e", size=6, symbol="diamond"),
            line=dict(color="#8b949e", width=1, dash="dot"),
            hovertemplate="<b>%{x}</b><br>ATR20: %{y:.1f} pips<extra></extra>",
        ))
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1117",
            margin=dict(l=10, r=10, t=10, b=10), height=280,
            legend=dict(font=dict(color="#8b949e", size=11), bgcolor="#161b22",
                        bordercolor="#21262d", orientation="h", x=0, y=1.08),
            xaxis=dict(tickfont=dict(color="#8b949e", size=9), tickangle=-35,
                       showgrid=False, linecolor="#21262d"),
            yaxis=dict(tickfont=dict(color="#8b949e"), showgrid=True,
                       gridcolor="#21262d", ticksuffix=" p"),
            bargap=0.25,
        )
        st.plotly_chart(fig2, use_container_width=True, config=dict(displayModeBar=False))

        # ── Volatility Ranking Table ───────────────────────────────
        st.markdown("### 🏆 Volatility Ranking")
        rank_rows = []
        for rank, pair in enumerate(pairs_loaded, 1):
            d2 = loaded[pair]
            _sp = d2.get("spread_pct", 0)
            rank_rows.append({
                "Rank":          rank,
                "Pair":          pair,
                "ATR14":         d2["atr14"],
                "ATR20":         d2["atr20"],
                "ATR14 Pips":    d2["atr14_pips"],
                "ATR20 Pips":    d2["atr20_pips"],
                "Ratio":         d2["ratio"],
                "Diff %":        f"{d2['pct_diff']:+.1f}%",
                "Spread/ATR %":  _sp,
                "SL Pips":       d2["sl_pips"],
                "TP1 Pips":      d2["tp1_pips"],
                "TP2 Pips":      d2["tp2_pips"],
                "5d Chg":        f"{d2['atr14_chg']:+.1f}%",
                "Status":        "✅ Pass" if d2["atr_ok"] else "❌ Caution",
            })

        df_rank = pd.DataFrame(rank_rows)
        st.dataframe(
            df_rank,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Rank":       st.column_config.NumberColumn("🏆", width="small"),
                "Pair":       st.column_config.TextColumn("Pair"),
                "ATR14":      st.column_config.NumberColumn("ATR14",      format="%.5f"),
                "ATR20":      st.column_config.NumberColumn("ATR20",      format="%.5f"),
                "ATR14 Pips": st.column_config.NumberColumn("ATR14 Pips", format="%.1f"),
                "ATR20 Pips": st.column_config.NumberColumn("ATR20 Pips", format="%.1f"),
                "Ratio":         st.column_config.NumberColumn("Ratio",        format="%.3f"),
                "Diff %":        st.column_config.TextColumn("Diff %"),
                "Spread/ATR %":  st.column_config.NumberColumn("Spread/ATR %",  format="%.1f"),
                "SL Pips":       st.column_config.NumberColumn("SL Pips",       format="%.1f"),
                "TP1 Pips":   st.column_config.NumberColumn("TP1 Pips",   format="%.1f"),
                "TP2 Pips":   st.column_config.NumberColumn("TP2 Pips",   format="%.1f"),
                "5d Chg":     st.column_config.TextColumn("5d Chg"),
                "Status":     st.column_config.TextColumn("Status"),
            }
        )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;color:#484f58;font-size:11px;margin-top:32px;padding-top:16px;border-top:1px solid #21262d;">
  📊 ATR Volatility Scanner · ATR = Wilder EWM · SL = 1.5×ATR14 · TP1 = 2:1 · TP2 = 3:1 · Not financial advice
</div>
""", unsafe_allow_html=True)