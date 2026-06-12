import streamlit as st
from src.pages_lib.navigation import render_sidebar_nav
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime

# ── Page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="R:R Calculator",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
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

    /* Verdict banners */
    .verdict-pass{ background:linear-gradient(135deg,#0d3a1f,#0d5e32);
                   border:2px solid #238636; border-radius:14px;
                   padding:22px 28px; text-align:center; margin-bottom:18px; }
    .verdict-warn{ background:linear-gradient(135deg,#1c1c0d,#2e2a0d);
                   border:2px solid #9e6a03; border-radius:14px;
                   padding:22px 28px; text-align:center; margin-bottom:18px; }
    .verdict-fail{ background:linear-gradient(135deg,#3a0d0d,#5e1414);
                   border:2px solid #8b2d2d; border-radius:14px;
                   padding:22px 28px; text-align:center; margin-bottom:18px; }

    /* R:R visual bar */
    .rr-track{ background:#21262d; border-radius:6px; height:28px;
               margin:10px 0; overflow:hidden; display:flex; }

    /* Trade levels table */
    .lvl-row{ display:flex; justify-content:space-between; align-items:center;
              padding:10px 16px; border-bottom:1px solid #21262d; font-size:13px; }
    .lvl-row:last-child{ border-bottom:none; }
    .lvl-label{ color:#8b949e; }
    .lvl-val  { font-family:'JetBrains Mono',monospace; font-weight:600; color:#c9d1d9; }

    /* Scenario table */
    .sc-row{ display:flex; gap:0; border-bottom:1px solid #21262d; font-size:12px; }
    .sc-row:last-child{ border-bottom:none; }
    .sc-cell{ flex:1; padding:8px 10px; font-family:'JetBrains Mono',monospace; }

    .explainer{ background:var(--background-color); border:1px solid #1e3a5f;
                border-left:3px solid #388bfd; border-radius:8px;
                padding:14px 18px; font-size:13px; color:var(--muted,#8b949e); line-height:1.7; }
    .formula-box{ background:#0d1117; border:1px solid #30363d; border-radius:8px;
                  padding:14px 18px; font-family:'JetBrains Mono',monospace;
                  font-size:13px; color:#c9d1d9; margin:10px 0; line-height:2.2; }
</style>
""", unsafe_allow_html=True)


# ── Instrument registry ────────────────────────────────────────────
INSTRUMENTS = {
    "EUR/USD":    {"ticker": "EURUSD=X", "pip": 10.0,  "pip_size": 0.0001},
    "GBP/USD":    {"ticker": "GBPUSD=X", "pip": 10.0,  "pip_size": 0.0001},
    "AUD/USD":    {"ticker": "AUDUSD=X", "pip": 10.0,  "pip_size": 0.0001},
    "NZD/USD":    {"ticker": "NZDUSD=X", "pip": 10.0,  "pip_size": 0.0001},
    "USD/JPY":    {"ticker": "USDJPY=X", "pip": 9.09,  "pip_size": 0.01},
    "USD/CHF":    {"ticker": "USDCHF=X", "pip": 10.8,  "pip_size": 0.0001},
    "USD/CAD":    {"ticker": "USDCAD=X", "pip": 7.4,   "pip_size": 0.0001},
    "EUR/GBP":    {"ticker": "EURGBP=X", "pip": 12.5,  "pip_size": 0.0001},
    "EUR/JPY":    {"ticker": "EURJPY=X", "pip": 9.09,  "pip_size": 0.01},
    "GBP/JPY":    {"ticker": "GBPJPY=X", "pip": 9.09,  "pip_size": 0.01},
    "AUD/JPY":    {"ticker": "AUDJPY=X", "pip": 9.09,  "pip_size": 0.01},
    "EUR/AUD":    {"ticker": "EURAUD=X", "pip": 6.3,   "pip_size": 0.0001},
    "GBP/AUD":    {"ticker": "GBPAUD=X", "pip": 6.3,   "pip_size": 0.0001},
    "EUR/CAD":    {"ticker": "EURCAD=X", "pip": 7.4,   "pip_size": 0.0001},
    "GBP/CAD":    {"ticker": "GBPCAD=X", "pip": 7.4,   "pip_size": 0.0001},
    "USD/ZAR":    {"ticker": "USDZAR=X", "pip": 0.55,  "pip_size": 0.0001},
    "EUR/ZAR":    {"ticker": "EURZAR=X", "pip": 0.55,  "pip_size": 0.0001},
    "GBP/ZAR":    {"ticker": "GBPZAR=X", "pip": 0.55,  "pip_size": 0.0001},
    "🥇 Gold":    {"ticker": "GC=F",     "pip": 10.0,  "pip_size": 0.10},
    "🥈 Silver":  {"ticker": "SI=F",     "pip": 10.0,  "pip_size": 0.01},
    "🪙 Platinum":{"ticker": "PL=F",     "pip": 10.0,  "pip_size": 0.10},
}


# ══════════════════════════════════════════════════════════════════
# DATA FETCH
# ══════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner=False)
def fetch_price(ticker: str) -> float | None:
    try:
        df = yf.download(ticker, interval="1d", period="5d",
                         progress=False, auto_adjust=True)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return float(df["Close"].iloc[-1])
    except Exception:
        return None

@st.cache_data(ttl=300, show_spinner=False)
def fetch_atr14(ticker: str) -> float | None:
    try:
        df = yf.download(ticker, interval="1d", period="60d",
                         progress=False, auto_adjust=True)
        if df.empty or len(df) < 15:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        prev = df["Close"].shift(1)
        tr   = pd.concat([
            df["High"] - df["Low"],
            (df["High"] - prev).abs(),
            (df["Low"]  - prev).abs(),
            ], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        return float(atr.iloc[-1])
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════
# CALCULATIONS
# ══════════════════════════════════════════════════════════════════

def compute_rr(entry: float, sl: float, tp1: float, tp2: float,
               pip_size: float, pip_val: float,
               account_bal: float, risk_pct: float,
               partial_pct: float) -> dict:

    sl_dist   = abs(entry - sl)
    tp1_dist  = abs(tp1   - entry)
    tp2_dist  = abs(tp2   - entry)

    sl_pips   = sl_dist  / pip_size
    tp1_pips  = tp1_dist / pip_size
    tp2_pips  = tp2_dist / pip_size

    rr_tp1    = tp1_dist / sl_dist if sl_dist else 0
    rr_tp2    = tp2_dist / sl_dist if sl_dist else 0

    risk_amt  = account_bal * (risk_pct / 100)
    lot_size  = risk_amt / (sl_pips * pip_val) if sl_pips else 0

    # Expected value at partial close
    partial_lots  = lot_size * (partial_pct / 100)
    runner_lots   = lot_size - partial_lots
    tp1_profit    = partial_lots  * tp1_pips * pip_val
    tp2_profit    = runner_lots   * tp2_pips * pip_val
    sl_loss       = lot_size      * sl_pips  * pip_val

    # Win rate breakeven
    be_winrate_tp1 = 1 / (1 + rr_tp1) * 100 if rr_tp1 else 0

    # Grade
    if rr_tp1 >= 3.0:
        grade, label, color, vclass = "excellent", "✅ EXCELLENT  R:R", "#3fb950", "verdict-pass"
    elif rr_tp1 >= 2.0:
        grade, label, color, vclass = "pass",      "✅ R:R MINIMUM MET", "#3fb950", "verdict-pass"
    elif rr_tp1 >= 1.5:
        grade, label, color, vclass = "marginal",  "⚠️ MARGINAL — REVIEW", "#e3b341", "verdict-warn"
    else:
        grade, label, color, vclass = "fail",      "❌ R:R BELOW MINIMUM", "#f85149", "verdict-fail"

    long_trade = tp1 > entry

    return {
        "entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2,
        "sl_dist": sl_dist, "tp1_dist": tp1_dist, "tp2_dist": tp2_dist,
        "sl_pips": sl_pips, "tp1_pips": tp1_pips, "tp2_pips": tp2_pips,
        "rr_tp1": rr_tp1, "rr_tp2": rr_tp2,
        "risk_amt": risk_amt, "lot_size": lot_size,
        "tp1_profit": tp1_profit, "tp2_profit": tp2_profit,
        "sl_loss": sl_loss, "total_profit": tp1_profit + tp2_profit,
        "be_winrate_tp1": be_winrate_tp1,
        "grade": grade, "label": label, "color": color, "vclass": vclass,
        "long_trade": long_trade,
        "partial_lots": partial_lots, "runner_lots": runner_lots,
    }


# ══════════════════════════════════════════════════════════════════
# CHART
# ══════════════════════════════════════════════════════════════════

def build_rr_chart(c: dict, pair: str) -> go.Figure:
    """
    Visual R:R diagram — horizontal bar showing SL zone, entry, TP1, TP2
    as proportional segments, plus a P&L curve chart.
    """
    direction = 1 if c["long_trade"] else -1
    levels = {
        "SL":    c["sl"],
        "Entry": c["entry"],
        "TP1":   c["tp1"],
        "TP2":   c["tp2"],
    }
    sorted_lvls = sorted(levels.items(), key=lambda x: x[1])
    y_min = min(v for _, v in sorted_lvls) * (0.9998 if c["long_trade"] else 0.9995)
    y_max = max(v for _, v in sorted_lvls) * (1.0002 if c["long_trade"] else 1.0005)

    fig = go.Figure()

    # ── SL zone (red) ─────────────────────────────────────────────
    fig.add_shape(type="rect",
                  x0=0, x1=1,
                  y0=min(c["sl"], c["entry"]), y1=max(c["sl"], c["entry"]),
                  fillcolor="rgba(248,81,73,0.13)", line_color="#f85149", line_width=1.5,
                  )
    # ── TP1 zone (green) ──────────────────────────────────────────
    fig.add_shape(type="rect",
                  x0=0, x1=1,
                  y0=min(c["tp1"], c["entry"]), y1=max(c["tp1"], c["entry"]),
                  fillcolor="rgba(63,185,80,0.13)", line_color="#3fb950", line_width=1.5,
                  )
    # ── TP2 extension (lighter green) ─────────────────────────────
    fig.add_shape(type="rect",
                  x0=0, x1=1,
                  y0=min(c["tp2"], c["tp1"]), y1=max(c["tp2"], c["tp1"]),
                  fillcolor="rgba(63,185,80,0.07)", line_color="#56d364", line_width=1,
                  line_dash="dot",
                  )

    colors = {"SL":"#f85149","Entry":"#ffffff","TP1":"#3fb950","TP2":"#56d364"}
    for name, val in levels.items():
        fig.add_hline(
            y=val, line_color=colors[name], line_width=2,
            line_dash="solid" if name in ("Entry","SL") else "dot",
            annotation_text=f"  {name}  {val:.5f}",
            annotation_position="right",
            annotation_font=dict(size=11, color=colors[name]),
        )

    # Labels for distances
    mid_sl  = (c["entry"] + c["sl"])  / 2
    mid_tp1 = (c["entry"] + c["tp1"]) / 2
    mid_tp2 = (c["tp1"]   + c["tp2"]) / 2

    for y, text, color in [
        (mid_sl,  f"SL: {c['sl_pips']:.1f} pips  (1R)",      "#f85149"),
        (mid_tp1, f"TP1: {c['tp1_pips']:.1f} pips  ({c['rr_tp1']:.1f}R)", "#3fb950"),
        (mid_tp2, f"TP2: {c['tp2_pips']:.1f} pips  ({c['rr_tp2']:.1f}R)", "#56d364"),
    ]:
        fig.add_annotation(
            x=0.5, y=y, text=text,
            showarrow=False, font=dict(size=11, color=color),
            bgcolor="rgba(13,17,23,0.80)", borderpad=4,
        )

    fig.update_layout(
        height=460,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#161b22",
        font=dict(family="Inter, sans-serif", size=11, color="#8b949e"),
        margin=dict(l=10, r=160, t=30, b=20),
        title=dict(text=f"<b>{pair}</b> — R:R Visual Diagram",
                   font=dict(color="#e6edf3", size=14), x=0.01),
        xaxis=dict(showticklabels=False, showgrid=False,
                   zeroline=False, range=[0,1]),
        yaxis=dict(showgrid=True, gridcolor="#21262d",
                   range=[y_min, y_max], zeroline=False),
        showlegend=False,
    )
    return fig


def build_winrate_chart(c: dict) -> go.Figure:
    """
    Shows the required win-rate breakeven curve across different R:R ratios.
    """
    rr_range = np.arange(0.5, 5.1, 0.1)
    be_wr    = [1/(1+r)*100 for r in rr_range]

    fig = go.Figure()

    # Breakeven curve
    fig.add_trace(go.Scatter(
        x=rr_range, y=be_wr,
        mode="lines", line=dict(color="#388bfd", width=2.5),
        name="Breakeven win rate",
        fill="tozeroy", fillcolor="rgba(56,139,253,0.07)",
    ))

    # Minimum 2:1 line
    fig.add_vline(x=2.0, line_color="rgba(227,179,65,0.40)", line_dash="dash", line_width=1.5,
                  annotation_text="Min 2:1", annotation_position="top",
                  annotation_font=dict(size=9, color="#e3b341"))

    # Current R:R marker
    current_be = 1/(1+c["rr_tp1"])*100 if c["rr_tp1"] else 50
    fig.add_trace(go.Scatter(
        x=[c["rr_tp1"]], y=[current_be],
        mode="markers+text",
        marker=dict(size=14, color=c["color"],
                    line=dict(color="#0d1117", width=2)),
        text=[f" {c['rr_tp1']:.2f}:1  ({current_be:.1f}% WR needed)"],
        textposition="middle right",
        textfont=dict(size=11, color=c["color"]),
        name="This trade",
    ))

    fig.update_layout(
        height=280,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#161b22",
        font=dict(family="Inter, sans-serif", size=11, color="#8b949e"),
        margin=dict(l=10, r=20, t=30, b=20),
        title=dict(text="Breakeven Win Rate required at each R:R",
                   font=dict(color="#e6edf3", size=13), x=0.01),
        xaxis=dict(title="R:R Ratio", gridcolor="#21262d",
                   showgrid=True, zeroline=False),
        yaxis=dict(title="Win rate needed (%)", gridcolor="#21262d",
                   showgrid=True, zeroline=False, range=[0, 80]),
        showlegend=False,
    )
    return fig


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### ⚖️ R:R Calculator")

    inst_keys    = list(INSTRUMENTS.keys())
    default_inst = st.session_state.get("selected_instrument", "EUR/USD")
    if default_inst not in inst_keys:
        default_inst = inst_keys[0]

    selected_pair = st.selectbox("Instrument", inst_keys,
                                 index=inst_keys.index(default_inst))

    inst     = INSTRUMENTS[selected_pair]
    pip_size = inst["pip_size"]
    pip_val  = inst["pip"]

    st.divider()

    # Auto-fetch live price
    if st.button("⚡ Fetch Live Price", use_container_width=True, type="primary"):
        st.cache_data.clear()
        st.rerun()

    with st.spinner("Fetching price…"):
        live_price = fetch_price(inst["ticker"])
        live_atr   = fetch_atr14(inst["ticker"])

    default_entry = live_price if live_price else 1.10000
    default_sl_p  = round(live_atr * 1.5 / pip_size, 1) if live_atr else 101.4
    default_tp1_p = round(default_sl_p * 2, 1)
    default_tp2_p = round(default_sl_p * 3, 1)

    st.divider()
    st.markdown("**📐 Trade Levels**")
    direction = st.radio("Direction", ["LONG", "SHORT"], horizontal=True)

    entry = st.number_input("Entry Price",
                            value=round(default_entry, 5),
                            format="%.5f", step=float(pip_size))

    sl_pips_in  = st.number_input("Stop Loss (pips)",
                                  value=float(default_sl_p),
                                  min_value=1.0, step=0.5, format="%.1f")
    tp1_pips_in = st.number_input("TP1 (pips)",
                                  value=float(default_tp1_p),
                                  min_value=1.0, step=0.5, format="%.1f")
    tp2_pips_in = st.number_input("TP2 (pips)",
                                  value=float(default_tp2_p),
                                  min_value=1.0, step=0.5, format="%.1f")

    if direction == "LONG":
        sl_price  = entry - sl_pips_in  * pip_size
        tp1_price = entry + tp1_pips_in * pip_size
        tp2_price = entry + tp2_pips_in * pip_size
    else:
        sl_price  = entry + sl_pips_in  * pip_size
        tp1_price = entry - tp1_pips_in * pip_size
        tp2_price = entry - tp2_pips_in * pip_size

    st.divider()
    st.markdown("**💰 Position Sizing**")
    account_bal = st.number_input("Account Balance ($)",
                                  value=float(st.session_state.get("account_bal", 10000.0)),
                                  step=500.0, format="%.2f")
    risk_pct    = st.slider("Risk per trade (%)", 0.25, 3.0,
                            float(st.session_state.get("risk_pct", 1.0)), 0.25)
    partial_pct = st.slider("Partial close at TP1 (%)", 25, 75, 50, step=5)

    st.divider()
    st.markdown("**📏 Minimum R:R**")
    min_rr = st.number_input("Minimum R:R to pass", value=2.0,
                             min_value=1.0, max_value=5.0, step=0.5, format="%.1f")

    st.caption(f"🕐 {datetime.now().strftime('%H:%M:%S')} local")


    st.divider()
    render_sidebar_nav()
# ══════════════════════════════════════════════════════════════════
# MAIN PAGE
# ══════════════════════════════════════════════════════════════════

c = compute_rr(
    entry=entry, sl=sl_price, tp1=tp1_price, tp2=tp2_price,
    pip_size=pip_size, pip_val=pip_val,
    account_bal=account_bal, risk_pct=risk_pct,
    partial_pct=partial_pct,
)

# Regrade against custom min_rr
if c["rr_tp1"] >= min_rr * 1.5:
    c["grade"]  = "excellent"
    c["label"]  = f"✅ EXCELLENT  R:R"
    c["color"]  = "#3fb950"
    c["vclass"] = "verdict-pass"
elif c["rr_tp1"] >= min_rr:
    c["grade"]  = "pass"
    c["label"]  = f"✅ R:R MINIMUM MET  ({min_rr:.1f}:1)"
    c["color"]  = "#3fb950"
    c["vclass"] = "verdict-pass"
elif c["rr_tp1"] >= min_rr * 0.75:
    c["grade"]  = "marginal"
    c["label"]  = f"⚠️ MARGINAL — {c['rr_tp1']:.2f}:1 (min {min_rr:.1f}:1)"
    c["color"]  = "#e3b341"
    c["vclass"] = "verdict-warn"
else:
    c["grade"]  = "fail"
    c["label"]  = f"❌ BELOW MINIMUM — {c['rr_tp1']:.2f}:1 (need {min_rr:.1f}:1)"
    c["color"]  = "#f85149"
    c["vclass"] = "verdict-fail"

# Header
st.markdown(f"""
<div style="background:linear-gradient(135deg,#0d1117 0%,#161b22 50%,#0d1117 100%);
            border:1px solid #21262d; border-radius:16px; padding:24px 28px;
            margin-bottom:20px;">
  <div style="font-size:24px; font-weight:700; color:#e6edf3;">⚖️ R:R Calculator</div>
  <div style="color:#8b949e; font-size:13px; margin-top:4px;">
    TP1 = {c['tp1_pips']:.1f} pips → R:R {c['rr_tp1']:.2f}:1
    {'✅' if c['rr_tp1'] >= min_rr else '❌'} · {selected_pair} · {direction}
  </div>
  <div style="font-size:12px; color:#388bfd; margin-top:6px;">
    Check #15 — R:R ≥ {min_rr:.1f}:1 to TP1 · {datetime.now().strftime('%A %d %B %Y  |  %H:%M')}
  </div>
</div>
""", unsafe_allow_html=True)

tab_scan, tab_detail = st.tabs(["📊 All-Pairs Scanner", "🔍 Pair Detail"])

with tab_scan:
    st.markdown("**ATR-Based Position Sizing — All Pairs**")
    st.caption(f"SL = ATR×1.5 · TP1 = SL×{min_rr:.1f} · Risk {risk_pct}% of ${account_bal:,.0f}")

    _prog  = st.progress(0, text="Scanning…")
    _pairs = list(INSTRUMENTS.items())
    _scan  = []

    for _i, (_name, _info) in enumerate(_pairs):
        _prog.progress((_i + 1) / len(_pairs), text=f"Fetching {_name}…")
        _price = fetch_price(_info["ticker"])
        _atr   = fetch_atr14(_info["ticker"])

        if _price is None or _atr is None:
            _scan.append({"pair": _name, "price": None, "atr": None,
                          "sl_pips": None, "tp1_pips": None,
                          "lot": None, "risk_usd": None, "status": "ERROR"})
            continue

        _sl_pips  = _atr * 1.5 / _info["pip_size"]
        _tp1_pips = _sl_pips * min_rr
        _risk_usd = account_bal * risk_pct / 100
        _lot      = _risk_usd / (_sl_pips * _info["pip"])
        _scan.append({"pair": _name, "price": _price, "atr": _atr,
                      "sl_pips": _sl_pips, "tp1_pips": _tp1_pips,
                      "lot": _lot, "risk_usd": _risk_usd, "status": "OK"})

    _prog.empty()

    _std_n  = sum(1 for r in _scan if r["status"] == "OK" and r["lot"] >= 1.0)
    _mini_n = sum(1 for r in _scan if r["status"] == "OK" and 0.10 <= r["lot"] < 1.0)
    _mic_n  = sum(1 for r in _scan if r["status"] == "OK" and r["lot"] < 0.10)
    _err_n  = sum(1 for r in _scan if r["status"] == "ERROR")

    _sc1, _sc2, _sc3, _sc4 = st.columns(4)
    for _col, _cnt, _lbl, _clr in [
        (_sc1, _std_n,  "≥1.0 Lot",    "#3fb950"),
        (_sc2, _mini_n, "0.1–1.0 Lot",  "#388bfd"),
        (_sc3, _mic_n,  "<0.1 Lot",     "#e3b341"),
        (_sc4, _err_n,  "Errors",        "#8b949e"),
    ]:
        with _col:
            st.markdown(
                f'<div class="metric-box">'
                f'<div class="metric-value" style="color:{_clr};font-size:22px;">{_cnt}</div>'
                f'<div class="metric-label">{_lbl}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    _gcols = st.columns(3)
    for _i, _r in enumerate(_scan):
        with _gcols[_i % 3]:
            _sel  = _r["pair"] == selected_pair
            _bclr = "#388bfd" if _sel else "#21262d"
            if _r["status"] == "ERROR":
                st.markdown(
                    f'<div style="background:#161b22;border:1px solid {_bclr};'
                    f'border-radius:10px;padding:14px;margin-bottom:10px;">'
                    f'<div style="font-weight:700;color:#e6edf3;font-size:13px;">{_r["pair"]}</div>'
                    f'<div style="color:#484f58;font-size:11px;margin-top:4px;">⚠️ Data unavailable</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                _lc  = "#3fb950" if _r["lot"] >= 0.10 else "#e3b341"
                _tag = ' <span style="color:#388bfd;font-size:10px;">◀ selected</span>' if _sel else ""
                _pfx = f"{_r['price']:.5f}" if _r["price"] < 100 else f"{_r['price']:.3f}"
                st.markdown(
                    f'<div style="background:#161b22;border:1px solid {_bclr};'
                    f'border-radius:10px;padding:14px;margin-bottom:10px;">'
                    f'<div style="font-weight:700;color:#e6edf3;font-size:13px;">{_r["pair"]}{_tag}</div>'
                    f'<div style="font-size:11px;color:#8b949e;margin-top:2px;">{_pfx}</div>'
                    f'<div style="display:flex;gap:6px;margin-top:8px;flex-wrap:wrap;">'
                    f'<span style="background:#0d1117;border:1px solid #30363d;border-radius:4px;'
                    f'padding:3px 7px;font-size:11px;color:#f85149;'
                    f'font-family:\'JetBrains Mono\',monospace;">SL {_r["sl_pips"]:.0f}p</span>'
                    f'<span style="background:#0d1117;border:1px solid #30363d;border-radius:4px;'
                    f'padding:3px 7px;font-size:11px;color:#3fb950;'
                    f'font-family:\'JetBrains Mono\',monospace;">TP1 {_r["tp1_pips"]:.0f}p</span>'
                    f'<span style="background:#0d1117;border:1px solid #30363d;border-radius:4px;'
                    f'padding:3px 7px;font-size:11px;color:{_lc};'
                    f'font-family:\'JetBrains Mono\',monospace;">{_r["lot"]:.2f} lots</span>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )

    with st.expander("📋 Full Scanner Table"):
        _rows = []
        for _r in _scan:
            if _r["status"] == "ERROR":
                _rows.append({"Pair": _r["pair"], "Price": "—", "ATR14": "—",
                               "SL (pips)": "—", "TP1 (pips)": "—",
                               "Lot Size": "—", "Risk $": "—"})
            else:
                _rows.append({
                    "Pair":       _r["pair"],
                    "Price":      round(_r["price"], 5) if _r["price"] < 100 else round(_r["price"], 2),
                    "ATR14":      round(_r["atr"],   5) if _r["atr"]   < 1   else round(_r["atr"],   3),
                    "SL (pips)":  round(_r["sl_pips"],  1),
                    "TP1 (pips)": round(_r["tp1_pips"], 1),
                    "Lot Size":   round(_r["lot"], 2),
                    "Risk $":     f"${_r['risk_usd']:.2f}",
                })
        st.dataframe(pd.DataFrame(_rows), use_container_width=True, hide_index=True)

with tab_detail:
    # ── Verdict banner ─────────────────────────────────────────────────
    risk_seg  = 1 / (1 + c["rr_tp1"]) * 100 if c["rr_tp1"] else 50
    rew_seg   = 100 - risk_seg
    min_mark  = 1 / (1 + min_rr) * 100

    st.markdown(f"""
    <div class="{c['vclass']}">
      <div style="font-size:23px; font-weight:800; color:{c['color']};
                  letter-spacing:1px; margin-bottom:6px;">{c['label']}</div>
      <div style="font-size:32px; font-weight:900; color:{c['color']};
                  font-family:'JetBrains Mono',monospace; margin:8px 0;">
        {c['rr_tp1']:.2f} : 1
      </div>
      <div style="font-size:13px; color:#8b949e; margin-bottom:14px;">
        TP1 = {c['tp1_pips']:.1f} pips &nbsp;·&nbsp;
        SL  = {c['sl_pips']:.1f} pips &nbsp;·&nbsp;
        Min required = {min_rr:.1f}:1
      </div>

      <!-- R:R bar -->
      <div style="margin:0 auto; max-width:600px;">
        <div style="display:flex; justify-content:space-between;
                    font-size:11px; color:#8b949e; margin-bottom:4px;">
          <span>Risk (SL)</span><span>Reward (TP1)</span>
        </div>
        <div class="rr-track">
          <div style="width:{risk_seg:.1f}%; background:rgba(248,81,73,0.80);
                      display:flex; align-items:center; justify-content:center;
                      font-size:11px; color:#fff; font-weight:700;">
            {risk_seg:.0f}%
          </div>
          <div style="width:{rew_seg:.1f}%; background:rgba(63,185,80,0.80);
                      display:flex; align-items:center; justify-content:center;
                      font-size:11px; color:#fff; font-weight:700;">
            {rew_seg:.0f}%
          </div>
        </div>
        <div style="font-size:10px; color:#484f58; text-align:left; margin-top:2px;">
          ▲ Min {min_rr:.1f}:1 threshold at {min_mark:.0f}% / {100-min_mark:.0f}%
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI strip ──────────────────────────────────────────────────────
    k1,k2,k3,k4,k5,k6 = st.columns(6)
    rr_color = c["color"]
    for col, val, lbl, color in [
        (k1, f"{c['rr_tp1']:.2f}:1",       "R:R to TP1",         rr_color),
        (k2, f"{c['rr_tp2']:.2f}:1",       "R:R to TP2",         "#56d364"),
        (k3, f"{c['sl_pips']:.1f}",        "SL (pips)",          "#f85149"),
        (k4, f"{c['tp1_pips']:.1f}",       "TP1 (pips)",         "#3fb950"),
        (k5, f"{c['tp2_pips']:.1f}",       "TP2 (pips)",         "#56d364"),
        (k6, f"{c['be_winrate_tp1']:.1f}%", "Breakeven Win Rate", "#388bfd"),
    ]:
        with col:
            st.markdown(
                f'<div class="metric-box">'
                f'<div class="metric-value" style="color:{color};font-size:18px;">{val}</div>'
                f'<div class="metric-label">{lbl}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── Trade levels + P&L ────────────────────────────────────────────
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown('<div class="section-title">📐 Trade Levels</div>', unsafe_allow_html=True)
        dir_icon = "🔼" if c["long_trade"] else "🔽"
        dir_col  = "#3fb950" if c["long_trade"] else "#f85149"

        def lvl_row(label, val, extra="", color="#c9d1d9"):
            return (f'<div class="lvl-row">'
                    f'<span class="lvl-label">{label}</span>'
                    f'<span class="lvl-val" style="color:{color};">{val}'
                    f'<span style="color:#8b949e;font-size:11px;font-weight:400;">'
                    f'&nbsp;&nbsp;{extra}</span></span></div>')

        st.markdown(f"""
        <div class="card">
          <div class="card-header">{dir_icon} {direction} — {selected_pair}</div>
          {lvl_row("Entry",  f"{c['entry']:.5f}", "", "#ffffff")}
          {lvl_row("Stop Loss", f"{c['sl']:.5f}",
                   f"−{c['sl_pips']:.1f} pips  (1R)", "#f85149")}
          {lvl_row("TP1",  f"{c['tp1']:.5f}",
                   f"+{c['tp1_pips']:.1f} pips  ({c['rr_tp1']:.2f}R)",
                   "#3fb950" if c['rr_tp1'] >= min_rr else "#e3b341")}
          {lvl_row("TP2",  f"{c['tp2']:.5f}",
                   f"+{c['tp2_pips']:.1f} pips  ({c['rr_tp2']:.2f}R)", "#56d364")}
          {lvl_row("Lot size", f"{c['lot_size']:.2f} lots",
                   f"${c['risk_amt']:.2f} risk ({risk_pct}%)", "#388bfd")}
          {lvl_row(f"Partial close ({partial_pct}%)",
                   f"{c['partial_lots']:.2f} lots at TP1", "", "#a29bfe")}
          {lvl_row(f"Runner ({100-partial_pct}%)",
                   f"{c['runner_lots']:.2f} lots to TP2", "", "#a29bfe")}
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="section-title">💰 P&L Projection</div>', unsafe_allow_html=True)

        tp1_usd  = c["partial_lots"] * c["tp1_pips"] * pip_val
        tp2_usd  = c["runner_lots"]  * c["tp2_pips"] * pip_val
        sl_usd   = c["lot_size"]     * c["sl_pips"]  * pip_val
        total    = tp1_usd + tp2_usd

        pnl_color = "#3fb950" if c["rr_tp1"] >= min_rr else "#f85149"

        st.markdown(f"""
        <div class="card">
          <div class="card-header">💵 Projected P&L</div>
          <div style="display:flex; justify-content:space-between; align-items:center;
                      padding:12px 0; border-bottom:1px solid #21262d;">
            <span style="color:#8b949e; font-size:13px;">
              TP1 profit &nbsp;({partial_pct}% of position)
            </span>
            <span style="font-family:'JetBrains Mono',monospace; font-weight:700;
                         color:#3fb950; font-size:15px;">+${tp1_usd:.2f}</span>
          </div>
          <div style="display:flex; justify-content:space-between; align-items:center;
                      padding:12px 0; border-bottom:1px solid #21262d;">
            <span style="color:#8b949e; font-size:13px;">
              TP2 profit &nbsp;({100-partial_pct}% of position)
            </span>
            <span style="font-family:'JetBrains Mono',monospace; font-weight:700;
                         color:#56d364; font-size:15px;">+${tp2_usd:.2f}</span>
          </div>
          <div style="display:flex; justify-content:space-between; align-items:center;
                      padding:12px 0; border-bottom:1px solid #21262d;">
            <span style="color:#8b949e; font-size:13px;">Total profit (both TPs hit)</span>
            <span style="font-family:'JetBrains Mono',monospace; font-weight:800;
                         color:#3fb950; font-size:17px;">+${total:.2f}</span>
          </div>
          <div style="display:flex; justify-content:space-between; align-items:center;
                      padding:12px 0; border-bottom:1px solid #21262d;">
            <span style="color:#8b949e; font-size:13px;">Max loss (SL hit)</span>
            <span style="font-family:'JetBrains Mono',monospace; font-weight:700;
                         color:#f85149; font-size:15px;">−${sl_usd:.2f}</span>
          </div>
          <div style="display:flex; justify-content:space-between; align-items:center;
                      padding:14px 0 4px 0;">
            <span style="color:#c9d1d9; font-size:13px; font-weight:600;">
              Net profit-to-risk ratio
            </span>
            <span style="font-family:'JetBrains Mono',monospace; font-weight:800;
                         color:{pnl_color}; font-size:18px;">
              {total/sl_usd:.2f}:1
            </span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Charts ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📊 Visuals</div>', unsafe_allow_html=True)

    chart_l, chart_r = st.columns([3, 2])
    with chart_l:
        st.plotly_chart(build_rr_chart(c, selected_pair), use_container_width=True)
    with chart_r:
        st.plotly_chart(build_winrate_chart(c), use_container_width=True)

    st.markdown("---")

    # ── Win-rate scenario table ────────────────────────────────────────
    st.markdown('<div class="section-title">📋 Win-Rate Scenario Analysis</div>',
                unsafe_allow_html=True)
    st.caption("Showing net P&L over 100 trades at varying win rates — "
               f"using {c['lot_size']:.2f} lots, SL=${sl_usd:.2f}, TP1=${tp1_usd + tp2_usd:.2f}")

    scenario_rows = []
    for wr in [30, 35, 40, 45, 50, 55, 60, 65, 70]:
        wins   = wr
        losses = 100 - wr
        gross  = wins * (tp1_usd + tp2_usd) - losses * sl_usd
        edge   = gross / 100
        status = "✅" if gross > 0 else "❌"
        scenario_rows.append({
            "Win Rate": f"{wr}%",
            "Wins / 100": wins,
            "Losses / 100": losses,
            "Gross P&L ($)": round(gross, 2),
            "Edge per trade ($)": round(edge, 2),
            "Profitable": status,
        })

    sc_df = pd.DataFrame(scenario_rows)
    st.dataframe(sc_df, use_container_width=True, hide_index=True,
                 column_config={
                     "Gross P&L ($)":      st.column_config.NumberColumn(format="$%.2f"),
                     "Edge per trade ($)": st.column_config.NumberColumn(format="$%.2f"),
                 })

    st.markdown("---")

    # ── Formula + explainer ────────────────────────────────────────────
    st.markdown('<div class="section-title">📖 R:R Logic & Rules</div>',
                unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"""
        <div class="explainer">
        <b style="color:#388bfd;">⚖️ Why R:R ≥ {min_rr:.1f}:1 matters</b><br><br>
        At a {min_rr:.1f}:1 R:R, you only need to be right <b style="color:#c9d1d9;">
        {1/(1+min_rr)*100:.0f}%</b> of the time to break even.<br><br>
        At a 2:1 R:R with a 40% win rate, 100 trades produce:<br>
        40 wins × $200 − 60 losses × $100 = <b style="color:#3fb950;">+$2,000 net profit</b>.<br><br>
        This is why a losing strategy can still be profitable — the R:R multiplies
        every winning trade relative to every loss. A minimum of 2:1 gives you
        enough margin to absorb losses while staying net positive.
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown(f"""
        <div class="explainer">
        <b style="color:#388bfd;">📐 How to use this checker</b><br><br>
        <b>1.</b> Fetch the live price (⚡ button in sidebar) or enter your entry manually.<br><br>
        <b>2.</b> Enter your SL in pips — pulled from Check #14 (ATR-based stop structure).<br><br>
        <b>3.</b> TP1 = at least {min_rr:.1f}× the SL distance.
        TP2 = 3× the SL distance for the runner.<br><br>
        <b>4.</b> If R:R is below {min_rr:.1f}:1 → <b style="color:#f85149;">do not take the trade.</b>
        Widen the TP target or wait for a closer entry.<br><br>
        This is <em>Check #15</em>. Must pass before confirming position size in Check #18.
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="formula-box">
    R:R to TP1 &nbsp;= &nbsp;TP1 distance ÷ SL distance
    &nbsp;= &nbsp;{c['tp1_pips']:.1f} pips ÷ {c['sl_pips']:.1f} pips
    &nbsp;= &nbsp;<b style="color:{c['color']};">{c['rr_tp1']:.2f} : 1</b>
    &nbsp; {'✅' if c['rr_tp1'] >= min_rr else '❌'} (min {min_rr:.1f}:1)<br>
    R:R to TP2 &nbsp;= &nbsp;{c['tp2_pips']:.1f} pips ÷ {c['sl_pips']:.1f} pips
    &nbsp;= &nbsp;<b style="color:#56d364;">{c['rr_tp2']:.2f} : 1</b><br>
    Breakeven &nbsp;&nbsp;= &nbsp;1 ÷ (1 + {c['rr_tp1']:.2f})
    &nbsp;= &nbsp;<b style="color:#388bfd;">{c['be_winrate_tp1']:.1f}% win rate needed</b>
    </div>
    """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div style="text-align:center;color:#484f58;font-size:11px;margin-top:32px;
                padding-top:16px;border-top:1px solid #21262d;">
      ⚖️ R:R Calculator · Check #15 · Risk to Reward · For educational purposes only
    </div>
    """, unsafe_allow_html=True)