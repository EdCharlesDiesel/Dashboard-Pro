"""Risk Suite — Stop Structure + R:R Calculator + Account Risk, merged.

Was three separate pages (stop-structure.py, rr-calculator.py,
account-risk.py) that each re-asked for instrument/account-balance/risk%
independently. They're one decision in practice — "where's the stop, what's
the size" — so this page shares that input once and tabs the three
calculators underneath it. Each tab's math is preserved byte-for-byte from
its original file (see archive/pages/ for the pre-merge versions); only the
shared inputs and page chrome (CSS/page-config/nav, previously duplicated
three times) were deduplicated. The two hardcoded local INSTRUMENTS dicts
(stop-structure.py, rr-calculator.py) are dropped in favor of the shared
registry, which account-risk.py already used and which carries the identical
22 pairs/tickers/pip values (verified before merging).
"""
import streamlit as st
from src.ui.theme import BloombergTheme
from src.pages_lib.navigation import render_sidebar_nav
from src.instruments.registry import INSTRUMENTS, TREND_COMMODITIES
from src.services.alert_service import NotifyCache
from src.services.tool_log import log_tool_usage
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.core.config import CANDLE_STYLE
from datetime import datetime

# ── Page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Risk Suite",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)
BloombergTheme.apply()

# ── CSS (union of the three original blocks — stop-structure.py's superset) ─
st.markdown("""
<style>
    .stApp {
      --border: color-mix(in srgb, var(--text-color) 12%, transparent);
      --muted:  color-mix(in srgb, var(--text-color) 55%, transparent);
    }
    html,body,[class*="css"]{ font-family:'JetBrains Mono','Fira Code',monospace; }
    .stApp{ background:var(--background-color); }
    section[data-testid="stSidebar"]{ background:var(--secondary-background-color)!important; border-right:1px solid var(--border,#2a2a2a); }
    #MainMenu,footer{ visibility:hidden; }
    [data-testid="stSidebarCollapsedControl"]{visibility:visible !important;}
    [data-testid="stSidebarNav"]{ display:none; }
    .block-container{ padding-top:1.5rem; max-width:1380px; }

    .card{ background:var(--secondary-background-color); border:1px solid var(--border,#2a2a2a); border-radius:12px;
           padding:20px; margin-bottom:16px; }
    .card-header{ font-size:13px; font-weight:600; letter-spacing:.08em;
                  text-transform:uppercase; color:#9a9a9a; margin-bottom:14px; }
    .metric-box{ background:var(--background-color); border:1px solid var(--border,#2a2a2a); border-radius:8px;
                 padding:14px; text-align:center; }
    .metric-value{ font-size:22px; font-weight:700; color:var(--text-color); }
    .metric-label{ font-size:11px; color:var(--muted,#9a9a9a); margin-top:2px; font-weight:500;
                   letter-spacing:.04em; text-transform:uppercase; }
    .section-title{ font-size:16px; font-weight:700; color:var(--text-color);
                    margin:24px 0 12px 0; padding-left:4px; border-left:3px solid #00ff41; }
    .prog-track{ background:var(--border,#2a2a2a); border-radius:8px; height:10px;
                 margin:6px 0 2px 0; overflow:hidden; }

    /* SL level cards (Stop Structure tab) */
    .sl-card{ border-radius:12px; padding:20px 24px; margin-bottom:12px; }
    .sl-card-long { background:linear-gradient(135deg,#0a2918,#0d3a1f);
                    border:1px solid #00a32a; border-left:4px solid #00ff66; }
    .sl-card-short{ background:linear-gradient(135deg,#2a0a0a,#3a0d0d);
                    border:1px solid #8b2d2d; border-left:4px solid #ff3344; }

    .str-row{ display:flex; justify-content:space-between; align-items:center;
              padding:9px 14px; border-bottom:1px solid #2a2a2a;
              font-size:13px; }
    .str-row:last-child{ border-bottom:none; }
    .str-label{ color:#9a9a9a; }
    .str-val  { font-family:monospace; font-weight:600; color:#e6e6e6; }

    /* Verdict banners (R:R Calculator tab) */
    .verdict-long { background:linear-gradient(135deg,#0d3a1f,#0d5e32);
                    border:2px solid #00a32a; border-radius:14px;
                    padding:22px 28px; text-align:center; margin-bottom:18px; }
    .verdict-short{ background:linear-gradient(135deg,#3a0d0d,#5e1414);
                    border:2px solid #8b2d2d; border-radius:14px;
                    padding:22px 28px; text-align:center; margin-bottom:18px; }
    .verdict-none { background:linear-gradient(135deg,#0a0a0a,#1c2128);
                    border:1px solid #2a2a2a; border-radius:14px;
                    padding:22px 28px; text-align:center; margin-bottom:18px; }
    .verdict-pass{ background:linear-gradient(135deg,#0d3a1f,#0d5e32);
                   border:2px solid #00a32a; border-radius:14px;
                   padding:22px 28px; text-align:center; margin-bottom:18px; }
    .verdict-warn{ background:linear-gradient(135deg,#1c1c0d,#2e2a0d);
                   border:2px solid #9e6a03; border-radius:14px;
                   padding:22px 28px; text-align:center; margin-bottom:18px; }
    .verdict-fail{ background:linear-gradient(135deg,#3a0d0d,#5e1414);
                   border:2px solid #8b2d2d; border-radius:14px;
                   padding:22px 28px; text-align:center; margin-bottom:18px; }

    .rr-track{ background:#2a2a2a; border-radius:6px; height:28px;
               margin:10px 0; overflow:hidden; display:flex; }

    .lvl-row{ display:flex; justify-content:space-between; align-items:center;
              padding:10px 16px; border-bottom:1px solid #2a2a2a; font-size:13px; }
    .lvl-row:last-child{ border-bottom:none; }
    .lvl-label{ color:#9a9a9a; }
    .lvl-val  { font-family:'JetBrains Mono',monospace; font-weight:600; color:#e6e6e6; }

    .sc-row{ display:flex; gap:0; border-bottom:1px solid #2a2a2a; font-size:12px; }
    .sc-row:last-child{ border-bottom:none; }
    .sc-cell{ flex:1; padding:8px 10px; font-family:'JetBrains Mono',monospace; }

    .explainer{ background:var(--background-color); border:1px solid #1e3a5f;
                border-left:3px solid #00ff41; border-radius:8px;
                padding:14px 18px; font-size:13px; color:var(--muted,#9a9a9a); line-height:1.7; }
    .formula-box{ background:#000000; border:1px solid #2a2a2a; border-radius:8px;
                  padding:14px 18px; font-family:monospace; font-size:13px;
                  color:#e6e6e6; margin:10px 0; line-height:2; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# SHARED DATA FETCH (identical across the original three files)
# ══════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner=False)
def fetch_ohlc(ticker: str) -> pd.DataFrame:
    from src.db.market_cache import cached_ohlc
    try:
        df = cached_ohlc(ticker, interval="1d", period="1y", ttl=300)
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def fetch_price(ticker: str) -> float | None:
    from src.db.market_cache import cached_ohlc
    try:
        df = cached_ohlc(ticker, interval="1d", period="5d", ttl=300)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return float(df["Close"].iloc[-1])
    except Exception:
        return None


def calc_atr(df: pd.DataFrame, period: int) -> pd.Series:
    prev = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev).abs(),
        (df["Low"] - prev).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


@st.cache_data(ttl=300, show_spinner=False)
def fetch_atr14(ticker: str) -> float | None:
    from src.db.market_cache import cached_ohlc
    try:
        df = cached_ohlc(ticker, interval="1d", period="60d", ttl=300)
        if df.empty or len(df) < 15:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        atr = calc_atr(df, 14)
        return float(atr.iloc[-1])
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def fetch_atr14_pips(ticker: str, pip_size: float) -> float | None:
    atr = fetch_atr14(ticker)
    return (atr / pip_size) if atr is not None else None


def fmt_price(p: float) -> str:
    return f"{p:.5f}" if abs(p) < 100 else f"{p:.3f}"


# ══════════════════════════════════════════════════════════════════
# STOP STRUCTURE — pure calc/chart functions (unchanged from stop-structure.py)
# ══════════════════════════════════════════════════════════════════

def find_structure_levels(df: pd.DataFrame, pivot_lb: int = 5) -> dict:
    """Identify the nearest swing high and swing low using a pivot lookback.
    These act as the structural levels to place the SL beyond."""
    H = df["High"].values
    L = df["Low"].values
    n = len(df)

    swing_high_idx, swing_low_idx = [], []
    lb = min(pivot_lb, 3)

    for i in range(lb, n - lb):
        if all(H[i] >= H[i-j] for j in range(1, lb+1)) and \
                all(H[i] >= H[i+j] for j in range(1, lb+1)):
            swing_high_idx.append(i)
        if all(L[i] <= L[i-j] for j in range(1, lb+1)) and \
                all(L[i] <= L[i+j] for j in range(1, lb+1)):
            swing_low_idx.append(i)

    recent_sh = df.iloc[swing_high_idx[-1]] if swing_high_idx else None
    recent_sl = df.iloc[swing_low_idx[-1]] if swing_low_idx else None

    sh_df = df.iloc[swing_high_idx] if swing_high_idx else pd.DataFrame()
    sl_df = df.iloc[swing_low_idx] if swing_low_idx else pd.DataFrame()

    return {
        "recent_sh": recent_sh, "recent_sl": recent_sl,
        "sh_df": sh_df, "sl_df": sl_df,
        "sh_idx": swing_high_idx, "sl_idx": swing_low_idx,
    }


def compute_stops(df: pd.DataFrame, atr_mult: float,
                  pip_size: float, pip_val: float,
                  account_bal: float, risk_pct: float,
                  struct: dict, atr_period: int = 14) -> dict:
    atr14 = calc_atr(df, atr_period)
    atr_now = float(atr14.iloc[-1])
    price = float(df["Close"].iloc[-1])
    sl_dist = atr_now * atr_mult
    sl_pips = round(sl_dist / pip_size, 1)
    tp1_pips = round(sl_pips * 2.0, 1)
    tp2_pips = round(sl_pips * 3.0, 1)
    tp1_dist = tp1_pips * pip_size
    tp2_dist = tp2_pips * pip_size

    atr_buf = atr_now * 0.20

    sh_price = float(struct["recent_sh"]["High"]) if struct["recent_sh"] is not None else None
    sl_price = float(struct["recent_sl"]["Low"]) if struct["recent_sl"] is not None else None

    sl_long_struct = (sl_price - atr_buf) if sl_price else (price - sl_dist)
    sl_short_struct = (sh_price + atr_buf) if sh_price else (price + sl_dist)

    sl_long_atr = price - sl_dist
    sl_short_atr = price + sl_dist

    use_struct_long = sl_price and abs(price - sl_long_struct) <= atr_now * 3
    use_struct_short = sh_price and abs(sl_short_struct - price) <= atr_now * 3

    sl_long_final = sl_long_struct if use_struct_long else sl_long_atr
    sl_short_final = sl_short_struct if use_struct_short else sl_short_atr

    sl_long_pips = round(abs(price - sl_long_final) / pip_size, 1)
    sl_short_pips = round(abs(sl_short_final - price) / pip_size, 1)

    tp1_long = price + tp1_dist
    tp2_long = price + tp2_dist
    tp1_short = price - tp1_dist
    tp2_short = price - tp2_dist

    risk_amount = account_bal * (risk_pct / 100)
    lot_long = round(risk_amount / (sl_long_pips * pip_val), 2) if sl_long_pips else 0
    lot_short = round(risk_amount / (sl_short_pips * pip_val), 2) if sl_short_pips else 0

    rr_tp1_long = round(tp1_pips / sl_long_pips, 2) if sl_long_pips else 0
    rr_tp2_long = round(tp2_pips / sl_long_pips, 2) if sl_long_pips else 0
    rr_tp1_short = round(tp1_pips / sl_short_pips, 2) if sl_short_pips else 0
    rr_tp2_short = round(tp2_pips / sl_short_pips, 2) if sl_short_pips else 0

    return {
        "price": price, "atr14": round(atr_now, 5),
        "sl_pips": sl_pips, "atr_buf": atr_buf,
        "sl_long": round(sl_long_final, 5), "sl_long_pips": sl_long_pips,
        "tp1_long": round(tp1_long, 5), "tp2_long": round(tp2_long, 5),
        "lot_long": lot_long, "rr_tp1_long": rr_tp1_long, "rr_tp2_long": rr_tp2_long,
        "use_struct_long": use_struct_long,
        "sl_short": round(sl_short_final, 5), "sl_short_pips": sl_short_pips,
        "tp1_short": round(tp1_short, 5), "tp2_short": round(tp2_short, 5),
        "lot_short": lot_short, "rr_tp1_short": rr_tp1_short, "rr_tp2_short": rr_tp2_short,
        "use_struct_short": use_struct_short,
        "risk_amount": round(risk_amount, 2),
        "tp1_pips": tp1_pips, "tp2_pips": tp2_pips,
        "struct_sh": sh_price, "struct_sl": sl_price,
        "atr_series": atr14,
    }


def build_structure_chart(df: pd.DataFrame, pair: str,
                          calc: dict, struct: dict,
                          direction: str, show_n: int = 80) -> go.Figure:
    plot = df.tail(show_n).copy()
    price = calc["price"]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.78, 0.22], vertical_spacing=0.03,
    )

    fig.add_trace(go.Candlestick(
        x=plot.index,
        open=plot["Open"], high=plot["High"],
        low=plot["Low"], close=plot["Close"],
        **CANDLE_STYLE,
        name="Daily", showlegend=False,
    ), row=1, col=1)

    vcol = ["rgba(63,185,80,0.33)" if c >= o else "rgba(248,81,73,0.33)"
            for c, o in zip(plot["Close"], plot["Open"])]
    fig.add_trace(go.Bar(
        x=plot.index, y=plot["Volume"],
        marker_color=vcol, name="Volume", showlegend=False,
    ), row=2, col=1)

    atr_plot = calc["atr_series"].reindex(plot.index).dropna()
    fig.add_trace(go.Scatter(
        x=atr_plot.index, y=atr_plot.values / atr_plot.values.max() * plot["Volume"].max(),
        mode="lines", line=dict(color="rgba(56,139,253,0.33)", width=1.5),
        name="ATR (scaled)", showlegend=False,
    ), row=2, col=1)

    sh_plot = struct["sh_df"].reindex(
        [i for i in struct["sh_df"].index if i in plot.index])
    sl_plot = struct["sl_df"].reindex(
        [i for i in struct["sl_df"].index if i in plot.index])

    if not sh_plot.empty:
        fig.add_trace(go.Scatter(
            x=sh_plot.index, y=sh_plot["High"] * 1.0005,
            mode="markers+text",
            marker=dict(symbol="triangle-down", size=10, color="#ff3344",
                        line=dict(color="#000000", width=1)),
            text=["SH"] * len(sh_plot), textposition="top center",
            textfont=dict(size=8, color="#ff3344"), name="Swing High",
        ), row=1, col=1)

    if not sl_plot.empty:
        fig.add_trace(go.Scatter(
            x=sl_plot.index, y=sl_plot["Low"] * 0.9995,
            mode="markers+text",
            marker=dict(symbol="triangle-up", size=10, color="#00ff66",
                        line=dict(color="#000000", width=1)),
            text=["SL"] * len(sl_plot), textposition="bottom center",
            textfont=dict(size=8, color="#00ff66"), name="Swing Low",
        ), row=1, col=1)

    fig.add_hline(
        y=price, line_color="#ffffff", line_dash="solid",
        line_width=1.5, opacity=0.85,
        annotation_text=f"  Price {price:.5f}",
        annotation_position="right",
        annotation_font=dict(size=10, color="#ffffff"),
        row=1, col=1,
    )

    def hline(y, color, dash, width, label, pos="right"):
        fig.add_hline(
            y=y, line_color=color, line_dash=dash,
            line_width=width, opacity=0.9,
            annotation_text=f"  {label}",
            annotation_position=pos,
            annotation_font=dict(size=9, color=color),
            row=1, col=1,
        )

    if direction in ("LONG", "BOTH"):
        fig.add_hrect(
            y0=calc["sl_long"] - calc["atr_buf"],
            y1=calc["sl_long"] + calc["atr_buf"],
            fillcolor="#ff3344", opacity=0.08, line_width=0, row=1, col=1,
        )
        hline(calc["sl_long"], "#ff3344", "dash", 2.0,
              f"SL LONG  {calc['sl_long']:.5f}  ({calc['sl_long_pips']:.0f} pips)")
        hline(calc["tp1_long"], "#00ff66", "dot", 1.5,
              f"TP1  {calc['tp1_long']:.5f}  ({calc['tp1_pips']:.0f} pips)")
        hline(calc["tp2_long"], "#56d364", "dot", 1.0,
              f"TP2  {calc['tp2_long']:.5f}  ({calc['tp2_pips']:.0f} pips)")
        if calc["struct_sl"]:
            hline(calc["struct_sl"], "#ffcc00", "longdash", 1.0,
                  f"Swing Low  {calc['struct_sl']:.5f}", pos="left")

    if direction in ("SHORT", "BOTH"):
        fig.add_hrect(
            y0=calc["sl_short"] - calc["atr_buf"],
            y1=calc["sl_short"] + calc["atr_buf"],
            fillcolor="#ff3344", opacity=0.08, line_width=0, row=1, col=1,
        )
        hline(calc["sl_short"], "#ff3344", "dash", 2.0,
              f"SL SHORT  {calc['sl_short']:.5f}  ({calc['sl_short_pips']:.0f} pips)")
        hline(calc["tp1_short"], "#00ff66", "dot", 1.5,
              f"TP1  {calc['tp1_short']:.5f}  ({calc['tp1_pips']:.0f} pips)")
        hline(calc["tp2_short"], "#56d364", "dot", 1.0,
              f"TP2  {calc['tp2_short']:.5f}  ({calc['tp2_pips']:.0f} pips)")
        if calc["struct_sh"]:
            hline(calc["struct_sh"], "#ffcc00", "longdash", 1.0,
                  f"Swing High  {calc['struct_sh']:.5f}", pos="left")

    fig.update_layout(
        height=620,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0a0a0a",
        font=dict(family="JetBrains Mono, monospace", size=11, color="#9a9a9a"),
        xaxis_rangeslider_visible=False,
        legend=dict(
            bgcolor="rgba(13,17,23,0.60)", bordercolor="#2a2a2a", borderwidth=1,
            font=dict(size=10), orientation="h",
            yanchor="bottom", y=1.01, xanchor="left", x=0,
        ),
        margin=dict(l=10, r=170, t=10, b=10),
    )
    fig.update_yaxes(gridcolor="#2a2a2a", zeroline=False)
    fig.update_xaxes(gridcolor="#2a2a2a", showgrid=False)
    return fig


def level_row(label, value, extra=""):
    return (f'<div class="str-row"><span class="str-label">{label}</span>'
            f'<span class="str-val">{value}&nbsp;&nbsp;<span style="color:#9a9a9a;'
            f'font-size:11px;">{extra}</span></span></div>')


# ══════════════════════════════════════════════════════════════════
# R:R CALCULATOR — pure calc/chart functions (unchanged from rr-calculator.py)
# ══════════════════════════════════════════════════════════════════

def compute_rr(entry: float, sl: float, tp1: float, tp2: float,
               pip_size: float, pip_val: float,
               account_bal: float, risk_pct: float,
               partial_pct: float) -> dict:
    sl_dist = abs(entry - sl)
    tp1_dist = abs(tp1 - entry)
    tp2_dist = abs(tp2 - entry)

    sl_pips = sl_dist / pip_size
    tp1_pips = tp1_dist / pip_size
    tp2_pips = tp2_dist / pip_size

    rr_tp1 = tp1_dist / sl_dist if sl_dist else 0
    rr_tp2 = tp2_dist / sl_dist if sl_dist else 0

    risk_amt = account_bal * (risk_pct / 100)
    lot_size = risk_amt / (sl_pips * pip_val) if sl_pips else 0

    partial_lots = lot_size * (partial_pct / 100)
    runner_lots = lot_size - partial_lots
    tp1_profit = partial_lots * tp1_pips * pip_val
    tp2_profit = runner_lots * tp2_pips * pip_val
    sl_loss = lot_size * sl_pips * pip_val

    be_winrate_tp1 = 1 / (1 + rr_tp1) * 100 if rr_tp1 else 0

    if rr_tp1 >= 3.0:
        grade, label, color, vclass = "excellent", "✅ EXCELLENT  R:R", "#00ff66", "verdict-pass"
    elif rr_tp1 >= 2.0:
        grade, label, color, vclass = "pass", "✅ R:R MINIMUM MET", "#00ff66", "verdict-pass"
    elif rr_tp1 >= 1.5:
        grade, label, color, vclass = "marginal", "⚠️ MARGINAL — REVIEW", "#ffcc00", "verdict-warn"
    else:
        grade, label, color, vclass = "fail", "❌ R:R BELOW MINIMUM", "#ff3344", "verdict-fail"

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


def build_rr_chart(c: dict, pair: str) -> go.Figure:
    levels = {"SL": c["sl"], "Entry": c["entry"], "TP1": c["tp1"], "TP2": c["tp2"]}
    sorted_lvls = sorted(levels.items(), key=lambda x: x[1])
    y_min = min(v for _, v in sorted_lvls) * (0.9998 if c["long_trade"] else 0.9995)
    y_max = max(v for _, v in sorted_lvls) * (1.0002 if c["long_trade"] else 1.0005)

    fig = go.Figure()

    fig.add_shape(type="rect", x0=0, x1=1,
                  y0=min(c["sl"], c["entry"]), y1=max(c["sl"], c["entry"]),
                  fillcolor="rgba(248,81,73,0.13)", line_color="#ff3344", line_width=1.5)
    fig.add_shape(type="rect", x0=0, x1=1,
                  y0=min(c["tp1"], c["entry"]), y1=max(c["tp1"], c["entry"]),
                  fillcolor="rgba(63,185,80,0.13)", line_color="#00ff66", line_width=1.5)
    fig.add_shape(type="rect", x0=0, x1=1,
                  y0=min(c["tp2"], c["tp1"]), y1=max(c["tp2"], c["tp1"]),
                  fillcolor="rgba(63,185,80,0.07)", line_color="#56d364", line_width=1,
                  line_dash="dot")

    colors = {"SL": "#ff3344", "Entry": "#ffffff", "TP1": "#00ff66", "TP2": "#56d364"}
    for name, val in levels.items():
        fig.add_hline(
            y=val, line_color=colors[name], line_width=2,
            line_dash="solid" if name in ("Entry", "SL") else "dot",
            annotation_text=f"  {name}  {val:.5f}",
            annotation_position="right",
            annotation_font=dict(size=11, color=colors[name]),
        )

    mid_sl = (c["entry"] + c["sl"]) / 2
    mid_tp1 = (c["entry"] + c["tp1"]) / 2
    mid_tp2 = (c["tp1"] + c["tp2"]) / 2

    for y, text, color in [
        (mid_sl, f"SL: {c['sl_pips']:.1f} pips  (1R)", "#ff3344"),
        (mid_tp1, f"TP1: {c['tp1_pips']:.1f} pips  ({c['rr_tp1']:.1f}R)", "#00ff66"),
        (mid_tp2, f"TP2: {c['tp2_pips']:.1f} pips  ({c['rr_tp2']:.1f}R)", "#56d364"),
    ]:
        fig.add_annotation(
            x=0.5, y=y, text=text,
            showarrow=False, font=dict(size=11, color=color),
            bgcolor="rgba(13,17,23,0.80)", borderpad=4,
        )

    fig.update_layout(
        height=460,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0a0a0a",
        font=dict(family="JetBrains Mono, monospace", size=11, color="#9a9a9a"),
        margin=dict(l=10, r=160, t=30, b=20),
        title=dict(text=f"<b>{pair}</b> — R:R Visual Diagram",
                   font=dict(color="#e6e6e6", size=14), x=0.01),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, range=[0, 1]),
        yaxis=dict(showgrid=True, gridcolor="#2a2a2a", range=[y_min, y_max], zeroline=False),
        showlegend=False,
    )
    return fig


def build_winrate_chart(c: dict) -> go.Figure:
    rr_range = np.arange(0.5, 5.1, 0.1)
    be_wr = [1/(1+r)*100 for r in rr_range]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=rr_range, y=be_wr,
        mode="lines", line=dict(color="#00ff41", width=2.5),
        name="Breakeven win rate",
        fill="tozeroy", fillcolor="rgba(56,139,253,0.07)",
    ))

    fig.add_vline(x=2.0, line_color="rgba(227,179,65,0.40)", line_dash="dash", line_width=1.5,
                  annotation_text="Min 2:1", annotation_position="top",
                  annotation_font=dict(size=9, color="#ffcc00"))

    current_be = 1/(1+c["rr_tp1"])*100 if c["rr_tp1"] else 50
    fig.add_trace(go.Scatter(
        x=[c["rr_tp1"]], y=[current_be],
        mode="markers+text",
        marker=dict(size=14, color=c["color"], line=dict(color="#000000", width=2)),
        text=[f" {c['rr_tp1']:.2f}:1  ({current_be:.1f}% WR needed)"],
        textposition="middle right",
        textfont=dict(size=11, color=c["color"]),
        name="This trade",
    ))

    fig.update_layout(
        height=280,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0a0a0a",
        font=dict(family="JetBrains Mono, monospace", size=11, color="#9a9a9a"),
        margin=dict(l=10, r=20, t=30, b=20),
        title=dict(text="Breakeven Win Rate required at each R:R",
                   font=dict(color="#e6e6e6", size=13), x=0.01),
        xaxis=dict(title="R:R Ratio", gridcolor="#2a2a2a", showgrid=True, zeroline=False),
        yaxis=dict(title="Win rate needed (%)", gridcolor="#2a2a2a",
                   showgrid=True, zeroline=False, range=[0, 80]),
        showlegend=False,
    )
    return fig


def lvl_row(label, val, extra="", color="#e6e6e6"):
    return (f'<div class="lvl-row"><span class="lvl-label">{label}</span>'
            f'<span class="lvl-val" style="color:{color};">{val}'
            f'<span style="color:#9a9a9a;font-size:11px;font-weight:400;">'
            f'&nbsp;&nbsp;{extra}</span></span></div>')


# ══════════════════════════════════════════════════════════════════
# ACCOUNT RISK — pure calc function (unchanged from account-risk.py)
# ══════════════════════════════════════════════════════════════════

def size_position(balance: float, risk_pct: float, sl_pips: float,
                  pip_val: float, n_trades: int, split_mode: str) -> dict:
    """Translate a risk budget into money-at-risk and a lot size.

    n_trades is the desk's trades-per-signal (2 here). split_mode decides
    whether the risk % is the budget for the whole signal (split across the
    trades) or applied per trade (so total exposure scales by n_trades).
    """
    risk_signal = balance * (risk_pct / 100)

    if split_mode == "Split across trades":
        risk_per_trade = risk_signal / n_trades if n_trades else risk_signal
        risk_total = risk_signal
    else:  # "Per trade"
        risk_per_trade = risk_signal
        risk_total = risk_signal * n_trades

    lots_per_trade = (risk_per_trade / (sl_pips * pip_val)) if sl_pips and pip_val else 0.0
    lots_total = lots_per_trade * n_trades

    return {
        "risk_signal": risk_signal,
        "risk_per_trade": risk_per_trade,
        "risk_total": risk_total,
        "lots_per_trade": lots_per_trade,
        "lots_total": lots_total,
        "units_per_trade": lots_per_trade * 100_000,
        "micro_per_trade": lots_per_trade * 100,
        "risk_pct_total": (risk_total / balance * 100) if balance else 0.0,
    }


# ══════════════════════════════════════════════════════════════════
# SIDEBAR — one shared instrument/account/risk block (the real merge win:
# previously re-entered on 3 separate pages), plus per-tab settings.
# ══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 🛡️ Risk Suite")

    inst_keys = list(INSTRUMENTS.keys())
    default_inst = st.session_state.get("selected_instrument", "EUR/USD")
    if default_inst not in inst_keys:
        default_inst = inst_keys[0]
    selected_pair = st.selectbox("Instrument", inst_keys,
                                 index=inst_keys.index(default_inst))
    st.session_state["selected_instrument"] = selected_pair

    inst = INSTRUMENTS[selected_pair]
    ticker = inst["ticker"]
    pip_size = inst["pip_size"]
    pip_val = inst["pip"]

    st.divider()
    st.markdown("**💰 Account**")
    account_bal = st.number_input(
        "Account Balance ($)",
        value=float(st.session_state.get("account_bal", 10000.0)),
        min_value=0.0, step=500.0, format="%.2f")
    st.session_state["account_bal"] = account_bal
    risk_pct = st.slider("Risk per Trade (%)", 0.25, 5.0,
                         float(st.session_state.get("risk_pct", 1.0)), 0.25)
    st.session_state["risk_pct"] = risk_pct

    st.divider()
    if st.button("🔄 Refresh All Data", width="stretch", type="primary"):
        st.cache_data.clear()
        st.rerun()
    st.caption(f"🕐 {datetime.now().strftime('%H:%M:%S')} local")

    st.divider()
    with st.expander("🛡️ Stop Structure settings", expanded=True):
        ss_direction = st.radio("Trade direction", ["LONG", "SHORT", "BOTH"],
                                horizontal=True, key="ss_direction")
        ss_atr_period = st.number_input("ATR Period", value=14, min_value=5, max_value=50,
                                        key="ss_atr_period")
        ss_atr_mult = st.number_input("ATR Multiplier (SL)", value=1.5,
                                      min_value=0.5, max_value=5.0, step=0.1, format="%.1f",
                                      key="ss_atr_mult")
        ss_pivot_lb = st.slider("Pivot lookback (bars)", 2, 10, 5,
                                help="Bars each side needed to confirm a swing pivot",
                                key="ss_pivot_lb")
        ss_show_n = st.slider("Candles on chart", 40, 200, 80, step=10, key="ss_show_n")

    with st.expander("⚖️ R:R Calculator settings", expanded=False):
        if st.button("⚡ Fetch Live Price", width="stretch", key="rr_fetch"):
            st.cache_data.clear()
            st.rerun()
        with st.spinner("Fetching price…"):
            rr_live_price = fetch_price(ticker)
            rr_live_atr = fetch_atr14(ticker)
        rr_default_entry = rr_live_price if rr_live_price else 1.10000
        rr_default_sl_p = round(rr_live_atr * 1.5 / pip_size, 1) if rr_live_atr else 101.4
        rr_default_tp1_p = round(rr_default_sl_p * 2, 1)
        rr_default_tp2_p = round(rr_default_sl_p * 3, 1)

        rr_direction = st.radio("Direction", ["LONG", "SHORT"], horizontal=True, key="rr_direction")
        rr_entry = st.number_input("Entry Price", value=round(rr_default_entry, 5),
                                   format="%.5f", step=float(pip_size), key="rr_entry")
        rr_sl_pips_in = st.number_input("Stop Loss (pips)", value=float(rr_default_sl_p),
                                        min_value=1.0, step=0.5, format="%.1f", key="rr_sl_pips")
        rr_tp1_pips_in = st.number_input("TP1 (pips)", value=float(rr_default_tp1_p),
                                         min_value=1.0, step=0.5, format="%.1f", key="rr_tp1_pips")
        rr_tp2_pips_in = st.number_input("TP2 (pips)", value=float(rr_default_tp2_p),
                                         min_value=1.0, step=0.5, format="%.1f", key="rr_tp2_pips")
        rr_partial_pct = st.slider("Partial close at TP1 (%)", 25, 75, 50, step=5, key="rr_partial")
        rr_min_rr = st.number_input("Minimum R:R to pass", value=2.0,
                                    min_value=1.0, max_value=5.0, step=0.5, format="%.1f",
                                    key="rr_min_rr")

        if rr_direction == "LONG":
            rr_sl_price = rr_entry - rr_sl_pips_in * pip_size
            rr_tp1_price = rr_entry + rr_tp1_pips_in * pip_size
            rr_tp2_price = rr_entry + rr_tp2_pips_in * pip_size
        else:
            rr_sl_price = rr_entry + rr_sl_pips_in * pip_size
            rr_tp1_price = rr_entry - rr_tp1_pips_in * pip_size
            rr_tp2_price = rr_entry - rr_tp2_pips_in * pip_size

    with st.expander("💵 Account Risk settings", expanded=False):
        ar_leverage = st.selectbox("Account Leverage", [30, 50, 100, 200, 400, 500],
                                   index=2, key="ar_leverage")
        ar_two_trades = st.checkbox("Take 2 trades per signal", value=True,
                                    help="Desk convention — every signal is entered as two positions.",
                                    key="ar_two_trades")
        ar_n_trades = 2 if ar_two_trades else 1
        ar_split_mode = st.radio(
            "Risk % applies as", ["Split across trades", "Per trade"], index=0,
            help="Split = the risk % is the budget for the whole signal. "
                 "Per trade = each of the two trades risks the full %, so total exposure doubles.",
            key="ar_split_mode")
        with st.spinner("Fetching…"):
            ar_live_price = fetch_price(ticker)
            ar_live_atr_pips = fetch_atr14_pips(ticker, pip_size)
        ar_sl_mode = st.radio("Stop method", ["Manual (pips)", "ATR × 1.5"],
                              index=1 if ar_live_atr_pips else 0, horizontal=True,
                              key="ar_sl_mode")
        if ar_sl_mode == "ATR × 1.5" and ar_live_atr_pips:
            ar_sl_pips = round(ar_live_atr_pips * 1.5, 1)
            st.caption(f"ATR14 = {ar_live_atr_pips:.1f} pips → SL = {ar_sl_pips:.1f} pips")
        else:
            ar_sl_pips = st.number_input("Stop Loss (pips)", value=30.0,
                                         min_value=1.0, step=0.5, format="%.1f", key="ar_sl_pips_in")
        st.caption(f"pip ${pip_val:.2f}/lot")

    st.divider()
    render_sidebar_nav()

# ══════════════════════════════════════════════════════════════════
# MAIN PAGE
# ══════════════════════════════════════════════════════════════════

st.markdown(f"""
<div style="background:linear-gradient(135deg,#000000 0%,#0a0a0a 50%,#000000 100%);
            border:1px solid #2a2a2a; border-radius:16px; padding:24px 28px;
            margin-bottom:20px;">
  <div style="font-size:24px; font-weight:700; color:#e6e6e6;">🛡️ Risk Suite</div>
  <div style="color:#9a9a9a; font-size:13px; margin-top:4px;">
    {selected_pair} · Stop Structure + R:R Calculator + Account Risk — one decision, three tabs
  </div>
  <div style="font-size:12px; color:#00ff41; margin-top:6px;">
    Checks #14–15 · {datetime.now().strftime('%A %d %B %Y  |  %H:%M')}
  </div>
</div>
""", unsafe_allow_html=True)

tab_stop, tab_rr, tab_account = st.tabs(
    ["🛡️ Stop Structure", "⚖️ R:R Calculator", "💵 Account Risk"]
)

# ══════════════════════════════════════════════
#  TAB 1 — Stop Structure (from stop-structure.py)
# ══════════════════════════════════════════════
with tab_stop:
    tab_scan, tab_detail = st.tabs(["📊 All-Pairs Scanner", "🔍 Pair Detail"])

    with tab_scan:
        st.markdown(
            f"Scanning all {len(inst_keys)} instruments · ATR({int(ss_atr_period)}) × {ss_atr_mult} · "
            f"pivot lookback {int(ss_pivot_lb)} bars …"
        )
        prog = st.progress(0)
        scan_results: list[dict] = []

        for idx, (pair_name, info) in enumerate(INSTRUMENTS.items()):
            prog.progress((idx + 1) / len(inst_keys), text=f"Scanning {pair_name} …")
            try:
                df_s = fetch_ohlc(info.ticker)
                if df_s.empty or len(df_s) < int(ss_atr_period) + 10:
                    scan_results.append({"pair": pair_name, "ok": False})
                    continue
                struct_s = find_structure_levels(df_s, pivot_lb=int(ss_pivot_lb))
                calc_s = compute_stops(
                    df_s, atr_mult=float(ss_atr_mult),
                    pip_size=info.pip_size, pip_val=info.pip,
                    account_bal=account_bal, risk_pct=risk_pct,
                    struct=struct_s, atr_period=int(ss_atr_period),
                )
                scan_results.append({
                    "pair": pair_name, "ok": True,
                    "price": calc_s["price"], "atr14": calc_s["atr14"],
                    "sl_long_pips": calc_s["sl_long_pips"],
                    "sl_short_pips": calc_s["sl_short_pips"],
                    "use_struct_l": calc_s["use_struct_long"],
                    "use_struct_s": calc_s["use_struct_short"],
                    "rr_tp1_long": calc_s["rr_tp1_long"],
                    "rr_tp1_short": calc_s["rr_tp1_short"],
                })
            except Exception:
                scan_results.append({"pair": pair_name, "ok": False})

        prog.empty()

        cnt_both = sum(1 for r in scan_results if r.get("ok") and r.get("use_struct_l") and r.get("use_struct_s"))
        cnt_long_s = sum(1 for r in scan_results if r.get("ok") and r.get("use_struct_l") and not r.get("use_struct_s"))
        cnt_short_s = sum(1 for r in scan_results if r.get("ok") and not r.get("use_struct_l") and r.get("use_struct_s"))
        cnt_atr = sum(1 for r in scan_results if r.get("ok") and not r.get("use_struct_l") and not r.get("use_struct_s"))

        sc1, sc2, sc3, sc4 = st.columns(4)
        for col_, val_, lbl_, c_ in [
            (sc1, cnt_both, "Both Structures", "#00ff66"),
            (sc2, cnt_long_s, "Long Struct Only", "#56d364"),
            (sc3, cnt_short_s, "Short Struct Only", "#ff3344"),
            (sc4, cnt_atr, "ATR Fallback", "#555555"),
        ]:
            with col_:
                st.markdown(
                    f'<div class="metric-box">'
                    f'<div class="metric-value" style="color:{c_};font-size:28px;">{val_}</div>'
                    f'<div class="metric-label">{lbl_}</div></div>',
                    unsafe_allow_html=True)

        st.markdown("---")

        cols3 = st.columns(3)
        for i, r in enumerate(scan_results):
            pair_name = r["pair"]
            is_sel = pair_name == selected_pair
            border = "border:2px solid #00ff41;" if is_sel else "border:1px solid #2a2a2a;"

            if not r.get("ok"):
                card_body = '<span style="font-size:11px;color:#ff3344;">No data</span>'
            else:
                l_note = "struct" if r["use_struct_l"] else "ATR"
                s_note = "struct" if r["use_struct_s"] else "ATR"
                l_col = "#00ff66" if r["use_struct_l"] else "#555555"
                s_col = "#ff3344" if r["use_struct_s"] else "#555555"
                rr_l = r["rr_tp1_long"]
                rr_s = r["rr_tp1_short"]
                rr_l_c = "#00ff66" if rr_l >= 2 else "#ffcc00"
                rr_s_c = "#00ff66" if rr_s >= 2 else "#ffcc00"
                card_body = (
                    f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">'
                    f'<span style="font-size:13px;font-weight:600;color:#e6e6e6;">{pair_name}</span>'
                    f'<span style="font-size:11px;color:#9a9a9a;font-family:monospace;">{r["price"]:,.4f}</span>'
                    f'</div>'
                    f'<div style="display:flex;gap:10px;flex-wrap:wrap;">'
                    f'<span style="font-size:11px;color:#9a9a9a;">🔼 SL <span style="color:{l_col};font-family:monospace;">{r["sl_long_pips"]:.0f}p</span> '
                    f'<span style="font-size:10px;color:#555555;">({l_note})</span> '
                    f'<span style="color:{rr_l_c};font-size:10px;">R:{rr_l:.1f}</span></span>'
                    f'<span style="font-size:11px;color:#9a9a9a;">🔽 SL <span style="color:{s_col};font-family:monospace;">{r["sl_short_pips"]:.0f}p</span> '
                    f'<span style="font-size:10px;color:#555555;">({s_note})</span> '
                    f'<span style="color:{rr_s_c};font-size:10px;">R:{rr_s:.1f}</span></span>'
                    f'</div>'
                    f'<div style="font-size:10px;color:#555555;margin-top:4px;">ATR {r["atr14"]:.5f}</div>'
                )

            with cols3[i % 3]:
                st.markdown(
                    f'<div style="background:#0a0a0a;{border}border-radius:8px;'
                    f'padding:12px 14px;margin-bottom:8px;">{card_body}</div>',
                    unsafe_allow_html=True)

        with st.expander("📋 Full Scanner Results"):
            table_rows = []
            for r in scan_results:
                table_rows.append({
                    "Pair": r["pair"],
                    "Price": f'{r["price"]:,.5f}' if r.get("ok") else "—",
                    f"ATR({int(ss_atr_period)})": f'{r["atr14"]:.5f}' if r.get("ok") else "—",
                    "SL Long (pips)": f'{r["sl_long_pips"]:.1f}' if r.get("ok") else "—",
                    "SL Short (pips)": f'{r["sl_short_pips"]:.1f}' if r.get("ok") else "—",
                    "Long Struct": "✅" if r.get("use_struct_l") else "ATR" if r.get("ok") else "—",
                    "Short Struct": "✅" if r.get("use_struct_s") else "ATR" if r.get("ok") else "—",
                    "R:R TP1 L": f'{r["rr_tp1_long"]:.2f}' if r.get("ok") else "—",
                    "R:R TP1 S": f'{r["rr_tp1_short"]:.2f}' if r.get("ok") else "—",
                })
            st.dataframe(pd.DataFrame(table_rows), width="stretch", hide_index=True)

    with tab_detail:
        detail_ok = True
        with st.spinner(f"Loading data for {selected_pair}…"):
            ss_df = fetch_ohlc(ticker)

        if ss_df.empty or len(ss_df) < int(ss_atr_period) + 10:
            st.error(f"⚠️ Not enough data for **{selected_pair}**.")
            detail_ok = False

        if detail_ok:
            struct = find_structure_levels(ss_df, pivot_lb=int(ss_pivot_lb))
            calc = compute_stops(ss_df, atr_mult=float(ss_atr_mult),
                                 pip_size=pip_size, pip_val=pip_val,
                                 account_bal=account_bal, risk_pct=risk_pct,
                                 struct=struct, atr_period=int(ss_atr_period))

            _ss_key = (f"{selected_pair}|{ss_direction}|{ss_atr_mult}|{ss_atr_period}|"
                      f"{ss_pivot_lb}|{calc['sl_pips']:.1f}")
            if NotifyCache("stop_structure_log").filter_new([_ss_key]):
                log_tool_usage("stop_structure", {
                    "pair": selected_pair, "direction": ss_direction,
                    "atr_mult": ss_atr_mult, "atr_period": ss_atr_period,
                    "pivot_lb": ss_pivot_lb, "price": calc["price"],
                    "atr14": calc["atr14"], "sl_pips": calc["sl_pips"],
                    "tp1_pips": calc["tp1_pips"], "tp2_pips": calc["tp2_pips"],
                    "risk_amount": calc["risk_amount"],
                })

            k1, k2, k3, k4, k5, k6 = st.columns(6)
            for col_, val_, lbl_, c_ in [
                (k1, f"{calc['price']:.5f}", "Current Price", "#e6e6e6"),
                (k2, f"{calc['atr14']:.5f}", f"ATR ({int(ss_atr_period)})", "#00ff41"),
                (k3, f"{calc['sl_pips']:.1f}", f"SL Pips ({ss_atr_mult}×ATR)", "#ff3344"),
                (k4, f"{calc['tp1_pips']:.1f}", "TP1 Pips (2:1)", "#00ff66"),
                (k5, f"{calc['tp2_pips']:.1f}", "TP2 Pips (3:1)", "#56d364"),
                (k6, f"${calc['risk_amount']:.2f}", "Risk Amount", "#ffcc00"),
            ]:
                with col_:
                    st.markdown(
                        f'<div class="metric-box">'
                        f'<div class="metric-value" style="color:{c_};font-size:17px;">{val_}</div>'
                        f'<div class="metric-label">{lbl_}</div></div>',
                        unsafe_allow_html=True)

            st.markdown("---")

            st.markdown('<div class="section-title">🎯 Stop Loss Levels</div>', unsafe_allow_html=True)
            col_l, col_r = st.columns(2)

            if ss_direction in ("LONG", "BOTH"):
                struct_note = "📐 Structure-based" if calc["use_struct_long"] else "📐 ATR-based"
                with col_l:
                    st.markdown(f"""
                    <div class="sl-card sl-card-long">
                      <div style="font-size:15px;font-weight:700;color:#00ff66;margin-bottom:14px;">🔼 LONG Setup</div>
                      {level_row("Entry (current price)", f"{calc['price']:.5f}")}
                      {level_row("Nearest swing low", f"{calc['struct_sl']:.5f}" if calc['struct_sl'] else "—", "structural anchor")}
                      {level_row(f"Stop Loss  ({calc['sl_long_pips']:.0f} pips)", f"{calc['sl_long']:.5f}", struct_note + " + buffer")}
                      {level_row(f"TP1  ({calc['tp1_pips']:.0f} pips)", f"{calc['tp1_long']:.5f}", f"R:R {calc['rr_tp1_long']:.1f}:1")}
                      {level_row(f"TP2  ({calc['tp2_pips']:.0f} pips)", f"{calc['tp2_long']:.5f}", f"R:R {calc['rr_tp2_long']:.1f}:1")}
                      {level_row("Position size", f"{calc['lot_long']:.2f} lots", f"${calc['risk_amount']:.2f} risk at {risk_pct}%")}
                    </div>
                    """, unsafe_allow_html=True)

            if ss_direction in ("SHORT", "BOTH"):
                struct_note = "📐 Structure-based" if calc["use_struct_short"] else "📐 ATR-based"
                with col_r if ss_direction == "BOTH" else col_l:
                    st.markdown(f"""
                    <div class="sl-card sl-card-short">
                      <div style="font-size:15px;font-weight:700;color:#ff3344;margin-bottom:14px;">🔽 SHORT Setup</div>
                      {level_row("Entry (current price)", f"{calc['price']:.5f}")}
                      {level_row("Nearest swing high", f"{calc['struct_sh']:.5f}" if calc['struct_sh'] else "—", "structural anchor")}
                      {level_row(f"Stop Loss  ({calc['sl_short_pips']:.0f} pips)", f"{calc['sl_short']:.5f}", struct_note + " + buffer")}
                      {level_row(f"TP1  ({calc['tp1_pips']:.0f} pips)", f"{calc['tp1_short']:.5f}", f"R:R {calc['rr_tp1_short']:.1f}:1")}
                      {level_row(f"TP2  ({calc['tp2_pips']:.0f} pips)", f"{calc['tp2_short']:.5f}", f"R:R {calc['rr_tp2_short']:.1f}:1")}
                      {level_row("Position size", f"{calc['lot_short']:.2f} lots", f"${calc['risk_amount']:.2f} risk at {risk_pct}%")}
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("---")

            st.markdown('<div class="section-title">📈 Daily Chart — Structure · SL · TP Levels</div>',
                        unsafe_allow_html=True)
            fig = build_structure_chart(ss_df, selected_pair, calc, struct, ss_direction, show_n=ss_show_n)
            st.plotly_chart(fig, width="stretch")

            st.markdown("---")

            with st.expander("📋 ATR History — Last 15 Sessions"):
                atr_hist = calc["atr_series"].tail(15).copy()[::-1]
                atr_df = pd.DataFrame({
                    "Date": atr_hist.index.strftime("%Y-%m-%d"),
                    f"ATR({int(ss_atr_period)})": atr_hist.values.round(5),
                    "SL Dist (price)": (atr_hist.values * float(ss_atr_mult)).round(5),
                    "SL Pips": (atr_hist.values * float(ss_atr_mult) / pip_size).round(1),
                    "TP1 Pips": (atr_hist.values * float(ss_atr_mult) / pip_size * 2).round(1),
                    "TP2 Pips": (atr_hist.values * float(ss_atr_mult) / pip_size * 3).round(1),
                })
                st.dataframe(atr_df, width="stretch", hide_index=True)

            st.markdown("---")

            st.markdown('<div class="section-title">📖 Stop Placement Logic</div>', unsafe_allow_html=True)
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"""
                <div class="explainer">
                <b style="color:#00ff66;">🔼 LONG — SL below structure</b><br><br>
                <b>Step 1 — Identify swing low</b><br>
                Find the most recent confirmed swing low on the daily chart.
                This is the structural level the market has already rejected — it acts as natural support.<br><br>
                <b>Step 2 — Add ATR buffer</b><br>
                Place the SL <em>below</em> the swing low by 20% of ATR({int(ss_atr_period)}).
                This prevents a wick through the level from stopping you out prematurely.<br><br>
                <b>Step 3 — Verify R:R</b><br>
                TP1 = 2× SL distance. TP2 = 3× SL distance.
                If R:R to TP1 is below 2:1, the zone is too far — skip the trade.
                </div>
                """, unsafe_allow_html=True)
            with col_b:
                st.markdown(f"""
                <div class="explainer">
                <b style="color:#ff3344;">🔽 SHORT — SL above structure</b><br><br>
                <b>Step 1 — Identify swing high</b><br>
                Find the most recent confirmed swing high on the daily chart.
                This is where the market rejected higher prices — it acts as natural resistance.<br><br>
                <b>Step 2 — Add ATR buffer</b><br>
                Place the SL <em>above</em> the swing high by 20% of ATR({int(ss_atr_period)}).
                Gives the trade room to breathe without being clipped by a false breakout wick.<br><br>
                <b>Step 3 — Verify R:R</b><br>
                TP1 = 2× SL distance. TP2 = 3× SL distance.
                A valid short trade needs TP1 R:R ≥ 2:1 before entry.
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="formula-box">
            ATR SL distance &nbsp;= &nbsp;ATR({int(ss_atr_period)}) &nbsp;×&nbsp; {ss_atr_mult} &nbsp;= &nbsp;
            {calc['atr14']:.5f} × {ss_atr_mult} &nbsp;= &nbsp;
            <b style="color:#ff3344;">{calc['atr14'] * float(ss_atr_mult):.5f} &nbsp; ({calc['sl_pips']:.1f} pips)</b>
            <br>
            SL LONG  &nbsp;= &nbsp;Entry &nbsp;−&nbsp; ATR SL distance
            &nbsp;=&nbsp; {calc['price']:.5f} − {calc['atr14'] * float(ss_atr_mult):.5f}
            &nbsp;=&nbsp; <b style="color:#ff3344;">{calc['sl_long']:.5f}</b>
            <br>
            SL SHORT &nbsp;= &nbsp;Entry &nbsp;+&nbsp; ATR SL distance
            &nbsp;=&nbsp; {calc['price']:.5f} + {calc['atr14'] * float(ss_atr_mult):.5f}
            &nbsp;=&nbsp; <b style="color:#ff3344;">{calc['sl_short']:.5f}</b>
            <br>
            TP1 &nbsp;= &nbsp;SL distance × 2.0 &nbsp;= &nbsp;
            <b style="color:#00ff66;">{calc['tp1_pips']:.1f} pips</b> &nbsp;&nbsp;
            TP2 &nbsp;= &nbsp;SL distance × 3.0 &nbsp;= &nbsp;
            <b style="color:#56d364;">{calc['tp2_pips']:.1f} pips</b>
            <br>
            Lot size &nbsp;= &nbsp;Risk $ ÷ (SL pips × pip value)
            &nbsp;= &nbsp;${calc['risk_amount']:.2f} ÷ ({calc['sl_pips']:.1f} × {pip_val})
            &nbsp;= &nbsp;<b style="color:#00ff41;">{calc['lot_long']:.2f} lots</b>
            </div>
            <div style="font-size:11px;color:#555555;margin-top:6px;">
              ⚠️ If structure-based SL is more than 3× ATR away from price the calculator falls back to
              the pure ATR-based distance. Adjust the pivot lookback in the sidebar if needed.
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  TAB 2 — R:R Calculator (from rr-calculator.py)
# ══════════════════════════════════════════════
with tab_rr:
    c = compute_rr(
        entry=rr_entry, sl=rr_sl_price, tp1=rr_tp1_price, tp2=rr_tp2_price,
        pip_size=pip_size, pip_val=pip_val,
        account_bal=account_bal, risk_pct=risk_pct,
        partial_pct=rr_partial_pct,
    )

    if c["rr_tp1"] >= rr_min_rr * 1.5:
        c["grade"] = "excellent"
        c["label"] = "✅ EXCELLENT  R:R"
        c["color"] = "#00ff66"
        c["vclass"] = "verdict-pass"
    elif c["rr_tp1"] >= rr_min_rr:
        c["grade"] = "pass"
        c["label"] = f"✅ R:R MINIMUM MET  ({rr_min_rr:.1f}:1)"
        c["color"] = "#00ff66"
        c["vclass"] = "verdict-pass"
    elif c["rr_tp1"] >= rr_min_rr * 0.75:
        c["grade"] = "marginal"
        c["label"] = f"⚠️ MARGINAL — {c['rr_tp1']:.2f}:1 (min {rr_min_rr:.1f}:1)"
        c["color"] = "#ffcc00"
        c["vclass"] = "verdict-warn"
    else:
        c["grade"] = "fail"
        c["label"] = f"❌ BELOW MINIMUM — {c['rr_tp1']:.2f}:1 (need {rr_min_rr:.1f}:1)"
        c["color"] = "#ff3344"
        c["vclass"] = "verdict-fail"

    _rr_key = (f"{selected_pair}|{rr_direction}|{c['entry']:.5f}|{c['sl']:.5f}|"
              f"{c['tp1']:.5f}|{c['tp2']:.5f}|{rr_min_rr:.1f}")
    if NotifyCache("rr_calculator_log").filter_new([_rr_key]):
        log_tool_usage("rr_calculator", {
            "pair": selected_pair, "direction": rr_direction,
            "entry": c["entry"], "sl": c["sl"], "tp1": c["tp1"], "tp2": c["tp2"],
            "sl_pips": c["sl_pips"], "tp1_pips": c["tp1_pips"], "tp2_pips": c["tp2_pips"],
            "rr_tp1": c["rr_tp1"], "rr_tp2": c["rr_tp2"],
            "lot_size": c["lot_size"], "risk_amt": c["risk_amt"],
            "account_bal": account_bal, "risk_pct": risk_pct,
            "min_rr": rr_min_rr, "grade": c["grade"],
        })

    st.markdown(f"""
    TP1 = {c['tp1_pips']:.1f} pips → R:R {c['rr_tp1']:.2f}:1
    {'✅' if c['rr_tp1'] >= rr_min_rr else '❌'} · {selected_pair} · {rr_direction}
    """)

    rr_tab_scan, rr_tab_detail = st.tabs(["📊 All-Pairs Scanner", "🔍 Pair Detail"])

    with rr_tab_scan:
        st.markdown("**ATR-Based Position Sizing — All Pairs**")
        st.caption(f"SL = ATR×1.5 · TP1 = SL×{rr_min_rr:.1f} · Risk {risk_pct}% of ${account_bal:,.0f}")

        rr_prog = st.progress(0, text="Scanning…")
        rr_scan = []

        for rr_i, (rr_name, rr_info) in enumerate(INSTRUMENTS.items()):
            rr_prog.progress((rr_i + 1) / len(inst_keys), text=f"Fetching {rr_name}…")
            rr_price = fetch_price(rr_info.ticker)
            rr_atr = fetch_atr14(rr_info.ticker)

            if rr_price is None or rr_atr is None:
                rr_scan.append({"pair": rr_name, "price": None, "atr": None,
                                "sl_pips": None, "tp1_pips": None,
                                "lot": None, "risk_usd": None, "status": "ERROR"})
                continue

            rr_sl_pips = rr_atr * 1.5 / rr_info.pip_size
            rr_tp1_pips = rr_sl_pips * rr_min_rr
            rr_risk_usd = account_bal * risk_pct / 100
            rr_lot = rr_risk_usd / (rr_sl_pips * rr_info.pip)
            rr_scan.append({"pair": rr_name, "price": rr_price, "atr": rr_atr,
                            "sl_pips": rr_sl_pips, "tp1_pips": rr_tp1_pips,
                            "lot": rr_lot, "risk_usd": rr_risk_usd, "status": "OK"})

        rr_prog.empty()

        rr_std_n = sum(1 for r in rr_scan if r["status"] == "OK" and r["lot"] >= 1.0)
        rr_mini_n = sum(1 for r in rr_scan if r["status"] == "OK" and 0.10 <= r["lot"] < 1.0)
        rr_mic_n = sum(1 for r in rr_scan if r["status"] == "OK" and r["lot"] < 0.10)
        rr_err_n = sum(1 for r in rr_scan if r["status"] == "ERROR")

        rr_sc1, rr_sc2, rr_sc3, rr_sc4 = st.columns(4)
        for rr_col, rr_cnt, rr_lbl, rr_clr in [
            (rr_sc1, rr_std_n, "≥1.0 Lot", "#00ff66"),
            (rr_sc2, rr_mini_n, "0.1–1.0 Lot", "#00ff41"),
            (rr_sc3, rr_mic_n, "<0.1 Lot", "#ffcc00"),
            (rr_sc4, rr_err_n, "Errors", "#9a9a9a"),
        ]:
            with rr_col:
                st.markdown(
                    f'<div class="metric-box">'
                    f'<div class="metric-value" style="color:{rr_clr};font-size:22px;">{rr_cnt}</div>'
                    f'<div class="metric-label">{rr_lbl}</div></div>',
                    unsafe_allow_html=True,
                )

        st.markdown("---")

        rr_gcols = st.columns(3)
        for rr_i, rr_r in enumerate(rr_scan):
            with rr_gcols[rr_i % 3]:
                rr_sel = rr_r["pair"] == selected_pair
                rr_bclr = "#00ff41" if rr_sel else "#2a2a2a"
                if rr_r["status"] == "ERROR":
                    st.markdown(
                        f'<div style="background:#0a0a0a;border:1px solid {rr_bclr};'
                        f'border-radius:10px;padding:14px;margin-bottom:10px;">'
                        f'<div style="font-weight:700;color:#e6e6e6;font-size:13px;">{rr_r["pair"]}</div>'
                        f'<div style="color:#555555;font-size:11px;margin-top:4px;">⚠️ Data unavailable</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    rr_lc = "#00ff66" if rr_r["lot"] >= 0.10 else "#ffcc00"
                    rr_tag = ' <span style="color:#00ff41;font-size:10px;">◀ selected</span>' if rr_sel else ""
                    rr_pfx = f"{rr_r['price']:.5f}" if rr_r["price"] < 100 else f"{rr_r['price']:.3f}"
                    st.markdown(
                        f'<div style="background:#0a0a0a;border:1px solid {rr_bclr};'
                        f'border-radius:10px;padding:14px;margin-bottom:10px;">'
                        f'<div style="font-weight:700;color:#e6e6e6;font-size:13px;">{rr_r["pair"]}{rr_tag}</div>'
                        f'<div style="font-size:11px;color:#9a9a9a;margin-top:2px;">{rr_pfx}</div>'
                        f'<div style="display:flex;gap:6px;margin-top:8px;flex-wrap:wrap;">'
                        f'<span style="background:#000000;border:1px solid #2a2a2a;border-radius:4px;'
                        f'padding:3px 7px;font-size:11px;color:#ff3344;'
                        f'font-family:\'JetBrains Mono\',monospace;">SL {rr_r["sl_pips"]:.0f}p</span>'
                        f'<span style="background:#000000;border:1px solid #2a2a2a;border-radius:4px;'
                        f'padding:3px 7px;font-size:11px;color:#00ff66;'
                        f'font-family:\'JetBrains Mono\',monospace;">TP1 {rr_r["tp1_pips"]:.0f}p</span>'
                        f'<span style="background:#000000;border:1px solid #2a2a2a;border-radius:4px;'
                        f'padding:3px 7px;font-size:11px;color:{rr_lc};'
                        f'font-family:\'JetBrains Mono\',monospace;">{rr_r["lot"]:.2f} lots</span>'
                        f'</div></div>',
                        unsafe_allow_html=True,
                    )

        with st.expander("📋 Full Scanner Table"):
            rr_rows = []
            for rr_r in rr_scan:
                if rr_r["status"] == "ERROR":
                    rr_rows.append({"Pair": rr_r["pair"], "Price": "—", "ATR14": "—",
                                    "SL (pips)": "—", "TP1 (pips)": "—",
                                    "Lot Size": "—", "Risk $": "—"})
                else:
                    rr_rows.append({
                        "Pair": rr_r["pair"],
                        "Price": round(rr_r["price"], 5) if rr_r["price"] < 100 else round(rr_r["price"], 2),
                        "ATR14": round(rr_r["atr"], 5) if rr_r["atr"] < 1 else round(rr_r["atr"], 3),
                        "SL (pips)": round(rr_r["sl_pips"], 1),
                        "TP1 (pips)": round(rr_r["tp1_pips"], 1),
                        "Lot Size": round(rr_r["lot"], 2),
                        "Risk $": f"${rr_r['risk_usd']:.2f}",
                    })
            st.dataframe(pd.DataFrame(rr_rows), width="stretch", hide_index=True)

    with rr_tab_detail:
        rr_risk_seg = 1 / (1 + c["rr_tp1"]) * 100 if c["rr_tp1"] else 50
        rr_rew_seg = 100 - rr_risk_seg
        rr_min_mark = 1 / (1 + rr_min_rr) * 100

        st.markdown(f"""
        <div class="{c['vclass']}">
          <div style="font-size:23px; font-weight:800; color:{c['color']};
                      letter-spacing:1px; margin-bottom:6px;">{c['label']}</div>
          <div style="font-size:32px; font-weight:900; color:{c['color']};
                      font-family:'JetBrains Mono',monospace; margin:8px 0;">
            {c['rr_tp1']:.2f} : 1
          </div>
          <div style="font-size:13px; color:#9a9a9a; margin-bottom:14px;">
            TP1 = {c['tp1_pips']:.1f} pips &nbsp;·&nbsp;
            SL  = {c['sl_pips']:.1f} pips &nbsp;·&nbsp;
            Min required = {rr_min_rr:.1f}:1
          </div>
          <div style="margin:0 auto; max-width:600px;">
            <div style="display:flex; justify-content:space-between;
                        font-size:11px; color:#9a9a9a; margin-bottom:4px;">
              <span>Risk (SL)</span><span>Reward (TP1)</span>
            </div>
            <div class="rr-track">
              <div style="width:{rr_risk_seg:.1f}%; background:rgba(248,81,73,0.80);
                          display:flex; align-items:center; justify-content:center;
                          font-size:11px; color:#fff; font-weight:700;">
                {rr_risk_seg:.0f}%
              </div>
              <div style="width:{rr_rew_seg:.1f}%; background:rgba(63,185,80,0.80);
                          display:flex; align-items:center; justify-content:center;
                          font-size:11px; color:#fff; font-weight:700;">
                {rr_rew_seg:.0f}%
              </div>
            </div>
            <div style="font-size:10px; color:#555555; text-align:left; margin-top:2px;">
              ▲ Min {rr_min_rr:.1f}:1 threshold at {rr_min_mark:.0f}% / {100-rr_min_mark:.0f}%
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        rk1, rk2, rk3, rk4, rk5, rk6 = st.columns(6)
        rr_color = c["color"]
        for col, val, lbl, color in [
            (rk1, f"{c['rr_tp1']:.2f}:1", "R:R to TP1", rr_color),
            (rk2, f"{c['rr_tp2']:.2f}:1", "R:R to TP2", "#56d364"),
            (rk3, f"{c['sl_pips']:.1f}", "SL (pips)", "#ff3344"),
            (rk4, f"{c['tp1_pips']:.1f}", "TP1 (pips)", "#00ff66"),
            (rk5, f"{c['tp2_pips']:.1f}", "TP2 (pips)", "#56d364"),
            (rk6, f"{c['be_winrate_tp1']:.1f}%", "Breakeven Win Rate", "#00ff41"),
        ]:
            with col:
                st.markdown(
                    f'<div class="metric-box">'
                    f'<div class="metric-value" style="color:{color};font-size:18px;">{val}</div>'
                    f'<div class="metric-label">{lbl}</div></div>',
                    unsafe_allow_html=True,
                )

        st.markdown("---")

        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.markdown('<div class="section-title">📐 Trade Levels</div>', unsafe_allow_html=True)
            dir_icon = "🔼" if c["long_trade"] else "🔽"

            st.markdown(f"""
            <div class="card">
              <div class="card-header">{dir_icon} {rr_direction} — {selected_pair}</div>
              {lvl_row("Entry", f"{c['entry']:.5f}", "", "#ffffff")}
              {lvl_row("Stop Loss", f"{c['sl']:.5f}",
                       f"−{c['sl_pips']:.1f} pips  (1R)", "#ff3344")}
              {lvl_row("TP1", f"{c['tp1']:.5f}",
                       f"+{c['tp1_pips']:.1f} pips  ({c['rr_tp1']:.2f}R)",
                       "#00ff66" if c['rr_tp1'] >= rr_min_rr else "#ffcc00")}
              {lvl_row("TP2", f"{c['tp2']:.5f}",
                       f"+{c['tp2_pips']:.1f} pips  ({c['rr_tp2']:.2f}R)", "#56d364")}
              {lvl_row("Lot size", f"{c['lot_size']:.2f} lots",
                       f"${c['risk_amt']:.2f} risk ({risk_pct}%)", "#00ff41")}
              {lvl_row(f"Partial close ({rr_partial_pct}%)",
                       f"{c['partial_lots']:.2f} lots at TP1", "", "#a29bfe")}
              {lvl_row(f"Runner ({100-rr_partial_pct}%)",
                       f"{c['runner_lots']:.2f} lots to TP2", "", "#a29bfe")}
            </div>
            """, unsafe_allow_html=True)

        with col_right:
            st.markdown('<div class="section-title">💰 P&L Projection</div>', unsafe_allow_html=True)

            rr_tp1_usd = c["partial_lots"] * c["tp1_pips"] * pip_val
            rr_tp2_usd = c["runner_lots"] * c["tp2_pips"] * pip_val
            rr_sl_usd = c["lot_size"] * c["sl_pips"] * pip_val
            rr_total = rr_tp1_usd + rr_tp2_usd

            rr_pnl_color = "#00ff66" if c["rr_tp1"] >= rr_min_rr else "#ff3344"

            st.markdown(f"""
            <div class="card">
              <div class="card-header">💵 Projected P&L</div>
              <div style="display:flex; justify-content:space-between; align-items:center;
                          padding:12px 0; border-bottom:1px solid #2a2a2a;">
                <span style="color:#9a9a9a; font-size:13px;">
                  TP1 profit &nbsp;({rr_partial_pct}% of position)
                </span>
                <span style="font-family:'JetBrains Mono',monospace; font-weight:700;
                             color:#00ff66; font-size:15px;">+${rr_tp1_usd:.2f}</span>
              </div>
              <div style="display:flex; justify-content:space-between; align-items:center;
                          padding:12px 0; border-bottom:1px solid #2a2a2a;">
                <span style="color:#9a9a9a; font-size:13px;">
                  TP2 profit &nbsp;({100-rr_partial_pct}% of position)
                </span>
                <span style="font-family:'JetBrains Mono',monospace; font-weight:700;
                             color:#56d364; font-size:15px;">+${rr_tp2_usd:.2f}</span>
              </div>
              <div style="display:flex; justify-content:space-between; align-items:center;
                          padding:12px 0; border-bottom:1px solid #2a2a2a;">
                <span style="color:#9a9a9a; font-size:13px;">Total profit (both TPs hit)</span>
                <span style="font-family:'JetBrains Mono',monospace; font-weight:800;
                             color:#00ff66; font-size:17px;">+${rr_total:.2f}</span>
              </div>
              <div style="display:flex; justify-content:space-between; align-items:center;
                          padding:12px 0; border-bottom:1px solid #2a2a2a;">
                <span style="color:#9a9a9a; font-size:13px;">Max loss (SL hit)</span>
                <span style="font-family:'JetBrains Mono',monospace; font-weight:700;
                             color:#ff3344; font-size:15px;">−${rr_sl_usd:.2f}</span>
              </div>
              <div style="display:flex; justify-content:space-between; align-items:center;
                          padding:14px 0 4px 0;">
                <span style="color:#e6e6e6; font-size:13px; font-weight:600;">
                  Net profit-to-risk ratio
                </span>
                <span style="font-family:'JetBrains Mono',monospace; font-weight:800;
                             color:{rr_pnl_color}; font-size:18px;">
                  {rr_total/rr_sl_usd:.2f}:1
                </span>
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        st.markdown('<div class="section-title">📊 Visuals</div>', unsafe_allow_html=True)

        chart_l, chart_r = st.columns([3, 2])
        with chart_l:
            st.plotly_chart(build_rr_chart(c, selected_pair), width="stretch")
        with chart_r:
            st.plotly_chart(build_winrate_chart(c), width="stretch")

        st.markdown("---")

        st.markdown('<div class="section-title">📋 Win-Rate Scenario Analysis</div>',
                    unsafe_allow_html=True)
        st.caption("Showing net P&L over 100 trades at varying win rates — "
                   f"using {c['lot_size']:.2f} lots, SL=${rr_sl_usd:.2f}, TP1=${rr_tp1_usd + rr_tp2_usd:.2f}")

        rr_scenario_rows = []
        for wr in [30, 35, 40, 45, 50, 55, 60, 65, 70]:
            wins = wr
            losses = 100 - wr
            gross = wins * (rr_tp1_usd + rr_tp2_usd) - losses * rr_sl_usd
            edge = gross / 100
            status = "✅" if gross > 0 else "❌"
            rr_scenario_rows.append({
                "Win Rate": f"{wr}%",
                "Wins / 100": wins,
                "Losses / 100": losses,
                "Gross P&L ($)": round(gross, 2),
                "Edge per trade ($)": round(edge, 2),
                "Profitable": status,
            })

        rr_sc_df = pd.DataFrame(rr_scenario_rows)
        st.dataframe(rr_sc_df, width="stretch", hide_index=True,
                     column_config={
                         "Gross P&L ($)": st.column_config.NumberColumn(format="$%.2f"),
                         "Edge per trade ($)": st.column_config.NumberColumn(format="$%.2f"),
                     })

        st.markdown("---")

        st.markdown('<div class="section-title">📖 R:R Logic & Rules</div>',
                    unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"""
            <div class="explainer">
            <b style="color:#00ff41;">⚖️ Why R:R ≥ {rr_min_rr:.1f}:1 matters</b><br><br>
            At a {rr_min_rr:.1f}:1 R:R, you only need to be right <b style="color:#e6e6e6;">
            {1/(1+rr_min_rr)*100:.0f}%</b> of the time to break even.<br><br>
            At a 2:1 R:R with a 40% win rate, 100 trades produce:<br>
            40 wins × $200 − 60 losses × $100 = <b style="color:#00ff66;">+$2,000 net profit</b>.<br><br>
            This is why a losing strategy can still be profitable — the R:R multiplies
            every winning trade relative to every loss. A minimum of 2:1 gives you
            enough margin to absorb losses while staying net positive.
            </div>
            """, unsafe_allow_html=True)

        with col_b:
            st.markdown(f"""
            <div class="explainer">
            <b style="color:#00ff41;">📐 How to use this checker</b><br><br>
            <b>1.</b> Fetch the live price (⚡ button in sidebar) or enter your entry manually.<br><br>
            <b>2.</b> Enter your SL in pips — pulled from the Stop Structure tab.<br><br>
            <b>3.</b> TP1 = at least {rr_min_rr:.1f}× the SL distance.
            TP2 = 3× the SL distance for the runner.<br><br>
            <b>4.</b> If R:R is below {rr_min_rr:.1f}:1 → <b style="color:#ff3344;">do not take the trade.</b>
            Widen the TP target or wait for a closer entry.<br><br>
            Must pass before confirming position size in the Account Risk tab.
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="formula-box">
        R:R to TP1 &nbsp;= &nbsp;TP1 distance ÷ SL distance
        &nbsp;= &nbsp;{c['tp1_pips']:.1f} pips ÷ {c['sl_pips']:.1f} pips
        &nbsp;= &nbsp;<b style="color:{c['color']};">{c['rr_tp1']:.2f} : 1</b>
        &nbsp; {'✅' if c['rr_tp1'] >= rr_min_rr else '❌'} (min {rr_min_rr:.1f}:1)<br>
        R:R to TP2 &nbsp;= &nbsp;{c['tp2_pips']:.1f} pips ÷ {c['sl_pips']:.1f} pips
        &nbsp;= &nbsp;<b style="color:#56d364;">{c['rr_tp2']:.2f} : 1</b><br>
        Breakeven &nbsp;&nbsp;= &nbsp;1 ÷ (1 + {c['rr_tp1']:.2f})
        &nbsp;= &nbsp;<b style="color:#00ff41;">{c['be_winrate_tp1']:.1f}% win rate needed</b>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  TAB 3 — Account Risk (from account-risk.py)
# ══════════════════════════════════════════════
with tab_account:
    ar_r = size_position(account_bal, risk_pct, ar_sl_pips, pip_val, ar_n_trades, ar_split_mode)

    ar_price = ar_live_price if ar_live_price else 0.0
    ar_notional_per_trade = ar_r["units_per_trade"] * ar_price
    ar_margin_per_trade = ar_notional_per_trade / ar_leverage if ar_leverage else 0.0
    ar_margin_total = ar_margin_per_trade * ar_n_trades
    ar_margin_pct = (ar_margin_total / account_bal * 100) if account_bal else 0.0

    if ar_r["risk_pct_total"] > 4:
        ar_band_color, ar_band_txt = "#ff3344", "⚠️ OVER-LEVERAGED — total risk above 4%"
    elif ar_r["risk_pct_total"] > 2:
        ar_band_color, ar_band_txt = "#ffcc00", "⚠️ ELEVATED — total risk above 2%"
    else:
        ar_band_color, ar_band_txt = "#00ff66", "✅ RISK WITHIN LIMITS"

    _ar_key = (f"{selected_pair}|{account_bal:.2f}|{risk_pct:.2f}|{ar_sl_pips:.1f}|"
              f"{ar_n_trades}|{ar_split_mode}")
    if NotifyCache("account_risk_log").filter_new([_ar_key]):
        log_tool_usage("account_risk", {
            "pair": selected_pair, "balance": account_bal, "risk_pct": risk_pct,
            "sl_pips": ar_sl_pips, "n_trades": ar_n_trades, "split_mode": ar_split_mode,
            "leverage": ar_leverage, "lots_per_trade": ar_r["lots_per_trade"],
            "lots_total": ar_r["lots_total"], "risk_per_trade": ar_r["risk_per_trade"],
            "risk_total": ar_r["risk_total"], "risk_pct_total": ar_r["risk_pct_total"],
        })

    st.markdown(f"""
    {selected_pair} · risk {risk_pct:.2f}% of ${account_bal:,.0f} · SL {ar_sl_pips:.1f} pips ·
    {ar_n_trades} trade{'s' if ar_n_trades > 1 else ''}/signal ({ar_split_mode.lower()})
    <div style="font-size:12px; color:{ar_band_color}; margin-top:6px;">{ar_band_txt}</div>
    """, unsafe_allow_html=True)

    ak1, ak2, ak3, ak4, ak5 = st.columns(5)
    for col, val, lbl, color in [
        (ak1, f"${ar_r['risk_total']:,.2f}", "Total $ at Risk", ar_band_color),
        (ak2, f"{ar_r['lots_total']:.2f}", "Total Lot Size", "#00ff41"),
        (ak3, f"{ar_r['lots_per_trade']:.2f}", "Lots / Trade", "#00ff66"),
        (ak4, f"${ar_r['risk_per_trade']:,.2f}", "$ Risk / Trade", "#56d364"),
        (ak5, f"{ar_r['risk_pct_total']:.2f}%", "Total Risk %", ar_band_color),
    ]:
        with col:
            st.markdown(
                f'<div class="metric-box">'
                f'<div class="metric-value" style="color:{color};font-size:20px;">{val}</div>'
                f'<div class="metric-label">{lbl}</div></div>',
                unsafe_allow_html=True)

    st.markdown("---")

    ar_col_left, ar_col_right = st.columns([1, 1])

    with ar_col_left:
        st.markdown('<div class="section-title">📐 Position Breakdown</div>', unsafe_allow_html=True)
        ar_units_note = "units (1 lot = 100k)" if selected_pair not in TREND_COMMODITIES else "≈ contract units"
        st.markdown(f"""
        <div class="card">
          <div class="card-header">{selected_pair} — {ar_n_trades} trade{'s' if ar_n_trades > 1 else ''}/signal</div>
          {lvl_row("Account balance", f"${account_bal:,.2f}")}
          {lvl_row("Risk budget / signal", f"${ar_r['risk_signal']:,.2f}", f"{risk_pct:.2f}%", ar_band_color)}
          {lvl_row("$ risk per trade", f"${ar_r['risk_per_trade']:,.2f}", "", "#56d364")}
          {lvl_row("Total $ at risk", f"${ar_r['risk_total']:,.2f}", f"{ar_r['risk_pct_total']:.2f}% of acct", ar_band_color)}
          {lvl_row("Lot size per trade", f"{ar_r['lots_per_trade']:.2f} lots", f"{ar_r['micro_per_trade']:.0f} micro", "#00ff41")}
          {lvl_row("Total lot size", f"{ar_r['lots_total']:.2f} lots", "", "#00ff41")}
          {lvl_row("Units per trade", f"{ar_r['units_per_trade']:,.0f}", ar_units_note, "#a29bfe")}
          {lvl_row("Stop loss", f"{ar_sl_pips:.1f} pips", f"${pip_val:.2f}/pip/lot", "#ff3344")}
        </div>
        """, unsafe_allow_html=True)

    with ar_col_right:
        st.markdown('<div class="section-title">💰 Margin &amp; Exposure</div>', unsafe_allow_html=True)
        if ar_price <= 0:
            st.markdown(
                '<div class="card"><div class="card-header">Margin</div>'
                '<div style="color:#9a9a9a;font-size:13px;">Fetch the live price '
                '(⚡ button in the sidebar) to estimate notional &amp; margin.</div></div>',
                unsafe_allow_html=True)
        else:
            ar_margin_color = "#ff3344" if ar_margin_pct > 50 else "#ffcc00" if ar_margin_pct > 25 else "#00ff66"
            st.markdown(f"""
            <div class="card">
              <div class="card-header">Live price {fmt_price(ar_price)} · {ar_leverage}:1 leverage</div>
              {lvl_row("Notional per trade", f"${ar_notional_per_trade:,.0f}", "", "#e6e6e6")}
              {lvl_row("Total notional", f"${ar_notional_per_trade * ar_n_trades:,.0f}", "", "#e6e6e6")}
              {lvl_row("Margin per trade", f"${ar_margin_per_trade:,.2f}", "", "#a29bfe")}
              {lvl_row("Total margin used", f"${ar_margin_total:,.2f}", f"{ar_margin_pct:.1f}% of acct", ar_margin_color)}
              {lvl_row("Free margin after", f"${max(account_bal - ar_margin_total, 0):,.2f}", "", "#56d364")}
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown('<div class="section-title">📊 Risk Ladder — $ &amp; Lots by Risk %</div>',
                unsafe_allow_html=True)
    st.caption(f"Same SL ({ar_sl_pips:.1f} pips) on {selected_pair}, {ar_n_trades} trade(s)/signal, "
               f"{ar_split_mode.lower()}.")

    ar_ladder_rows = []
    for rp in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        rr = size_position(account_bal, rp, ar_sl_pips, pip_val, ar_n_trades, ar_split_mode)
        ar_ladder_rows.append({
            "Risk %": f"{rp:.1f}%",
            "$ / Signal": round(rr["risk_signal"], 2),
            "$ / Trade": round(rr["risk_per_trade"], 2),
            "Total $ Risk": round(rr["risk_total"], 2),
            "Lots / Trade": round(rr["lots_per_trade"], 2),
            "Total Lots": round(rr["lots_total"], 2),
        })
    st.dataframe(
        pd.DataFrame(ar_ladder_rows), width="stretch", hide_index=True,
        column_config={
            "$ / Signal": st.column_config.NumberColumn(format="$%.2f"),
            "$ / Trade": st.column_config.NumberColumn(format="$%.2f"),
            "Total $ Risk": st.column_config.NumberColumn(format="$%.2f"),
        })

    st.markdown('<div class="section-title">🌐 All-Pairs Lot Sizer</div>', unsafe_allow_html=True)
    st.caption(f"Lots needed to risk {risk_pct:.2f}% (${ar_r['risk_per_trade']:,.2f}/trade) "
               f"with each pair's live ATR×1.5 stop.")

    if st.button("📡 Scan all pairs (live ATR)", width="content", key="ar_scan_btn"):
        st.cache_data.clear()
        st.rerun()

    with st.expander("📋 All-Pairs Sizing Table", expanded=False):
        ar_prog = st.progress(0, text="Scanning…")
        ar_scan_rows = []
        ar_pairs = list(INSTRUMENTS.items())
        for ar_i, (ar_name, ar_info) in enumerate(ar_pairs):
            ar_prog.progress((ar_i + 1) / len(ar_pairs), text=f"Fetching {ar_name}…")
            ar_atr_pips = fetch_atr14_pips(ar_info.ticker, ar_info.pip_size)
            if ar_atr_pips is None:
                ar_scan_rows.append({"Pair": ar_name, "ATR Stop (pips)": None,
                                     "Lots / Trade": None, "Total Lots": None,
                                     "$ / Trade": None})
                continue
            ar_sl = round(ar_atr_pips * 1.5, 1)
            ar_rr = size_position(account_bal, risk_pct, ar_sl, ar_info.pip, ar_n_trades, ar_split_mode)
            ar_scan_rows.append({
                "Pair": ar_name,
                "ATR Stop (pips)": round(ar_sl, 1),
                "Lots / Trade": round(ar_rr["lots_per_trade"], 2),
                "Total Lots": round(ar_rr["lots_total"], 2),
                "$ / Trade": round(ar_rr["risk_per_trade"], 2),
            })
        ar_prog.empty()
        st.dataframe(
            pd.DataFrame(ar_scan_rows), width="stretch", hide_index=True,
            column_config={"$ / Trade": st.column_config.NumberColumn(format="$%.2f")})

    st.markdown("---")

    ar_col_a, ar_col_b = st.columns(2)
    with ar_col_a:
        st.markdown(f"""
        <div class="explainer">
        <b style="color:#00ff41;">📏 How the lot size is derived</b><br><br>
        Money at risk is fixed first — <b style="color:#e6e6e6;">{risk_pct:.2f}%</b> of your
        ${account_bal:,.0f} balance = <b style="color:{ar_band_color};">${ar_r['risk_signal']:,.2f}</b>
        per signal.<br><br>
        Lot size then falls out of the stop distance: a wider stop forces a smaller
        lot so the dollar loss stays constant. This is why position size is an
        <em>output</em> of risk, never an input.
        </div>
        """, unsafe_allow_html=True)
    with ar_col_b:
        st.markdown(f"""
        <div class="explainer">
        <b style="color:#00ff41;">🎯 Two trades per signal</b><br><br>
        Every signal is entered as <b style="color:#e6e6e6;">two positions</b>. In
        <b>{ar_split_mode.lower()}</b> mode each trade carries
        <b style="color:#56d364;">{ar_r['lots_per_trade']:.2f} lots</b>
        (${ar_r['risk_per_trade']:,.2f} risk), for a combined
        <b style="color:{ar_band_color};">{ar_r['lots_total']:.2f} lots</b> /
        ${ar_r['risk_total']:,.2f} on the signal.<br><br>
        Switch to <em>per-trade</em> mode in the sidebar if each entry should carry
        the full risk % independently.
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="formula-box">
    $ at risk &nbsp;= &nbsp;balance × risk%
    &nbsp;= &nbsp;${account_bal:,.0f} × {risk_pct:.2f}%
    &nbsp;= &nbsp;<b style="color:{ar_band_color};">${ar_r['risk_signal']:,.2f}</b><br>
    Lots / trade &nbsp;= &nbsp;$ risk per trade ÷ (SL pips × pip value)
    &nbsp;= &nbsp;${ar_r['risk_per_trade']:,.2f} ÷ ({ar_sl_pips:.1f} × ${pip_val:.2f})
    &nbsp;= &nbsp;<b style="color:#00ff41;">{ar_r['lots_per_trade']:.2f} lots</b><br>
    Total exposure &nbsp;= &nbsp;{ar_r['lots_per_trade']:.2f} × {ar_n_trades} trades
    &nbsp;= &nbsp;<b style="color:#00ff41;">{ar_r['lots_total']:.2f} lots</b>
    &nbsp;(${ar_r['risk_total']:,.2f} total risk)
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center;color:#555555;font-size:11px;margin-top:32px;
            padding-top:16px;border-top:1px solid #2a2a2a;">
  🛡️ Risk Suite · Stop Structure + R:R Calculator + Account Risk · For educational purposes only
</div>
""", unsafe_allow_html=True)
