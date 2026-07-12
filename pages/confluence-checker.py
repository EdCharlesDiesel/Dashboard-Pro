import streamlit as st
from src.ui.theme import BloombergTheme
from src.core.config import CANDLE_STYLE
from src.pages_lib.navigation import render_sidebar_nav
from src.services.signal_store import persist_signals
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import pytz

# ── Page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="2/3 Confluence Checker",
    page_icon="🔀",
    layout="wide",
    initial_sidebar_state="expanded",
)
BloombergTheme.apply()

# ── CSS — matches main app dark theme ─────────────────────────────
st.markdown("""
<style>
    /* Theme-adaptive layout vars */
    .stApp {
      --border: color-mix(in srgb, var(--text-color) 12%, transparent);
      --muted:  color-mix(in srgb, var(--text-color) 55%, transparent);
    }
    html,body,[class*="css"]{ font-family:'JetBrains Mono','Fira Code',monospace; }
    .stApp { background:var(--background-color); }
    section[data-testid="stSidebar"]{ background:var(--secondary-background-color)!important; border-right:1px solid var(--border,#2a2a2a); }

    .card{ background:var(--secondary-background-color); border:1px solid var(--border,#2a2a2a); border-radius:12px;
           padding:20px; margin-bottom:16px; }
    .card-header{ font-size:13px; font-weight:600; letter-spacing:.08em;
                  text-transform:uppercase; color:#9a9a9a; margin-bottom:14px; }
    .metric-box{ background:var(--background-color); border:1px solid var(--border,#2a2a2a); border-radius:8px;
                 padding:14px; text-align:center; }
    .metric-value{ font-size:22px; font-weight:700; color:var(--text-color); }
    .metric-label{ font-size:11px; color:var(--muted,#9a9a9a); margin-top:2px; font-weight:500;
                   letter-spacing:.04em; text-transform:uppercase; }
    .section-title{ font-size:16px; font-weight:700; color:var(--text-color); margin:24px 0 12px 0;
                    padding-left:4px; border-left:3px solid #00ff41; }
    .prog-track{ background:var(--border,#2a2a2a); border-radius:8px; height:12px;
                 margin:6px 0 2px 0; overflow:hidden; }
    [data-testid="stSidebarNav"]{ display:none; }
    #MainMenu,footer{ visibility:hidden; }
    [data-testid="stSidebarCollapsedControl"]{visibility:visible !important;}
    .block-container{ padding-top:1.5rem; max-width:1380px; }

    /* Element confirmation cards */
    .el-card-pass{ background:linear-gradient(135deg,#0d3a1f,#0a2918);
                   border:1px solid #00a32a; border-left:4px solid #00ff66;
                   border-radius:10px; padding:18px 20px; margin-bottom:12px; }
    .el-card-fail{ background:linear-gradient(135deg,#1c1c1c,#0a0a0a);
                   border:1px solid #2a2a2a; border-left:4px solid #555555;
                   border-radius:10px; padding:18px 20px; margin-bottom:12px; }
    .el-card-partial{ background:linear-gradient(135deg,#2a1f00,#201800);
                      border:1px solid #9e6a03; border-left:4px solid #ffcc00;
                      border-radius:10px; padding:18px 20px; margin-bottom:12px; }
    .el-title{ font-size:15px; font-weight:700; }
    .el-value{ font-size:13px; font-family:monospace; margin-top:6px; }
    .el-detail{ font-size:12px; color:#9a9a9a; margin-top:4px; line-height:1.6; }

    /* Verdict banners */
    .verdict-pass3{ background:linear-gradient(135deg,#0d3a1f,#0d5e32);
                    border:2px solid #00a32a; border-radius:14px;
                    padding:22px 28px; text-align:center; margin-bottom:18px; }
    .verdict-pass2{ background:linear-gradient(135deg,#1c2a00,#243300);
                    border:2px solid #00ff66; border-radius:14px;
                    padding:22px 28px; text-align:center; margin-bottom:18px; }
    .verdict-fail { background:linear-gradient(135deg,#3a0d0d,#5e1414);
                    border:2px solid #8b2d2d; border-radius:14px;
                    padding:22px 28px; text-align:center; margin-bottom:18px; }
    .verdict-wait { background:linear-gradient(135deg,#1c1c0d,#2e2a0d);
                    border:2px solid #9e6a03; border-radius:14px;
                    padding:22px 28px; text-align:center; margin-bottom:18px; }

    /* Proximity gauge */
    .prox-row{ display:flex; justify-content:space-between; align-items:center;
               padding:8px 0; border-bottom:1px solid #2a2a2a; font-size:13px; }
    .prox-row:last-child{ border-bottom:none; }
    .prox-label{ color:#9a9a9a; }
    .prox-val{ font-family:monospace; font-weight:600; }

    .explainer{ background:var(--background-color); border:1px solid #1e3a5f;
                border-left:3px solid #00ff41; border-radius:8px;
                padding:14px 18px; font-size:13px; color:var(--muted,#9a9a9a); line-height:1.7; }
</style>
""", unsafe_allow_html=True)


# ── Instrument registry ────────────────────────────────────────────
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
    "XAU/USD":    {"ticker": "GC=F",     "pip_size": 0.10},
    "XAG/USD":  {"ticker": "SI=F",     "pip_size": 0.01},
    "XPT/USD":{"ticker": "PL=F",     "pip_size": 0.10},
    "WTI/USD":    {"ticker": "CL=F",     "pip_size": 0.01},
}


# ══════════════════════════════════════════════════════════════════
# DATA & CALCULATIONS
# ══════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner=False)
def fetch_4h(ticker: str, days: int = 90) -> pd.DataFrame:
    from src.db.market_cache import cached_ohlc
    try:
        end   = datetime.now(pytz.utc)
        start = end - timedelta(days=days)
        df    = cached_ohlc(ticker, start=start, end=end, interval="1h", ttl=300)
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open","High","Low","Close","Volume"]].dropna()
        df = df.resample("4h").agg({
            "Open":"first","High":"max","Low":"min",
            "Close":"last","Volume":"sum"
        }).dropna()
        return df
    except Exception:
        return pd.DataFrame()


def calc_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def calc_fibonacci(swing_high: float, swing_low: float) -> dict:
    diff = swing_high - swing_low
    return {
        "0.0%":   swing_high,
        "23.6%":  swing_high - 0.236 * diff,
        "38.2%":  swing_high - 0.382 * diff,
        "50.0%":  swing_high - 0.500 * diff,
        "61.8%":  swing_high - 0.618 * diff,
        "78.6%":  swing_high - 0.786 * diff,
        "100%":   swing_low,
    }


def calc_pivots(df: pd.DataFrame) -> dict:
    prev = df.iloc[-2]
    H, L, C = float(prev["High"]), float(prev["Low"]), float(prev["Close"])
    PP = (H + L + C) / 3
    return {
        "PP": PP,
        "R1": 2*PP - L,   "R2": PP + (H-L),   "R3": H + 2*(PP-L),
        "S1": 2*PP - H,   "S2": PP - (H-L),   "S3": L - 2*(H-PP),
    }


def nearest_level(price: float, levels: dict, tol_pct: float) -> dict:
    """Return the closest level and whether it's within tolerance."""
    tol      = price * tol_pct
    best_lbl = None
    best_val = None
    best_dist = float("inf")
    for lbl, val in levels.items():
        dist = abs(val - price)
        if dist < best_dist:
            best_dist = dist
            best_lbl  = lbl
            best_val  = val
    pct_away = (best_val - price) / price * 100 if best_val else None
    within   = best_dist <= tol if best_val else False
    return {
        "label":   best_lbl,
        "value":   best_val,
        "dist":    best_dist,
        "pct_away":pct_away,
        "within":  within,
    }


def check_ema_proximity(price: float, ema_vals: dict, tol_pct: float) -> dict:
    """Find the closest EMA and whether price is within tolerance of it."""
    tol       = price * tol_pct
    best_lbl  = None
    best_val  = None
    best_dist = float("inf")
    for lbl, val in ema_vals.items():
        dist = abs(val - price)
        if dist < best_dist:
            best_dist = dist
            best_lbl  = lbl
            best_val  = val
    pct_away = (best_val - price) / price * 100 if best_val else None
    within   = best_dist <= tol if best_val else False
    return {
        "label":   best_lbl,
        "value":   best_val,
        "dist":    best_dist,
        "pct_away":pct_away,
        "within":  within,
    }


def evaluate_confluence(price, fib_res, pivot_res, ema_res, pdh_pdl_bonus=False) -> dict:
    elements = {
        "Fibonacci": fib_res["within"],
        "Pivot":     pivot_res["within"],
        "EMA":       ema_res["within"],
    }
    passed = sum(elements.values())
    if passed == 3:
        verdict, grade = "3/3 — PERFECT CONFLUENCE", "pass3"
    elif passed == 2:
        verdict, grade = "2/3 — MIN CONFLUENCE MET ✅", "pass2"
    elif passed == 1:
        verdict, grade = "1/3 — INSUFFICIENT", "wait"
    else:
        verdict, grade = "0/3 — NO CONFLUENCE", "fail"
    if pdh_pdl_bonus and passed >= 2:
        verdict += " + PDH/PDL KEY LEVEL"
    return {"elements": elements, "passed": passed, "verdict": verdict, "grade": grade,
            "pdh_pdl_bonus": pdh_pdl_bonus}


@st.cache_data(ttl=300, show_spinner=False)
def get_pdh_pdl(ticker: str):
    """Return previous day High/Low from daily data."""
    from src.db.market_cache import cached_ohlc
    try:
        df = cached_ohlc(ticker, period="5d", interval="1d", ttl=300)
        if df.empty or len(df) < 2:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        prev = df.iloc[-2]
        return {"pdh": float(prev["High"]), "pdl": float(prev["Low"])}
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════
# CHART
# ══════════════════════════════════════════════════════════════════

def build_chart(df: pd.DataFrame, pair: str,
                fib_levels: dict, pivots: dict, ema_vals: dict,
                price: float, tol_pct: float,
                fib_res: dict, pivot_res: dict, ema_res: dict,
                show_n: int = 80) -> go.Figure:

    plot = df.tail(show_n).copy()
    tol  = price * tol_pct

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.80, 0.20], vertical_spacing=0.03,
    )

    # Candles
    fig.add_trace(go.Candlestick(
        x=plot.index,
        open=plot["Open"], high=plot["High"],
        low=plot["Low"],   close=plot["Close"],
        **CANDLE_STYLE,
        name="4H Candles", showlegend=False,
    ), row=1, col=1)

    # Volume
    vol_col = ["rgba(63,185,80,0.33)" if c >= o else "rgba(248,81,73,0.33)"
               for c, o in zip(plot["Close"], plot["Open"])]
    fig.add_trace(go.Bar(
        x=plot.index, y=plot["Volume"],
        marker_color=vol_col, name="Volume", showlegend=False,
    ), row=2, col=1)

    # ── Fibonacci levels ──────────────────────────────────────────
    fib_colors = {
        "0.0%":"rgba(255,255,255,0.27)","23.6%":"#00d4ff","38.2%":"#a29bfe",
        "50.0%":"#ffcc00", "61.8%":"#ff6b35","78.6%":"#ff3344","100%":"rgba(255,255,255,0.27)",
    }
    key_fibs = {"38.2%","50.0%","61.8%"}   # show labels only on key levels
    for lbl, val in fib_levels.items():
        is_nearest = (lbl == fib_res["label"])
        width = 2.0 if is_nearest else 0.8
        dash  = "solid" if is_nearest else "dot"
        color = fib_colors.get(lbl, "#555555")
        fig.add_hline(
            y=val, line_dash=dash, line_color=color,
            line_width=width, opacity=0.75 if is_nearest else 0.45,
            annotation_text=f"Fib {lbl}" if lbl in key_fibs else "",
            annotation_position="right",
            annotation_font=dict(size=9, color=color),
            row=1, col=1,
        )

    # ── Pivot levels ──────────────────────────────────────────────
    pivot_colors = {
        "PP":"#ffffff","R1":"#00d4ff","R2":"#0077ff","R3":"#003f8a",
        "S1":"#ff6b35","S2":"#ff3344","S3":"#8a1a00",
    }
    for lbl, val in pivots.items():
        is_nearest = (lbl == pivot_res["label"])
        width = 2.0 if is_nearest else 0.8
        dash  = "solid" if is_nearest else "dash"
        color = pivot_colors.get(lbl, "#555555")
        fig.add_hline(
            y=val, line_dash=dash, line_color=color,
            line_width=width, opacity=0.75 if is_nearest else 0.40,
            annotation_text=f"  {lbl}" if is_nearest else "",
            annotation_position="left",
            annotation_font=dict(size=9, color=color),
            row=1, col=1,
        )

    # ── EMAs ──────────────────────────────────────────────────────
    ema_line_colors = {
        list(ema_vals.keys())[0]: "#f59e0b",
        list(ema_vals.keys())[1]: "#a78bfa",
        list(ema_vals.keys())[2]: "#34d399",
    }
    for col_key, ema_col_name in zip(
            ["EMA_s","EMA_m","EMA_l"],
            list(ema_vals.keys()),
    ):
        if col_key in plot.columns:
            is_nearest = (ema_col_name == ema_res["label"])
            fig.add_trace(go.Scatter(
                x=plot.index, y=plot[col_key],
                mode="lines",
                line=dict(color=ema_line_colors.get(ema_col_name,"#888"),
                          width=2.5 if is_nearest else 1.2),
                name=ema_col_name, opacity=1.0 if is_nearest else 0.5,
            ), row=1, col=1)

    # ── Confluence zone band around price ─────────────────────────
    fig.add_hrect(
        y0=price - tol, y1=price + tol,
        fillcolor="#00ff41", opacity=0.07, line_width=0,
        row=1, col=1,
    )
    fig.add_hline(
        y=price, line_dash="solid",
        line_color="#ffffff", line_width=1.5, opacity=0.90,
        annotation_text=f"  Price {price:.5f}",
        annotation_position="right",
        annotation_font=dict(size=10, color="#ffffff"),
        row=1, col=1,
    )

    fig.update_layout(
        height=620,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0a0a0a",
        font=dict(family="JetBrains Mono, monospace", size=11, color="#9a9a9a"),
        xaxis_rangeslider_visible=False,
        legend=dict(
            bgcolor="rgba(13,17,23,0.60)", bordercolor="#2a2a2a", borderwidth=1,
            font=dict(size=10), orientation="h",
            yanchor="bottom", y=1.01, xanchor="left", x=0,
        ),
        margin=dict(l=10, r=110, t=10, b=10),
    )
    fig.update_yaxes(gridcolor="#2a2a2a", zeroline=False)
    fig.update_xaxes(gridcolor="#2a2a2a", showgrid=False)
    return fig


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 🔀 2/3 Confluence Checker")

    inst_keys    = list(INSTRUMENTS.keys())
    default_inst = st.session_state.get("selected_instrument", "EUR/USD")
    if default_inst not in inst_keys:
        default_inst = inst_keys[0]

    selected_pair = st.selectbox("Instrument", inst_keys,
                                 index=inst_keys.index(default_inst))

    lookback_days = st.slider("Lookback (days)", 30, 120, 60, step=15)
    show_candles  = st.slider("Candles on chart",  30, 200,  80, step=10)

    st.divider()
    st.markdown("**📐 EMA Periods**")
    ema_s = st.number_input("EMA Short", value=20,  min_value=5,   max_value=200)
    ema_m = st.number_input("EMA Mid",   value=50,  min_value=10,  max_value=500)
    ema_l = st.number_input("EMA Long",  value=200, min_value=20,  max_value=800)

    st.divider()
    st.markdown("**🎯 Confluence Tolerance**")
    tol_pct = st.slider("Price tolerance (%)", 0.1, 2.0, 0.5, step=0.1) / 100

    st.divider()
    st.markdown("**📏 Fibonacci Swing**")
    swing_lb = st.slider("Swing lookback (candles)", 10, 100, 50, step=5)

    st.divider()
    if st.button("🔄 Refresh Data", width="stretch", type="primary"):
        st.cache_data.clear()
        st.rerun()

    st.caption(f"🕐 {datetime.now().strftime('%H:%M:%S')} local")


    st.divider()
    render_sidebar_nav()
# ══════════════════════════════════════════════════════════════════
# MAIN PAGE
# ══════════════════════════════════════════════════════════════════

ticker   = INSTRUMENTS[selected_pair]["ticker"]
pip_size = INSTRUMENTS[selected_pair]["pip_size"]

# Page header
st.markdown(f"""
<div style="background:linear-gradient(135deg,#000000 0%,#0a0a0a 50%,#000000 100%);
            border:1px solid #2a2a2a; border-radius:16px; padding:24px 28px;
            margin-bottom:20px; position:relative; overflow:hidden;">
  <div style="font-size:24px; font-weight:700; color:#e6e6e6;">
    🔀 Min 2/3 Confluence Elements
  </div>
  <div style="color:#9a9a9a; font-size:13px; margin-top:4px;">
    At least 2 of 3 elements confirmed at zone · Fibonacci · Pivot Points · EMA · {selected_pair}
  </div>
  <div style="font-size:12px; color:#00ff41; margin-top:6px;">
    Check #11 — 2/3 Confluence Checker · {datetime.now().strftime('%A %d %B %Y  |  %H:%M')}
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Tabs
# ─────────────────────────────────────────────
tab_scan, tab_detail = st.tabs(["📊 All-Pairs Scanner", "🔍 Pair Detail"])

# ══════════════════════════════════════════════
#  TAB 1 — All-Pairs Scanner
# ══════════════════════════════════════════════
with tab_scan:
    st.markdown("Scanning all 21 instruments for 2/3 confluence …")
    prog = st.progress(0)
    all_instruments = list(INSTRUMENTS.items())
    n_inst = len(all_instruments)
    scan_results: list[dict] = []

    for idx, (pair_name, info) in enumerate(all_instruments):
        prog.progress((idx + 1) / n_inst, text=f"Scanning {pair_name} …")
        try:
            df_s = fetch_4h(info["ticker"], 60)
            if df_s.empty or len(df_s) < 15:
                scan_results.append({"pair": pair_name, "ok": False})
                continue
            df_s["EMA_s"] = calc_ema(df_s["Close"], 20)
            df_s["EMA_m"] = calc_ema(df_s["Close"], 50)
            df_s["EMA_l"] = calc_ema(df_s["Close"], 200)
            price_s    = float(df_s["Close"].iloc[-1])
            sh_s       = float(df_s["High"].rolling(50, min_periods=1).max().iloc[-1])
            sl_s       = float(df_s["Low"].rolling(50, min_periods=1).min().iloc[-1])
            fib_s      = calc_fibonacci(sh_s, sl_s)
            piv_s      = calc_pivots(df_s)
            ema_s_vals = {
                "EMA 20":  float(df_s["EMA_s"].iloc[-1]),
                "EMA 50":  float(df_s["EMA_m"].iloc[-1]),
                "EMA 200": float(df_s["EMA_l"].iloc[-1]),
            }
            tol_s     = 0.005
            fib_r     = nearest_level(price_s, fib_s, tol_s)
            piv_r     = nearest_level(price_s, piv_s, tol_s)
            ema_r     = check_ema_proximity(price_s, ema_s_vals, tol_s)
            result_s  = evaluate_confluence(price_s, fib_r, piv_r, ema_r)
            scan_results.append({
                "pair":    pair_name,
                "ok":      True,
                "price":   price_s,
                "passed":  result_s["passed"],
                "grade":   result_s["grade"],
                "verdict": result_s["verdict"],
            })
        except Exception:
            scan_results.append({"pair": pair_name, "ok": False})

    prog.empty()

    # Persist confluence-met events (≥2/3 elements). Confluence is non-directional
    # (price sitting at a key zone), so bias is recorded Neutral. source='confluence_checker'.
    persist_signals("confluence_checker", [{
        "pair": r["pair"],
        "bias": "Neutral",
        "entry": r["price"],
        "conviction": r["verdict"],
        "thesis": f"Confluence Checker — {r['verdict']} ({r.get('passed', 0)}/3 elements)",
    } for r in scan_results if r.get("ok") and r.get("grade") in ("pass3", "pass2")])

    # ── Summary counts ──────────────────────────
    cnt_3 = sum(1 for r in scan_results if r.get("grade") == "pass3")
    cnt_2 = sum(1 for r in scan_results if r.get("grade") == "pass2")
    cnt_1 = sum(1 for r in scan_results if r.get("grade") == "wait")
    cnt_0 = sum(1 for r in scan_results if r.get("grade") == "fail")

    sc1, sc2, sc3, sc4 = st.columns(4)
    for col_, val_, lbl_, c_ in [
        (sc1, cnt_3, "3/3 Perfect",    "#00ff66"),
        (sc2, cnt_2, "2/3 Min Met",    "#a3e635"),
        (sc3, cnt_1, "1/3 Partial",    "#ffcc00"),
        (sc4, cnt_0, "0/3 No Confluence", "#ff3344"),
    ]:
        with col_:
            st.markdown(
                f'<div class="metric-box">'
                f'<div class="metric-value" style="color:{c_};font-size:28px;">{val_}</div>'
                f'<div class="metric-label">{lbl_}</div>'
                f'</div>',
                unsafe_allow_html=True)

    st.markdown("---")

    # ── Card grid ───────────────────────────────
    GRADE_COLOR = {
        "pass3": "#00ff66",
        "pass2": "#a3e635",
        "wait":  "#ffcc00",
        "fail":  "#ff3344",
    }
    GRADE_BG = {
        "pass3": "rgba(63,185,80,.10)",
        "pass2": "rgba(163,230,53,.08)",
        "wait":  "rgba(227,179,65,.08)",
        "fail":  "rgba(248,81,73,.08)",
    }

    cols3 = st.columns(3)
    for i, r in enumerate(scan_results):
        pair_name = r["pair"]
        is_sel    = pair_name == selected_pair
        border    = "border:2px solid #00ff41;" if is_sel else "border:1px solid #2a2a2a;"

        if not r.get("ok"):
            card_body = '<span style="font-size:11px;color:#ff3344;">No data</span>'
        else:
            grade  = r["grade"]
            color  = GRADE_COLOR.get(grade, "#555555")
            bg     = GRADE_BG.get(grade, "transparent")
            passed = r["passed"]
            dots   = "".join(
                f'<span style="display:inline-block;width:10px;height:10px;border-radius:50%;'
                f'background:{"#00ff66" if j < passed else "#2a2a2a"};margin:0 2px;"></span>'
                for j in range(3)
            )
            card_body = (
                f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                f'<span style="font-size:13px;font-weight:600;color:#e6e6e6;">{pair_name}</span>'
                f'<span style="background:{bg};border:1px solid {color}44;border-radius:4px;'
                f'padding:2px 8px;font-size:11px;color:{color};font-weight:700;">{passed}/3</span>'
                f'</div>'
                f'<div style="margin-top:6px;">{dots}</div>'
                f'<div style="font-size:11px;color:#9a9a9a;margin-top:4px;font-family:monospace;">'
                f'{r["price"]:,.5f}</div>'
            )

        with cols3[i % 3]:
            st.markdown(
                f'<div style="background:#0a0a0a;{border}border-radius:8px;'
                f'padding:12px 14px;margin-bottom:8px;">{card_body}</div>',
                unsafe_allow_html=True)

    # ── Expander table ───────────────────────────
    with st.expander("📋 Full Scanner Results"):
        table_rows = []
        for r in scan_results:
            table_rows.append({
                "Pair":    r["pair"],
                "Price":   f'{r["price"]:,.5f}' if r.get("ok") else "—",
                "Score":   f'{r["passed"]}/3'   if r.get("ok") else "—",
                "Grade":   {"pass3": "3/3 Perfect", "pass2": "2/3 Min Met",
                           "wait": "1/3 Partial", "fail": "0/3 None"}.get(
                           r.get("grade",""), "Error") if r.get("ok") else "Error",
                "Verdict": r.get("verdict", "No data"),
            })
        st.dataframe(pd.DataFrame(table_rows), width="stretch", hide_index=True)

# ══════════════════════════════════════════════
#  TAB 2 — Pair Detail
# ══════════════════════════════════════════════
with tab_detail:
    detail_ok = True

    # Fetch
    with st.spinner(f"Loading 4H data for {selected_pair}…"):
        df = fetch_4h(ticker, lookback_days)

    if df.empty or len(df) < 15:
        st.error(f"⚠️ Not enough 4H data for **{selected_pair}**. "
                 "Try increasing the lookback or a different instrument.")
        detail_ok = False

    if detail_ok:
        # ── Compute indicators ───────────────────────
        df["EMA_s"] = calc_ema(df["Close"], int(ema_s))
        df["EMA_m"] = calc_ema(df["Close"], int(ema_m))
        df["EMA_l"] = calc_ema(df["Close"], int(ema_l))

        price      = float(df["Close"].iloc[-1])
        swing_high = float(df["High"].rolling(swing_lb).max().iloc[-1])
        swing_low  = float(df["Low"].rolling(swing_lb).min().iloc[-1])

        fib_levels = calc_fibonacci(swing_high, swing_low)
        pivots     = calc_pivots(df)
        ema_vals   = {
            f"EMA {int(ema_s)}":  float(df["EMA_s"].iloc[-1]),
            f"EMA {int(ema_m)}":  float(df["EMA_m"].iloc[-1]),
            f"EMA {int(ema_l)}":  float(df["EMA_l"].iloc[-1]),
        }

        fib_res   = nearest_level(price, fib_levels, tol_pct)
        pivot_res = nearest_level(price, pivots,     tol_pct)
        ema_res   = check_ema_proximity(price, ema_vals, tol_pct)

        # PDH/PDL bonus check
        _pdh_pdl = get_pdh_pdl(ticker)
        _pdh_pdl_bonus = False
        _pdh_pdl_label = ""
        if _pdh_pdl:
            _tol = price * tol_pct
            if abs(price - _pdh_pdl["pdh"]) <= _tol:
                _pdh_pdl_bonus = True
                _pdh_pdl_label = f"PDH @ {_pdh_pdl['pdh']:.5f}"
            elif abs(price - _pdh_pdl["pdl"]) <= _tol:
                _pdh_pdl_bonus = True
                _pdh_pdl_label = f"PDL @ {_pdh_pdl['pdl']:.5f}"

        result = evaluate_confluence(price, fib_res, pivot_res, ema_res, _pdh_pdl_bonus)

        # ── Verdict banner ───────────────────────────
        VERDICT_STYLE = {
            "pass3": ("verdict-pass3", "#00ff66", "🎯 PERFECT CONFLUENCE",
                      "All 3 elements confirmed at current price zone — highest conviction"),
            "pass2": ("verdict-pass2", "#a3e635", "✅ MIN CONFLUENCE MET",
                      "2 of 3 elements align — minimum threshold passed for entry consideration"),
            "wait":  ("verdict-wait",  "#ffcc00", "⚠️ PARTIAL — WAIT",
                      "Only 1 element confirmed — do not enter, wait for price to reach a stronger zone"),
            "fail":  ("verdict-fail",  "#ff3344", "❌ NO CONFLUENCE",
                      "No elements confirmed at this price — zone is invalid, stay flat"),
        }
        vcls, vcolor, vtitle, vdesc = VERDICT_STYLE[result["grade"]]

        dots = "".join(
            f'<span style="display:inline-block;width:22px;height:22px;border-radius:50%;'
            f'background:{"" + vcolor if i < result["passed"] else "#2a2a2a"};margin:0 4px;"></span>'
            for i in range(3)
        )

        _key_badge_html = (
            f'<div style="margin-top:10px;">'
            f'<span style="background:rgba(255,215,0,.2);border:1px solid rgba(255,215,0,.6);'
            f'border-radius:6px;padding:5px 16px;font-size:13px;font-weight:700;color:#ffd700;">'
            f'📌 KEY LEVEL: {_pdh_pdl_label}</span></div>'
        ) if _pdh_pdl_bonus else ""

        st.markdown(f"""
        <div class="{vcls}">
          <div style="font-size:23px; font-weight:800; color:{vcolor};
                      letter-spacing:1px; margin-bottom:8px;">{vtitle}</div>
          <div style="margin:8px 0;">{dots}</div>
          {_key_badge_html}
          <div style="font-size:13px; color:#9a9a9a; margin-top:8px;">{vdesc}</div>
          <div style="font-size:12px; color:#555555; margin-top:10px;">
            Current price &nbsp;
            <code style="color:#e6e6e6;">{price:.5f}</code>
            &nbsp;·&nbsp; Tolerance &nbsp;
            <code style="color:#e6e6e6;">±{tol_pct*100:.1f}%</code>
            &nbsp;·&nbsp; Zone &nbsp;
            <code style="color:#e6e6e6;">{price*(1-tol_pct):.5f} – {price*(1+tol_pct):.5f}</code>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── KPI row ──────────────────────────────────
        tol_pips        = price * tol_pct / pip_size
        fib_dist_pips   = fib_res["dist"]   / pip_size if fib_res["value"]   else 0
        pivot_dist_pips = pivot_res["dist"] / pip_size if pivot_res["value"] else 0
        ema_dist_pips   = ema_res["dist"]   / pip_size if ema_res["value"]   else 0

        k1, k2, k3, k4, k5 = st.columns(5)
        for col_, val_, lbl_, color_ in [
            (k1, f"{result['passed']}/3",   "Elements Confirmed",    vcolor),
            (k2, f"{tol_pips:.1f}",         "Tolerance (pips)",      "#00ff41"),
            (k3, f"{fib_dist_pips:.1f}",    "Fib Distance (pips)",   "#a29bfe" if fib_res["within"] else "#555555"),
            (k4, f"{pivot_dist_pips:.1f}",  "Pivot Distance (pips)", "#00d4ff" if pivot_res["within"] else "#555555"),
            (k5, f"{ema_dist_pips:.1f}",    "EMA Distance (pips)",   "#f59e0b" if ema_res["within"] else "#555555"),
        ]:
            with col_:
                st.markdown(
                    f'<div class="metric-box">'
                    f'<div class="metric-value" style="color:{color_};">{val_}</div>'
                    f'<div class="metric-label">{lbl_}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("---")

        # ── Element confirmation cards ───────────────
        st.markdown('<div class="section-title">🧩 Element Breakdown</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)

        def element_card(col, icon, name, passed, nearest_lbl, nearest_val,
                         dist_pips, pct_away, detail_lines):
            card_cls = "el-card-pass" if passed else "el-card-fail"
            status   = "✅ CONFIRMED" if passed else "❌ NOT AT ZONE"
            color    = "#00ff66" if passed else "#555555"
            with col:
                lines_html = "".join(
                    f'<div style="display:flex;justify-content:space-between;'
                    f'padding:3px 0;border-bottom:1px solid rgba(33,38,45,0.40);">'
                    f'<span style="color:#9a9a9a;font-size:11px;">{k}</span>'
                    f'<span style="font-family:monospace;font-size:11px;color:#e6e6e6;">{v}</span>'
                    f'</div>'
                    for k, v in detail_lines.items()
                )
                st.markdown(f"""
                <div class="{card_cls}">
                  <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div class="el-title">{icon} {name}</div>
                    <span style="font-size:12px;font-weight:700;color:{color};">{status}</span>
                  </div>
                  <div class="el-value" style="color:{color};">
                    {nearest_lbl} &nbsp;@ &nbsp;{nearest_val:.5f}
                  </div>
                  <div style="margin-top:10px;">{lines_html}</div>
                </div>
                """, unsafe_allow_html=True)

        element_card(
            c1, "📐", "Fibonacci",
            fib_res["within"],
            fib_res["label"], fib_res["value"],
            fib_dist_pips, fib_res["pct_away"],
            {
                "Nearest level": fib_res["label"],
                "Level price":   f'{fib_res["value"]:.5f}',
                "Distance":      f'{fib_dist_pips:.1f} pips',
                "% from price":  f'{fib_res["pct_away"]:+.3f}%',
                "Swing high":    f'{swing_high:.5f}',
                "Swing low":     f'{swing_low:.5f}',
                "Tolerance":     f'±{tol_pips:.1f} pips',
            }
        )
        element_card(
            c2, "🔵", "Pivot Point",
            pivot_res["within"],
            pivot_res["label"], pivot_res["value"],
            pivot_dist_pips, pivot_res["pct_away"],
            {
                "Nearest level": pivot_res["label"],
                "Level price":   f'{pivot_res["value"]:.5f}',
                "Distance":      f'{pivot_dist_pips:.1f} pips',
                "% from price":  f'{pivot_res["pct_away"]:+.3f}%',
                "PP":            f'{pivots["PP"]:.5f}',
                "R1 / S1":       f'{pivots["R1"]:.5f} / {pivots["S1"]:.5f}',
                "Tolerance":     f'±{tol_pips:.1f} pips',
            }
        )
        element_card(
            c3, "〰️", "EMA",
            ema_res["within"],
            ema_res["label"], ema_res["value"],
            ema_dist_pips, ema_res["pct_away"],
            {
                "Nearest EMA":      ema_res["label"],
                "EMA price":        f'{ema_res["value"]:.5f}',
                "Distance":         f'{ema_dist_pips:.1f} pips',
                "% from price":     f'{ema_res["pct_away"]:+.3f}%',
                f"EMA {int(ema_s)}": f'{ema_vals[f"EMA {int(ema_s)}"]:.5f}',
                f"EMA {int(ema_m)}": f'{ema_vals[f"EMA {int(ema_m)}"]:.5f}',
                f"EMA {int(ema_l)}": f'{ema_vals[f"EMA {int(ema_l)}"]:.5f}',
            }
        )

        # ── Progress bar summary ─────────────────────
        st.markdown("---")
        st.markdown('<div class="section-title">📊 Confluence Score</div>', unsafe_allow_html=True)

        for el_name, el_passed in result["elements"].items():
            el_color = "#00ff66" if el_passed else "#ff3344"
            el_icon  = "✅" if el_passed else "❌"
            el_pct   = 100 if el_passed else 0
            st.markdown(f"""
            <div style="margin-bottom:12px;">
              <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                <span style="font-size:13px;color:#e6e6e6;">{el_icon} &nbsp;{el_name}</span>
                <span style="font-size:12px;font-weight:700;color:{el_color};">
                  {"CONFIRMED" if el_passed else "NOT AT ZONE"}
                </span>
              </div>
              <div class="prog-track">
                <div style="background:{el_color};width:{el_pct}%;height:100%;border-radius:8px;"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        overall_pct = int(result["passed"] / 3 * 100)
        overall_col = "#00ff66" if result["passed"] >= 2 else "#ff3344"
        st.markdown(f"""
        <div style="margin-top:16px;">
          <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
            <span style="font-size:14px;font-weight:700;color:#e6e6e6;">Overall Confluence</span>
            <span style="font-size:14px;font-weight:800;color:{overall_col};">
              {result['passed']}/3 · {overall_pct}%
            </span>
          </div>
          <div class="prog-track" style="height:16px;">
            <div style="background:linear-gradient(90deg,{overall_col},{overall_col}cc);
                        width:{overall_pct}%;height:100%;border-radius:8px;
                        transition:width .4s;"></div>
          </div>
          <div style="font-size:12px;color:#9a9a9a;margin-top:6px;">
            Minimum required: 2/3 (66%) · {"✅ Threshold met" if result["passed"] >= 2 else "❌ Below threshold — do not enter"}
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # ── Chart ────────────────────────────────────
        st.markdown('<div class="section-title">📈 4H Chart — All 3 Elements</div>',
                    unsafe_allow_html=True)
        fig = build_chart(
            df, selected_pair,
            fib_levels=fib_levels, pivots=pivots, ema_vals=ema_vals,
            price=price, tol_pct=tol_pct,
            fib_res=fib_res, pivot_res=pivot_res, ema_res=ema_res,
            show_n=show_candles,
        )
        st.plotly_chart(fig, width="stretch")

        st.markdown("---")

        # ── Full levels reference table ──────────────
        st.markdown('<div class="section-title">📋 All Levels Reference</div>', unsafe_allow_html=True)
        rows = []
        for lbl, val in fib_levels.items():
            dist = abs(val - price)
            rows.append({"Element": "Fibonacci", "Label": lbl,
                         "Level": round(val, 5), "Δ Pips": round(dist / pip_size, 1),
                         "Δ %": round((val - price) / price * 100, 3),
                         "In Zone": "✅" if dist <= price * tol_pct else "—"})
        for lbl, val in pivots.items():
            dist = abs(val - price)
            rows.append({"Element": "Pivot", "Label": lbl,
                         "Level": round(val, 5), "Δ Pips": round(dist / pip_size, 1),
                         "Δ %": round((val - price) / price * 100, 3),
                         "In Zone": "✅" if dist <= price * tol_pct else "—"})
        for lbl, val in ema_vals.items():
            dist = abs(val - price)
            rows.append({"Element": "EMA", "Label": lbl,
                         "Level": round(val, 5), "Δ Pips": round(dist / pip_size, 1),
                         "Δ %": round((val - price) / price * 100, 3),
                         "In Zone": "✅" if dist <= price * tol_pct else "—"})

        ref_df = pd.DataFrame(rows).sort_values("Δ Pips").reset_index(drop=True)
        st.dataframe(ref_df, width="stretch", hide_index=True,
                     column_config={
                         "Element": st.column_config.TextColumn("Element", width="small"),
                         "Label":   st.column_config.TextColumn("Label",   width="small"),
                         "Level":   st.column_config.NumberColumn("Level",  format="%.5f"),
                         "Δ Pips":  st.column_config.NumberColumn("Δ Pips", format="%.1f"),
                         "Δ %":     st.column_config.NumberColumn("Δ %",    format="%.3f"),
                         "In Zone": st.column_config.TextColumn("In Zone",  width="small"),
                     })

        st.markdown("---")

        # ── Strategy explainer ───────────────────────
        st.markdown('<div class="section-title">📖 How the 2/3 Rule Works</div>',
                    unsafe_allow_html=True)
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown(f"""
            <div class="explainer">
            <b style="color:#a29bfe;">📐 Fibonacci</b><br><br>
            Calculates retracement levels from the {swing_lb}-candle swing high to swing low.<br><br>
            <b>Key levels:</b> 38.2%, 50.0%, 61.8%<br>
            These represent likely zones where price pauses or reverses after a trend move.<br><br>
            <b>Confirmed when:</b> current price is within ±{tol_pct*100:.1f}% of any Fib level.
            </div>
            """, unsafe_allow_html=True)
        with col_b:
            st.markdown(f"""
            <div class="explainer">
            <b style="color:#00d4ff;">🔵 Pivot Points</b><br><br>
            Classic floor-trader pivots calculated from the previous completed 4H candle's High, Low, Close.<br><br>
            <b>Key levels:</b> PP, S1, R1<br>
            PP is the gravitational centre. Price reacts to pivot levels across all timeframes.<br><br>
            <b>Confirmed when:</b> current price is within ±{tol_pct*100:.1f}% of any pivot level.
            </div>
            """, unsafe_allow_html=True)
        with col_c:
            st.markdown(f"""
            <div class="explainer">
            <b style="color:#f59e0b;">〰️ EMA Stack</b><br><br>
            Three exponential moving averages: EMA {int(ema_s)} / {int(ema_m)} / {int(ema_l)}.<br><br>
            <b>Key role:</b> Dynamic support and resistance. The EMA acts as a magnet for price during pullbacks in trending markets.<br><br>
            <b>Confirmed when:</b> current price is within ±{tol_pct*100:.1f}% of any EMA.
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="explainer" style="margin-top:12px;">
        <b style="color:#00ff41;">⚙️ The 2/3 Minimum Rule</b><br><br>
        A single indicator at a price level is coincidence. Two or more is confluence — a statistically stronger
        reaction zone where multiple market participants are watching the same level.<br><br>
        <b>Score guide:</b>&nbsp;
        <span style="color:#ff3344;">0/3 — No trade</span> &nbsp;·&nbsp;
        <span style="color:#ffcc00;">1/3 — Wait</span> &nbsp;·&nbsp;
        <span style="color:#a3e635;">2/3 — Minimum to consider entry</span> &nbsp;·&nbsp;
        <span style="color:#00ff66;">3/3 — Highest conviction setup</span><br><br>
        This is <em>Check #11</em>. Pair with Check #10 (4H confluence zone confirmed) and
        Check #12 (15M rejection candle) before triggering entry.
        </div>
        """, unsafe_allow_html=True)

        # Footer
        st.markdown("""
        <div style="text-align:center;color:#555555;font-size:11px;margin-top:32px;
                    padding-top:16px;border-top:1px solid #2a2a2a;">
          🔀 Min 2/3 Confluence Elements · Check #11 · Fibonacci · Pivot · EMA · For educational purposes only
        </div>
        """, unsafe_allow_html=True)